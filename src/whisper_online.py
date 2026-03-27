#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import logging

from rapidfuzz import fuzz, utils

logger = logging.getLogger(__name__)

MAX_DEDUP_NGRAM_SIZE = 10


@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


# Leave the hierarchy for possible extensions


class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
    # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(
        self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr
    ):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version."""

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        #        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        # print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr, **kwargs):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        # Confirmation logic args
        self.last_commited_time = 0
        self.last_commited_word = None
        # Fuzzy Confirmation logic and fallback args
        self.fuzz_threshold = kwargs.get("qratio_threshold", 95)
        self.dedup_threshold = kwargs.get("dedup_threshold", 98)
        self.use_fallback = kwargs.get("use_fallback", False)
        _fb_thresh = kwargs.get("fallback_threshold", 1)
        if _fb_thresh < 1:
            logger.warning(
                "fallback_threshold must be >= 1, got %s — clamping to 1", _fb_thresh
            )
        self.fallback_threshold = max(1, _fb_thresh)
        self.unconfirmed_amount = 0

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            # TODO: consider tweaking the 1 second threshold for the time difference
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Remove overlapping prefix: compare committed tail vs new head
                    # using n-grams from largest to smallest to find maximum overlap.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(min(cn, nn, MAX_DEDUP_NGRAM_SIZE), 0, -1):
                        c = " ".join(w[2] for w in self.commited_in_buffer[-i:])
                        tail = " ".join(w[2] for w in self.new[:i])
                        if (
                            fuzz.QRatio(c, tail, processor=utils.default_process)
                            >= self.dedup_threshold
                        ):
                            for j in range(i):
                                self.new.pop(0)
                            logger.debug(f"dedup: removed {i} overlapping word(s)")
                            break

    ### Changes from the original code
    # helper for flush not used in any other place
    def __commit_and_pop(self, num_new_pops, num_buffer_pops, commit):
        if num_new_pops <= 0:
            return

        new_not_popped = self.new[:num_new_pops]
        for w in new_not_popped:
            commit.append((w[0], w[1], w[2]))

        self.last_commited_word = new_not_popped[-1][2]  # nt
        self.last_commited_time = new_not_popped[-1][1]  # nb

        self.new = self.new[num_new_pops:]
        self.buffer = self.buffer[num_buffer_pops:]

    def __fallback(self, commit):  # TODO: more inspection in further fallback logics
        if not self.buffer or not self.new:
            return
        half = len(self.buffer) // 2

        # TODO: consider half new instead of half byffer
        prefixes = [
            " ".join([j[-1] for j in self.buffer[: i + 1]]) for i in range(half + 1)
        ]

        max_score = 0
        half_time = self.buffer[half][1] + 1
        new_filtered = [e for e in self.new if e[1] <= half_time]
        # best = { "buffer": "", "num_drops_buffer": 0,"new": "", "num_drops_new": 0,"score": 0,#}
        # uses a dict fot debugging purpose only
        num_drops_buffer = 0
        num_drops_new = 0

        for i in range(len(new_filtered)):
            for j, p in enumerate(prefixes):
                new_p = " ".join(k[-1] for k in new_filtered[: i + 1])
                this_score = fuzz.QRatio(p, new_p)
                if this_score > max_score:
                    max_score = this_score
                    num_drops_buffer = j + 1
                    num_drops_new = i + 1
        self.__commit_and_pop(num_drops_new, num_drops_buffer, commit)

    def flush(self):
        ### Changes from the original code for reduced delay at cost of more errors.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if (
                fuzz.QRatio(nt, self.buffer[0][2], processor=utils.default_process)
                >= self.fuzz_threshold
            ):
                self.__commit_and_pop(1, 1, commit)
            else:
                break

        if self.use_fallback:
            if commit:
                self.unconfirmed_amount = 0
            elif not self.buffer:
                pass  # populate flush — no prior buffer to compare against
            elif self.unconfirmed_amount >= self.fallback_threshold:
                self.__fallback(commit)
                self.unconfirmed_amount = 0
            else:
                self.unconfirmed_amount += 1

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenizer=None,
        buffer_trimming=("segment", 15),
        logfile=sys.stderr,
        **kwargs,
    ):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.kwargs = kwargs

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile, **self.kwargs)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(
            f"transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # there is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if (
                len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec
            ):  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30  # if the audio buffer is longer than 30s, trim it

        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            # l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k][1] > l:
            #    k -= 1
            # t = self.commited[k][1]
            logger.debug("chunking segment")
            # self.chunk_at(t)

        logger.debug(
            f"len of buffer now: {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f}"
        )
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE) :]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)
