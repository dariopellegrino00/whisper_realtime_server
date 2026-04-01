import logging
import sys

import numpy as np

from swim.runtime.hypothesis import HypothesisBuffer

logger = logging.getLogger(__name__)


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
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.kwargs = kwargs
        self.init()
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile, **self.kwargs)
        self.buffer_time_offset = 0 if offset is None else offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        prompt_words = [text for _, _, text in self.commited[:k]]
        prompt = []
        prompt_len = 0
        while prompt_words and prompt_len < 200:
            word = prompt_words.pop(-1)
            prompt_len += len(word) + 1
            prompt.append(word)

        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(text for _, _, text in non_prompt)

    def process_iter(self):
        prompt, non_prompt = self.prompt()
        logger.debug("PROMPT: %s", prompt)
        logger.debug("CONTEXT: %s", non_prompt)
        logger.debug(
            "transcribing %.2f seconds from %.2f",
            len(self.audio_buffer) / self.SAMPLING_RATE,
            self.buffer_time_offset,
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)
        completed = self.to_flush(committed)
        logger.debug(">>>>COMPLETE NOW: %s", completed)
        remainder = self.to_flush(self.transcript_buffer.complete())
        logger.debug("INCOMPLETE: %s", remainder)

        if committed and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        max_buffer_seconds = (
            self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        )
        if len(self.audio_buffer) / self.SAMPLING_RATE > max_buffer_seconds:
            self.chunk_completed_segment(res)
            logger.debug("chunking segment")

        logger.debug(
            "len of buffer now: %.2f",
            len(self.audio_buffer) / self.SAMPLING_RATE,
        )
        return self.to_flush(committed)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        logger.debug(self.commited)
        sentences = self.words_to_sentences(self.commited)
        for sentence in sentences:
            logger.debug("\t\tSENT: %s", sentence)
        if len(sentences) < 2:
            return
        while len(sentences) > 2:
            sentences.pop(0)
        chunk_at = sentences[-2][1]
        logger.debug("--- sentence chunked at %.2f", chunk_at)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.commited[-1][1]

        if len(ends) > 1:
            end_time = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and end_time > last_committed_time:
                ends.pop(-1)
                end_time = ends[-2] + self.buffer_time_offset
            if end_time <= last_committed_time:
                logger.debug("--- segment chunked at %.2f", end_time)
                self.chunk_at(end_time)
            else:
                logger.debug("--- last segment not within commited area")
        else:
            logger.debug("--- not enough segments to chunk")

    def chunk_at(self, time):
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE) :]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        cwords = [word for word in words]
        sentence_text = " ".join(word[2] for word in cwords)
        sentences = self.tokenizer.split(sentence_text)
        output = []
        while sentences:
            beg = None
            end = None
            sentence = sentences.pop(0).strip()
            full_sentence = sentence
            while cwords:
                word_beg, word_end, word = cwords.pop(0)
                word = word.strip()
                if beg is None and sentence.startswith(word):
                    beg = word_beg
                elif end is None and sentence == word:
                    end = word_end
                    output.append((beg, end, full_sentence))
                    break
                sentence = sentence[len(word) :].strip()
        return output

    def finish(self):
        remaining = self.transcript_buffer.complete()
        flushed = self.to_flush(remaining)
        logger.debug("last, noncommited: %s", flushed)
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return flushed

    def to_flush(self, sents, sep=None, offset=0):
        if sep is None:
            sep = self.asr.sep
        text = sep.join(sentence[2] for sentence in sents)
        if len(sents) == 0:
            beg = None
            end = None
        else:
            beg = offset + sents[0][0]
            end = offset + sents[-1][1]
        return (beg, end, text)


class ParallelOnlineASRProcessor(OnlineASRProcessor):
    def __init__(self, asr, logger=None, **kwargs):
        super().__init__(asr, **kwargs)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.buffer_trimming_sec = kwargs.get("buffer_trimming_sec", 15)
        self._result = None
        self._hypothesis = None
        self.timed_out = False

    @property
    def buffer_time_seconds(self):
        return len(self.audio_buffer) / self.SAMPLING_RATE

    def update(self, results):
        self.logger.debug("ITERATION START\n")
        self.logger.debug(
            "transcribing %.2f seconds from %.2f",
            self.buffer_time_seconds,
            self.buffer_time_offset,
        )

        self.transcript_buffer.insert(results, self.buffer_time_offset)
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        self._result = self.to_flush(committed)
        self.logger.debug(">>>>COMPLETE NOW: %s", self._result)

        self._hypothesis = self.to_flush(self.transcript_buffer.complete())
        self.logger.debug("INCOMPLETE: %s", self._hypothesis)

        self._chunk_buffer_at()

        self.logger.info("len of buffer now: %.2f", self.buffer_time_seconds)
        self.logger.debug("ITERATION END \n")

    @property
    def hypothesis(self):
        return self._hypothesis

    @property
    def results(self):
        return self._result

    def _chunk_buffer_at(self):
        k = len(self.commited) - 1
        if self.buffer_time_seconds > self.buffer_trimming_sec and k >= 0:
            limit = (
                self.buffer_time_offset + self.buffer_time_seconds - (self.buffer_trimming_sec / 2)
            )
            while k > 0 and self.commited[k][1] > limit:
                k -= 1
            chunk_time = self.commited[k][1]
            self.logger.debug(
                "chunking segment at word %s at %s",
                self.commited[-1],
                chunk_time,
            )
            self.chunk_at(chunk_time)


__all__ = [
    "OnlineASRProcessor",
    "ParallelOnlineASRProcessor",
]
