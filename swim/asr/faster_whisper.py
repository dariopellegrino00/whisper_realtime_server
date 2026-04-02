import logging

from swim.asr.base import ASRBase

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASRBase):
    """Faster-Whisper backend used by the realtime runtime."""

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(
                "Loading whisper model from model_dir %s. modelsize and cache_dir parameters are not used.",
                model_dir,
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        return WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )

    def transcribe(self, audio, init_prompt=""):
        segments, _info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments):
        output = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                output.append((word.start, word.end, word.word))
        return output

    def segments_end_ts(self, res):
        return [segment.end for segment in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"
