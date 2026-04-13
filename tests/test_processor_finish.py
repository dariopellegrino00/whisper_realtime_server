import logging

import numpy as np

from swim.runtime import ParallelOnlineASRProcessor


class DummyASR:
    sep = " "


def make_words(texts, start=0.0, step=0.5):
    words = []
    current = start
    for text in texts:
        words.append((current, current + step, text))
        current += step
    return words


def make_processor():
    processor = ParallelOnlineASRProcessor(DummyASR(), logger=logging.getLogger("test"))
    processor.init()
    return processor


def test_finish_flushes_current_hypothesis_when_last_update_was_already_emitted():
    processor = make_processor()

    processor.update(make_words(["tail"], start=1.0))
    processor.mark_update_emitted()

    assert processor.finish() == (1.0, 1.5, "tail")


def test_finish_includes_unemitted_results_and_hypothesis_after_final_decode():
    processor = make_processor()

    processor.update(make_words(["hello"]))
    processor.mark_update_emitted()

    processor.insert_audio_chunk(np.zeros(8000, dtype=np.float32))
    processor.update(make_words(["hello", "world"]))

    assert processor.finish() == (0.0, 1.0, "hello world")


def test_has_audio_since_last_decode_detects_pending_audio():
    processor = make_processor()

    processor.insert_audio_chunk(np.zeros(8000, dtype=np.float32))
    processor.update(make_words(["hello"]))

    assert processor.has_audio_since_last_decode() is False

    processor.insert_audio_chunk(np.zeros(8000, dtype=np.float32))

    assert processor.has_audio_since_last_decode() is True
