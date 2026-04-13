import logging
from types import SimpleNamespace

import numpy as np
import pytest

import swim.runtime.shared_asr as pwo


def _segment(start, end, words, no_speech_prob=0.0):
    return SimpleNamespace(
        start=start,
        end=end,
        no_speech_prob=no_speech_prob,
        words=[
            SimpleNamespace(start=word_start, end=word_end, word=text)
            for word_start, word_end, text in words
        ],
    )


class DummyModel:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio_len": len(audio), "kwargs": kwargs})
        return iter(self._segments), None


@pytest.mark.parametrize(
    ("backend", "expected_audio_len", "expected_clip_timestamps", "expected_extra_kwargs"),
    [
        ("plain", 32800, [0.0, 2.0], {}),
        (
            "batched",
            32100,
            [{"start": 0, "end": 32000}],
            {"without_timestamps": False, "batch_size": 16},
        ),
    ],
)
def test_transcribe_parallel_restores_original_timestamps_when_vad_enabled(
    monkeypatch, backend, expected_audio_len, expected_clip_timestamps, expected_extra_kwargs
):
    dummy_model = DummyModel(
        [
            _segment(
                0.1,
                1.2,
                [
                    (0.1, 0.2, " hello"),
                    (0.9, 1.2, " world"),
                ],
            )
        ]
    )

    monkeypatch.setattr(
        pwo.MultiProcessingFasterWhisperASR,
        "load_model",
        lambda self, modelsize=None, cache_dir=None, model_dir=None: dummy_model,
    )
    monkeypatch.setattr(
        pwo,
        "get_speech_timestamps",
        lambda audio, _options: [
            {"start": 0, "end": 16000},
            {"start": 64000, "end": 80000},
        ],
    )

    asr = pwo.MultiProcessingFasterWhisperASR(
        "auto",
        modelsize="tiny",
        logfile=logging.getLogger("test"),
        use_vad=True,
        backend=backend,
    )
    buffer = pwo.ParallelAudioBuffer()
    buffer.append_token("clip-1", np.zeros(80000, dtype=np.float32))

    results = asr.transcribe_parallel(buffer)

    assert results == [
        (
            "clip-1",
            [
                (0.1, 0.2, " hello"),
                (3.9, 4.2, " world"),
            ],
        )
    ]
    assert dummy_model.calls == [
        {
            "audio_len": expected_audio_len,
            "kwargs": {
                "beam_size": 5,
                "condition_on_previous_text": False,
                "multilingual": True,
                "word_timestamps": True,
                "clip_timestamps": expected_clip_timestamps,
                **expected_extra_kwargs,
            },
        }
    ]


@pytest.mark.parametrize(
    ("backend", "expected_audio_len", "expected_clip_timestamps", "expected_extra_kwargs"),
    [
        ("plain", 80800, [0.0, 5.0], {}),
        (
            "batched",
            80100,
            [{"start": 0, "end": 80000}],
            {"without_timestamps": False, "batch_size": 16},
        ),
    ],
)
def test_transcribe_parallel_keeps_compact_timestamps_when_vad_disabled(
    monkeypatch, backend, expected_audio_len, expected_clip_timestamps, expected_extra_kwargs
):
    dummy_model = DummyModel(
        [
            _segment(
                0.1,
                1.2,
                [
                    (0.1, 0.2, " hello"),
                    (0.9, 1.2, " world"),
                ],
            )
        ]
    )

    monkeypatch.setattr(
        pwo.MultiProcessingFasterWhisperASR,
        "load_model",
        lambda self, modelsize=None, cache_dir=None, model_dir=None: dummy_model,
    )

    asr = pwo.MultiProcessingFasterWhisperASR(
        "auto",
        modelsize="tiny",
        logfile=logging.getLogger("test"),
        use_vad=False,
        backend=backend,
    )
    buffer = pwo.ParallelAudioBuffer()
    buffer.append_token("clip-1", np.zeros(80000, dtype=np.float32))

    results = asr.transcribe_parallel(buffer)

    assert results == [
        (
            "clip-1",
            [
                (0.1, 0.2, " hello"),
                (0.9, 1.2, " world"),
            ],
        )
    ]
    assert dummy_model.calls == [
        {
            "audio_len": expected_audio_len,
            "kwargs": {
                "beam_size": 5,
                "condition_on_previous_text": False,
                "multilingual": True,
                "word_timestamps": True,
                "clip_timestamps": expected_clip_timestamps,
                **expected_extra_kwargs,
            },
        }
    ]


def test_warmup_bypasses_vad_so_it_always_runs_one_inference(monkeypatch):
    dummy_model = DummyModel([])

    monkeypatch.setattr(
        pwo.MultiProcessingFasterWhisperASR,
        "load_model",
        lambda self, modelsize=None, cache_dir=None, model_dir=None: dummy_model,
    )
    monkeypatch.setattr(
        pwo, "load_audio_chunk", lambda filepath, beg, end: np.zeros(16000, dtype=np.float32)
    )
    monkeypatch.setattr(pwo, "get_speech_timestamps", lambda audio, _options: [])
    monkeypatch.setattr(pwo.os.path, "isfile", lambda filepath: True)

    asr = pwo.MultiProcessingFasterWhisperASR(
        "auto",
        modelsize="tiny",
        logfile=logging.getLogger("test"),
        use_vad=True,
        backend="plain",
    )

    asr.warmup("dummy.wav")

    assert dummy_model.calls == [
        {
            "audio_len": 16800,
            "kwargs": {
                "beam_size": 5,
                "condition_on_previous_text": False,
                "multilingual": True,
                "word_timestamps": True,
                "clip_timestamps": [0.0, 1.0],
            },
        }
    ]
