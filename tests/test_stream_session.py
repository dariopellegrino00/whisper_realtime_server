import asyncio
import logging
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from grpc import StatusCode

fake_generated = ModuleType("swim.transports.grpc.generated")
fake_speech_pb2 = ModuleType("swim.transports.grpc.generated.speech_pb2")
fake_speech_pb2_grpc = ModuleType("swim.transports.grpc.generated.speech_pb2_grpc")


class FakeTranscript:
    def __init__(self, start_time_millis=0, end_time_millis=0, text=""):
        self.start_time_millis = start_time_millis
        self.end_time_millis = end_time_millis
        self.text = text


class FakeStreamingRecognizeResponse:
    def __init__(self, confirmed=None, interim=None):
        self.confirmed = confirmed
        self.interim = interim

    def HasField(self, field_name):
        return getattr(self, field_name) is not None


fake_speech_pb2.Transcript = FakeTranscript
fake_speech_pb2.StreamingRecognizeResponse = FakeStreamingRecognizeResponse
fake_speech_pb2_grpc.SpeechToTextServicer = type("SpeechToTextServicer", (), {})
fake_generated.speech_pb2_grpc = fake_speech_pb2_grpc
fake_generated.speech_pb2 = fake_speech_pb2
sys.modules.setdefault("swim.transports.grpc.generated", fake_generated)
sys.modules.setdefault("swim.transports.grpc.generated.speech_pb2", fake_speech_pb2)
sys.modules.setdefault("swim.transports.grpc.generated.speech_pb2_grpc", fake_speech_pb2_grpc)

from swim.transports.grpc.session import SpeechStreamSession  # noqa: E402
from tests.conftest import AsyncIterator  # noqa: E402


class AbortCalled(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class FakeContext:
    async def abort(self, code, details):
        raise AbortCalled(code, details)


class FakeProcessorManager:
    def __init__(self):
        self.id = "test-stream"
        self.logger = logging.getLogger("test-stream")
        self.server_logger = logging.getLogger("test-server")
        self.audio_queue = asyncio.Queue()
        self.inserted_batches = []
        self.stream_closed_calls = 0
        self.processor = type(
            "Processor",
            (),
            {
                "results": None,
                "hypothesis": None,
                "chunk_duration_seconds": None,
                "mark_update_emitted": lambda proc: None,
                "insert_audio_chunk": lambda proc, audio: self.inserted_batches.append(audio),
            },
        )()

    async def insert_audio(self, already_collected_chunks=None):
        self.inserted_batches.append(np.array(already_collected_chunks, dtype=np.float32))

    def mark_stream_closed(self):
        self.stream_closed_calls += 1


class FakeRequest:
    def __init__(self, *, config=None, encoding=None, audio_bytes=None):
        self.config = None
        self.audio_chunk = None
        if config is not None:
            self.config = SimpleNamespace(chunk_duration_millis=config)
            if encoding is not None:
                self.config.encoding = encoding
        elif audio_bytes is not None:
            self.audio_chunk = SimpleNamespace(audio_bytes=audio_bytes)

    def WhichOneof(self, field_name):
        assert field_name == "payload"
        if self.config is not None:
            return "config"
        if self.audio_chunk is not None:
            return "audio_chunk"
        return None


def make_session():
    return SpeechStreamSession(FakeProcessorManager())


def test_manage_first_message_requires_config():
    async def scenario():
        session = make_session()
        with pytest.raises(AbortCalled) as excinfo:
            await session.manage_first_message(FakeRequest(audio_bytes=b"abc"), FakeContext())
        assert excinfo.value.code is StatusCode.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_manage_first_message_validates_chunk_duration():
    async def scenario():
        session = make_session()
        with pytest.raises(AbortCalled) as excinfo:
            await session.manage_first_message(FakeRequest(config=1001), FakeContext())
        assert excinfo.value.code is StatusCode.INVALID_ARGUMENT
        assert "<= 1000" in excinfo.value.details

    asyncio.run(scenario())


def test_manage_first_message_sets_chunk_duration_and_max_chunk_bytes():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500), FakeContext())
        assert session.chunk_duration_millis == 500
        assert session.processor_manager.processor.chunk_duration_seconds == 0.5
        assert session.max_chunk_bytes == 32000

    asyncio.run(scenario())


def test_manage_first_message_accepts_pcm_s16le_and_updates_max_chunk_bytes():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500, encoding=2), FakeContext())
        assert session.chunk_duration_millis == 500
        assert session.audio_encoding == 2
        assert session.max_chunk_bytes == 16000

    asyncio.run(scenario())


def test_manage_first_message_rejects_unknown_encoding():
    async def scenario():
        session = make_session()
        with pytest.raises(AbortCalled) as excinfo:
            await session.manage_first_message(FakeRequest(config=500, encoding=999), FakeContext())
        assert excinfo.value.code is StatusCode.INVALID_ARGUMENT
        assert "encoding" in excinfo.value.details

    asyncio.run(scenario())


def test_consume_initial_audio_request_uses_same_validation():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500), FakeContext())
        samples = np.array([0.1, 0.2], dtype=np.float32)
        await session.consume_initial_audio_request(
            FakeRequest(audio_bytes=samples.tobytes()), FakeContext()
        )
        assert len(session.processor_manager.inserted_batches) == 1
        assert np.array_equal(session.processor_manager.inserted_batches[0], samples)

    asyncio.run(scenario())


def test_consume_initial_audio_request_decodes_pcm_s16le():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500, encoding=2), FakeContext())
        samples = np.array([16384, -8192], dtype="<i2")
        await session.consume_initial_audio_request(
            FakeRequest(audio_bytes=samples.tobytes()), FakeContext()
        )
        assert len(session.processor_manager.inserted_batches) == 1
        assert np.allclose(
            session.processor_manager.inserted_batches[0],
            np.array([0.5, -0.25], dtype=np.float32),
        )

    asyncio.run(scenario())


def test_enqueue_audio_request_rejects_oversized_chunk():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500), FakeContext())
        oversized = np.zeros(8001, dtype=np.float32).tobytes()
        with pytest.raises(AbortCalled) as excinfo:
            await session.enqueue_audio_request(FakeRequest(audio_bytes=oversized), FakeContext())
        assert excinfo.value.code is StatusCode.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_request_enqueuer_logs_when_client_closes_request_stream(caplog):
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500), FakeContext())

        with caplog.at_level(logging.INFO, logger="swim.transports.grpc.session"):
            await session.request_enqueuer(AsyncIterator([]), FakeContext())

        assert "Client closed request stream for test-stream" in caplog.text
        assert session.processor_manager.stream_closed_calls == 1

    asyncio.run(scenario())


def test_create_response_emits_confirmed_and_interim_segments():
    session = make_session()
    session.processor_manager.processor.results = (0.1, 0.4, "confirmed")
    session.processor_manager.processor.hypothesis = (0.4, 0.7, "interim")

    responses = session.create_response()

    assert len(responses) == 1
    response = responses[0]
    assert response.confirmed.text == "confirmed"
    assert response.confirmed.start_time_millis == 100
    assert response.interim.text == "interim"
    assert response.interim.start_time_millis == 400


def test_create_response_emits_interim_without_confirmed():
    session = make_session()
    session.processor_manager.processor.results = None
    session.processor_manager.processor.hypothesis = (0.2, 0.5, "interim")

    responses = session.create_response()

    assert len(responses) == 1
    response = responses[0]
    assert not response.HasField("confirmed")
    assert response.interim.text == "interim"
