import asyncio
import logging
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from grpc import StatusCode

fake_generated = ModuleType("swim.transports.grpc.generated")
fake_speech_pb2 = ModuleType("swim.transports.grpc.generated.speech_pb2")
fake_generated.speech_pb2 = fake_speech_pb2
sys.modules.setdefault("swim.transports.grpc.generated", fake_generated)
sys.modules.setdefault("swim.transports.grpc.generated.speech_pb2", fake_speech_pb2)

from swim.transports.grpc.session import StandardWhispStreamSession
from tests.conftest import AsyncIterator


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
                "insert_audio_chunk": lambda proc, audio: self.inserted_batches.append(audio),
            },
        )()

    async def insert_audio(self, already_collected_chunks=None):
        self.inserted_batches.append(np.array(already_collected_chunks, dtype=np.float32))

    def mark_stream_closed(self):
        self.stream_closed_calls += 1


class FakeRequest:
    def __init__(self, *, config=None, audio_bytes=None):
        self.config = None
        self.audio_chunk = None
        if config is not None:
            self.config = SimpleNamespace(chunk_duration_millis=config)
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
    return StandardWhispStreamSession(FakeProcessorManager())


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


def test_consume_initial_audio_request_uses_same_validation():
    async def scenario():
        session = make_session()
        await session.manage_first_message(FakeRequest(config=500), FakeContext())
        samples = np.array([0.1, 0.2], dtype=np.float32)
        await session.consume_initial_audio_request(FakeRequest(audio_bytes=samples.tobytes()), FakeContext())
        assert len(session.processor_manager.inserted_batches) == 1
        assert np.array_equal(session.processor_manager.inserted_batches[0], samples)

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
