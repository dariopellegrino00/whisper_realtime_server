import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest
from grpc import StatusCode

fake_generated = ModuleType("src.generated")
fake_speech_pb2_grpc = ModuleType("src.generated.speech_pb2_grpc")
fake_speech_pb2 = ModuleType("src.generated.speech_pb2")
fake_speech_pb2_grpc.SpeechToTextServicer = type("SpeechToTextServicer", (), {})
fake_speech_pb2_grpc.SpeechToTextWithHypothesisServicer = type(
    "SpeechToTextWithHypothesisServicer", (), {}
)
fake_generated.speech_pb2_grpc = fake_speech_pb2_grpc
fake_generated.speech_pb2 = fake_speech_pb2
sys.modules.setdefault("src.generated", fake_generated)
sys.modules.setdefault("src.generated.speech_pb2_grpc", fake_speech_pb2_grpc)
sys.modules.setdefault("src.generated.speech_pb2", fake_speech_pb2)

from src.server.whisper_server import BaseSpeechToTextServicer


class AbortCalled(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class FakeContext:
    async def abort(self, code, details):
        raise AbortCalled(code, details)


class FakeProcessorManager:
    def __init__(self, stream_session):
        self.id = "fake-stream"
        self.audio_queue = asyncio.Queue()
        self._finished = False
        self._stream_session = stream_session
        self.insert_calls = 0
        self.get_transcription_calls = 0

    @asynccontextmanager
    async def context(self):
        yield

    async def insert_audio(self):
        self.insert_calls += 1
        if not self.audio_queue.empty():
            self.audio_queue.get_nowait()

    async def get_transcription(self):
        self.get_transcription_calls += 1
        self._finished = True

    def is_finished(self):
        return self._finished


class FakeStreamSession:
    def __init__(self):
        self.processor_manager = FakeProcessorManager(self)
        self.id = self.processor_manager.id
        self.chunk_duration_millis = 10
        self.reader_cancelled = False
        self.create_response_calls = 0
        self.final_response_calls = 0

    async def manage_first_message(self, first_request, context):
        return None

    async def consume_initial_audio_request(self, request, context):
        return None

    async def request_enqueuer(self, request_iterator, context):
        try:
            async for item in request_iterator:
                await self.processor_manager.audio_queue.put(item)
        except asyncio.CancelledError:
            self.reader_cancelled = True
            raise
        finally:
            self.reader_cancelled = True

    def create_response(self):
        self.create_response_calls += 1
        return []

    def final_response(self):
        self.final_response_calls += 1
        return []


class FakeServicer(BaseSpeechToTextServicer):
    def __init__(self, stream_session):
        self._fake_stream_session = stream_session
        super().__init__(shared_asr=self, main_server_logger=logging.getLogger("test"))

    def StreamSessionType(self):
        raise NotImplementedError

    def create_stream_session(self):
        return self._fake_stream_session

    async def set_processor_ready(self, _id):
        return None


class AsyncIterator:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class HangingSecondMessageIterator:
    def __init__(self, first_request):
        self._first_request = first_request
        self._used = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._used:
            self._used = True
            return self._first_request
        await asyncio.sleep(3600)


def test_streaming_recognize_times_out_if_first_audio_never_arrives(monkeypatch):
    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)
        monkeypatch.setattr(servicer, "_first_audio_timeout_seconds", lambda _stream_session: 0.01)

        with pytest.raises(AbortCalled) as excinfo:
            async for _ in servicer.StreamingRecognize(
                HangingSecondMessageIterator(object()), context=FakeContext()
            ):
                pass

        assert excinfo.value.code is StatusCode.DEADLINE_EXCEEDED

    asyncio.run(scenario())


def test_streaming_recognize_drains_queued_audio_before_exit():
    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)
        request_iterator = AsyncIterator([object(), object(), [0.0], [0.1]])

        async for _ in servicer.StreamingRecognize(request_iterator, context=None):
            pass

        assert stream_session.processor_manager.insert_calls >= 1
        assert stream_session.processor_manager.get_transcription_calls == 1
        assert stream_session.final_response_calls == 0

    asyncio.run(scenario())
