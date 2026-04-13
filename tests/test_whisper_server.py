import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest
from grpc import StatusCode

# We need to mock the gRPC generated modules BEFORE importing the servicer.
# This prevents ImportErrors since we don't want to rely on the actual protoc-generated files during unit tests.
fake_generated = ModuleType("swim.transports.grpc.generated")
fake_speech_pb2_grpc = ModuleType("swim.transports.grpc.generated.speech_pb2_grpc")
fake_speech_pb2 = ModuleType("swim.transports.grpc.generated.speech_pb2")
fake_speech_pb2_grpc.SpeechToTextServicer = type("SpeechToTextServicer", (), {})
fake_generated.speech_pb2_grpc = fake_speech_pb2_grpc
fake_generated.speech_pb2 = fake_speech_pb2
sys.modules.setdefault("swim.transports.grpc.generated", fake_generated)
sys.modules.setdefault("swim.transports.grpc.generated.speech_pb2_grpc", fake_speech_pb2_grpc)
sys.modules.setdefault("swim.transports.grpc.generated.speech_pb2", fake_speech_pb2)

from swim.transports.grpc.server import BaseSpeechToTextServicer, build_parser  # noqa: E402
from tests.conftest import AsyncIterator  # noqa: E402

# These mock classes allow us to isolate the gRPC servicer logic without
# spinning up a real ASR engine or a full network stack.


class FakeProcessorManager:
    def __init__(self, stream_session):
        self.id = "fake-stream"
        self.audio_queue = asyncio.Queue()
        self._finished = False
        self._stream_session = stream_session
        self.insert_calls = 0
        self.get_transcription_calls = 0
        self.processor = type(
            "Processor",
            (),
            {
                "pending_audio_since_last_decode": False,
                "has_audio_since_last_decode": lambda proc: proc.pending_audio_since_last_decode,
            },
        )()

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
        self.ready_calls = 0
        super().__init__(shared_asr=self, main_server_logger=logging.getLogger("test"))

    def StreamSessionType(self):
        raise NotImplementedError

    def create_stream_session(self):
        return self._fake_stream_session

    async def set_processor_ready(self, _id):
        self.ready_calls += 1
        return None


class HangingIterator:
    """Iterates first item then hangs indefinitely by waiting on a never-triggered event."""

    def __init__(self, first_item):
        self.first_item = first_item
        self.yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            return self.first_item
        await asyncio.Event().wait()


class CancelAfterFirstItemIterator:
    """Returns the config message, then simulates a client-side cancellation."""

    def __init__(self, first_item):
        self.first_item = first_item
        self.yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            return self.first_item
        raise asyncio.CancelledError


class FinalDecodeProcessorManager(FakeProcessorManager):
    async def get_transcription(self):
        self.get_transcription_calls += 1
        self.processor.pending_audio_since_last_decode = self.get_transcription_calls == 1


class FinalDecodeStreamSession(FakeStreamSession):
    def __init__(self):
        super().__init__()
        self.processor_manager = FinalDecodeProcessorManager(self)
        self.id = self.processor_manager.id


# Start of tests for the BaseSpeechToTextServicer


def test_streaming_recognize_times_out_if_first_audio_never_arrives(
    monkeypatch, fake_context, abort_exception, run_test
):
    """Verify that the servicer doesn't hang forever if the client stops sending audio."""

    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)
        # Force a very short timeout for testing purposes
        monkeypatch.setattr(servicer, "_first_audio_timeout_seconds", lambda _session: 0.01)

        with pytest.raises(abort_exception) as excinfo:
            async for _ in servicer.StreamingRecognize(
                HangingIterator(object()), context=fake_context
            ):
                pass

        assert excinfo.value.code is StatusCode.DEADLINE_EXCEEDED

    run_test(scenario())


def test_streaming_recognize_drains_queued_audio_before_exit(run_test):
    """Ensure that any pending audio in the queue is processed before closing the stream."""

    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)
        # Simulate a stream with config, first audio, and then more audio chunks
        request_iterator = AsyncIterator([object(), object(), [0.0], [0.1]])

        async for _ in servicer.StreamingRecognize(request_iterator, context=None):
            pass

        assert stream_session.processor_manager.insert_calls >= 1
        assert stream_session.processor_manager.get_transcription_calls == 1
        assert stream_session.final_response_calls == 0

    run_test(scenario())


def test_streaming_recognize_transcribes_initial_chunk_before_eof(run_test):
    """Ensure that we trigger a transcription even if the stream closes immediately after the first chunk."""

    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)
        # Stream closes immediately after sending the initial mandatory chunks
        request_iterator = AsyncIterator([object(), object()])

        async for _ in servicer.StreamingRecognize(request_iterator, context=None):
            pass

        assert stream_session.processor_manager.get_transcription_calls == 1
        assert stream_session.final_response_calls == 0

    run_test(scenario())


def test_streaming_recognize_runs_one_last_shared_decode_before_final_response(run_test):
    async def scenario():
        stream_session = FinalDecodeStreamSession()
        servicer = FakeServicer(stream_session)
        request_iterator = AsyncIterator([object(), object()])

        async for _ in servicer.StreamingRecognize(request_iterator, context=None):
            pass

        assert servicer.ready_calls == 2
        assert stream_session.processor_manager.get_transcription_calls == 2
        assert stream_session.final_response_calls == 1

    run_test(scenario())


def test_streaming_recognize_logs_startup_cancellation_without_claiming_disconnect(
    run_test, caplog
):
    """Ensure startup cancellations are traced without attributing them to the client."""

    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)

        with pytest.raises(asyncio.CancelledError):
            with caplog.at_level(logging.INFO, logger="test"):
                async for _ in servicer.StreamingRecognize(
                    CancelAfterFirstItemIterator(object()), context=None
                ):
                    pass

        assert "RPC cancelled during startup for fake-stream" in caplog.text

    run_test(scenario())


def test_streaming_recognize_logs_disconnect_if_client_closes_after_config(run_test, caplog):
    """Ensure EOF after the config message is still traced as a startup disconnect."""

    async def scenario():
        stream_session = FakeStreamSession()
        servicer = FakeServicer(stream_session)

        with caplog.at_level(logging.INFO, logger="test"):
            async for _ in servicer.StreamingRecognize(AsyncIterator([object()]), context=None):
                pass

        assert "Service fake-stream closed before sending the first audio chunk" in caplog.text

    run_test(scenario())


def test_build_parser_enables_vad_by_default():
    args = build_parser().parse_args([])

    assert args.vad is True


def test_build_parser_enables_fallback_by_default():
    args = build_parser().parse_args([])

    assert args.fallback is True


def test_build_parser_disables_fallback_with_no_fallback_flag():
    args = build_parser().parse_args(["--no-fallback"])

    assert args.fallback is False


def test_build_parser_rejects_removed_fallback_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--fallback"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments: --fallback" in captured.err


def test_build_parser_disables_vad_with_no_vad_flag():
    args = build_parser().parse_args(["--no-vad"])

    assert args.vad is False
