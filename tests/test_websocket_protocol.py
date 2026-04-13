import asyncio
import json
import logging
from contextlib import asynccontextmanager

import numpy as np
import pytest

from swim.transports.websocket.messages import (
    PCM_F32_LE,
    PCM_S16_LE,
    WebsocketProtocolError,
    build_completed_event,
    build_error_event,
    build_transcript_event,
    parse_finish_message,
    parse_start_message,
)
from swim.transports.websocket.server import (
    WebsocketTranscriptionServer,
    build_parser,
    websocket_max_message_size_bytes,
    websocket_start_timeout_seconds,
)
from swim.transports.websocket.session import WebsocketStreamSession


class FakeProcessor:
    def __init__(self):
        self.results = None
        self.hypothesis = None
        self.chunk_duration_seconds = None
        self.timed_out = False
        self.pending_audio_since_last_decode = False

    def finish(self):
        return self.results

    def mark_update_emitted(self):
        return None

    def has_audio_since_last_decode(self):
        return self.pending_audio_since_last_decode


class FakeProcessorManager:
    def __init__(self):
        self.id = "test-stream"
        self.logger = logging.getLogger("test-stream")
        self.server_logger = logging.getLogger("test-server")
        self.audio_queue = asyncio.Queue()
        self.inserted_batches = []
        self.stream_closed_calls = 0
        self.processor = FakeProcessor()

    async def insert_audio(self, already_collected_chunks=None):
        self.inserted_batches.append(np.array(already_collected_chunks, dtype=np.float32))

    def mark_stream_closed(self):
        self.stream_closed_calls += 1


class FakeWebsocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_messages = []
        self.closed = None

    async def recv(self):
        if not self._messages:
            raise AssertionError("No more fake websocket messages available")
        message = self._messages.pop(0)
        if isinstance(message, BaseException):
            raise message
        return message

    async def send(self, message):
        self.sent_messages.append(message)

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)


class FakeSharedASR:
    def __init__(self):
        self.ready_calls = 0

    async def set_processor_ready(self, _stream_id):
        self.ready_calls += 1
        return None


class FakeServerProcessorManager(FakeProcessorManager):
    @asynccontextmanager
    async def context(self):
        yield

    async def get_transcription(self):
        return None

    def is_finished(self):
        return self.processor.timed_out


class FakeServerSession:
    def __init__(self):
        self.processor_manager = FakeServerProcessorManager()
        self.id = self.processor_manager.id
        self.logger = logging.getLogger("test-stream")
        self.chunk_duration_millis = 500
        self.final_response_calls = 0

    async def manage_start_message(self, _first_message):
        return None

    async def consume_initial_audio_message(self, _message):
        return None

    async def request_enqueuer(self, _websocket):
        raise WebsocketProtocolError("Only finish is allowed here")

    def create_response(self):
        return []

    def final_response(self):
        self.final_response_calls += 1
        return []


class FakeGenericErrorServerSession(FakeServerSession):
    async def request_enqueuer(self, _websocket):
        raise RuntimeError("boom")


class FakeWebsocketServer(WebsocketTranscriptionServer):
    def __init__(self, session=None):
        super().__init__(
            shared_asr=FakeSharedASR(),
            main_server_logger=logging.getLogger("test-server"),
            max_chunk_duration_seconds=20.0,
        )
        self.session = FakeServerSession() if session is None else session
        self.create_stream_session_calls = 0

    def create_stream_session(self):
        self.create_stream_session_calls += 1
        return self.session


def make_session():
    return WebsocketStreamSession(FakeProcessorManager())


class FinalDecodeServerProcessorManager(FakeServerProcessorManager):
    async def get_transcription(self):
        self.processor.pending_audio_since_last_decode = self.processor.results is None
        self.processor.results = (0.0, 0.5, "confirmed")
        self.processor.hypothesis = (
            (0.5, 1.0, "tail") if self.processor.pending_audio_since_last_decode else None
        )


class FinalDecodeServerSession(FakeServerSession):
    def __init__(self):
        super().__init__()
        self.processor_manager = FinalDecodeServerProcessorManager()
        self.id = self.processor_manager.id

    async def request_enqueuer(self, websocket):
        while True:
            message = await websocket.recv()
            if isinstance(message, str):
                parse_finish_message(message)
                self.processor_manager.mark_stream_closed()
                return


def test_parse_start_message_accepts_expected_payload():
    message = json.dumps(
        {
            "type": "start",
            "chunk_duration_millis": 500,
            "audio_format": {
                "encoding": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
        }
    )

    parsed = parse_start_message(message, max_chunk_duration_millis=1000)

    assert parsed.chunk_duration_millis == 500
    assert parsed.encoding == PCM_S16_LE


def test_parse_start_message_accepts_pcm_f32le():
    message = json.dumps(
        {
            "type": "start",
            "chunk_duration_millis": 500,
            "audio_format": {
                "encoding": "pcm_f32le",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
        }
    )

    parsed = parse_start_message(message, max_chunk_duration_millis=1000)

    assert parsed.encoding == PCM_F32_LE


def test_parse_start_message_rejects_wrong_audio_format():
    message = json.dumps(
        {
            "type": "start",
            "chunk_duration_millis": 500,
            "audio_format": {
                "encoding": "pcm_u8",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
        }
    )

    with pytest.raises(WebsocketProtocolError) as excinfo:
        parse_start_message(message, max_chunk_duration_millis=1000)

    assert "pcm_s16le or pcm_f32le" in excinfo.value.message


def test_parse_finish_message_rejects_non_finish_event():
    with pytest.raises(WebsocketProtocolError) as excinfo:
        parse_finish_message(json.dumps({"type": "ping"}))

    assert "finish" in excinfo.value.message


def test_manage_start_message_sets_chunk_duration_and_max_chunk_bytes():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        assert session.chunk_duration_millis == 500
        assert session.audio_encoding == PCM_S16_LE
        assert session.processor_manager.processor.chunk_duration_seconds == 0.5
        assert session.max_chunk_bytes == 16000

    asyncio.run(scenario())


def test_manage_start_message_accepts_pcm_f32le_and_updates_max_chunk_bytes():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": PCM_F32_LE,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        assert session.audio_encoding == PCM_F32_LE
        assert session.max_chunk_bytes == 32000

    asyncio.run(scenario())


def test_consume_initial_audio_message_stores_audio_samples():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        raw_samples = np.array([16384, -8192], dtype="<i2")
        await session.consume_initial_audio_message(raw_samples.tobytes())
        assert len(session.processor_manager.inserted_batches) == 1
        assert np.allclose(
            session.processor_manager.inserted_batches[0],
            np.array([0.5, -0.25], dtype=np.float32),
        )

    asyncio.run(scenario())


def test_parse_audio_message_uses_little_endian_int16():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        samples = np.array([8192, -16384], dtype="<i2")
        parsed = session._parse_audio_message(samples.tobytes())
        assert np.allclose(parsed, np.array([0.25, -0.5], dtype=np.float32))

    asyncio.run(scenario())


def test_parse_audio_message_accepts_pcm_f32le():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": PCM_F32_LE,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        samples = np.array([0.25, -0.5], dtype=np.float32)
        parsed = session._parse_audio_message(samples.tobytes())
        assert np.allclose(parsed, samples)

    asyncio.run(scenario())


def test_request_enqueuer_stops_on_finish_message():
    async def scenario():
        session = make_session()
        await session.manage_start_message(
            json.dumps(
                {
                    "type": "start",
                    "chunk_duration_millis": 500,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    },
                }
            )
        )
        audio = np.array([3277, 6553], dtype="<i2").tobytes()
        websocket = FakeWebsocket([audio, json.dumps({"type": "finish"})])

        await session.request_enqueuer(websocket)

        queued = await session.processor_manager.audio_queue.get()
        assert np.allclose(queued, np.array([3277 / 32768.0, 6553 / 32768.0], dtype=np.float32))
        assert session.processor_manager.stream_closed_calls == 1

    asyncio.run(scenario())


def test_create_response_emits_confirmed_and_interim():
    session = make_session()
    session.processor_manager.processor.results = (0.1, 0.4, "confirmed")
    session.processor_manager.processor.hypothesis = (0.4, 0.7, "interim")

    responses = session.create_response()

    assert len(responses) == 1
    payload = json.loads(responses[0])
    assert payload["type"] == "transcript"
    assert payload["confirmed"]["text"] == "confirmed"
    assert payload["interim"]["text"] == "interim"


def test_build_transcript_event_omits_missing_fields():
    payload = json.loads(build_transcript_event(interim=(200, 500, "interim")))

    assert payload == {
        "type": "transcript",
        "interim": {
            "start_time_millis": 200,
            "end_time_millis": 500,
            "text": "interim",
        },
    }


def test_build_error_and_completed_events():
    assert json.loads(build_completed_event()) == {"type": "completed"}
    assert json.loads(build_error_event("invalid_argument", "bad")) == {
        "type": "error",
        "code": "invalid_argument",
        "message": "bad",
    }


def test_build_parser_enables_vad_and_fallback_by_default():
    args = build_parser().parse_args([])

    assert args.vad is True
    assert args.fallback is True


def test_websocket_start_timeout_uses_chunk_duration_limit():
    assert websocket_start_timeout_seconds(1.0) == 2.0
    assert websocket_start_timeout_seconds(0.2) == 1.0


def test_websocket_max_message_size_grows_with_max_chunk_duration():
    assert websocket_max_message_size_bytes(1.0) == 2**20
    assert websocket_max_message_size_bytes(10.0) == 2**20
    assert websocket_max_message_size_bytes(20.0) == 1280000


def test_handle_connection_turns_request_task_protocol_error_into_clean_close():
    async def scenario():
        server = FakeWebsocketServer()
        websocket = FakeWebsocket(['{"type":"start"}', np.array([0.1], dtype=np.float32).tobytes()])

        await server.handle_connection(websocket)

        assert len(websocket.sent_messages) == 1
        payload = json.loads(websocket.sent_messages[0])
        assert payload == {
            "type": "error",
            "code": "invalid_argument",
            "message": "Only finish is allowed here",
        }
        assert websocket.closed == (1008, "Only finish is allowed here")

    asyncio.run(scenario())


def test_handle_connection_turns_request_task_runtime_error_into_internal_error():
    async def scenario():
        server = FakeWebsocketServer(session=FakeGenericErrorServerSession())
        websocket = FakeWebsocket(['{"type":"start"}', np.array([0.1], dtype=np.float32).tobytes()])

        await server.handle_connection(websocket)

        assert len(websocket.sent_messages) == 1
        payload = json.loads(websocket.sent_messages[0])
        assert payload == {
            "type": "error",
            "code": "internal_error",
            "message": "Internal server error",
        }
        assert websocket.closed == (1011, "Internal server error")

    asyncio.run(scenario())


def test_handle_connection_times_out_before_allocating_stream_session(monkeypatch):
    async def scenario():
        server = FakeWebsocketServer()
        websocket = FakeWebsocket([])

        async def timeout_wait_for(awaitable, timeout):
            del timeout
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", timeout_wait_for)

        await server.handle_connection(websocket)

        assert server.create_stream_session_calls == 0
        assert len(websocket.sent_messages) == 1
        payload = json.loads(websocket.sent_messages[0])
        assert payload["type"] == "error"
        assert payload["code"] == "deadline_exceeded"
        assert "initial start event" in payload["message"]

    asyncio.run(scenario())


def test_handle_connection_allows_finish_before_first_audio():
    async def scenario():
        server = FakeWebsocketServer()
        websocket = FakeWebsocket(['{"type":"start"}', json.dumps({"type": "finish"})])

        await server.handle_connection(websocket)

        assert websocket.sent_messages == [build_completed_event()]
        assert websocket.closed == (1000, "completed")

    asyncio.run(scenario())


def test_handle_connection_runs_one_last_shared_decode_before_final_response():
    async def scenario():
        session = FinalDecodeServerSession()
        server = FakeWebsocketServer(session=session)
        websocket = FakeWebsocket(
            [
                '{"type":"start"}',
                np.array([1], dtype="<i2").tobytes(),
                json.dumps({"type": "finish"}),
            ]
        )

        await server.handle_connection(websocket)

        assert server._shared_asr.ready_calls == 2
        assert session.final_response_calls == 1
        assert websocket.closed == (1000, "completed")

    asyncio.run(scenario())
