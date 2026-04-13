import json
from dataclasses import dataclass
from typing import Any

WEBSOCKET_TRANSCRIBE_PATH = "/v1/transcribe"
PCM_S16_LE = "pcm_s16le"
SAMPLE_RATE_HZ = 16000
CHANNELS = 1


class WebsocketProtocolError(Exception):
    def __init__(self, message, *, code="invalid_argument", close_code=1008):
        super().__init__(message)
        self.message = message
        self.code = code
        self.close_code = close_code


@dataclass(frozen=True)
class StartMessage:
    chunk_duration_millis: int


def encode_json(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


def parse_json_message(raw_message: str) -> dict[str, Any]:
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise WebsocketProtocolError("Text frames must contain valid JSON messages") from exc

    if not isinstance(message, dict):
        raise WebsocketProtocolError("Text frames must contain a JSON object")

    return message


def parse_start_message(raw_message: str, *, max_chunk_duration_millis: int) -> StartMessage:
    message = parse_json_message(raw_message)
    if message.get("type") != "start":
        raise WebsocketProtocolError("The first websocket message must be a start event")

    chunk_duration_millis = message.get("chunk_duration_millis")
    if not isinstance(chunk_duration_millis, int) or isinstance(chunk_duration_millis, bool):
        raise WebsocketProtocolError("chunk_duration_millis must be an integer")
    if chunk_duration_millis <= 0 or chunk_duration_millis > max_chunk_duration_millis:
        raise WebsocketProtocolError(
            f"chunk_duration_millis must be > 0 and <= {max_chunk_duration_millis}"
        )

    audio_format = message.get("audio_format")
    if not isinstance(audio_format, dict):
        raise WebsocketProtocolError("audio_format must be an object")
    if audio_format.get("encoding") != PCM_S16_LE:
        raise WebsocketProtocolError("audio_format.encoding must be pcm_s16le")
    if audio_format.get("sample_rate_hz") != SAMPLE_RATE_HZ:
        raise WebsocketProtocolError("audio_format.sample_rate_hz must be 16000")
    if audio_format.get("channels") != CHANNELS:
        raise WebsocketProtocolError("audio_format.channels must be 1")

    return StartMessage(chunk_duration_millis=chunk_duration_millis)


def parse_finish_message(raw_message: str) -> None:
    message = parse_json_message(raw_message)
    if message.get("type") != "finish":
        raise WebsocketProtocolError("Only a finish event is allowed after the start message")


def build_transcript_event(confirmed=None, interim=None) -> str:
    payload: dict[str, Any] = {"type": "transcript"}
    if confirmed is not None:
        payload["confirmed"] = {
            "start_time_millis": confirmed[0],
            "end_time_millis": confirmed[1],
            "text": confirmed[2],
        }
    if interim is not None:
        payload["interim"] = {
            "start_time_millis": interim[0],
            "end_time_millis": interim[1],
            "text": interim[2],
        }
    return encode_json(payload)


def build_completed_event() -> str:
    return encode_json({"type": "completed"})


def build_error_event(code: str, message: str) -> str:
    return encode_json({"type": "error", "code": code, "message": message})
