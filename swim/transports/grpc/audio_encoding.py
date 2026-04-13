from swim.transports.audio_encoding import PCM_F32_LE, PCM_S16_LE
from swim.transports.grpc.generated import speech_pb2 as whisp_speech

AUDIO_ENCODING_UNSPECIFIED = getattr(whisp_speech, "AUDIO_ENCODING_UNSPECIFIED", 0)
AUDIO_ENCODING_PCM_F32LE = getattr(whisp_speech, "AUDIO_ENCODING_PCM_F32LE", 1)
AUDIO_ENCODING_PCM_S16LE = getattr(whisp_speech, "AUDIO_ENCODING_PCM_S16LE", 2)

_CANONICAL_TO_PROTO = {
    PCM_F32_LE: AUDIO_ENCODING_PCM_F32LE,
    PCM_S16_LE: AUDIO_ENCODING_PCM_S16LE,
}
_PROTO_TO_CANONICAL = {value: key for key, value in _CANONICAL_TO_PROTO.items()}


def normalize_proto_audio_encoding(encoding: int) -> str:
    if encoding == AUDIO_ENCODING_UNSPECIFIED:
        return PCM_F32_LE
    try:
        return _PROTO_TO_CANONICAL[encoding]
    except KeyError as exc:
        raise ValueError(f"Unsupported audio encoding {encoding!r}") from exc


def proto_audio_encoding_for(encoding: str) -> int:
    try:
        return _CANONICAL_TO_PROTO[encoding]
    except KeyError as exc:
        raise ValueError(f"Unsupported audio encoding {encoding!r}") from exc


__all__ = [
    "AUDIO_ENCODING_PCM_F32LE",
    "AUDIO_ENCODING_PCM_S16LE",
    "AUDIO_ENCODING_UNSPECIFIED",
    "normalize_proto_audio_encoding",
    "proto_audio_encoding_for",
]
