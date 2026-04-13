from __future__ import annotations

import numpy as np
import numpy.typing as npt

PCM_F32_LE = "pcm_f32le"
PCM_S16_LE = "pcm_s16le"
SUPPORTED_AUDIO_ENCODINGS = (PCM_F32_LE, PCM_S16_LE)

_BYTES_PER_SAMPLE = {
    PCM_F32_LE: 4,
    PCM_S16_LE: 2,
}


def bytes_per_sample_for_encoding(encoding: str) -> int:
    try:
        return _BYTES_PER_SAMPLE[encoding]
    except KeyError as exc:
        raise ValueError(f"Unsupported audio encoding {encoding!r}") from exc


def decode_audio_bytes(audio_bytes: bytes, encoding: str) -> npt.NDArray[np.float32]:
    if encoding == PCM_F32_LE:
        # Use little-endian float32 explicitly ('<f4')
        return np.frombuffer(audio_bytes, dtype="<f4")
    if encoding == PCM_S16_LE:
        # Use little-endian int16 explicitly ('<i2')
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32)
        return samples / 32768.0
    raise ValueError(f"Unsupported audio encoding {encoding!r}")


def encode_audio_samples(samples: npt.ArrayLike, encoding: str) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    if encoding == PCM_F32_LE:
        return bytes(clipped.astype("<f4").tobytes())
    if encoding == PCM_S16_LE:
        return bytes(np.rint(clipped * 32767.0).astype("<i2").tobytes())
    raise ValueError(f"Unsupported audio encoding {encoding!r}")


__all__ = [
    "PCM_F32_LE",
    "PCM_S16_LE",
    "SUPPORTED_AUDIO_ENCODINGS",
    "bytes_per_sample_for_encoding",
    "decode_audio_bytes",
    "encode_audio_samples",
]
