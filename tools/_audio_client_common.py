from __future__ import annotations

import sys
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import sounddevice as sd

AudioSamples = npt.NDArray[np.float32]


@dataclass(slots=True)
class AudioConfig:
    chunk_duration_seconds: float
    sample_rate: int = 16000
    channels: int = 1
    input_dtype: str = "float32"

    @property
    def chunk_duration_millis(self) -> int:
        return int(round(self.chunk_duration_seconds * 1000))

    @property
    def effective_chunk_duration_seconds(self) -> float:
        return self.chunk_duration_millis / 1000.0

    def chunk_size(self, sample_rate: int | None = None) -> int:
        effective_sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        return int(self.chunk_duration_millis * effective_sample_rate / 1000)


@dataclass(frozen=True, slots=True)
class LiveInputSettings:
    device_id: int
    device_name: str
    capture_sample_rate: int
    source_sample_rate: int
    channels: int


def normalize_samples(samples: npt.ArrayLike) -> AudioSamples:
    normalized = np.asarray(samples, dtype=np.float32)
    if normalized.ndim == 2:
        normalized = normalized.mean(axis=1)
    return normalized.reshape(-1)


def load_simulated_audio(filepath: str, sample_rate: int) -> AudioSamples:
    audio_data, _ = librosa.load(
        filepath,
        sr=sample_rate,
        mono=True,
        dtype=np.float32,
    )
    return normalize_samples(audio_data)


def print_live_input_banner(live_input: LiveInputSettings, audio_config: AudioConfig) -> None:
    print("Capturing real-time audio from the microphone...")
    print(
        f"Using input device {live_input.device_id}: {live_input.device_name} "
        f"({live_input.channels} channel(s), {live_input.capture_sample_rate} Hz)"
    )
    if live_input.capture_sample_rate != audio_config.sample_rate:
        print(
            f"Resampling microphone audio from {live_input.capture_sample_rate} Hz "
            f"to {audio_config.sample_rate} Hz"
        )


def resolve_live_input_settings(
    audio_config: AudioConfig,
    *,
    sound_device_id: int | None,
) -> LiveInputSettings:
    attempted: list[str] = []

    for device_id in _candidate_device_ids(sound_device_id):
        try:
            device_info = sd.query_devices(device_id)
        except sd.PortAudioError as exc:
            attempted.append(f"{device_id}: {exc}")
            continue

        if int(device_info["max_input_channels"]) <= 0:
            continue

        for channels in _candidate_channels(device_info):
            for sample_rate in _candidate_sample_rates(audio_config, device_info):
                try:
                    sd.check_input_settings(
                        device=device_id,
                        channels=channels,
                        dtype=audio_config.input_dtype,
                        samplerate=sample_rate,
                    )
                except sd.PortAudioError as exc:
                    attempted.append(
                        f"{device_id}@{sample_rate}Hz/{channels}ch ({device_info['name']}): {exc}"
                    )
                    continue

                return LiveInputSettings(
                    device_id=int(device_id),
                    device_name=str(device_info["name"]),
                    capture_sample_rate=sample_rate,
                    source_sample_rate=sample_rate,
                    channels=channels,
                )

    if sound_device_id is not None:
        prefix = f"Could not open input device {sound_device_id}."
    else:
        prefix = "Could not find a working microphone input device."

    details = "\n".join(attempted[:5])
    if len(attempted) > 5:
        details += f"\n... and {len(attempted) - 5} more failed combinations."
    raise RuntimeError(f"{prefix}\n{details}" if details else prefix)


class LiveAudioChunkProducer:
    def __init__(self, audio_config: AudioConfig, live_input: LiveInputSettings):
        self.audio_config = audio_config
        self.live_input = live_input
        self.chunk_samples = max(1, audio_config.chunk_size(live_input.capture_sample_rate))
        self.audio_queue: Queue[AudioSamples] = Queue(maxsize=100)
        self.buffered_chunks: list[AudioSamples] = []
        self.buffered_samples = 0
        self._stream: sd.InputStream | None = None
        self._running = False

    def __enter__(self) -> LiveAudioChunkProducer:
        self._running = True
        try:
            self._stream = sd.InputStream(
                device=self.live_input.device_id,
                channels=self.live_input.channels,
                samplerate=self.live_input.capture_sample_rate,
                blocksize=0,
                dtype=self.audio_config.input_dtype,
                latency="low",
                callback=self._audio_callback,
            )
            self._stream.__enter__()
        except sd.PortAudioError as exc:
            self._running = False
            raise RuntimeError(
                f"Could not start microphone input on device {self.live_input.device_id} "
                f"({self.live_input.device_name}): {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.__exit__(exc_type, exc, tb)
            self._stream = None

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if not self._running:
            return

        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        chunk = normalize_samples(indata)
        try:
            self.audio_queue.put_nowait(chunk)
        except Full:
            try:
                self.audio_queue.get_nowait()
            except Empty:
                pass
            try:
                self.audio_queue.put_nowait(chunk)
            except Full:
                pass

    def read_chunk(self) -> AudioSamples:
        while self._running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                return self._buffer_chunk(chunk)
            except Empty:
                continue
        raise EOFError("Producer is stopped")

    def _buffer_chunk(self, chunk: AudioSamples) -> AudioSamples:
        self.buffered_chunks.append(chunk)
        self.buffered_samples += len(chunk)

        while self._running and self.buffered_samples < self.chunk_samples:
            try:
                next_chunk = self.audio_queue.get(timeout=0.1)
                self.buffered_chunks.append(next_chunk)
                self.buffered_samples += len(next_chunk)
            except Empty:
                continue

        if not self._running:
            raise EOFError("Producer is stopped during buffering")

        chunk_native = normalize_samples(np.concatenate(self.buffered_chunks))
        if len(chunk_native) > self.chunk_samples:
            chunk_to_send = chunk_native[: self.chunk_samples]
            remainder = chunk_native[self.chunk_samples :]
            self.buffered_chunks = [remainder] if len(remainder) else []
            self.buffered_samples = len(remainder)
        else:
            chunk_to_send = chunk_native
            self.buffered_chunks = []
            self.buffered_samples = 0

        if self.live_input.source_sample_rate != self.audio_config.sample_rate:
            chunk_to_send = normalize_samples(
                librosa.resample(
                    chunk_to_send,
                    orig_sr=self.live_input.source_sample_rate,
                    target_sr=self.audio_config.sample_rate,
                )
            )

        return chunk_to_send


def add_common_client_args(parser: Any) -> None:
    parser.add_argument(
        "--all-updates",
        action="store_true",
        help="Print each response packet in a grouped block, including interim updates",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        help="Simulation mode: path to the audio file used to simulate realtime streaming",
    )
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="Display confirmed and interim text in a live preview view",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.0,
        help="Chunk duration in seconds for realtime streaming. Must be > 0",
    )
    parser.add_argument(
        "--sound-device-id",
        type=int,
        default=None,
        help=(
            "ID of the sound device to use for live audio capture. If not specified, "
            "the default input device will be used."
        ),
    )


def _default_input_device_id() -> int | None:
    default_device = sd.default.device
    if hasattr(default_device, "__getitem__"):
        default_input = default_device[0]
    else:
        default_input = default_device
    if default_input is None or default_input < 0:
        return None
    return int(default_input)


def _candidate_device_ids(sound_device_id: int | None) -> list[int]:
    if sound_device_id is not None:
        return [sound_device_id]

    candidates: list[int] = []
    default_device_id = _default_input_device_id()
    if default_device_id is not None:
        candidates.append(default_device_id)

    for device_id, device_info in enumerate(sd.query_devices()):
        if int(device_info["max_input_channels"]) > 0 and device_id not in candidates:
            candidates.append(device_id)
    return candidates


def _candidate_sample_rates(audio_config: AudioConfig, device_info: Any) -> list[int]:
    candidates: list[int] = []

    def add(rate: int) -> None:
        if rate > 0 and rate not in candidates:
            candidates.append(rate)

    add(audio_config.sample_rate)
    add(int(round(float(device_info["default_samplerate"]))))
    add(48000)
    add(44100)
    add(32000)
    add(16000)
    return candidates


def _candidate_channels(device_info: Any) -> list[int]:
    max_input_channels = int(device_info["max_input_channels"])
    candidates: list[int] = []
    for channels in (1, 2):
        if 0 < channels <= max_input_channels and channels not in candidates:
            candidates.append(channels)
    return candidates
