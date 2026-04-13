#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import sounddevice as sd
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from swim.transports.websocket.messages import (
    CHANNELS,
    PCM_FLOAT32_LE,
    SAMPLE_RATE_HZ,
    WEBSOCKET_TRANSCRIBE_PATH,
)

# WARNING: This code is just a test client for testing the multiclient realtime whisper server

LIVE_PREVIEW_INTERIM_COLOR = "38;5;215"
ALL_UPDATES_SEPARATOR = "--------------------"

AudioSamples = npt.NDArray[np.float32]


@dataclass(slots=True)
class AudioConfig:
    chunk_duration_seconds: float
    sample_rate: int = SAMPLE_RATE_HZ
    channels: int = CHANNELS
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
class ClientOptions:
    host: str
    port: int
    path: str
    all_updates: bool
    simulate_filepath: str | None
    live_preview: bool
    sound_device_id: int | None


@dataclass(frozen=True, slots=True)
class LiveInputSettings:
    device_id: int
    device_name: str
    capture_sample_rate: int
    source_sample_rate: int
    channels: int


class TranscriptRenderer:
    def __init__(self, *, all_updates: bool, live_preview: bool):
        self.all_updates = all_updates
        self.live_preview = live_preview
        self.confirmed_text = ""
        self.last_confirmed = ""
        self.last_interim = ""

    def render_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "transcript":
            return
        if self.live_preview:
            self._render_live_preview(message)
            return
        self._render_standard(message)

    def _print_live_preview(self, confirmed_text: str, interim_text: str) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        sys.stdout.write(confirmed_text)
        if interim_text:
            sys.stdout.write(f"\033[{LIVE_PREVIEW_INTERIM_COLOR}m{interim_text}\033[0m")
        sys.stdout.flush()

    @staticmethod
    def _get_transcript(message: dict[str, Any], field_name: str) -> dict[str, Any] | None:
        transcript = message.get(field_name)
        if not isinstance(transcript, dict):
            return None
        if not transcript.get("text"):
            return None
        return transcript

    @staticmethod
    def _format_update_line(prefix: str, transcript: dict[str, Any]) -> str:
        return (
            f"{prefix} {int(transcript['start_time_millis']):04d} "
            f"{int(transcript['end_time_millis']):04d} {transcript['text']}"
        )

    @staticmethod
    def _format_standard_line(transcript: dict[str, Any]) -> str:
        return (
            f"{int(transcript['start_time_millis']):04d} "
            f"{int(transcript['end_time_millis']):04d} {transcript['text']}"
        )

    def _render_standard(self, message: dict[str, Any]) -> None:
        confirmed = self._get_transcript(message, "confirmed")
        interim = self._get_transcript(message, "interim")

        if not self.all_updates:
            if confirmed is not None:
                print(self._format_standard_line(confirmed))
            return

        if confirmed is None and interim is None:
            return

        print(ALL_UPDATES_SEPARATOR)
        if confirmed is not None:
            print(self._format_update_line("CONF", confirmed))
        if interim is not None:
            print(self._format_update_line("INT ", interim))
        print()

    def _render_live_preview(self, message: dict[str, Any]) -> None:
        confirmed = self._get_transcript(message, "confirmed")
        interim = self._get_transcript(message, "interim")
        updated = False

        if confirmed is not None and confirmed["text"] != self.last_confirmed:
            self.confirmed_text += str(confirmed["text"])
            self.last_confirmed = str(confirmed["text"])
            self.last_interim = ""
            updated = True

        if interim is not None and interim["text"] != self.last_interim:
            self.last_interim = str(interim["text"])
            updated = True

        if updated:
            self._print_live_preview(self.confirmed_text, self.last_interim)


class WebsocketTranscriptionClient:
    def __init__(self, options: ClientOptions, audio_config: AudioConfig):
        self.options = options
        self.audio_config = audio_config
        self.renderer = TranscriptRenderer(
            all_updates=options.all_updates,
            live_preview=options.live_preview,
        )

    def _start_message(self) -> str:
        return json.dumps(
            {
                "type": "start",
                "chunk_duration_millis": self.audio_config.chunk_duration_millis,
                "audio_format": {
                    "encoding": PCM_FLOAT32_LE,
                    "sample_rate_hz": self.audio_config.sample_rate,
                    "channels": self.audio_config.channels,
                },
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _normalize_samples(samples: npt.ArrayLike) -> AudioSamples:
        normalized = np.asarray(samples, dtype=np.float32)
        if normalized.ndim == 2:
            normalized = normalized.mean(axis=1)
        return normalized.reshape(-1)

    @staticmethod
    def _default_input_device_id() -> int | None:
        default_device = sd.default.device
        if hasattr(default_device, "__getitem__"):
            default_input = default_device[0]
        else:
            default_input = default_device
        if default_input is None or default_input < 0:
            return None
        return int(default_input)

    def _candidate_device_ids(self) -> list[int]:
        if self.options.sound_device_id is not None:
            return [self.options.sound_device_id]

        candidates: list[int] = []
        default_device_id = self._default_input_device_id()
        if default_device_id is not None:
            candidates.append(default_device_id)

        for device_id, device_info in enumerate(sd.query_devices()):
            if int(device_info["max_input_channels"]) > 0 and device_id not in candidates:
                candidates.append(device_id)
        return candidates

    def _candidate_sample_rates(self, device_info: Any) -> list[int]:
        candidates: list[int] = []

        def add(rate: int) -> None:
            if rate > 0 and rate not in candidates:
                candidates.append(rate)

        add(self.audio_config.sample_rate)
        add(int(round(float(device_info["default_samplerate"]))))
        add(48000)
        add(44100)
        add(32000)
        add(16000)
        return candidates

    @staticmethod
    def _candidate_channels(device_info: Any) -> list[int]:
        max_input_channels = int(device_info["max_input_channels"])
        candidates: list[int] = []
        for channels in (1, 2):
            if 0 < channels <= max_input_channels and channels not in candidates:
                candidates.append(channels)
        return candidates

    def _resolve_live_input_settings(self) -> LiveInputSettings:
        attempted: list[str] = []

        for device_id in self._candidate_device_ids():
            try:
                device_info = sd.query_devices(device_id)
            except sd.PortAudioError as exc:
                attempted.append(f"{device_id}: {exc}")
                continue

            if int(device_info["max_input_channels"]) <= 0:
                continue

            for channels in self._candidate_channels(device_info):
                for sample_rate in self._candidate_sample_rates(device_info):
                    try:
                        sd.check_input_settings(
                            device=device_id,
                            channels=channels,
                            dtype=self.audio_config.input_dtype,
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

        if self.options.sound_device_id is not None:
            prefix = f"Could not open input device {self.options.sound_device_id}."
        else:
            prefix = "Could not find a working microphone input device."

        details = "\n".join(attempted[:5])
        if len(attempted) > 5:
            details += f"\n... and {len(attempted) - 5} more failed combinations."
        raise RuntimeError(f"{prefix}\n{details}" if details else prefix)

    async def _send_simulated_audio(self, websocket) -> None:
        assert self.options.simulate_filepath is not None

        audio_data, _ = librosa.load(
            self.options.simulate_filepath,
            sr=self.audio_config.sample_rate,
            mono=True,
            dtype=np.float32,
        )
        normalized_audio = self._normalize_samples(audio_data)
        print(f"Loaded {len(normalized_audio)} samples from file {self.options.simulate_filepath}")

        await websocket.send(self._start_message(), text=True)
        for start in range(0, len(normalized_audio), self.audio_config.chunk_size()):
            chunk_samples = normalized_audio[start : start + self.audio_config.chunk_size()]
            await websocket.send(chunk_samples.tobytes())
            await asyncio.sleep(self.audio_config.effective_chunk_duration_seconds)
        await websocket.send(json.dumps({"type": "finish"}), text=True)

    async def _queue_get(self, audio_queue: Queue[AudioSamples]) -> AudioSamples:
        while True:
            try:
                return audio_queue.get(timeout=0.1)
            except Empty:
                await asyncio.sleep(0)

    async def _send_live_audio(self, websocket, *, live_input: LiveInputSettings) -> None:
        chunk_samples = max(1, self.audio_config.chunk_size(live_input.capture_sample_rate))
        audio_queue: Queue[AudioSamples] = Queue(maxsize=100)

        def audio_callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status:
                print(f"Audio callback status: {status}", file=sys.stderr)

            chunk = self._normalize_samples(indata)
            try:
                audio_queue.put_nowait(chunk)
            except Full:
                try:
                    audio_queue.get_nowait()
                except Empty:
                    pass
                try:
                    audio_queue.put_nowait(chunk)
                except Full:
                    pass

        try:
            with sd.InputStream(
                device=live_input.device_id,
                channels=live_input.channels,
                samplerate=live_input.capture_sample_rate,
                blocksize=0,
                dtype=self.audio_config.input_dtype,
                latency="low",
                callback=audio_callback,
            ):
                await websocket.send(self._start_message(), text=True)

                buffered_chunks: list[AudioSamples] = []
                buffered_samples = 0

                while True:
                    chunk = await self._queue_get(audio_queue)
                    buffered_chunks.append(chunk)
                    buffered_samples += len(chunk)

                    if buffered_samples < chunk_samples:
                        continue

                    chunk_native = self._normalize_samples(np.concatenate(buffered_chunks))
                    if len(chunk_native) > chunk_samples:
                        chunk_to_send = chunk_native[:chunk_samples]
                        remainder = chunk_native[chunk_samples:]
                        buffered_chunks = [remainder] if len(remainder) else []
                        buffered_samples = len(remainder)
                    else:
                        chunk_to_send = chunk_native
                        buffered_chunks = []
                        buffered_samples = 0

                    if live_input.source_sample_rate != self.audio_config.sample_rate:
                        chunk_to_send = self._normalize_samples(
                            librosa.resample(
                                chunk_to_send,
                                orig_sr=live_input.source_sample_rate,
                                target_sr=self.audio_config.sample_rate,
                            )
                        )

                    await websocket.send(chunk_to_send.tobytes())
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f"Could not start microphone input on device {live_input.device_id} "
                f"({live_input.device_name}): {exc}"
            ) from exc

    async def _send_audio_stream(self, websocket, live_input: LiveInputSettings | None) -> None:
        if self.options.simulate_filepath:
            await self._send_simulated_audio(websocket)
            return

        if live_input is None:
            raise RuntimeError("Live input settings must be resolved before starting live capture.")

        print("Capturing real-time audio from the microphone...")
        print(
            f"Using input device {live_input.device_id}: {live_input.device_name} "
            f"({live_input.channels} channel(s), {live_input.capture_sample_rate} Hz)"
        )
        if live_input.capture_sample_rate != self.audio_config.sample_rate:
            print(
                f"Resampling microphone audio from {live_input.capture_sample_rate} Hz "
                f"to {self.audio_config.sample_rate} Hz"
            )

        await self._send_live_audio(websocket, live_input=live_input)

    async def _receive_events(self, websocket) -> None:
        while True:
            message = await websocket.recv()
            if not isinstance(message, str):
                raise RuntimeError("Received an unexpected binary frame from the websocket server")

            payload = json.loads(message)
            event_type = payload.get("type")
            if event_type == "transcript":
                self.renderer.render_message(payload)
                continue
            if event_type == "completed":
                return
            if event_type == "error":
                print(
                    f"Websocket server error [{payload.get('code', 'unknown')}]: "
                    f"{payload.get('message', '')}",
                    file=sys.stderr,
                )
                return

    async def run(self) -> None:
        live_input = None
        if self.options.simulate_filepath is None:
            try:
                live_input = self._resolve_live_input_settings()
            except RuntimeError as exc:
                print(f"Audio input error: {exc}", file=sys.stderr)
                return

        uri = f"ws://{self.options.host}:{self.options.port}{self.options.path}"
        print("Started connection")

        try:
            async with connect(uri, compression=None) as websocket:
                sender_task = asyncio.create_task(self._send_audio_stream(websocket, live_input))
                try:
                    await self._receive_events(websocket)
                finally:
                    if not sender_task.done():
                        sender_task.cancel()
                    try:
                        await sender_task
                    except asyncio.CancelledError:
                        pass
        except RuntimeError as exc:
            print(f"Audio input error: {exc}", file=sys.stderr)
        except ConnectionClosed as exc:
            print(f"Websocket closed: code={exc.code} reason={exc.reason}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Websocket transcription client")
    parser.add_argument("--host", type=str, default="localhost", help="Websocket server address")
    parser.add_argument("--port", type=int, default=8000, help="Websocket server port")
    parser.add_argument(
        "--path",
        type=str,
        default=WEBSOCKET_TRANSCRIBE_PATH,
        help="Websocket endpoint path",
    )
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_duration <= 0:
        parser.error("--chunk-duration must be > 0 seconds")

    audio_config = AudioConfig(chunk_duration_seconds=args.chunk_duration)
    if audio_config.chunk_duration_millis <= 0:
        parser.error("--chunk-duration must round to at least 1 millisecond")

    options = ClientOptions(
        host=args.host,
        port=args.port,
        path=args.path,
        all_updates=args.all_updates,
        simulate_filepath=args.simulate,
        live_preview=args.live_preview,
        sound_device_id=args.sound_device_id,
    )
    asyncio.run(WebsocketTranscriptionClient(options, audio_config).run())


if __name__ == "__main__":
    main()
