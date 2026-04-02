#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

import grpc
import librosa
import numpy as np
import numpy.typing as npt
import sounddevice as sd

from swim.transports.grpc.generated import speech_pb2, speech_pb2_grpc

# WARNING: This code is just a test client for testing the multiclient realtime whisper server

PB2: Any = speech_pb2
PB2_GRPC: Any = speech_pb2_grpc
LIVE_PREVIEW_INTERIM_COLOR = "38;5;215"
ALL_UPDATES_SEPARATOR = "--------------------"

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
class ClientOptions:
    host: str
    port: int
    all_updates: bool
    simulate_filepath: str | None
    live_preview: bool
    sound_device_id: int | None


class TranscriptRenderer:
    def __init__(self, *, all_updates: bool, live_preview: bool):
        self.all_updates = all_updates
        self.live_preview = live_preview

    def render(self, responses: Any) -> None:
        if self.live_preview:
            self._render_live_preview(responses)
            return
        self._render_standard(responses)

    def _print_live_preview(self, confirmed_text: str, interim_text: str) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        sys.stdout.write(confirmed_text)
        if interim_text:
            sys.stdout.write(f"\033[{LIVE_PREVIEW_INTERIM_COLOR}m{interim_text}\033[0m")
        sys.stdout.flush()

    @staticmethod
    def _has_transcript(response: Any, field_name: str) -> bool:
        return response.HasField(field_name) and bool(getattr(response, field_name).text)

    @staticmethod
    def _format_update_line(prefix: str, transcript: Any) -> str:
        return (
            f"{prefix} {transcript.start_time_millis:04d} "
            f"{transcript.end_time_millis:04d} {transcript.text}"
        )

    @staticmethod
    def _format_standard_line(transcript: Any) -> str:
        return (
            f"{transcript.start_time_millis:04d} {transcript.end_time_millis:04d} {transcript.text}"
        )

    def _render_standard(self, responses: Any) -> None:
        for response in responses:
            has_confirmed = self._has_transcript(response, "confirmed")
            has_interim = self._has_transcript(response, "interim")

            if not self.all_updates:
                if has_confirmed:
                    print(self._format_standard_line(response.confirmed))
                continue

            if not (has_confirmed or has_interim):
                continue

            print(ALL_UPDATES_SEPARATOR)
            if has_confirmed:
                print(self._format_update_line("CONF", response.confirmed))
            if has_interim:
                print(self._format_update_line("INT ", response.interim))
            print()

    def _render_live_preview(self, responses: Any) -> None:
        confirmed_text = ""
        last_confirmed = ""
        last_interim = ""

        for response in responses:
            updated = False

            if self._has_transcript(response, "confirmed"):
                confirmed = response.confirmed
                if confirmed.text != last_confirmed:
                    confirmed_text += confirmed.text
                    last_confirmed = confirmed.text
                    last_interim = ""
                    updated = True

            if self._has_transcript(response, "interim"):
                interim = response.interim
                if interim.text != last_interim:
                    last_interim = interim.text
                    updated = True

            if updated:
                self._print_live_preview(confirmed_text, last_interim)


class TranscriptionClient:
    def __init__(self, options: ClientOptions, audio_config: AudioConfig):
        self.options = options
        self.audio_config = audio_config
        self.renderer = TranscriptRenderer(
            all_updates=options.all_updates,
            live_preview=options.live_preview,
        )

    def _build_config_request(self) -> Any:
        return PB2.StreamingRecognizeRequest(
            config=PB2.StreamingConfig(
                chunk_duration_millis=self.audio_config.chunk_duration_millis,
            )
        )

    @staticmethod
    def _normalize_samples(samples: npt.ArrayLike) -> AudioSamples:
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    def _build_audio_request(self, samples: npt.ArrayLike) -> Any:
        normalized_samples = self._normalize_samples(samples)
        return PB2.StreamingRecognizeRequest(
            audio_chunk=PB2.AudioChunk(audio_bytes=normalized_samples.tobytes())
        )

    def _input_device_id(self) -> Any:
        if self.options.sound_device_id is not None:
            return self.options.sound_device_id

        default_device = sd.default.device
        if isinstance(default_device, tuple):
            return default_device[0]
        return default_device[0]

    def _stream_simulated_audio(self) -> Any:
        assert self.options.simulate_filepath is not None

        audio_data, _ = librosa.load(
            self.options.simulate_filepath,
            sr=self.audio_config.sample_rate,
            mono=True,
            dtype=np.float32,
        )
        normalized_audio = self._normalize_samples(audio_data)
        print(f"Loaded {len(normalized_audio)} samples from file {self.options.simulate_filepath}")

        yield self._build_config_request()
        for start in range(0, len(normalized_audio), self.audio_config.chunk_size()):
            chunk_samples = normalized_audio[start : start + self.audio_config.chunk_size()]
            yield self._build_audio_request(chunk_samples)
            time.sleep(self.audio_config.effective_chunk_duration_seconds)

    def _stream_live_audio(self) -> Any:
        print("Capturing real-time audio from the microphone...")

        device_id = self._input_device_id()
        device_info = sd.query_devices(device_id)
        native_sample_rate = int(device_info["default_samplerate"])
        print(
            "Mic sample rate: "
            f"{native_sample_rate} Hz, preferred capture sample rate: {self.audio_config.sample_rate} Hz"
        )

        try:
            yield from self._stream_live_audio_with_rate(
                device_id=device_id,
                capture_sample_rate=self.audio_config.sample_rate,
                source_sample_rate=self.audio_config.sample_rate,
            )
            return
        except sd.PortAudioError as exc:
            if native_sample_rate == self.audio_config.sample_rate:
                raise
            print(f"Falling back to native microphone sample rate {native_sample_rate} Hz: {exc}")

        yield from self._stream_live_audio_with_rate(
            device_id=device_id,
            capture_sample_rate=native_sample_rate,
            source_sample_rate=native_sample_rate,
        )

    def _stream_live_audio_with_rate(
        self,
        *,
        device_id: Any,
        capture_sample_rate: int,
        source_sample_rate: int,
    ) -> Any:
        callback_blocksize = max(1, int(capture_sample_rate * 0.05))
        chunk_samples = max(1, self.audio_config.chunk_size(capture_sample_rate))
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

        with sd.InputStream(
            device=device_id,
            channels=self.audio_config.channels,
            samplerate=capture_sample_rate,
            blocksize=callback_blocksize,
            dtype=self.audio_config.input_dtype,
            latency="low",
            callback=audio_callback,
        ):
            yield self._build_config_request()

            buffered_chunks: list[AudioSamples] = []
            buffered_samples = 0

            while True:
                chunk = audio_queue.get()
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

                if source_sample_rate != self.audio_config.sample_rate:
                    chunk_to_send = self._normalize_samples(
                        librosa.resample(
                            chunk_to_send,
                            orig_sr=source_sample_rate,
                            target_sr=self.audio_config.sample_rate,
                        )
                    )

                yield self._build_audio_request(chunk_to_send)

    def stream_requests(self) -> Any:
        print("Started connection")
        if self.options.simulate_filepath:
            return self._stream_simulated_audio()
        return self._stream_live_audio()

    def run(self) -> None:
        channel = grpc.insecure_channel(f"{self.options.host}:{self.options.port}")
        stub = PB2_GRPC.SpeechToTextStub(channel)
        requests = self.stream_requests()
        responses = stub.StreamingRecognize(requests)

        try:
            self.renderer.render(responses)
        except grpc.RpcError as exc:
            print("gRPC Error:", exc)
        finally:
            channel.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gRPC transcription client")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
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
        all_updates=args.all_updates,
        simulate_filepath=args.simulate,
        live_preview=args.live_preview,
        sound_device_id=args.sound_device_id,
    )
    TranscriptionClient(options, audio_config).run()


if __name__ == "__main__":
    main()
