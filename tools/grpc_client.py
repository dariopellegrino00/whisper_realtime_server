#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any

import grpc

from swim.transports.audio_encoding import (
    PCM_S16_LE,
    SUPPORTED_AUDIO_ENCODINGS,
    encode_audio_samples,
)
from swim.transports.grpc.audio_encoding import proto_audio_encoding_for
from swim.transports.grpc.generated import speech_pb2, speech_pb2_grpc
from tools._audio_client_common import (
    AudioConfig,
    LiveAudioChunkProducer,
    LiveInputSettings,
    add_common_client_args,
    load_simulated_audio,
    normalize_samples,
    print_live_input_banner,
    resolve_live_input_settings,
)

# WARNING: This code is just a test client for testing the multiclient realtime whisper server

PB2: Any = speech_pb2
PB2_GRPC: Any = speech_pb2_grpc
LIVE_PREVIEW_INTERIM_COLOR = "38;5;215"
ALL_UPDATES_SEPARATOR = "--------------------"


@dataclass(frozen=True, slots=True)
class ClientOptions:
    host: str
    port: int
    audio_encoding: str
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
                encoding=proto_audio_encoding_for(self.options.audio_encoding),
            )
        )

    def _build_audio_request(self, samples) -> Any:
        audio_bytes = encode_audio_samples(normalize_samples(samples), self.options.audio_encoding)
        return PB2.StreamingRecognizeRequest(audio_chunk=PB2.AudioChunk(audio_bytes=audio_bytes))

    def _stream_simulated_audio(self) -> Any:
        assert self.options.simulate_filepath is not None

        normalized_audio = load_simulated_audio(
            self.options.simulate_filepath,
            self.audio_config.sample_rate,
        )
        print(f"Loaded {len(normalized_audio)} samples from file {self.options.simulate_filepath}")

        yield self._build_config_request()
        for start in range(0, len(normalized_audio), self.audio_config.chunk_size()):
            chunk_samples = normalized_audio[start : start + self.audio_config.chunk_size()]
            yield self._build_audio_request(chunk_samples)
            time.sleep(self.audio_config.effective_chunk_duration_seconds)

    def _stream_live_audio_with_rate(
        self,
        *,
        live_input: LiveInputSettings,
    ) -> Any:
        with LiveAudioChunkProducer(self.audio_config, live_input) as producer:
            yield self._build_config_request()
            while True:
                yield self._build_audio_request(producer.read_chunk())

    def stream_requests(self, live_input: LiveInputSettings | None = None) -> Any:
        print("Started connection")
        if self.options.simulate_filepath:
            return self._stream_simulated_audio()
        if live_input is None:
            raise RuntimeError("Live input settings must be resolved before starting live capture.")

        print_live_input_banner(live_input, self.audio_config)

        return self._stream_live_audio_with_rate(live_input=live_input)

    def run(self) -> None:
        live_input = None
        if self.options.simulate_filepath is None:
            try:
                live_input = resolve_live_input_settings(
                    self.audio_config,
                    sound_device_id=self.options.sound_device_id,
                )
            except RuntimeError as exc:
                print(f"Audio input error: {exc}", file=sys.stderr)
                return

        channel = grpc.insecure_channel(f"{self.options.host}:{self.options.port}")
        stub = PB2_GRPC.SpeechToTextStub(channel)
        requests = self.stream_requests(live_input=live_input)
        responses = stub.StreamingRecognize(requests)

        try:
            self.renderer.render(responses)
        except RuntimeError as exc:
            print(f"Audio input error: {exc}", file=sys.stderr)
        except grpc.RpcError as exc:
            print("gRPC Error:", exc)
        finally:
            channel.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gRPC transcription client")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument(
        "--audio-encoding",
        type=str,
        choices=SUPPORTED_AUDIO_ENCODINGS,
        default=PCM_S16_LE,
        help="Wire encoding for outbound gRPC audio chunks",
    )
    add_common_client_args(parser)
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
        audio_encoding=args.audio_encoding,
        all_updates=args.all_updates,
        simulate_filepath=args.simulate,
        live_preview=args.live_preview,
        sound_device_id=args.sound_device_id,
    )
    TranscriptionClient(options, audio_config).run()


if __name__ == "__main__":
    main()
