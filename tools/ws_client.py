#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from swim.transports.audio_encoding import (
    PCM_S16_LE,
    SUPPORTED_AUDIO_ENCODINGS,
    encode_audio_samples,
)
from swim.transports.websocket.messages import WEBSOCKET_TRANSCRIBE_PATH
from tools._audio_client_common import (
    AudioConfig,
    LiveAudioChunkProducer,
    LiveInputSettings,
    add_common_client_args,
    load_simulated_audio,
    print_live_input_banner,
    resolve_live_input_settings,
)

# WARNING: This code is just a test client for testing the multiclient realtime whisper server

LIVE_PREVIEW_INTERIM_COLOR = "38;5;215"
ALL_UPDATES_SEPARATOR = "--------------------"


@dataclass(frozen=True, slots=True)
class ClientOptions:
    host: str
    port: int
    path: str
    audio_encoding: str
    all_updates: bool
    simulate_filepath: str | None
    live_preview: bool
    sound_device_id: int | None


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
                    "encoding": self.options.audio_encoding,
                    "sample_rate_hz": self.audio_config.sample_rate,
                    "channels": self.audio_config.channels,
                },
            },
            separators=(",", ":"),
        )

    def _encode_audio_chunk(self, samples) -> bytes:
        return encode_audio_samples(samples, self.options.audio_encoding)

    async def _send_simulated_audio(self, websocket, normalized_audio) -> None:
        await websocket.send(self._start_message(), text=True)
        for start in range(0, len(normalized_audio), self.audio_config.chunk_size()):
            chunk_samples = normalized_audio[start : start + self.audio_config.chunk_size()]
            await websocket.send(self._encode_audio_chunk(chunk_samples))
            await asyncio.sleep(self.audio_config.effective_chunk_duration_seconds)
        await websocket.send(json.dumps({"type": "finish"}), text=True)

    async def _send_live_audio(self, websocket, *, live_input: LiveInputSettings) -> None:
        with LiveAudioChunkProducer(self.audio_config, live_input) as producer:
            await websocket.send(self._start_message(), text=True)
            while True:
                chunk = await asyncio.to_thread(producer.read_chunk)
                await websocket.send(self._encode_audio_chunk(chunk))

    async def _send_audio_stream(
        self,
        websocket,
        live_input: LiveInputSettings | None,
        simulated_audio=None,
    ) -> None:
        if self.options.simulate_filepath:
            if simulated_audio is None:
                raise RuntimeError("Simulated audio must be loaded before opening the websocket.")
            await self._send_simulated_audio(websocket, simulated_audio)
            return

        if live_input is None:
            raise RuntimeError("Live input settings must be resolved before starting live capture.")

        print_live_input_banner(live_input, self.audio_config)

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
        simulated_audio = None
        if self.options.simulate_filepath is None:
            try:
                live_input = resolve_live_input_settings(
                    self.audio_config,
                    sound_device_id=self.options.sound_device_id,
                )
            except RuntimeError as exc:
                print(f"Audio input error: {exc}", file=sys.stderr)
                return
        else:
            simulated_audio = load_simulated_audio(
                self.options.simulate_filepath,
                self.audio_config.sample_rate,
            )

        uri = f"ws://{self.options.host}:{self.options.port}{self.options.path}"
        print("Started connection")
        if simulated_audio is not None:
            print(
                f"Loaded {len(simulated_audio)} samples from file {self.options.simulate_filepath}"
            )

        try:
            async with connect(uri, compression=None) as websocket:
                sender_task = asyncio.create_task(
                    self._send_audio_stream(websocket, live_input, simulated_audio)
                )
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
        "--audio-encoding",
        type=str,
        choices=SUPPORTED_AUDIO_ENCODINGS,
        default=PCM_S16_LE,
        help="Wire encoding for outbound websocket audio chunks",
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
        path=args.path,
        audio_encoding=args.audio_encoding,
        all_updates=args.all_updates,
        simulate_filepath=args.simulate,
        live_preview=args.live_preview,
        sound_device_id=args.sound_device_id,
    )
    asyncio.run(WebsocketTranscriptionClient(options, audio_config).run())


if __name__ == "__main__":
    main()
