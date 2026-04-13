import logging

import numpy as np
from websockets.exceptions import ConnectionClosed

from swim.runtime import ParallelOnlineASRProcessor
from swim.transports.grpc.stream_utils import ProcessorManager, TranscriptionManager
from swim.transports.websocket.messages import (
    PCM_F32_LE,
    PCM_S16_LE,
    WebsocketProtocolError,
    build_transcript_event,
    parse_finish_message,
    parse_start_message,
)

BYTES_PER_SAMPLE = {
    PCM_F32_LE: 4,
    PCM_S16_LE: 2,
}


class WebsocketStreamSession:
    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        self.processor_manager = processor_manager
        self.id = processor_manager.id
        self.server_logger = (
            server_logger if server_logger is not None else logging.getLogger(__name__)
        )
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.chunk_duration_millis = None
        self.max_chunk_bytes = None
        self.max_chunk_duration_millis = int(
            round(getattr(processor_manager, "max_chunk_duration_seconds", 1.0) * 1000)
        )
        self.transcription_managers = {
            "confirmed": TranscriptionManager(),
            "interim": TranscriptionManager(),
        }
        self.audio_encoding = PCM_F32_LE

    async def manage_start_message(self, first_message):
        if not isinstance(first_message, str):
            raise WebsocketProtocolError("The first websocket message must be a JSON start event")

        start_message = parse_start_message(
            first_message,
            max_chunk_duration_millis=self.max_chunk_duration_millis,
        )
        self.chunk_duration_millis = start_message.chunk_duration_millis
        self.audio_encoding = start_message.encoding
        self.max_chunk_bytes = int(
            ParallelOnlineASRProcessor.SAMPLING_RATE
            * (self.chunk_duration_millis / 1000.0)
            * BYTES_PER_SAMPLE[self.audio_encoding]
        )
        self.processor_manager.processor.chunk_duration_seconds = (
            self.chunk_duration_millis / 1000.0
        )
        self.logger.debug(
            "Accepted websocket start: chunk_duration_millis=%s encoding=%s max_chunk_bytes=%s",
            self.chunk_duration_millis,
            self.audio_encoding,
            self.max_chunk_bytes,
        )

    def _parse_audio_message(self, message):
        if isinstance(message, str):
            raise WebsocketProtocolError(
                "Only binary audio frames are allowed after the initial start event"
            )
        audio_bytes = bytes(message)
        if self.max_chunk_bytes is not None and len(audio_bytes) > self.max_chunk_bytes:
            raise WebsocketProtocolError("Audio frame exceeds the configured chunk_duration_millis")
        try:
            if self.audio_encoding == PCM_F32_LE:
                return np.frombuffer(audio_bytes, dtype=np.float32)
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            return samples / 32768.0
        except ValueError as exc:
            raise WebsocketProtocolError(
                "Audio frames must match the encoding declared in the start event"
            ) from exc

    async def enqueue_audio_message(self, message):
        audio_samples = self._parse_audio_message(message)
        await self.processor_manager.audio_queue.put(audio_samples)

    async def consume_initial_audio_message(self, message):
        audio_samples = self._parse_audio_message(message)
        await self.processor_manager.insert_audio(audio_samples)

    async def request_enqueuer(self, websocket):
        try:
            while True:
                message = await websocket.recv()
                if isinstance(message, str):
                    parse_finish_message(message)
                    self.logger.info("Client closed request stream for %s", self.id)
                    return
                await self.enqueue_audio_message(message)
        except ConnectionClosed:
            self.logger.info("Client disconnected from %s", self.id)
        except Exception:
            self.logger.exception("Request enqueuer failed for %s", self.id)
            raise
        finally:
            self.processor_manager.mark_stream_closed()

    def create_response(self):
        results = self.processor_manager.processor.results
        interim = self.processor_manager.processor.hypothesis
        has_confirmed, confirmed_fmt = self.transcription_managers["confirmed"].format_transcript(
            results
        )
        has_interim, interim_fmt = self.transcription_managers["interim"].format_transcript(
            interim, use_last_end=False
        )
        responses = []
        if has_confirmed or has_interim:
            responses.append(
                build_transcript_event(
                    confirmed_fmt if has_confirmed else None,
                    interim_fmt if has_interim else None,
                )
            )
        self.processor_manager.processor.mark_update_emitted()
        return responses

    def final_response(self):
        results = self.processor_manager.processor.finish()
        has_confirmed, confirmed_fmt = self.transcription_managers["confirmed"].format_transcript(
            results
        )
        return [build_transcript_event(confirmed_fmt)] if has_confirmed else []


__all__ = ["ProcessorManager", "TranscriptionManager", "WebsocketStreamSession"]
