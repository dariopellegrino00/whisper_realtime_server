import asyncio
import logging
from abc import ABC, abstractmethod

import grpc
import numpy as np
from grpc import StatusCode

from swim.runtime import ParallelOnlineASRProcessor
from swim.transports.grpc.generated import speech_pb2 as whisp_speech
from swim.transports.grpc.stream_utils import ProcessorManager, TranscriptionManager

BYTES_PER_SAMPLE = 4


class StreamSession(ABC):
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

    @abstractmethod
    async def request_enqueuer(self, request_iterator, context):
        pass

    @abstractmethod
    async def manage_first_message(self, first_request, context):
        pass

    @abstractmethod
    def create_response(self) -> list:
        pass

    @abstractmethod
    def final_response(self) -> list:
        pass


class SpeechStreamSession(StreamSession):
    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        super().__init__(processor_manager, server_logger, logger)
        self.transcription_managers = {
            "confirmed": TranscriptionManager(),
            "interim": TranscriptionManager(),
        }

    async def _parse_audio_request(self, request, context):
        if request.WhichOneof("payload") != "audio_chunk":
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                "Only audio_chunk messages are allowed after the initial config",
            )
        audio_bytes = request.audio_chunk.audio_bytes
        if self.max_chunk_bytes is not None and len(audio_bytes) > self.max_chunk_bytes:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                "Audio chunk exceeds the configured chunk_duration_ms",
            )
        return np.frombuffer(audio_bytes, dtype=np.float32)

    async def enqueue_audio_request(self, request, context):
        audio_samples = await self._parse_audio_request(request, context)
        await self.processor_manager.audio_queue.put(audio_samples)

    async def consume_initial_audio_request(self, request, context):
        audio_samples = await self._parse_audio_request(request, context)
        await self.processor_manager.insert_audio(audio_samples)

    async def request_enqueuer(self, request_iterator, context):
        try:
            async for request in request_iterator:
                await self.enqueue_audio_request(request, context)
        except asyncio.CancelledError:
            raise
        except grpc.RpcError as exc:
            if exc.code() == StatusCode.CANCELLED:
                self.logger.info("Client disconnected from %s", self.id)
            raise
        except Exception:
            self.logger.exception("Request enqueuer failed for %s", self.id)
            raise
        else:
            self.logger.info("Client closed request stream for %s", self.id)
        finally:
            self.processor_manager.mark_stream_closed()

    async def manage_first_message(self, first_request, context):
        if first_request.WhichOneof("payload") != "config":
            await context.abort(
                StatusCode.INVALID_ARGUMENT, "The first streaming message must be a config"
            )

        chunk_duration_millis = first_request.config.chunk_duration_millis
        if chunk_duration_millis <= 0 or chunk_duration_millis > self.max_chunk_duration_millis:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                f"chunk_duration_millis must be > 0 and <= {self.max_chunk_duration_millis}",
            )

        self.chunk_duration_millis = chunk_duration_millis
        self.max_chunk_bytes = int(
            ParallelOnlineASRProcessor.SAMPLING_RATE
            * (chunk_duration_millis / 1000.0)
            * BYTES_PER_SAMPLE
        )
        self.processor_manager.processor.chunk_duration_seconds = chunk_duration_millis / 1000.0
        self.logger.debug(
            "Accepted stream config: chunk_duration_millis=%s max_chunk_bytes=%s",
            self.chunk_duration_millis,
            self.max_chunk_bytes,
        )

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
                self._create_response(
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
        return [self._create_response(confirmed_fmt)] if has_confirmed else []

    @staticmethod
    def _create_transcript(start, end, text):
        return whisp_speech.Transcript(
            start_time_millis=start,
            end_time_millis=end,
            text=text,
        )

    def _create_response(self, confirmed=None, interim=None):
        fields = {}
        if confirmed is not None:
            fields["confirmed"] = self._create_transcript(*confirmed)
        if interim is not None:
            fields["interim"] = self._create_transcript(*interim)
        return whisp_speech.StreamingRecognizeResponse(**fields)
