import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import grpc
import numpy as np
from grpc import StatusCode

from src.generated import speech_pb2 as whisp_speech
from src.server.stream_utils import *

BYTES_PER_SAMPLE = 4


class StreamSession(ABC):
    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        self.processor_manager = processor_manager
        self.id = processor_manager.id
        self.server_logger = server_logger if server_logger is not None else logging.getLogger(__name__)
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
    def create_response(self) -> List:
        pass

    @abstractmethod
    def final_response(self) -> List:
        pass

class WhispStreamSession(StreamSession):
    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        super().__init__(processor_manager, server_logger, logger)
        self.transcription_managers = self.create_transcription_managers()

    def create_transcription_managers(self) -> Dict[str, TranscriptionManager]:
        return {"confirmed": TranscriptionManager()}

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
                self.server_logger.info(f"Client disconnected from {self.id}")
            raise
        except Exception as exc:
            self.processor_manager.logger.error(
                f"Exception in request_enqueuer {self.processor_manager.id}: {exc}"
            )
            raise
        else:
            self.server_logger.info(f"Client closed request stream for {self.id}")
        finally:
            self.processor_manager.mark_stream_closed()

    async def manage_first_message(self, first_request, context):
        if first_request.WhichOneof("payload") != "config":
            await context.abort(StatusCode.INVALID_ARGUMENT, "The first streaming message must be a config")

        chunk_duration_millis = first_request.config.chunk_duration_millis
        if chunk_duration_millis <= 0 or chunk_duration_millis > self.max_chunk_duration_millis:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                f"chunk_duration_millis must be > 0 and <= {self.max_chunk_duration_millis}",
            )

        self.chunk_duration_millis = chunk_duration_millis
        self.max_chunk_bytes = int(
            ParallelOnlineASRProcessor.SAMPLING_RATE * (chunk_duration_millis / 1000.0) * BYTES_PER_SAMPLE
        )
        self.processor_manager.processor.chunk_duration_seconds = chunk_duration_millis / 1000.0


class StandardWhispStreamSession(WhispStreamSession):
    def create_response(self):
        results = self.processor_manager.processor.results
        exist, fmt = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt)] if exist else []

    def final_response(self):
        results = self.processor_manager.processor.finish()
        exist, fmt = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt)] if exist else []

    def _create_response(self, start, end, text):
        return whisp_speech.Transcript(
            start_time_millis=start,
            end_time_millis=end,
            text=text,
        )


class HypothesisWhispStreamSession(WhispStreamSession):
    def create_transcription_managers(self):
        return {"confirmed": TranscriptionManager(), "hypothesis": TranscriptionManager()}

    def create_response(self):
        results = self.processor_manager.processor.results
        hypothesis = self.processor_manager.processor.hypothesis
        exist1, fmt_t = self.transcription_managers["confirmed"].format_transcript(results)
        exist2, fmt_h = self.transcription_managers["hypothesis"].format_transcript(
            hypothesis, use_last_end=False
        )
        return [self._create_response(*fmt_t, *fmt_h)] if exist1 or exist2 else []

    def final_response(self):
        results = self.processor_manager.processor.finish()
        exist, fmt_t = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt_t, 0, 0, "")] if exist else []

    def _create_response(self, start_t, end_t, text, start_h, end_h, hypothesis):
        return whisp_speech.TranscriptWithHypothesis(
            confirmed=whisp_speech.Transcript(
                start_time_millis=start_t,
                end_time_millis=end_t,
                text=text,
            ),
            hypothesis=whisp_speech.Transcript(
                start_time_millis=start_h,
                end_time_millis=end_h,
                text=hypothesis,
            ),
        )
