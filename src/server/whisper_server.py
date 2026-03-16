#!/usr/bin/env python3
import argparse
import asyncio
import itertools
import logging
import signal
import sys
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Type

import grpc
from grpc import StatusCode, aio

from src.generated import speech_pb2_grpc
from src.parallel_whisper_online import ParallelRealtimeASR
from src.server.stream_session import (
    HypothesisWhispStreamSession,
    StandardWhispStreamSession,
    StreamSession,
)
from src.server.stream_utils import ProcessorManager, setup_logging


class BaseSpeechToTextServicer(ABC):
    _service_id = itertools.count()
    log_every_processor = False
    _logger_level = logging.DEBUG

    def __init__(self, shared_asr, main_server_logger, **kwargs):
        self._shared_asr = shared_asr
        self._main_server_logger = main_server_logger
        self._kwargs = kwargs

    @abstractmethod
    def StreamSessionType(self) -> Type[StreamSession]:
        pass

    def create_stream_session(self) -> StreamSession:
        id = self.get_unique_name()
        logger = self.log_setup(id)
        processor_manager = ProcessorManager(
            id,
            self._shared_asr,
            logger=logger,
            server_logger=self._main_server_logger,
            **self._kwargs,
        )
        return self.StreamSessionType()(
            processor_manager=processor_manager,
            server_logger=self._main_server_logger,
            logger=logger,
        )

    @classmethod
    def get_unique_name(cls):
        return f"Whisper-service-{next(cls._service_id)}"

    @classmethod
    def log_setup(cls, id):
        if cls.log_every_processor:
            return setup_logging(f"{id}", level=cls._logger_level)
        return logging.getLogger(__name__)

    @staticmethod
    def _first_audio_timeout_seconds(stream_session):
        chunk_duration_seconds = (stream_session.chunk_duration_millis or 1000) / 1000.0
        return max(chunk_duration_seconds * 2, 1.0)

    async def StreamingRecognize(self, request_iterator, context):
        stream_session = self.create_stream_session()
        id = stream_session.id

        self._main_server_logger.info(f"Started connection on {id}")

        try:
            first_request = await request_iterator.__anext__()
            await stream_session.manage_first_message(first_request, context)
            try:
                first_audio_request = await asyncio.wait_for(
                    request_iterator.__anext__(),
                    timeout=self._first_audio_timeout_seconds(stream_session),
                )
            except StopAsyncIteration:
                return
            await stream_session.consume_initial_audio_request(first_audio_request, context)
        except StopAsyncIteration:
            self._main_server_logger.info(f"Service {id} closed prematurely by client")
            return
        except asyncio.TimeoutError:
            await context.abort(
                StatusCode.DEADLINE_EXCEEDED,
                f"{id} did not send the first audio chunk within "
                f"{self._first_audio_timeout_seconds(stream_session):.3f}s",
            )
        except grpc.RpcError:
            raise
        except Exception as exc:
            self._main_server_logger.exception(f"Exception in {id}: {exc}")
            raise

        request_task = asyncio.create_task(stream_session.request_enqueuer(request_iterator, context))
        has_unsubmitted_audio = True

        try:
            async with stream_session.processor_manager.context():
                while not stream_session.processor_manager.is_finished():
                    if (
                        request_task.done()
                        and stream_session.processor_manager.audio_queue.empty()
                        and not has_unsubmitted_audio
                    ):
                        break

                    if stream_session.processor_manager.audio_queue.empty() and not has_unsubmitted_audio:
                        await asyncio.sleep(0.001)
                        continue

                    await stream_session.processor_manager.insert_audio()
                    await asyncio.sleep(0.001)
                    await self._shared_asr.set_processor_ready(id)
                    await stream_session.processor_manager.insert_audio()
                    await stream_session.processor_manager.get_transcription()
                    has_unsubmitted_audio = False
                    if stream_session.processor_manager.is_finished():
                        break

                    responses = stream_session.create_response()
                    for response in responses:
                        yield response
        finally:
            if not request_task.done():
                request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass

        if not stream_session.processor_manager.is_finished():
            final_responses = stream_session.final_response()
            for response in final_responses:
                yield response


class StandardWhispSpeechToTextServicer(
    BaseSpeechToTextServicer, speech_pb2_grpc.SpeechToTextServicer
):
    def StreamSessionType(self) -> Type[StreamSession]:
        return StandardWhispStreamSession


class HypothesisWhispSpeechToTextServicer(
    BaseSpeechToTextServicer, speech_pb2_grpc.SpeechToTextWithHypothesisServicer
):
    def StreamSessionType(self) -> Type[StreamSession]:
        return HypothesisWhispStreamSession


async def serve(args):
    log_level_name = args.log_level.upper()
    if log_level_name not in logging._nameToLevel:
        print("Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL", file=sys.stderr)
        sys.exit(1)
    log_level = logging._nameToLevel[log_level_name]

    server_logger = setup_logging("Layer-server", use_stdout=True, level=log_level)
    BaseSpeechToTextServicer._logger_level = log_level

    server_logger.info("Starting server...")

    if args.log_every_processor:
        BaseSpeechToTextServicer.log_every_processor = True
        server_logger.info(
            "Logging every processor in a separate file, be careful with the number of files generated, this should be used for debugging reasons only"
        )

    if args.qratio_threshold <= 0 or args.qratio_threshold > 100:
        server_logger.error("qratio threshold must be between 0 and 100")
        sys.exit(1)

    if args.dedup_threshold <= 0 or args.dedup_threshold > 100:
        server_logger.error("dedup threshold must be between 0 and 100")
        sys.exit(1)

    if args.fallback:
        server_logger.info("Fallback logic enabled")
        if args.fallback_threshold <= 0:
            server_logger.error("Fallback threshold must be greater than 0")
            sys.exit(1)

    if args.buffer_trimming_sec <= 0:
        server_logger.error("Buffer trimming must be greater than 0")
        sys.exit(1)

    if args.max_chunk_duration_seconds <= 0:
        server_logger.error("Max chunk duration must be greater than 0")
        sys.exit(1)

    server_logger.info(f"Using faster-whisper model {args.model}")
    shared_asr = ParallelRealtimeASR(
        modelsize=args.model,
        logger=setup_logging("asr", level=log_level),
        warmup_file=args.warmup_file,
    )
    server_logger.info("Model loaded")

    await shared_asr.start()
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        maximum_concurrent_rpcs=args.max_workers,
        options=[
            ("grpc.keepalive_time_ms", 1000),
            ("grpc.keepalive_timeout_ms", 1000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    processor_args = {
        "use_fallback": args.fallback,
        "fallback_threshold": args.fallback_threshold,
        "qratio_threshold": args.qratio_threshold,
        "dedup_threshold": args.dedup_threshold,
        "buffer_trimming_sec": args.buffer_trimming_sec,
        "max_chunk_duration_seconds": args.max_chunk_duration_seconds,
    }

    speech_pb2_grpc.add_SpeechToTextServicer_to_server(
        StandardWhispSpeechToTextServicer(shared_asr, server_logger, **processor_args),
        server,
    )
    speech_pb2_grpc.add_SpeechToTextWithHypothesisServicer_to_server(
        HypothesisWhispSpeechToTextServicer(shared_asr, server_logger, **processor_args),
        server,
    )

    for port in args.ports:
        server.add_insecure_port(f"[::]:{port}")

    await server.start()
    server_logger.info("Server started")

    shutdown_event = asyncio.Event()

    def _shutdown():
        server_logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _shutdown)
    loop.add_signal_handler(signal.SIGTERM, _shutdown)

    await shutdown_event.wait()

    server_logger.info("Stopping ASR...")
    await shared_asr.stop()
    server_logger.info("ASR stopped")

    server_logger.info("Stopping gRPC server...")
    await server.stop(0)
    server_logger.info("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Argument parser for the whisper-realtme-server")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable fallback logic when similarity local agreement fails for a mltitude of times",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=1,
        help="threshold t for fallback logic after t+1 similarity local agreement fails (ignored if --fallback is not set)",
    )
    parser.add_argument(
        "--qratio-threshold",
        type=float,
        default=95,
        help="Threshold for qratio to confirm and insert new words using the hypothesis buffer (between 0 and 100), lower values than 90 are not recommended",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=98,
        help="Threshold for qratio to deduplicate overlapping words between committed and new in the hypothesis buffer (between 0 and 100)",
    )
    parser.add_argument(
        "--buffer-trimming-sec",
        type=int,
        default=15,
        help="Buffer trimming is the threshold in seconds that triggers the service processor audio buffer to be trimmed. This is useful to avoid memory leaks and to keep the buffer size under control. Default value is 15 seconds",
    )
    parser.add_argument(
        "--max-chunk-duration-seconds",
        type=float,
        default=1.0,
        help="Maximum chunk duration accepted from a client session config",
    )

    parser.add_argument("--ports", type=int, nargs="+", default=[50051, 50052], help="Ports to run the server on")
    parser.add_argument("--max-workers", type=int, default=20, help="Max workers for the server")
    parser.add_argument("--log-every-processor", action="store_true", help="Log every processor in a separate file")

    parser.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo,turbo".split(","),
        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir",
    )
    parser.add_argument("--model-cache-dir", type=str, default=None, help="Directory for the whisper model caching")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory for a custom ct2 whisper model skipping if --model provided")
    parser.add_argument("--warmup-file", type=str, default="resources/sample1.wav", help="File to warm up the model and speed up the first request")

    parser.add_argument("--lan", type=str, default="en", help="Language for the whisper model to translate to (unused at the moment)")
    parser.add_argument("--vad", action="store_true", help="Use VAD for the model (unused at the moment)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Log level for the server and shared ASR logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    asyncio.run(serve(parser.parse_args()))
