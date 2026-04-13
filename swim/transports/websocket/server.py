#!/usr/bin/env python3
import argparse
import asyncio
import itertools
import logging
import signal
import sys

from websockets.asyncio.server import serve as websocket_serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from swim.runtime import ParallelOnlineASRProcessor, ParallelRealtimeASR, resolve_asr_backend
from swim.transports.grpc.stream_utils import (
    ProcessorManager,
    setup_application_logging,
    setup_stream_logger,
)
from swim.transports.websocket.messages import (
    WEBSOCKET_TRANSCRIBE_PATH,
    WebsocketProtocolError,
    build_completed_event,
    build_error_event,
    parse_finish_message,
)
from swim.transports.websocket.session import WebsocketStreamSession

DEFAULT_WEBSOCKET_MAX_SIZE_BYTES = 2**20
BYTES_PER_SAMPLE = 2


def websocket_start_timeout_seconds(max_chunk_duration_seconds: float) -> float:
    return max(max_chunk_duration_seconds * 2, 1.0)


def websocket_max_message_size_bytes(max_chunk_duration_seconds: float) -> int:
    expected_audio_bytes = int(
        round(
            ParallelOnlineASRProcessor.SAMPLING_RATE * max_chunk_duration_seconds * BYTES_PER_SAMPLE
        )
    )
    return max(expected_audio_bytes, DEFAULT_WEBSOCKET_MAX_SIZE_BYTES)


class WebsocketTranscriptionServer:
    _service_id = itertools.count()
    log_every_processor = False
    _logger_level = logging.DEBUG

    def __init__(self, shared_asr, main_server_logger, **kwargs):
        self._shared_asr = shared_asr
        self._main_server_logger = main_server_logger
        self._kwargs = kwargs
        self._max_chunk_duration_seconds = kwargs.get("max_chunk_duration_seconds", 1.0)

    def create_stream_session(self) -> WebsocketStreamSession:
        stream_id = self.get_unique_name()
        logger = self.log_setup(stream_id)
        processor_manager = ProcessorManager(
            stream_id,
            self._shared_asr,
            logger=logger,
            server_logger=self._main_server_logger,
            **self._kwargs,
        )
        return WebsocketStreamSession(
            processor_manager=processor_manager,
            server_logger=self._main_server_logger,
            logger=logger,
        )

    @classmethod
    def get_unique_name(cls):
        return f"Whisper-service-{next(cls._service_id)}"

    @classmethod
    def log_setup(cls, stream_id):
        return setup_stream_logger(
            stream_id,
            level=cls._logger_level,
            log_every_processor=cls.log_every_processor,
        )

    @staticmethod
    def _first_audio_timeout_seconds(stream_session):
        chunk_duration_seconds = (stream_session.chunk_duration_millis or 1000) / 1000.0
        return max(chunk_duration_seconds * 2, 1.0)

    def _start_timeout_seconds(self):
        return websocket_start_timeout_seconds(self._max_chunk_duration_seconds)

    async def _send_error_and_close(self, websocket, exc: WebsocketProtocolError):
        try:
            await websocket.send(build_error_event(exc.code, exc.message), text=True)
        except ConnectionClosed:
            return
        await websocket.close(code=exc.close_code, reason=exc.message)

    async def handle_connection(self, websocket):
        stream_session = None
        stream_id = "unassigned-websocket-stream"
        stream_logger = self._main_server_logger
        request_task = None

        try:
            try:
                first_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=self._start_timeout_seconds(),
                )
            except asyncio.TimeoutError as exc:
                raise WebsocketProtocolError(
                    "Client did not send the initial start event within "
                    f"{self._start_timeout_seconds():.3f}s",
                    code="deadline_exceeded",
                ) from exc

            stream_session = self.create_stream_session()
            stream_id = stream_session.id
            stream_logger = getattr(stream_session, "logger", self._main_server_logger)
            stream_logger.info("Started websocket connection on %s", stream_id)
            await stream_session.manage_start_message(first_message)
            try:
                first_audio_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=self._first_audio_timeout_seconds(stream_session),
                )
            except asyncio.TimeoutError as exc:
                raise WebsocketProtocolError(
                    f"{stream_id} did not send the first audio frame within "
                    f"{self._first_audio_timeout_seconds(stream_session):.3f}s",
                    code="deadline_exceeded",
                ) from exc

            if isinstance(first_audio_message, str):
                parse_finish_message(first_audio_message)
                stream_logger.info(
                    "Client closed request stream for %s before sending audio",
                    stream_id,
                )
                await websocket.send(build_completed_event(), text=True)
                await websocket.close(code=1000, reason="completed")
                return

            await stream_session.consume_initial_audio_message(first_audio_message)

            request_task = asyncio.create_task(stream_session.request_enqueuer(websocket))
            has_unsubmitted_audio = True

            async with stream_session.processor_manager.context():
                while not stream_session.processor_manager.is_finished():
                    if request_task.done():
                        task_error = request_task.exception()
                        if task_error is not None:
                            raise task_error
                        if (
                            stream_session.processor_manager.audio_queue.empty()
                            and not has_unsubmitted_audio
                        ):
                            if stream_session.processor_manager.processor.has_audio_since_last_decode():
                                await self._shared_asr.set_processor_ready(stream_id)
                                await stream_session.processor_manager.get_transcription()
                            break

                    if (
                        stream_session.processor_manager.audio_queue.empty()
                        and not has_unsubmitted_audio
                    ):
                        await asyncio.sleep(0.001)
                        continue

                    await stream_session.processor_manager.insert_audio()
                    await asyncio.sleep(0.001)
                    await self._shared_asr.set_processor_ready(stream_id)
                    await stream_session.processor_manager.insert_audio()
                    await stream_session.processor_manager.get_transcription()
                    has_unsubmitted_audio = False
                    if stream_session.processor_manager.is_finished():
                        break

                    for response in stream_session.create_response():
                        await websocket.send(response, text=True)

            if not stream_session.processor_manager.is_finished():
                for response in stream_session.final_response():
                    await websocket.send(response, text=True)

            await websocket.send(build_completed_event(), text=True)
            await websocket.close(code=1000, reason="completed")
        except ConnectionClosed:
            stream_logger.info("Websocket closed for %s", stream_id)
        except asyncio.CancelledError:
            stream_logger.info("Websocket handler cancelled for %s", stream_id)
            raise
        except WebsocketProtocolError as exc:
            stream_logger.warning("Protocol error in %s: %s", stream_id, exc.message)
            await self._send_error_and_close(websocket, exc)
        except Exception:
            stream_logger.exception("Unhandled websocket exception in %s", stream_id)
            await self._send_error_and_close(
                websocket,
                WebsocketProtocolError(
                    "Internal server error",
                    code="internal_error",
                    close_code=1011,
                ),
            )
        finally:
            if request_task is not None and not request_task.done():
                request_task.cancel()
            if request_task is not None:
                try:
                    await request_task
                except (asyncio.CancelledError, ConnectionClosed, WebsocketProtocolError):
                    pass
                except Exception:
                    pass


def _build_not_found_response():
    return Response(
        404,
        "Not Found",
        Headers({"Content-Type": "text/plain; charset=utf-8"}),
        b"Not Found\n",
    )


async def _process_request(_connection, request):
    if request.path != WEBSOCKET_TRANSCRIBE_PATH:
        return _build_not_found_response()
    return None


async def serve(args):
    log_level_name = args.log_level.upper()
    if log_level_name not in logging._nameToLevel:
        print(
            "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
            file=sys.stderr,
        )
        sys.exit(1)
    log_level = logging._nameToLevel[log_level_name]

    app_logger = setup_application_logging(level=log_level, use_stdout=True)
    server_logger = app_logger.getChild("server")
    WebsocketTranscriptionServer._logger_level = log_level

    server_logger.info("Starting websocket server...")

    if args.log_every_processor:
        WebsocketTranscriptionServer.log_every_processor = True
        server_logger.warning(
            "Per-stream file logging enabled. Debug-only: busy or long-running servers "
            "can keep many log files open and create many files."
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
    else:
        server_logger.info("Fallback logic disabled")

    if args.buffer_trimming_sec <= 0:
        server_logger.error("Buffer trimming must be greater than 0")
        sys.exit(1)

    if args.max_chunk_duration_seconds <= 0:
        server_logger.error("Max chunk duration must be greater than 0")
        sys.exit(1)

    server_logger.info(
        "Using faster-whisper model %s with %s backend (vad=%s)",
        args.model,
        args.backend,
        "on" if args.vad else "off",
    )
    shared_asr = ParallelRealtimeASR(
        modelsize=args.model,
        cache_dir=args.model_cache_dir,
        model_dir=args.model_dir,
        logger=app_logger.getChild("asr"),
        warmup_file=args.warmup_file,
        use_vad=args.vad,
        backend=args.backend,
    )
    server_logger.info("Model loaded")

    await shared_asr.start()
    processor_args = {
        "use_fallback": args.fallback,
        "fallback_threshold": args.fallback_threshold,
        "qratio_threshold": args.qratio_threshold,
        "dedup_threshold": args.dedup_threshold,
        "buffer_trimming_sec": args.buffer_trimming_sec,
        "max_chunk_duration_seconds": args.max_chunk_duration_seconds,
    }
    transport_server = WebsocketTranscriptionServer(shared_asr, server_logger, **processor_args)

    shutdown_event = asyncio.Event()

    def _shutdown():
        server_logger.info("Shutdown signal received")
        shutdown_event.set()

    wait_for_signal = False
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
        wait_for_signal = True
    except NotImplementedError:
        server_logger.info(
            "Signal handlers are not supported on this platform. Use Ctrl+C to stop the server."
        )

    async with websocket_serve(
        transport_server.handle_connection,
        args.host,
        args.port,
        process_request=_process_request,
        compression=None,
        logger=server_logger,
        max_size=websocket_max_message_size_bytes(args.max_chunk_duration_seconds),
    ):
        server_logger.info(
            "Websocket server started on %s:%s%s",
            args.host,
            args.port,
            WEBSOCKET_TRANSCRIBE_PATH,
        )
        try:
            if wait_for_signal:
                await shutdown_event.wait()
            else:
                await asyncio.Future()
        finally:
            server_logger.info("Stopping ASR...")
            await shared_asr.stop()
            server_logger.info("ASR stopped")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Argument parser for the swim websocket server",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--no-fallback",
        dest="fallback",
        action="store_false",
        default=True,
        help="Disable fallback logic when similarity local agreement fails repeatedly",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=1,
        help="threshold t for fallback logic after t+1 similarity local agreement fails (ignored if fallback is disabled)",
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
        help="Maximum chunk duration accepted from a client start message",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind the websocket server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the websocket server on",
    )
    parser.add_argument(
        "--log-every-processor",
        action="store_true",
        help=(
            "Write one log file per stream. Debug-only: busy or long-running servers "
            "can keep many log files open and create many files."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo,turbo".split(
            ","
        ),
        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Directory for the whisper model caching",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory for a custom ct2 whisper model skipping if --model provided",
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default="resources/sample1.wav",
        help="File to warm up the model and speed up the first request",
    )
    parser.add_argument(
        "--lan",
        type=str,
        default="en",
        help="Language for the whisper model to translate to (unused at the moment)",
    )
    parser.set_defaults(vad=True)
    parser.add_argument(
        "--no-vad",
        dest="vad",
        action="store_false",
        help="Disable the shared VAD preprocessing step before transcription",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=resolve_asr_backend(),
        choices=("batched", "plain"),
        help="Shared ASR backend to use: batched inference pipeline or plain WhisperModel",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Log level for the server and shared ASR logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser
