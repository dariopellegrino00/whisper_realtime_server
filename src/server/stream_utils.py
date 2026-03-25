import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Iterable, Optional

import numpy as np

from src.parallel_whisper_online import ParallelOnlineASRProcessor


def _configure_stdout_utf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(
    log_name, use_stdout=False, log_folder="server_logs", level=logging.DEBUG
):
    os.makedirs(log_folder, exist_ok=True)

    log_path = os.path.join(
        log_folder, f"{datetime.now():%Y%m%d_%H%M%S}_{log_name}.log"
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if use_stdout:
        _configure_stdout_utf8()
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class TranscriptionManager:
    def __init__(self):
        self.last_end = None

    def format_transcript(self, t, use_last_end=True):
        if t and t[0] is not None:
            beg, end = t[0] * 1000, t[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            if beg < 0:
                beg = 0
            if use_last_end:
                self.last_end = end
            return True, (int(round(beg)), int(round(end)), t[2])
        return False, (0, 0, "")


class ProcessorManager:
    def __init__(self, id, shared_asr, logger=None, server_logger=None, **kwargs):
        self.kwargs = kwargs
        self.id = id
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.server_logger = (
            server_logger if server_logger is not None else logging.getLogger(__name__)
        )
        self.processor = ParallelOnlineASRProcessor(
            asr=shared_asr.asr,
            logger=self.logger,
            **self.kwargs,
        )
        self.processor.init()
        self.audio_queue = asyncio.Queue()
        self._shared_asr = shared_asr
        self.max_chunk_duration_seconds = kwargs.get("max_chunk_duration_seconds", 1.0)
        self._stream_closed_event = asyncio.Event()
        self._registered = False

    async def insert_audio(
        self, already_collected_chunks: Optional[Iterable[float]] = None
    ):
        audio_batch = []
        if already_collected_chunks is not None:
            audio_batch.extend(already_collected_chunks)

        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
            audio_batch.extend(chunk)

        if audio_batch:
            self.processor.insert_audio_chunk(np.array(audio_batch, dtype=np.float32))

    async def get_transcription(self):
        await self._shared_asr.wait(self.id)

    def mark_stream_closed(self):
        self._stream_closed_event.set()

    @asynccontextmanager
    async def context(self, re_init_processor=False):
        if re_init_processor:
            self.processor.init()
        try:
            while (
                self.audio_queue.qsize() < 2 and not self._stream_closed_event.is_set()
            ):
                await asyncio.sleep(0.001)

            await self._shared_asr.register_processor(self.id, self.processor)
            self._registered = True
            self.server_logger.debug(
                f"{self.id} accumulated {self.audio_queue.qsize()} queued chunks before registration"
            )
            yield
        except Exception as exc:
            self.server_logger.error(f"Exception in context manager of {self.id}:")
            self.server_logger.exception(exc)
            raise
        finally:
            if self._registered:
                await self._shared_asr.unregister_processor(self.id)
            self.server_logger.debug(f"{self.id} finished processing")

    def is_finished(self):
        return self.processor.timed_out
