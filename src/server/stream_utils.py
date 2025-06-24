import os
import sys
import logging
import asyncio
import numpy as np
from typing import Iterable, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from src.parallel_whisper_online import ParallelOnlineASRProcessor

# LOGGING SETUP FUNCTION
def setup_logging(log_name, use_stdout=False, log_folder="server_logs"):
    os.makedirs(log_folder, exist_ok=True)

    log_path = os.path.join(log_folder, f"{datetime.now():%Y%m%d_%H%M%S}_{log_name}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    handlers = [logging.FileHandler(log_path)]
    if use_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

### Here for the purpose of future expansion,
class TranscriptionManager:
    def __init__(self):
        self.last_end = None

    def format_transcript(self, t):
        if t and t[0] is not None: # what if t is null
            beg, end = t[0]*1000, t[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            if beg < 0: beg = 0
            self.last_end = end
            return True, (int(round(beg)), int(round(end)), t[2])
        else:
            return False, (0, 0, "")

# Parallel ASR Processors manager, common for every service 
# make indipendent processors work with a shared ASR
class ProcessorManager:
    def __init__(self, id, shared_asr, logger=None, server_logger=None, **kwargs):
        self.kwargs = kwargs
        self.id = id
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.server_logger = server_logger if server_logger is not None else logging.getLogger(__name__)
        self.processor = ParallelOnlineASRProcessor(asr=shared_asr.asr, logger=self.logger, **self.kwargs)
        self.processor.init()
        self.audio_queue = asyncio.Queue()
        self._shared_asr = shared_asr 

    async def insert_audio(self, already_collected_chunks: Optional[Iterable[float]] = None):
        """
        Insert the audio chunks collected by the async audio_queue into the processor.
        If a chunk was already collected, it will be inserted into the processor.
        """
        audio_batch = []
        if already_collected_chunks is not None:
            audio_batch.extend(already_collected_chunks)

        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
            audio_batch.extend(chunk)

        if audio_batch:
            self.processor.insert_audio_chunk(np.array(audio_batch, dtype=np.float32))

    async def get_transcription(self):
        """
        Get the transcription from the processor.
        TODO: for future preprocessing while waiting for the transcription. VAD or other.
        """
        await self._shared_asr.wait() 

    @asynccontextmanager
    async def context(self, re_init_processor=False):
        if re_init_processor: 
            self.processor.init()
        try:
            while self.audio_queue.qsize() < 2:
                await asyncio.sleep(0.001)
            await self._shared_asr.register_processor(self.id, self.processor)
            self.server_logger.debug(f"{self.id} accumulated {self.audio_queue.qsize()} chunks for the first time")
            yield # implement logic here, this is were the code inside with statement is executed
        except Exception as e:
            self.server_logger.error(f"Exception in context manager of {self.id}: ") 
            self.server_logger.exception(e)
        finally:
            await self._shared_asr.unregister_processor(self.id)
            self.server_logger.debug(f"{self.id} finished processing")

