import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict

import numpy as np

from src.whisper_online import *


class ParallelAudioBuffer:
    def __init__(self):
        self._audio_chunks = []
        self._audio_size = 0
        self._segment_times = []
        self._ids = []

    def reset(self):
        self._audio_chunks = []
        self._audio_size = 0
        self._segment_times = []
        self._ids = []

    def append_token(self, id, audio):
        audio_length = len(audio)
        if audio_length == 0:
            return

        self._segment_times.append({"start": self._audio_size, "end": self._audio_size + audio_length})
        self._ids.append(id)
        self._audio_chunks.append(audio)
        self._audio_chunks.append(np.zeros(100, dtype=np.float32))
        self._audio_size += audio_length + 100

    @property
    def size(self):
        return self._audio_size

    def __len__(self):
        return self._audio_size

    def parameters(self):
        audio = np.concatenate(self._audio_chunks) if self._audio_chunks else np.array([], dtype=np.float32)
        ns = SimpleNamespace(ids=self._ids, audio=audio, segment_times=self._segment_times)
        return copy.deepcopy(ns)


class MultiProcessingFasterWhisperASR(FasterWhisperASR):
    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=logging.getLogger(__name__)):
        self._client_events = []
        self._last_transcript_time = 0.0
        self._log = logfile
        super().__init__(lan, modelsize, cache_dir, model_dir, logfile)

    @staticmethod
    def normalize_segment(start, segment):
        output = []
        for word in segment.words:
            if segment.no_speech_prob > 0.9:
                continue
            output.append((round(word.start - start, 5), round(word.end - start, 5), word.word))
        return output

    def warmup(self, filepath):
        if filepath:
            if os.path.isfile(filepath):
                audio = load_audio_chunk(filepath, 0, 1)
                buffer = ParallelAudioBuffer()
                buffer.append_token(1, audio)
                self.transcribe_parallel(buffer)
                self._log.info("asr is warmed up")
            else:
                self._log.info(f"{filepath} not found")
        else:
            self._log.info("no warmup file provided or file not found")

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import BatchedInferencePipeline

        model = super().load_model(modelsize, cache_dir, model_dir)
        return BatchedInferencePipeline(model)

    def transcribe_parallel(self, audio_buffer: ParallelAudioBuffer):
        if not len(audio_buffer):
            return []

        parameters = audio_buffer.parameters()
        timestamp = time.time()

        segments, _ = self.model.transcribe(
            parameters.audio,
            beam_size=5,
            condition_on_previous_text=False,
            multilingual=True,
            word_timestamps=True,
            clip_timestamps=parameters.segment_times,
            batch_size=16,
        )

        segment_list = list(segments)
        results_tagged = [
            (id, self.normalize_segment(start, seg))
            for (id, start, seg) in zip(
                parameters.ids,
                [segment["start"] / OnlineASRProcessor.SAMPLING_RATE for segment in parameters.segment_times],
                segment_list,
            )
        ]

        self._last_transcript_time = time.time() - timestamp
        self._log.debug(f"transcription time: {self._last_transcript_time} seconds")
        return results_tagged


class ParallelOnlineASRProcessor(OnlineASRProcessor):
    def __init__(self, asr, logger=logging.getLogger(__name__), **kwargs):
        super().__init__(asr, **kwargs)
        self.logger = logger
        self.buffer_trimming_sec = kwargs.get("buffer_trimming_sec", 15)
        self._result = None
        self._hypothesis = None
        self.timed_out = False

    @property
    def buffer_time_seconds(self):
        return len(self.audio_buffer) / self.SAMPLING_RATE

    def update(self, results):
        self.logger.debug("ITERATION START\n")
        self.logger.debug(
            f"transcribing {self.buffer_time_seconds:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )

        self.transcript_buffer.insert(results, self.buffer_time_offset)

        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        self._result = self.to_flush(committed)
        self.logger.debug(f">>>>COMPLETE NOW: {self._result}")

        self._hypothesis = self.to_flush(self.transcript_buffer.complete())
        self.logger.debug(f"INCOMPLETE: {self._hypothesis}")

        self._chunk_buffer_at()

        self.logger.info(f"len of buffer now: {self.buffer_time_seconds:2.2f}")
        self.logger.debug("ITERATION END \n")

    @property
    def hypothesis(self):
        return self._hypothesis

    @property
    def results(self):
        return self._result

    def _chunk_buffer_at(self):
        k = len(self.commited) - 1
        s = self.buffer_trimming_sec
        if self.buffer_time_seconds > s and k >= 0:
            limit = self.buffer_time_offset + self.buffer_time_seconds - (s / 2)
            while k > 0 and self.commited[k][1] > limit:
                k -= 1
            t = self.commited[k][1]
            self.logger.debug(f"chunking segment at word {self.commited[-1]} at {t}")
            self.chunk_at(t)


@dataclass
class RegisteredProcess:
    asr_processor: ParallelOnlineASRProcessor
    ready_flag: bool = False
    never_commited_flag: bool = True


class ParallelRealtimeASR:
    def __init__(self, modelsize="large-v3-turbo", logger=None, warmup_file=None):
        self._registered_pids: Dict[int, RegisteredProcess] = {}
        self._register_lock = asyncio.Lock()
        self._transcription_event = asyncio.Event()
        self._audio_buffer = ParallelAudioBuffer()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._asr = MultiProcessingFasterWhisperASR("auto", modelsize=modelsize, logfile=self._logger)
        self._stopped = False
        self._loop_task = None
        self._last_transcript_time_seconds = 0.0
        self._estimated_transcription_time = 1.0
        self._transcript_timeout_seconds = 2.0
        self._timed_out = False

        if warmup_file:
            self._asr.warmup(warmup_file)

    @property
    def asr(self):
        return self._asr

    async def start(self):
        self._loop_task = asyncio.create_task(self._asr_loop())

    async def stop(self):
        self._stopped = True
        if self._loop_task:
            await self._loop_task

    async def register_processor(self, id, asr_processor):
        async with self._register_lock:
            self._registered_pids[id] = RegisteredProcess(asr_processor=asr_processor, ready_flag=False)

    async def unregister_processor(self, id):
        async with self._register_lock:
            return self._registered_pids.pop(id, None)

    def append_audio(self, id, audio):
        self._audio_buffer.append_token(id, audio)

    async def set_processor_ready(self, id):
        async with self._register_lock:
            if id in self._registered_pids:
                self._registered_pids[id].ready_flag = True
            else:
                raise ValueError(f"{id} is not a registered processor.")

    async def wait(self):
        try:
            timeout = max(self._transcript_timeout_seconds, len(self._registered_pids) * 0.15)
            await asyncio.wait_for(self._transcription_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._timed_out = True
            await asyncio.sleep(0.001)

    async def _all_pid_ready(self):
        async with self._register_lock:
            return len(self._registered_pids) > 0 and all(proc.ready_flag for proc in self._registered_pids.values())

    async def _reset_ready_pids(self):
        async with self._register_lock:
            for pid in self._registered_pids:
                self._registered_pids[pid].ready_flag = False
                self._registered_pids[pid].never_commited_flag = False

    async def _exclude_timed_out_processors(self):
        async with self._register_lock:
            to_remove = [pid for pid, proc in self._registered_pids.items() if not proc.ready_flag]
            for pid in to_remove:
                rp = self._registered_pids[pid]
                if rp.never_commited_flag:
                    self._logger.info(f"Processor {pid} is new and not ready, giving grace period.")
                    rp.never_commited_flag = False
                else:
                    self._logger.warning(f"Processor {pid} timed out and will be excluded.")
                    rp.asr_processor.timed_out = True
                    self._registered_pids.pop(pid)

    async def _asr_loop(self):
        self._logger.info("ASR loop started")
        timestamp = time.time()

        try:
            while not self._stopped:
                self._transcription_event.clear()

                if not await self._all_pid_ready():
                    if self._timed_out:
                        self._logger.error("Timeout waiting for transcription")
                        await self._exclude_timed_out_processors()
                        self._timed_out = False
                    await asyncio.sleep(0.001)
                    continue

                self._timed_out = False
                await self._reset_ready_pids()
                waiting_time = time.time() - timestamp
                self._logger.debug(f"Time lost waiting {waiting_time} seconds")
                self._logger.info("Transcribing")
                timestamp = time.time()

                async with self._register_lock:
                    current_processors = self._registered_pids.copy()
                    for pid, processor in self._registered_pids.items():
                        self.append_audio(pid, processor.asr_processor.audio_buffer)

                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, self._asr.transcribe_parallel, self._audio_buffer)
                self._audio_buffer.reset()

                async with self._register_lock:
                    for processor_id, result in results:
                        processor = current_processors[processor_id]
                        processor.asr_processor.update(result)
                        self._logger.debug(f"Result {processor_id}: {processor.asr_processor.results}")

                self._transcription_event.set()
                self._last_transcript_time_seconds = time.time() - timestamp
                self._estimated_transcription_time = (
                    self._estimated_transcription_time * 0.7 + self._last_transcript_time_seconds * 0.3
                )
                self._transcript_timeout_seconds = self._estimated_transcription_time * 2.0
                timestamp = time.time()
        except Exception as exc:
            self._logger.exception(exc)
