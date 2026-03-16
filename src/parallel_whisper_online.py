import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict

import numpy as np

from src.whisper_online import *

DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS = 1.0


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
        timestamp = time.monotonic()

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

        self._last_transcript_time = time.monotonic() - timestamp
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
    never_committed_flag: bool = True
    chunk_duration_seconds: float = DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS
    transcription_event: asyncio.Event = field(default_factory=asyncio.Event)


class ParallelRealtimeASR:
    def __init__(self, modelsize="large-v3-turbo", logger=None, warmup_file=None):
        self._registered_pids: Dict[int, RegisteredProcess] = {}
        self._register_lock = asyncio.Lock()
        self._audio_buffer = ParallelAudioBuffer()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._asr = MultiProcessingFasterWhisperASR("auto", modelsize=modelsize, logfile=self._logger)
        self._stopped = False
        self._loop_task = None
        self._loop_failure = None

        if warmup_file:
            self._asr.warmup(warmup_file)

    @property
    def asr(self):
        return self._asr

    async def start(self):
        self._loop_task = asyncio.create_task(self._asr_loop())

    async def stop(self):
        self._stopped = True
        async with self._register_lock:
            for process in self._registered_pids.values():
                process.transcription_event.set()
        if self._loop_task:
            await self._loop_task

    async def register_processor(self, id, asr_processor):
        async with self._register_lock:
            if self._loop_failure is not None:
                raise RuntimeError("Shared ASR loop failed") from self._loop_failure
            self._registered_pids[id] = RegisteredProcess(
                asr_processor=asr_processor,
                ready_flag=False,
                chunk_duration_seconds=getattr(
                    asr_processor,
                    "chunk_duration_seconds",
                    DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS,
                ),
            )

    async def unregister_processor(self, id):
        async with self._register_lock:
            registered = self._registered_pids.pop(id, None)
            if registered is not None:
                registered.transcription_event.set()
            return registered

    def append_audio(self, id, audio):
        self._audio_buffer.append_token(id, audio)

    async def set_processor_ready(self, id):
        async with self._register_lock:
            if id in self._registered_pids:
                self._registered_pids[id].transcription_event.clear()
                self._registered_pids[id].ready_flag = True
            else:
                raise ValueError(f"{id} is not a registered processor.")

    async def wait(self, processor_id):
        if self._loop_failure is not None:
            raise RuntimeError("Shared ASR loop failed") from self._loop_failure
        async with self._register_lock:
            registered = self._registered_pids.get(processor_id)
            if registered is None:
                return
            transcription_event = registered.transcription_event
        await transcription_event.wait()
        if self._loop_failure is not None:
            raise RuntimeError("Shared ASR loop failed") from self._loop_failure

    def _barrier_timeout_seconds(self):
        max_chunk_duration = max(
            (proc.chunk_duration_seconds for proc in self._registered_pids.values()),
            default=DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS,
        )
        return max_chunk_duration * 2

    async def _ready_counts(self):
        async with self._register_lock:
            registered_count = len(self._registered_pids)
            ready_count = sum(1 for proc in self._registered_pids.values() if proc.ready_flag)
            return ready_count, registered_count

    async def _claim_ready_processors(self):
        async with self._register_lock:
            claimed = {}
            for pid, process in self._registered_pids.items():
                if not process.ready_flag:
                    continue
                process.ready_flag = False
                process.never_committed_flag = False
                claimed[pid] = process.asr_processor
            return claimed

    async def _timed_out_processor_candidates(self):
        async with self._register_lock:
            return {
                pid
                for pid, process in self._registered_pids.items()
                if not process.ready_flag
            }

    async def _exclude_timed_out_processors(self):
        async with self._register_lock:
            to_remove = [pid for pid, proc in self._registered_pids.items() if not proc.ready_flag]
            for pid in to_remove:
                rp = self._registered_pids[pid]
                if rp.never_committed_flag:
                    self._logger.info(f"Processor {pid} is new and not ready, giving grace period.")
                    rp.never_committed_flag = False
                else:
                    self._logger.warning(f"Processor {pid} timed out and will be excluded.")
                    rp.asr_processor.timed_out = True
                    rp.transcription_event.set()
                    self._registered_pids.pop(pid)

    async def _exclude_still_not_ready_processors(self, processor_ids):
        async with self._register_lock:
            for pid in processor_ids:
                process = self._registered_pids.get(pid)
                if process is None or process.ready_flag:
                    continue
                if process.never_committed_flag:
                    self._logger.info(f"Processor {pid} is new and not ready, giving grace period.")
                    process.never_committed_flag = False
                    continue
                self._logger.warning(f"Processor {pid} timed out and will be excluded.")
                process.asr_processor.timed_out = True
                process.transcription_event.set()
                self._registered_pids.pop(pid)

    async def _transcribe_current_processors(self, current_processors, waiting_time):
        processor_ids = sorted(current_processors)
        total_audio_seconds = sum(
            len(processor.audio_buffer) / OnlineASRProcessor.SAMPLING_RATE
            for processor in current_processors.values()
        )
        self._logger.debug(f"Time lost waiting {waiting_time} seconds")
        self._logger.info(
            "Transcribing %d processor(s): ids=%s total_audio_seconds=%.2f waited=%.3fs registered=%d",
            len(current_processors),
            processor_ids,
            total_audio_seconds,
            waiting_time,
            len(self._registered_pids),
        )

        async with self._register_lock:
            for pid, processor in current_processors.items():
                self.append_audio(pid, processor.audio_buffer)

        loop = asyncio.get_running_loop()
        transcription_started_at = time.monotonic()
        results = await loop.run_in_executor(None, self._asr.transcribe_parallel, self._audio_buffer)
        transcription_elapsed = time.monotonic() - transcription_started_at
        self._audio_buffer.reset()
        self._logger.info(
            "Transcribed %d processor(s) in %.3fs",
            len(current_processors),
            transcription_elapsed,
        )

        results_by_id = {processor_id: result for processor_id, result in results}

        for processor_id, processor in current_processors.items():
            result = results_by_id.get(processor_id, [])
            processor.update(result)
            self._logger.debug(f"Result {processor_id}: {processor.results}")
            async with self._register_lock:
                registered = self._registered_pids.get(processor_id)
                if registered is not None:
                    registered.transcription_event.set()

    async def _asr_loop(self):
        self._logger.info("ASR loop started")
        batch_wait_started_at = None

        try:
            while not self._stopped:
                ready_count, registered_count = await self._ready_counts()

                if ready_count != registered_count or registered_count == 0:
                    if ready_count == 0:
                        batch_wait_started_at = None
                        await asyncio.sleep(0.001)
                        continue

                    if batch_wait_started_at is None:
                        batch_wait_started_at = time.monotonic()

                    if time.monotonic() - batch_wait_started_at >= self._barrier_timeout_seconds():
                        self._logger.error("Timeout waiting for transcription")
                        waiting_time = time.monotonic() - batch_wait_started_at
                        timed_out_candidates = await self._timed_out_processor_candidates()
                        current_processors = await self._claim_ready_processors()
                        if current_processors:
                            await self._transcribe_current_processors(current_processors, waiting_time)
                            await self._exclude_still_not_ready_processors(timed_out_candidates)
                        else:
                            await self._exclude_timed_out_processors()
                        batch_wait_started_at = time.monotonic()
                        continue
                    await asyncio.sleep(0.001)
                    continue

                waiting_time = 0.0 if batch_wait_started_at is None else time.monotonic() - batch_wait_started_at
                current_processors = await self._claim_ready_processors()
                await self._transcribe_current_processors(current_processors, waiting_time)
                batch_wait_started_at = None
        except Exception as exc:
            self._loop_failure = exc
            async with self._register_lock:
                for process in self._registered_pids.values():
                    process.transcription_event.set()
            self._logger.exception(exc)
