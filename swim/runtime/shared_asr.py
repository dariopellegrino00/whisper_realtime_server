import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

from swim.asr.faster_whisper import FasterWhisperASR
from swim.runtime.audio import load_audio_chunk
from swim.runtime.processor import OnlineASRProcessor, ParallelOnlineASRProcessor

DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS = 1.0
DEFAULT_ASR_BACKEND = "plain"
DEFAULT_BATCHED_INFERENCE_BATCH_SIZE = 16
DEFAULT_BATCHED_BACKEND_CLIP_SEPARATOR_SAMPLES = 100
DEFAULT_PLAIN_BACKEND_CLIP_SEPARATOR_SAMPLES = 800
VALID_ASR_BACKENDS = {"batched", "plain"}


def resolve_asr_backend(backend=None):
    if backend is None:
        backend = os.getenv("USE_BATCHED_INFERENCE")

    if backend is None:
        return DEFAULT_ASR_BACKEND

    normalized = str(backend).strip().lower()
    if normalized in VALID_ASR_BACKENDS:
        return normalized
    if normalized in {"1", "true", "yes", "on"}:
        return "batched"
    if normalized in {"0", "false", "no", "off"}:
        return "plain"

    raise ValueError(f"Unsupported ASR backend {backend!r}. Expected one of: batched, plain")


class FasterWhisperBackendAdapter:
    name: str | None = None

    def load_model(self, asr, modelsize=None, cache_dir=None, model_dir=None):
        raise NotImplementedError

    def build_clip_timestamps(self, clip_windows):
        raise NotImplementedError

    def build_transcribe_kwargs(self):
        return {}

    def separator_samples(self):
        return DEFAULT_BATCHED_BACKEND_CLIP_SEPARATOR_SAMPLES


class PlainWhisperBackendAdapter(FasterWhisperBackendAdapter):
    name = "plain"

    def load_model(self, asr, modelsize=None, cache_dir=None, model_dir=None):
        return FasterWhisperASR.load_model(asr, modelsize, cache_dir, model_dir)

    def build_clip_timestamps(self, clip_windows):
        timestamps = []
        for clip in clip_windows:
            timestamps.extend([clip["start_seconds"], clip["end_seconds"]])
        return timestamps

    def separator_samples(self):
        return DEFAULT_PLAIN_BACKEND_CLIP_SEPARATOR_SAMPLES


class BatchedWhisperBackendAdapter(FasterWhisperBackendAdapter):
    name = "batched"

    def load_model(self, asr, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import BatchedInferencePipeline

        model = FasterWhisperASR.load_model(asr, modelsize, cache_dir, model_dir)
        return BatchedInferencePipeline(model)

    def build_clip_timestamps(self, clip_windows):
        return [{"start": clip["start_sample"], "end": clip["end_sample"]} for clip in clip_windows]

    def build_transcribe_kwargs(self):
        return {
            "without_timestamps": False,
            "batch_size": DEFAULT_BATCHED_INFERENCE_BATCH_SIZE,
        }


def create_asr_backend_adapter(backend=None):
    resolved_backend = resolve_asr_backend(backend)
    if resolved_backend == "plain":
        return PlainWhisperBackendAdapter()
    if resolved_backend == "batched":
        return BatchedWhisperBackendAdapter()
    raise ValueError(
        f"Unsupported ASR backend {resolved_backend!r}. Expected one of: batched, plain"
    )


class ParallelAudioBuffer:
    def __init__(self, separator_samples=DEFAULT_BATCHED_BACKEND_CLIP_SEPARATOR_SAMPLES):
        self._audio_chunks = []
        self._audio_size = 0
        self._segment_times = []
        self._ids = []
        self._separator_samples = separator_samples

    def reset(self):
        self._audio_chunks = []
        self._audio_size = 0
        self._segment_times = []
        self._ids = []

    def append_token(self, processor_id, audio):
        audio_length = len(audio)
        if audio_length == 0:
            return

        self._segment_times.append(
            {"start": self._audio_size, "end": self._audio_size + audio_length}
        )
        self._ids.append(processor_id)
        self._audio_chunks.append(audio)
        self._audio_chunks.append(np.zeros(self._separator_samples, dtype=np.float32))
        self._audio_size += audio_length + self._separator_samples

    @property
    def size(self):
        return self._audio_size

    def __len__(self):
        return self._audio_size

    def parameters(self):
        audio = (
            np.concatenate(self._audio_chunks)
            if self._audio_chunks
            else np.array([], dtype=np.float32)
        )
        ns = SimpleNamespace(ids=self._ids, audio=audio, segment_times=self._segment_times)
        return copy.deepcopy(ns)


class MultiProcessingFasterWhisperASR(FasterWhisperASR):
    def __init__(
        self,
        lan,
        modelsize=None,
        cache_dir=None,
        model_dir=None,
        logfile=None,
        use_vad=False,
        backend=None,
    ):
        self._client_events = []
        self._last_transcript_time = 0.0
        self._log = logfile if logfile is not None else logging.getLogger(__name__)
        self._use_vad = use_vad
        self._vad_options = VadOptions()
        self._backend_adapter = create_asr_backend_adapter(backend)
        super().__init__(lan, modelsize, cache_dir, model_dir, self._log)

    @property
    def backend(self):
        return self._backend_adapter.name

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
                self.transcribe_parallel(buffer, use_vad=False)
                self._log.info("asr is warmed up")
            else:
                self._log.info("%s not found", filepath)
        else:
            self._log.info("no warmup file provided or file not found")

    def _prepare_clip(self, clip_audio, use_vad=None):
        if use_vad is None:
            use_vad = self._use_vad

        if not use_vad:
            return clip_audio, None

        speech_chunks = get_speech_timestamps(clip_audio, self._vad_options)
        audio_chunks, _chunks_metadata = collect_chunks(clip_audio, speech_chunks)
        compact_audio = (
            np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        )
        timestamp_map = (
            SpeechTimestampsMap(
                speech_chunks,
                OnlineASRProcessor.SAMPLING_RATE,
                time_precision=5,
            )
            if speech_chunks
            else None
        )
        return compact_audio, timestamp_map

    def _prepare_shared_audio(self, parameters, use_vad=None):
        sample_rate = OnlineASRProcessor.SAMPLING_RATE
        shared_buffer = ParallelAudioBuffer(
            separator_samples=self._backend_adapter.separator_samples()
        )
        clip_windows = []

        for processor_id, segment in zip(parameters.ids, parameters.segment_times, strict=True):
            clip_audio = parameters.audio[segment["start"] : segment["end"]]
            if len(clip_audio) == 0:
                continue

            compact_audio, timestamp_map = self._prepare_clip(clip_audio, use_vad=use_vad)
            if len(compact_audio) == 0:
                continue

            shared_start_sample = shared_buffer.size
            shared_buffer.append_token(processor_id, compact_audio)
            clip_windows.append(
                {
                    "id": processor_id,
                    "start_sample": shared_start_sample,
                    "end_sample": shared_start_sample + len(compact_audio),
                    "start_seconds": shared_start_sample / sample_rate,
                    "end_seconds": (shared_start_sample + len(compact_audio)) / sample_rate,
                    "timestamp_map": timestamp_map,
                }
            )

        return shared_buffer.parameters(), clip_windows

    @staticmethod
    def _segment_clip(segment, clip_windows):
        midpoint = (segment.start + segment.end) / 2
        for clip in clip_windows:
            if clip["start_seconds"] <= midpoint <= clip["end_seconds"]:
                return clip
        return None

    @staticmethod
    def _restore_word_timestamps(segment, clip):
        output = []
        for word in segment.words:
            if segment.no_speech_prob > 0.9:
                continue

            local_start = word.start - clip["start_seconds"]
            local_end = word.end - clip["start_seconds"]

            if clip["timestamp_map"] is None:
                restored_start = local_start
                restored_end = local_end
            else:
                middle = (local_start + local_end) / 2
                chunk_index = clip["timestamp_map"].get_chunk_index(middle)
                restored_start = clip["timestamp_map"].get_original_time(
                    local_start,
                    chunk_index,
                )
                restored_end = clip["timestamp_map"].get_original_time(
                    local_end,
                    chunk_index,
                )

            output.append((round(restored_start, 5), round(restored_end, 5), word.word))
        return output

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        return self._backend_adapter.load_model(
            self, modelsize=modelsize, cache_dir=cache_dir, model_dir=model_dir
        )

    def _clip_timestamps(self, clip_windows):
        return self._backend_adapter.build_clip_timestamps(clip_windows)

    def transcribe_parallel(self, audio_buffer: ParallelAudioBuffer, use_vad=None):
        if not len(audio_buffer):
            return []

        parameters = audio_buffer.parameters()
        shared_parameters, clip_windows = self._prepare_shared_audio(parameters, use_vad=use_vad)
        if not clip_windows:
            return []

        timestamp = time.monotonic()
        segments, _info = self.model.transcribe(
            shared_parameters.audio,
            beam_size=5,
            condition_on_previous_text=False,
            multilingual=True,
            word_timestamps=True,
            clip_timestamps=self._clip_timestamps(clip_windows),
            **self._backend_adapter.build_transcribe_kwargs(),
        )

        results_by_id: dict[object, list[tuple[float, float, str]]] = {
            clip["id"]: [] for clip in clip_windows
        }
        for segment in segments:
            clip = self._segment_clip(segment, clip_windows)
            if clip is None:
                self._log.warning(
                    "Unable to map segment start=%s end=%s to a shared clip window",
                    segment.start,
                    segment.end,
                )
                continue

            results_by_id[clip["id"]].extend(self._restore_word_timestamps(segment, clip))

        results_tagged = [
            (clip["id"], results_by_id[clip["id"]])
            for clip in clip_windows
            if results_by_id[clip["id"]]
        ]

        self._last_transcript_time = time.monotonic() - timestamp
        self._log.debug("transcription time: %s seconds", self._last_transcript_time)
        return results_tagged


@dataclass
class RegisteredProcess:
    asr_processor: ParallelOnlineASRProcessor
    ready_flag: bool = False
    never_committed_flag: bool = True
    chunk_duration_seconds: float = DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS
    transcription_event: asyncio.Event = field(default_factory=asyncio.Event)


class ParallelRealtimeASR:
    def __init__(
        self,
        modelsize="large-v3-turbo",
        cache_dir=None,
        model_dir=None,
        logger=None,
        warmup_file=None,
        use_vad=False,
        backend=None,
    ):
        self._registered_pids: dict[object, RegisteredProcess] = {}
        self._register_lock = asyncio.Lock()
        self._audio_buffer = ParallelAudioBuffer()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._asr = MultiProcessingFasterWhisperASR(
            "auto",
            modelsize=modelsize,
            cache_dir=cache_dir,
            model_dir=model_dir,
            logfile=self._logger,
            use_vad=use_vad,
            backend=backend,
        )
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

    async def register_processor(self, processor_id, asr_processor):
        async with self._register_lock:
            if self._loop_failure is not None:
                raise RuntimeError("Shared ASR loop failed") from self._loop_failure
            self._registered_pids[processor_id] = RegisteredProcess(
                asr_processor=asr_processor,
                ready_flag=False,
                chunk_duration_seconds=getattr(
                    asr_processor,
                    "chunk_duration_seconds",
                    DEFAULT_EXPECTED_CHUNK_DURATION_SECONDS,
                ),
            )

    async def unregister_processor(self, processor_id):
        async with self._register_lock:
            registered = self._registered_pids.pop(processor_id, None)
            if registered is not None:
                registered.transcription_event.set()
            return registered

    def append_audio(self, processor_id, audio):
        self._audio_buffer.append_token(processor_id, audio)

    async def set_processor_ready(self, processor_id):
        async with self._register_lock:
            if processor_id in self._registered_pids:
                self._registered_pids[processor_id].transcription_event.clear()
                self._registered_pids[processor_id].ready_flag = True
            else:
                raise ValueError(f"{processor_id} is not a registered processor.")

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
            for processor_id, process in self._registered_pids.items():
                if not process.ready_flag:
                    continue
                process.ready_flag = False
                process.never_committed_flag = False
                claimed[processor_id] = process.asr_processor
            return claimed

    async def _timed_out_processor_candidates(self):
        async with self._register_lock:
            return {
                processor_id
                for processor_id, process in self._registered_pids.items()
                if not process.ready_flag
            }

    async def _exclude_timed_out_processors(self):
        async with self._register_lock:
            to_remove = [
                processor_id
                for processor_id, proc in self._registered_pids.items()
                if not proc.ready_flag
            ]
            for processor_id in to_remove:
                registered_process = self._registered_pids[processor_id]
                if registered_process.never_committed_flag:
                    self._logger.info(
                        "Processor %s is new and not ready, giving grace period.",
                        processor_id,
                    )
                    registered_process.never_committed_flag = False
                else:
                    self._logger.warning(
                        "Processor %s timed out and will be excluded.",
                        processor_id,
                    )
                    registered_process.asr_processor.timed_out = True
                    registered_process.transcription_event.set()
                    self._registered_pids.pop(processor_id)

    async def _exclude_still_not_ready_processors(self, processor_ids):
        async with self._register_lock:
            for processor_id in processor_ids:
                process = self._registered_pids.get(processor_id)
                if process is None or process.ready_flag:
                    continue
                if process.never_committed_flag:
                    self._logger.info(
                        "Processor %s is new and not ready, giving grace period.",
                        processor_id,
                    )
                    process.never_committed_flag = False
                    continue
                self._logger.warning(
                    "Processor %s timed out and will be excluded.",
                    processor_id,
                )
                process.asr_processor.timed_out = True
                process.transcription_event.set()
                self._registered_pids.pop(processor_id)

    async def _transcribe_current_processors(self, current_processors, waiting_time):
        processor_ids = sorted(current_processors)
        total_audio_seconds = sum(
            len(processor.audio_buffer) / OnlineASRProcessor.SAMPLING_RATE
            for processor in current_processors.values()
        )
        self._logger.debug("Time lost waiting %s seconds", waiting_time)
        self._logger.info(
            "Transcribing %d processor(s): ids=%s total_audio_seconds=%.2f waited=%.3fs registered=%d",
            len(current_processors),
            processor_ids,
            total_audio_seconds,
            waiting_time,
            len(self._registered_pids),
        )

        async with self._register_lock:
            for processor_id, processor in current_processors.items():
                self.append_audio(processor_id, processor.audio_buffer)

        loop = asyncio.get_running_loop()
        transcription_started_at = time.monotonic()
        results = await loop.run_in_executor(
            None, self._asr.transcribe_parallel, self._audio_buffer
        )
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
            self._logger.debug("Result %s: %s", processor_id, processor.results)
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
                            await self._transcribe_current_processors(
                                current_processors, waiting_time
                            )
                            await self._exclude_still_not_ready_processors(timed_out_candidates)
                        else:
                            await self._exclude_timed_out_processors()
                        batch_wait_started_at = time.monotonic()
                        continue
                    await asyncio.sleep(0.001)
                    continue

                waiting_time = (
                    0.0
                    if batch_wait_started_at is None
                    else time.monotonic() - batch_wait_started_at
                )
                current_processors = await self._claim_ready_processors()
                await self._transcribe_current_processors(current_processors, waiting_time)
                batch_wait_started_at = None
        except Exception as exc:
            self._loop_failure = exc
            async with self._register_lock:
                for process in self._registered_pids.values():
                    process.transcription_event.set()
            self._logger.exception(exc)
