import asyncio
import logging

import numpy as np
import pytest

import src.server.stream_utils as stream_utils

# We use the following mock classes to simulate various parts of the ASR
# and processor lifecycle without needing real models or audio hardware.

class DummyProcessor:
    """A minimal mock for ParallelOnlineASRProcessor used in high-level tests."""
    def __init__(self, asr=None, logger=None, **kwargs):
        self.asr = asr
        self.logger = logger
        self.kwargs = kwargs
        self.timed_out = False

    def init(self):
        return None


class DummySharedASR:
    """Mocks the ParallelRealtimeASR shared service to verify registrations."""
    def __init__(self):
        self.asr = object()
        self.registered = []
        self.unregistered = []

    async def register_processor(self, processor_id, processor):
        self.registered.append((processor_id, processor))

    async def unregister_processor(self, processor_id):
        self.unregistered.append(processor_id)

    async def wait(self, processor_id):
        return None


class DummyRuntimeProcessor:
    """Mocks a processor that tracks its transcription state for verification."""
    def __init__(self, chunk_duration_seconds=1.0):
        self.chunk_duration_seconds = chunk_duration_seconds
        self.timed_out = False
        self.updated = []

    def update(self, result):
        self.updated.append(result)

    @property
    def results(self):
        return self.updated[-1] if self.updated else None

    @property
    def audio_buffer(self):
        return np.array([0.0, 0.1], dtype=np.float32)

# Tests for the core logic of the Shared ASR (ParallelRealtimeASR)

def test_shared_asr_barrier_timeout_uses_max_chunk_duration(monkeypatch, run_test):
    """Verify that the barrier timeout adjusts based on the clients' chunk durations."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): return [("active", [])]

    # Patch the real model with our dummy to avoid actual model loading
    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        await shared_asr.register_processor("fast", DummyRuntimeProcessor(0.5))
        await shared_asr.register_processor("slow", DummyRuntimeProcessor(1.0))

    run_test(scenario())
    # Timeout is calculated as 2x the maximum chunk duration
    assert shared_asr._barrier_timeout_seconds() == 2.0


def test_shared_asr_excludes_processor_still_not_ready_after_partial_batch(monkeypatch, run_test):
    """Verify that unresponsive clients are timed out and cleaned up after a batch."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): return [("active", [])]

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("paused", paused)
        
        # Simulate 'paused' being an existing processor that has stopped sending data
        shared_asr._registered_pids["paused"].never_committed_flag = False
        await shared_asr.set_processor_ready("active")

        candidates = await shared_asr._timed_out_processor_candidates()
        assert candidates == {"paused"}

        claimed = await shared_asr._claim_ready_processors()
        assert set(claimed) == {"active"}

        await shared_asr._exclude_still_not_ready_processors(candidates)

        assert active.timed_out is False
        assert paused.timed_out is True
        assert "paused" not in shared_asr._registered_pids

    run_test(scenario())


def test_shared_asr_keeps_timed_out_candidate_if_it_recovers_before_post_batch_check(monkeypatch, run_test):
    """Verify that a processor isn't removed if it becomes ready just before the cleanup happens."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): return [("active", [])]

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        late = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("late", late)
        await shared_asr.set_processor_ready("active")

        # Capture candidates, then 'late' becomes ready just before cleanup
        candidates = await shared_asr._timed_out_processor_candidates()
        await shared_asr._claim_ready_processors()
        await shared_asr.set_processor_ready("late")
        await shared_asr._exclude_still_not_ready_processors(candidates)

        assert late.timed_out is False
        assert "late" in shared_asr._registered_pids

    run_test(scenario())


def test_shared_asr_wait_does_not_release_processor_omitted_from_partial_batch(monkeypatch, run_test):
    """Ensure processors omitted from a transcription batch don't accidentally return."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("paused", paused)
        shared_asr._registered_pids["paused"].never_committed_flag = False
        await shared_asr.set_processor_ready("active")

        await shared_asr.start()
        paused_wait = asyncio.create_task(shared_asr.wait("paused"))

        candidates = await shared_asr._timed_out_processor_candidates()
        await shared_asr._claim_ready_processors()
        # Mark transcription as done for 'active' only
        shared_asr._registered_pids["active"].transcription_event.set()

        await asyncio.sleep(0.01)
        assert not paused_wait.done()

        # Only after final cleanup should the 'paused' client return
        await shared_asr._exclude_still_not_ready_processors(candidates)
        await asyncio.wait_for(paused_wait, timeout=0.2)
        await asyncio.wait_for(shared_asr.stop(), timeout=0.2)

    run_test(scenario())


def test_shared_asr_transcribe_updates_claimed_processor_with_empty_result_when_asr_omits_it(monkeypatch, run_test):
    """Ensure that we properly clear states if the ASR engine misses a claimed processor."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        # Simulate ASR returning results only for the first processor
        def transcribe_parallel(self, audio_buffer): return [("active", [("ok",)])]

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        silent = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("silent", silent)

        claimed = {"active": active, "silent": silent}

        await shared_asr._transcribe_current_processors(claimed, waiting_time=0.0)

        assert active.updated == [[("ok",)]]
        assert silent.updated == [[]]
        assert shared_asr._registered_pids["active"].transcription_event.is_set()
        assert shared_asr._registered_pids["silent"].transcription_event.is_set()

    run_test(scenario())


def test_shared_asr_loop_transcribes_ready_processors_and_excludes_late_ones_on_timeout(monkeypatch, run_test):
    """Verify the main ASR loop behavior under mixed client readiness and timeouts."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): return [("active", [("ok",)])]

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("paused", paused)
        shared_asr._registered_pids["paused"].never_committed_flag = False
        shared_asr._barrier_timeout_seconds = lambda: 0.01

        await shared_asr.start()
        await shared_asr.set_processor_ready("active")

        active_wait = asyncio.create_task(shared_asr.wait("active"))
        await asyncio.wait_for(active_wait, timeout=0.5)

        assert active.updated == [[("ok",)]]
        assert paused.timed_out is True
        assert "paused" not in shared_asr._registered_pids

        await asyncio.wait_for(shared_asr.stop(), timeout=0.5)

    run_test(scenario())


def test_shared_asr_wait_reraises_if_loop_fails_after_timeout(monkeypatch, run_test):
    """Verify that exceptions in the ASR loop are propagated to waiting clients."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): raise RuntimeError("boom")

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        await shared_asr.register_processor("paused", paused)
        shared_asr._registered_pids["paused"].never_committed_flag = False
        shared_asr._barrier_timeout_seconds = lambda: 0.01

        await shared_asr.start()
        await shared_asr.set_processor_ready("active")

        with pytest.raises(RuntimeError, match="Shared ASR loop failed"):
            await asyncio.wait_for(shared_asr.wait("active"), timeout=0.5)

        await asyncio.wait_for(shared_asr.stop(), timeout=0.5)

    run_test(scenario())


def test_shared_asr_wait_reraises_if_loop_fails_before_timeout(monkeypatch, run_test):
    """Verify loop failure propagation even during normal (no-timeout) operation."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass
        def transcribe_parallel(self, audio_buffer): raise RuntimeError("boom")

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        active = DummyRuntimeProcessor()

        await shared_asr.register_processor("active", active)
        shared_asr._barrier_timeout_seconds = lambda: 10.0

        await shared_asr.start()
        await shared_asr.set_processor_ready("active")

        with pytest.raises(RuntimeError, match="Shared ASR loop failed"):
            await asyncio.wait_for(shared_asr.wait("active"), timeout=0.5)

        await asyncio.wait_for(shared_asr.stop(), timeout=0.5)

    run_test(scenario())


def test_shared_asr_wait_returns_when_processor_is_unregistered(monkeypatch, run_test):
    """Ensure wait() unblocks if the client disconnects manually."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("paused", paused)

        paused_wait = asyncio.create_task(shared_asr.wait("paused"))
        await asyncio.sleep(0.01)
        await shared_asr.unregister_processor("paused")
        await asyncio.wait_for(paused_wait, timeout=0.2)

    run_test(scenario())


def test_shared_asr_wait_returns_when_stopped_after_timeout(monkeypatch, run_test):
    """Ensure wait() unblocks when the shared ASR service is stopped."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")

    async def scenario():
        paused = DummyRuntimeProcessor()

        await shared_asr.register_processor("paused", paused)
        shared_asr._barrier_timeout_seconds = lambda: 0.01
        await shared_asr.start()

        shared_asr._registered_pids["paused"].transcription_event.clear()
        paused_wait = asyncio.create_task(shared_asr.wait("paused"))
        await asyncio.sleep(0.02)
        await asyncio.wait_for(shared_asr.stop(), timeout=0.2)
        await asyncio.wait_for(paused_wait, timeout=0.2)

    run_test(scenario())


def test_shared_asr_register_and_wait_fail_fast_after_loop_failure(monkeypatch, run_test):
    """Verify that we can't interact with the service if the loop has already crashed."""
    import src.parallel_whisper_online as pwo

    class DummyASR:
        def __init__(self, lan, modelsize=None, logfile=None): pass
        def warmup(self, filepath): pass

    monkeypatch.setattr(pwo, "MultiProcessingFasterWhisperASR", DummyASR)
    shared_asr = pwo.ParallelRealtimeASR(modelsize="tiny")
    shared_asr._loop_failure = RuntimeError("boom")

    async def scenario():
        with pytest.raises(RuntimeError, match="Shared ASR loop failed"):
            await shared_asr.register_processor("active", DummyRuntimeProcessor())

        shared_asr._registered_pids["active"] = pwo.RegisteredProcess(
            asr_processor=DummyRuntimeProcessor(),
        )
        with pytest.raises(RuntimeError, match="Shared ASR loop failed"):
            await shared_asr.wait("active")

    run_test(scenario())


# Tests for the ProcessorManager and its lifecycle context

def test_context_registers_after_two_queued_chunks(monkeypatch, run_test):
    """Verify registration only triggers once enough initial audio is buffered."""
    async def scenario():
        monkeypatch.setattr(stream_utils, "ParallelOnlineASRProcessor", DummyProcessor)
        shared_asr = DummySharedASR()
        manager = stream_utils.ProcessorManager(
            id="p1",
            shared_asr=shared_asr,
            logger=logging.getLogger("test"),
            server_logger=logging.getLogger("test"),
        )
        manager.audio_queue.put_nowait([0.0])
        manager.audio_queue.put_nowait([0.1])

        async with manager.context():
            assert shared_asr.registered == [("p1", manager.processor)]

        assert shared_asr.unregistered == ["p1"]

    run_test(scenario())


def test_context_registers_when_stream_closes_early(monkeypatch, run_test):
    """Ensure registration happens if the stream ends even with a small buffer."""
    async def scenario():
        monkeypatch.setattr(stream_utils, "ParallelOnlineASRProcessor", DummyProcessor)
        shared_asr = DummySharedASR()
        manager = stream_utils.ProcessorManager(
            id="p1",
            shared_asr=shared_asr,
            logger=logging.getLogger("test"),
            server_logger=logging.getLogger("test"),
        )
        manager.audio_queue.put_nowait([0.0])
        manager.mark_stream_closed()

        async with manager.context():
            assert shared_asr.registered == [("p1", manager.processor)]

        assert shared_asr.unregistered == ["p1"]

    run_test(scenario())


def test_context_reraises_body_exceptions(monkeypatch, run_test):
    """Verify that exceptions inside the context block are propagated normally."""
    async def scenario():
        monkeypatch.setattr(stream_utils, "ParallelOnlineASRProcessor", DummyProcessor)
        shared_asr = DummySharedASR()
        manager = stream_utils.ProcessorManager(
            id="p1",
            shared_asr=shared_asr,
            logger=logging.getLogger("test"),
            server_logger=logging.getLogger("test"),
        )
        manager.audio_queue.put_nowait([0.0])
        manager.audio_queue.put_nowait([0.1])

        with pytest.raises(RuntimeError, match="boom"):
            async with manager.context():
                raise RuntimeError("boom")

    run_test(scenario())
