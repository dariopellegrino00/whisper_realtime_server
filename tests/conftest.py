import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

# --- Mocks and Utilities shared across tests ---


class AbortCalled(Exception):
    """Custom exception to catch gRPC context.abort() calls in tests."""

    def __init__(self, code, details):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class FakeContext:
    """Mocks the gRPC 'context' object passed to servicer methods."""

    async def abort(self, code, details):
        raise AbortCalled(code, details)


class AsyncIterator:
    """Helper to convert a list into an async iterator for streaming tests."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.fixture
def fake_context():
    """Returns a fake gRPC context for unit testing servicers."""
    return FakeContext()


@pytest.fixture
def abort_exception():
    """Provides the AbortCalled exception class to test for gRPC aborts."""
    return AbortCalled


# --- Shared Dummy Fixtures for Processor and ASR logic ---


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


@pytest.fixture
def dummy_processor_factory():
    """Returns a factory for DummyRuntimeProcessor."""
    return DummyRuntimeProcessor


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


@pytest.fixture
def dummy_shared_asr():
    """Returns an instance of DummySharedASR."""
    return DummySharedASR()


@pytest.fixture
def dummy_asr_cls_factory():
    """Returns a factory that creates a Dummy ASR class with configurable behavior."""

    def _create_dummy_asr(results=None, exc=None):
        class DummyASR:
            def __init__(self, lan=None, modelsize=None, logfile=None, **kwargs):
                pass

            def warmup(self, filepath):
                pass

            def transcribe_parallel(self, audio_buffer):
                if exc:
                    raise exc
                return results if results is not None else []

        return DummyASR

    return _create_dummy_asr


# --- Async Lifecycle Helpers ---


@asynccontextmanager
async def asr_lifecycle(shared_asr):
    """Context manager to handle start/stop of a ParallelRealtimeASR in tests."""
    await shared_asr.start()
    try:
        yield shared_asr
    finally:
        # Use wait_for to avoid hanging forever if stop() blocks
        await asyncio.wait_for(shared_asr.stop(), timeout=1.0)


@pytest.fixture
def run_asr_loop():
    """Provides the asr_lifecycle helper as a fixture."""
    return asr_lifecycle


def _run_async(coro):
    """Helper to run a coroutine in a dedicated event loop."""
    return asyncio.run(coro)


@pytest.fixture
def run_test():
    """Provides a way to run async scenarios inside standard sync tests."""
    return _run_async
