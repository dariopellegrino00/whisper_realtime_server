import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

# Ignore playground scripts during test collection
collect_ignore_glob = ["playground/*"]

# The following mocks and utilities are shared across multiple test files 
# to keep the actual test code clean and focused.

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

# Async helper fixtures for running coroutines in standard sync tests

def _run_async(coro):
    """Helper to run a coroutine in a dedicated event loop."""
    return asyncio.run(coro)

@pytest.fixture
def run_test():
    """Provides a way to run async scenarios inside standard sync tests."""
    return _run_async
