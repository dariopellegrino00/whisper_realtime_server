import asyncio
import logging

import pytest

import src.server.stream_utils as stream_utils


class DummyProcessor:
    def __init__(self, asr, logger=None, **kwargs):
        self.asr = asr
        self.logger = logger
        self.kwargs = kwargs
        self.timed_out = False

    def init(self):
        return None


class DummySharedASR:
    def __init__(self):
        self.asr = object()
        self.registered = []
        self.unregistered = []

    async def register_processor(self, processor_id, processor):
        self.registered.append((processor_id, processor))

    async def unregister_processor(self, processor_id):
        self.unregistered.append(processor_id)

    async def wait(self):
        return None


def test_context_registers_after_two_queued_chunks(monkeypatch):
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

    asyncio.run(scenario())


def test_context_registers_when_stream_closes_early(monkeypatch):
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

    asyncio.run(scenario())


def test_context_reraises_body_exceptions(monkeypatch):
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

    asyncio.run(scenario())
