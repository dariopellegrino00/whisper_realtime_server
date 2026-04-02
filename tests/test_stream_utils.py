import logging
from pathlib import Path

from swim.transports.grpc.stream_utils import build_logger_name, setup_stream_logger


def _close_logger_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def test_setup_stream_logger_without_per_stream_files_uses_propagation_only():
    stream_id = "stream-no-file"
    logger = setup_stream_logger(
        stream_id,
        level=logging.INFO,
        log_every_processor=False,
    )

    try:
        assert logger.name == build_logger_name("stream", stream_id)
        assert logger.level == logging.INFO
        assert logger.propagate is True
        assert logger.handlers == []
    finally:
        _close_logger_handlers(logger)


def test_setup_stream_logger_with_per_stream_files_creates_file_handler(tmp_path):
    stream_id = "stream-with-file"
    logger = setup_stream_logger(
        stream_id,
        level=logging.INFO,
        log_every_processor=True,
        log_folder=str(tmp_path),
    )

    try:
        assert logger.name == build_logger_name("stream", stream_id)
        assert logger.level == logging.INFO
        assert logger.propagate is True

        file_handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

        logger.info("test log line")
        for handler in file_handlers:
            handler.flush()

        log_path = Path(file_handlers[0].baseFilename)
        assert log_path.exists()
        assert log_path.parent == tmp_path
        assert log_path.name.endswith(f"_{logger.name.replace('.', '_')}.log")
        assert "test log line" in log_path.read_text(encoding="utf-8")
    finally:
        _close_logger_handlers(logger)
