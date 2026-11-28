"""Shared console, rotating project-file and workflow logging."""

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from threading import RLock

from team_wtm.config import SETTINGS

_CONTEXT = ContextVar("team_wtm_logging_context", default=("-", "-"))
_CONFIGURATION_LOCK = RLock()


class _ContextFilter(logging.Filter):
    def filter(self, record):
        record.workflow, record.run_id = _CONTEXT.get()
        return True


def _formatter():
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s "
        "[workflow=%(workflow)s run=%(run_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    return formatter


class _WorkflowHandler(logging.Handler):
    """Route module messages to the current run without mixing concurrent calls."""

    def __init__(self):
        super().__init__()
        self.streams = {}
        self.addFilter(_ContextFilter())

    def emit(self, record):
        handler = self.streams.get(record.run_id)
        if handler is not None and record.levelno >= handler.level:
            handler.handle(record)

    def register(self, run_id, handler):
        with self.lock:
            self.streams[run_id] = handler

    def unregister(self, run_id):
        with self.lock:
            self.streams.pop(run_id).close()


_ROUTER = _WorkflowHandler()


def _package_logger():
    logger = logging.getLogger("team_wtm")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if _ROUTER not in logger.handlers:
        logger.addHandler(_ROUTER)
    return logger


def shutdown_logging():
    """Close handlers owned by this application while preserving active run logs."""
    with _CONFIGURATION_LOCK:
        logger = logging.getLogger("team_wtm")
        for handler in list(logger.handlers):
            if getattr(handler, "_team_wtm_output", False):
                logger.removeHandler(handler)
                handler.close()


def configure_logging(config=None, *, console=True):
    """Configure project logging at application startup; repeated calls replace owned outputs."""
    config = config or SETTINGS
    with _CONFIGURATION_LOCK:
        shutdown_logging()
        logger = _package_logger()
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
            encoding="utf-8",
        )
        handlers = [file_handler]
        if console:
            handlers.append(logging.StreamHandler(sys.stderr))
        for handler in handlers:
            handler.setLevel(config.log_level)
            handler.setFormatter(_formatter())
            handler.addFilter(_ContextFilter())
            handler._team_wtm_output = True
            logger.addHandler(handler)


@contextmanager
def workflow_logging(workflow, run_id, path, level):
    """Capture normal module loggers in one workflow's file."""
    with _CONFIGURATION_LOCK:
        _package_logger()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_formatter())
    _ROUTER.register(run_id, handler)
    token = _CONTEXT.set((workflow, run_id))
    try:
        yield logging.getLogger("team_wtm.runtime")
    finally:
        _CONTEXT.reset(token)
        _ROUTER.unregister(run_id)
