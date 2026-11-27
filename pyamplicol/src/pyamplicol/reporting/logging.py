# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, TextIO

LOGGER_NAME = "pyamplicol"
DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_pyamplicol_cli_handler"
_NULL_MARKER = "_pyamplicol_null_handler"
_PREVIOUS_STATE_MARKER = "_pyamplicol_cli_previous_state"
# A record's user-facing label (``notice``): the CLI shows such INFO records
# as its own short lines at the default level.
NOTICE_ATTRIBUTE = "pyamplicol_notice"
_NOTICE_MARKER = "_pyamplicol_cli_notice_handler"


def notice(logger: logging.Logger, label: str, message: str, *args: object) -> None:
    """Log ``message`` at INFO as a user-facing notice labelled ``label``
    (e.g. "process", "widths"): an ordinary INFO record for library users;
    the CLI shows it at its default level as one short ``label  message``
    line (``configure_cli_logging``)."""

    logger.info(message, *args, extra={NOTICE_ATTRIBUTE: label})


class _NoticeFormatter(logging.Formatter):
    """``  label     message``: the CLI's own short notice lines."""

    def __init__(self, *, color: bool) -> None:
        super().__init__("%(message)s")
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        import shutil
        import textwrap

        label = str(getattr(record, NOTICE_ATTRIBUTE, ""))
        # Wrapped within the terminal at word boundaries (a path stays
        # whole), continuation lines under the message.
        width = shutil.get_terminal_size((100, 24)).columns - 10
        text = "\n".join(
            textwrap.wrap(
                super().format(record),
                width=max(width, 20),
                subsequent_indent=" " * 10,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        if self._color:
            try:
                colorama: Any = importlib.import_module("colorama")
            except ImportError:
                pass
            else:
                label = f"{colorama.Fore.CYAN}{label:<9}{colorama.Style.RESET_ALL}"
                return f"{label} {text}"
        return f"{label:<9} {text}"


class _CliFormatter(logging.Formatter):
    """Color terminal diagnostics while leaving ordinary informational logs alone."""

    def __init__(self, fmt: str, *, color: bool) -> None:
        super().__init__(fmt)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        label = getattr(record, NOTICE_ATTRIBUTE, None)
        if label:
            # A notice keeps its label in the logger format (--log-level
            # info / debug): "model: built-in-sm", not a bare value.
            record = logging.makeLogRecord(
                {**record.__dict__, "msg": f"{label}: {record.getMessage()}"}
                | {"args": None}
            )
        rendered = super().format(record)
        if not self._color:
            return rendered
        try:
            colorama: Any = importlib.import_module("colorama")
        except ImportError:
            return rendered
        if record.levelno >= logging.ERROR:
            foreground = colorama.Fore.RED
        elif record.levelno >= logging.WARNING:
            foreground = colorama.Fore.YELLOW
        else:
            return rendered
        return f"{foreground}{rendered}{colorama.Style.RESET_ALL}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or a named child without configuring logging."""

    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _ensure_null_handler() -> None:
    logger = get_logger()
    if any(getattr(handler, _NULL_MARKER, False) for handler in logger.handlers):
        return
    handler = logging.NullHandler()
    setattr(handler, _NULL_MARKER, True)
    logger.addHandler(handler)


def _normalize_level(level: int | str) -> int:
    if isinstance(level, bool):
        raise ValueError(f"invalid logging level {level!r}")
    if isinstance(level, int):
        return level
    if not isinstance(level, str):
        raise ValueError(f"invalid logging level {level!r}")
    normalized = logging.getLevelNamesMapping().get(level.upper())
    if normalized is None:
        raise ValueError(f"unknown logging level {level!r}")
    return normalized


def configure_cli_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    color: bool = False,
) -> logging.Logger:
    """Install one package-local CLI handler, leaving the root logger untouched."""

    selected_level = _normalize_level(level)
    logger = get_logger()
    handler = next(
        (
            candidate
            for candidate in logger.handlers
            if getattr(candidate, _HANDLER_MARKER, False)
            and not getattr(candidate, _NOTICE_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr if stream is None else stream)
        setattr(handler, _HANDLER_MARKER, True)
        setattr(
            handler,
            _PREVIOUS_STATE_MARKER,
            (logger.level, logger.propagate),
        )
        logger.addHandler(handler)
    elif stream is not None and isinstance(handler, logging.StreamHandler):
        handler.setStream(stream)
    handler.setLevel(selected_level)
    handler.setFormatter(_CliFormatter(fmt, color=color))
    for candidate in tuple(logger.handlers):
        if getattr(candidate, _NOTICE_MARKER, False):
            logger.removeHandler(candidate)
            candidate.close()
    effective = selected_level
    if selected_level > logging.INFO:
        # Above INFO the user-facing notices still show, as the CLI's own
        # short lines (never in the logger format).
        notices = logging.StreamHandler(getattr(handler, "stream", sys.stderr))
        setattr(notices, _NOTICE_MARKER, True)
        setattr(notices, _HANDLER_MARKER, True)
        notices.setLevel(logging.INFO)
        notices.addFilter(
            lambda record: (
                record.levelno == logging.INFO and hasattr(record, NOTICE_ATTRIBUTE)
            )
        )
        notices.setFormatter(_NoticeFormatter(color=color))
        logger.addHandler(notices)
        effective = logging.INFO
    logger.setLevel(effective)
    logger.propagate = False
    return logger


def reset_cli_logging() -> None:
    """Remove CLI handlers and restore the embedding application's logger state."""

    logger = get_logger()
    previous_state: tuple[int, bool] | None = None
    for handler in tuple(logger.handlers):
        if not getattr(handler, _HANDLER_MARKER, False):
            continue
        candidate_state = getattr(handler, _PREVIOUS_STATE_MARKER, None)
        if (
            previous_state is None
            and isinstance(candidate_state, tuple)
            and len(candidate_state) == 2
            and isinstance(candidate_state[0], int)
            and isinstance(candidate_state[1], bool)
        ):
            previous_state = candidate_state
        logger.removeHandler(handler)
        handler.close()
    if previous_state is not None:
        logger.setLevel(previous_state[0])
        logger.propagate = previous_state[1]
    _ensure_null_handler()


_ensure_null_handler()


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "LOGGER_NAME",
    "NOTICE_ATTRIBUTE",
    "configure_cli_logging",
    "get_logger",
    "notice",
    "reset_cli_logging",
]
