# SPDX-License-Identifier: 0BSD
"""Scoped capture of known, benign dependency log events (m-lit
ux-besteffort, USER 2026-11-11 items 7-8): returned as warning metadata
(:class:`pyamplicol.api.result_warnings.ResultWarning`) alongside the result
of the operation that triggered them, instead of being printed.

Two channels, each restricted to an explicit list of understood classes:

- Python Symbolica's own ``tracing`` subscriber writes its events to the
  process's standard output (file descriptor 1; its ``use_custom_logger``
  only drops them, it routes nothing to Python). During a capture fd 1 is a
  pipe read by a thread: a line of a known class (``SYMBOLICA_CLASSES``) is
  counted, every other byte is forwarded to the original standard output at
  once, unchanged (nothing else is muted; the forwarding is real time).
- ``ufo_model_loader`` (the UFO reader) reports through Python ``logging``:
  a filter on its logger counts and drops the records of its known classes
  (``LOADER_CLASSES``) during the capture; any other record passes.

Known classes and why they are benign (each root-caused):

- ``WARN symbolica::coefficient: Created infinity by raising 0 to the power
  of -1``: loop_qcd_qed_sm's ``aEW0 = 0 if aEW0M1 == 0 else 1/aEW0M1`` with
  the restriction's ``aEW0M1 = 0``; HEPKit's ``Model.from_json`` substitutes
  the zero into both branches of the lowered conditional, so the UNTAKEN
  branch normalises to an infinity; the conditional selects 0, and ``aEW0``
  is referenced by no other parameter, coupling or vertex: the infinity
  never reaches a value.
- ``Expression contains the following floating point values ...`` (and its
  advice line): a UFO float literal (``1.0``, ``4.0``) mapped exactly to the
  rational it denotes (no value changes).

The capture is process-wide for its duration (a file-descriptor level
redirect: other threads' standard output is forwarded too, a line at a
time), never nested (an inner capture adds to the outer one).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field

# Optional ANSI colour codes and the tracing timestamp before the level.
_ANSI = re.compile(rb"\x1b\[[0-9;]*m")
SYMBOLICA_CLASSES = (
    (
        "WARN symbolica::coefficient",
        re.compile(
            rb"WARN\s+symbolica::coefficient:\s+"
            rb"(Created infinity by raising 0 to the power of -1)"
        ),
    ),
)
LOADER_LOGGER = "UFOModelLoader"
LOADER_CLASSES = (
    re.compile(r"^Expression contains the following floating point values"),
    re.compile(r"^It is typically best for the UFO model to be adjusted"),
)


@dataclass
class Captured:
    """The events of one capture: per (source, kind) the first message and
    the count, in first-seen order."""

    where: str
    events: dict[tuple[str, str], list[object]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, source: str, kind: str, message: str) -> None:
        with self._lock:
            entry = self.events.get((source, kind))
            if entry is None:
                self.events[(source, kind)] = [message, 1]
            else:
                entry[1] = int(entry[1]) + 1  # type: ignore[call-overload]

    def warnings(self) -> tuple[object, ...]:
        """The captured events as ``ResultWarning`` records."""

        from pyamplicol.api.result_warnings import ResultWarning

        return tuple(
            ResultWarning(source, kind, str(message), int(count), self.where)  # type: ignore[arg-type]
            for (source, kind), (message, count) in self.events.items()
        )


_ACTIVE: list[Captured] = []
_GUARD = threading.Lock()


class _LoaderFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _ACTIVE:
            return True
        message = record.getMessage()
        for pattern in LOADER_CLASSES:
            if pattern.search(message):
                _ACTIVE[0].add("ufo-model-loader", "WARNING UFOModelLoader", message)
                return False
        return True


_FILTER = _LoaderFilter()


def _classify(line: bytes) -> tuple[str, str] | None:
    plain = _ANSI.sub(b"", line)
    for kind, pattern in SYMBOLICA_CLASSES:
        found = pattern.search(plain)
        if found is not None:
            return kind, found.group(1).decode(errors="replace")
    return None


def _forward(read_fd: int, out_fd: int, sink: Captured) -> None:
    """Read the pipe line by line; count the known classes, forward the rest
    to the original standard output (``out_fd``) unchanged."""

    pending = b""
    while True:
        try:
            chunk = os.read(read_fd, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            break
        pending += chunk
        *lines, pending = pending.split(b"\n")
        passed = []
        for line in lines:
            known = _classify(line)
            if known is None:
                passed.append(line + b"\n")
            else:
                sink.add("symbolica", known[0], known[1])
        if passed:
            os.write(out_fd, b"".join(passed))
    if pending:
        known = _classify(pending)
        if known is None:
            os.write(out_fd, pending)
        else:
            sink.add("symbolica", known[0], known[1])


@contextmanager
def capture(where: str) -> Iterator[Captured]:
    """Capture the known benign classes (module docstring) for the duration
    of the block; the yielded ``Captured`` holds them (``.warnings()``)."""

    with _GUARD:
        nested = bool(_ACTIVE)
        sink = _ACTIVE[0] if nested else Captured(where)
        if not nested:
            _ACTIVE.append(sink)
    if nested:
        yield sink
        return
    logger = logging.getLogger(LOADER_LOGGER)
    logger.addFilter(_FILTER)
    redirected = None
    try:
        try:
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            saved = os.dup(1)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 1)
            os.close(write_fd)
            reader = threading.Thread(
                target=_forward, args=(read_fd, saved, sink), daemon=True
            )
            reader.start()
            redirected = (saved, read_fd, reader)
        except (OSError, ValueError, AttributeError):
            # No file-descriptor level standard output (an embedded
            # interpreter): nothing is captured on this channel.
            redirected = None
        yield sink
    finally:
        if redirected is not None:
            saved, read_fd, reader = redirected
            with suppress(OSError, ValueError, AttributeError):
                sys.stdout.flush()
            os.dup2(saved, 1)  # closes the pipe's last write end
            reader.join()
            os.close(read_fd)
            os.close(saved)
        logger.removeFilter(_FILTER)
        with _GUARD:
            _ACTIVE.clear()


__all__ = ["LOADER_CLASSES", "SYMBOLICA_CLASSES", "Captured", "capture"]
