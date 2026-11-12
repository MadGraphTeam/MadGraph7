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

A third channel needs no list: ``ufo_source_warnings(source)``, entered
wherever pyAmpliCol imports or compiles a UFO model's Python, records the
Python warnings attributed to a file of that model (for example the
``SyntaxWarning: invalid escape sequence '\\l'`` Python >= 3.12 raises when it
compiles a TeX name such as ``'\\lambda_{HV}'`` in a non-raw string) into the
active capture as ``source="ufo-model"`` (``kind`` = the warning category,
``message`` = ``file:line: text``), or, outside a capture, logs them at
DEBUG on ``pyamplicol.models``; every other warning raised in the block is
re-emitted unchanged after it, under the caller's filters.

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
time), never nested (an inner capture adds to the outer one). A partial
line is forwarded promptly: everything up to a carriage return at once (a
progress line), any other unterminated text once the pipe has been quiet
for ``PARTIAL_SECONDS``. Leaving the capture never waits indefinitely: a
descendant process that inherited the redirected descriptor and outlives the
block keeps the pipe open, so after ``LINGER_SECONDS`` the reader is left
to forward that process's output unchanged (nothing more is captured) and
closes its descriptors itself at the end of the pipe.
"""

from __future__ import annotations

import logging
import os
import re
import select
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

# Optional ANSI colour codes and the tracing timestamp before the level.
_ANSI = re.compile(rb"\x1b\[[0-9;]*m")
SYMBOLICA_CLASSES = (
    (
        "symbolica::coefficient",
        re.compile(
            rb"WARN\s+symbolica::coefficient:\s+"
            rb"(Created infinity by raising 0 to the power of -1)"
        ),
    ),
)
# A partial line is forwarded once the pipe is quiet this long (seconds).
PARTIAL_SECONDS = 0.05
# How long leaving a capture waits for the end of the pipe (seconds).
LINGER_SECONDS = 2.0
_CHUNK = 65536
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
                _ACTIVE[0].add("ufo-model-loader", "UFOModelLoader", message)
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


def _forward(read_fd: int, out_fd: int, sink: Captured, done: threading.Event) -> None:
    """Read the pipe until its end: count the lines of the known classes
    (while the capture lasts, until ``done``), forward every other byte to
    the original standard output (``out_fd``) unchanged; a partial line is
    forwarded up to its last carriage return at once and in full once the
    pipe is quiet (module docstring). Closes both descriptors at the end."""

    def emit(line: bytes) -> bytes:
        known = None if done.is_set() else _classify(line)
        if known is None:
            return line
        sink.add("symbolica", known[0], known[1])
        return b""

    def quiet() -> bool:
        try:
            ready, _, _ = select.select([read_fd], [], [], PARTIAL_SECONDS)
        except (OSError, ValueError):
            return False
        return not ready

    pending = b""
    try:
        while True:
            if pending and (len(pending) >= _CHUNK or quiet()):
                os.write(out_fd, emit(pending))
                pending = b""
            try:
                chunk = os.read(read_fd, _CHUNK)
            except OSError:
                chunk = b""
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b"\n")
            passed = [emit(line + b"\n") for line in lines]
            cut = pending.rfind(b"\r")
            if cut >= 0:
                passed.append(emit(pending[: cut + 1]))
                pending = pending[cut + 1 :]
            if any(passed):
                os.write(out_fd, b"".join(passed))
        if pending:
            os.write(out_fd, emit(pending))
    except OSError:
        pass
    finally:
        for fd in (read_fd, out_fd):
            with suppress(OSError):
                os.close(fd)


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
            done = threading.Event()
            reader = threading.Thread(
                target=_forward, args=(read_fd, saved, sink, done), daemon=True
            )
            reader.start()
            redirected = (saved, reader, done)
        except (OSError, ValueError, AttributeError):
            # No file-descriptor level standard output (an embedded
            # interpreter): nothing is captured on this channel.
            redirected = None
        yield sink
    finally:
        if redirected is not None:
            saved, reader, done = redirected
            with suppress(OSError, ValueError, AttributeError):
                sys.stdout.flush()
            os.dup2(saved, 1)  # closes this process's write end of the pipe
            # The end of the pipe, unless a descendant still holds a write
            # end: then the reader forwards its output on its own, capturing
            # nothing more, and closes the descriptors when the pipe ends.
            reader.join(LINGER_SECONDS)
            done.set()
        logger.removeFilter(_FILTER)
        with _GUARD:
            _ACTIVE.clear()


_MODEL_LOG = logging.getLogger("pyamplicol.models")


def _inside(filename: object, root: Path, directory: bool) -> bool:
    """Whether ``filename`` is a file of the model: inside its directory
    ``root``, or ``root`` itself when the model source is one file."""

    try:
        path = Path(str(filename)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return path.is_relative_to(root) if directory else path == root


@contextmanager
def ufo_source_warnings(source: os.PathLike[str] | str) -> Iterator[None]:
    """Record the Python warnings attributed to a file of the UFO model
    ``source`` (a directory: any file inside it; a single file: that file
    only) raised while the block imports or compiles it, instead of printing
    them (module docstring). Scoped: the block runs under
    ``warnings.catch_warnings(record=True)`` with the caller's filters
    unchanged (an ``error`` filter still raises); the warnings of other files
    are re-emitted after the block, also when it raises. Each compile counts:
    a model compiled twice (an import and a source check) reports each of its
    warnings twice. ``catch_warnings`` swaps process-wide state, so model
    loads are not meant to overlap in threads (they do not today)."""

    path = Path(source).expanduser().resolve()
    directory = path.is_dir()
    recorded: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            try:
                yield
            finally:
                recorded = list(caught)
    finally:
        # After the caller's filters are restored, also when the block
        # raises: re-emitted warnings meet the caller's own filters.
        _dispatch(recorded, path, directory)


def _dispatch(
    recorded: list[warnings.WarningMessage], path: Path, directory: bool
) -> None:
    with _GUARD:
        sink = _ACTIVE[0] if _ACTIVE else None
    name = path.name if directory else path.parent.name
    for item in recorded:
        if not _inside(item.filename, path, directory):
            # Re-emitted at the original location. The original module's
            # warning registry is not recorded by catch_warnings, so a
            # 'default' / 'once' filter deduplicates against the module
            # derived from the file name (warn_explicit's own default).
            warnings.warn_explicit(
                item.message,
                item.category,
                item.filename,
                item.lineno,
                source=item.source,
            )
            continue
        kind = item.category.__name__
        message = f"{Path(item.filename).name}:{item.lineno}: {item.message}"
        if sink is not None:
            sink.add("ufo-model", kind, message)
        else:
            _MODEL_LOG.debug("UFO model %s: %s: %s", name, kind, message)


__all__ = [
    "LINGER_SECONDS",
    "LOADER_CLASSES",
    "PARTIAL_SECONDS",
    "SYMBOLICA_CLASSES",
    "Captured",
    "capture",
    "ufo_source_warnings",
]
