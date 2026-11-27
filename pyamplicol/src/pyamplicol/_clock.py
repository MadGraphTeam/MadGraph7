# SPDX-License-Identifier: 0BSD
"""The command's clock: when the pyAmpliCol command the user waits for
started, portable (no ``/proc``). ``IMPORTED`` is the first import of the
package (the interpreter's own start before it is a few tens of ms);
``cli.main`` marks it as the start of a command-line run, an in-process
``run_cli`` starts the clock at its own entry, and ``command_start`` is
what a time budget (``profile --completion-time``) counts from."""

from __future__ import annotations

import time

IMPORTED = time.perf_counter()
_entry: list[float | None] = [None]
_start: list[float | None] = [None]


def mark_entry(at: float) -> None:
    """The next command starts at ``at`` (``time.perf_counter`` seconds)."""

    _entry[0] = at


def begin_command() -> None:
    """A command starts now, or at the time ``mark_entry`` gave."""

    _start[0] = _entry[0] if _entry[0] is not None else time.perf_counter()
    _entry[0] = None


def end_command() -> None:
    """The running command is over."""

    _start[0] = None


def command_start() -> float | None:
    """When the running command started (None: no command, e.g. the
    library called directly)."""

    return _start[0]


__all__ = ["IMPORTED", "begin_command", "command_start", "end_command", "mark_entry"]
