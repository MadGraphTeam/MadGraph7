# SPDX-License-Identifier: 0BSD
"""The fd-1 capture of known dependency log lines (m-lit ux-besteffort;
ux-polish item 5): known lines are counted, every other byte is forwarded
unchanged and promptly (a carriage-return progress line at once, any other
partial line once the pipe is quiet), and leaving the capture never waits
indefinitely for a descendant that keeps the redirected descriptor."""

from __future__ import annotations

import os
import subprocess
import sys
import time

from pyamplicol._internal import symbolica_log

KNOWN = (
    b"\x1b[33m WARN\x1b[0m symbolica::coefficient: "
    b"Created infinity by raising 0 to the power of -1\n"
)


def test_known_lines_are_counted_and_the_rest_forwarded(capfd):
    with symbolica_log.capture("generate") as sink:
        os.write(1, b"before\n" + KNOWN + KNOWN + b"after\n")
    out = capfd.readouterr().out
    assert out == "before\nafter\n"
    (warning,) = sink.warnings()
    assert (warning.kind, warning.count, warning.where) == (
        "symbolica::coefficient",
        2,
        "generate",
    )


def test_partial_lines_are_forwarded_promptly(capfd):
    with symbolica_log.capture("check"):
        os.write(1, b"progress 10%\r")
        time.sleep(0.3)
        assert capfd.readouterr().out == "progress 10%\r"
        os.write(1, b"no newline yet")
        time.sleep(0.3)
        assert capfd.readouterr().out == "no newline yet"
        os.write(1, b" and the rest\n")
    assert capfd.readouterr().out == " and the rest\n"


def test_a_lingering_descendant_does_not_block_the_exit(capfd):
    start = time.monotonic()
    with symbolica_log.capture("generate"):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time; time.sleep(4); sys.stdout.write('late\\n')",
            ]
        )
    left = time.monotonic() - start
    try:
        assert left < symbolica_log.LINGER_SECONDS + 1.5
    finally:
        child.wait(timeout=30)
    time.sleep(0.3)
    # The descendant's output is still forwarded, unchanged.
    assert "late\n" in capfd.readouterr().out
