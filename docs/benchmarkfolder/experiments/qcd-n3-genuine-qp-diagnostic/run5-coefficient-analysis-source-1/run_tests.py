"""Run pure synthetic tests under explicit native/process invocation guards."""
import contextlib
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


def forbidden(*args, **kwargs):
    raise AssertionError("Native/process invocation forbidden in this pure audit")


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    guards = [(subprocess, n) for n in ("Popen", "run", "call", "check_call", "check_output")]
    guards += [(ctypes, n) for n in ("CDLL", "PyDLL")]
    guards += [(os, n) for n in dir(os) if n == "system" or n == "fork" or n.startswith(("exec", "spawn", "posix_spawn"))]
    with contextlib.ExitStack() as stack:
        for obj, name in guards:
            if callable(getattr(obj, name, None)):
                stack.enter_context(mock.patch.object(obj, name, forbidden))
        suite = unittest.defaultTestLoader.discover(str(root), pattern="test_summary.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("Native/process guard active; all fixtures synthetic; no run5/raw/native execution.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
