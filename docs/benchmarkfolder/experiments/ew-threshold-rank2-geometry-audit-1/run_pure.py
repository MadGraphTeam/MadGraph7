"""Guard native/process functions while testing/replaying closed rank2 bytes."""
import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import audit


def forbidden(*args,**kwargs):
    raise AssertionError("Native/process invocation forbidden in pure rank2 audit")


def run_tests():
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern="test_audit.py")
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError("Synthetic tests failed")
    return result.testsRun


@contextlib.contextmanager
def no_native():
    guards=[(subprocess,n) for n in ("Popen","run","call","check_call","check_output")]
    guards += [(ctypes,n) for n in ("CDLL","PyDLL")]
    guards += [(os,n) for n in dir(os) if n=="system" or n=="fork" or n.startswith(("exec","spawn","posix_spawn"))]
    with contextlib.ExitStack() as stack:
        for obj,name in guards:
            if callable(getattr(obj,name,None)):stack.enter_context(mock.patch.object(obj,name,forbidden))
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");parser.add_argument("--tests-only",action="store_true");args=parser.parse_args()
    with no_native():
        count=run_tests()
        if args.tests_only:
            print("PASS: %d pure tests; native/process invocation blocked"%count);return
        result=audit.evaluate();encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
        destination=audit.HERE/"summary.json"
        if args.check:
            if destination.read_text()!=encoded:raise ValueError("Closed rank2 summary replay mismatch")
        else:
            if destination.exists():raise ValueError("Refusing to overwrite closed summary")
            destination.write_text(encoded)
    print(json.dumps({"tests":count,"summary_sha256":audit.sha(destination),"cases":len(result["records"]),"native_execution":False,"check":args.check}))


if __name__=="__main__":main()
