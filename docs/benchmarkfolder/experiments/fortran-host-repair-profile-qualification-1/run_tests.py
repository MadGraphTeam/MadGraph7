#!/usr/bin/env python3
"""Standard-library capsule tests; no compiler, loader or model execution."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
import test_verify
import test_replay_measurements


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-index');args=parser.parse_args()
    test_verify.EXPECTED_INDEX=args.expected_index
    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromModule(test_verify),
                             unittest.defaultTestLoader.loadTestsFromModule(test_replay_measurements)))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    outcome={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
             'errors':len(result.errors),'failures':len(result.failures),'optimized_python':not __debug__,
             'scope':'Portable source/data replay, clean relocation, corruption and explicit guard probes; no native numerical work.',
             'native_execution':False,'details':stream.getvalue()}
    print(json.dumps(outcome,indent=2,sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
