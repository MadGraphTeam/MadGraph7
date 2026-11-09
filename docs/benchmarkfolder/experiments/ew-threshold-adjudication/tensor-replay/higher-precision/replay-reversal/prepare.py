#!/usr/bin/env python3
"""Frozen source bindings for the derived reversal 512-bit replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[6]
TRACE=ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication'
WORK=ROOT/'.tirust-build/ew-threshold-512-replay-reversal'
RUNTIME=ROOT/'.tirust-build/ew-threshold-512/runtime'
spec=importlib.util.spec_from_file_location('captured_reversal_inputs',ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/prepare.py')
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
extract=original.extract
def sources():
    data=json.loads((REPORT/'source-manifest.json').read_text())
    for path,expected in data['sources_and_link_inputs'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    return data
