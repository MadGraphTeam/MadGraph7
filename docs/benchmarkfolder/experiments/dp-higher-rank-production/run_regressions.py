#!/usr/bin/env python3
"""Record focused source and existing QP/polynomial regression checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = (
    'madgraph/loop/dp_terminal_proof.py',
    'madgraph/loop/loop_exporters.py',
    'madgraph/various/q_polynomial.py',
    'tests/unit_tests/loop/test_dp_terminal_wavefunctions.py',
    'tests/unit_tests/loop/test_dp_terminal_export.py',
    'tests/unit_tests/various/test_dp_terminal_compiled.py',
)
TESTS = (
    'tests/unit_tests/loop/test_dp_terminal_wavefunctions.py',
    'tests/unit_tests/loop/test_dp_terminal_export.py',
    'tests/unit_tests/loop/test_terminal_wavefunctions.py',
    'tests/unit_tests/various/test_q_polynomial.py',
)


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


before = hashes()
command = [sys.executable, '-B', '-m', 'pytest', '-q', *TESTS]
start = time.monotonic()
with (HERE/'regressions.log').open('w') as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
record = {'command': command, 'returncode': result.returncode,
          'source_sha256_before': before, 'source_sha256_after': hashes(),
          'elapsed_seconds': time.monotonic()-start,
          'scope': 'Source proof/exporter and existing polynomial/QP tests; not a performance measurement.'}
record['sources_unchanged'] = before == record['source_sha256_after']
(HERE/'regressions.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
print(json.dumps({'returncode': result.returncode, 'sources_unchanged': record['sources_unchanged']}))
sys.exit(result.returncode if record['sources_unchanged'] else 1)
