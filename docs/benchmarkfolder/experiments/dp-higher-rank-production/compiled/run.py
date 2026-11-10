#!/usr/bin/env python3
"""Record bounded production compiled tests; contains no timing benchmark."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from tests.unit_tests.various import test_dp_terminal_compiled as cases

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
NAME = sys.argv[1] if len(sys.argv) > 1 else 'qualification'
SOURCES = (
    'tests/unit_tests/various/test_dp_terminal_compiled.py',
    'madgraph/various/q_polynomial.py',
    'madgraph/loop/loop_exporters.py',
    'madgraph/iolibs/template_files/loop_optimized/polynomial.inc',
    'docs/benchmarkfolder/experiments/dp-terminal/oracle.f90',
    'docs/benchmarkfolder/experiments/dp-terminal/guard-driver.f90',
    'docs/benchmarkfolder/experiments/dp-terminal/trap-driver.f90',
    'docs/benchmarkfolder/experiments/dp-higher-rank/kernels/public.f',
)


def hashes():
    return {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


original_run = subprocess.run
before = hashes()
records = []


def recorded_run(command, *args, **kwargs):
    started = time.monotonic()
    result = original_run(command, *args, **kwargs)
    record = dict(command=list(map(str, command)), cwd=str(kwargs.get('cwd', ROOT)),
                  returncode=result.returncode, stdout=result.stdout,
                  stderr=result.stderr, elapsed_seconds=time.monotonic()-started)
    if command[0] == 'gfortran':
        cwd = Path(kwargs['cwd'])
        record['source_sha256'] = {
            p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(cwd.iterdir())
            if p.suffix in ('.f', '.f90', '.inc')}
    records.append(record)
    with (HERE/(NAME+'-commands.jsonl')).open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    return result


(HERE/(NAME+'-commands.jsonl')).write_text('')
metadata = dict(scope='Production compiled contracts; no performance evidence',
                argv=sys.argv, cwd=str(ROOT), affinity=sorted(os.sched_getaffinity(0)),
                python=sys.version, source_sha256_before=before,
                compiler=original_run(['gfortran', '--version'], text=True,
                                      capture_output=True, check=True).stdout)
subprocess.run = recorded_run
started = time.monotonic()
with (HERE/(NAME+'.log')).open('w') as stream:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(cases.DPTerminalCompiledTest)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
metadata.update(passed=result.wasSuccessful(), tests=result.testsRun,
                failures=len(result.failures), errors=len(result.errors),
                elapsed_seconds=time.monotonic()-started,
                source_sha256_after=hashes(), subprocess_count=len(records))
metadata['sources_unchanged'] = before == metadata['source_sha256_after']
(HERE/(NAME+'-summary.json')).write_text(json.dumps(metadata, indent=2)+'\n')
with (HERE/(NAME+'.log')).open('a') as stream:
    stream.write('\nSOURCE PROVENANCE (SHA256)\n')
    stream.write(json.dumps({key: metadata[key] for key in
        ('source_sha256_before', 'source_sha256_after', 'sources_unchanged')}, indent=2)+'\n')
print(json.dumps({key: metadata[key] for key in
                  ('passed', 'tests', 'failures', 'errors', 'subprocess_count',
                   'sources_unchanged', 'elapsed_seconds')}), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
