#!/usr/bin/env python3
"""Illustrative isolated-kernel timing; never an amplitude acceptance gate."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path.cwd()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/dp-terminal/packed-scan'
WORK = ROOT / '.tirust-build/dp-terminal/packed-scan'
COMPILER = shutil.which('gfortran')
CPU = int(os.environ.get('DP_TERMINAL_CPU', '9'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                            text=True, timeout=60)
    record = {'command': list(map(str, command)), 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode:
        raise RuntimeError(record)
    return record


records = []
for name, flags in [('default-o0', ['-O0']), ('o3', ['-O3'])]:
    directory = WORK / ('micro-' + name)
    directory.mkdir(exist_ok=True)
    executable = directory / 'microbenchmark'
    command = [COMPILER, *flags, '-ffree-line-length-none', '-J', directory,
               '-I', directory, REPORT / 'oracle.f90', REPORT / 'terminal.f90',
               REPORT / 'microbenchmark.f90', '-o', executable]
    build = run(command)
    calibration = run(['taskset', '-c', str(CPU), executable, '1000', '1'])
    rows = list(csv.DictReader(io.StringIO(calibration['stdout'])))
    slowest = max(float(row['seconds']) for row in rows)
    iterations = max(1000, min(250000, int(1000 * .15 / slowest)))
    measured = run(['taskset', '-c', str(CPU), executable, iterations, '7'])
    rows = list(csv.DictReader(io.StringIO(measured['stdout'])))
    ratios = []
    for pair in range(1, 8):
        timing = {int(row['algorithm']): float(row['seconds']) for row in rows
                  if int(row['pair']) == pair}
        ratios.append(timing[1] / timing[0])
    (REPORT / ('microbenchmark-' + name + '.csv')).write_text(measured['stdout'])
    record = {'name': name, 'flags': flags, 'build': build,
              'executable_sha256': digest(executable), 'calibration': calibration,
              'measured': measured, 'terminal_over_dense_paired_ratios': ratios,
              'median_terminal_over_dense_ratio': statistics.median(ratios)}
    records.append(record)
    print(name, 'median terminal/dense', statistics.median(ratios), flush=True)

record = {
    'scope': 'Synthetic full-shape DP rank1x1 kernel only, with caller IEEE_GET_FLAG cost; '
             'illustrative diagnostic, no amplitude performance claim or acceptance gate',
    'comparison': 'The untouched extracted dense kernel versus conservative guarded '
                  'diagonal terminal arithmetic. The prior QP optimization is not involved.',
    'inputs': 'Deterministic finite fractional complex components; IEEE_INEXACT already '
              'set, nearest rounding, all exception traps disabled; all calls admitted',
    'cpu': CPU, 'load_average_after': os.getloadavg(),
    'compiler': run([COMPILER, '--version']),
    'source_sha256': {name: digest(REPORT / name) for name in
                      ['oracle.f90', 'terminal.f90', 'microbenchmark.f90',
                       'run_microbenchmark.py']},
    'results': records,
}
(REPORT / 'microbenchmark-results.json').write_text(json.dumps(record, indent=2) + '\n')
