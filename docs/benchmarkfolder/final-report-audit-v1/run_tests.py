#!/usr/bin/env python3
"""Independent final reader/report regressions; no reduction engine execution."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
FILES = [
    'tools/ninjars_report_v2.py', 'tools/ninjars_supplement_v2.py',
    'tools/ninjars_precision_v2.py', 'tools/ninjars_precision_driver.c',
    'tests/unit_tests/loop/test_ninjars_report_v2.py',
    'tests/unit_tests/various/test_ninjars_supplement.py',
    'tests/unit_tests/various/test_ninjars_precision.py',
    'docs/benchmarkfolder/policy-v2/template/benchmark.typ',
    'docs/benchmarkfolder/style.typ',
    'docs/benchmarkfolder/policy-v2/final-regressions-v1/verify_validation.py',
    'docs/benchmarkfolder/policy-v2/final-regressions-v1/validation.json',
    'docs/benchmarkfolder/policy-v2/final-report-pipeline-v1/validation.json',
    'docs/benchmarkfolder/policy-v2/final-report-pipeline-v1/manifest.json',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.sched_getaffinity(0) == {5}
    before = {name: sha(ROOT / name) for name in FILES}
    command = [sys.executable, '-B', '-m', 'unittest', '-v',
               'tests.unit_tests.loop.test_ninjars_report_v2',
               'tests.unit_tests.various.test_ninjars_precision',
               'tests.unit_tests.various.test_ninjars_supplement']
    started = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE / 'unittest.stdout').open('x') as out, (HERE / 'unittest.stderr').open('x') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                   start_new_session=True)
        code = process.wait(timeout=300)
    elapsed = time.monotonic() - started
    after = {name: sha(ROOT / name) for name in FILES}
    assert before == after, 'Frozen source changed during independent test run'
    text = (HERE / 'unittest.stderr').read_text()
    footer = re.findall(r'^Ran (\d+) tests? in ([0-9.]+)s$', text, re.M)
    assert len(footer) == 1
    children = []
    for line in subprocess.check_output(['ps', '-eo', 'pid=,pgid=,stat=,args='], text=True).splitlines():
        row = line.split(None, 3)
        if len(row) == 4 and int(row[1]) == process.pid:
            children.append(line.strip())
    result = {
        'schema': 'independent-final-report-audit-v1',
        'scope': 'Report/precision/supplement readers, negative proof controls and synthetic PDF layout only; no physical acceptance claim',
        'command': command, 'cpu': 5, 'started_utc': stamp,
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'wall_seconds': elapsed, 'exit_code': code,
        'tests_run': int(footer[0][0]), 'reported_seconds': float(footer[0][1]),
        'passed': code == 0 and re.search(r'^OK$', text, re.M) is not None,
        'source_sha256': before, 'source_unchanged_during_tests': True,
        'test_process_id': process.pid, 'remaining_process_group_members': children,
        'all_children_exited': not children,
        'builds_of_numerical_code': 0, 'reduction_engine_calls': 0,
        'typst_compilation': 'Synthetic fixtures and byte-reproducibility controls in report suite',
        'logs': {name: sha(HERE / name) for name in ('unittest.stdout', 'unittest.stderr')},
        'final_goal_claimed': False,
    }
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('source_sha256', 'command', 'logs')}, indent=2))
    assert result['passed'] and result['tests_run'] == 55 and result['all_children_exited']


if __name__ == '__main__':
    main()
