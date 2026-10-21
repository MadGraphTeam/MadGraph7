#!/usr/bin/env python3
"""Verify retained software results and source hashes without running engines."""
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['schema'] == 'ninjars-final-pin-software-manifest-v1'
    for name, record in manifest['payloads'].items():
        path = HERE/name
        assert path.is_relative_to(HERE)
        assert path.stat().st_size == record['bytes'], name
        assert sha(path) == record['sha256'], name
    summary = json.loads((HERE/'summary.json').read_text())
    sources = json.loads(gzip.decompress((HERE/'source-hashes.json.gz').read_bytes()))
    for name, digest in sources['repository_files'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT)
        assert sha(path) == digest, name
    total = passed = skipped = 0
    for suite in summary['final_suites']:
        result = json.loads((HERE/suite['result']).read_text())
        assert result['suite_successful']
        assert not result['failures'] and not result['errors']
        assert len(result['test_ids']) == result['tests']
        assert result['relevant_source_unchanged']
        output = (HERE/suite['log']).read_text()
        footer = re.findall(r'Ran (\d+) tests? in ([0-9.]+)s', output)
        assert len(footer) == 1 and int(footer[0][0]) == result['tests']
        assert re.search(r'^OK(?: \(skipped=\d+\))?$',output,re.M)
        total += result['tests']
        skipped += len(result['skipped'])
        passed += result['tests']-len(result['skipped'])
    assert (total,passed,skipped) == (
        summary['tests_run'],summary['tests_passed'],summary['tests_skipped'])
    assert summary['software_passed']
    assert summary['actual_reduction_engine_evaluations'] == 0
    assert summary['correctness_gate_passed'] is False
    assert summary['performance_gate_passed'] is False
    print(json.dumps({'verified':True,'tests':total,'passed':passed,
        'skipped':skipped,'actual_reduction_engine_evaluations':0},sort_keys=True))

if __name__ == '__main__':
    main()
