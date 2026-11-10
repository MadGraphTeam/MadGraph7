#!/usr/bin/env python3
"""Verify portable observer evidence, full traces and optional retained live inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from decode_trace import load_trace
from run import rows

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    inventory = json.loads((HERE / 'evidence-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    assert actual == set(inventory['files']) | {'evidence-index.json'}
    for name, entry in inventory['files'].items():
        path = HERE / name
        assert not path.is_symlink() and '..' not in Path(name).parts
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], name
        assert not raw.startswith(b'\x7fELF') and path.suffix not in ('.pyc', '.o', '.mod', '.a', '.so')
    build = json.loads((HERE / 'attempt1/build.json').read_text())
    assert build['complete'] and build['protected_inputs_unchanged'] and not build['native_executed']
    analysis = json.loads((HERE / 'analysis1/summary.json').read_text())
    for name, digest in analysis['bindings'].items():
        assert sha(ROOT / name) == digest, name
    checks = json.loads((HERE / 'decoder-checks.json').read_text())
    assert checks['passed'] and not checks['native_execution']
    assert len(checks['exact_ieee_checks']) == 18 and len(checks['rejected_mutations']) == 20
    assert all(row['passed'] for row in checks['exact_ieee_checks'])
    assert all(row['passed'] for row in checks['rejected_mutations'])
    summaries = {}
    live = dict(build['protected_inputs_before'])
    live.update(build['artifacts_sha256'])
    for backend in (9, 8):
        directory = HERE / 'attempt1' / f'backend-{backend}'
        execution = json.loads((directory / 'execution.json').read_text())
        assert execution['complete'] and execution['all_original_rows_exact'] and execution['protected_inputs_unchanged']
        assert execution['protected_inputs_before'] == execution['protected_inputs_after']
        assert execution['runtime_sha256_before'] == execution['runtime_sha256_after']
        assert execution['returncode'] == 0 and execution['wall_seconds'] <= 180
        assert execution['cpu'] == 11 and not execution['formal_performance_eligible']
        assert sha(directory / 'observed.log') == execution['log_sha256']
        assert sha(directory / 'trace.dat') == execution['trace_sha256']
        assert rows((directory / 'observed.log').read_text()) == execution['reference_numeric_rows'] == execution['observed_numeric_rows']
        trace = load_trace(directory / 'trace.dat', directory / 'observed.log')
        assert trace['summary'] == analysis['trace_summaries'][str(backend)]
        assert [p['host_code'] for p in trace['points']] == [210 + backend, 420, 210 + backend]
        stress = trace['points'][1]
        assert Counter((c['ctmode'], c['group']) for c in stress['calls']) == Counter(
            (mode, group) for mode in (4, 5) for group in range(1, 117))
        assert all(c['numerical_summaries']['NUM_ORIGINAL']['all_finite'] for c in stress['calls'])
        summaries[str(backend)] = trace['summary']
        live.update(execution['protected_inputs_before'])
        live.update(execution['runtime_sha256_before'])
    if args.live:
        for name, digest in live.items():
            assert sha(ROOT / name) == digest, name
    print(json.dumps({'portable_files': len(inventory['files']), 'complete_traces': summaries,
                      'native_executed_by_verifier': False, 'live_inputs_checked': len(live) if args.live else 0,
                      'verification': 'pass'}, sort_keys=True))


if __name__ == '__main__':
    main()
