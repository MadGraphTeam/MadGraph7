#!/usr/bin/env python3
"""Recheck retained combined correctness without a live crate or engine."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(record):
    path = ROOT / record['path']
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], path
    return path


def summarize():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    build_path = HERE / 'builds/attempt-1/result.json'
    build = json.loads(build_path.read_text())
    assert build['status'] == 'compiled-not-executed' and build['exit_code'] == 0
    assert build['affinity'] == [234] and build['sources'] == provenance['combined_files']
    assert bound(build['provenance']) == HERE / 'provenance.json'
    for name, digest in provenance['original_files'].items():
        assert sha(HERE / 'original' / name) == digest
    for name, digest in build['sources'].items():
        assert sha(build_path.parent / 'source' / name) == digest
    for target, record in provenance['reused'].items():
        path = bound(record)
        assert sha(HERE / 'reused' / path.name) == record['sha256'] == build['sources'][target]
    assert build['sources']['src/tensor_dense_reference.rs'] == provenance['original_files']['src/tensor.rs']
    assert provenance['changed'] == ['src/tensor.rs']
    assert provenance['added'] == ['src/tensor_allocation_tests.rs', 'src/tensor_capture_tests.rs', 'src/tensor_dense_reference.rs']
    capture = ROOT / provenance['capture']['path']
    assert sha(capture) == provenance['capture']['sha256']
    with gzip.open(capture, 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == provenance['capture']['uncompressed_sha256']
    original = [json.loads(line) for line in raw.splitlines()]
    assert len(original) == 1392 and all(row['status'] == 0 and len(row['output']) == 8 for row in original)
    results = {}
    for mode in ['unit', 'golden', 'callback-replay', 'state-replay']:
        directory = HERE / 'runs' / mode / 'attempt-1'
        result = json.loads((directory / 'result.json').read_text())
        assert result['status'] == 'passed' and result['exit_code'] == 0 and result['mode'] == mode
        assert result['affinity'] == [234] and result['executable'] == build['executable']
        assert bound(result['build']) == build_path
        assert bound(result['provenance']) == HERE / 'provenance.json'
        bound(result['controller'])
        for file in result['raw_files'].values():
            bound(file)
        stdout, stderr = (directory / 'stdout.log').read_text(), (directory / 'stderr.log').read_text()
        tests = 22 if mode == 'unit' else 1
        assert re.search(rf'test result: ok\. {tests} passed; 0 failed;', stdout)
        if mode == 'unit':
            for name in result['checks']['mandatory_transport_and_tensor_tests']:
                assert re.search(r'test [^\n]*::' + name + r' \.\.\. ok', stdout)
            convergence = [json.loads(line.split('NINJARS_CONVERGENCE ', 1)[1])
                for line in stdout.splitlines() if 'NINJARS_CONVERGENCE ' in line]
            assert len(convergence) == 1 and convergence[0] == result['checks']['convergence']
            assert convergence[0]['passed'] and convergence[0]['production_bits'] == 256
            assert convergence[0]['reference_bits'] == 384
        elif mode in ['golden', 'callback-replay']:
            if mode == 'golden':
                label, count = 'golden exact lanes', 21
                assert sum(line.startswith('golden ') and ' lane ' in line for line in stderr.splitlines()) == count
            else:
                label, count = 'captured exact requests', 1392
                assert sum(line.startswith('captured request n=') for line in stderr.splitlines()) == count
            lines = [line for line in stderr.splitlines() if line.startswith(label + '=')]
            assert len(lines) == 1 and lines[0].startswith(f'{label}={count};')
            counts = ast.literal_eval(lines[0].split('counts=', 1)[1])
            expected = ({'mu': 24, 'numerator_values': 48, 't2': 1554, 't3': 1056} if mode == 'golden'
                else {'mu': 108, 'numerator_values': 2184, 't2': 47304, 't3': 36840})
            assert counts == expected
            results[mode + '-callback-counts'] = counts
        else:
            rows = [json.loads(line) for line in (directory / 'state-replay.jsonl').read_text().splitlines()]
            assert len(rows) == 1393 and rows[-1] == {'complete': True, 'records': 1392,
                'changed_records': 0, 'changed_fields': 0, 'failures': 0}
            for index, (row, old) in enumerate(zip(rows[:-1], original)):
                assert row['record'] == index and row['status'] == 0 and row['changed_fields'] == []
                assert (row['n'], row['rank']) == (old['n'], old['rank'])
                assert row['actual'] == row['expected'] == old['output']
        results[mode] = {'result_sha256': sha(directory / 'result.json'), 'checks': result['checks'],
                         'elapsed_seconds': result['elapsed_seconds']}
    pause = json.loads((HERE / 'execution-events/pause-20261020.json').read_text())
    resume = json.loads((HERE / 'execution-events/resume-20261020.json').read_text())
    assert resume['pause_sha256'] == sha(HERE / 'execution-events/pause-20261020.json')
    return {'schema': 'corrected-transport-allocation-summary-v1', 'passed': True,
        'production_revision': provenance['production_revision'], 'build_result_sha256': sha(build_path),
        'combined_tensor_sha256': build['sources']['src/tensor.rs'],
        'tests_passed': 22, 'inherited_tests': 18, 'added_tensor_tests': 4, 'golden_lanes': 21,
        'callback_replay_records': 1392, 'callback_complex_outputs': 86436,
        'state_replay_records': 1392, 'exact_ascii_fields_per_replay': 11136,
        'negative_zero_and_beyond_dp_transport_passed': True,
        'callback_value_precision_sign_exact': True, 'results': results,
        'build_elapsed_includes_explicit_user_pause': {'paused_utc': pause['utc'], 'resumed_utc': resume['utc']},
        'production_changed': False, 'acceptance_evidence': False, 'full_process_performance_measured': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    result = summarize()
    if args.write:
        with (HERE / 'summary.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    else:
        assert result == json.loads((HERE / 'summary.json').read_text())
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}))
