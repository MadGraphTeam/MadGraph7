#!/usr/bin/env python3
"""Serial, explicitly coordinated combined correctness gates; never timings."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

from prepare import HERE, ROOT, WORK, binding, sha, write
from build import sources


def validate(mode, destination):
    stdout = (destination / 'stdout.log').read_text()
    stderr = (destination / 'stderr.log').read_text()
    expected = 22 if mode == 'unit' else 1
    assert re.search(rf'test result: ok\. {expected} passed; 0 failed;', stdout)
    if mode == 'unit':
        mandatory = ['mp_transport_preserves_binary128_information',
            'mp_transport_preserves_binary128_finite_dyadics_and_sign',
            'mp_transport_preserves_complex_branch_lips',
            'allocation_tensor_matches_dense_values_precision_and_signed_zero',
            'allocation_tensor_nonfinite_inputs_match_original',
            'allocation_evaluation_rebinds_scratch_and_lanes_exactly',
            'borrowed_product_and_actual_zero_one_resets_preserve_state']
        for name in mandatory:
            assert re.search(r'test [^\n]*::' + name + r' \.\.\. ok', stdout), name
        convergence = [json.loads(line.split('NINJARS_CONVERGENCE ', 1)[1])
            for line in stdout.splitlines() if 'NINJARS_CONVERGENCE ' in line]
        assert len(convergence) == 1 and convergence[0]['passed']
        assert convergence[0]['production_bits'] == 256 and convergence[0]['reference_bits'] == 384
        return {'passed_tests': 22, 'inherited_tests': 18, 'added_tensor_tests': 4,
                'mandatory_transport_and_tensor_tests': mandatory, 'convergence': convergence[0]}
    if mode == 'golden':
        assert 'golden exact lanes=21;' in stderr
        assert sum(line.startswith('golden ') and ' lane ' in line for line in stderr.splitlines()) == 21
        return {'golden_lanes': 21, 'exact_dense_callback_oracle': True}
    if mode == 'callback-replay':
        assert 'captured exact requests=1392;' in stderr
        assert sum(line.startswith('captured request n=') for line in stderr.splitlines()) == 1392
        return {'records': 1392, 'exact_ascii_fields': 11136, 'exact_dense_callback_oracle': True}
    rows = [json.loads(line) for line in (destination / 'state-replay.jsonl').read_text().splitlines()]
    assert rows[-1] == {'complete': True, 'records': 1392, 'changed_records': 0,
                        'changed_fields': 0, 'failures': 0}
    expected_rows = [json.loads(line) for line in (WORK / 'actual-mp-requests.jsonl').read_text().splitlines()]
    assert len(rows) == 1393 and len(expected_rows) == 1392
    for index, (row, original) in enumerate(zip(rows[:-1], expected_rows)):
        assert row['record'] == index and row['status'] == 0 and row['changed_fields'] == []
        assert (row['n'], row['rank']) == (original['n'], original['rank'])
        assert row['actual'] == row['expected'] == original['output'] and len(row['actual']) == 8
    return {'records': 1392, 'exact_ascii_fields': 11136, 'state_replay': binding(destination / 'state-replay.jsonl')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['unit', 'golden', 'callback-replay', 'state-replay'])
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--authorized-run', required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT and args.authorized_run.strip()
    assert Path(os.environ['TMPDIR']).resolve().is_relative_to(ROOT)
    os.sched_setaffinity(0, {234})
    build_path = args.build.resolve()
    build = json.loads(build_path.read_text())
    assert build['status'] == 'compiled-not-executed' and build['exit_code'] == 0
    assert binding(HERE / 'provenance.json') == build['provenance']
    provenance = json.loads((HERE / 'provenance.json').read_text())
    assert sources() == build['sources'] == provenance['combined_files']
    for name, digest in build['sources'].items():
        assert sha(build_path.parent / 'source' / name) == digest
    original = {name: sha(ROOT / 'vendor/ninjars' / name) for name in provenance['original_files']}
    assert original == provenance['original_files']
    assert sha(WORK / 'actual-mp-requests.jsonl') == provenance['capture']['uncompressed_sha256']
    executable = ROOT / build['executable']['path']
    assert binding(executable) == build['executable']
    attempts = HERE / 'runs' / args.mode
    attempts.mkdir(parents=True, exist_ok=True)
    destination = attempts / f'attempt-{len(list(attempts.iterdir())) + 1}'
    destination.mkdir()
    command = [str(executable), '--test-threads=1', '--nocapture']
    tests = {'golden': 'tensor::capture_tests::all_golden_mp_inputs_match_original_numerator_state',
        'callback-replay': 'tensor::capture_tests::actual_madloop_requests_match_dense_expansions_exactly',
        'state-replay': 'transport_replay_tests::all_stored_requests_retain_outputs_after_transport_fix'}
    if args.mode != 'unit':
        command += ['--ignored', tests[args.mode], '--exact']
    overrides = {'RUST_BACKTRACE': '1', 'OMP_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'}
    if args.mode == 'callback-replay':
        overrides.update(NINJARS_TENSOR_REPLAY=str(WORK / 'actual-mp-requests.jsonl'), NINJARS_REPLAY_ALL='1')
    if args.mode == 'state-replay':
        overrides.update(NINJARS_TRANSPORT_REPLAY=str(WORK / 'actual-mp-requests.jsonl'),
            NINJARS_TRANSPORT_REPLAY_OUTPUT=str(destination / 'state-replay.jsonl'))
    record = {'schema': 'corrected-transport-allocation-execution-v1', 'mode': args.mode,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'running',
        'coordination': args.authorized_run, 'command': command, 'environment': overrides,
        'affinity': sorted(os.sched_getaffinity(0)), 'build': binding(build_path),
        'executable': binding(executable), 'controller': binding(Path(__file__).resolve()),
        'provenance': binding(HERE / 'provenance.json'),
        'installed_archive': binding(ROOT / 'HEPTools/ninjars/lib/libninjars.a'),
        'acceptance_evidence': False, 'production_changed': False}
    write(destination / 'started.json', record)
    start = time.monotonic()
    try:
        with (destination / 'stdout.log').open('x') as stdout, (destination / 'stderr.log').open('x') as stderr:
            completed = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **overrides), stdout=stdout, stderr=stderr)
        record['exit_code'] = completed.returncode
        assert completed.returncode == 0
        record['checks'] = validate(args.mode, destination)
        assert sources() == build['sources']
        assert binding(executable) == record['executable']
        assert binding(ROOT / 'HEPTools/ninjars/lib/libninjars.a') == record['installed_archive']
        assert {name: sha(ROOT / 'vendor/ninjars' / name) for name in original} == original
        record['status'] = 'passed'
    except BaseException as exc:
        record['status'] = 'failed'
        record['failure'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['raw_files'] = {path.name: binding(path) for path in sorted(destination.iterdir())
                               if path.is_file() and path.name != 'started.json'}
        write(destination / 'result.json', record)
    print(json.dumps({'status': record['status'], 'mode': args.mode,
        'result': str((destination / 'result.json').relative_to(ROOT)), 'checks': record['checks']}))


if __name__ == '__main__':
    main()
