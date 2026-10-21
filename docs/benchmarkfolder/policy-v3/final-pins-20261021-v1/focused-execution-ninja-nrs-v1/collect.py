#!/usr/bin/env python3
"""New-runtime fresh/persistent parity using unchanged representative workers."""
import argparse
import datetime
import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
HERE = ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1"
SPECIFICATION = Path(__file__).with_name("specification-held.json")
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

CPU = 190
COLLECTIONS = (
    ('complex', ['control-complex'], [6, 8], 'docs/benchmarkfolder/points'),
    ('mixed-ninja', ['control-mixed'], [6], 'docs/benchmarkfolder/points'),
    ('near-singular-nrs', ['control-forced-qp'], [8],
     'docs/benchmarkfolder/policy-v2/representative-point-streams/near-singular'),
)


@contextmanager
def owned_collection(report, label):
    def interrupted(signum, _frame):
        raise SystemExit('Focused collection interrupted by signal ' + str(signum))
    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    try:
        for sig in signals:
            signal.signal(sig, interrupted)
        with pair.owned_preparation(report, label=label, fresh_cpu=CPU):
            yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)



def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def production_sources():
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', *bench.GENERATOR_PATHS, 'vendor'],
                                    cwd=ROOT).decode().split('\0')
    return {name: workers.digest(ROOT / name) if (ROOT / name).is_file() else None
            for name in names if name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--execute-numerical', action='store_true')
    args = parser.parse_args()
    specification = read(SPECIFICATION)
    for record in specification['bound_inputs']:
        policy.check_file(record)
    generation = HERE / 'generation'
    controls = HERE / 'controls-generation'
    freeze = read(generation / 'executed-freeze.json')
    core_status, control_status = read(generation / 'status.json'), read(controls / 'status.json')
    expected_handoff = {'root_authorized': True, 'phase': 'final-pins-focused-ninja-nrs-v1', 'cpu': CPU,
        'source_sha256': workers.digest(Path(__file__)),
        'specification_sha256': workers.digest(SPECIFICATION),
        'freeze_sha256': workers.digest(generation / 'executed-freeze.json'),
        'core_status_sha256': workers.digest(generation / 'status.json'),
        'control_status_sha256': workers.digest(controls / 'status.json')}
    if not args.execute_numerical or read(args.handoff) != expected_handoff:
        raise ValueError('A matching explicit root representative MP handoff is required')
    if CPU not in os.sched_getaffinity(0) or any(os.environ.get(name) != '1' for name in pair.THREADS):
        raise ValueError('CPU190 and all four single-thread settings are required')
    for status in (core_status, control_status):
        if not all(status.get(key) is True for key in ('complete', 'preserved_existing_inputs',
                'dependencies_unchanged', 'original_archive_unchanged')):
            raise ValueError('Fresh generation/preservation checks are incomplete')
        if status['git_revision'] != freeze['committed_revision']:
            raise ValueError('Control/core CLI generation differs from the committed freeze')
    inventory = policy.verified(read(generation / 'runtime-inventory.json'), 'fresh-generated-runtime-inventory-v2')
    if len(inventory['runtime_groups']['8']) != 1:
        raise ValueError('Extend representative selection for every actual runtime group')
    runtime, group = next(iter(inventory['runtime_groups']['8'].items()))
    signature = policy.verified(group['signature'], 'reducer-runtime-signature-v2')
    cohort = {'numerical_revision': freeze['committed_revision'],
              'generation_revision': core_status['git_revision'],
              'generator_identity_sha256': core_status['generator_tree_state_sha256'],
              'runtime_identity_sha256': runtime,
              'archive_sha256': freeze['installation_files_sha256']['lib/libninjars.a']}
    frozen = {'candidate_cohort': cohort, 'production_sources': production_sources(), 'sources': {}}
    control_work = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/controls-generation'
    def check_sources():
        pair.check_production(frozen)
        if bench.content_digest(bench.generator_tree_state()) != cohort['generator_identity_sha256']:
            raise ValueError('The live generator changed after the final CLI exports')
        for name, digest in freeze['installation_files_sha256'].items():
            if workers.digest(ROOT / 'HEPTools/ninjars' / name) != digest:
                raise ValueError('The accepted installed runtime changed: ' + name)
        for cid, record in control_status['cases'].items():
            output = control_work / 'outputs/candidate' / cid
            manifest = bench.generated_source_manifest(output)
            if (record['status'] != 'complete' or
                    read(output / 'generation.json')['source_identity']['revision'] != cohort['generation_revision'] or
                    manifest['files_sha256'] != record['compilation']['generated_sources_sha256'] or
                    pair.actual_runtime(output, manifest, 8) != signature):
                raise ValueError('A generated representative control changed: ' + cid)
    check_sources()
    report = HERE / 'focused-ninja-nrs-v1'
    work = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/focused-ninja-nrs-v1'
    if report.exists() or work.exists():
        raise ValueError('Retain this representative attempt; use a fresh namespace for a retry')
    report.mkdir(); work.mkdir(parents=True)
    shutil.copy2(Path(__file__), report / 'focused.py')
    shutil.copy2(SPECIFICATION, report / 'input-specification.json')
    shutil.copy2(args.handoff, report / 'root-handoff.json')
    write(report / 'release-cohort.json', cohort)
    write(report / 'production-sources.json', frozen['production_sources'])
    correction = ROOT / 'docs/benchmarkfolder/policy-v2/model-correction.json'
    results, bindings = {}, []
    for name, cases, backends, points in COLLECTIONS:
        spec = {'run_id': 'final-pins-focused-ninja-nrs-' + name,
            'configuration': 'docs/benchmarkfolder/policy-v2/controls.json',
            'report': policy.relative(report / 'runs' / name), 'work': policy.relative(work / name),
            'output_roots': {'baseline': '.benchmark-work/outputs/baseline',
                             'candidate': policy.relative(control_work / 'outputs/candidate')},
            'points_root': points, 'cases': cases, 'sources': ['candidate'], 'backends': backends,
            'cpu': CPU, 'representative': True, 'fresh_parity': True,
            'timeout': 86400., 'minimum_seconds': 1e-12}
        if spec != read(HERE / 'focused-readiness-ninja-nrs-v1/specs' / (name + '.json')):
            raise ValueError('Focused target scope differs from the reviewed retained specification')
        spec_path = report / 'specs' / (name + '.json'); write(spec_path, spec)
        started = time.monotonic()
        print(name, 'started', flush=True)
        try:
            check_sources()
            with owned_collection(report, label=name):
                collected = references.collect_timing(spec_path)
            run = policy.path(spec['report'])
            role = run / 'role-bindings.json'; bound = report / 'runtime-bindings' / (name + '.json')
            policy.bind_run(collected['report'], collected['role_map'], collected['outputs'], correction, role)
            policy.refresh_runtime_bindings(role, correction, bound)
            actual = policy.checked_bindings(bound, policy.validate_correction(correction), live=True)
            protocol, rows = workers.checked_series(run, minimum=2)
            targets = {}
            for key, target in actual['targets'].items():
                if target['reducer_runtime'] != signature:
                    raise ValueError('A representative worker used another full runtime')
                first = read(run / 'data' / (key + '-0.json'))
                fresh = read(run / 'data' / (key + '-fresh.json'))
                parity = workers.numerical_parity(first, fresh, protocol['configuration'])
                targets[key] = {'status_counts': first['status_counts'], 'parity': parity,
                                'runtime': runtime, 'executable': target['executable_sha256']}
            if name == 'near-singular-nrs' and any(
                    code != '322' for target in targets.values() for code in target['status_counts']):
                raise ValueError('The forced-QP representative failed to exercise genuine rescue')
            results[name] = {'status': 'passed', 'seconds': time.monotonic() - started,
                             'runtime_binding': supplement.bound(bound), 'targets': targets, 'records': len(rows)}
            bindings.append(policy.relative(bound))
            check_sources(); write(report / (name + '-result.json'), results[name])
            print(name, 'passed', flush=True)
        except BaseException as error:
            write(report / (name + '-failure.json'), {'status': 'failed', 'error_type': type(error).__name__,
                  'error': str(error), 'seconds': time.monotonic() - started})
            (report / (name + '-traceback.txt')).write_text(traceback.format_exc())
            raise
    write(report / 'candidate-results.json', {'schema': 'final-candidate-focused-ninja-nrs-results-v1',
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runtime': runtime,
        'candidate_cohort': cohort, 'collections': results, 'runtime_bindings': bindings,
        'targets': sum(len(item['targets']) for item in results.values()),
        'status': 'passed', 'final_role_proof_pending': True})


if __name__ == '__main__':
    main()
