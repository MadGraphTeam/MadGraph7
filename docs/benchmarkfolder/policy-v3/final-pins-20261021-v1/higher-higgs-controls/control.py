#!/usr/bin/env python3
"""Four held fresh-process calls for two matched higher-Higgs points per case."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_policy_v3 as policy3
import ninjars_supplement_v2 as supplement
from madgraph.various.banner import MadLoopParam

module = importlib.util.spec_from_file_location('higher_higgs_generation', HERE.parent / 'higher-higgs-generation/prepare.py')
generation = importlib.util.module_from_spec(module); module.loader.exec_module(generation)
PREPARATION = generation.REPORT
SPECIFICATION = HERE / 'held-specification-v2.json'
RUN = HERE / 'run-v1'
SOURCE_PATHS = [Path(__file__).resolve(), HERE.parent / 'higher-higgs-generation/prepare.py'] + [
    ROOT / 'tools' / name for name in ('ninjars_benchmark.py', 'ninjars_benchmark_driver.f',
    'ninjars_benchmark_workers.py', 'ninjars_benchmark_pair_v3.py', 'ninjars_benchmark_policy_v2.py',
    'ninjars_benchmark_policy_v3.py', 'ninjars_supplement_v2.py')]
CONFIGURATION = {'cases': generation.CASES, 'points_per_case': 2, 'energy_gev': 1000,
                 'seed': 940086, 'relative_tolerance': 1e-7, 'scale_tolerance': 1e-10,
                 'madloop_parameters': {}}


def read(path):
    return json.loads(path.read_text())


def prepare():
    if SPECIFICATION.exists():
        raise ValueError('Preserve the existing fixed-point proposal')
    status = read(PREPARATION / 'status.json')
    if status['status'] != 'complete' or status['amplitude_evaluations'] != 0:
        raise ValueError('Both genuine exports/builds must be complete')
    previous = bench.REPORT
    bench.REPORT = HERE
    try:
        points, origins = {}, {}
        for case in generation.CASES:
            proof_path = supplement.checked(status['cases'][case['id']]['proof'])
            proof = read(proof_path)
            path = bench.make_points(case, proof['external_masses'], CONFIGURATION)
            points[case['id']] = {'data': supplement.bound(path), 'metadata': supplement.bound(path.with_suffix('.json'))}
            origins[case['id']] = supplement.bound(proof_path)
    finally:
        bench.REPORT = previous
    spec = {'schema': 'higher-higgs-fixed-two-point-proposal-v1', 'configuration': CONFIGURATION,
            'generation_status': supplement.bound(PREPARATION / 'status.json'), 'origins': origins, 'points': points,
            'production_inventory': status['source_inventory_after'],
            'sources': {str(p.relative_to(ROOT)): supplement.bound(p) for p in SOURCE_PATHS},
            'targets': [6, 8], 'cpu': 190, 'loops': 1, 'fresh_process_invocations': 4,
            'evaluation_wall_budget_seconds': 300, 'per_process_timeout_seconds': 120,
            'driver_semantics': 'Per invocation: one cold call, up to1002 warm calls until two consecutive stable points, one untimed two-point traversal, one retained two-point traversal.',
            'normal_rescue': True, 'performance_measurement': False, 'automatic_retry': False,
            'scope': 'Exactly two shared physical points per process, full Born/Laurent and split orders, current generated Ninja6 versus NinjaRS8. No additional random or seven-pair campaign.'}
    bench.dump(SPECIFICATION, spec)
    return supplement.bound(SPECIFICATION)


def check_inputs(spec, live=False):
    if (spec['schema'] != 'higher-higgs-fixed-two-point-proposal-v1' or spec['configuration'] != CONFIGURATION or
            spec['targets'] != [6, 8] or spec['loops'] != 1 or spec['fresh_process_invocations'] != 4 or
            spec['evaluation_wall_budget_seconds'] != 300 or spec['per_process_timeout_seconds'] != 120):
        raise ValueError('Changed two-point scope')
    if set(spec['sources']) != {str(p.relative_to(ROOT)) for p in SOURCE_PATHS}:
        raise ValueError('Missing or changed collector dependency')
    for name, item in spec['sources'].items():
        if item['path'] != name:
            raise ValueError('Source binding key/path mismatch')
        supplement.checked(item)
    inventory = read(supplement.checked(spec['production_inventory']))
    if live and generation.source_inventory() != inventory:
        raise ValueError('Actual generator/vendor bytes changed after preparation')
    for case in generation.CASES:
        cid = case['id']; proof = read(supplement.checked(spec['origins'][cid]))
        if proof['case'] != case:
            raise ValueError('Changed actual process definition')
        data = supplement.checked(spec['points'][cid]['data']); metadata = read(supplement.checked(spec['points'][cid]['metadata']))
        if metadata['points'] != 2 or metadata['energy_gev'] != 1000 or metadata['seed'] != case['seed'] or workers.digest(data) != metadata['sha256']:
            raise ValueError('Changed authorized two-point stream')
        for name in ('generation', 'compilation', 'compiled_manifest', 'physical_card', 'runtime_signature', 'madloop_origin_card'):
            supplement.checked(proof[name])
        if live:
            output = ROOT / proof['output']
            pair.validate_origin_records('candidate', output, read(supplement.checked(proof['generation'])),
                read(supplement.checked(proof['compilation'])), read(supplement.checked(proof['compiled_manifest'])))
            if pair.actual_runtime(output, read(supplement.checked(proof['compiled_manifest'])), 8) != read(supplement.checked(proof['runtime_signature'])):
                raise ValueError('Actual numerical runtime changed')


def collect(handoff):
    spec = read(SPECIFICATION); check_inputs(spec, live=True)
    authorization = read(handoff)
    if (authorization.get('root_authorized') is not True or authorization.get('cpu') != 190 or
            authorization.get('specification_sha256') != workers.digest(SPECIFICATION) or
            authorization.get('purpose') != 'higher-higgs-two-point-correctness'):
        raise ValueError('An explicit matching root CPU190 handoff is required')
    if os.sched_getaffinity(0) != {190} or any(os.environ.get(n) != '1' for n in pair.THREADS):
        raise ValueError('CPU190 alone and four one-thread settings are required')
    if RUN.exists():
        raise ValueError('Preserve every earlier attempt; no automatic retry')
    RUN.mkdir(); shutil.copy2(handoff, RUN / 'root-handoff.json')
    shutil.copy2(SPECIFICATION, RUN / 'input-spec.json')
    status = {'status': 'incomplete', 'completed': [], 'requested': [c['id'] for c in generation.CASES],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    bench.dump(RUN / 'status.json', status)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    expired = [False]
    def interrupted(signum, frame):
        if expired[0]:
            raise TimeoutError('Two-point control budget exhausted; preserve partial output')
        raise InterruptedError('Two-point control interrupted by signal ' + str(signum))
    for sig in previous:
        signal.signal(sig, interrupted)
    def deadline():
        expired[0] = True; os.kill(os.getpid(), signal.SIGTERM)
    timer = threading.Timer(300, deadline); timer.daemon = True; timer.start()
    previous_roots = bench.WORK, bench.REPORT
    bench.WORK, bench.REPORT = generation.WORK, RUN
    try:
        with pair.owned_preparation(RUN, label='higher-higgs-two-point'):
            owned_run = bench.run
            def finite_run(command, cwd, log, **kwargs):
                if command[:4] != ['taskset', '-c', '190', './check'] or len(command) != 6 or command[-1] != '1':
                    raise ValueError('Only the four explicit two-point invocations are allowed')
                return owned_run(command, cwd, log, timeout=120)
            bench.run = finite_run
            try:
                for case in generation.CASES:
                    cid = case['id']; points = supplement.checked(spec['points'][cid]['data'])
                    for backend in (6, 8):
                        check_inputs(spec, live=True)
                        record = bench.measure('candidate', case, backend, points, 1, 0, 190, config=CONFIGURATION)
                        directory = bench.subprocess_dir('candidate', case)
                        raw = RUN / 'raw' / f'{cid}-{backend}'; raw.mkdir(parents=True)
                        record['raw_capture'] = {}
                        for name in (*workers.OUTPUTS, 'dimensions.dat'):
                            shutil.copy2(directory / name, raw / name)
                            record['raw_capture'][name] = supplement.bound(raw / name)
                        record['performance_measurement'] = False
                        bench.dump(bench.measurement_path(record), record)
                        check_inputs(spec, live=True)
                        status['completed'].append([cid, backend]); bench.dump(RUN / 'status.json', status)
            finally:
                bench.run = owned_run
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', failure=repr(error))
        raise
    finally:
        timer.cancel(); status['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bench.dump(RUN / 'status.json', status)
        bench.WORK, bench.REPORT = previous_roots
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result = validate()
    bench.dump(RUN / 'result.json', result)
    return result


def validate():
    retained_spec = RUN / 'input-spec.json'
    if retained_spec.read_bytes() != SPECIFICATION.read_bytes():
        raise ValueError('Retained input spec differs from the authorized held proposal')
    authorization = read(RUN / 'root-handoff.json')
    if (authorization.get('root_authorized') is not True or authorization.get('cpu') != 190 or
            authorization.get('purpose') != 'higher-higgs-two-point-correctness' or
            authorization.get('specification_sha256') != workers.digest(retained_spec)):
        raise ValueError('Retained proposal differs from the original root handoff')
    spec = read(retained_spec); check_inputs(spec)
    status = read(RUN / 'status.json')
    if status['status'] != 'complete' or status['completed'] != [[c['id'], b] for c in generation.CASES for b in (6, 8)]:
        raise ValueError('Incomplete two-point controls')
    ledger_path = RUN / 'process-ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    spawns = [item for item in ledger if item['event'] == 'spawn']
    reaps = [item for item in ledger if item['event'] == 'reaped']
    if len(spawns) != 4 or len(reaps) != 4 or len(ledger) != 8:
        raise ValueError('Expected exactly four owned fresh-process invocations')
    for index, item in enumerate(spawns):
        cid = generation.CASES[index // 2]['id']; proof = read(supplement.checked(spec['origins'][cid]))
        command = ['taskset', '-c', '190', './check', str(supplement.checked(spec['points'][cid]['data'])), '1']
        matches = [r for r in reaps if all(r.get(k) == item.get(k) for k in ('pid', 'pgid', 'start_ticks'))]
        if (item['command'] != command or Path(item['cwd']) != ROOT / proof['output'] / 'SubProcesses' / proof['subprocess'] or
                not isinstance(item.get('start_ticks'), int) or item['pid'] != item['pgid'] or
                len(matches) != 1 or matches[0].get('returncode') != 0 or matches[0].get('reaped') is not True):
            raise ValueError('Changed actual fixed-stream dispatch or terminal ownership')
    results = {}; previous = bench.REPORT; bench.REPORT = RUN
    try:
        for case in generation.CASES:
            cid = case['id']; records = {}
            for backend in (6, 8):
                path = RUN / 'data' / f'candidate-{cid}-{backend}-0.json'; record = read(path)
                raw = {name: supplement.checked(item) for name, item in record['raw_capture'].items()}
                if set(raw) != {*workers.OUTPUTS, 'dimensions.dat'} or len({p.parent for p in raw.values()}) != 1:
                    raise ValueError('Incomplete or mixed retained raw capture')
                parsed = workers.parse_outputs(next(iter(raw.values())).parent, backend, 2, 1)
                if any(record[k] != value for k, value in parsed.items()):
                    raise ValueError('Record differs from actual raw two-point output')
                proof = read(supplement.checked(spec['origins'][cid])); compilation = read(supplement.checked(proof['compilation']))
                card_path = RUN / 'inputs/candidate' / cid / f'MadLoopParams-{backend}.dat'
                card = MadLoopParam(str(card_path)); expected = MadLoopParam(str(supplement.checked(proof['madloop_origin_card'])))
                for key, value in {'MLReductionLib': str(backend), 'CTModeRun': -1, 'CTModeInit': 1,
                                   'UseLoopFilter': False, 'COLLIERComputeUVpoles': True, 'COLLIERComputeIRpoles': True}.items():
                    expected[key] = value
                if dict(card) != dict(expected) or record['madloop_card_sha256'] != workers.digest(card_path):
                    raise ValueError('Actual backend card differs from normal-rescue control settings')
                if (record['case'] != cid or record['backend'] != backend or record['loops'] != 1 or record['cpu'] != 190 or
                        record['points_sha256'] != spec['points'][cid]['data']['sha256'] or
                        record['parameter_card_sha256'] != proof['physical_card']['sha256'] or
                        record['executable_sha256'] != compilation['executable_sha256'] or
                        record['generated_sources_sha256'] != compilation['generated_sources_sha256'] or
                        record['generation_source_identity'] != proof['generated_source_identity'] or
                        record['answer_dimensions']['born_orders'] != 0 or any(r['coefficients'][0] != 0 for r in record['rows'])):
                    raise ValueError('Changed target/source/input or nonzero loop-induced Born term')
                records[f'candidate-{backend}'] = record
            if records['candidate-6']['shared_numerical_settings'] != records['candidate-8']['shared_numerical_settings']:
                raise ValueError('Different shared numerical settings')
            agreement = policy3.compare_rows({name: record['rows'] for name, record in records.items()})
            results[cid] = {'passed': agreement['passed'], 'agreement': agreement, 'points': spec['points'][cid],
                            'records': {str(b): supplement.bound(RUN / 'data' / f'candidate-{cid}-{b}-0.json') for b in (6, 8)}}
    finally:
        bench.REPORT = previous
    value = {'schema': 'higher-higgs-fixed-two-point-result-v1', 'specification': supplement.bound(RUN / 'input-spec.json'),
            'cases': results, 'passed': all(c['passed'] for c in results.values()), 'unique_process_points': 4,
            'fresh_process_invocations': 4, 'process_ledger': supplement.bound(ledger_path),
            'performance_measurement': False, 'seven_pair_campaign': False}
    if (RUN / 'result.json').exists() and read(RUN / 'result.json') != value:
        raise ValueError('Published higher-Higgs conclusion differs from retained raw proof')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'collect', 'validate'))
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--execute-numerical', action='store_true')
    args = parser.parse_args()
    if args.action == 'collect' and (not args.handoff or not args.execute_numerical):
        parser.error('Numerical collection requires explicit root handoff and execution flag')
    value = prepare() if args.action == 'prepare' else collect(args.handoff) if args.action == 'collect' else validate()
    print(json.dumps(value, indent=2))
