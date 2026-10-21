#!/usr/bin/env python3
"""Light six-case timing, reusing the frozen Fortran and pair primitives."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
from functools import lru_cache
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT / 'tools'))
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_policy_v3 as release
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

module_spec = importlib.util.spec_from_file_location('light_observed_pair_helpers', BASE / 'loop-execution/loop_pair.py')
observed = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(observed)
SCHEMA = 'final-light-explicit-pair-spec-v1'
BINDINGS = 'final-light-explicit-pair-bindings-v1'
CASE_IDS = ('qcd-1', 'ew-1', 'qcd-2', 'ew-2', 'supplement-gg-h', 'supplement-gg-hh')
TARGETS = (('baseline', 6), ('candidate', 8))
ROLES = {'baseline': 'original', 'candidate': 'candidate_corrected'}
POINTS, REPETITIONS, MINIMUM_SECONDS = 10, 7, .5
CPU, WALL_BUDGET_SECONDS, PHASE_TIMEOUT_SECONDS = 207, 600., 120.
SAMPLING_LABEL = 'First10 fixed events of each original100-event stream; no100-point timing claim'
PARENTS = {
    'core': (BASE / 'timing-pair/specification-held-v1.json', 'de65bb6bb168bf427be3fc5c9fc1cb2eb2dc77a386c24528015228a1c4e8f813'),
    'loop': (BASE / 'loop-execution/specification-pair-held-v2.json', '56cee094b28452ba481c998410d6095b0f46d4737f6b200dea31587cf08c2d47'),
}


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prefix_bytes(parent, nfinal, points):
    lines = parent.read_bytes().splitlines(keepends=True)
    total = int(lines[0])
    if (type(points) is not int or not 0 < points <= total or
            len(lines) != 1 + total * (nfinal + 2)):
        raise ValueError('Invalid fixed point-prefix dimensions')
    return str(points).encode() + b'\n' + b''.join(lines[1:1 + points * (nfinal + 2)])


def parent_inputs():
    """Read retired specs only as original source/point provenance; never execute them."""
    parents = {}
    for key, (path, digest) in PARENTS.items():
        if workers.digest(path) != digest:
            raise ValueError('Historical source/point provenance changed')
        parents[key] = read(path)
    cases, origins, roots, points = [], {}, {}, {}
    for cid in CASE_IDS:
        parent = parents['loop' if cid.startswith('supplement-') else 'core']
        config = read(supplement.checked(parent['configuration']))
        case = next(c for c in config['cases'] if c['id'] == cid)
        cases.append(case); roots[cid] = parent['output_roots']; points[cid] = parent['points'][cid]
        origins[cid] = {}
        for source, backend in TARGETS:
            item = parent['origins'][cid][source]
            if 'source_proof' in item:
                anchor = policy.verified(read(supplement.checked(item['source_proof'])), 'supplement-generation-anchor-v1')
                generation, compilation, manifest = (anchor[k] for k in ('generation', 'compilation', 'compiled_manifest'))
            else:
                generation, compilation, manifest = (read(supplement.checked(item[k]))
                                                     for k in ('generation', 'compilation', 'compiled_manifest'))
            origins[cid][source] = {'role': ROLES[source], 'generation': generation,
                'compilation': compilation, 'compiled_manifest': manifest,
                'physical_card': item['physical_card'], 'reducer_runtime': item['reducer_runtime']}
    config = read(supplement.checked(parents['core']['configuration']))
    config.update(points_per_case=POINTS, repetitions=REPETITIONS, cases=cases)
    return config, origins, roots, points


def source_files():
    result = observed.source_files()
    result['observed_pair_helpers.py'] = result.pop('collector')
    result['collector'] = supplement.bound(Path(__file__))
    return result


def parent_seeds():
    seeds = {key: read(supplement.checked(read(path)['configuration']))['seed']
             for key, (path, _) in PARENTS.items()}
    return {cid: seeds['loop' if cid.startswith('supplement-') else 'core'] for cid in CASE_IDS}


@lru_cache(maxsize=8)
def committed_production(revision):
    """Read historical committed bytes; never substitute the current checkout."""
    paths = (*bench.GENERATOR_PATHS, 'vendor')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '-z', revision, '--', *paths], cwd=ROOT).decode().split('\0')
    archive = subprocess.check_output(['git', 'archive', revision, '--', *paths], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members = {m.name: m for m in stream if not m.isdir()}
        def contents(name, seen=()):
            if name in seen:
                raise ValueError('Cyclic committed source symlink')
            member = members.get(name)
            if member is None:
                entry = subprocess.check_output(['git', 'ls-tree', revision, '--', name], cwd=ROOT).decode()
                if not entry:
                    return None
                value = policy.git_bytes(revision, name)
                symbolic, target = entry.startswith('120000 '), value.decode() if entry.startswith('120000 ') else None
            else:
                if not member.isfile() and not member.issym():
                    raise ValueError('Unsupported committed source entry')
                symbolic, target = member.issym(), member.linkname
                value = None if symbolic else stream.extractfile(member).read()
            if symbolic:
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                if resolved.startswith('../') or resolved.startswith('/'):
                    raise ValueError('Committed source symlink escapes the repository')
                return contents(resolved, (*seen, name))
            return value
        return {name: hashlib.sha256(blob).hexdigest() if (blob := contents(name)) is not None else None
                for name in names if name}


def record_source_checkpoint(spec, report, context):
    pair.check_production(spec)  # Unchanged live guard, before and after every command.
    actual = {name: workers.digest(ROOT / name) if (ROOT / name).is_file() else None
              for name in spec['production_sources']}
    if actual != spec['production_sources']:
        raise ValueError('Observed live source inventory changed after guard')
    inventory = {'schema': 'final-light-observed-production-inventory-v1', 'production_sources': actual}
    inventory_path = report / 'source-checkpoints' / (bench.content_digest(inventory) + '.json')
    if inventory_path.exists():
        if read(inventory_path) != inventory:
            raise ValueError('Recorded source inventory was replaced')
    else:
        exclusive(inventory_path, inventory)
    value = {'schema': 'final-light-source-checkpoint-v1', **context,
        'utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'orchestration_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'numerical_revision': spec['candidate_cohort']['numerical_revision'],
        'generator_identity_sha256': spec['candidate_cohort']['generator_identity_sha256'],
        'observed_inventory': supplement.bound(inventory_path)}
    path = report / 'source-checkpoints' / (str(context['index']) + '.json')
    exclusive(path, value)
    return supplement.bound(path)


class GuardedDispatch(pair.Dispatch):
    def __init__(self, report, spec):
        super().__init__(report)
        self.report, self.spec = report, spec

    def event(self, case, phase, event, key, repetition=None, **fields):
        if phase in ('calibration', 'measurement') and event in ('start', 'complete'):
            context = {'index': self.index, 'case': case['id'], 'phase': phase, 'event': event,
                       'source': key[0], 'backend': key[1], 'repetition': repetition}
            fields['source_checkpoint'] = record_source_checkpoint(self.spec, self.report, context)
        return super().event(case, phase, event, key, repetition, **fields)


def check_source_checkpoints(events, spec):
    expected = spec['production_sources']
    if committed_production(spec['candidate_cohort']['numerical_revision']) != expected:
        raise ValueError('Held production inventory differs from committed numerical source')
    for event in events:
        if event['phase'] not in ('calibration', 'measurement'):
            continue
        value = read(supplement.checked(event['source_checkpoint']))
        if (value.get('schema') != 'final-light-source-checkpoint-v1' or
                any(value.get(k) != event[k] for k in ('index', 'case', 'phase', 'event', 'source', 'backend', 'repetition')) or
                value.get('numerical_revision') != spec['candidate_cohort']['numerical_revision'] or
                value.get('generator_identity_sha256') != spec['candidate_cohort']['generator_identity_sha256']):
            raise ValueError('Missing or mismatched before/after measurement source context')
        inventory = read(supplement.checked(value['observed_inventory']))
        if (inventory.get('schema') != 'final-light-observed-production-inventory-v1' or
                inventory.get('production_sources') != expected or
                committed_production(value['orchestration_revision']) != expected):
            raise ValueError('Recorded source context differs from exact historical production bytes')


def validate_spec(spec, live=True):
    policy.verified(spec, SCHEMA)
    config, origins, roots, parent_points = parent_inputs()
    if (spec['configuration'] != config or spec['origins'] != origins or spec['output_roots'] != roots or
            spec['pair_targets'] != [list(t) for t in TARGETS] or spec['role_map'] != ROLES or
            spec['repetitions'] != REPETITIONS or spec['minimum_seconds'] != MINIMUM_SECONDS or
            spec['cpu'] != CPU or spec['wall_budget_seconds'] != WALL_BUDGET_SECONDS or
            spec['timeout'] != PHASE_TIMEOUT_SECONDS or spec['sources'] != source_files() or
            spec['sampling_label'] != SAMPLING_LABEL or spec['parent_sampling_seeds'] != parent_seeds() or
            spec['old_full8_runtime_gate_required_for_completion'] is not False or
            spec['new_cuttools_or_collier_evaluations'] != 0 or
            spec['historical_parent_specs_source_only'] != {k: supplement.bound(p) for k, (p, _) in PARENTS.items()}):
        raise ValueError('The approved light six-case source/sampling/target plan changed')
    if spec['production_sources'] != read(PARENTS['core'][0])['production_sources']:
        raise ValueError('Held production inventory changed')
    if live:
        pair.check_production(spec)
    elif committed_production(spec['candidate_cohort']['numerical_revision']) != spec['production_sources']:
        raise ValueError('Historical production source differs from committed numerical bytes')
    cohort = release.validate_cohort(supplement.checked(spec['cohort_descriptor']))
    if pair.candidate_cohort(spec) != {k: cohort[k] for k in pair.candidate_cohort(spec)}:
        raise ValueError('Light timing uses another numerical/runtime cohort')
    proof = policy.verified(read(supplement.checked(spec['runtime_qualification'])), 'representative-policy-validation-v2')
    if cohort['runtime_identity_sha256'] not in proof['reducer_runtimes']['candidate_corrected/8']:
        raise ValueError('Light runtime lacks its actual representative proof')
    if set(spec['points']) != set(CASE_IDS) or spec['parent_points'] != parent_points:
        raise ValueError('Missing/injected fixed point prefix')
    for case in config['cases']:
        cid = case['id']; prefix = supplement.checked(spec['points'][cid])
        if prefix.read_bytes() != prefix_bytes(supplement.checked(parent_points[cid]), case['nfinal'], POINTS):
            raise ValueError('Light points differ from the exact authorized parent prefix')
        cards = []
        for source, backend in TARGETS:
            anchor = origins[cid][source]; output = policy.path(roots[cid][source]) / cid
            generation = anchor['generation']['source_identity']
            runtime = policy.verified(anchor['reducer_runtime'], 'reducer-runtime-signature-v2')
            if source == 'baseline':
                policy.original_identity(generation)
            elif (generation['revision'] != cohort['generation_revision'] or
                    generation['tree_state_sha256'] != cohort['generator_identity_sha256'] or
                    runtime['identity_sha256'] != cohort['runtime_identity_sha256'] or
                    cohort['archive_sha256'] not in runtime['libraries'].values()):
                raise ValueError('Light candidate source/runtime differs from the pinned release')
            card = supplement.checked(anchor['physical_card']).read_bytes(); cards.append(card)
            if live:
                if output.is_symlink() or not output.is_dir():
                    raise ValueError('Expected a real original/candidate source directory')
                pair.validate_origin_records(source, output, anchor['generation'], anchor['compilation'], anchor['compiled_manifest'])
                if ((output / 'Cards/param_card.dat').read_bytes() != card or
                        pair.actual_runtime(output, anchor['compiled_manifest'], backend) != runtime):
                    raise ValueError('Actual light source card or full runtime changed')
        if cards[0] != cards[1]:
            raise ValueError('Literal original/final physical cards differ')
    return spec, config


def prepare_spec(output):
    config, origins, roots, parents = parent_inputs()
    points = {}
    for case in config['cases']:
        path = HERE / 'points' / (case['id'] + '.dat'); path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:
            stream.write(prefix_bytes(supplement.checked(parents[case['id']]), case['nfinal'], POINTS))
        points[case['id']] = supplement.bound(path)
    old = read(PARENTS['core'][0])
    spec = policy.seal({'schema': SCHEMA, 'status': 'held-light-scope-not-launched',
        'configuration': config, 'origins': origins, 'output_roots': roots,
        'points': points, 'parent_points': parents,
        'historical_parent_specs_source_only': {k: supplement.bound(p) for k, (p, _) in PARENTS.items()},
        'cohort_descriptor': supplement.bound(BASE / 'candidate-cohort.json'),
        'candidate_cohort': pair.candidate_cohort(old), 'runtime_qualification': old['runtime_qualification'],
        'pair_targets': [list(t) for t in TARGETS], 'role_map': ROLES, 'sources': source_files(),
        'production_sources': old['production_sources'], 'cpu': CPU, 'repetitions': REPETITIONS,
        'minimum_seconds': MINIMUM_SECONDS, 'timeout': PHASE_TIMEOUT_SECONDS,
        'wall_budget_seconds': WALL_BUDGET_SECONDS,
        'run_id': 'final-pins-light-six-case-pairs-v1',
        'report': policy.relative(HERE / 'runs/run-v1'),
        'work': '.benchmark-work/policy-v3/final-pins-20261021-v1/light-timing/runs/run-v1',
        'sampling_label': SAMPLING_LABEL, 'parent_sampling_seeds': parent_seeds(),
        'old_full8_runtime_gate_required_for_completion': False,
        'new_cuttools_or_collier_evaluations': 0})
    validate_spec(spec)
    exclusive(output, spec)
    return {'specification': supplement.bound(output), 'actual_timing_pairs': 0}


def validate_handoff(specification, handoff):
    value = read(handoff)
    required = {'root_authorized': True, 'phase': SCHEMA, 'cpu': CPU,
                'specification_sha256': workers.digest(specification), 'quiet_cpu207': True,
                'all_other_numerical_and_compiler_work_released': True}
    if any(value.get(k) != v or (type(v) is bool and type(value.get(k)) is not bool) for k, v in required.items()):
        raise ValueError('A matching fresh root quiet207 handoff is required')
    if not isinstance(value.get('release_evidence'), list) or not value['release_evidence']:
        raise ValueError('The quiet handoff must bind actual release evidence')
    for record in value['release_evidence']:
        supplement.checked(record)
    return value


def collect(specification, handoff):
    spec, config = validate_spec(read(specification)); validate_handoff(specification, handoff)
    if CPU not in os.sched_getaffinity(0) or any(os.environ.get(n) != '1' for n in pair.THREADS):
        raise ValueError('CPU207 and all four one-thread limits are required')
    report, work = policy.path(spec['report']), policy.path(spec['work'])
    if report.exists() or work.exists():
        raise ValueError('Preserve the existing light attempt; no automatic restart/append')
    report.mkdir(parents=True); work.mkdir(parents=True)
    for name in ('sources', 'points'):
        (report / name).mkdir()
    for name, item in spec['sources'].items():
        shutil.copy2(supplement.checked(item), report / 'sources' / ('light_pair.py' if name == 'collector' else name))
    for cid, item in spec['points'].items():
        shutil.copy2(supplement.checked(item), report / 'points' / (cid + '.dat'))
    shutil.copy2(specification, report / 'input-spec.json'); shutil.copy2(handoff, report / 'root-handoff.json')
    descriptor = workers.protocol_descriptor()
    bench.dump(report / 'protocol.json', {'schema': workers.SCHEME, 'configuration': config,
        'protocol_descriptor': descriptor, 'protocol_identity_sha256': bench.content_digest(descriptor),
        'cpu': CPU, 'repetitions': REPETITIONS, 'sources': 'both', 'backends': [6, 8],
        'pair_targets': [list(t) for t in TARGETS], 'role_map': ROLES, 'fresh_parity': False,
        'candidate_cohort': spec['candidate_cohort'], 'acceptance_eligible': False,
        'input_spec': supplement.bound(report / 'input-spec.json'), 'sampling_label': spec['sampling_label'],
        'orchestration_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    status = {'status': 'incomplete', 'requested_cases': list(CASE_IDS), 'completed_cases': [], 'actual_points': POINTS}
    bench.dump(report / 'run-status.json', status)
    dispatch = GuardedDispatch(report, spec)
    args = SimpleNamespace(cpu=CPU, repetitions=REPETITIONS, minimum_seconds=MINIMUM_SECONDS, timeout=PHASE_TIMEOUT_SECONDS)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    expired = [False]
    def interrupted(signum, _frame):
        if expired[0]:
            raise TimeoutError('Light evaluation/build wall budget exhausted; retain partial pairs')
        raise InterruptedError('Light controller interrupted by signal ' + str(signum))
    def deadline():
        expired[0] = True
        os.kill(os.getpid(), signal.SIGTERM)
    for sig in previous:
        signal.signal(sig, interrupted)
    timer = threading.Timer(WALL_BUDGET_SECONDS, deadline); timer.daemon = True; timer.start()
    sentinel = work / 'input-root-sentinel'
    try:
        for case in config['cases']:
            pair.check_production(spec)
            with supplement.harness(sentinel, report), supplement.input_roots(spec['output_roots'][case['id']], sentinel):
                with pair.owned_preparation(report, label='light-' + case['id']):
                    pair.run_pair_case(case, config, args, work / 'workers', report, spec['run_id'], dispatch)
            for source, backend in TARGETS:
                observed.bind_target(report, work, case, source, backend, spec)
            status['completed_cases'].append(case['id']); bench.dump(report / 'run-status.json', status)
        status['status'] = 'complete'; bench.dump(report / 'run-status.json', status)
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        bench.dump(report / 'run-status.json', status)
        raise
    finally:
        timer.cancel(); timer.join(timeout=1.)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    try:
        result = snapshot(report)
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        bench.dump(report / 'run-status.json', status)
        raise
    exclusive(report / 'result.json', result)
    return result


def check_worker_delta(case, anchor, build, manifest, worker_file):
    source = worker_file.read_text()
    prefix = re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION', source, re.I).group(1)
    nborn = re.search(r'PARAMETER\s*\(NSQSO_BORN\s*=\s*(\d+)\)', source, re.I).group(1)
    def rendered(name):
        return (ROOT / 'tools' / name).read_text().replace('@PREFIX@', prefix).replace(
            '@NEXTERNAL@', str(case['nfinal'] + 2)).replace('@NBORN@', nborn).encode()
    drivers = [p for p in anchor['compiled_manifest']['files'] if p.startswith('SubProcesses/P') and p.endswith('/check_sa.f')]
    if len(drivers) != 1:
        raise ValueError('Expected one original observation driver')
    fresh = rendered('ninjars_benchmark_driver.f'); relative = drivers[0]
    if (hashlib.sha256(fresh).hexdigest() != anchor['compilation']['driver_sha256'] or
            anchor['compiled_manifest']['files'][relative] != {'sha256': hashlib.sha256(fresh).hexdigest(), 'bytes': len(fresh)}):
        raise ValueError('Worker prefix/Born dimensions differ from actual original driver')
    worker = rendered('ninjars_benchmark_worker.f'); files = dict(anchor['compiled_manifest']['files'])
    files[relative] = {'sha256': hashlib.sha256(worker).hexdigest(), 'bytes': len(worker)}
    if (worker_file.read_bytes() != worker or build['driver_sha256'] != hashlib.sha256(worker).hexdigest() or
            manifest['files'] != files or manifest['files_sha256'] != bench.content_digest(files)):
        raise ValueError('Numerical source changed beyond the exact rendered diagnostic driver')


def check_raw(record, points, inputs):
    capture = record['raw_capture']
    if set(capture) != {*workers.OUTPUTS, 'dimensions.dat'}:
        raise ValueError('Incomplete raw output capture')
    paths = {name: supplement.checked(item) for name, item in capture.items()}
    if len({p.parent for p in paths.values()}) != 1 or any(p.name != name for name, p in paths.items()):
        raise ValueError('Mixed raw output capture directories')
    for path in paths.values():
        inputs[policy.relative(path)] = workers.digest(path)
    parsed = workers.parse_outputs(next(iter(paths.values())).parent, record['backend'], points, record['loops'])
    if any(record[k] != v for k, v in parsed.items()):
        raise ValueError('Published record differs from actual-count raw observations')


def snapshot(report):
    report = policy.path(report); protocol = read(report / 'protocol.json')
    spec_path = supplement.checked(protocol['input_spec']); spec, config = validate_spec(read(spec_path), live=False)
    npoints = config['points_per_case']; status = read(report / 'run-status.json')
    if (status['status'] != 'complete' or status['requested_cases'] != list(CASE_IDS) or
            status['completed_cases'] != list(CASE_IDS) or status['actual_points'] != npoints or
            list((report / 'rejected').glob('*.json'))):
        raise ValueError('Light series is incomplete, failed or rejected')
    descriptor = workers.protocol_descriptor()
    if (protocol['configuration'] != config or protocol['pair_targets'] != [list(t) for t in TARGETS] or
            protocol['cpu'] != CPU or protocol['repetitions'] != REPETITIONS or protocol['backends'] != [6, 8] or
            protocol['sources'] != 'both' or protocol['role_map'] != ROLES or protocol['fresh_parity'] is not False or
            protocol['candidate_cohort'] != spec['candidate_cohort'] or protocol['protocol_descriptor'] != descriptor or
            protocol['protocol_identity_sha256'] != bench.content_digest(descriptor)):
        raise ValueError('Changed light protocol/source/sampling scope')
    for name, item in spec['sources'].items():
        if workers.digest(report / 'sources' / ('light_pair.py' if name == 'collector' else name)) != item['sha256']:
            raise ValueError('Retained light source changed')
    events = [json.loads(line) for line in (report / 'dispatch.jsonl').read_text().splitlines()]
    observed.check_dispatch(events, CASE_IDS)
    check_source_checkpoints(events, spec)
    ledger = report / 'process-ledger.jsonl'; observed.check_ownership([json.loads(x) for x in ledger.read_text().splitlines()])
    expected, cases, inputs = set(), {}, {}
    for case in config['cases']:
        cid = case['id']; copied = report / 'points' / (cid + '.dat')
        if copied.read_bytes() != supplement.checked(spec['points'][cid]).read_bytes():
            raise ValueError('Copied light stream differs from authorized prefix')
        records, targets = {}, {}
        for source, backend in TARGETS:
            key = f'{cid}-{source}-{backend}'; build_path = report / 'builds' / key / 'build.json'; build = read(build_path)
            manifest = read(build_path.parent / 'compiled-sources.json'); anchor = spec['origins'][cid][source]
            target_path = report / 'targets' / (key + '.json'); target = policy.verified(read(target_path), observed.TARGET_SCHEMA)
            if (target['case'] != cid or target['source'] != source or target['backend'] != backend or target['role'] != ROLES[source] or
                    target['compiled_manifest'] != manifest or read(supplement.checked(target['build'])) != build):
                raise ValueError('Observed light target/build changed')
            for item in target['runtime_metadata'].values():
                supplement.checked(item)
            if policy.reducer_signature(target) != target['reducer_runtime'] or target['reducer_runtime'] != anchor['reducer_runtime']:
                raise ValueError('Observed full runtime differs from qualified source')
            check_worker_delta(case, anchor, build, manifest, build_path.parent / 'worker.f')
            if (build['generation_source_identity'] != anchor['generation']['source_identity'] or
                    build['origin_compilation'] != anchor['compilation'] or
                    build['original_generated_sources_sha256'] != anchor['compiled_manifest']['files_sha256'] or
                    manifest['files_sha256'] != build['generated_sources_sha256']):
                raise ValueError('Light source/build provenance changed')
            values = []; calibration = None
            for repetition in ('calibration', *range(REPETITIONS)):
                path = report / 'data' / f'{key}-{repetition}.json'; expected.add(path); record = read(path)
                release.validate_rows(record, backend, points=npoints); check_raw(record, npoints, inputs)
                if (record['case'] != cid or record['source'] != source or record['backend'] != backend or record['repetition'] != repetition or
                        record['cpu'] != CPU or record['measurement_run_id'] != spec['run_id'] or record['paired_baseline_candidate'] is not True or
                        record['generation_source_identity'] != build['generation_source_identity'] or
                        record['protocol_identity_sha256'] != protocol['protocol_identity_sha256'] or
                        record['executable_sha256'] != build['executable_sha256'] or record['generated_sources_sha256'] != build['generated_sources_sha256'] or
                        record['points_sha256'] != workers.digest(copied) or record['parameter_card_sha256'] != workers.digest(supplement.checked(anchor['physical_card'])) or
                        (repetition == 'calibration' and record['loops'] != 1)):
                    raise ValueError('Light record target/source/actual-count stream changed')
                if cid.startswith('supplement-') and (record['answer_dimensions']['born_orders'] != 0 or any(r['coefficients'][0] != 0 for r in record['rows'])):
                    raise ValueError('Loop-induced Born component is not zero')
                starts = [e for e in events if e['index'] == record['dispatch_index']]
                required = {'case': cid, 'source': source, 'backend': backend, 'event': 'start', 'repetition': repetition,
                            'phase': 'calibration' if repetition == 'calibration' else 'measurement',
                            'loops': record['loops'], 'worker_sequence': record['worker_sequence']}
                if len(starts) != 1 or any(starts[0].get(k) != v for k, v in required.items()):
                    raise ValueError('Raw observation differs from actual dispatched command')
                end = events[record['dispatch_index'] + 1]
                if (supplement.checked(end['record']) != path or end['loops'] != record['loops'] or end['worker_sequence'] != record['worker_sequence']):
                    raise ValueError('Dispatch completion does not bind actual raw record')
                if repetition == 'calibration':
                    calibration = record
                else:
                    bench.validate_repetition(record, calibration, config); values.append(record)
                inputs[policy.relative(path)] = workers.digest(path)
            records[source, backend] = values; targets[f'{source}/{backend}'] = supplement.bound(target_path)
        left, right = records['baseline', 6], records['candidate', 8]; comparisons = []
        for a, b in zip(left, right):
            if any(a[k] != b[k] for k in ('points_sha256', 'parameter_card_sha256', 'shared_numerical_settings', 'answer_dimensions')):
                raise ValueError('Light original/candidate physical or numerical drift')
            comparisons.append(policy.compare_rows({'original/6': a['rows'], 'candidate_corrected/8': b['rows']}))
        stats = bench.runtime_statistics([r['seconds_per_point'] for r in left], [r['seconds_per_point'] for r in right], 1.05, config['seed'])
        cases[cid] = {'agreement': {'passed': all(x['passed'] for x in comparisons), 'comparisons': comparisons},
            'timing': stats, 'actual_pairs': REPETITIONS, 'actual_points': npoints, 'sampling_label': spec['sampling_label'],
            'targets': targets, 'status_counts': {f'{s}/{b}': [r['status_counts'] for r in rows] for (s, b), rows in records.items()}}
    if set((report / 'data').glob('*.json')) != expected or {p.name for p in (report / 'targets').glob('*.json')} != {
            f'{cid}-{s}-{b}.json' for cid in CASE_IDS for s, b in TARGETS}:
        raise ValueError('Extra/missing light observation or runtime target')
    passed = all(c['agreement']['passed'] for c in cases.values())
    return policy.seal({'schema': BINDINGS, 'specification': supplement.bound(spec_path),
        'protocol': supplement.bound(report / 'protocol.json'), 'dispatch': supplement.bound(report / 'dispatch.jsonl'),
        'process_ledger': supplement.bound(ledger), 'candidate_cohort': spec['candidate_cohort'],
        'cases': cases, 'ninja_ninjars_correctness_passed': passed, 'timing_complete': True,
        'status': 'correctness-passed-timing-reported' if passed else 'correctness-failed',
        'reference_1p05_passed': all(c['timing']['passed'] for c in cases.values()),
        'old_full8_runtime_gate_required_for_completion': False, 'new_cuttools_or_collier_evaluations': 0,
        'sampling_label': spec['sampling_label'], 'parent_sampling_seeds': spec['parent_sampling_seeds'],
        'bootstrap_seed': config['seed'], 'inputs_sha256': dict(sorted(inputs.items()))})


def validate(filename):
    value = policy.verified(read(filename), BINDINGS)
    if snapshot(supplement.checked(value['protocol']).parent) != value:
        raise ValueError('Stale or forged light timing conclusion')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-spec', 'collect', 'validate'))
    parser.add_argument('--spec', type=Path, default=HERE / 'specification-held-v1.json')
    parser.add_argument('--handoff', type=Path); parser.add_argument('--result', type=Path)
    parser.add_argument('--execute-numerical', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare-spec':
        result = prepare_spec(args.spec)
    elif args.action == 'validate':
        result = validate(args.result)
    elif not args.execute_numerical or not args.handoff:
        parser.error('Actual collection requires an explicit root quiet handoff')
    else:
        result = collect(args.spec, args.handoff)
    print(json.dumps(result, indent=2))
