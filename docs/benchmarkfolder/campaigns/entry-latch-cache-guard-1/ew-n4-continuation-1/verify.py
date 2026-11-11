#!/usr/bin/env python3
"""Pure replay of five closed lanes and a separate, complete EW n4 continuation."""
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

CAMPAIGN = 'entry-latch-cache-guard-1'
CASE = 'ew_n4_epem_mupmumaa'
REPORT = 'docs/benchmarkfolder/campaigns/' + CAMPAIGN
HERE = REPORT + '/ew-n4-continuation-1'
ORIGINAL = REPORT + '/experiments/numerical/full100-warmup-' + CASE + '-1'
OLD_CASE = REPORT + '/remaining-qualification-1/' + CASE
STAGE = HERE + '/numerical-' + CASE
MANIFEST = '4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076'
REPLAY = 'docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-qualification-1/run.py'
REPLAY_SHA = '36622d7aa41cd6fe19cb01a992a6ec527fb9899579f57ddd6c5f7725b2fd58b0'
EXPECTED = [('baseline', 1), *[('candidate', b) for b in (1, 6, 7, 8, 9)]]
PROTOCOLS = {'tirust_benchmark.py', 'tirust_benchmark_campaign.py', 'tirust_benchmark_compiler.py',
             'tirust_benchmark_numerical.py', 'tirust_benchmark_warmup.py',
             'tirust_benchmark_worker.py', 'tirust_benchmark_worker.f'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name, 'Invalid relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and name != '.'
            and '..' not in path.parts, 'Noncanonical relative path: ' + name)
    return name


def contained(root, name):
    target = root
    for part in PurePosixPath(relative(name)).parts:
        target /= part
        require(not target.is_symlink(), 'Symlink in retained path: ' + name)
    require(target.resolve().is_relative_to(root), 'Path escapes repository')
    return target


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def extract(raw, names, namespace):
    """Load selected definitions/literal constants, never module top-level code."""
    selected = []
    for node in ast.parse(raw).body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
        if name in names:
            require(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)), 'Unexpected parser node')
            require(not getattr(node, 'decorator_list', []), 'Decorated parser definition')
            if isinstance(node, ast.Assign):
                ast.literal_eval(node.value)
            require(not any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in ast.walk(node)),
                    'Parser definition imports external code')
            selected.append(node)
    require(len(selected) == len(names), 'Missing/duplicated parser definition')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<retained pure definitions>', 'exec'), namespace)


class Evidence:
    def __init__(self, root, name, expected):
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Explicit SHA-256 required')
        self.root = Path(root).resolve()
        raw = contained(self.root, name).read_bytes()
        require(sha(raw) == expected, 'Index hash mismatch')
        self.index, self.blobs = decode(raw), {}
        for path, identity in self.index['files'].items():
            relative(path)
            require(type(identity['bytes']) is int and identity['bytes'] >= 0, 'Invalid payload byte count')
            data = contained(self.root, path).read_bytes()
            require(len(data) == identity['bytes'] and sha(data) == identity['sha256'],
                    'Payload size/hash mismatch: ' + path)
            self.blobs[path] = data
        require(len(self.blobs) == self.index['file_count']
                and sum(map(len, self.blobs.values())) == self.index['total_bytes'], 'Inventory count/size mismatch')

    def data(self, name):
        name = name.name if isinstance(name, RetainedPath) else str(name)
        require(name in self.blobs, 'Read outside verified allowlist: ' + name)
        return self.blobs[name]

    def load(self, name):
        return decode(self.data(name))

    def digest(self, name):
        return sha(self.data(name))


class RetainedPath:
    """No filesystem operations: only bytes verified by the caller's index."""
    def __init__(self, name, evidence=None):
        self.name = name.name if isinstance(name, RetainedPath) else relative(str(name))
        self.evidence = evidence if evidence is not None else getattr(name, 'evidence', None)

    def __truediv__(self, name):
        return RetainedPath(self.name + '/' + relative(name), self.evidence)

    def is_file(self):
        return self.name in self.evidence.blobs

    def open(self, encoding='utf-8', newline=None):
        require(encoding == 'utf-8' and newline == '' and self.name.endswith(('/raw/warmup.dat', '/raw/dimensions.dat')),
                'Unexpected raw parser access')
        return io.StringIO(self.evidence.data(self.name).decode(encoding), newline=newline)


def descriptors(fresh=False):
    result = [{'source': source, 'backend': backend, 'directory': ORIGINAL + f'/{source}-{backend}',
               'origin': 'original-complete'} for source, backend in EXPECTED[:5]]
    if fresh:
        result.append({'source': 'candidate', 'backend': 9, 'directory': HERE + '/lane/candidate-9',
                       'origin': 'fresh-continuation'})
    return result


def helpers(evidence, prefix):
    config = evidence.load(REPORT + '/cases.json')
    spec = evidence.load(prefix + '/specification.json')
    case = next(item for item in config['cases'] if item['id'] == CASE)
    require(config['relative_tolerance'] == 3e-6 and config['points_per_case'] == 100, 'Different numerical policy')
    require(spec['case'] == case and spec['campaign_binding'] == {'id': CAMPAIGN, 'manifest_sha256': MANIFEST},
            'Numerical specification identity changed')
    require(spec['measurement_phase'] == 'steady-warmup' and spec['performance_eligible'] is False,
            'Unexpected numerical scope')
    require(spec['configuration_sha256'] == evidence.digest(REPORT + '/cases.json')
            == evidence.digest(prefix + '/cases.json') and evidence.digest(prefix + '/campaign-manifest.json')
            == MANIFEST == evidence.digest(REPORT + '/manifest.json'), 'Manifest/configuration mismatch')
    manifest = evidence.load(REPORT + '/manifest.json')
    require(spec['worker_timeout_seconds'] == manifest['administrative_worker_timeout_seconds'][CASE], 'Worker timeout changed')
    require(set(spec['protocols']) == PROTOCOLS
            and spec['protocols'] == evidence.load(ORIGINAL + '/specification.json')['protocols'],
            'Continuation protocol differs from original admitted parser')
    for name, expected in spec['protocols'].items():
        require(evidence.digest(prefix + '/' + relative(name)) == expected, 'Retained protocol changed')
    worker = {}
    extract(evidence.data(prefix + '/tirust_benchmark_worker.py'),
            {'BACKEND_CODES', 'MeasurementError', 'validate_status'}, worker)
    parser = {'math': math, 'Path': lambda name: RetainedPath(name, evidence),
              'worker': SimpleNamespace(**worker), 'MeasurementError': worker['MeasurementError']}
    extract(evidence.data(prefix + '/tirust_benchmark_warmup.py'), {'_call_groups', 'parse_warmup'}, parser)
    bench = {'json': json, 'hashlib': hashlib}
    extract(evidence.data(prefix + '/tirust_benchmark.py'),
            {'compare_rows', 'content_digest', 'parse_chain', 'validate_worker_modes'}, bench)
    require(evidence.digest(REPLAY) == REPLAY_SHA, 'Pinned lane replay source changed')
    replay = {'parse_warmup': parser['parse_warmup'], 'read': evidence.load, 'digest': evidence.digest,
              'bench': SimpleNamespace(**bench), 'CAMPAIGN': CAMPAIGN, 'EXPECTED_MANIFEST': MANIFEST}
    extract(evidence.data(REPLAY), {'runtime_files', 'replay_lane'}, replay)
    return config, case, spec, bench['compare_rows'], replay['replay_lane']


def stage(evidence, directory, entry, passed):
    require(entry['status'] == ('passed' if passed else 'failed') and entry['cleanup']['survivors'] == [],
            'Stage status/cleanup mismatch')
    if passed:
        require(entry.get('returncode') == 0, 'Stage did not exit successfully')
        require(evidence.digest(directory + '/console.log') == entry['console_sha256'], 'Stage console changed')
    else:
        require(entry.get('error') == "InterruptedError('Supervisor received signal 2')",
                'Original numerical stage is not the retained interruption')


def interruption(evidence, value):
    require(value['handoff_path'] == HERE + '/interruption/handoff.json' and value['last_saved_point'] == 74
            and value['original_case_status_path'] == OLD_CASE + '/status.json', 'Interruption identity changed')
    require(evidence.digest(value['handoff_path']) == value['sha256'], 'Interruption handoff changed')
    handoff = evidence.load(value['handoff_path'])
    require(handoff['campaign'] == CAMPAIGN and handoff['case'] == CASE
            and handoff['last_raw_saved_point'] == 74 and handoff['partial_raw_lines'] == 151
            and handoff['checkpoint_created'] is False and handoff['full_case_qualified'] is False
            and handoff['performance_acceptance_eligible'] is False
            and handoff['classification'].startswith('interrupted-incomplete')
            and handoff['cleanup']['survivors'] == [], 'Original interruption was relabelled')
    copies = value['file_copies']
    require(set(copies) == set(handoff['files']), 'Incomplete interruption payload copies')
    for original, copied in copies.items():
        raw = evidence.data(copied)
        record = handoff['files'][original]
        require(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'Interrupted payload changed')
        if record['original'].startswith(REPORT + '/') and record['original'] in evidence.blobs:
            require(raw == evidence.data(record['original']), 'Original interruption evidence changed')
        if original.endswith('/worker-raw/warmup.dat'):
            lines = raw.decode().splitlines()
            rows = [line.split() for line in lines if line.strip() and not line.lstrip().startswith('#')]
            require(len(lines) == 151 and len(rows) == 150
                    and [(int(row[0]), int(row[1]), int(row[2])) for row in rows]
                    == [(point, max(1, point), order) for point in range(75) for order in (0, 1)],
                    'Interrupted trace is not the retained partial traversal through point 74')
    status = evidence.load(value['original_case_status_path'])
    require(status['case'] == CASE and status['status'] == 'failed' and status['full_matrix_complete'] is False
            and status['performance_acceptance_eligible'] is False
            and status['error'] == "InterruptedError('Supervisor received signal 2')", 'Original case was relabelled')
    require([item['id'] for item in status['stages']] ==
            ['reuse-baseline', 'generate-candidate', 'build-both', 'numerical-' + CASE], 'Original stages changed')
    for index, item in enumerate(status['stages']):
        stage(evidence, OLD_CASE + '/' + item['id'], item, index < 3)
    require([(item['lane'], item['status'], item['performance_eligible']) for item in handoff['completed_lanes']]
            == [(f'{s}-{b}', 'complete', False) for s, b in EXPECTED[:5]], 'Original completed lane list changed')
    for item, expected in zip(handoff['completed_lanes'], descriptors()):
        require(item['path'] == expected['directory'] + '/result.json'
                and evidence.digest(item['path']) == item['sha256'], 'Original completed result changed')
    return {'classification': handoff['classification'], 'original_case_status': 'failed',
            'last_saved_point': 74, 'partial_raw_lines': 151, 'cleanup_survivors': [],
            'handoff_path': value['handoff_path'], 'sha256': value['sha256']}


def replay_lanes(evidence, selected, records, reference=None):
    lanes = []
    for lane in selected:
        source, backend, directory = lane['source'], lane['backend'], lane['directory']
        prefix = str(PurePosixPath(directory).parent)
        config, case, spec, compare, replay = helpers(evidence, prefix)
        value = evidence.load(directory + '/result.json')
        require(value['source'] == source and value['backend'] == backend and value['status'] == 'complete', 'Incomplete/wrong lane')
        require(directory + '/failure.json' not in evidence.blobs, 'Lane has retained failure')
        record = records[source + '/' + CASE]
        inputs = evidence.load(directory + '/' + source + '-inputs.json')
        if backend == 9:
            interrupted = evidence.load(ORIGINAL + '/candidate-9/candidate-inputs.json')
            require(inputs['settings'] == interrupted['settings']
                    and inputs['numerical_cards'] == interrupted['numerical_cards'],
                    'Continuation numerical card/settings differ from original candidate9')
        for kind in ('generation', 'compilation'):
            require(inputs[kind] == evidence.load(REPORT + f'/inputs/{source}/{CASE}/{kind}.json'), 'Retained build record mismatch')
        require(record['points_sha256'] == evidence.digest(REPORT + '/points/' + CASE + '.dat')
                and record['points_metadata_sha256'] == evidence.digest(REPORT + '/points/' + CASE + '.json')
                and record['param_card_sha256'] == evidence.digest(REPORT + f'/inputs/{source}/{CASE}/param_card.dat'),
                'Admitted physical input differs from retained files')
        require(inputs['compilation']['link_inputs_before'] == inputs['compilation']['link_inputs_after']
                == {key: record['runtime'][key] for key in ('archives', 'metadata')}, 'Static dependencies changed')
        for name in ('warmup.dat', 'dimensions.dat', 'ready.json'):
            evidence.data(directory + '/raw/' + name)
        try:
            observation = replay(RetainedPath(directory, evidence), source, backend, value, config, record)
        except RuntimeError as error:
            raise ValueError('Retained warmup replay failed: ' + str(error)) from error
        require(observation['seconds'] is None and observation['performance_eligible'] is False
                and observation['measurement_phase'] == 'steady-warmup'
                and sorted({row['point'] for row in observation['rows']}) == list(range(1, 101)), 'Expected full untimed 100-point traversal')
        if reference is None:
            require(source == 'baseline' and backend == 1, 'Missing baseline reference')
            reference = observation['rows']
        comparison = compare(reference, observation['rows'], config['relative_tolerance'])
        require(comparison == value['comparison'], 'Numerical comparison differs from raw replay')
        residuals = comparison['residuals']
        misses = [row for row in residuals if not (
            row['symmetric_relative_error'] is not None and row['symmetric_relative_error'] < 1e-8
            or row['pole_to_finite'] is not None and row['pole_to_finite'] < 1e-8)]
        def maximum(rows):
            values = [row['symmetric_relative_error'] for row in rows]
            return None if any(value is None for value in values) else max(values, default=0.)
        lanes.append({'source': source, 'backend': backend, 'status': 'complete', 'error': None,
            'numerical_agreement_passed': comparison['passed'], 'ordinary_target_passed': comparison['ordinary_target_passed'],
            'status_counts': observation['status_counts'], 'accepted_backend_precision_counts': observation['backend_precision_counts'],
            'matched_saved_points': 100, 'fpe_flags': None, 'internal_retry_counts': None,
            'performance_acceptance_eligible': False, 'regression_passed': comparison['passed'],
            'points': 100, 'rows': len(observation['rows']), 'scalar_comparisons': len(residuals),
            'maximum_symmetric_finite_error': maximum([row for row in residuals if row['column'] == 1]),
            'maximum_symmetric_non_negligible_residual': maximum([row for row in residuals
                if row['pole_to_finite'] is None or row['pole_to_finite'] >= 1e-8]),
            'ordinary_target_miss_count': len(misses), 'target_miss_points': sorted({row['point'] for row in misses}),
            'target_miss_columns': {str(column): sum(row['column'] == column for row in misses) for column in range(4)},
            'target_miss_residuals': misses})
    return lanes, reference


def case_summary(lanes):
    """Exact scientific projection; administrative lineage stays in its own record."""
    keys = ('source', 'backend', 'status', 'error', 'numerical_agreement_passed', 'ordinary_target_passed',
            'status_counts', 'accepted_backend_precision_counts', 'matched_saved_points', 'fpe_flags',
            'internal_retry_counts', 'performance_acceptance_eligible')
    return {'case': CASE, 'status': 'passed' if all(row['numerical_agreement_passed'] for row in lanes) else 'failed',
            'lanes': [{key: row[key] for key in keys} for row in lanes]}


def original(evidence, index):
    require(index['schema'] == 'madloop-tirust-ew-continuation-inputs-v1' and index['campaign'] == CAMPAIGN
            and index['case'] == CASE and index['performance_acceptance_eligible'] is False, 'Wrong original input index')
    require(index['original_experiment'] == ORIGINAL and index['original_case_directory'] == OLD_CASE
            and index['selected_original_lanes'] == descriptors(), 'Original lane selection changed')
    preserved = interruption(evidence, index['interruption'])
    aggregate = evidence.load(ORIGINAL + '/results.json')
    case = next(row for row in evidence.load(REPORT + '/cases.json')['cases'] if row['id'] == CASE)
    require(aggregate['case'] == case and aggregate['performance_eligible'] is False
            and aggregate['results'] == [evidence.load(item['directory'] + '/result.json') for item in descriptors()],
            'Original aggregate does not contain exactly its five completed lanes')
    records = evidence.load(OLD_CASE + '/numerical-' + CASE + '/compiled-inputs.json')
    require(records == evidence.load(OLD_CASE + '/build-both/verified-builds.json'), 'Original admitted builds disagree')
    lanes, baseline = replay_lanes(evidence, descriptors(), records)
    return lanes, baseline, records, preserved


def replay_original(repo_root, input_index_path, input_index_sha):
    """Preflight only: the original partial case never becomes a six-lane pass."""
    require(input_index_path == HERE + '/inputs-index.json', 'Wrong original index path')
    evidence = Evidence(repo_root, input_index_path, input_index_sha)
    lanes, baseline, records, preserved = original(evidence, evidence.index)
    return {'verified': True, 'index_sha256': input_index_sha, 'payloads_verified': len(evidence.blobs),
            'cases': [CASE], 'lanes_replayed': 5, 'lanes': lanes, 'baseline_rows': baseline,
            'admitted_builds': records, 'interruption': preserved, 'full_case_qualified': False,
            'native_executions': 0, 'performance_acceptance': False}


def verify(repo_root, checkpoint_path, expected_sha):
    """Verify an explicitly selected closed continuation; never discover live data."""
    require(checkpoint_path == HERE + '/checkpoint.json', 'Wrong continuation checkpoint path')
    evidence = Evidence(repo_root, checkpoint_path, expected_sha)
    index = evidence.index
    require(index['schema'] == 'madloop-tirust-combined-one-case-closure-v1'
            and index['campaign'] == CAMPAIGN and index['performance_acceptance_eligible'] is False,
            'Unexpected closure schema/scope')
    continuation = index['continuation']
    require(continuation['input_index_path'] == HERE + '/inputs-index.json'
            and evidence.digest(continuation['input_index_path']) == continuation['input_index_sha256'], 'Original input index changed')
    old = evidence.load(continuation['input_index_path'])
    require(all(evidence.index['files'].get(path) == identity for path, identity in old['files'].items()),
            'Original evidence is missing/changed in combined closure')
    require(index['interruption'] == old['interruption'], 'Interruption binding changed')
    lanes, baseline, records, preserved = original(evidence, old)
    require(index['lanes'] == descriptors(True), 'Combined closure requires exact five original lanes plus fresh candidate9')
    require(index['manifest_sha256'] == MANIFEST == evidence.digest(REPORT + '/manifest.json'), 'Manifest changed')
    require(index['source_index_sha256'] == evidence.digest(HERE + '/source-index.json')
            and index['replay_index_sha256'] == evidence.digest(HERE + '/replay-index.json'), 'Source/replay seal changed')
    source_seal = evidence.load(HERE + '/source-index.json')
    replay_seal = evidence.load(HERE + '/replay-index.json')
    require(source_seal['schema'] == 'madloop-tirust-ew-single-lane-source-v1'
            and source_seal['native_lanes'] == [['candidate', 9]] and source_seal['points'] == 100
            and source_seal['worker_timeout_seconds'] == 14400 and source_seal['stage_cap_seconds'] == 28800
            and source_seal['cleanup_reserve_seconds'] == 60 and source_seal['automatic_retry'] is False
            and source_seal['timing_authorized'] is False, 'Unexpected continuation admission scope')
    require(replay_seal['schema'] == 'madloop-tirust-ew-continuation-replay-source-v1'
            and {'verify.py', 'test_verify.py'} <= replay_seal['files'].keys(), 'Missing replay source seal')
    require(set(source_seal['files']) == {'README.md', 'inputs-index.json', 'preflight.json', 'run.py', 'test_run.py'},
            'Incomplete continuation source seal')
    for seal in (source_seal, replay_seal):
        for name, identity in seal['files'].items():
            raw = evidence.data(HERE + '/' + relative(name))
            require(len(raw) == identity['bytes'] and sha(raw) == identity['sha256'], 'Sealed continuation source changed')
    require(source_seal['dependencies'], 'Missing sealed continuation dependencies')
    for name, identity in source_seal['dependencies'].items():
        raw = evidence.data(relative(name))
        require(len(raw) == identity['bytes'] and sha(raw) == identity['sha256'],
                'Sealed continuation dependency changed: ' + name)
    require(continuation['status_path'] == HERE + '/status.json' and continuation['stage_directory'] == STAGE
            and continuation['configuration_before_path'] == STAGE + '/configuration-before.json'
            and continuation['configuration_after_path'] == STAGE + '/configuration-after.json', 'Continuation administrative paths changed')
    require(evidence.load(continuation['configuration_before_path']) == evidence.load(continuation['configuration_after_path']),
            'Configuration changed during continuation')
    require(evidence.load(STAGE + '/compiled-inputs.json') == evidence.load(STAGE + '/verified-builds.json') == records,
            'Continuation differs from original admitted builds')
    status = evidence.load(continuation['status_path'])
    require(status['status'] == 'passed' and len(status['stages']) == 1
            and status['stages'][0]['id'] == 'numerical-' + CASE, 'Continuation is not terminal/successful')
    stage(evidence, STAGE, status['stages'][0], True)
    fresh, _ = replay_lanes(evidence, descriptors(True)[-1:], records, baseline)
    lanes += fresh
    case = next(row for row in evidence.load(REPORT + '/cases.json')['cases'] if row['id'] == CASE)
    regression = all(row['numerical_agreement_passed'] for row in lanes)
    target = all(row['ordinary_target_passed'] for row in lanes)
    summary = case_summary(lanes)
    require(index['case'] == case and index['cases'] == [summary], 'Combined case summary differs from raw replay')
    return {'verified': True, 'index_sha256': expected_sha, 'payloads_verified': len(evidence.blobs),
            'payload_bytes': sum(map(len, evidence.blobs.values())), 'cases': [CASE], 'lanes_replayed': 6, 'lanes': lanes,
            'regression_passed': regression, 'ordinary_target_passed': target, 'regression_and_target_passed': regression and target,
            'native_executions': 0, 'performance_acceptance': False, 'fpe_flags_observed': False,
            'internal_retries_observed': False, 'external_objects_rehashed': False,
            'runtime_evidence': 'retained dependency hashes only; no worker /proc maps',
            'continuation_bindings': continuation,
            'continuation': {'reason': 'user-requested daemon update',
                'original_attempt': {'path': OLD_CASE, 'status': 'interrupted',
                    'selected_completed_lanes': [f'{source}-{backend}' for source, backend in EXPECTED[:5]],
                    'partial_backend9_selected': False, 'last_saved_point': 74},
                'fresh_attempt': {'path': HERE + '/lane', 'selected_lane': 'candidate-9', 'points': 100},
                'concatenated': False}, 'interruption': preserved}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo_root')
    parser.add_argument('checkpoint_path')
    parser.add_argument('expected_sha')
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, args.checkpoint_path, args.expected_sha), indent=2, sort_keys=True))
