#!/usr/bin/env python3
"""Verify this closed, zero-amplitude CMS failure using retained text only.

No driver is imported and no recorded binary or absolute receipt path is read.
Verification establishes integrity and the failed attempt's accounting, rather
than numerical success or a new runtime identity check.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

INDEX = 'evidence-index.json'
CASE = 'cms_ew_n2_epem_mupmum'
REQUEST_IDS = ['baseline-1-auto', 'candidate-1-auto', 'candidate-6-auto',
               'candidate-7-auto', 'candidate-8-auto', 'candidate-9-auto']
FIXED_BINDING = {
    'campaign_manifest_sha256': 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0',
    'generator_inventory_sha256': 'a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19',
    'pre_generator_binding_index_sha256': '3fc31b25e2e3a9a6c11f5d0c15054cb78f32536655f63ba188763120b1d5bfc7',
    'proposal_index_sha256': '9c1703b96ec3933b9f8644a4850db2d9742579a109d4fdb7e72d3d776f0eba2e',
    'supervisor_index_sha256': 'edf6d8cacfe1ad72cdb94584cb6d9a49e022e976c52d0668d670b702c28c84d6',
    'current_revision': '2c13246ffa63aa357ad1e29e5e79aff63d88dc7e',
    'baseline_revision': '9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f',
    'current_native_dso_sha256': 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
    'points_sha256': 'a0b0bf140915a5b65600eda9dc6ca87f7e4749f6d3b1cab3ecf4b28616051b60',
    'physical_card_sha256': '96158fedb398f3a7957bff10db0a7101281eb57ef0e9973e9c6e43c727efb8ec',
    'point_count': 21, 'MLStabThres': 1e-3, 'OSThres': 1e-8,
    'regression_tolerance': 3e-6, 'ordinary_target': 1e-8,
    'complex_squared_mass': 'M*M - i*M*Gamma',
    'width_sign': 'positive widths; negative imaginary squared masses',
    'geometry_source': 'source/proposal1/points/cms_ew_n2_epem_mupmum.dat',
    'geometry_modified': False,
}


class EvidenceError(ValueError):
    pass


def require(value, message):
    if not value:
        raise EvidenceError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _path(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid relative evidence path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
            'Unsafe or noncanonical evidence path: ' + name)
    return name


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _float(text):
    value = float(text)
    require(math.isfinite(value), 'Nonfinite JSON number')
    return value


def _json(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_float=_float,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON constant: ' + value))


def filesystem_reader(root):
    """Create a portable, strict inventory of regular files without symlinks."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Evidence root must be a real directory')
    paths = []
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlink in evidence: ' + str(path))
        require(path.is_file() or path.is_dir(), 'Special file in evidence: ' + str(path))
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())

    def read(name):
        name = _path(name)
        target = root / name
        require(not target.is_symlink() and target.is_file(), 'Missing/symlink evidence: ' + name)
        return target.read_bytes()
    read.paths = tuple(paths)
    return read


class _Reader:
    def __init__(self, read):
        self.read = read
        self.paths = set(_path(name) for name in read.paths) if hasattr(read, 'paths') else None
        self.cache = {}
        self.retention = self.json('retention.json')

    def raw(self, name):
        name = _path(name)
        if name not in self.cache:
            value = self.read(name)
            require(isinstance(value, bytes), 'Reader must return bytes: ' + name)
            self.cache[name] = value
        return self.cache[name]

    def json(self, name):
        return _json(self.raw(name))

    def closed(self, logical):
        _path(logical)
        record = self.retention['closed_files'][logical]
        require(record['path'] == 'closed/' + logical + '.gz', 'Closed path mapping differs: ' + logical)
        raw = gzip.decompress(self.raw(record['path']))
        raw.decode('utf-8')
        require(len(raw) == record['bytes'] and digest(raw) == record['sha256'],
                'Closed decoded digest/size differs: ' + logical)
        return raw

    def closed_json(self, logical):
        return _json(self.closed(logical))


def _hash_map(value, label):
    require(isinstance(value, dict) and value, 'Missing hash map: ' + label)
    require(all(isinstance(path, str) and isinstance(checksum, str)
                and re.fullmatch('[0-9a-f]{64}', checksum) for path, checksum in value.items()),
            'Malformed hash map: ' + label)


def _subset(actual, bound, label):
    _hash_map(actual, label)
    _hash_map(bound, label + ' bound')
    require(all(bound.get(path) == checksum for path, checksum in actual.items()),
            'Actual LOAD/object absent from bound inputs: ' + label)


def _original_logical(path, reader):
    original = PurePosixPath(reader.retention['original_closed_root'])
    value = PurePosixPath(path)
    require(value.is_relative_to(original), 'Receipt path leaves original closed root')
    return _path(value.relative_to(original).as_posix())


def _source_bindings(reader):
    manifest = reader.json('source/ancestry/docs/benchmarkfolder/campaigns/invariant-symmetry-1/manifest.json')
    require(manifest['candidate']['exporter_identity']['files_sha256'] == FIXED_BINDING['generator_inventory_sha256'],
            'Generator inventory binding differs')
    lineage = reader.json('lineage.json')
    require(lineage == {
        'schema_version': 1,
        'kind': 'additive-generator-summary-binding',
        'preserved_predecessor': 'current-cms-replay-retention-1-before-generator-binding-1',
        'index_sha256': FIXED_BINDING['pre_generator_binding_index_sha256'],
        'summary_sha256': '0b9a39d5f3cfd02d01abbc5777c1a6ef3cb0734fbd55cc4cb2625efe5e38d171',
        'verifier_sha256': '520c9b075a0b54530bd681510a1e0849c76a81091860b413fcf68c65a100e7e8',
        'scope': 'Explicit generator inventory summary binding and lineage; original closed evidence and scientific policy unchanged',
    }, 'Preserved provisional capsule lineage differs')
    for prefix, key in (('source/proposal1', 'proposal_index_sha256'),
                        ('source/supervisor1', 'supervisor_index_sha256')):
        index_name = prefix + '/source-index.json'
        require(digest(reader.raw(index_name)) == FIXED_BINDING[key], 'Original source index differs: ' + prefix)
        index = reader.json(index_name)
        expected = {index_name}
        for name, record in index['files'].items():
            target = prefix + '/' + _path(name)
            raw = reader.raw(target)
            require(digest(raw) == record['sha256'] and len(raw) == record['size'],
                    'Original sealed source bytes differ: ' + target)
            expected.add(target)
        if reader.paths is not None:
            require({name for name in reader.paths if name.startswith(prefix + '/')} == expected,
                    'Original source inventory differs: ' + prefix)
    proposal_index = reader.json('source/proposal1/source-index.json')
    require(proposal_index['proposal_sha256'] == digest(reader.raw('source/proposal1/proposal.json')),
            'Proposal self binding differs')
    require(proposal_index['external_input_bindings_sha256'] == digest(reader.raw('source/proposal1/input-bindings.json')),
            'Ancestry map binding differs')
    ancestry = reader.json('source/proposal1/input-bindings.json')
    require(len(ancestry) == 43, 'Expected 43 retained ancestry inputs')
    for name, checksum in ancestry.items():
        require(digest(reader.raw('source/ancestry/' + _path(name))) == checksum,
                'Ancestry bytes differ: ' + name)
    if reader.paths is not None:
        require({name for name in reader.paths if name.startswith('source/ancestry/')} ==
                {'source/ancestry/' + name for name in ancestry}, 'Ancestry inventory differs')


def _model_proof(raw, expected):
    lines = raw.decode('utf-8').splitlines()
    require(len(lines) == 3 and float(lines[0]) == 1., 'Runtime CMSParam/row count differs')
    result = {'CMSParam': 1., 'masses': {}}
    for name, line in zip(('W', 'Z'), lines[1:]):
        values = list(map(float, line.split()))
        require(len(values) == 4 and all(math.isfinite(value) for value in values), 'Nonfinite CMS model values')
        mass, width, real, imag = values
        require(mass == expected[name]['mass'] and width == expected[name]['width'] and width > 0,
                'Frozen model mass/width differs: ' + name)
        error = abs(complex(real, imag) - complex(mass * mass, -mass * width)) / mass ** 2
        require(imag < 0 and error <= 1e-13, 'Complex mass sign/reconstruction differs: ' + name)
        result['masses'][name] = {'mass': mass, 'width': width, 'squared_mass': [real, imag],
                                  'relative_reconstruction_error': error}
    return result


def _build_receipts(reader, plan, source):
    base = 'builds/' + source + '-' + CASE + '/'
    build = reader.closed_json(base + 'build.json')
    generation = reader.closed_json(base + 'generation.json')
    prelink = reader.closed_json(base + 'prelink.json')
    probe_prelink = reader.closed_json(base + 'cms-probe-prelink.json')
    preflight = reader.closed_json(base + 'preflight.json')
    _hash_map({build['executable']: build['executable_sha256']}, 'recorded main executable')
    require(build['generation_sha256'] == digest(reader.closed(base + 'generation.json'))
            and build['prelink_sha256'] == digest(reader.closed(base + 'prelink.json'))
            and preflight['build_sha256'] == digest(reader.closed(base + 'build.json')),
            'Build/generation/prelink/preflight receipt hash differs: ' + source)
    require(generation['source'] == source and generation['revision'] == plan[source + '_revision']
            and generation['case'] == plan['case'], 'Executed generator identity differs: ' + source)
    require(generation['physical_card_sha256'] == build['physical_card_sha256'] ==
            preflight['physical_card_sha256'] == FIXED_BINDING['physical_card_sha256'], 'Physical card receipt differs')
    require(generation['command_sha256'] == digest(reader.raw('source/proposal1/commands/' + source + '-' + CASE + '.mg5')),
            'CLI command receipt differs')
    admission = generation['cli_admission']
    require(admission['entrypoint'] == 'bin/madgraph' and admission['sha256'] ==
            reader.json('source/proposal1/cli-entrypoints.json')[source]['sha256']
            and admission['materialized_receipt_sha256'] == digest(reader.closed('materialized-' + source + '.json')),
            'Tracked CLI admission receipt differs')
    require(build['fresh_full_objects'] is True and build['loader_executed'] is False
            and build['masses_executed'] is False and prelink['completed_before_link'] is True
            and probe_prelink['completed_before_link'] is True, 'Preparation execution boundaries differ')
    require(prelink['bound'] == build['bound_inputs_before_after'], 'Prelink/build bound closure differs')
    _subset(build['actual_link_loads'], prelink['bound'], source + ' main link')
    _subset(build['objects'], prelink['bound'], source + ' objects')
    _subset(prelink['closure']['files'], prelink['bound'], source + ' main transitive closure')
    probe = build['probe']
    _hash_map({probe['executable']: probe['sha256']}, 'recorded probe executable')
    require(probe['prelink_sha256'] == digest(reader.closed(base + 'cms-probe-prelink.json'))
            and _original_logical(probe['prelink_receipt'], reader) == base + 'cms-probe-prelink.json'
            and probe['bound'] == probe_prelink['bound'] and preflight['probe'] == probe,
            'CMS probe receipt binding differs')
    _subset(probe['actual_LOAD'], probe_prelink['bound'], source + ' probe link')
    _subset(probe_prelink['closure']['files'], probe_prelink['bound'], source + ' probe transitive closure')
    for path_field, hash_field in (('source', 'source_sha256'), ('object', 'object_sha256'), ('model_archive', 'model_sha256')):
        require(probe['bound'].get(probe[path_field]) == probe[hash_field], 'Probe source/object/model binding differs')
    require(probe['source_sha256'] == digest(reader.raw('source/proposal1/protocol/cms_probe.f')),
            'CMS probe source differs')
    require(build['scientific_sources'] == generation['scientific_sources'], 'Scientific source receipts differ')
    for name, checksum in generation['scientific_sources'].items():
        require(generation['all_generation_files'][name]['sha256'] == checksum, 'Generated scientific inventory differs')
    archive = generation['raw_cli_archive']
    for path, checksum in archive['bound_files'].items():
        require(digest(reader.closed(_original_logical(path, reader))) == checksum, 'Raw fresh CLI archive receipt differs')
    for name, checksum in generation['raw_cli_scientific_sources'].items():
        logical = _original_logical(archive['directory'] + '/' + name, reader)
        require(digest(reader.closed(logical)) == checksum, 'Raw CLI scientific source differs')
    raw = reader.closed(base + 'cms_masses.dat')
    proof = _model_proof(raw, plan['expected_CMS_model'])
    require(preflight['cms_masses_sha256'] == digest(raw) and preflight['complex_model'] == proof,
            'Raw CMS proof differs from preflight receipt')
    require(preflight['masses'] == plan['expected_CMS_model']['external_masses'] == [0., 0., 0., 0.],
            'External mass preflight differs')
    for runtime in (preflight['runtime'], preflight['probe_runtime']):
        _hash_map(runtime['files'], 'recorded runtime identity')
        require(runtime['files'].get(runtime['loader']) == runtime['loader_sha256'], 'Recorded interpreter binding differs')
    if source == 'candidate':
        dso = plan['candidate_native_prefix'] + '/lib/libmadloop_tirust.so'
        require(preflight['runtime']['files'].get(dso) == FIXED_BINDING['current_native_dso_sha256'],
                'Recorded current native DSO differs')
    return {'registered_source_revision': plan[source + '_revision'],
            'executed_generator_revision': generation['revision'],
            'recorded_executable_sha256': build['executable_sha256'],
            'recorded_probe_sha256': probe['sha256'], 'recorded_model_archive_sha256': probe['model_sha256'],
            'complex_model': proof, 'external_masses': preflight['masses'],
            'binary_identities_recorded_only': True}


def _build_summary(reader):
    _source_bindings(reader)
    retention = reader.retention
    require(retention['role'] == 'current-cms-replay' and retention['schema_version'] == 1
            and retention['copied_before_after_sha256_equal'] is True, 'Closed retention manifest differs')
    require(retention['original_files'] == len(retention['closed_files']) == 945
            and retention['original_bytes'] == sum(record['bytes'] for record in retention['closed_files'].values()),
            'Closed retention counts differ')
    for name in retention['closed_files']:
        reader.closed(name)
    if reader.paths is not None:
        require({name for name in reader.paths if name.startswith('closed/')} ==
                {record['path'] for record in retention['closed_files'].values()}, 'Closed gzip inventory differs')
    plan = reader.json('source/proposal1/proposal.json')
    require(plan['candidate_revision'] == FIXED_BINDING['current_revision']
            and plan['baseline_revision'] == FIXED_BINDING['baseline_revision']
            and plan['manifest_sha256'] == FIXED_BINDING['campaign_manifest_sha256']
            and plan['points_sha256'] == FIXED_BINDING['points_sha256']
            and plan['physical_card_sha256'] == FIXED_BINDING['physical_card_sha256']
            and plan['point_count'] == 21 and plan['strict_host_accuracy_limit'] == 1e-3
            and plan['regression_tolerance'] == 3e-6 and plan['ordinary_target'] == 1e-8,
            'Fixed scientific proposal binding differs')
    for field, name in (('points_sha256', FIXED_BINDING['geometry_source']),
                        ('physical_card_sha256', 'source/proposal1/cards/param_card.dat')):
        require(digest(reader.raw(name)) == FIXED_BINDING[field], 'Frozen physical bytes differ')
    policies = reader.json('source/proposal1/policy-reference.json')
    require(all(policy['mlstabthres'] == 1e-3 and policy['osthres'] == 1e-8 for policy in policies.values()),
            'Primary CMS policy differs')
    freeze = reader.closed_json('source-freeze.json')
    require(freeze['proposal_index_sha256'] == FIXED_BINDING['proposal_index_sha256']
            and freeze['source_index_sha256'] == FIXED_BINDING['supervisor_index_sha256']
            and freeze['campaign_manifest_sha256'] == FIXED_BINDING['campaign_manifest_sha256'], 'Source freeze binding differs')
    stages = {name: reader.closed_json('stage-' + name + '.json') for name in ('prepare', 'preflight', 'run')}
    for name, stage in stages.items():
        require(stage['stage'] == name and stage['status'] == ('failed' if name == 'run' else 'passed')
                and stage['source_index_sha256'] == FIXED_BINDING['supervisor_index_sha256']
                and stage['campaign_manifest_sha256'] == FIXED_BINDING['campaign_manifest_sha256']
                and stage['performance_eligible'] is False
                and stage['native_execution'] is (name != 'prepare')
                and (name == 'run' or stage['exception'] is None),
                'Closed stage status/binding differs: ' + name)
    accounting = reader.closed_json('request-accounting.json')
    aborted = reader.closed_json('aborted-request.json')
    error = stages['run']['exception']
    require(isinstance(error, str) and 'Worker link resolves outside private/bound external inputs:' in error
            and '/baseline-1-auto/' in error and '/lib/ninjars_lib' in error
            and accounting['native_stage_failure'] == aborted['exception'] == error, 'Known pre-launch refusal differs')
    require(accounting['requested'] == 6 and accounting['completed'] == 0
            and [item['id'] for item in accounting['not_completed']] == REQUEST_IDS
            and [item['id'] for item in plan['requests']] == REQUEST_IDS
            and aborted['request'] == accounting['not_completed'][0]
            and aborted['numerically_qualified'] is False and aborted['state'] == 'aborted-during-request',
            'Closed request accounting differs')
    for expected, observed in zip(plan['requests'], accounting['not_completed']):
        require(all(observed.get(key) == value for key, value in expected.items()), 'Uncompleted request identity differs')
    require(not any(name.startswith(('runs/', 'request-completions/', 'execution/query-run', 'execution/request-'))
                    for name in retention['closed_files']), 'Unexpected amplitude/runtime request evidence in zero-launch capsule')
    closure = reader.json('closure.json')
    require(closure['root_observed_terminal'] is True and closure['schema_version'] == 1,
            'Root terminal attestation missing')
    for name, session in closure['sessions'].items():
        require(name in stages and session['exit_code'] == (1 if name == 'run' else 0), 'Terminal exit status differs')
    require(set(closure['sessions']) == set(stages), 'Terminal sessions incomplete')
    counters = ('amplitude_launch_count', 'worker_launch_count', 'physics_loader_launch_count', 'completed_requests')
    require(all(type(closure[name]) is int and closure[name] == 0 for name in counters), 'Zero-run launch/completion counter differs')
    require(closure['preflight_counts'] == {'CMS_model_probes': 2, 'loader_listings': 4, 'mass_probes': 2},
            'Preflight execution counts differ')
    preflight = {source: _build_receipts(reader, plan, source) for source in ('baseline', 'candidate')}
    requests = []
    for position, request in enumerate(plan['requests']):
        outcomes = lambda: [{'point': point, 'order': order, 'state': 'unexecuted'}
                            for point in range(1, 22) for order in (0, 1)]
        requests.append({'id': request['id'], 'source': request['source'], 'backend': request['backend'],
                         'state': 'failed' if position == 0 else 'unexecuted',
                         'controller_setup_started': position == 0, 'execution_started': False,
                         'numerically_qualified': False, 'amplitude_observed': False,
                         'phase_outcomes': {'warmup': outcomes(), 'final': outcomes()},
                         'phase_counts': {phase: {'accepted': 0, 'rejected': 0, 'unchecked': 0, 'unexecuted': 42}
                                          for phase in ('warmup', 'final')}})
    return {'schema_version': 1, 'role': 'current-cms-replay', 'closed': True, 'stage': 'native_terminal',
            'status': 'failed', 'overall_status': 'failed', 'terminal_exit_code': 1,
            'binding': dict(FIXED_BINDING),
            'stages': {name: {'status': stage['status'], 'exit_code': closure['sessions'][name]['exit_code']}
                       for name, stage in stages.items()},
            'gates': {'preparation_passed': True, 'preflight_passed': True, 'native_run_passed': False,
                      'numerically_qualified': False, 'independent_finite_term_oracle': False},
            **{name: 0 for name in counters}, 'amplitude_evaluations': 0, 'requested_requests': 6,
            'preflight_counts': closure['preflight_counts'], 'preflight': preflight,
            'requests': requests, 'failure': error, 'comparisons': [], 'primary_policy_agreement': None,
            'five_current_reducers_complete': False, 'performance_eligible': False,
            'binary_identities_recorded_only': True}


def build_summary(reader):
    """Derive the capsule's failed result without consulting summary.json."""
    try:
        return _build_summary(_Reader(reader))
    except EvidenceError:
        raise
    except Exception as error:
        raise EvidenceError('Malformed retained evidence: ' + str(error)) from error


def verify(root=None, *, read_bytes=None):
    """Check exact inventory, digests and receipt semantics, including failure."""
    try:
        read = filesystem_reader(Path(__file__).resolve().parent if root is None else root) if read_bytes is None else read_bytes
        require(hasattr(read, 'paths'), 'Injected reader must expose .paths for strict inventory')
        paths = tuple(_path(name) for name in read.paths)
        require(len(paths) == len(set(paths)), 'Duplicate reader inventory path')
        reader = _Reader(read)
        index_raw = reader.raw(INDEX)
        index = _json(index_raw)
        require(index['schema_version'] == 1 and index['role'] == 'current-cms-replay'
                and index['closed'] is True and index['stage'] == 'native_terminal'
                and index['binding'] == FIXED_BINDING and index['summary'] == 'summary.json', 'Evidence index header/binding differs')
        require(INDEX not in index['files'] and set(paths) == set(index['files']) | {INDEX},
                'Evidence index inventory differs')
        for name, record in index['files'].items():
            raw = reader.raw(name)
            require(type(record['bytes']) is int and len(raw) == record['bytes'] and digest(raw) == record['sha256'],
                    'Indexed digest/size differs: ' + name)
            require(record['codec'] in ('utf8', 'gzip-utf8'), 'Unknown evidence codec: ' + name)
            decoded = gzip.decompress(raw) if record['codec'] == 'gzip-utf8' else raw
            decoded.decode('utf-8')
            if 'decoded_sha256' in record:
                require(digest(decoded) == record['decoded_sha256'], 'Decoded digest differs: ' + name)
            if 'decoded_bytes' in record:
                require(len(decoded) == record['decoded_bytes'], 'Decoded size differs: ' + name)
        summary = _build_summary(reader)
        require(reader.json(index['summary']) == summary, 'Stored summary differs from retained evidence')
        return {'verified': True, 'index_sha256': digest(index_raw), 'summary': summary}
    except EvidenceError:
        raise
    except Exception as error:
        raise EvidenceError('Malformed retained evidence: ' + str(error)) from error


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
