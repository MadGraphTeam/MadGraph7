"""Read the sealed ten-worker prefix without executing or reparsing native traces."""
from decimal import Decimal
from itertools import combinations
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import re

PREFIX_SHA = 'fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361'
PREFIX_SPECS = tuple((b, m, o) for b in (1, 6, 8, 9) for m in (4, 5) for o in (0, 1))[:10]
SUFFIX_SPECS = ((8, 5, 0), (8, 5, 1), (9, 4, 0), (9, 4, 1), (9, 5, 0), (9, 5, 1))
MATRIX_SPECS = PREFIX_SPECS + SUFFIX_SPECS
OLD_ELAPSED = Decimal('152.56272240309045')
TOTAL_BUDGET = Decimal('3000')
NEW_BUDGET = TOTAL_BUDGET - OLD_ELAPSED
ORIGINAL_FAILURE = {'type': 'ValueError', 'message': 'malformed decimal transport'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def worker_name(spec):
    return 'b%d-m%d-o%d' % spec


class Capsule:
    """Hash every outer member; decompress original files only when requested."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        require(not any(path.is_symlink() for path in (self.root, *self.root.parents)),
                'Capsule root or ancestor is symlinked')
        members = list(self.root.rglob('*'))
        require(not any(path.is_symlink() for path in members), 'Capsule directory or member is symlinked')
        index_raw = (self.root / 'evidence-index.json').read_bytes()
        require(digest(index_raw) == PREFIX_SHA, 'Closed run4 capsule index changed')
        self.index = loads(index_raw)
        require(self.index['schema'] == 'qcd-n3-genuine-qp-failed-run4-retention-v1', 'Unexpected prefix schema')
        require({str(path.relative_to(self.root)) for path in members if path.is_file()} ==
                set(self.index['files']) | {'evidence-index.json'}, 'Capsule outer membership differs')
        self.outer = {}
        for name, record in self.index['files'].items():
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                    'Unsafe capsule member')
            target = self.root / name
            require(target.is_file() and not target.is_symlink(), 'Capsule member missing or symlinked: ' + name)
            raw = target.read_bytes()
            require(len(raw) == record['bytes'] and digest(raw) == record['sha256'], 'Capsule member changed: ' + name)
            self.outer[name] = raw
        self.inventory = loads(self.outer['retention-inventory.json'])

    def raw(self, name):
        record = self.inventory['original_files'][name]
        require(record['codec'] in ('identity', 'gzip'), 'Unexpected original-file codec')
        value = self.outer[record['member']]
        if record['codec'] == 'gzip':
            value = gzip.decompress(value)
        require(len(value) == record['raw_bytes'] and digest(value) == record['raw_sha256'],
                'Retained original bytes changed: ' + name)
        return value

    def json(self, name):
        return loads(self.raw(name))


def validate(capsule, preflight, parser_sha, fixture_hashes, binding):
    """Validate pinned replay proof and native closure, without a second parser run."""
    j, raw, inventory = capsule.json, capsule.raw, capsule.inventory
    expected_names = [worker_name(spec) for spec in PREFIX_SPECS]
    worker_dirs = {name.split('/')[0] for name in inventory['actual_attempt_directories'] if name.startswith('b')}
    worker_files = {name.split('/')[1] for name in inventory['original_files'] if name.startswith('actual-run4/b')}
    require(worker_dirs == worker_files == set(expected_names), 'Prefix directory or started-worker inventory differs')
    result = j('actual-run4/result.json')
    require(result['passed'] is False and result['workers_completed'] == 9 and
            result['failure'] == ORIGINAL_FAILURE, 'Original failed run4 status changed')
    require(Decimal(str(result['elapsed_seconds'])) == OLD_ELAPSED, 'Original elapsed budget changed')
    require(result['recompiled'] is False and result['original_donor_runtime_preserved'] is True,
            'Original runtime/build policy changed')
    require(result['source_index_sha256'] == digest(raw('controller/source-index.json')), 'Original controller binding differs')
    old_binding = j('controller/bindings.json')
    manifest = j('controller/donor-closure.json')
    expected = {entry['resolved_path']: entry['sha256'] for entry in manifest['entries']}
    require(len(expected) == 15 and old_binding['runtime'] == binding['runtime'] ==
            {entry['source_path']: entry['sha256'] for entry in manifest['entries']}, 'Original fifteen-library runtime changed')
    build = j('controller/inherited-build2.json')
    require(build['runtime'] == binding['runtime'] and build['target_interpreter'] == binding['target_interpreter'] and
            result['build_sha256'] == digest(raw('controller/inherited-build2.json')), 'Original inherited build changed')
    require(inventory['actual_attempt_symlinks'] == {
            'closed-runtime/' + entry['soname']: entry['resolved_path'] for entry in manifest['entries']},
            'Original alias inventory differs')
    alias_dir = old_binding['native_library_path']
    absolute_aliases = {alias_dir + '/' + entry['soname']: entry['resolved_path'] for entry in manifest['entries']}
    proof_names = ['actual-run4/closed-runtime-proof.json']
    proof_names += ['actual-run4/' + name + '/' + leaf for name in expected_names for leaf in
                    ('closed-runtime-pretrace.json', 'closed-runtime-preworker.json')]
    for name in proof_names:
        proof = j(name)
        require(proof['passed'] is True and proof['library_count'] == 15 and proof['alias_directory'] == alias_dir,
                'Original alias guard failed')
        require({entry['soname']: (entry['alias_target'], entry['resolved_path'], entry['sha256'])
                 for entry in proof['libraries']} == {entry['soname']: (
                     entry['resolved_path'], entry['resolved_path'], entry['sha256']) for entry in manifest['entries']},
                'Original alias provider differs')
    for prefix in ['actual-run4'] + ['actual-run4/' + name for name in expected_names]:
        trace = j(prefix + '/loader.json')
        require(trace['native'] is False and trace['returncode'] == 0 and trace['process_reaped'] is True and
                trace.get('failure') is None and not trace.get('timed_out') and 0 < trace['seconds_cap'] <= 30,
                'Original loader trace did not close cleanly')
        require(trace['target_loader_environment_at_execve'] == {
                'LD_LIBRARY_PATH': alias_dir, 'LD_TRACE_LOADED_OBJECTS': '1'}, 'Original loader environment differs')
        text = raw(prefix + '/loader.log').decode()
        paths = set(re.findall(r'/[^\s()]+', text))
        require('not found' not in text and {absolute_aliases.get(path, path) for path in paths} == set(expected),
                'Original pre-entry loader closure differs')
    for name, expected_sha in fixture_hashes.items():
        require(digest(raw('controller/' + name)) == expected_sha, 'Original fixture differs: ' + name)
    require(preflight['passed'] is True and preflight['parser_sha256'] == parser_sha and
            preflight['fixture_sha256'] == fixture_hashes, 'Pinned preflight parser or fixture identity differs')
    require(preflight['closed_failed_run4_retention_index_sha256'] == PREFIX_SHA and
            preflight['candidate_parser_full_workers_passed'] == 10 and preflight['matrix_workers_required'] == 16 and
            preflight['original_source4_attempt_remains_failed'] is True and
            preflight['original_source4_parser_failure'] == ORIGINAL_FAILURE and
            preflight['adjudication_executed'] is False and preflight['scientific_acceptance'] is False and
            preflight['public_plain_observed_bytes_equal'] is True and
            preflight['all_partial_common_inputs_equal'] is True, 'Preflight scope or status changed')
    workers = preflight['workers']
    require([worker['worker'] for worker in workers] == expected_names, 'Preflight prefix is missing, extra, reordered or duplicated')
    parsed_workers = []
    for spec, worker in zip(PREFIX_SPECS, workers):
        name = worker_name(spec)
        prefix = 'actual-run4/' + name
        execution, mapping = j(prefix + '/worker.json'), j(prefix + '/mapped-runtime.json')
        require(execution['native'] is True and execution['returncode'] == 0 and execution['process_spawned'] is True and
                execution['process_reaped'] is True and execution.get('failure') is None and
                not execution.get('timed_out') and 0 < execution['seconds_cap'] <= 180,
                'Inherited native worker did not close cleanly: ' + name)
        require(execution['target_loader_environment_at_execve'] == {'LD_LIBRARY_PATH': alias_dir}, 'Original worker environment differs')
        require(mapping['expected_runtime_sha256'] == mapping['mapped_files_sha256_after_exit'] == expected and
                set(mapping['mapped_elf_paths']) == set(expected) | {build['executable']} and
                mapping['executable'] == build['executable'] and mapping['cpu_allowed_list'] == '12',
                'Inherited actual mapped closure differs: ' + name)
        maps = raw(prefix + '/maps.txt')
        require(mapping['maps_text'].encode() == maps and mapping['maps_sha256'] == digest(maps) ==
                execution['mapping_capture']['complete_snapshot_sha256'], 'Inherited maps capture changed')
        for leaf, expected_sha in j(prefix + '/outputs.json').items():
            require(digest(raw(prefix + '/' + leaf)) == expected_sha, 'Inherited output inventory differs')
        cwd = execution['cwd'].rstrip('/') + '/'
        for path, expected_sha in j(prefix + '/inputs.json').items():
            require(path.startswith(cwd) and digest(raw(prefix + '/' + path[len(cwd):])) == expected_sha,
                    'Inherited input inventory differs')
        trace, stdout = raw(prefix + '/trace.txt'), raw(prefix + '/worker.log')
        parsed = worker['parsed']
        require(worker['trace_sha256'] == parsed['trace_sha256'] == digest(trace) and
                worker['stdout_sha256'] == parsed['stdout_sha256'] == digest(stdout), 'Preflight raw-worker binding differs')
        require((parsed['backend'], parsed['mode'], parsed['observed']) == spec and
                parsed['host_checked'] is False and parsed['public_calls'] == 5 and parsed['diagnostic_calls'] == 3 and
                parsed['warnings'] == [] and j(prefix + '/warnings.json') == [], 'Inherited parsed identity or protocol differs')
        require([point['point'] for point in parsed['points']] == [1, 93, 39], 'Inherited diagnostic point IDs differ')
        for point in parsed['points']:
            require(point['status'] == 120 and point['accuracy'] == [-1., -1.] and point['host_checked'] is False,
                    'Inherited unchecked diagnostic status differs')
            if spec[2]:
                require(isinstance(point['inputs'], dict) and re.fullmatch(
                        r'[0-9a-f]{64}', point['inputs']['shared_input_sha256']) is not None,
                        'Inherited shared-input digest missing or malformed')
            else:
                require(point['inputs'] is None, 'Plain arm unexpectedly claims observed inputs')
        parsed_workers.append(parsed)
    for index in range(0, 10, 2):
        require(parsed_workers[index]['public_raw'] == parsed_workers[index + 1]['public_raw'],
                'Inherited public plain/observed parity differs')
    checks = []
    observed_workers = [worker for worker in parsed_workers if worker['observed']]
    for left, right in combinations(observed_workers, 2):
        if left['backend'] != right['backend'] and left['mode'] != right['mode']:
            continue
        for a, b in zip(left['points'], right['points']):
            equal = a['inputs']['shared_input_sha256'] == b['inputs']['shared_input_sha256']
            require(equal, 'Inherited prefix common-input digest differs')
            checks.append({'left': [left['backend'], left['mode']], 'right': [right['backend'], right['mode']],
                           'point': a['point'], 'shared_input_equal': equal})
    require(len(checks) == 18 and preflight['partial_common_input_checks'] == checks,
            'Pinned partial common-input checks differ')
    return {'workers': parsed_workers, 'worker_ids': expected_names,
            'inherited_elapsed_seconds': float(OLD_ELAPSED), 'new_phase_budget_seconds': float(NEW_BUDGET),
            'new_phase_budget_seconds_decimal': str(NEW_BUDGET), 'prefix_index_sha256': PREFIX_SHA,
            'original_source4_attempt_remains_failed': True}


def load(here, binding):
    here = Path(here)
    source_index = loads((here / 'source-index.json').read_bytes())
    preflight_raw = (here / 'closed-run4-preflight.json').read_bytes()
    entry = source_index['files']['closed-run4-preflight.json']
    require(len(preflight_raw) == entry['bytes'] and digest(preflight_raw) == entry['sha256'], 'Pinned preflight bytes changed')
    fixtures = {'fixtures/' + str(path.relative_to(here / 'fixtures')): digest(path.read_bytes())
                for path in sorted((here / 'fixtures').rglob('*')) if path.is_file()}
    capsule = Capsule(here.parent / 'failed-run4-retention-1')
    result = validate(capsule, loads(preflight_raw), digest((here / 'analyze.py').read_bytes()), fixtures, binding)
    result['preflight_sha256'] = digest(preflight_raw)
    return result


def require_complete_matrix(workers):
    actual = [(worker['backend'], worker['mode'], worker['observed']) for worker in workers]
    require(actual == list(MATRIX_SPECS) and len(set(actual)) == 16,
            'Adjudication requires exactly sixteen unique workers in the original order')
