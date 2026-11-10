#!/usr/bin/env python3
"""Verify this closed CMS startup attempt using retained text bytes only.

Recorded binary names are identities, never paths to read or execute.  The
ancestor verifier is executed only after its exact reviewed source hash matches;
its complete evidence I/O remains inside a prefix-restricted byte callback.
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
ANCESTOR_VERIFIER_SHA256 = 'af78d87e1c1ba85c9c4a64e8b0e4f872d4c3daff7825d1d8464b4d71e822352c'
FIXED_BINDING = {
    'campaign_manifest_sha256': 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0',
    'generator_inventory_sha256': 'a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19',
    'proposal_index_sha256': '59370d67a1ef278fd9de76b660beeac065bea6ebbe05b7cc8619627a98b60068',
    'scientific_proposal_index_sha256': '9c1703b96ec3933b9f8644a4850db2d9742579a109d4fdb7e72d3d776f0eba2e',
    'supervisor_index_sha256': '50112efc92cfab1c90a3147bc2acf0ff6729a9dd7ceec0206973369fc921a934',
    'ancestor_failed1_index_sha256': '7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed',
    'inherited_checkpoint_sha256': 'b3e38892b2cc86b679c9ed6c05bbad0b125a65ba6a67cb41eca6824dabb09949',
    'link_closure_audit_sha256': 'db5663ece3fa81217a11119f556a2cffa57d9fa09da7540685632cd1b768463e',
    'current_revision': '2c13246ffa63aa357ad1e29e5e79aff63d88dc7e',
    'baseline_revision': '9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f',
    'current_native_dso_sha256': 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
    'points_sha256': 'a0b0bf140915a5b65600eda9dc6ca87f7e4749f6d3b1cab3ecf4b28616051b60',
    'physical_card_sha256': '96158fedb398f3a7957bff10db0a7101281eb57ef0e9973e9c6e43c727efb8ec',
    'point_count': 21, 'MLStabThres': 1e-3, 'OSThres': 1e-8,
    'regression_tolerance': 3e-6, 'ordinary_target': 1e-8,
    'complex_squared_mass': 'M*M - i*M*Gamma',
    'width_sign': 'positive widths; negative imaginary squared masses',
    'geometry_source': 'ancestor1/source/proposal1/points/cms_ew_n2_epem_mupmum.dat',
    'geometry_modified': False,
}


class EvidenceError(ValueError):
    pass


def require(value, message):
    if not value:
        raise EvidenceError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


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


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    return value


def _json(raw, diagnostic=False):
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
    def walk(item, path=()):
        if isinstance(item, float) and not math.isfinite(item):
            # Only numerical payloads of retained parsed rows may be nonfinite.
            allowed = (diagnostic and 'diagnostics' in path and
                       any(key in path for key in ('bootstrap_rows', 'warmup_rows', 'traversal_rows'))
                       and any(key in path for key in ('coefficients', 'accuracy')))
            require(allowed, 'Nonfinite JSON outside diagnostic payload: ' + repr(path))
            return _safe(item)
        if isinstance(item, list):
            return [walk(child, path + (index,)) for index, child in enumerate(item)]
        if isinstance(item, dict):
            return {key: walk(child, path + (key,)) for key, child in item.items()}
        return item
    return walk(value)


def filesystem_reader(root):
    """Inventory only real directories and regular files; reject all symlinks."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Evidence root must be a real directory')
    paths = []
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlink in evidence: ' + str(path))
        require(path.is_file() or path.is_dir(), 'Special evidence file: ' + str(path))
        if path.is_file():
            paths.append(path.relative_to(root).as_posix())
    def read(name):
        target = root / _path(name)
        require(target.is_file() and not target.is_symlink(), 'Missing/symlink evidence: ' + name)
        return target.read_bytes()
    read.paths = tuple(paths)
    return read


class _Reader:
    def __init__(self, read):
        require(hasattr(read, 'paths'), 'Injected reader must expose .paths for strict inventory')
        paths = tuple(_path(name) for name in read.paths)
        require(len(paths) == len(set(paths)), 'Duplicate reader inventory path')
        self.paths, self.read, self.cache = set(paths), read, {}
        self.retention = self.json('retention.json')

    def raw(self, name):
        name = _path(name)
        require(name in self.paths, 'Read outside declared evidence inventory: ' + name)
        if name not in self.cache:
            raw = self.read(name)
            require(isinstance(raw, bytes), 'Reader must return bytes: ' + name)
            self.cache[name] = raw
        return self.cache[name]

    def json(self, name):
        return _json(self.raw(name))

    def closed(self, name):
        _path(name)
        record = self.retention['closed_files'][name]
        require(record['path'] == 'closed2/' + name + '.gz', 'Closed path mapping differs: ' + name)
        raw = gzip.decompress(self.raw(record['path']))
        raw.decode('utf-8')
        require(type(record['bytes']) is int and len(raw) == record['bytes'] and digest(raw) == record['sha256'],
                'Decoded closed digest/size differs: ' + name)
        return raw

    def closed_json(self, name):
        return _json(self.closed(name), diagnostic=True)


def _ancestor(reader):
    raw = reader.raw('ancestor1/verify_evidence.py')
    require(digest(raw) == ANCESTOR_VERIFIER_SHA256, 'Pinned pure ancestor verifier differs')
    def read(name):
        return reader.raw('ancestor1/' + _path(name))
    read.paths = tuple(name[len('ancestor1/'):] for name in reader.paths if name.startswith('ancestor1/'))
    scope = {'__name__': '_pinned_cms_failed1_verifier', '__file__': 'ancestor1/verify_evidence.py'}
    exec(compile(raw, 'ancestor1/verify_evidence.py', 'exec'), scope)
    try:
        result = scope['verify'](read_bytes=read)
    except Exception as error:
        raise EvidenceError('Ancestor verification failed: ' + str(error)) from error
    require(result['index_sha256'] == FIXED_BINDING['ancestor_failed1_index_sha256'], 'Ancestor index differs')
    return result['summary'], scope['_Reader'](read)


def _sources(reader):
    for prefix, key in (('source/supervisor2', 'supervisor_index_sha256'),
                        ('source/proposal2', 'proposal_index_sha256')):
        name = prefix + '/source-index.json'
        require(digest(reader.raw(name)) == FIXED_BINDING[key], 'Sealed successor index differs: ' + prefix)
        index = reader.json(name)
        expected = {name}
        for relative, record in index['files'].items():
            target = prefix + '/' + _path(relative)
            raw = reader.raw(target)
            require(digest(raw) == record['sha256'] and len(raw) == record['size'], 'Sealed source bytes differ: ' + target)
            expected.add(target)
        require({name for name in reader.paths if name.startswith(prefix + '/')} == expected,
                'Sealed source inventory differs: ' + prefix)
    require(digest(reader.raw('source/proposal2/link-closure-audit.json')) == FIXED_BINDING['link_closure_audit_sha256'],
            'Pinned link closure audit differs')


def _status(status, backend):
    mapping = {(2, 1): (1, 'DP'), (3, 9): (1, 'QP'), (2, 6): (6, 'DP'),
               (3, 8): (6, 'QP'), (2, 7): (7, 'DP'), (2, 8): (8, 'DP'),
               (3, 2): (8, 'QP'), (2, 9): (9, 'DP'), (3, 3): (9, 'QP')}
    actual = mapping.get((status // 100, status % 10))
    return actual if status // 10 % 10 in (1, 2) and actual and actual[0] == backend else None


def _warmup(raw, backend):
    rows, bootstrap, outcomes = [], [], []
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        require(len(parts) == 9, 'Malformed raw warmup row')
        call, point, order, status = map(int, parts[:4])
        coefficients = [float(token.replace('D', 'E').replace('d', 'e')) for token in parts[4:8]]
        accuracy = float(parts[8].replace('D', 'E').replace('d', 'e'))
        require(1 <= point <= 21 and order in (0, 1) and call >= 0, 'Warmup point/order/call differs')
        actual = _status(status, backend)
        finite = all(math.isfinite(value) for value in coefficients + [accuracy])
        eligible = actual is not None and finite and 0 <= accuracy <= 1e-3 and call > 0
        reasons = []
        if actual is None:
            reasons.append('Unexpected status %s for backends [%s]; initialization, failure, or another backend was used' % (status, backend))
        if not finite:
            reasons.append('Nonfinite coefficient or accuracy')
        elif not 0 <= accuracy <= 1e-3:
            reasons.append('Accuracy outside [0, 1e-3]')
        reason = '; '.join(reasons) or None
        if call == 0:
            eligible, reason = False, 'Bootstrap only; never a final observation'
        row = {'phase': 'warmup', 'line_number': number, 'raw_line': line,
               'call_index': call, 'point': point, 'order': order, 'status': status,
               'coefficients': _safe(coefficients), 'accuracy': _safe(accuracy),
               'eligible': eligible, 'reason': reason}
        if actual:
            row.update(actual_backend=actual[0], precision=actual[1])
        (bootstrap if call == 0 else rows).append(row)
    require([(row['call_index'], row['point'], row['order']) for row in bootstrap] == [(0, 1, 0), (0, 1, 1)],
            'Exactly two ordered bootstrap rows required')
    maximum = 21 if backend == 7 else 20
    require([(row['call_index'], row['point'], row['order']) for row in rows] ==
            [(point, point, order) for point in range(1, maximum + 1) for order in (0, 1)],
            'Complete ordered retained warmup call identities differ')
    for point in range(1, 22):
        for order in (0, 1):
            observed = [row for row in rows if row['point'] == point and row['order'] == order]
            last = observed[-1] if observed else None
            outcomes.append({'point': point, 'order': order,
                             'state': 'accepted' if last and last['eligible'] else 'rejected' if last else 'unexecuted',
                             'reason': last['reason'] if last else 'No row produced in this phase',
                             'observations': len(observed), 'last_call_index': last['call_index'] if last else None})
    return bootstrap, rows, outcomes


def _empty():
    return [{'point': point, 'order': order, 'state': 'unexecuted', 'reason': 'No row produced in this phase',
             'observations': 0, 'last_call_index': None} for point in range(1, 22) for order in (0, 1)]


def _counts(rows):
    return {state: sum(row['state'] == state for row in rows) for state in ('accepted', 'rejected', 'unchecked', 'unexecuted')}


def _receipt(reader, name):
    receipt = reader.closed_json(name + '.json')
    require(receipt['returncode'] == 0 and receipt['exception'] is None and receipt['cleanup_exception'] is None
            and receipt['cleanup']['survivors'] == [] and receipt['cleanup']['observed_owned_processes'],
            'Owned controller/loader receipt did not close cleanly: ' + name)
    require(receipt['log_sha256'] == digest(reader.closed(name + '.txt')), 'Owned receipt log hash differs: ' + name)
    environment = receipt['environment']
    repository = str(PurePosixPath(reader.retention['original_closed_root']).parents[3])
    private = repository + '/.tirust-build/current-cms-replay-2/'
    expected = {'PWD': receipt['cwd'], 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1',
                'MPLCONFIGDIR': private + 'matplotlib', 'XDG_CACHE_HOME': private + 'xdg/cache',
                'XDG_CONFIG_HOME': private + 'xdg/config', 'XDG_DATA_HOME': private + 'xdg/data',
                'XDG_STATE_HOME': private + 'xdg/state'}
    require(all(environment.get(key) == value for key, value in expected.items())
            and not any(key in environment for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'PYTHONPATH'))
            and environment['MAKEFLAGS'].startswith('-j1 SHELL=/nix/store/')
            and receipt['cpu'] == 12 and receipt['cpu_exclusive'] is False,
            'Recorded private environment/CPU contract differs: ' + name)
    return receipt


def _maps(reader, prefix, result, specification):
    require(result['complete_runtime_map'] is True and result['runtime_identity_failure'] is False
            and result['mapping_errors'] == [] and len(result['actual_map_captures']) == 1, 'Incomplete runtime capture')
    cap = result['actual_map_captures'][0]
    require(cap['path'] == 'maps-000.txt' and cap['sha256'] == digest(reader.closed(prefix + cap['path']))
            and cap['exact_expected'] is True and cap['elf_files'] == specification['expected_runtime'],
            'Runtime map capture metadata differs')
    mapped = {}
    for line in reader.closed(prefix + cap['path']).decode('utf-8').splitlines():
        fields = line.split(None, 5)
        require(len(fields) >= 5, 'Malformed raw runtime map')
        if len(fields) == 6 and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)'), 'Deleted raw map path')
            mapped.setdefault(fields[5], []).append(fields)
    require(set(mapped) == set(cap['elf_files']) | set(cap['data_files']), 'Raw mapped path membership differs')
    for path, record in cap['data_files'].items():
        require(path.startswith('/nix/store/') and path.endswith('/lib/locale/locale-archive')
                and record['permissions'] == 'r--p' and re.fullmatch('[0-9a-f]{128}', record['header64_hex'])
                and not record['header64_hex'].startswith('7f454c46')
                and re.fullmatch('[0-9a-f]{64}', record['sha256']) and record['size'] >= 64
                and all(fields[1] == record['permissions'] and fields[3] == record['map_device']
                        and int(fields[4]) == record['inode'] for fields in mapped[path]), 'Raw data mapping identity differs')


def _request(reader, ancestor, checkpoint, plan_request, position, closed_summary):
    request_id = plan_request['id']
    prefix = 'runs/' + CASE + '/' + request_id + '/'
    spec = reader.closed_json(prefix + 'specification.json')
    result = reader.closed_json(prefix + 'result.json')
    completion = reader.closed_json(prefix + 'completion.json')
    archived = reader.closed_json('request-completions/%02d.json' % position)
    request = spec['request']
    require(request['id'] == request_id and all(request.get(key) == value for key, value in plan_request.items()
                                               if key not in ('cwd', 'fifo_argument_appended_by_worker')),
            'Request scientific identity differs: ' + request_id)
    backend, source = request['backend'], request['source']
    original = PurePosixPath(reader.retention['original_closed_root'])
    repository = original.parents[3]
    expected_cwd = str(repository / '.tirust-build/current-cms-replay-2/workers' / request_id / CASE / 'SubProcesses/P0_epem_mupmum')
    require(spec['argv_without_fifo'] == request['argv_without_fifo'] == plan_request['argv_without_fifo']
            and request['cwd'] == spec['cwd'] == expected_cwd
            and request['fifo_argument_appended_by_worker'] == expected_cwd + '/worker-protocol.fifo'
            and spec['actual_stdin_after_READY'] == 'RUN 1 1\nSTOP 0 0\n', 'Frozen native invocation/rebased worker paths differ')
    build = ancestor.closed_json('builds/' + source + '-' + CASE + '/build.json')
    preflight = ancestor.closed_json('builds/' + source + '-' + CASE + '/preflight.json')
    inherited = checkpoint['builds'][source]
    require(spec['build'] == build and digest(ancestor.closed('builds/' + source + '-' + CASE + '/build.json')) == inherited['build_sha256']
            and spec['preflight_sha256'] == inherited['preflight_sha256']
            and digest(ancestor.closed('builds/' + source + '-' + CASE + '/preflight.json')) == inherited['preflight_sha256']
            and all(spec['runtime'][key] == inherited['runtime'][key] == preflight['runtime'][key]
                    for key in ('files', 'loader', 'loader_sha256'))
            and spec['complex_model'] == inherited['complex_model'] == preflight['complex_model']
            and spec['scientific_sources'] == build['scientific_sources'], 'Inherited build/preflight identity differs')
    require(spec['inherited_checkpoint_sha256'] == FIXED_BINDING['inherited_checkpoint_sha256']
            and spec['ancestor_failed_native_attempt'] == FIXED_BINDING['ancestor_failed1_index_sha256']
            and spec['backend'] == backend and spec['cwd'] == request['cwd']
            and spec['host_accuracy_limit'] == 1e-3 and spec['phase_timeout_seconds'] == 300
            and spec['expected_dimensions'] == {'allocated': 1, 'born_orders': 1, 'loop_orders': 1} and spec['points_count'] == 21
            and spec['points_sha256'] == FIXED_BINDING['points_sha256']
            and spec['physical_card_sha256'] == FIXED_BINDING['physical_card_sha256'], 'Specification policy differs')
    expected_runtime = dict(inherited['runtime']['files'])
    expected_runtime[spec['cwd'] + '/check'] = inherited['executable_sha256']
    require(spec['expected_runtime'] == expected_runtime, 'Expected private executable/runtime differs')
    card = reader.raw('ancestor1/source/proposal1/cards/%s-%s-MadLoopParams.dat' % (source, backend))
    settings = dict(ancestor.json('source/proposal1/policy-reference.json')[source])
    settings['mlreductionlib'] = str(backend)
    ignored = {key: value for key, value in settings.items() if key in
               ('tirustcachemaxbytes', 'tirustcomputefinite', 'tirustglobalcache', 'useqpintegrandfortirust')} if source == 'baseline' else {}
    require(reader.closed(prefix + 'MadLoopParams.dat') == card
            and spec['settings']['requested'] == settings and spec['settings']['not_applicable_to_baseline'] == ignored
            and spec['settings']['serialized_sha256'] == digest(card)
            and set(spec['numerical_cards']) == {'Cards/MadLoopParams.dat', 'SubProcesses/MadLoopParams.dat',
                                               'SubProcesses/MadLoop5_resources/MadLoopParams.dat'}
            and set(spec['numerical_cards'].values()) == {digest(card)}
            and reader.closed(prefix + 'param_card.dat') == reader.raw('ancestor1/source/proposal1/cards/param_card.dat'),
            'Frozen numerical/physical card bytes differ')
    for name in ('inherited-before.json', 'inherited-after.json'):
        require(reader.closed_json(prefix + name) == checkpoint, 'Inherited guard checkpoint differs')
    require(result['request'] == request and completion['result'] == archived['result'] == closed_summary['result'] == result
            and archived == closed_summary and archived['request'] == request,
            'Duplicate request completion/result copies differ')
    require(completion['outer_failure'] is None and completion['fatal_infrastructure_failure'] is None
            and completion['inputs_unchanged'] is True and completion['performance_eligible'] is False,
            'Outer request completion failed')
    require(result['orchestration_complete'] is True and result['fatal_controller_failure'] is False
            and result['numerically_qualified'] is False and result['performance_eligible'] is False
            and result['observation'] is None and result['failure_phase'] == 'startup'
            and result['cleanup_failure'] is None and result['collection_errors'] == []
            and result['accuracy_limit'] == 1e-3 and result['point_count'] == 21
            and result['phase_timeout_seconds'] == 300, 'Result startup/refusal contract differs')
    require(len(result['cleanup_receipts']) == 1, 'Exactly one native cleanup receipt required')
    cleanup = result['cleanup_receipts'][0]
    require(cleanup['complete'] is True and cleanup['worker_closed'] is True and cleanup['graceful_requested'] is False
            and cleanup['returncode'] == (-15 if backend == 1 else 3), 'Native worker cleanup differs')
    require(reader.closed(prefix + 'dimensions.dat').decode().split() == ['1', '1', '1'], 'Allocated dimensions differ')
    require(result['raw_outputs_sha256'] == {name: digest(reader.closed(prefix + name)) for name in ('dimensions.dat', 'warmup.dat')},
            'Raw output digest set differs or final output was invented')
    stdout = reader.closed(prefix + 'stdout.txt')
    require(result['stdout_sha256'] == digest(stdout), 'Raw stdout hash differs')
    bootstrap, rows, outcomes = _warmup(reader.closed(prefix + 'warmup.dat'), backend)
    diagnostic = result['diagnostics']
    require(diagnostic == {'bootstrap_rows': bootstrap, 'warmup_rows': rows, 'traversal_rows': [], 'parse_issues': [],
                           'phase_outcomes': {'warmup': outcomes, 'traversal': _empty()}},
            'Retained diagnostics differ from independently parsed raw warmup')
    lines = stdout.decode().splitlines()
    warnings = [{'line_number': index + 1, 'line': line, 'context_start_line': max(0, index - 3) + 1,
                 'context': '\n'.join(lines[max(0, index - 3):min(len(lines), index + 25)])}
                for index, line in enumerate(lines) if '##W03' in line]
    warning_lines = [(item['line_number'], item['line']) for item in warnings]
    require(result['incomplete_warnings'] == warnings, 'W03 warning lines/full context differ from raw stdout')
    if backend == 1:
        require(result['failure']['class'] == 'ProtocolError' and result['failure']['message'].startswith('Worker protocol timeout; see ')
                and result['protocol_messages'] == [{'error': result['failure']}]
                and len(warning_lines) == 0 and _counts(outcomes) == {'accepted': 40, 'rejected': 0, 'unchecked': 0, 'unexecuted': 2},
                'CutTools startup timeout contract differs')
    else:
        point, status = (21, 410) if backend == 7 else (20, 420)
        require(result['failure']['class'] == 'MeasurementError'
                and ('at point %s, status %s;' % (point, status)) in result['failure']['message']
                and result['protocol_messages'] == [{'message': ['ERROR', 'warmup-status', str(point), str(status)]}]
                and all(row['status'] == status and not row['eligible'] for row in rows if row['point'] == point)
                and len(warning_lines) == (2 if backend == 7 else 1), 'Warmup failure protocol differs')
    _maps(reader, prefix, result, spec)
    query = _receipt(reader, 'execution/query-run-%04d' % position)
    owned = _receipt(reader, 'execution/request-' + CASE + '-' + request_id)
    require(query['argv'] == ['taskset', '-c', '12', spec['runtime']['loader'], '--list', spec['cwd'] + '/check']
            and reader.closed('execution/query-run-%04d.txt' % position).decode() == spec['runtime']['listing'],
            'Runtime query argv differs')
    argv = owned['argv']
    python = ancestor.json('source/proposal1/proposal.json')['environment']['shell_setup']['python']
    require(len(argv) == 7 and argv[:3] == ['taskset', '-c', '12'] and argv[4] == '-B'
            and argv[3] == python + '3'
            and argv[5] == str(original.parent / 'current-cms-replay-supervisor-1/native_job.py')
            and argv[6] == str(original) + '/' + prefix + 'specification.json' and owned['cwd'] == spec['cwd'],
            'Owned native adapter invocation differs')
    require(cleanup['pid'] in {item['pid'] for item in owned['cleanup']['observed_owned_processes']},
            'Native worker cleanup PID absent from owned process receipt')
    return {'id': request_id, 'source': source, 'backend': backend,
            'state': 'startup_timeout' if backend == 1 else 'warmup_rejected',
            'controller_setup_started': True, 'execution_started': True, 'amplitude_observed': True,
            'numerically_qualified': False, 'orchestration_complete': True,
            'failure': result['failure'], 'failure_phase': 'startup',
            'bootstrap_rows': len(bootstrap), 'warmup_observed_rows': len(rows), 'W03_count': len(warning_lines),
            'native_cleanup_returncode': cleanup['returncode'], 'worker_closed': True,
            'phase_outcomes': {'warmup': outcomes, 'final': _empty()},
            'phase_counts': {'warmup': _counts(outcomes), 'final': _counts(_empty())}}


def _build_summary(reader):
    ancestor_summary, ancestor = _ancestor(reader)
    _sources(reader)
    retention = reader.retention
    require(retention['schema_version'] == 1 and retention['role'] == 'current-cms-replay'
            and retention['copied_before_after_sha256_equal'] is True
            and retention['original_files'] == len(retention['closed_files']) == 101
            and retention['original_bytes'] == sum(item['bytes'] for item in retention['closed_files'].values()),
            'Closed successor retention counts differ')
    for name in retention['closed_files']:
        reader.closed(name)
    require({name for name in reader.paths if name.startswith('closed2/')} ==
            {item['path'] for item in retention['closed_files'].values()}, 'Closed successor gzip inventory differs')
    require(digest(reader.closed('inherited-checkpoint.json')) == FIXED_BINDING['inherited_checkpoint_sha256'],
            'Inherited checkpoint bytes differ')
    inheritance = reader.closed_json('inherited-checkpoint.json')
    checkpoint = inheritance['checkpoint']
    require(checkpoint['ancestor_index_sha256'] == FIXED_BINDING['ancestor_failed1_index_sha256']
            and checkpoint['audit_sha256'] == FIXED_BINDING['link_closure_audit_sha256']
            and inheritance['source2_executed_setup'] is False and inheritance['subprocess_launch_count'] == 0
            and inheritance['mass_probe_launch_count'] == 0 and inheritance['CMS_model_probe_launch_count'] == 0,
            'Read-only inherited preparation boundary differs')
    closure = reader.json('closure.json')
    require(closure['schema_version'] == 1 and closure['root_observed_terminal'] is True
            and closure['ancestor_failed1_index_sha256'] == FIXED_BINDING['ancestor_failed1_index_sha256']
            and closure['inherited_checkpoint_sha256'] == FIXED_BINDING['inherited_checkpoint_sha256']
            and set(closure['sessions']) == {'inherit', 'run'}
            and all(item['exit_code'] == 0 for item in closure['sessions'].values()), 'Root terminal attestation differs')
    stages = {name: reader.closed_json('stage-' + name + '.json') for name in ('inherit', 'run')}
    for name, stage in stages.items():
        require(stage['stage'] == name and stage['status'] == 'passed' and stage['exception'] is None
                and stage['native_execution'] is (name == 'run') and stage['performance_eligible'] is False
                and stage['source_index_sha256'] == FIXED_BINDING['supervisor_index_sha256']
                and stage['campaign_manifest_sha256'] == FIXED_BINDING['campaign_manifest_sha256']
                and stage['ancestor_failed_native_attempt'] == FIXED_BINDING['ancestor_failed1_index_sha256']
                and stage['inherited_checkpoint_sha256'] == FIXED_BINDING['inherited_checkpoint_sha256'],
                'Closed successor stage differs: ' + name)
    require(reader.closed_json('request-accounting.json') == {'completed': 6, 'native_stage_failure': None, 'not_completed': [], 'requested': 6},
            'Completed controller request accounting differs')
    old_summary = reader.closed_json('summary.json')
    require(old_summary['request_count'] == 6 and len(old_summary['requests']) == 6
            and old_summary['complete_checked_streams'] == 0 and old_summary['comparisons'] == []
            and old_summary['primary_policy_agreement'] is None and old_summary['baseline_vs_current_CutTools'] is None
            and old_summary['all_requested_streams_checked_and_consistent'] is False
            and old_summary['five_current_backend_complete'] is False and old_summary['performance_eligible'] is False,
            'Closed controller summary promoted incomplete streams')
    plan = ancestor.json('source/proposal1/proposal.json')
    require(retention['original_closed_root'] == str(PurePosixPath(plan['environment']['shell_setup']['cwd']) /
                                                    'docs/benchmarkfolder/experiments/current-cms-replay-qualification-2'),
            'Original successor report root differs from sealed repository binding')
    require([item['id'] for item in plan['requests']] == REQUEST_IDS, 'Six frozen scientific requests differ')
    requests = [_request(reader, ancestor, checkpoint, request, number, old_summary['requests'][number - 1])
                for number, request in enumerate(plan['requests'], 1)]
    expected_files = {'inherited-checkpoint.json', 'stage-inherit.json', 'stage-run.json', 'request-accounting.json', 'summary.json'}
    for number, request_id in enumerate(REQUEST_IDS, 1):
        prefix = 'runs/' + CASE + '/' + request_id + '/'
        expected_files.update(prefix + name for name in ('warmup.dat', 'dimensions.dat', 'param_card.dat', 'MadLoopParams.dat',
                              'inherited-before.json', 'inherited-after.json', 'specification.json', 'stdout.txt', 'maps-000.txt', 'completion.json', 'result.json'))
        expected_files.add('request-completions/%02d.json' % number)
        for stem in ('execution/query-run-%04d' % number, 'execution/request-' + CASE + '-' + request_id):
            expected_files.update(stem + extension for extension in ('.json', '.txt'))
    require(set(retention['closed_files']) == expected_files, 'Closed successor exact logical inventory differs')
    return {'schema_version': 1, 'role': 'current-cms-replay', 'closed': True, 'stage': 'native_terminal',
            'status': 'incomplete', 'overall_status': 'incomplete', 'terminal_exit_code': 0,
            'failure_kind': 'warmup_incomplete_or_rejected', 'binding': dict(FIXED_BINDING),
            'stages': {name: {'status': 'passed', 'exit_code': 0} for name in stages},
            'gates': {'preparation_passed': True, 'preflight_passed': True, 'native_run_passed': True,
                      'numerically_qualified': False, 'independent_finite_term_oracle': False},
            'requested_requests': 6, 'completed_requests': 6, 'amplitude_launch_count': 6,
            'worker_launch_count': 6, 'physics_loader_launch_count': 6, 'amplitude_evaluations': None,
            'bootstrap_rows': 12, 'warmup_observed_rows': sum(item['warmup_observed_rows'] for item in requests),
            'phase_counts': {phase: {state: sum(item['phase_counts'][phase][state] for item in requests)
                                    for state in ('accepted', 'rejected', 'unchecked', 'unexecuted')}
                             for phase in ('warmup', 'final')},
            'W03_count': sum(item['W03_count'] for item in requests), 'requests': requests,
            'preflight': ancestor_summary['preflight'], 'preflight_counts': ancestor_summary['preflight_counts'],
            'preparation_origin': 'verified failed1 ancestor; no new preparation execution',
            'preflight_origin': 'inherited',
            'setup_provenance': {'kind': 'inherited', 'ancestor_index_sha256': FIXED_BINDING['ancestor_failed1_index_sha256'],
                                 'inherited_checkpoint_sha256': FIXED_BINDING['inherited_checkpoint_sha256'],
                                 'inherited_stages': inheritance['inherited_stages'],
                                 'read_only_inherit_session': closure['sessions']['inherit'],
                                 'successor_setup_subprocess_launch_count': 0, 'successor_mass_probe_launch_count': 0,
                                 'successor_CMS_model_probe_launch_count': 0},
            'numerically_qualified': False, 'complete_checked_streams': 0, 'full_final_streams': 0,
            'comparisons': [], 'primary_policy_agreement': None, 'five_current_reducers_complete': False,
            'independent_finite_term_oracle': False, 'performance_eligible': False,
            'binary_identities_recorded_only': True}


def build_summary(reader):
    """Derive the finite summary without consulting this capsule's summary.json."""
    try:
        summary = _build_summary(_Reader(reader))
        json.dumps(summary, allow_nan=False)
        return summary
    except EvidenceError:
        raise
    except Exception as error:
        raise EvidenceError('Malformed retained evidence: ' + str(error)) from error


def verify(root=None, *, read_bytes=None):
    """Verify strict byte inventory and incomplete startup accounting, portably."""
    try:
        read = filesystem_reader(Path(__file__).resolve().parent if root is None else root) if read_bytes is None else read_bytes
        reader = _Reader(read)
        raw = reader.raw(INDEX)
        index = _json(raw)
        require(index['schema_version'] == 1 and index['role'] == 'current-cms-replay'
                and index['closed'] is True and index['stage'] == 'native_terminal'
                and index['binding'] == FIXED_BINDING and index['summary'] == 'summary.json', 'Evidence index header/binding differs')
        require(INDEX not in index['files'] and reader.paths == set(index['files']) | {INDEX}, 'Evidence index inventory differs')
        for name, record in index['files'].items():
            payload = reader.raw(name)
            require(type(record['bytes']) is int and len(payload) == record['bytes'] and digest(payload) == record['sha256'],
                    'Indexed digest/size differs: ' + name)
            require(record['codec'] in ('utf8', 'gzip-utf8'), 'Unknown evidence codec: ' + name)
            decoded = gzip.decompress(payload) if record['codec'] == 'gzip-utf8' else payload
            decoded.decode('utf-8')
            if 'decoded_sha256' in record:
                require(digest(decoded) == record['decoded_sha256'], 'Indexed decoded digest differs: ' + name)
            if 'decoded_bytes' in record:
                require(len(decoded) == record['decoded_bytes'], 'Indexed decoded size differs: ' + name)
        summary = _build_summary(reader)
        require(reader.json(index['summary']) == summary, 'Stored summary differs from retained evidence')
        return {'verified': True, 'index_sha256': digest(raw), 'summary': summary}
    except EvidenceError:
        raise
    except Exception as error:
        raise EvidenceError('Malformed retained evidence: ' + str(error)) from error


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
