"""Replay a closed raw stability capsule without following recorded host paths."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import types

SOURCE5_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
PREPARATION_SHA = '1588341143201f03700361de79b9bfa43e2dd4485d5cb0fd9887cccddbbef682'
CASE_IDS = ('qcd_n2_gg_ttx', 'qcd_n3_uux_zgg', 'ew_n2_epem_mupmum', 'qcd_n4_uux_zzgg')
BACKENDS = (1, 6, 7, 8, 9)
MAX_RAW_TEXT_BYTES = 512 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def local_path(root, name):
    path = Path(name)
    require(isinstance(name, str) and name and not path.is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe capsule path')
    result = root
    for part in path.parts:
        result = result / part
        require(not result.is_symlink(), 'Capsule symlink is forbidden')
    require(result.is_file(), 'Missing capsule file: ' + name)
    return result


def load_module(root, name, files, namespace):
    blob = local_path(root, name).read_bytes()
    require(sha(blob) == files[name]['sha256'], 'Source checksum differs: ' + name)
    module = types.ModuleType(namespace)
    module.__file__ = str(root / name)
    exec(compile(blob, module.__file__, 'exec'), module.__dict__)
    return module


def check_admission(index, retention):
    """Join historical capture claims to the actual indexed original bytes."""
    terminal = index['payloads']['receipts/run-terminal.json']['sha256']
    require(index['root_admitted_run_terminal_sha256'] == retention['run_terminal_sha256'] == terminal,
            'Root-admitted run terminal hash differs')
    require(index['source5_index_sha256'] == retention['source5_index_sha256'] == SOURCE5_SHA,
            'Retention source identity differs')
    require(index['payloads']['receipts/prepare-terminal.json']['sha256'] == PREPARATION_SHA,
            'Fixed inherited preparation differs')
    require(retention['original_hashes_rechecked_after_capture'] is True,
            'Original capture recheck was not retained')
    logical = []
    for item in retention['original_allowlist'].values():
        name = item['logical_payload']; stored = index['payloads'][name]
        require(item['bytes'] == stored['bytes'] and item['sha256'] == stored['sha256'],
                'Original allowlist binding differs')
        logical.append(name)
    require(len(logical) == len(set(logical)) and set(logical) == set(index['payloads']),
            'Original allowlist does not cover every logical payload exactly once')


def summarize_records(dataset, records, metrics):
    """Fixed physical-point and primary-call populations; no truth inference."""
    by_id = {row['request']['id']: row for row in records}
    rows = []
    for case in dataset['cases']:
        for lane in case['lanes']:
            backend = lane['backend']; record = by_id[case['id'] + '-backend' + str(backend)]
            parsed = record.get('diagnostics', {})
            primary = [row for row in parsed.get('rows', []) if row['phase'] == 1 and row['order'] == 0]
            bootstrap = [row for row in parsed.get('rows', []) if row['phase'] == 0 and row['order'] == 0]
            outcomes = parsed.get('primary_call_outcomes', record.get('primary_call_outcomes', []))
            require(len(outcomes) == 300, 'Every lane must retain 300 intended primary-call outcomes')
            groups = [row for row in parsed.get('loop_group_calls', []) if row['phase'] == 1]
            values = [metrics.metric_value(value) for value in lane['values']]
            failure_counts = {
                'coverage_missing': sum(not row['coverage'] for row in groups),
                'active_group_failure': sum(row['failed_groups'] > 0 for row in groups),
                'nonfinite_coefficients': sum(row['nonfinite_coefficient_groups'] > 0 for row in groups),
                'nonfinite_results': sum(row['nonfinite_result_groups'] > 0 for row in groups),
                'floating_point_failure_latch': sum(row['fpe_dp'] or row['fpe_qp']
                    or row['fpe_dp_before'] or row['fpe_qp_before'] for row in groups),
            }
            rows.append({'case': case['id'], 'backend': backend,
                'result_state': record['state'], 'native_started': record['native_started'],
                'error': record.get('error'), 'diagnostic_error': record.get('diagnostic_error'),
                'readonly_guard_error': record.get('readonly_guard_error'),
                'intended_points': len(values), 'available_points': sum(value is not None for value in values),
                'unavailable_points': sum(value is None for value in values),
                'zero_points': sum(value == 0 for value in values if value is not None),
                'infinite_points': sum(value == float('inf') for value in values if value is not None),
                'return_code_counts': dict(sorted(Counter(str(row['status']) for row in primary).items())),
                'bootstrap_return_code_counts': dict(sorted(Counter(str(row['status']) for row in bootstrap).items())),
                'primary_outcome_counts': dict(sorted(Counter(row['state'] for row in outcomes).items())),
                'group_failure_call_counts': failure_counts,
                'tail_at_1e3': metrics.tail_bounds(lane['values'], 1e-3)})
    require(len(rows) == 20, 'The complete four-case/five-backend table is required')
    return {'schema': 'madloop-raw-stability-summary-v1', 'rows': rows,
        'interpretation': '100 cut-qualified physical points per case; host forced-DP/card-selected backend; unchecked internal consistency, not independent true error or performance acceptance',
        'precision_observation': 'reducer-internal precision and actual backend are unobserved; 110 is an unchecked host diagnostic return code',
        'tail_interpretation': 'all 100 intended points remain in the denominator; unavailable outcomes give lower/upper bounds, not measured failure fractions; ratios require complete paired populations and a nonzero Ninja tail'}


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    root = root.resolve()
    index_blob = (root / 'index.json').read_bytes(); index = json.loads(index_blob)
    require(index['schema'] == 'madloop-raw-stability-capsule-v1', 'Unexpected capsule schema')
    files = index['files']
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() or path.is_symlink()}
    require(actual == set(files) | {'index.json'}, 'Capsule inventory differs')
    for name, item in files.items():
        path = local_path(root, name)
        require(path.stat().st_size == item['bytes'], 'File size differs: ' + name)
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        require(hasher.hexdigest() == item['sha256'], 'File checksum differs: ' + name)
    codec = load_module(root, 'journal_codec.py', files, 'capsule_journal_codec')
    check_admission(index, json.loads(local_path(root, 'retention.json').read_bytes()))

    def payload(name):
        storage = index['payloads'][name]
        require(type(storage['bytes']) is int and 0 <= storage['bytes'] <= MAX_RAW_TEXT_BYTES,
                'Text payload budget exceeded')
        if storage['kind'] == 'file':
            path = local_path(root, storage['path'])
            require(path.stat().st_size == storage['bytes'], 'Direct text payload size differs before allocation')
            blob = path.read_bytes()
        elif storage['kind'] == 'gzip-chunks':
            blob = codec.decode_journal(storage['descriptor'], root, total_raw_budget=MAX_RAW_TEXT_BYTES)
        else:
            raise ValueError('Unknown payload storage')
        require(len(blob) == storage['bytes'] and sha(blob) == storage['sha256'], 'Original payload differs: ' + name)
        return blob

    read = lambda name: json.loads(payload(name))
    source_index_blob = local_path(root, 'source5/source-index.json').read_bytes()
    require(sha(source_index_blob) == SOURCE5_SHA == index['source5_index_sha256'], 'Frozen source5 seal differs')
    source_index = json.loads(source_index_blob)
    for name, item in source_index['files'].items():
        blob = local_path(root, 'source5/' + name).read_bytes()
        require(len(blob) == item['bytes'] and sha(blob) == item['sha256'], 'Frozen source5 payload differs')
    pilot = load_module(root, 'source5/pilot.py', files, 'capsule_frozen_pilot')
    metrics = load_module(root, 'source5/plot_metrics.py', files, 'capsule_frozen_metrics')
    require(files['point_validator.py']['sha256'] == source_index['inputs']['tools/tirust_benchmark_points.py'],
            'Frozen physical-point validator differs')
    validator = load_module(root, 'point_validator.py', files, 'capsule_physical_points')
    value = json.loads(local_path(root, 'source5/protocol.json').read_bytes())
    terminal = read('receipts/run-terminal.json'); prepared = read('receipts/prepare-terminal.json')
    require(terminal['source_index_sha256'] == SOURCE5_SHA and terminal['request_count'] == 20,
            'Run source/request identity differs')
    require(terminal['preparation_sha256'] == index['payloads']['receipts/prepare-terminal.json']['sha256']
            and prepared['status'] == 'passed' and prepared['source_index_sha256'] == SOURCE5_SHA,
            'Preparation identity differs')
    require(tuple(prepared['completed_cases']) == CASE_IDS and set(prepared['cases']) == set(CASE_IDS),
            'Preparation must contain all four cases')
    require([row['request'] for row in terminal['requests']] == pilot.requests(value), 'Fixed request schedule differs')
    card_observations = json.loads(local_path(root, 'card-observations.json').read_bytes())
    require(set(card_observations) == {row['request']['id'] for row in terminal['requests'] if 'effective_card' in row},
            'Canonical card observation population differs')
    for case_id, case in prepared['cases'].items():
        data = case['donor']; point_name = 'points/' + case_id
        require(sha(payload('cards/' + case_id + '/param_card.dat')) == data['physical_card_sha256'],
                'Frozen physical parameter-card bytes differ')
        point_blob = payload(point_name + '.dat'); metadata = read(point_name + '.json')
        require(sha(point_blob) == data['points_sha256']
                and index['payloads'][point_name + '.json']['sha256'] == data['point_metadata_sha256'],
                'Frozen physical point-file binding differs')
        inputs = metadata['inputs']
        require(inputs['incoming_masses'] + inputs['masses'] == data['compiled']['masses']
                and inputs['resolved_physical_card_sha256'] == data['physical_card_sha256']
                and inputs['seed'] == value['sampling']['seeds'][case_id]
                and inputs['count'] == len(metadata['points']) == 100 and inputs['energy'] == 1000.,
                'Frozen point sampling/mass/card identity differs')
        validator.validate_points(metadata['points'], inputs['masses'], incoming_masses=inputs['incoming_masses'])
        serialized = '100\n' + ''.join(' '.join('%.17e' % x for x in vector) + '\n'
            for point in metadata['points'] for vector in point['momenta'])
        require(serialized.encode() == point_blob, 'Point metadata and Fortran vectors differ')
        cuts = pilot.cut_report(pilot.point_array(point_blob.decode(), len(data['compiled']['masses'])), 50., .5)
        require(cuts['passed'] == 100, 'Frozen points fail declared input cuts')
    request_hashes = {}
    for record in terminal['requests']:
        request = record['request']; rid = request['id']; case = prepared['cases'][request['case']]
        logical = 'requests/' + rid + '/result.json'
        if logical in index['payloads']:
            require(read(logical) == record, 'Request/terminal receipts disagree')
            request_hashes[logical] = index['payloads'][logical]['sha256']
        else:
            require(record['state'] == 'unexecuted' and record['native_started'] is False,
                    'Executed request lacks its original receipt')
        if 'effective_card' in record:
            effective = record['effective_card']
            observation = card_observations[rid]
            logical_card = 'cards/' + rid + '/MadLoopParams.dat'
            require(observation['status'] in ('regular', 'missing', 'symlink', 'nonregular'),
                    'Unknown canonical card observation')
            if observation['status'] == 'regular':
                card = payload(logical_card)
                require(len(card) == observation['bytes'] and sha(card) == observation['sha256'],
                        'Canonical post-run card observation differs')
            else:
                require(logical_card not in index['payloads'] and record.get('readonly_guard_passed') is not True,
                        'Guard-passed card is not a retained regular file')
            if record.get('readonly_guard_passed') is True:
                require(all(checksum == sha(card) for checksum in effective['files'].values())
                        and pilot.verify_card_tokens(card.decode(), pilot.card_settings(request['backend'])) == effective['settings']
                        and all(observed == effective['settings'] for observed in effective['readback'].values()),
                        'Effective numerical-card bytes/settings differ')
        raw_files = record.get('raw_files', {})
        for filename, item in raw_files.items():
            require(filename in ('stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat'),
                    'Unknown collected diagnostic filename')
            logical_raw = 'requests/' + rid + '/' + filename
            require(index['payloads'][logical_raw]['sha256'] == item['sha256']
                    and index['payloads'][logical_raw]['bytes'] == item['size'], 'Collected raw-file binding differs')
        if record.get('native_started'):
            prefix = 'requests/' + rid + '/native/'
            command = read(prefix + 'command.json')
            require(command == record['native'], 'Native command receipt differs')
            console = payload(prefix + 'stdout.txt')
            require(sha(console) == command['stdout_sha256'], 'Native stdout differs')
            maps = {}
            for sample in command['runtime_map_samples']:
                require(Path(sample['path']).name == sample['path'], 'Unsafe map-sample name')
                blob = payload(prefix + sample['path'])
                require(sha(blob) == sample['sha256'], 'Native map sample differs')
                maps[sample['path']] = blob.decode()
            binding = record.get('runtime_binding', {})
            if binding.get('verified') is True:
                expected = dict(case['runtime_closure']['files'])
                executable = command['target_argv'][0]
                expected[executable] = case['executable_sha256']
                require(binding.get('backend_observed') is False and binding['samples'], 'Unchecked backend identity was fabricated')
                complete_map = False
                for sample in binding['samples']:
                    require(sample['path'] in maps, 'Runtime-binding map is missing')
                    mapped_rows = [fields for line in maps[sample['path']].splitlines()
                        if len(fields := line.split(maxsplit=5)) == 6 and fields[5].startswith('/')]
                    mapped_paths = {fields[5] for fields in mapped_rows}
                    require(set(sample['elf_files']).issubset(mapped_paths)
                            and all(expected.get(path) == checksum for path, checksum in sample['elf_files'].items()),
                            'Runtime observer receipt disagrees with retained map/closure')
                    require(set(sample['elf_files']) | set(sample['read_only_locale']) == mapped_paths,
                            'Runtime observer omitted a retained file-backed mapping')
                    require(all(path.startswith('/nix/store/') and path.endswith('/lib/locale/locale-archive')
                                and all(fields[1] == 'r--p' for fields in mapped_rows if fields[5] == path)
                                for path in sample['read_only_locale']),
                            'Runtime locale observation has unsupported path or permissions')
                    complete_map |= set(sample['elf_files']) == set(expected)
                require(complete_map, 'Exact application/runtime closure was not observed')
            required = {'stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat'}
            if required.issubset(raw_files):
                try:
                    parsed = pilot.parse_diagnostics(
                        payload('requests/' + rid + '/stability-dimensions.dat').decode(),
                        payload('requests/' + rid + '/stability-values.dat').decode(), console.decode(errors='replace'),
                        payload('requests/' + rid + '/stability-groups.dat').decode(), case['group_layout'])
                except Exception as error:
                    require('diagnostics' not in record and record.get('diagnostic_error', {}).get('class') == type(error).__name__,
                            'Raw parser failure disagrees with retained outcome')
                else:
                    require(parsed == record.get('diagnostics'), 'Full raw diagnostic/group replay differs')
            else:
                require('diagnostics' not in record, 'Diagnostics claim unavailable raw inputs')
    dataset = read('point-metrics.json'); metrics.validate_dataset(dataset)
    expected = pilot.point_metrics(value, terminal['requests'], metrics.relative_spread)
    expected['provenance'].update({
        'closed_run_terminal_sha256': index['payloads']['receipts/run-terminal.json']['sha256'],
        'preparation_checkpoint_sha256': index['payloads']['receipts/prepare-terminal.json']['sha256'],
        'fresh_point_streams': {case: {'sha256': item['donor']['points_sha256'],
            'metadata_sha256': item['donor']['point_metadata_sha256'],
            'physical_card_sha256': item['donor']['physical_card_sha256']} for case, item in prepared['cases'].items()},
        'closed_request_records': request_hashes})
    require(dataset == expected, 'All-point metrics or unavailable reasons differ from frozen implementation')
    summary = summarize_records(dataset, terminal['requests'], metrics)
    if 'summary.json' in files:
        require(json.loads(local_path(root, 'summary.json').read_bytes()) == summary, 'Summary differs')
    return {'verified': True, 'index_sha256': sha(index_blob), 'summary': summary,
        'scope': 'lossless local raw-parser/group/point-metric replay; historical runtime observer receipts are checked against retained maps, without reopening original host binaries or paths'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
