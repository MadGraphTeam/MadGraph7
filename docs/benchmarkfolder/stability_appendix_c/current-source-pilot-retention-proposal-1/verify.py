"""Pure replay of the one explicitly closed current-source QCD n3 pilot.

Recorded host paths are documentary strings. All bytes come from the local
capsule or an explicit byte-reader; no controller stage is invoked.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import types
import zlib

CASE = 'qcd_n3_uux_zgg'
BACKENDS = (1, 6, 7, 8, 9)
CURRENT_SHA = 'c071ec1c87ce6d5034e8a2daf07f093e1c71581ba79bcaa1c9176ca0c79e05ef'
SOURCE5_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
CLASSIFIER_SHA = '7bc7a08c2a96483100ea47c2d652694a34a4f426af2b5ba9a8317798d6003032'
CODEC_SHA = '33505d7db5186df76c974cf419b2333a8efbc1ce90ecee67a19c9edfd9de7a7a'
PREP_SHA = '9c1b45cdbf2783d2da63df8c55a744c2cb976fe4316ed715973dfbdae0e34d85'
DRY_SHA = '5bac8cc40c16eda6504764bbcb0a7675cd1fe6a819019bcb95843a595c4b50a5'
RUN_SHA = '4e6cc378cf14b9f9bab387efb02c63307a3f1d6dc2a1cbff03f4905a4d022fac'
MAX_RAW = 512 * 1024 * 1024
RAW_NAMES = ('stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat')
MAP_NAMES = tuple('runtime-maps-%02d.txt' % n for n in range(17))
ROOT_SPELLING = '/shared/external/madloop_tirust/madgraph7_madloop_tirust'
WORK_SPELLING = ROOT_SPELLING + '/.tirust-build/current-source-qcd3-raw-pilot-1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def json_value(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def safe_path(name):
    require(isinstance(name, str) and name and '\\' not in name and '\0' not in name
            and ':' not in name and not PurePosixPath(name).is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe payload path')
    return name


def checked_module(blob, name):
    module = types.ModuleType(name)
    module.__file__ = 'retained-source/' + name + '.py'
    exec(compile(blob, module.__file__, 'exec'), module.__dict__)
    return module


def pure_functions(blob, names, namespace):
    tree = ast.parse(blob)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == set(names) and len(nodes) == len(names), 'Pure function selection differs')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'checked-pure-functions', 'exec'), namespace)
    return namespace


def decode_journal(descriptor, read_bytes, codec):
    """Callback counterpart of the unchanged, pinned filesystem codec.

    The same descriptor, per-chunk and whole-file budgets and hashes apply.
    Each compressed byte string is processed as exactly one gzip member.
    """
    require(isinstance(descriptor, dict) and set(descriptor) == {'schema', 'raw_sha256', 'raw_bytes', 'chunks'}, 'Invalid descriptor')
    require(descriptor['schema'] == codec.SCHEMA, 'Wrong journal schema')
    codec._sha(descriptor['raw_sha256']); codec._size(descriptor['raw_bytes'], MAX_RAW)
    chunks = descriptor['chunks']; require(isinstance(chunks, list) and chunks, 'Missing chunks')
    names = set(); total = 0
    for item in chunks:
        require(isinstance(item, dict) and set(item) == {'path', 'raw_sha256', 'raw_bytes', 'gzip_sha256', 'gzip_bytes'}, 'Invalid chunk')
        name = str(codec._relative(item['path']))
        require(name not in names, 'Duplicate chunk'); names.add(name)
        total += codec._size(item['raw_bytes'], codec.MAX_RAW_CHUNK)
        codec._size(total, MAX_RAW)
        require(codec._size(item['gzip_bytes'], codec.MAX_GZIP_CHUNK) > 0, 'Empty gzip')
        codec._sha(item['raw_sha256']); codec._sha(item['gzip_sha256'])
    require(total == descriptor['raw_bytes'], 'Chunk total differs')
    pieces = []
    for item in chunks:
        compressed = read_bytes(item['path'])
        require(len(compressed) == item['gzip_bytes'] and sha(compressed) == item['gzip_sha256'], 'Compressed bytes differ')
        decoder = zlib.decompressobj(31)
        try:
            raw = decoder.decompress(compressed, item['raw_bytes'] + 1)
        except zlib.error as error:
            raise ValueError('Corrupt gzip') from error
        require(len(raw) == item['raw_bytes'] and decoder.eof and not decoder.unused_data
                and not decoder.unconsumed_tail and sha(raw) == item['raw_sha256'], 'Invalid/trailing/oversized gzip member')
        pieces.append(raw)
    blob = b''.join(pieces)
    require(len(blob) == descriptor['raw_bytes'] and sha(blob) == descriptor['raw_sha256'], 'Whole journal differs')
    return blob


def requests():
    return [{'id': CASE + '-backend' + str(backend), 'case': CASE, 'backend': backend,
             'lock_required': backend in (8, 9), 'primary_calls_expected': 300,
             'bootstrap_calls_maximum': 16} for backend in BACKENDS]


def replay_diagnostics(record, parser, texts, layout):
    """Preserve assignment-before-completeness-error and missing dimensions."""
    if texts['stability-dimensions.dat']:
        try:
            parsed = parser(texts['stability-dimensions.dat'], texts['stability-values.dat'],
                            texts['stdout.txt'], texts['stability-groups.dat'], layout)
        except Exception as error:
            require('diagnostics' not in record and record['state'] == 'failed' and record.get('diagnostic_error') ==
                    {'class': type(error).__name__, 'message': str(error)}, 'Parser exception differs')
            return None
        require(parsed == record.get('diagnostics'), 'Qualified raw diagnostics/annotations differ')
        if not (parsed['structurally_complete'] and parsed['loop_groups_structurally_complete']):
            require(record.get('diagnostic_error') == {'class': 'ValueError', 'message': 'Raw diagnostic journals are incomplete or structurally invalid'}
                    and record['state'] == 'failed', 'Incomplete journal outcome differs')
        return parsed
    require('diagnostics' not in record, 'Missing dimensions cannot have parsed diagnostics')
    if 'raw_diagnostics' in record or record.get('native_started'):
        require(record['state'] == 'failed' and record.get('diagnostic_unavailable') ==
                'Dimensions not produced; all 100 intended points remain unavailable', 'Missing-dimension outcome differs')
    return None


def metric_replay(current_blob, source5_blob, manifest, value, results, spread, protocol_hash):
    """Execute only the exact two frozen aggregation functions, with local pins."""
    pspace = {'math': math, 'BACKENDS': BACKENDS, 'HERE': Path('retained-current'),
              'REVISION': manifest['candidate']['exporter_identity']['revision'],
              'MANIFEST_SHA': '4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076',
              'digest': lambda path: protocol_hash}
    pure_functions(source5_blob, ('point_metrics',), pspace)
    p = types.SimpleNamespace(point_metrics=pspace['point_metrics'])
    current_space = {'types': types, 'HERE': Path('retained-current'), 'BACKENDS': BACKENDS,
                     'MANIFEST': Path('retained-manifest.json'), 'MANIFEST_SHA': pspace['MANIFEST_SHA'],
                     'read': lambda path: manifest, 'digest': lambda path: protocol_hash}
    pure_functions(current_blob, ('point_metrics',), current_space)
    return current_space['point_metrics'](p, value, results, spread)


def summarize(dataset, records, metrics):
    require(len(dataset['cases']) == 1 and dataset['cases'][0]['id'] == CASE, 'Wrong metric population')
    case = dataset['cases'][0]
    require(case['point_ids'] == list(range(1, 101)) and [lane['backend'] for lane in case['lanes']] == list(BACKENDS), 'Missing fixed slots')
    rows = []
    for record, lane in zip(records, case['lanes']):
        values = [metrics.metric_value(item) for item in lane['values']]
        require(len(values) == 100, 'All 100 points are required')
        parsed = record.get('diagnostics', {})
        groups = [item for item in parsed.get('loop_group_calls', []) if item['phase'] == 1]
        primary = [item for item in parsed.get('rows', []) if item['phase'] == 1 and item['order'] == 0]
        bootstrap = [item for item in parsed.get('rows', []) if item['phase'] == 0 and item['order'] == 0]
        outcomes = parsed.get('primary_call_outcomes')
        if outcomes is not None:
            require(len(outcomes) == 300, 'All 300 intended primary outcomes are required')
        outcome_counts = Counter(item['state'] for item in outcomes) if outcomes is not None else Counter({'unexecuted': 300})
        rows.append({'case': CASE, 'backend': lane['backend'], 'state': record['state'],
            'native_started': record['native_started'], 'error': record.get('error'),
            'diagnostic_error': record.get('diagnostic_error'), 'readonly_guard_error': record.get('readonly_guard_error'),
            'protected_guard_error': record.get('protected_guard_error'), 'intended_points': 100,
            'available_points': sum(item is not None for item in values),
            'unavailable_points': sum(item is None for item in values),
            'zero_points': sum(item == 0 for item in values if item is not None),
            'infinite_points': sum(item == math.inf for item in values if item is not None),
            'return_code_counts': dict(sorted(Counter(str(item['status']) for item in primary).items())),
            'bootstrap_return_code_counts': dict(sorted(Counter(str(item['status']) for item in bootstrap).items())),
            'primary_outcome_counts': dict(sorted(outcome_counts.items())),
            'active_group_failure_calls': sum(item['failed_groups'] > 0 for item in groups),
            'active_group_failure_observations': sum(item['failed_groups'] for item in groups),
            'coverage_missing_calls': sum(not item['coverage'] for item in groups),
            'nonfinite_coefficient_calls': sum(item['nonfinite_coefficient_groups'] > 0 for item in groups),
            'nonfinite_result_calls': sum(item['nonfinite_result_groups'] > 0 for item in groups),
            'incoming_software_latch_calls': sum(item['fpe_dp_before'] or item['fpe_qp_before'] for item in groups),
            'outgoing_software_latch_calls': sum(item['fpe_dp'] or item['fpe_qp'] for item in groups),
            'qualified_incoming_clear_calls': sum(item.get('incoming_software_latch_cleared') is True for item in groups),
            'legacy_extra_ineligible_calls': sum(item['eligible'] and not item.get('legacy_source5_eligible', item['eligible']) for item in groups),
            'tail_at_1e3': metrics.tail_bounds(lane['values'], 1e-3)})
    return {'schema': 'madloop-current-source-raw-stability-summary-v1', 'rows': rows,
            'interpretation': '100 fixed physical points; host forced-DP/card-selected internal consistency; qualified current-source software-latch semantics; no true-error, observed internal precision or performance acceptance',
            'historical_reclassification': False, 'hardware_ieee_state_observed': False,
            'numerical_acceptance_claimed': False, 'performance_acceptance_claimed': False}


def verify(root=None, *, read_bytes=None):
    if read_bytes is None:
        base = Path(root or Path(__file__).parent)
        require(base.is_dir() and not base.is_symlink(), 'Regular capsule directory required')
        def read_bytes(name):
            path = base
            for part in safe_path(name).split('/'):
                path = path / part; require(not path.is_symlink(), 'No capsule symlinks')
            require(path.is_file(), 'Missing capsule payload')
            require(0 <= path.stat().st_size <= MAX_RAW, 'Physical file allocation budget exceeded')
            return path.read_bytes()
        read_bytes.paths = {str(path.relative_to(base)) for path in base.rglob('*') if path.is_file() or path.is_symlink()}
    index_blob = read_bytes('index.json')
    require(isinstance(index_blob, bytes) and len(index_blob) <= MAX_RAW, 'Invalid/oversized index bytes')
    index = json_value(index_blob)
    require(index['schema'] == 'madloop-current-source-raw-stability-capsule-v1' and
            index['source_index_sha256'] == CURRENT_SHA and index['root_admitted_run_terminal_sha256'] == RUN_SHA,
            'Wrong capsule schema/source/closed-terminal binding')
    files = index['files']
    require(isinstance(files, dict) and files and 'index.json' not in files, 'Invalid physical inventory')
    declared_total = 0
    for name, item in files.items():
        safe_path(name)
        require(isinstance(item, dict) and set(item) == {'bytes', 'sha256'} and type(item['bytes']) is int
                and 0 <= item['bytes'] <= MAX_RAW and re.fullmatch('[0-9a-f]{64}', str(item['sha256'])), 'Invalid physical descriptor')
        declared_total += item['bytes']
        require(declared_total <= MAX_RAW, 'Total physical allocation budget exceeded')
    require(set(read_bytes.paths) == set(files) | {'index.json'}, 'Unindexed/missing physical payload')
    physical = {}
    for name, item in files.items():
        safe_path(name); blob = read_bytes(name)
        require(isinstance(blob, bytes) and {'bytes': len(blob), 'sha256': sha(blob)} == item, 'Physical bytes differ: ' + name)
        physical[name] = blob
    codec_blob = physical['journal_codec.py']; require(sha(codec_blob) == CODEC_SHA, 'Pinned codec differs')
    codec = checked_module(codec_blob, 'journal_codec')
    payloads = index['payloads']; decoded = {}
    def payload(name):
        # Frozen source packets and generated metadata are indexed physical
        # files; original closed inputs alone have logical storage descriptors.
        if name not in payloads:
            return physical[safe_path(name)]
        if name in decoded:
            return decoded[name]
        safe_path(name); item = payloads[name]
        codec._size(item['bytes'], MAX_RAW); codec._sha(item['sha256'])
        if item['kind'] == 'file':
            blob = physical[safe_path(item['path'])]
        else:
            require(item['kind'] == 'gzip-chunks', 'Unknown representation')
            blob = decode_journal(item['descriptor'], lambda name: physical[safe_path(name)], codec)
        require({'bytes': len(blob), 'sha256': sha(blob)} == {k: item[k] for k in ('bytes', 'sha256')}, 'Logical bytes differ')
        decoded[name] = blob
        return blob
    def read(name):
        return json_value(payload(name))
    admission = read('retention.json')
    require(admission['run_terminal_sha256'] == RUN_SHA and admission['preparation_sha256'] == PREP_SHA
            and admission['dryrun_sha256'] == DRY_SHA and admission['source_index_sha256'] == CURRENT_SHA
            and admission['root_reported_run_terminal'] == {'session_id': 29683, 'exit_code': 0}
            and admission['original_hashes_rechecked_after_capture'] is True, 'Root closure binding differs')
    allowlist = admission['original_allowlist']
    require(len({item['logical_payload'] for item in allowlist.values()}) == len(allowlist)
            and {item['logical_payload'] for item in allowlist.values()} == set(payloads), 'Original allowlist coverage differs')
    for item in allowlist.values():
        require({k: item[k] for k in ('bytes', 'sha256')} == {k: payloads[item['logical_payload']][k] for k in ('bytes', 'sha256')}, 'Original allowlist hash differs')
    for name in payloads:
        safe_path(name); payload(name)
    def sealed(prefix, pin):
        blob = payload(prefix + '/source-index.json'); require(sha(blob) == pin, 'Source seal differs')
        source = json_value(blob)
        for name, item in source['files'].items():
            blob = payload(prefix + '/' + safe_path(name))
            require({'bytes': len(blob), 'sha256': sha(blob)} == item, 'Source payload differs')
        return source
    current_index = sealed('pilot', CURRENT_SHA)
    source5_index = sealed('source5', SOURCE5_SHA)
    sealed('classifier', CLASSIFIER_SHA)
    value = read('pilot/protocol.json'); manifest_blob = payload('qualification/manifest.json'); manifest = json_value(manifest_blob)
    classifier = checked_module(payload('classifier/classifier.py'), 'current_classifier')
    metrics = checked_module(payload('source5/plot_metrics.py'), 'frozen_metrics')
    binding = read('pilot/closed-input.json')
    require(sha(payload('qualification/case-checkpoint.json')) == binding['checkpoint']['sha256'], 'Closed owner receipt differs')
    require(sha(payload('qualification/numerical-inputs.json')) == binding['numerical_inputs']['sha256'], 'Closed numerical input differs')
    prepared = read('receipts/prepare-terminal.json'); dry = read('receipts/dryrun-terminal.json'); terminal = read('receipts/run-terminal.json')
    require(sha(payload('receipts/prepare-terminal.json')) == PREP_SHA and sha(payload('receipts/dryrun-terminal.json')) == DRY_SHA
            and sha(payload('receipts/run-terminal.json')) == RUN_SHA, 'Closed receipts differ')
    require(prepared['status'] == dry['status'] == 'passed' and prepared['requests'] == requests()
            and [row['request'] for row in terminal['results']] == requests()
            and [row['request'] for row in dry['results']] == requests(), 'Five-worker schedule differs')
    protocol_sha = sha(payload('pilot/protocol.json'))
    for receipt in (prepared, dry, terminal):
        require(receipt['source_index_sha256'] == CURRENT_SHA and receipt['protocol_sha256'] == protocol_sha, 'Source/protocol join differs')
    require(dry['preparation_sha256'] == terminal['preparation_sha256'] == PREP_SHA, 'Preparation ancestry differs')
    case = prepared['case']; donor = case['donor']; subprocess = donor['subprocess']
    require(subprocess == 'SubProcesses/P0_uux_zgg', 'Unexpected subprocess')
    private = {name: item for name, item in case.items() if name != 'classifier_binding'}
    qualification = classifier.qualify_sources(manifest=manifest_blob, template=payload('qualification/entry-template.inc'),
        generation=payload('qualification/generation.json'), compilation=payload('qualification/compilation.json'),
        matrix=payload('qualification/loop_matrix.f'), worker_template=payload('source5/stability_worker.f'),
        driver=payload('qualification/worker_driver.f'), private_build=canonical(private),
        matrix_source_key=subprocess + '/loop_matrix.f', worker_parameters=donor['compiled']['worker_render_parameters'])
    require(qualification._asdict() == case['classifier_binding'], 'Recomputed source qualification differs')
    source5_blob = payload('source5/pilot.py')
    parser = classifier.parsers(source5_blob, qualification=qualification)['parse_diagnostics']
    legacy_parser = classifier.parsers(source5_blob)['parse_diagnostics']
    metadata = read('points/' + CASE + '.json'); points = payload('points/' + CASE + '.dat')
    require(sha(points) == value['points']['sha256'] == donor['points_sha256'] and
            sha(payload('points/' + CASE + '.json')) == value['points']['metadata_sha256'] == donor['point_metadata_sha256'], 'Saved point binding differs')
    require(sha(payload('cards/param_card.dat')) == donor['physical_card_sha256'] == metadata['inputs']['resolved_physical_card_sha256'], 'Physical card differs')
    validator_blob = payload('point_validator.py')
    require(sha(validator_blob) == source5_index['inputs']['tools/tirust_benchmark_points.py'], 'Point validator differs')
    validator = checked_module(validator_blob, 'point_validator')
    inputs = metadata['inputs']; require(inputs['count'] == len(metadata['points']) == 100 and inputs['energy'] == 1000., 'Wrong point population')
    require(inputs['incoming_masses'] + inputs['masses'] == donor['compiled']['masses'], 'Physical masses differ')
    validator.validate_points(metadata['points'], inputs['masses'], incoming_masses=inputs['incoming_masses'])
    serialized = '100\n' + ''.join(' '.join('%.17e' % x for x in vector) + '\n' for point in metadata['points'] for vector in point['momenta'])
    require(serialized.encode() == points, 'Physical vectors differ')
    functions = pure_functions(source5_blob, ('require', 'point_array', 'cut_report', 'card_settings', 'verify_card_tokens'), {'math': math, 're': re, 'BACKENDS': BACKENDS})
    require(functions['cut_report'](functions['point_array'](points.decode(), 5), 50., .5)['passed'] == 100, 'Point cuts differ')
    observations = read('card-observations.json')
    require(set(observations) == {row['request']['id'] for row in terminal['results'] if 'effective_card' in row}, 'Card observation population differs')
    for record in terminal['results']:
        rid = record['request']['id']; prefix = 'requests/' + rid + '/'
        raw = record.get('raw_diagnostics', {})
        require(set(raw).issubset(RAW_NAMES), 'Unexpected raw filename')
        for name, item in raw.items():
            require(item['path'] == WORK_SPELLING + '/run/' + rid + '/' + subprocess + '/' + name, 'Raw path escaped fixed allowlist')
            blob = payload(prefix + name)
            require({'bytes': len(blob), 'sha256': sha(blob)} == {k: item[k] for k in ('bytes', 'sha256')}, 'Raw receipt differs')
        command = record.get('command')
        if command is not None:
            require(read(prefix + 'command.json') == command, 'Command receipt differs')
            console = payload(prefix + 'stdout.txt'); require(sha(console) == command['stdout_sha256'], 'Complete console differs')
            maps = {}
            for sample in command['runtime_map_samples']:
                require(sample['path'] in MAP_NAMES and sample['path'] not in maps, 'Unexpected/duplicate runtime map')
                blob = payload(prefix + sample['path']); require(sha(blob) == sample['sha256'], 'Runtime map differs')
                maps[sample['path']] = blob.decode()
            runtime_binding(record, maps, case)
        else:
            require(not record['native_started'], 'Started worker lacks command receipt')
            console = payload(prefix + 'stdout.txt') if prefix + 'stdout.txt' in payloads else b''
        texts = {name: payload(prefix + name).decode() if name in raw else '' for name in RAW_NAMES}
        texts['stdout.txt'] = console.decode(errors='replace')
        parsed = replay_diagnostics(record, parser, texts, case['group_layout'])
        if parsed is not None:
            legacy = legacy_parser(texts['stability-dimensions.dat'], texts['stability-values.dat'], texts['stdout.txt'], texts['stability-groups.dat'], case['group_layout'])
            old = {(item['phase'], item['call']): item for item in legacy['loop_group_calls']}
            for call in parsed['loop_group_calls']:
                require(call['legacy_source5_eligible'] == old[(call['phase'], call['call'])]['eligible']
                        and call['legacy_source5_reason'] == old[(call['phase'], call['call'])]['reason']
                        and call['hardware_ieee_state_observed'] is False, 'Legacy annotation differs')
        if 'effective_card' in record:
            observation = observations[rid]; card_name = prefix + 'MadLoopParams.dat'
            guarded = record.get('readonly_guard_passed') is True and record.get('protected_guards_passed') is True
            require(observation['status'] in ('regular', 'missing', 'symlink', 'nonregular'), 'Unknown card observation')
            if observation['status'] == 'regular':
                card = payload(card_name); require({'bytes': len(card), 'sha256': sha(card)} == {k: observation[k] for k in ('bytes', 'sha256')}, 'Postcard differs')
                if guarded:
                    effective = record['effective_card']
                    require(all(checksum == sha(card) for checksum in effective['files'].values()) and
                            functions['verify_card_tokens'](card.decode(), functions['card_settings'](record['request']['backend'])) == effective['settings'] and
                            all(item == effective['settings'] for item in effective['readback'].values()), 'Guarded numerical card differs')
            else:
                require(not guarded and card_name not in payloads, 'Guarded canonical card is absent/nonregular')
    dataset = read('point-metrics.json'); metrics.validate_dataset(dataset)
    expected = metric_replay(payload('pilot/controller.py'), source5_blob, manifest, value, terminal['results'], metrics.relative_spread, protocol_sha)
    expected['provenance'] = {
        'campaign_manifest_sha256': classifier.MANIFEST_SHA,
        'source_revision': manifest['candidate']['exporter_identity']['revision'],
        'generation_receipt_revision': donor['generation_revision'],
        'source_index_sha256': CURRENT_SHA, 'protocol_sha256': protocol_sha,
        'classifier_source_index_sha256': CLASSIFIER_SHA,
        'preparation_sha256': PREP_SHA, 'points_sha256': value['points']['sha256'],
        'point_count_per_case': 100,
        'scope': 'host forced-DP/card-selected backend internal consistency; no host accuracy or true-error qualification',
        'observable': 'finite/Born; common alpha_s/(2*pi) factor cancels in relative spread',
        'backend_observation': 'effective singleton numerical card; unchecked110 does not identify reducer/internal precision',
        'unavailable_reasons': expected['provenance']['unavailable_reasons']}
    require(dataset == expected, 'Current combined-guard point metrics differ')
    provenance = dataset['provenance']
    require(provenance['campaign_manifest_sha256'] == classifier.MANIFEST_SHA and provenance['source_index_sha256'] == CURRENT_SHA
            and provenance['protocol_sha256'] == protocol_sha and provenance['classifier_source_index_sha256'] == CLASSIFIER_SHA
            and provenance['preparation_sha256'] == PREP_SHA and provenance['points_sha256'] == sha(points)
            and provenance['point_count_per_case'] == 100, 'Dataset provenance differs')
    summary = summarize(dataset, terminal['results'], metrics)
    require(read('summary.json') == summary, 'Fixed five-row summary differs')
    return {'verified': True, 'index_sha256': sha(index_blob), 'summary': summary,
            'scope': 'pure local/callback full-journal and qualified/legacy parser replay; no host paths reopened or native work'}


def runtime_binding(record, maps, case):
    binding = record.get('runtime_binding', {})
    if binding.get('verified') is not True:
        return
    expected = dict(case['runtime_closure']['files'])
    executable = record['command']['target_argv'][0]; expected[executable] = case['executable_sha256']
    require(binding.get('backend_observed') is False and binding['samples'], 'Observed reducer claim or missing maps')
    complete = False
    for sample in binding['samples']:
        require(sample['path'] in maps, 'Missing observed map')
        rows = [fields for line in maps[sample['path']].splitlines()
                if len(fields := line.split(maxsplit=5)) == 6 and fields[5].startswith('/')]
        paths = {row[5] for row in rows}
        require(set(sample['elf_files']) | set(sample['read_only_locale']) == paths and
                all(expected.get(path) == checksum for path, checksum in sample['elf_files'].items()), 'Runtime map closure differs')
        require(all(path.startswith('/nix/store/') and path.endswith('/lib/locale/locale-archive') and
                    all(row[1] == 'r--p' for row in rows if row[5] == path) for path in sample['read_only_locale']), 'Unsupported locale mapping')
        complete |= sample['elf_files'] == expected
    require(complete, 'Complete runtime closure not observed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?')
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True, allow_nan=False))
