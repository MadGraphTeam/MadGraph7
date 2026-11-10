"""Pure portable verification of a closed first-failure capture capsule.

Original absolute paths are identifiers only. Every opened payload is indexed
under the supplied capsule, or supplied by its explicit byte-reader callback.
No compiler, loader, bridge, controller stage or external path is invoked.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

SOURCE_INDEX_SHA256 = '2b4bac95022fe550902164a0636699f46798bf4733185610d8f9829f6cd3880d'
CONTROLLER_SHA256 = 'fb121299e65adba6d49ece5577d8586c32f0cd89b5d998b7dae599a391833645'
SOURCE2_INDEX_SHA256 = 'bb990d419677675095d5c230b189a3986d428283e5c3df74edb2907e071a9734'
READINESS2_SHA256 = '18cb676644201fd52e3808ea3fc1d57742d6a5a11c6e0e523aefe97ec98e518d'
ADMISSION2_SHA256 = '57d0d59873391e70cb9ca68d18f083338a1c2447cc3e3a1cf4522a8debc33564'
SPECS = ((1, 'qcd_n3_uux_zgg', 25, (59, 61), 5, 4),
         (2, 'qcd_n4_uux_zzgg', 2, (408, 422), 6, 5))
HEADER = ('case_id', 'phase', 'point', 'method', 'group', 'n', 'rank',
          'plen', 'slen', 'mlen', 'olen', 'poles', 'finite', 'check', 'cache')
KEY = ('case_id', 'phase', 'point', 'method', 'group')
WORD = re.compile(r'[0-9a-f]{16}\Z')


def require(value, message):
    if not value:
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


def expected_slots():
    return [dict(case_id=cid, case=case, phase=1, point=point, method=method,
                 group=group, nprop=n, rank=rank,
                 name='tirust-input-p%03d-m%d-g%03d.dat' % (point, method, group))
            for cid, case, point, groups, n, rank in SPECS
            for method in (1, 2, 3) for group in groups]


def key(row):
    require(all(type(row.get(name)) is int for name in KEY), 'Malformed integer input key')
    return tuple(row[name] for name in KEY)


def parse_input(blob):
    """The strict ASCII MLTCAP_INPUT_V1 grammar accepted by replay.c."""
    require(isinstance(blob, bytes) and blob.endswith(b'\n'), 'Input must have LF-terminated records')
    require(all(byte == 10 or 32 <= byte <= 126 for byte in blob), 'Input is not printable ASCII/LF')
    lines = blob.decode('ascii').split('\n')[:-1]
    require(len(lines) >= 3 and lines[0] == 'MLTCAP_INPUT_V1', 'Wrong input schema')
    require(re.fullmatch(r'[0-9]+(?: [0-9]+){14}', lines[1]), 'Expected exactly fifteen unsigned decimal fields')
    values = list(map(int, lines[1].split(' ')))
    require(all(value <= 2 ** 31 - 1 for value in values), 'Input field exceeds int32')
    h = dict(zip(HEADER, values))
    expected = next((slot for slot in expected_slots() if key(slot) == key(h)), None)
    require(expected is not None, 'Unregistered input key')
    n, rank = expected['nprop'], expected['rank']
    require((h['n'], h['rank'], h['plen'], h['slen'], h['mlen'], h['olen']) ==
            (n, rank, 4 * n, n * n, n, 4 * math.comb(rank + 4, 4)), 'Input tensor shape differs')
    require((h['poles'], h['finite'], h['check'], h['cache']) ==
            (0, 1, int(h['method'] == 2), 0), 'Input evaluation controls differ')
    require(len(lines) == 3 + h['plen'] + h['slen'] + 2 * h['mlen'], 'Missing or trailing input records')
    require(all(re.fullmatch(r'[0-9a-fA-F]{16}', word) for word in lines[2:]), 'Malformed binary64 input word')
    words = [word.lower() for word in lines[2:]]
    p_end, s_end = 1 + h['plen'], 1 + h['plen'] + h['slen']
    return {'header': h, 'input_key': {name: h[name] for name in KEY},
            'mu2_bits': words[0], 'p_bits': words[1:p_end],
            's_bits': words[p_end:s_end], 'mass_bits': words[s_end:]}


def status(row):
    value = row.get('status')
    require(type(value) is int and -(2 ** 31) <= value < 2 ** 31, 'Status is not int32')
    return value


def error_copy(row, *, exact=False):
    required, truncated, text = row.get('error_required'), row.get('error_truncated'), row.get('error')
    require(type(required) is int and required >= 1, 'Missing required C error capacity')
    require(type(truncated) is bool and truncated == (required > 4096), 'C error truncation differs')
    require(row.get('error_encoding') == 'latin1-bytes' and isinstance(text, str)
            and all(ord(c) <= 255 for c in text), 'C error bytes are not losslessly encoded')
    limit = min(required - 1, 4095)
    require(len(text) == limit if exact else len(text) <= limit, 'C error copy length differs')


def output_bits(row, rank, *, boundary=False):
    good = status(row) == 0 and (not boundary or row.get('input_shape_valid') is True)
    if 'output_bits' in row:
        require(good, 'Failed or invalid-shape output was inspected')
    if good:
        bits = row.get('output_bits')
        require(isinstance(bits, list) and len(bits) == 8 * math.comb(rank + 4, 4)
                and all(isinstance(word, str) and WORD.fullmatch(word) for word in bits),
                'Successful output has wrong raw word shape')


def json_lines(blob):
    require(isinstance(blob, bytes), 'JSONL input must be bytes')
    return [json_value(line) for line in blob.splitlines() if line.strip()]


def parse_capture_events(blob, case_id):
    rows = json_lines(blob)
    require(rows and isinstance(rows[0], dict) and rows[0].get('record_type') == 'init' and rows[0].get('case_id') == case_id
            and rows[0].get('schema') == 1, 'Missing capture initialization')
    spec = next(item for item in SPECS if item[0] == case_id)
    primary = {key(slot): {'input_key': {name: slot[name] for name in KEY},
                          'c_boundaries': [], 'fortran_stages': [], 'group_ends': []}
               for slot in expected_slots() if slot['case_id'] == case_id}
    active, ordinal, contexts = None, 0, []
    for row in rows[1:]:
        require(isinstance(row, dict) and row.get('case_id') == case_id, 'Wrong capture event case')
        identity = key(row)
        require(row.get('group') in spec[3] and row.get('n') == spec[4] and row.get('rank') == spec[5],
                'Capture group shape differs')
        is_primary = identity in primary
        require(is_primary or (case_id == 2 and row['phase'] == 0 and row['point'] == 2
                               and row['method'] == 0), 'Unexpected watched capture point')
        kind = row.get('record_type')
        lifecycle = tuple(row.get(name) for name in ('case_id', 'phase', 'call', 'point', 'method', 'group', 'sqso', 'n', 'rank'))
        require(type(row.get('call')) is int and row['call'] > 0 and row.get('sqso') == 1,
                'Capture call/order identity differs')
        require(row.get('mode') == (2 if row['method'] == 2 else 1), 'Capture DP mode differs')
        if kind == 'group_enter':
            require(active is None, 'Overlapping capture group lifecycle')
            active = lifecycle
            require(type(row.get('entry_buffers_valid')) is bool, 'Missing entry-buffer validity')
            if row['entry_buffers_valid']:
                for name, count in (('pl_bits', 4 * spec[4]), ('mass_bits', 2 * spec[4])):
                    require(isinstance(row.get(name), list) and len(row[name]) == count
                            and all(isinstance(word, str) and WORD.fullmatch(word) for word in row[name]),
                            'Malformed group-entry raw words')
        else:
            require(active == lifecycle, 'Capture event lacks its group entry')
            if kind == 'c_boundary':
                status(row); error_copy(row); output_bits(row, spec[5], boundary=True)
                require(type(row.get('input_shape_valid')) is bool and row.get('full_request') is is_primary,
                        'Capture boundary request classification differs')
                require(type(row.get('forwarded_ordinal')) is int and row['forwarded_ordinal'] > ordinal,
                        'C forwarding ordinal is not increasing')
                ordinal = row['forwarded_ordinal']
                (primary[identity]['c_boundaries'] if is_primary else contexts).append(row)
            elif kind == 'fortran_stage':
                require(type(row.get('stage')) is int and type(row.get('detail')) is int,
                        'Malformed Fortran stage')
                if is_primary: primary[identity]['fortran_stages'].append(row)
            elif kind == 'fortran_status':
                status(row)
                require(row.get('error_required') is None and row.get('error_truncation') == 'unobserved'
                        and row.get('error_buffer_bytes') == 512 and isinstance(row.get('error'), str)
                        and len(row['error']) <= 512 and all(ord(c) <= 255 for c in row['error'])
                        and 'output_bits' not in row, 'Malformed limited Fortran error copy')
                if is_primary: primary[identity]['fortran_stages'].append(row)
            elif kind == 'group_end':
                require(type(row.get('stable')) is bool, 'Missing group stability result')
                if is_primary: primary[identity]['group_ends'].append(row)
                active = None
            else:
                raise ValueError('Unknown capture record type')
    return {'primary_slots': list(primary.values()), 'context_c_boundaries': contexts,
            'open_group': list(active) if active else None, 'record_count': len(rows)}


def load_replay_summary(controller_blob):
    """Use exact frozen function bodies without importing controller I/O code."""
    require(sha(controller_blob) == CONTROLLER_SHA256, 'Frozen controller bytes differ')
    tree = ast.parse(controller_blob, filename='retained/source/controller.py')
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('require', 'replay_summary')]
    require([node.name for node in nodes] == ['require', 'replay_summary'], 'Frozen replay functions are absent')
    namespace = {'json': json, 'math': math, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'retained/source/controller.py', 'exec'), namespace)
    return namespace['replay_summary']


def verify_replay(raw, manifest_rows, controller_blob):
    rows = json_lines(raw)
    for row in rows:
        error_copy(row, exact=True)
        require(isinstance(row.get('input_key'), dict), 'Missing replay key')
        spec = next((item for item in SPECS if item[0] == row['input_key'].get('case_id')), None)
        require(spec is not None, 'Unknown replay case')
        if row.get('record_type') == 'evaluate':
            output_bits(row, spec[5])
        else:
            require('output_bits' not in row, 'Non-evaluation output was inspected')
        if row.get('record_type') == 'destroy':
            require(row.get('destroy_succeeded') is (status(row) == 0), 'Destroy success label differs')
            require(row.get('context_present') is (False if status(row) == 0 else None),
                    'Failed destruction does not imply a known context lifetime')
    return load_replay_summary(controller_blob)(raw.decode('utf-8'), manifest_rows)


def logical_path(name):
    require(isinstance(name, str) and name and '\\' not in name and
            not PurePosixPath(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe logical retention path')
    return name


def verify_files(index, reader):
    require(index.get('schema') == 'madloop-tirust-first-failure-capture-closed-v1', 'Wrong closed capsule schema')
    files = index.get('files')
    require(isinstance(files, dict) and files and 'index.json' not in files, 'Missing retained payload inventory')
    if hasattr(reader, 'paths'):
        require(set(reader.paths) == set(files) | {'index.json'}, 'Missing or unindexed retained files')
    blobs = {}
    for name, expected in files.items():
        logical_path(name)
        blob = reader(name)
        require(isinstance(blob, bytes) and {'sha256': sha(blob), 'bytes': len(blob)} == expected,
                'Retained payload differs: ' + name)
        require(not blob.startswith((b'\x7fELF', b'!<arch>\n')), 'Binary payload is outside lean retention scope')
        blobs[name] = blob
    return blobs


def original_payload(reference, originals, blobs):
    """Resolve an original-path identifier exclusively through the registry."""
    original = reference.get('path')
    require(isinstance(original, str) and original in originals, 'Unindexed original-path identifier')
    item = originals[original]
    name = logical_path(item['logical'])
    require(name in blobs and sha(blobs[name]) == reference.get('sha256')
            and len(blobs[name]) == reference.get('size', reference.get('bytes')),
            'Closed original reference differs from retained bytes')
    return blobs[name]


def verify_command(prefix, result, blobs):
    command = result.get('command')
    if command is None:
        return {'executed': False, 'returncode': None, 'error': result.get('error')}
    require(json_value(blobs[prefix + '/command.json']) == command, 'Command receipt differs from terminal')
    require(sha(blobs[prefix + '/stdout.txt']) == command['stdout_sha256'], 'Command stdout differs')
    for sample in command.get('runtime_map_samples', []):
        name = logical_path(sample['path'])
        require(sha(blobs[prefix + '/' + name]) == sample['sha256'], 'Retained runtime map differs')
    require(command.get('cpu') == 12 and command.get('cpu_exclusive') is False
            and command.get('memory_bytes') == 4294967296, 'Native execution policy differs')
    return {'executed': command.get('process_started') is True,
            'returncode': command.get('returncode'), 'error': command.get('error'),
            'runtime_verified_in_receipt': result.get('runtime_binding', {}).get('verified') is True,
            'controller_state': result.get('state'), 'guard_error': result.get('guard_error')}


def terminal_confirmation(value, blob):
    require(isinstance(value, dict) and value.get('observed_terminal') is True
            and type(value.get('session_id')) is int and value['session_id'] > 0
            and type(value.get('exit_code')) is int and value.get('sha256') == sha(blob),
            'Missing or mismatched external terminal confirmation')


def admission_failure(value, blobs):
    require(isinstance(value, dict) and value.get('native_started') is False,
            'Pre-admission failure must explicitly retain zero native work')
    name = logical_path(value.get('logical'))
    require(name in blobs, 'Missing replay admission failure transcript')
    terminal_confirmation(value, blobs[name])
    require(value['exit_code'] != 0, 'Admission failure cannot have a successful terminal')
    return value


def template_changes(admission, source2, policy):
    """Only the two reviewed inventory substitutions are admitted, never read."""
    changes = admission.get('approved_template_changes')
    expected_names = {'madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc',
                      'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc'}
    require(isinstance(changes, dict) and set(changes) == expected_names
            and changes == policy, 'Additive admission changed more than the exact reviewed two hashes')
    for name, change in changes.items():
        require(set(change) == {'old', 'new'} and change['old'] != change['new']
                and all(re.fullmatch('[0-9a-f]{64}', value) for value in change.values())
                and source2['inputs'].get(name) == change['new'], 'Template hash join differs')
    return changes


def verify_source2(blobs, registry, capture_blob, prepare_blob, available):
    """Check additive replay admission as bytes; do not execute its loader."""
    seal_blob = blobs['source2/source-index.json']
    require(sha(seal_blob) == SOURCE2_INDEX_SHA256, 'Unreviewed replay admission source seal')
    source2 = json_value(seal_blob)
    for name, expected in source2['files'].items():
        name = 'source2/' + logical_path(name)
        require(name in blobs and {'sha256': sha(blobs[name]), 'bytes': len(blobs[name])} == expected,
                'Additive replay source payload differs')
    admission_blob = blobs['receipts/additive-admission.json']
    require(sha(admission_blob) == ADMISSION2_SHA256, 'Closed additive admission differs')
    admission = json_value(admission_blob)
    require(admission.get('source1_index_sha256') == SOURCE_INDEX_SHA256 and
            admission.get('source2_index_sha256') == SOURCE2_INDEX_SHA256 and
            admission.get('prepare_terminal_sha256') == sha(prepare_blob) and
            admission.get('failed_capture_terminal_sha256') == sha(capture_blob), 'Additive receipt ancestry differs')
    require(admission.get('inputs') == available and len(available) == 6
            and all(row['case_id'] == 1 for row in available)
            and admission.get('missing_input_count') == 6 and
            admission.get('maximum_tensor_evaluations') == 12, 'Additive replay input scope differs')
    for name in ('compiler_invocations', 'loader_invocations', 'amplitude_evaluations', 'standalone_evaluations'):
        require(admission.get(name) == 0, 'Admission receipt claims native work')
    tree = ast.parse(blobs['source2/replay_admission.py'])
    policies = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'CHANGES' for target in node.targets)]
    require(len(policies) == 1, 'Frozen exact-two-template policy is absent')
    changes = template_changes(admission, source2, policies[0])
    # Original production paths below remain historical identifiers only.
    historical_inputs = source2['inputs']
    require(historical_inputs['.tirust-build/tirust-first-failure-capture-1/evidence/capture-terminal.json'] == sha(capture_blob)
            and historical_inputs['.tirust-build/tirust-first-failure-capture-1/evidence/prepare-terminal.json'] == sha(prepare_blob)
            and historical_inputs['docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/source-index.json'] == SOURCE_INDEX_SHA256,
            'Additive source input identifiers do not join retained receipts')
    guard = admission.get('additive_guard_proof_sha256')
    require(guard == historical_inputs['.tirust-build/tirust-first-failure-closed-validation-1/helfilter-proof.json']
            and any(sha(blob) == guard for blob in blobs.values()), 'Additive guard proof is not retained')
    readiness_blob = blobs['receipts/replay-readiness-2.json']
    require(sha(readiness_blob) == READINESS2_SHA256, 'Additive read-only readiness proof differs')
    readiness = json_value(readiness_blob)
    require(readiness.get('verified') is True and readiness.get('admission') == admission
            and readiness.get('denied_attempts') == [] and readiness.get('work2_absent') is True,
            'Readiness did not admit the exact closed replay inventory')
    for name in ('commands_executed', 'compiler_invocations', 'loader_invocations',
                 'amplitude_evaluations', 'standalone_tensor_evaluations'):
        require(readiness.get(name) == 0, 'Readiness proof claims native work')
    prior = json_value(blobs['source2/prior-attempt.json'])
    require(prior.get('source1_capture_unchanged') == sha(capture_blob) and
            prior.get('replay_started_exists') is False and prior.get('replay_terminal_exists') is False,
            'Predecessor admission failure lineage differs')
    predecessor = admission_failure(registry.get('replay_admission_confirmation'), blobs)
    require(predecessor['logical'] == 'source2/prior-attempt.txt' and
            all(predecessor.get(name) == prior.get(name) for name in
                ('observed_terminal', 'session_id', 'exit_code', 'sha256', 'native_started')),
            'Predecessor failure confirmation differs from the frozen source2 history')
    return {'source2_index_sha256': SOURCE2_INDEX_SHA256,
            'approved_template_changes': changes, 'predecessor_failure': predecessor,
            'readiness_sha256': READINESS2_SHA256, 'admission_sha256': ADMISSION2_SHA256}


def derive_closed(blobs, registry):
    """Derive diagnostics without promoting failures or unavailable target slots."""
    originals = registry['originals']
    prepare_blob = blobs['receipts/prepare-terminal.json']
    prepare = json_value(prepare_blob)
    require(prepare.get('source_index_sha256') == SOURCE_INDEX_SHA256 and
            prepare.get('status') == 'passed' and prepare.get('amplitude_evaluations') == 0,
            'Passed zero-amplitude preparation receipt required')
    capture_blob = blobs['receipts/capture-terminal.json']
    capture = json_value(capture_blob)
    terminal_confirmation(registry.get('capture_terminal_confirmation'), capture_blob)
    require(capture.get('source_index_sha256') == SOURCE_INDEX_SHA256 and
            capture.get('preparation_sha256') == sha(prepare_blob), 'Capture ancestry differs')
    require(capture.get('status') in ('passed', 'failed') and
            capture.get('maximum_unique_inputs') == 12 and
            capture.get('maximum_standalone_evaluations') == 24 and
            capture.get('missing_inputs_are_not_replaced') is True and
            capture.get('automatic_retry') is False, 'Capture closure policy differs')
    require([row.get('case') for row in capture['results']] == [s[1] for s in SPECS],
            'Capture omitted a registered case')
    manifest_rows, slots, cases = [], [], []
    for spec, result in zip(SPECS, capture['results']):
        cid, case = spec[:2]
        expected = [slot for slot in expected_slots() if slot['case_id'] == cid]
        require(len(result.get('inputs', [])) == 6, 'Capture omitted an expected target slot')
        ref = result.get('tirust-capture-events.jsonl')
        if ref is None:
            events = {'primary_slots': [{'input_key': {k: slot[k] for k in KEY},
                                         'c_boundaries': [], 'fortran_stages': [], 'group_ends': []}
                                        for slot in expected],
                      'context_c_boundaries': [], 'record_count': 0, 'open_group': None}
        else:
            events = parse_capture_events(original_payload(ref, originals, blobs), cid)
        by_key = {key(row['input_key']): row for row in events['primary_slots']}
        for fixed, row in zip(expected, result['inputs']):
            require(all(row.get(name) == fixed[name] for name in
                        ('case', 'case_id', 'point', 'method', 'group', 'nprop', 'rank', 'name')),
                    'Capture terminal target key differs')
            require(row.get('state') in ('captured', 'missing', 'unavailable-collection-failure'),
                    'Unknown input capture state')
            manifest_rows.append(row)
            observed = by_key[key(fixed)]
            require(len(observed['c_boundaries']) <= 1, 'Repeated full C request for one target key')
            if row['state'] == 'captured':
                parsed = parse_input(original_payload(row, originals, blobs))
                require(key(parsed['input_key']) == key(fixed), 'Captured input header does not match its slot')
                require(len(observed['c_boundaries']) == 1 and
                        observed['c_boundaries'][0]['input_shape_valid'] is True,
                        'Captured input has no corresponding complete C-boundary request')
            slots.append({'input_key': {name: fixed[name] for name in KEY},
                          'input_state': row['state'], 'input_sha256': row.get('sha256'),
                          'c_boundary_count': len(observed['c_boundaries']),
                          'c_statuses': [r['status'] for r in observed['c_boundaries']],
                          'c_errors': [{name: r[name] for name in
                                        ('error_required', 'error_truncated', 'error_encoding', 'error')}
                                       for r in observed['c_boundaries']],
                          'fortran_stage_only': bool(observed['fortran_stages']) and not observed['c_boundaries'],
                          'group_end_stable': [r['stable'] for r in observed['group_ends']]})
        cases.append({'case': case, 'capture': verify_command('capture/' + case, result, blobs),
                      'event_records': events['record_count'],
                      'primary_c_boundaries': sum(len(r['c_boundaries']) for r in events['primary_slots']),
                      'bootstrap_context_c_boundaries': len(events['context_c_boundaries']),
                      'bootstrap_context_statuses': {str(s): sum(r['status'] == s for r in events['context_c_boundaries'])
                                                     for s in sorted({r['status'] for r in events['context_c_boundaries']})},
                      'open_group': events['open_group']})
    replay_name = 'receipts/replay-terminal.json'
    if replay_name not in blobs:
        replay_status = registry.get('replay_status')
        require(replay_status in ('not-executed', 'pre-admission-failed'),
                'Absent replay needs an explicit unexecuted state')
        require('receipts/replay-started.json' not in blobs, 'Replay started but no terminal was retained')
        require(not any(name.startswith('replay/') for name in blobs), 'Unexecuted replay has unexpected native artifacts')
        replay = {'status': replay_status, 'evaluate_count': 0,
                  'captured_inputs_not_replayed': sum(row['state'] == 'captured' for row in manifest_rows)}
        if replay_status == 'pre-admission-failed':
            replay['admission_failure'] = admission_failure(registry.get('replay_admission_confirmation'), blobs)
    else:
        require(registry.get('replay_status') == 'closed', 'Replay terminal requires an explicit closed state')
        terminal = json_value(blobs[replay_name])
        terminal_confirmation(registry.get('replay_terminal_confirmation'), blobs[replay_name])
        require(terminal.get('source_index_sha256') == SOURCE_INDEX_SHA256 and
                terminal.get('preparation_sha256') == sha(prepare_blob) and
                len(terminal.get('results', [])) == 1, 'Replay ancestry or schedule differs')
        result = terminal['results'][0]
        require(result.get('case') == SPECS[0][1], 'Replay schedule differs')
        available = [row for row in manifest_rows if row['state'] == 'captured']
        additive = None
        if 'source2/source-index.json' in blobs or 'receipts/additive-admission.json' in blobs:
            additive = verify_source2(blobs, registry, capture_blob, prepare_blob, available)
        elif 'replay_admission_confirmation' in registry:
            admission_failure(registry['replay_admission_confirmation'], blobs)
        prefix = 'replay/' + SPECS[0][1]
        manifest_name = prefix + '/outputs/request-manifest.txt'
        if manifest_name in blobs:
            expected_manifest = ''.join(row['path'] + '\n' for row in available).encode('ascii')
            require(blobs[manifest_name] == expected_manifest and
                    sha(expected_manifest) == result.get('request_manifest_sha256'), 'Replay manifest differs')
        else:
            require(result.get('native_started') is False, 'Started replay omitted its input manifest')
        raw_name = prefix + '/outputs/tirust-replay.jsonl'
        raw = blobs.get(raw_name, b'')
        if 'tirust-replay.jsonl' in result:
            require(original_payload(result['tirust-replay.jsonl'], originals, blobs) == raw,
                    'Replay terminal journal reference differs')
        summary = verify_replay(raw, manifest_rows, blobs['source/controller.py'])
        if 'replay_summary' in result:
            require(result['replay_summary'] == summary, 'Exact frozen controller replay summary differs')
        require(result.get('closed_capture_terminal_sha256') in (None, sha(capture_blob)), 'Replay capture binding differs')
        replay = {'status': terminal.get('status'), 'protocol': summary,
                  'execution': verify_command(prefix, result, blobs),
                  'evaluate_count': summary['evaluate_count'], 'additive_admission': additive}
        evaluated = [row for row in json_lines(raw) if row.get('record_type') == 'evaluate']
        replay['status_counts'] = {str(s): sum(row['status'] == s for row in evaluated)
                                   for s in sorted({row['status'] for row in evaluated})}
        replay['fresh_status_counts'] = {str(s): sum(row['status'] == s and row['repetition'] == 1 for row in evaluated)
                                         for s in sorted({row['status'] for row in evaluated})}
        replay['repeat_status_counts'] = {str(s): sum(row['status'] == s and row['repetition'] == 2 for row in evaluated)
                                          for s in sorted({row['status'] for row in evaluated})}
        replay['successful_tensor_output_count'] = sum(row['status'] == 0 for row in evaluated)
    return {'schema': 'madloop-tirust-first-failure-capture-summary-v1',
            'capture_terminal_status': capture['status'], 'capture_terminal_sha256': sha(capture_blob),
            'expected_input_slots': 12, 'captured_input_slots': sum(r['input_state'] == 'captured' for r in slots),
            'missing_or_unavailable_input_slots': sum(r['input_state'] != 'captured' for r in slots),
            'cases': cases, 'slots': slots, 'replay': replay,
            'interpretation': 'Captured tensor requests and explicit missing slots; no failed tensor output inspected.',
            'numerical_acceptance_claimed': False, 'performance_acceptance_claimed': False}


def verify(root=None, *, read_bytes=None):
    """Verify a closed capsule without following any original-path identifier."""
    if read_bytes is None:
        base = Path(root or Path(__file__).resolve().parent)
        require(base.is_dir() and not base.is_symlink(), 'Expected a regular retained directory')
        def read_bytes(name):
            logical_path(name)
            path = base
            for part in name.split('/'):
                path = path / part
                require(not path.is_symlink(), 'Retention paths cannot follow symlinks')
            require(path.is_file(), 'Missing retained file: ' + name)
            return path.read_bytes()
        read_bytes.paths = {str(path.relative_to(base)) for path in base.rglob('*') if path.is_file() or path.is_symlink()}
    index_blob = read_bytes('index.json')
    index = json_value(index_blob)
    blobs = verify_files(index, read_bytes)
    registry = json_value(blobs['retention.json'])
    originals = registry.get('originals')
    require(isinstance(originals, dict), 'Missing original-to-logical retention registry')
    for original, item in originals.items():
        require(isinstance(original, str) and original.startswith('/'), 'Original path is not an absolute identifier')
        name = logical_path(item['logical'])
        require(name in blobs and item == {'logical': name, 'sha256': sha(blobs[name]), 'bytes': len(blobs[name])},
                'Original path registry differs from indexed payload')
    require(sha(blobs['source/source-index.json']) == SOURCE_INDEX_SHA256, 'Source seal differs')
    source = json_value(blobs['source/source-index.json'])
    for name, expected in source['files'].items():
        require('source/' + name in blobs and
                {'sha256': sha(blobs['source/' + name]), 'bytes': len(blobs['source/' + name])} == expected,
                'Retained frozen source differs: ' + name)
    require(sha(blobs['source/controller.py']) == CONTROLLER_SHA256, 'Controller source differs')
    summary = derive_closed(blobs, registry)
    if 'summary.json' in blobs:
        require(json_value(blobs['summary.json']) == summary, 'Stored derived summary differs')
    return {'verified': True, 'index_sha256': sha(index_blob), 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))
