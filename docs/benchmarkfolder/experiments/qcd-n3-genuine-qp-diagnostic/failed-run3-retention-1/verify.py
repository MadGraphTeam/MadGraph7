#!/usr/bin/env python3
"""Replay the frozen parser's failure from retained bytes, never original paths."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import re
import struct
import types


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replay(payload, inventory):
    j = lambda name: json.loads(payload[name])
    source_index = j('controller/source-index.json')
    for name, item in source_index['files'].items():
        raw = payload['controller/' + name]
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Controller copy changed: ' + name)
    result = j('actual-run3/result.json')
    require(result['source_index_sha256'] == digest(payload['controller/source-index.json']), 'Controller binding differs')
    require(result['passed'] is False and result['workers_completed'] == 1 and result['failure'] == {
            'type': 'ValueError', 'message': 'Born is not using observed repaired DP inputs'}, 'Unexpected parser failure')
    require(result['recompiled'] is False and result['original_donor_runtime_preserved'] is True,
            'Unexpected rebuild/runtime policy')
    binding = j('controller/bindings.json')
    manifest = j('controller/donor-closure.json')
    expected = {x['resolved_path']: x['sha256'] for x in manifest['entries']}
    require(len(expected) == 15 and {x['source_path']: x['sha256'] for x in manifest['entries']} == binding['runtime'],
            'Original fifteen-library identities differ')
    build = j('controller/inherited-build2.json')
    require(result['build_sha256'] == digest(payload['controller/inherited-build2.json']), 'Inherited build hash differs')
    require(build['runtime'] == binding['runtime'] and build['target_interpreter'] == binding['target_interpreter'],
            'Inherited runtime/interpreter differs')
    links = inventory['actual_attempt_symlinks']
    aliases = {'closed-runtime/' + x['soname']: x['resolved_path'] for x in manifest['entries']}
    require(links == aliases and len(links) == 15, 'Actual closed symlink inventory differs')
    alias_dir = binding['native_library_path']
    absolute_aliases = {alias_dir + '/' + x['soname']: x['resolved_path'] for x in manifest['entries']}
    proof_names = ['actual-run3/closed-runtime-proof.json']
    for worker in ('b1-m4-o0', 'b1-m4-o1'):
        proof_names += ['actual-run3/' + worker + '/' + name for name in
                        ('closed-runtime-pretrace.json', 'closed-runtime-preworker.json')]
    for name in proof_names:
        proof = j(name)
        require(proof['passed'] is True and proof['library_count'] == 15 and proof['alias_directory'] == alias_dir,
                'Alias guard failed')
        require({x['soname']: (x['alias_target'], x['resolved_path'], x['sha256']) for x in proof['libraries']} ==
                {x['soname']: (x['resolved_path'], x['resolved_path'], x['sha256']) for x in manifest['entries']},
                'Alias guard identity mismatch')
    trace_prefixes = ['actual-run3', 'actual-run3/b1-m4-o0', 'actual-run3/b1-m4-o1']
    for prefix in trace_prefixes:
        r = j(prefix + '/loader.json')
        require(r['native'] is False and r['returncode'] == 0 and r['process_reaped'] is True and
                r.get('failure') is None and not r.get('timed_out') and r['seconds_cap'] <= 30,
                'Loader trace failed')
        require(r['target_loader_environment_at_execve'] ==
                {'LD_LIBRARY_PATH': alias_dir, 'LD_TRACE_LOADED_OBJECTS': '1'}, 'Trace environment mismatch')
        text = payload[prefix + '/loader.log'].decode()
        names = set(re.findall(r'/[^\s()]+', text))
        resolved = {absolute_aliases.get(name, name) for name in names}
        require('not found' not in text and resolved == set(expected), 'Strict pre-entry trace differs')
    parser = types.ModuleType('retained_qcd_n3_frozen_analyzer')
    exec(compile(payload['controller/analyze.py'], 'retained/controller/analyze.py', 'exec'), parser.__dict__)
    selected = j('controller/fixtures/selected-inputs.json')
    point_bytes = {p: b''.join(struct.pack('<d', float.fromhex(x)) for row in
                               selected['points'][str(p)]['binary64_hex'] for x in row) for p in (1, 93, 39)}
    workers, total_counts, public_rows = [], Counter(), []
    for observed in (0, 1):
        worker = 'b1-m4-o' + str(observed)
        prefix = 'actual-run3/' + worker
        r, mapping = j(prefix + '/worker.json'), j(prefix + '/mapped-runtime.json')
        require(r['native'] is True and r['returncode'] == 0 and r['process_spawned'] is True and
                r['process_reaped'] is True and r.get('failure') is None and not r.get('timed_out') and
                r['seconds_cap'] <= 180, 'Native worker did not close cleanly')
        require(r['target_loader_environment_at_execve'] == {'LD_LIBRARY_PATH': alias_dir}, 'Worker environment differs')
        require(mapping['expected_runtime_sha256'] == mapping['mapped_files_sha256_after_exit'] == expected and
                set(mapping['mapped_elf_paths']) == set(expected) | {build['executable']} and
                mapping['executable'] == build['executable'] and mapping['cpu_allowed_list'] == '12',
                'Actual mapped runtime closure differs')
        require(mapping['maps_text'].encode() == payload[prefix + '/maps.txt'] and
                mapping['maps_sha256'] == digest(payload[prefix + '/maps.txt']) ==
                r['mapping_capture']['complete_snapshot_sha256'], 'Mapping snapshot hash mismatch')
        for name, sha in j(prefix + '/outputs.json').items():
            require(digest(payload[prefix + '/' + name]) == sha, 'Native output inventory mismatch: ' + name)
        raw_prefix = r['cwd'].rstrip('/') + '/'
        for name, sha in j(prefix + '/inputs.json').items():
            require(name.startswith(raw_prefix) and digest(payload[prefix + '/' + name[len(raw_prefix):]]) == sha,
                    'Worker fixture inventory mismatch')
        text, stdout = payload[prefix + '/trace.txt'].decode(), payload[prefix + '/worker.log'].decode()
        actual, rows = parser.parse_trace(text)
        require(actual == observed, 'Observation header differs')
        entry = [row for row in rows if row['kind'] == 1 and row['part'] == 0]
        returned = [row for row in rows if row['kind'] == 1 and row['part'] == 1]
        calls = [row['call'] for row in entry]
        require(calls == [row['call'] for row in returned] == [0, 1, 101, 102, 103], 'Actual call/return sequence differs')
        require(parser.ints(parser.payload(parser.one(rows, 9, 0))) == (1, 4, observed, 1, 113, 128),
                'Completion footer differs')
        require(len(re.findall(r'\bQG_DONE\s+1\s+4\s+' + str(observed) + r'\s+1\b', stdout)) == 1 and
                not re.search(r'##W03\b|QG_FAILED|(?:^|\n)\s*(?:ERROR )?STOP\b', stdout, re.I) and
                j(prefix + '/warnings.json') == [], 'Unexpected warning or STOP')
        public_rows.append([row['data'].hex() for row in rows if row['kind'] == 1])
        returned_diagnostics = []
        for row in returned:
            body = parser.payload(row)
            status = parser.ints(body[328:336])[0]
            if row['call'] >= 101:
                require(status == 120 and parser.floats(body[312:328]) == (-1., -1.), 'Unchecked status changed')
                returned_diagnostics.append({'call': row['call'], 'status': status, 'host_checked': False})
        counts = {'bootstrap_attempted': 1, 'bootstrap_returned': 1, 'initialization_attempted': 1,
                  'initialization_returned': 1, 'diagnostic_attempted': 3, 'diagnostic_returned': 3,
                  'public_attempted': 5, 'public_returned': 5}
        total_counts.update(counts)
        failure = None
        try:
            parsed = parser.parse_worker(text, stdout, 1, 4, observed, point_bytes)
        except ValueError as error:
            failure = {'type': 'ValueError', 'message': str(error)}
        if observed == 0:
            require(failure is None and json.loads(json.dumps(parsed,allow_nan=False)) == j(prefix + '/parsed.json'), 'Plain parser replay differs')
        else:
            require(failure == result['failure'] and prefix + '/parsed.json' not in payload,
                    'Frozen observed parser failure not reproduced')
        workers.append({'worker': worker, 'native_returncode': 0, 'complete_trace': True,
                        'counts': counts, 'returned_diagnostics': returned_diagnostics,
                        'record_count': len(rows), 'payload_bytes': sum(len(row['data']) for row in rows),
                        'kind_counts': {str(k): v for k, v in sorted(Counter(row['kind'] for row in rows).items())},
                        'frozen_parser_failure': failure, 'frozen_parser_passed': failure is None})
    return {'passed': True, 'scope': 'failure retention replay, not scientific adjudication acceptance',
            'overall_native_attempt_passed': False, 'overall_failure': result['failure'],
            'loader_trace_count': 3, 'loader_traces_passed': True,
            'strict_actual_runtime_closure_passed_for_attempted_workers': True,
            'required_original_runtime_libraries': 15, 'workers_planned': 16, 'workers_native_attempted': 2,
            'workers_native_exited_zero': 2, 'workers_frozen_parser_passed': 1, 'workers_unexecuted': 14,
            'counts': dict(total_counts), 'public_plain_observed_bytes_equal': public_rows[0] == public_rows[1],
            'full_plain_observed_protocol_parity_qualified': False,
            'adjudication_executed': False, 'coefficient_comparison_qualified': False,
            'unchecked_diagnostic_order_columns_returned': 12,
            'internal_native_calls_during_bootstrap_or_plain_arm': 'unobserved', 'workers': workers}


def verify(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    index = json.loads((root / 'evidence-index.json').read_bytes())
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Retained package contains symlinks')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(index['files']) | {'evidence-index.json'}, 'Package membership changed')
    stored = {}
    for name, item in index['files'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name, 'Unsafe retained member')
        raw = (root / p).read_bytes()
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Changed retained member: ' + name)
        stored[name] = raw
    inventory = json.loads(stored['retention-inventory.json'])
    payload = {}
    for logical, item in inventory['original_files'].items():
        raw = stored[item['member']]
        require(len(raw) == item['stored_bytes'] and digest(raw) == item['stored_sha256'], 'Stored original differs')
        require(item['codec'] in ('gzip', 'identity'), 'Unsupported retention codec')
        raw = gzip.decompress(raw) if item['codec'] == 'gzip' else raw
        require(len(raw) == item['raw_bytes'] and digest(raw) == item['raw_sha256'], 'Raw original differs')
        payload[logical] = raw
    result = replay(payload, inventory)
    require(result == json.loads(stored['summary.json']), 'Retained summary replay differs')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True, indent=2))
