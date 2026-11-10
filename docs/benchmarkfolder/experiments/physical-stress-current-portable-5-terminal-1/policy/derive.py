"""Pure policy for the exact20-request continuation and24-row aggregate.

Binders authenticate closed proofs, full raw parsing, runtime maps and owned
cleanup before invoking these functions. No filesystem or execution API.
"""
import ast
import hashlib
import math

WORKER_SHA = '71a460579b2cabe8a1485fa0eed406e98988ec019b6c195bd78af55eedf99d57'
COMPARATOR_SHA = 'bc405b44d8d794cc4396ef3da3d48af238ac4dfb619c58901d7bab910a7e1370'


def require(value, message):
    if not value:
        raise ValueError(message)


def key(request):
    return request['case'] + '/' + request['label']


def analyzer(worker, comparator):
    """Load only frozen pure parser/status definitions and the comparator."""
    require(hashlib.sha256(worker).hexdigest() == WORKER_SHA, 'Worker source differs')
    require(hashlib.sha256(comparator).hexdigest() == COMPARATOR_SHA, 'Comparator source differs')
    names = {'MeasurementError', 'validate_status', 'parse_outputs'}
    selected = [node for node in ast.parse(worker).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BACKEND_CODES' for t in node.targets)]
    ns = {'math': math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<frozen pure worker parser>', 'exec'), ns)
    compare = [node for node in ast.parse(comparator).body if isinstance(node, ast.FunctionDef) and node.name == 'compare_rows']
    exec(compile(ast.Module(body=compare, type_ignores=[]), '<frozen comparator>', 'exec'), ns)

    def parse(raw, backend):
        ns['raw_outputs'] = lambda unused: raw
        return ns['parse_outputs'](None, [backend], 2, 1)
    return parse, ns['validate_status'], ns['compare_rows'], ns['BACKEND_CODES']


def trace(raw, phase, aliases):
    """Keep tokens and partial rows; nonfinite diagnostics are not JSON numbers."""
    rows = []; malformed = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split(); width = 9 if phase == 'warmup' else 8
        try:
            require(len(fields) == width, 'row width')
            offset = 1 if phase == 'warmup' else 0
            point, order, status = map(int, fields[offset:offset + 3])
            actual = aliases.get((status // 100, status % 10))
            numeric = fields[offset + 3:]
            nonfinite = [token for token in numeric if not math.isfinite(float(token))]
            rows.append({'line': number, 'call_index': int(fields[0]) if offset else None,
                         'point': point, 'order': order, 'status': status,
                         'alias_backend': None if actual is None else actual[0],
                         'alias_precision': None if actual is None else actual[1],
                         'raw_coefficients': numeric[:4], 'raw_accuracy': numeric[4],
                         'nonfinite_tokens': nonfinite})
        except (ValueError, TypeError):
            malformed.append({'line': number, 'raw': line})
    return {'rows': rows, 'malformed': malformed,
            'observed_call_keys': [list(v) for v in sorted({(r['call_index'], r['point']) for r in rows})] if phase == 'warmup' else [],
            'observed_points': sorted({r['point'] for r in rows}),
            'nonfinite_row_count': sum(bool(r['nonfinite_tokens']) for r in rows)}


def derive(plan, records, stage, accounting, aborted, protocol):
    """Derive exactly the original20 suffix states, never the inherited four.

    Records already have authenticated artifact bindings.

    runtime_verified is the future binder's computed verdict, never a saved
    success field trusted from execution. Failures remain valid retained data.
    """
    parse, validate_status, compare, aliases = protocol
    original, requests, case_ids = selection(plan)
    keys = [key(r) for r in requests]
    require(set(records) <= set(keys), 'Unknown request evidence')
    require(stage['status'] in ('passed', 'failed'), 'Stage is not terminal')
    require(accounting['requested'] == 20, 'Accounting request count differs')
    aborted_key = None if aborted is None else key(aborted['request'])
    require(aborted_key is None or aborted_key in keys, 'Unknown aborted request')
    rows = []; accepted = {}; completed = []
    for request in requests:
        rid = key(request); record = records.get(rid, {})
        rc = record.get('request_completion'); completion = record.get('completion')
        result = record.get('result'); raw = record.get('raw', {}); stdout = record.get('stdout')
        if rc is not None:
            require(rc['request'] == request, 'Request completion identity differs')
            completed.append(rid)
        if result is not None:
            require(result['request'] == request, 'Result identity differs')
        warm = trace(raw.get('warmup.dat', ''), 'warmup', aliases)
        final = trace(raw.get('values.dat', ''), 'final', aliases)
        warnings = None if stdout is None else [line for line in stdout.splitlines() if '##W03' in line]
        parsed = None; parse_error = None
        if raw:
            try:
                parsed = parse(raw, request['backend'])
            except Exception as error:
                parse_error = {'class': type(error).__name__, 'message': str(error)}
        if result is not None:
            require(result['incomplete_warnings'] == (warnings or []), 'W03 scan differs')
            if result['observation'] is not None:
                require(parsed == result['observation'], 'Saved numerical observation differs')
        checked = bool(parsed is not None and warnings == [] and result is not None
                       and result['failure'] is None and not result['mapping_errors']
                       and result['complete_runtime_map'] and record.get('runtime_verified') is True
                       and not record.get('binding_errors')
                       and all(0 <= row['accuracy'] <= 1e-6 for row in parsed['rows']))
        if result is not None and record.get('runtime_verified') is True and not record.get('binding_errors'):
            require(result['numerically_qualified'] == checked, 'Saved numerical qualification differs')
        fatal = (completion or {}).get('fatal_infrastructure_failure')
        aborted_here = aborted_key == rid
        if aborted_here and aborted['state'] == 'deferred-before-launch':
            require(not record, 'Deferred request unexpectedly has launched artifacts')
            state = 'deferred_before_launch'; checked = False
        elif fatal or record.get('binding_errors') or record.get('runtime_verified') is False or aborted_here:
            state = 'controller_failed'; checked = False
        elif rc is not None:
            require(completion is not None, 'Completed request lacks completion receipt')
            require(rc['result'] == completion['result'], 'Outer completion differs')
            if completion.get('outer_failure') is not None:
                checked = False
            require(completion['result']['numerically_qualified'] == checked, 'Outer qualification differs')
            state = 'host_checked_complete' if checked else 'physical_rejected'
        else:
            require(not record, 'Unaccounted request artifacts')
            state = 'not_started'; checked = False
        if checked:
            accepted[rid] = parsed['rows']
        rows.append({'request_id': rid, 'case': request['case'], 'label': request['label'],
                     'source': request['source'], 'backend': request['backend'], 'state': state,
                     'host_checked_complete_stream': checked, 'expected_final_points': 2,
                     'qualified_final_points': [1, 2] if checked else [],
                     'raw_final_points_observed': final['observed_points'],
                     'warmup_calls_observed': len(warm['observed_call_keys']),
                     'warmup_trace': warm, 'final_trace': final,
                     'accepted_precision_counts': parsed['backend_precision_counts'] if checked else {},
                     'warnings': warnings, 'nonfinite_rows': warm['nonfinite_row_count'] + final['nonfinite_row_count'],
                     'parser_failure': parse_error, 'native_failure': None if result is None else result['failure'],
                     'controller_failure': aborted['exception'] if aborted_here else (completion or {}).get('outer_failure'),
                     'fatal_infrastructure_failure': fatal, 'binding_errors': record.get('binding_errors', []),
                     'coefficient_comparison_eligible': checked,
                     'request_completion_recorded': rc is not None,
                     'runtime_verified': record.get('runtime_verified'),
                     'mapping_errors': [] if result is None else result['mapping_errors'],
                     'complete_runtime_map': None if result is None else result['complete_runtime_map'],
                     'failure_flags': {
                         'status120_observed': any(r['status'] == 120 for r in warm['rows'] + final['rows']),
                         'W03_observed': bool(warnings),
                         'nonfinite_observed': bool(warm['nonfinite_row_count'] + final['nonfinite_row_count']),
                         'runtime_identity_failure': bool(record and (record.get('runtime_verified') is False
                             or record.get('binding_errors') or result is not None
                             and (result['mapping_errors'] or not result['complete_runtime_map']))),
                         'native_failure': result is not None and result['failure'] is not None,
                         'timeout_observed': any('timeout' in str(value).lower() for value in
                             ((None if result is None else result['failure']),
                              (completion or {}).get('outer_failure'), fatal,
                              (aborted['exception'] if aborted_here else None))),
                         'controller_failure': aborted_here or (completion or {}).get('outer_failure') is not None,
                         'fatal_infrastructure_failure': bool(fatal)}})
    require(accounting['completed'] == len(completed), 'Completed request accounting differs')
    require([key(r) for r in accounting['not_completed']] == [rid for rid in keys if rid not in completed],
            'Unexecuted request accounting differs')
    if stage['status'] == 'passed':
        require(len(completed) == 20 and aborted is None, 'Passed stage lacks complete orchestration')
    elif aborted is None:
        require(len(completed) == 20, 'Failed partial stage lacks abort record')
    comparisons, coverage = compare_coverage(rows, protocol)
    return {'scope': 'twenty neverlaunched suffix requests under original strict automatic policy',
            'stage_status': stage['status'], 'request_rows': rows, 'case_coverage': coverage,
            'comparisons': comparisons, 'requested': 20, 'completed_orchestration': len(completed),
            'host_checked_complete_streams': len(accepted),
            'original_request_order': [key(r) for r in original],
            'inherited_request_ids': [key(r) for r in original[:4]],
            'continuation_request_ids': keys,
            'missing_or_unqualified_request_ids': [r['request_id'] for r in rows
                if not r['host_checked_complete_stream']],
            'not_completed_orchestration_request_ids': [r['request_id'] for r in rows
                if not r['request_completion_recorded']],
            'unexecuted_request_ids': [r['request_id'] for r in rows
                if r['state'] in ('not_started', 'deferred_before_launch')],
            'requested_subset_all_cases_agree': all(v['requested_subset_agreement'] for v in coverage),
            'six_lane_all_cases_agree': False,
            'full24_coverage_requires_two_independently_verified_closed_proofs': True,
            'independent_physical_accuracy_qualified': False, 'EW_analytic_oracle_qualified': False,
            'performance_eligible': False, 'warmup_counts_are_observed_only': True}


def selection(plan):
    """Check original24 identities and return exactly its neverlaunched suffix."""
    original = plan['requests']; ids = [key(r) for r in original]
    require(len(ids) == len(set(ids)) == 24, 'Expected24 unique original requests')
    cases = [entry['case']['id'] for entry in plan['cases']]
    require(len(cases) == len(set(cases)) == 4, 'Expected four original cases')
    require([r['case'] for r in original] == [cid for cid in cases for unused in range(6)],
            'Original six-lane case order differs')
    expected = [('baseline', 1), ('candidate', 1), ('candidate', 6),
                ('candidate', 7), ('candidate', 8), ('candidate', 9)]
    for start in range(0, 24, 6):
        require([(r['source'], r['backend']) for r in original[start:start + 6]] == expected,
                'Original source/backend lane order differs')
    requests = original[4:]
    require([sum(r['case'] == cid for r in requests) for cid in cases] == [2, 6, 6, 6],
            'Continuation is not the exact2+6+6+6 suffix')
    return original, requests, cases


def accepted_rows(row, protocol):
    """Reconstruct comparator rows from authenticated accepted final-trace tokens.

    The owning binder has already run the SHA-pinned full raw parser, including
    dimensions and histograms. This second pass does not invent those inputs or
    timing values. It uses that parser's frozen status validator and verifies the
    retained coefficient/accuracy tokens and ordered point/order sequence again.
    """
    checked = row['host_checked_complete_stream']
    require(isinstance(checked, bool), 'Nonboolean checked-stream state')
    require(row['coefficient_comparison_eligible'] is checked, 'Eligibility differs from checked-stream state')
    require((row['state'] == 'host_checked_complete') is checked, 'Checked-stream state differs')
    require(row['qualified_final_points'] == ([1, 2] if checked else []), 'Qualified points differ')
    if not checked:
        return None
    require(row['warnings'] == [] and row['native_failure'] is None
            and row['controller_failure'] is None and not row['fatal_infrastructure_failure']
            and not row['binding_errors'], 'Accepted stream has retained failures')
    require(row['raw_final_points_observed'] == [1, 2], 'Accepted final points differ')
    trace_data = row['final_trace']
    require(not trace_data['malformed'] and trace_data['nonfinite_row_count'] == 0,
            'Accepted final trace is malformed or nonfinite')
    require(trace_data['observed_points'] == [1, 2], 'Accepted trace point index differs')
    parsed = []; usage = {}; validate_status = protocol[1]
    for saved in trace_data['rows']:
        require(saved['call_index'] is None and not saved['nonfinite_tokens'], 'Accepted final trace row differs')
        status = saved['status']; actual = validate_status(status, row['backend'])
        require(saved['alias_backend'] == actual['backend']
                and saved['alias_precision'] == actual['precision'], 'Accepted status alias differs')
        tokens = saved['raw_coefficients']
        require(isinstance(tokens, list) and len(tokens) == 4 and all(isinstance(t, str) for t in tokens)
                and isinstance(saved['raw_accuracy'], str), 'Accepted numerical tokens differ')
        coefficients = list(map(float, tokens)); accuracy = float(saved['raw_accuracy'])
        require(all(math.isfinite(x) for x in coefficients + [accuracy])
                and 0 <= accuracy <= 1e-6, 'Accepted numerical tokens failed finite/accuracy check')
        point, order = saved['point'], saved['order']
        require(type(point) is int and type(order) is int and order >= 0, 'Accepted point/order type differs')
        parsed.append({'point': point, 'order': order, 'status': status,
                       'coefficients': coefficients, 'accuracy': accuracy,
                       'actual_backend': actual['backend'], 'precision': actual['precision']})
        if order == 0:
            label = str(actual['backend']) + ':' + actual['precision']
            usage[label] = usage.get(label, 0) + 1
    orders = [r['order'] for r in parsed if r['point'] == 1]
    require(orders and orders == list(range(len(orders)))
            and [(r['point'], r['order']) for r in parsed] == [(p, o) for p in (1, 2) for o in orders],
            'Accepted point/order sequence differs')
    for point in (1, 2):
        require(len({r['status'] for r in parsed if r['point'] == point}) == 1,
                'Accepted split-order status differs')
    require(usage == row['accepted_precision_counts'], 'Accepted precision count differs')
    return parsed


def compare_coverage(rows, protocol):
    """Recompute every eligible pair; missing lanes never qualify six-lane coverage."""
    ids = [row['request_id'] for row in rows]
    require(len(ids) == len(set(ids)), 'Repeated request cannot enter comparisons')
    cases = list(dict.fromkeys(row['case'] for row in rows))
    accepted = {row['request_id']: value for row in rows
                if (value := accepted_rows(row, protocol)) is not None}
    comparisons = []; coverage = []
    for cid in cases:
        case_rows = [row for row in rows if row['case'] == cid]
        ready = [row['request_id'] for row in case_rows if row['request_id'] in accepted]
        current = []
        for i, left in enumerate(ready):
            for right in ready[i + 1:]:
                value = {'case': cid, 'left': left, 'right': right,
                         **protocol[2](accepted[left], accepted[right], 3e-6)}
                current.append(value); comparisons.append(value)
        subset = bool(current) and len(ready) == len(case_rows) and all(v['passed'] for v in current)
        coverage.append({'case': cid, 'requested_lanes': len(case_rows), 'host_checked_lanes': len(ready),
                         'six_lane_coverage': len(case_rows) == len(ready) == 6,
                         'compared_pairs': len(current),
                         'compared_subset_agreement': None if not current else all(v['passed'] for v in current),
                         'requested_subset_agreement': subset,
                         'six_lane_agreement': len(case_rows) == len(ready) == 6
                             and len(current) == 15 and all(v['passed'] for v in current)})
    return comparisons, coverage


def aggregate(original_verified_summary, continuation_summary, protocol):
    """Combine only independently verified original-four and continuation20 proofs.

    This is pure derivation, not proof authentication. The caller verifies both
    complete portable capsules and the continuation's admitted original ancestor
    first. Unexecuted original suffix rows are replaced exactly once; accepted
    original prefix rows and their failing comparisons cannot be discarded.
    """
    original = original_verified_summary['request_rows']
    current = continuation_summary['request_rows']
    original_ids = [row['request_id'] for row in original]
    require(len(original_ids) == len(set(original_ids)) == 24, 'Expected24 unique ancestor identities')
    require(original_verified_summary['stage_status'] == 'failed'
            and original_verified_summary['completed_orchestration'] == 4
            and original_verified_summary['host_checked_complete_streams'] == 4,
            'Ancestor is not the admitted four-completed stage')
    require(continuation_summary['requested'] == 20
            and continuation_summary['original_request_order'] == original_ids
            and continuation_summary['inherited_request_ids'] == original_ids[:4]
            and continuation_summary['continuation_request_ids'] == original_ids[4:],
            'Continuation ancestry or suffix differs')
    left = original[:4]; right = current
    require([r['request_id'] for r in right] == original_ids[4:]
            and not {r['request_id'] for r in left} & {r['request_id'] for r in right},
            'Coverage is not the exact disjoint prefix and suffix')
    for i, row in enumerate(original):
        require(row['request_id'] == row['case'] + '/' + row['label'], 'Ancestor request identity differs')
        expected = 'host_checked_complete' if i < 4 else ('deferred_before_launch' if i == 4 else 'not_started')
        require(row['state'] == expected and row['host_checked_complete_stream'] is (i < 4),
                'Ancestor completed/deferred/not-started states differ')
        if i >= 4:
            require(not row['qualified_final_points'] and not row['raw_final_points_observed']
                    and row['warmup_calls_observed'] == 0, 'Ancestor suffix was already observed')
    for old, new in zip(original[4:], right):
        require(all(old[name] == new[name] for name in ('request_id', 'case', 'label', 'source', 'backend')),
                'Continuation request identity differs')
    ids_by_case = list(dict.fromkeys(row['case'] for row in original))
    require(len(ids_by_case) == 4 and [r['case'] for r in original] == [cid for cid in ids_by_case for unused in range(6)],
            'Aggregate is not four ordered six-lane cases')
    prefix_comparisons, unused = compare_coverage(left, protocol)
    require(original_verified_summary['comparisons'] == prefix_comparisons,
            'Inherited comparisons differ from accepted original tokens')
    require(len(current) == 20 and sum(bool(r['request_completion_recorded']) for r in current)
            == continuation_summary['completed_orchestration'], 'Continuation completion count differs')
    require(continuation_summary['stage_status'] in ('passed', 'failed')
            and continuation_summary['host_checked_complete_streams']
                == sum(r['host_checked_complete_stream'] for r in current),
            'Continuation checked-stream count or terminal state differs')
    if continuation_summary['stage_status'] == 'passed':
        require(continuation_summary['completed_orchestration'] == 20,
                'Passed continuation lacks complete orchestration')
    current_comparisons, current_coverage = compare_coverage(current, protocol)
    require(continuation_summary['comparisons'] == current_comparisons
            and continuation_summary['case_coverage'] == current_coverage,
            'Continuation comparisons differ from accepted tokens')
    rows = left + right
    comparisons, coverage = compare_coverage(rows, protocol)
    inherited_pairs = {(v['left'], v['right']): v for v in prefix_comparisons}
    require(all(inherited_pairs[(v['left'], v['right'])] == v for v in comparisons
                if (v['left'], v['right']) in inherited_pairs), 'Inherited comparison changed')
    return {'scope': 'four frozen stress streams: independently verified original4 plus continuation20',
            'request_rows': rows, 'case_coverage': coverage, 'comparisons': comparisons,
            'requested': 24, 'completed_orchestration': 4 + continuation_summary['completed_orchestration'],
            'host_checked_complete_streams': sum(r['host_checked_complete_stream'] for r in rows),
            'inherited_request_ids': original_ids[:4], 'continuation_request_ids': original_ids[4:],
            'inherited_comparisons': prefix_comparisons,
            'inherited_failing_comparisons': [v for v in prefix_comparisons if not v['passed']],
            'missing_or_unqualified_request_ids': [r['request_id'] for r in rows
                if not r['host_checked_complete_stream']],
            'not_completed_orchestration_request_ids': [r['request_id'] for r in current
                if not r['request_completion_recorded']],
            'unexecuted_request_ids': [r['request_id'] for r in rows
                if r['state'] in ('not_started', 'deferred_before_launch')],
            'original_stage_status': original_verified_summary['stage_status'],
            'continuation_stage_status': continuation_summary['stage_status'],
            'full24_orchestration_complete': continuation_summary['stage_status'] == 'passed'
                and continuation_summary['completed_orchestration'] == 20,
            'six_lane_all_cases_agree': continuation_summary['stage_status'] == 'passed'
                and all(v['six_lane_agreement'] for v in coverage),
            'independent_physical_accuracy_qualified': False, 'EW_analytic_oracle_qualified': False,
            'performance_eligible': False, 'warmup_counts_are_observed_only': True}
