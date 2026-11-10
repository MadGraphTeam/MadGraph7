"""Pure mocked/closed-data summary policy; no filesystem or numerical execution.

The future portable binder must authenticate specifications, raw files, runtime
maps and owned cleanup first. This module cannot turn unbound records into proof.
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
    """Derive all24 states. records have already authenticated artifact bindings.

    runtime_verified is the future binder's computed verdict, never a saved
    success field trusted from execution. Failures remain valid retained data.
    """
    parse, validate_status, compare, aliases = protocol
    requests = plan['requests']; keys = [key(r) for r in requests]
    require(len(keys) == len(set(keys)) == 24, 'Expected24 unique requests')
    require(set(records) <= set(keys), 'Unknown request evidence')
    require(stage['status'] in ('passed', 'failed'), 'Stage is not terminal')
    require(accounting['requested'] == 24, 'Accounting request count differs')
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
        elif fatal or record.get('binding_errors') or aborted_here:
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
                     'coefficient_comparison_eligible': checked})
    require(accounting['completed'] == len(completed), 'Completed request accounting differs')
    require([key(r) for r in accounting['not_completed']] == [rid for rid in keys if rid not in completed],
            'Unexecuted request accounting differs')
    if stage['status'] == 'passed':
        require(len(completed) == 24 and aborted is None, 'Passed stage lacks complete orchestration')
    elif aborted is None:
        require(len(completed) == 24, 'Failed partial stage lacks abort record')
    comparisons = []; coverage = []
    for case in plan['cases']:
        cid = case['case']['id']; case_keys = [rid for rid in keys if rid.startswith(cid + '/')]
        ready = [rid for rid in case_keys if rid in accepted]
        current = []
        for i, left in enumerate(ready):
            for right in ready[i + 1:]:
                value = {'case': cid, 'left': left, 'right': right, **compare(accepted[left], accepted[right], 3e-6)}
                current.append(value); comparisons.append(value)
        coverage.append({'case': cid, 'requested_lanes': 6, 'host_checked_lanes': len(ready),
                         'six_lane_coverage': len(ready) == 6, 'compared_pairs': len(current),
                         'compared_subset_agreement': None if not current else all(v['passed'] for v in current),
                         'six_lane_agreement': len(ready) == 6 and len(current) == 15 and all(v['passed'] for v in current)})
    return {'scope': 'four frozen physical stress streams under original strict automatic policy',
            'stage_status': stage['status'], 'request_rows': rows, 'case_coverage': coverage,
            'comparisons': comparisons, 'requested': 24, 'completed_orchestration': len(completed),
            'host_checked_complete_streams': len(accepted),
            'six_lane_all_cases_agree': all(v['six_lane_agreement'] for v in coverage),
            'independent_physical_accuracy_qualified': False, 'EW_analytic_oracle_qualified': False,
            'performance_eligible': False, 'warmup_counts_are_observed_only': True}
