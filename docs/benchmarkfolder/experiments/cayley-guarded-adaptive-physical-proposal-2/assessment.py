"""Pure receipt/fixed-QP checks; the checked qualifier stays unchanged."""
import math
import re

def require(condition, message):
    if not condition:
        raise ValueError(message)

def accept_build_receipt(record, *, source_index_path, source_index_sha256, build_command):
    require(record.get('complete') is True, 'Build did not complete')
    require(record.get('loader_or_native_test_executed') is False, 'Unexpected build-stage native execution')
    require(record.get('root_admission_received') is True, 'Build admission missing')
    require(record.get('cpu_exclusive') is False, 'Unexpected CPU reservation')
    for name, expected in [('cpu', 11), ('jobs', 1), ('automatic_retries', 0), ('overall_seconds', 1800)]:
        require(type(record.get(name)) is int and record[name] == expected, 'Wrong build condition ' + name)
    require(record.get('address_space_limit_bytes_per_build_process') == 8589934592, 'Wrong admitted build resource limit')
    require(record.get('address_space_limit_bytes_per_cache_copy_process') == 4294967296, 'Wrong cache-copy resource limit')
    commands = record.get('commands', [])
    require(len(commands) == 2 and [c['label'] for c in commands] == ['copy-cache', 'release-build'], 'Wrong build command sequence')
    require(commands[1]['command'] == build_command, 'Build command changed')
    require(all(c['progress'].get('process_reaped') is True and type(c['progress'].get('returncode')) is int and c['progress']['returncode'] == 0 for c in commands), 'Build child failed or was not reaped')
    before = record.get('protected_inputs_before', {})
    require(before and before == record.get('protected_inputs_after'), 'Build source closure absent or changed')
    require(before.get(source_index_path) == source_index_sha256, 'Wrong build source proposal')
    elapsed = record.get('elapsed_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed < 1800, 'Build budget not satisfied')
    library = record.get('private_library_not_loaded', {})
    require(re.fullmatch('[0-9a-f]{64}', library.get('sha256', '')) is not None, 'Missing actual DSO hash')
    require(type(library.get('bytes')) is int and library['bytes'] > 0, 'Missing DSO size')
    return library

def finite_complex_row(row):
    kind, _, *words = row
    if kind == 'C32' and len(words) == 4:
        return all((int(words[i], 16) >> 48) & 0x7fff != 0x7fff for i in (1, 3))
    if kind == 'C16' and len(words) == 2:
        return all((int(w, 16) >> 52) & 0x7ff != 0x7ff for w in words)
    return False

def qualify_fixed(capture, controls, parser, *, omissions):
    errors = []
    if not capture['complete'] or set(capture['points']) != {1, 2}:
        errors.append('incomplete point set')
    if type(omissions) is not int or omissions != 0:
        errors.append('W03 missing contributions')
    if set(controls) != set(capture['points']):
        errors.append('driver point set differs')
    for point in capture['points'].values():
        number = point['number']
        def check(condition, reason):
            if not condition:
                errors.append(f'point{number}: {reason}')
        check(parser.scalar(point, 'PUBLIC_STATUS') == '120', 'fixed-QP public code is not unchecked120')
        control = controls.get(number, {})
        check(tuple(control.get(k) for k in ('H', 'T', 'U')) == (1, 2, 0), 'driver public-code mismatch')
        accuracy = point['data'].get('ACCURACY', [])
        check(len(accuracy) == 2 and all(row[2:] == ['BFF0000000000000'] for row in accuracy), 'accuracy is not the unchanged -1 sentinel')
        check(control.get('accuracy') == [-1.0, -1.0], 'driver accuracy mismatch')
        check(len(point['groups']) == capture['expected_groups'], 'not exactly one QP traversal')
        check(len(point['model_snapshots']) == 1, 'not exactly one genuine-QP model snapshot')
        check(all(parser.scalar(m, 'GENUINE_QP_INTEGRAND') == 'T' for m in point['model_snapshots']), 'genuine QP integrand absent')
        for row in point['data'].get('AMPLITUDE', []):
            check(parser.finite_value(row[2:]) is not None, 'nonfinite amplitude')
        for group in point['groups']:
            check(group['key'][3] == 4, 'non-fixed QP mode')
            check(len(group['native']) == 1 and group['native'][0]['backend'] == 9 and group['native'][0]['status'] == 0, 'native call missing/refused')
            for label, expected in [('ACTUAL_BACKEND', '9'), ('GROUP_STABLE', 'T'), ('CHECK_PASS', 'F'), ('CACHE_ENABLED', 'T')]:
                check(parser.scalar(group, label) == expected, label + ' changed')
            for label in (*parser.RAW, 'GROUP_RESULT'):
                rows = group['data'].get(label, [])
                check(bool(rows) and all(finite_complex_row(row) for row in rows), 'nonfinite/missing ' + label)
    return {'fixed_qp_outputs_qualified': not errors, 'errors': errors,
            'stability_checked': False, 'public120_means_unchecked': True,
            'per_call_working_precision_qualified': False,
            'independent_physical_accuracy_qualified': False}

def previously_served_preserved(before, after, parser):
    failures = []
    count = 0
    missing = []
    matched = 0
    for number, point in before['points'].items():
        current = after['points'].get(number, {})
        groups = {tuple(g['key']): g for g in current.get('groups', [])}
        for group in point['groups']:
            if not group['native'] or group['native'][0]['status'] != 0:
                continue
            count += 1
            other = groups.get(tuple(group['key']))
            if other is None:
                missing.append([number, group['key'][0]])
                continue
            matched += 1
            for label in (*parser.RAW, 'GROUP_RESULT'):
                if group['data'].get(label) != other['data'].get(label):
                    failures.append([number, group['key'][0], label])
    return {'previously_served_groups': count, 'preserved_exact': not failures and not missing,
            'matched_groups': matched, 'matched_groups_bit_exact': not failures,
            'missing_groups_are_coverage_changes': missing,
            'failures': failures, 'independent_physical_truth_qualified': False}

def compare_amplitudes(left, right, parser, benchmark):
    """Numerical ANS comparison independent of status, flags and rescue path."""
    if not left['complete'] or not right['complete'] or set(left['points']) != {1, 2} or set(right['points']) != {1, 2}:
        return {'passed': False, 'reason': 'incomplete physical outputs', 'ordinary_target_passed': False}
    def rows(capture):
        result = []
        for number in (1, 2):
            words = capture['points'][number]['data'].get('AMPLITUDE', [])
            require(len(words) == 8, 'Incomplete two-column amplitude')
            values = [parser.finite_value(row[2:]) for row in words]
            require(all(value is not None for value in values), 'Nonfinite amplitude')
            for order in (0, 1):
                result.append({'point': number, 'order': order,
                               'coefficients': [float(v) for v in values[4*order:4*order+4]]})
        return result
    try:
        left_rows, right_rows = rows(left), rows(right)
    except ValueError as error:
        return {'passed': False, 'reason': str(error), 'ordinary_target_passed': False}
    comparison = benchmark.compare_rows(left_rows, right_rows, tolerance=3e-6)
    comparison['public_status_accuracy_bits_and_precision_usage_compared_separately'] = True
    comparison['independent_physical_truth_qualified'] = False
    return comparison

def classify_paths(left, right, parser):
    def view(capture):
        summary = {p['point']: p for p in parser.summarize(capture)['points']}
        result = {}
        for number, point in capture['points'].items():
            raw = parser.scalar(point, 'PUBLIC_STATUS')
            code = int(raw) if raw is not None else None
            result[number] = {'public_code': code,
                'accepted_precision': 'DP' if code == 219 else 'QP' if code == 323 else 'unchecked_QP' if code == 120 else 'not_accepted',
                'accuracy_bits': point['data'].get('ACCURACY', []),
                **{key: summary[number][key] for key in ('groups', 'pass_modes', 'model_snapshots', 'native_calls', 'served_calls', 'native_refusals')}}
        return result
    first, second = view(left), view(right)
    changes = []
    for number in sorted(set(first) | set(second)):
        a, b = first.get(number), second.get(number)
        changes.append({'point': number, 'before': a, 'after': b,
                        'accepted_precision_changed': a is not None and b is not None and a['accepted_precision'] != b['accepted_precision'],
                        'observed_call_count_changed': a is not None and b is not None and a['native_calls'] != b['native_calls'],
                        'status_or_accuracy_or_pass_usage_changed': a != b})
    return {'points': changes, 'FPE_flags_can_change_rescue_selection': True,
            'path_identity_is_not_a_numerical_acceptance_requirement': True,
            'cleaner_flags_do_not_qualify_host_stability': True,
            'per_call_working_precision_qualified': False}

def separate_gates(observations, transport_valid, fault_controls_valid=False):
    """Keep numerical agreement, MadLoop checks, QP use and traps distinct."""
    result = {}
    for mode in ('checked', 'forced-qp'):
        row = observations.get(mode, {})
        valid = bool(transport_valid)
        both = lambda label: all(row.get(label, {}).get(policy, False) for policy in ('masked', 'trapped'))
        result[mode] = {
            'numerical_agreement': bool(valid and both('numerical_to_closed_adaptive')),
            'host_stability_acceptance': bool(valid and mode == 'checked' and both('host_stability_acceptance')),
            'QP_path_coverage': bool(valid and both('observed_QP_path_coverage')),
            'trap_completion': bool(valid and fault_controls_valid and row.get('IEEE_policy_worker_completion', {}).get('trapped', False)),
            'masked_trapped_numerical_agreement': bool(valid and row.get('policy_numerical_agreement', False)),
            'per_call_working_precision_qualified': False,
        }
    return result
