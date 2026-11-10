#!/usr/bin/env python3
"""Strict, offline reader for the existing genuine-QP mass observer journal.

Parsing is structural evidence, not numerical qualification.  A final trapped
prefix may be retained with allow_partial=True; it can never qualify.  Passes
are observed full group traversals, without inventing a rotation label absent
from EW_QP_MASS_HEX_V1.  No floating-point flag is changed by this reader.
"""
from fractions import Fraction
import math
from pathlib import Path
import re


FORMAT = 'FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words'
COUNTS = {'R8': 1, 'R16': 2, 'C16': 2, 'C32': 4}
POINT_ARRAYS = {'INPUT_P': ('R8', 20), 'RETURN_P': ('R8', 20),
                'AMPLITUDE': ('R8', 8), 'ACCURACY': ('R8', 2)}
POINT_SCALARS = {'PUBLIC_STATUS': 'INTEGER', **{
    name: 'LOGICAL' for name in ('FPE_DP_BEGIN', 'FPE_QP_BEGIN',
                                 'FPE_DP_END', 'FPE_QP_END')}}
MODEL_ARRAYS = {'CONSUMED_MP_PS': ('R16', 20), 'MODEL_MU_R': ('R8', 1),
                'MODEL_MP__MU_R': ('C32', 1), 'MODEL_GAL': ('C16', 2),
                'MODEL_MP__GAL': ('C32', 2)}
for _mass in ('MT', 'MZ', 'MW', 'MH'):
    MODEL_ARRAYS['MODEL_MDL_' + _mass] = ('R8', 1)
    MODEL_ARRAYS['MODEL_MP__MDL_' + _mass] = ('R16', 1)
MODEL_SCALARS = {'MODEL_UPDATE': 'INTEGER', 'GENUINE_QP_INTEGRAND': 'LOGICAL'}
GROUP_SCALARS = {'ACTUAL_BACKEND': 'INTEGER', 'GROUP_STABLE': 'LOGICAL',
                 'CHECK_PASS': 'LOGICAL', 'CACHE_ENABLED': 'LOGICAL'}
RAW = ('RAW_UV1', 'RAW_IR1', 'RAW_IR2', 'RAW_FINITE')
NATIVE_ARRAYS = ('NATIVE_SHIFTS', 'KINEMATIC_S', 'NATIVE_INVARIANTS',
                 'NATIVE_MASSES', 'NATIVE_SCALE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(word):
    require(re.fullmatch(r'-?(0|[1-9][0-9]*)', word) is not None,
            'Noncanonical integer: ' + word)
    return int(word)


def flags(word):
    require(re.fullmatch('[TF]{5}', word) is not None, 'Invalid IEEE flags')
    return word


def group_arrays(group):
    n, rank = group['key'][4:6]
    nc = math.comb(rank + 4, 4)
    result = {'DP_M2L': ('C16', n), 'QP_M2L': ('C32', n),
              'HELPER_DP_PL': ('R8', 4*n), 'HELPER_QP_PL': ('R16', 4*n),
              'REPAIRED_MP_PS': ('R16', 20), 'CURRENT_MP_P': ('R16', 20),
              'NUMERATOR_COEFS': ('C32', nc), 'ENTRY_PL': ('R16', 4*n),
              'ENTRY_M2_QP': ('C32', n), 'ORDERED_PL': ('R16', 4*n),
              'ORDERED_MASSES': ('C32', n), 'NATIVE_SHIFTS': ('R16', 4*n),
              'KINEMATIC_S': ('C32', n*n), 'NATIVE_INVARIANTS': ('R16', n*n),
              'NATIVE_MASSES': ('C32', n), 'NATIVE_SCALE': ('R16', 1),
              'GROUP_RESULT': ('C16', 3)}
    result.update({name: ('C32', nc) for name in RAW})
    for i in range(1, n+1):
        result['ARG_M' + str(i)] = ('C16', 1)
        result['MP_ARG_M' + str(i)] = ('C32', 1)
    return result


def check_data(target, arrays, scalars, *, partial=False, optional=()):
    data = target['data']
    require(set(data) <= set(arrays) | set(scalars), 'Unknown data label')
    for label, (kind, size) in arrays.items():
        rows = data.get(label, [])
        require(len(rows) <= size, 'Oversized array ' + label)
        require([r[1] for r in rows] == list(range(1, len(rows)+1)),
                'Nonconsecutive indices for ' + label)
        require(all(r[0] == kind for r in rows), 'Wrong kind for ' + label)
        if not partial and label not in optional:
            require(len(rows) == size, 'Missing/incomplete array ' + label)
        if label in optional and rows and not partial:
            require(len(rows) == size, 'Incomplete optional array ' + label)
    for label, kind in scalars.items():
        rows = data.get(label, [])
        require(len(rows) <= 1 and all(r[0] == kind for r in rows),
                'Repeated/wrong scalar ' + label)
        if not partial and label not in optional:
            require(len(rows) == 1, 'Missing scalar ' + label)


def scalar(target, label):
    rows = target['data'].get(label, [])
    return rows[0][1] if rows else None


def check_model(model, partial=False):
    check_data(model, MODEL_ARRAYS, MODEL_SCALARS, partial=partial)
    require(scalar(model, 'MODEL_UPDATE') in (None, '1'), 'Unexpected MODEL_UPDATE')
    require(scalar(model, 'GENUINE_QP_INTEGRAND') in (None, 'T'),
            'Genuine QP integrand disabled')


def check_group(group, partial=False):
    arrays = group_arrays(group)
    native = group['native']
    require(len(native) <= 1, 'Repeated native call')
    if native:
        require(native[0]['backend'] == 9, 'Unexpected native backend')
    served = bool(native and native[0]['status'] == 0)
    optional = set(RAW) if not served else set()
    if not native:
        optional |= set(NATIVE_ARRAYS) | {'ORDERED_PL', 'ORDERED_MASSES',
                                         'CHECK_PASS', 'CACHE_ENABLED'}
    check_data(group, arrays, GROUP_SCALARS, partial=partial, optional=optional)
    require(scalar(group, 'ACTUAL_BACKEND') in (None, '9'), 'Unexpected chosen backend')
    require(scalar(group, 'CHECK_PASS') in
            (None, 'T' if group['key'][3] == 5 else 'F'), 'Wrong check-pass store')
    require(scalar(group, 'CACHE_ENABLED') in (None, 'T'), 'Cache policy changed')
    if not served:
        require(not any(label in group['data'] for label in RAW),
                'RAW result without successful native call')
    require(not group['errors'] or bool(native and native[0]['status'] != 0),
            'Error bytes without a native refusal')
    require(len(group['errors']) <= 1, 'Repeated native error record')
    if not partial:
        # Native-free early adapter returns remain visible and never become QP truth.
        require(bool(native) or scalar(group, 'GROUP_STABLE') == 'F',
                'Early adapter return marked stable')
        require(not native or native[0]['status'] == 0 or
                scalar(group, 'GROUP_STABLE') == 'F', 'Refusal marked stable')
        if not native:
            require(not any(name in group['data'] for name in NATIVE_ARRAYS)
                    and scalar(group, 'CHECK_PASS') is None
                    and scalar(group, 'CACHE_ENABLED') is None,
                    'Native argument snapshot without native call')
            require(('ORDERED_PL' in group['data']) == ('ORDERED_MASSES' in group['data']),
                    'Incomplete early-return ordered arguments')


def parse(path, allow_partial=False, *, expected_groups=200):
    """Read complete journal, or retain only an explicitly unqualified prefix.

    expected_groups=200 is the frozen physical process.  The override exists for
    compact synthetic tests, and is retained in the returned evidence identity.
    """
    require(type(expected_groups) is int and expected_groups > 0,
            'Invalid expected group count')
    text = Path(path).read_text(encoding='utf-8')
    lines = text.splitlines()
    require(lines and lines[0] == FORMAT, 'Missing exact format marker')
    points = {}; point = None; group = None; model = None
    footer = None; records = 0; trailing_fragment = False
    # A trap may interrupt a write.  Only a final non-newline fragment is dropped;
    # all preceding complete records retain their ordinary validation.
    if allow_partial and not text.endswith('\n') and len(lines) > 1:
        trailing_fragment = True
        lines = lines[:-1]
    for number, line in enumerate(lines[1:], 2):
        require(bool(line.strip()), f'Blank record at line {number}')
        row = line.split(); tag = row[0]
        require(footer is None, 'Record after footer')
        if tag not in ('BEGIN', 'END', 'FOOTER'):
            records += 1
        if tag == 'BEGIN':
            require(len(row) == 3 and point is None, 'Invalid/nested BEGIN')
            k = integer(row[1])
            require(k == len(points)+1 and k in (1, 2), 'Point sequence changed')
            point = {'number': k, 'begin_flags': flags(row[2]), 'data': {},
                     'groups': [], 'model_snapshots': [], 'ended': False}
            points[k] = point; group = None; model = None
        elif tag == 'GROUP':
            require(len(row) == 9 and point is not None, 'Invalid GROUP')
            if group is not None:
                check_group(group)
            if model is not None:
                check_model(model)
            require(integer(row[1]) == point['number'], 'GROUP point mismatch')
            key = [integer(x) for x in row[2:8]]
            ordinal, gid, sqso, mode, n, rank = key
            require(ordinal == len(point['groups'])+1, 'Nonconsecutive group ordinal')
            require(gid == 1+(ordinal-1) % expected_groups, 'Nonconsecutive pass group ID')
            require(sqso == 1 and mode in (4, 5) and 1 <= n <= 5 and 0 <= rank <= 3,
                    'Unsupported frozen process group shape/mode')
            require(model is not None, 'QP group has no model snapshot')
            if (ordinal-1) % expected_groups:
                require(mode == point['groups'][-1]['key'][3], 'Mode changed within pass')
            group = {'key': key, 'flags': flags(row[8]), 'data': {},
                     'native': [], 'errors': [], 'model_snapshot': len(point['model_snapshots'])}
            point['groups'].append(group)
        elif tag in COUNTS or tag in ('INTEGER', 'LOGICAL'):
            require(point is not None, 'Data outside a point')
            require(len(row) >= 2, 'Missing data label')
            label = row[1]
            if label == 'MODEL_UPDATE':
                require(tag == 'INTEGER' and len(row) == 4, 'Invalid snapshot marker')
                if model is not None:
                    check_model(model)
                model = {'data': {}}; point['model_snapshots'].append(model)
            if label in POINT_ARRAYS or label in POINT_SCALARS:
                target = point
            elif label in MODEL_ARRAYS or label in MODEL_SCALARS:
                require(model is not None, 'Snapshot data before MODEL_UPDATE')
                target = model
            else:
                require(group is not None, 'Unknown/ungrouped data label ' + label)
                require(label in group_arrays(group) or label in GROUP_SCALARS,
                        'Unknown group label ' + label)
                target = group
            if tag in COUNTS:
                require(len(row) == 3+COUNTS[tag], 'Wrong real/complex word count')
                index = integer(row[2]); require(index > 0, 'Nonpositive array index')
                require(all(re.fullmatch('[0-9A-F]{16}', word) for word in row[3:]),
                        'Invalid hexadecimal word')
                record = [tag, index, *row[3:]]
            else:
                require(len(row) == 4, 'Wrong scalar width')
                if tag == 'INTEGER':
                    integer(row[2])
                else:
                    require(row[2] in ('T', 'F'), 'Invalid logical value')
                record = [tag, row[2], flags(row[3])]
            target['data'].setdefault(label, []).append(record)
            # Reject wrong kinds/indices immediately, also in trapped prefixes.
            if target is point:
                check_data(target, POINT_ARRAYS, POINT_SCALARS, partial=True)
            elif target is model:
                check_model(target, partial=True)
            else:
                check_group(target, partial=True)
        elif tag == 'NATIVE':
            require(group is not None and len(row) == 4, 'Invalid NATIVE')
            group['native'].append({'backend': integer(row[1]), 'status': integer(row[2]),
                                    'flags': flags(row[3])})
            check_group(group, partial=True)
        elif tag == 'ERROR_BYTES':
            require(group is not None and all(re.fullmatch('[0-9A-F]{2}', x) for x in row[1:]),
                    'Invalid error bytes')
            group['errors'].append(bytes(int(x, 16) for x in row[1:]).decode('utf-8', 'replace'))
            check_group(group, partial=True)
        elif tag == 'END':
            require(point is not None and len(row) == 5, 'Invalid END')
            require(integer(row[1]) == point['number'] and integer(row[2]) == len(point['groups'])
                    and integer(row[3]) == (point['groups'][-1]['key'][1] if point['groups'] else 0),
                    'END group counts disagree')
            if group is not None:
                check_group(group)
            if model is not None:
                check_model(model)
            check_data(point, POINT_ARRAYS, POINT_SCALARS)
            require(len(point['groups']) % expected_groups == 0, 'Incomplete group traversal')
            require(bool(point['groups']) or not point['model_snapshots'],
                    'Model snapshots without any QP group')
            point['end_flags'] = flags(row[4]); point['ended'] = True
            point = None; group = None; model = None
        elif tag == 'FOOTER':
            require(point is None and len(row) == 5, 'Invalid FOOTER')
            footer = dict(zip(('records', 'dropped', 'flag_changes', 'limit'),
                              (integer(x) for x in row[1:])))
            require(footer == {'records': records, 'dropped': 0, 'flag_changes': 0,
                               'limit': 1000000}, 'Capped/flag-changing/miscounted footer')
        elif tag == 'OBSERVER_FLAG_CHANGE':
            raise ValueError('Observer changed IEEE flags')
        else:
            raise ValueError('Unknown observer record ' + tag)
    complete = (set(points) == {1, 2} and all(p['ended'] for p in points.values())
                and footer is not None and not trailing_fragment)
    require(complete or allow_partial, 'Incomplete observer journal')
    if point is not None:
        if group is not None:
            check_group(group, partial=True)
        if model is not None:
            check_model(model, partial=True)
    return {'points': points, 'footer': footer, 'complete': complete,
            'partial': not complete, 'trailing_fragment': trailing_fragment,
            'expected_groups': expected_groups, 'observed_records': records}


def finite_value(words, bits=64):
    """Exact finite binary64/binary128 value, or None for either infinity/NaN."""
    require(bits in (64, 128) and len(words) == bits//64, 'Invalid IEEE word count')
    value = sum(int(word, 16) << (64*i) for i, word in enumerate(words))
    mantissa_bits, exp_bits, bias = (52, 11, 1023) if bits == 64 else (112, 15, 16383)
    exponent = (value >> mantissa_bits) & ((1 << exp_bits)-1)
    if exponent == (1 << exp_bits)-1:
        return None
    mantissa = value & ((1 << mantissa_bits)-1)
    if exponent:
        mantissa += 1 << mantissa_bits
    sign = -1 if value >> (bits-1) else 1
    power = (exponent-bias if exponent else 1-bias)-mantissa_bits
    return sign*Fraction(mantissa)*Fraction(2)**power


def summarize(capture):
    points = []
    size = capture['expected_groups']
    for p in capture['points'].values():
        groups = p['groups']; native = [n for g in groups for n in g['native']]
        modes = [groups[i]['key'][3] for i in range(0, len(groups), size)]
        points.append({'point': p['number'], 'ended': p['ended'], 'groups': len(groups),
                       'complete_passes': len(groups)//size, 'pass_modes': modes,
                       'model_snapshots': len(p['model_snapshots']), 'native_calls': len(native),
                       'native_refusals': sum(n['status'] != 0 for n in native),
                       'early_adapter_returns': sum(not g['native'] for g in groups),
                       'served_calls': sum(n['status'] == 0 for n in native),
                       'public_status': scalar(p, 'PUBLIC_STATUS'),
                       'accuracy_bits': p['data'].get('ACCURACY', []),
                       'mode4_mode5_repeat_sequence': modes == [4, 5, 4] and len(groups) == 3*size,
                       'rotation_label_available': False})
    return {'complete': capture['complete'], 'partial': capture['partial'], 'points': points,
            'scope': 'Observed traversals only; the journal does not label rotation identity.'}


def physical_data(data):
    """Only scalar records carry a trailing event flag word; retain their value."""
    return {label: [row[:2] if row[0] in ('INTEGER', 'LOGICAL') else row[:]
                    for row in rows] for label, rows in data.items()}


def flag_observations(capture):
    """Keep every journal event flag, independently of input-value comparison."""
    result = []
    for k, point in capture['points'].items():
        for label in ('begin_flags', 'end_flags'):
            if label in point:
                result.append({'point': k, 'scope': 'point', 'label': label,
                               'flags': point[label]})
        targets = [('point', 0, point)]
        targets.extend(('model', i, model)
                       for i, model in enumerate(point['model_snapshots'], 1))
        targets.extend(('group', group['key'][0], group) for group in point['groups'])
        for scope, index, target in targets:
            if 'flags' in target:
                result.append({'point': k, 'scope': scope, 'index': index,
                               'label': 'GROUP', 'flags': target['flags']})
            for label, rows in target['data'].items():
                for row in rows:
                    if row[0] in ('INTEGER', 'LOGICAL'):
                        result.append({'point': k, 'scope': scope, 'index': index,
                                       'label': label, 'flags': row[-1]})
            for native in target.get('native', []):
                result.append({'point': k, 'scope': scope, 'index': index,
                               'label': 'NATIVE', 'flags': native['flags']})
    return result


def compare_inputs_and_born(left, right):
    failures = []; sequence_differences = []
    pass_comparable = True; model_comparable = True
    if set(left['points']) != set(right['points']):
        failures.append({'scope': 'point_set'})
    for k in sorted(set(left['points']) & set(right['points'])):
        a, b = left['points'][k], right['points'][k]
        for label in ('INPUT_P', 'RETURN_P'):
            if a['data'].get(label) != b['data'].get(label):
                failures.append({'point': k, 'scope': 'point_input', 'label': label})
        # ANS(0:3,0:1) is storage order: Born is element0 of each column.
        for i in (0, 4):
            av = a['data'].get('AMPLITUDE', []); bv = b['data'].get('AMPLITUDE', [])
            if len(av) <= i or len(bv) <= i or av[i] != bv[i]:
                failures.append({'point': k, 'scope': 'born', 'flat_index': i+1})
        models_a = [physical_data(x['data']) for x in a['model_snapshots']]
        models_b = [physical_data(x['data']) for x in b['model_snapshots']]
        if models_a != models_b:
            sequence_differences.append({'point': k, 'scope': 'model_snapshot_sequence',
                                         'left_count': len(models_a), 'right_count': len(models_b),
                                         'event_flags_excluded': True})
        if len(models_a) != len(models_b):
            model_comparable = False
        # Compare every shared snapshot value, preserving all unmatched extra
        # snapshots in capture and classifying unequal coverage separately.
        for i, (x, y) in enumerate(zip(models_a, models_b), 1):
            for label in sorted(set(x) | set(y)):
                if x.get(label) != y.get(label):
                    failures.append({'point': k, 'scope': 'model_input',
                                     'snapshot': i, 'label': label})
        if [g['key'] for g in a['groups']] != [g['key'] for g in b['groups']]:
            pass_comparable = False
            sequence_differences.append({'point': k, 'scope': 'group_sequence',
                                         'left_keys': [g['key'] for g in a['groups']],
                                         'right_keys': [g['key'] for g in b['groups']]})
        else:
            excluded = set(RAW) | {'GROUP_RESULT', 'GROUP_STABLE'}
            for x, y in zip(a['groups'], b['groups']):
                dx, dy = physical_data(x['data']), physical_data(y['data'])
                labels = (set(dx) | set(dy)) - excluded
                for label in sorted(labels):
                    if dx.get(label) != dy.get(label):
                        failures.append({'point': k, 'ordinal': x['key'][0],
                                         'scope': 'group_input', 'label': label})
    old_flags, new_flags = flag_observations(left), flag_observations(right)
    software_flags = {
        side: {k: {label: [row[:] for row in p['data'].get(label, [])]
                   for label in POINT_SCALARS if label.startswith('FPE_')}
               for k, p in capture['points'].items()}
        for side, capture in (('left', left), ('right', right))}
    return {'inputs_and_born_exact': not failures, 'failures': failures,
            'both_complete': left['complete'] and right['complete'],
            'pass_sequence_comparable': pass_comparable,
            'model_sequence_comparable': model_comparable,
            'all_input_records_comparable': pass_comparable and model_comparable and set(left['points']) == set(right['points']),
            'sequence_differences': sequence_differences,
            'event_flags_equal': old_flags == new_flags,
            'event_flags': {'left': old_flags, 'right': new_flags},
            'software_reduction_flags': software_flags,
            'scope': 'Bitwise shared input/Born comparison; unmatched passes are separate coverage gaps, not input differences. No loop-output accuracy claim.'}


def qualify_checked(capture, controls, *, omissions=0, threshold=1e-6):
    """Check observer output against separately parsed driver controls.

    controls maps point integers to {H,T,U,accuracy:[binary64,binary64]}.
    Omissions must include all W03 diagnostics detected in the complete stdout.
    Native refusals may be recovered by later checked passes; all stay visible.
    This cannot establish independent physical truth or identify a rotation.
    """
    errors = []
    if not capture['complete']:
        errors.append('partial journal')
    if type(omissions) is not int or omissions != 0:
        errors.append('missing contributions or invalid omission count')
    limit = Fraction(str(threshold))
    require(limit > 0, 'Nonpositive accuracy threshold')
    if set(controls) != set(capture['points']):
        errors.append('driver point set differs')
    per_point = []
    for entry in summarize(capture)['points']:
        k = entry['point']; p = capture['points'][k]; failures = []
        control = controls.get(k, {})
        code = scalar(p, 'PUBLIC_STATUS')
        status = integer(code) if code is not None else -1
        htu = (status//100, status//10 % 10, status % 10)
        if tuple(control.get(name) for name in ('H', 'T', 'U')) != htu:
            failures.append('driver status mismatch')
        if htu not in ((2, 1, 9), (3, 2, 3)):
            failures.append('not checked backend9 success')
        acc = [finite_value(row[2:]) for row in p['data'].get('ACCURACY', [])]
        if len(acc) != 2 or any(v is None or v < 0 or v >= limit for v in acc):
            failures.append('accuracy missing/nonfinite/unknown/above threshold')
        ca = control.get('accuracy')
        try:
            agree = len(ca) == 2 and all(math.isfinite(v) for v in ca) and [
                Fraction.from_float(float(v)) for v in ca] == acc
        except (TypeError, ValueError, OverflowError):
            agree = False
        if not agree:
            failures.append('driver accuracy mismatch')
        for row in p['data'].get('AMPLITUDE', []):
            if finite_value(row[2:]) is None:
                failures.append('nonfinite amplitude'); break
        qp_observed = bool(entry['groups'])
        qp_sequence = entry['mode4_mode5_repeat_sequence']
        if htu[0] == 2 and qp_observed:
            failures.append('DP-only success conflicts with observed QP traversal')
        if htu[0] == 3 and not qp_sequence:
            failures.append('QP checked three-pass traversal not observed')
        per_point.append({'point': k, 'status': list(htu), 'errors': failures,
                          'qp_observed': qp_observed, 'qp_checked_sequence_observed': qp_sequence,
                          'rotation_identity_qualified': False})
        errors.extend(f'point{k}: {failure}' for failure in failures)
    return {'checked_outputs_qualified': not errors, 'errors': errors, 'points': per_point,
            'qp_coverage_observed': bool(per_point) and any(p['qp_checked_sequence_observed'] for p in per_point),
            'independent_physical_accuracy_qualified': False,
            'rotation_identity_qualified': False,
            'scope': 'Separate driver controls and complete stdout omission scan are required.'}
