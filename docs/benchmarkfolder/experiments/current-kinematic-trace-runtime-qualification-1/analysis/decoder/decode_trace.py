#!/usr/bin/env python3
"""Exact integer-word decoding and bounded kinematic observations; no native code."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re

DERIVED = Path('.tirust-build/kinematic-provenance-current-executor-1/results/derived')
DERIVED_SHA = '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184'
HEADER = ('point', 'serial', 'id', 'mode', 'rank', 'n', 'i', 'j', 'branch', 'split_order')
WORD = re.compile(r'[0-9A-Fa-f]{16}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ieee(words, bits, qp_word_order):
    """Decode IEEE binary64/128 without a float conversion; preserve NaN/zero bits."""
    require(bits in (64, 128) and len(words) == bits // 64, 'IEEE word count differs')
    require(qp_word_order == 'low-first', 'Explicit GNU x86_64 low-first ABI required')
    require(all(WORD.fullmatch(w) for w in words), 'Malformed INTEGER*8 hexadecimal word')
    value = sum(int(w, 16) << (64 * k) for k, w in enumerate(words))
    fraction_bits, exponent_bits = (52, 11) if bits == 64 else (112, 15)
    sign = value >> (bits - 1)
    exponent = (value >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = value & ((1 << fraction_bits) - 1)
    result = {'words': [w.upper() for w in words], 'bits': bits, 'sign': sign}
    if exponent == (1 << exponent_bits) - 1:
        return {**result, 'kind': 'nan' if fraction else 'infinity',
                'payload_hex': hex(fraction), 'value': None}
    bias = (1 << (exponent_bits - 1)) - 1
    significand = fraction + ((1 << fraction_bits) if exponent else 0)
    power = (exponent or 1) - bias - fraction_bits
    number = F(significand << max(power, 0), 1 << max(-power, 0))
    return {**result, 'kind': 'zero' if significand == 0 else 'finite',
            'significand': str(significand), 'exponent2': power,
            'value': -number if sign else number}


def wire(value):
    """Hex ratios remain exact even for the smallest binary128 subnormal."""
    if isinstance(value, F):
        with localcontext() as ctx:
            ctx.prec = 36
            approx = str(Decimal(value.numerator) / Decimal(value.denominator))
        return {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator),
                'decimal_approximation_36_digits': approx}
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def source_binding(case):
    raw = (DERIVED / 'index.json').read_bytes()
    require(digest(raw) == DERIVED_SHA, 'Derived source index differs')
    index = json.loads(raw)
    require(index['schema'] == 'current-generated-trace-derived-index-v1', 'Derived schema differs')
    names = [case + '/' + name for name in ('CT_interface.f', 'TIRUST_interface.f', 'derivation.json', 'routing.json')]
    data = {}
    for name in names:
        path = DERIVED / name
        require(not path.is_symlink(), 'Derived source cannot be a link')
        content = path.read_bytes()
        require(index['files'][name] == {'bytes': len(content), 'sha256': digest(content)}, 'Derived bytes differ: ' + name)
        data[name] = content
    derivation = json.loads(data[case + '/derivation.json'])
    routing = json.loads(data[case + '/routing.json'])
    require(derivation['header_fields'] == list(HEADER), 'Trace header differs')
    route = next(iter(routing['cases'].values()))
    return derivation, route, {name: index['files'][name] for name in names}


def payload(tag, words, n, order):
    bits = 128 if tag.endswith('_QP') else 64
    cursor = 0

    def take(count, storage_bits=bits):
        nonlocal cursor
        size = storage_bits // 64
        require(cursor + count * size <= len(words), 'Truncated payload: ' + tag)
        values = [ieee(words[cursor + k * size:cursor + (k + 1) * size], storage_bits, order) for k in range(count)]
        cursor += count * size
        return values

    name = tag[3:-3]
    if name == 'LOOP':
        result = {'corners': take(4 * n)}
        if bits == 64:
            result['squared_masses'] = take(2 * n)
    elif name == 'EXTERNAL':
        result = {'complex_momentum': take(8)}
    elif name == 'RESIDUAL':
        result = {'residual': take(4), 'corners': take(4 * n)}
    elif name == 'MATRIX':
        result = {'shifts_component_major': take(4 * n), 'squared_masses': take(2 * n)}
    elif name == 'RAW':
        result = {'diffsq': take(2), 'matrix_entry': take(2)}
    elif name == 'SCALE':
        result = {'reference_normalization': take(1), 'osthres_dp': take(1, 64)}
    elif name == 'TIRUST':
        result = {'shifts_corner_major': take(4 * n), 'squared_masses': take(2 * n),
                  'invariants_column_major': take(n * n), 'scale': take(1)}
    elif name == 'FINAL' or name in {label + '_' + edge for label in ('MASSI', 'MASSJ', 'ZERO') for edge in ('BEFORE', 'AFTER')}:
        result = {'matrix_entry': take(2)}
    else:
        raise ValueError('Unknown payload tag: ' + tag)
    require(cursor == len(words), 'Unexpected payload words: ' + tag)
    return result


def parse(raw, order):
    require(order == 'low-first', 'Explicit low-first ABI selection required')
    records, schemas, section = [], [], -1
    for line_number, line in enumerate(raw.decode('ascii').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == 'KT_SCHEMA_V2':
            require(len(fields) == 6, 'Schema field count differs')
            schema = list(map(int, fields[1:]))
            require(schema[1:] == [64, 128, 64, 32], 'Unsupported Fortran storage schema')
            schemas.append(schema)
            section += 1
            continue
        require(section >= 0 and len(fields) >= 11, 'Missing schema or truncated header')
        header = dict(zip(HEADER, map(int, fields[1:11])))
        require(header['point'] == schemas[-1][0] and header['n'] == 4
                and header['mode'] in (1, 4), 'Outside initial four-propagator mode1/4 scope')
        require(header['serial'] > 0 and header['split_order'] > 0, 'Invalid call identity')
        family = tag[3:-3]
        if family == 'EXTERNAL':
            require(header['i'] > 0 and header['j'] == header['branch'] == 0, 'External record indices differ')
        elif family in ('RAW', 'SCALE', 'FINAL') or family.startswith(('MASSI_', 'MASSJ_', 'ZERO_')):
            expected_branch = next((k for label, k in (('MASSI_', 1), ('MASSJ_', 2), ('ZERO_', 3)) if family.startswith(label)), 0)
            require(1 <= header['i'] <= 4 and 1 <= header['j'] <= 4
                    and header['branch'] == expected_branch, 'Matrix branch indices differ')
        else:
            require(header['i'] == header['j'] == header['branch'] == 0, 'Boundary indices differ')
        if tag in ('KT_STATUS_DP', 'KT_STATUS_QP'):
            require(len(fields) == 12, 'Status record count differs')
            data = {'native_status': int(fields[11])}
        else:
            data = payload(tag, fields[11:], header['n'], order)
        records.append({'line': line_number, 'section': section, 'tag': tag, **header, 'data': data})
    require(schemas, 'No schema record observed')
    return schemas, records


def values(items):
    require(all(v['value'] is not None for v in items), 'Nonfinite captured input; rational analysis unavailable')
    return [v['value'] for v in items]


def vectors(items, n, component_major=False):
    x = values(items)
    return [[x[n * k + i] if component_major else x[4 * i + k] for k in range(4)] for i in range(n)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def squared(a):
    return a[0] * a[0] - sum(x * x for x in a[1:])


def norm2(z):
    return sum(x * x for x in z)


def one(rows, tag, i=None, j=None):
    found = [r for r in rows if r['tag'] == tag and (i is None or r['i'] == i) and (j is None or r['j'] == j)]
    require(len(found) <= 1, 'Repeated boundary inside one reduction call: ' + tag)
    return found[0] if found else None


def analyze_call(rows, case, derivation, route):
    h = rows[0]
    suffix = 'QP' if h['mode'] == 4 else 'DP'
    identity = next((r for r in derivation['target_identities'] if r['reduction_identifier'] == h['id']), None)
    require(identity is not None and identity['rank'] == h['rank'], 'Trace identity differs from derived group')
    pair = (2, 4) if case == 'cms' else (1, 2)
    residual = one(rows, 'KT_RESIDUAL_' + suffix)
    matrix = one(rows, 'KT_MATRIX_' + suffix)
    native = one(rows, 'KT_TIRUST_' + suffix)
    status = one(rows, 'KT_STATUS_' + suffix)
    result = {'identity': {k: h[k] for k in ('section', 'point', 'serial', 'id', 'mode', 'rank', 'n', 'split_order')},
              'source_mass_expressions': identity['modes'][str(h['mode'])]['squared_mass_source_expressions'],
              'critical_pair': list(pair), 'interpretation': 'p1-p3-r' if case == 'cms' else 'p3+r',
              'native_status': status['data']['native_status'] if status else None,
              'boundary_observed': {'residual': residual is not None, 'matrix': matrix is not None,
                                    'native_inputs': native is not None, 'native_return': status is not None},
              'numerically_qualified': False}
    if status is None:
        result['missing_boundary_scope'] = 'Early return, interrupted call, or missing trace remains unresolved; no native success inferred.'
    try:
        external = {}
        for p in route['external_provenance']:
            row = one(rows, 'KT_EXTERNAL_' + suffix, p['leg'])
            if row:
                raw = values(row['data']['complex_momentum'])
                require(all(v == 0 for v in raw[1::2]), 'Complex external momenta: real physical routing not assigned')
                external[p['leg']] = [p['stored_helas_sign'] * v for v in raw[::2]]
        if len(external) == len(route['external_provenance']):
            coeffs = identity['conservation_coefficients']
            exact_r = [sum(coeffs[leg - 1] * p[k] for leg, p in external.items()) for k in range(4)]
            nominal = subtract(external[1], external[3]) if case == 'cms' else external[3]
            signed_r = exact_r if case == 'cms' else [-v for v in exact_r]
            routed = subtract(nominal, signed_r)
            result['exact_routing_on_captured_external_words'] = {'residual_r': exact_r, 'nominal_vector': nominal,
                'routed_vector': routed, 'nominal_square': squared(nominal), 'routed_square': squared(routed),
                'square_shift_due_to_r': squared(routed) - squared(nominal)}
            if matrix:
                shifts = vectors(matrix['data']['shifts_component_major'], h['n'], True)
                actual = subtract(shifts[pair[0] - 1], shifts[pair[1] - 1])
                result['matrix_shift_pair'] = {'vector': actual, 'exact_square': squared(actual),
                    'difference_from_symbolic_routed_vector': subtract(actual, routed)}
        if residual:
            corners = vectors(residual['data']['corners'], h['n'])
            exact_sum = [sum(p[k] for p in corners) for k in range(4)]
            observed = values(residual['data']['residual'])
            result['conservation'] = {'captured_zero_seeded_residual': observed, 'exact_sum_of_captured_corners': exact_sum,
                                      'summation_difference': subtract(observed, exact_sum)}
        if matrix:
            masses = values(matrix['data']['squared_masses'])
            dp = one(rows, 'KT_LOOP_DP')
            if dp:
                source_masses = values(dp['data']['squared_masses'])
                result['matrix_masses_equal_dp_squared_inputs_exactly'] = masses == source_masses
                result['mass_transport_scope'] = 'Exact numeric comparison of observed squared inputs; signed zeros remain in decoded words. No model mass/offshell assertion.'
            decisions = []
            for i in range(1, h['n'] + 1):
                for j in range(1, h['n'] + 1):
                    raw = one(rows, 'KT_RAW_' + suffix, i, j)
                    scale = one(rows, 'KT_SCALE_' + suffix, i, j)
                    final = one(rows, 'KT_FINAL_' + suffix, i, j)
                    if not raw or not scale:
                        continue
                    diff = values(raw['data']['diffsq'])
                    ref = values(scale['data']['reference_normalization'])[0]
                    threshold = values(scale['data']['osthres_dp'])[0]
                    require(threshold >= 0, 'Negative threshold outside proposal scope')
                    mi, mj = masses[2*(i-1):2*i], masses[2*(j-1):2*j]
                    ideal = {label: norm2(m) != 0 and norm2(subtract(diff, m)) < threshold**2 * norm2(m)
                             for label, m in (('MASSI', mi), ('MASSJ', mj))}
                    ideal['ZERO'] = ref != 0 and norm2(diff) < threshold**2 * ref**2
                    observed = {}
                    for branch in ('MASSI', 'MASSJ', 'ZERO'):
                        before, after = (one(rows, 'KT_' + branch + '_' + edge + '_' + suffix, i, j) for edge in ('BEFORE', 'AFTER'))
                        observed[branch] = ('taken' if before and after else 'incomplete' if before or after
                                            else 'not taken (complete pair boundaries)' if final else 'not observed')
                    decisions.append({'pair': [i, j], 'original_branches': observed, 'final_entry_observed': final is not None,
                        'exact_rational_predicates_not_fortran_replay': ideal, 'captured_diffsq': diff,
                        'reference_normalization': ref, 'osthres_dp': threshold,
                        'critical_pair': (i, j) == pair})
            result['threshold_decisions'] = decisions
        if native:
            inv = values(native['data']['invariants_column_major'])
            i, j = pair
            invariant = inv[(j-1)*h['n']+i-1]
            result['critical_native_invariant'] = invariant
            result['critical_native_transpose'] = inv[(i-1)*h['n']+j-1]
            comparison = {}
            raw_pair = one(rows, 'KT_RAW_' + suffix, i, j)
            if raw_pair:
                comparison['native_minus_captured_raw_real'] = invariant - values(raw_pair['data']['diffsq'])[0]
            if 'matrix_shift_pair' in result:
                comparison['native_minus_exact_shift_square'] = invariant - result['matrix_shift_pair']['exact_square']
            if 'exact_routing_on_captured_external_words' in result:
                comparison['native_minus_exact_nominal_square'] = invariant - result['exact_routing_on_captured_external_words']['nominal_square']
            result['critical_invariant_rational_differences'] = comparison
    except ValueError as error:
        result['rational_analysis_unavailable'] = str(error)
    return result


def analyze(raw, case, order, derivation, route):
    schemas, records = parse(raw, order)
    groups = {}
    keys = ('section', 'point', 'serial', 'id', 'mode', 'rank', 'n', 'split_order')
    for row in records:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return {'schema': 'private-kinematic-observation-analysis-v1', 'case': case,
            'abi_assumption_explicit': 'GNU x86_64 IEEE binary64/binary128; INTEGER*8 low-word-first; root must authenticate build',
            'qp_word_order': order, 'storage_headers': schemas, 'trace_sha256': digest(raw),
            'native_executions': 0, 'accuracy_qualified': False, 'production_changes': False,
            'scope': 'Exact decoding and rational algebra on captured words only; original branch records authoritative; no threshold, model, or accuracy correction.',
            'no_selected_calls_observed': not groups,
            'records': records, 'calls': [analyze_call(rows, case, derivation, route) for rows in groups.values()]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('cms', 'ew'))
    p.add_argument('--trace', type=Path)
    p.add_argument('--trace-sha256')
    p.add_argument('--qp-word-order', choices=('low-first',))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.trace is None:
        require(not any((args.case, args.trace_sha256, args.qp_word_order, args.output)), 'Analysis requires all explicit inputs')
        print(json.dumps({'status': 'source-only-proposal-no-trace-read', 'native_executions': 0,
                          'derived_index_sha256': DERIVED_SHA, 'accuracy_qualified': False}))
        return
    require(all((args.case, args.trace_sha256, args.qp_word_order, args.output)), 'Analysis requires all explicit inputs')
    require(not args.output.exists() and args.output.parent.is_dir(), 'Fresh output in an existing directory required')
    raw = args.trace.read_bytes()
    require(digest(raw) == args.trace_sha256, 'Explicit trace hash differs')
    derivation, route, binding = source_binding(args.case)
    result = analyze(raw, args.case, args.qp_word_order, derivation, route)
    result.update(derived_index_sha256=DERIVED_SHA, derived_files=binding)
    require(args.trace.read_bytes() == raw, 'Trace changed during analysis')
    with args.output.open('x') as output:
        json.dump(wire(result), output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
