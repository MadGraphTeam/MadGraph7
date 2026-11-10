#!/usr/bin/env python3
"""Exact offline analysis of fully validated observer streams; no native calls."""
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json

from decode_trace import load_trace, decode_payload, finite_fraction

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
DESIGN = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(q):
    q = Fraction(q)
    with localcontext() as ctx:
        ctx.prec = 65
        decimal = str(Decimal(q.numerator) / Decimal(q.denominator))
    return {'numerator': str(q.numerator), 'denominator': str(q.denominator), 'decimal': decimal}


def reals(event, order):
    return [finite_fraction(v) for v in decode_payload(event, order)]


def complexes(event, order):
    return [(finite_fraction(v['real']), finite_fraction(v['imag'])) for v in decode_payload(event, order)]


def minkowski(v):
    return v[0] * v[0] - sum(x * x for x in v[1:])


def one(events, tag):
    selected = [e for e in events if e['tag'] == tag]
    assert len(selected) == 1, (tag, len(selected))
    return selected[0]


def analyze_call(call, seq, order, routing):
    route = next(r for r in routing['groups_by_mode'][str(call['ctmode'])] if r['group'] == call['group'])
    assert (route['n'], route['rank']) == (call['n'], call['rank'])
    events = call['events']
    record = {key: call[key] for key in ('call', 'backend', 'ctmode', 'group', 'sqso', 'n', 'rank',
                                        'native_status', 'stable', 'fpe', 'fpe_entry', 'outcome')}
    record['seq'] = seq
    record['errors'] = [e['error']['text'] for e in events if e['tag'] == 'BACKEND_ERROR']
    record['numerical_summaries'] = call['numerical_summaries']
    record['all_original_numerator_components_finite'] = one(events, 'NUM_ORIGINAL')['numerical_summary']['all_finite']
    record['target_group'] = any(p['target_p1_minus_p4'] for p in route['pairs'])
    record['pair_count'] = 0
    if not any(e['tag'] == 'SHIFTS' for e in events):
        record['geometry_available'] = False
        return record, []
    record['geometry_available'] = True
    n = call['n']
    qflat = reals(one(events, 'SHIFTS'), order)
    q = [qflat[4 * i:4 * i + 4] for i in range(n)]
    distances = reals(one(events, 'DISTANCES'), order)
    sflat = complexes(one(events, 'S_COMPLEX'), order)
    masses = complexes(one(events, 'MASS2_ORDERED'), order)
    pflat = reals(one(events, 'CONTEXT_MP_P'), order)
    p = [pflat[4 * i:4 * i + 4] for i in range(5)]
    literal = [[sum(Fraction(c) * p[k][mu] for k, c in enumerate(coeffs)) for mu in range(4)]
               for coeffs in route['shift_coefficients_physical_p']]
    record['maximum_exact_routing_component_residual'] = value(max(abs(q[i][mu] - literal[i][mu])
                                                                   for i in range(n) for mu in range(4)))
    record['native_scale'] = value(reals(one(events, 'NATIVE_SCALE'), order)[0])
    record['literal_q1_cartesian'] = [value(x) for x in q[0]]
    grouped = defaultdict(list)
    for event in events:
        if event['tag'].startswith('PAIR_'):
            grouped[(event['i'], event['j'])].append(event)
    rows = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            pair = grouped[(i, j)]
            mapped = next(r for r in route['pairs'] if r['propagators'] == [i, j])
            diffsq = complexes(one(pair, 'PAIR_DIFFSQ'), order)[0]
            initial = complexes(one(pair, 'PAIR_S_INITIAL'), order)[0]
            final = complexes(one(pair, 'PAIR_S_FINAL'), order)[0]
            reference = reals(one(pair, 'PAIR_E_REFERENCE'), order)[0]
            cartesian = minkowski([q[i - 1][mu] - q[j - 1][mu] for mu in range(4)])
            physical = [sum(Fraction(c) * p[k][mu] for k, c in enumerate(mapped['literal_coefficients_physical_p']))
                        for mu in range(4)]
            physical_squared = minkowski(physical)
            distance = distances[(j - 1) * n + i - 1]
            assert final == sflat[(j - 1) * n + i - 1]
            assert distance == distances[(i - 1) * n + j - 1], 'Supplied matrix is asymmetric.'
            clamps = []
            for index, event in enumerate(pair):
                if event['tag'] == 'PAIR_CLAMP':
                    after = pair[index + 1]
                    assert after['tag'] == 'PAIR_S_CLAMPED'
                    z = complexes(after, order)[0]
                    clamps.append({'branch': event['value'],
                                   'cause': {1: 'mass_i_shell', 2: 'mass_j_shell', 3: 'massless_energy_threshold'}[event['value']],
                                   's_real': value(z[0]), 's_imag': value(z[1])})
            row = {key: record[key] for key in ('seq', 'call', 'backend', 'ctmode', 'group', 'n', 'rank',
                                               'native_status', 'stable', 'outcome')}
            row.update({'i': i, 'j': j, 'routing': mapped,
                        'diffsq_real': value(diffsq[0]), 'diffsq_imag': value(diffsq[1]),
                        'initial_s_real': value(initial[0]), 'initial_s_imag': value(initial[1]),
                        'final_s_real': value(final[0]), 'final_s_imag': value(final[1]),
                        'energy_reference': value(reference), 'taken_clamps': clamps,
                        'native_distance': value(distance), 'cartesian_exact_squared': value(cartesian),
                        'literal_physical_channel_squared': value(physical_squared),
                        'diffsq_rounding_difference': value(diffsq[0] - cartesian),
                        'supplied_vs_cartesian_difference': value(distance - cartesian),
                        'supplied_vs_computed_diffsq_difference': value(distance - diffsq[0]),
                        'native_distance_zero': distance == 0,
                        'cartesian_exact_nonzero': cartesian != 0,
                        'computed_diffsq_nonzero': diffsq != (0, 0),
                        'clamp_changed_s': final != initial,
                        'exact_abs_diffsq_over_energy_reference': value(abs(diffsq[0] / reference)) if reference else None,
                        'mass_i': {'real': value(masses[i - 1][0]), 'imag': value(masses[i - 1][1])},
                        'mass_j': {'real': value(masses[j - 1][0]), 'imag': value(masses[j - 1][1])}})
            rows.append(row)
    record['pair_count'] = len(rows)
    record['clamped_pairs'] = sum(bool(row['taken_clamps']) for row in rows)
    record['nonzero_composite_pairs_supplied_as_zero'] = [
        [r['i'], r['j']] for r in rows if r['routing']['classification'] == 'composite'
        and r['cartesian_exact_nonzero'] and r['native_distance_zero']]
    record['target_pairs'] = [[r['i'], r['j']] for r in rows if r['routing']['target_p1_minus_p4']]
    return record, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not replace an analysis attempt.'
    args.output.mkdir(parents=True)
    routing = json.loads((DESIGN / 'routing-modes.json').read_text())
    calls, pairs, traces, bindings = [], [], {}, {}
    for backend in (9, 8):
        directory = HERE / 'attempt1' / f'backend-{backend}'
        execution = json.loads((directory / 'execution.json').read_text())
        assert execution['complete'] and execution['all_original_rows_exact'] and execution['protected_inputs_unchanged']
        assert sha(directory / 'trace.dat') == execution['trace_sha256']
        assert sha(directory / 'observed.log') == execution['log_sha256']
        trace = load_trace(directory / 'trace.dat', directory / 'observed.log')
        traces[backend] = trace
        for file in ('trace.dat', 'observed.log', 'execution.json'):
            path = directory / file
            bindings[str(path.relative_to(ROOT))] = sha(path)
        for point in trace['points']:
            for call in point['calls']:
                row, geometry = analyze_call(call, point['seq'], trace['byte_order'], routing)
                calls.append(row)
                pairs.extend(geometry)
    comparisons = []
    indexes = {}
    for backend, trace in traces.items():
        index = {}
        for point in trace['points']:
            seen = Counter()
            for call in point['calls']:
                stem = (point['seq'], call['ctmode'], call['group'], call['sqso'])
                seen[stem] += 1
                index[(*stem, seen[stem])] = call
        indexes[backend] = index
    assert indexes[8].keys() == indexes[9].keys(), 'Backend visit coverage differs; explicit adjudication required.'
    for key in indexes[9]:
        left, right = indexes[9][key], indexes[8][key]
        same = {}
        for tag in ('EDGE_INITIAL', 'MASS2_DP', 'NUM_ORIGINAL', 'CONTEXT_MP_PS', 'CONTEXT_MP_P',
                    'EDGE_ORDERED', 'MASS2_ORDERED', 'SHIFTS', 'S_COMPLEX', 'DISTANCES', 'NATIVE_SCALE'):
            l = [e['words'] for e in left['events'] if e['tag'] == tag]
            r = [e['words'] for e in right['events'] if e['tag'] == tag]
            same[tag] = l == r
        comparisons.append({'seq': key[0], 'ctmode': key[1], 'group': key[2], 'sqso': key[3], 'visit': key[4],
                            'exact_payload_equal': same,
                            'tirust_status': left['native_status'], 'ninjars_status': right['native_status'],
                            'tirust_stable': left['stable'], 'ninjars_stable': right['stable']})
    summary = {'schema': 1, 'scope': 'Offline exact analysis after full trace validation and exact unobserved-host parity.',
               'native_executed_by_analyzer': False, 'bindings': bindings,
               'trace_summaries': {str(k): v['summary'] for k, v in traces.items()},
               'calls': len(calls), 'unordered_pair_rows': len(pairs),
               'all_original_numerators_finite': all(c['all_original_numerator_components_finite'] for c in calls),
               'all_cross_backend_input_payloads_equal': all(all(c['exact_payload_equal'].values()) for c in comparisons),
               'cross_backend_payload_differences': [c for c in comparisons if not all(c['exact_payload_equal'].values())],
               'by_backend_mode': {},
               'limits': ['Clamps and S/Cartesian inconsistencies are observed associations, not a repaired-input acceptance proof.',
                          'The physical host remains rejected420; no threshold or working-precision setting changed.',
                          'Exact Cartesian arithmetic treats recorded binary128 coordinates as fixed data.',
                          'Native status is the public Fortran wrapper status and may include transport errors.',
                          'Ordinary control points used the unchanged automatic DP path and contain no QP group calls.']}
    for backend in (9, 8):
        for mode in (4, 5):
            selected = [c for c in calls if c['backend'] == backend and c['ctmode'] == mode]
            selected_pairs = [p for p in pairs if p['backend'] == backend and p['ctmode'] == mode]
            targets = [p for p in selected_pairs if p['routing']['target_p1_minus_p4']]
            assert len(targets) == 19
            summary['by_backend_mode'][f'{backend}:{mode}'] = {
                'calls': len(selected), 'statuses': dict(Counter(str(c['native_status']) for c in selected)),
                'stable': sum(c['stable'] for c in selected),
                'refused_groups': [c['group'] for c in selected if not c['stable']],
                'target_count': len(targets),
                'targets_with_nonzero_cartesian_supplied_zero': sum(p['cartesian_exact_nonzero'] and p['native_distance_zero'] for p in targets),
                'targets_with_computed_nonzero_supplied_zero': sum(p['computed_diffsq_nonzero'] and p['native_distance_zero'] for p in targets),
                'all_targets_took_massless_clamp': all(any(x['branch'] == 3 for x in p['taken_clamps']) for p in targets),
                'clamped_pairs_by_literal_class': dict(Counter(p['routing']['classification'] for p in selected_pairs if p['taken_clamps'])),
            }
    for path in (HERE / 'analyze.py', HERE / 'decode_trace.py', DESIGN / 'routing-modes.json', DESIGN / 'evidence-index.json'):
        summary['bindings'][str(path.relative_to(ROOT))] = sha(path)
    for name, data in (('summary.json', summary), ('calls.json', calls), ('cross-backend.json', comparisons)):
        (args.output / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    with (args.output / 'pairs.jsonl').open('x') as stream:
        for row in pairs:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps({key: summary[key] for key in ('calls', 'unordered_pair_rows', 'all_original_numerators_finite',
                                                  'all_cross_backend_input_payloads_equal', 'by_backend_mode')}, sort_keys=True))


if __name__ == '__main__':
    main()
