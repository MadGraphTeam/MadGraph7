#!/usr/bin/env python3
"""Offline exact matrix/clamp/Gram analysis of the six bounded private runs."""
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import json
import re
import sys

from common import ROOT, HERE, DESIGN, OLD, ARMS, sha, rel, require, pins, validate_pins

sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD))
from decode_trace import load_trace
from analyze import analyze_call, one, reals, complexes, minkowski
from gram import gram_variant

INDEX = '97e5c6e16d52c166f3344921149ba0909cc844ab2a099f7c016100f8a95073cf'
ROUTING = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer/routing-modes.json'


def gram_record(call, point, order, pair_index, nrs):
    require(call['native_status'] == 1 and not call['stable'], 'Unexpected failed native status.')
    message = one(call['events'], 'BACKEND_ERROR')['error']['text']
    parsed = re.search(r'(ill-conditioned|singular) Gram matrix for (\[[0-9, ]+\])(?:\: conditioning ([0-9.e+-]+))?', message)
    require(parsed is not None, 'Unrecognized native error: ' + message)
    reason, members, captured = parsed[1], json.loads(parsed[2]), parsed[3]
    n = call['n']
    flat = reals(one(call['events'], 'DISTANCES'), order)
    supplied = {(i, j): flat[j*n+i] for i in range(n) for j in range(n)}
    qflat = reals(one(call['events'], 'SHIFTS'), order)
    q = [qflat[4*i:4*i+4] for i in range(n)]
    cartesian = {(i, j): minkowski([q[i][mu]-q[j][mu] for mu in range(4)])
                 for i in range(n) for j in range(n)}
    masses = complexes(one(call['events'], 'MASS2_ORDERED'), order)
    require(all(imag == 0 for _, imag in masses), 'QCD audit requires real masses.')
    variants = {label: gram_variant(distances, members, [real for real, _ in masses])
                for label, distances in (('supplied', supplied), ('exact_cartesian', cartesian))}
    error = None
    if reason == 'singular':
        require(variants['supplied']['determinant_is_zero'], 'Reported singularity differs from supplied exact determinant.')
    else:
        require(not variants['supplied']['determinant_is_zero'] and captured, 'Missing finite conditioning indicator.')
        with localcontext() as ctx:
            ctx.prec = 90
            error = abs(Decimal(variants['supplied']['condition'])-Decimal(captured))/abs(Decimal(captured))
        require(error < Decimal('1e-50'), 'Native condition indicator not reproduced.')
    subset = [pair_index[(9, point['seq'], call['ctmode'], call['group'], i+1, j+1)]
              for ai, i in enumerate(members) for j in members[ai+1:]]
    counterpart = nrs[(point['seq'], call['ctmode'], call['group'])]
    return {'seq': point['seq'], 'ctmode': call['ctmode'], 'group': call['group'], 'n': n,
            'rank': call['rank'], 'native_error': message, 'reported_kind': reason,
            'members_zero_based': members, 'captured_condition': captured,
            'relative_condition_reconstruction_error': str(error) if error is not None else None,
            'variants': variants, 'ninjars_same_group_status': counterpart['native_status'],
            'clamped_pairs_in_reported_subset': [
                {'pair_one_based': [p['i'], p['j']], 'classification': p['routing']['classification'],
                 'target': p['routing']['target_p1_minus_p4'], 'clamps': p['taken_clamps']}
                for p in subset if p['taken_clamps']]}


def main():
    require(sha(HERE/'source-index.json') == INDEX, 'Frozen controller changed.')
    protected = pins()
    before = dict(protected)
    routing = json.loads(ROUTING.read_text())
    out = HERE/'analysis1'/'matrices'
    require(not out.exists(), 'Do not replace an analysis attempt.')
    out.mkdir(parents=True)
    aggregate = {'schema': 1, 'scope': 'Offline matrix and clamp attribution; no new native run or physics acceptance.',
                 'arms': {}, 'bindings': {rel(p): sha(p) for p in
                   (Path(__file__), HERE/'source-index.json', OLD/'evidence-index.json',
                    OLD/'analyze.py', OLD/'gram.py', OLD/'decode_trace.py', ROUTING)},
                 'limits': ['A source-defined Gram condition indicator is a heuristic, not a certified error bound.',
                            'Exact Cartesian distances are diagnostic; no replacement integral is evaluated.',
                            'Checked host status alone does not establish physical truth or coordinate consistency.',
                            'The B/C and all nonmatrix input comparisons are recorded separately in handoff analysis.']}
    for arm in ARMS:
        calls, pairs, traces, executions = [], [], {}, {}
        directory = out/arm
        directory.mkdir()
        for backend in (9, 8):
            run = HERE/'attempt1'/arm/f'backend-{backend}'
            execution = json.loads((run/'execution.json').read_text())
            require(execution['complete'] and execution['ordinary_raw_words_exact'] and
                    execution['protected_inputs_unchanged'] and execution['runtime_sha256_unchanged'],
                    'Private protocol incomplete or changed.')
            for filename, key in (('trace.dat', 'trace_sha256'), ('observed.log', 'log_sha256')):
                require(sha(run/filename) == execution[key], 'Captured native output changed.')
            validate_pins(execution['protected_inputs_before'])
            validate_pins(execution['runtime_sha256_before'])
            executions[backend] = execution
            trace = traces[backend] = load_trace(run/'trace.dat', run/'observed.log')
            for filename in ('execution.json', 'observed.log', 'trace.dat'):
                aggregate['bindings'][rel(run/filename)] = sha(run/filename)
            for point in trace['points']:
                for call in point['calls']:
                    row, geometry = analyze_call(call, point['seq'], trace['byte_order'], routing)
                    calls.append(row)
                    pairs.extend(geometry)
        require(len(calls) == 464 and len(pairs) == 1908, 'Group/pair coverage changed.')
        require(all(c['all_original_numerator_components_finite'] for c in calls), 'Nonfinite numerator observed.')
        if arm.startswith('topology'):
            composite = [p for p in pairs if p['routing']['classification'] == 'composite']
            require(all(not p['taken_clamps'] and p['initial_s_real'] == p['final_s_real'] and
                        p['initial_s_imag'] == p['final_s_imag'] for p in composite),
                    'Composite channel was clamped or its computed S changed.')
        summary = {'by_backend_mode': {}, 'stress_rows': {str(b): executions[b]['stress_numeric_rows'] for b in (9, 8)},
                   'all_numerators_finite': True, 'ordinary_raw_and_printed_parity': True,
                   'supplied_cartesian_exact_equality_not_assumed': True}
        for backend in (9, 8):
            for mode in (4, 5):
                group = [c for c in calls if c['backend'] == backend and c['ctmode'] == mode]
                pp = [p for p in pairs if p['backend'] == backend and p['ctmode'] == mode]
                targets = [p for p in pp if p['routing']['target_p1_minus_p4']]
                require(len(group) == 116 and len(pp) == 477 and len(targets) == 19, 'Direction coverage changed.')
                summary['by_backend_mode'][f'{backend}:{mode}'] = {
                    'calls': len(group), 'native_statuses': dict(Counter(str(c['native_status']) for c in group)),
                    'refused_groups': [c['group'] for c in group if not c['stable']],
                    'target_channels': len(targets),
                    'target_nonzero_cartesian_supplied_zero': sum(p['cartesian_exact_nonzero'] and p['native_distance_zero'] for p in targets),
                    'target_nonzero_computed_supplied_zero': sum(p['computed_diffsq_nonzero'] and p['native_distance_zero'] for p in targets),
                    'clamped_pairs_by_class': dict(Counter(p['routing']['classification'] for p in pp if p['taken_clamps'])),
                    'composite_clamps': sum(bool(p['taken_clamps']) for p in pp if p['routing']['classification'] == 'composite')}
        pindex = {(p['backend'], p['seq'], p['ctmode'], p['group'], p['i'], p['j']): p for p in pairs}
        nrs = {(c['seq'], c['ctmode'], c['group']): c for c in calls if c['backend'] == 8}
        grams = [gram_record(call, point, traces[9]['byte_order'], pindex, nrs)
                 for point in traces[9]['points'] for call in point['calls'] if call['native_status'] != 0]
        summary['gram'] = {'refusals': len(grams), 'reported_kind': dict(Counter(g['reported_kind'] for g in grams)),
                           'exact_supplied_singular': sum(g['variants']['supplied']['determinant_is_zero'] for g in grams),
                           'exact_cartesian_singular': sum(g['variants']['exact_cartesian']['determinant_is_zero'] for g in grams),
                           'all_conditions_reconstructed': True}
        aggregate['arms'][arm] = summary
        for name, data in (('calls.json', calls), ('gram.json', grams), ('summary.json', summary)):
            (directory/name).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
        with (directory/'pairs.jsonl').open('x') as stream:
            for pair in pairs:
                stream.write(json.dumps(pair, sort_keys=True)+'\n')
    validate_pins(before)
    aggregate['protected_inputs_unchanged'] = True
    (out/'summary.json').write_text(json.dumps(aggregate, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete': True, 'arms': list(aggregate['arms']),
                      'gram': {arm: s['gram'] for arm, s in aggregate['arms'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
