#!/usr/bin/env python3
"""Exact supplied/Cartesian Gram audit for the actually reported native refusals."""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import argparse
import json
import re

from analyze import HERE, ROOT, sha, value, one, reals, complexes, minkowski
from decode_trace import load_trace


def determinant(matrix):
    a = [row[:] for row in matrix]
    result = Fraction(1)
    for col in range(len(a)):
        pivot = next((i for i in range(col, len(a)) if a[i][col]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            result = -result
        diagonal = a[col][col]
        result *= diagonal
        for row in range(col + 1, len(a)):
            factor = a[row][col] / diagonal
            for j in range(col + 1, len(a)):
                a[row][j] -= factor * a[col][j]
    return result


def gram_variant(distances, members, masses):
    origin, others = members[0], members[1:]
    matrix = [[distances[origin, i] + distances[origin, j] - distances[i, j] for j in others] for i in others]
    det = determinant(matrix)
    f = [distances[origin, i] - masses[i] + masses[origin] for i in others]
    scale = max([abs(masses[origin]), *map(abs, f), *(abs(v) for row in matrix for v in row)])
    with localcontext() as context:
        context.prec = 90
        decimal = lambda q: Decimal(q.numerator) / Decimal(q.denominator)
        hadamard = Decimal(1)
        for row in matrix:
            hadamard *= decimal(sum(v * v for v in row)).sqrt()
        relative = decimal(abs(det)) / hadamard if hadamard else Decimal(0)
        scaled = decimal(abs(det)) ** (Decimal(1) / len(others)) / decimal(scale) if scale else Decimal(0)
        condition = min(relative, scaled)
    return {'matrix': [[value(v) for v in row] for row in matrix],
            'determinant': value(det), 'determinant_is_zero': det == 0,
            'scale': value(scale), 'hadamard_relative_determinant': str(relative),
            'scaled_root_determinant': str(scaled), 'condition': str(condition)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    source_proof = ROOT / 'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/gram-consistency/summary.json'
    previous = json.loads(source_proof.read_text())
    source_paths = [ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs',
                    ROOT / '.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs']
    for path in source_paths:
        assert sha(path) == previous['sources'][str(path.relative_to(ROOT))], 'Gram formula source changed.'
    directory = HERE / 'attempt1/backend-9'
    execution = json.loads((directory / 'execution.json').read_text())
    assert execution['complete'] and execution['all_original_rows_exact'] and execution['protected_inputs_unchanged']
    assert sha(directory / 'trace.dat') == execution['trace_sha256']
    trace = load_trace(directory / 'trace.dat', directory / 'observed.log')
    pair_rows = [json.loads(line) for line in (HERE / 'analysis1/pairs.jsonl').read_text().splitlines()]
    pair_index = {(r['backend'], r['seq'], r['ctmode'], r['group'], r['i'], r['j']): r for r in pair_rows}
    nrs = {(r['seq'], r['ctmode'], r['group']): r for r in json.loads((HERE / 'analysis1/calls.json').read_text())
           if r['backend'] == 8}
    records = []
    for point in trace['points']:
        for call in point['calls']:
            if call['native_status'] == 0:
                continue
            assert call['native_status'] == 1 and not call['stable']
            message = one(call['events'], 'BACKEND_ERROR')['error']['text']
            parsed = re.search(r'(ill-conditioned|singular) Gram matrix for (\[[0-9, ]+\])(?:\: conditioning ([0-9.e+-]+))?', message)
            assert parsed, message
            reason, members, captured = parsed[1], json.loads(parsed[2]), parsed[3]
            assert len(members) in (3, 4)
            n = call['n']
            flat = reals(one(call['events'], 'DISTANCES'), trace['byte_order'])
            supplied = {(i, j): flat[j * n + i] for i in range(n) for j in range(n)}
            pflat = reals(one(call['events'], 'SHIFTS'), trace['byte_order'])
            q = [pflat[4 * i:4 * i + 4] for i in range(n)]
            derived = {(i, j): minkowski([q[i][mu] - q[j][mu] for mu in range(4)])
                       for i in range(n) for j in range(n)}
            masses_complex = complexes(one(call['events'], 'MASS2_ORDERED'), trace['byte_order'])
            assert all(imag == 0 for _, imag in masses_complex)
            masses = [real for real, _ in masses_complex]
            variants = {label: gram_variant(distances, members, masses)
                        for label, distances in (('supplied', supplied), ('exact_cartesian', derived))}
            relative = None
            if reason == 'singular':
                assert variants['supplied']['determinant_is_zero'], 'Reported singularity is not exact in supplied data.'
            else:
                assert not variants['supplied']['determinant_is_zero'] and captured
                with localcontext() as context:
                    context.prec = 90
                    relative = abs(Decimal(variants['supplied']['condition']) - Decimal(captured)) / abs(Decimal(captured))
                assert relative < Decimal('1e-50'), ('Condition formula differs from native report.', relative)
            changed = []
            for ai, i in enumerate(members):
                for j in members[ai + 1:]:
                    row = pair_index[(9, point['seq'], call['ctmode'], call['group'], i + 1, j + 1)]
                    if row['taken_clamps']:
                        changed.append({'pair_one_based': [i + 1, j + 1],
                                        'classification': row['routing']['classification'],
                                        'target_p1_minus_p4': row['routing']['target_p1_minus_p4'],
                                        'native_distance_zero': row['native_distance_zero'],
                                        'cartesian_exact_nonzero': row['cartesian_exact_nonzero'],
                                        'native_distance': row['native_distance'],
                                        'cartesian_exact_squared': row['cartesian_exact_squared'],
                                        'taken_clamps': row['taken_clamps']})
            other = nrs[(point['seq'], call['ctmode'], call['group'])]
            records.append({'seq': point['seq'], 'ctmode': call['ctmode'], 'group': call['group'],
                            'n': n, 'rank': call['rank'], 'native_error': message,
                            'reported_kind': reason, 'members_zero_based': members,
                            'variants': variants, 'captured_condition': captured,
                            'relative_condition_reconstruction_error': str(relative) if relative is not None else None,
                            'clamped_pairs_in_reported_subset': changed,
                            'target_composite_clamped_in_subset': any(r['target_p1_minus_p4'] for r in changed),
                            'ninjars_same_group_status': other['native_status'],
                            'ninjars_same_group_stable': other['stable']})
    summary = {'refusals': len(records), 'reported_kind': dict(Counter(r['reported_kind'] for r in records)),
               'supplied_singular_cartesian_nonsingular': sum(r['variants']['supplied']['determinant_is_zero'] and
                                                            not r['variants']['exact_cartesian']['determinant_is_zero'] for r in records),
               'all_reported_singular_subsets_contain_clamped_target': all(r['target_composite_clamped_in_subset']
                                                                         for r in records if r['reported_kind'] == 'singular'),
               'common_refusals': sum(r['ninjars_same_group_status'] != 0 for r in records),
               'common_refusals_containing_clamped_target': sum(r['ninjars_same_group_status'] != 0 and
                                                               r['target_composite_clamped_in_subset'] for r in records)}
    sources = [Path(__file__), HERE / 'analyze.py', HERE / 'decode_trace.py', source_proof,
               directory / 'trace.dat', directory / 'observed.log', directory / 'execution.json',
               HERE / 'analysis1/pairs.jsonl', HERE / 'analysis1/calls.json', *source_paths]
    result = {'schema': 1, 'summary': summary, 'records': records,
              'sources': {str(path.relative_to(ROOT)): sha(path) for path in sources},
              'scope': 'Exact arithmetic on already captured inputs; no native run or replacement input.',
              'definition': 'Zij=d(origin,i)+d(origin,j)-d(i,j). Supplied d is authoritative; Cartesian d derives exactly from fixed recorded shifts.',
              'limits': ['The condition indicator is the source-defined heuristic, not a certified error bound.',
                         'No scalar or tensor integral is evaluated with the Cartesian replacement matrix.',
                         'NinjaRS reports unstable status without detailed error text; a common group does not prove the same internal failure mechanism.',
                         'Changed external-shell residuals and clamped composite channels are retained separately.',
                         'This audit does not establish that a new threshold policy yields an accepted or accurate amplitude.']}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
