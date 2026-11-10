#!/usr/bin/env python3
"""Portable integrity and retained exact-arithmetic checks; no native code."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def determinant(a):
    if not a:
        return F(1)
    return sum(((-1) ** j * x * determinant(
        [row[:j] + row[j + 1:] for row in a[1:]])
        for j, x in enumerate(a[0])), F(0))


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[7]
    index = json.loads((here / 'evidence-index.json').read_text())
    for name, expected in index['files'].items():
        path = here / name
        need(not path.is_symlink() and sha(path) == expected, 'Changed evidence: ' + name)
    for name, expected in index['retained_source_and_input_bindings'].items():
        need(sha(root / name) == expected, 'Changed retained binding: ' + name)
    for name, expected in index['parent_indices'].items():
        path = root / name
        need(sha(path) == expected, 'Changed parent index: ' + name)
        for child, digest in json.loads(path.read_text())['files'].items():
            need(sha(path.parent / child) == digest, 'Changed parent evidence: ' + child)
    geometry = json.loads((here / 'analysis.json').read_text())
    rounding = json.loads((here / 'rounding.json').read_text())
    tensors = json.loads((here / 'rank-one.json').read_text())
    need(len(geometry['geometry_records']) == 800, 'Geometry coverage changed')
    matrices = pairs = decompositions = identities = 0
    for record in geometry['geometry_records']:
        for pair in record['pairs']:
            need(F(pair['cartesian_invariant']['fraction']) - F(pair['native_invariant']['fraction'])
                 == F(pair['cartesian_minus_native']['fraction']), 'Invariant difference')
            pairs += 1
        for group in record['grams'].values():
            for kind in ('cartesian', 'supplied'):
                matrix = [[F(x) for x in row] for row in group[kind]['gram_exact']]
                need(determinant(matrix) == F(group[kind]['determinant']['fraction']), 'Gram determinant')
                matrices += 1
            difference = [[F(x) - F(y) for x, y in zip(a, b, strict=True)]
                          for a, b in zip(group['cartesian']['gram_exact'],
                                          group['supplied']['gram_exact'], strict=True)]
            need(difference == [[F(x) for x in row] for row in group['cartesian_minus_supplied_gram_exact']],
                 'Gram matrix difference')
    terms = ('cartesian_minus_sequential_qp_momentum_arithmetic',
             'sequential_qp_momentum_arithmetic_minus_post_snap_distance',
             'post_snap_distance_minus_recovered_native_invariant')
    need(rounding['all_800_group_S_and_native_invariant_arrays_reproduced'] is True, 'Rounding coverage')
    for group in rounding['selected_group_records']:
        for pair in group['pairs']:
            need(sum((F(pair[k]['fraction']) for k in terms), F(0))
                 == F(pair['sum_cartesian_minus_native']['fraction']), 'Rounding decomposition')
            decompositions += 1
    need(tensors['native_execution_performed'] is False and tensors['physical_accuracy_qualified'] is False,
         'Scope changed')
    need(len(tensors['cohorts']) == 4, 'Tensor cohort count')
    for cohort in tensors['cohorts']:
        threshold = cohort['point'] == 2
        need(len(cohort['native_refused_groups']) == (37 if threshold else 0), 'Refusal count')
        need(cohort['complete_parent_equations'] == (19 if threshold else 115), 'Equation count')
        need(len(cohort['records']) == (17 if threshold else 33), 'Parent count')
        refused = set(cohort['native_refused_groups'])
        for parent in cohort['records']:
            need(parent['native_status'] == 0 and parent['group'] not in refused, 'Invalid RAW use')
            for equation in parent['equations']:
                for slot in equation['slots'].values():
                    need([F(a) - F(b) for a, b in zip(slot['lhs']['exact'], slot['rhs']['exact'], strict=True)]
                         == [F(x) for x in slot['residual']['exact']], 'Tensor residual')
                    identities += 1
    need(identities == 1340, 'Identity coverage')
    print(json.dumps({'verified_files': len(index['files']), 'geometry_records': 800,
        'exact_determinants': matrices, 'invariant_differences': pairs,
        'rounding_decompositions': decompositions, 'exact_complex_residuals': identities,
        'scope': 'Portable retained-evidence integrity and algebra; no reducer or generator replay',
        'checked_qp_qualified': False, 'threshold_tirust_aggregate_complete': False}, sort_keys=True))


if __name__ == '__main__':
    main()
