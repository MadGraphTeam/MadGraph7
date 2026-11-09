#!/usr/bin/env python3
"""Correct the soft-leg target using frozen input and symbolic routing."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DESIGN = HERE.parent / 'design'
POINTS = ROOT / 'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tokens = POINTS.read_text().split()
    assert tokens[0] == '2' and len(tokens) == 41
    point = [[Fraction(float(tokens[21 + 4 * leg + mu])) for mu in range(4)] for leg in range(5)]
    assert point[3][0] < Fraction(1, 1000) and point[4][0] > 400
    q = [point[0][mu] - point[3][mu] for mu in range(4)]
    q2 = q[0] ** 2 - sum(value ** 2 for value in q[1:])
    routing = json.loads((DESIGN / 'topology/routing.json').read_text())
    targets = []
    for group in routing['groups']:
        for pair in group['pairs']:
            if pair['literal_difference_mask'] in (9, 22):
                targets.append({'group': group['group'], 'n': group['n'], 'rank': group['rank'],
                                'source_call': group['call'], 'pair': pair})
    assert len(targets) == 19
    report = {
        'schema': 1, 'native_executed': False,
        'correction': 'The soft leg is 4. The tiny composite channel is p1-p4; p1-p5 is a hard-channel comparator.',
        'bindings': {str(path.relative_to(ROOT)): sha(path) for path in
                     (POINTS, DESIGN / 'evidence-index.json', DESIGN / 'topology/routing.json', HERE / 'derive.py')},
        'original_binary64_q14_squared': {'numerator': str(q2.numerator), 'denominator': str(q2.denominator)},
        'direct_wavefunction': 10,
        'explicit_W10_groups': [20, 21, 23, 24, 27, 28, 29],
        'target_pair_count': len(targets), 'target_pairs': targets,
        'limitation': 'Formal incidence and exact original-input arithmetic; no observed clamp or reducer status.',
    }
    (HERE / 'targets.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('Corrected target leg4: 19 source-proven channel/complement pairs; no native execution.')


if __name__ == '__main__':
    main()
