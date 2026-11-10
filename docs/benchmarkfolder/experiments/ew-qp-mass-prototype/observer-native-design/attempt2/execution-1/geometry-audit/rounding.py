#!/usr/bin/env python3
"""Append-only offline decomposition of frozen real QP kinematic inputs."""
import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_geometry_audit', HERE / 'audit.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert Path.cwd().resolve() == a.ROOT
    original = json.loads((HERE / 'analysis.json').read_text())
    assert a.sha(HERE / 'audit.py') == original['sources_and_inputs_sha256'][str((HERE / 'audit.py').relative_to(a.ROOT))]
    p = a.load_parser()
    # OSTHRES is a REAL*8 control; exact nearest-even value of the frozen card.
    threshold = a.rne(Fraction(Decimal('1e-8')), 53)
    card = a.EXECUTION / 'raw/candidate-9/MadLoop5_resources/MadLoopParams.dat'
    assert '#OSThres\n1.000000d-08\n' in card.read_text()
    source = a.DESIGN / 'sources/candidate/CT_interface.f'
    assert a.sha(source) == original['sources_and_inputs_sha256'][str(source.relative_to(a.ROOT))]
    assert 'S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)' in source.read_text()
    r = lambda x: a.rne(x, 113)
    records = []
    counters = Counter()
    paths = [Path(__file__), HERE / 'audit.py', HERE / 'analysis.json', card, source]
    for arm in a.ARMS:
        path = a.EXECUTION / 'raw' / (arm + '-9') / 'mass-observer.hex'
        paths.append(path)
        assert a.sha(path) == original['sources_and_inputs_sha256'][str(path.relative_to(a.ROOT))]
        capture = p.parse(path, arm, 9)
        for point, actual in capture['points'].items():
            for g in actual['groups']:
                n = g['key'][4]
                q = a.vectors(a.values(p, g, 'NATIVE_SHIFTS'))
                masses = a.values(p, g, 'NATIVE_MASSES')
                smat = a.matrix(a.values(p, g, 'KINEMATIC_S'), n)
                inv = a.matrix(a.values(p, g, 'NATIVE_INVARIANTS'), n)
                cart = a.distances(q)
                selected = g['key'][1] == 5 or g['native'][0]['status'] != 0
                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        delta = [r(x - y) for x, y in zip(q[i], q[j])]
                        squared = [r(x * x) for x in delta]
                        diff = squared[0]
                        for value in squared[1:]:
                            diff = r(diff - value)
                        s = r(r(diff - masses[i]) - masses[j])
                        target = diff
                        branches = []
                        if masses[i] and abs(r(r(diff - masses[i]) / masses[i])) < threshold:
                            s = -masses[j]
                            target = masses[i]
                            branches.append('mass_i')
                        if masses[j] and abs(r(r(diff - masses[j]) / masses[j])) < threshold:
                            s = -masses[i]
                            target = masses[j]
                            branches.append('mass_j')
                        ref = r(abs(q[i][0]) + abs(q[j][0]))
                        ref = r(r(ref / 2) ** 2)
                        if ref and abs(r(diff / ref)) < threshold:
                            s = -r(masses[i] + masses[j])
                            target = Fraction(0)
                            branches.append('zero')
                        recovered = r(r(s + masses[i]) + masses[j])
                        assert s == smat[i][j], (arm, point, g['key'], i, j, 'S')
                        assert recovered == inv[i][j], (arm, point, g['key'], i, j, 'invariant')
                        parts = [cart[i][j] - diff, diff - target, target - recovered]
                        assert sum(parts) == cart[i][j] - inv[i][j]
                        counters[arm + ':' + ('/'.join(branches) or 'default')] += 1
                        if selected:
                            pairs.append({'pair_zero_based': [i, j], 'source_emulated_snap_sequence': branches,
                                'cartesian_minus_sequential_qp_momentum_arithmetic': a.exact(parts[0]),
                                'sequential_qp_momentum_arithmetic_minus_post_snap_distance': a.exact(parts[1]),
                                'post_snap_distance_minus_recovered_native_invariant': a.exact(parts[2]),
                                'sum_cartesian_minus_native': a.exact(sum(parts))})
                if selected:
                    records.append({'arm': arm, 'point': point, 'group': g['key'][1],
                                    'native_refused': g['native'][0]['status'] != 0, 'pairs': pairs})
    result = {'scope': 'Offline exact nearest-even emulation on captured real inputs; no reducer execution or source edit.',
              'sources_sha256': {str(path.relative_to(a.ROOT)): a.sha(path) for path in paths},
              'all_800_group_S_and_native_invariant_arrays_reproduced': True,
              'exact_OSTHRES': a.exact(threshold), 'off_diagonal_unordered_pair_branch_counts': dict(counters),
              'definitions': ['Cartesian geometry evaluates the captured binary128 shifts with exact rational arithmetic.',
                  'Sequential QP arithmetic rounds each subtraction, square, and running Minkowski subtraction to113 bits.',
                  'Snap sequence follows the frozen Fortran source: mass i, then mass j, then massless; last assignment wins.',
                  'Residual is decomposed additively into momentum arithmetic, threshold assignment, and S mass subtraction/recovery.',
                  'Branch sequence is a source emulation verified against every captured S and native invariant, not an instrumented branch trace.'],
              'limitations': ['This emulation covers the actual real-mass fixture only; no complex-mass or exception-flag emulation.',
                  'Reproduction of matrices establishes their source arithmetic, not the validity of tensor or amplitude outputs.',
                  'All37 threshold native refusals remain; no failed tensor value is read.'], 'selected_group_records': records}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'all_800_groups_reproduced': True, 'selected_records': len(records),
                      'output_sha256': a.sha(args.output)}))


if __name__ == '__main__':
    main()
