#!/usr/bin/env python3
"""Validate and summarize the private generated-code geometry probe."""
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[4]
D = Decimal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tolerance=D('1e-25')):
    assert abs(a-b) <= tolerance, (a, b)


def analyze():
    prepared = json.loads((REPORT/'prepared.json').read_text())
    executed = json.loads((REPORT/'execution.json').read_text())
    original = ROOT/prepared['origin']
    work = ROOT/prepared['work']
    historical_path = ROOT/'docs/benchmarkfolder/experiments/physical-stress/diagnostics/ew_n3_epem_ttxa/candidate-9-qp-init-dp/diagnostic-compilation.json'
    historical = json.loads(historical_path.read_text())
    links = json.loads((REPORT/'link-inputs-post-execution.json').read_text())
    assert prepared['original_compilation_sha256'] == digest(historical_path)
    assert links['historical_compilation_sha256'] == digest(historical_path)
    assert links['archives_matching_historical'] == historical['link_inputs_after']['archives']
    assert executed['returncode'] == 0
    assert executed['origin_numerical_inputs_unchanged']
    assert executed['changed_original_objects'] == []
    assert executed['added_private_objects'] == ['SubProcesses/P0_epem_ttxa/geometry_probe.o']
    assert executed['driver_sha256'] == prepared['driver_sha256'] == digest(REPORT/'probe.f')
    assert prepared['run_script_sha256'] == digest(REPORT/'run.py')
    assert executed['raw_log_sha256'] == digest(REPORT/'raw.log')
    assert executed['build_log_sha256'] == digest(REPORT/'build.log')
    assert prepared['overlay_sha256'] == digest(REPORT/'geometry.mk')
    assert executed['point_sha256'] == prepared['point_sha256'] == digest(REPORT/'selected.dat')
    assert executed['private_dso_sha256'] == prepared['private_dso_sha256']
    assert prepared['private_dso_sha256'] == digest(work/'private-libraries/libmadloop_tirust.so')
    for name, sha in prepared['card_sha256'].items():
        assert executed['card_sha256'][name] == sha == digest(REPORT/name)
    for name, sha in prepared['objects_archives_before'].items():
        assert executed['objects_archives_after'][name] == sha
        assert digest(original/name) == sha == digest(work/name)
    for name, sha in prepared['original_source_hashes'].items():
        assert digest(original/name) == sha == digest(work/name)
    records = {}
    recognized = {'IMPROVEPSPOINT', 'INITIAL_OSTHRES', 'DP_TOP_PROMOTED', 'MP_TOP',
                  'EXACT_DECIMAL_TOP', 'DP_SQUARE_PROMOTED', 'MP_SQUARE',
                  'MP_MINUS_DP_SQUARE', 'DP_AEW', 'MP_AEW', 'DP_GAL_PROMOTED',
                  'MP_GAL', 'SCALE', 'RAW', 'MP_PS', 'MP_P', 'SHELL',
                  'CONSERVATION', 'PAIR', 'SHIFT', 'MASS2', 'MATRIX'}
    for line in (REPORT/'raw.log').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in recognized:
            records.setdefault(fields[0], []).append(fields[1:])
    counts = {k: len(v) for k, v in records.items()}
    for key, count in {'SCALE': 2, 'RAW': 10, 'MP_PS': 10, 'MP_P': 10,
                       'SHELL': 20, 'CONSERVATION': 4, 'PAIR': 4,
                       'SHIFT': 20, 'MASS2': 60, 'MATRIX': 900}.items():
        assert counts[key] == count, (key, counts.get(key))
    assert records['IMPROVEPSPOINT'] == [['2']]
    close(D(records['INITIAL_OSTHRES'][0][0]), D.from_float(1e-8), D('1e-42'))
    assert records['MP_PS'] == records['MP_P']
    scalars = {key: [D(v) for v in rows[0]] for key, rows in records.items()
               if key not in {'SCALE', 'RAW', 'MP_PS', 'MP_P', 'SHELL',
                              'CONSERVATION', 'PAIR', 'SHIFT', 'MASS2', 'MATRIX'}}
    close(scalars['MP_TOP'][0], D.from_float(173.3), D('1e-32'))
    assert scalars['MP_TOP'] == scalars['DP_TOP_PROMOTED']
    close(scalars['MP_MINUS_DP_SQUARE'][0],
          scalars['MP_SQUARE'][0]-scalars['DP_SQUARE_PROMOTED'][0], D('1e-29'))
    raw = {(int(r[0]), int(r[1])): list(map(D, r[2:])) for r in records['RAW']}
    repaired = {(int(r[0]), int(r[1])): list(map(D, r[2:])) for r in records['MP_P']}
    exact_lines = (REPORT/'selected.dat').read_text().splitlines()
    assert int(exact_lines[0]) == 2
    for k in (1, 2):
        for i in range(1, 6):
            for observed, value in zip(raw[k, i], exact_lines[(k-1)*5+i].split()):
                close(observed, D.from_float(float(value)), D('1e-31'))
    pairs = {(int(r[0]), int(r[1])): list(map(D, r[2:])) for r in records['PAIR']}
    shells = {(int(r[0]), int(r[1]), int(r[2])): D(r[3]) for r in records['SHELL']}
    conservation = {(int(r[0]), int(r[1])): list(map(D, r[2:]))
                    for r in records['CONSERVATION']}
    matrices = {tuple(map(int, r[:6])): list(map(D, r[6:])) for r in records['MATRIX']}
    assert len(matrices) == 900
    for (k, stage, pattern, h, i, j), row in matrices.items():
        expected = D.from_float({1: 1e-8, 2: 1e-14, 3: 1e-24}[h])
        close(row[0], expected, D('1e-40'))
        assert row[4] == 0  # all internal masses and invariants are real
        for a, b in zip(row, matrices[k, stage, pattern, h, j, i]):
            close(a, b)
    changes = []
    for k in (1, 2):
        for stage in (0, 1):
            for pattern in range(3):
                for i in range(1, 6):
                    for j in range(i+1, 6):
                        rows = [matrices[k, stage, pattern, h, i, j] for h in (1, 2, 3)]
                        vals = [r[2] for r in rows]
                        if max(vals)-min(vals) > D('1e-24'):
                            changes.append({'point': k, 'stage': stage, 'mass_pattern': pattern,
                                            'i': i, 'j': j, 'raw_invariant': rows[0][1],
                                            'adjusted_by_threshold': vals})
    points = []
    for k in (1, 2):
        scale = list(map(D, records['SCALE'][k-1][1:]))
        scale_difference = scale[2]-scale[1]
        shellmax = {str(stage): max(abs(shells[k, stage, i]) for i in range(1, 6))
                    for stage in (0, 1)}
        consmax = {str(stage): max(map(abs, conservation[k, stage])) for stage in (0, 1)}
        close(shellmax['1'], D(0), D('1e-25'))
        close(consmax['1'], D(0), D('1e-25'))
        points.append({
            'point': k, 'max_shell_residual': shellmax, 'max_conservation_residual': consmax,
            'raw_pair_s_delta_beta_excess': pairs[k, 0],
            'repaired_pair_s_delta_beta_excess': pairs[k, 1],
            'pair_delta_relative_change': pairs[k, 1][1]/pairs[k, 0][1]-1,
            'beta_relative_change': pairs[k, 1][2]/pairs[k, 0][2]-1,
            'mass_square_rounding_relative_to_repaired_pair_delta':
                4*scalars['MP_MINUS_DP_SQUARE'][0]/pairs[k, 1][1],
            'max_momentum_component_change': max(abs(repaired[k, i][j]-raw[k, i][j])
                                                  for i in range(1, 6) for j in range(4)),
            'scale_mu_dp_square_qp_square': scale,
            'scale_square_qp_minus_dp': scale_difference,
            'scale_square_relative_difference': scale_difference/scale[2],
            'scale_square_difference_over_repaired_pair_delta': scale_difference/pairs[k, 1][1],
        })
    return {
        'scope': 'Geometry probe only; no reducer, amplitude, accuracy or timing acceptance.',
        'canonical_shift_order': ['-p1', '-p2', 'p3', 'p4', 'p5'],
        'mass_patterns': {'0': [0]*5, '1': ['top']*5, '2': [0, 'top', 0, 'top', 0]},
        'scalar_inputs': scalars, 'point_results': points, 'matrix_changes': changes,
        'record_counts': counts,
        'validation': 'PASS: exact inputs, historical objects/sources, private links, numerical invariants.',
        'evidence_sha256': {p: digest(REPORT/p) for p in
                            ('probe.f', 'run.py', 'analyze.py', 'prepared.json', 'execution.json',
                             'build-status.json', 'build.log', 'raw.log', 'selected.dat',
                             'param_card.dat', 'MadLoopParams.dat', 'geometry.mk',
                             'link-inputs-post-execution.json', 'README.md',
                             'source-audit.md', 'source-audit-hashes.json')},
        'limitations': ['Canonical five-leg shifts are representative, not captured per-diagram shifts.',
                        'Mass-pattern probes do not identify physical scalar topologies.',
                        'All QP model parameters here retain the generated DP-reader promotion semantics.']}


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 100
        result = analyze()
    (REPORT/'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=str)+'\n')
    print(result['validation'])
    for point in result['point_results']:
        print('point', point['point'], 'repaired beta', point['repaired_pair_s_delta_beta_excess'][2],
              'relative beta shift', point['beta_relative_change'])
