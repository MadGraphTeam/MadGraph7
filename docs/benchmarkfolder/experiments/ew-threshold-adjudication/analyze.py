#!/usr/bin/env python3
"""Reduce the logging-only trace; never promote unchecked outputs to acceptance."""
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI = Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    execution = json.loads((HERE / 'execution.json').read_text())
    if not execution['complete'] or not execution['final_values_identical_to_historical']:
        raise ValueError('The trace did not reproduce the historical result')
    prepared = json.loads((HERE / 'prepared.json').read_text())
    compiled = json.loads((HERE / 'compilation.json').read_text())
    historical = HERE.parent / 'physical-stress/diagnostics/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
    values = HERE / 'values.dat'
    if not (digest(values) == execution['values_sha256'] == execution['historical_values_sha256']
            == digest(historical / 'raw/values.dat') == digest(HERE / 'raw/values.dat')):
        raise ValueError('Raw result binding mismatch')
    checks = {'trace.f': prepared['trace_sha256'], 'run.py': prepared['run_script_sha256'],
        'selected.dat': prepared['point_sha256'], 'trace.mk': prepared['overlay_sha256'],
        'sources/tirust_benchmark_worker.py': prepared['controller_sha256'],
        'build.log': compiled['build_log_sha256'], 'ldd.txt': execution['ldd_sha256']}
    checks.update(prepared['card_sha256'])
    checks.update({'sources/' + p: h for p, h in prepared['patched_source_hashes'].items()})
    for path, expected in checks.items():
        if digest(HERE / path) != expected:
            raise ValueError('Evidence changed: ' + path)
    if digest(historical / 'diagnostic-compilation.json') != prepared['original_compilation_sha256']:
        raise ValueError('Historical compilation binding changed')
    if execution['private_dso_sha256'] != prepared['private_dso_sha256']:
        raise ValueError('Runtime snapshot binding mismatch')
    points, point, call = {}, None, None
    for line in (HERE / 'ew-tensor-trace.dat').read_text().splitlines():
        fields = line.split()
        tag = fields[0]
        if tag == 'BEGIN':
            point = {'index': int(fields[1]), 'calls': [], 'groups': {}, 'P': [], 'MP_PS': [], 'MP_P': [], 'answers': {}}
            points[point['index']] = point
        elif tag == 'CALL':
            p, seq, group, so, mode, n, rank, status = map(int, fields[1:])
            if p != point['index'] or seq != len(point['calls']) + 1:
                raise ValueError('Disordered calls')
            call = dict(sequence=seq, group=group, split_order=so, mode=mode, n=n, rank=rank, status=status)
            point['calls'].append(call)
            if group not in point['groups']:
                point['groups'][group] = dict(call, shifts=[], masses=[], invariants=[], raw=[])
        elif tag == 'ERROR':
            call['error'] = line.split('ERROR ', 1)[1]
            point['groups'][call['group']]['error'] = call['error']
        elif tag == 'SCALE':
            point['groups'][call['group']]['scale'] = fields[1]
        elif tag in ('SHIFT', 'MASS', 'S', 'RAW'):
            key = {'SHIFT': 'shifts', 'MASS': 'masses', 'S': 'invariants', 'RAW': 'raw'}[tag]
            point['groups'][call['group']][key].append(fields[1:])
        elif tag == 'END':
            p, count, status = map(int, fields[1:4])
            if p != point['index'] or count != len(point['calls']):
                raise ValueError('Inconsistent end marker')
            point.update(host_status=status, fpe_dp=fields[4] == 'T', fpe_qp=fields[5] == 'T')
        elif tag in ('P', 'MP_PS', 'MP_P'):
            point[tag].append(fields[1:])
        elif tag == 'ANS':
            point['answers'][int(fields[1])] = fields[2:]
        elif tag in ('GAL', 'MP_GAL', 'MT'):
            point[tag.lower()] = fields[1:]
        else:
            raise ValueError('Unknown trace record: ' + tag)
    if set(points) != {1, 2}:
        raise ValueError('Missing point')
    summary = {'scope': 'Unchecked fixed-QP diagnostic; no numerical or performance acceptance.', 'points': []}
    with localcontext() as ctx:
        ctx.prec = 80
        for index, point in points.items():
            counts = Counter(call['status'] for call in point['calls'])
            failures = sorted({call['group'] for call in point['calls'] if call['status'] != 0})
            errors = Counter(call.get('error', '') for call in point['calls'] if call['status'] != 0)
            born = Decimal(point['answers'][0][0])
            gal_re, gal_im = map(Decimal, point['gal'])
            # Massless e-/e+ contribute Q_i^2=1 each; massive top legs and
            # external photon do not contribute a double infrared pole.
            alpha = (gal_re**2 + gal_im**2) / (4 * PI)
            prediction = -2 * born * alpha / (2 * PI)
            actual = Decimal(point['answers'][0][3])
            rel = abs(actual - prediction) / abs(prediction)
            summary['points'].append({
                'index': index, 'host_status': point['host_status'],
                'fpe_dp': point['fpe_dp'], 'fpe_qp': point['fpe_qp'],
                'calls': len(point['calls']), 'unique_groups': len(point['groups']),
                'call_coverage': 'Only reductions reaching the native TIRUST_TENSOR_EVALUATE call are counted; pre-call adapter failures are outside this hook.',
                'native_status_counts': dict(counts), 'failed_group_ids': failures,
                'native_error_counts': dict(errors),
                'omitted_groups_from_existing_adapter_failure_return': failures,
                'double_pole_prediction_from_returned_born': str(prediction),
                'double_pole_returned': str(actual), 'double_pole_relative_residual': str(rel),
                'alpha_from_DP_GAL': str(alpha),
                'prediction_limit': 'Uses returned DP Born and DP model coupling, sufficient for gross pole consistency, not an independent higher-precision Born certificate.',
                'answers': point['answers'],
            })
    summary['source_policy'] = {
        'fks_double_pole': 'Template/NLO/SubProcesses/fks_singular.f: massless external charged-leg QED Casimir sum',
        'adapter_failure': 'MP_TIRUST_TENSOR_DIAGNOSTIC returns STABLE false on nonzero native status; MP_TIRUST_LOOP sets FPE_IN_QP_REDUCTION and returns RES=0.',
        'fixed_mode_status': 'CTModeRun=4 skips automatic FPE accuracy handling; status120 alone does not certify a complete amplitude.',
        'fpe_limit': 'The common FPE flag can persist from warmup or earlier calls; per-call native status is the direct evidence of a rejected tensor.'}
    subordinate_packages = {'geometry', 'ct-qp-transport', 'tensor-replay'}
    summary['evidence_sha256'] = {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and p.name not in ('summary.json', 'trace.json', 'artifact-index.json')
        and not subordinate_packages.intersection(p.relative_to(HERE).parts)}
    (HERE / 'trace.json').write_text(json.dumps(points, indent=2, sort_keys=True) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary['points'], indent=2))


if __name__ == '__main__':
    analyze()
