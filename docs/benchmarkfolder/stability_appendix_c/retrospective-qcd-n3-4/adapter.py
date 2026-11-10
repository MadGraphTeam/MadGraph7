#!/usr/bin/env python3
"""Reproduce plots using only the explicitly admitted closed portable inputs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
CAPSULE = HERE.parents[1] / 'experiments/invariant-symmetry-qcd-n3-target-analysis/portable-analysis2'
PLOTTER = HERE.parent / 'plotter-source-5'
CAPSULE_SHA = '79f57a8d83ea74249af83d12908de299372714cd7587d59d4ae31cfa9f63e457'
ANALYSIS_SHA = 'e0dcc8460091021db3089f56b181d4b184b4f11df1302fd14b4a383bd1568778'
PLOTTER_SHA = 'fe16afaae081694af3eb3a7e7c8ec9b9d037933b1938e6afa6ba56522c507e41'
RESULT_SHA = '59bf41a90d581fb757b2240b57162fc85c5e1f29ec14a1d54713a2255188b373'
CURRENT = 'docs/benchmarkfolder/campaigns/invariant-symmetry-1/experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/results.json'
BACKENDS = (1, 6, 7, 8, 9)
STATUS = {211: (1, 'DP'), 216: (6, 'DP'), 217: (7, 'DP'),
          218: (8, 'DP'), 219: (9, 'DP'), 323: (9, 'QP')}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def strict_json(blob):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(blob, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def require(value, message):
    if not value: raise ValueError(message)


def checked_package(root, index_name, expected):
    root = Path(root)
    require(not any(path.is_symlink() for path in (root, *root.parents)), 'Symlink package root')
    blob = (root / index_name).read_bytes()
    require(sha(blob) == expected, 'Source package index changed')
    index = strict_json(blob)
    require(set(path.name for path in root.iterdir()) == set(index['files']) | {index_name},
            'Source package inventory differs')
    payloads = {}
    for name, item in index['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Unsafe payload name')
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'Expected regular payload: ' + name)
        source = path.read_bytes()
        require(sha(source) == item['sha256'] and len(source) == item['bytes'], 'Payload changed: ' + name)
        payloads[name] = source
    return payloads


def module_from_bytes(name, path, source):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def verified_inputs(capsule=CAPSULE, plotter=PLOTTER):
    capsule, plotter = Path(capsule), Path(plotter)
    files = checked_package(capsule, 'index.json', CAPSULE_SHA)
    require(sha(files['analysis.json']) == ANALYSIS_SHA, 'Closed analysis identity differs')
    inputs = module_from_bytes('portable_inputs', capsule / 'portable_inputs.py', files['portable_inputs.py'])
    previous = sys.modules.get('portable_inputs')
    sys.modules['portable_inputs'] = inputs
    try:
        verifier = module_from_bytes('closed_qcd_n3_verifier', capsule / 'verify.py', files['verify.py'])
        verification = verifier.verify(capsule)
        require(verification['passed'] and verification['analysis_sha256'] == ANALYSIS_SHA
                and verification['accepted_rows'] == 1200
                and verification['new_native_evaluations'] == 0, 'Portable analysis verification failed')
        mapping = strict_json(files['input-map.json'])['inputs']
        pins = {name: item['raw']['sha256'] for name, item in mapping.items()}
        require(pins[CURRENT] == RESULT_SHA and mapping[CURRENT]['gzip_file'] == 'input-01.gz',
                'Wrong closed result selected')
        raw = inputs.read_input(capsule, CURRENT, pins)
        require(sha(raw) == RESULT_SHA, 'Closed result hash differs')
    finally:
        if previous is None: sys.modules.pop('portable_inputs', None)
        else: sys.modules['portable_inputs'] = previous
    plotting = checked_package(plotter, 'index.json', PLOTTER_SHA)
    module = module_from_bytes('admitted_stability_plotter', plotter / 'plot_metrics.py', plotting['plot_metrics.py'])
    return strict_json(raw), verification, module


def same_value(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, float): return a.hex() == b.hex()
    if isinstance(a, list): return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict): return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def collapse(lane):
    rows = lane['observation']['rows']
    require(len(rows) == 200, 'Expected two answer orders for 100 physical points')
    collapsed = []
    for point in range(1, 101):
        zero, one = rows[2 * (point - 1):2 * point]
        require((zero['point'], zero['order'], one['point'], one['order']) == (point, 0, point, 1),
                'Physical-point/order sequence differs')
        require(same_value(zero | {'order': 1}, one), 'Duplicate answer orders disagree')
        require(STATUS.get(zero['status']) == (zero['actual_backend'], zero['precision'])
                and zero['actual_backend'] == lane['backend'], 'Accepted status/backend/precision differs')
        require(len(zero['coefficients']) == 4 and all(math.isfinite(x) for x in zero['coefficients'])
                and math.isfinite(zero['accuracy']) and zero['accuracy'] >= 0, 'Invalid accepted numerical row')
        collapsed.append(zero)
    return collapsed


def normalized(results, plotting):
    require(results['case']['id'] == 'qcd_n3_uux_zgg', 'Unexpected process')
    lanes = [lane for lane in results['results'] if lane['source'] == 'candidate']
    require([lane['backend'] for lane in lanes] == list(BACKENDS), 'All five current lanes required')
    selected = {lane['backend']: collapse(lane) for lane in lanes}
    reference = selected[1]
    outcomes = {}
    for backend, rows in selected.items():
        outcomes[str(backend)] = {
            'accepted_precision_counts': dict(Counter(row['precision'] for row in rows)),
            'accepted_status_counts': dict(Counter(str(row['status']) for row in rows)),
            'accepted_QP_points': [row['point'] for row in rows if row['precision'] == 'QP'],
            'exact_zero_internal_estimates': sum(row['accuracy'] == 0 for row in rows),
            'QP_work_indicated_points': [row['point'] for row in rows if row['status'] % 100 // 10 in (2, 4)],
            'interpretation': 'accepted precision and status, not all internal attempts or certified digits'}
    require(outcomes['9']['accepted_precision_counts'] == {'DP': 87, 'QP': 13}, 'TIRust accepted outcome changed')
    provenance = {
        'closed_capsule_index_sha256': CAPSULE_SHA, 'closed_analysis_sha256': ANALYSIS_SHA,
        'closed_results_sha256': RESULT_SHA, 'plotter_source_index_sha256': PLOTTER_SHA,
        'campaign': 'invariant-symmetry-1', 'new_native_evaluations': 0,
        'sampling': 'Original ordinary campaign sample; not asserted to satisfy Appendix C cuts',
        'answer_orders': 'exact duplicate orders 0/1 asserted; order 0 used once per physical point',
        'post_rescue': True, 'pre_rescue_DP_values_available': False,
        'actual_policy_marker': 'unavailable: current effective numerical card bytes are absent from selected closed capsule',
        'actual_policy_binding_available': False,
        'accepted_outcomes': outcomes,
        'warnings': 'this selected capsule does not provide an independently complete per-call console-warning qualification',
        'independent_true_finite_error_available': False,
        'reference': 'current candidate CutTools backend 1, not an independent truth reference',
        'reference_label': 'current CutTools backend 1'}
    datasets = {}
    for metric, key in (('accepted-composite-internal-estimate', 'accepted-internal-estimate'),
                        ('accepted-reducer-disagreement', 'finite-disagreement-current-cuttools')):
        plot_lanes = []
        for backend, rows in selected.items():
            values = ([row['accuracy'] for row in rows] if metric == 'accepted-composite-internal-estimate'
                      else [plotting.symmetric_disagreement(row['coefficients'][1], ref['coefficients'][1])
                            for row, ref in zip(rows, reference)])
            values = ['inf' if math.isinf(value) else value for value in values]
            plot_lanes.append({'backend': backend, 'values': values})
        data = {'schema': 'madloop-stability-point-metrics-v1', 'metric': metric,
                'provenance': provenance, 'cases': [{'id': 'qcd_n3_uux_zgg',
                    'label': 'u ū → Z g g [QCD]: accepted outputs', 'point_ids': list(range(1, 101)),
                    'lanes': plot_lanes}]}
        plotting.validate_dataset(data)
        datasets[key] = data
    return datasets, outcomes


def save_new(path, text):
    with Path(path).open('x') as stream: stream.write(text)


def execute(destination, capsule=CAPSULE, plotter=PLOTTER):
    destination = Path(destination)
    require(not destination.exists(), 'Fresh output directory required')
    results, verification, plotting = verified_inputs(capsule, plotter)
    datasets, outcomes = normalized(results, plotting)
    destination.mkdir(parents=True)
    save_new(destination / 'closed-input-verification.json', encoded(verification))
    save_new(destination / 'accepted-outcomes.json', encoded(outcomes))
    for name, data in datasets.items():
        text = encoded(data)
        save_new(destination / (name + '-points.json'), text)
        plotting.render(data, destination / name, input_sha256=sha(text.encode()))
    files = {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in sorted(destination.iterdir()) if p.is_file()}
    save_new(destination / 'index.json', encoded({
        'schema': 'madloop-stability-retrospective-qcd-n3-v1', 'files': files,
        'adapter_sha256': sha(Path(__file__).read_bytes()), 'closed_capsule_index_sha256': CAPSULE_SHA,
        'closed_analysis_sha256': ANALYSIS_SHA, 'plotter_source_index_sha256': PLOTTER_SHA,
        'new_native_evaluations': 0, 'performance_eligible': False,
        'scope': 'two retrospective accepted-output plots; not raw DP or independently known true error'}))
    return {'output': str(destination), 'index_sha256': sha((destination / 'index.json').read_bytes()),
            'new_native_evaluations': 0, 'files': len(files)}


def verify(destination, capsule=CAPSULE, plotter=PLOTTER):
    destination = Path(destination)
    results, verification, plotting = verified_inputs(capsule, plotter)
    datasets, outcomes = normalized(results, plotting)
    index = strict_json((destination / 'index.json').read_bytes())
    require(index['adapter_sha256'] == sha(Path(__file__).read_bytes()), 'Adapter changed')
    require(set(p.name for p in destination.iterdir()) == set(index['files']) | {'index.json'}, 'Output inventory differs')
    for name, item in index['files'].items():
        path = destination / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink(), 'Invalid output path')
        blob = path.read_bytes()
        require(sha(blob) == item['sha256'] and len(blob) == item['bytes'], 'Output changed: ' + name)
    require((destination / 'closed-input-verification.json').read_text() == encoded(verification), 'Verification differs')
    require((destination / 'accepted-outcomes.json').read_text() == encoded(outcomes), 'Accepted outcomes differ')
    for name, data in datasets.items():
        require((destination / (name + '-points.json')).read_text() == encoded(data), 'Point metrics differ')
        summary = strict_json((destination / (name + '.json')).read_bytes())
        expected = plotting.summarize(data, summary['thresholds'])
        expected['input_sha256'] = sha(encoded(data).encode())
        require(summary == expected, 'Derived tail summary differs')
    return {'verified': True, 'index_sha256': sha((destination / 'index.json').read_bytes()),
            'new_native_evaluations': 0, 'accepted_TIRust_DP': 87, 'accepted_TIRust_QP': 13}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, default=CAPSULE)
    parser.add_argument('--plotter', type=Path, default=PLOTTER)
    parser.add_argument('--destination', type=Path, default=HERE / 'figures')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    require(args.execute != args.verify_only, 'Select exactly one of --execute and --verify-only')
    result = (verify if args.verify_only else execute)(args.destination, args.capsule, args.plotter)
    print(encoded(result), end='')


if __name__ == '__main__': main()
