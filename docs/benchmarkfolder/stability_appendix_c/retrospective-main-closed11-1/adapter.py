#!/usr/bin/env python3
"""Plot only four explicitly selected cases from the closed eleven-case capsule."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import types

HERE = Path(__file__).resolve().parent
CAPSULE = HERE.parents[1] / 'campaigns/invariant-symmetry-1/numerical-retention-1/sealed-1'
PLOTTER = HERE.parent / 'plotter-source-5'
CAPSULE_SHA = '8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be'
PLOTTER_SHA = 'fe16afaae081694af3eb3a7e7c8ec9b9d037933b1938e6afa6ba56522c507e41'
PREFIX = 'docs/benchmarkfolder/campaigns/invariant-symmetry-1/'
BACKENDS = (1, 6, 7, 8, 9)
CASES = {
    'qcd_n2_gg_ttx': ('g g → t t̄ [QCD]', '23e7075c8afab1cca898ad1734621d9c6ea3f6b626303dc5d45b414c513a4597'),
    'qcd_n3_uux_zgg': ('u ū → Z g g [QCD]', '59bf41a90d581fb757b2240b57162fc85c5e1f29ec14a1d54713a2255188b373'),
    'ew_n2_epem_mupmum': ('e⁺ e⁻ → μ⁺ μ⁻ [EW]', 'a2bd40f2d32deb5ba79112fe2a40238eb29fda389fbceca55eb3506351da418a'),
    'qcd_n4_uux_zzgg': ('u ū → Z Z g g [QCD]', 'a5630ade23fccd0da72fa9517148b5112eb647d2f286b159730c941fa35a169c'),
}
# H and U are the hundreds and units of H*100 + T*10 + U. U aliases
# depend on H: in particular (2,9)=TIRust DP but (3,9)=CutTools QP.
STATUS = {211: (1, 'DP'), 216: (6, 'DP'), 217: (7, 'DP'),
          218: (8, 'DP'), 219: (9, 'DP'), 329: (1, 'QP'),
          328: (6, 'QP'), 322: (8, 'QP'), 323: (9, 'QP')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def checked_package(root, expected):
    root = Path(root)
    require(not any(path.is_symlink() for path in (root, *root.parents)), 'Symlink package root')
    raw = (root / 'index.json').read_bytes()
    require(sha(raw) == expected, 'Pinned package index changed')
    index = strict_json(raw)
    require(set(path.name for path in root.iterdir()) == set(index['files']) | {'index.json'},
            'Pinned package inventory differs')
    files = {}
    for name, identity in index['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Unsafe payload name')
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'Nonregular package payload')
        payload = path.read_bytes()
        require({'sha256': sha(payload), 'bytes': len(payload)} == identity, 'Package payload changed: ' + name)
        files[name] = payload
    return files


def module_from_bytes(name, path, source):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def verified_inputs(capsule=CAPSULE, plotter=PLOTTER):
    capsule, plotter = Path(capsule), Path(plotter)
    files = checked_package(capsule, CAPSULE_SHA)
    retained = module_from_bytes('closed_main_retainer', capsule / 'retain.py', files['retain.py'])
    index, store = retained.load(capsule, CAPSULE_SHA)
    summary = retained.validate(store, store.json('closed-case-confirmation.json'), capsule)
    require(summary == strict_json(files['summary.json']), 'Closed numerical replay differs')
    require(summary['full_matrix_closed'] and summary['full_matrix_regression_passed']
            and summary['selected_case_count'] == 11 and summary['selected_lane_slots'] == 66,
            'Expected the fully closed eleven-case numerical campaign')
    plotting = checked_package(plotter, PLOTTER_SHA)
    module = module_from_bytes('admitted_stability_plotter', plotter / 'plot_metrics.py', plotting['plot_metrics.py'])
    verification = {'passed': True, 'closed_capsule_index_sha256': CAPSULE_SHA,
                    'closed_summary_sha256': sha(files['summary.json']),
                    'full_matrix_closed': True, 'full_matrix_regression_passed': True,
                    'closed_cases': 11, 'closed_lanes': 66, 'new_native_evaluations': 0}
    return store, verification, module


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex()
    if isinstance(a, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def collapse(lane):
    require(lane['status'] == 'complete' and lane['performance_eligible'] is False,
            'Only completed numerical lanes are admitted')
    rows = lane['observation']['rows']
    require(len(rows) == 200, 'Expected two answer orders for 100 physical points')
    collapsed = []
    for point in range(1, 101):
        zero, one = rows[2 * (point - 1):2 * point]
        require((zero['point'], zero['order'], one['point'], one['order']) == (point, 0, point, 1),
                'Physical-point/order sequence differs')
        require(same_value(zero | {'order': 1}, one), 'Duplicate answer orders disagree')
        require(type(zero['status']) is int and type(zero['actual_backend']) is int
                and STATUS.get(zero['status']) == (zero['actual_backend'], zero['precision'])
                and zero['actual_backend'] == lane['backend'], 'Accepted status/backend/precision differs')
        require(len(zero['coefficients']) == 4 and all(math.isfinite(x) for x in zero['coefficients'])
                and math.isfinite(zero['accuracy']) and zero['accuracy'] >= 0, 'Invalid accepted numerical row')
        collapsed.append(zero)
    return collapsed


def normalized(store, plotting):
    datasets, all_outcomes = {}, {}
    for case_id, (label, expected_sha) in CASES.items():
        directory = PREFIX + 'experiments/numerical/full100-warmup-' + case_id + '-1/'
        result_name = directory + 'results.json'
        require(sha(store.members[result_name]) == expected_sha, 'Selected closed result changed')
        results = store.json(result_name)
        require(results['case']['id'] == case_id, 'Wrong closed process selected')
        lanes = [lane for lane in results['results'] if lane['source'] == 'candidate']
        require([lane['backend'] for lane in lanes] == list(BACKENDS), 'All five current lanes required')
        selected = {lane['backend']: collapse(lane) for lane in lanes}
        outcomes, policies = {}, {}
        for lane in lanes:
            backend, rows = lane['backend'], selected[lane['backend']]
            lane_prefix = directory + f'candidate-{backend}/'
            inputs_name = lane_prefix + 'candidate-inputs.json'
            inputs = store.json(inputs_name)
            require(inputs['settings']['mlreductionlib'] == str(backend)
                    and inputs['settings']['ctmoderun'] == -1
                    and inputs['settings']['nrotations_dp'] == 0,
                    'Unexpected accepted-output policy')
            console_name = lane_prefix + 'candidate-output.txt'
            console = store.members[console_name]
            w03 = len(re.findall(rb'##W03\b', console, re.IGNORECASE))
            require(w03 == 0, 'Closed lane contains a native instability warning')
            policies[str(backend)] = {
                'retained_input_record': inputs_name, 'retained_input_record_sha256': sha(store.members[inputs_name]),
                'numerical_card_sha256_by_path': inputs['numerical_cards'],
                'effective_numerical_card_bytes_retained': False,
                'recorded_settings': inputs['settings'], 'physical_card_sha256': inputs['param_card_sha256'],
                'points_sha256': inputs['points_sha256'], 'points_metadata_sha256': inputs['points_metadata_sha256'],
                'retained_console': console_name, 'retained_console_sha256': sha(console),
                'W03_count_case_insensitive': w03,
                'compiled_executable_sha256': inputs['compilation']['executable_sha256'],
                'runtime_libraries_sha256': sha(encoded(inputs['runtime']['libraries']).encode()),
            }
            outcomes[str(backend)] = {
                'accepted_precision_counts': dict(Counter(row['precision'] for row in rows)),
                'accepted_status_counts': dict(Counter(str(row['status']) for row in rows)),
                'accepted_H_U_counts': dict(Counter(f"{row['status']//100},{row['status']%10}" for row in rows)),
                'accepted_QP_points': [row['point'] for row in rows if row['precision'] == 'QP'],
                'exact_zero_internal_estimates': sum(row['accuracy'] == 0 for row in rows),
                'unavailable_points': 0, 'accepted_rows': rows,
                'comparison_to_starting_baseline': {
                    'regression_passed': lane['comparison']['passed'],
                    'ordinary_1e8_target_passed': lane['comparison']['ordinary_target_passed'],
                    'maximum_relative_error': lane['comparison']['maximum_relative_error'],
                },
                'interpretation': 'accepted status and precision, not all internal attempts or certified digits',
            }
        point_hashes = {policy['points_sha256'] for policy in policies.values()}
        require(len(point_hashes) == 1, 'Backend point streams differ')
        provenance = {
            'closed_capsule_index_sha256': CAPSULE_SHA, 'closed_results_sha256': expected_sha,
            'plotter_source_index_sha256': PLOTTER_SHA, 'campaign': 'invariant-symmetry-1',
            'new_native_evaluations': 0,
            'sampling': 'Ordinary saved campaign fixtures, not asserted to satisfy Appendix C cuts',
            'answer_orders': 'exact duplicate orders 0/1 asserted; order 0 used once per physical point',
            'post_rescue': True, 'pre_rescue_DP_values_available': False,
            'actual_policy_binding_available': False,
            'actual_policy_marker': 'omitted: numerical card hashes/settings are retained, but effective card bytes are absent',
            'recorded_policy_and_native_identity_by_backend': policies,
            'accepted_outcomes': outcomes,
            'independent_true_finite_error_available': False,
            'reference': 'current candidate CutTools backend 1, not an independent truth reference',
            'reference_label': 'current CutTools backend 1',
            'fpe_flags_observed': False, 'internal_retry_counts_observed': False,
            'COLLIER_estimator': 'may differ from the other host composite estimates',
        }
        abbreviation = {1: 'CT', 6: 'N', 7: 'C', 8: 'NR', 9: 'T'}
        def count(backend):
            counts = outcomes[str(backend)]['accepted_precision_counts']
            return f"{abbreviation[backend]} {counts.get('DP',0)}/{counts.get('QP',0)}"
        case_label = (label + '; 100 ordinary points\nAccepted DP/QP: ' + '; '.join(count(b) for b in BACKENDS[:3])
                      + '\n' + '; '.join(count(b) for b in BACKENDS[3:]))
        for metric, name in (('accepted-composite-internal-estimate', 'accepted-internal-estimate'),
                             ('accepted-reducer-disagreement', 'finite-disagreement-current-cuttools')):
            plotted = []
            for backend, rows in selected.items():
                values = ([row['accuracy'] for row in rows] if metric == 'accepted-composite-internal-estimate'
                          else [plotting.symmetric_disagreement(row['coefficients'][1], reference['coefficients'][1])
                                for row, reference in zip(rows, selected[1])])
                plotted.append({'backend': backend, 'values': ['inf' if math.isinf(v) else v for v in values]})
            data = {'schema': 'madloop-stability-point-metrics-v1', 'metric': metric, 'provenance': provenance,
                    'cases': [{'id': case_id, 'label': case_label, 'point_ids': list(range(1,101)), 'lanes': plotted}]}
            plotting.validate_dataset(data)
            datasets[case_id + '-' + name] = data
        all_outcomes[case_id] = {'case': results['case'], 'accepted_outcomes': outcomes,
                                 'policy_and_native_identity_by_backend': policies}
    return datasets, all_outcomes


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(encoded(value))


def execute(destination, capsule=CAPSULE, plotter=PLOTTER):
    destination = Path(destination)
    require(not destination.exists(), 'Fresh output directory required')
    store, verification, plotting = verified_inputs(capsule, plotter)
    datasets, outcomes = normalized(store, plotting)
    destination.mkdir(parents=True)
    write_new(destination / 'closed-input-verification.json', verification)
    write_new(destination / 'accepted-outcomes.json', outcomes)
    for name, data in datasets.items():
        write_new(destination / (name + '-points.json'), data)
        plotting.render(data, destination / name, input_sha256=sha(encoded(data).encode()))
    files = {path.name: {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}
             for path in sorted(destination.iterdir()) if path.is_file()}
    write_new(destination / 'index.json', {
        'schema': 'madloop-stability-retrospective-main-closed11-figures-v1', 'files': files,
        'adapter_sha256': sha(Path(__file__).read_bytes()), 'closed_capsule_index_sha256': CAPSULE_SHA,
        'plotter_source_index_sha256': PLOTTER_SHA, 'case_ids': list(CASES), 'figures': 8,
        'new_native_evaluations': 0, 'performance_eligible': False,
        'scope': 'accepted-output retrospective diagnostics; not raw DP, paper-cut-qualified fixtures, or true errors'})
    return {'output': str(destination), 'index_sha256': sha((destination / 'index.json').read_bytes()),
            'new_native_evaluations': 0, 'figures': 8, 'files': len(files)}


def verify(destination, capsule=CAPSULE, plotter=PLOTTER):
    destination = Path(destination)
    store, verification, plotting = verified_inputs(capsule, plotter)
    datasets, outcomes = normalized(store, plotting)
    index = strict_json((destination / 'index.json').read_bytes())
    require(index['adapter_sha256'] == sha(Path(__file__).read_bytes()), 'Adapter changed')
    require(set(path.name for path in destination.iterdir()) == set(index['files']) | {'index.json'},
            'Figure inventory differs')
    for name, expected in index['files'].items():
        path = destination / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink(), 'Invalid output path')
        raw = path.read_bytes()
        require({'sha256': sha(raw), 'bytes': len(raw)} == expected, 'Figure payload changed: ' + name)
    require((destination / 'closed-input-verification.json').read_text() == encoded(verification), 'Replay differs')
    require((destination / 'accepted-outcomes.json').read_text() == encoded(outcomes), 'Accepted outcomes differ')
    for name, data in datasets.items():
        require((destination / (name + '-points.json')).read_text() == encoded(data), 'Point metric differs')
        summary = strict_json((destination / (name + '.json')).read_bytes())
        expected = plotting.summarize(data, plotting.threshold_grid(data))
        expected['input_sha256'] = sha(encoded(data).encode())
        require(summary == expected, 'Exact empirical tail summary differs')
    return {'verified': True, 'index_sha256': sha((destination / 'index.json').read_bytes()),
            'new_native_evaluations': 0, 'cases': 4, 'figures': 8, 'physical_points_per_case': 100}


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


if __name__ == '__main__':
    main()
