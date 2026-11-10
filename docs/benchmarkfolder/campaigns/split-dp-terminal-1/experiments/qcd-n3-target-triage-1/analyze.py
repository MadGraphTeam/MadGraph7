#!/usr/bin/env python3
"""Replay retained QCD n3 evidence without compilation or native execution."""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
REPORT = ROOT / 'docs/benchmarkfolder'
CAMPAIGN = REPORT / 'campaigns/split-dp-terminal-1'
CURRENT = CAMPAIGN / 'experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1'
HISTORICAL = REPORT / 'experiments/numerical/strict-policy-qcd-n3'
QP = REPORT / 'experiments/qcd-n3-adjudication'
LANES = [('baseline', 1), *[('candidate', value) for value in (1, 6, 7, 8, 9)]]
NAMES = {1: 'CutTools', 6: 'Ninja', 7: 'COLLIER', 8: 'NinjaRS', 9: 'TIRust'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def bits(value):
    return struct.pack('>d', value).hex()


def verify_inputs():
    ledger = read(HERE / 'inputs.json')
    for name, record in ledger['files'].items():
        path = ROOT / name
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    return ledger


def retained_helpers():
    # Execute only the retained pure comparator function and parser modules.
    # No campaign activation, live provenance check, subprocess or reducer call.
    tree = ast.parse((CURRENT / 'tirust_benchmark.py').read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == 'compare_rows')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<retained compare_rows>', 'exec'), namespace)
    for name in ('tirust_benchmark_worker', 'tirust_benchmark_warmup'):
        spec = importlib.util.spec_from_file_location(name, CURRENT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return namespace['compare_rows'], sys.modules['tirust_benchmark_warmup'].parse_warmup


def condensed(comparison):
    target_misses = [row for row in comparison['residuals']
                     if not ((row['symmetric_relative_error'] is not None
                              and row['symmetric_relative_error'] < 1e-8)
                             or (row['pole_to_finite'] is not None and row['pole_to_finite'] < 1e-8))]
    worst = max(comparison['residuals'], key=lambda row: row['symmetric_relative_error'] or 0)
    return {key: comparison[key] for key in ('passed', 'ordinary_target_passed', 'maximum_relative_error')} | {
        'maximum_symmetric_relative_error': worst['symmetric_relative_error'],
        'worst_symmetric_residual': worst,
        'target_missed_points': sorted({row['point'] for row in target_misses}),
        'target_missed_components': sorted({row['column'] for row in target_misses}),
        'target_missed_coefficient_comparisons': len(target_misses),
        'target_misses': target_misses}


def raw_rows(path, mapping):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        assert len(fields) == 8
        point, order, status = map(int, fields[:3])
        values = list(map(float, fields[3:]))
        assert all(math.isfinite(value) for value in values)
        rows.append({'point': mapping[point - 1], 'order': order, 'status': status,
                     'coefficients': values[:4], 'accuracy': values[4]})
    assert [(row['point'], row['order']) for row in rows] == [
        (point, order) for point in mapping for order in (0, 1)]
    return rows


def same_raw(recorded, parsed):
    assert len(recorded) == len(parsed)
    for a, b in zip(recorded, parsed):
        for key in ('point', 'order', 'status'):
            assert a[key] == b[key]
        assert [bits(value) for value in a['coefficients']] == [bits(value) for value in b['coefficients']]
        assert bits(a['accuracy']) == bits(b['accuracy'])


def collect():
    ledger = verify_inputs()
    compare, parse_warmup = retained_helpers()
    points_path = CAMPAIGN / 'points/qcd_n3_uux_zgg.dat'
    metadata = read(points_path.with_suffix('.json'))
    current, histories, equality, inputs = {}, {}, {}, {}
    for source, backend in LANES:
        key = f'{source}-{backend}'
        directory = CURRENT / key
        result = read(directory / 'result.json')
        binding = read(directory / (source + '-inputs.json'))
        after = read(directory / 'inputs-after.json')
        assert result['status'] == 'complete' and result['performance_eligible'] is False
        assert binding['points_sha256'] == sha(points_path) == after['points_sha256']
        assert binding['points_metadata_sha256'] == sha(points_path.with_suffix('.json'))
        assert binding['param_card_sha256'] == sha(HERE / 'source/param_card.dat')
        assert all(binding['runtime'][kind] == after['runtime'][kind]
                   for kind in ('libraries', 'archives', 'metadata'))
        assert binding['numerical_cards'] == after['numerical_cards']
        assert binding['compilation']['executable_sha256'] == after['executable_sha256']
        observation = parse_warmup(directory / 'raw', backend, 100, read(directory / 'raw/ready.json'))
        assert observation == result['observation'] == read(directory / 'raw/observation.json')
        assert observation['seconds'] is None and observation['performance_eligible'] is False
        current[key] = observation['rows']
        inputs[key] = binding
        if source == 'baseline':
            baseline = observation['rows']
        assert compare(baseline, observation['rows']) == result['comparison']
        assert result['comparison']['passed']
        old = read(HISTORICAL / key / 'result.json')['observation']['rows']
        old_binding = read(HISTORICAL / key / (source + '-inputs.json'))
        assert old_binding['points_sha256'] == binding['points_sha256']
        assert old_binding['param_card_sha256'] == binding['param_card_sha256']
        same_raw(old, raw_rows(HISTORICAL / key / 'raw/values.dat', list(range(1, 101))))
        histories[key] = old
        changes = []
        for left, right in zip(old, current[key]):
            assert (left['point'], left['order']) == (right['point'], right['order'])
            for column, (a, b) in enumerate(zip(left['coefficients'], right['coefficients'])):
                if bits(a) != bits(b):
                    changes.append({'point': left['point'], 'order': left['order'], 'column': column,
                                    'historical': a, 'current': b, 'historical_bits': bits(a),
                                    'current_bits': bits(b), 'absolute_difference': abs(a-b)})
        equality[key] = {'coefficient_rows': len(old), 'coefficient_binary64_identical': not changes,
                         'statuses_identical': [row['status'] for row in old] == [row['status'] for row in current[key]],
                         'changed_coefficients': changes}

    # Verify complete copied normalization sources against retained compilation
    # and source-generation hashes; no ignored live build is needed for replay.
    for name, record in ledger['normalization_sources'].items():
        assert sha(HERE / 'source' / name) == record['sha256']
        if record['binding'] == 'compilation':
            assert inputs['candidate-9']['compilation']['sources'][record['source']] == record['sha256']
        elif record['binding'] == 'generation':
            assert inputs['candidate-9']['generation']['source_identity']['files'][record['source']] == record['sha256']
        else:
            assert name == 'param_card.dat' and inputs['candidate-9']['param_card_sha256'] == record['sha256']

    original = {key: condensed(compare(baseline, rows)) for key, rows in current.items()}
    target_points = sorted(set().union(*(set(value['target_missed_points']) for value in original.values())))
    backend_rows = {backend: current[f'candidate-{backend}'] for backend in NAMES}
    assert all([bits(row['coefficients'][0]) for row in rows] ==
               [bits(row['coefficients'][0]) for row in baseline] for rows in backend_rows.values())
    pairwise = {f'{a}-{b}': condensed(compare(backend_rows[a], backend_rows[b]))
                for a, b in itertools.combinations(NAMES, 2)}
    points = []
    for point in target_points:
        observations = {}
        for backend, rows in backend_rows.items():
            selected = [row for row in rows if row['point'] == point]
            assert len(selected) == 2 and selected[0]['coefficients'] == selected[1]['coefficients']
            assert selected[0]['status'] == selected[1]['status']
            observations[str(backend)] = selected[0]
        points.append({'point': point, 'components': ['finite'], 'orders': [0, 1],
                       'accepted_results': observations,
                       'finite_pairwise': {key: next(row for row in compare(backend_rows[int(key.split('-')[0])],
                                   backend_rows[int(key.split('-')[1])])['residuals']
                                   if row['point'] == point and row['order'] == 0 and row['column'] == 1)
                           for key in pairwise}})

    qp_rows, qp_provenance = {}, {}
    for group, source, backend in [('minimal', 'baseline', 1), ('minimal', 'candidate', 6),
                                   ('minimal', 'candidate', 9), ('target-misses', 'baseline', 1)]:
        key = f'{group}/{source}-{backend}-init-dp-unchecked'
        directory = QP / 'runs' / key
        result = read(directory / 'result.json')
        record = result['point_record']
        assert result['status'] == 'diagnostic' and result['stability_certified'] is False
        assert result['performance_eligible'] is False
        assert record['parent_sha256'] == sha(points_path)
        assert record['parameter_card_sha256'] == sha(HERE / 'source/param_card.dat') == sha(directory / 'param_card.dat')
        assert record['mu_r_gev'] == 1000.0 and record['mu_r_squared_gev2'] == 1e6
        mapping = record['source_point_indices']
        assert record['points'] == [metadata['points'][point - 1] for point in mapping]
        assert result['settings'] == dict(inputs[f'{source}-{backend}']['settings'], ctmoderun=4)
        rows = result['observation']['rows']
        same_raw(rows, raw_rows(directory / 'raw/values.dat', mapping))
        assert all(row['status'] == 120 and row['precision'] == 'QP' and row['accuracy'] == -1
                   and row['stability_checked'] is False for row in rows)
        assert result['pilot_results_sha256'] == sha(HISTORICAL / 'results.json')
        qp_rows[key] = rows
        qp_provenance[key] = {'point_indices': mapping, 'result_sha256': sha(directory / 'result.json'),
                              'raw_values_sha256': sha(directory / 'raw/values.dat'),
                              'scope': result['scope'], 'stability_certified': False,
                              'performance_eligible': False}
    qp_reference = qp_rows['target-misses/baseline-1-init-dp-unchecked']
    assert sorted({row['point'] for row in qp_reference}) == target_points
    overlap = [row for row in qp_reference if row['point'] in (39, 60, 91, 93)]
    same_raw(qp_rows['minimal/baseline-1-init-dp-unchecked'], overlap)
    qp_pairs = {}
    for a, b in itertools.combinations((1, 6, 9), 2):
        ka = f'minimal/{"baseline" if a == 1 else "candidate"}-{a}-init-dp-unchecked'
        kb = f'minimal/{"baseline" if b == 1 else "candidate"}-{b}-init-dp-unchecked'
        qp_pairs[f'{a}-{b}'] = condensed(compare(qp_rows[ka], qp_rows[kb]))
    against_qp = {str(backend): condensed(compare(qp_reference, [row for row in rows if row['point'] in target_points]))
                  for backend, rows in backend_rows.items()}

    # Source-reconstructed G, with binary64 operations matching the generated
    # expression. This is a diagnostic normalization, not a native G trace.
    pi = 3.141592653589793
    alpha_s = .118
    coupling = 2 * math.sqrt(alpha_s * pi)
    factor = -(2*(4/3) + 2*3) * coupling**2 / (8*pi**2)
    assert all(point['momenta'][:2] == [[500., 0., 0., 500.], [500., 0., 0., -500.]]
               for point in metadata['points'])
    analytic = {}
    for backend, rows in backend_rows.items():
        residuals = []
        for row in rows:
            expected = row['coefficients'][0] * factor
            actual = row['coefficients'][3]
            residuals.append({'point': row['point'], 'order': row['order'], 'born': row['coefficients'][0],
                              'expected_double_pole': expected, 'observed_double_pole': actual,
                              'relative_to_expected': abs(actual-expected)/abs(expected),
                              'symmetric_relative_error': abs(actual-expected)/abs(actual+expected)})
        analytic[str(backend)] = {'maximum_relative_to_expected': max(row['relative_to_expected'] for row in residuals),
                                  'maximum_symmetric_relative_error': max(row['symmetric_relative_error'] for row in residuals),
                                  'rows': residuals}
    return {'schema_version': 1, 'case': 'qcd_n3_uux_zgg', 'campaign': 'split-dp-terminal-1',
            'scope': 'Read-only numerical diagnosis; original comparisons and campaign gates unchanged',
            'performance_eligible': False, 'new_native_evaluations': 0,
            'metrics': {'gate': 'abs(a-b)/abs(a+b)<3e-6, or pole magnitude/finite maximum<3e-6',
                        'ordinary_target': 'same criterion at 1e-8',
                        'raw_relative': 'abs(a-b)/max(abs(a),abs(b))',
                        'columns': ['Born', 'finite', 'single pole', 'double pole']},
            'original_comparisons': original, 'pairwise_all100': pairwise,
            'target_points': target_points, 'target_point_results': points,
            'historical_replay': equality,
            'higher_precision': {'provenance': qp_provenance, 'three_reducer_four_point_pairs': qp_pairs,
                                 'current_against_ct_qp28': against_qp,
                                 'qp28_rows': qp_reference, 'stability_certified': False},
            'universal_double_pole': {'formula': '-Born*(2*CF+2*CA)*G**2/(8*pi**2)',
                                     'alpha_s_card': alpha_s, 'CF': 4/3, 'CA': 3,
                                     'G_reconstructed': coupling, 'G_binary64_bits': bits(coupling),
                                     'factor': factor, 'scale_gev': 1000.0,
                                     'caveat': 'G is source-reconstructed, not directly recorded; Born uses P_USER before loop momentum repair. No finite or single-pole truth claim.',
                                     'backends': analytic},
            'fpe_flags_observed': False, 'internal_retries_observed': False,
            'accepted_status_scope': 'Final accepted backend/precision only; no count of internal attempts'}


def main():
    if sys.flags.optimize:
        raise RuntimeError('Verification requires enabled assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='write new derived files once')
    args = parser.parse_args()
    if not args.freeze:
        index = read(HERE / 'index.json')
        for name, value in index['files'].items():
            path = HERE / name
            assert not path.is_symlink() and path.resolve().is_relative_to(HERE)
            assert path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], name
        assert sorted(index['files']) == sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*')
                                                if p.is_file() and p.name != 'index.json')
    result = collect()
    if args.freeze:
        write(HERE / 'analysis.json', result)
        files = {str(p.relative_to(HERE)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'index.json'}
        write(HERE / 'index.json', {'schema_version': 1, 'scope': result['scope'],
                                  'new_native_evaluations': 0, 'performance_eligible': False,
                                  'files': files})
    else:
        assert result == read(HERE / 'analysis.json')
    print(json.dumps({'status': 'passed', 'target_points': result['target_points'],
                      'analysis_sha256': sha(HERE / 'analysis.json'),
                      'index_sha256': sha(HERE / 'index.json'),
                      'new_native_evaluations': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
