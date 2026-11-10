#!/usr/bin/env python3
"""Replay the closed QCD n4 checkpoint without native execution or live builds."""
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
REPORT = ROOT / 'docs/benchmarkfolder/campaigns/split-dp-terminal-1'
CASE = 'qcd_n4_uux_zzgg'
CURRENT = REPORT / ('experiments/numerical/full100-warmup-' + CASE + '-1')
LANES = [('baseline', 1), *[('candidate', value) for value in (1, 6, 7, 8, 9)]]
NAMES = {1: 'CutTools', 6: 'Ninja', 7: 'COLLIER', 8: 'NinjaRS', 9: 'TIRust'}
# Independently inspected, immutable source versions establishing the literals
# used below. These fingerprints are also checked against compilation/generation
# provenance; the replay does not attempt to interpret arbitrary Fortran.
NORMALIZATION_SOURCES = {
    'born_matrix.f': 'a3521e23fa855bb8e82281c3d7b68ffd5056693213c1b1dcd9ba9ee2ea84b900',
    'loop_matrix.f': 'a07804ebcb4088c8b82da923b630f845c3d4339d4bacbccdca976308ecdc437a',
    'TIR_interface.f': '05ae4e14af288945845ba259f8f185b7257d113fa5ca8b49e80d8b43c4ef3904',
    'intparam_definition.inc': '4d14529f6c3cb833a2cb434d8bfca120d83d00f132ce414e37e1033cbcb83084',
    'couplings.f': 'e732847aa93cdcae502cd37808eb68b45d7dc3c9665bdd5d4de443be19747936',
    'param_card.dat': '5a535b7044777bd87c7088dc14bf9605d165ebae843dbcbde91a02f545c05433',
    'fks_singular.f': '65c9ba67697b697deac3d626cb425ee0dbc747afba5abefbd0808a0a8fba26a7',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def bits(value):
    return struct.pack('>d', value).hex()


def verify_inputs():
    ledger = read(HERE / 'inputs.json')
    relative = Path(ledger['checkpoint']['path'])
    assert str(relative) == 'numerical-checkpoints/qcd-n4-1/index.json'
    checkpoint = REPORT / relative
    assert not any(path.is_symlink() for path in [checkpoint, *checkpoint.parents]
                   if path.is_relative_to(REPORT))
    assert info(checkpoint) == ledger['checkpoint']['identity']
    index = read(checkpoint)
    assert index['schema'] == 'madloop-tirust-closed-numerical-checkpoint-v1'
    assert index['case_count'] == 1 and index['lane_count'] == 6
    assert index['performance_acceptance_eligible'] is False
    for name, identity in index['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = REPORT / relative
        assert path.is_file() and not any(parent.is_symlink() for parent in [path, *path.parents]
                                          if parent.is_relative_to(REPORT))
        assert info(path) == identity, name
    assert len(index['files']) == index['file_count']
    assert sum(value['bytes'] for value in index['files'].values()) == index['total_bytes']
    assert not any('/full100-warmup-ew_n4_' in name or name.endswith('numerical-qualification-2/status.json')
                   for name in index['files'])
    required = {REPORT/'cases.json', REPORT/('points/'+CASE+'.dat'), REPORT/('points/'+CASE+'.json'),
                checkpoint.parent/'summary.json'}
    required.update(CURRENT/(name+'.py') for name in
                    ('tirust_benchmark', 'tirust_benchmark_worker', 'tirust_benchmark_warmup'))
    for source, backend in LANES:
        directory = CURRENT/f'{source}-{backend}'
        required.update(directory/name for name in ('result.json',source+'-inputs.json','inputs-after.json'))
        required.update(directory/'raw'/name for name in
                        ('ready.json','observation.json','warmup.dat','dimensions.dat'))
    assert {str(path.relative_to(REPORT)) for path in required} <= index['files'].keys()
    summary = read(checkpoint.parent/'summary.json')
    assert summary['schema'] == 'madloop-tirust-completed-qcd-n4-numerical-v1'
    assert summary['case_ids'] == [CASE] and summary['case_count'] == 1 and summary['lane_count'] == 6
    assert summary['cases'][0]['passed'] is True and summary['saved_points_per_lane'] == 100
    assert ledger['normalization_sources'].keys() == NORMALIZATION_SOURCES.keys()
    for name, record in ledger['normalization_sources'].items():
        path = HERE / 'source' / name
        assert path.resolve().is_relative_to(HERE / 'source') and not path.is_symlink()
        assert info(path) == record['identity']
        assert record['identity']['sha256'] == NORMALIZATION_SOURCES[name]
    return ledger, index


def retained_helpers():
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


def missed(row):
    return not ((row['symmetric_relative_error'] is not None and row['symmetric_relative_error'] < 1e-8)
                or (row['pole_to_finite'] is not None and row['pole_to_finite'] < 1e-8))


def condensed(comparison):
    misses = [row for row in comparison['residuals'] if missed(row)]
    worst = max(comparison['residuals'], key=lambda row: row['symmetric_relative_error'] or 0)
    return {key: comparison[key] for key in ('passed', 'ordinary_target_passed', 'maximum_relative_error')} | {
        'maximum_symmetric_relative_error': worst['symmetric_relative_error'],
        'worst_symmetric_residual': worst,
        'target_missed_points': sorted({row['point'] for row in misses}),
        'target_missed_components': sorted({row['column'] for row in misses}),
        'target_missed_coefficient_comparisons': len(misses), 'target_misses': misses}


def relative(a, b, symmetric=False):
    denominator = abs(a+b) if symmetric else abs(b)
    return abs(a-b)/denominator if denominator else (0.0 if a == b else None)


def collect():
    ledger, checkpoint = verify_inputs()
    compare, parse_warmup = retained_helpers()
    points_path = REPORT / ('points/' + CASE + '.dat')
    metadata = read(points_path.with_suffix('.json'))
    assert len(metadata['points']) == 100
    config = read(REPORT / 'cases.json')
    assert config['relative_tolerance'] == 3e-6
    current, bindings, usage = {}, {}, {}
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
        observation = parse_warmup(directory / 'raw', [backend], 100, read(directory / 'raw/ready.json'))
        assert observation == result['observation'] == read(directory / 'raw/observation.json')
        assert observation['seconds'] is None and observation['performance_eligible'] is False
        assert sorted({row['point'] for row in observation['rows']}) == list(range(1, 101))
        current[key], bindings[key] = observation['rows'], binding
        usage[key] = {'accepted_backend_precision_counts': observation['backend_precision_counts'],
                      'status_counts': observation['status_counts']}
        if source == 'baseline':
            baseline = observation['rows']
        assert compare(baseline, observation['rows'], config['relative_tolerance']) == result['comparison']
        assert result['comparison']['passed']
    for name, record in ledger['normalization_sources'].items():
        fingerprint = record['identity']['sha256']
        if record['binding'] == 'compilation':
            assert bindings['candidate-9']['compilation']['sources'][record['source']] == fingerprint
        elif record['binding'] == 'generation':
            assert bindings['candidate-9']['generation']['source_identity']['files'][record['source']] == fingerprint
        else:
            assert name == 'param_card.dat' and bindings['candidate-9']['param_card_sha256'] == fingerprint

    original = {key: condensed(compare(baseline, rows, config['relative_tolerance'])) for key, rows in current.items()}
    target_points = sorted(set().union(*(set(value['target_missed_points']) for value in original.values())))
    backend_rows = {backend: current[f'candidate-{backend}'] for backend in NAMES}
    assert all([bits(row['coefficients'][0]) for row in rows] ==
               [bits(row['coefficients'][0]) for row in baseline] for rows in backend_rows.values())
    pairwise_full = {f'{a}-{b}': compare(backend_rows[a], backend_rows[b], config['relative_tolerance'])
                     for a, b in itertools.combinations(NAMES, 2)}
    membership = {}
    for point in target_points:
        key = '|'.join(str(backend) for backend in NAMES
                       if point in original[f'candidate-{backend}']['target_missed_points'])
        membership.setdefault(key, []).append(point)
    point_results = []
    for point in target_points:
        components = sorted({row['column'] for value in original.values()
                             for row in value['target_misses'] if row['point'] == point})
        point_results.append({'point': point, 'missed_components': components,
            'accepted_results': {str(backend): [row for row in rows if row['point'] == point]
                                 for backend, rows in backend_rows.items()},
            'pairwise_residuals': {key: [row for row in comparison['residuals']
                                        if row['point'] == point and row['column'] in components]
                                   for key, comparison in pairwise_full.items()}})

    # Source-derived normalization, not a direct trace of the native coupling.
    pi, alpha_s = 3.141592653589793, .118
    coupling = 2 * math.sqrt(alpha_s*pi)
    factor = -(2*(4/3)+2*3)*coupling**2/(8*pi**2)
    assert all(point['momenta'][:2] == [[500., 0., 0., 500.], [500., 0., 0., -500.]]
               for point in metadata['points'])
    analytic = {}
    for backend, rows in backend_rows.items():
        residuals = []
        for row in rows:
            expected, observed = row['coefficients'][0]*factor, row['coefficients'][3]
            residuals.append({'point': row['point'], 'order': row['order'], 'born': row['coefficients'][0],
                'expected_double_pole': expected, 'observed_double_pole': observed,
                'relative_to_expected': relative(observed, expected),
                'symmetric_relative_error': relative(observed, expected, True)})
        analytic[str(backend)] = {'rows': residuals,
            'undefined_relative_to_expected': sum(row['relative_to_expected'] is None for row in residuals),
            'maximum_relative_to_expected': max((row['relative_to_expected'] for row in residuals
                                                 if row['relative_to_expected'] is not None), default=None),
            'maximum_symmetric_relative_error': max((row['symmetric_relative_error'] for row in residuals
                                                     if row['symmetric_relative_error'] is not None), default=None)}
    return {'schema_version': 1, 'case': CASE, 'campaign': 'split-dp-terminal-1',
        'scope': 'Read-only diagnosis of closed QCD n4 evidence; original comparisons and campaign gates unchanged',
        'checkpoint_sha256': ledger['checkpoint']['identity']['sha256'],
        'performance_eligible': False, 'new_native_evaluations': 0,
        'metrics': {'gate': 'abs(a-b)/abs(a+b)<3e-6, or pole magnitude/finite maximum<3e-6',
                    'ordinary_target': 'same criterion at 1e-8',
                    'raw_relative': 'abs(a-b)/max(abs(a),abs(b))',
                    'columns': ['Born', 'finite', 'single pole', 'double pole']},
        'accepted_usage': usage, 'original_comparisons': original,
        'pairwise_all100': {key: condensed(value) for key, value in pairwise_full.items()},
        'target_points': target_points, 'target_point_results': point_results,
        'target_membership_groups': membership,
        'target_membership_scope': 'Backends missing the ordinary target against baseline CutTools; no truth attribution',
        'higher_precision_adjudication': None,
        'truth_scope': 'No independent finite or single-pole reference supplied; agreement between reducers is not a truth certificate',
        'universal_double_pole': {'formula': '-Born*(2*CF+2*CA)*G**2/(8*pi**2)',
            'alpha_s_card': alpha_s, 'CF': 4/3, 'CA': 3, 'G_reconstructed': coupling,
            'G_binary64_bits': bits(coupling), 'factor': factor, 'scale_gev': 1000.0,
            'born_and_loop_iden': 144, 'broken_symmetry_factor': 1,
            'caveat': 'G is source-reconstructed, not directly recorded; Born uses P_USER before loop momentum repair. No finite or single-pole truth claim.',
            'backends': analytic},
        'fpe_flags_observed': False, 'internal_retries_observed': False,
        'accepted_status_scope': 'Final accepted backend/precision only; no count of internal attempts'}


def main():
    if sys.flags.optimize:
        raise RuntimeError('Verification requires enabled assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if not args.freeze:
        index = read(HERE / 'index.json')
        for name, value in index['files'].items():
            path = HERE / name
            assert not path.is_symlink() and path.resolve().is_relative_to(HERE)
            assert info(path) == value
        assert sorted(index['files']) == sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*')
                                                if p.is_file() and p.name != 'index.json')
    result = collect()
    if args.freeze:
        write(HERE / 'analysis.json', result)
        files = {str(p.relative_to(HERE)): info(p) for p in sorted(HERE.rglob('*'))
                 if p.is_file() and p.name != 'index.json'}
        write(HERE / 'index.json', {'schema_version': 1, 'scope': result['scope'],
            'new_native_evaluations': 0, 'performance_eligible': False, 'files': files})
    else:
        assert result == read(HERE / 'analysis.json')
    print(json.dumps({'status': 'passed', 'target_points': result['target_points'],
                      'analysis_sha256': sha(HERE / 'analysis.json'),
                      'index_sha256': sha(HERE / 'index.json'), 'new_native_evaluations': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
