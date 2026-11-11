"""Portable double-pole calculation and exact joins to its closed parent packet."""
import gzip
from decimal import Decimal as D, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
SOURCE = '5748eeb64f53f4a201ae0e5a355221c03e5dac39790d55531e52b0d290e057e3'
PARENT = 'kinematic-invariant-amplitude-qualification-1'
PARENT_SEAL = 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4'
COMPARISON = '7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d'
WORK = '.tirust-build/kinematic-invariant-amplitude-executor-2/'
WORKERS = ('cms-p20-mode4-lib1-original', 'cms-p20-mode4-lib9-candidate')


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe packet path')
    return name


def packet(directory, expected=None):
    raw = (directory / 'evidence-index.json').read_bytes()
    require(expected is None or pin(raw)['sha256'] == expected, 'Dependency seal differs')
    index = json.loads(raw)
    paths = list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in paths), 'Symlink in packet')
    require({str(path.relative_to(directory)) for path in paths if path.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set differs')
    for name, identity in index['files'].items():
        raw = (directory / safe(name)).read_bytes()
        require(pin(raw) == identity, 'Stored file differs: ' + name)
        text = gzip.decompress(raw) if name.endswith('.gz') else raw
        require(b'\0' not in text and not text.startswith((b'\x7fELF', b'!<arch>')), 'Native payload retained')
        text.decode('utf-8')

    def data(name):
        name = safe(name)
        compressed = index.get('compressed_text', {}).get(name)
        if compressed:
            require(compressed['path'] in index['files'], 'Unindexed compressed file')
            raw = gzip.decompress((directory / safe(compressed['path'])).read_bytes())
            require(pin(raw) == {'bytes': compressed['raw_bytes'], 'sha256': compressed['raw_sha256']},
                    'Decompressed file differs')
            return raw
        require(name in index['files'], 'Unindexed file: ' + name)
        return (directory / name).read_bytes()

    for name, identity in index.get('origins', {}).items():
        require(pin(data(name)) == {key: identity[key] for key in ('bytes', 'sha256')},
                'Retained origin identity differs')
    return index, data


def main():
    index, data = packet(HERE)
    require(index['source_index_sha256'] == SOURCE and index['dependencies'] == {PARENT: PARENT_SEAL},
            'Audit source/dependency scope differs')
    require(pin(data('source/index.json'))['sha256'] == SOURCE, 'Original source seal differs')
    source = json.loads(data('source/index.json'))
    require(source['file_count'] == len(source['files']) == 8 and source['native_executions'] == 0,
            'Original source scope differs')
    for name, identity in source['files'].items():
        require(pin(data('source/' + safe(name))) == identity, 'Original payload differs')
    require({name for name in index['files'] if name.startswith('source/')} ==
            {'source/index.json'} | {'source/' + name for name in source['files']}, 'Original source file set differs')

    parent_index, parent = packet(HERE.parent / PARENT, PARENT_SEAL)
    comparison_raw = parent('results/root-point20-comparison.json')
    require(pin(comparison_raw)['sha256'] == COMPARISON, 'Closed comparison differs')
    comparison = json.loads(comparison_raw)
    require(comparison['subset_scope']['closed_parent_stress_sha256'] == parent_index['closed_stages']['stress'] and
            comparison['subset_scope']['not_a_replacement_full_stress_comparison'] and
            comparison['candidate_gate_passed'] is False and comparison['independent_accuracy_qualified'] is False,
            'Parent failure/scope was erased')
    observed = json.loads(data('source/observations.json'))
    require(observed['source'] == WORK + 'results/root-point20-comparison.json' and
            observed['source_sha256'] == COMPARISON, 'Observation ancestry differs')
    expected = []
    for worker in WORKERS:
        observation = comparison['observations'][worker]
        require(observation['eligible'] and observation['failures'] == [] and
                observation['protocol_complete'] and observation['request']['requested_mode'] == 4 and
                observation['request']['point'] == 20, 'Selected worker was not eligible fixed-QP evidence')
        rows = observation['raw_rows']
        require([(row['phase'], row['call'], row['point'], row['order'], row['code']) for row in rows] ==
                [(0, 1, 1, 0, 120), (0, 1, 1, 1, 120), (1, 2, 20, 0, 120), (1, 2, 20, 1, 120)],
                'Selected phase/call/point/order/code coverage differs')
        expected.extend({'worker': worker, 'eligible': observation['eligible'], **row} for row in rows)
    require(observed['rows'] == expected and len(expected) == 8,
            'Observation words/phase/call/point/order/code do not match the parent')

    card = data('source/param_card.dat')
    require(card == parent('proposal/inputs/cms-param_card.dat'), 'Actual physical card differs')
    pins = json.loads(data('source/source-pins.json'))
    require(pins[observed['source']] == pin(comparison_raw) and
            pins[WORK + 'work/builds/cms/original/Cards/param_card.dat'] == pin(card),
            'Source/card ancestry identity differs')
    original_build = json.loads(parent('results/builds/original/build.json'))
    require(original_build['physical_card'] == pin(card), 'Built physical card differs')
    generated = {name: identity for name, identity in pins.items() if name in original_build['sources']}
    require(len(generated) == 7 and all(original_build['sources'][name] == identity
                                      for name, identity in generated.items()),
            'Historical generated-source inventory differs')
    for worker in WORKERS:
        specification = json.loads(parent('results/runs/' + worker + '/specification.json'))
        require(specification['physical_card'] == pin(card) and specification['request']['physical_card'] ==
                'inputs/cms-param_card.dat', 'Selected worker physical card differs')

    spec = importlib.util.spec_from_file_location('retained_universal_double_pole', HERE / 'source/calculate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.calculate()
    require(result == json.loads(data('source/result.json')), 'Pure double-pole recalculation differs')
    require(result['native_executions'] == 0 and result['decimal_working_digits'] == 70 and
            result['massless_legs'] == result['sum_charge_squared'] == 4 and
            all(result[key] is False for key in ('model_runtime_coupling_bits_observed',
                'complete_Fortran_rounding_emulated', 'independent_full_amplitude_accuracy_qualified',
                'finite_or_single_pole_truth_qualified', 'checked_stability_qualified',
                'performance_qualified', 'cuttools_specific_scalar_cause_proven')),
            'Recalculation overstates its scope')
    # Independent algebra for the modulus, using the same explicitly assumed
    # binary64 external-card parsing boundary and retained model pi literal.
    values = module.card_values(card.decode('ascii'))
    with localcontext() as context:
        context.prec = 90
        pi = D('3.1415926535897932384626433832795')
        mw, mz = values['MASS', 24], values['MASS', 23]
        w = (mw * mw, -mw * values['WIDTH', 24])
        z = (mz * mz, -mz * values['WIDTH', 23])
        modulus_squared = ((w[0] ** 2 + w[1] ** 2) *
            ((z[0] - w[0]) ** 2 + (z[1] - w[1]) ** 2) / (z[0] ** 2 + z[1] ** 2))
        alpha = D(2).sqrt() * values['SMINPUTS', 2] * modulus_squared.sqrt() / pi
        require(abs(alpha - D(result['alpha_reconstructed'])) < D('1e-65'), 'Independent alpha algebra differs')
        require(len(result['comparisons']) == len(expected), 'Calculation row coverage differs')
        for row, answer in zip(expected, result['comparisons']):
            for key in ('worker', 'phase', 'point', 'order'):
                require(answer[key] == row[key], 'Calculation row identity differs')
            born = D.from_float(struct.unpack('>d', bytes.fromhex(row['words'][0]))[0])
            double = D.from_float(struct.unpack('>d', bytes.fromhex(row['words'][3]))[0])
            pole = -D(2) * born * alpha / pi
            residual = abs((double - pole) / pole)
            require(abs(pole - D(answer['expected_four_massless_unit_charge_legs'])) < D('1e-65') and
                    abs(residual - D(answer['relative_residual'])) < D('1e-65'),
                    'Independent double-pole/residual algebra differs')
    print(json.dumps({'passed': True, 'original_source_files': 8, 'parent_observation_rows_exactly_joined': 8,
        'physical_card_byte_equal': True, 'historical_generated_source_identities_joined': 7,
        'independent_modulus_algebra_replayed': True, 'card_binary64_parsing_boundary': 'reconstruction assumption',
        'fresh_full_source_reparse': False, 'runtime_coupling_bits_observed': False,
        'new_native_executions': 0, 'external_binary_reads': 0,
        'finite_single_full_amplitude_accuracy_qualified': False, 'candidate_promoted': False,
        'point20_double_pole_relative_residuals': {row['worker']: row['relative_residual']
            for row in result['comparisons'] if row['phase'] == 1 and row['order'] == 0}}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
