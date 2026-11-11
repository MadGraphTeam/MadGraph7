"""Portable ancestry joins and pure replay of the closed single-IR-pole audit."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
SOURCE = '501302116affa5358b566282580d1f172a51499d93684625f27e73190094858e'
PARENT = 'kinematic-invariant-amplitude-qualification-1'
TRACE = 'current-kinematic-trace-runtime-qualification-1'
DOUBLE = 'cms-universal-qed-pole-audit-1'
DEPENDENCIES = {
    PARENT: 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4',
    TRACE: '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
    DOUBLE: '1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77',
}
COMPARISON = '7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d'
WORK = '.tirust-build/kinematic-invariant-amplitude-executor-2/'
OLD_WORK = '.tirust-build/kinematic-provenance-current-executor-1/'
WORKERS = ('cms-p20-mode4-lib1-original', 'cms-p20-mode4-lib9-candidate')
STREAMS = {
    1: '646bcb1e45dd7872b0de2bb26682008aa1438ee8ea13a7e2fa773ab3aad5b39d',
    20: '844b17ac85366c144571236a0af6b3c70d64607e2af0d2fe0232c59a0e7c2ac5',
}
CLOSURES = {
    1: ('ordinary', '56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf'),
    20: ('stress', 'df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e'),
}


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
        require(pin(data(name)) == {key: identity[key] for key in ('bytes', 'sha256')}, 'Origin identity differs')
    return index, data


def parsed_rows(raw):
    rows = []
    for line in raw.decode('ascii').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 10 and all(re.fullmatch('[0-9A-F]{16}', word) for word in fields[5:]),
                'Driver bit schema differs')
        rows.append({**dict(zip(('phase', 'call', 'point', 'order', 'code'), map(int, fields[:5]))),
                     'words': fields[5:]})
    return rows


def fraction128(words):
    require(len(words) == 2 and all(re.fullmatch('[0-9A-F]{16}', w) for w in words), 'Bad binary128 word')
    bits = int(words[0], 16) | int(words[1], 16) << 64
    exponent = (bits >> 112) & 32767
    require(exponent != 32767, 'Nonfinite historical input')
    mantissa = bits & ((1 << 112) - 1)
    if exponent:
        mantissa += 1 << 112
    shift = (exponent or 1) - 16383 - 112
    value = F(mantissa << shift) if shift >= 0 else F(mantissa, 1 << -shift)
    return -value if bits >> 127 else value


def rational(record):
    return F(int(record['numerator']), int(record['denominator']))


def dot(a, b):
    return a[0] * b[0] - sum(a[i] * b[i] for i in (1, 2, 3))


def main():
    index, data = packet(HERE)
    require(index['source_index_sha256'] == SOURCE and index['dependencies'] == DEPENDENCIES, 'Scope differs')
    require(pin(data('source/index.json'))['sha256'] == SOURCE, 'Original source seal differs')
    source = json.loads(data('source/index.json'))
    require(source['file_count'] == len(source['files']) == 28 and source['native_executions'] == 0,
            'Original source inventory differs')
    for name, identity in source['files'].items():
        require(pin(data('source/' + safe(name))) == identity, 'Original source payload differs')
    require({name for name in index['files'] if name.startswith('source/')} ==
            {'source/index.json'} | {'source/' + name for name in source['files']}, 'Original source set differs')
    # Authenticate all declared public ancestors before importing calculations
    # or trusting rows, source identities, card identities, or trace metadata.
    ancestors = {name: packet(HERE.parent / name, seal) for name, seal in DEPENDENCIES.items()}
    parent_index, parent = ancestors[PARENT]
    trace_index, trace = ancestors[TRACE]
    _, double = ancestors[DOUBLE]
    require(data('source/trace-parent-index.json') == (HERE.parent / TRACE / 'evidence-index.json').read_bytes(),
            'Historical trace index differs from the public ancestor')
    old_index = json.loads(data('source/double-pole/index.json'))
    require(pin(data('source/double-pole/index.json'))['sha256'] ==
            '5748eeb64f53f4a201ae0e5a355221c03e5dac39790d55531e52b0d290e057e3', 'Model source seal differs')
    for name in ['index.json', *old_index['files']]:
        require(data('source/double-pole/' + name) == double('source/' + name), 'Published model/card source differs')

    comparison_raw = parent('results/root-point20-comparison.json')
    require(pin(comparison_raw)['sha256'] == COMPARISON, 'Current comparison differs')
    comparison = json.loads(comparison_raw)
    closure_raw = parent('results/stress-closure.json')
    require(pin(closure_raw)['sha256'] == parent_index['closed_stages']['stress'], 'Current stress closure differs')
    closure = json.loads(closure_raw)
    require(comparison['candidate_gate_passed'] is False and closure['passed'] is False and
            comparison['subset_scope']['closed_parent_stress_sha256'] == pin(closure_raw)['sha256'],
            'Parent failed gate or incomplete stress scope was erased')
    observed = json.loads(data('source/double-pole/observations.json'))
    require(observed['source'] == WORK + 'results/root-point20-comparison.json' and
            observed['source_sha256'] == COMPARISON, 'Observation ancestry differs')
    expected = []
    for worker in WORKERS:
        observation = comparison['observations'][worker]
        require(observation['eligible'] and observation['failures'] == [] and observation['protocol_complete'],
                'Selected current worker is not eligible')
        logical = 'results/runs/' + worker + '/driver-bits.dat'
        raw = parent(logical)
        require(pin(raw) == observation['source_evidence']['driver-bits.dat'] == closure['payloads'][logical[8:]],
                'Current bit file is not joined to its closure')
        rows = parsed_rows(raw)
        require(rows == observation['raw_rows'] and
                [(r['phase'], r['call'], r['point'], r['order'], r['code']) for r in rows] ==
                [(0, 1, 1, 0, 120), (0, 1, 1, 1, 120), (1, 2, 20, 0, 120), (1, 2, 20, 1, 120)],
                'Current word/phase/call/point/order/code coverage differs')
        expected.extend({'worker': worker, 'eligible': True, **row} for row in rows)
    require(observed['rows'] == expected and len(expected) == 8, 'Copied current observations differ')
    trial_raw = data('source/trial-specification.json')
    require(trial_raw == parent('results/runs/' + WORKERS[1] + '/specification.json'), 'Trial specification differs')
    trial = json.loads(trial_raw)
    physical, numerical = data('source/double-pole/param_card.dat'), data('source/MadLoopParams.dat')
    require(physical == parent('proposal/inputs/cms-param_card.dat') == trace('inputs/cms-param_card.dat') and
            numerical == parent('proposal/inputs/cms-backend9-mode4-MadLoopParams.dat') ==
            trace('inputs/cms-mode4-MadLoopParams.dat') and trial['physical_card'] == pin(physical) and
            trial['numerical_card'] == pin(numerical), 'Physical/numerical cards differ across captures')
    for worker in WORKERS:
        specification = json.loads(parent('results/runs/' + worker + '/specification.json'))
        require(specification['physical_card'] == pin(physical) and specification['bootstrap_pin'] ==
                pin(data('source/point1.dat')) and specification['point_pin'] == pin(data('source/point20.dat')),
                'Current worker physical points/cards differ')
    stream_fingerprints = {}
    for point in (1, 20):
        raw = data(f'source/point{point}.dat')
        require(raw == parent(f'proposal/inputs/cms-point{point}.dat') == trace(f'inputs/cms-point{point}.dat'),
                'Saved point stream differs across public ancestors')
        tokens = raw.decode('ascii').split()
        require(tokens[0] == '1' and len(tokens) == 17, 'Saved point stream shape differs')
        words = b''.join(struct.pack('>d', float(token)) for token in tokens[1:])
        stream_fingerprints[str(point)] = pin(words)['sha256']
        require(stream_fingerprints[str(point)] == STREAMS[point], 'Reconstructed binary64 point stream differs')

    pins = json.loads(data('source/source-pins.json'))
    for entry in json.loads(data('source/source-excerpts.json')).values():
        require(entry['path'] in pins, 'Unpinned source excerpt')
        for section in entry['ranges']:
            require(pin(section['text'].encode())['sha256'] == section['sha256'], 'Source excerpt differs')
    original_build = json.loads(parent('results/builds/original/build.json'))
    generated = {name: identity for name, identity in pins.items() if name in original_build['sources']}
    require(len(generated) == 4 and all(original_build['sources'][name] == identity for name, identity in generated.items()),
            'Generated momentum/counterterm source identity differs')
    require(pins[WORK + 'results/runs/' + WORKERS[1] + '/specification.json'] == pin(trial_raw), 'Source trial pin differs')

    retained = json.loads(data('source/trace-retention.json'))
    require(len(retained) == 6, 'Historical trace retention scope differs')
    history = {}
    for logical, binding in retained.items():
        stored = data('source/' + binding['local'])
        require(stored == trace(binding['parent_storage']) and pin(stored) == trace_index['files'][binding['parent_storage']],
                'Copied historical trace storage differs')
        raw = gzip.decompress(stored)
        require(raw == trace(logical) and pin(raw) == {'bytes': binding['raw_bytes'], 'sha256': binding['raw_sha256']} and
                trace_index['origins'][logical] == {'path': binding['origin_path'], **pin(raw)},
                'Copied historical trace origin differs')
        history[logical] = raw
    for point, (stage, seal) in CLOSURES.items():
        raw = history['results/' + stage + '-terminal-closure.json']
        require(pin(raw)['sha256'] == seal, 'Historical closed stage differs')
        closed = json.loads(raw)
        require(closed['stage']['passed'] is (point == 1) and closed['terminal_exit_code'] == (0 if point == 1 else 1),
                'Historical stress failure erased')
        name = f'cms-point{point}-mode4-enabled'
        worker = closed['workers'][name]
        require(worker['actual_mode'] == 4 and worker['native_returncode'] == 0 and worker['outer_entries_observed'] == 2,
                'Historical input capture scope differs')
        for suffix in ('trace.dat', 'specification.json'):
            logical = 'results/runs/' + name + '/' + suffix
            require(worker['files'][OLD_WORK + logical] == pin(history[logical]), 'Historical closure/worker-file join differs')
        require(worker['trace_pin'] == pin(history['results/runs/' + name + '/trace.dat']), 'Historical trace pin differs')
        specification = json.loads(history['results/runs/' + name + '/specification.json'])
        require(specification['physical_card'] == pin(physical) and specification['numerical_card'] == pin(numerical) and
                specification['point_pin'] == pin(data(f'source/point{point}.dat')) and
                specification['bootstrap_pin'] == pin(data('source/point1.dat')) and
                specification['request']['point'] == point and specification['request']['requested_mode'] == 4 and
                specification['request']['trace_enabled'], 'Historical card/point/precision scope differs')

    spec = importlib.util.spec_from_file_location('retained_single_IR_calculation', HERE / 'source/calculate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.calculate()
    require(result == json.loads(data('source/result.json')), 'Retained single-pole calculation differs')
    require(result['native_executions'] == 0 and result['all_outgoing_charges'] == [-1, 1, 1, -1] and
            all(result[key] is False for key in ('model_runtime_coupling_bits_observed',
                'current_trial_new_external_word_capture', 'finite_truth_qualified', 'full_amplitude_truth_qualified',
                'checked_stability_qualified', 'performance_qualified', 'UV_cancellation_independently_measured',
                'failed_tensor_outputs_read')), 'Pole calculation overstates scope')
    require([p['point'] for p in result['points']] == [1, 20], 'Calculated point coverage differs')
    residuals = {}
    with localcontext() as context:
        context.prec = 90
        alpha = D(result['alpha_reconstructed'])
        require(str(alpha) == json.loads(double('source/result.json'))['alpha_reconstructed'], 'Alpha model ancestry differs')
        pi = D('3.1415926535897932384626433832795')
        dec = lambda value: D(value.numerator) / D(value.denominator)
        for calculated in result['points']:
            point, phase = calculated['point'], calculated['phase']
            require((point, phase) in ((1, 0), (20, 1)), 'Calculated phase differs')
            raw = history[f'results/runs/cms-point{point}-mode4-enabled/trace.dat']
            lines = raw.decode('ascii').splitlines()
            require(lines[0].split() == ['KT_SCHEMA_V2', str(point), '64', '128', '64', '32'], 'Trace ABI differs')
            inputs = calculated['captured_inputs']
            require(inputs['context'] == [point, 3, 3, 4, 2, 4, 1] and inputs['external_lines'] == [8, 9, 10, 11] and
                    inputs['native_input_line'] == (70 if point == 1 else 74), 'Historical selected context differs')
            vectors = []
            for leg, (line, expected_words) in enumerate(zip(inputs['external_lines'], inputs['external_complex_words']), 1):
                fields = lines[line - 1].split(); header = list(map(int, fields[1:11]))
                require(fields[0] == 'KT_EXTERNAL_QP' and fields[11:] == expected_words and len(expected_words) == 16 and
                        [header[i] for i in (0, 1, 2, 3, 4, 5, 9)] == inputs['context'] and header[6] == leg,
                        'Historical external leaf words/header differ')
                require(all(fraction128(expected_words[i+2:i+4]) == 0 for i in range(0, 16, 4)), 'Nonreal external leaf')
                vectors.append([fraction128(expected_words[i:i+2]) for i in range(0, 16, 4)])
            fields = lines[inputs['native_input_line'] - 1].split(); header = list(map(int, fields[1:11]))
            require(fields[0] == 'KT_TIRUST_QP' and len(fields[11:]) == 82 and fields[-2:] == inputs['native_mu_squared_words'] and
                    [header[i] for i in (0, 1, 2, 3, 4, 5, 9)] == inputs['context'], 'Historical scale words/header differ')
            mu2 = fraction128(fields[-2:])
            require(mu2 == rational(calculated['native_mu_squared']) == 1000000 and
                    vectors == [[rational(value) for value in row] for row in calculated['all_outgoing_vectors']],
                    'Exact historical input decoding differs')
            require([dot(v, v) for v in vectors] == [rational(v) for v in calculated['external_shell_residuals']] and
                    [sum(v[i] for v in vectors) for i in range(4)] ==
                    [rational(v) for v in calculated['external_conservation_residual']], 'Input residuals differ')
            pairs = {(i, j): abs(2 * dot(vectors[i-1], vectors[j-1])) for i in range(1, 5) for j in range(i+1, 5)}
            require(all(value > 0 for value in pairs.values()), 'Fixed noncollinear formula unavailable')
            require({tuple(row['legs']): rational(row['positive_pair_invariant']) for row in calculated['pair_terms']} == pairs,
                    'Exact pair invariants differ')
            ratio = pairs[1, 2] * pairs[1, 3] * pairs[2, 4] * pairs[3, 4] / (pairs[1, 4] * pairs[2, 3] * mu2 ** 2)
            factor = -D(6) + 2 * dec(ratio).ln()
            require(abs(factor - D(calculated['single_factor'])) < D('1e-60'), 'Independent factored pole differs')
            charges = [-1, 1, 1, -1]
            for multiplier in (F(1), F(13, 7), F(7, 13)):
                qes = mu2 * multiplier
                fks = -D(6) - 2 * sum(charges[i-1] * charges[j-1] * dec(value/qes).ln() for (i, j), value in pairs.items())
                fks -= 4 * dec(mu2/qes).ln()
                require(abs(fks - factor) < D('1e-80'), 'Independent FKS auxiliary scale cancellation differs')
            rows = [row for row in expected if row['phase'] == phase]
            require(len(calculated['comparisons']) == len(rows) == 4, 'Pole comparison coverage differs')
            for row, answer in zip(rows, calculated['comparisons']):
                require(answer['worker'] == row['worker'] and answer['order'] == row['order'], 'Pole row identity differs')
                born = D.from_float(struct.unpack('>d', bytes.fromhex(row['words'][0]))[0])
                single = D.from_float(struct.unpack('>d', bytes.fromhex(row['words'][2]))[0])
                pole = born * alpha / (2 * pi) * factor
                residual = abs((single - pole) / pole)
                require(abs(pole - D(answer['expected_IR_single_pole'])) < D('1e-60') and
                        abs(residual - D(answer['relative_residual'])) < D('1e-60'), 'Independent pole/residual differs')
                if row['order'] == 0:
                    residuals[f'{point}:' + row['worker']] = answer['relative_residual']
    print(json.dumps({'passed': True, 'original_payloads': 28, 'current_rows_joined_to_comparison_and_raw_bits': 8,
        'historical_trace_payloads_joined': 6, 'historical_input_contexts': 2,
        'reconstructed_saved_stream_sha256': stream_fingerprints, 'current_candidate_leaf_bits_observed': False,
        'old_stress_failure_preserved': True, 'independent_factored_formula_and_three_QES_checks': True,
        'card_binary64_parsing_boundary': 'reconstruction assumption', 'runtime_coupling_bits_observed': False,
        'UV_cancellation_independently_measured': False, 'fresh_all_source_reparse': False,
        'finite_full_amplitude_accuracy_qualified': False, 'checked_stability_qualified': False,
        'production_candidate_promoted': False, 'new_native_executions': 0, 'external_binary_reads': 0,
        'single_pole_relative_residuals': residuals}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
