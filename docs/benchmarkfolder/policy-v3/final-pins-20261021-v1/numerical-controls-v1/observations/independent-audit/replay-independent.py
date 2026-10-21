"""Independent, engine-free arithmetic audit of retained release observations."""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal, localcontext
from collections import Counter
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / '.benchmark-work/final-numerical-release-20261021-v1'


def read(path):
    return json.loads(path.read_text())


def rounded(value):
    if value == 0:
        return value
    sign = 1 if value > 0 else -1
    numerator, denominator = abs(value).as_integer_ratio()
    step = Q(2) ** (numerator.bit_length() - denominator.bit_length() - 112)
    while abs(value) < step * 2**112:
        step /= 2
    while abs(value) >= step * 2**113:
        step *= 2
    numerator, denominator = (abs(value) / step).as_integer_ratio()
    integral, remainder = divmod(numerator, denominator)
    upward = 2 * remainder > denominator or (2 * remainder == denominator and integral % 2 == 1)
    return sign * step * (integral + upward)


def ascii113(text):
    return rounded(Q(text)), text.lstrip().startswith('-')


def show(value):
    with localcontext() as context:
        context.prec = 100
        return str(Decimal(value.numerator) / Decimal(value.denominator))


summary = read(REPORT / 'replay-exact-classification.json')
for name in ('input', 'capture', 'controller', 'exact_checks'):
    record = summary[name]
    assert hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() == record['sha256']
original = [json.loads(line) for line in (ROOT / summary['input']['path']).read_text().splitlines()]
capture = [json.loads(line) for line in (ROOT / summary['capture']['path']).read_text().splitlines()]
assert len(original) == 1392 and len(capture) == 2785
changed = ascii_changes = checked_fields = 0
components, shapes, ascii_rounding = Counter(), Counter(), Counter()
maxima = dict.fromkeys(('laurent', 'CC', 'R1', 'split', 'old_laurent', 'old_R1', 'ascii_rounding'), Q(0))
names = ('finite.re', 'finite.im', 'single.re', 'single.im', 'double.re', 'double.im', 'rational.re', 'rational.im')
for index, (old, low, high) in enumerate(zip(original, capture[:1392], capture[1392:2784])):
    assert low['kind'] == 'replay' and high['kind'] == 'convergence'
    assert low['sequence'] == high['sequence'] == index and low['input'] == high['input'] == old
    observations = [low['observed'], high['observed_384']]
    values = []
    for bits, observed in zip((256, 384), observations):
        assert observed['status'] == 0 and observed['working_bits'] == bits
        assert observed['input_unchanged'] is True and observed['output_bytes'] == 512
        assert len(observed['raw']) == 10 and len(observed['output']) == 8
        assert all(item['finite'] for item in observed['raw'])
        represented = [Q(int(item['numerator']), int(item['denominator'])) for item in observed['raw']]
        values.append(represented)
        scale = max(map(abs, represented[:6]))
        for field, (text, raw_index) in enumerate(zip(observed['output'], (0, 1, 2, 3, 4, 5, 8, 9))):
            actual, sign = ascii113(text)
            assert sign == observed['raw'][raw_index]['negative']
            difference = abs(actual - rounded(represented[raw_index]))
            if scale:
                difference /= scale
            else:
                assert difference == 0
            checked_fields += 1
            if difference:
                ascii_rounding[f'{bits}/{names[field]}'] += 1
                maxima['ascii_rounding'] = max(maxima['ascii_rounding'], difference)
    a, b = values
    scale = max(map(abs, b[:6]))
    for name, indices in (('laurent', range(6)), ('CC', (6, 7)), ('R1', (8, 9))):
        error = max(abs(a[i] - b[i]) for i in indices)
        if scale:
            error /= scale
        else:
            assert error == 0
        assert error <= Q(1, 10**31)
        maxima[name] = max(maxima[name], error)
    defect = max(abs(value[i] - value[6+i] - value[8+i]) for value in values for i in (0, 1))
    if scale:
        defect /= scale
    else:
        assert defect == 0
    assert defect <= Q(1, 10**60)
    maxima['split'] = max(maxima['split'], defect)
    assert [item['negative'] for item in observations[0]['raw']] == [item['negative'] for item in observations[1]['raw']]
    assert high['comparison']['passed'] is True and high['comparison']['sign_changed_indices'] == []
    historical = list(map(ascii113, old['output']))
    current = list(map(ascii113, observations[0]['output']))
    differing = [i for i, (a, b) in enumerate(zip(historical, current)) if a != b]
    assert bool(differing) == (not low['historical_binary128_values_and_zero_signs_equal'])
    changed += bool(differing)
    ascii_changes += old['output'] != observations[0]['output']
    components.update(names[i] for i in differing)
    if differing:
        shapes[f"n{old['n']}-r{old['rank']}"] += 1
    common = max(abs(value[0]) for value in historical[:6] + current[:6])
    deltas = [abs(a[0] - b[0]) for a, b in zip(historical, current)]
    if common:
        maxima['old_laurent'] = max(maxima['old_laurent'], max(deltas[:6]) / common)
        maxima['old_R1'] = max(maxima['old_R1'], max(deltas[6:]) / common)
assert changed == summary['historical_binary128_changed_records'] == 468
assert ascii_changes == summary['historical_ascii_changed_records'] == 482
assert dict(components) == summary['historical_changed_component_counts']
assert dict(shapes) == summary['historical_changed_topology_rank_counts']
for name, field in (('laurent', 'maximum_laurent_scaled_256_384_difference'),
                    ('R1', 'maximum_same_basis_rational_scaled_256_384_difference'),
                    ('old_laurent', 'maximum_historical_complete_laurent_component_scaled_difference'),
                    ('old_R1', 'maximum_historical_basis_rational_component_scaled_difference')):
    assert show(maxima[name]) == summary[field]
process = read(REPORT / 'replay-convergence-final-process.json')
equivalence = read(REPORT / 'committed-source-equivalence.json')
assert process['exit_code'] == 101 and process['source_sha256_before_equals_after'] is True
assert process['bridge_source_sha256'] == process['source_sha256_after'] == equivalence['complete_source45_sha256']
for name, checksum in equivalence['complete_source45_sha256'].items():
    data = subprocess.check_output(['git', 'show', equivalence['committed_revision'] + ':vendor/ninjars/' + name], cwd=ROOT)
    assert hashlib.sha256(data).hexdigest() == checksum
result = {'schema': 'independent-final1392-exact-rational-audit-v1', 'records_each_precision': 1392,
    'all_inputs_and_order_unchanged': True, 'new256_384_laurent_CC_R1_convergence_passed': True,
    'maximum_scaled_errors': {key: show(value) for key, value in maxima.items()},
    'historical_changed_binary128_records': changed, 'historical_changed_ascii_records': ascii_changes,
    'historical_component_changes': dict(components), 'historical_shape_changes': dict(shapes),
    'ASCII_output_fields_examined': checked_fields, 'ASCII_adjacent113bit_rounding_fields': dict(ascii_rounding),
    'ASCII_rounding_scope': '40-digit text can round to the adjacent binary128 value versus exact raw Float rounding; observed effect is far below1e-31. Input113-bit recovery is a separate guarantee.',
    'original_test_exit_code': 101,
    'original_test_failure_retained': 'Strict historical bit identity failed for468 requests. This separately recomputed convergence result does not relabel the Rust test as passed.',
    'historical_outputs_are_oracle': False, 'committed_source_revision': equivalence['committed_revision'],
    'all45_tested_source_files_equal_commit': True,
    'Fortran_semantics': 'Both wrappers return complete NRS_RES Laurent values; their local R1 is not added to RES. Rust result[4] is the separate rational diagnostic, result[3] is CC. Full finite=CC+R1 is independently checked.',
    'scope': 'Focused convergence and historical classification only; core scientific and performance gates remain pending.',
    'new_engine_calls': 0,
    'inputs_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__), REPORT / 'replay-exact-classification.json', REPORT / 'replay-convergence-final-process.json',
        REPORT / 'committed-source-equivalence.json', ROOT / summary['input']['path'], ROOT / summary['capture']['path'],
        ROOT / 'vendor/ninjars/src/lib.rs', ROOT / 'vendor/ninjars/src/final_precision_tests.rs',
        ROOT / 'madgraph/iolibs/template_files/loop_optimized/NINJARS_interface.inc')}}
destination = HERE / 'replay1392-independent-exact-audit.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'audit': str(destination.relative_to(ROOT)),
    'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'maximum_scaled_errors': result['maximum_scaled_errors'], 'records_each_precision':1392}, indent=2))
