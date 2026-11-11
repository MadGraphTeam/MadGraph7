"""Verify retained synthetic bits with exact arithmetic, without native execution."""
import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_SEAL = '1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964'
KERNEL_SHA = '0cb41b0c10233027d65c85f63625556eefb819ce2570390447bdceff4d8f7088'
REFERENCE_SHA = '7b52f707d16b5e248160b8fb7df0a162be41285d02a4cea4d5ae1fe2bac22de1'
ZERO = -1000000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def power(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def floor2(value):
    value = abs(value)
    require(value != 0, 'floor2 needs nonzero value')
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - int(value < power(exponent))


def decode(word, width):
    if width not in (64, 128) or word >= 1 << width:
        return None
    fraction_bits, exponent_bits, bias = (52, 11, 1023) if width == 64 else (112, 15, 16383)
    exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return None
    mantissa = (word & ((1 << fraction_bits) - 1)) + ((1 << fraction_bits) if exponent else 0)
    return (-1 if word >> (width - 1) else 1) * mantissa * power((exponent or 1) - bias - fraction_bits)


def upper_power_sum(terms):
    """Power-of-two enclosure matching the documented conservative aggregation."""
    positive = [term for term in terms if term]
    require(all(term > 0 and term == power(floor2(term)) for term in positive), 'Not a positive binary bound')
    return max(positive) * power((len(positive) - 1).bit_length()) if positive else Fraction(0)


def replay(inputs, output, reference_source):
    # Import no retained module: select only five pure arithmetic functions from
    # the sealed reference text. Controller/main and DAG helpers never execute.
    require(pin(reference_source)['sha256'] == REFERENCE_SHA, 'Reference source changed')
    selected = {'pow2', 'floor2', 'psum', 'certificate', 'exact_error_bound'}
    nodes = [n for n in ast.parse(reference_source).body if isinstance(n, ast.FunctionDef) and n.name in selected]
    require({n.name for n in nodes} == selected, 'Reference function selection changed')
    ns = {'F': Fraction, 'ZERO': ZERO}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'sealed-pure-reference', 'exec'), ns)
    rows, results = inputs.decode('ascii').splitlines(), output.decode('ascii').splitlines()
    require(len(rows) == len(results) == 4287, 'Expected 4287 retained cases')
    counts = Counter()
    seeds = {64: set(), 128: set()}
    accepted_seeds = {64: set(), 128: set()}
    exception_controls = {64: set(), 128: set()}
    for number, (line, result) in enumerate(zip(rows, results), 1):
        fields = list(map(int, line.split()))
        require(len(fields) == 16, 'Input shape differs')
        identifier, width, cls, mass, have, env, seed = fields[:7]
        errors = fields[7:11]
        words = fields[11:]
        require(identifier == number and 0 <= seed < 32 and all(-(1 << 127) <= w < (1 << 127) for w in words),
                'Input identity/seed/int128 range differs')
        bits = [word % (1 << 128) for word in words]
        tokens = result.split()
        require(len(tokens) == 3 and tokens[0] == str(number) and tokens[1] in ('T', 'F') and tokens[2] == 'T',
                'Native result or sticky-flag comparison differs')
        actual = tokens[1] == 'T'
        valid = (width in (64, 128) and cls == 2 and mass == 0 and have != 0 and env != 0 and
                 all(e == ZERO or -100000 <= e <= 100000 for e in errors))
        values = [decode(word, width) for word in bits] if width in (64, 128) else [None] * 5
        valid = valid and all(value is not None for value in values)
        expected = False
        if valid:
            q, raw = values[:4], values[4]
            precision, emin = (53, -1022) if width == 64 else (113, -16382)
            e = [Fraction(0) if value == ZERO else power(value) for value in errors]
            upper = [power(floor2(value) + 1) if value else Fraction(0) for value in q]
            terms = [error * error for error in e if error]
            terms += [2 * value * error for value, error in zip(upper, e) if value and error]
            squares = upper_power_sum([value * value for value in upper])
            if squares:
                terms.append(8 * power(-precision) * squares)
            terms.append(16 * power(emin - precision))
            threshold = upper_power_sum(terms)
            expected = raw < 0 and power(floor2(raw)) > threshold
            reference = ns['certificate'](q, raw, errors, precision, emin)
            require(expected == reference, 'Independent bound differs from sealed reference')
            exact = sum(2 * abs(value) * error + error * error for value, error in zip(q, e))
            exact += 8 * power(-precision) * sum(value * value for value in q) + 16 * power(emin - precision)
            require(exact == ns['exact_error_bound'](q, errors, precision, emin) and threshold >= exact,
                    'Conservative enclosure failed')
            if actual:
                require(raw < 0 and -raw > exact, 'Accepted case violates exact conditional enclosure')
                counts['accepted_exact_enclosures'] += 1
            counts['valid_finite_contract_inputs'] += 1
            if q == [0, 1, 0, 0] and raw == -1 and errors == [-100] * 4:
                exception_controls[width].add(seed)
        require(actual == expected, 'Native decision differs from independent/reference expectations')
        counts['accepted' if actual else 'rejected'] += 1
        counts['width_' + str(width)] += 1
        if width in seeds:
            seeds[width].add(seed)
            if actual:
                accepted_seeds[width].add(seed)
    require(counts['accepted'] == counts['accepted_exact_enclosures'] == 1572 and counts['rejected'] == 2715,
            'Decision totals differ')
    require(all(value == set(range(32)) for group in (seeds, accepted_seeds, exception_controls) for value in group.values()),
            'Not all 32 seeds reached accepted controls in both widths')
    return {'cases': len(rows), **dict(counts), 'exact_enclosure_false_positives': 0,
            'reference_mismatches': 0, 'retained_sticky_flag_comparison_failures': 0,
            'accepted_controls_cover_32_seeds_per_width': True,
            'native_executions': 0, 'external_binary_reads': 0,
            'source_dag_or_caller_qualified': False, 'amplitude_accuracy_qualified': False}


def main():
    index = json.loads((HERE / 'evidence-index.json').read_bytes())
    all_paths = list(HERE.rglob('*'))
    require(not any(path.is_symlink() for path in all_paths), 'Symlink in portable packet')
    require({str(path.relative_to(HERE)) for path in all_paths if path.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Portable file set differs')
    for name, identity in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe packet path')
        raw = (HERE / name).read_bytes()
        require(pin(raw) == identity, 'Stored file identity differs: ' + name)
        decoded = gzip.decompress(raw) if name.endswith('.gz') else raw
        decoded.decode('utf-8')
        require(b'\0' not in decoded and not decoded.startswith((b'\x7fELF', b'!<arch>')), 'Native binary retained')

    def data(name):
        compressed = index['compressed_text'].get(name)
        stored = compressed['path'] if compressed else name
        require(stored in index['files'], 'Unindexed logical payload')
        raw = (HERE / stored).read_bytes()
        if compressed:
            raw = gzip.decompress(raw)
            require(pin(raw) == {'bytes': compressed['raw_bytes'], 'sha256': compressed['raw_sha256']}, 'Raw gzip identity differs')
        return raw

    for name, identity in index['origins'].items():
        require(pin(data(name)) == {k: identity[k] for k in ('bytes', 'sha256')}, 'Retained origin identity differs')
    require(pin(data('source/index.json'))['sha256'] == SOURCE_SEAL and
            pin(data('certificate.f90'))['sha256'] == KERNEL_SHA, 'Sealed source/kernel differs')
    source = json.loads(data('source/index.json'))
    require(pin(data('certificate.f90')) == source['files']['certificate.f90'] and
            pin(data('source/certificate_reference.py')) == source['files']['certificate_reference.py'], 'Source payload pin differs')
    review = json.loads(data('independent-review.json'))
    require(pin(data('independent-review.json'))['sha256'] ==
            '8b887fc6d9f24f2585cdcbe658401f0bae8f4e4343e323da0a55e68b6e63f281' == index['review_sha256'],
            'Independent review changed')
    work_prefix = '.tirust-build/spacelike-kernel-native-qualification-1/'
    source_prefix = '.tirust-build/kinematic-invariant-general-proposal-1/'
    expected_origins = {'independent-review.json': work_prefix + 'independent-review.json'}
    for name, identity in review['reviewed_files'].items():
        require(pin(data(name)) == identity, 'Reviewed synthetic payload changed')
        expected_origins[name] = work_prefix + name
    for name, identity in review['source_provenance'].items():
        require(pin(data('source/' + name)) == identity, 'Reviewed reference provenance changed')
        expected_origins['source/' + name] = source_prefix + name
    require({name: identity['path'] for name, identity in index['origins'].items()} == expected_origins,
            'Retained origins differ from explicit reviewed selection')
    summary = json.loads(data('summary.json'))
    require(summary['kernel_sha256'] == KERNEL_SHA and summary['input_sha256'] == pin(data('inputs.txt'))['sha256'] and
            summary['tensor_calls'] == 0 and not summary['source_dag_bound_factory_qualified'] and
            not summary['amplitude_accuracy_qualified'] and not summary['performance_measurement'], 'Summary identity/scope differs')
    commands = json.loads(data('commands.json'))
    require(len(commands) == 3 and all(c['returncode'] == 0 for c in commands), 'Recorded command failed')
    require(commands[0]['argv'] == [summary['compiler'], '--version'] and commands[1]['argv'] ==
            [summary['compiler'], '-O2', '-Wall', '-Wextra', '-fcheck=all', '-fno-fast-math',
             'certificate.f90', 'check.f90', '-o', 'check'] and len(commands[2]['argv']) == 1 and
            commands[2]['argv'][0].endswith('/spacelike-kernel-native-qualification-1/check'), 'Compiler/run command differs')
    empty = pin(b'')['sha256']
    require(commands[0]['stdout_sha256'] == pin(data('compiler-version.txt'))['sha256'] and
            commands[0]['stderr_sha256'] == empty and data('compiler-version.txt').startswith(b'GNU Fortran (GCC) 15.2.0\n'),
            'Compiler version transcript differs')
    for idx, prefix in [(1, 'build'), (2, 'run')]:
        for channel in ('stdout', 'stderr'):
            require(commands[idx][channel + '_sha256'] == pin(data(prefix + '.' + channel))['sha256'], 'Command output pin differs')
    result = replay(data('inputs.txt'), data('run.stdout'), data('source/certificate_reference.py'))
    for key in ('cases', 'accepted', 'rejected', 'exact_enclosure_false_positives', 'reference_mismatches'):
        require(result[key] == summary[key], 'Summary numerical accounting differs')
    require(summary['ieee_flag_changes'] == 0 and summary['all_32_flag_combinations'] and summary['widths'] == [64, 128],
            'Summary flag/width scope differs')
    require(result == review['pure_replay'], 'Independent review replay result differs')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
