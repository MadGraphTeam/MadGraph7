#!/usr/bin/env python3
"""Decode one frozen 256-bit refusal; never load native code."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
DECODER = ROOT/'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
SOURCE = ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-native'
DECODER_SHA = '6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac'
SOURCE_SHA = 'c7d185c4991fdfacf5c76710d1790b7ccf31c5c2ec0f9270cc4f72ec3dfdfc32'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path, expected):
    assert sha(path/'evidence-index.json') == expected
    data = json.loads((path/'evidence-index.json').read_text())['files']
    for name, record in data.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p = path/name
        assert not p.is_symlink() and p.stat().st_size == record['bytes'] and sha(p) == record['sha256']


def binary128_round(value):
    if not value: return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = lambda n: Fraction(1 << n) if n >= 0 else Fraction(1, 1 << -n)
    if value < power(exponent): exponent -= 1
    quantum = power(max(exponent - 112, -16494))
    scaled = value/quantum
    q, rem = divmod(scaled.numerator, scaled.denominator)
    if 2*rem > scaled.denominator or (2*rem == scaled.denominator and q % 2): q += 1
    return sign*q*quantum


def main():
    inventory(DECODER, DECODER_SHA); inventory(SOURCE, SOURCE_SHA)
    spec = importlib.util.spec_from_file_location('adaptive_fixture_decoder', DECODER/'decode_trace.py')
    decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
    run = SOURCE/'attempt1/topology-c/backend-9'
    trace = decoder.load_trace(run/'trace.dat', run/'observed.log')
    call = next(c for c in trace['points'][1]['calls'] if c['ctmode'] == 4 and c['group'] == 10)
    assert (call['n'], call['rank'], call['native_status']) == (4, 3, 1)
    output = HERE/'fixtures'; output.mkdir(exist_ok=False)
    fields = {}
    for name, tag in [('p','SHIFTS'), ('s','DISTANCES'), ('m','MASS2_ORDERED'), ('scale','NATIVE_SCALE')]:
        event, = [e for e in call['events'] if e['tag'] == tag]
        values = decoder.decode_payload(event, trace['byte_order'])
        if values and 'real' in values[0]:
            values = [part for value in values for part in (value['real'], value['imag'])]
        records = []
        for component in values:
            exact = decoder.finite_fraction(component)
            with localcontext() as ctx:
                ctx.prec = 100
                text = '-0' if component['negative_zero'] else format(Decimal(exact.numerator)/Decimal(exact.denominator), '.39e')
            assert binary128_round(Fraction(Decimal(text))) == exact
            assert len(text) <= 64
            records.append(text.ljust(64).encode('ascii'))
        path = output/f'group10-{name}.dat'; path.write_bytes(b''.join(records))
        fields[name] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'raw_words': event['words'], 'tag': tag}
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in [DECODER/'evidence-index.json', DECODER/'decode_trace.py',
        SOURCE/'evidence-index.json', run/'execution.json', run/'trace.dat', run/'observed.log']}
    result = {'schema': 1, 'n': 4, 'rank': 3, 'group': 10, 'ctmode': 4, 'point': 2,
              'source_native_status': 1, 'record_width': 64, 'records_parse_at_bits': 113,
              'exact_binary128_roundtrip_checked': True, 'fields': fields, 'bindings': bindings,
              'native_executed': False, 'physics_reference': False}
    (output/'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'fixture': 'N4R3 group10 CT4', 'exact_roundtrip': True, 'native_executed': False}))


if __name__ == '__main__': main()
