"""Strict raw-tensor/transport assertions; no physical truth or timing claim."""
from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('private_tier_telemetry', HERE / 'telemetry.py')
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


def require(value, message):
    if not value:
        raise ValueError(message)


def binary128(text):
    """Exact decimal -> IEEE binary128 round-to-nearest/ties-to-even."""
    decimal = Decimal(text.strip().replace('D', 'E'))
    require(decimal.is_finite(), 'Nonfinite decimal transport')
    sign = int(decimal.is_signed()) << 127
    value = abs(Fraction(decimal))
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = lambda n: Fraction(2**n) if n >= 0 else Fraction(1, 2**-n)
    if value < power(exponent):
        exponent -= 1
    require(exponent <= 16383, 'binary128 overflow')
    exponent = max(exponent, -16382)
    scaled = value / power(exponent - 112)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    quotient += 2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and quotient % 2)
    if quotient == 2**113:
        quotient >>= 1
        exponent += 1
    if quotient < 2**112:
        return sign | quotient
    require(exponent <= 16383, 'binary128 rounded overflow')
    return sign | ((exponent + 16383) << 112) | (quotient - 2**112)


def fraction(bits):
    exponent = (bits >> 112) & 32767
    require(exponent != 32767, 'Nonfinite returned binary128')
    mantissa = bits & (2**112 - 1)
    if exponent:
        mantissa += 2**112
    shift = (exponent or 1) - 16383 - 112
    value = Fraction(mantissa * 2**shift) if shift >= 0 else Fraction(mantissa, 2**-shift)
    return -value if bits >> 127 else value


def records(raw_hex, expected):
    raw = bytes.fromhex(raw_hex)
    require(len(raw) == expected * 64, 'Wrong decimal record count')
    return [binary128(raw[k:k+64].decode('ascii')) for k in range(0, len(raw), 64)]


def word_bits(words):
    require(len(words) % 2 == 0 and all(re.fullmatch('[0-9A-Fa-f]{16}', x) for x in words), 'Malformed binary128 words')
    return [int(words[k], 16) | (int(words[k+1], 16) << 64) for k in range(0, len(words), 2)]


def schedule(action):
    large = 4 <= action <= 9
    return [action, 4 if large else 1, 3 if large else 0, 99 if action == 10 else 0,
            int(not 7 <= action <= 9), int(action in (3, 6, 8)), 1]


def parse_worker(directory, arm):
    lines = iter((directory / 'records.txt').read_text().splitlines())
    def row(tag, count=None):
        values = next(lines, '').split()
        require(values and values.pop(0) == tag, 'Expected ' + tag)
        require(count is None or len(values) == count, 'Wrong field count: ' + tag)
        return values
    require(list(map(int, row('HEADER', 4))) == [1, arm, 113, arm], 'Header/capability mismatch')
    require(list(map(int, row('REGISTER', 4))) == [1, 0, 0, 1], 'Ordinary registration failed')
    if arm:
        require(list(map(int, row('REGISTER', 4))) == [4, 3, 0, 1], 'Captured registration failed')
    actions = {}
    for action in range(1, 11 if arm else 2):
        require(list(map(int, row('ACTION', 7))) == schedule(action), 'Unexpected action stream')
        status = int(row('STATUS', 1)[0])
        require(status == (2 if action == 10 else 0), 'Tensor status mismatch')
        encoded = row('ERRORS', 1)[0]
        require(re.fullmatch('[0-9A-Fa-f]{2048}', encoded), 'Wrong numerical error transport')
        errors = bytes.fromhex(encoded)
        before, after = (part.decode('utf8').rstrip(' ') for part in (errors[:512], errors[512:]))
        require(before == after, 'Telemetry replaced numerical error')
        require(before == ('invalid TIRust pole mode or logical flag' if action == 10 else ''), 'Unexpected numerical error')
        count = 35 if 4 <= action <= 9 else 1
        values = []
        for slot in range(1, 5):
            for coefficient in range(1, count + 1):
                tokens = row('VALUE', 6)
                require(list(map(int, tokens[:2])) == [coefficient, slot], 'Missing/reordered tensor output')
                values.extend(word_bits(tokens[2:]))
        cache = list(map(int, row('CACHE', 7)))
        require(cache[0] == 0 and all(x >= 0 for x in cache), 'Cache query failed')
        require(list(map(int, row('END', 2))) == [action, 0], 'Observer health/END failed')
        actions[action] = {'status': status, 'values': values, 'error': before, 'cache': cache[1:]}
    require(list(map(int, row('CACHE_BOUNDARY', 2))) == [0, 0], 'Cache/budget boundary failed')
    require(list(map(int, row('FINALIZE', 3))) == [0, 1, -1], 'Finalize failed')
    require(list(map(int, row('RECREATE', 2))) == [0, 2], 'Context generation did not advance')
    require(list(map(int, row('DONE', 3))) == [0, 0, -1], 'Final completion failed')
    require(next(lines, None) is None, 'Unexpected trailing records')
    return actions


def analyze(root):
    root = Path(root)
    manifest = json.loads((HERE / 'fixtures/manifest.json').read_text())
    fixture = {key: word_bits(value['raw_words']) for key, value in manifest['fields'].items()}
    def significant_bits(value):
        mantissa=(value & (2**112-1)) + (2**112 if (value>>112)&32767 else 0)
        return (mantissa//(mantissa & -mantissa)).bit_length() if mantissa else 0
    beyond_dp = sum(significant_bits(v) > 53 for key in ('p', 's') for v in fixture[key])
    require(beyond_dp > 0, 'Fixture was reduced to binary64 precision')
    deltas = [
        [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0], [1,0,0,0,0,0,0,1,0],
        [1,1,1,1,0,0,0,0,0], [1,1,0,0,0,0,1,0,0], [1,1,1,1,0,0,0,1,1],
        [1,1,1,1,0,0,0,0,0], [1,1,1,1,0,0,0,1,1], [1,1,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0]]
    results, parsed = {}, {}
    for arm_name, arm in (('legacy', 0), ('adaptive', 1)):
        directory = root / arm_name
        parsed[arm_name] = actions = parse_worker(directory, arm)
        snaps = telemetry.parse_snapshots((directory / 'telemetry.txt').read_text())
        expected_keys = [(n, phase) for n in list(actions) + [101, 102] for phase in ('BEFORE', 'AFTER')]
        require(list(snaps) == expected_keys, 'Snapshot stream missing/extra/reordered')
        observed = [json.loads(x) for x in (directory / 'observer.jsonl').read_text().splitlines()]
        raw_snaps = [json.loads(x) for x in (directory / 'snapshots.jsonl').read_text().splitlines()]
        require(len(observed) == len(actions), 'Native call count mismatch')
        require(len(raw_snaps) == (len(snaps) if arm else 0), 'Raw C snapshot count mismatch')
        for ordinal, snapshot in enumerate(snaps.values()):
            require(snapshot['generation'] == (2 if (snapshot['point'],snapshot['phase']) == (102,'AFTER') else 1), 'Snapshot epoch mismatch')
            require(snapshot['status'] == (0 if arm else 3), 'Snapshot capability/status mismatch')
            if arm:
                raw = raw_snaps[ordinal]
                require(raw['ordinal'] == ordinal and raw['status'] == 0 and raw['count'] == 11, 'C getter metadata mismatch')
                require(raw['words'] == snapshot['words_hex'], 'C -> Fortran eleven-word transport changed bits')
                require(snapshot['telemetry_error'] == '', 'Unexpected getter error')
        serialized = []
        previous_after = None
        if arm:
            require(not any(int(x,16) for x in snaps[1,'BEFORE']['words_hex']), 'Initial context precision counters are not empty')
        for native, (action, data) in zip(observed, actions.items()):
            require([native[k] for k in ('action','n','rank','poles','finite','check','cache')] == schedule(action), 'Native dispatch arguments differ')
            require(native['present'] == 1 and native['health'] == 0 and native['status'] == data['status'], 'Native observer/status failed')
            require(native['flags'][1] == native['flags'][2] and native['rounding'][1] == native['rounding'][2], 'Observer changed floating environment')
            require(bytes.fromhex(native['error']).decode() == data['error'], 'Rust -> Fortran error preservation mismatch')
            expected = fixture if 4 <= action <= 9 else {'p':[0]*4,'s':[0], 'm':[binary128('4'),1<<127], 'scale':[binary128('4')]}
            for key, values in expected.items():
                require(records(native[key], len(values)) == values, 'Input binary128 bits changed: '+key)
            if action == 10:
                require(native['output'] == '' and not any(data['values']), 'Failure exposed native output')
            else:
                require(records(native['output'], len(data['values'])) == data['values'], 'Rust decimal -> Fortran output bits changed')
            for value in data['values']:
                fraction(value)
            before, after = snaps[action,'BEFORE'], snaps[action,'AFTER']
            if arm:
                if previous_after is not None:
                    require(before['words_hex'] == previous_after['words_hex'], 'Counters changed outside scheduled dispatch')
                delta = [int(b,16)-int(a,16) for a,b in zip(before['words_hex'][:9], after['words_hex'][:9])]
                require(delta == deltas[action-1], 'Unexpected mixed-tier/check/cache delta')
                require(after['last_output_bits'] == (256 if action <= 3 else 512), 'Last dispatched tensor tier mismatch')
                require((after['last_eligible_condition'] > 0) == (action >= 4), 'Eligible-refusal condition missing/unexpected')
                if action == 10:
                    require(before['words_hex'] == after['words_hex'], 'Invalid foreign call changed dispatcher counters')
                require(after['native_error_before_query'] == data['error'], 'Snapshot did not capture numerical error')
                previous_after = after
            else:
                delta = None
            serialized.append({'action': action, 'tensor_status': data['status'], 'delta': None if delta is None else dict(zip(telemetry.FIELDS,delta)),
                               'attribution_gap': None if arm else 'getter_unavailable', 'last_dispatched_tensor_bits': after['last_output_bits'] if arm else None,
                               'successes_256': None, 'accepted_amplitude_tier': None})
        require(snaps[101,'BEFORE']['words_hex'] == snaps[101,'AFTER']['words_hex'], 'Cache/budget boundary reset precision counters')
        require(snaps[102,'BEFORE']['generation'] != snaps[102,'AFTER']['generation'], 'Recreation not identified as attribution gap')
        require(not any(int(x,16) for x in snaps[102,'AFTER']['words_hex']), 'New context counters not reset')
        if arm:
            for group in ((1,2,3),(4,5,6),(7,8,9)):
                require(all(actions[k]['values'] == actions[group[0]]['values'] for k in group), 'Cache/check tensor outputs disagree')
            for a,b in zip(actions[4]['values'][:210],actions[7]['values'][:210]):
                av,bv=fraction(a),fraction(b)
                require(abs(av-bv) <= Fraction(1,2**100)*(abs(av)+abs(bv)+1), 'Full/partial poles disagree')
            require(all(fraction(x) == 0 for x in actions[7]['values'][210:]), 'Partial finite ABI placeholder changed')
            # Entries: ordinary256, captured full512, captured partial512.
            require([actions[k]['cache'][2] for k in actions] == [1,1,1,2,2,2,3,3,3,3], 'Full/partial/check cache namespace mismatch')
        results[arm_name] = {'tensor_calls': len(actions), 'snapshots': len(snaps), 'raw_c_snapshots': len(raw_snaps), 'actions': serialized,
                             'cache_boundary_preserves_precision_counters': True, 'recreation_attribution_gap': 'context_generation_changed'}
    require(parsed['legacy'][1]['values'] == parsed['adaptive'][1]['values'], 'Ordinary256 changed between released and adaptive bridges')
    result = {'schema':1,'passed':True,'workers':results,'genuine_binary128_input_components':beyond_dp,
              'scope':'Standalone tensor/Fortran/optional telemetry transport; no amplitude, Born, kinematic-fix or performance qualification.',
              'partial_finite':'unevaluated; zero ABI placeholder only', 'successes_256':None,'accepted_amplitude_tier':None}
    json.dumps(result, allow_nan=False)
    return result
