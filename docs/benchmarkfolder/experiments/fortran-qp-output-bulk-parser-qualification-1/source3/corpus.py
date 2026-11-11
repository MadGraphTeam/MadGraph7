"""Deterministic, compact public-parser corpus; integer-only binary128 oracle."""
from fractions import Fraction
from pathlib import Path
import re

WIDTH = 64


def record(text):
    value = text.encode('ascii') if isinstance(text, str) else bytes(text)
    if len(value) > WIDTH:
        raise ValueError('record exceeds transport width')
    return value.ljust(WIDTH, b' ').hex()


def canonical_integer(value):
    sign = '-' if value < 0 else ''
    digits = str(abs(value))
    assert 1 <= len(digits) <= 39 and value
    return sign + digits[0] + '.' + digits[1:].ljust(38, '0') + 'e' + str(len(digits)-1)


def canonical(text):
    """Independent grammar mirror; source tests bind this to Fortran fixtures."""
    data = bytes.fromhex(text)
    if len(data) != WIDTH:
        return False
    if data.rstrip(b' ') in (b'0', b'-0'):
        return True
    m = re.fullmatch(rb'-?[1-9]\.[0-9]{38}e(-?)(0|[1-9][0-9]{0,3}) *', data)
    return bool(m and not (m[1] and m[2] == b'0') and int(m[2]) <= 4900)


def count_for_rank(rank):
    return (rank+1)*(rank+2)*(rank+3)*(rank+4)//24


def records(case):
    size = 8*count_for_rank(case['rank'])
    values = [case['pattern_hex'][i % len(case['pattern_hex'])] for i in range(size)]
    for item in case['overrides']:
        values[item['index']] = item['hex']
    assert all(len(bytes.fromhex(v)) == WIDTH for v in values)
    return values


def fraction(text):
    text = text.strip()
    match = re.fullmatch(r'([+-]?)(\d+)(?:\.(\d*))?(?:[eEdD]([+-]?\d+))?', text)
    if not match:
        raise ValueError('Not a defined singleton decimal: '+repr(text))
    sign, whole, tail, exponent = match.groups()
    tail = tail or ''
    numerator = int(whole+tail)
    power = int(exponent or 0)-len(tail)
    value = Fraction(numerator*10**max(power,0), 10**max(-power,0))
    return (-value if sign == '-' else value), sign == '-'


def binary128(text, rounding=0):
    """Low/high uint64 words using exact rational conversion; no binary64."""
    value, negative = fraction(text)
    if not value:
        return (0, (1 << 63) if negative else 0)
    n, d = abs(value).numerator, value.denominator
    e = n.bit_length()-d.bit_length()
    if (n < (d << e)) if e >= 0 else ((n << -e) < d):
        e -= 1
    shift = 112-max(e, -16382)
    numerator, denominator = (n << shift, d) if shift >= 0 else (n, d << -shift)
    q, remainder = divmod(numerator, denominator)
    if rounding == 0:
        increment = remainder*2 > denominator or (remainder*2 == denominator and q % 2)
    elif rounding == 1:
        increment = bool(remainder and not negative)
    elif rounding == 2:
        increment = bool(remainder and negative)
    elif rounding == 3:
        increment = False
    else:
        raise ValueError('Unknown rounding mode')
    q += bool(increment)
    if e < -16382:
        exponent = 0 if q < (1 << 112) else 1
        significand = q if exponent == 0 else q-(1 << 112)
    else:
        if q >= (1 << 113):
            q >>= 1
            e += 1
        exponent = e+16383
        significand = q-(1 << 112)
        if exponent >= 0x7fff:
            away = rounding == 0 or (rounding == 1 and not negative) or (rounding == 2 and negative)
            exponent, significand = (0x7fff, 0) if away else (0x7ffe, (1 << 112)-1)
    bits = (int(negative) << 127) | (exponent << 112) | significand
    assert 0 <= bits < 1 << 128
    return bits & ((1 << 64)-1), bits >> 64


def build_corpus():
    unit = '1.'+'0'*38+'e0'
    quarter = '1.25'+'0'*36+'e0'
    boundary = [canonical_integer((1 << 113)+k) for k in [-1,0,1,2,3]]
    pattern = ['0','-0',unit,'-'+quarter,*boundary,
               '1.'+'2'*38+'e4900','-1.'+'2'*38+'e-4900']
    cases = []
    def add(name, n=1, rank=0, tokens=None, overrides=None, **options):
        case = dict(id=name,n=n,rank=rank,poles=0,finite=1,check=0,cache=1,
                    clear_fails=0,rounding=0,sticky_mask=0,halt_flag=0,
                    expected='success',execution='public',pattern_hex=[record(v) for v in (tokens or pattern)],
                    overrides=[])
        case.update(options)
        for index, value in overrides or []:
            case['overrides'].append({'index':index,'hex':record(value)})
        cases.append(case)
    # All qualified N/rank shapes and exact coefficient counts.
    for n in range(1,7):
        for rank in range(7):
            add(f'shape-n{n}-r{rank}',n=n,rank=rank)
    # Defined malformed singleton at first/middle/last of every count.
    for rank in range(7):
        count = 8*count_for_rank(rank)
        for label,index in [('first',0),('middle',count//2),('last',count-1)]:
            add(f'invalid-r{rank}-{label}', n=1+rank%6,rank=rank,
                overrides=[(index,'not-a-number')],expected='error')
    # Defined scalar fallback: no acceptance inferred for undefined null/slash.
    accepted = ['1.0e4901','1.0e-4901','1.0e-4950','1.0e-5000',
                '1.25E+0','+1.25',' 1.25','1.25\t','1.25D0','1.25e00',
                '1.25e+0','1.25e-0','-0.0','0.0','1.25 8', '1.25,8']
    for i,token in enumerate(accepted):
        # Extra token variants have the explicitly retained first scalar token.
        first = '1.25' if token in ('1.25 8','1.25,8') else token.strip()
        add(f'fallback-value-{i:02}',tokens=[unit],overrides=[(3,token)],
            expected_decimal_overrides={'3':first})
    for i,token in enumerate(['NaN','Infinity','-Infinity','1.0E+5000','1.25x']):
        add(f'fallback-error-{i:02}',tokens=[unit],overrides=[(3,token)],expected='error')
    ambiguous = [b'',b' ',b'/',b',',b'2*',b'2*1.25',b'1\t2',b'\x001',b'1\x00',b'1.2\x005',
                 b'\xff',b'1.0\r2',b'1.25/',b'1.25\n8']
    for i,token in enumerate(ambiguous):
        add(f'lexical-only-{i:02}',tokens=[unit],overrides=[(3,token)],execution='lexical-only',
            expected='undefined-or-platform-parser-syntax')
    # All mode combinations qualify dispatch/parser only, not physical poles.
    for finite in (0,1):
        for poles in range(4):
            add(f'options-f{finite}-p{poles}',finite=finite,poles=poles,
                expected='unsupported' if not finite and poles == 3 else 'success')
    for check,cache in [(1,1),(0,0),(1,0)]:
        for bad in (False,True):
            add(f'cache-c{check}-e{cache}-bad{int(bad)}',check=check,cache=cache,
                overrides=[(3,'invalid')] if bad else [], expected='error' if bad else 'success')
    add('clear-failure',overrides=[(3,'invalid')],expected='error',clear_fails=1)
    for rounding in range(4):
        for sticky in (0,1,2,4,8,16):
            add(f'ieee-round{rounding}-sticky{sticky}',rounding=rounding,sticky_mask=sticky)
    for flag in range(1,6):
        add(f'ieee-halt{flag}-exact',tokens=[quarter],halt_flag=flag)
        add(f'ieee-halt{flag}-inexact',tokens=[boundary[2]],halt_flag=flag)
    # Test-only forced unexpected bulk failure; not the scientific candidate.
    for clear in (0,1):
        add(f'forced-bulk-failure-clear{clear}',tokens=[unit],clear_fails=clear,
            execution='forced-hook',expected='error')
    assert len({x['id'] for x in cases}) == len(cases)
    for case in cases:
        values = records(case)
        case['whole_batch_canonical'] = all(canonical(v) for v in values)
        if case['execution'] == 'lexical-only':
            assert not case['whole_batch_canonical']
    return {'schema':'madloop-tirust-bulk-unit-corpus-v1','cases':cases,
            'coefficient_count_formula':'C(rank+4,4)','binary128_oracle':'exact integer rational conversion',
            'halting_scope':'Actual ABI and observer markers identify phase; no output-decoder claim before ABI entry',
            'undefined_parser_syntax':'lexical-only; no allocation-dependent numeric comparison',
            'performance_acceptance':False}


def write_case(case,path):
    fields = ['id','n','rank','poles','finite','check','cache','clear_fails','rounding','sticky_mask','halt_flag']
    Path(path).write_text(' '.join(str(case[key]) for key in fields)+'\n'+'\n'.join(records(case))+'\n')
