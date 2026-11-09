#!/usr/bin/env python3
"""Exact-rational expected REAL16 input bits; no float or native reducer."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REFERENCE=ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/input-consistency/audit.py'
spec=importlib.util.spec_from_file_location('reviewed_binary128_audit',REFERENCE)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def bits(text):
    value=helper.binary128(text)
    negative=Decimal(text).is_signed()
    if value==0:
        code=int(negative)<<127
    else:
        value=abs(value)
        exponent=value.numerator.bit_length()-value.denominator.bit_length()
        if value<helper.power2(exponent):exponent-=1
        assert -16382<=exponent<=16383
        mantissa=value/helper.power2(exponent-112)
        assert mantissa.denominator==1
        code=(int(negative)<<127)|((exponent+16383)<<112)|(int(mantissa)-(1<<112))
    return code.to_bytes(16,'little').hex().upper()

def main():
    records=json.loads((HERE/'selected-inputs.json').read_text())['records']
    result=[]
    for r in records:
        inv={(i,j):value for i,j,value in r['invariants']}
        result.append({'point':r['point'],'group':r['group'],
            'INPUT_P':''.join(bits(value) for row in r['shifts'] for value in row[1:]),
            'INPUT_S':''.join(bits(inv[i,j]) for j in range(1,r['n']+1) for i in range(1,r['n']+1)),
            'INPUT_M':''.join(bits(value) for row in r['masses'] for value in row[1:]),
            'INPUT_MU':bits(r['scale'])})
    data={'scope':'Expected GNU/Linux little-endian IEEE binary128 bytes reconstructed with exact rational ties-even rounding',
          'saved_token_precision':'36 significant decimal digits, sufficient for finite binary128 round-trip',
          'helper_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
          'records':result}
    (HERE/'expected-input-bits.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Prepared exact expected input bytes for',len(result),'saved parents; no native code loaded.')

if __name__=='__main__':main()
