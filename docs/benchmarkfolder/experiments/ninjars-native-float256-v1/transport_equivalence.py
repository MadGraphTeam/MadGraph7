#!/usr/bin/env python3
"""Compare the represented binary128 return values, retaining ASCII differences."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    helper = HERE/'independent/focused_audit.py'
    spec = importlib.util.spec_from_file_location('native_independent_rounding',helper)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    independent.validate_rounding()
    path = HERE/'run-replay/raw.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==1393 and rows[-1]['kind']=='summary'
    minimum_normal=Fraction(1,1<<16382)
    maximum_finite=Fraction((2**113-1)*2**(16383-112))
    changes=[]
    binary_changes=[]
    for row in rows[:-1]:
        baseline=row['arms'][0]
        for arm in row['arms'][1:]:
            for field,(a,b) in enumerate(zip(arm['output'],baseline['output'])):
                av,bv=independent.recover_binary(a),independent.recover_binary(b)
                assert all(value==0 or minimum_normal<=abs(value)<=maximum_finite for value in (av,bv))
                sign_changed=(a.strip().startswith('-')!=b.strip().startswith('-')) if av==bv==0 else False
                record={'sequence':row['sequence'],'policy':arm['policy'],'field':field,
                    'native_ascii':a,'interpreted_ascii':b,
                    'represented_binary128_equal':av==bv and not sign_changed,
                    'zero_sign_changed':sign_changed}
                if a!=b:changes.append(record)
                if av!=bv or sign_changed:binary_changes.append(record)
    print(json.dumps({'complete':True,'engine_calls':0,'performance_claim':False,
        'script_sha256':sha(Path(__file__)),'rounding_helper_sha256':sha(helper),
        'raw_sha256':sha(path),'records':1392,'binary128_fields_compared_per_native_policy':11136,
        'binary128_rounding':'Independent exact-rational nearest-even113significantbit recovery; finite normal outputs only; zero signs compared separately',
        'ascii_changes':changes,'represented_binary128_changes':binary_changes,
        'all_returned_binary128_values_and_zero_signs_equal':not binary_changes,
        'does_not_claim_raw256bit_equality':True},indent=2))


if __name__=='__main__':main()
