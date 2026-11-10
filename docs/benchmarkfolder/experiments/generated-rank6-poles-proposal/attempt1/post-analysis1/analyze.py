"""Adjudicate immutable attempt1; fixture105 remains uncovered as complex mass."""
from decimal import localcontext
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent
ROOT=ORIGINAL.parents[4]
WORK=ROOT/'.tirust-build/generated-rank6-poles-1'

def analyze(workdirs=None):
    source=(ORIGINAL/'analyze.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='d3e6583162f3e3efe74bf933cda34e82eb0d8f606f752c475d15c6a3ac06eb4e'
    old='a[:2]==b[:2] and binary128(a[2:])<0,"Complex mass mutation differs"'
    new='a[:2]==b[:2] and a[2:]==["0000000000000000","8000000000000000"],"Attempt1 zero-width mass mutation differs"'
    assert source.count(old)==1
    namespace={'__name__':'rank6_attempt1_post_analysis','__file__':str(ORIGINAL/'analyze.py')}
    exec(compile(source.replace(old,new),str(ORIGINAL/'analyze.py'),'exec'),namespace)
    with localcontext() as context:
        context.prec=100
        result=namespace['analyze']([WORK/'ew',WORK/'qcd'] if workdirs is None else workdirs)
    result['status']='open_acceptance_items'
    result['open_items'].insert(0,{'kind':'complex_mass_fixture_uncovered','action':105,
        'detail':'Frozen WZ is zero; fixture105 has imaginary mass squared negative zero. Its rows are retained zero-width controls and provide no complex-mass qualification.'})
    result['complex_mass_qualification']=False
    result['post_analysis_scope']='One exact fixture relationship amended to its observed frozen negative-zero value; all remaining protocol/numerical assertions unchanged. No new native execution.'
    return result

if __name__=='__main__':
    print(json.dumps(analyze(),indent=2,sort_keys=True))
