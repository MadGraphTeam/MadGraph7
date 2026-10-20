#!/usr/bin/env python3
"""Compare immutable helper with historical installed source on boundary inputs."""
import ast
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from build import HERE, ROOT, binding, write
from comparison import compare, function_text

ORIGINAL = ROOT / '.benchmark-work/policy-v2/install-export-abi3-zero-sign-v1/source/tools/ninjars_report_v2.py'
text = ORIGINAL.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == '93f942dfa74b2c3354d066e5436f4aadd7a009d0f08fc03a05190aa8f5c21c85'
node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'compare')
assert ast.get_source_segment(text,node) == function_text
namespace = {'defaultdict':defaultdict,'itertools':itertools,'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ORIGINAL),'exec'),namespace)
original = namespace['compare']

def row(values,point=1,order=0):
    return {'point':point,'order':order,'coefficients':values,'status':218,'accuracy':0.}

def outcome(fn,groups,pairs):
    try:
        return {'return':fn(groups,pairs)}
    except Exception as exc:
        return {'error_type':type(exc).__name__,'error':str(exc)}

t = 1e-7+1e-10
critical = t/(1-t)
cases = [
    ('zeros',{'a':[row([0.,-0.,0.,0.])],'b':[row([-0.,0.,0.,0.])]},None,True),
    ('inside_positive_boundary',{'a':[row([1.,0.,0.,0.])],'b':[row([1.+.999*critical,0.,0.,0.])]},None,True),
    ('outside_positive_boundary',{'a':[row([1.,0.,0.,0.])],'b':[row([1.+1.001*critical,0.,0.,0.])]},None,False),
    ('inside_negative_boundary',{'a':[row([-1.,0.,0.,0.])],'b':[row([-1.-.999*critical,0.,0.,0.])]},None,True),
    ('outside_negative_boundary',{'a':[row([-1.,0.,0.,0.])],'b':[row([-1.-1.001*critical,0.,0.,0.])]},None,False),
    ('common_cohort_scale',{'a':[row([0.,0.,0.,0.])],'b':[row([1e-6,0.,0.,0.])],'c':[row([1e6,0.,0.,0.])]},[('a','b')],True),
    ('common_split_order_scale',{'a':[row([0.,0.,0.,0.]),row([1e6,0.,0.,0.],order=1)],'b':[row([1e-6,0.,0.,0.]),row([1e6,0.,0.,0.],order=1)]},None,True),
    ('empty',{},None,'ValueError'),
    ('duplicate',{'a':[row([1.,0.,0.,0.]),row([1.,0.,0.,0.])]},None,'ValueError'),
    ('mismatched_order',{'a':[row([1.,0.,0.,0.])],'b':[row([1.,0.,0.,0.],order=1)]},None,'ValueError'),
    ('nonfinite',{'a':[row([math.inf,0.,0.,0.])]},None,'ValueError'),
    ('nan',{'a':[row([math.nan,0.,0.,0.])]},None,'ValueError'),
    ('missing_coefficient',{'a':[row([1.,0.,0.])]},None,'ValueError'),
]
results = []
for name,groups,pairs,expected in cases:
    a,b = outcome(original,groups,pairs),outcome(compare,groups,pairs)
    assert a == b,name
    if isinstance(expected,bool):
        assert a['return']['passed'] is expected,name
    else:
        assert a['error_type'] == expected,name
    results.append({'name':name,'identical':True,'expected':expected,'outcome':a})
result = {'schema':'allocation-private-comparison-contract-v1','passed':True,
    'controller':binding(Path(__file__)),'loader':binding(HERE / 'comparison.py'),
    'immutable_source':binding(HERE / 'bound-support/ninjars_report_v2.py'),
    'historical_source':binding(ORIGINAL),'provenance':binding(HERE / 'bound-support/provenance.json'),
    'compare_function_bytes':len(function_text.encode()),
    'compare_function_sha256':hashlib.sha256(function_text.encode()).hexdigest(),
    'function_bytes_identical':True,'cases':results,'numerical_engine_execution':False}
write(HERE / 'comparison-contract.json',result)
print(json.dumps({'passed':True,'cases':len(results),'result':binding(HERE / 'comparison-contract.json')}),flush=True)
