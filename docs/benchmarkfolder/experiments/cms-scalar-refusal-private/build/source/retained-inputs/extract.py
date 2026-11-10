#!/usr/bin/env python3
"""Select exact printed scalar-input records from the frozen CMS diagnosis."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/baseline-original-threshold'
EXPECTED={'a0':['mm'],'b11':['pp','m1','m2'],
          'c0':['p1','p2','p3','m1','m2','m3'],
          'd0':['p1','p2','p3','p4','p12','p23','m1','m2','m3','m4']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def value(token):
    token=token.strip()
    if token.startswith('('):
        assert token.endswith(')')
        parts=token[1:-1].split(',');assert len(parts)==2
        return tuple(Decimal(x.strip()) for x in parts)
    return (Decimal(token),Decimal(0))

def parse(text):
    starts=list(re.finditer(r'(?m)^\s*onshell:\s*(\S+)\s*$',text));records=[]
    for i,start in enumerate(starts):
        end=starts[i+1].start() if i+1<len(starts) else len(text)
        block=text[start.end():end]
        result_types=set(re.findall(r'(?m)^\s*(a0|b11|c0|d0)\(2\):',block))
        assert len(result_types)==1, (i,result_types)
        kind=result_types.pop()
        scales=re.findall(r'(?m)^\s*muscale:\s*(\S+)\s*$',block);assert len(scales)==1
        inputs=re.findall(r'(?m)^\s*(pp|p[1234]|p12|p23|mm|m[1234]):(.*)$',block)
        assert [k for k,v in inputs]==EXPECTED[kind],(i,inputs)
        for _,token in inputs:assert all(x.is_finite() for x in value(token))
        records.append({'ordinal':i+1,'family':kind,'source_line':text.count('\n',0,start.start())+1,
                        'threshold':start[1],'muscale':scales[0],
                        'inputs':[[k,v.strip()] for k,v in inputs]})
    return records

def features(record):
    vals=[(k,*value(v)) for k,v in record['inputs']]
    return {'has_complex_mass':any(k.startswith('m') and im!=0 for k,re,im in vals),
            'has_physical_small_transfer':any(k.startswith('p') and Decimal('1e-8')<abs(re)<Decimal('1e-6') for k,re,im in vals),
            'has_small_external_residual':any(k.startswith('p') and 0<abs(re)<Decimal('1e-10') for k,re,im in vals),
            'all_masses_exact_zero':all(re==0 and im==0 for k,re,im in vals if k.startswith('m'))}

def main():
    assert Path.cwd()==ROOT
    status=json.loads((SOURCE/'status.json').read_text())
    assert sha(SOURCE/'output.log')==status['output_sha256']
    assert status['backend']==1 and status['mode']==4 and not status['stability_certified']
    rows=parse((SOURCE/'output.log').read_text())
    assert sum(len(r['inputs']) for r in rows)==2708
    assert sum(k.startswith('p') for r in rows for k,v in r['inputs'])==1076
    threshold=str(Decimal.from_float(1e-6))
    assert all(abs(Decimal(r['threshold'])-Decimal(threshold))<Decimal('1e-40') for r in rows)
    selected={}
    for row in rows:
        selected.setdefault((row['family'],'first'),row)
        for label,present in features(row).items():
            if present:selected.setdefault((row['family'],label),row)
    examples=[]
    for key,row in selected.items():examples.append({'selection':list(key),**row,'features':features(row)})
    payload={'schema':'cms-scalar-refusal-retained-inputs-v1',
        'source_files':{str((SOURCE/name).relative_to(ROOT)):sha(SOURCE/name) for name in ('output.log','status.json','PS.input','MadLoopParams.dat')},
        'record_count':len(rows),'input_record_count':2708,'invariant_record_count':1076,
        'family_counts':dict(Counter(r['family'] for r in rows)),
        'ordered_inputs_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'selected':examples,
        'scope':'Exact printed input tokens from historical unchecked fixed-QP point20; selection labels are raw-input features, not classifier outcomes',
        'limitations':['No native parser/replay was launched','DP denominator transport remains in this source trace',
                       'No group/direction IDs were printed by the old scalar logger','No amplitude or performance qualification']}
    target=HERE/'point20-inputs.json';assert not target.exists()
    target.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'records':len(rows),'selected':len(examples),'sha256':sha(target)}))

if __name__=='__main__':main()
