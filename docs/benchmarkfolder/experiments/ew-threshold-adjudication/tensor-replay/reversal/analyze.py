#!/usr/bin/env python3
"""Future replay validator; no compilation or native execution is performed."""
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
from prepare import extract

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
D=Decimal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a,b):
    ar,ai,br,bi=map(D,[*a,*b])
    absolute=((ar-br)**2+(ai-bi)**2).sqrt()
    scale=max((ar**2+ai**2).sqrt(),(br**2+bi**2).sqrt())
    return {'absolute':str(absolute),'relative':str(absolute/scale if scale else D(0))}


def analyze():
    inputs=json.loads((HERE/'selected-inputs.json').read_text())
    sources=json.loads((HERE/'source-manifest.json').read_text())
    execution=json.loads((HERE/'execution.json').read_text())
    assert execution['complete'] and execution['returncode']==0
    assert execution['output_sha256']==digest(HERE/'output.log')
    assert execution['input_stream_sha256']==inputs['input_stream_sha256']
    assert execution['source_manifest_sha256']==inputs['source_manifest_sha256']
    required_artifacts={'.tirust-build/ew-threshold-tensor-replay-reversal/replay.o',
                        '.tirust-build/ew-threshold-tensor-replay-reversal/replay'}
    assert required_artifacts<=set(execution['artifacts_sha256'])
    for name,expected_hash in execution['artifacts_sha256'].items():
        assert digest(ROOT/name)==expected_hash,name
    runtime=sources['historical_tirust_dso']
    assert execution['loaded_libraries_sha256'][runtime]==sources['sources_and_link_inputs'][runtime]
    for name,expected_hash in execution['loaded_libraries_sha256'].items():
        assert digest(ROOT/name)==expected_hash,name
    assert digest(HERE/'source-manifest.json')==inputs['source_manifest_sha256']
    assert digest(HERE/'selected-inputs.dat')==inputs['input_stream_sha256']
    for name,expected_hash in sources['sources_and_link_inputs'].items():
        assert digest(ROOT/name)==expected_hash,name
    private_work=ROOT/'.tirust-build/ew-threshold-tensor-replay-reversal'
    assert digest(private_work/'replay.f90')==digest(HERE/'replay.f90')
    assert digest(private_work/'selected-inputs.dat')==inputs['input_stream_sha256']
    for name in ('trace','summary','execution'):
        assert digest(HERE.parents[1]/(name+'.json'))==inputs['parent_'+name+'_sha256'],name
    trace=json.loads((HERE.parents[1]/'trace.json').read_text())
    expected={(r['point'],r['group']):r for r in inputs['records']}
    assert len(expected)==len(inputs['records'])
    for key,record in expected.items():
        assert extract(trace[str(key[0])],key[1])==record,'Captured-input binding changed'
    observed={}
    current=None
    for line in (HERE/'output.log').read_text().splitlines():
        fields=line.split()
        if not fields: continue
        tag=fields[0]
        if tag=='EW_TENSOR_REPLAY_V1':
            assert not observed
            continue
        if tag=='CASES':
            assert int(fields[1])==len(expected)
            continue
        if tag=='BEGIN':
            assert current is None or current.get('complete')
            point,group,n,rank,status,count=map(int,fields[1:])
            key=(point,group)
            assert key not in observed and key in expected
            e=expected[key]
            assert (n,rank,status,count)==(e['n'],e['rank'],e['historical_status'],e['count'])
            current={'point':point,'group':group,'shifts':[],'masses':[],'invariants':[],
                     'raw':{},'units':{},'comparisons':[]}
            observed[key]=current
        elif tag=='SCALE': current['scale']=fields[1]
        elif tag in ('SHIFT','MASS','S'):
            ni=2 if tag=='S' else 1
            row=[*map(int,fields[1:1+ni]),*fields[1+ni:]]
            current[{'SHIFT':'shifts','MASS':'masses','S':'invariants'}[tag]].append(row)
        elif tag in ('CACHE_BEFORE','CACHE_AFTER'):
            current[tag.lower()]=list(map(int,fields[1:]))
        elif tag=='TIR_STATUS': current['tir_status']=int(fields[1])
        elif tag=='TIR_ERROR': current['tir_error']=line.split('TIR_ERROR ',1)[1]
        elif tag=='TIR_RAW':
            raw_key=tuple(map(int,fields[1:3]))
            assert raw_key not in current['raw'] and len(fields)==5
            current['raw'][raw_key]=fields[3:]
        elif tag=='NRS_UNIT':
            i,degree,mapping,status=map(int,fields[1:])
            assert i not in current['units']
            current['units'][i]={'degree':degree,'mapping':mapping,'status':status,'results':{}}
            unit=current['units'][i]
        elif tag=='NRS_ERROR': unit['error']=line.split('NRS_ERROR ',1)[1]
        elif tag=='NRS_RESULT':
            i,k=map(int,fields[1:3])
            assert k not in current['units'][i]['results'] and len(fields)==5
            current['units'][i]['results'][k]=fields[3:]
        elif tag=='NRS_R1': current['units'][int(fields[1])]['r1']=fields[2:]
        elif tag=='COMPARE':
            i,k=map(int,fields[1:3])
            current['comparisons'].append({'unit':i,'slot':k,'tir':fields[3:5],'nrs':fields[5:7]})
        elif tag=='END':
            assert tuple(map(int,fields[1:]))==(current['point'],current['group'])
            current['complete']=True
        else: raise ValueError('Unrecognized replay output record: '+tag)
    assert set(observed)==set(expected)
    results=[]
    for key,e in expected.items():
        p=observed[key]
        assert p.get('complete')
        # Exact canonical ES45.35E4 tokens verify round-trip binary128 inputs,
        # including signed zero, without any Python float conversion.
        for field in ('scale','shifts','masses','invariants'): assert p[field]==e[field],(key,field)
        assert p['cache_before'][2:4]==[0,0] and p['cache_after'][2:4]==[0,0]
        assert p['cache_after'][5]==p['cache_before'][5]+1
        assert set(p['units'])==set(range(e['count']))
        for i,u in p['units'].items():
            assert u['degree']==e['unit_degrees'][i]
            assert u['mapping']==sources['mapping_MadLoop_index_to_IREGI_one_based'][i]
            assert set(u['results'])==({0,1,2} if u['status']==0 else set())
            assert ('r1' in u)==(u['status']==0)
        if p['tir_status']==0:
            assert set(p['raw'])=={(i,k) for i in range(1,e['count']+1) for k in range(1,5)}
        else:
            assert not p['raw'] and not p['comparisons']
        expected_comparisons=3*sum(u['status']==0 for u in p['units'].values()) if p['tir_status']==0 else 0
        assert len(p['comparisons'])==expected_comparisons
        expected_pairs={(i,k) for i,u in p['units'].items() if u['status']==0 for k in range(3)} if p['tir_status']==0 else set()
        assert {(r['unit'],r['slot']) for r in p['comparisons']}==expected_pairs
        for row in p['comparisons']:
            u=p['units'][row['unit']]
            assert u['status']==0 and row['nrs']==u['results'][row['slot']]
            raw_row=u['mapping']
            if row['slot'] in (0,2):
                assert row['tir']==p['raw'][raw_row,{0:4,2:3}[row['slot']]]
            else:
                assert row['slot']==1
                for component in range(2):
                    uv=D(p['raw'][raw_row,1][component])
                    ir=D(p['raw'][raw_row,2][component])
                    # Permit only binary128 addition/decimal-print rounding.
                    bound=max(abs(uv),abs(ir))*D(2)**-111+D(2)**-16492
                    assert abs(D(row['tir'][component])-(uv+ir))<=bound
            row['difference']=distance(row['tir'],row['nrs'])
        historical_differences=[]
        if e['historical_status']==0 and p['tir_status']==0:
            for i,k,re,im in e['historical_raw_IREGI_UV1_IR1_IR2_finite']:
                historical_differences.append({'row':int(i),'slot':int(k),
                    'difference':distance(p['raw'][int(i),int(k)],[re,im])})
        results.append({'point':key[0],'group':key[1],'historical_status':e['historical_status'],
            'tir_status':p['tir_status'],'tir_error':p.get('tir_error'),
            'NRS_units':p['units'],'comparisons':p['comparisons'],
            'TIR_historical_differences':historical_differences,
            'exact_input_echo':'PASS','cache_disabled':'PASS'})
    return {'scope':'Tensor diagnostic only; no amplitude reconstruction or numerical acceptance.',
        'complex_metric':'abs(a-b)/max(abs(a),abs(b)); zero-zero=0; also report absolute difference.',
        'limitations':['Shared oneloopmaster ancestry limits scalar independence.',
                      'Common captured TIR scale differs from original NinjaRS DP-square scale.',
                      'Failed native outputs are excluded; zero arrays are never interpreted as successful tensors.',
                      'Current parent trace has no numerator coefficients, so no amplitude reconstruction is claimed.'],
        'records':results,'input_manifest_sha256':digest(HERE/'selected-inputs.json'),
        'execution_manifest_sha256':digest(HERE/'execution.json'),
        'raw_output_sha256':digest(HERE/'output.log'),'analyzer_sha256':digest(Path(__file__))}


if __name__=='__main__':
    with localcontext() as context:
        context.prec=100
        result=analyze()
    (HERE/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS: exact captured inputs, unit degrees, output coverage and failed-output exclusion.')
