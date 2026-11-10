#!/usr/bin/env python3
"""Read/hash the closed prefix once, then fully replay all ten proposed-parser arms."""
from pathlib import Path
from itertools import combinations
import ast
import gzip
import hashlib
import json
import struct
import analyze

HERE=Path(__file__).resolve().parent
RETENTION=HERE.parent/'failed-run4-retention-1'
RETENTION_SHA='fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361'
SPECS=[(b,m,o) for b in (1,6,8,9) for m in (4,5) for o in (0,1)]


def require(condition,message):
    if not condition: raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def replay(retention=RETENTION):
    retention=Path(retention)
    require(digest((retention/'evidence-index.json').read_bytes())==RETENTION_SHA,'Closed run4 seal differs')
    index=json.loads((retention/'evidence-index.json').read_bytes())
    actual=set()
    for path in retention.rglob('*'):
        require(not path.is_symlink(),'Symlink in retained prefix')
        if path.is_file():actual.add(path.relative_to(retention).as_posix())
    require(actual==set(index['files'])|{'evidence-index.json'},'Retained prefix membership differs')
    stored={}
    for name,item in index['files'].items():
        p=Path(name)
        require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'Unsafe prefix member')
        raw=(retention/p).read_bytes()
        require(len(raw)==item['bytes'] and digest(raw)==item['sha256'],'Retained prefix bytes differ: '+name)
        stored[name]=raw
    inventory=json.loads(stored['retention-inventory.json'])
    def raw(name):
        r=inventory['original_files'][name];value=stored[r['member']]
        value=gzip.decompress(value) if r['codec']=='gzip' else value
        require(len(value)==r['raw_bytes'] and digest(value)==r['raw_sha256'],'Original prefix bytes differ')
        return value
    original=json.loads(raw('actual-run4/result.json'))
    require(original['passed'] is False and original['workers_completed']==9 and original['failure']==
            {'type':'ValueError','message':'malformed decimal transport'},'Original failed result differs')
    old_parser=raw('controller/analyze.py').decode();new_parser=(HERE/'analyze.py').read_text()
    functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
                           if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before,after=functions(old_parser),functions(new_parser)
    require(set(after)-set(before)=={'ascii_zero_signs'} and set(before)<=set(after),'Unexpected parser helper change')
    allowed={'rounded113','ascii113','native_calls'}
    for name in before:
        if name not in allowed:require(before[name]==after[name],'Unrelated input/protocol gate changed: '+name)
    original_index=json.loads(raw('controller/source-index.json'))
    require(digest(raw('controller/source-index.json'))==original['source_index_sha256'],'Original source4 seal differs')
    for name,item in original_index['files'].items():
        require(digest(raw('controller/'+name))==item['sha256'],'Original source4 payload differs')
    fixture_sha={}
    for name in original_index['files']:
        if name.startswith('fixtures/'):
            candidate=(HERE/name).read_bytes()
            require(candidate==raw('controller/'+name),'Physical fixture changed')
            fixture_sha[name]=digest(candidate)
    for name in ('driver.f','observer.c','observer.f90','supervise.py','exec_target.py','shapes.py','prepare.py'):
        require((HERE/name).read_bytes()==raw('controller/'+name),'Numerical/scientific source changed')
    started={name for name in inventory['actual_attempt_directories'] if name.startswith('b') and '/' not in name}
    prefix_names=['b%d-m%d-o%d'%spec for spec in SPECS[:10]]
    require(started==set(prefix_names),'Retained started-worker set differs')
    selected=json.loads(raw('controller/fixtures/selected-inputs.json'))
    points={p:b''.join(struct.pack('<d',float.fromhex(v)) for row in
                      selected['points'][str(p)]['binary64_hex'] for v in row) for p in (1,93,39)}
    workers=[]
    for backend,mode,observed in SPECS[:10]:
        name='b%d-m%d-o%d'%(backend,mode,observed);prefix='actual-run4/'+name
        receipt=json.loads(raw(prefix+'/worker.json'))
        require(receipt['native'] is True and receipt['returncode']==0 and receipt['process_reaped'] is True,
                'Native prefix worker did not close successfully')
        trace,stdout=raw(prefix+'/trace.txt'),raw(prefix+'/worker.log')
        parsed=analyze.parse_worker(trace.decode(),stdout.decode(),backend,mode,observed,points)
        workers.append({'worker':name,'trace_sha256':digest(trace),'stdout_sha256':digest(stdout),'parsed':parsed})
    for i in range(0,10,2):
        require(workers[i]['parsed']['public_raw']==workers[i+1]['parsed']['public_raw'],
                'Prefix plain/observed public input/output/status/IEEE parity differs')
    observed=[w['parsed'] for w in workers if w['parsed']['observed']]
    comparisons=[]
    for left,right in combinations(observed,2):
        if left['backend']!=right['backend'] and left['mode']!=right['mode']:continue
        for a,b in zip(left['points'],right['points']):
            equal=a['inputs']['shared_input_sha256']==b['inputs']['shared_input_sha256']
            comparisons.append({'left':[left['backend'],left['mode']],'right':[right['backend'],right['mode']],
                                'point':a['point'],'shared_input_equal':equal})
    require(all(c['shared_input_equal'] for c in comparisons),'Prefix common input mismatch; no coefficient comparison')
    return {'schema':'source5-closed-run4-ten-worker-pure-replay-v1','passed':True,
            'scope':'Ten inherited native arms reparsed offline; six suffix arms unexecuted, no new amplitude',
            'parser_sha256':digest((HERE/'analyze.py').read_bytes()),'fixture_sha256':fixture_sha,
            'closed_failed_run4_retention_index_sha256':RETENTION_SHA,
            'original_source4_attempt_remains_failed':True,'original_source4_parser_failure':original['failure'],
            'original_run4_elapsed_seconds':original['elapsed_seconds'],
            'remaining_original_budget_seconds':3000-original['elapsed_seconds'],
            'candidate_parser_full_workers_passed':10,'matrix_workers_required':16,'matrix_workers_not_replayed':6,
            'exact_neverlaunched_suffix':[list(spec) for spec in SPECS[10:]],
            'public_plain_observed_bytes_equal':True,'partial_common_input_checks':comparisons,
            'all_partial_common_inputs_equal':True,'adjudication_executed':False,
            'coefficient_comparison_qualified':False,'scientific_acceptance':False,
            'normal_range_113_only':True,'zero_sign_scope':'Lexical metadata retained; Fraction equality is not zero-sign bit parity',
            'delta_guard':{'unchanged_analyzer_functions':[name for name in before if name not in allowed],
                           'changed_functions':sorted(allowed),'new_helper':'ascii_zero_signs'},'workers':workers}


if __name__=='__main__':print(json.dumps(replay(),sort_keys=True,indent=2,allow_nan=False))
