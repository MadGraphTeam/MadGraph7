#!/usr/bin/env python3
"""Pure extraction of two unchanged EW N4/rank3 requests from closed traces."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def require(ok,message):
    if not ok:raise ValueError(message)

def pin(path):
    raw=Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def checked(path,expected):
    raw=Path(path).read_bytes()
    require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==
            {k:expected[k] for k in ('bytes','sha256')},'Bound bytes differ: '+str(path))
    return raw

def seal(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'Index differs: '+str(path))
    index=json.loads(raw)
    for name,value in index['files'].items():
        p=Path(name)
        require(not p.is_absolute() and '..' not in p.parts,'Unsafe index path')
        checked(Path(path).parent/p,value)
    return index

def write_once_or_verify(path,raw,verify):
    if verify:require(path.read_bytes()==raw,'Saved extraction differs: '+str(path))
    else:
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)

def main():
    require(sys.argv[1:] in ([],['--verify-only']),'Only --verify-only supported')
    verify=sys.argv[1:]==['--verify-only']
    require(Path.cwd()==ROOT,'Run from repository root')
    bindings=json.loads((HERE/'bindings.json').read_bytes())
    for name,value in bindings['indices'].items():seal(ROOT/name,value)
    for name,value in bindings['files'].items():checked(ROOT/name,value)
    for key in ('toolchain','fixed_supplemental_providers'):
        for name,value in bindings[key].items():checked(name,value)
    closures={k:json.loads(checked(ROOT/v['path'],v)) for k,v in bindings['closures'].items()}
    retained_root=ROOT/'docs/benchmarkfolder/experiments/current-kinematic-trace-runtime-qualification-1'
    retained=json.loads((retained_root/'evidence-index.json').read_bytes())
    # Authenticate the exact private source captures against their already
    # published, sealed raw bytes. No external native provider is consulted.
    for bound in bindings['traces'].values():
        logical=next(name for name,p in retained['origins'].items() if p['path']==bound['path'])
        compressed=retained['compressed_text'][logical]
        raw=gzip.decompress((retained_root/compressed['path']).read_bytes())
        require(raw==checked(ROOT/bound['path'],bound),'Private trace differs from retained packet')
    path=ROOT/'.tirust-build/kinematic-provenance-analysis-1/decode_trace.py'
    spec=importlib.util.spec_from_file_location('ew_refusal_bound_decoder',path)
    decoder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=decoder;spec.loader.exec_module(decoder)
    derivation,route,sourcepins=decoder.source_binding('ew')
    identity=next(x for x in derivation['target_identities'] if x['reduction_identifier']==5)
    require(identity['rank']==3 and identity['identity_kind']=='group','Wrong EW source identity')
    tir=ROOT/'.tirust-build/kinematic-provenance-current-executor-1/results/derived/ew/TIRUST_interface.f'
    plans=[list(map(int,pair)) for pair in re.findall(r'CALL TIRUST_REGISTER_PLAN\((\d+),(\d+),STATUS\)',tir.read_text())]
    require(plans==bindings['scope']['plan_registration_sequence'] and len(plans)==13,'Plan sequence differs')
    requests=[];certificates=[]
    for label,bound in bindings['traces'].items():
        require(closures[bound['closure']]['workers'][label]['trace_pin']==
                {k:bound[k] for k in ('bytes','sha256')},'Trace not bound to closed worker')
        raw=checked(ROOT/bound['path'],bound)
        _,records=decoder.parse(raw,'low-first')
        rows=[r for r in records if r['id']==5]
        keys={(r['section'],r['point'],r['serial'],r['id'],r['mode'],r['rank'],r['n'],r['split_order']) for r in rows}
        require(len(keys)==1 and rows and all(r['point']==2 and r['mode']==bound['mode'] and
                r['n']==4 and r['rank']==3 for r in rows),'EW group5 input call not unique/bound')
        suffix='DP' if bound['mode']==1 else 'QP'
        native=decoder.one(rows,'KT_TIRUST_'+suffix)
        matrix=decoder.one(rows,'KT_MATRIX_'+suffix)
        status=decoder.one(rows,'KT_STATUS_'+suffix)
        require(all(r is not None for r in (native,matrix,status)),'Missing input/status boundary')
        require(status['data']['native_status']==1,'Original EW refusal not reproduced in capture')
        data=native['data']
        require(all(v['value'] is not None for values in data.values() for v in values),'Nonfinite native input')
        for corner in range(4):
            for component in range(4):
                require(data['shifts_corner_major'][4*corner+component]['words']==
                        matrix['data']['shifts_component_major'][4*component+corner]['words'],
                        'Shift transpose changes input words')
        words=[word for field in ('shifts_corner_major','squared_masses','invariants_column_major','scale')
               for value in data[field] for word in value['words']]
        require(len(words)==(41 if bound['mode']==1 else 82),'Native input shape differs')
        name=label.replace('-enabled','-group5-stock')
        p=HERE/'inputs'/(name+'.words')
        raw=('KPROJ_WORDS_V1\n'+f"2 {bound['mode']} 4 3 {len(words)}\n"+'\n'.join(words)+'\n').encode()
        write_once_or_verify(p,raw,verify)
        requests.append({'id':name,'variant':'stock','source_trace':label,'point':2,
                         'mode':bound['mode'],'input':{'path':str(p.relative_to(ROOT)),**pin(p)}})
        certificates.append({'trace':bound,'identity':dict(zip(('section','point','serial','id','mode','rank','n','split_order'),next(iter(keys)))),
                             'source_lines':{'native':native['line'],'matrix':matrix['line'],'status':status['line']},
                             'original_native_status':1,'native_words':words,'intervention':False})
    require(len(requests)==2 and [r['mode'] for r in requests]==[1,4],'Exactly DP/QP stock requests required')
    result={'schema':'private-exact-ew-original-refusal-certificate-v1','source_only':True,'native_executions':0,
            'failed_tensor_outputs_read':False,'input_trace_only':True,'source_pins':sourcepins,'source_identity':identity,
            'plan_registration_source':{'path':str(tir.relative_to(ROOT)),**pin(tir)},'plan_registration_sequence':plans,
            'tensor_component_count':35,'requests':requests,'certificates':certificates,'input_interventions':False,
            'production_fixture_or_policy_change':False,'success_status_is_not_accuracy_proof':True}
    raw=(json.dumps(decoder.wire(result),sort_keys=True,indent=2)+'\n').encode()
    write_once_or_verify(HERE/'input-certificate.json',raw,verify)
    print(json.dumps({'requests':2,'plans':13,'native_executions':0,'verify_only':verify,
                      'certificate':pin(HERE/'input-certificate.json')},indent=2))

if __name__=='__main__':main()
