#!/usr/bin/env python3
"""Create or verify the source qualification index without native execution."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
INDEX=HERE/'evidence-index.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(index):
    for section in ('source_files','native_artifacts_hash_only'):
        for name,digest in index[section].items():assert sha(ROOT/name)==digest,name
    prepared=json.loads((HERE/'prepared.json').read_text())
    for name,digest in prepared['inputs'].items():assert sha(ROOT/name)==digest,name
    for name,digest in prepared['system_files'].items():assert sha(name)==digest,name
    assert prepared['cpu']==8 and prepared['jobs']==1 and prepared['build_seconds_max']==180
    assert [len(s['jobs']) for s in prepared['stages']]==[218,4,2]
    assert len(prepared['build_jobs'])==19
    return prepared

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--check',action='store_true');args=cli.parse_args()
    if args.check:index=json.loads(INDEX.read_text())
    else:
        assert not INDEX.exists(), 'Frozen index exists'
        assert not (HERE/'build.json').exists() and not (HERE/'run.json').exists()
        source={};artifacts={}
        for p in sorted(HERE.rglob('*')):
            if p.is_file():
                store=artifacts if p.parent==HERE/'current-runtime' else source
                store[str(p.relative_to(ROOT))]=sha(p)
        index={'schema':'combined-qp-mass-source-qualification-v1','state':'source-prepared-not-native-qualified',
            'source_files':source,'native_artifacts_hash_only':artifacts,
            'required_review':'Root review of exact prepared.json commands and control.py before compiler/native launch',
            'scope':'General MP model mass routing and caller guard; current eb3d runtime with retained historical EW generated objects/oracle',
            'source_tests':{'controller':8,'caller_fixture':11,'amplitude':11},
            'resource_caps':{'cpu':8,'jobs':1,'compile_link_commands':19,'build_global_seconds':180,
                'loader_list_commands':7,'loader_global_seconds':30,'caller_cases':218,'caller_total_seconds':60,
                'ew_workers':6,'ew_each_seconds':120,'native_total_seconds':780,
                'parser_caps_seconds':[20,30,30],'admission_cap_seconds':7200},
            'no_rust_build':True,'no_automatic_retry':True,'production_and_installations_unchanged':True,
            'not_qualified':['automatic checked QP rescue','complete threshold amplitude','physical CMS','all-five-backend ABI','performance']}
        verify(index)
        INDEX.write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    spec=verify(index)
    print(json.dumps({'index_sha256':sha(INDEX),'prepared_sha256':sha(HERE/'prepared.json'),
        'source_files':len(index['source_files']),'native_artifacts_hash_only':len(index['native_artifacts_hash_only']),
        'bound_inputs':len(spec['inputs']),'system_files':len(spec['system_files']),
        'compiler_launched':False,'native_launched':False}))

if __name__=='__main__':main()
