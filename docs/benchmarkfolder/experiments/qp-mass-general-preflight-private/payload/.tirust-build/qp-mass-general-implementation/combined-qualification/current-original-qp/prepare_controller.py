#!/usr/bin/env python3
"""Freeze one additive link/two-worker plan; source-only, no process launch."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def absolute(p):
    p=Path(p);return p if p.is_absolute() else ROOT/p

def rel(p):return str(absolute(p).relative_to(ROOT))

def main():
    assert Path.cwd()==ROOT and not (HERE/'prepared.json').exists()
    child=json.loads((HERE/'child-prepared.json').read_text())
    prior=json.loads((PARENT/'prepared.json').read_text())
    assert sha(PARENT/'prepared.json')==child['parent_prepared_sha256']
    assert sha(PARENT/'build.json')==child['parent_build_sha256']
    inputs={**prior['inputs']};system={**prior['system_files']}
    def add(p,digest=None):
        p=absolute(p);actual=sha(p);assert digest is None or actual==digest,str(p)
        store,key=(inputs,rel(p)) if p.is_relative_to(ROOT) else (system,str(p))
        assert key not in store or store[key]==actual
        store[key]=actual
    for p,d in child['inputs'].items():add(p,d)
    for p,d in child['system_files'].items():add(p,d)
    for p in sorted(HERE.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.o','.mod','.so') and p.name not in ('build.json','run.json','prepared.json')
            add(p)
    for name in ('control.py','prepared.json','build.json','evidence-index.json'):add(PARENT/name)
    validation={'name':'validate-current-original-qp',
        'argv':[child['python'],'-B',str(HERE/'analyze.py')],
        'cwd':rel(HERE),'timeout_seconds':30}
    spec={'schema':'additive-current-original-qp-controller-v1','state':'source-prepared-not-executed',
        'parent_manifest_sha256':child['parent_prepared_sha256'],
        'shared_controller_sha256':sha(PARENT/'control.py'),
        'inputs':dict(sorted(inputs.items())),'system_files':dict(sorted(system.items())),
        'compiler':child['compiler'],'python':child['python'],'taskset':child['taskset'],
        'compiler_environment':child['compiler_environment'],
        'cpu':8,'jobs':1,'build_seconds_max':30,'loader_seconds_max':10,
        'execution_seconds_max':240,'lock_admission_seconds_max':7200,
        'build_jobs':child['build_jobs'],'build_outputs':[rel(p) for p in child['build_outputs']],
        'module_outputs':[], 'runtime_checks':child['runtime_checks'],
        'runtime_library_directories':child['runtime_library_directories'],
        'stages':[{'name':'current-original-qp','execution_seconds_max':240,
                   'jobs':child['runs'],'validation':validation}],
        'qualification':'Original versus mass-only guarded with same current eb3d runtime; fixed4 unchecked diagnostics',
        'prerequisite':'Frozen combined parent run must complete all caller/DP/QP parser gates before these two workers',
        'automatic_retry':False,'new_fortran_compiles':0,'rust_builds':0}
    assert len(spec['build_jobs'])==1 and len(spec['stages'][0]['jobs'])==2
    (HERE/'prepared.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest_sha256':sha(HERE/'prepared.json'),'inputs':len(inputs),'system_files':len(system),
                      'compiler_launched':False,'native_launched':False}))

if __name__=='__main__':main()
