#!/usr/bin/env python3
"""Assemble reviewed child source manifests; never compile or execute native code."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def absolute(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p

def rel(p):
    return str(absolute(p).relative_to(ROOT))

def main():
    assert Path.cwd()==ROOT
    assert not (HERE/'prepared.json').exists(), 'Preparation is append-only'
    fixture=json.loads((HERE/'fixture/commands.json').read_text())
    amp=json.loads((HERE/'amplitude/prepared.json').read_text())
    prefix=ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust'
    current_pins={'libmadloop_tirust.so':'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
                  'libtirust.a':'0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a',
                  'mtirust.mod':'d7838e75335c908a31e644d909bdd6dc71d98467ef8bdca3ca06f8bfad20eaf5'}
    runtime=HERE/'current-runtime';runtime.mkdir(exist_ok=False)
    current_sources={}
    for name,digest in current_pins.items():
        source=prefix/'lib'/name
        assert sha(source)==digest
        shutil.copyfile(source,runtime/name)
        assert sha(runtime/name)==digest
        current_sources[str(source)]=digest
    # The public .mod is byte-identical across these installations. The new
    # include path still binds the current copy, and links use its matching
    # archive instead of retaining historical QP transport cleanup.
    historical_module=ROOT/'.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2/includes/5/mtirust.mod'
    assert sha(historical_module)==current_pins['mtirust.mod']
    inputs={};system={}
    def add(p,wanted=None):
        p=absolute(p)
        digest=sha(p)
        assert wanted is None or digest==wanted, str(p)
        store,key=(inputs,str(p.relative_to(ROOT))) if p.is_relative_to(ROOT) else (system,str(p))
        assert key not in store or store[key]==digest
        store[key]=digest
    for p,d in fixture['inputs'].items():add(p,d)
    for section in ('copied_sha256','readonly_sha256'):
        for p,d in amp[section].items():add(p,d)
    for p,d in current_sources.items():add(p,d)
    add(ROOT/'.tirust-build/qp-transport-install/package-result.json')
    for p in sorted(HERE.rglob('*')):
        if p.is_file():
            assert p.suffix not in ('.o','.mod','.so') or p.parent==runtime, 'Unexpected native artifact'
            assert p.name not in ('build.json','run.json','prepared.json') or p==HERE/'amplitude/prepared.json'
            add(p)
    cleanup=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py'
    add(cleanup,'acf00431793415817c29f26d997ae272da968f4b024482cfb72c60333b61ed50')
    compiler=amp['compiler']['path'];py=fixture['python']
    assert compiler==fixture['compiler']
    add(compiler,amp['compiler']['sha256']);add(py)
    taskset=shutil.which('taskset');assert taskset;add(taskset)
    for info in amp['runtime_expected'].values():add(info['path'],info['sha256'])
    build_jobs=list(fixture['build_jobs'])
    for original in amp['build_jobs']:
        assert original['argv'][:3]==[taskset,'-c','8']
        argv=original['argv'][3:]
        if '-c' in argv:argv=[argv[0],'-I'+str(runtime),*argv[1:]]
        replaced={}
        for i,arg in enumerate(argv):
            if Path(arg).name in ('libtirust.a','libmadloop_tirust.so'):
                replaced[arg]=str(runtime/Path(arg).name);argv[i]=replaced[arg]
        build_jobs.append({**original,'argv':argv,
                           'current_install_input_replacements':replaced,
                           'source_command_had_taskset_prefix':original['argv'][:3]})
    build_outputs=[rel(p) for p in fixture['build_outputs']+amp['build_outputs']]
    loader=amp['loader']['path']
    current_expected={**amp['runtime_expected'],
       'libmadloop_tirust.so':{'path':str(runtime/'libmadloop_tirust.so'),'sha256':current_pins['libmadloop_tirust.so']}}
    checks=[]
    exe=absolute(fixture['runs'][0]['argv'][0])
    allowed={k:v for k,v in amp['runtime_expected'].items() if not Path(v['path']).is_relative_to(ROOT)}
    checks.append({'executable':rel(exe),'loader':loader,'expected':allowed,
        'allow_system_subset':True,'required':['ELF_INTERPRETER','libc.so.6','libgfortran.so.5','libquadmath.so.0'],
        'job':{'name':'loader-fixture','argv':[loader,'--list',str(exe)],
               'cwd':rel(HERE/'fixture'),'log_file':rel(HERE/'fixture/loader-list.txt'),
               'raw_stdout':True,'timeout_seconds':5}})
    native=[]
    for original in amp['runs']:
        directory=absolute(original['cwd'])
        assert original['runtime_check_argv'][:3]==[taskset,'-c','8']
        checks.append({'executable':rel(original['argv'][0]),'loader':loader,'expected':current_expected,
            'job':{'name':'loader-'+original['name'],'argv':original['runtime_check_argv'][3:],
                   'cwd':rel(directory),'log_file':rel(directory/'loader-list.txt'),
                   'raw_stdout':True,'timeout_seconds':5}})
        native.append({**original,'cwd':rel(directory),'log_file':rel(directory/'stdout.log'),
            'raw_stdout':True,'expected_outputs':original['expected_outputs']+['stdout.log']})
    validation={**fixture['validation'],'name':'validate-fixture'}
    stages=[{'name':'caller-poison-fixture','execution_seconds_max':60,
             'jobs':fixture['runs'],'validation':validation}]
    for name,mode,limit in [('dp',1,480),('qp',4,240)]:
        stages.append({'name':'ew-'+name,'execution_seconds_max':limit,
            'jobs':[j for j in native if j['mode']==mode],
            'validation':{'name':'validate-ew-'+name,'argv':[py,'-B',str(HERE/'amplitude/analyze.py'),'--stage',name],
                          'cwd':rel(HERE/'amplitude'),'timeout_seconds':30}})
    assert len(build_jobs)==19 and len(stages[0]['jobs'])==218
    spec={'schema':'combined-qp-mass-qualification-v1','state':'prepared-not-executed',
        'cpu':8,'jobs':1,'build_seconds_max':180,'execution_seconds_max':780,
        'loader_seconds_max':30,'lock_admission_seconds_max':7200,
        'compiler':compiler,'python':py,'taskset':taskset,
        'compiler_environment':amp['historical_environment'],
        'inputs':dict(sorted(inputs.items())),'system_files':dict(sorted(system.items())),
        'build_jobs':build_jobs,'build_outputs':build_outputs,'module_outputs':fixture['module_outputs'],
        'runtime_library_directories':[rel(runtime),rel(ROOT/'.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2/libraries')],
        'current_install':{'source_prefix':rel(prefix),'source_revision':'059bb9c979194bfda783cad83443223def8df280',
            'artifacts':current_pins,'module_identical_to_historical':True,
            'historical_child_preparation_preserved':True,
            'other_generated_objects':'Historical frozen EW model/numerator/host objects; unchanged'},
        'runtime_checks':checks,'stages':stages,
        'qualifications':{'native_or_compiler_launched':False,'automatic_retry':False,
          'scope':'Caller-safety fixture plus fixed DP and unchecked fixed QP EW transport diagnostics',
          'qp_oracle':'Historical new-mass fixed4 fixture; 37 TIRust threshold refusals retained',
          'not_qualified':['checked-QP rescue','all-five-backend ABI','CMS physical amplitudes','performance'],
          'fixture_runtime_policy':'Exact pinned system-library subset, required interpreter/libc/gfortran/quadmath; reducer libraries forbidden',
          'lock_admission':'At most7200s per admission, outside execution limits; no worker starts before acquired'}}
    (HERE/'prepared.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest_sha256':sha(HERE/'prepared.json'),'inputs':len(inputs),
        'system_files':len(system),'build_jobs':len(build_jobs),'native_cases':sum(len(s['jobs']) for s in stages)}))

if __name__=='__main__':main()
