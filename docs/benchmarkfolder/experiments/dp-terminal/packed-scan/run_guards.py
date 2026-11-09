#!/usr/bin/env python3
"""Run isolated Fortran correctness probes; no tensor backend or acceptance timing."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess

ROOT=Path.cwd()
REPORT=ROOT/'docs/benchmarkfolder/experiments/dp-terminal/packed-scan'
WORK=ROOT/'.tirust-build/dp-terminal/packed-scan'
COMPILER=shutil.which('gfortran')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def run(command):
    value=subprocess.run(command,cwd=ROOT,env=dict(os.environ,GFORTRAN_ERROR_BACKTRACE='0'),
                         capture_output=True,text=True,timeout=60,preexec_fn=no_core)
    return {'command':list(map(str,command)),'returncode':value.returncode,
            'stdout':value.stdout,'stderr':value.stderr}

results=[]
for name,flags in [('o0-round',['-O0','-frounding-math']),('o3',['-O3']),
                   ('o3-round',['-O3','-frounding-math']),
                   ('o3-round-static',['-O3','-frounding-math','-fno-automatic'])]:
    directory=WORK/name;directory.mkdir(exist_ok=True)
    base=[COMPILER,*flags,'-fcheck=all','-ffree-line-length-none','-J',str(directory),'-I',str(directory),
          str(REPORT/'oracle.f90'),str(REPORT/'terminal.f90')]
    guard=directory/'guards';trap=directory/'traps'
    record={'name':name,'flags':flags,'builds':[],'traps':[]}
    results.append(record)
    for source,output in [('guard-driver.f90',guard),('trap-driver.f90',trap)]:
        build=run([*base,str(REPORT/source),'-o',str(output)])
        record['builds'].append(build)
        if build['returncode']!=0:raise RuntimeError(build)
    probe=run([str(guard)]);record['guard_probe']=probe
    if probe['returncode']!=0 or '384000' not in probe['stdout']:raise RuntimeError(probe)
    for mode in range(1,9):
        dense=run([str(trap),str(mode),'0']);candidate=run([str(trap),str(mode),'1'])
        check={'mode':mode,'dense':dense,'candidate':candidate}
        record['traps'].append(check)
        if dense['returncode']!=candidate['returncode']:
            raise RuntimeError(check)
        if mode<=4 and dense['returncode']!=-8:raise RuntimeError(check)
        if mode in (5,6,8) and dense['returncode']!=0:raise RuntimeError(check)
        if dense['returncode']==0 and dense['stdout']!=candidate['stdout']:
            raise RuntimeError(check)
    record['passed']=True
    print(name,probe['stdout'].strip(),'trap pairs',len(record['traps']),flush=True)
    (REPORT/'guard-results-in-progress.json').write_text(json.dumps(results,indent=2)+'\n')
record={'scope':'DP terminal rank1x1 isolated kernel correctness; no production edits, physical acceptance, or performance claim',
        'compiler':run([COMPILER,'--version']),
        'source_sha256':{p.name:digest(p) for p in sorted(REPORT.glob('*.f90'))},
        'runner_sha256':digest(Path(__file__)),'results':results,
        'guard_comparisons':384000*len(results),'trap_comparison_pairs':8*len(results),'passed':True}
(REPORT/'guard-results.json').write_text(json.dumps(record,indent=2)+'\n')
(REPORT/'guard-results-in-progress.json').unlink()
