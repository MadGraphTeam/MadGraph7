"""Bounded mock-only transport checks; no real reduction library is linked."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/adaptive-qp-tier-telemetry-private/attempt1'
SUPERVISOR = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
FC = '/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran'
CC = '/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/gcc'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert Path.cwd() == ROOT
    assert sha(SUPERVISOR) == 'ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
    assert sha(FC) == '30c6495dcb18cc324629fc230302d9cd8d81143f0ca1635d752b3f8552d48cae'
    assert sha(CC) == '3c48a46fb1dc134c0f10c2da6ec1a8d34a381b93c2ac588b7d19dd3e804eeeef'
    source = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    WORK.mkdir(parents=True, exist_ok=False)
    helper = runpy.run_path(str(SUPERVISOR))
    os.sched_setaffinity(0, {12})
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('LD_'):
            del env[key]
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               TELEMETRY_MOCK_OUTPUT=str(WORK), PYTHONPATH=str(HERE), OMP_NUM_THREADS='1')
    fflags = [FC, '-O0', '-g', '-fcheck=all', '-J'+str(WORK), '-I'+str(WORK)]
    commands = []
    for name in ('mtirust', 'qp_tier_records', 'mock_driver'):
        commands.append((name+'-compile.log', fflags+['-c', str(HERE/(name+'.f90')), '-o', str(WORK/(name+'.o'))]))
    commands.append(('shim-compile.log', [CC,'-std=c11','-O0','-Wall','-Wextra','-Werror','-c',str(HERE/'optional_getter.c'),'-o',str(WORK/'shim.o')]))
    for variant in ('present', 'absent'):
        commands.append((variant+'-compile.log', [CC,'-std=c11','-O0','-Wall','-Wextra','-Werror']+
                         (['-DOMIT_GETTER'] if variant=='absent' else [])+
                         ['-c',str(HERE/'mock_bridge.c'),'-o',str(WORK/(variant+'.o'))]))
        commands.append((variant+'-link.log', [FC,'-o',str(WORK/variant)]+[str(WORK/(name+'.o')) for name in
                         ('mtirust','qp_tier_records','mock_driver','shim',variant)]))
        commands.append((variant+'.log', [str(WORK/variant),variant]))
    commands.append(('pytest.log',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(HERE/'test_telemetry.py')]))
    record={'scope':'C/Fortran mocks and Python serialization only; no real ABI8/9 libraries or integrals',
            'complete':False,'cpu':12,'jobs':1,'total_cap_seconds':120,'sources_before':source,'commands':[]}
    start=time.monotonic(); handlers={}
    def save():
        (WORK/'results.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    def interrupt(signum,frame):
        raise InterruptedError('Mock qualification interrupted by signal %d'%signum)
    try:
        for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            handlers[signum]=signal.signal(signum,interrupt)
        for logname,argv in commands:
            remaining=120-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Shared mock-test cap expired')
            row={'argv':argv,'log':logname,'progress':{}}
            record['commands'].append(row);save()
            with (WORK/logname).open('xb') as log:
                row['exit_code']=helper['run_plain_capped'](argv,log,cwd=ROOT,env=env,progress=row['progress'],seconds=remaining)
            row['log_sha256']=sha(WORK/logname);save()
            print(logname,row['exit_code'],flush=True)
            if row['exit_code']:raise RuntimeError('Mock command failed: '+logname)
        record['sources_after']={name:sha(HERE/name) for name in source}
        assert record['sources_after']==source
        record['complete']=True
    except BaseException as error:
        record['failure']=repr(error)
        raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)
        record['elapsed_seconds']=time.monotonic()-start
        record['artifacts']={p.name:sha(p) for p in sorted(WORK.iterdir()) if p.is_file() and p.name!='results.json'}
        save()

if __name__=='__main__':main()
