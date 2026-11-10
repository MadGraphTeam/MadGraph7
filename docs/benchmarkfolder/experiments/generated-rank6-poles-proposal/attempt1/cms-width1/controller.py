"""One private driver rebuild; later one bounded CMS-only native worker."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[4]
OLD_WORK=ROOT/'.tirust-build/generated-rank6-poles-1'
WORK=OLD_WORK/'cms-width1'
BUILD=WORK/'build1'
RUN=WORK/'run1'
INDEX=HERE/'source-index.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def verify():
    assert Path.cwd()==ROOT
    index=json.loads(INDEX.read_text())
    for name,digest in index['files'].items():assert sha(HERE/name)==digest,name
    for name,digest in index['dependencies'].items():assert sha(ROOT/name)==digest,name
    previous=runpy.run_path(str(PARENT/'run.py'))
    plan,build=previous['verify']()
    old=(PARENT/'materialized/ew/driver.f').read_text()
    before='M20(3,5)=CMPLX(MDL_MZ**2,-MDL_MZ*MDL_WZ,KIND=8)'
    after='M20(3,5)=CMPLX(MDL_MZ**2,-MDL_MZ*2.498770D0,KIND=8)'
    assert old.count(before)==1 and (HERE/'driver.f').read_text()==old.replace(before,after)
    fixture=json.loads((HERE/'fixture.json').read_text())
    assert sha(HERE/'width-origin-card.dat')==sha(ROOT/fixture['origin'])==fixture['sha256']
    assert re.findall(r'^DECAY\s+23\s+(\S+)',(HERE/'width-origin-card.dat').read_text(),re.M)==['2.498770e+00']
    allrows=json.loads((PARENT/'schedule.json').read_text());rows=[r for r in allrows if r['fixture']==105]
    assert rows==json.loads((HERE/'schedule.json').read_text()) and len(rows)==15
    keys=('action','fixture','backend','ct','mode','finite','cache','budget','clear','n','rank')
    assert (HERE/'schedule.dat').read_text()=='15\n'+''.join(' '.join(str(r[k]) for k in keys)+'\n' for r in rows)
    return previous,plan,build

def handlers():
    def interrupted(signum,frame):raise InterruptedError('CMS diagnostic interrupted by signal %d'%signum)
    return {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}

def commands(build):
    old=OLD_WORK/'ew';source=PARENT/'materialized/ew/driver.f'
    compile_row=build['commands'][4];link_row=build['commands'][6]
    compile_argv=[compile_row['compiler_path']]+compile_row['argv'][1:]
    replace={str(source):str(HERE/'driver.f'),str(old/'driver.o'):str(BUILD/'driver.o'),'-J'+str(old):'-J'+str(BUILD)}
    compile_argv=[replace.get(arg,arg) for arg in compile_argv]
    link_argv=[link_row['compiler_path']]+link_row['argv'][1:]
    replace={str(old/'rank6-poles'):str(BUILD/'rank6-cms'),str(old/'driver.o'):str(BUILD/'driver.o'),
             '-Wl,-Map,'+str(old/'link.map'):'-Wl,-Map,'+str(BUILD/'link.map')}
    return [compile_argv,[replace.get(arg,arg) for arg in link_argv]]

def build():
    prior,plan,old=verify();helper,adapted=prior['supervisor']()
    BUILD.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{12})
    resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
    env={key:value for key,value in old['compiler_environment'].items() if value is not None}
    if 'HOME' in os.environ:env['HOME']=os.environ['HOME']
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    original_loads={str(Path(name).resolve()) for name in re.findall(r'^LOAD\s+(.+)$',(OLD_WORK/'ew/link.map').read_text(),re.M)}
    bound={name:sha(name) for name in sorted(original_loads)}
    record={'complete':False,'scope':'one private driver compile and relink only; no loader/native call','source_index_sha256':sha(INDEX),
            'parent_build_sha256':sha(PARENT/'build1/build.json'),'cpu':12,'exclusive_cpu':False,'jobs':1,'shared_cap_seconds':120,
            'address_space_bytes':4294967296,'environment':env,'prebound_link_inputs':bound,'commands':[]}
    start=time.monotonic();previous=handlers()
    try:
        for number,argv in enumerate(commands(old)):
            remaining=120-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Shared CMS compile cap expired')
            row={'argv':argv,'progress':{},'log':str(number)+'.log'};record['commands'].append(row);save(BUILD/'build.json',record)
            if number==1:bound[str(BUILD/'driver.o')]=sha(BUILD/'driver.o')
            with (BUILD/row['log']).open('xb') as log:
                row['exit_code']=helper['run_plain_capped'](argv,log,cwd=ROOT,env=env,progress=row['progress'],seconds=remaining)
            row['log_sha256']=sha(BUILD/row['log']);save(BUILD/'build.json',record)
            assert row['exit_code']==0
        actual={str(Path(name).resolve()) for name in re.findall(r'^LOAD\s+(.+)$',(BUILD/'link.map').read_text(),re.M)}
        expected=original_loads-{str(OLD_WORK/'ew/driver.o')}|{str(BUILD/'driver.o')}
        assert actual==expected
        assert all(sha(name)==bound[name] for name in actual)
        record['actual_link_inputs']={name:bound[name] for name in sorted(actual)}
        verify();record['complete']=True
    except BaseException as error:record['failure']=repr(error);raise
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
        record['elapsed_seconds']=time.monotonic()-start
        record['artifacts']={p.name:sha(p) for p in BUILD.iterdir() if p.is_file() and p.name!='build.json'}
        save(BUILD/'build.json',record)

def built():
    prior,plan,_=verify();build=json.loads((BUILD/'build.json').read_text())
    assert build['complete'] and build['source_index_sha256']==sha(INDEX)
    assert len(build['commands'])==2 and all(r['exit_code']==0 for r in build['commands'])
    for name,digest in build['artifacts'].items():assert sha(BUILD/name)==digest
    for name,digest in build['actual_link_inputs'].items():assert sha(name)==digest
    prior['interpreter'](BUILD/'rank6-cms')
    return prior,plan,build

def run():
    prior,plan,_=built();helper,adapted=prior['supervisor']();os.sched_setaffinity(0,{12})
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(75)
        RUN.mkdir(exist_ok=False)
        for name in ('point.dat','nominal/param_card.dat','nominal/ident_card.dat'):
            target=RUN/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD_WORK/'ew'/name,target)
        shutil.copyfile(HERE/'schedule.dat',RUN/'schedule.dat')
        inputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file()}
        env=prior['worker_environment'](plan['models']['ew'])
        resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
        record={'complete':False,'source_index_sha256':sha(INDEX),'build_sha256':sha(BUILD/'build.json'),
                'scope':'two captures and15 private CMS tensor rows only; no timing/physical-amplitude qualification',
                'cpu':12,'exclusive_cpu':False,'worker_seconds':900,'address_space_bytes':4294967296,'inputs':inputs,
                'loader':{'argv':[str(prior['LOADER']),'--list',str(BUILD/'rank6-cms')],'progress':{}},'environment_overrides':{
                    k:env[k] for k in ('LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS')}}
        start=time.monotonic();previous=handlers()
        def checkpoint():save(RUN/'execution.json',record)
        try:
            checkpoint()
            with (RUN/'loader.log').open('xb') as log:
                record['loader']['exit_code']=helper['run_plain_capped'](record['loader']['argv'],log,cwd=ROOT,env=env,progress=record['loader']['progress'],seconds=30)
            assert record['loader']['exit_code']==0
            record['runtime_sha256']=prior['closure']((RUN/'loader.log').read_text(),plan)
            old_runtime=json.loads((PARENT/'runtime1/execution.json').read_text())['workers']['ew']['runtime_sha256']
            assert record['runtime_sha256']==old_runtime
            record['argv']=[str(BUILD/'rank6-cms')];record['progress']={};checkpoint()
            with (RUN/'console.log').open('xb') as log:
                record['exit_code'],record['mapping']=helper['run_captured'](record['argv'],log,cwd=RUN,env=env,
                    expected_runtime=record['runtime_sha256'],executable=str(BUILD/'rank6-cms'),selected_library=str(prior['BRIDGE']),
                    maps_path=RUN/'maps.txt',progress=record['progress'],seconds=900)
            checkpoint();assert record['exit_code']==0
            for name,digest in inputs.items():assert sha(RUN/name)==digest
            result=runpy.run_path(str(HERE/'analyze.py'))['analyze'](RUN)
            save(RUN/'results.json',result);record['results_sha256']=sha(RUN/'results.json')
            built();record['complete']=True
        except BaseException as error:record['failure']=repr(error);raise
        finally:
            for signum,handler in previous.items():signal.signal(signum,handler)
            record['elapsed_seconds']=time.monotonic()-start
            record['outputs']={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name!='execution.json'}
            checkpoint()

if __name__=='__main__':
    assert len(sys.argv)<=2
    stage=sys.argv[1] if len(sys.argv)==2 else 'verify'
    if stage=='verify':verify();print('CMS private source/dependency verification passed; no compiler/native execution.')
    elif stage=='build':build()
    elif stage=='run':run()
    else:raise SystemExit('Use verify, build, or run')
