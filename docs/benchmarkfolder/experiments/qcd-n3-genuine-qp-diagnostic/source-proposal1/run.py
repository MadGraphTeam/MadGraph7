#!/usr/bin/env python3
"""Future one-shot compile/run stages. Neither stage is authorized by this file.
Source proposal tests use injected fake supervisors, never these real stages.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import sys
import time
import analyze
import supervise

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
WORK=ROOT/'.tirust-build/qcd-n3-genuine-qp-diagnostic'
WRAPS=('ml5_0_smatrixhel_splitorders_','ml5_0_improve_ps_point_precision_',
       'ml5_0_mp_improve_ps_point_precision_','ml5_0_mp_compute_loop_coefs_',
       'mp_update_as_param_','ml5_0_mp_helas_calls_ampb_1_','ml5_0_mp_coef_construction_1_',
       'ml5_0_mploopnum_','ml5_0_mp_switch_order_','ml5_0_loop_ct_calls_1_',
       'ctsxcut_','quadninja_tensor_evaluate_smat_cm_',
       'ninjars_evaluate_mp_with_poles','ml_tirust_evaluate_mp')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value): path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def verify_files(files):
    for name,expected in files.items():
        if sha(Path(name))!=expected: raise ValueError('Bound input changed: '+name)

def source_check():
    index=json.loads((HERE/'source-index.json').read_text())
    verify_files({str(HERE/name):item['sha256'] for name,item in index['files'].items()})
    binding=json.loads((HERE/'bindings.json').read_text())
    verify_files(binding['dependencies'])
    return binding

def env(binding):
    result=dict(os.environ)
    for key in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH','LD_DEBUG','LD_PROFILE','GCONV_PATH',
                'FFLAGS','FCFLAGS','CFLAGS','CPPFLAGS','CXXFLAGS','LDFLAGS','FC','CC','CXX',
                'LIBRARY_PATH','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','COMPILER_PATH',
                'GCC_EXEC_PREFIX','GFORTRAN_UNBUFFERED_ALL','GFORTRAN_ERROR_BACKTRACE'):
        result.pop(key,None)
    result.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  RAYON_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    result['LD_LIBRARY_PATH']=':'.join(dict.fromkeys(str(Path(p).parent) for p in binding['runtime']))
    return result

def prefix(binding):
    p=binding['programs']
    return [p['taskset'],'-c','12',p['prlimit'],'--as=4294967296','--']

@contextmanager
def interruptions():
    def stop(signum,frame): raise InterruptedError('Controller interrupted: '+str(signum))
    old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try: yield
    finally:
        for sig,handler in old.items(): signal.signal(sig,handler)

@contextmanager
def numerical_lock():
    # Acquire before making an attempt. Busy75 is a hold, not a native attempt.
    path=ROOT/'.tirust-build/numerical-tests.lock'
    with path.open('a+') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit(75)
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)

def run_command(argv,folder,label,*,binding,deadline,cap=120,native=False,exe=None,runner=None):
    """One reviewed owned-session call; metadata retained even on interruption."""
    progress={'argv':argv,'cwd':str(folder),'native':native,'started_unix':time.time(),
              'seconds_cap':min(cap,deadline-time.monotonic()),'environment':{k:v for k,v in env(binding).items() if k in ('LC_ALL','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','LD_LIBRARY_PATH')}}
    if progress['seconds_cap']<=0: raise TimeoutError('Shared stage budget expired')
    runner=runner or (supervise.run_captured if native else supervise.run_plain_capped)
    try:
        with (folder/(label+'.log')).open('xb') as log:
            if native:
                code,maps=runner(argv,log,cwd=folder,env=env(binding),progress=progress,
                    seconds=progress['seconds_cap'],expected_runtime={str(Path(p).resolve()):h for p,h in binding['runtime'].items()},
                    executable=str(exe.resolve()),selected_library=next(p for p in binding['runtime'] if Path(p).name=='libmadloop_tirust.so'),
                    maps_path=folder/'maps.txt')
                dump(folder/'mapped-runtime.json',maps)
            else:
                code=runner(argv,log,cwd=folder,env=env(binding),progress=progress,seconds=progress['seconds_cap'])
            progress['returncode']=code
            if code: raise RuntimeError(f'{label} returned {code}')
    except BaseException as error:
        progress['failure']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        progress['finished_unix']=time.time()
        dump(folder/(label+'.json'),progress)

def commands(binding,folder):
    p=binding['programs']; process=binding['process']; donor=binding['donor']
    common=[p['fc'],'-w','-fPIC','-I'+process,'-I'+str(folder)]
    objects=[str(folder/name) for name in ('observer_f.o','observer_c.o','driver.o')]
    link=[p['fc'],'-w','-fPIC','-o',str(folder/'diagnostic'),*objects,*binding['objects'],
          '-L'+donor+'/lib','-ldhelas','-lmodel',*binding['link_loop_libs'],
          *['-Wl,-rpath,'+d for d in dict.fromkeys(str(Path(x).parent) for x in binding['runtime'])],
          '-Wl,-Map,'+str(folder/'link.map'),*['-Wl,--wrap='+x for x in WRAPS]]
    return [
      [*common,'-ffree-line-length-none','-J'+str(folder),'-c',str(HERE/'observer.f90'),'-o',objects[0]],
      [p['cc'],'-std=c11','-O0','-fPIC','-fno-fast-math','-c',str(HERE/'observer.c'),'-o',objects[1]],
      [*common,'-ffixed-line-length-132','-c',str(HERE/'driver.f'),'-o',objects[2]],link]

def build():
    binding=source_check(); folder=WORK/'build1'; folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+120
    plan=commands(binding,folder);dump(folder/'plan.json',plan)
    result={'source_index_sha256':sha(HERE/'source-index.json'),'passed':False}
    try:
        for i,command in enumerate(plan):
            run_command(prefix(binding)+command,folder,f'command-{i+1}',binding=binding,deadline=deadline)
        source_check()
        if time.monotonic()>deadline: raise TimeoutError('Shared build budget expired after verification')
        result.update(passed=True,executable=str(folder/'diagnostic'),executable_sha256=sha(folder/'diagnostic'),
            objects={str(p):sha(p) for p in folder.glob('*.o')},original_objects={x:sha(Path(x)) for x in binding['objects']},
            link_map_sha256=sha(folder/'link.map'),runtime=binding['runtime'])
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        if result['elapsed_seconds']>120: result['passed']=False
        dump(folder/'build.json',result)

def card_for_backend(raw,backend):
    replacement,count=re.subn(rb'(?im)^(#MLReductionLib\s*\n)[^\r\n]+',
        lambda m:m.group(1)+str(backend).encode(),raw)
    if count!=1: raise ValueError('Expected exactly one backend field')
    return replacement

def fixture_inputs(folder,backend):
    # Original physical bytes/resources; only the public backend card differs.
    original=HERE/'fixtures'
    for p in sorted(original.rglob('*')):
        if not p.is_file() or p.name=='selected-inputs.json': continue
        target=folder/p.relative_to(original);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(card_for_backend(p.read_bytes(),backend) if p.name=='MadLoopParams.dat' else p.read_bytes())
    return {str(p):sha(p) for p in folder.rglob('*') if p.is_file()}

def point_bytes():
    meta=json.loads((HERE/'fixtures/selected-inputs.json').read_text())
    return {p:b''.join(struct.pack('<d',float.fromhex(x)) for row in meta['points'][str(p)]['binary64_hex'] for x in row) for p in (1,93,39)}

def validate_loader(text,binding):
    if 'not found' in text: raise ValueError('Missing runtime library')
    names={str(Path(m.group(0)).resolve()) for m in re.finditer(r'/[^\s()]+',text)}
    expected={str(Path(p).resolve()) for p in binding['runtime']}
    if names!=expected: raise ValueError('Loader closure differs from closed binding: '+repr((sorted(names-expected),sorted(expected-names))))

def run():
    binding=source_check(); build_path=WORK/'build1/build.json'; proof=json.loads(build_path.read_text())
    if proof.get('passed') is not True or proof['source_index_sha256']!=sha(HERE/'source-index.json'):
        raise ValueError('Successful matching private build required')
    exe=Path(proof['executable']); verify_files({str(exe):proof['executable_sha256'],**proof['objects'],**proof['original_objects']})
    with numerical_lock():
        attempt=WORK/'run1';attempt.mkdir(exist_ok=False)
        started=time.monotonic();deadline=started+3000; all_workers=[]
        result={'passed':False,'build_sha256':sha(build_path),'source_index_sha256':sha(HERE/'source-index.json')}
        try:
            run_command(prefix(binding)+[binding['programs']['ldd'],str(exe)],attempt,'loader',binding=binding,deadline=deadline,cap=30)
            validate_loader((attempt/'loader.log').read_text(),binding)
            for backend in analyze.BACKENDS:
                for mode in (4,5):
                    for observation in (0,1):
                        folder=attempt/f'b{backend}-m{mode}-o{observation}';folder.mkdir()
                        inputs=fixture_inputs(folder,backend); dump(folder/'inputs.json',inputs)
                        try:
                            run_command(prefix(binding)+[str(exe),str(backend),str(mode),str(observation)],folder,'worker',
                                binding=binding,deadline=deadline,cap=180,native=True,exe=exe)
                            verify_files(inputs);source_check(); verify_files({str(exe):proof['executable_sha256']})
                            parsed=analyze.parse_worker((folder/'trace.txt').read_text(),(folder/'worker.log').read_text(),
                                backend,mode,observation,point_bytes())
                            dump(folder/'parsed.json',parsed);all_workers.append(parsed)
                        finally:
                            # Retain every actual warning even if process or trace validation failed.
                            text=(folder/'worker.log').read_text(errors='replace') if (folder/'worker.log').exists() else ''
                            dump(folder/'warnings.json',[{'line':i,'text':line} for i,line in enumerate(text.splitlines(),1) if re.search(r'##W03\b',line,re.I)])
                            dump(folder/'outputs.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='outputs.json'})
            analysis=analyze.adjudicate(all_workers); dump(attempt/'analysis.json',analysis)
            if time.monotonic()>deadline: raise TimeoutError('Shared native-stage budget expired after verification')
            result.update(passed=analysis['all_shared_input_gates_pass'],analysis_sha256=sha(attempt/'analysis.json'),workers_completed=16)
            if not result['passed']: raise ValueError('Shared effective input equality gate failed')
        except BaseException as error:
            result.update(failure={'type':type(error).__name__,'message':str(error)},workers_completed=len(all_workers))
            raise
        finally:
            result['elapsed_seconds']=time.monotonic()-started
            if result['elapsed_seconds']>3000: result['passed']=False
            dump(attempt/'result.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('verify','plan','build','run'))
    args=parser.parse_args()
    with interruptions():
        if args.stage=='verify': source_check(); print('Source/dependency binding PASS; no compilation or native execution.')
        elif args.stage=='plan': print(json.dumps(commands(source_check(),WORK/'build1'),indent=2))
        elif args.stage=='build': build()
        else: run()
if __name__=='__main__': main()
