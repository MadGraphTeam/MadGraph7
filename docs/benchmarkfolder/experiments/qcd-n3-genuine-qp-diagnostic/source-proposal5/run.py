#!/usr/bin/env python3
"""Proposed one-shot run5: inherit ten closed workers and run exactly six new ones.
This source does not authorize native execution; proposal tests use fake runners.
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
import exec_target
import donor_closure
import prefix_reader

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
    validate_target_closure(binding)
    donor=Path(binding['process'])/'check'
    if exec_target.interpreter(donor.read_bytes())!=binding['target_interpreter']:
        raise ValueError('Donor interpreter differs from static environment evidence')
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
    # Never expose target loader settings to utilities, compilers or Python.
    for key in tuple(result):
        if key.startswith('LD_'): result.pop(key)
    return result

def target_command(binding,exe,checksum,arguments,*,trace=False):
    return prefix(binding)+[binding['programs']['python'],'-I','-B',str(HERE/'exec_target.py'),
        'trace' if trace else 'run',str(exe),checksum,binding['target_interpreter'],
        binding['native_library_path'],*map(str,arguments)]

def validate_target_closure(binding):
    historical=binding['historical_ldd_runtime']
    excluded=binding['excluded_host_ldd_interpreter']
    allowed='/nix/store/0d8g8n0a11v6f5m2h416ajyxmnkwc3md-glibc-2.42-67/lib64/ld-linux-x86-64.so.2'
    if set(excluded)!={allowed} or excluded[allowed]!=historical[allowed]:
        raise ValueError('Unexpected historical loader exclusion')
    if binding['runtime']!={p:h for p,h in historical.items() if p!=allowed}:
        raise ValueError('Target closure changed beyond the single host-loader exclusion')
    interp=binding['target_interpreter']
    if interp not in binding['runtime'] or interp==allowed:
        raise ValueError('Target interpreter must remain in the closure')
    prior=json.loads((HERE/'prior-source2-bindings.json').read_text())
    prior_index=json.loads((HERE/'prior-source2-index.json').read_text())
    if sha(HERE/'prior-source2-bindings.json')!=prior_index['files']['bindings.json']['sha256']:
        raise ValueError('Prior source2 binding copy differs from its sealed index')
    if binding['runtime']!=prior['runtime'] or binding['target_interpreter']!=prior['target_interpreter']:
        raise ValueError('Original source2 donor runtime or target interpreter changed')
    expected=str(WORK/'run5/closed-runtime')
    if binding['native_library_path']!=expected:
        raise ValueError('Native library path differs from private closed-runtime directory')
    manifest=json.loads((HERE/'donor-closure.json').read_text())
    donor_closure.validate_manifest(manifest,binding['runtime'])
    audit=json.loads((HERE/'static-closure-audit.json').read_text())
    metadata={x['literal_path']:x for x in audit['expected_runtime']}
    for entry in manifest['entries']:
        old=metadata[entry['source_path']]
        if (entry['soname'],entry['resolved_path'],entry['sha256'])!=(old['soname'],old['resolved_path'],old['sha256']):
            raise ValueError('Alias differs from the pinned ELF SONAME/provider metadata')

def inherited_build(binding):
    record=json.loads((HERE/'build2-reuse.json').read_text())
    build_path=WORK/'build2/build.json'
    if sha(build_path)!=record['build2_json_sha256'] or sha(build_path)!=sha(HERE/'inherited-build2.json'):
        raise ValueError('Inherited build2 proof changed')
    proof=json.loads(build_path.read_text())
    source2_sha=sha(HERE/'prior-source2-index.json')
    if proof.get('passed') is not True or proof['source_index_sha256']!=source2_sha or record['source2_index_sha256']!=source2_sha:
        raise ValueError('Successful matching source2 build2 required')
    if proof['executable_sha256']!=record['build2_executable_sha256'] or proof['link_map_sha256']!=record['build2_link_map_sha256']:
        raise ValueError('Inherited build2 executable or link map identity differs')
    if proof['runtime']!=binding['runtime'] or proof['target_interpreter']!=binding['target_interpreter']:
        raise ValueError('Inherited build2 donor runtime changed')
    exe=Path(proof['executable'])
    verify_files({str(exe):proof['executable_sha256'],**proof['objects'],**proof['original_objects']})
    if exec_target.interpreter(exe.read_bytes())!=binding['target_interpreter']:
        raise ValueError('Inherited build2 target interpreter mismatch')
    return build_path,proof,exe

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
              'seconds_cap':min(cap,deadline-time.monotonic()),'environment':{k:v for k,v in env(binding).items() if k in ('LC_ALL','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
              'target_loader_environment_at_execve':({'LD_LIBRARY_PATH':binding.get('native_library_path'),**({'LD_TRACE_LOADED_OBJECTS':'1'} if label=='loader' else {})} if native or label=='loader' else {})}
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
    raise ValueError('Source5 reuses immutable build2; no compilation stage is admitted')

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

def run_worker(spec,attempt,*,binding,proof,exe,manifest,deadline):
    """The original per-worker traces, maps, caps, parser and integrity gates."""
    backend,mode,observation=spec
    folder=attempt/prefix_reader.worker_name(spec);folder.mkdir()
    inputs=fixture_inputs(folder,backend); dump(folder/'inputs.json',inputs)
    try:
        source_check();inherited_build(binding)
        dump(folder/'closed-runtime-pretrace.json',donor_closure.verify(attempt/'closed-runtime',manifest,binding['runtime']))
        # Separate pre-entry loader trace; actual worker maps remain observational.
        run_command(target_command(binding,exe,proof['executable_sha256'],[],trace=True),folder,'loader',
            binding=binding,deadline=deadline,cap=30)
        validate_loader((folder/'loader.log').read_text(),binding)
        source_check();inherited_build(binding)
        dump(folder/'closed-runtime-preworker.json',donor_closure.verify(attempt/'closed-runtime',manifest,binding['runtime']))
        run_command(target_command(binding,exe,proof['executable_sha256'],[backend,mode,observation]),folder,'worker',
            binding=binding,deadline=deadline,cap=180,native=True,exe=exe)
        donor_closure.verify(attempt/'closed-runtime',manifest,binding['runtime'])
        verify_files(inputs);source_check();verify_files({str(exe):proof['executable_sha256']})
        parsed=analyze.parse_worker((folder/'trace.txt').read_text(),(folder/'worker.log').read_text(),
            backend,mode,observation,point_bytes())
        dump(folder/'parsed.json',parsed)
        return parsed
    finally:
        # Retain every actual warning even if process or trace validation failed.
        text=(folder/'worker.log').read_text(errors='replace') if (folder/'worker.log').exists() else ''
        dump(folder/'warnings.json',[{'line':i,'text':line} for i,line in enumerate(text.splitlines(),1) if re.search(r'##W03\b',line,re.I)])
        dump(folder/'outputs.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='outputs.json'})


def run():
    with numerical_lock():
        # Busy75 exits before this point. All continuation validation/IO counts.
        started=time.monotonic();deadline=started+float(prefix_reader.NEW_BUDGET)
        attempt=WORK/'run5';attempt.mkdir(exist_ok=False)
        all_workers=[];inherited_count=0;new_count=0
        result={'passed':False,'original_donor_runtime_preserved':True,'recompiled':False,
                'original_source4_attempt_remains_failed':True,
                'inherited_elapsed_seconds':float(prefix_reader.OLD_ELAPSED),
                'new_phase_budget_seconds':float(prefix_reader.NEW_BUDGET),
                'new_phase_budget_seconds_decimal':str(prefix_reader.NEW_BUDGET),
                'combined_budget_seconds':3000,'required_inherited_workers':10,'required_new_workers':6,
                'new_worker_order':[prefix_reader.worker_name(spec) for spec in prefix_reader.SUFFIX_SPECS]}
        try:
            binding=source_check();build_path,proof,exe=inherited_build(binding)
            result.update(build_sha256=sha(build_path),source_index_sha256=sha(HERE/'source-index.json'),
                          inherited_source2_index_sha256=proof['source_index_sha256'])
            prefix=prefix_reader.load(HERE,binding)
            all_workers=list(prefix['workers']);inherited_count=len(all_workers)
            dump(attempt/'inherited-prefix.json',{key:value for key,value in prefix.items() if key!='workers'})
            if time.monotonic()>=deadline: raise TimeoutError('Shared continuation budget expired during prefix verification')
            manifest=json.loads((HERE/'donor-closure.json').read_text())
            dump(attempt/'closed-runtime-proof.json',donor_closure.create(attempt/'closed-runtime',manifest,binding['runtime']))
            run_command(target_command(binding,exe,proof['executable_sha256'],[],trace=True),attempt,'loader',binding=binding,deadline=deadline,cap=30)
            validate_loader((attempt/'loader.log').read_text(),binding)
            for spec in prefix_reader.SUFFIX_SPECS:
                if time.monotonic()>=deadline: raise TimeoutError('Shared continuation budget expired before new worker')
                all_workers.append(run_worker(spec,attempt,binding=binding,proof=proof,exe=exe,manifest=manifest,deadline=deadline))
                new_count+=1
            prefix_reader.require_complete_matrix(all_workers)
            analysis=analyze.adjudicate(all_workers);dump(attempt/'analysis.json',analysis)
            if time.monotonic()>deadline: raise TimeoutError('Shared continuation budget expired after verification')
            result.update(passed=analysis['all_shared_input_gates_pass'],analysis_sha256=sha(attempt/'analysis.json'))
            if not result['passed']: raise ValueError('Shared effective input equality gate failed')
        except BaseException as error:
            result['failure']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            result.update(inherited_workers=inherited_count,new_workers_parsed=new_count,workers_completed=len(all_workers))
            execution_records=[json.loads(path.read_text()) for path in attempt.glob('b*/worker.json')]
            result['new_workers_native_started']=sum(record.get('process_spawned') is True for record in execution_records)
            result['new_workers_native_reaped']=sum(record.get('process_reaped') is True for record in execution_records)
            result['elapsed_seconds']=time.monotonic()-started
            result['combined_elapsed_seconds']=float(prefix_reader.OLD_ELAPSED)+result['elapsed_seconds']
            if (result['elapsed_seconds']>float(prefix_reader.NEW_BUDGET) or
                result['combined_elapsed_seconds']>3000): result['passed']=False
            dump(attempt/'result.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('verify','run'))
    args=parser.parse_args()
    with interruptions():
        if args.stage=='verify':
            binding=source_check();inherited_build(binding)
            print('Source/dependency/inherited-build2 binding PASS; no compilation or native execution.')
        else: run()
if __name__=='__main__': main()
