#!/usr/bin/env python3
"""One bounded private Fortran build, separately admitted diagnostic workers."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path.cwd()
HERE=Path(__file__).resolve().parent
PROPOSAL=ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-1'
CAMPAIGN='entry-latch-cache-guard-1'
CASE_ID='qcd_n3_uux_zgg'
PROC_NAME='P0_uux_zgg'
CASE=ROOT/'.benchmark-work/tirust/campaigns'/CAMPAIGN/'outputs/candidate'/CASE_ID
PROC=CASE/'SubProcesses'/PROC_NAME
WORK=ROOT/'.tirust-build/fortran-qp-transport-profile-controller-1'
BUILD=WORK/'build'
PARENT_SEAL='049c26fbf460e0102269bdd237a71fd8d2438160f0fa47afeb57be59aa218ba2'
DSO_SHA='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
CPU=12
CAPS={'build':180.,'mode-0':300.,'mode-1':300.,'mode-2':300.}
CLEANUP=60.
LIMITS={'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0}
EXPECTED_SOURCE=None
REVIEWED_BUILD=None
SUPPORT={
 'cleanup':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py','240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
 'deadline':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py','726f629210a155b894f423ad76fa1de9d96b1f98270d4aceba2ba240250d6901'),
}
sys.path.insert(0,str(ROOT/'tools'))
import tirust_benchmark as bench
import tirust_benchmark_compiler as compiler
from tirust_benchmark_worker import Worker
from tirust_benchmark_warmup import parse_warmup


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def dump(path,value,exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load(name,path,expected=None):
    if expected is not None and digest(path)!=expected:raise ValueError('Changed support: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def support():
    return tuple(load('qp_profile_'+name,ROOT/path,sha) for name,(path,sha) in SUPPORT.items())


def safe_files(base,files):
    for relative,expected in files.items():
        name=Path(relative);path=base/name
        if name.is_absolute() or '..' in name.parts or not path.is_file():raise ValueError('Unsafe/missing sealed file: '+relative)
        if digest(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise ValueError('Changed sealed file: '+relative)


def verify_seal():
    if EXPECTED_SOURCE is None or digest(HERE/'source-index.json')!=EXPECTED_SOURCE:
        raise ValueError('Explicit reviewed source index is required and must match')
    if digest(PROPOSAL/'index.json')!=PARENT_SEAL:raise ValueError('Original proposal seal changed')
    parent_files=read(PROPOSAL/'index.json')['files']
    safe_files(PROPOSAL,parent_files)
    if {str(p.relative_to(PROPOSAL)) for p in PROPOSAL.rglob('*') if p.is_file()}!=set(parent_files)|{'index.json'}:
        raise ValueError('Original proposal exact file inventory changed')
    safe_files(HERE,read(HERE/'source-index.json')['files'])
    safe_files(ROOT,read(HERE/'source-index.json')['support'])


def protected():
    """Pure file hashes, including exact source absence; no loader/compiler call."""
    pin=read(HERE/'input-pins.json')
    for path,expected in pin['files'].items():
        file=ROOT/path if not Path(path).is_absolute() else Path(path)
        if expected is None:
            wanted=pin['null_symlink_targets'].get(path)
            actual_link=str(file.readlink()) if file.is_symlink() else None
            if file.exists() or actual_link!=wanted:raise ValueError('Expected absent target/link identity changed: '+path)
        actual=digest(file) if file.is_file() else None
        if actual!=expected:raise ValueError('Protected input changed: '+path)
    original=read(PROPOSAL/'input-pins.json')
    for item in original['files']:
        p=ROOT/item['path']
        if digest(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('Original proposal input changed: '+str(p))
    receipt=read(CASE/'compilation.json')
    if bench.generated_sources(CASE)!=receipt['sources']:raise ValueError('Original generated sources changed')
    if bench.numerical_objects(CASE)!=pin['original_objects']:raise ValueError('Original numerical objects changed')
    if bench.dependency_inputs(PROC)!=receipt['link_inputs_after']:raise ValueError('Original static dependencies changed')
    if (PROC/'worker-protocol.fifo').exists():raise ValueError('Original process has a worker FIFO')
    return {'input_pins_sha256':digest(HERE/'input-pins.json'),'original_objects':pin['original_objects'],
            'original_executable_sha256':digest(PROC/'check'),'files_checked':len(pin['files'])}


def env():
    native=read(PROPOSAL/'input-pins.json')['runtime']['libraries']
    dirs=sorted({str(Path(p).parent) for p in native if 'libmadloop_tirust' in p})
    if len(dirs)!=1:raise ValueError('Expected one frozen native prefix')
    # runtime_identity() internally calls the existing environment helper too.
    bench.TIRUST_PREFIX=Path(dirs[0]).parent
    value=bench.environment()
    value['LD_LIBRARY_PATH']=os.pathsep.join(dirs+[value.get('LD_LIBRARY_PATH','')])
    return value


def preflight():
    verify_seal()
    if CPU not in os.sched_getaffinity(0):raise ValueError('Nonexclusive CPU12 unavailable')
    evidence=protected()
    for path,sha in read(PROPOSAL/'input-pins.json')['runtime']['libraries'].items():
        if digest(path)!=sha:raise ValueError('Runtime file changed')
    return {'status':'source-only-preflight-passed','protected':evidence,'native_executions':0,
        'compiler_invocations':0,'caps_seconds':CAPS,'cleanup_reserve_seconds':CLEANUP,'resource_limits':LIMITS,
        'source_index_sha256':EXPECTED_SOURCE}


def commands(command,cwd,logs,name,environment):
    logs.mkdir(exist_ok=True,parents=True)
    record={'command':list(map(str,command)),'cwd':str(cwd),'started_utc':now(),
            'owned_controller_pid':os.getpid(),'controller_session':os.getsid(0),'controller_process_group':os.getpgrp(),
            'subprocess_new_session':False}
    dump(logs/(name+'.command.json'),record)
    with (logs/(name+'.stdout.txt')).open('xb') as stream:
        done=subprocess.run(command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    record.update(returncode=done.returncode,finished_utc=now(),stdout_sha256=digest(logs/(name+'.stdout.txt')))
    dump(logs/(name+'.terminal.json'),record)
    if done.returncode:raise RuntimeError('Command failed: '+name)
    return (logs/(name+'.stdout.txt')).read_text()


def clone(source,destination):
    # Dereference source/include symlinks into private copies. Libraries remain
    # read-only references; no build command targets that shared directory.
    def ignore(directory,names):
        excluded={n for n in names if n=='worker-protocol.fifo' or n=='worker-raw'
                  or n.endswith(('HelFilter.dat','LoopFilter.dat'))}
        if Path(directory)==source:excluded.add('lib')
        return excluded
    shutil.copytree(source,destination,symlinks=False,ignore=ignore)
    (destination/'lib').symlink_to((source/'lib').resolve(),target_is_directory=True)
    for p in destination.rglob('*'):
        if p.is_symlink() and p!=destination/'lib':raise ValueError('Unexpected mutable private symlink')


def private_makefile(text):
    old='$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)'
    new='$(FC) $(FFLAGS) -Wl,-Map,private-link.map,--cref -o $(PROG) check_sa.o qp_profile.o mtirust.o $(PROCESS) $(LINKLIBS)'
    if text.count(old)!=1:raise ValueError('Ambiguous original link recipe')
    return text.replace(old,new)+'''
# Exact private profiling units: explicit recipes, including every new .f90.
qp_profile.o: qp_profile.f90
\t$(FC) -O3 -fPIC -c $< -o $@
mtirust.o: mtirust.f90 qp_profile.o
\t$(FC) -O3 -fPIC -c $< -o $@
check_sa.o TIRUST_interface.o TIR_interface.o: qp_profile.o mtirust.o
'''


def verify_recipes(original,actual):
    if original['environment']!=actual['environment']:raise ValueError('Compiler injected environment changed')
    if set(original['scopes'])!=set(actual['scopes']):raise ValueError('Compilation scopes changed')
    for scope,left_scope in original['scopes'].items():
        left,right=left_scope['recipes'],actual['scopes'][scope]['recipes']
        extras={'qp_profile.f90','mtirust.f90'} if scope=='process' else set()
        if set(right)!=set(left)|extras:raise ValueError('Missing or extra actual source recipe')
        for name in left:
            if compiler._signature(left[name])!=compiler._signature(right[name]):raise ValueError('Changed original recipe: '+name)
        for name in extras:
            if right[name]['compiler']!=left['TIRUST_interface.f']['compiler'] or right[name]['flags']!=['-O3','-fPIC']:
                raise ValueError('Unexpected explicit private module recipe: '+name)


def expanded_commands(directory,environment,logs):
    output=commands(['make','--no-print-directory','--dry-run','--always-make','check'],directory,logs,'expanded-check',environment)
    compile_by_name={};links=[]
    for line in output.replace('\\\n',' ').splitlines():
        tokens=shlex.split(line)
        if not tokens:continue
        if any(t in {';','&&','||','|','>','>>','<'} for t in tokens):raise ValueError('Compound build recipe')
        if '-c' in tokens:
            sources=[x for x in tokens if Path(x).suffix.lower() in compiler.FORTRAN_SUFFIXES and not x.startswith('-')]
            if len(sources)!=1:raise ValueError('Unresolved compile source')
            name=Path(sources[0]).name
            if name in compile_by_name and compile_by_name[name]!=tokens:raise ValueError('Duplicate conflicting source recipe')
            compile_by_name[name]=tokens
        elif '-o' in tokens and tokens[tokens.index('-o')+1]=='check':links.append(tokens)
    if len(links)!=1:raise ValueError('Expected exactly one check link')
    needed=['qp_profile.f90','mtirust.f90','TIR_interface.f','TIRUST_interface.f','check_sa.f']
    if not set(needed)<=set(compile_by_name):raise ValueError('Private build lacks a complete recipe')
    link=links[0]
    if link.count('mtirust.o')!=1 or link.index('mtirust.o')>=link.index('-ltirust'):raise ValueError('Private MTIRUST link precedence differs')
    return [(name,compile_by_name[name]) for name in needed],link


def defined_symbols(text):
    answer={}
    for line in text.splitlines():
        words=line.split()
        if len(words)>=3 and '__mtirust_MOD_' in words[-1]:answer[words[-1]]=answer.get(words[-1],0)+1
    return answer


def verify_symbols(module_text,executable_text,map_text):
    own,linked=defined_symbols(module_text),defined_symbols(executable_text)
    if not own or any(v!=1 for v in own.values()) or own!=linked:raise ValueError('MTIRUST symbols are missing/duplicated')
    if re.search(r'libtirust\.a\([^)]*mtirust[^)]*\)',map_text):raise ValueError('Original MTIRUST archive member was pulled')
    if 'LOAD mtirust.o' not in map_text:raise ValueError('Private MTIRUST object absent from link map')
    return {'definitions':own,'original_archive_member_loaded':False}


def loader_support():
    return load('qp_profile_loader_checks',HERE/'loader_checks.py')


def expected_actual_libraries():
    """Preserve old ldd evidence; exclude only its explicit foreign-loader alias."""
    old=read(PROPOSAL/'input-pins.json')['runtime'];files={str(Path(p).resolve()):h for p,h in old['libraries'].items()}
    actual=loader_support()._support('run.py',{'interpreter'}).interpreter(PROC/'check')
    removed={}
    for left,right in re.findall(r'^\s*(/[^\s]+)\s*=>\s*(/[^\s]+)\s*\(',old['ldd'],re.M):
        if Path(left).resolve()==actual and Path(right).resolve()!=actual:
            key=str(Path(right).resolve())
            if 'ld-linux' not in Path(key).name or key not in files:raise ValueError('Unexpected historical loader alias')
            removed[key]=files.pop(key)
    if str(actual) not in files:raise ValueError('Actual interpreter absent from frozen identity')
    return files,{'actual_interpreter':str(actual),'historical_ldd_only_interpreter_aliases':removed}


def guarded_runtime(directory,prebound,logs,label):
    expected,interpretation=expected_actual_libraries()
    callback=lambda argv,name:commands(argv,directory,logs,label+'-'+name,env())
    result=loader_support().query_runtime(directory/'check',expected,prebound,callback)
    return {'libraries':read(PROPOSAL/'input-pins.json')['runtime']['libraries'],
            **bench.dependency_inputs(directory),'actual_pt_interp':result,'historical_ldd_interpretation':interpretation}


def build_child():
    require_admission('build');verify_seal();before=protected();environment=env()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a+b') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        build_locked(before,environment)


def build_locked(before,environment):
    WORK.mkdir(parents=True,exist_ok=True);clone(CASE,BUILD)
    directory=BUILD/'SubProcesses'/PROC_NAME
    flags=load('qp_profile_flags',HERE/'worker_flags.py')
    for name in ['TIRUST_interface.f','TIR_interface.f','mtirust.f90','check_sa.f']:
        text=(PROPOSAL/'source-preview'/name).read_text()
        if name=='check_sa.f':text=flags.instrument_worker(text)
        (directory/name).write_text(text)
    shutil.copyfile(PROPOSAL/'qp_profile.f90',directory/'qp_profile.f90')
    (directory/'makefile').write_text(private_makefile((directory/'makefile').read_text()))
    original=read(CASE/'compilation.json')['effective_compiler_flags']
    actual=compiler.capture_effective_flags(directory,env=environment);verify_recipes(original,actual)
    logs=HERE/'build/commands';dump(HERE/'build/compiler-before.json',actual)
    recipes,link=expanded_commands(directory,environment,logs)
    original_objects=bench.numerical_objects(BUILD)
    for i,(source,command) in enumerate(recipes):commands(command,directory,logs,f'{i:02d}-compile-'+source,environment)
    loader=loader_support()
    prebound=loader.prelink(link,directory,environment,lambda argv,name:commands(argv,directory,logs,name,environment))
    expected,_=expected_actual_libraries()
    for path,sha in expected.items():
        if path in prebound['bound'] and prebound['bound'][path]!=sha:raise ValueError('Frozen runtime/prelink conflict')
        if digest(path)!=sha:raise ValueError('Frozen runtime changed before link')
        prebound['bound'][path]=sha
    dump(HERE/'build/prelink.json',prebound)
    commands(link,directory,logs,'05-link',environment)
    actual_loads=loader.requireactualmap((directory/'private-link.map').read_text(),directory,prebound,directory/'check')
    current=compiler.capture_effective_flags(directory,env=environment);verify_recipes(original,current)
    if not bench.same_compiler_policy(actual,current):raise ValueError('Compiler recipes changed during build')
    dump(HERE/'build/compiler-after.json',current)
    prefix='SubProcesses/'+PROC_NAME+'/'
    changed={prefix+n for n in ('TIR_interface.o','TIRUST_interface.o','check_sa.o')}
    added={prefix+n for n in ('mtirust.o','qp_profile.o','mtirust.mod','qp_profile_state.mod')}
    unchanged={k:v for k,v in original_objects.items() if k not in changed}
    observed={k:v for k,v in bench.numerical_objects(BUILD).items() if k not in changed|added}
    if unchanged!=observed:raise ValueError('Frozen non-profile numerical object changed')
    own=commands(['nm','--defined-only','mtirust.o'],directory,logs,'06-module-symbols',environment)
    linked=commands(['nm','--defined-only','check'],directory,logs,'07-executable-symbols',environment)
    symbols=verify_symbols(own,linked,(directory/'private-link.map').read_text())
    runtime=guarded_runtime(directory,prebound,logs,'build')
    expected=read(PROPOSAL/'input-pins.json')['runtime']
    if runtime['libraries']!=expected['libraries'] or runtime['archives']!=expected['archives'] or runtime['metadata']!=expected['metadata']:
        raise ValueError('Private link changed runtime providers or archive identities')
    if protected()!=before:raise ValueError('Protected originals changed')
    dump(HERE/'build/build-result.json',{'status':'passed','source_index_sha256':digest(HERE/'source-index.json'),
        'prelink_sha256':digest(HERE/'build/prelink.json'),'actual_link_loads':actual_loads,
        'compiler_flags':current,'sources':bench.generated_sources(BUILD),'objects':bench.numerical_objects(BUILD),
        'executable_sha256':digest(directory/'check'),'runtime':runtime,'symbols':symbols,
        'link_map_sha256':digest(directory/'private-link.map'),'private_driver_sha256':digest(directory/'check_sa.f'),
        'native_executions':0,'model_setup_calls':0,'protected':before})


def verify_build():
    value=read(HERE/'build/build-result.json')
    stage=read(HERE/'build/terminal.json')
    if (REVIEWED_BUILD is None or digest(HERE/'build/terminal.json')!=REVIEWED_BUILD
        or stage.get('build_result_sha256')!=digest(HERE/'build/build-result.json')):
        raise ValueError('Explicit coordinator-reviewed closed build hash is required')
    if value['status']!='passed' or stage['status']!='passed' or stage['cleanup']['survivors']:
        raise ValueError('Build is not closed and qualified')
    directory=BUILD/'SubProcesses'/PROC_NAME
    if value['prelink_sha256']!=digest(HERE/'build/prelink.json'):raise ValueError('Prelink receipt changed')
    prebound=read(HERE/'build/prelink.json')
    loads=loader_support().requireactualmap((directory/'private-link.map').read_text(),directory,prebound,directory/'check')
    if loads!=value['actual_link_loads']:raise ValueError('Actual link LOADs changed')
    if (value['source_index_sha256']!=digest(HERE/'source-index.json') or value['sources']!=bench.generated_sources(BUILD)
        or value['objects']!=bench.numerical_objects(BUILD) or value['executable_sha256']!=digest(directory/'check')):
        raise ValueError('Private build changed')
    return value


def exact_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        f=line.split()
        if len(f)!=9:raise ValueError('Malformed anchor row')
        call,point,order,status=map(int,f[:4]);values=list(map(float,f[4:]))
        if call!=point or point not in (1,16) or not all(math.isfinite(x) for x in values):raise ValueError('Invalid anchor row')
        rows.append({'point':point,'order':order,'status':status,'coefficients':values[:4],'accuracy':values[4]})
    expected=read(PROPOSAL/'expected.json')['rows']
    if len(rows)!=len(expected):raise ValueError('Incomplete selected anchors')
    for row,wanted in zip(rows,expected):
        if any(row[k]!=wanted[k] for k in ('point','order','status')):raise ValueError('Anchor status/order mismatch')
        if [struct.pack('>d',x).hex() for x in row['coefficients']]!=wanted['coefficient_binary64_big_endian_hex']:
            raise ValueError('Anchor coefficient words changed')
        if struct.pack('>d',row['accuracy']).hex()!=wanted['accuracy_binary64_big_endian_hex']:
            raise ValueError('Anchor accuracy words changed')
    return rows


def read_flags(path,mode):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        fields=list(map(int,line.split()))
        if len(fields)!=14 or fields[0]!=mode:raise ValueError('Malformed actual IEEE state')
        rows.append(fields[1:])
    if len(rows)!=4 or [x[:2] for x in rows]!=[[1,1],[1,2],[16,1],[16,2]]:raise ValueError('Incomplete actual IEEE state')
    if any(x not in (-1,0,1) for row in rows for x in row[3:]):raise ValueError('Invalid observed flag/halting value')
    if any(row[2] not in range(-1,5) for row in rows):raise ValueError('Invalid rounding observation')
    return rows


def validate_profile(prefix,mode):
    timer=Path(str(prefix)+'.timers.tsv');text=timer.read_text();rows=[]
    headers=dict(re.findall(r'^# (mode|clock_rate) ([0-9]+)$',text,re.M))
    if headers.get('mode')!=str(mode) or int(headers.get('clock_rate','0'))<=0:
        raise ValueError('Invalid clock/mode evidence')
    rate=int(headers['clock_rate']);seen=set()
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):continue
        values=list(map(int,line.split()))
        if len(values)!=7:raise ValueError('Malformed stage clock row')
        stage,ct,n,rank,precision,count,ticks=values;key=tuple(values[:5])
        if not (1<=stage<=7 and 0<=ct<=9 and 0<=n<=6 and 0<=rank<=6 and 0<=precision<=2 and count>0 and ticks>=0):
            raise ValueError('Invalid stage clock values')
        if key in seen or not math.isfinite(ticks/rate):raise ValueError('Invalid/duplicate stage clock scope')
        seen.add(key);rows.append({'key':list(key),'calls':count,'ticks':ticks,'seconds':ticks/rate})
    if (mode!=1 and rows) or (mode==1 and not rows):raise ValueError('Clock rows differ from selected mode')
    snapshots=[];record_count=0
    if mode:
        for line in Path(str(prefix)+'.calls.tsv').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            fields=line.split()
            if len(fields)!=24:raise ValueError('Malformed native counter trace')
            numbers=list(map(int,fields[:-1]));record_count+=1
            if numbers[0]!=record_count or numbers[1] not in range(1,7) or numbers[3]<0:
                raise ValueError('Invalid trace sequence/kind/epoch')
            absent_context=(numbers[1]==5 and numbers[13]==-1 and numbers[15:21]==[-1]*6)
            if any(x<0 for x in numbers[15:21]) and not absent_context:raise ValueError('Negative native cache counter')
            if numbers[1] in (2,3,5):
                if numbers[13]!=0 and not absent_context:raise ValueError('Native cache statistics query failed')
                snapshots.append({'row':numbers[0],'kind':numbers[1],'epoch':numbers[3],
                    'generation':numbers[12],'status':numbers[13],'counts':numbers[15:21]})
    return {'clock_rate':rate,'timers':rows,'native_counter_snapshots':snapshots,'trace_records':record_count,
            'interpretation':'Raw scoped intervals; overlapping stages must not be summed; no gain estimate.'}


def mapped_files(pid):
    raw=Path(f'/proc/{pid}/maps').read_text()
    files={}
    for line in raw.splitlines():
        fields=line.split(None,5)
        if len(fields)==6 and fields[-1].startswith('/'):
            path=fields[-1]
            if path.endswith(' (deleted)'):raise ValueError('Deleted mapped runtime file')
            files[path]=digest(path)
    native={p:h for p,h in files.items() if Path(p).name=='libmadloop_tirust.so'}
    if len(native)!=1 or next(iter(native.values()))!=DSO_SHA:raise ValueError('Actual native mapping differs')
    resolved={str(Path(p).resolve()):h for p,h in files.items()}
    for path,sha in expected_actual_libraries()[0].items():
        if resolved.get(str(Path(path).resolve()))!=sha:raise ValueError('Expected provider absent/different in actual process maps: '+path)
    return {'maps':raw,'files':files}


def private_binding(output,built,logs,label):
    directory=output/'SubProcesses'/PROC_NAME
    lane=read(ROOT/'docs/benchmarkfolder/campaigns'/CAMPAIGN/'experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/candidate-9/candidate-inputs.json')
    physical=bench.resolved_parameter_card(directory)
    if physical is None or digest(physical)!=lane['param_card_sha256']:raise ValueError('Private physical card differs')
    cards=bench.numerical_card_inputs(output)
    if cards!=lane['numerical_cards']:raise ValueError('Private numerical cards differ')
    runtime=guarded_runtime(directory,read(HERE/'build/prelink.json'),logs,label)
    if any(runtime[k]!=built['runtime'][k] for k in ('libraries','archives','metadata')):raise ValueError('Private loader identities changed')
    if (bench.generated_sources(output)!=built['sources'] or bench.numerical_objects(output)!=built['objects']
        or digest(directory/'check')!=built['executable_sha256']):raise ValueError('Private execution copy changed')
    return {'runtime':runtime,'param_card_sha256':digest(physical),'numerical_cards':cards,
            'executable_sha256':digest(directory/'check')}


def run_child(mode):
    require_admission(f'mode-{mode}');verify_seal();before=protected();built=verify_build()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a+b') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        run_locked(mode,before,built)


def run_locked(mode,before,built):
    report=HERE/f'mode-{mode}';output=WORK/f'mode-{mode}';clone(BUILD,output)
    directory=output/'SubProcesses'/PROC_NAME;environment=env()
    private_before=private_binding(output,built,report/'commands','before')
    dump(report/'private-inputs-before.json',private_before)
    point=ROOT/'docs/benchmarkfolder/campaigns'/CAMPAIGN/'points'/f'{CASE_ID}.dat'
    with Worker(['taskset','-c',str(CPU),str(directory/'check'),str(point)],directory,report/'worker.log',env=environment,timeout=240.) as worker:
        ready={k:getattr(worker,k) for k in ('points','ndim','nso','nborn','cold_seconds')}
        dump(report/'ready.json',ready)
        warmup=parse_warmup(directory,9,100,ready)
        reference=read(ROOT/'docs/benchmarkfolder/campaigns'/CAMPAIGN/'experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/candidate-9/raw/observation.json')
        if warmup['rows']!=reference['rows']:raise ValueError('Full learning traversal differs from closed current source')
        dump(report/'warmup.json',warmup)
        dump(report/'maps-before.json',mapped_files(worker.process.pid))
        worker._send(f'PROFILE 1 {mode}')
        if worker._message()!=['PROFILED','1',str(mode)]:raise ValueError('Private PROFILE did not complete exactly once')
        dump(report/'maps-after.json',mapped_files(worker.process.pid))
        rows=exact_rows(directory/'profile-values.dat')
        flags=read_flags(directory/'ieee-flags.dat',mode)
        if mode:
            reference=read(HERE/'mode-0/result.json')
            if reference['rows']!=rows or reference['ieee_state']!=flags:raise ValueError('Profile changed coefficients/status/accuracy/observed IEEE state')
    if (directory/'worker-protocol.fifo').exists():raise ValueError('Worker FIFO survived normal cleanup')
    raw=report/'raw';raw.mkdir()
    for path in sorted(directory.iterdir()):
        if path.name in ('warmup.dat','dimensions.dat','profile-values.dat','ieee-flags.dat') or path.name.startswith('point-'):
            shutil.copyfile(path,raw/path.name)
    profile={str(p):validate_profile(raw/f'point-{p:02d}',mode) for p in (1,16)}
    shadow=None
    if mode==2:
        analyzer=load('qp_shadow_analysis',PROPOSAL/'analyze_shadow.py')
        shadow={str(p):analyzer.inspect(raw/f'point-{p:02d}') for p in (1,16)}
    if protected()!=before:raise ValueError('Inputs changed during profile')
    private_after=private_binding(output,built,report/'commands','after')
    if any(private_after[k]!=private_before[k] for k in ('param_card_sha256','numerical_cards','executable_sha256')):
        raise ValueError('Private inputs changed during profile')
    dump(report/'private-inputs-after.json',private_after)
    warm_calls={int(line.split()[0]) for line in (raw/'warmup.dat').read_text().splitlines() if line.strip() and not line.startswith('#')}
    dump(report/'result.json',{'status':'passed','mode':mode,'rows':rows,'ieee_state':flags,'shadow':shadow,'profile':profile,'resource_limits':LIMITS,
         'warmup_points':100,'warmup_public_calls':len(warm_calls),'actual_anchor_calls':2,'performance_acceptance':False,'optimized':False,
         'private_runtime':private_after['runtime'],'build_result_sha256':digest(HERE/'build/build-result.json'),'protected':before})


def cleanup_owned(owned,end):
    budget=min(55.,max(0.,end-time.monotonic()))
    result=owned.stop(cooperative_seconds=budget*45/55,terminate_seconds=budget*5/55,kill_seconds=budget*5/55)
    if result['survivors']:raise RuntimeError('Owned cleanup is unresolved')
    return result


def require_admission(name):
    value=read(HERE/name/'admission.json')
    if (value['stage']!=name or value['cap_seconds']!=CAPS[name]
        or value['source_index_sha256']!=digest(HERE/'source-index.json')
        or value['reviewed_build_sha256']!=REVIEWED_BUILD or value['resource_limits']!=LIMITS):
        raise ValueError('Child lacks matching bounded-supervisor admission')
    actual={label:list(resource.getrlimit(kind)) for kind,label in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes'))}
    if actual!={k:[v,v] for k,v in LIMITS.items()}:raise ValueError('Actual child resource limits differ')
    dump(HERE/name/'resource-limits.json',{'actual':actual,'controller_pid':os.getpid(),'session':os.getsid(0),'process_group':os.getpgrp()})


def stage(name):
    cleanup,deadline_support=support();deadline_support.STAGE_SECONDS=CAPS[name];deadline_support.CLEANUP_RESERVE=CLEANUP
    directory=HERE/name;directory.mkdir(exist_ok=False)
    entry={'stage':name,'cap_seconds':CAPS[name],'cleanup_reserve_seconds':CLEANUP,'resource_limits':LIMITS,'started_utc':now(),'status':'running'}
    owned=None;cleaned=False
    try:
        with cleanup.SignalRequests() as requests,deadline_support.StageDeadline() as deadline:
            entry['deadline_monotonic']=deadline.end;requests.check();verify_seal();protected()
            command=['taskset','-c',str(CPU),sys.executable,'-B',str(Path(__file__).relative_to(ROOT)),
                     '--build-child' if name=='build' else '--run-child']
            if name!='build':command.append(name[-1])
            command.extend(['--source',EXPECTED_SOURCE])
            if name!='build':command.extend(['--reviewed-build',REVIEWED_BUILD])
            entry['command']=command
            dump(directory/'admission.json',{'stage':name,'cap_seconds':CAPS[name],
                 'source_index_sha256':digest(HERE/'source-index.json'),'command':command,
                 'reviewed_build_sha256':REVIEWED_BUILD,'resource_limits':LIMITS})
            with (directory/'console.log').open('xb') as stream:
                deadline.handoff=True
                try:
                    process=subprocess.Popen(command,cwd=ROOT,env=env(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    owned=cleanup.OwnedChild(process)
                finally:
                    deadline.handoff=False
                    if deadline.deferred:deadline.expired(None,None)
                entry['child_pid']=process.pid;dump(directory/'started.json',entry)
                print('START '+name+' child_pid='+str(process.pid),flush=True)
                entry['returncode']=owned.wait(requests);requests.check()
                entry['cleanup']=cleanup_owned(owned,deadline.end);cleaned=True
            if entry['returncode']:raise RuntimeError('Private stage failed')
            protected();verify_seal();requests.check()
            if name=='build':entry['build_result_sha256']=digest(directory/'build-result.json')
            else:entry['result_sha256']=digest(directory/'result.json')
            entry.update(status='passed',finished_utc=now())
            dump(directory/'terminal.json',entry)
    except BaseException as error:
        if owned is not None and not cleaned:
            try:entry['cleanup']=cleanup_owned(owned,entry['deadline_monotonic'])
            except BaseException as cleanup_error:entry['cleanup_error']=repr(cleanup_error)
        entry.update(status='failed',error=repr(error),finished_utc=now(),publication_scope='administrative failure closure; no measurement pass')
        dump(directory/'terminal.json',entry)
        raise


def set_limits():
    for kind,name in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        value=LIMITS[name];soft,hard=resource.getrlimit(kind)
        if hard!=resource.RLIM_INFINITY and hard<value:raise ValueError('Inherited resource limit is smaller than reviewed scope')
        resource.setrlimit(kind,(value,value))


def main():
    global EXPECTED_SOURCE,REVIEWED_BUILD
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true');group.add_argument('--build',action='store_true')
    group.add_argument('--run',action='store_true');group.add_argument('--build-child',action='store_true')
    group.add_argument('--run-child',type=int,choices=(0,1,2))
    parser.add_argument('--source',required=True);parser.add_argument('--reviewed-build')
    args=parser.parse_args();EXPECTED_SOURCE=args.source;REVIEWED_BUILD=args.reviewed_build
    if not re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE):parser.error('Expected exact source SHA256')
    if (args.run or args.run_child is not None) and (REVIEWED_BUILD is None or not re.fullmatch('[0-9a-f]{64}',REVIEWED_BUILD)):
        parser.error('RUN requires the separately reviewed build/terminal.json SHA256')
    if args.build_child or args.run_child is not None:set_limits()
    if args.preflight:print(json.dumps(preflight(),indent=2));return
    if args.build_child:build_child();return
    if args.run_child is not None:run_child(args.run_child);return
    preflight()
    if args.build:stage('build');return
    verify_build()
    for mode in (0,1,2):stage(f'mode-{mode}')


if __name__=='__main__':main()
