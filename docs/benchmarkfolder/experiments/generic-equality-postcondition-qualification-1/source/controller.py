"""Dormant standalone integer-helper BUILD/RUN controller; separate admissions."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'.tirust-build/generic-equality-postcondition-qualification-work-1'
CPU=12
CAPS={'build':180.,'run':180.}
CLEANUP=60.
LIMITS={'address_space_bytes':4*1024**3,'file_bytes':256*1024**2,'core_bytes':0}
EXPECTED_SOURCE=None
REVIEWED_BUILD=None
PHASE=None
CLEANUP_PATH='docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py'
LOADER_PATH='docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/loader_checks.py'


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def dump(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def load(name,path,checksum):
    require(digest(path)==checksum,'Changed imported source: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def local(name):
    seal=read(HERE/'source-index.json')
    return load('exact_geometry_qualification_'+name,HERE/(name+'.py'),seal['files'][name+'.py']['sha256'])


def support(path):
    return load('exact_geometry_support_'+Path(path).stem,ROOT/path,read(HERE/'input-pins.json')['support'][path])


def verify_files(base,files):
    for relative,want in files.items():
        p=Path(relative)
        require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
        item=base/p
        require(not item.is_symlink() and item.is_file() and item.stat().st_size==want['bytes'] and
                digest(item)==want['sha256'],'Changed source: '+relative)


def environment():
    ancestry=read(HERE/'ancestry.json');pin=read(HERE/'toolchain-pins.json')
    return local('tool_environment').reconstruct(pin,ancestry['compiler_environment_names'],{},[],os.environ)


def protected():
    ancestry=read(HERE/'ancestry.json');pin=read(HERE/'input-pins.json')
    require(os.uname().sysname=='Linux' and os.uname().machine=='x86_64','This qualification targets GNU/Linux x86_64')
    for name,checksum in {**ancestry['source_files'],**pin['support'],**pin['closed_standalone_build'],**pin['files']}.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        require(path.is_file() and digest(path)==checksum,'Changed protected source/tool/provider: '+name)
    for name,want in pin['additional_compiler_files'].items():
        item=Path(name)
        require(item.is_file() and item.stat().st_size==want['bytes'] and digest(item)==want['sha256'] and
                (not want['executable'] or os.access(item,os.X_OK)),'Changed C/backend compiler input: '+name)
    for name in pin['additional_compiler_absences']:
        require(not Path(name).exists() and not Path(name).is_symlink(),'Unexpected C compiler hook appeared')
    parent=ROOT/ancestry['helper_parent'];index=parent/'source-index.json'
    require(digest(index)==ancestry['helper_index_sha256'],'Frozen helper parent changed')
    values=read(index);verify_files(parent,values['files'])
    require({str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}==
            set(values['files'])|{'source-index.json'},'Frozen helper inventory changed')
    require(digest(HERE/'exact_geometry.f90')==ancestry['helper_Fortran_sha256'],
            'Standalone helper differs from frozen exact source')
    toolpin=read(HERE/'toolchain-pins.json')
    for name,checksum in toolpin['donor_lineage'].items():
        require(digest(ROOT/name)==checksum,'Successful toolchain donor changed')
    admitted=local('tool_environment').validate(toolpin,environment())
    return {'helper_parent_index_sha256':ancestry['helper_index_sha256'],
            'helper_sha256':ancestry['helper_Fortran_sha256'],'pinned_runtime_providers':len(pin['files']),
            'tool_admission':admitted,'compiler_queries':0,'loader_queries':0,'native_executions':0}


def verify_source():
    require(Path.cwd()==ROOT,'Run only from the MadGraph working repository')
    require(EXPECTED_SOURCE and digest(HERE/'source-index.json')==EXPECTED_SOURCE,'Exact reviewed source SHA required')
    seal=read(HERE/'source-index.json');verify_files(HERE,seal['files'])
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==
            set(seal['files'])|{'source-index.json'},'Source inventory changed')
    return protected()


def preflight():
    before=verify_source();require(CPU in os.sched_getaffinity(0),'Nonexclusive CPU12 unavailable')
    corpus=local('corpus');cases=corpus.build()
    require(corpus.encode(cases)==(HERE/'fixtures.hex').read_text(),'Frozen hex fixtures changed')
    require(corpus.expectations(cases)==read(HERE/'expectations.json'),'Frozen expected Fraction answers changed')
    if PHASE=='run':verify_build()
    return {'status':'source-only-preflight-passed','source_index_sha256':EXPECTED_SOURCE,
            'protected':before,'caps_seconds':CAPS,'cleanup_reserve_seconds':CLEANUP,'resource_limits':LIMITS,
            'CPU':CPU,'CPU_reserved':False,'compiler_or_loader_queries':0,'native_executions':0}


def command(argv,cwd,logs,label):
    logs.mkdir(exist_ok=True,parents=True)
    dump(logs/(label+'.command.json'),{'argv':list(map(str,argv)),'cwd':str(cwd),'started_utc':now(),
         'controller_pid':os.getpid(),'session':os.getsid(0),'process_group':os.getpgrp()})
    with (logs/(label+'.stdout.txt')).open('xb') as output:
        done=subprocess.run(argv,cwd=cwd,env=environment(),stdout=output,stderr=subprocess.STDOUT)
    dump(logs/(label+'.terminal.json'),{'returncode':done.returncode,'finished_utc':now(),
         'stdout_sha256':digest(logs/(label+'.stdout.txt'))})
    require(done.returncode==0,'Admitted command failed: '+label)
    return (logs/(label+'.stdout.txt')).read_text()


def build_locked(directory):
    pin=read(HERE/'toolchain-pins.json');tools={k:v['path'] for k,v in pin['required_tools'].items()}
    loader=support(LOADER_PATH);runtime=local('unit_runtime');logs=directory/'commands'
    for name in ('gfortran','cc'):
        text=command([tools[name],'--version'],directory,logs,name+'-version')
        require('15.2.0' in text,'Compiler version is not GNU15.2.0')
    command([tools['cc'],'-std=c11','-O0','-fPIC','-frounding-math','-c',str(HERE/'fenv_observer.c'),
             '-o','fenv_observer.o'],directory,logs,'fenv-observer-compile')
    variants={}
    for variant,flags in (('checked',['-O0','-fcheck=all','-ftrapv']),('optimized',['-O3','-fno-fast-math'])):
        case=directory/variant;case.mkdir()
        command([tools['gfortran'],*flags,'-fPIC','-c',str(HERE/'exact_geometry.f90'),'-o','exact_geometry.o'],
                case,logs,variant+'-helper-compile')
        command([tools['gfortran'],'-O0','-fPIC','-fcheck=all','-frounding-math','-c',
                 str(HERE/'driver.f90'),'-o','driver.o'],case,logs,variant+'-driver-compile')
        link=[tools['gfortran'],'-fPIC','-Wl,-Map,link.map,--cref','-o','check',
              'driver.o','exact_geometry.o',str(directory/'fenv_observer.o'),'-lm']
        callback=lambda argv,label:command(argv,case,logs,variant+'-'+label)
        bound=loader.prelink(link,case,environment(),callback)
        # Explicit source-protected standalone runtime join BEFORE link/publication.
        # Do not borrow the full-amplitude15-provider closure for this small unit.
        extra=read(HERE/'input-pins.json')['runtime_before_link_augmentation']
        for name,checksum in extra.items():
            require(bound['bound'].get(name,checksum)==checksum and digest(name)==checksum,
                    'Standalone pre-link runtime join conflict')
            bound['bound'][name]=checksum
        bound['explicit_standalone_runtime_join_before_link']=extra
        dump(case/'prelink.json',bound)
        command(link,case,logs,variant+'-link')
        loads=loader.requireactualmap((case/'link.map').read_text(),case,bound,case/'check')
        closure=runtime.derive(case/'check',bound)
        dump(case/'static-runtime.json',closure)
        observed=loader.query_runtime(case/'check',closure['files'],bound,callback)
        symbols=command([tools['nm'],'--defined-only',str(case/'check')],case,logs,variant+'-symbols')
        names=re.findall(r'^[0-9a-fA-F]+\s+[A-Za-z]\s+(\S+)$',symbols,re.M)
        require(names.count('__ml_private_exact_geometry_MOD_exact_geometry_difference')==1,'Helper ownership is not unique')
        undefined=command([tools['nm'],'--undefined-only',str(case/'exact_geometry.o')],case,logs,variant+'-helper-undefined')
        names=re.findall(r'\bU\s+(\S+)',undefined)
        require(not any(re.search(r'float|quadmath|multc|trunc|extend|feset|ieee',n,re.I) for n in names),
                'Unexpected FP/environment helper dependency')
        variants[variant]={'path':str(case/'check'),'sha256':digest(case/'check'),'flags':flags,
            'objects':{n:digest(case/n) for n in ('exact_geometry.o','driver.o')},
            'prelink_sha256':digest(case/'prelink.json'),'link_map_sha256':digest(case/'link.map'),
            'static_runtime_sha256':digest(case/'static-runtime.json'),'runtime':observed,
            'actual_link_loads':loads,'helper_undefined_symbols':names}
    dump(directory/'result.json',{'status':'passed','source_index_sha256':EXPECTED_SOURCE,
        'helper_sha256':digest(HERE/'exact_geometry.f90'),'fenv_object_sha256':digest(directory/'fenv_observer.o'),
        'variants':variants,'helper_main_calls':0,'scientific_reducer_calls':0,'protected':protected()})


def verify_build():
    directory=WORK/'build';terminal=directory/'terminal.json'
    require(REVIEWED_BUILD and digest(terminal)==REVIEWED_BUILD,'Separate exact reviewed BUILD terminal required')
    closed=read(terminal)
    require(closed['status']=='passed' and closed['returncode']==0 and not closed['cleanup']['survivors'],
            'BUILD was not successfully closed')
    require(digest(directory/'result.json')==closed['result_sha256'],'BUILD result changed')
    result=read(directory/'result.json')
    require(result['source_index_sha256']==EXPECTED_SOURCE and result['helper_sha256']==digest(HERE/'exact_geometry.f90'),
            'Reviewed BUILD/source mismatch')
    require(digest(directory/'fenv_observer.o')==result['fenv_object_sha256'],'Observer object changed')
    for variant,row in result['variants'].items():
        case=directory/variant
        require(digest(row['path'])==row['sha256'],'Executable changed')
        for name,checksum in row['objects'].items():require(digest(case/name)==checksum,'Helper/driver object changed')
        for name,key in [('prelink.json','prelink_sha256'),('link.map','link_map_sha256'),('static-runtime.json','static_runtime_sha256')]:
            require(digest(case/name)==row[key],'Link/runtime proof changed')
        require(local('unit_runtime').derive(row['path'],read(case/'prelink.json'))==read(case/'static-runtime.json'),
                'Standalone static runtime derivation changed')
        for name,checksum in {**row['actual_link_loads'],**row['runtime']['files']}.items():
            require(digest(name)==checksum,'Admitted runtime/link provider changed')
    return result


def mapped(process,row):
    raw=Path('/proc/'+str(process.pid)+'/maps').read_text();actual={};data={}
    allowed=read(HERE/'input-pins.json').get('allowed_data_mappings',{})
    for line in raw.splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        name=fields[5];require(not name.endswith(' (deleted)'),'Deleted native mapping')
        path=Path(name)
        with path.open('rb') as stream:header=stream.read(4)
        resolved=str(path.resolve())
        if header==b'\x7fELF':actual[resolved]=digest(path)
        else:
            require(allowed.get(resolved)==digest(path),'Unexpected mapped data file')
            data[resolved]=allowed[resolved]
    expected=dict(row['runtime']['files']);expected[str(Path(row['path']).resolve())]=row['sha256']
    require(actual==expected,'Actual worker ELF mappings differ from admitted providers')
    require(sorted(os.sched_getaffinity(process.pid))==[CPU],'Worker affinity differs')
    return {'pid':process.pid,'identity':support(CLEANUP_PATH).identity(process.pid),
            'raw':raw,'ELF':actual,'data':data,'affinity':[CPU],'exclusive_CPU':False}


def run_locked(directory):
    built=verify_build();analyzer=local('analyze');results={}
    for variant,row in built['variants'].items():
        folder=directory/variant;folder.mkdir();output=folder/'observations.txt'
        argv=[row['path'],str(HERE/'fixtures.hex'),str(output)]
        dump(folder/'command.json',{'argv':argv,'started_utc':now(),'cwd':str(folder),
            'session':os.getsid(0),'process_group':os.getpgrp(),'new_session':False})
        process=None
        try:
            with (folder/'stderr.txt').open('xb') as error:
                process=subprocess.Popen(argv,cwd=folder,env=environment(),stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1)
                dump(folder/'started.json',{'pid':process.pid,'identity':support(CLEANUP_PATH).identity(process.pid),
                     'session':os.getsid(process.pid),'process_group':os.getpgid(process.pid),'new_session':False})
                ready=process.stdout.readline();(folder/'ready.txt').write_text(ready)
                require(ready.strip()=='READY '+str(process.pid),'Standalone READY protocol failed')
                dump(folder/'maps-before.json',mapped(process,row))
                process.stdin.write('GO\n');process.stdin.flush()
                done=process.stdout.readline();(folder/'done.txt').write_text(done)
                require(done.strip()=='DONE '+str(process.pid),'Standalone DONE protocol failed')
                dump(folder/'maps-after.json',mapped(process,row))
                process.stdin.write('EXIT\n');process.stdin.flush()
                tail=process.communicate()[0]
        except BaseException as error:
            # The owning stage supervises this whole session and every descendant.
            # Preserve partial evidence before handing cleanup back to that owner.
            dump(folder/'worker-failure.json',{'error':repr(error),'finished_utc':now(),
                'pid':process.pid if process is not None else None,
                'returncode_at_failure':process.poll() if process is not None else None,
                'observations_sha256_at_failure':digest(output) if output.exists() else None,
                'stderr_sha256_at_failure':digest(folder/'stderr.txt') if (folder/'stderr.txt').exists() else None,
                'cleanup_owner':'Closed stage supervisor; no success on this path'})
            raise
        dump(folder/'terminal.json',{'returncode':process.returncode,'finished_utc':now(),
            'stdout_tail':tail,'observations_sha256':digest(output) if output.exists() else None,
            'stderr_sha256':digest(folder/'stderr.txt')})
        require(process.returncode==0 and not tail.strip(),'Standalone worker failed or emitted unexpected trailing output')
        result=analyzer.analyze(output.read_text(),read(HERE/'expectations.json'))
        dump(folder/'analysis.json',result);results[variant]=result
    require(results['checked']['case_results']==results['optimized']['case_results'],
            'Checked and optimized helper decisions differ')
    verify_build();verify_source()
    dump(directory/'result.json',{'status':'passed','source_index_sha256':EXPECTED_SOURCE,
        'reviewed_build_sha256':REVIEWED_BUILD,'variants':results,'scientific_reducer_calls':0,
        'physical_or_performance_qualification':False,'snapshot_ownership_qualified':False,
        'scope':'Standalone exact integer helper only; explicit driver-owned fenv perturbations.'})


def set_limits():
    for kind,name in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes')):
        value=LIMITS[name];_,hard=resource.getrlimit(kind)
        require(hard==resource.RLIM_INFINITY or hard>=value,'Inherited resource limit too small')
        resource.setrlimit(kind,(value,value))


def child():
    set_limits();directory=WORK/PHASE;admission=read(directory/'admission.json')
    require(admission['source_index_sha256']==EXPECTED_SOURCE and admission['phase']==PHASE and
            admission['reviewed_build_sha256']==REVIEWED_BUILD,'Child admission changed')
    actual={name:list(resource.getrlimit(kind)) for kind,name in
            ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CORE,'core_bytes'))}
    require(actual=={k:[v,v] for k,v in LIMITS.items()},'Child resource limits differ')
    dump(directory/'limits.json',{'actual':actual,'pid':os.getpid(),'session':os.getsid(0),'process_group':os.getpgrp()})
    verify_source()
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        dump(directory/'lock.json',{'acquired_utc':now(),'path':str(Path(lock.name))})
        if PHASE=='build':build_locked(directory)
        else:run_locked(directory)


def stage():
    cleanup=support(CLEANUP_PATH);ownership=local('ownership')
    WORK.mkdir(exist_ok=True);directory=WORK/PHASE;directory.mkdir(exist_ok=False)
    entry={'phase':PHASE,'cap_seconds':CAPS[PHASE],'cleanup_reserve_seconds':CLEANUP,'started_utc':now(),'status':'running'}
    owned=None;process=None;raw_record=None;cleaned=False
    with cleanup.SignalRequests() as requests,ownership.StageDeadline(CAPS[PHASE]) as deadline:
        entry['deadline_monotonic']=deadline.end
        try:
            requests.check();preflight()
            tools=read(HERE/'toolchain-pins.json')['required_tools']
            argv=[tools['taskset']['path'],'-c',str(CPU),tools['python3']['path'],'-B',str(Path(__file__).relative_to(ROOT)),
                  '--child','--phase',PHASE,'--source',EXPECTED_SOURCE]
            if REVIEWED_BUILD:argv+=['--reviewed-build',REVIEWED_BUILD]
            entry['command']=argv
            dump(directory/'admission.json',{'phase':PHASE,'source_index_sha256':EXPECTED_SOURCE,
                 'reviewed_build_sha256':REVIEWED_BUILD,'command':argv,'resource_limits':LIMITS})
            with (directory/'console.log').open('xb') as output:
                deadline.handoff=True
                try:
                    process=subprocess.Popen(argv,cwd=ROOT,env=environment(),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    raw_record=cleanup.identity(process.pid);owned=cleanup.OwnedChild(process)
                finally:
                    deadline.handoff=False
                    if deadline.deferred:deadline.expired(None,None)
                entry['child_pid']=process.pid;dump(directory/'started.json',entry)
                print('START '+PHASE+' child_pid='+str(process.pid),flush=True)
                entry['returncode']=owned.wait(requests);requests.check()
                entry['cleanup']=ownership.cleanup_owned(owned,deadline.end);cleaned=True
            require(entry['returncode']==0,'Standalone stage failed')
            verify_source();requests.check()
            entry.update(status='passed',finished_utc=now(),result_sha256=digest(directory/'result.json'))
            dump(directory/'terminal.json',entry)
        except BaseException as error:
            if process is not None and not cleaned:
                try:
                    entry['cleanup']=ownership.cleanup_owned(owned,deadline.end) if owned is not None else \
                        ownership.raw_cleanup(process,raw_record,cleanup,deadline.end)
                except BaseException as cleanup_error:entry['cleanup_error']=repr(cleanup_error)
            entry.update(status='failed',error=repr(error),finished_utc=now(),
                         scope='Administrative failure closure; never a qualification pass')
            dump(directory/'terminal.json',entry);raise


def main():
    global EXPECTED_SOURCE,REVIEWED_BUILD,PHASE
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true');mode.add_argument('--child',action='store_true')
    parser.add_argument('--phase',choices=tuple(CAPS),required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--reviewed-build');args=parser.parse_args()
    EXPECTED_SOURCE=args.source;REVIEWED_BUILD=args.reviewed_build;PHASE=args.phase
    require(re.fullmatch('[0-9a-f]{64}',EXPECTED_SOURCE),'Exact source SHA required')
    if PHASE=='run':require(REVIEWED_BUILD and re.fullmatch('[0-9a-f]{64}',REVIEWED_BUILD),'Separate reviewed BUILD SHA required')
    if args.preflight:print(json.dumps(preflight(),sort_keys=True,indent=2));return
    if args.child:child();return
    stage()


if __name__=='__main__':main()
