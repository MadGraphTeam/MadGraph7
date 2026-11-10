"""Frozen source/compile/replay stages. Each execution stage is one-shot.

Default invocation only verifies source bindings. Build and run require separate
root admission; build never runs a loader, executable, or real reduction.
"""
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
import struct
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=PACKAGE.parents[3]
WORK=ROOT/'.tirust-build/adaptive-qp-release-fortran-qualification'
ORIGINAL=ROOT/'docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification'
ORIGINAL_BUILD=ROOT/'.tirust-build/adaptive-qp-fortran-qualification/build1'
ORIGINAL_SOURCE_SHA='80f83d5384ada33b609ac94bb9f9be935eb92e56ee706f979494e4d053fcc9b5'
ORIGINAL_BUILD_INDEX_SHA='7d92500bc6fb518f8904f831f305519b07b6b4c011d3e15785146fffee5923d9'
ORIGINAL_BUILD_SHA='a213c73aadab7d898e1a3699ab7c1c4ae09dae8afa1c7916a8e6671ebfb14b15'
BUILD=WORK/'build1'
RUN=WORK/'run1'
INDEX=PACKAGE/'source-index.json'
SUPERVISOR=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
SUPERVISOR_SHA='ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
FC=Path('/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran')
CC=Path('/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/gcc')
LIBC=Path('/nix/store/57iz36553175g3178pvxjij8z5rcsd4n-glibc-2.42-61/lib')
LIBGCC=Path('/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib')
LOADER=LIBC/'ld-linux-x86-64.so.2'
BRIDGES={'legacy':ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so',
         'adaptive':ROOT/'.tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so'}
BRIDGE_SHAS={'legacy':'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
             'adaptive':'c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def verify():
    assert Path.cwd()==ROOT
    index=json.loads(INDEX.read_text())
    for name,digest in index['sources'].items():assert sha(PACKAGE/name)==digest,name
    for name,digest in index['dependencies'].items():assert sha(ROOT/name if not name.startswith('/') else name)==digest,name
    origins=json.loads((HERE/'origins.json').read_text())
    for name,origin in origins.items():assert sha(HERE/name)==sha(ROOT/origin['path'])==origin['sha256'],name
    manifest=json.loads((HERE/'fixtures/manifest.json').read_text())
    for name,field in manifest['fields'].items():
        assert sha(HERE/'fixtures'/('group10-'+name+'.dat'))==field['sha256']
        words=field['raw_words']
        exact=str(len(words)//2)+'\n'+''.join(' '.join(words[k:k+2])+'\n' for k in range(0,len(words),2))
        assert (HERE/'fixtures'/('group10-'+name+'.words')).read_text()==exact
    for arm,path in BRIDGES.items():assert sha(path)==BRIDGE_SHAS[arm]
    assert sha(SUPERVISOR)==SUPERVISOR_SHA
    assert sha(ORIGINAL/'source-index.json')==ORIGINAL_SOURCE_SHA
    original_index=json.loads((ORIGINAL/'source-index.json').read_text())
    for name,digest in original_index['sources'].items():assert sha(ORIGINAL/name)==digest,name
    assert sha(ORIGINAL/'build1/evidence-index.json')==ORIGINAL_BUILD_INDEX_SHA
    frozen=json.loads((ORIGINAL/'build1/evidence-index.json').read_text())['files']
    for name,digest in frozen.items():
        assert sha(ORIGINAL/'build1'/name)==digest,name
    assert sha(ORIGINAL_BUILD/'build.json')==ORIGINAL_BUILD_SHA
    assert sha(ORIGINAL/'build1/build.json')==ORIGINAL_BUILD_SHA
    old=json.loads((ORIGINAL_BUILD/'build.json').read_text())
    assert old['complete'] and len(old['commands'])==7 and all(row['exit_code']==0 for row in old['commands'])
    for name,digest in old['artifacts'].items():assert sha(ORIGINAL_BUILD/name)==digest,name
    for inputs in old['link_inputs'].values():
        for name,digest in inputs.items():assert sha(name)==digest,name
    for path in (ORIGINAL/'source').rglob('*'):
        if path.is_file() and path.name!='controller.py':
            assert sha(path)==sha(HERE/path.relative_to(ORIGINAL/'source')),str(path)
    return index


def supervisor():
    source=SUPERVISOR.read_text()
    assert source.count("affinity!=['11']")==source.count('differs from CPU11.')==1
    source=source.replace("affinity!=['11']","affinity!=['12']").replace('differs from CPU11.','differs from CPU12.')
    namespace={'__name__':'adaptive_fortran_owned_supervisor','__file__':str(SUPERVISOR)}
    exec(compile(source,str(SUPERVISOR),'exec'),namespace)
    return namespace,hashlib.sha256(source.encode()).hexdigest()


def environment():
    # Do not inherit loader/compiler overrides or serialize unrelated secrets.
    env={key:os.environ[key] for key in ('PATH','HOME','TMPDIR') if key in os.environ}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',LC_ALL='C')
    return env


def handlers():
    def interrupt(signum,frame):raise InterruptedError('Adaptive Fortran stage interrupted by signal %d'%signum)
    return {signum:signal.signal(signum,interrupt) for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}


def link_command():
    old=json.loads((ORIGINAL_BUILD/'build.json').read_text())
    argv=list(old['commands'][-1]['argv'])
    assert argv[0]==str(FC) and argv[1:3]==['-o',str(ORIGINAL_BUILD/'adaptive')]
    old_bridge=ROOT/'.tirust-build/adaptive-qp-rescue-cdylib-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so'
    assert argv.count(str(old_bridge))==1
    replacements={str(ORIGINAL_BUILD/'adaptive'):str(BUILD/'adaptive'),
                  str(old_bridge):str(BRIDGES['adaptive']),
                  '-Wl,-rpath,'+str(old_bridge.parent):'-Wl,-rpath,'+str(BRIDGES['adaptive'].parent),
                  '-Wl,-Map,'+str(ORIGINAL_BUILD/'adaptive.map'):'-Wl,-Map,'+str(BUILD/'adaptive.map')}
    replacements.update({str(ORIGINAL_BUILD/(name+'.o')):str(BUILD/(name+'.o'))
                         for name in ('mtirust','qp_tier_records','driver','optional_getter','observer')})
    assert all(argv.count(key)==1 for key in replacements)
    return [replacements.get(arg,arg) for arg in argv]


def build():
    index=verify();helper,adapted=supervisor()
    BUILD.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{12})
    resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
    old_build=json.loads((ORIGINAL_BUILD/'build.json').read_text())
    copied={}
    # Reuse the five identical numerical/observer objects and legacy control;
    # there is no Fortran/C recompilation and no new fixture or arithmetic.
    for name,digest in old_build['artifacts'].items():
        if name.endswith(('.o','.mod')) or name in ('legacy','legacy.map'):
            shutil.copyfile(ORIGINAL_BUILD/name,BUILD/name)
            assert sha(BUILD/name)==digest
            copied[name]=digest
    common=[str(BUILD/(name+'.o')) for name in ('mtirust','qp_tier_records','driver','optional_getter','observer')]
    assert set(Path(name).name for name in common).issubset(copied)
    command=link_command()
    protected={str(PACKAGE/name):digest for name,digest in index['sources'].items()}
    protected.update({str(ROOT/name if not name.startswith('/') else Path(name)):digest for name,digest in index['dependencies'].items()})
    protected[str(INDEX)]=sha(INDEX)
    protected.update({str(BUILD/name):digest for name,digest in copied.items()})
    record={'complete':False,'scope':'one release relink only; no source compilation/loader/native execution',
            'source_index_sha256':sha(INDEX),'original_source_index_sha256':ORIGINAL_SOURCE_SHA,
            'original_build_sha256':ORIGINAL_BUILD_SHA,'cpu':12,'cpu_reservation':False,'jobs':1,
            'shared_cap_seconds':120,'rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS)),
            'supervisor_sha256':SUPERVISOR_SHA,'adapted_supervisor_sha256':adapted,'commands':[],
            'environment':environment(),'reused_artifacts_sha256':copied,'protected_inputs_before':protected}
    old=handlers();start=time.monotonic()
    try:
        row={'argv':command,'log':'link-adaptive.log','progress':{}};record['commands'].append(row);save(BUILD/'build.json',record)
        with (BUILD/row['log']).open('xb') as log:
            row['exit_code']=helper['run_plain_capped'](command,log,cwd=ROOT,env=record['environment'],progress=row['progress'],seconds=120)
        row['log_sha256']=sha(BUILD/row['log']);save(BUILD/'build.json',record)
        if row['exit_code']:raise RuntimeError('Release relink failed; no retry')
        paths={str(Path(name).resolve()) for name in re.findall(r'^LOAD\s+(.+)$',(BUILD/'adaptive.map').read_text(),re.M)}
        assert str(BRIDGES['adaptive'].resolve()) in paths and set(common).issubset(paths)
        record['link_inputs']={'legacy':old_build['link_inputs']['legacy'],
                              'adaptive':{name:sha(name) for name in sorted(paths)}}
        verify();assert all(sha(BUILD/name)==digest for name,digest in copied.items());record['complete']=True
    except BaseException as error:
        record['failure']=repr(error);raise
    finally:
        for signum,handler in old.items():signal.signal(signum,handler)
        record['elapsed_seconds']=time.monotonic()-start
        record['protected_inputs_after']={name:sha(name) for name in protected}
        if record['protected_inputs_after']!=protected:record['complete']=False
        record['artifacts']={p.name:sha(p) for p in sorted(BUILD.iterdir()) if p.is_file() and p.name!='build.json'}
        save(BUILD/'build.json',record)
    assert record['complete']


def built():
    verify();record=json.loads((BUILD/'build.json').read_text())
    assert record['complete'] and record['source_index_sha256']==sha(INDEX)
    assert len(record['commands'])==1 and record['commands'][0]['exit_code']==0
    assert record['commands'][0]['argv']==link_command()
    assert record['original_build_sha256']==ORIGINAL_BUILD_SHA
    assert record['protected_inputs_before']==record['protected_inputs_after']
    for name,digest in record['protected_inputs_after'].items():assert sha(name)==digest,name
    for name,digest in record['reused_artifacts_sha256'].items():
        assert sha(BUILD/name)==sha(ORIGINAL_BUILD/name)==digest,name
    for name,digest in record['artifacts'].items():assert sha(BUILD/name)==digest,name
    for inputs in record['link_inputs'].values():
        for name,digest in inputs.items():assert sha(name)==digest,name
    # Inspect ELF PT_INTERP without invoking a loader.
    for arm in BRIDGES:
        with (BUILD/arm).open('rb') as stream:
            header=stream.read(64);assert header[:6]==b'\x7fELF\x02\x01'
            offset=struct.unpack_from('<Q',header,32)[0];size,count=struct.unpack_from('<HH',header,54);found=[]
            for number in range(count):
                stream.seek(offset+number*size);entry=stream.read(size)
                if struct.unpack_from('<I',entry)[0]==3:
                    start=struct.unpack_from('<Q',entry,8)[0];length=struct.unpack_from('<Q',entry,32)[0]
                    stream.seek(start);found.append(stream.read(length).rstrip(b'\0').decode())
            assert found==[str(LOADER)]
    return record


def runtime_closure(text,arm):
    assert 'not found' not in text
    paths={str(Path(name).resolve()) for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)',text,re.M)}
    assert str(LOADER.resolve()) in paths and str(BRIDGES[arm].resolve()) in paths
    assert str(BRIDGES['legacy' if arm=='adaptive' else 'adaptive'].resolve()) not in paths
    assert str((LIBC/'libc.so.6').resolve()) in paths
    assert all(name.startswith('/nix/store/') or name==str(BRIDGES[arm].resolve()) for name in paths)
    return {name:sha(name) for name in sorted(paths)}


def run():
    built();helper,adapted=supervisor();os.sched_setaffinity(0,{12})
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(75)
        RUN.mkdir(exist_ok=False)
        record={'complete':False,'scope':'tensor transport only, no physical amplitude or performance claim',
                'source_index_sha256':sha(INDEX),'build_sha256':sha(BUILD/'build.json'),'cpu':12,'worker_cap_seconds':180,
                'loader_cap_seconds':30,'supervisor_sha256':SUPERVISOR_SHA,'adapted_supervisor_sha256':adapted,'workers':{}}
        env=environment();env['LD_LIBRARY_PATH']=str(LIBC)+':'+str(LIBGCC)
        resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296));record['rlimit_as']=list(resource.getrlimit(resource.RLIMIT_AS))
        start=time.monotonic();old=handlers()
        def checkpoint():save(RUN/'execution.json',record)
        try:
            for arm in BRIDGES:
                directory=RUN/arm;directory.mkdir()
                for path in (HERE/'fixtures').glob('*.words'):shutil.copyfile(path,directory/path.name)
                item={'argv':[str(BUILD/arm),arm],'environment':env,'inputs':{p.name:sha(p) for p in directory.iterdir()},'loader':{'progress':{}}}
                record['workers'][arm]=item;checkpoint()
                argv=[str(LOADER),'--list',str(BUILD/arm)];item['loader']['argv']=argv
                with (RUN/(arm+'-loader.log')).open('xb') as log:
                    item['loader']['exit_code']=helper['run_plain_capped'](argv,log,cwd=ROOT,env=env,progress=item['loader']['progress'],seconds=30)
                assert item['loader']['exit_code']==0
                item['runtime_sha256']=runtime_closure((RUN/(arm+'-loader.log')).read_text(),arm);checkpoint()
            for arm,item in record['workers'].items():
                directory=RUN/arm;item['progress']={};checkpoint()
                original_capture=helper['capture_maps']
                def captured(*args,**kwargs):
                    result=original_capture(*args,**kwargs)
                    if result is not None:(directory/'maps-ready').write_text('actual ELF mapping captured\n')
                    return result
                helper['capture_maps']=captured
                try:
                    with (RUN/(arm+'-console.log')).open('xb') as log:
                        item['exit_code'],item['mapping']=helper['run_captured'](item['argv'],log,cwd=directory,env=env,
                            expected_runtime=item['runtime_sha256'],executable=str(BUILD/arm),selected_library=str(BRIDGES[arm]),
                            maps_path=RUN/(arm+'-maps.txt'),progress=item['progress'],seconds=180)
                finally:helper['capture_maps']=original_capture
                checkpoint();assert item['exit_code']==0
                for name,digest in item['inputs'].items():assert sha(directory/name)==digest
                assert {p.name for p in directory.iterdir()}==set(item['inputs'])|{'records.txt','telemetry.txt','observer.jsonl','snapshots.jsonl','maps-ready'}
            result=runpy.run_path(str(HERE/'analyze.py'))['analyze'](RUN)
            save(RUN/'results.json',result);record['results_sha256']=sha(RUN/'results.json')
            built();record['complete']=True
        except BaseException as error:record['failure']=repr(error);raise
        finally:
            for signum,handler in old.items():signal.signal(signum,handler)
            record['elapsed_seconds']=time.monotonic()-start
            record['retained_files']={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='execution.json'}
            checkpoint()


if __name__=='__main__':
    assert len(sys.argv)<=2
    stage=sys.argv[1] if len(sys.argv)==2 else 'verify'
    if stage=='verify':verify();print('Source/dependency verification passed; no compiler/loader/native execution.')
    elif stage=='build':build()
    elif stage=='run':run()
    else:raise SystemExit('Use verify, build, or run')
