"""Bounded CLI exports/builds only. Reuses the previously reviewed child owner."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare_source import BIND, ROOT, WORK, protected, require, digest

SUPPORT={'cleanup':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py',
    '240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84'),
    'deadline':('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py',
    '726f629210a155b894f423ad76fa1de9d96b1f98270d4aceba2ba240250d6901')}
PLAN=json.loads((HERE/'commands.json').read_text())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def verify_source():
    seal=json.loads((HERE/'source-index.json').read_text())
    for name,checksum in seal['files'].items():
        require(digest(HERE/name)==checksum,'Proposal source changed: '+name)
    for name,checksum in seal['inputs'].items():
        require(digest(ROOT/name)==checksum,'Bound preparation/support input changed: '+name)
    prep=json.loads((WORK/'preparation.json').read_text())
    for name,checksum in prep['copied_files'].items():
        require(digest(WORK/'mg'/name)==checksum,'Private implementation changed: '+name)
    for name,checksum in BIND['candidate']['exporter_identity']['files'].items():
        path=ROOT/name
        require((digest(path) if path.is_file() else None)==checksum,'Production source changed: '+name)
    protected()


def support():
    for name,(relative,checksum) in SUPPORT.items():
        require(digest(ROOT/relative)==checksum,'Ownership/deadline helper changed')
    path=ROOT/SUPPORT['cleanup'][0]
    spec=importlib.util.spec_from_file_location('deployment_owned_cleanup',path)
    cleanup=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup)
    raw=(ROOT/SUPPORT['deadline'][0]).read_text()
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='StageDeadline')
    namespace={'time':time,'signal':signal,'STAGE_SECONDS':300,'CLEANUP_RESERVE':15}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/SUPPORT['deadline'][0]),'exec'),namespace)
    return cleanup,namespace


def environment():
    env=os.environ.copy()
    for name in PLAN['environment']['unset']:env.pop(name,None)
    env.update(PLAN['environment']['set'])
    # Resolve compiler search consistently with the prebound tool selection.
    for name in ['PATH','NIX_CFLAGS_COMPILE','NIX_LDFLAGS']:
        value=BIND['tools'].get('env:'+name)
        if value is None:env.pop(name,None)
        else:env[name]=value
    env.update(FC=BIND['tools']['gfortran']['path'],CC=BIND['tools']['gcc']['path'],
               CXX=BIND['tools']['g++']['path'],PWD=str(ROOT))
    return env


def child_setup():
    os.sched_setaffinity(0,set(PLAN['build_affinity']))
    resource.setrlimit(resource.RLIMIT_AS,(PLAN['address_space_bytes'],PLAN['address_space_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def run_stage(item,end,requests,cleanup,namespace,state):
    cap=min(float(item['cap_seconds']),end-time.monotonic())
    require(cap>15,'Build campaign deadline exhausted')
    namespace['STAGE_SECONDS']=cap
    path=WORK/'stages'/item['label'];path.mkdir(parents=True,exist_ok=False)
    row={**item,'actual_cap_seconds':cap,'status':'running','child_pid':None,'returncode':None}
    state['stages'].append(row)
    owned=None;cleaned=False;deadline=None
    try:
        with namespace['StageDeadline']() as deadline:
            row['deadline_monotonic']=deadline.end
            requests.check();verify_source()
            with (path/'console.log').open('xb') as log:
                deadline.handoff=True
                try:
                    child=subprocess.Popen(item['command'],cwd=ROOT,env=environment(),stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_setup)
                    row['child_pid']=child.pid
                    owned=cleanup.OwnedChild(child)
                finally:
                    deadline.handoff=False
                    if deadline.deferred:deadline.expired(None,None)
                print('START '+item['label']+' pid='+str(child.pid),flush=True)
                row['returncode']=owned.wait(requests)
                budget=min(15,max(0,deadline.end-time.monotonic()))
                row['cleanup']=owned.stop(cooperative_seconds=budget/3,
                    terminate_seconds=budget/3,kill_seconds=budget/3)
                cleaned=True
                require(not row['cleanup']['survivors'],'Unresolved owned child cleanup')
            require(row['returncode']==0,'Build command failed: '+item['label'])
            requests.check();verify_source()
            row['status']='passed'
    except BaseException as error:
        row['status']='failed';row['error']=repr(error)
        if owned is not None and not cleaned:
            budget=min(15,max(0,(deadline.end if deadline is not None else end)-time.monotonic()))
            try:
                row['cleanup']=owned.stop(cooperative_seconds=budget/3,
                    terminate_seconds=budget/3,kill_seconds=budget/3)
            except BaseException as cleanup_error:row['cleanup_error']=repr(cleanup_error)
        raise
    finally:
        if (path/'console.log').exists():row['console_sha256']=digest(path/'console.log')
        write(path/'receipt.json',row)
    print('DONE '+item['label'],flush=True)
    return (path/'console.log').read_text()


def inventory(directory):
    out={}
    for path in sorted(directory.rglob('*')):
        relative=str(path.relative_to(directory))
        if path.is_symlink():
            out[relative]={'symlink':os.readlink(path),'sha256':digest(path) if path.is_file() else None}
        elif path.is_file():out[relative]={'sha256':digest(path),'bytes':path.stat().st_size}
    return out


def bounded_action(label,action,end,namespace,state,cap=180):
    namespace['STAGE_SECONDS']=min(cap,end-time.monotonic())
    require(namespace['STAGE_SECONDS']>15,'Build deadline before '+label)
    row={'label':label,'kind':'pure file validation/move','status':'running'}
    state['stages'].append(row)
    try:
        with namespace['StageDeadline']() as deadline:
            row['deadline_monotonic']=deadline.end
            result=action()
        row['status']='passed'
        return result
    except BaseException as error:
        row.update(status='failed',error=repr(error))
        raise


def export_checks():
    sys.path.insert(0,str(WORK/'mg'))
    from madgraph.various import tirust_install,ninjars_install
    result={}
    for label in ['standalone','nlo']:
        output=WORK/(label+'-original')
        require(output.is_dir(),'Missing CLI export')
        process=output/('SubProcesses/P0_uux_z' if label=='standalone' else 'SubProcesses/P0_uux_z/V0_uux_z')
        require((process/'check_sa.f').is_file() and (process/'loop_matrix.f').is_file(),
                'CLI failed to generate numerical sources')
        require(not (output/'Source/IREGI').exists(),'Excluded optional IREGI unexpectedly deployed')
        copied={}
        for backend,module in [('tirust',tirust_install),('ninjars',ninjars_install)]:
            source=WORK/'source-prefixes'/backend/'lib'
            for name in module.export_files(str(source)):
                path=Path(name) if Path(name).is_absolute() else source/name
                target=output/'lib'/path.name
                require(target.is_file() and not target.is_symlink() and digest(target)==digest(path),
                        'Internal native artifact not copied exactly: '+str(target))
                copied[path.name]={'sha256':digest(target),'bytes':target.stat().st_size}
        require(copied['libninjars.a']['bytes']==298293528 and copied['libninjars.a']['sha256']==
            '0edfa89d7d83161a6ce3fdf0bb1fbca928b73d2b5ec44211a3b2efa186a155d6','NinjaRS changed')
        result[label]={'copied_native':copied,'generated_before_relocation':inventory(output)}
    return result


def relocate():
    result=export_checks()
    prefix=WORK/'source-prefixes'
    require(prefix.is_dir() and not prefix.is_symlink() and prefix.resolve()==prefix,'Unsafe private prefix')
    original=json.loads((WORK/'preparation.json').read_text())['source_prefix_files']
    actual={str(path.relative_to(prefix)):{'sha256':digest(path),
        'symlink':os.readlink(path) if path.is_symlink() else None}
        for path in sorted(prefix.rglob('*')) if path.is_file()}
    require(actual==original,'Private source prefix changed before removal')
    write(WORK/'before-relocation.json',result)
    shutil.rmtree(prefix)
    (WORK/'relocated').mkdir()
    for label in ['standalone','nlo']:
        (WORK/(label+'-original')).rename(WORK/'relocated'/label)
    require(not prefix.exists(),'Private source prefix was not removed')
    write(WORK/'relocation.json',{'source_prefixes_removed':True,'outputs_moved':['standalone','nlo'],
                                 'original_native_installations_removed':False})


def bind_build(end,requests,cleanup,namespace,state):
    files={};providers={};exports={}
    for label,suffix in [('standalone','SubProcesses/P0_uux_z'),('nlo','SubProcesses/P0_uux_z/V0_uux_z')]:
        export=WORK/'relocated'/label;process=export/suffix;check=process/'check'
        require(check.is_file(),'Missing executable')
        text=(process/'loop_matrix.f').read_text()
        require('FPE_IN_DP_REDUCTION=.FALSE.' in text and 'FPE_IN_QP_REDUCTION=.FALSE.' in text,
                'Missing current independent-entry latches')
        require('IF(TIRUST_CONTEXT_GENERATION().GE.0) THEN' in (process/'TIRUST_interface.f').read_text(),
                'Missing current lazy context clear')
        require('PARAMETER (NPSPOINTS = 4)' in (process/'check_sa.f').read_text(), 'Unexpected driver repetition')
        commands=[('readelf',[BIND['tools']['readelf']['path'],'-d',str(check)]),
                  ('ldd',[BIND['tools']['ldd']['path'],str(check)])]
        metadata={}
        for name,command in commands:
            item={'label':label+'-'+name,'command':command,'cwd':str(ROOT),'cap_seconds':60}
            metadata[name]=run_stage(item,end,requests,cleanup,namespace,state)
        require('$ORIGIN' in metadata['readelf'] and str(WORK/'source-prefixes') not in metadata['readelf'],
                'Missing relocatable runtime path')
        require('not found' not in metadata['ldd'],'Unresolved runtime library')
        observed={}
        for line in metadata['ldd'].splitlines():
            match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
            if match:
                path=Path(match.group(1)).resolve()
                require(path.is_file(),'Missing linked provider')
                observed[str(path)]=digest(path)
        dso=export/'lib/libmadloop_tirust.so'
        require(dso.is_file() and not dso.is_symlink() and dso.resolve()==dso,'Nonlocal native DSO')
        require(observed.get(str(dso))==BIND['candidate']['native']['files']['lib/libmadloop_tirust.so'],
                'Loader does not resolve relocated eb3 library')
        require(all(name==str(dso) or name.startswith('/nix/store/') for name in observed),
                'Unexpected non-system native dependency')
        providers.update(observed)
        def retain():
            for path in sorted(export.rglob('*')):
                keep=(path.suffix.lower() in {'.f','.f90','.inc','.mod','.o','.a','.so','.h','.c','.cc','.cpp','.mk'}
                      or path.name in {'check','makefile','Makefile','make_opts','param_card.dat','ident_card.dat','MadLoopParams.dat'}
                      or path.parent==export/'lib')
                if path.is_file() and keep:
                    resolved=path.resolve()
                    require(resolved.is_relative_to(export),'Build input escapes relocated export: '+str(path))
                    files[str(path)]=digest(path)
            return {'process':str(process),'executable_sha256':digest(check),
                'check_sa_sha256':digest(process/'check_sa.f'), 'ldd_initialized_scope':False,
                'runtime_card':str(export/('Cards/MadLoopParams.dat' if label=='standalone' else 'SubProcesses/MadLoopParams.dat')),
                'inventory':inventory(export)}
        exports[label]=bounded_action(label+'-input-binding',retain,end,namespace,state)
    return {'schema':'tirust-final-deployment-build-v1','revision':'0fcd2cd873d61d134837e1d9e87afb13a47824fd',
        'exporter_inventory':BIND['candidate']['exporter_identity']['files_sha256'],'passed':True,
        'native_evaluations':0,'scope_inventory_complete':True,'files':files,'runtime_providers':providers,
        'exports':exports,'effective_recipes':'Exact generated makefiles/make_opts plus actual stage compiler/link command logs',
        'environment':{k:environment().get(k) for k in ['PATH','FC','CC','CXX','MAKEFLAGS','NIX_CFLAGS_COMPILE',
            'NIX_LDFLAGS','FFLAGS','FCFLAGS','LDFLAGS','LIBRARY_PATH','LD_LIBRARY_PATH','LD_PRELOAD']}}


def main():
    require(Path.cwd().resolve()==ROOT,'Run from required repository root')
    require(sys.argv[1:]==['--execute-build'],'Explicit build admission required')
    require(not (WORK/'build-start.json').exists(),'Build already attempted')
    verify_source();cleanup,namespace=support()
    start=time.monotonic();end=start+PLAN['build_total_cap_seconds']
    state={'stages':[],'native_evaluations':0,'passed':False}
    write(WORK/'build-start.json',{'pid':os.getpid(),'deadline_monotonic':end,
        'source_index_sha256':digest(HERE/'source-index.json'),'commands_sha256':digest(HERE/'commands.json')})
    try:
        with cleanup.SignalRequests() as requests:
            for item in PLAN['export_commands']:run_stage(item,end,requests,cleanup,namespace,state)
            bounded_action('verify-copy-and-relocate',relocate,end,namespace,state)
            requests.check();verify_source()
            for item in PLAN['build_commands']:run_stage(item,end,requests,cleanup,namespace,state)
            receipt=bind_build(end,requests,cleanup,namespace,state)
            require(time.monotonic()<end,'Build administrative deadline exhausted')
            requests.check();verify_source()
            write(WORK/'build-admitted.json',receipt)
            state['passed']=True
    except BaseException as error:
        state['error']=repr(error)
        raise
    finally:
        state['elapsed_seconds_diagnostic_only']=time.monotonic()-start
        try:protected();state['protected_unchanged']=True
        except BaseException as error:state.update(protected_unchanged=False,protected_error=repr(error),passed=False)
        write(WORK/'build-result.json',state)
    require(state['passed'],'Build qualification incomplete')


if __name__=='__main__':main()
