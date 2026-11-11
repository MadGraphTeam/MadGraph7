"""Portable text-only replay. Never execute historical controllers/review scripts."""
import argparse
import builtins
from contextlib import contextmanager
import copy
import ctypes
import dataclasses
from datetime import datetime
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import posixpath
import random
import re
import subprocess
import sys
import types

sys.dont_write_bytecode=True
def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def normalized(value):return json.loads(json.dumps(value,sort_keys=True))


@contextmanager
def blocked_apis(memory):
    """Numerical replay gets only exact preloaded Path reads; no live filesystem."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden API during portable replay')
    def memory_bytes(path,*args,**kwargs):
        key=str(path)
        if key not in memory:return deny(key)
        return memory[key]
    def memory_text(path,encoding=None,errors=None,*args,**kwargs):
        return memory_bytes(path).decode(encoding or 'utf-8',errors or 'strict')
    def patch(obj,name,value=deny):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        patch(Path,'read_bytes',memory_bytes);patch(Path,'read_text',memory_text)
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob',
                     'write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        patch(builtins,'open');patch(io,'open');patch(io,'open_code')
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp',
                     'execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty',
                     'mkdir','makedirs','unlink','remove','rename','replace','rmdir','removedirs','chmod','chown',
                     'truncate','utime','link','symlink'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        if 'ctypes' in sys.modules:
            module=sys.modules['ctypes']
            for name in ('CDLL','PyDLL','_dlopen'):patch(module,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def load_source(name,data,path):
    module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module


SOURCE='41797bbf9e25eb1836932307f66ad74f413714dd38cf1485e07cc8f2d74160d9'
PARENT='faae6af3bb71d490fde1ab7bb46a77c43e99991ec314d4a6a9eeb921ff9a1015'
MODULE='307b8f7615b3c83c8ca15713b180d8cd70013f441ca0598f466efd6386966706'
BUILD='cb09bd997719180a66ad18d157d7d0830876ff12611621abeae97537fd342ec1'
BUILD_RESULT='0a6737daa72d18f9031aa53e7a4c7d8df8f7b967dddf86372f432d34e6c344a4'
RUN='3b9b46fd23b59f29e0ef571edf8024f078aeaff1892c8a8f89bca630f636100d'
RUN_RESULT='0e696761b43e899c78e8457dee150066b948738d43a18633a1bfa38227cb4e0d'
RAW='6b75d55910c8f00b0ae10d9a52544913ea2dfa7cf20a477326eda69b32e2c31b'
REVIEWS={
 'independent-build':('221cff5b413732eef32cd344e103bb7ccf105a75ab44402178ed29d5c5745a4b','PASS'),
 'independent-run':('a204cfd9bf9b2f992d9e1ff6ea760e1ba5d3aac99b35a1b096d28e45dcfe406b','PASS'),
 'parent-build':('673339d0415a2a551ef75cfb3764d0a24f85ce58b2a145d7ecca2b4361b81ea0','clear_for_root_separate_RUN_decision'),
 'parent-build-runtime':('6ccc37cc45855136a99b8b2d374c5c15f74d3deba9c8203b7526bc78b92fa99e','pass-closed-BUILD-only'),
 'parent-run':('7b26424c80d982c5de95de7d528216f594cb3c50687f8520f495345a7ccf6bc9','clear_within_declared_standalone_scope'),
 'parent-run-state':('209dd8c5ced44bf4f60ed29df6701742f4c4434db69d9d5ecf72aba3a025973e','pass-exact-standalone-fixture-lane')}


def verify(root,expected_index_sha256=None):
    root=Path(root).resolve();index_data=(root/'index.json').read_bytes()
    if expected_index_sha256 is not None:
        require(sha(index_data)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_data)
    require(index['schema']=='owned-snapshot-portable-qualification-v1','Wrong capsule schema')
    require(index['source_index_sha256']==SOURCE and index['parent_receipt_sha256']==PARENT and
            index['native_binaries_packaged']==index['new_native_calls']==0 and index['public_writes'] is False,
            'Contradictory source/native/publication metadata')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Capsule symlinks are forbidden')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Capsule exact inventory changed')
    physical={}
    for name,row in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe owned path')
        value=(root/name).read_bytes()
        require(len(value)==row['bytes'] and sha(value)==row['sha256'],'Owned payload changed: '+name)
        require(not value.startswith((b'\x7fELF',b'!<arch>\n')),'Native binary was packaged')
        physical[name]=value
    blobs={}
    for name,row in index['logical_files'].items():
        value=physical[row['path']]
        require(row['compression'] in ('none','gzip'),'Unknown retention encoding')
        if row['compression']=='gzip':
            require(value[:3]==b'\x1f\x8b\x08' and value[4:8]==b'\0'*4,'Non-deterministic gzip header')
            value=gzip.decompress(value)
        require(len(value)==row['raw_bytes'] and sha(value)==row['raw_sha256'],'Logical payload changed')
        value.decode('utf-8');blobs[name]=value
    def data(name,checksum=None):
        value=blobs[name]
        if checksum is not None:require(sha(value)==checksum,'Frozen identity changed: '+name)
        return value
    def record(name,checksum=None):return json.loads(data(name,checksum))
    snapshot_math=load_source('portable_snapshot_math',physical['snapshot_math.py'],root/'snapshot_math.py')
    with blocked_apis({}) as attempts:
        source=record('source/source-index.json',SOURCE);parent=record('parent/source-receipt.json',PARENT)
        for prefix,files,seal,extra in [('source',source['files'],'source-index.json',set()),
                ('parent',parent['payloads'],'source-receipt.json',{'source-receipt.sha256'})]:
            names={n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/')}
            require(names==set(files)|{seal}|extra,'Nested source inventory differs')
            for name,row in files.items():
                require(len(data(prefix+'/'+name,row['sha256']))==row['bytes'],'Nested source size differs')
        require(data('parent/source-receipt.sha256').decode().strip()==PARENT,'Parent receipt sidecar differs')
        require(len(source['files'])==72 and len(parent['payloads'])==31 and
                all(source['files'][n]==r for n,r in parent['payloads'].items()),'Exact31-parent join differs')
        require(data('source/owned_snapshots.f90',MODULE)==data('parent/owned_snapshots.f90',MODULE),
                'Compiled module source changed')
        for name,row in parent['external_inputs'].items():
            require(len(data('external/'+name,row['sha256']))==row['bytes'],'External parent input changed')
        ancestry=record('source/ancestry.json');pins=record('source/input-pins.json')
        for name,checksum in {**ancestry['source_files'],**pins['support']}.items():
            data('external/'+name,checksum)
        unit_math=snapshot_math.replay(blobs,load_source)
        build_terminal=record('build/terminal.json',BUILD);built=record('build/result.json',BUILD_RESULT)
        run_terminal=record('run/terminal.json',RUN);ran=record('run/result.json',RUN_RESULT)
        for phase,terminal,result in [('build',build_terminal,built),('run',run_terminal,ran)]:
            require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and
                    terminal['cleanup']['survivors']==[],'Native phase was not successfully closed')
            require(terminal['result_sha256']==sha(data(phase+'/result.json')) and
                    result['source_index_sha256']==SOURCE,'Phase source/result binding differs')
            require(terminal['cap_seconds']==180 and terminal['cleanup_reserve_seconds']==60 and
                    terminal['original_end_monotonic']==terminal['deadline_monotonic'] and
                    terminal['original_end_monotonic']-terminal['original_start_monotonic']==180,
                    'Original phase admission window changed')
            require(0<=(datetime.fromisoformat(terminal['finished_utc'])-
                        datetime.fromisoformat(terminal['started_utc'])).total_seconds()<180,
                    'Recorded phase duration exceeded budget')
            require(record(phase+'/limits.json')['actual']=={'address_space_bytes':[4294967296]*2,
                    'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Actual phase limits changed')
            require(record(phase+'/admission.json')['source_index_sha256']==SOURCE,'Phase admission differs')
        require(ran['reviewed_build_sha256']==BUILD and record('run/admission.json')['reviewed_build_sha256']==BUILD,
                'Separately reviewed BUILD missing')
        require(built['module_sha256']==MODULE and built['snapshot_main_calls']==built['scientific_reducer_calls']==0,
                'BUILD main/scientific scope changed')
        for flag in ('actual_constructor_completion_qualified','final_dot_runtime_qualified','full_caller_qualified',
                     'physical_or_performance_qualification','q_at_use_lowering_qualified',
                     'scientific_dispatch_enabled','snapshot_ownership_qualified'):
            require(ran[flag] is False,'Standalone result inflated into physical/caller qualification')
        require(ran['scientific_reducer_calls']==0,'A scientific reduction was relabelled')
        commands=sorted(n for n in blobs if n.startswith('build/commands/') and n.endswith('.command.json'))
        require(len(commands)==57,'BUILD command coverage differs')
        for name in commands:
            stem=name[:-len('.command.json')];closed=record(stem+'.terminal.json')
            require(closed['returncode']==0 and closed['stdout_sha256']==sha(data(stem+'.stdout.txt')),
                    'BUILD command/log binding differs')
            require(record(name)['argv'][0] not in [r['path'] for r in built['variants'].values()],
                    'BUILD executed a snapshot main')
        wanted={'__ml_private_owned_snapshots_MOD_owned_'+operation+'_'+precision
                for operation in ('reset','prepare','commit') for precision in ('dp','qp')}
        for variant,row in built['variants'].items():
            names=re.findall(r'^\s*[0-9a-fA-F]+\s+[Tt]\s+(\S+)\s*$',
                             data('build/commands/'+variant+'-symbols.stdout.txt').decode(),re.M)
            require(all(names.count(name)==1 for name in wanted),'Six real module API definitions are not unique')
            require(row['flags']==(['-O0','-fcheck=all','-ftrapv'] if variant=='checked' else
                                    ['-O3','-fno-fast-math']),'Qualified module flags changed')
        require(built['variants']['checked']['objects']['driver.o']==
                built['variants']['optimized']['objects']['driver.o'],'Fixed driver changed between variants')
        source_pins=record('source/input-pins.json');excluded=index['excluded_binaries']
        require(len(excluded)==9 and not any(n in blobs for n in excluded),'Binary inventory changed')
        require(excluded['build/fenv_observer.o']['sha256']==built['fenv_object_sha256'],'Observer binary identity lost')
        replayed={};serial=[]
        for variant,row in built['variants'].items():
            prefix='build/'+variant;folder='run/'+variant
            require(excluded[prefix+'/check']['sha256']==row['sha256'],'Executable binary identity lost')
            for name,checksum in row['objects'].items():
                require(excluded[prefix+'/'+name]['sha256']==checksum,'Object binary identity lost')
            prebound=record(prefix+'/prelink.json',row['prelink_sha256'])
            data(prefix+'/link.map',row['link_map_sha256'])
            static=record(prefix+'/static-runtime.json',row['static_runtime_sha256'])
            require(row['runtime']['files']==static['files']==source_pins['runtime_before_link_augmentation'],
                    'Standalone runtime identity set changed')
            require(len(static['files'])==5 and all(prebound['bound'].get(n)==h for n,h in
                    {**row['actual_link_loads'],**static['files']}.items()),'Actual recorded LOAD/provider was not prebound')
            aliases={n:n for n in static['files']}
            aliases.update({x['selected_spelling']:x['selected'] for x in static['resolutions'] if 'selected_spelling' in x})
            providers=set()
            for line in row['runtime']['listing'].splitlines():
                line=line.strip()
                if not line or line.startswith('linux-vdso.so.1 '):continue
                match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
                require(match is not None and match[1] in aliases,'Unrecognized recorded actual loader provider')
                providers.add(aliases[match[1]])
            providers.add(row['runtime']['loader'])
            require(providers==set(static['files']),'Actual PT_INTERP listing/provider join differs')
            require(not any(re.search(r'float|quadmath|multc|trunc|extend|feset|ieee',n,re.I)
                            for n in row['module_undefined_symbols']),'Snapshot module FP/environment dependency claim invalid')
            command=record(folder+'/command.json');started=record(folder+'/started.json')
            closed=record(folder+'/terminal.json');pid=started['pid']
            require(closed['returncode']==0 and not closed['stdout_tail'] and not data(folder+'/stderr.txt'),
                    'Standalone worker failure lost')
            require(sha(data(folder+'/stderr.txt'))==closed['stderr_sha256'],'Worker stderr identity changed')
            require(data(folder+'/ready.txt').decode().strip()=='READY '+str(pid) and
                    data(folder+'/done.txt').decode().strip()=='DONE '+str(pid),'Worker handshakes differ')
            require(command['new_session'] is False,'Unexpected worker session')
            serial.append((datetime.fromisoformat(command['started_utc']),datetime.fromisoformat(closed['finished_utc'])))
            for edge in ('before','after'):
                mapped=record(folder+'/maps-'+edge+'.json')
                require(mapped['ELF']=={**static['files'],row['path']:row['sha256']} and mapped['affinity']==[12] and
                        not mapped['exclusive_CPU'],'Actual worker provider/CPU set differs')
                paths={posixpath.normpath(line.split(None,5)[5]) for line in mapped['raw'].splitlines()
                       if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['ELF'])|set(mapped['data']) and
                        all(source_pins['allowed_data_mappings'].get(n)==h for n,h in mapped['data'].items()),
                        'Recorded raw maps disagree with stored identities')
                require(mapped['pid']==mapped['identity']['pid']==pid,'Mapped worker PID differs')
                keys=('pid','parent_pid','process_group','session','start_time_ticks')
                require(all(mapped['identity'][k]==started['identity'][k] for k in keys),'Worker identity changed')
                require(any(all(x[k]==mapped['identity'][k] for k in ('pid','process_group','session','start_time_ticks'))
                            for x in run_terminal['cleanup']['observed_owned_processes']),'Worker absent from ownership cleanup')
            raw=data(folder+'/observations.txt',RAW)
            require(len(raw)==2156955 and closed['observations_sha256']==RAW,'Raw observation coverage changed')
            result=unit_math['variant_results'][variant]
            require(result==record(folder+'/analysis.json')==ran['variants'][variant],
                    'Strict raw native replay differs from retained results')
            require(result['analysis_only'] is True and result['case_count']==37 and result['cohorts']==20 and
                    result['observations']==740 and result['module_calls']==1840 and
                    result['complete_integer_state_words']==47360 and
                    result['execution_authorization_checked'] is False and
                    result['native_snapshot_transport_qualified'] is False,
                    'Analysis-only component fields or observed unit coverage changed')
            replayed[variant]=result
        require(serial[0][1]<=serial[1][0],'Workers were not serial')
        require(replayed['checked']==replayed['optimized'] and
                data('run/checked/observations.txt')==data('run/optimized/observations.txt'),
                'Checked and optimized raw outcomes differ')
        for label,(checksum,status) in REVIEWS.items():
            require(record('reviews/'+label+'/receipt.json',checksum)['status']==status,
                    'Independent/parent review changed')
        note=json.loads(physical['root-pure-review-note.json'])
        require([r['status'] for r in note['BUILD_review']]==['failed','passed'] and
                note['RUN_review']=={'tool_chunk':'23067c','status':'passed'} and note['new_native_calls']==0,
                'Honest root pure-review history changed')
        require(not attempts,'Forbidden effects during portable numerical replay')
        facts={'fixture_cases':37,'bounded_scenarios':27,'variants':2,'executed_cohorts':20,
            'total_observations':1480,'module_API_outcomes':3680,'complete_integer_state_words':94720,
            'checked_optimized_raw_identical':True,'observed_environment_and_restoration_exact':True,
            'actual_compiled_module_definitions_verified':True,'observed_standalone_module_cases_verified':True,
            'component_analysis_only_flags_preserved':True,'build_commands_passed':57,
            'runtime_providers_per_variant':5,'native_binaries_packaged':0,'excluded_binary_identities':9,
            'new_compiler_loader_native_calls':0,'canonical_current_point_origin_qualified':False,
            'actual_constructor_completion_qualified':False,'q_at_use_lowering_qualified':False,
            'final_dot_runtime_qualified':False,'full_caller_qualified':False,
            'same_buffer_mixed_precision_qualified':False,'runtime_precision_tags_enforced':False,
            'snapshot_ownership_qualified':False,'scientific_dispatch_enabled':False,
            'physical_amplitude_qualified':False,'performance_acceptance':False,
            'composite_prototype_2f07_qualified':False}
        require(index['claims']==facts,'Published claims differ from observed standalone evidence')
    return {**facts,'verified':True,'forbidden_API_attempts':len(attempts),
            'runtime_identity_scope':'Recorded LOAD/PT_INTERP/maps only; no live provider/binary reads.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index-sha256');args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))


