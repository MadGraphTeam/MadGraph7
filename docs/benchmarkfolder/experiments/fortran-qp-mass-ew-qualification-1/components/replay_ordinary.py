"""File-only replay of the closed ordinary EW pair; no new native work or final seal."""
import argparse
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import types

sys.dont_write_bytecode=True

def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


@contextmanager
def blocked_apis(memory):
    """Only preloaded logical text reads are available during semantic replay."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden API during portable replay')
    def memory_bytes(path,*args,**kwargs):
        if str(path) not in memory:return deny(str(path))
        return memory[str(path)]
    def memory_text(path,encoding=None,errors=None,*args,**kwargs):
        return memory_bytes(path).decode(encoding or 'utf-8',errors or 'strict')
    def patch(obj,name,value=deny):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        patch(Path,'read_bytes',memory_bytes);patch(Path,'read_text',memory_text)
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob',
                     'write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        for name in ('open',):patch(builtins,name)
        for name in ('open','open_code'):patch(io,name)
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp',
                     'execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty',
                     'mkdir','makedirs','unlink','remove','rename','replace','rmdir','chmod','chown','truncate','utime',
                     'link','symlink','kill','killpg'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        for name in ('CDLL','PyDLL','_dlopen'):patch(ctypes,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        for name in ('socket','create_connection'):patch(socket,name)
        for name in ('signal','setitimer','alarm','pthread_kill'):patch(signal,name)
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def load_source(name,raw,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module



SOURCE='0c6792e4531a9da678f098f9b9ceb93479f21734c9c06e731316139a9f45cd41'
PAIR={'original':('9663923d31ee9a87238aa564380c435dc3a81ab631649d9b604cfa05a4110173','e9e8c7f35dacc0264157d9edf93d0d7b1c2654b924a169ce4b533e714f7eef30'),
 'candidate':('0614cdc57afeb21551151e7ea53c2f5ff9e1a64700d08e0d5a86bbc00f3e3b60','f1c3c7eaae4f7dc8175860637964073f94aadfcfc4a3db1c55ea093d2036a89e')}
def verify(root,expected_state_sha256=None):
    import math
    import struct
    from collections import Counter
    root=Path(root).resolve();raw=(root/'collection-state.json').read_bytes()
    if expected_state_sha256:require(sha(raw)==expected_state_sha256,'Ordinary section state changed')
    state=json.loads(raw);require(state['schema']=='EW-QP-mass-ordinary-pair-draft-v1' and not state['final_capsule_sealed'],'Ordinary section must stay unsealed')
    require(state['independent_pair_review_pending'],'Independent pair review has not yet been bound')
    require(state['closed_chain_state_sha256']=='6804182635afed62c4e0f35308816806834a4905760eca502791301cab1bb706' and state['source_sha256']==SOURCE,'Closed GEN/source/BUILD or ordinary source ancestry changed')
    require(state['observed_QP_points']==[] and not state['QP_mass_path_covered'],'DP-only pair cannot acquire QP path coverage')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Ordinary evidence symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual=={v['path'] for v in state['logical_files'].values()}|{'collection-state.json'},'Ordinary exact inventory changed')
    blobs={}
    for name,pin in state['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not Path(pin['path']).is_absolute() and '..' not in Path(pin['path']).parts,'Unsafe ordinary logical/physical path')
        value=(root/pin['path']).read_bytes()
        if pin['compression']=='gzip':require(value[:3]==b'\x1f\x8b\x08' and value[4:8]==b'\0'*4,'Nondeterministic ordinary gzip');value=gzip.decompress(value)
        else:require(pin['compression']=='none','Unknown ordinary text compression')
        require(sha(value)==pin['raw_sha256'] and len(value)==pin['raw_bytes'],'Changed ordinary logical text: '+name);value.decode('utf-8');require(not value.startswith(b'\x7fELF'),'Ordinary native payload refused');blobs[name]=value
    def data(name,wanted=None):
        require(name in blobs,'Missing ordinary logical text: '+name);value=blobs[name]
        if wanted:require(sha(value)==wanted,'Frozen ordinary closure changed: '+name)
        return value
    def record(name,wanted=None):return json.loads(data(name,wanted))
    source=record('ordinary-source/source-index.json',SOURCE)
    require({n[len('ordinary-source/'):] for n in blobs if n.startswith('ordinary-source/')}==set(source['files'])|{'source-index.json'},'Ordinary exact source namespace changed')
    for name,pin in source['files'].items():require(len(data('ordinary-source/'+name,pin['sha256']))==pin['bytes'],'Ordinary frozen source size changed')
    memory={str(root/name):value for name,value in blobs.items()}
    with blocked_apis(memory) as attempts:
        collector=load_source('EW_ordinary_frozen_collector',data('ordinary-source/collector.py'),root/'ordinary-source/collector.py')
        comparator=load_source('EW_ordinary_frozen_comparison',data('ordinary-source/comparison.py'),root/'ordinary-source/comparison.py')
        pins=record('ordinary-source/input-pins.json');gate=record('closure/reviewed-build-gate.json','d19814faae3c2e5e6ae2235e2b7b87dbb25337021ff497cdec8bc7b204576ebf')
        roles={};maps=0;expected_native=set()
        for role,(term_sha,result_sha) in PAIR.items():
            prefix='ordinary/'+role;term=record(prefix+'/terminal.json',term_sha);result=record(prefix+'/result.json',result_sha);started=record(prefix+'/started.json')
            require(term['status']==result['status']=='passed' and term['returncode']==0 and term['result_sha256']==result_sha and not term['cleanup']['survivors'] and
                term['source_index_sha256']==result['source_index_sha256']==SOURCE,'Ordinary closed terminal/result/source changed')
            require(term['pid']==started['pid'] and term['deadline_monotonic']==term['original_end_monotonic']==started['deadline_monotonic'] and term['cap_seconds']==started['cap_seconds']==600 and
                term['cleanup_reserve_seconds']==started['cleanup_reserve_seconds']==60 and term['cpu']==12 and not term['cpu_reservation'] and
                term['resource_limits']=={'address_space_bytes':17179869184,'core_bytes':0,'file_bytes':536870912},'Ordinary original bounds changed')
            require(result['BUILD_terminal_sha256']=='f7cac8b007f41e09964b95c9a0fb6172cb0f4eaa249ed3192188f9fa1af46b22' and result['BUILD_independent_review_sha256']=='fa3e423b60d76d840efe66376ec47d00b8ffb73642192f3504eb3bfcbbefa9be' and
                result['BUILD_raw_inventory_sha256']=='c1e46e69465817ec0e43b6df490eeac8ec5d0d7fd96548ea4b36baf71edec82a' and result['BUILD_gate_sha256']==term['BUILD_gate_sha256']=='d19814faae3c2e5e6ae2235e2b7b87dbb25337021ff497cdec8bc7b204576ebf','Ordinary reviewed BUILD chain changed')
            for name,wanted in result['raw_files'].items():data(prefix+'/'+name,wanted)
            raw={name:data(prefix+'/'+name).decode() for name in ('dimensions.dat','warmup.dat','exact-words.dat','exact-ieee.dat','software-flags.dat')}
            collected=collector.collect(raw)
            require(collected==record(prefix+'/collected.json')==result['lane'],'Ordinary raw/collected/result lane join changed')
            require(len(collected['rows'])==200 and len(collected['learning']['rows'])==200 and len(collected['learning']['history'])==101 and len(collected['learning']['all_raw_records'])==202 and
                collected['cohort']['backend_precision_counts']=={'9:DP':100} and collected['cohort']['status_counts']=={'219':100} and not collected['cohort']['accepted_QP_points'],'Actual ordinary all100 DP cohort/learning coverage changed')
            require(result['UseQPIntegrandForTIRust'] and result['finite'] and result['LoopPoleMode']==0 and result['normal_CTModeRun']==-1 and result['TIRust_caches']=={'DP':0,'QP':0} and
                result['threshold']==1e-6 and result['forced_independent_stability'] and result['full100_learning'] and not result['historical_cohort_assumed'] and not result['QP_cohort_required'] and
                not result['performance_acceptance'] and not result['source_promoted'],'Ordinary numerical policy or scope changed')
            command=record(prefix+'/worker-command.json');check=command['argv'][3];elf=command['executable_sha256']
            require(command['driver_source_sha256']=='cf2816e4d8ef9a176522f319ae44982e9f07c2003088daccab4313700c5b990b' and command['argv'][1:3]==['-c','12'] and
                command['cwd']==str(Path(check).parent),'Actual ordinary worker command/driver binding changed')
            expected_native.add('ordinary/'+role+'/check');require(state['excluded_binaries']['ordinary/'+role+'/check']['sha256']==elf,'Excluded ordinary ELF identity changed')
            saved=[]
            for edge in ('before','after'):
                mapped=record(prefix+'/maps-'+edge+'.json');maps+=1;saved.append(mapped)
                require(mapped['files']=={**pins['expected_runtime']['files'],check:elf} and mapped['affinity']==[12] and mapped['parent_pid']==term['pid'] and
                    mapped['pid']==mapped['process_group']==mapped['session'],'Actual ordinary runtime/provider/process binding changed')
                paths={os.path.normpath(line.split(None,5)[5]) for line in mapped['raw_maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['files']),'Actual ordinary kernel map closure changed')
                require(any(all(row[k]==mapped[k] for k in ('pid','parent_pid','process_group','session','start_time_ticks')) for row in term['cleanup']['observed_owned_processes']),'Ordinary mapped process identity absent from cleanup')
                for path,wanted in mapped['files'].items():
                    expected_native.add('mapped/'+path);require(state['excluded_binaries']['mapped/'+path]['sha256']==wanted,'Excluded mapped identity changed')
            require(all(saved[0][k]==saved[1][k] for k in ('pid','parent_pid','process_group','session','start_time_ticks','files')),'Ordinary before/after worker identity/provider closure changed')
            ready=record(prefix+'/ready.json');protocol=record(prefix+'/protocol.json')
            require({k:ready[k] for k in ('points','nborn','ndim','nso')}=={'points':100,'nborn':1,'ndim':1,'nso':1} and len(protocol)==6 and
                protocol[0]['direction']=='receive' and protocol[0]['fields'][:6]==['READY','1','100','1','1','1'] and float(protocol[0]['fields'][6])==ready['cold_seconds'] and
                protocol[1:]==[{'direction':'send','line':'EXACT 1 0'},{'direction':'receive','fields':['EXACTED','1']},{'direction':'send','line':'STOP 0 0'},{'direction':'receive','fields':['BYE']},{'direction':'exit','returncode':0,'trailing_bytes':0}] and result['protocol_complete'],'Actual ordinary READY/EXACT/STOP/exit protocol changed')
            require(result['reviewed_original']==term['reviewed_original']==(None if role=='original' else PAIR['original'][0]),'Candidate original-terminal admission changed')
            roles[role]=collected
        require(set(state['excluded_binaries'])==expected_native and maps==4,'Ordinary native-exclusion/map inventory changed')
        compared=collector.compare(roles['original'],roles['candidate'],comparator.compare_rows)
        require(roles['original']['learning']['all_raw_records']==roles['candidate']['learning']['all_raw_records'],'Complete bootstrap/full100 learning raw record parity changed')
        require(compared==record('ordinary/candidate/comparison.json')==record('ordinary/candidate/result.json')['comparison'] and compared['passed'] and
            not compared['final']['word_differences'] and not compared['learning']['word_differences'] and not compared['accepted_QP_observed_in_each_arm'] and
            compared['learning_history_equal'] and compared['observed_IEEE_equal'] and compared['software_latches_equal'],'Ordinary exact pair parity/scope changed')
        require(not attempts,'Forbidden ordinary replay API attempted')
    return {'ordinary_pair_replayed':True,'workers':2,'physical_points':100,'final_answer_rows_each':200,'steady_learning_answer_rows_each':200,
        'learning_calls_each':101,'learning_raw_rows_each':202,'reported_final_status':'219','backend_precision_counts_each':{'9:DP':100},
        'candidate_all_words_status_IEEE_software_equal':True,'observed_QP_points':[],'QP_mass_path_covered':False,'runtime_map_snapshots':4,
        'forbidden_API_attempts':0,'independent_pair_review_pending':state['independent_pair_review_pending'],'independent_physical_truth':False,
        'performance_acceptance':False,'production_promotion':False,'final_capsule_sealed':False,'new_native_executions':0}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--draft',type=Path,required=True);cli.add_argument('--state-sha256');args=cli.parse_args()
    print(json.dumps(verify(args.draft,args.state_sha256),sort_keys=True,indent=2))
