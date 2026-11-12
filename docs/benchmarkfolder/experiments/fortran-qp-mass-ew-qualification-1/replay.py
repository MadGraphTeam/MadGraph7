"""Standalone contained EW source/GEN/BUILD/ordinary replay; no native execution."""
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



@contextmanager
def contained_reads(memory,directories):
    """Serve exact preloaded capsule files/metadata; deny every outside effect."""
    directories=set(directories);files=set(memory);known=directories|files
    def key(path):
        value=os.path.normpath(str(path))
        if value not in known:
            attempts.append(value);raise RuntimeError('Outside contained capsule path refused: '+value)
        return value
    with blocked_apis(memory) as attempts:
        saved={name:getattr(Path,name) for name in ('resolve','rglob','is_file','is_dir','is_symlink')}
        def resolve(path,*args,**kwargs):return Path(key(path))
        def rglob(path,pattern,*args,**kwargs):
            value=key(path)
            if value not in directories or pattern!='*':
                attempts.append(value);raise RuntimeError('Only contained exact inventory enumeration is allowed')
            return iter(Path(p) for p in sorted(known) if p.startswith(value+'/'))
        try:
            Path.resolve=resolve;Path.rglob=rglob;Path.is_file=lambda p:key(p) in files;Path.is_dir=lambda p:key(p) in directories;Path.is_symlink=lambda p:(key(p) and False)
            yield attempts
        finally:
            for name,value in saved.items():setattr(Path,name,value)

def verify(root,expected_index_sha256=None):
    # Preload the standard-library dependencies of both unchanged contained
    # component replayers before blocking import-related filesystem access.
    import ast
    import difflib
    import fractions
    import math
    import posixpath
    import struct
    from collections import Counter
    from pathlib import PurePosixPath
    root=Path(root).resolve();index_raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(index_raw)==expected_index_sha256,'Final EW capsule seal changed')
    index=json.loads(index_raw);require(index['schema']=='EW-QP-mass-contained-qualification-v1','Wrong final EW schema')
    actual=set();directories={str(root)}
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Final EW capsule symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
        elif path.is_dir():directories.add(str(path))
    require(actual==set(index['files'])|{'index.json'},'Final EW exact physical inventory changed')
    physical={}
    for name,pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe final EW member')
        raw=(root/name).read_bytes();require(sha(raw)==pin['sha256'] and len(raw)==pin['bytes'],'Changed final EW physical member: '+name)
        require(not raw.startswith(b'\x7fELF'),'Native ELF bytes packaged');physical[name]=raw
    memory={str(root/name):raw for name,raw in physical.items()};memory[str(root/'index.json')]=index_raw
    def review_text(name):
        pin=index['review_text_members'][name];raw=physical[pin['path']]
        if pin['compression']=='gzip':require(raw[:3]==b'\x1f\x8b\x08' and raw[4:8]==b'\0'*4,'Nondeterministic review gzip');raw=gzip.decompress(raw)
        else:require(pin['compression']=='none','Unknown review compression')
        require(sha(raw)==pin['raw_sha256'] and len(raw)==pin['raw_bytes'],'Changed review text');raw.decode('utf-8');return raw
    reviews={name:review_text(name) for name in index['review_text_members']}
    require(sha(physical['closed-chain/collection-state.json'])=='6804182635afed62c4e0f35308816806834a4905760eca502791301cab1bb706' and
        sha(physical['ordinary-section/collection-state.json'])=='f74f42bfea3e45d651eba0e22e9dbe4c457f31a913038637f95153fd0f3b2cc3','Unchanged component draft ancestry changed')
    for name,wanted in index['component_sha256'].items():require(sha(physical['components/'+name])==wanted,'Contained component replayer changed')
    with contained_reads(memory,directories) as attempts:
        chain=load_source('EW_contained_closed_chain',physical['components/replay_closed.py'],root/'components/replay_closed.py')
        ordinary=load_source('EW_contained_ordinary_pair',physical['components/replay_ordinary.py'],root/'components/replay_ordinary.py')
        chain_result=chain.verify(root/'closed-chain','6804182635afed62c4e0f35308816806834a4905760eca502791301cab1bb706')
        ordinary_result=ordinary.verify(root/'ordinary-section','f74f42bfea3e45d651eba0e22e9dbe4c457f31a913038637f95153fd0f3b2cc3')
        require(chain_result['closed_chain_replayed'] and ordinary_result['ordinary_pair_replayed'] and chain_result['forbidden_API_attempts']==ordinary_result['forbidden_API_attempts']==0,'Contained chain/ordinary replay failed')
        # Their draft-state flags remain historical and byte-for-byte unchanged;
        # the actual closed review is authenticated separately here.
        require(ordinary_result['independent_pair_review_pending'],'Historical ordinary collection state was rewritten')
        pair_raw=reviews['reviews/ordinary-pair/receipt.json'];require(sha(pair_raw)=='f34cf67f3b21ecd934d67e40c14534c9d2a8a7ba8332464286a3234e6b4ce391','Closed independent ordinary review changed')
        pair=json.loads(pair_raw);require(pair['status']=='passed' and not pair['blockers'] and pair['qualification']['ordinary_DP_regression'] and
            not pair['qualification']['changed_QP_mass_path'] and not pair['qualification']['independent_full_amplitude_finite_truth'],'Independent ordinary review scope changed')
        for name,pin in pair['artifacts'].items():require(sha(reviews['reviews/ordinary-pair/'+name])==pin['sha256'] and len(reviews['reviews/ordinary-pair/'+name])==pin['bytes'],'Independent review replay-code/stdout changed')
        require(sha(reviews['reviews/ordinary-pair/checks.json'])==sha(reviews['reviews/ordinary-pair/checks-optimized.json'])=='2ff3d8bf0714a8909a1f5d4ec36a8193eaf39da4c13e102a3a93f39b96219140','Independent normal/optimized replay outputs changed')
        source_review=reviews['reviews/ordinary-source/receipt.json'];require(sha(source_review)=='5993ed5bad1bf8b76fa66ed9d9cfb729357b47771ea70b5873bc343ca42719fd','Independent ordinary source review changed')
        require(pair['verification']['normal_checks']==pair['verification']['optimized_checks']==28 and pair['verification']['maximum_relative_difference']==0 and
            pair['verification']['coefficient_regression_tolerance']==3e-6 and pair['verification']['coefficient_ordinary_target_tolerance']==1e-8,'Independent pair comparison scope changed')
        base=json.loads(physical['closed-chain/collection-state.json']);ordinary_state=json.loads(physical['ordinary-section/collection-state.json'])
        facts={'source_GEN_correspondence_BUILD_replayed':True,'independent_ordinary_DP_pair_review_closed':True,
            'generated_inputs_each':427,'raw_identical_bindings':421,'explained_changed_bindings':6,'public_units_raw_exact':25,'private_helpers':9,
            'guarded_calls':200,'DP_MP_mass_actual_pairs':626,'selected_DP_calls_logically_exact':200,'selected_DP_calls_raw_exact':0,
            'BUILD_commands':168,'fresh_compiles':42,'links':2,'unique_LOAD_bindings_each':54,'runtime_providers_each':15,'MTIRUST_definitions_each':23,
            'ordinary_workers':2,'ordinary_physical_points':100,'returned_final_rows_each':200,'returned_coefficient_slots_each':800,'returned_ACC_slots_each':200,
            'learning_calls_each':101,'learning_raw_rows_each':202,'steady_learning_rows_each':200,'accepted_DP219_points_each':100,'accepted_QP_points_each':0,
            'all_observed_words_status_history_IEEE_software_equal':True,'maximum_reported_ACC_each':1.0098069303873654e-7,
            'pair_regression_tolerance':3e-6,'pair_target_tolerance':1e-8,'independent_absolute_accuracy_qualified':False,
            'changed_QP_mass_path_observed':False,'unobserved_internal_QP_attempts_excluded':False,'native_mass_input_witness_qualified':False,
            'genuine113bit_physical_mass_inputs_qualified':False,'complex_or_derived_MW_qualified':False,'bootstrap_scientifically_qualified':False,
            'independent_full_amplitude_truth':False,'warning_free_execution_qualified':False,'IEEE_lowering_or_trap_freedom_qualified':False,
            'cache_qualified':False,'performance_acceptance':False,'production_promotion':False,'component_draft_states_unchanged':True,
            'excluded_native_identity_records':len(base['excluded_binaries'])+len(ordinary_state['excluded_binaries']),'native_binaries_packaged':0,'new_native_executions':0}
        require(index['claims']==facts,'Final EW claims exceed the actual closed evidence');require(not attempts,'Outer contained guard observed a forbidden attempt')
    return {**facts,'verified':True,'outer_forbidden_API_attempts':0,'component_forbidden_API_attempts':0}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
