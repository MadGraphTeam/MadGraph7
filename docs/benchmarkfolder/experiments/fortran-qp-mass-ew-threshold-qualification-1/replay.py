"""Standalone contained EW threshold refusal replay; no native execution."""
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



# No retained controller, worker, review script or project module is executed.
# Only unchanged pure parsers plus selected collector FunctionDefs are evaluated.
import ast
import difflib
import math
import struct
from collections import Counter

RUN='543cf39207106d354add7972072aa0bfebf20e22882be2bd1199a733647b2009'
BUILD='d00a9c97b5c4a47264ac1a8e2dcc9d07984313fb618f0d7a993f75702f98a3c5'
BT='30c92c137088fe014c1b939149def320fc2f3888bc7b830f748c7ba850231591'
BR='bc9fcee46366192bb6544593a0220c0055575c12ab6c9b99dadff04234d19f2c'
PINS={
 'root/build2.json':'c9bf073afcf9e2dc62eefbbda97609cfa2730a333542ec95e7be38f32553f6ba',
 'root/original2.json':'23b9d4a142da757360abe8a7e72e27152ae058b1fab9b1d9c80b10510e5b8c17',
 'root/candidate2.json':'b0fd8695dec1f6c2e5b0ed417b5b540cacdaa240d872c4c600794dca94a14bd0',
 'root/original1.json':'aa311d30f18381d21cd5193368b9c5590cd9093a1bfee31eb4727b3b563180ac',
 'root/build1.json':'4dc3332fcbe9e92a00939f4f3954759378ee4f63059592c0b26306ce3dfde7b0',
 'root/build1-gate.json':'9429ebafb63ba633cb1d8ce95d179cdff0365c6b7b6a7291469db7e24965e18b',

 'run2/original/terminal.json':'3e8facada671f781aa10976958265789d192fa2deddd95a656120bb277858c15',
 'run2/original/result.json':'edb82a9f99972b3f6eaef714e2a42835a6acff499c1870cd77444a5d026c854f',
 'run2/candidate/terminal.json':'f5e47c5b7e3f897815d6ff3fecd5efa10464225cddccf7f00090bcc8b45d8e84',
 'run2/candidate/result.json':'04b7df18b738899627fa8e7eadfb7d4abd9774fa205edca0ae940f01b9433a29',
 'reviews/pair/receipt.json':'544b8f427cd3e30bf91b3d6dd3966e6cc8053e3808a64a4b2ca325c3ba519d0c',
 'reviews/original/receipt.json':'4e85daa0cb5f4010f2c5777783b80e7d18f3730991411dc8785b98315bab02d6',
 'reviews/pair-science/receipt.json':'5d31adfd95bcf0f4b76b7cff626516bda1a0fa4b79bc69cd638ff5715371347c',
 'reviews/original-science/receipt.json':'244f05e2ede3c139d28f18f3aaf3fb8eda0f7a4a845ff0ece995db3efba0f754',
 'reviews/build2/receipt.json':'5913f0ff3ac2dc7b225574935713887e79e31dcbb088bbb931dba7fe74ffb5d0',
 'reviews/build2/raw-inventory.json':'01309a6b45c1d10ab055f9a9e611435e022181375fc6443ac8a7600f126f0dfd',
 'reviews/build2-objects/receipt.json':'e166226b30df739e3f96e90c1f1ee742548b55de0c6bc97054fd7feb15553d2a',
 'reviews/failure1/receipt.json':'00e4246eb29c1a285dfc7c312bbd70647b192df382d992ffce89fe0571aabe4e',
 'history/run1/original/terminal.json':'40d966bcf175d2698a8065ea6f8175d9733c26cabf865670f3999aea69611064',
 'source/run2/source-index.json':RUN,'source/build2/source-index.json':BUILD,
 'build2/terminal.json':BT,'build2/result.json':BR,
 'ancestry/ordinary-public-index.json':'48a8b22cdb652cb173ebe2714aab7cf8efee07ec5793b1a30ceedfabdec6b04a',
 'ancestry/ordinary-collection-state.json':'f74f42bfea3e45d651eba0e22e9dbe4c457f31a913038637f95153fd0f3b2cc3',
 'ancestry/ordinary-original-result.json':'e9e8c7f35dacc0264157d9edf93d0d7b1c2654b924a169ce4b533e714f7eef30',
 'ancestry/ordinary-candidate-result.json':'f1c3c7eaae4f7dc8175860637964073f94aadfcfc4a3db1c55ea093d2036a89e',
 'ancestry/base-build-result.json':'60a99e609b0aae3576c31f3f49cf6a250f21dfa1e7dbc670e95f4c49ac72d603',
 'ancestry/base-build-terminal.json':'f7cac8b007f41e09964b95c9a0fb6172cb0f4eaa249ed3192188f9fa1af46b22',
 'source/run2/target-BUILD-gate.json':'712c40a4634fc62bd0648769e18bd72c0abfbfb5214f920b529c3a02977c6138',
 'source/run2/ordinary-reference-gate.json':'c20854f8e1f447ff8c9f99052caa9fde2c27e786b257367eb7ed06f49196accf'}
DIM={'NCOMB':32,'NLOOPGROUPS':200,'NSQUAREDSO':1};PROC='SubProcesses/P0_epem_ttxa'

def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}
def functions_only(name,raw,names,globals):
    tree=ast.parse(raw,filename=name);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Missing actual collector functions')
    obj=types.ModuleType(name);obj.__dict__.update(globals)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),obj.__dict__);return obj

def validate_observed_scope(lane,result):
    """Refuse promotion of a recorded failure or stored INIT bookkeeping."""
    require(result['stored_filters_learned'] is False,'Stored filters falsely called learned')
    require(result['scientific_qualified'] is False and result['independent_physical_truth'] is False and result['source_promoted'] is False,'Observation promoted to science')
    statuses=lane['observed_statuses'];require([r['code'] for r in statuses]==[219,420],'False threshold success')
    require(statuses[0]['precision']=='DP' and statuses[0]['comparison_eligible'] is True and statuses[1]['precision'] is None and statuses[1]['accepted_public_status'] is False and statuses[1]['comparison_eligible'] is False,'False accepted QP claim')
    for state in lane['readiness']['states']:
        require(state['stored_INIT_flags_are_learned'] is False and state['CHECKPHASE']==1 and state['HELDOUBLECHECKED']==0,'False filter-learning claim')
    for row in lane['values']['targets'][1]['answers']:
        require(row['status']==420 and row['public']['accepted_public_status'] is False and row['public']['precision'] is None and row['comparison_eligible'] is False and row['buffers_are_diagnostics_only'] is True and row['accuracy']==1.0,'Failed buffers falsely admitted as QP/scientific truth')

def decode(root,expected_index_sha256=None):
    root=Path(root).resolve();raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(raw)==expected_index_sha256,'Capsule seal changed')
    ix=json.loads(raw);require(ix['schema']=='EW-threshold-contained-refusal-v1','Wrong capsule schema')
    actual=set()
    for f in root.rglob('*'):
        require(not f.is_symlink(),'Capsule symlink refused')
        if f.is_file():actual.add(str(f.relative_to(root)))
    require(actual==set(ix['files'])|{'index.json'},'Exact physical inventory changed')
    physical={}
    for name,pin in ix['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe capsule member')
        b=(root/name).read_bytes();require(identity(b)==pin,'Changed physical capsule member '+name);require(not b.startswith(b'\x7fELF'),'Native ELF packaged');physical[name]=b
    c=json.loads(physical['collection.json']);data={}
    for name,pin in c['members'].items():
        b=physical[pin['path']];require(b[:3]==b'\x1f\x8b\x08' and b[4:8]==b'\0'*4,'Nondeterministic gzip')
        b=gzip.decompress(b);b.decode('utf8');require(identity(b)=={k:pin[k] for k in ('bytes','sha256')},'Logical member changed');require(not b.startswith(b'\x7fELF'),'Decoded native member');data[name]=b
    return ix,c,data,physical

def semantic(data,collection,physical):
    J=lambda n:json.loads(data[n]);T=lambda n:data[n].decode('utf8')
    for n,pin in PINS.items():require(sha(data[n])==pin,'Closed identity changed '+n)
    for ns,spec in collection['source_namespaces'].items():
        index=J(ns+'/source-index.json');require(sha(data[ns+'/source-index.json'])==spec['index_sha256'],'Source index changed')
        names={n[len(ns)+1:] for n in data if n.startswith(ns+'/')}-{'source-index.json'}
        require(names==set(spec['retained_payloads']),'Source namespace drift')
        if spec['complete_namespace']:require(names==set(index['files']),'Complete source namespace omitted payload')
        for n in names:require(identity(data[ns+'/'+n])==index['files'][n],'Source payload identity '+n)
    require(len(J('source/run2/source-index.json')['files'])==24 and len(J('source/build2/source-index.json')['files'])==17,'Source payload counts')
    require(data['source/build2/protocol.py']==data['source/run2/target_protocol.py'],'Compiled/collector policy grammar differs')
    p=J('source/run2/input-pins.json');bp=J('source/build2/input-pins.json');br=J('build2/result.json');bt=J('build2/terminal.json');ex=json.loads(physical['excluded-native-identities.json'])
    require(bp['filter_source']['sha256']==sha(data['policy/loop_matrix.f'])=='87e1c28b7ffa50db9416f88a38883e43ddb289ca8336c493a46eb3b35886cb0b','Pole0 source changed')
    require('BYPASS_CHECK.OR.LOOPPOLEMODE.GE.0' in T('policy/loop_matrix.f'),'Source bypass rule missing')
    require(bt['status']=='passed' and bt['returncode']==0 and bt['result_sha256']==BR and bt['cleanup']['survivors']==[],'BUILD did not close cleanly')
    require(br['status']=='passed' and br['new_driver_compiles']==br['new_links']==2 and br['native_main_calls']==br['reducer_calls']==br['model_probes']==0 and not br['scientific_qualified'],'BUILD scope')
    rawix=J('reviews/build2/raw-inventory.json')['files'];actual={n[len('build2/'):] for n in data if n.startswith('build2/')}
    require(actual==set(rawix) and len(actual)==341,'All BUILD report files required')
    for n,pin in rawix.items():require(identity(data['build2/'+n])==pin,'BUILD raw identity')
    commands=[n for n in actual if n.endswith('.command.json')];require(len(commands)==110,'Actual110 BUILD commands required')
    for n in commands:
        base='build2/'+n[:-len('.command.json')];terminal=J(base+'.terminal.json');require(terminal['returncode']==0 and terminal['stdout_sha256']==sha(data[base+'.stdout.txt']),'BUILD command failed')
    oldbase=J('ancestry/base-build-result.json');ancestry=J('ancestry/ordinary-collection-state.json');require(ancestry['observed_QP_points']==[] and ancestry['QP_mass_path_covered'] is False,'Ordinary ancestor overclaims QP')
    for role,v in br['variants'].items():
        require(len(v['sources'])==428 and len(v['closed_process_objects'])==21 and len(v['actual_link_loads'])==54 and len(v['runtime']['files'])==15,'Actual BUILD coverage counts')
        require(v['sources'][PROC+'/target_driver.f']==identity(data['source/build2/target_driver.f']),'Actual driver source binding')
        require(sha(data['build2-link/'+role+'/target-link.map'])==v['link_map_sha256'],'Actual link map changed')
        loads={line.split(maxsplit=1)[1] for line in T('build2-link/'+role+'/target-link.map').splitlines() if line.startswith('LOAD ')}
        aliases=json.loads(physical['link-aliases.json'])[role];require(loads==set(aliases['resolutions']) and set(aliases['resolutions'].values())==set(v['actual_link_loads']),'Actual54 LOAD alias bindings differ');require(all(v['actual_link_loads'][n]==J('build2/'+role+'/target-prelink.json')['bound'][n] for n in v['actual_link_loads']),'Actual LOAD not prebound')
        require(ex[role]['closed_process_objects']==v['closed_process_objects'] and ex[role]['actual_link_loads']==v['actual_link_loads'] and ex[role]['runtime_provider_files']==v['runtime']['files'] and ex[role]['target_driver_object']==v['target_driver_object'] and ex[role]['executable']['sha256']==v['executable_sha256'] and ex[role]['binary_bytes_retained'] is False,'Native exclusion identity drift')
        require(v['normal_model_and_process_sources_unchanged'] and v['old_main_object_retained_but_not_linked'],'Old science source/object invariant')
        # Exact compiled-input bindings retain COMMON and model includes; do not
        # read excluded compiler modules or any live provider during replay.
        bound=v['target_compile_inputs']['bound_files']
        for n in ('MadLoopParams.inc','coupl.inc','pmass.inc'):require(bound[v['output']+'/'+PROC+'/'+n]==identity(data['policy/'+n]),'Actual include binding')
        require(v['compile_recipe'][1:4]==['-w','-fPIC','-ffixed-line-length-132'] and v['link_recipe'][1:4]==v['compile_recipe'][1:4],'Original scientific flags changed')
        for n,pin in v['closed_process_objects'].items():
            require(oldbase['variants'][role]['process_objects'][n]==pin,'Reused process object changed')
    # Execute the actual contained pure functions with every filesystem/process/
    # loader/network/signal/mutation API blocked. Bootstrap imports are excluded.
    with blocked_apis({}) as effects:
        P=load_source('contained_target_protocol',data['source/run2/target_protocol.py'],'contained:protocol')
        O=load_source('contained_ordinary_collector',data['source/run2/ordinary_collector.py'],'contained:ordinary')
        C=load_source('contained_comparison',data['source/run2/comparison.py'],'contained:comparison')
        F=functions_only('contained_collector',data['source/run2/collector.py'],('require','initial_readiness','collect','compare'),{'P':P,'O':O})
        numeric_calls=[]
        def accepted_compare(a,b,**kwargs):
            require(all(row['status']==219 for row in a+b),'Refused buffer entered numerical comparison')
            numeric_calls.append((len(a),len(b),kwargs.get('tolerance')));return C.compare_rows(a,b,**kwargs)
        lanes={}
        for role in ('original','candidate'):
            r=J('run2/'+role+'/result.json');t=J('run2/'+role+'/terminal.json');prefix='run2/'+role+'/'
            raw={n:T(prefix+n) for n in ('dimensions.dat','warmup.dat','target-raw.dat','target-readiness.dat','ready-readiness.dat')}
            prior=J('ancestry/ordinary-'+role+'-result.json');lane=F.collect(raw,T('source/run2/inputs/targets2.dat'),prior,accepted_compare,DIM);lanes[role]=lane
            require(lane==r['lane']==J(prefix+'collected.json'),'Actual raw replay differs from retained lane');validate_observed_scope(lane,r)
            require(lane['learning']==prior['lane']['learning'],'Full101 initialization history/raw/IEEE/software differs from ordinary reference')
            require(len(lane['learning']['history'])==101 and len(lane['learning']['all_raw_records'])==202 and len(lane['learning']['rows'])==200,'Warmup scope counts')
            require(all(row['status']==219 and row['precision']=='DP' for row in lane['learning']['rows']) and lane['learning_comparison']['passed'] and not lane['learning_comparison']['word_differences'] and not lane['learning_comparison']['status_changes'],'Warmup DP exactness')
            require(lane['ordinary_control']['passed'] and not lane['ordinary_control']['word_differences'],'Accepted target1 parity')
            require(t['status']=='passed' and t['returncode']==0 and t['result_sha256']==sha(data[prefix+'result.json']) and t['cleanup']['survivors']==[],'RUN observation did not close')
            require(t['source_index_sha256']==r['source_index_sha256']==RUN and r['compiled_source_index_sha256']==BUILD and r['BUILD_terminal_sha256']==BT,'RUN source/BUILD join');require(r['BUILD_gate_sha256']==sha(data['source/run2/target-BUILD-gate.json']) and r['BUILD_independent_review_sha256']==sha(data['reviews/build2/receipt.json']) and r['BUILD_raw_inventory_sha256']==sha(data['reviews/build2/raw-inventory.json']),'RUN review/gate bindings')
            require(t['cap_seconds']==900 and t['cleanup_reserve_seconds']==60 and t['original_end_monotonic']==t['deadline_monotonic']==t['bootstrap_monotonic']+900 and t['cpu']==12 and t['cpu_reservation'] is False and t['resource_limits']=={'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0},'Actual RUN caps')
            require(r['status']=='passed' and r['full100_ordinary_initialization_sweep'] and r['all_declared_targets_completed'] and r['requested_targets']==2 and not r['stored_filters_learned'] and not r['QP_cohort_required'] and not r['target2_numeric_success_required'],'Observation-only declared policy')
            require(all(r[n] is False for n in ('independent_physical_truth','scientific_qualified','source_promoted','performance_acceptance')),'Unqualified result promoted')
            names={n[len(prefix):] for n in data if n.startswith(prefix)};require(names==set(r['raw_files'])|{'result.json','terminal.json'},'Exact closed raw report inventory')
            for n,h in r['raw_files'].items():require(sha(data[prefix+n])==h,'Actual raw file identity '+n)
            require(data[prefix+'console.log']==b'' and r['full_stdout_retained'] and r['stdout']==identity(data[prefix+'worker.log']),'Full stdout changed')
            warning=T(prefix+'worker.log');require('##W03 WARNING An unstable PS point was detected.' in warning and 'last QP reduction was deemed unstable because a floating point exception was encountered.' in warning,'Full refused-point warning missing')
            for n,pin in J(prefix+'pre-shutdown-raw-identities.json').items():require(identity(data[prefix+n])==pin,'Raw changed after shutdown')
            att=J('reviews/'+('original' if role=='original' else 'pair')+'/verified-input-bindings.json')
            planned={**p['data_pins'][role],PROC+'/target-check':{'sha256':br['variants'][role]['executable_sha256']}}
            require(set(att['private_runtime_inputs'])==set(planned) and len(planned)==r['runtime_binding']['runtime_files']==17,'Runtime input inventory scope')
            for n,pin in planned.items():require(att['private_runtime_inputs'][n]['sha256']==pin['sha256'],'Retained runtime input binding')
            for n,pin in p['frozen_inputs'].items():require(identity(data['source/run2/inputs/'+n])==pin==att['frozen_source_inputs'][n],'Frozen stream/card binding')
            require(r['ordinary_reference']['gate_sha256']==sha(data['source/run2/ordinary-reference-gate.json']) and r['ordinary_reference']['result_sha256']==sha(data['ancestry/ordinary-'+role+'-result.json']),'Own ordinary reference join')
            require(r['runtime_binding']['physical_card_sha256']==p['frozen_inputs']['param_card.dat']['sha256'] and all(h==p['frozen_inputs']['cache0.dat']['sha256'] for h in r['runtime_binding']['numerical_cards'].values()),'Runtime card policy changed')
            protocol=J(prefix+'protocol.json');require(len(protocol)==6 and protocol[0]['fields'][:6]==['READY','1','100','1','1','1'] and protocol[0]['direction']=='receive' and protocol[1:]==[{'direction':'send','line':'TARGETS 1 0'},{'direction':'receive','fields':['TARGETED','1','2']},{'direction':'send','line':'STOP 0 0'},{'direction':'receive','fields':['BYE']},{'direction':'exit','returncode':0,'trailing_bytes':0}],'Strict READY/TARGETS/STOP/BYE closure')
            ready=J(prefix+'ready.json');require(ready=={'points':100,'ndim':1,'nso':1,'nborn':1,'cold_seconds':float(protocol[0]['fields'][6])},'READY dimensions')
            before=J(prefix+'maps-before.json');after=J(prefix+'maps-after.json');identitykeys=('pid','parent_pid','process_group','session','start_time_ticks','affinity','files')
            require(all(before[k]==after[k] for k in identitykeys) and before['pid']==before['process_group']==before['session'] and before['parent_pid']==t['pid'] and before['affinity']==[12],'Worker identity across map snapshots')
            observed={z['pid']:z for z in t['cleanup']['observed_owned_processes']};require(observed[before['pid']]['start_time_ticks']==before['start_time_ticks'],'Cleanup identity join')
            w=J(prefix+'worker-command.json');v=br['variants'][role];exe=w['argv'][3]
            require(w['executable_sha256']==v['executable_sha256'] and w['driver_source_sha256']==p['driver']['sha256'] and w['compiled_source_index_sha256']==BUILD and w['argv'][1:3]==['-c','12'] and exe==w['cwd']+'/target-check','Actual worker command/source')
            expected={**p['expected_runtime']['files'],exe:v['executable_sha256']}
            require(len(expected)==16 and r['worker_identity']=={'same_process_and_file_providers':True,'anonymous_VMA_changes_allowed':True},'Runtime closure scope')
            for snapshot in (before,after):
                names={line.split(None,5)[5] for line in snapshot['raw_maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(snapshot['files']==expected and names==set(expected) and not any(n.endswith(' (deleted)') for n in names),'Actual15provider+ELF map closure')
            for state in lane['readiness']['states']:
                require(state['CHECKPHASE']==1 and state['HELDOUBLECHECKED']==0 and state['HELOFFSET']==10000 and state['GOODHEL']==[1]*32 and state['GOODAMP']==[1]*200 and state['stored_INIT_flags_are_learned'] is False,'Stored INIT profile is not learned state')
                policy=state['policy'];require(policy['LoopPoleMode']==0 and policy['BYPASS_CHECK']==0 and policy['ALWAYS_TEST_STABILITY']==1 and policy['derived_bypass_from_observed_policy_and_source']==1 and policy['MLStabThres']['word']=='3EB0C6F7A0B5ED8D','Actual V2 policy')
            require(len(lane['readiness']['states'])==5,'Five V2 snapshots required')
            for target in lane['values']['targets']:
                k=target['target'];require(target['before']==target['after'],'Caller P/MUR changed')
                require(target['environment_before']['IEEE']==target['environment_after']['IEEE']==[1]+[0]*10,'Observed IEEE changed')
                require(target['environment_before']['software']==[0,0] and target['environment_after']['software']==([0,0] if k==1 else [0,1]),'Software QP_FPE observation changed')
                for row in target['answers']:
                    require(row['status']==(219 if k==1 else 420),'Unexpected observed public status')
                    require(row['public']['accepted_public_status']==(k==1) and row['comparison_eligible']==(k==1) and row['buffers_are_diagnostics_only']==(k==2) and row['public']['precision']==('DP' if k==1 else None),'Refusal counted as accepted precision')
                    if k==2:require(row['accuracy']==1.0,'Refusal ACC changed')
        pair=F.compare(lanes['original'],lanes['candidate'],accepted_compare)
        require(pair==J('run2/candidate/comparison.json')==J('run2/candidate/result.json')['comparison'],'Conditional actual pair replay changed')
        require(pair['ordinary_control_pair']['passed'] and pair['learning_history_equal'] and pair['observed_readiness_equal'] and pair['target2_both_comparison_eligible'] is False and pair['target2_numerical_comparison'] is None and pair['target2_statuses']=={'original':[420,420],'candidate':[420,420]},'Refused buffers entered scientific comparison')
        diffs=[]
        for a,b in zip(lanes['original']['values']['targets'],lanes['candidate']['values']['targets'],strict=True):
            require(a['before']==b['before'] and a['after']==b['after'],'Cross-arm caller input differs')
            for x,y in zip(a['answers'],b['answers'],strict=True):
                for j,(v,w) in enumerate(zip(x['words'],y['words'],strict=True)):
                    if v!=w:diffs.append({'target':a['target'],'order':x['order'],'column':j,'original_word':v,'candidate_word':w,'both_status420':x['status']==y['status']==420})
        diagnostic=J('reviews/pair/diagnostic-pair-word-changes.json');require(diffs==diagnostic['word_differences'] and len(diffs)==4 and all(x['both_status420'] and x['column'] in (1,2) and x['target']==2 for x in diffs),'Failed-buffer diagnostic words differ')
        # Preserve and independently reproduce the historical strict predicate
        # rejection from its complete initial V1 snapshot; no missing RUN1 data.
        oldP=load_source('contained_old_protocol',data['history/source/run1/target_protocol.py'],'contained:old-protocol')
        oldF=functions_only('contained_old_initial',data['history/source/run1/collector.py'],('require','initial_readiness'),{'P':oldP})
        try:oldF.initial_readiness(T('history/run1/original/failure-target-readiness.dat'),DIM)
        except ValueError as error:require(str(error)=='Actual ordinary-only readiness not achieved','Historical refusal reason changed')
        else:raise ValueError('Old readiness predicate falsely accepted stored INIT')
        oldt=J('history/run1/original/terminal.json');require(oldt['status']=='failed' and oldt['returncode']==1 and oldt['cleanup']['survivors']==[] and 'history/run1/original/result.json' not in data,'Historical failure rewritten')
        require('IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) GOTO 800' in T('history/source/build1/target_driver.f'),'Original wrong readiness predicate missing')
        require(J('history/run1/original/first-failure.json')['retry'] is False,'Failed attempt retried')
        require(not effects,'Forbidden API attempted by contained parser')
    return {'verified':True,'actual_source_and_BUILD_replayed':True,'BUILD_recorded_commands':110,'BUILD_fresh_compiles':2,'BUILD_links':2,'BUILD_unique_LOAD_bindings_each':54,'runtime_providers_each':15,'actual_RUN_workers':2,'ordinary_initialization_calls_each':101,'ordinary_initialization_raw_rows_each':202,'ordinary_steady_rows_each':200,'accepted_control_DP219_points_each':1,'threshold_refused420_points_each':1,'accepted_QP_points_each':0,'failed_buffer_word_differences_diagnostic_only':4,'threshold_numerical_comparison':None,'QP_software_latch_each':[0,1],'observed_IEEE_unchanged':True,'V2_policy_snapshots_each':5,'stored_filters_learned':False,'caller_P_and_MUR_unchanged':True,'first_pre_READY_failure_preserved':True,'old_readiness_predicate_rejection_reproduced':True,'mass_fix_resolved_EW_stress':False,'actual_native_QP_mass_inputs_observed':False,'genuine_QP_physics_qualified':False,'independent_physical_truth':False,'IEEE_lowering_or_trap_freedom_qualified':False,'cache_qualified':False,'performance_acceptance':False,'source_promoted':False,'native_binaries_packaged':0,'new_native_executions':0,'forbidden_API_attempts':0}

def verify(root,expected_index_sha256=None):
    ix,c,data,physical=decode(root,expected_index_sha256);facts=semantic(data,c,physical)
    require(ix['claims']==facts,'Capsule claims exceed replayed evidence');return facts

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
