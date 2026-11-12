"""Standalone contained closed CMS cache RUN4 replay; no native execution."""
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



import ast
import copy
import math
import struct
from fractions import Fraction

SOURCE='fca076eeb821a457f13f971a3ae8bc0568022db7eb6622a7c645b08c527c9554'
COMPILED='6417ad4e1905765b3e2ae17fd67cc4aa01ac7dfe4b93ec75318c7558e02b5f93'
TERM='83034229e1377efc0d2ec2c6ab400065b11a0c2bc778aa5421c8e2970ae7a719'
RESULT='ee95d391ecf7dc9cf2a7b28cf8ed349e0869c58da22d384632c33f4d2fc41325'
CACHE='82b80686d5206c94c5a7ead314aa63b9756e7588961d135eb6a94998b766128d'
BT='4110ed1280ce47cf55a93fd45cc4b848b84906411b255a7c2f0a41090260eb44'
BR='cb9b8ff22f9ceacc60d66b908a060b90e4a6d3322d514955224c31b3ee993137'
PINS={
 'reviews/run4/receipt-final.json':'94aab9f8f817cc6f41f382cf0872bb42fbe0f70fdddfa6103914d84ad808f049',
 'reviews/run4/raw-inventory.json':'a37f52b048a7629e3733f92b6c12c02e624418485d9c823a2042bf6dad525f46',
 'source/run4/source-index.json':SOURCE,'source/compiled6417/source-index.json':COMPILED,
 'run4/terminal.json':TERM,'run4/result.json':RESULT,'run4/cache-observations.json':CACHE,
 'build/terminal.json':BT,'build/result.json':BR,
 'owner-readout.json':'e8685f52b02a8d62f094d498c9c435670323d9952beeedf3fd1e9c31d045f20a',
 'retention-plan/source-index.json':'44774f2f5baa89a652fd17a8fdd23f2a6d30812a77de5d2142d8352dc6c17232',
 'root-BUILD-gate.json':'c379a5e6f61d5464c786b327c3e99b2f0101a729b22a6ae5849957542dbdfbc5',
 'reviews/build/receipt.json':'c88fe3bb30543c01c77d5d6d4b4af3b82a14944a5814b39d790a26522e4c1238',
 'reviews/source4/receipt.json':'26b132ef9663818ef0bd3e82c30d0804402d0f53da6c7b518d295fb3313a5821',
 'reviews/blocked-source1/receipt.json':'d6f6e489f532b2b5951924ec2ac9365b8818a4aeaa3fdecd81fd47ca1219e87c',
 'reviews/failure2/receipt.json':'7c1737a8de86258e5fb3f8d211f3f5f392f3009e0c3398feca139ec7bc0b19e8',
 'reviews/failure3/receipt.json':'e1382577f5865b91b357f044ff9427b9bc68b7964933806d42d80bb2ec650316',
 'reviews/eb3-source-artifact/receipt.json':'a94d65a6b62cdeab9ea92d09a5b80434736ddf65570e24e7e299ca71eb2282d8',
 'history/run2/terminal.json':'d48625b5591ec24bd01d6596d9910db83b507d5373b70d07685fd1c89630867f',
 'history/run3/terminal.json':'02b53e803756f53e356326068895145b29be640696bc9dbbc70bfb42b9d87cdc',
 'prior-plain/terminal.json':'182714a03f54cea2ea344e92bb31922a86cb603615176fb5aef8b060eda041bc',
 'prior-plain/result.json':'b0a3eb3596a8c183775d37a1a791a2aee3e3ca671958f8e2884776d79cb042eb',
 'reviews/prior-plain-run/receipt.json':'efb558d49c9050c33c7755915a689db699178d0f6fdf68183298a2cbe1c4b692'}
SUB='SubProcesses/P0_epem_mupmum'

def wire(value):return json.loads(json.dumps(value))

def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}
def select_functions(name,raw,names,globals):
    tree=ast.parse(raw,filename=name);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Missing exact selected pure functions')
    obj=types.ModuleType(name);obj.__dict__.update(globals)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),obj.__dict__);return obj

def decode(root,expected_index_sha256=None):
    root=Path(root).resolve();raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(raw)==expected_index_sha256,'Capsule seal changed')
    ix=json.loads(raw);require(ix['schema']=='CMS-cache-RUN4-contained-portable-v1','Wrong portable capsule schema')
    actual=set()
    for f in root.rglob('*'):
        require(not f.is_symlink(),'Capsule symlink refused')
        if f.is_file():actual.add(str(f.relative_to(root)))
    require(actual==set(ix['files'])|{'index.json'},'Exact physical inventory changed')
    physical={}
    for name,pin in ix['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe capsule member')
        b=(root/name).read_bytes();require(identity(b)==pin,'Changed physical member '+name);require(not b.startswith(b'\x7fELF'),'Native ELF packaged');physical[name]=b
    c=json.loads(physical['collection.json']);data={}
    for name,pin in c['members'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical member')
        b=physical[pin['path']];require(b[:3]==b'\x1f\x8b\x08' and b[4:8]==b'\0'*4,'Non-deterministic gzip')
        b=gzip.decompress(b);b.decode('utf8');require(identity(b)=={k:pin[k] for k in ('bytes','sha256')},'Logical member changed');require(not b.startswith(b'\x7fELF'),'Decoded native binary');data[name]=b
    return ix,c,data,physical

def modules(data,namespace='source/run4'):
    out={}
    for name in ('normal_run_analysis','observer_analysis','cache_comparison','cache_metrics','protocol_parity','accuracy_policy','run_plan','cache_run'):
        if namespace+'/'+name+'.py' in data:out[name]=load_source('owned_'+namespace.replace('/','_')+'_'+name,data[namespace+'/'+name+'.py'],'contained:'+namespace+'/'+name+'.py')
    out['cache_worker']=select_functions('owned_point_words',data[namespace+'/cache_worker.py'],('require','point_words','validate_policy','validate_runtime_card'),{'Path':Path,'struct':struct,'hashlib':hashlib,'SUB':Path(SUB),'ORIGINAL_CARD_SHA':'8b8f556a4d52f570a4f80d81d105bb1b4636badd232f097ec8afe19a923ef969'})
    return out

def normal_view(a):return a.get('normal_analysis',a)

def authenticate_maps(worker,terminal,runtime):
    pid=worker['pid'];identity_keys=('pid','parent_pid','process_group','session','start_time_ticks')
    observed={r['pid']:r for r in terminal['cleanup']['observed_owned_processes']}
    expected={**runtime,worker['executable_path']:worker['executable_sha256']}
    require(len(expected)==16,'Exact15 providers plus worker ELF required')
    for label,m in worker['maps'].items():
        require(m['pid']==pid and m['affinity']==[12] and m['exclusive_CPU'] is False and m['ELF']==expected,'Map CPU/PID/provider set changed')
        require(all(m['identity'][k]==m['started_identity'][k] for k in identity_keys),'Map lifetime mismatch')
        require(all(m['identity'][k]==observed[pid][k] for k in identity_keys),'Map/cleanup owned identity mismatch')
        require(m['identity']['parent_pid']==terminal['child_pid']==m['identity']['process_group']==m['identity']['session'],'Map outside owned supervisor group')
        paths={line.split(None,5)[5] for line in m['raw'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
        require(paths==set(expected) and all(not n.endswith(' (deleted)') for n in paths),'Raw map file closure mismatch')
    return True

def replay_workers(data,physical):
    J=lambda n:json.loads(data[n]);T=lambda n:data[n].decode('utf8')
    br=J('build/result.json');r=J('run4/result.json');terminal=J('run4/terminal.json');p=J('source/run4/input-pins.json');policies=J('source/run4/runtime-policies.json')['policies']
    b_sources=json.loads(physical['compiled-source-bindings.json']);runtime_inputs=json.loads(physical['runtime-input-bindings.json'])
    memory={n:b for n,b in data.items()}
    # Alias only actual compiled source bytes to local owned member names. Native
    # path labels are checked separately and never opened by parser replay.
    with blocked_apis(memory) as effects:
        M=modules(data);requests=M['run_plan'].requests();M['run_plan'].validate_requests(requests)
        R=M['cache_run'];N=M['normal_run_analysis'];A=M['accuracy_policy'];C=M['cache_comparison'];P=M['protocol_parity'];W=M['cache_worker'];metrics=M['cache_metrics']
        prior=J('prior-plain/result.json');references={}
        for cohort,ids in (('controls',[1,7,12]),('critical20',[20]),('critical21',[21])):
            row=prior['requests']['candidate-'+cohort];raw_name='prior-plain/evidence/candidate-'+cohort+'/driver.raw64'
            require(sha(data[raw_name])==row['raw_sha256'],'Historical plain raw hash')
            points=W.point_words('source/run4/inputs/training.dat','source/run4/inputs/'+cohort+'.dat')
            parsed=N.parse_raw(T(raw_name),lane='candidate',request_ids=ids,expected_points=points)
            require(parsed['pid']==row['pid'] and parsed['accepted_for_continuation'],'Historical reference not accepted')
            exact=C.compare_exact_histories(row['analysis'],parsed,label='closed original cache0 plaintext replay');require(exact['passed'],'Historical plain parse differs');A.enforce(parsed,word_value=N._value)
            references[cohort]={'raw_path':raw_name,'raw_sha256':row['raw_sha256'],'analysis':parsed,'pid':row['pid']}
        replays={};own={};policy_comparisons={};prefixes=0;normal_records=0;veto=0;frames=0;genuine=0;dumped={}
        for index,request in enumerate(requests):
            ident=request['id'];worker=J('run4/validated-'+ident+'.json');arm=request['arm'];policy=policies[request['policy']];prefix='run4/evidence/'+ident+'/'
            require(worker['request']==request and worker['returncode']==0 and worker['orchestration_closed'] and worker['no_retry'],'Closed worker contract')
            require(worker['executable_sha256']==br['variants'][arm]['sha256'],'Actual compiled worker identity')
            require(sha(data[prefix+'driver.raw64'])==worker['raw_sha256']==r['worker_results'][ident]['raw_sha256'],'Raw fingerprint')
            require(identity(data['run4/validated-'+ident+'.json'])['sha256']==r['worker_results'][ident]['validated_sha256'],'Validated report binding')
            require(sha(data[prefix+'result.json'])==worker['worker_result_sha256'],'Native result binding')
            require(worker['analysis']==J(prefix+'result.json')['analysis'] and worker['protocol']==J(prefix+'protocol.json'),'Worker raw/protocol attachment')
            command=J(prefix+'command.json');started=J(prefix+'started.json')
            require(command['source_index_sha256']==SOURCE and command['compiled_source_index_sha256']==COMPILED and command['compiled_result_sha256']==BR and command['reviewed_build']==BT and command['ELF_sha256']==worker['executable_sha256'] and command['request']==request and not command['new_session'],'Actual command/source join')
            require(started['pid']==worker['pid'] and started['source_executable_sha256']==worker['executable_sha256'],'Actual READY process binding')
            require(command['argv']==[worker['executable_path'],command['cwd']+'/training.dat',command['cwd']+'/measurement.dat',command['cwd']+'/native-protocol.fifo',str(Path(worker['raw_path']))] and command['cwd']==str(Path(worker['executable_path']).parent),'Actual worker argv/path coordinates')
            authenticate_maps(worker,terminal,p['runtime'])
            for label,m in worker['maps'].items():require(m==J(prefix+'maps-'+label+'.json'),'Attached maps differ')
            require(sha(data[prefix+'stderr.txt'])==worker['stderr_sha256'],'Full stderr changed')
            W.validate_policy(policy,request);W.validate_runtime_card(data['source/run4/cards/original-MadLoopParams.dat'],data['source/run4/'+policy['runtime_card']],policy)
            require(command['runtime_card_sha256']==policy['runtime_sha256'],'Actual runtime numerical card')
            for n,h in J('reviews/run4/runtime-input-bindings.json')['worker_inputs'][ident].items():require(sha(data['runtime/'+ident+'/'+n])==h,'Independent actual staged input binding')
            for n,pin in runtime_inputs[ident]['files'].items():
                raw=data['runtime/'+ident+'/'+n];require(identity(raw)==pin,'Staged runtime input changed')
                if n.endswith('MadLoopParams.dat'):require(sha(raw)==policy['runtime_sha256'],'Staged numerical card policy')
                elif n.endswith('param_card.dat'):require(raw==data['source/run4/cards/param_card.dat'],'Staged physical model card')
                elif n.endswith('training.dat'):require(raw==data['source/run4/inputs/training.dat'],'Staged learning stream')
                elif n.endswith('measurement.dat'):require(raw==data['source/run4/'+request['measurement']],'Staged physical request stream')
                elif Path(n).name in {Path(v).name for v in p['resources']}:
                    expected=next(pin for k,pin in p['resources'].items() if Path(k).name==Path(n).name);require(identity(raw)==expected,'Staged resource/ident identity')
            reference=(references[request['cohort']] if request['policy']=='cache0' else None) if arm=='plain' else replays[request['policy']+'-plain-'+request['cohort']]
            if reference is not None and arm=='observed':reference={**reference,'raw_path':'run4/evidence/'+reference['request']['id']+'/driver.raw64'}
            b=types.SimpleNamespace(HERE=Path('source/run4'),CACHE_RUN_STAGE=Path('run4'),local=lambda name:M[name],dump=lambda path,value:dumped.__setitem__(str(path),value))
            row={**br['variants'][arm],'path':'compiled-actual/'+arm+'/check'}
            adapter=R.AnalysisAdapter(b,request,row,policy,reference);adapter.bind(worker['pid'])
            points=W.point_words('source/run4/inputs/training.dat','source/run4/'+request['measurement']);raw=T(prefix+'driver.raw64')
            last=None
            for count in range(len(request['ids'])+1):
                saved=J(prefix+('learning-analysis.json' if count==0 else 'measurement-'+str(count)+'.json'))
                # callback receipts contain the analysis itself, not an aggregate.
                size=saved['raw_bytes'];piece=raw[:size];require(sha(piece.encode())==saved['raw_sha256'],'Actual flushed prefix boundary')
                last=adapter.parse_raw(piece,lane='candidate',request_ids=request['ids'][:count],expected_points=points)
                require(wire(last)==saved,'Actual frozen adapter prefix replay differs '+ident+' '+str(count)+' '+str(C.differences(wire(last),saved)[:2]));prefixes+=1
            require(adapter.finish(last) and wire(last)==worker['analysis'],'Final complete worker analysis differs')
            replays[ident]={**worker,'analysis':last,'raw_path':prefix+'driver.raw64'}
            normal=R.normal_view(last);require(normal['learning']['outer_calls']==16 and P.learning_coordinate(normal['learning'])==15 and normal['learning']['streak']==12,'Bootstrap0/lastK15/streak12 conflation')
            normal_records+=len(normal['learning']['records'])+len(normal['measurements'])
            P.protocol_core(worker['protocol'],pid=worker['pid'],ids=request['ids'],measurements=normal['measurements'],learning=normal['learning'])
            require(all(m['code']==(219 if request['cohort']=='controls' else 323) for m in normal['measurements']),'Observed measurement status/rescue changed')
            if arm=='observed':
                plain=replays[request['policy']+'-plain-'+request['cohort']]
                exact=C.compare_own_parity(R.normal_view(plain['analysis']),normal);runtime=P.own_runtime_parity(plain,replays[ident],runtime=p['runtime']);own[ident]={'normal':exact,'protocol_maps':runtime}
                require(own[ident]==J('run4/own-parity-'+ident+'.json')==r['own_plain_observed_parity'][ident] and exact['passed'] and runtime['passed'],'Six actual own parity gates')
                journals=last['journals'];veto+=sum(j['observed_veto_count'] for j in journals);frames+=sum(len(j['frames']) for j in journals);genuine+=sum(f.get('observed_genuine_QP') is True for j in journals for f in j['frames'])
            if request['policy']=='cacheall':
                zero=replays['cache0-'+arm+'-'+request['cohort']];comparison=C.compare_policy_numerics(R.normal_view(zero['analysis']),normal,word_value=N._value,ratio=N._ratio,journals0=zero['analysis'].get('journals'),journals_all=last.get('journals'));policy_comparisons[ident]=comparison
                require(comparison==J('run4/policy-comparison-'+ident+'.json')==r['cache_policy_numerical_comparisons'][ident] and comparison['passed'],'Six actual cache-policy comparisons')
            before=J('run4/progress-'+ident+'-starting.json');after=J('run4/progress-'+ident+'-closed.json');priorids=[q['id'] for q in requests[:index]]
            require(before=={'request':ident,'state':'starting','completed':priorids,'compiled_source_index_sha256':COMPILED,'RUN_source_index_sha256':SOURCE} and after=={**before,'state':'closed','completed':priorids+[ident]},'Immutable ordered unique progress receipt')
        for name,value in dumped.items():require(wire(value)==J(name),'Actual cache0 reference-prefix gate differs')
        samples=[{'worker_id':key,'arm':v['request']['arm'],'policy_label':v['request']['policy'],'policy':policies[v['request']['policy']]['GlobalCache'],'budget':268435456,'analysis':v['analysis']} for key,v in replays.items()]
        cache=metrics.summarize_cache_observations(samples);require(wire(cache)==J('run4/cache-observations.json'),'Recorded failure-censored cache output differs')
        require(not effects,'Forbidden pure parser API effect')
    return {'worker_replays':replays,'own':own,'policy_comparisons':policy_comparisons,'cache':cache,'prefixes':prefixes,'normal_records':normal_records,'frames':frames,'genuine':genuine,'veto':veto,'forbidden_API_attempts':len(effects)}


def replay_history(data,physical):
    J=lambda n:json.loads(data[n]);T=lambda n:data[n].decode('utf8');memory=dict(data)
    with blocked_apis(memory) as effects:
        M=modules(data);N=M['normal_run_analysis'];W=M['cache_worker'];C=M['cache_comparison'];br=J('build/result.json');prior=J('prior-plain/result.json');requests=M['run_plan'].requests()
        reference_row=prior['requests']['candidate-controls'];reference={'raw_path':'prior-plain/evidence/candidate-controls/driver.raw64','raw_sha256':reference_row['raw_sha256'],'analysis':reference_row['analysis'],'pid':reference_row['pid']}
        blocked=modules(data,'history/source1');dumped={};b=types.SimpleNamespace(HERE=Path('history/source1'),CACHE_RUN_STAGE=Path('history/blocked'),local=lambda name:blocked[name],dump=lambda path,value:dumped.__setitem__(str(path),value))
        row={**br['variants']['plain'],'path':'compiled-actual/plain/check'};request=requests[0];policy=J('source/run4/runtime-policies.json')['policies']['cache0'];adapter=blocked['cache_run'].AnalysisAdapter(b,request,row,policy,reference);adapter.bind(reference['pid'])
        prefix=blocked['cache_run'].reference_prefixes(T(reference['raw_path']),request['ids'])[0];points=blocked['cache_worker'].point_words('history/source1/inputs/training.dat','history/source1/inputs/controls.dat')
        try:adapter.parse_raw(prefix,lane='candidate',request_ids=[],expected_points=points)
        except blocked['cache_run'].PrefixRefusal:pass
        else:raise ValueError('Blocked source1 falsely accepted a partial history')
        require(J('reviews/blocked-source1/receipt.json')['native_or_model_execution'] is False if 'native_or_model_execution' in J('reviews/blocked-source1/receipt.json') else True,'Blocked source1 reported native execution')
        # Exact old dump() function with a memory-only exclusive-create Path.
        # Execute the two actual source2 dump call expressions, preserving its
        # FileExistsError without writing a real filesystem or running a worker.
        written={}
        class Stream(io.StringIO):
            def __init__(self,key):super().__init__();self.key=key
            def __exit__(self,*unused):written[self.key]=self.getvalue();self.close()
        class VirtualPath:
            def __init__(self,key):self.key=str(key)
            def open(self,mode):
                require(mode=='x','Historical writer stopped using exclusive creation')
                if self.key in written:raise FileExistsError(self.key)
                written[self.key]='';return Stream(self.key)
        writer=select_functions('contained_actual_exclusive_writer',data['history/exclusive-writer-controller.py'],('dump',),{'Path':VirtualPath,'json':json})
        oldtree=ast.parse(data['history/source2/cache_run.py']);calls=[node for node in ast.walk(oldtree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='dump' and node.args and isinstance(node.args[0],ast.BinOp) and isinstance(node.args[0].right,ast.Constant) and node.args[0].right.value=='progress.json'];calls.sort(key=lambda n:n.lineno);require(len(calls)==2,'Actual source2 duplicate-write call sites changed')
        env={'b':types.SimpleNamespace(dump=writer.dump,EXPECTED_SOURCE=J('history/source2/source-index.json') and sha(data['history/source2/source-index.json'])),'directory':Path('memory:history2'),'request':request,'results':{},'plan':M['run_plan']}
        exec(compile(ast.Expression(calls[0]),'contained:source2-starting','eval'),env)
        env['results']={request['id']:J('history/run2/validated-'+request['id']+'.json')}
        try:exec(compile(ast.Expression(calls[1]),'contained:source2-closed','eval'),env)
        except FileExistsError:pass
        else:raise ValueError('Source2 duplicate progress write falsely accepted')
        require(json.loads(written['memory:history2/progress.json'])==J('history/run2/progress.json'),'Historical first progress contents differ')
        require('FileExistsError' in T('history/run2/console.log'),'Actual RUN2 exclusive-writer exception missing')
        oldprotocol=load_source('contained_failed_source3_protocol',data['history/source3/protocol_parity.py'],'contained:old-protocol3');failed3=J('reviews/failure3/receipt.json');failed_workers=failed3['attempted_workers'];require(len(failed_workers)==4,'Actual four attempted RUN3 workers required')
        refused=0
        for ident in failed_workers:
            prefix='history/run3/evidence/'+ident+'/';actual=J(prefix+'result.json');request=actual['request'];expected=W.point_words('source/run4/inputs/training.dat','source/run4/'+request['measurement']);raw=T(prefix+'driver.raw64')
            if request['arm']=='observed':
                plain=J('history/run3/evidence/cache0-plain-controls/result.json');parsed=M['observer_analysis'].analyze_observed(raw,T('history/run3/evidence/cache0-plain-controls/driver.raw64'),lane='candidate',request_ids=request['ids'],observed_pid=actual['pid'],plain_pid=plain['pid'],parse_normal=N.parse_raw,expected_points=expected,compiled_dispatch_source=data['compiled-actual/observed/CT_interface.f'],cache_policy=0,cache_budget=268435456)
            else:parsed=N.parse_raw(raw,lane='candidate',request_ids=request['ids'],expected_points=expected)
            normal=normal_view(parsed);require(C.compare_exact_histories(normal_view(actual['analysis']),normal,label='preserved failed-stage worker only')['passed'],'Failed-stage raw parse changed')
            require(M['protocol_parity'].learning_coordinate(normal['learning'])==15 and normal['learning']['outer_calls']==16,'Historical normal counter changed')
            transcript=J(prefix+'protocol.json')
            try:oldprotocol.protocol_core(transcript,pid=actual['pid'],ids=request['ids'],measurements=normal['measurements'],learning_outer_calls=16)
            except ValueError as error:require(str(error)=='protocol message/coordinate mismatch','Historical RUN3 refusal changed');refused+=1
            else:raise ValueError('Old bootstrap-inclusive protocol counter falsely accepted')
            # Explicit source diagnosis only, never an adopted RUN3 success.
            oldprotocol.protocol_core(transcript,pid=actual['pid'],ids=request['ids'],measurements=normal['measurements'],learning_outer_calls=15)
        for version in (2,3):
            prefix='history/run'+str(version)+'/'
            t=J(prefix+'terminal.json');require(t['status']=='failed' and t['returncode']==1 and not t['cleanup']['survivors'] and prefix+'result.json' not in data,'Failed stage adopted/promoted')
            inventory=J('reviews/failure'+str(version)+'/closed-relevant-files.json')
            entries=inventory.get('files',inventory)
            # The original audit may key by documentary root paths. All retained
            # relevant report values must join an owned member without I/O.
            for name,pin in entries.items():
                suffix=name.split('/run/',1)[-1] if '/run/' in name else name
                logical=prefix+suffix
                require(logical in data and identity(data[logical])==pin,'Failed-stage exact relevant record changed '+name)
        require(not effects,'Historical pure replay attempted an effect')
    return {'blocked_source1_partial_prefix_refused':True,'failed_RUN2_exclusive_progress_refused':True,'failed_RUN3_actual_protocols_refused':refused,'historical_partial_workers_adopted':False,'forbidden_API_attempts':len(effects)}


def cache_facts(cache):
    require(cache['hit_rates'] is None and cache['lifetime_totals'] is None and cache['initial_creation_observed'] is False and cache['failed_interval_hit_counts_censored'] is True and cache['Fortran_group_cache']=='N/A','Counters promoted to lifetime rates/Fortran cache coverage')
    intervals=cache['intervals'];lifecycle=cache['lifecycle_events'];require(len(intervals)==728 and len(lifecycle)==752 and len(cache['censored_intervals'])==60 and cache['successful_visible_interval_count']==692,'Visible/failure/reset interval accounting changed')
    failures=[e for e in intervals if e['status']!=0];require(len(failures)==36 and all(e['width']==64 and e['successful_visible_delta'] is None and e['censored_for_work_metrics'] for e in failures),'Failed native intervals counted as successful')
    clears=[e for e in lifecycle if e.get('lifecycle')=='explicit_all15_clear'];require(len(clears)==24 and all(e['after15']==[0]*15 for e in clears),'Explicit clear evidence changed')
    hits=[e for e in intervals if e['successful_visible_delta'] and e['successful_visible_delta']['QP6'][0]>0]
    require(len(hits)==2 and all(e['group']==19 and e['n']==1 and e['rank']==0 and e['width']==128 and e['mode']==4 and e['provenance']['actual_consumer_MP_genuine'] is True and e['provenance']['group3_frame_index'] is None and e['provenance']['observed_genuine_QP'] is None for e in hits),'QP hits promoted beyond actual consumer-only group19 context')
    checks=[e for e in intervals if e['cache_enabled'] and e['check_pass'] and e['status']==0 and e['before15'][6]>0];require(len(checks)==141,'Actual populated independent check count changed')
    require(cache['successful_visible_counter_sums']['DP_component9'][2]==701 and cache['successful_visible_counter_sums']['QP6'][0]==2,'Visible reuse counts changed')
    require(not cache['performance_qualified'] and not cache['independent_truth'] and not cache['full_caller_qualified'],'Counter evidence overqualified')
    return {'native_intervals':len(intervals),'successful_visible_intervals':692,'failed_DP_intervals_censored':len(failures),'explicit_clears':len(clears),'lifecycle_records':len(lifecycle),'DP_visible_master_reuses':701,'QP_visible_tensor_hits':2,'populated_independent_checks':len(checks),'hit_rates':None,'lifetime_work_totals':None,'initial_context_creation_observed':False,'QP_hits_scope':'group19 N1 rank0 actual consumer context only; no per-native owned frame'}

def semantic(data,collection,physical,*,require_closed_review=True):
    J=lambda n:json.loads(data[n]);T=lambda n:data[n].decode('utf8')
    for n,wanted in PINS.items():require(sha(data[n])==wanted,'Closed identity changed '+n)
    for ns,spec in collection['source_namespaces'].items():
        index=J(ns+'/source-index.json');require(sha(data[ns+'/source-index.json'])==spec['index_sha256'],'Source index drift')
        names={n[len(ns)+1:] for n in data if n.startswith(ns+'/')}-{'source-index.json'};require(names==set(spec['retained_payloads']),'Source namespace drift')
        if spec['complete_namespace']:require(names==set(index['files']),'Complete namespace omitted source payload')
        for n in names:require(identity(data[ns+'/'+n])==index['files'][n],'Frozen source member drift')
    br=J('build/result.json');bt=J('build/terminal.json');r=J('run4/result.json');t=J('run4/terminal.json')
    require(br['status']=='passed' and br['source_index_sha256']==COMPILED and br['native_main_calls']==br['model_calls']==br['reducer_calls']==0 and not br['runtime_cache_behavior_qualified'],'BUILD scope/source')
    require(bt['status']=='passed' and bt['returncode']==0 and bt['result_sha256']==BR and bt['cleanup']['survivors']==[],'BUILD closure')
    inventory=J('reviews/build/raw-command-inventory.json');require(len(inventory)==411,'All137 BUILD command triples required')
    for n,pin in inventory.items():require(identity(data['build/'+n])==pin,'Actual BUILD command record drift')
    for n in inventory:
        if n.endswith('.command.json'):
            prefix='build/'+n[:-len('.command.json')];terminal=J(prefix+'.terminal.json');require(terminal['returncode']==0 and terminal['stdout_sha256']==sha(data[prefix+'.stdout.txt']),'Actual BUILD command failure')
    bindings=json.loads(physical['compiled-source-bindings.json']);excluded=json.loads(physical['excluded-native-identities.json'])
    for arm,v in br['variants'].items():
        require(bindings[arm]['source_inventory']==v['source_inventory'],'Full original source inventory drift')
        for n in bindings[arm]['retained_sources']:require(identity(data['compiled-actual/'+arm+'/'+n])==v['source_inventory'][SUB+'/'+n],'Actual compiled source bytes differ')
        require(v['link_map_sha256']==sha(data['build-identity/'+arm+'/owned-link.map']) and v['prelink_sha256']==sha(data['build-identity/'+arm+'/owned-prelink.json']) and v['static_runtime_sha256']==sha(data['build-identity/'+arm+'/owned-static-runtime.json']),'Actual link/prebound/static identity drift')
        require(excluded['compiled/'+arm]['objects']==v['objects'] and excluded['compiled/'+arm]['module_files']==v['module_files'] and excluded['compiled/'+arm]['actual_link_loads']==v['actual_link_loads'] and not excluded['compiled/'+arm]['binary_contents_retained'],'Excluded native identity drift')
        require(v['link_command'][1:4]==['-w','-fPIC','-ffixed-line-length-132'],'Original flags changed')
    require(data['compiled-actual/observed/CT_interface.f']==data['source/compiled6417/emittedsrc/observed/CT_interface.f'],'Compiled source/parser CT byte join')
    require(t['status']=='passed' and t['returncode']==0 and t['result_sha256']==RESULT and t['cleanup']['survivors']==[] and t['cleanup']['events']==[],'RUN4 not cleanly closed')
    require(t['cap_seconds']==1800 and t['cleanup_reserve_seconds']==60 and t['deadline_monotonic']==t['original_end_monotonic']==t['original_start_monotonic']+1800,'Absolute all-in RUN4 cap')
    require(r['status']=='passed' and r['RUN_source_index_sha256']==SOURCE and r['compiled_source_index_sha256']==COMPILED and r['compiled_result_sha256']==BR and r['reviewed_BUILD_terminal_sha256']==BT and r['cache_observations_sha256']==CACHE and r['native_processes']==12,'RUN4 source/compiled/result identity')
    for key in ('compiler_calls','loader_queries'):require(r[key]==0,'RUN scope became a new build/query')
    for key in ('Fortran_group_cache_applicable','live_error_API_state_compared','unobserved_software_state_equivalence','independent_physical_truth','full_caller_qualified','performance_qualified','production_dispatch_enabled'):require(r[key] is False,'Observation promoted beyond scope')
    result=replay_workers(data,physical)
    history=replay_history(data,physical);counters=cache_facts(wire(result['cache']))
    review=J('reviews/run4/receipt-final.json');require(review['status']=='PASS-limited-declared-normal-cache-qualification' and not review['blockers'] and review['runtime_cache_qualified_in_declared_scope'] is True,'Final independent RUN4 receipt is not closed/clear')
    require(review['terminal_sha256']==TERM and review['result_sha256']==RESULT and review['RUN_source_index_sha256']==SOURCE and review['compiled_source_index_sha256']==COMPILED and review['compiled_BUILD_terminal_sha256']==BT and review['cache_observations_sha256']==CACHE,'Final independent closed chain mismatch')
    require(all(review[k] is False for k in ('general_cache_lifecycle_qualified','independent_physical_truth','performance_qualified','production_promoted')),'Final independent review overqualified')
    require(sha(data['reviews/run4/receipt.json'])==review['label_only_successor_of']['sha256'],'Original label-only receipt history changed')
    for name,pin in review['component_receipts'].items():
        relative=pin['path'].split('generic-owned-CMS-cache-run-independent-review-4/',1)[1];require(sha(data['reviews/run4/'+relative])==pin['sha256'],'Independent component review drift')
    raw_inventory=J('reviews/run4/raw-inventory.json');owned={n[len('run4/'):] for n in data if n.startswith('run4/')};require(owned==set(raw_inventory['files']) and len(owned)==212,'Exact all212 current RUN4 report members required')
    for n,pin in raw_inventory['files'].items():require(identity(data['run4/'+n])==pin,'Current RUN4 record differs from independent inventory')
    require(sha(data['history/exclusive-writer-controller.py'])=='bf281d5f97e6d442da8740d21954ceac1cf817314063f7bcfe17a6c6fe9b3ca9','Qualified historical exclusive writer source drift')
    require(J('source/run4/source-index.json')['blocked_parent_executed'] is False and J('source/run4/source-index.json')['failed_parent_worker_not_adopted'] is True,'Blocked/failed source history adopted')
    require(len(result['worker_replays'])==12 and len(result['own'])==len(result['policy_comparisons'])==6 and result['prefixes']==32 and result['normal_records']==212 and result['frames']==28 and result['genuine']==8 and result['veto']==32,'Observed typed worker/frame counts differ')
    return {'verified':True,'RUN4_independent_receipt_closed':True,'RUN4_source_index_sha256':SOURCE,'compiled_source_index_sha256':COMPILED,'actual_BUILD_command_triples':137,'native_binaries_packaged':0,'new_native_executions':0,'actual_RUN_workers':12,'normal_learning_records':192,'measured_outer_records':20,'prefix_callbacks':32,'actual_last_normal_K':15,'bootstrap_index':0,'learning_streak':12,'own_plain_observed_parities':6,'cache_policy_comparisons':6,'comparison_tolerance':3e-6,'measured_DP219_records':12,'measured_QP323_records':8,'group3_frames':28,'observed_genuine_QP_frames':8,'recorded_veto_decisions':32,'normal_returned_state_fields':15,'all_observed_own_words_states_exact':True,'counter_observations':counters,'failure_history':history,'runtime_cache_qualified_in_declared_scope':True,'component_metrics_runtime_qualified_flag_preserved':False,'budget_partition_bounds_qualified':True,'budget_exhaustion_qualified':False,'Fortran_group_cache':'N/A; NSQSO1 export','LoopPoleMode':-1,'finite':True,'strict_recorded_ACC_threshold':1e-6,'independent_physical_truth':False,'general_cache_lifecycle_qualified':False,'full_caller_or_numerator_lifetime_qualified':False,'unobserved_software_or_error_API_state_equivalence':False,'initial_context_creation_observed':False,'eb3_source_artifact_provenance_retained':True,'clean_rebuild_attestation':False,'performance_qualified':False,'production_promoted':False,'forbidden_API_attempts':0}


def verify(root,expected_index_sha256=None):
    ix,c,data,physical=decode(root,expected_index_sha256);facts=semantic(data,c,physical)
    require(ix['claims']==facts,'Portable claims exceed replayed evidence');return facts

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
