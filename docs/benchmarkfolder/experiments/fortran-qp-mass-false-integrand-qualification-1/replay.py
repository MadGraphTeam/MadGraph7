"""Portable text-only False/False replay. Historical controllers never execute."""
import argparse
import ast
import builtins
from contextlib import contextmanager
import ctypes
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import types
sys.dont_write_bytecode=True
SOURCES={'1':'eab42489ae3fb3b0b4bd9b51b5ad4e83607a62a4fa09323a3b156aac8bc26a79','2':'8feb2de2034c96c10951b9840c68ae6b72953d52ddd7dac5f4e0812f5c5c7eba','3':'05a51c4fc579c3977ad3ddae7e69044160ca70fd4012e0c56e998cb323d6e1aa'}
PARENTS={'current':'893c51f75dedbb6bbb1a70f78bed84b00d97b958ac678c194dc28cbb3aaa6b0b','fresh':'c5d0025ca581a466535db6d8196e1599f245bc4b45777c20c0d601781773751a'}
TERMINALS={'original':'6108741ded610fef1a2e46f1c23ae48e680b4b7522c5a9d4cd0d3cabd5a1a4fe','candidate':'b90a8bb1d1f621b01241658051aacccd9aa898a0d4821625d97666c6710d17ce'}
RESULTS={'original':'798c440b67334375c135482b6ce5327bcaaac90a406af30dd6bb0160c59f77c0','candidate':'4b98893973e8c956458d40aff1280f2a49bf50e6897ff583b5c28e516b9d8e3f'}
FAILED='4038bbb4b9281d66dae098bdb361b09deaa3574ef70b3353532cc19feec62072'
COMPARE='e381bc20682bc5972dbc264d63dbb7b2da727469bf19182dec4907850c956eb0'
PAIR_REVIEW='389e27899edee0d5fb2f3fbdeab0e6faa870498dbcb2a04429692a480d8beee6'
LIMITS={'address_space_bytes':17179869184,'file_bytes':536870912,'core_bytes':0}
SCOPE={'model':'loop_sm','process':'u u~ > z g g [virt=QCD]','UseQPIntegrandForTIRust':False,'CTModeRun':-1,'forced_independent_stability':True,'MLStabThres':1e-6,'TIRustComputeFinite':True,'LoopPoleMode':0,'DP_cache':0,'QP_cache':0,'physical_points_per_final_traversal':100,'returned_point_order_slots_per_final_traversal':200,'returned_coefficient_slots_per_final_traversal':800,'accuracy_word_slots_per_final_traversal':200,'orders_are_overall_and_sole_split_order':True,'independent_finite_truth_claimed':False,'internal_copied_coefficient_observation_claimed':False,'internal_native_mass_or_entry_observation_claimed':False,'performance_acceptance':False,'production_promotion':False}
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
@contextmanager
def blocked_apis():
    saved=[];attempts=[]
    def deny(*args,**kwargs):attempts.append(repr(args[:1]));raise RuntimeError('Forbidden effect in portable replay')
    def patch(obj,name):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,deny)
    try:
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob','read_bytes','read_text','write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):patch(Path,name)
        patch(builtins,'open');patch(io,'open');patch(io,'open_code');patch(io,'FileIO')
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp','execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty','mkdir','makedirs','unlink','remove','rename','replace','rmdir','chmod','chown','truncate','utime','link','symlink','kill','killpg','read','readv','pread','write','writev','pwrite'):patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        for name in ('CDLL','PyDLL','_dlopen'):patch(ctypes,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        for name in ('socket','create_connection'):patch(socket,name)
        for name in ('signal','setitimer','alarm'):patch(signal,name)
        patch(resource,'setrlimit');patch(fcntl,'flock')
        for name in ('copyfile','copy','copy2','copytree','move','rmtree'):patch(shutil,name)
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)
def preload(root,expected_index=None,draft=False):
    root=Path(root).resolve();raw=(root/'index.json').read_bytes()
    if expected_index:require(sha(raw)==expected_index,'Capsule index seal changed')
    index=json.loads(raw);require(index['schema']=='false-integrand-portable-qualification-v1','Wrong capsule schema')
    require(index['sealed'] is True or draft,'Capsule is not sealed')
    actual=set();physical={}
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Capsule alias forbidden')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Exact capsule inventory changed')
    for name,row in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe physical member')
        data=(root/name).read_bytes();require(sha(data)==row['sha256'] and len(data)==row['bytes'],'Physical payload changed')
        require(not data.startswith((b'\x7fELF',b'!<arch>\n')) and Path(name).suffix.lower() not in ('.o','.mod','.a','.so','.dll','.dylib'),'Native payload forbidden');physical[name]=data
    blobs={}
    for name,row in index['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical member')
        data=physical[row['path']];require(row['compression'] in ('none','gzip'),'Unknown encoding')
        if row['compression']=='gzip':
            require(data[:3]==b'\x1f\x8b\x08' and data[4:8]==b'\0'*4,'Nondeterministic gzip');data=gzip.decompress(data)
        require(sha(data)==row['raw_sha256'] and len(data)==row['raw_bytes'],'Decoded payload changed')
        require(not data.startswith((b'\x7fELF',b'!<arch>\n')) and Path(name).suffix.lower() not in ('.o','.mod','.a','.so','.dll','.dylib'),'Decoded native payload forbidden');data.decode('utf-8');blobs[name]=data
    return index,blobs

def verify(root,expected_index=None,draft=False):
    index,blobs=preload(root,expected_index,draft)
    def data(name,wanted=None):
        require(name in blobs,'Missing logical member: '+name);raw=blobs[name]
        if wanted:require(sha(raw)==wanted,'Closed identity changed: '+name)
        return raw
    def record(name,wanted=None):return json.loads(data(name,wanted))
    with blocked_apis() as attempts:
        require(index['scope']==SCOPE,'Qualification scope changed')
        for number,wanted in SOURCES.items():
            prefix='source'+number+'/';source=record(prefix+'source-index.json',wanted)
            require({name[len(prefix):] for name in blobs if name.startswith(prefix)}==set(source['files'])|{'source-index.json'},'Exact source lineage inventory changed')
            for name,row in source['files'].items():require(len(data(prefix+name,row['sha256']))==row['bytes'],'Source lineage payload changed')
        pins=record('source3/input-pins.json');oldpins=record('source2/input-pins.json')
        for name,wanted in {'original-root-admission-2':'76083074c8d3d80cca206bf681baf855443b7992eb2fd235cfc4c78cd4479eb7','original-root-admission-3':'2e75d369e3ba982be1aedb37005ff346b7ef23dfb96cb57512fbe37a170b8d91','candidate-root-admission-3':'a48bf87e3207fe5a848130e61679626824d89a3ecf9e31d00bf6e6c2d600e014'}.items():record('admissions/'+name+'.json',wanted)
        for number,wanted in {'1':'a762fd9a839551b3c6446da9df398f6dc2d91800e64db10070186435f8cf72be','2':'65ada7cdf186bf2b4667e713b69b5e33394473e511bfec7d78f19e7ffa6d78fb','3':'f6857c232a5c41cd0184bfcd539941737476c7a2f0a9a0778d9fb8483b01d19c'}.items():record('reviews/qp-mass-false-integrand-independent-source-review-'+number+'.json',wanted)
        failure_audit=record('reviews/qp-mass-false-integrand-original2-failed-startup-audit.json','ceccbca680ea366654adf37d47f149caaa27819cddb6e9158a76cf068495b898')
        require(failure_audit['actual_program_entered'] and failure_audit['completed_numerical_points']==0 and not failure_audit['protocol_READY_observed'],'Historical startup failure scope changed')
        data('failed-source2/worker.log',failure_audit['worker_log_sha256']);data('failed-source2/console.log',failure_audit['console_sha256'])
        for name in ('ownership','tool_environment'):data('support/'+name+'.py',pins['external_files'][pins['support_paths'][name]]['sha256'])
        data('support/toolchain-pins.json',pins['external_files'][pins['toolchain']]['sha256']);data('inputs/points.dat',pins['external_files'][pins['points']]['sha256'])
        require(data('source1/collector.py')==data('source2/collector.py')==data('source3/collector.py'),'Collector lineage changed')
        require(data('source1/false-cache0.dat')==data('source2/false-cache0.dat')==data('source3/false-cache0.dat'),'False card lineage changed')
        require(b'#UseQPIntegrandForTIRust\n.false.\n' in data('source3/false-cache0.dat'),'False policy missing')
        for label,wanted in PARENTS.items():
            parent=record('parents/'+label+'/index.json',wanted)
            expected=set(index['parent_members'][label]);prefix='parents/'+label+'/'
            require({n[len(prefix):] for n in blobs if n.startswith(prefix)}==expected|{'index.json'},'Parent subset inventory changed')
            for name in expected:
                row=parent['logical_files'][name];require(len(data(prefix+name,row['raw_sha256']))==row['raw_bytes'],'Authenticated parent member changed')
        parents={'original':record('parents/current/stages/build/result.json'),'candidate':record('parents/fresh/build/result.json')}
        build_rows={'original':parents['original']['variants']['original'],'candidate':parents['candidate']}
        for role,row in build_rows.items():
            require(len(row['sources'])==195 and len(row['objects'])==144,'Physical source/object scope changed')
            label='current' if role=='original' else 'fresh';terminal=record('parents/'+label+('/stages/build/terminal.json' if role=='original' else '/build/terminal.json'))
            require(terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Parent BUILD not closed')
            require(terminal['result_sha256']==sha(data('parents/'+label+('/stages/build/result.json' if role=='original' else '/build/result.json'))),'Parent BUILD result join changed')
            require(sha(data('parents/'+label+('/stages/build/result.json' if role=='original' else '/build/result.json')))==pins['external_files'][pins['builds'][role]['result']]['sha256'],'Source3 BUILD parent identity changed')
            require(row['runtime']['files']==pins['runtime']['files'] and len(row['runtime']['files'])==15,'Physical provider scope changed')
            for name,wanted in row['objects'].items():require(index['excluded_binaries'][row['output']+'/'+name]['sha256']==wanted,'Excluded numerical object identity changed')
            for path,wanted in {**row['actual_link_loads'],**row['runtime']['files']}.items():require(index['excluded_binaries'][path]['sha256']==wanted,'Excluded native LOAD/provider identity changed')
            require(set(pins['runtime_copies'][role])-set(oldpins['runtime_copies'][role])=={'Cards/ident_card.dat'},'Wrong resource fix')
            for name,rowpin in oldpins['runtime_copies'][role].items():require(pins['runtime_copies'][role][name]==rowpin,'Existing runtime binding changed')
            for name,rowpin in pins['runtime_copies'][role].items():
                if name.endswith('/check'):continue
                require(len(data('donor-inputs/'+role+'/'+name,rowpin['sha256']))==rowpin['bytes'],'Donor runtime data changed')
                worker=data('worker-inputs/'+role+'/'+name)
                require(worker==(data('source3/false-cache0.dat') if name in pins['numerical_cards'] else data('donor-inputs/'+role+'/'+name)),'Private worker card/resource changed')
            for name in [n for n in blobs if n.startswith('bound-source/'+role+'/')]:
                relative=name[len('bound-source/'+role+'/'):];data(name,row['sources'][relative])
        benchmark=data('support/benchmark.py',pins['external_files'][next(n for n in pins['external_files'] if n.endswith('/tools/tirust_benchmark.py'))]['sha256'])
        function=next(node for node in ast.parse(benchmark).body if isinstance(node,ast.FunctionDef) and node.name=='compare_rows');namespace={}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'retained-exact-compare_rows','exec'),namespace);comparator=namespace['compare_rows']
        collector=types.ModuleType('false_portable_collector');exec(compile(data('source3/collector.py'),'retained-pure-collector','exec'),collector.__dict__)
        lanes={};results={};summaries={}
        for role in ('original','candidate'):
            prefix='run/'+role+'/';terminal=record(prefix+'terminal.json',TERMINALS[role]);result=record(prefix+'result.json',RESULTS[role]);results[role]=result;row=build_rows[role]
            require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Unclosed False arm')
            require(terminal['result_sha256']==sha(data(prefix+'result.json')) and terminal['source_index_sha256']==result['source_index_sha256']==SOURCES['3'],'Arm source/result join changed')
            require(terminal['cap_seconds']==600 and terminal['cleanup_reserve_seconds']==60 and terminal['deadline_monotonic']==terminal['original_end_monotonic'] and terminal['resource_limits']==LIMITS and terminal['cpu']==12 and not terminal['cpu_reservation'],'Frozen phase controls changed')
            require(result['integrand_policy'] is False and result['normal_CTModeRun']==-1 and result['forced_independent_stability'] and result['threshold']==1e-6 and result['finite'] and result['LoopPoleMode']==0 and result['TIRust_cache_layers']=={'DP':0,'QP':0},'False physical card policy changed')
            for name,wanted in result['raw_files'].items():data(prefix+name,wanted)
            raw={name:data(prefix+name).decode() for name in ('dimensions.dat','warmup.dat','exact-words.dat','exact-ieee.dat','software-flags.dat')};lane=collector.collect(raw);require(lane==result['lane'],'Raw numerical collector join changed');lanes[role]=lane
            ready=record(prefix+'ready.json');require([ready[n] for n in ('points','ndim','nso','nborn')]==[100,1,1,1],'Wrong physical dimensions')
            require(record(prefix+'exact-command.json')['command']==['EXACT',1,0],'Wrong final traversal')
            for phase in ('rows','learning'):
                rows=lane['rows'] if phase=='rows' else lane['learning']['rows']
                require(all({k:v for k,v in rows[i].items() if k!='order'}=={k:v for k,v in rows[i+1].items() if k!='order'} for i in range(0,200,2)),'Overall/sole split-order duplication changed')
            exe=record(prefix+'worker-command.json')['argv'][-2];expected={**pins['runtime']['files'],exe:row['executable_sha256']};allowed={**expected,**pins['extra_mappings']}
            require(index['excluded_binaries'][exe]['sha256']==row['executable_sha256'],'Private executable identity missing')
            maps=[record(prefix+name) for name in ('maps-before.json','maps-after.json')]
            require(maps[0]['files']==maps[1]['files'],'Initialized maps changed')
            for entry in maps:require(entry['required_provider_count']==15 and all(entry['files'].get(n)==h for n,h in expected.items()) and all(allowed.get(n)==h for n,h in entry['files'].items()),'Initialized15 providers+executable scope changed')
            require(not result['candidate_promoted'] and not result['performance_acceptance'],'Unjustified claim')
            summaries[role]={'cohort':{k:v for k,v in lane['cohort'].items() if k!='history'},'physical_points':100,'returned_point_order_slots':len(lane['rows']),'returned_coefficient_slots':4*len(lane['rows']),'accuracy_word_slots':len(lane['rows']),'overall_and_sole_split_order_duplicate':True}
        require(results['candidate']['reviewed_original']==TERMINALS['original'],'Separate original review join changed')
        comparison=collector.compare(lanes['original'],lanes['candidate'],comparator)
        require(comparison==record('run/candidate/comparison.json',COMPARE)==results['candidate']['comparison'] and comparison['passed'],'Paired numerical replay failed')
        failed=record('failed-source2/terminal.json',FAILED)
        require(failed['status']=='failed' and failed['returncode']==1 and not failed['cleanup']['survivors'] and failed['source_index_sha256']==SOURCES['2'],'Historical failed startup retagged')
        require(b'not found in the parent directories!(not found for mp_)' in data('failed-source2/worker.log') and b'ident_card.dat' in data('failed-source2/worker.log'),'Failed resource diagnostic missing')
        require(not any(n in blobs for n in ('failed-source2/result.json','failed-source2/warmup.dat','failed-source2/exact-words.dat','failed-source2/dimensions.dat')),'Failed startup assigned numerical output')
        if index['sealed']:
            review=index['pair_review'];require(review is not None and review['sha256']==PAIR_REVIEW,'Independent pair review missing/changed')
            independent=record(review['logical_path'],PAIR_REVIEW)
            require(independent['status']=='clear_for_retention_of_this_closed_False_False_pair' and independent['source_index_sha256']==SOURCES['3'] and independent['comparison_sha256']==COMPARE,'Independent pair review scope changed')
            for role in ('original','candidate'):
                arm=independent['arms'][role];require(arm['terminal_sha256']==TERMINALS[role] and arm['result_sha256']==RESULTS[role] and arm['derived_public_precision_counts']==lanes[role]['cohort']['backend_precision_counts'] and arm['physical_points']==100 and arm['returned_point_order_rows']==200,'Independent arm/slot join changed')
        require(not attempts,'Portable replay attempted an external effect')
        return {'status':'passed','sealed':index['sealed'],'source3':SOURCES['3'],'arms':summaries,'False_False_comparison_passed':True,'regression_tolerance':3e-6,'ordinary_target_tolerance':1e-8,'coefficient_word_differences':sum(l['words'][j]!=r['words'][j] for l,r in zip(lanes['original']['rows'],lanes['candidate']['rows']) for j in range(4)),'accuracy_word_differences':sum(l['words'][4]!=r['words'][4] for l,r in zip(lanes['original']['rows'],lanes['candidate']['rows'])),'learning_history_equal':comparison['learning_history_equal'],'observed_IEEE_equal':comparison['observed_IEEE_equal'],'software_equal':comparison['software_latches_equal'],'source2_failed_startup_retained':True,'physical_provider_count':15,'excluded_native_identities':len(index['excluded_binaries']),'new_native_executions':0,'effects_attempted':0,'scope':SCOPE}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index');p.add_argument('--draft',action='store_true');args=p.parse_args();print(json.dumps(verify(Path(__file__).resolve().parent,args.index,args.draft),sort_keys=True))
