"""Portable file-only plain CMS replay; historical controllers are never run."""
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


BUILD_SOURCE='fb73546348a4daaf736c8c17b91670225cafa4f36fe93e5a066b81c3464a7ccb'
RUN_SOURCE='7cc733efccfcfca4aa9a84a36ef3c64ea3cc4471f71e7a48bafaa16ebef3f925'
SCIENTIFIC='3c7b6023f406a9599c1ae38c87b9b4fec5e032572cf419916ec596c6b4a3fd59'
BUILD='be89129beb16a82d65624748aa6948d1d955358d910efc9956845ae6efff98f2'
RUN='182714a03f54cea2ea344e92bb31922a86cb603615176fb5aef8b060eda041bc'
REVIEWS={'generic-owned-CMS-plain-build-independent-review-2':'015ca5d7035252f48128509632f50e3b8ff214fbada765124565158a1ac43377',
 'generic-owned-CMS-plain-run-independent-review-4':'efb558d49c9050c33c7755915a689db699178d0f6fdf68183298a2cbe1c4b692',
 'generic-owned-CMS-plain-run-failure-review-2':'353811a4fc993ad2ce8cee415b7947966479dd1c2dc2893b2acedcd316108f38',
 'generic-owned-CMS-plain-run-source-review-3':'5c6cbc401d64890c49411f067a29e60f7cafcb0240f026dae49f5c4be4805cd6',
 'generic-owned-CMS-plain-run-source-review-4':'4dfa86a03536790f21f6ecc21e4da345b646be0c6e7cacb53c20ce7ecbc0b6a8'}

def verify(root,expected_index_sha256=None):
    root=Path(root).resolve();raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(raw)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(raw);require(index['schema']=='plain-CMS-normal-portable-v1','Wrong capsule schema')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Portable symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Portable exact inventory changed')
    physical={}
    for name,pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe physical member')
        value=(root/name).read_bytes();require(sha(value)==pin['sha256'] and len(value)==pin['bytes'],'Changed physical payload: '+name)
        require(not value.startswith(b'\x7fELF'),'Native binary retained');physical[name]=value
    blobs={}
    for name,pin in index['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical member')
        value=physical[pin['path']]
        if pin['compression']=='gzip':
            require(value[:3]==b'\x1f\x8b\x08' and value[4:8]==b'\0'*4,'Nondeterministic gzip')
            value=gzip.decompress(value)
        else:require(pin['compression']=='none','Unknown compression')
        require(sha(value)==pin['raw_sha256'] and len(value)==pin['raw_bytes'],'Changed logical text: '+name)
        value.decode('utf-8');require(not value.startswith(b'\x7fELF'),'Native logical member');blobs[name]=value
    def data(name,wanted=None):
        require(name in blobs,'Missing logical member: '+name);value=blobs[name]
        if wanted:require(sha(value)==wanted,'Changed frozen closure: '+name)
        return value
    def record(name,wanted=None):return json.loads(data(name,wanted))
    for prefix,wanted in [('build-source',BUILD_SOURCE),('run-source',RUN_SOURCE)]:
        source=record(prefix+'/source-index.json',wanted)
        require({n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/')}==set(source['files'])|{'source-index.json'},'Nested source inventory changed')
        for name,pin in source['files'].items():require(len(data(prefix+'/'+name,pin['sha256']))==pin['bytes'],'Nested source size changed')
    for prefix,wanted in [('history/build-source1','691aaa51ea9f39e9d810455db4cd28a5c6cda5f7de7235da86229ba3e8b6bfae'),
            ('history/run-source2','47e858475b6c5fa571885ed665607497db72153d41f7101abf85b2d48c7661be'),
            ('history/dormant-run-source3','9ccd017f09ccf8ac61b384c11aee0b20edc06c0e3ddee6358d42439949a1819a')]:
        subset=record(prefix+'/source-index.json',wanted)
        for member in (n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/') and n!=prefix+'/source-index.json'):
            require(member in subset['files'],'Historical partial source namespace changed')
            pin=subset['files'][member];require(len(data(prefix+'/'+member,pin['sha256']))==pin['bytes'],'Historical partial source bytes changed')
    analysis=load_source('portable_CMS_frozen_analysis',data('run-source/normal_run_analysis.py'),root/'run-source/normal_run_analysis.py')
    protocol=load_source('portable_CMS_frozen_protocol',data('run-source/cms_run.py'),root/'run-source/cms_run.py')
    memory={str(root/name):value for name,value in blobs.items()}
    with blocked_apis(memory) as attempts:
        scientific=record('scientific/source-receipt.json',SCIENTIFIC)
        require(scientific['active_candidate']=='candidate-2' and scientific['changed_existing_Fortran_units']==8 and
                scientific['added_Fortran_modules']==1 and scientific['private_routines']==18 and
                sum(n.endswith('.f90') for n in index['active_scientific_subset'])==7,
                'Active scientific source identity changed')
        for name in index['active_scientific_subset']:
            pin=scientific['payloads']['candidate-2/'+name]
            require(len(data('scientific/candidate-2/'+name,pin['sha256']))==pin['bytes'],'Active source subset changed')
        census=record('scientific/candidate-2/owned-composite-source.json','1417f2ee8dde5ff41b7cd9958bbf4115152373bf9f000c0e52f72ea1a5360651')
        lineage=census['lineage'];require(len(lineage['includes'])==302 and len(lineage['source_pins'])==269 and
            lineage['literal_include_closure_complete'] and not lineage['borrowed_includes'],'Clean269/include302 lineage changed')
        generation=record('lineage/generation.json','73d32bc26ee0641faaf90c7b1060471c2339a64d9dbc33d6f064356b106a093a')
        require(generation['generation_only'] and generation['amplitude_evaluations']==0 and len(generation['raw_sources'])==269,'Clean CLI generation lineage changed')
        pins=record('build-source/input-pins.json');require(len(pins['clean_sources'])==269 and len(pins['runtime'])==15 and len(pins['resources'])==4 and len(pins['shared_aliases'])==2,'Frozen input closure changed')
        for name,pin in pins['resources'].items():require(len(data('static-resources/'+name,pin['sha256']))==pin['bytes'],'Fixed resource payload changed')
        for name,pin in pins['clean_sources'].items():
            require(generation['raw_sources'][pin['path']]['sha256']==pin['sha256'],'Clean source/generation join changed')
        built=record('build/result.json','a3085e4db3ad63f4484e73c37c73ee24ce9ac06efaf7c65b25715a9e0144e072')
        ran=record('run/result.json','b0a3eb3596a8c183775d37a1a791a2aee3e3ca671958f8e2884776d79cb042eb')
        terminals={}
        for phase,wanted,result,source in [('build',BUILD,built,BUILD_SOURCE),('run',RUN,ran,RUN_SOURCE)]:
            term=record(phase+'/terminal.json',wanted);terminals[phase]=term
            require(term['status']=='passed' and term['returncode']==0 and not term['cleanup']['survivors'] and
                term['result_sha256']==sha(data(phase+'/result.json')) and result['source_index_sha256']==source,'Closed phase chain changed')
            require(term['cap_seconds']==1800 and term['cleanup_reserve_seconds']==60 and
                term['original_end_monotonic']==term['deadline_monotonic'] and abs(term['original_end_monotonic']-term['original_start_monotonic']-1800)<1e-8,'Original total phase cap changed')
            require(record(phase+'/limits.json')['actual']=={'address_space_bytes':[17179869184]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Actual resource limits changed')
        require(ran['reviewed_build_sha256']==BUILD and ran['native_processes']==6 and not ran['all_requests_numerically_valid'],'RUN scope changed')
        commands=[n for n in blobs if n.startswith('build/') and '/commands/' in n and n.endswith('.command.json')]
        require(len(commands)==147,'BUILD command coverage changed')
        for name in commands:
            stem=name[:-len('.command.json')];term=record(stem+'.terminal.json')
            require(term['returncode']==0 and term['stdout_sha256']==sha(data(stem+'.stdout.txt')),'BUILD command/log failure')
        excluded=index['excluded_binaries']
        for lane,row in built['variants'].items():
            require(excluded['build/'+lane+'/check']['sha256']==row['sha256'],'Excluded BUILD ELF changed')
            for name,pin in row['objects'].items():require(excluded['build/'+lane+'/'+name]['sha256']==pin['sha256'],'Excluded object identity changed')
            data('build/'+lane+'/owned-link.map',row['link_map_sha256'])
            pre=record('build/'+lane+'/owned-prelink.json',row['prelink_sha256']);static=record('build/'+lane+'/owned-static-runtime.json',row['static_runtime_sha256'])
            require(static['files']==row['runtime']['files']==pins['runtime'] and all(pre['bound'].get(n)==h for n,h in
                {**row['actual_link_loads'],**row['runtime']['files']}.items()),'Recorded LOAD/static/runtime identity changed')
            require(row['immutable_shared_aliases']==pins['shared_aliases'] and len(row['canonical_reused_library_leaves'])==5,'Immutable aliases or five reused leaves changed')
            data('build/'+lane+'/check_sa.f',analysis.DRIVER_SHA256[lane])
            symbols=[line.split()[-1] for line in data('build/'+lane+'/commands/symbols.stdout.txt').decode().splitlines() if line.split()]
            expected=pins['candidate_symbols'] if lane=='candidate' else []
            require(all(symbols.count(name)==1 for name in expected),'Private candidate definitions missing/duplicate')
        policy=protocol.validate_runtime_card(data('build-source/cards/MadLoopParams.dat'),data('run-source/cards/MadLoopParams.dat'))
        require(policy==record('run/runtime-card-policy.json')==ran['runtime_card_policy'] and policy['stability_threshold']==1e-6,'Pre-native card correction changed')
        expected_points=protocol.point_words(*[root/'run-source/inputs'/name for name in ('training.dat','controls.dat','critical20.dat','critical21.dat')])
        replayed={};maps=0;learning_calls=0;measured=0;all_calls=0
        for request in protocol.request_matrix():
            name=request['id'];prefix='run/evidence/'+name;row=ran['requests'][name];command=record(prefix+'/command.json')
            require(command['request']==request and command['reviewed_build']==BUILD and command['source_index_sha256']==RUN_SOURCE and not command['new_session'],'Worker request/source/session changed')
            lane=request['lane'];built_row=built['variants'][lane];elf=command['argv'][0]
            require(excluded['run/'+name+'/check']['sha256']==built_row['sha256'],'Excluded RUN ELF changed')
            raw=data(prefix+'/driver.raw64',row['raw_sha256']);parsed=analysis.parse_raw(raw.decode(),lane=lane,request_ids=request['ids'],expected_points=expected_points)
            require(parsed==row['analysis']==record(prefix+'/result.json')['analysis'],'Raw frozen parser result changed')
            require(parsed['learning']['success'] and parsed['learning']['outer_calls']==16 and parsed['accepted_for_continuation'],'Worker learning/continuation changed')
            require(parsed['pid']==row['pid'] and row['returncode']==0 and row['orchestration_closed'] and row['no_retry'],'Worker did not close once')
            data(prefix+'/stderr.txt',row['stderr_sha256'])
            p=parsed['pid'];expected=[{'direction':'RECV','fields':['READY',str(p)]},{'direction':'SEND','fields':['LEARN','1','0']},
                {'direction':'RECV','fields':['LEARNED',str(p),'1','1','15','12']}]
            for i,measurement in enumerate(parsed['measurements'],1):
                seq=i+1;expected.extend([{'direction':'SEND','fields':['RUN',str(seq),str(i)]},
                  {'direction':'RECV','fields':['DONE',str(p),str(seq),str(i),str(measurement['point_id']),str(measurement['code']),str(measurement['class'])]}])
            seq=len(request['ids'])+2;expected.extend([{'direction':'SEND','fields':['EXIT',str(seq),'0']},{'direction':'RECV','fields':['EXIT',str(p),str(seq)]}])
            require(record(prefix+'/protocol.json')==expected,'Six-request protocol changed')
            for member in sorted(n for n in blobs if n.startswith(prefix+'/maps-') and n.endswith('.json')):
                mapped=record(member);maps+=1
                require(mapped['ELF']=={**pins['runtime'],elf:built_row['sha256']} and mapped['data']=={} and mapped['affinity']==[12] and not mapped['exclusive_CPU'],'Actual15+ELF map changed')
                paths={os.path.normpath(line.split(None,5)[5]) for line in mapped['raw'].splitlines()
                       if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['ELF'])|set(mapped['data']),'Raw kernel maps and recorded identities differ')
                identity=mapped['identity'];started=mapped['started_identity'];keys=('pid','parent_pid','process_group','session','start_time_ticks')
                require(mapped['pid']==p and all(identity[k]==started[k] for k in keys),'Worker identity changed')
                require(any(all(x[k]==identity[k] for k in ('pid','process_group','session','start_time_ticks')) for x in terminals['run']['cleanup']['observed_owned_processes']),'Mapped worker absent from owned cleanup')
            replayed[name]=parsed;learning_calls+=len(parsed['learning']['records']);measured+=len(parsed['measurements']);all_calls+=raw.decode().count('\nCALL ')
        comparison=analysis.compare_ordinary(replayed['legacy-controls'],replayed['candidate-controls'])
        require(comparison==record('run/ordinary-comparison.json')==ran['ordinary_comparison'],'Ordinary comparison changed')
        require(len(comparison['residuals'])==24 and all(r['bitwise_equal'] for r in comparison['residuals']) and
            all(r['legacy']==r['candidate'] for r in comparison['accuracy_words']),'Ordinary returned-slot/ACC parity changed')
        for point in (20,21):
            legacy=replayed['legacy-critical'+str(point)]['measurements'][0];candidate=replayed['candidate-critical'+str(point)]['measurements'][0]
            require(legacy['code']==420 and legacy['expected_legacy_critical_failure'] and candidate['code']==323 and
                candidate['rawACC']==['0000000000000000']*2,'Critical observations changed')
        require(learning_calls==96 and measured==10 and all_calls==106 and maps==22,'Actual cohort coverage changed')
        for name,wanted in REVIEWS.items():
            review=record('reviews/'+name+'/receipt.json',wanted)
            for member,pin in review.get('payloads',{}).items():require(len(data('reviews/'+name+'/'+member,pin['sha256']))==pin['bytes'],'Closed review companion changed')
        failed_build=record('history/failed-build1/terminal.json','76b7362e4f8799cd62194dc5435bf5b6f244eacbc3beeefbecc2e16015d7853d')
        failed_run=record('history/failed-run2/terminal.json','6aab64a38dd4a5e7701e2b06c7bb86676ba15839920c9054b366d7c0ea549f62')
        require(failed_build['status']==failed_run['status']=='failed' and not failed_build['cleanup']['survivors'] and not failed_run['cleanup']['survivors'],'Administrative failure history changed')
        failed_raw=[data(n) for n in blobs if n.startswith('history/failed-run2/evidence/') and n.endswith('driver.raw64')]
        require(len(failed_raw)==1 and len(failed_raw[0])==84 and b'\nCALL ' not in failed_raw[0],'Failed RUN completed-call scope changed')
        summary=record('sidecars/generic-owned-CMS-plain-run-summary-4.json','6c884510084b6b0f3e9f806cdc1da61ad88e2c9804f5b0d3908cf6c9fc13789d')
        require(not summary['all_requests_numerically_valid'] and summary['completed_outer_learning_calls']==96,'Honest numerical-refusal summary changed')
        facts={'workers':6,'outer_learning_calls':96,'measurements':10,'completed_outer_calls':106,'maps_exact15_plus_ELF':22,
            'ordinary_points':3,'ordinary_bit_identical_returned_slots':24,'ordinary_bit_identical_ACC_words':6,
            'ordinary_reported_status':219,'legacy_critical_reported_status':420,'candidate_critical_reported_status':323,
            'candidate_critical_reported_ACC_zero':True,'all_requests_numerically_valid':False,'build_commands':147,
            'active_changed_units':8,'active_Fortran_modules':7,'active_C_modules':1,'private_functions':18,
            'clean_source_inventory':269,'literal_include_edges':302,'fixed_static_resources':4,'immutable_shared_aliases':2,'excluded_native_identities':len(excluded),
            'native_binaries_packaged':0,'new_native_executions':0,'administrative_failure_history_retained':True,
            'independent_physical_truth':False,'constructor_point_certificate_q_veto_qualification':False,
            'numerator_lifetime_qualification':False,'IEEE_lowering_qualification':False,'performance_acceptance':False,'production_promotion':False}
        require(index['claims']==facts,'Published scope differs from closed actual evidence');require(not attempts,'Forbidden API attempted')
    return {**facts,'verified':True,'forbidden_API_attempts':0}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
