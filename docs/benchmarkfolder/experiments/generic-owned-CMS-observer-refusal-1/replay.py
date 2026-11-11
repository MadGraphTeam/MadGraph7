"""Portable file-only observed CMS refusal replay; historical controllers are never run."""
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



SOURCE='2333cb5216b178da1d77083de30ee4c72226d2b82ec6753d0ddc1fd6459b9e29'
BUILD='c04a37845f1e5fd9a02cea25375d4f6a8e523071829b19fcb853ae27c73025fd'
BUILD_RESULT='814b24d4ec582b6a89888438da0c5cbb7f6c7773dce7b41a811eb5707a035cf8'
RUN='6bd9c810016f56594320043c30e3cd3b79b71517508b6b578eb6ffd24d6d1a33'
CANDIDATE_RAW='feed964f80f8b7306658c4dc70375378bb81ff09d2f20e058107da342c6804ef'
FAILURE_REVIEW='a477572d395057ad39c3eecb3a7dfcd4cb3b76e967faa54f6a57ebb893572f7c'
SEMANTICS='04f6df9b1abac2743cee377dcdf5d5355f89302a488fa17751e2236a67f4eb17'
DIAGNOSIS='3f9a5f528019f3a56b1f8f998878ab8848dee6df97f28cec64578c36f61c8e15'
REVIEWS={'generic-owned-CMS-observer-source-review-1':'f19322450aec81293edcf8fe968fd116b5d3bb596963f1bfe62cda5625abb420',
 'generic-owned-CMS-observer-composition-review-1':'84c09aa0dbe0667e70ffd6db558168b3b5b31edaece3e6ec3c499b081dab7083',
 'generic-owned-CMS-observer-build-independent-review-1':'f670a433f7a269bc748ce9e716ce4e53f380a13edbccd1ceae32a2453df4ac44'}

def verify(root,expected_index_sha256=None):
    root=Path(root).resolve();index_raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(index_raw)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_raw);require(index['schema']=='observed-CMS-refusal-portable-v1','Wrong capsule schema')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Portable symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Portable exact inventory changed')
    physical={}
    for name,pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe physical member')
        raw=(root/name).read_bytes();require(sha(raw)==pin['sha256'] and len(raw)==pin['bytes'],'Changed physical text: '+name)
        require(not raw.startswith(b'\x7fELF'),'Native binary retained');physical[name]=raw
    blobs={}
    for name,pin in index['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical member')
        raw=physical[pin['path']]
        if pin['compression']=='gzip':
            require(raw[:3]==b'\x1f\x8b\x08' and raw[4:8]==b'\0'*4,'Nondeterministic gzip');raw=gzip.decompress(raw)
        else:require(pin['compression']=='none','Unknown compression')
        require(sha(raw)==pin['raw_sha256'] and len(raw)==pin['raw_bytes'],'Changed logical text: '+name)
        raw.decode('utf-8');require(not raw.startswith(b'\x7fELF'),'Native logical member');blobs[name]=raw
    def data(name,wanted=None):
        require(name in blobs,'Missing logical member: '+name);raw=blobs[name]
        if wanted:require(sha(raw)==wanted,'Changed frozen closure: '+name)
        return raw
    def record(name,wanted=None):return json.loads(data(name,wanted))
    source=record('source/source-index.json',SOURCE)
    require({n[7:] for n in blobs if n.startswith('source/')}==set(source['files'])|{'source-index.json'},'Exact source inventory changed')
    for name,pin in source['files'].items():require(len(data('source/'+name,pin['sha256']))==pin['bytes'],'Source size changed')
    plain_index=record('plain-reference/source-subset/source-index.json','7cc733efccfcfca4aa9a84a36ef3c64ea3cc4471f71e7a48bafaa16ebef3f925')
    for name in ('normal_run_analysis.py','cms_run.py','cards/MadLoopParams.dat'):
        data('plain-reference/source-subset/'+name,plain_index['files'][name]['sha256'])
    normal=load_source('refusal_original_normal',data('plain-reference/source-subset/normal_run_analysis.py'),root/'plain-reference/source-subset/normal_run_analysis.py')
    protocol=load_source('refusal_original_protocol',data('plain-reference/source-subset/cms_run.py'),root/'plain-reference/source-subset/cms_run.py')
    observer=load_source('refusal_original_observer',data('source/observer_analysis.py','5f2b1e4d6f2b1ee9a3e8d722bf27270985026b188cf4e872d4407f9213a7edc3'),root/'source/observer_analysis.py')
    observed_run=load_source('refusal_original_run_helpers',data('source/observer_run.py'),root/'source/observer_run.py')
    memory={str(root/name):raw for name,raw in blobs.items()}
    with blocked_apis(memory) as attempts:
        require('failed-run/result.json' not in blobs,'Original failed stage cannot acquire a success result')
        built=record('build/result.json',BUILD_RESULT);build_terminal=record('build/terminal.json',BUILD)
        terminal=record('failed-run/terminal.json',RUN)
        require(build_terminal['status']=='passed' and build_terminal['returncode']==0 and not build_terminal['cleanup']['survivors'],'BUILD did not close')
        require(build_terminal['result_sha256']==BUILD_RESULT and built['source_index_sha256']==SOURCE and built['native_main_calls']==0,'BUILD source/result/scope changed')
        require(terminal['status']=='failed' and terminal['returncode']==1 and not terminal['cleanup']['survivors'],'Original RUN failure changed')
        for phase,row in [('build',build_terminal),('failed-run',terminal)]:
            require(row['cap_seconds']==1800 and row['cleanup_reserve_seconds']==60 and row['original_end_monotonic']==row['deadline_monotonic'] and
                abs(row['original_end_monotonic']-row['original_start_monotonic']-1800)<1e-8,'Original bounded phase changed')
            require(record(phase+'/limits.json')['actual']=={'address_space_bytes':[17179869184]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Actual phase limits changed')
        commands=[n for n in blobs if n.startswith('build/') and '/commands/' in n and n.endswith('.command.json')]
        require(len(commands)==138,'BUILD command coverage changed')
        for name in commands:
            stem=name[:-len('.command.json')];row=record(stem+'.terminal.json')
            require(row['returncode']==0 and row['stdout_sha256']==sha(data(stem+'.stdout.txt')),'BUILD command/log failure')
        pins=record('source/input-pins.json');excluded=index['excluded_binaries']
        require(len(excluded)==559,'Excluded binary identity count changed')
        for lane,row in built['variants'].items():
            require(excluded['build/'+lane+'/check']['sha256']==row['sha256'],'Excluded BUILD ELF changed')
            for name,pin in row['objects'].items():require(excluded['build/'+lane+'/'+name]['sha256']==pin['sha256'],'Excluded object/module changed')
            data('build/'+lane+'/owned-link.map',row['link_map_sha256'])
            pre=record('build/'+lane+'/owned-prelink.json',row['prelink_sha256']);static=record('build/'+lane+'/owned-static-runtime.json',row['static_runtime_sha256'])
            require(static['files']==row['runtime']['files']==pins['runtime'] and len(static['files'])==15 and all(pre['bound'].get(n)==h for n,h in
                {**row['actual_link_loads'],**static['files']}.items()),'Recorded LOAD/runtime identities changed')
            data('build/'+lane+'/check_sa.f',source['files']['emittedsrc/'+lane+'/check_sa.f']['sha256'])
        for name,wanted in REVIEWS.items():
            review=record('reviews/'+name+'/receipt.json',wanted)
            for member,pin in review.get('payloads',{}).items():require(len(data('reviews/'+name+'/'+member,pin['sha256']))==pin['bytes'],'Closed BUILD/source review companion changed')
        failure_review=record('reviews/failed-run/receipt.json',FAILURE_REVIEW)
        for name,pin in failure_review['payloads'].items():data('reviews/failed-run/'+name,pin['sha256'])
        semantics=record('diagnosis/generic-owned-CMS-frame-semantics-review-1/receipt.json',SEMANTICS)
        diagnosis=record('diagnosis/generic-owned-CMS-observer-join-diagnosis-1/receipt.json',DIAGNOSIS)
        for pin in diagnosis['payloads']:data('diagnosis/generic-owned-CMS-observer-join-diagnosis-1/'+pin['path'],pin['sha256'])
        data('admissions/root-file-only-retention.json','e3a4e61d8735b65cee64d1bf5e93e538e4b77cf12ba445f5e591453c55849a34')
        data('admissions/root-failed-run.json','a580ea2bcd9596b8d04c30d66a4feb4bf3d1f46e222bf4ad73a4b584a7752606')
        plain_result=record('plain-reference/closed-run/result.json','b0a3eb3596a8c183775d37a1a791a2aee3e3ca671958f8e2884776d79cb042eb')
        expected_points=protocol.point_words(*[root/'source/inputs'/name for name in ('training.dat','controls.dat','critical20.dat','critical21.dat')])
        names={'legacy-controls','legacy-critical20','legacy-critical21','candidate-controls'}
        require({n.split('/')[2] for n in blobs if n.startswith('failed-run/evidence/')}==names,'Actual attempted worker set changed')
        maps=0;raw_calls=0;raw_measured=0;legacy_journals=0;observed={}
        for name in sorted(names):
            prefix='failed-run/evidence/'+name;command=record(prefix+'/command.json');request=command['request'];lane=request['lane']
            require(command['source_index_sha256']==SOURCE and command['reviewed_build']==BUILD and command['observed_ELF_sha256']==built['variants'][lane]['sha256'] and not command['new_session'],'Worker source/BUILD/ELF/session changed')
            require(excluded['failed-run/'+name+'/check']['sha256']==command['observed_ELF_sha256'],'Excluded attempted ELF changed')
            plain=plain_result['requests'][name];plain_raw=data('plain-reference/evidence/'+name+'/driver.raw64',plain['raw_sha256']).decode()
            raw=data(prefix+'/driver.raw64');text=raw.decode();raw_calls+=text.count('\nCALL ');raw_measured+=text.count('\nCALL RUN ')
            pid=record(prefix+'/started.json')['pid'];messages=record(prefix+'/protocol.json')
            require(messages[0]=={'direction':'RECV','fields':['READY',str(pid)]},'Actual READY/PID changed')
            for member in sorted(n for n in blobs if n.startswith(prefix+'/maps-') and n.endswith('.json')):
                mapped=record(member);maps+=1;elf=command['argv'][0]
                require(mapped['ELF']=={**pins['runtime'],elf:command['observed_ELF_sha256']} and mapped['data']=={} and mapped['affinity']==[12] and not mapped['exclusive_CPU'],'Actual mapped15+ELF closure changed')
                paths={os.path.normpath(line.split(None,5)[5]) for line in mapped['raw'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['ELF']),'Raw kernel map differs')
                identity=mapped['identity'];require(mapped['pid']==pid and all(identity[k]==mapped['started_identity'][k] for k in ('pid','parent_pid','process_group','session','start_time_ticks')),'Mapped worker start identity changed')
                require(any(all(x[k]==identity[k] for k in ('pid','process_group','session','start_time_ticks')) for x in terminal['cleanup']['observed_owned_processes']),'Mapped worker absent from owned cleanup')
            if lane=='legacy':
                row=record(prefix+'/result.json');require(row['orchestration_closed'] and row['returncode']==0 and row['no_retry'],'Legacy worker did not close')
                replay=observer.analyze_observed(text,plain_raw,lane=lane,request_ids=request['ids'],observed_pid=pid,
                    plain_pid=plain['analysis']['pid'],parse_normal=normal.parse_raw,expected_points=expected_points)
                replay['observed_driver_source_sha256']=sha(data('source/emittedsrc/'+lane+'/check_sa.f'))
                replay['observed_driver_identity_bound_by_actual_ELF_maps']=True
                # Native controller serializes width-keyed dictionaries to JSON.
                # Normalize representation only; no analyzer arithmetic is replaced.
                replay=json.loads(json.dumps(replay,sort_keys=True))
                require(replay==row['analysis'] and replay['parity']['passed'] and replay['journal_schema_complete'],'Original complete legacy journal/parity replay changed')
                require(messages[-1]['fields'][:1]==['EXIT'] and len([r for r in messages if r['direction']=='RECV' and r['fields'][0]=='DONE'])==len(request['ids']),'Legacy protocol completion changed')
                legacy_journals+=len(replay['journals']);observed[name]=replay
            else:
                require(prefix+'/result.json' not in blobs and prefix+'/maps-done-1.json' not in blobs,'Refused candidate cannot gain success result/map')
                require(sha(raw)==CANDIDATE_RAW and len(raw)==150580,'Original candidate raw changed')
                require(messages[-1]=={'direction':'RECV','fields':['DONE',str(pid),'2','1','1','219','2']} and len(messages)==5,'Candidate stopped-request protocol changed')
                require(not any(r['fields'][:1]==['EXIT'] for r in messages),'Refused candidate cannot gain clean EXIT')
                failure=record(prefix+'/worker-failure.json')
                require(failure['returncode_at_failure']==-15 and failure['cleanup_error'] is None and not failure['worker_cleanup']['survivors'] and failure['observations_sha256_at_failure']==CANDIDATE_RAW,'Retained candidate failure/cleanup changed')
                require(failure['error']=="ObserverError('journal line1419: owned frame not bound to actual width-selected consumer')",'Original worker refusal changed')
                prefix_plain=observed_run.reference_prefixes(plain_raw,request['ids'])[1]
                try:
                    observer.analyze_observed(text,prefix_plain,lane=lane,request_ids=[1],observed_pid=pid,
                        plain_pid=plain['analysis']['pid'],parse_normal=normal.parse_raw,expected_points=expected_points)
                except observer.ObserverError as error:
                    require(str(error)=='journal line1419: owned frame not bound to actual width-selected consumer','Frozen analyzer refusal changed')
                else:raise ValueError('Original strict candidate analyzer unexpectedly passed')
                normalized,blocks=observer._split(text)
                parsed=normal.parse_raw(normalized,lane=lane,request_ids=[1],expected_points=expected_points)
                reference=normal.parse_raw(prefix_plain,lane=lane,request_ids=[1],expected_points=expected_points)
                core=lambda a:{k:a[k] for k in ('request_ids','learning','measurements')}
                require(not observer._differences(core(parsed),core(reference)) and len(blocks)==1,'Candidate pre-journal normal parity differs')
                # Read-only field diagnosis, never a repaired journal parser.
                rows=[line.split() for line in blocks[0].splitlines()]
                frames=[list(map(int,r[1:])) for r in rows if r[0]=='OBS_FRAME']
                consumers={int(r[1]):list(map(int,r[1:])) for r in rows if r[0]=='OBS_CONSUMER'}
                def words(label,key):
                    matches=[r[4:] for r in rows if r[0]==label and list(map(int,r[1:4]))==key]
                    require(len(matches)==1,'Diagnosis raw word key missing/duplicate');return matches[0]
                require(len(frames)==2,'Retained candidate DP frame count changed')
                for f in frames:
                    c=consumers[f[3]]
                    require(f[2]==f[5]==64 and f[6] in (1,2) and f[6]==c[3] and f[11]==1 and c[6]==0 and
                        f[12:16]==c[9:13]==[1]*4 and f[10]==c[7]==1,'Exact DP sentinel/current-QP diagnosis changed')
                    for leg in range(1,5):
                        for mu in range(4):
                            require(words('OBS_FRAME_REF',[f[0],leg,mu])==words('OBS_CREF_DP',[c[0],leg,mu]) and
                                words('OBS_FRAME_CURRENT',[f[0],leg,mu])==words('OBS_CURRENT_DP',[c[0],leg,mu]),'Retained reference/current raw words differ')
        require(raw_calls==70 and raw_measured==6 and legacy_journals==5 and maps==13,'Actual partial coverage changed')
        for route in semantics['all_four_wrapper_routes']:
            lines=data('source/emittedsrc/candidate/CT_interface.f').decode().splitlines()
            for width in ('dp','qp'):
                span=route[width];text=''.join(''.join(line[6:].split()) for line in lines[span['first']-1:span['last']]).upper()
                require(text==span['text'],'Source-coordinate typed-tail diagnosis changed')
        require(semantics['status']=='confirmed-width-dependent-parser-contract-defect' and not semantics['actual_inputview_mismatch_found'] and
                not semantics['scientific_source_defect_found'] and semantics['parser_edits']==semantics['frozen_edits']==0,'Source diagnosis scope changed')
        require(failure_review['candidate_accepted_measurements']==0 and failure_review['started_workers']==4 and failure_review['complete_legacy_workers']==3 and
            not failure_review['whole_four_ELF_gate_passed'],'Independent original failure scope changed')
        facts={'original_observed_RUN_passed':False,'strict_original_analyzer_refusal_reproduced':True,'started_workers':4,
            'complete_legacy_workers':3,'complete_legacy_measurement_journals':5,'raw_outer_calls':70,'raw_outer_learning_calls':64,
            'raw_measured_calls':6,'candidate_measured_raw_records':1,'candidate_accepted_measurements':0,'unattempted_candidate_point_ids':[7,12,20,21],
            'actual_maps_exact15_plus_ELF':13,'build_commands':138,'candidate_DP_frames':2,'candidate_QP_frames':0,
            'DP_frame_argument_true_vs_consumer_actual_QP_false':True,'other_checked_reference_current_mask_OK_fields_match':True,
            'source_contract_diagnosis_reproduced':True,'stage_success_result_absent':True,'candidate_success_result_absent':True,
            'excluded_native_identities':559,'native_binaries_packaged':0,'new_native_executions':0,'repaired_parser_executed':False,
            'native_retry_performed':False,'existing_Rust_C_bridge_changed':False,'private_Fortran_observer_interfaces_compiled':True,
            'whole_observer_runtime_gate_passed':False,'candidate_constructor_q_veto_qualified':False,'independent_physical_truth':False,
            'numerator_lifetime_qualified':False,'IEEE_lowering_qualified':False,'cache_qualified':False,'performance_acceptance':False,'production_promotion':False}
        require(index['claims']==facts,'Published claims differ from original failed evidence');require(not attempts,'Forbidden replay API attempted')
    return {**facts,'verified':True,'forbidden_API_attempts':0,'verification_scope':'Portable evidence and unchanged refusing analyzer; original RUN remains failed'}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
