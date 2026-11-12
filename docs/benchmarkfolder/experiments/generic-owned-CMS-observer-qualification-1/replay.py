"""Portable file-only corrected observed CMS replay; historical controllers are never run."""
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

def verify(root,expected_index_sha256=None,test_probes=False):
    import ast
    import fractions
    import struct
    root=Path(root).resolve();index_raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(index_raw)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_raw);require(index['schema']=='observed-CMS-success-portable-v1','Wrong capsule schema')
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
    source2_sha='1a1cebf262c5fe7767a1bf6db54591a3baae9d72326370524dc918a41fd69e5d'
    run2_sha='f6a77100a5da0f87cb0fb915a5c32567070d81af1361877ff8cf80808a17da91'
    result2_sha='10db46c9885340846d5865f8cf184c2363e82e05b7e393d9a7ec20416d41b82a'
    source2=record('run-source2/source-index.json',source2_sha)
    require({n[len('run-source2/'):] for n in blobs if n.startswith('run-source2/')}==set(source2['files'])|{'source-index.json'},'Corrected RUN-only exact source inventory changed')
    for name,pin in source2['files'].items():require(len(data('run-source2/'+name,pin['sha256']))==pin['bytes'],'Corrected source payload changed')
    ancestry=record('history/original-refusal-index.json','0278b7ecb1e2baf346470b28de344d385b250137aea8f44c33eaf65cdb726b3c')
    for name,pin in ancestry['logical_files'].items():require(len(data(name,pin['raw_sha256']))==pin['raw_bytes'],'Original refusal ancestry changed')
    for name in ('README.md','replay.py','test_replay.py'):data('history/original-refusal-'+name,ancestry['files'][name]['sha256'])
    plain_index=record('plain-reference/source-subset/source-index.json','7cc733efccfcfca4aa9a84a36ef3c64ea3cc4471f71e7a48bafaa16ebef3f925')
    for name in ('normal_run_analysis.py','cms_run.py','cards/MadLoopParams.dat'):data('plain-reference/source-subset/'+name,plain_index['files'][name]['sha256'])
    require(source2['compiled_source_index_sha256']==SOURCE and source2['compiled_BUILD_terminal']==BUILD and source2['compiled_payloads_unchanged'] and
        not source2['new_Fortran_or_driver_changes'] and not source2['parent_failure_reinterpreted_as_completed_gate'],'Compiled and corrected Python provenance collapsed')
    reuse=record('run-source2/compiled-source-reuse.json')
    compiled=data('build/candidate/CT_interface.f','5dbe44e671be8e9bad32a6910b5ec9da18f0005e19bcdbf2cb1d19ff28501619')
    total_compiled=0
    for lane,files in reuse['payloads'].items():
        for name,pin in files.items():
            require(data('source/emittedsrc/'+lane+'/'+name,pin['sha256'])==data('run-source2/emittedsrc/'+lane+'/'+name,pin['sha256']),'Compiled21 payloads changed');total_compiled+=1
    require(total_compiled==21 and compiled==data('source/emittedsrc/candidate/CT_interface.f'),'Actual compiled/source CT join changed')
    memory={str(root/name):raw for name,raw in blobs.items()}
    def functions(raw,names,label,extra=None):
        nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes}==set(names),'Frozen pure helper definition missing')
        scope={'Path':Path,'hashlib':hashlib,'struct':struct,**(extra or {})};exec(compile(ast.Module(body=nodes,type_ignores=[]),label,'exec'),scope)
        return types.SimpleNamespace(**scope)
    normalize=lambda value:json.loads(json.dumps(value,sort_keys=True))
    with blocked_apis(memory) as attempts:
        normal=load_source('success_normal',data('plain-reference/source-subset/normal_run_analysis.py'),root/'plain-reference/source-subset/normal_run_analysis.py')
        observer1=load_source('success_original_strict_observer',data('source/observer_analysis.py'),root/'source/observer_analysis.py')
        observer2=load_source('success_corrected_typed_observer',data('run-source2/observer_analysis.py'),root/'run-source2/observer_analysis.py')
        protocol=functions(data('plain-reference/source-subset/cms_run.py'),{'require','point_words','request_matrix'},'frozen_plain_protocol_helpers')
        helpers=functions(data('run-source2/observer_run.py'),{'require','digest','reference_prefixes','compact_result'},'frozen_corrected_observer_helpers')
        built=record('build/result.json',BUILD_RESULT);build_terminal=record('build/terminal.json',BUILD)
        terminal=record('run/terminal.json',run2_sha);result=record('run/result.json',result2_sha)
        require(build_terminal['status']=='passed' and built['native_main_calls']==0 and build_terminal['result_sha256']==BUILD_RESULT and built['source_index_sha256']==SOURCE,'Compiled BUILD closure changed')
        require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and terminal['result_sha256']==result2_sha and not terminal['cleanup']['survivors'],'Corrected RUN2 not cleanly closed')
        require(result['source_index_sha256']==source2_sha and result['compiled_source_index_sha256']==SOURCE and result['reviewed_build_sha256']==BUILD,'RUN2 source/BUILD chain changed')
        for phase,row in [('build',build_terminal),('run',terminal),('failed-run',record('failed-run/terminal.json',RUN))]:
            require(row['cap_seconds']==1800 and row['cleanup_reserve_seconds']==60 and row['original_end_monotonic']==row['deadline_monotonic'] and
                abs(row['original_end_monotonic']-row['original_start_monotonic']-1800)<1e-8,'Original phase bounds changed')
            require(record(phase+'/limits.json')['actual']=={'address_space_bytes':[17179869184]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]},'Actual phase limits changed')
        commands=[n for n in blobs if n.startswith('build/') and '/commands/' in n and n.endswith('.command.json')]
        require(len(commands)==138,'BUILD command coverage changed')
        for name in commands:
            stem=name[:-len('.command.json')];row=record(stem+'.terminal.json')
            require(row['returncode']==0 and row['stdout_sha256']==sha(data(stem+'.stdout.txt')),'BUILD command/log failure')
        pins=record('source/input-pins.json');excluded=index['excluded_binaries']
        require({k:v for k,v in excluded.items() if not k.startswith('run/')}==ancestry['excluded_binaries'],'Original excluded native identities changed')
        for lane,row in built['variants'].items():
            require(excluded['build/'+lane+'/check']['sha256']==row['sha256'],'Excluded compiled ELF changed')
            for name,pin in row['objects'].items():require(excluded['build/'+lane+'/'+name]['sha256']==pin['sha256'],'Excluded numerical object/module changed')
            data('build/'+lane+'/owned-link.map',row['link_map_sha256'])
            pre=record('build/'+lane+'/owned-prelink.json',row['prelink_sha256']);static=record('build/'+lane+'/owned-static-runtime.json',row['static_runtime_sha256'])
            require(static['files']==row['runtime']['files']==pins['runtime'] and len(static['files'])==15 and all(pre['bound'].get(n)==h for n,h in
                {**row['actual_link_loads'],**static['files']}.items()),'Recorded prebound LOAD/runtime set changed')
            data('build/'+lane+'/check_sa.f',source['files']['emittedsrc/'+lane+'/check_sa.f']['sha256'])
        for name,wanted in REVIEWS.items():record('reviews/'+name+'/receipt.json',wanted)
        record('reviews/generic-owned-CMS-observer-source-review-2/receipt.json','78dc707ed20cfed257be25cf1d531e510f1cda78ee9d3a2e3852c60803af9c1f')
        record('reviews/generic-owned-CMS-typed-parser-review-1/receipt.json','8b509b3954b9a6ad32e4af272f49fada0ff8bab1d40c0b10f213fc44f454f322')
        record('reviews/failed-run/receipt.json',FAILURE_REVIEW);record('diagnosis/generic-owned-CMS-frame-semantics-review-1/receipt.json',SEMANTICS)
        record('diagnosis/generic-owned-CMS-observer-join-diagnosis-1/receipt.json',DIAGNOSIS)
        data('admissions/root-run2.json','b4ed8b5b80316cb30858d378f955164d3d342480a1af8b190f16d178b638d265')
        plain_result=record('plain-reference/closed-run/result.json','b0a3eb3596a8c183775d37a1a791a2aee3e3ca671958f8e2884776d79cb042eb')
        require(result['plain_reference_RUN_result']=='b0a3eb3596a8c183775d37a1a791a2aee3e3ca671958f8e2884776d79cb042eb' and
            result['plain_reference_RUN_terminal']=='182714a03f54cea2ea344e92bb31922a86cb603615176fb5aef8b060eda041bc','Own-plain closure changed')
        expected_points=protocol.point_words(*[root/'run-source2/inputs'/name for name in ('training.dat','controls.dat','critical20.dat','critical21.dat')])
        # Reproduce the unchanged strict source1 refusal; source2 never
        # retroactively turns the partial original RUN into a passed gate.
        require('failed-run/result.json' not in blobs and record('failed-run/terminal.json')['status']=='failed','Original RUN1 failure reinterpreted')
        original_prefix='failed-run/evidence/candidate-controls';oldcommand=record(original_prefix+'/command.json');oldraw=data(original_prefix+'/driver.raw64',CANDIDATE_RAW).decode()
        plain=plain_result['requests']['candidate-controls'];plainraw=data('plain-reference/evidence/candidate-controls/driver.raw64',plain['raw_sha256']).decode()
        try:
            observer1.analyze_observed(oldraw,helpers.reference_prefixes(plainraw,[1,7,12])[1],lane='candidate',request_ids=[1],observed_pid=record(original_prefix+'/started.json')['pid'],
                plain_pid=plain['analysis']['pid'],parse_normal=normal.parse_raw,expected_points=expected_points)
        except observer1.ObserverError as error:require(str(error)=='journal line1419: owned frame not bound to actual width-selected consumer','Frozen original analyzer refusal changed')
        else:raise ValueError('Original strict refusing analyzer unexpectedly passed')
        require(original_prefix+'/result.json' not in blobs,'Original refused worker cannot acquire a result')
        names={r['id'] for r in protocol.request_matrix()};require(set(result['requests'])==names and
            {n.split('/')[2] for n in blobs if n.startswith('run/evidence/')}==names,'Six actual RUN2 worker set changed')
        analyzed={};maps=0;learning_calls=0;measurements=0;journal_total=0;veto_total=0;protocol_records=0;frames=0;producers=0;consumers=0;pairs_count=0;native_words=0
        for request in protocol.request_matrix():
            name=request['id'];lane=request['lane'];prefix='run/evidence/'+name;linked=result['requests'][name]
            command=record(prefix+'/command.json');row=record(prefix+'/result.json',linked['result_sha256']);started=record(prefix+'/started.json');pid=started['pid']
            require(command['request']==row['request']==request and command['source_index_sha256']==source2_sha and command['compiled_source_index_sha256']==SOURCE and command['reviewed_build']==BUILD and
                command['observed_ELF_sha256']==built['variants'][lane]['sha256'] and not command['new_session'],'Actual worker request/source/ELF/session changed')
            require(excluded['run/'+name+'/check']['sha256']==command['observed_ELF_sha256'] and excluded['run/'+name+'/check']['original_path']==command['argv'][0],'Excluded RUN2 ELF identity changed')
            require(started['identity']['pid']==pid==row['pid']==linked['pid'] and started['source_executable_sha256']==command['observed_ELF_sha256'],'Worker started PID/ELF binding changed')
            require(row['orchestration_closed'] and row['returncode']==0 and row['no_retry'] and row['stderr_sha256']==sha(data(prefix+'/stderr.txt')),'Observed worker not cleanly closed')
            plain=plain_result['requests'][name];plain_prefix='plain-reference/evidence/'+name;plain_raw=data(plain_prefix+'/driver.raw64',plain['raw_sha256']).decode()
            require(record(plain_prefix+'/result.json')==plain,'Full own-plain worker/top-result join changed')
            plain_command=record(plain_prefix+'/command.json')
            require(plain_command['request']==request and plain_command['source_index_sha256']=='7cc733efccfcfca4aa9a84a36ef3c64ea3cc4471f71e7a48bafaa16ebef3f925' and
                plain_command['reviewed_build']=='be89129beb16a82d65624748aa6948d1d955358d910efc9956845ae6efff98f2' and not plain_command['new_session'],'Own-plain command source/BUILD changed')
            require(normal.parse_raw(plain_raw,lane=lane,request_ids=request['ids'],expected_points=expected_points)==plain['analysis'],'Full own-plain raw analysis changed')
            raw=data(prefix+'/driver.raw64',linked['raw_sha256']).decode();require(sha(raw.encode())==row['raw_sha256'],'Worker raw/result digest join changed')
            replay=observer2.analyze_observed(raw,plain_raw,lane=lane,request_ids=request['ids'],observed_pid=pid,plain_pid=plain['analysis']['pid'],parse_normal=normal.parse_raw,
                expected_points=expected_points,compiled_dispatch_source=compiled if lane=='candidate' else None)
            replay['observed_driver_source_sha256']=sha(data('source/emittedsrc/'+lane+'/check_sa.f'));replay['observed_driver_identity_bound_by_actual_ELF_maps']=True
            replay=normalize(replay);require(replay==row['analysis'] and replay['parity']['passed'] and replay['journal_schema_complete'],'Source-bound raw journal/full plain parity replay changed')
            compact=helpers.compact_result(root/prefix/'result.json',row);compact['result_path']=linked['result_path']
            require(compact==linked,'Full worker/compact stage join changed')
            messages=record(prefix+'/protocol.json');expected=[{'direction':'RECV','fields':['READY',str(pid)]},{'direction':'SEND','fields':['LEARN','1','0']},
                {'direction':'RECV','fields':['LEARNED',str(pid),'1','1','15','12']}]
            for i,measurement in enumerate(replay['measurements'],1):
                expected.extend([{'direction':'SEND','fields':['RUN',str(i+1),str(i)]},{'direction':'RECV','fields':['DONE',str(pid),str(i+1),str(i),str(measurement['point_id']),str(measurement['code']),str(measurement['class'])]}])
            expected.extend([{'direction':'SEND','fields':['EXIT',str(len(request['ids'])+2),'0']},{'direction':'RECV','fields':['EXIT',str(pid),str(len(request['ids'])+2)]}])
            require(messages==expected,'Actual protocol READY/LEARNED/DONE/EXIT join changed');protocol_records+=len(messages)
            plain_protocol=[{'direction':m['direction'],'fields':[str(plain['pid']) if f==str(pid) else f for f in m['fields']]} for m in expected]
            require(record(plain_prefix+'/protocol.json')==plain_protocol,'Full own-plain protocol parity changed')
            for member in sorted(n for n in blobs if n.startswith(prefix+'/maps-') and n.endswith('.json')):
                mapped=record(member);maps+=1
                require(mapped['ELF']=={**pins['runtime'],command['argv'][0]:command['observed_ELF_sha256']} and mapped['data']=={} and mapped['affinity']==[12] and not mapped['exclusive_CPU'],'Actual mapped15+ELF closure changed')
                paths={os.path.normpath(line.split(None,5)[5]) for line in mapped['raw'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(mapped['ELF']),'Raw kernel map file closure differs')
                identity=mapped['identity'];require(mapped['pid']==identity['pid']==pid and all(identity[k]==mapped['started_identity'][k]==started['identity'][k] for k in ('pid','parent_pid','process_group','session','start_time_ticks')),'Actual mapped worker identity changed')
                require(any(all(x[k]==identity[k] for k in ('pid','process_group','session','start_time_ticks')) for x in terminal['cleanup']['observed_owned_processes']),'Mapped worker missing from owned cleanup')
            require(sum(n.startswith(prefix+'/maps-') and n.endswith('.json') for n in blobs)==len(request['ids'])+2,'Actual worker snapshot coverage changed')
            learning_calls+=replay['learning']['outer_calls'];measurements+=len(replay['measurements']);journal_total+=len(replay['journals']);veto_total+=replay['observed_veto_count'];analyzed[name]=replay
            for journal in replay['journals']:
                frames+=len(journal['frames']);producers+=len(journal['producers']);consumers+=len(journal['consumers'])
                for frame in journal['frames']:
                    pairs_count+=len(frame['pairs']);native_words+=sum(len(row) for matrix in frame['native'] for row in matrix)
            codes=[m['code'] for m in replay['measurements']]
            require(codes==([219]*3 if request['cohort']=='controls' else [420 if lane=='legacy' else 323]),'Ordinary/candidate/legacy numerical status scope changed')
            if request['cohort']!='controls':require(all(w==('3FF0000000000000' if lane=='legacy' else '0000000000000000') for m in replay['measurements'] for w in m['rawACC']),'Retained critical reported accuracy words changed')
        ordinary=normal.compare_ordinary(analyzed['legacy-controls'],analyzed['candidate-controls'])
        require(ordinary==record('run/ordinary-comparison.json')==result['ordinary_comparison'] and ordinary['passed'],'Ordinary amplitude comparison changed')
        changes={};upper={}
        for cohort in ('controls','critical20','critical21'):
            delta=normalize(observer2.compare_native(analyzed['legacy-'+cohort],analyzed['candidate-'+cohort]))
            require(delta==record('run/native-delta-'+cohort+'.json')==result['native_deltas'][cohort] and not delta['unmatched_frames'],'Actual native delta/unmatched-frame join changed')
            changes[cohort]=len(delta['changes']);upper[cohort]=sum(r['upper_native_consumed'] for r in delta['changes'])
            require(changes[cohort]==(0 if cohort=='controls' else 8) and upper[cohort]==(0 if cohort=='controls' else 4),'Actual matrix/consumed-upper word delta count changed')
            if cohort!='controls':
                critical_frames=analyzed['candidate-'+cohort]['journals'][0]['frames'];pairs=[(f,p) for f in critical_frames for p in f['pairs'] if p['used']]
                require(len(pairs)==8 and {(f['width'],f['ctmode']) for f,p in pairs}=={(64,1),(64,2),(128,4),(128,5)} and
                    all({p['i'],p['j']}=={2,4} and p['class']==2 and f['producer_lifetime_complete'] for f,p in pairs),'Observed symmetric-pair veto/lifetime scope changed')
        require(learning_calls==96 and measurements==journal_total==10 and maps==22 and veto_total==16 and
            (frames,producers,consumers,pairs_count,native_words,protocol_records)==(28,252,28,448,1344,50),'Actual six-worker coverage totals changed')
        require(result['all_plain_observer_parity_passed'] and not result['all_requests_numerically_valid'] and result['expected_legacy_critical_failures']==['legacy-critical20','legacy-critical21'] and
            result['observed_veto_count']==16 and result['native_processes']==6 and result['compiler_calls']==result['loader_queries']==0,'Success orchestration versus numerical refusal scope changed')
        require(index['RUN2_review_sha256']=='51d5c291c9aacfe9ea6fd9f75ecc6c8d1f9f3c0f5f84a22604746b3c00dd17a4' and
            index['positive_provenance_review_sha256']=='04e2d4618943d74384d5ca6f79fa87e6dac754b119e422a68ba0b876f42822b6','Independent closed review pins changed')
        review=record('reviews/actual-run2/receipt.json',index['RUN2_review_sha256'])
        positive=record('reviews/positive-provenance2/receipt.json',index['positive_provenance_review_sha256'])
        for pin in review['audit_payloads']:require(len(data('reviews/actual-run2/'+Path(pin['path']).name,pin['sha256']))==pin['bytes'],'Independent full RUN2 review companion changed')
        for pin in positive['payloads']:require(len(data('reviews/positive-provenance2/'+pin['path'],pin['sha256']))==pin['bytes'],'Independent positive-provenance companion changed')
        require(review['stage']['status']=='passed' and review['qualification']['actual_bounded_observer_parity_gate_closed_pass'] and not review['reported_outcomes']['all_requests_numerically_valid'] and
            review['raw_byte_joins']['outer_CALLs']==106 and review['raw_byte_joins']['OBS_NATIVE']==native_words and not positive['independent_finite_truth'],'Independent review scope changed')
        require(not index['independent_RUN2_review_pending'],'Cannot seal before independent closed RUN2 review')
        require(data('run-source2/cards/MadLoopParams.dat')==data('source/cards/MadLoopParams.dat') and
            record('run/runtime-card-policy.json')==record('plain-reference/closed-run/runtime-card-policy.json') and
            sha(data('run-source2/cards/MadLoopParams.dat'))=='01db62b7a861aa832063c5216473aafb8adff0a44bffc2956fd7b684452093e0','Runtime card1e-6 policy changed')
        typed_denials=0
        if test_probes:
            raw=data('run/evidence/candidate-critical20/driver.raw64').decode();block=observer2._split(raw)[1][0];journal=analyzed['candidate-critical20']['journals'][0]
            kwargs={'sequence':journal['sequence'],'request':journal['request'],'point_id':20}
            def refused(call,diagnostic):
                nonlocal typed_denials
                try:call()
                except observer2.ObserverError as error:require(diagnostic in str(error),'Wrong typed tamper refusal');typed_denials+=1
                else:raise ValueError('Typed tamper probe unexpectedly passed')
            for wrong in (None,True,compiled+b'changed'):
                refused(lambda wrong=wrong:observer2.parse_journal(block,**kwargs,compiled_dispatch_source=wrong),'exact pinned compiled dispatch-source bytes')
            def changed_frame(width):
                rows=block.splitlines(keepends=True)
                for i,line in enumerate(rows):
                    fields=line.split()
                    if fields[0]=='OBS_FRAME' and int(fields[3])==width:
                        fields[12]='0' if fields[12]=='1' else '1';rows[i]=' '.join(fields)+'\n';return ''.join(rows)
                raise ValueError('Actual typed fixture frame absent')
            refused(lambda:observer2.parse_journal(changed_frame(64),**kwargs,compiled_dispatch_source=compiled),'DP frame lacks pinned caller TRUE')
            refused(lambda:observer2.parse_journal(changed_frame(128),**kwargs,compiled_dispatch_source=compiled),'QP flag not bound to actual consumer')
            rows=block.splitlines(keepends=True)
            for i,line in enumerate(rows):
                fields=line.split()
                if fields[0]=='OBS_FRAME_CURRENT':
                    fields[-1]=f'{int(fields[-1],16)^1:032X}';rows[i]=' '.join(fields)+'\n';break
            refused(lambda:observer2.parse_journal(''.join(rows),**kwargs,compiled_dispatch_source=compiled),'actual width-selected consumer')
            rows=block.splitlines(keepends=True);removed=False
            for i,line in enumerate(rows):
                if line.startswith('OBS_PRODUCER '):del rows[i];removed=True;break
            require(removed,'Actual producer probe absent')
            refused(lambda:observer2.parse_journal(''.join(rows),**kwargs,compiled_dispatch_source=compiled),'wrong tag/cardinality')
        facts={'corrected_observer_RUN2_orchestration_passed':True,'original_RUN1_remains_failed':True,'strict_original_analyzer_refusal_reproduced':True,
            'compiled_source1_unchanged':True,'compiled_payloads_unchanged':21,'new_Fortran_or_bridge_changes':False,'build_commands':138,
            'fresh_workers':6,'outer_learning_calls':learning_calls,'measured_calls':measurements,'completed_outer_calls':learning_calls+measurements,
            'accepted_complete_journals':journal_total,'actual_maps_exact15_plus_ELF':maps,'all15_state_fields_and_RAW_history_parity':True,
            'journal_frames':frames,'producer_records':producers,'producer_triplets':producers//3,'consumer_records':consumers,'matrix_pairs':pairs_count,'native_words':native_words,'protocol_records':protocol_records,
            'ordinary_points':3,'ordinary_reported_status':219,'ordinary_native_word_changes':changes['controls'],
            'candidate_critical_reported_status':323,'candidate_critical_ACC_zero_reported':True,'legacy_critical_reported_status':420,'all_requests_numerically_valid':False,
            'candidate_critical_veto_records_each':8,'candidate_critical_matrix_word_changes_each':8,'candidate_critical_consumed_upper_changes_each':4,
            'observed_veto_records_total':veto_total,'DP_eligibility_sentinel_is_genuine_QP':False,'source_bound_veto_observations_only':True,
            'excluded_native_identities':len(excluded),'native_binaries_packaged':0,'new_native_executions':0,
            'independent_physical_truth':False,'full_caller_qualified':False,'numerator_lifetime_qualified':False,'IEEE_lowering_qualified':False,
            'generic_constructor_coverage_qualified':False,'cache_qualified':False,'performance_acceptance':False,'production_promotion':False}
        require(index['claims']==facts,'Published claims differ from corrected closed evidence');require(not attempts,'Forbidden replay API attempted')
    return {**facts,'verified':True,'forbidden_API_attempts':0,'typed_guard_test_denials':typed_denials}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
