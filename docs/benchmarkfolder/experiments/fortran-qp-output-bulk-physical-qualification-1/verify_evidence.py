#!/usr/bin/env python3
"""Replay the closed physical experiment with indexed package text reads only.

Absolute paths in captured records are identity strings, never filesystem inputs.
No compiler, loader, process API, native library or recorded live path is opened.
"""
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import types

SOURCE='8e19ec4f1e422e5022cb3161cfe248bbc3a11cc6a2e3912e3693718ccea571ce'
TERMINALS={
 'build':'5fe62f2983e1c80e919fb404d167f1cbce2e004a1a89bab06c465d01860c932c',
 'parity':'f2623135240aab116af97f27eb09cf64d8a8f30c44db91852f471411d237c656',
 'timing':'a851754444b8c0e644871df82b8c79173e36a5f5c8229bccf33b498051f80067',
}
AUDITS={
 'build':'d1f5802b8bf51cba5740513de3620c6431b5f67e728a5189c4bbb086c69dba36',
 'parity':'e5a3577d3a68dabf2226728844c968d97c54e6dc678cee5a1a2915c36d61d4aa',
 'timing':'53561ddf8bae67ee3652083ee99c82de24466cef95122a439367a96c7bdb54ad',
}
UNIT='d0d816c0bb7ebe4cd5b8e051aaf206552597c65c8a1e26fdf6d678d8ae685546'
VARIANTS=('original-parity','candidate-parity','wire-parity','original-timing','candidate-timing')
PROC='SubProcesses/P0_uux_zgg/'

def verify(root=None, *, read_bytes=None, derive_summary=False, expected_index_sha256=None):
    def require(ok,message):
        if not ok:raise ValueError(message)
    def sha(raw):return hashlib.sha256(raw).hexdigest()
    def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}
    def safe(name):
        require(isinstance(name,str) and name and '\\' not in name,'Invalid package name')
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'Unsafe package-relative name')
        return name
    local=read_bytes is None
    if local:
        root=Path(root or Path(__file__).parent).absolute()
        require(all(not p.is_symlink() for p in (root,*root.parents)),'Symlink package root')
        def read_local(name):
            safe(name);path=root
            for part in PurePosixPath(name).parts:
                path/=part;require(not path.is_symlink(),'Symlink package member')
            require(path.is_file(),'Missing package file')
            return path.read_bytes()
        read_bytes=read_local
    index_raw=read_bytes('evidence-index.json')
    if expected_index_sha256 is not None:require(sha(index_raw)==expected_index_sha256,'Different package index')
    index=json.loads(index_raw)
    require(index.get('schema_version')==1 and index.get('kind')=='madloop-tirust-bulk-physical-qualification-v1','Wrong package schema')
    require(index.get('native_execution') is False and index.get('performance_acceptance') is False,'Package scope changed')
    files=index['files'];packed=index['compressed_text']
    for name in files:safe(name);require(name!='evidence-index.json','Self-indexed payload')
    for name,row in packed.items():safe(name);safe(row['path']);require(row['path'] in files,'Missing packed payload')
    packed_paths={row['path'] for row in packed.values()}
    require(len(packed_paths)==len(packed) and not set(packed)&set(files),'Aliased or colliding packed payloads')
    def physical(name):
        safe(name);require(name in files and files[name]['kind']=='file','Unindexed package read')
        raw=read_bytes(name);m=files[name]
        require(len(raw)==m['size'] and sha(raw)==m['sha256'],'Changed package payload: '+name)
        return raw
    def read(name):
        safe(name)
        if name in packed:
            m=packed[name];raw=gzip.decompress(physical(m['path']))
            require(len(raw)==m['raw_bytes'] and sha(raw)==m['raw_sha256'],'Changed decoded gzip text')
        else:raw=physical(name)
        require(b'\0' not in raw and not raw.startswith(b'\x7fELF'),'Binary payload retained')
        raw.decode('utf-8');return raw
    def text(name):return read(name).decode('utf-8')
    def invalid(value):raise ValueError('Nonfinite JSON '+value)
    def load(name):return json.loads(read(name),parse_constant=invalid)
    logical=(set(files)-packed_paths)|set(packed)
    for name in files:physical(name)
    for name in logical:read(name)
    if local:
        paths=list(root.rglob('*'));require(all(not p.is_symlink() for p in paths),'Symlink package inventory')
        require({str(p.relative_to(root)) for p in paths if p.is_file()}==set(files)|{'evidence-index.json'},'Package inventory changed')
    retention=load('retention.json')
    require(retention['schema']=='madloop-tirust-bulk-physical-retention-v1' and retention['source_index_sha256']==SOURCE,'Wrong retention scope')
    require(retention['binary_bytes_retained'] is False and retention['native_or_compiler_executions_during_capture']==0 and
            retention['candidate_promoted'] is False and retention['performance_acceptance'] is False,'Retention claims changed')
    for name,want in retention['origins'].items():
        require(identity(read(safe(name)))=={k:want[k] for k in ('bytes','sha256')},'Original retained bytes changed')
    require(sha(read('source/source-index.json'))==SOURCE,'Source seal changed')
    seal=load('source/source-index.json');require(len(seal['files'])==29,'Source inventory differs')
    for name,want in seal['files'].items():require(identity(read('source/'+safe(name)))==want,'Source payload changed')
    require({n for n in logical if n.startswith('source/')}=={'source/'+n for n in seal['files']}|{'source/source-index.json'},'Source extras or omissions')
    pins=load('source/input-pins.json');require(len(pins['files'])==3539 and len(pins['original_sources'])==195 and len(pins['original_objects'])==144,'Frozen numerical inventory differs')
    for name,want in pins['original_sources'].items():require(sha(read('donor/'+safe(name)))==want,'Original process source differs')
    require(sha(read('inputs/points.dat'))==pins['point_sha256'],'Frozen physical points differ')
    require(sha(read('donor/compilation.json'))==pins['case_compilation_sha256'],'Donor compiler receipt differs')
    for name,want in pins['expected_numerical_cards'].items():require(sha(read('donor/'+name))==want,'Donor card differs')
    require(sha(read('donor/Cards/param_card.dat'))==pins['expected_physical_card_sha256'],'Parameter card differs')

    def module(name):
        m=types.ModuleType('physical_packet_'+name);exec(compile(text(name+'.py'),'<indexed:'+name+'>','exec'),m.__dict__);return m
    # Only pure reviewed parser/symbol helpers execute; controller/native helpers do not.
    a=module('source/analyze');b=module('source/build_helpers');warm=module('replay_warmup')
    stages={};elapsed={}
    for phase,cap in (('build',300),('parity',600),('timing',480)):
        require(sha(read(phase+'/terminal.json'))==TERMINALS[phase],'Closed terminal changed')
        t=load(phase+'/terminal.json');stages[phase]=t
        require(t['status']=='passed' and t['returncode']==0 and not t['cleanup']['survivors'],'Original stage not closed successfully')
        require(t['cap_seconds']==cap and t['cleanup_reserve_seconds']==60,'Stage bounds changed')
        elapsed[phase]=(datetime.fromisoformat(t['finished_utc'])-datetime.fromisoformat(t['started_utc'])).total_seconds()
        require(0<elapsed[phase]<cap,'Stage exceeded cap')
        require(sha(read(phase+'/result.json'))==t['result_sha256'],'Closed result changed')
        adm=load(phase+'/admission.json');lim=load(phase+'/limits.json')['actual']
        require(lim=={'address_space_bytes':[16*1024**3]*2,'file_bytes':[512*1024**2]*2,'core_bytes':[0,0]},'Resource limits changed')
        require(adm['command']==t['command'] and t['command'][1:3]==['-c','12'] and adm['phase']==phase and adm['source_index_sha256']==SOURCE,'Stage command/source/CPU differs')
        if phase!='build':require(adm['reviewed_build_sha256']==TERMINALS['build'],'Unreviewed BUILD used')
        if phase=='timing':require(adm['reviewed_parity_sha256']==TERMINALS['parity'],'Unreviewed PARITY used')
        require(sha(read('audits/'+phase+'/receipt.json'))==AUDITS[phase],'Independent audit receipt differs')
        au=load('audits/'+phase+'/receipt.json');require(au['status']=='pass','Independent audit did not pass')
        require(sha(read('audits/'+phase+'/review.py'))==au['audit_source_sha256'],'Independent audit source changed')
        for k,v in au.items():
            if k in ('audit_compiler_calls','audit_loader_calls','audit_reducer_or_amplitude_calls','audit_native_calls'):require(v==0,'Independent audit executed native work')
        require(au[phase+'_terminal_sha256']==TERMINALS[phase] and au[phase+'_result_sha256']==t['result_sha256'],'Independent audit stage join differs')
    build=load('build/result.json')
    require(build['status']=='passed' and build['source_index_sha256']==SOURCE and build['Rust_builds']==0 and build['native_amplitude_calls']==0 and build['reused_numerical_objects'] is True,'BUILD scope differs')
    require(build['protected']=={'dso_sha256':pins['dso_sha256'],'files':3539,'input_pins_sha256':sha(read('source/input-pins.json')),'original_objects':144},'BUILD protected input join differs')
    require(set(build['executables'])==set(VARIANTS),'Compiled variants differ')
    commands=sorted(n for n in logical if n.startswith('build/commands/') and n.endswith('.terminal.json'))
    require(len(commands)==292,'BUILD tool count differs')
    intervals=[]
    for name in commands:
        t=load(name);c=load(name.replace('.terminal.json','.command.json'))
        require(t['returncode']==0 and not t.get('timed_out') and t['subprocess_new_session'] is False,'Failed original build command/owned-session receipt differs')
        require(all(t.get(k)==v for k,v in c.items()),'Compiler/query command-terminal join differs')
        require(sha(read(name.replace('.terminal.json','.stdout.txt')))==t['stdout_sha256'],'Compiler/query log differs')
        intervals.append((datetime.fromisoformat(t['started_utc']),datetime.fromisoformat(t['finished_utc'])))
    intervals.sort();require(all(x[0]<=x[1] for x in intervals) and all(x[1]<=y[0] for x,y in zip(intervals,intervals[1:])),'Build commands overlapped')
    aliases=load('link-runtime-resolution.json');runtime=pins['expected_runtime'];require(len(runtime)==15,'Expected runtime closure differs')
    for variant,row in build['executables'].items():
        prefix='build/'+variant+'/';wire=variant=='wire-parity';candidate=variant.startswith('candidate')
        want_module=read('source/'+('candidate' if candidate else 'original')+'-mtirust.f90')
        want_driver=read('source/'+('wire' if wire else 'timing' if variant.endswith('timing') else 'parity')+'-driver.f')
        require(read(prefix+'mtirust.f90')==want_module and read(prefix+'check_sa.f')==want_driver,'Compiled scientific source differs')
        require(row['module_sha256']==sha(want_module) and row['driver_sha256']==sha(want_driver),'Module/driver identity join differs')
        require(text(prefix+'makefile')==b.private_makefile(text('donor/'+PROC+'makefile'),wire=wire),'Private makefile differs')
        for name,checksum in row['sources'].items():
            source=prefix+name[len(PROC):] if name.startswith(PROC) and name[len(PROC):] in ('mtirust.f90','check_sa.f','makefile','wire_capture.c') else 'donor/'+name
            require(sha(read(source))==checksum,'Compiled/reused source identity differs')
        changes={k for k,v in row['objects'].items() if pins['original_objects'].get(k)!=v}
        allowed={PROC+n for n in ('check_sa.o','mtirust.o','mtirust.mod')}|({PROC+'wire_capture.o'} if wire else set())
        require(changes==allowed and set(pins['original_objects'])-set(row['objects'])=={'lib/libdhelas.a','lib/libcts.a','lib/libiregi.a','lib/mpmodule.mod','lib/libmodel.a'},'Numerical object reuse differs')
        pre=load(prefix+'prelink.json');require(sha(read(prefix+'prelink.json'))==row['prelink_sha256'] and sha(read(prefix+'private-link.map'))==row['link_map_sha256'],'Prelink/map identity differs')
        require(pre['completed_before_link'] is True and pre['explicit_frozen_runtime_before_link']==runtime,'Frozen provider prejoin missing')
        require(all(pre['bound'].get(k)==v for k,v in runtime.items()),'Provider not prebound')
        mapping=aliases[variant]['map_aliases']
        require(re.findall(r'^LOAD\s+(.+)$',text(prefix+'private-link.map'),re.M)==[v['spelling'] for v in mapping],'Actual LOAD spellings differ')
        require({v['resolved']:v['sha256'] for v in mapping}==row['actual_link_loads'],'Actual LOAD resolution differs')
        require(all(pre['bound'].get(v['resolved'])==v['sha256'] for v in mapping),'Actual linker input not prebound')
        listing=row['runtime']['listing'];runtime_alias=aliases[variant]['runtime_aliases'];spellings=[]
        for line in listing.splitlines():
            line=line.strip()
            if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)',line):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line);require(match is not None,'Malformed runtime listing');spellings.append(match[1])
        require(spellings==[v['spelling'] for v in runtime_alias] and {v['resolved']:v['sha256'] for v in runtime_alias}==row['runtime']['files']==runtime,'Actual runtime closure differs')
        require(runtime.get(row['runtime']['loader'])==row['runtime']['loader_sha256'],'PT_INTERP loader not bound')
        au=load('audits/build/receipt.json')['executables'][variant]
        require(au['actual_PT_INTERP']==row['runtime']['loader'] and au['executable_sha256']==row['executable_sha256'] and au['private_module_sha256']==row['module_sha256'] and au['runtime_provider_count']==15,'Independent binary identity proof differs')
        require(au['MP_external_relocations'] and all(v['type']=='R_X86_64_PLT32' for v in au['MP_external_relocations']),'External native MP binding absent')
        own=text('build/commands/'+variant+'-module-symbols.stdout.txt');exe=text('build/commands/'+variant+'-executable-symbols.stdout.txt')
        require(b.verify_symbols(own,exe,text(prefix+'private-link.map'),wire=wire)==row['symbols'],'Unique private module ownership differs')
        command=load('build/commands/'+variant+'-link.command.json');require(command['argv']==row['link_command'],'Actual link argv differs')
        require(command['argv'].index('mtirust.o')<command['argv'].index('-ltirust'),'Private module does not precede archive')
        require(command['argv'].count('-Wl,--wrap=ml_tirust_evaluate_mp')==int(wire),'Wrapper leaked into scientific variants')
        for name,want in pins['expected_numerical_cards'].items():require(sha(read(prefix+name))==want,'Private numerical card differs')
        require(set(row['card_binding']['numerical_cards'].values())==set(pins['expected_numerical_cards'].values()) and row['card_binding']['parameter_card_sha256']==pins['expected_physical_card_sha256'],'Build card binding differs')
    # All runtime proofs are replayed as recorded text, not claimed fresh local ELF checks.
    def raw_inventory(prefix):
        return {name[len(prefix):]:identity(read(name)) for name in logical if name.startswith(prefix)}
    def worker_checks(phase,name,variant):
        prefix=phase+'/'+name+'/';row=build['executables'][variant];ready=load(prefix+'ready.json')
        require({k:ready[k] for k in ('points','ndim','nso','nborn')}=={'points':100,'ndim':1,'nso':1,'nborn':1},'Worker cohort/dimensions differ')
        observed=load(prefix+'warmup.json');expected=load('source/expected-warmup.json')
        parsed=warm.replay_warmup(observed,expected,ready)
        require(parsed==observed,'Warm-up parser disagrees with recorded full trace')
        for key in ('rows','status_counts','backend_precision_counts','warmup_call_point_order','warmup_first_call','warmup_last_call','raw_outputs'):
            require(parsed[key]==expected[key],'Warm-up differs from frozen reference: '+key)
        process=load(prefix+'process.json');require(process['affinity']==[12] and process['cpu_exclusive'] is False,'Worker affinity/exclusivity differs')
        require(process['identity'] in stages[phase]['cleanup']['observed_owned_processes'] or any(
          all(v.get(k)==process['identity'].get(k) for k in ('pid','parent_pid','start_time_ticks','session','process_group')) for v in stages[phase]['cleanup']['observed_owned_processes']),'Worker not observed by cleanup owner')
        for which in ('before','after'):
            maps=load(prefix+'maps-'+which+'.json');elf=dict(maps['ELF_files']);extra=set(elf)-set(runtime)
            require(len(extra)==1,'Mapped executable count differs');exe=extra.pop()
            require(exe.endswith('/'+phase+'/'+name+'/export/'+PROC+'check') and elf.pop(exe)==row['executable_sha256'] and elf==runtime,'Actual mapped runtime/executable differs')
            paths=set()
            for line in maps['maps'].splitlines():
                fields=line.split(maxsplit=5)
                if len(fields)==6 and fields[5].startswith('/'):
                    path=fields[5];require(not path.endswith(' (deleted)'),'Deleted mapped provider');paths.add(path)
            require(paths==set(maps['ELF_files'])|set(maps['readonly_locale_data']),'Raw map text paths differ from recorded identities')
            require(all(not any(line.split()[1][1]=='w' for line in maps['maps'].splitlines() if line.endswith(path)) for path in maps['readonly_locale_data']),'Writable unbound locale mapping')
        require('W03' not in text(prefix+'worker.log'),'Original worker emitted W03')
        return ready
    def protocol(prefix,command,response,names):
        c=load(prefix+'command.json');r=load(prefix+'response.json')
        require(c['command']==command and c['expected_response']==r['response']==response,'Raw command/response protocol differs')
        require(r['files']=={n:sha(read(prefix+n)) for n in names},'Response raw hash join differs')
        before=datetime.fromisoformat(c['before_utc']);after=datetime.fromisoformat(r['after_utc']);require(before<after,'Invalid command interval')
        return before,after
    parity=load('parity/result.json');require(parity['status']=='passed' and parity['source_index_sha256']==SOURCE and parity['reviewed_build_sha256']==TERMINALS['build'],'PARITY ancestry differs')
    require(parity['full_learning_points']==300 and parity['post_learning_physical_calls']==306 and parity['candidate_promoted'] is False and parity['performance_acceptance'] is False,'PARITY scope differs')
    parsed_parity={};wire=None;init={}
    require(set(parity['variants'])=={'original-parity','candidate-parity','wire-parity'},'PARITY variants differ')
    for variant,row in parity['variants'].items():
        prefix='parity/'+variant+'/';ready=worker_checks('parity',variant,variant);init[variant]=ready['cold_seconds']
        require(row['initialization']==ready and row['card_binding']==build['executables'][variant]['card_binding'] and row['raw_files']==raw_inventory(prefix),'PARITY raw/card inventory differs')
        protocol(prefix+'parity/','PARITY 1',['PARITYDONE','1'],('parity-words.dat','parity-ieee.dat'))
        value=a.parse_parity(text(prefix+'parity/parity-words.dat'),text(prefix+'parity/parity-ieee.dat'),ready)
        require(value==row['observations'],'PARITY raw word/IEEE parser differs');a.closed_anchor_words(load('source/expected.json')['rows'],value)
        require(Counter(v['status'] for v in value['values'].values())=={219:88,323:14},'Actual precision/status counts differ')
        parsed_parity[variant]=value
        if variant=='wire-parity':
            wire=a.parse_wire(text(prefix+'real-wire.hex'),text(prefix+'wire-health.dat'));require(wire==row['wire'],'Raw wire replay differs')
        else:require(row['wire'] is None,'Wire instrumentation on scientific variant')
    for value in parsed_parity.values():a.compare_parity(parsed_parity['original-parity'],value)
    require(wire['native_QP_calls']==wire['successful_native_QP_calls']==wire['whole_batch_eligible']==3248 and wire['raw_records']==818720,'Actual native wire coverage differs')
    timing=load('timing/result.json')
    require(timing['status']=='passed' and timing['source_index_sha256']==SOURCE and timing['reviewed_build_sha256']==TERMINALS['build'] and timing['reviewed_parity_sha256']==TERMINALS['parity'],'TIMING ancestry differs')
    require(timing['candidate_promoted'] is False and timing['performance_acceptance'] is False and set(timing['anchors'])=={'1','16'},'Timing scope changed')
    readiness={}
    for variant in ('original','candidate'):
        readiness[variant]=worker_checks('timing',variant,variant+'-timing');init[variant+'-timing']=readiness[variant]['cold_seconds']
        require(raw_inventory('timing/'+variant+'/')==timing['raw_files'][variant],'TIMING raw inventory differs')
    chronological=[];seq={'original':0,'candidate':0};calls=Counter();stats={};batches=0;drifts=0
    def measure(variant,point,loops,label,expected):
        nonlocal batches,drifts
        seq[variant]+=1;number=seq[variant];prefix='timing/'+variant+'/'+label+'/'
        interval=protocol(prefix,f'ANCHORRUN {number} {point} {loops}',['ANCHORDONE',str(number),str(point),str(loops)],('anchor-timing.dat','anchor-words.dat','anchor-status.dat'))
        chronological.append((variant,label,*interval));value=a.parse_timing(text(prefix+'anchor-timing.dat'),text(prefix+'anchor-words.dat'),text(prefix+'anchor-status.dat'),readiness[variant],number,point,loops)
        ref=parsed_parity['original-parity']['values'][str(1 if point==1 else 2)]
        require(value['last_output']['ans_words']==ref['ans_words'] and value['last_output']['status']==ref['status'],'Timed anchor numerical/status words differ')
        value['accuracy_differs_from_earlier_parity_anchor']=value['last_output']['accuracy_words']!=ref['accuracy_words'];drifts+=int(value['accuracy_differs_from_earlier_parity_anchor'])
        # JSON object keys are strings in both frozen measurement/result files.
        value['accepted_status_counts']={str(k):v for k,v in value['accepted_status_counts'].items()}
        require(value==expected==load(prefix+'measurement.json'),'Raw timing measurement/result join differs')
        calls[(variant,value['last_output']['status'])]+=loops;batches+=1;return value
    for point in (1,16):
        row=timing['anchors'][str(point)];first={v:measure(v,point,1,f'p{point}-calibration1',row['calibration1'][v]) for v in ('original','candidate')}
        pilot=a.common_pilot(first['original']['seconds'],first['candidate']['seconds']);require(pilot==row['pilot_loops'],'Common pilot not derived from frozen rule')
        second={v:measure(v,point,pilot,f'p{point}-calibration2',row['calibration2'][v]) for v in ('candidate','original')}
        loops=a.common_fixed(pilot,second['original']['seconds'],second['candidate']['seconds']);require(loops==row['fixed_loops'],'Common fixed loops not derived from frozen rule')
        pairs=[];ratios=[]
        for i,pair in enumerate(row['pairs']):
            order=['original','candidate'] if i%2==0 else ['candidate','original'];require(pair['order']==order,'Alternating paired order changed')
            values={v:measure(v,point,loops,f'p{point}-pair{i+1}',pair[v]) for v in order};pairs.append({'order':order,**values})
            def seconds(x):return Fraction(x['end_ticks']-x['start_ticks'],x['clock_rate'])
            ratios.append(seconds(values['candidate'])/seconds(values['original']))
        summary=a.summarize_pairs(pairs);require(summary==row['summary'],'Paired summary differs')
        require(abs(float(statistics.median(ratios))-summary['median_paired_ratio'])<=2*math.ulp(summary['median_paired_ratio']),'Independent integer-clock ratio disagrees')
        stats[str(point)]={**summary,'fixed_loops':loops,'pairs':7}
    require(seq=={'original':18,'candidate':18} and batches==36 and sum(calls.values())==260,'Missing or duplicated timing batches/calls')
    require(all(x[2]<x[3] and x[3]<=y[2] for x,y in zip(chronological,chronological[1:])),'Actual timed command chronology differs')
    require(calls=={('original',219):121,('candidate',219):121,('original',323):9,('candidate',323):9} and drifts==0,'Timed precision or accuracy history differs')
    # Join prior rejected build/oracle failure to the indexed public unit packet.
    require(sha(read('unit-reference/evidence-index.json'))==UNIT,'Unit packet index differs')
    ui=load('unit-reference/evidence-index.json');join=load('unit-reference/join.json');require(join['index_sha256']==UNIT,'Unit ancestry join differs')
    for name,want in join['files'].items():
        raw=read('unit-reference/'+safe(name));require(identity(raw)==want,'Unit subset bytes changed')
        if name in ui['compressed_text']:m=ui['compressed_text'][name];require(want=={'bytes':m['raw_bytes'],'sha256':m['raw_sha256']},'Unit decoded index join differs')
        else:m=ui['files'][name];require(want=={'bytes':m['size'],'sha256':m['sha256']},'Unit index join differs')
    require(load('unit-reference/build2/terminal.json')['status']=='failed' and load('unit-reference/run3/terminal.json')['status']=='failed' and load('unit-reference/run4/terminal.json')['status']=='passed','Historical unit failure/success changed')
    require(sha(read('unit-reference/source2/source-index.json'))==pins['unit_source_index_sha256'],'Physical unit-source ancestry differs')
    unit_sources=[load('unit-reference/'+name+'/source-index.json')['files'] for name in ('source2','source3','source4')]
    for name in ('provider.c','driver.f90','corpus.py','corpus.json','analyze.py','observer-mtirust.f90','forced-mtirust.f90'):
        require(unit_sources[0][name]==unit_sources[1][name]==unit_sources[2][name],'Referenced unit scientific corpus changed across continuation')
    proposal=load('unit-reference/proposal/index.json');require(sha(read('unit-reference/proposal/index.json'))==pins['bulk_proposal_index_sha256'],'Original candidate proposal differs')
    for variant in ('original','candidate'):require(identity(read('source/'+variant+'-mtirust.f90'))==proposal['files'][variant+'-mtirust.f90'],'Physical candidate/original differ from unit proposal')
    profile=load('profile-reference/index.json');require(sha(read('profile-reference/index.json'))==pins['profile_index_sha256'],'Prior profile index differs')
    expected_profile=pins['files']['docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1/analysis.json']
    require(sha(read('profile-reference/analysis.json'))==expected_profile,'Prior profile analysis differs')
    previous=load('profile-reference/analysis.json')
    intervals=next(row for row in previous['points'] if row['point']==16)['qp_recorded_sequential_intervals']
    fractions={str(v['stage']):v['fraction_of_recorded_qp_intervals'] for v in intervals['stages']}
    summary={'schema':'madloop-tirust-bulk-physical-summary-v1','status':'passed','source_index_sha256':SOURCE,
       'stage_terminal_sha256':TERMINALS,'elapsed_seconds':elapsed,'build_tool_commands':292,'reused_original_objects_modules_archives':144,
       'learning_workers':5,'learning_points':500,'parity_workers':3,'post_learning_parity_calls':306,'parity_statuses_each':{'219':88,'323':14},
       'all_public_coefficient_accuracy_status_IEEE_words_equal':True,'original_wire_observer':{k:v for k,v in wire.items() if k!='calls'},
       'timing_batches':36,'timing_calls_including_calibration':260,'timing_statuses_each':{'219':121,'323':9},'accuracy_drift_batches':drifts,
       'initialization_seconds':init,'anchors':stats,'candidate_promoted':False,'performance_acceptance':False,
       'decision':'No convincing useful measured benefit; observed effects are smaller than paired scatter. Keep candidate private and unpromoted.',
       'scope':'Matched-history targeted DP/QP anchor observations on a shared host; no full-process, baseline-chain or local acceptance claim.',
       'prior_profile_interpretation':{'native_FFI_fraction_of_recorded_QP_intervals':fractions['4'],
          'output_decode_and_tensor_mapping_fraction':fractions['5'],'profile_scope':intervals['scope'],
          'inference':'An output-only change targets a small interval. Eligibility alone does not establish useful full-amplitude benefit; the separate profile does not causally explain these tiny timing differences.'},
       'unit_packet_index_sha256':UNIT,'portable_verifier_native_executions':0,'binary_bytes_retained':False,
       'runtime_proof':'Recorded bound binary identities, LOAD/PT_INTERP/runtime/map text and independent live-hash audit receipts; no fresh native verification during portable replay.'}
    if not derive_summary:require(load('qualification-summary.json')==summary,'Published summary differs from raw replay')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',nargs='?');parser.add_argument('--expected-index-sha256');args=parser.parse_args()
    print(json.dumps(verify(args.root,expected_index_sha256=args.expected_index_sha256),sort_keys=True,indent=2))
