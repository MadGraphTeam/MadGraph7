"""Pure portable source/build/ancestry joins for the closed checked CMS20 run.

Callbacks supply authenticated text bytes only. Marker stripping and card/source
joins are freshly replayed. ELF/PT_INTERP/provider hashes, process absence and
locks remain retained historical observations: no external binary is opened here.
No scientific output coefficient is decoded or compared by this module.
"""
import ast
from collections import Counter
import difflib
import hashlib
import json
import re
import shlex

SCIENCE='7dab8a04296a44e6eef1cf017bee751491220b77804ebe2a34faa328f85cb2c9'
PLAN='9387a2858df75bf34d0940121da6e006fe7648a9d63d241276ba05f94487f7b4'
CONTROLLER1='fd482474221715397a0dff7b47244ef25a07f077e79b4b8a47c5e3beca398c03'
CONTROLLER2='88cc660d0deecc3b8f87eba23bc10deed729a048cb4bddfcbe3ca4265f8830da'
AMPLITUDE='ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'
BUILD='3071f3887244a34ebdf134e3867484cae6ddec9a6623309d3668d8817d26f09a'
BUILD_RECEIPT='f172832d12ef1a568467fe44bd3ec2c80f8f9717c628e1ab689a3773a57e0145'
BUILD_SUMMARY='4a32825da2511600228eb11c4a2dfaca2577adb8647e8fb36426cb17893a260e'
FAILED_RUN='754078acb9700ffa10b9c040bc6c0de86784e4da66adaaef0e8fdcb8441985af'
RUN='046d83232bd40235f2401341c0806fb9446599b321da23e72b663b9430d68aa1'
AMP='kinematic-invariant-amplitude-qualification-1'
MATCHED='matched-native-input-qualification-1'
LEGACY='kinematic-legacy-reference-adjudication-1'
ROOT='/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
OUTPUT=ROOT+'.tirust-build/kinematic-normal-checked-executor-1/work/builds/cms/checked'
SUB='SubProcesses/P0_epem_mupmum/'
WORKER='cms-p20-auto-lib9-checked'
SIX=('CT_interface.f','TIRUST_interface.f','TIR_interface.f','loop_matrix.f','loop_CT_calls_1.f','mp_compute_loop_coefs.f')
TARGETS=['check_sa.o','loop_matrix.o','mp_compute_loop_coefs.o','CT_interface.o','TIR_interface.o','TIRUST_interface.o','checked_trace.o']
COMPILE_ORDER=['checked_trace.o']+TARGETS[:-1]


def require(ok,message):
    if not ok:
        raise ValueError('Closed checked build: '+message)


def pin(raw):
    require(isinstance(raw,bytes),'data callback must return bytes')
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def identity(item):
    return {key:item[key] for key in ('bytes','sha256')}


def safe(name):
    require(isinstance(name,str) and name and not name.startswith('/') and
            all(part not in ('','.','..') for part in name.split('/')),'unsafe logical path')
    return name


def checked(data,name,expected):
    raw=data(safe(name));actual=pin(raw)
    require(actual['sha256']==(expected if isinstance(expected,str) else expected['sha256']) and
            (isinstance(expected,str) or actual['bytes']==expected['bytes']),'changed payload '+name)
    return raw


def seal(data,prefix,sha,count):
    source=json.loads(checked(data,prefix+'/index.json',sha))
    require(len(source['files'])==count,'sealed payload count differs: '+prefix)
    for name,expected in source['files'].items():
        checked(data,prefix+'/'+safe(name),expected)
    return source


def suffix(name,prefix):
    require(name.startswith(prefix),'foreign recorded path')
    return safe(name[len(prefix):])


def source_inventory(build):
    prefix=suffix(build['output'],ROOT)+'/'
    return {suffix(name,prefix):value for name,value in build['sources'].items()}


def strip_markers(raw):
    """Strict balanced insertion removal, not execution of the historical generator."""
    text=raw.decode('ascii');out=[];active=None;count=0
    for line in text.splitlines(True):
        begin=re.fullmatch(r'C NCHECK_BEGIN ([A-Z0-9_]+)\n',line)
        end=re.fullmatch(r'C NCHECK_END ([A-Z0-9_]+)\n',line)
        if begin:
            require(active is None,'nested instrumentation block')
            active=begin.group(1);count+=1
        elif end:
            require(active==end.group(1),'unbalanced instrumentation block')
            active=None
        else:
            require(not line.startswith(('C NCHECK_BEGIN','C NCHECK_END')),'malformed marker')
            if active is None:
                out.append(line)
    require(active is None,'unterminated instrumentation block')
    return ''.join(out).encode('ascii'),count


def subroutine(text,name):
    anchor='      SUBROUTINE '+name+'('
    require(text.count(anchor)==1,'subroutine anchor differs: '+name)
    start=text.index(anchor);end=re.search(r'^      END[ \t]*$',text[start:],re.M)
    require(end is not None,'subroutine end missing: '+name)
    return text[start:start+end.end()]


def once(text,old,new):
    require(text.count(old)==1,'literal transformation anchor differs')
    return text.replace(old,new,1)


def replay_source(data,document,amp_data,matched_data):
    proof=document('science/source-proof.json');policy=document('plan/policy.json')
    require(proof['schema']=='private-checked-source-inverse-v1' and proof['base_sha256']==AMPLITUDE and
            proof['plan_sha256']==PLAN and proof['base_payloads']==55 and proof['new_native_actions']==0 and
            proof['card_inverse'] and proof['protected_legacy8_restored'] and not proof['scientific_arithmetic_changed'],
            'source inverse claim differs')
    require(set(proof['sources'])==set(SIX),'source inverse domain differs')
    counts={}
    for name in SIX:
        old=amp_data('proposal/candidate/'+name);new=data('science/candidate/'+name)
        restored,count=strip_markers(new);row=proof['sources'][name]
        require(restored==old and pin(old)==row['original'] and pin(new)==row['derived'] and
                row['exact_inverse'] and row['instrumentation_blocks']==count,'fresh exact inverse failed: '+name)
        patch=''.join(difflib.unified_diff(old.decode('ascii').splitlines(True),new.decode('ascii').splitlines(True),
                    fromfile='candidate2/'+name,tofile='checked/'+name)).encode('ascii')
        require(patch==data('science/'+name+'.patch'),'source patch differs: '+name)
        counts[name]=count
    require(counts=={'CT_interface.f':40,'TIRUST_interface.f':18,'TIR_interface.f':2,
                     'loop_matrix.f':14,'loop_CT_calls_1.f':0,'mp_compute_loop_coefs.f':2},'insertion scope differs')
    original=amp_data('proposal/inputs/cms-backend9-mode4-MadLoopParams.dat')
    require(original.count(b'#CTModeRun\n4\n')==1,'literal original card anchor differs')
    card=original.replace(b'#CTModeRun\n4\n',b'#CTModeRun\n-1\n',1)
    require(card==data('science/inputs/backend9-auto-MadLoopParams.dat') and
            pin(original)==policy['literal_card_source_pin'] and pin(card)==policy['planned_literal_card_pin'],
            'literal card one-value derivative differs')
    values={}
    lines=card.decode('ascii').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('#'):
            require(line[1:] not in values and i+1<len(lines),'duplicate/incomplete card value')
            values[line[1:]]=lines[i+1]
    require(values==policy['card_values'],'literal card policy differs')
    for name in ('cms-point1.dat','cms-point20.dat','cms-param_card.dat'):
        raw=amp_data('proposal/inputs/'+name)
        require(data('science/inputs/'+name)==data('controller1/inputs/'+name)==data('controller2/inputs/'+name)==raw,
                'physical/point input changed')
    for prefix in ('controller1','controller2'):
        require(data(prefix+'/backend9-auto-MadLoopParams.dat')==card,'controller card differs')
    require(data('science/expected-routes.json')==matched_data('science/expected-routes.json'),'route table ancestry differs')
    driver=data('science/driver-checked.f').decode('ascii')
    require(driver.count('CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)')==1 and
            driver.count('CALL ML5_0_SLOOPMATRIX_THRES(')==2 and '1000+NPOINTS' not in driver,
            'bounded checked driver policy differs')
    require(proof['driver']['checked_policy']=={'force_stability_check':True,'long_warmup_removed':True,
            'requested_precision':'-1D0','thres_requests':2},'driver policy proof differs')
    helpers=amp_data('proposal/driver-amplitude.f').decode('ascii')
    card_helper=subroutine(helpers,'KTRACE_DRIVER_CARD')
    card_helper=once(card_helper,'      INTEGER MODE,BACKEND\n',
        '      INTEGER MODE,BACKEND\n      LOGICAL NC_BYPASS,NC_FORCE\n'
        '      COMMON/ML5_0_BYPASS_CHECK/NC_BYPASS,NC_FORCE\n')
    card_helper=once(card_helper,'      IF(LoopPoleMode.NE.0) STOP 24\n',
        '      IF(LoopPoleMode.NE.0.OR.MODE.NE.-1.OR.BACKEND.NE.9)\n'
        '     $ STOP 24\n      IF(.NOT.NC_FORCE.OR.NC_BYPASS) STOP 24\n'
        '      IF(NRotations_DP.NE.0.OR.NRotations_QP.NE.0) STOP 24\n'
        '      IF(MLStabThres.NE.1.000000D-03) STOP 24\n'
        '      IF(ImprovePSPoint.NE.2.OR.CheckCycle.NE.3) STOP 24\n')
    state=subroutine(helpers,'KTRACE_DRIVER_STATE')
    state=once(state,"      IF(STAGE.EQ.'A'.AND.(CODE.EQ.110.OR.CODE.EQ.120))\n",
        "      IF(STAGE.EQ.'A'.AND.(CODE.EQ.219.OR.CODE.EQ.323))\n")
    state=once(state,'C     returning an unchecked fixed-mode code. INIT has then been restored.\n',
        'C     returning a recognized checked code. INIT has then been restored.\n')
    values=subroutine(helpers,'KTRACE_DRIVER_VALUES');reader=subroutine(helpers,'KTRACE_READ_POINT')
    main=data('science/driver-main.f.in').decode('ascii')
    derived=main.rstrip()+'\n\n'+'\n\n'.join((card_helper,state,values,reader))+'\n'
    require(derived==driver and pin(values.encode())==proof['driver']['raw_value_helper_unchanged'] and
            pin(reader.encode())==proof['driver']['point_reader_unchanged'],'fresh bounded driver derivation differs')
    require(proof['declared_compile_objects']==COMPILE_ORDER,'declared object order differs')
    return {'fresh_marker_inverse_files':6,'instrumentation_blocks':counts,'literal_card_changes':{'CTModeRun':['4','-1']},
            'fresh_marker_inverse_only_not_generator_execution':True,'original_backend8_bytes_restored':True,
            'source_bound_routes':26,'fresh_bounded_driver_derivation':True,'declared_compile_order':COMPILE_ORDER,'external_source_generation_reexecuted':False}


def closed_stage(data,prefix,stage,sha,source,passed,count,ncommands,nloaders):
    closure=json.loads(checked(data,prefix+'/'+stage+'-closure.json',sha))
    require(closure['stage']==stage and closure['source_index_sha256']==source and closure['passed'] is passed and
            closure['postcheck_failure'] is None and len(closure['payloads'])==count and
            closure['owned_command_count']==ncommands and closure['loader_query_count']==nloaders and
            not closure['host_stability_qualified'] and not closure['performance_measurement'],'closed stage scope differs')
    commands={}
    for name,expected in closure['payloads'].items():
        raw=checked(data,prefix+'/'+safe(name),expected)
        if name.startswith('execution/') and name.endswith('.json'):
            item=json.loads(raw);commands[name]=item
            require(item['cleanup_exception'] is None and item['cleanup']['survivors']==[],'command cleanup failed')
            log=name[:-5]+'.txt'
            require(log in closure['payloads'] and pin(data(prefix+'/'+log))['sha256']==item['log_sha256'],'command/log join differs')
            require(item['cpu']==12 and item['cpu_exclusive'] is False and item['argv'][:3]==['taskset','-c','12'],
                    'owned placement differs')
    require(len(commands)==ncommands and sum('--list' in c['argv'] for c in commands.values())==nloaders,'command count differs')
    require(data(prefix+'/'+stage+'-before.json')==data(prefix+'/'+stage+'-after.json'),'protected state changed')
    return closure,commands


def verify_successor(data,document,source1,source2,matched_data,legacy_data):
    for name in ('native_job.py','limit_exec.py','backend9-auto-MadLoopParams.dat'):
        require(data('controller1/'+name)==data('controller2/'+name),'successor changed unchanged native/policy input: '+name)
    for name in ('inputs/cms-param_card.dat','inputs/cms-point1.dat','inputs/cms-point20.dat'):
        require(data('controller1/'+name)==data('controller2/'+name),'successor changed numerical input')
    old_plan=document('controller1/validation-plan.json');new_plan=document('controller2/validation-plan.json')
    require({k:v for k,v in old_plan.items() if k not in ('schema','build')}==
            {k:v for k,v in new_plan.items() if k not in ('schema','build','available_stages','executor','prior_failed_run_sha256')},
            'successor changed checked policy/request/cap')
    require(new_plan['available_stages']==['run'] and new_plan['executor']=='kinematic-normal-checked-executor-2' and
            new_plan['prior_failed_run_sha256']==FAILED_RUN and new_plan['build']=={
                'loader_queries':0,'native_main_calls':0,'new_compilations':0,'new_links':0,
                'original_receipt_sha256':BUILD_RECEIPT,'original_source_index_sha256':CONTROLLER1,
                'reuse_only':True,'reviewed_closure_sha256':BUILD},'run-only build reuse policy differs')
    old=data('controller1/worker_entry.py')
    expected=old.replace(b'kinematic-normal-checked-executor-1',b'kinematic-normal-checked-executor-2')
    require(old.count(b'kinematic-normal-checked-executor-1')==1 and data('controller2/worker_entry.py')==expected,
            'native owner entry changed beyond executor identity')
    require(data('controller1/limit_exec.py')==matched_data('controller/limit_exec.py')==legacy_data('source/limit_exec.py'),
            'inherited limiter changed')
    # Authenticate the RUN-only guard structurally without importing/running it.
    tree=ast.parse(data('controller2/run.py'));functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    require('build' not in functions and 'build_checked' not in functions,'successor gained build implementation')
    main=ast.get_source_segment(data('controller2/run.py').decode(),functions['main'])
    setup=ast.get_source_segment(data('controller2/run.py').decode(),functions['setup_owned'])
    require("parser.add_argument('stage', choices=('run',))" in main and
            "require(stage == 'run', 'Successor permits RUN only; compilation is forbidden')" in setup and
            "L.STAGES = ('run',)" in setup and "require(args == loader or (len(args) == 6" in setup,
            'run-only command boundary differs')
    require(source1['native_actions']==source2['native_actions']==0 and
            source1['schema']=='private-normal-checked-controller-source-index-v1' and
            source2['schema']=='private-normal-checked-run-only-controller-source-index-v1','source seal scope differs')
    return {'run_only_successor':True,'native_owner_unchanged':True,'worker_entry_only_executor_path_changed':True,
            'source_controller_payloads':[24,23],'new_controller_execution':False}


def verify_build(data,document,ancestors):
    """API: ancestors[full_public_packet_name]=(authenticated_index,data,document)."""
    _,ad,aj=ancestors[AMP];_,md,mj=ancestors[MATCHED];_,ld,lj=ancestors[LEGACY]
    science=seal(data,'science',SCIENCE,36);plan=seal(data,'plan',PLAN,6)
    first=seal(data,'controller1',CONTROLLER1,24);second=seal(data,'controller2',CONTROLLER2,23)
    seal(ad,'proposal',AMPLITUDE,55)
    require(science['plan_index_sha256']==PLAN and science['scientific_base_index_sha256']==AMPLITUDE,
            'scientific seal ancestry differs')
    source_proof=replay_source(data,document,ad,md)
    successor=verify_successor(data,document,first,second,md,ld)
    ancestry={
        'amplitude_source':(ad,'proposal/index.json'),'approved_plan':(data,'plan/index.json'),
        'candidate_build':(ad,'results/builds/candidate/build.json'),'candidate_build_closure':(ad,'results/build-closure.json'),
        'matched_comparison':(md,'results/matched-comparison.json'),'matched_controller':(md,'controller/index.json'),
        'matched_run_closure':(md,'results/run-closure.json'),'ordinary_closure':(ad,'results/ordinary-closure.json'),
        'ordinary_comparison':(ad,'results/ordinary-comparison.json'),'scientific_source':(data,'science/index.json')}
    for prefix in ('controller1','controller2'):
        bindings=document(prefix+'/bindings.json')
        targets=dict(ancestry)
        if prefix=='controller2':
            targets.update({'checked_build':(data,'results1/builds/checked/build.json'),
                'checked_build_closure':(data,'results1/build-closure.json'),'checked_build_summary':(data,'results1/build-summary.json'),
                'prior_controller':(data,'controller1/index.json'),'prior_failed_run':(data,'results1/run-closure.json')})
        require(set(bindings['files'])==set(targets) and not bindings['scientific_source_pending'],'binding domain differs')
        for key,(reader,logical) in targets.items():
            require(pin(reader(logical))==identity(bindings['files'][key]),'binding ancestry differs: '+key)
    closure,commands=closed_stage(data,'results1','build',BUILD,CONTROLLER1,True,138,64,1)
    require(closure['failure'] is None and closure['reviewed_build_sha256'] is None,'build failure/admission differs')
    require(all(c['returncode']==0 and c['exception'] is None for c in commands.values()),'build command failed')
    programs=Counter(c['argv'][3].split('/')[-1] for c in commands.values())
    require(programs=={'make':10,'gfortran':53,'ld-linux-x86-64.so.2':1},'unexpected build-owned program')
    query_kinds=[]
    for c in commands.values():
        if c['argv'][3].endswith('/gfortran'):
            require(len(c['argv'])==5,'compiler query argv shape differs')
            arg=c['argv'][-1]
            query_kinds.append('version' if arg=='--version' else 'search' if arg=='-print-search-dirs'
                               else 'file' if arg.startswith('-print-file-name=') else 'unknown')
    require(Counter(query_kinds)=={'version':2,'search':1,'file':50},'compiler query kinds differ')
    owned_ids=sorted({(row['pid'],row['start_time_ticks']) for c in commands.values()
                      for row in c['cleanup']['observed_owned_processes']})
    donor=aj('results/builds/candidate/build.json')
    require(pin(ad('results/builds/candidate/build.json'))==aj('results/build-closure.json')['payloads']['builds/candidate/build.json'],
            'donor receipt not closed')
    before=document('results1/builds/checked/before-compile.json');attempt=document('results1/builds/checked/attempt.json')
    build=json.loads(checked(data,'results1/builds/checked/build.json',BUILD_RECEIPT))
    link=document('results1/builds/checked/link-variant-receipt.json');prelink=document('results1/builds/checked/prelink.json')
    runtime=document('results1/runtime/checked.json');summary=json.loads(checked(data,'results1/build-summary.json',BUILD_SUMMARY))
    # The original f172 receipt lacks the summary-level scientific-source field.
    require('scientific_source_index_sha256' not in build and summary['scientific_source_index_sha256']==SCIENCE and
            summary['checked_build']==pin(data('results1/builds/checked/build.json')),'unaltered original build-summary linkage differs')
    require(summary['executables']==summary['actual_make_jobs']==1 and summary['checked_trace_only'] and
            summary['existing_modules_preserved'] and not summary['private8_geometry_policy_changed'] and
            not summary['scientific_library_rebuilds'] and summary['native_main_calls']==summary['tensor_calls']==0,'build summary scope differs')
    require(build['case']==link['case']=='cms' and build['variant']==link['variant']=='checked' and
            build['output']==link['output']==OUTPUT and build['compiled_targets']==summary['compiled_objects']==TARGETS,'target identity differs')
    require(build['fortran_main_calls']==build['tensor_evaluations']==0 and not any(build[k] for k in
            ('library_rebuilds','independent_accuracy_qualified','performance_measurement')) and
            build['unmodified_numerical_arithmetic'] is link['unmodified_numerical_arithmetic'] is False,
            'original low-level build scope differs')
    require(build['sources']==before['sources']==attempt['sources_before']==link['sources'],'source inventory changed during build')
    current,old=source_inventory(build),source_inventory(donor)
    changed={SUB+n for n in ('check_sa.f','loop_matrix.f','mp_compute_loop_coefs.f','CT_interface.f','TIR_interface.f','TIRUST_interface.f','makefile')}
    require(len(current)==275 and len(old)==274 and set(current)-set(old)=={SUB+'checked_trace.f90'} and
            set(old)<=set(current) and {n for n in old if current[n]!=old[n]}==changed,'source change set differs')
    for name in SIX:
        require(current[SUB+name]==pin(data('science/candidate/'+name)),'built scientific source differs')
    require(current[SUB+'check_sa.f']==pin(data('science/driver-checked.f')) and
            current[SUB+'checked_trace.f90']==pin(data('science/checked_trace.f90')) and
            build['physical_card']==link['physical_card']==pin(data('science/inputs/cms-param_card.dat')),'built driver/recorder/card differs')
    objects=attempt['objects_before'];changed_objects={SUB+n for n in TARGETS[:-1]}
    require(objects==donor['all_objects_after'] and len(objects)==211 and len(build['all_objects_after'])==212 and
            set(build['all_objects_after'])-set(objects)=={SUB+'checked_trace.o'} and
            {n for n in objects if objects[n]!=build['all_objects_after'][n]}==changed_objects and
            build['preserved_object_hashes']=={n:v for n,v in objects.items() if n not in changed_objects} and
            len(build['preserved_object_hashes'])==205,'preserved object receipt differs')
    require(build['objects']==link['objects'] and len(build['objects'])==24 and
            all(build['all_objects_after'][n]['sha256']==v for n,v in build['objects'].items()),'link object inventory differs')
    require(build['reused_libraries']==attempt['reused_libraries_before']==donor['reused_libraries'] and
            len(build['reused_libraries'])==4,'archive preservation differs')
    require(build['module_inputs']==donor['modules_after'] and len(build['module_inputs'])==10 and
            len(build['modules_after'])==11 and set(build['modules_after'])-set(build['module_inputs'])==
            {SUB+'ml_private_checked_trace.mod'} and all(build['modules_after'][n]==v for n,v in build['module_inputs'].items()),
            'module preservation differs')
    require(build['compiler_flags']==before['compiler_flags']==link['compiler_flags'],'compiler snapshots differ')
    for scope,count in (('MODEL',14),('DHELAS',175),('process',25)):
        recipes=build['compiler_flags']['scopes'][scope]['recipes'];prior=donor['compiler_flags']['scopes'][scope]['recipes']
        require(len(recipes)==count and all(recipes[n]==v for n,v in prior.items()),'existing compiler recipe changed')
        if scope=='process':
            require(set(recipes)-set(prior)=={'checked_trace.f90'} and recipes['checked_trace.f90']==recipes['check_sa.f'],
                    'recorder compiler flags differ')
    require(build['build_commands']==attempt['commands'] and len(build['build_commands'])==5 and
            attempt['failure'] is None and all(attempt[k] for k in ('compile_seen','link_seen','runtime_seen')) and
            attempt['fortran_main_calls']==attempt['tensor_evaluations']==0 and
            build['c_dependencies'] is attempt['c_dependencies'] is None,'closed build attempt differs')
    transforms={row['label']:row for row in summary['inherited_make_job_requests_and_actual_transform']}
    require(set(transforms)=={'compile-cms-checked','link-cms-checked'},'make override domain differs')
    for request in build['build_commands']:
        command=commands['execution/build-'+request['label']+'.json']
        actual=['-j1' if a=='-j2' else a for a in request['argv']]
        tail=['SHELL=/nix/store/1mv3qz005gkbalxfghk4y3s8ayv4d2dl-bash-interactive-5.3p15/bin/bash'] if actual[0]=='make' else []
        require(command['argv']==['taskset','-c','12']+actual+tail and command['cwd']==request['cwd'],'actual build argv differs')
        if request['label'] in transforms:
            require(transforms[request['label']]['inherited_request']==request['argv'] and
                    transforms[request['label']]['actual_argv']==actual and '-j1' in actual,'make single-job transform differs')
    compiled=[]
    for line in data('results1/execution/build-compile-cms-checked.txt').decode('utf-8').splitlines():
        args=shlex.split(line)
        require(args and args[0].endswith('/gfortran') and args.count('-c')==args.count('-o')==1,'unexpected compile log command')
        source=args[args.index('-c')+1];target=args[args.index('-o')+1]
        require(source.rsplit('.',1)[0]+'.o'==target,'compile source/target differs')
        compiled.append(target)
    require(compiled==COMPILE_ORDER,'actual compile order differs')
    require(prelink['completed_before_link'] and len(prelink['bound'])==152 and prelink['bound']==build['prelink_bound']==link['prelink_bound'] and
            all(prelink['bound'].get(n)==v for n,v in prelink['closure']['files'].items()),'prelink binding differs')
    require(build['actual_link_loads']==link['actual_link_loads'] and len(build['actual_link_loads'])==57 and
            all(prelink['bound'].get(n)==v['sha256'] for n,v in build['actual_link_loads'].items()),'actual LOAD not prebound')
    require(build['runtime']==link['runtime']==runtime['runtime'] and runtime['passed'] and runtime['failure'] is None and
            runtime['loader_query_started'] and runtime['actual_interpreter']==build['runtime']['loader'] and
            build['runtime']['files']==donor['runtime']['files'] and len(build['runtime']['files'])==15,'runtime provider/PT_INTERP receipt differs')
    supplemental=aj('proposal/bindings.json')['fixed_supplemental_providers']
    allowed={**prelink['bound'],**{n:v['sha256'] for n,v in supplemental.items()}}
    require(all(allowed.get(n)==v for n,v in build['runtime']['files'].items()) and
            build['runtime']['files'][build['runtime']['loader']]==build['runtime']['loader_sha256'],'runtime provider not admitted')
    require(build['executable']==link['executable']==runtime['executable']==summary['executable']==OUTPUT+'/'+SUB+'check' and
            build['executable_pin']==link['executable_pin']==runtime['executable_pin']==summary['executable_pin'],'ELF identity join differs')
    name,loader=next((n,c) for n,c in commands.items() if '--list' in c['argv'])
    require(loader['argv']==['taskset','-c','12',build['runtime']['loader'],'--list',build['executable']] and
            data('results1/'+name[:-5]+'.txt').decode()==build['runtime']['loader_listing'],'loader listing join differs')
    protected=document('results1/build-before.json')
    require(protected['source_index']==pin(data('controller1/index.json')) and protected['bindings']==pin(data('controller1/bindings.json')) and
            protected['scientific_source_index']==pin(data('science/index.json')) and protected['upstream']==aj('results/build-before.json'),
            'protected source/ancestor join differs')
    admitted=protected['candidate_donor_current']
    require(admitted['receipt']==pin(ad('results/builds/candidate/build.json')) and admitted['executable']==donor['executable_pin'] and
            admitted['current_files_verified'] and admitted['native_actions']==0 and
            (admitted['libraries'],admitted['modules'],admitted['objects'],admitted['prelink_inputs'],admitted['providers'])==(4,10,211,151,15),
            'historical donor admission differs')
    terminal=document('results1/build-terminal-review.json');root_review=document('results1/root-build-review.json')
    require(terminal['passed'] and terminal['terminal_exit_code']==0 and terminal['build_closure']==pin(data('results1/build-closure.json')) and
            terminal['build_receipt']==pin(data('results1/builds/checked/build.json')) and terminal['protected_equal'] and
            terminal['protected_before']==terminal['protected_after']==pin(data('results1/build-before.json')) and
            terminal['remaining_owned_identities']==[] and terminal['shared_lock_free_after_terminal'] and
            terminal['owned_process_identities']==[list(row) for row in owned_ids] and len(owned_ids)==85 and terminal['source_index']==CONTROLLER1 and
            terminal['scientific_source_index']==SCIENCE and terminal['fortran_main_calls']==terminal['tensor_calls']==0,
            'historical build terminal review differs')
    require(root_review['passed'] and root_review['build_closure_sha256']==BUILD and root_review['commands']==64 and
            root_review['owned_start_identities_absent']==85 and root_review['lock_free'] and root_review['physics_calls']==0 and
            root_review['actual_interpreter']==runtime['actual_interpreter'],'historical root build review differs')
    failed,fcommands=closed_stage(data,'results1','run',FAILED_RUN,CONTROLLER1,False,20,2,1)
    require(failed['failure']=="RuntimeError('Command returned 1')" and failed['reviewed_build_sha256']==BUILD,'failed first run altered')
    gate=document('results1/runs/'+WORKER+'/bootstrap-gate.json');result=document('results1/runs/'+WORKER+'/result.json')
    require(gate['passed'] is False and gate['measurement_requested'] is False and gate['failures']==["'scientific_source_index_sha256'"] and
            result['bootstrap_gate_passed'] is False and not result['measurement_command_attempted'] and not result['measurement_completed'] and
            result['native_returncode']==0 and [row[0] for row in result['protocol']]==['READY','BYE'] and
            result['failure']['message']=='Checked bootstrap gate failed; measurement not requested','first-run KeyError/no-measurement evidence differs')
    require(len(result['runtime_maps'])==1 and result['runtime_maps'][0]['boundary']=='READY' and
            result['close_receipts']==[{'complete':True,'graceful_requested':True,'pid':result['native_pid'],
                                      'returncode':0,'worker_closed':True}] and result['cleanup_failure'] is None,
            'first-run graceful closure differs')
    require(sorted(c['returncode'] for c in fcommands.values())==[0,1],'first-run command outcomes differ')
    failed_terminal=document('results1/run-terminal-review.json')
    require(failed_terminal['passed'] is False and failed_terminal['terminal_exit_code']==1 and
            failed_terminal['run_closure']==pin(data('results1/run-closure.json')) and not failed_terminal['measurement_requested'] and
            not failed_terminal['measurement_completed'] and failed_terminal['retention_cleanup_passed'] and
            failed_terminal['remaining_owned_identities']==[] and failed_terminal['shared_lock_free_after_terminal'],
            'failed first run terminal evidence differs')
    run,rcommands=closed_stage(data,'results2','run',RUN,CONTROLLER2,True,26,2,1)
    require(run['failure'] is None and run['reviewed_build_sha256']==BUILD and
            all(c['returncode']==0 and c['exception'] is None for c in rcommands.values()),'successor run closure differs')
    for prefix,controller in (('results1','controller1'),('results2','controller2')):
        snapshot=document(prefix+'/run-before.json')
        require(snapshot['source_index']==pin(data(controller+'/index.json')) and
                snapshot['bindings']==pin(data(controller+'/bindings.json')) and
                snapshot['scientific_source_index']==pin(data('science/index.json')),'run protected source binding differs')
    terminal2=document('results2/run-terminal-review.json');root2=document('results2/root-run-review.json')
    require(terminal2['passed'] and terminal2['terminal_exit_code']==0 and
            terminal2['run_closure']==pin(data('results2/run-closure.json')) and terminal2['original_failed_run_preserved'] and
            terminal2['protected_equal'] and terminal2['protected_before']==terminal2['protected_after']==pin(data('results2/run-before.json')) and
            terminal2['remaining_owned_identities']==[] and terminal2['shared_lock_free_after_terminal'] and
            not terminal2['failed_tensor_outputs_read'] and not terminal2['performance_qualified'],
            'retained successor terminal provenance differs')
    donor2=root2['donor_ancestry']
    require(root2['run_closure']==pin(data('results2/run-closure.json')) and root2['original_run_remains_failed'] and
            donor2['build_closure']==pin(data('results1/build-closure.json')) and
            donor2['build_receipt']==pin(data('results1/builds/checked/build.json')) and
            donor2['build_summary']==pin(data('results1/build-summary.json')) and
            donor2['build_source_index']==pin(data('controller1/index.json')) and
            donor2['scientific_source_index']==pin(data('science/index.json')) and
            donor2['prior_failed_run']==pin(data('results1/run-closure.json')) and
            donor2['prior_run_remains_failed'] and not donor2['prior_measurement_requested'],
            'root retained donor ancestry differs')
    # Do not inspect successful or failed coefficients here. Worker scientific replay is separate.
    return {'passed':True,'source_payloads':36,'plan_payloads':6,'controller_payloads':[24,23],
            'source_inverse':source_proof,'successor':successor,'closed_build_payloads':138,
            'owned_build_commands':64,'owned_build_programs':dict(programs),'recorded_owned_process_identities':85,'loader_queries_at_build':1,'compiled_objects':TARGETS,'actual_compile_order':compiled,
            'recorded_sources':275,'preserved_objects':205,'unchanged_old_modules':10,'reused_archives':4,
            'prelink_inputs':152,'recorded_actual_LOADs':57,'recorded_runtime_providers':15,
            'unchanged_original_build_receipt':BUILD_RECEIPT,'scientific_source_join_via_summary':BUILD_SUMMARY,
            'failed_first_run_preserved':True,'failed_first_run_payloads':20,'first_run_CMS20_requested':False,
            'closed_successor_run_payloads':26,'historical_process_and_lock_reviews_only':True,
            'new_ELF_PT_INTERP_parse':False,'new_external_artifact_hashing':False,'new_native_executions':0,
            'scientific_output_arithmetic_performed':False,'normal_host_stability_qualified_here':False,'performance_qualified':False}
