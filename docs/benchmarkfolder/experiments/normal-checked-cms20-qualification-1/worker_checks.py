"""Portable replay of the sealed checked-run scientific gates.

All reads go to authenticated retained-byte callbacks. Absolute historical paths
are provenance strings only. Original scientific function ASTs are compiled
unchanged; native/controller entrypoints and live filesystem access are excluded.
"""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct
from types import ModuleType, SimpleNamespace

ROOT='/shared/external/madloop_tirust/madgraph7_madloop_tirust'
SCIENCE='7dab8a04296a44e6eef1cf017bee751491220b77804ebe2a34faa328f85cb2c9'
C1='fd482474221715397a0dff7b47244ef25a07f077e79b4b8a47c5e3beca398c03'
C2='88cc660d0deecc3b8f87eba23bc10deed729a048cb4bddfcbe3ca4265f8830da'
BUILD='3071f3887244a34ebdf134e3867484cae6ddec9a6623309d3668d8817d26f09a'
FAILED='754078acb9700ffa10b9c040bc6c0de86784e4da66adaaef0e8fdcb8441985af'
RUN='046d83232bd40235f2401341c0806fb9446599b321da23e72b663b9430d68aa1'
COMPARISON='8ff34fa6a4648ebb84cab4b2cb8a0a1d2f3e25591c42b55aa8f398b308a27465'
AMP='kinematic-invariant-amplitude-qualification-1'
MATCH='matched-native-input-qualification-1'
DOUBLE='cms-universal-qed-pole-audit-1'
SINGLE='cms-universal-qed-single-pole-audit-1'
ANCESTORS={AMP:'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4',
 MATCH:'eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0',
 DOUBLE:'1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77',
 SINGLE:'99603d41d662e90f071b8052eb4303bcba8af451c5c96654febc276e620ad11e'}
DECIMAL_TOKEN=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z')


def require(ok,message):
    if not ok:raise ValueError('Checked portable worker: '+message)


def pin(raw):
    require(isinstance(raw,bytes),'Callback did not return bytes')
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def checked(data,name,expected):
    raw=data(name);actual=pin(raw)
    require(actual['sha256']==expected if isinstance(expected,str) else
            actual=={k:expected[k] for k in ('bytes','sha256')},'Changed retained bytes: '+name)
    return raw


def safe(name):
    require(isinstance(name,str) and name and not name.startswith('/') and '\\' not in name and
            all(p not in ('','.','..') for p in name.split('/')),'Unsafe logical path')
    return name


def definitions(raw,names,constants=(),*,injected=None,label='retained',cached=False):
    """Exact original function nodes, selected original pure assignments only."""
    tree=ast.parse(raw);nodes=[];found=set();assigned=set()
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            require(not node.decorator_list or cached and node.name=='references' and
                    len(node.decorator_list)==1 and ast.unparse(node.decorator_list[0])=='lru_cache(maxsize=1)',
                    'Unexpected scientific decorator')
            nodes.append(node);found.add(node.name)
        elif isinstance(node,ast.Assign):
            targets={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
            if targets and targets<=set(constants):nodes.append(node);assigned.update(targets)
    require(found==set(names) and assigned==set(constants),'Selected definitions/constants differ: '+label)
    result=ModuleType(label);result.__dict__.update(injected or {})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),label,'exec'),result.__dict__)
    return result


def verify_maps(raw,capture,expected):
    require(pin(raw)['sha256']==capture['sha256'] and capture['elf']==expected,'Map receipt differs')
    extra=capture['data'];observed=set();inodes={}
    for name,identity in extra.items():
        require(name.startswith('/nix/store/') and name.endswith('/lib/locale/locale-archive') and
                identity['bytes']>0 and re.fullmatch('[0-9a-f]{64}',identity['sha256']),'Unknown data mapping')
    for line in raw.decode('ascii').splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        name=fields[5]
        require(not name.endswith(' (deleted)') and int(fields[4])>0 and name in set(expected)|set(extra),
                'Unadmitted/deleted raw mapping')
        if name in extra:require(fields[1]=='r--p','Locale mapping permissions differ')
        observed.add(name);inodes.setdefault(name,set()).add(tuple(fields[3:5]))
    require(observed==set(expected)|set(extra) and all(len(x)==1 for x in inodes.values()),'Map coverage differs')


def verify_workers(data,document,ancestors):
    def doc(name,expected=None):
        raw=data(name) if expected is None else checked(data,name,expected)
        value=json.loads(raw);require(value==document(name),'Byte/document callback differs: '+name)
        return value
    sources={};source_indices={}
    for prefix,sha,count in (('science',SCIENCE,36),('controller1',C1,24),('controller2',C2,23)):
        index=doc(prefix+'/index.json',sha);require(len(index['files'])==count,'Source inventory differs')
        source_indices[prefix]=index
        sources[prefix]={name:checked(data,prefix+'/'+safe(name),record) for name,record in index['files'].items()}
    stages={}
    for name,sha,passed in (('results1/build',BUILD,True),('results1/run',FAILED,False),('results2/run',RUN,True)):
        close=doc(name+'-closure.json',sha);require(close['passed'] is passed and close['postcheck_failure'] is None,
            'Historical stage status differs')
        for member,record in close['payloads'].items():checked(data,name.split('/')[0]+'/'+safe(member),record)
        stages[name]=close
    require(stages['results1/run']['source_index_sha256']==C1 and stages['results2/run']['source_index_sha256']==C2,
            'Run source differs')
    require(stages['results1/run']['failure']=="RuntimeError('Command returned 1')" and
            stages['results2/run']['failure'] is None,'Historical failure reclassified')
    build_raw=checked(data,'results1/builds/checked/build.json',stages['results1/build']['payloads']['builds/checked/build.json'])
    build=json.loads(build_raw);build_pin=pin(build_raw)
    require(build_pin['sha256']=='f172832d12ef1a568467fe44bd3ec2c80f8f9717c628e1ab689a3773a57e0145' and
            'scientific_source_index_sha256' not in build,'Original build schema relabeled')
    summary=doc('results1/build-summary.json');require(summary['checked_build']==build_pin and
            summary['scientific_source_index_sha256']==SCIENCE,'Corrected donor provenance differs')
    plan=json.loads(sources['controller2']['validation-plan.json']);requests=plan['requests']
    require(len(requests)==1 and requests[0]['id']=='cms-p20-auto-lib9-checked','Unexpected worker scope')
    request=requests[0];identifier=request['id'];work_prefix={n:f'.tirust-build/kinematic-normal-checked-executor-{n}/' for n in (1,2)}
    before=doc('results2/run-before.json');require(before==doc('results2/run-after.json'),'Protected source snapshot differs')
    source_evidence={'controller_index_sha256':C2,'scientific_index_sha256':SCIENCE,
        'controller_payloads':23,'scientific_payloads':36,'preflight':before['verified'],
        'preparation_only':False,'controller_snapshot':None,
        'criterion_source_sha256':'ecce1e1f99f0de7d1798880cc9137251b4e553accc708f3cee357ed794b99272'}
    require(before['verified']['passed'] and before['verified']['donor_build']['build_receipt']==build_pin and
            before['verified']['donor_build']['scientific_source_index']['sha256']==SCIENCE,
            'Historical source preflight does not join donor/science')

    # The enclosing packet reader authenticated each ancestor; additionally bind
    # its exact canonical index and every stored file this helper may consume.
    ancestor_bytes={}
    for name,sha in ANCESTORS.items():
        index,callback,documents=ancestors[name]
        raw=callback('evidence-index.json')
        require(pin(raw)['sha256']==sha and json.loads(raw)==index,'Ancestor index changed: '+name)
        for member,record in index['files'].items():checked(callback,safe(member),record)
        ancestor_bytes[name]=raw
    amp=ancestors[AMP][1];matched=ancestors[MATCH][1]
    registries=[
        ('.tirust-build/kinematic-normal-checked-source-1/','science/',data),
        ('.tirust-build/kinematic-normal-checked-controller-1/','controller1/',data),
        ('.tirust-build/kinematic-normal-checked-controller-2/','controller2/',data),
        (work_prefix[1]+'results/','results1/',data),(work_prefix[2]+'results/','results2/',data),
        ('.tirust-build/kinematic-invariant-amplitude-proposal-2/','proposal/',amp),
        ('.tirust-build/kinematic-invariant-amplitude-executor-2/results/','results/',amp),
        ('.tirust-build/kinematic-matched-native-input-proposal-1/','science/',matched),
        ('.tirust-build/kinematic-matched-native-input-executor-2/results/','results/',matched)]
    for name in (DOUBLE,SINGLE):
        registries.append(('docs/benchmarkfolder/experiments/'+name+'/', '',ancestors[name][1]))
    view={'bootstrap2':False};reads=set()
    boot2=doc('results2/runs/'+identifier+'/bootstrap-gate.json')
    def read_path(path):
        original=str(path);require(original.startswith(ROOT+'/'),'Read outside declared repository provenance')
        relative=safe(original[len(ROOT)+1:]);reads.add(relative)
        for name in (DOUBLE,SINGLE):
            if relative=='docs/benchmarkfolder/experiments/'+name+'/evidence-index.json':return ancestor_bytes[name]
        for prefix,logical,callback in registries:
            if relative.startswith(prefix):
                name=logical+safe(relative[len(prefix):]);raw=callback(name)
                if view['bootstrap2'] and callback is data and name.startswith('results2/runs/'+identifier+'/'):
                    tail=name.rsplit('/',1)[-1]
                    if tail in boot2['source_evidence']:
                        identity=boot2['source_evidence'][tail]
                        if 'bytes' in identity:
                            raw=raw[:identity['bytes']];require(pin(raw)==identity,'READY snapshot prefix differs: '+tail)
                return raw
        return data('helpers/'+relative)

    class RetainedPath:
        """Lexical paths only, with no host filesystem methods or fallback."""
        def __init__(self,value):self.path=PurePosixPath(str(value))
        def __truediv__(self,value):return RetainedPath(self.path/str(value))
        def __str__(self):return str(self.path)
        def __eq__(self,other):return str(self)==str(other)
        def __hash__(self):return hash(str(self))
        @property
        def parts(self):return self.path.parts
        @property
        def name(self):return self.path.name
        @property
        def parent(self):return RetainedPath(self.path.parent)
        @property
        def parents(self):return [RetainedPath(x) for x in self.path.parents]
        def is_absolute(self):return self.path.is_absolute()
        def relative_to(self,other):return self.path.relative_to(str(other))
        def resolve(self):require(self.is_absolute(),'Unexpected relative resolve');return self
        def read_bytes(self):return read_path(self)
    R=RetainedPath(ROOT)
    old_index=json.loads(checked(amp,'proposal/index.json','ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'))
    old_raw=checked(amp,'proposal/amplitude_compare.py',source_evidence['criterion_source_sha256'])
    require(pin(old_raw)==old_index['files']['amplitude_compare.py'],'Legacy comparator membership differs')
    legacy=definitions(old_raw,{'require','pin','decode64','wire','_decimal_shape','compare_coefficients'},
        {'COLUMNS'},injected={'hashlib':hashlib,'re':re,'F':Fraction,'Decimal':Decimal,'localcontext':localcontext,
         'DECIMAL_TOKEN':DECIMAL_TOKEN,'TOL':Fraction(3,1000000),'TARGET':Fraction(1,100000000)},label='sealed-amplitude-math')
    trace=definitions(sources['science']['checked_trace.py'],
        {'require','integers','finite_word','exact_word','environment_valid','_routes','parse_trace','_word64','qualify_phase'},
        {'SCHEMA','MAX_BYTES','MAX_GROUPS','MAX_EVENTS','MAX_ERROR','HEX','HEX64','INTEGER','FIELDS','EVENT_SHAPES','META_TAGS','WORD_TAGS','CMS_SHAPES'},
        injected={'Fraction':Fraction,'hashlib':hashlib,'re':re},label='sealed-checked-trace')
    def geometry_module(name,raw):
        if name=='checked_pinned_factory':
            return definitions(raw,{'producer_kind_ok','environment_ok','upper_exp','psum','route_matches','build','finish'},
                               {'ZERO','MARK'},label=name)
        if name=='checked_pinned_certificate':
            return definitions(raw,{'pow2','floor2','psum','certificate','exact_error_bound','source_chord_bound_exp'},
                               {'ZERO'},injected={'F':Fraction},label=name)
        require(name=='checked_pinned_matched','Unexpected geometry module')
        tree=ast.parse(raw)
        functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}-{'verify_source_contract'}
        constants={v.id for n in tree.body if isinstance(n,ast.Assign) for t in n.targets for v in ast.walk(t) if isinstance(v,ast.Name)}
        return definitions(raw,functions,constants,injected={'hashlib':hashlib,'math':math,'re':re},label=name)
    geometry_names={'require','_pin','_packet','references','fmt','word','value','encode','textword','neg','add','sub','mul',
        '_complex_add','_complex_neg','_complex_sub','_external','replay_shifts','_flat_matrix','expected_metadata',
        '_pair_replay','validate_geometry','join_matched_mode4'}
    geometry=definitions(sources['science']['checked_geometry.py'],geometry_names,{'PACKETS','MATCHED_CLOSURE','HEX','NR'},
        injected={'ROOT':R,'MATCHED_RESULTS':R/'.tirust-build/kinematic-matched-native-input-executor-2/results',
                  'hashlib':hashlib,'json':json,'re':re,'F':Fraction,'lru_cache':lru_cache,'_module':geometry_module},
        label='sealed-source-geometry',cached=True)

    # Retained pole calculations retain their original functions and formulas.
    # The sole dynamic import in single.calculate is a fixed retained double
    # helper; this tiny loader accepts only that exact path and runs no imports.
    double_names={'require','digest','card_values','binary64','calculate'}
    def double_module(path,raw):
        return definitions(raw,double_names,injected={'D':Decimal,'localcontext':localcontext,'hashlib':hashlib,
            'json':json,'struct':struct,'HERE':path.parent},label='sealed-double-pole')
    def pole_module(path,raw):
        text=str(path)
        if text==str(R/'.tirust-build/kinematic-invariant-amplitude-proposal-2/amplitude_compare.py'):return legacy
        if text==str(R/('docs/benchmarkfolder/experiments/'+DOUBLE+'/source/calculate.py')):
            return double_module(path,raw)
        require(text==str(R/('docs/benchmarkfolder/experiments/'+SINGLE+'/source/calculate.py')),'Unadmitted pole module')
        expected=path.parent/'double-pole/calculate.py'
        def spec_from_file_location(name,p):
            require(name=='retained_double_pole_calculation' and p==expected,'Unadmitted retained Python import')
            raw_double=p.read_bytes()
            require(raw_double==ancestors[DOUBLE][1]('source/calculate.py'),'Copied double helper differs')
            obj=double_module(p,raw_double)
            def load_module(target):require(target is obj,'Retained module identity differs')
            return SimpleNamespace(module=obj,loader=SimpleNamespace(exec_module=load_module))
        util=SimpleNamespace(spec_from_file_location=spec_from_file_location,module_from_spec=lambda spec:spec.module)
        return definitions(raw,{'require','pin','binary128','dot','decimal','wire','load_trace_payloads','external_inputs',
            'verify_numerical_card','calculate'},{'TRACE_SEAL','DOUBLE_SEAL','CLOSURES'},
            injected={'D':Decimal,'F':Fraction,'localcontext':localcontext,'gzip':gzip,'hashlib':hashlib,
                      'json':json,'struct':struct,'HERE':path.parent,'importlib':SimpleNamespace(util=util)},label='sealed-single-pole')
    poles=definitions(sources['controller2']['pole_checks.py'],
        {'require','pin','checked','references','finite_word','_proof','_factor','_metric','check_poles'},
        {'AUDITS','OLD_SHA','COMPARE_SHA','HEX','HEX64'},
        injected={'ROOT':R,'OLD':R/'.tirust-build/kinematic-invariant-amplitude-proposal-2','Path':RetainedPath,
            'D':Decimal,'F':Fraction,'localcontext':localcontext,'hashlib':hashlib,'json':json,'re':re,'module':pole_module},
        label='sealed-pole-gates')
    routes={int(k):v for k,v in json.loads(sources['science']['expected-routes.json']).items()}
    scientific={'trace':trace,'geometry':geometry,'poles':poles,'routes':routes,'legacy':legacy,
                'verifier':None,'evidence':source_evidence}

    def retained_spec(directory,spec,controller_sha,supplied_build,supplied_pin,*,historical=False):
        number=1 if historical else 2;prefix=f'results{number}/runs/{identifier}/'
        require(str(directory)==ROOT+'/'+work_prefix[number]+'results/runs/'+identifier,'Specification directory differs')
        raw=checked(data,prefix+'specification.json',pin((json.dumps(spec,indent=2,sort_keys=True)+'\n').encode()))
        require(spec['request']==request and controller_sha==(C1 if historical else C2) and
                spec['source_index_sha256']==controller_sha and spec['scientific_source_index_sha256']==SCIENCE and
                spec['build_receipt_sha256']==build_pin['sha256'] and supplied_build==build and supplied_pin==build_pin,
                'Specification source/build scope differs')
        if not historical:
            require(spec['donor_build_closure_sha256']==BUILD and spec['donor_build_source_index_sha256']==C1,'Donor ancestry differs')
        cwd=ROOT+'/'+work_prefix[number]+'work/workers/'+identifier+'/SubProcesses/P0_epem_mupmum'
        require(spec['cwd']==cwd and spec['executable']==cwd+'/check' and spec['evidence']==str(directory) and
                spec['executable_pin']==build['executable_pin'],'Worker path/executable identity differs')
        require(spec['expected_runtime']=={**build['runtime']['files'],spec['executable']:build['executable_pin']['sha256']},
                'Admitted provider set differs')
        evidence={'specification.json':pin(raw),'build_receipt':build_pin}
        for key,field in (('card','numerical_card'),('physical_card','physical_card'),('bootstrap','bootstrap_pin'),('measurement','point_pin')):
            name=request[key];source=sources['controller2'][name]
            require(source==sources['controller1'][name] and source==sources['science']['inputs/'+name.rsplit('/',1)[-1]],
                    'Retained scientific literal differs')
            require(spec[field]==pin(source),'Specification literal pin differs');evidence[field]=pin(source)
        require(spec['bootstrap_file']==cwd+'/bootstrap-input.dat' and spec['point_file']==cwd+'/measurement-input.dat',
                'Materialized point path differs')
        return evidence
    def retained_build(runs,provided=None,*,verifier):
        require(provided is None or provided==build,'Provided donor differs');return build,build_pin
    def retained_sources(expected,*,preparation=False):
        require(expected==C2 and not preparation,'Only sealed historical source is replayed');return scientific
    outer_names={'require','pin','checked','validate_requests','_protocol','states','coverage','_outer','_phase','_saved','_view',
        '_poles','_evaluate_phase','_bootstrap_snapshot_gate','_closed','compare_stage'}
    constants={'DONOR_CONTROLLER_SHA','DONOR_BUILD_SHA','FAILED_RUN_SHA','SCIENCE_SHA','OLD_SHA','OLD_COMPARE_SHA',
        'ORDINARY_CLOSURE_SHA','ORDINARY_COMPARE_SHA','MATCHED_CLOSURE_SHA','MATCHED_COMPARE_SHA','OUTPUTS','BOOT_OUTPUTS','HEX64','DECIMAL','REQUEST'}
    outer=definitions(sources['controller2']['compare.py'],outer_names,constants,
        injected={'hashlib':hashlib,'json':json,'re':re,'Path':RetainedPath,
          'ORDINARY':R/'.tirust-build/kinematic-invariant-amplitude-executor-2/results',
          'MATCHED':R/'.tirust-build/kinematic-matched-native-input-executor-2/results',
          '_spec':retained_spec,'_build':retained_build,'sources':retained_sources},label='sealed-checked-outer')
    arithmetic=[];original_compare=legacy.compare_coefficients
    def compare_guard(reference,candidate):
        require(reference['eligible'] is True and candidate['eligible'] is True,'Failed output arithmetic')
        arithmetic.append((reference['id'],candidate['id']))
        return original_compare(reference,candidate)
    legacy.compare_coefficients=compare_guard
    # Reproduce the first historical failure without pretending its gate passed.
    oldconsumer=definitions(sources['controller1']['compare.py'],{'require','pin','checked','_build'}, {'SCIENCE_SHA'},
        injected={'hashlib':hashlib,'json':json,'re':re,'Path':RetainedPath},label='original-failed-build-consumer')
    try:oldconsumer._build(R/(work_prefix[1]+'results/runs'))
    except KeyError as exc:require(str(exc)=="'scientific_source_index_sha256'",'Original gate failure differs')
    else:raise ValueError('Original missing-key failure did not reproduce')
    failedgate=doc('results1/runs/'+identifier+'/bootstrap-gate.json')
    require(failedgate=={'schema':'private-checked-bootstrap-gate-v1','passed':False,
        'failures':["'scientific_source_index_sha256'"],'measurement_requested':False,'native_actions':0,'source_evidence':{}},
        'Original failed admission receipt differs')
    native1=doc('results1/runs/'+identifier+'/result.json');spec1=doc('results1/runs/'+identifier+'/specification.json')
    require(native1['protocol'][1:]==[['BYE']] and native1['bootstrap_gate_passed'] is False and
            native1['measurement_command_attempted'] is False and native1['measurement_completed'] is False and
            native1['native_returncode']==0 and native1['cleanup_failure'] is None,'Failed first worker reclassified')
    retrospective=outer._bootstrap_snapshot_gate(R/(work_prefix[1]+'results/runs/'+identifier),spec1,
        native1['protocol'][:1],native1['runtime_maps'],scientific,build,build_pin,C1,historical=True)
    retrospective.update(retrospective_only=True,original_run_passed=False,original_bootstrap_gate_passed=False,
        original_failure_preserved=True,original_run_closure=pin(data('results1/run-closure.json')),
        retained_payloads_verified=20,new_native_actions=0,historical_protocol=native1['protocol'],
        historical_native_pid=native1['native_pid'],donor_build_receipt=build_pin,
        retained_raw_directory=ROOT+'/'+work_prefix[1]+'results/runs/'+identifier,
        scope='Retrospective ordinary READY admission only; CMS20 was never requested')
    require(retrospective==doc('results2/retrospective-bootstrap-preflight.json') and retrospective['passed'],
            'Full corrected retrospective gate differs')
    # Replay live READY using exact prefixes whose hashes are in its receipt.
    native2=doc('results2/runs/'+identifier+'/result.json');spec2=doc('results2/runs/'+identifier+'/specification.json')
    view['bootstrap2']=True
    ready=outer._bootstrap_snapshot_gate(R/(work_prefix[2]+'results/runs/'+identifier),spec2,
        native2['protocol'][:1],native2['runtime_maps'][:1],scientific,build,build_pin,C2)
    view['bootstrap2']=False
    require(ready==boot2 and ready['passed'],'Exact live bootstrap gate does not reproduce')
    replay=outer.compare_stage(R/(work_prefix[2]+'results/runs'),requests,controller_source_sha=C2,build_receipt=build)
    raw=checked(data,'results2/checked-comparison.json',COMPARISON)
    require(replay==json.loads(raw)==document('results2/checked-comparison.json') and
            raw==(json.dumps(replay,indent=2,sort_keys=True)+'\n').encode(),'Exact complete checked comparison does not reproduce')
    require(replay['passed'] and replay['host_stability_qualified_this_fixture'] and
            not replay['broad_accuracy_qualified'] and not replay['performance_qualified'],'Qualified scope differs')
    expected_calls=[('cms-p1-mode4-lib9-candidate',identifier)]*3+[
        ('cms-p20-mode4-lib8-matched',identifier),('cms-p20-mode4-lib9-matched',identifier)]
    require(arithmetic==expected_calls,'Bootstrap/measurement arithmetic sequence differs')
    for number,spec,native in ((1,spec1,native1),(2,spec2,native2)):
        prefix=f'results{number}/runs/{identifier}/'
        require([x['boundary'] for x in native['runtime_maps']]==(['READY'] if number==1 else ['READY','DONE']),
                'Actual runtime boundaries differ')
        for capture in native['runtime_maps']:verify_maps(data(prefix+safe(capture['file'])),capture,spec['expected_runtime'])
        require(native['close_receipts'] and all(x['complete'] and x['worker_closed'] and x['returncode']==0 and
                x['pid']==native['native_pid'] for x in native['close_receipts']),'Native ownership closure differs')
    require(native1['native_pid']!=native2['native_pid'],'Worker PID reused')
    require(replay['actual_SLOOP_entries']==2 and replay['actual_native_calls']==156 and replay['rescues_exercised']==[1],
            'Actual checked coverage differs')
    q0=replay['phases']['0'];q1=replay['phases']['1']
    require(q0['host_qualification']['accepted_modes']==[1,2] and q1['host_qualification']['accepted_modes']==[4,5] and
            q1['matched_mode4_join']['passed'] and all(p['full_coefficients_passed'] and p['pole_checks']['passed'] for p in (q0,q1)),
            'Checked rescue/input/coefficients/poles differ')
    return {'comparison_serialized_bytes_reproduced':True,'original_admission_failure_reproduced':True,
        'retrospective_bootstrap_reproduced':True,'live_bootstrap_reproduced':True,
        'scientific_definitions_unchanged':True,'failed_first_run_preserved':True,
        'ordinary_return':q0['host_qualification']['code'],'CMS20_return':q1['host_qualification']['code'],
        'checked_native_calls':156,'checked_outer_returns':2,'historical_failed_bootstrap_outer_returns':1,
        'rescues_exercised':[1],'raw_runtime_maps_verified':3,'coefficient_comparison_calls':len(arithmetic),
        'accepted_QP_primary_input_join':True,'failed_DP_tensor_outputs_read':False,
        'normal_host_stability_this_fixture':True,'broad_accuracy_qualified':False,'production_promoted':False,
        'performance_qualified':False,'new_native_executions':0,'external_binary_reads':0,
        'historical_path_callback_reads':len(reads),'materialized_inputs_rehashed':False,
        'historical_source_admission_preserved_from_protected_snapshot':True}
