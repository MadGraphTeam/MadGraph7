#!/usr/bin/env python3
"""Two loop-induced timing pairs using the unchanged reviewed pair primitives.

Only original Ninja6 and final NinjaRS8 are dispatched.  The older Cartesian
supplement specification is retained as history and is never executed here.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_policy_v3 as release
import ninjars_benchmark_workers as workers
import ninjars_supplement_v2 as supplement

HERE = Path(__file__).resolve().parent
SCHEMA = 'final-loop-explicit-pair-spec-v1'
BINDINGS = 'final-loop-explicit-pair-bindings-v1'
TARGET_SCHEMA = 'final-loop-observed-pair-target-v1'
TARGETS = (('baseline',6),('candidate',8))
ROLES = {'baseline':'original','candidate':'candidate_corrected'}
CASE_IDS = ('supplement-gg-h','supplement-gg-hh')
COHORT_KEYS = ('numerical_revision','generation_revision','generator_identity_sha256',
               'runtime_identity_sha256','archive_sha256')
POINT_SHA256 = {'supplement-gg-h':'146ab1c08480c2687352d1800ba12c00633097d109c9f3b084bb69c9a41dfa5b',
    'supplement-gg-hh':'3f05e279caa8057906bce524597066c479454b89d455ca57a42691f5060004f6'}


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def source_files():
    names = {*workers.SOURCE_NAMES,'ninjars_benchmark_policy_v2.py',
        'ninjars_benchmark_policy_v3.py','ninjars_benchmark_pair_v3.py',
        'ninjars_benchmark_references_v2.py','ninjars_supplement_v2.py','ninjars_benchmark_driver.f'}
    return {'collector':supplement.bound(Path(__file__)),
            **{name:supplement.bound(ROOT/'tools'/name) for name in sorted(names)}}


def validate_spec(spec,live=True):
    policy.verified(spec,SCHEMA)
    if (spec['pair_targets'] != [list(t) for t in TARGETS] or spec['role_map'] != ROLES or
            spec['cpu'] != 207 or spec['repetitions'] != 7 or spec['minimum_seconds'] != 1. or
            spec['timeout'] != 86400. or set(spec['origins']) != set(CASE_IDS)):
        raise ValueError('Loop timing requires exactly seven original6/final8 pairs on CPU207')
    config=read(supplement.checked(spec['configuration']))
    supplement.validate_configuration(config)
    if tuple(case['id'] for case in config['cases']) != CASE_IDS:
        raise ValueError('Unexpected loop process or ordering')
    pair.check_production(spec)
    cohort=release.validate_cohort(supplement.checked(spec['cohort_descriptor']))
    if pair.candidate_cohort(spec) != {k:cohort[k] for k in pair.candidate_cohort(spec)}:
        raise ValueError('The explicit loop cohort differs from its installed source/runtime proof')
    proof=policy.verified(read(supplement.checked(spec['runtime_qualification'])),'representative-policy-validation-v2')
    if cohort['runtime_identity_sha256'] not in proof['reducer_runtimes']['candidate_corrected/8']:
        raise ValueError('Final loop runtime lacks its own representative qualification')
    for case in config['cases']:
        cid=case['id'];point_file=supplement.checked(spec['points'][cid]);cards=[]
        if (int(point_file.read_text().splitlines()[0]) != 100 or
                workers.digest(point_file)!=POINT_SHA256[cid]):
            raise ValueError('Exactly100 retained loop points are required')
        for source,backend in TARGETS:
            item=spec['origins'][cid][source]
            anchor=policy.verified(read(supplement.checked(item['source_proof'])),'supplement-generation-anchor-v1')
            output=policy.path(spec['output_roots'][source])/cid
            definition=anchor['case']
            if (anchor['role'] != ROLES[source] or
                    any(definition[k] != case[k] for k in ('model','process','nfinal'))):
                raise ValueError('A source alias changed its actual process/role or is a top symlink')
            identity=anchor['generation']['source_identity']
            if live:
                if output.is_symlink() or not output.is_dir():raise ValueError('Real original loop output required')
                pair.validate_origin_records(source,output,anchor['generation'],anchor['compilation'],anchor['compiled_manifest'])
            card_path=supplement.checked(item['physical_card']);card=card_path.read_bytes()
            if (workers.digest(card_path) != anchor['parameter_card_sha256'] or
                    (live and (output/'Cards/param_card.dat').read_bytes()!=card)):
                raise ValueError('A loop physical card changed')
            command=supplement.checked(anchor['cli_command']).read_text()
            log=supplement.checked(anchor['cli_log']).read_text()
            if (f"generate {case['process']}\n" not in command or
                    f"import model {case['model']}\n" not in command or
                    'Error detected in sub-command' in log or 'interrupted in sub-command' in log):
                raise ValueError('Missing or failed actual loop CLI generation')
            runtime=policy.verified(item['reducer_runtime'],'reducer-runtime-signature-v2')
            if live and pair.actual_runtime(output,anchor['compiled_manifest'],backend) != runtime:
                raise ValueError('The actual loop runtime changed')
            if source=='baseline':policy.original_identity(identity)
            if source=='candidate' and (identity['revision'] != cohort['generation_revision'] or
                    identity['tree_state_sha256'] != cohort['generator_identity_sha256'] or
                    runtime['identity_sha256'] != cohort['runtime_identity_sha256'] or
                    cohort['archive_sha256'] not in runtime['libraries'].values()):
                raise ValueError('Loop candidate differs from final numerical/generator/runtime cohort')
            cards.append(card)
        if cards[0] != cards[1]:
            raise ValueError('Original Ninja/final NinjaRS physical cards differ')
    return spec,config


def prepare_spec(output):
    old=read(HERE/'specification-held-v1.json')
    cohort=release.validate_cohort(HERE.parent/'candidate-cohort.json')
    aliases=policy.verified(read(supplement.checked(old['archive_aliases'])),'final-loop-archived-input-aliases-v1')
    origins={}
    for cid in CASE_IDS:
        saved=aliases['targets'][cid]
        origins[cid]={}
        for source,_ in TARGETS:
            card=(ROOT/'docs/benchmarkfolder/policy-v2/final-supplement-generation-v1/inputs/baseline'/cid/'param_card.dat'
                  if source=='baseline' else HERE.parent/'controls-generation/inputs/candidate'/saved['actual_candidate_cli_case']['id']/'param_card.dat')
            origins[cid][source]={'source_proof':old['source_proofs'][source][cid],
                'physical_card':supplement.bound(card),
                'reducer_runtime':saved['baseline_runtime_6' if source=='baseline' else 'candidate_runtime']}
    names=subprocess.check_output(['git','ls-files','-z','--',*bench.GENERATOR_PATHS,'vendor'],cwd=ROOT).decode().split('\0')
    spec=policy.seal({'schema':SCHEMA,'status':'held-no-numerical-launch',
        'superseded_schedule':supplement.bound(HERE/'specification-held-v1.json'),
        'superseded_schedule_must_not_execute':True,
        'configuration':old['configuration'],'points':old['points'],'output_roots':old['output_roots'],
        'archive_aliases':old['archive_aliases'],'runtime_qualification':old['runtime_qualification'],
        'cohort_descriptor':supplement.bound(HERE.parent/'candidate-cohort.json'),
        'candidate_cohort':{k:cohort[k] for k in COHORT_KEYS},
        'pair_targets':[list(t) for t in TARGETS],'role_map':ROLES,'origins':origins,
        'cpu':207,'repetitions':7,'minimum_seconds':1.,'timeout':86400.,
        'evaluation_thread_environment':{name:'1' for name in pair.THREADS},
        'sources':source_files(),'production_sources':{name:workers.digest(ROOT/name) if (ROOT/name).is_file() else None for name in names if name},
        'run_id':'final-pins-loop-explicit-pair-v2',
        'report':policy.relative(HERE/'runs/loop-pair-v2'),
        'work':policy.relative(ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/loop-execution/runs/loop-pair-v2'),
        'performance_gate':{'denominator':'original/6','max_runtime_ratio':1.05,
            'uncertainty_rule':'Unresolved confidence bounds do not pass'},
        'new_cuttools_or_collier_evaluations':0})
    validate_spec(spec)
    exclusive(output,spec)
    return {'specification':supplement.bound(output),'status':'held-no-numerical-launch',
        'actual_timing_pairs':0,'new_cuttools_or_collier_evaluations':0}


def collect(specification,handoff):
    spec,config=validate_spec(read(specification))
    authorization=read(handoff)
    if authorization != {'root_authorized':True,'phase':SCHEMA,'cpu':207,
            'specification_sha256':supplement.digest(specification),'quiet_cpu207':True}:
        raise ValueError('Explicit root quiet-CPU207 handoff required')
    if 207 not in os.sched_getaffinity(0) or any(os.environ.get(name)!='1' for name in pair.THREADS):
        raise ValueError('Unavailable CPU207 or changed one-thread policy')
    report,work=policy.path(spec['report']),policy.path(spec['work'])
    if report.exists() or work.exists():
        raise ValueError('Immutable loop pair namespace exists')
    report.mkdir(parents=True);work.mkdir(parents=True)
    (report/'sources').mkdir();(report/'points').mkdir()
    for name,item in spec['sources'].items():
        shutil.copy2(supplement.checked(item),report/'sources'/('loop_pair.py' if name=='collector' else name))
    shutil.copy2(specification,report/'input-spec.json')
    for cid,item in spec['points'].items():shutil.copy2(supplement.checked(item),report/'points'/(cid+'.dat'))
    descriptor=workers.protocol_descriptor()
    bench.dump(report/'protocol.json',{'schema':workers.SCHEME,'configuration':config,
        'protocol_descriptor':descriptor,'protocol_identity_sha256':bench.content_digest(descriptor),
        'cpu':207,'repetitions':7,'sources':'both','backends':[6,8],
        'pair_targets':[list(t) for t in TARGETS],'role_map':ROLES,'fresh_parity':False,
        'candidate_cohort':spec['candidate_cohort'],'acceptance_eligible':False,
        'input_spec':supplement.bound(report/'input-spec.json'),
        'ordering':'Adjacent explicit original6/final8; reverse on every odd repetition'})
    status={'status':'incomplete','requested_cases':list(CASE_IDS),'completed_cases':[],
            'acceptance_eligible':False,'new_cuttools_or_collier_evaluations':0}
    bench.dump(report/'run-status.json',status)
    dispatch=pair.Dispatch(report);args=SimpleNamespace(cpu=207,repetitions=7,minimum_seconds=1.,timeout=86400.)
    sentinel=work/'input-root-sentinel'
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGHUP)}
    def interrupted(sig,_frame):raise InterruptedError('Loop pair controller received signal '+str(sig))
    for sig in previous:signal.signal(sig,interrupted)
    try:
        with supplement.harness(sentinel,report),supplement.input_roots(spec['output_roots'],sentinel):
            for case in config['cases']:
                validate_spec(spec)
                with pair.owned_preparation(report,label='loop-pair'):
                    pair.run_pair_case(case,config,args,work/'workers',report,spec['run_id'],dispatch)
                for source,backend in TARGETS:
                    bind_target(report,work,case,source,backend,spec)
                status['completed_cases'].append(case['id']);bench.dump(report/'run-status.json',status)
        status['status']='complete';bench.dump(report/'run-status.json',status)
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error));bench.dump(report/'run-status.json',status)
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    try:
        result=snapshot(report)
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        bench.dump(report/'run-status.json',status)
        raise
    exclusive(report/'result.json',result)
    return result


def bind_target(report,work,case,source,backend,spec):
    """Retain the observed built runtime, never substitute the expected runtime."""
    key=f"{case['id']}-{source}-{backend}"
    output=work/'workers'/key/'outputs'/source/case['id']
    build=read(report/'builds'/key/'build.json')
    manifest=bench.generated_source_manifest(output)
    if (manifest!=read(report/'builds'/key/'compiled-sources.json') or
            workers.digest(next(output.glob('SubProcesses/P*_*/check')))!=build['executable_sha256']):
        raise ValueError('Actual loop worker source or executable changed')
    assets=report/'runtime-assets';assets.mkdir(exist_ok=True)
    metadata={}
    for name,path in policy.runtime_metadata_files(output).items():
        target=assets/workers.digest(path)
        if not target.exists():shutil.copy2(path,target)
        metadata[name]=supplement.bound(target)
    item={'schema':TARGET_SCHEMA,'case':case['id'],'source':source,'role':ROLES[source],'backend':backend,
        'compiled_manifest':manifest,'libraries':policy.library_hashes(output),
        'runtime_metadata':metadata,'build':supplement.bound(report/'builds'/key/'build.json')}
    item['reducer_runtime']=policy.reducer_signature(item)
    if item['reducer_runtime']!=spec['origins'][case['id']][source]['reducer_runtime']:
        raise ValueError('Observed actual loop worker runtime differs from its qualified origin')
    exclusive(report/'targets'/(key+'.json'),policy.seal(item))


def check_dispatch(events,cases):
    if [e['index'] for e in events] != list(range(len(events))):
        raise ValueError('Dispatch indices are missing or duplicated')
    for event in events:
        if ((event['source'],event['backend']) not in TARGETS or event['case'] not in cases or
                event['event'] not in ('start','complete') or event['phase'] not in
                ('prepare','initialize','calibration','measurement','cleanup')):
            raise ValueError('Unexpected dispatch target, phase or failed event')
    expected=[]
    def add(cid,phase,source,backend,repetition=None):
        expected.extend((cid,phase,event,source,backend,repetition) for event in ('start','complete'))
    for cid in cases:
        for source,backend in TARGETS:add(cid,'prepare',source,backend)
        for source,backend in TARGETS:
            add(cid,'initialize',source,backend)
            add(cid,'calibration',source,backend,'calibration')
        for repetition in range(7):
            order=TARGETS if repetition%2==0 else tuple(reversed(TARGETS))
            for source,backend in order:add(cid,'measurement',source,backend,repetition)
        for source,backend in reversed(TARGETS):add(cid,'cleanup',source,backend)
    actual=[tuple(e[k] for k in ('case','phase','event','source','backend','repetition')) for e in events]
    if actual!=expected:
        raise ValueError('Loop dispatch is not the complete adjacent alternating6/8 protocol')


def check_worker_delta(case,anchor,build,manifest,worker_file):
    """Only the exact rendered observation driver may differ from CLI sources."""
    drivers=[name for name in anchor['compiled_manifest']['files']
             if name.startswith('SubProcesses/P') and name.endswith('/check_sa.f')]
    if len(drivers)!=1:raise ValueError('Expected exactly one original loop driver')
    relative=drivers[0]
    prefix=re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION',worker_file.read_text(),re.I).group(1)
    # Both fixed loop-induced exports have no Born orders. Reconstructing the
    # original fresh driver proves this prefix against its observed source hash.
    fresh=(ROOT/'tools/ninjars_benchmark_driver.f').read_text().replace('@PREFIX@',prefix).replace(
        '@NEXTERNAL@',str(case['nfinal']+2)).replace('@NBORN@','0').encode()
    if (hashlib.sha256(fresh).hexdigest()!=anchor['compilation']['driver_sha256'] or
            anchor['compiled_manifest']['files'][relative] !=
            {'sha256':hashlib.sha256(fresh).hexdigest(),'bytes':len(fresh)}):
        raise ValueError('Rendered observation driver has a different original process prefix')
    expected=(ROOT/'tools/ninjars_benchmark_worker.f').read_text().replace('@PREFIX@',prefix).replace(
        '@NEXTERNAL@',str(case['nfinal']+2)).replace('@NBORN@','0').encode()
    digest=hashlib.sha256(expected).hexdigest()
    files=dict(anchor['compiled_manifest']['files'])
    files[relative]={'sha256':digest,'bytes':len(expected)}
    if (worker_file.read_bytes()!=expected or build['driver_sha256']!=digest or
            manifest['files']!=files or manifest['files_sha256']!=bench.content_digest(files)):
        raise ValueError('Worker source differs from exact CLI sources plus rendered diagnostic driver')


def check_ownership(events):
    """Require the owned registry's actual terminal receipt for every spawn."""
    live={}
    for event in events:
        identity=tuple(event[k] for k in ('pid','start_ticks','pgid'))
        if (type(event['pid']) is not int or type(event['pgid']) is not int or
                event['pid']<=0 or event['pgid']!=event['pid']):
            raise ValueError('Invalid owned process-group identity')
        if event['event']=='spawn':
            known=(type(event['start_ticks']) is int and event['start_ticks']>0)
            exited=(event['start_ticks'] is None and type(event.get('exited_before_identity')) is int
                    and event['exited_before_identity']==0)
            if identity in live or not (known or exited):
                raise ValueError('Duplicate or unknown live owned identity')
            live[identity]=event
        elif event['event']=='reaped':
            if (identity not in live or event.get('reaped') is not True or
                    type(event.get('returncode')) is not int or event['returncode']!=0):
                raise ValueError('Unmatched or failed terminal owned receipt')
            del live[identity]
        else:raise ValueError('Unexpected ownership event')
    if not events or live:raise ValueError('Missing terminal owned process receipts')


def check_points(cid,copied,authorized):
    if copied.read_bytes()!=authorized.read_bytes() or workers.digest(copied)!=POINT_SHA256[cid]:
        raise ValueError('Copied loop point stream differs from the authorized fixed100 inputs')


def snapshot(report):
    report=policy.path(report);protocol=read(report/'protocol.json')
    specification=supplement.checked(protocol['input_spec']);spec,config=validate_spec(read(specification),live=False)
    status=read(report/'run-status.json')
    if (status['status']!='complete' or status['requested_cases']!=list(CASE_IDS) or
            status['completed_cases']!=list(CASE_IDS) or list((report/'rejected').glob('*.json'))):
        raise ValueError('Loop pair collection is incomplete, failed or contains rejected observations')
    descriptor=workers.protocol_descriptor()
    if (protocol['configuration']!=config or protocol['pair_targets']!=[list(t) for t in TARGETS] or
            protocol['backends']!=[6,8] or protocol['sources']!='both' or protocol['role_map']!=ROLES or
            protocol['cpu']!=207 or protocol['repetitions']!=7 or protocol['fresh_parity'] is not False or
            protocol['candidate_cohort']!=spec['candidate_cohort'] or
            protocol['protocol_descriptor']!=descriptor or protocol['protocol_identity_sha256']!=bench.content_digest(descriptor)):
        raise ValueError('Changed loop pair protocol/cohort')
    for name,digest in descriptor['sources'].items():
        if workers.digest(report/'sources'/name)!=digest:raise ValueError('Changed retained worker source')
    for name,item in spec['sources'].items():
        retained=report/'sources'/('loop_pair.py' if name=='collector' else name)
        if workers.digest(retained)!=item['sha256']:raise ValueError('Changed retained pair adapter source')
    events=[json.loads(line) for line in (report/'dispatch.jsonl').read_text().splitlines()]
    check_dispatch(events,CASE_IDS)
    ownership_path=report/'process-ledger.jsonl'
    ownership=[json.loads(line) for line in ownership_path.read_text().splitlines()]
    check_ownership(ownership)
    expected=set();cases={};inputs={}
    for case in config['cases']:
        cid=case['id'];records={};targets={}
        authorized=supplement.checked(spec['points'][cid]);copied=report/'points'/(cid+'.dat')
        check_points(cid,copied,authorized)
        for source,backend in TARGETS:
            key=f'{cid}-{source}-{backend}';build_path=report/'builds'/key/'build.json';build=read(build_path)
            manifest=read(build_path.parent/'compiled-sources.json')
            target_path=report/'targets'/(key+'.json');target=policy.verified(read(target_path),TARGET_SCHEMA)
            if (target.get('case')!=cid or target.get('source')!=source or target.get('backend')!=backend or
                    target.get('role')!=ROLES[source] or target['compiled_manifest']!=manifest or
                    read(supplement.checked(target['build']))!=build):
                raise ValueError('Observed loop target/build binding changed')
            for item in target['runtime_metadata'].values():supplement.checked(item)
            if (policy.reducer_signature(target)!=target['reducer_runtime'] or
                    target['reducer_runtime']!=spec['origins'][cid][source]['reducer_runtime']):
                raise ValueError('Observed full runtime differs from qualified origin')
            anchor=policy.verified(read(supplement.checked(spec['origins'][cid][source]['source_proof'])),'supplement-generation-anchor-v1')
            check_worker_delta(case,anchor,build,manifest,build_path.parent/'worker.f')
            if (build['generation_source_identity']!=anchor['generation']['source_identity'] or
                    build['origin_compilation']!=anchor['compilation'] or
                    build['original_generated_sources_sha256']!=anchor['compiled_manifest']['files_sha256'] or
                    manifest['files_sha256']!=build['generated_sources_sha256'] or
                    bench.content_digest(manifest['files'])!=manifest['files_sha256']):
                raise ValueError('Changed actual loop source/build identity')
            values=[];calibration=None
            for repetition in ('calibration',*range(7)):
                path=report/'data'/f'{key}-{repetition}.json';expected.add(path);record=read(path)
                release.validate_rows(record,backend);release._check_raw_capture(record,inputs)
                if (record['case']!=cid or record['source']!=source or record['backend']!=backend or
                        record['repetition']!=repetition or record['cpu']!=207 or
                        record['measurement_run_id']!=spec['run_id'] or record['paired_baseline_candidate'] is not True or
                        record['generation_source_identity']!=build['generation_source_identity'] or
                        record['protocol_identity_sha256']!=protocol['protocol_identity_sha256'] or
                        record['executable_sha256']!=build['executable_sha256'] or
                        record['generated_sources_sha256']!=build['generated_sources_sha256'] or
                        record['points_sha256']!=supplement.digest(report/'points'/(cid+'.dat')) or
                        record['parameter_card_sha256']!=anchor['parameter_card_sha256'] or
                        record['answer_dimensions']['born_orders']!=0 or
                        any(row['coefficients'][0]!=0. for row in record['rows']) or
                        (repetition=='calibration' and record['loops']!=1)):
                    raise ValueError('Changed loop record target, physical input or zero-Born condition')
                matched=[e for e in events if e['index']==record['dispatch_index']]
                if len(matched)!=1 or any(matched[0][k]!=v for k,v in
                        {'case':cid,'source':source,'backend':backend,'event':'start','repetition':repetition,
                         'phase':'calibration' if repetition=='calibration' else 'measurement',
                         'loops':record['loops'],'worker_sequence':record['worker_sequence']}.items()):
                    raise ValueError('Measurement record differs from its actual dispatch command')
                completed=events[record['dispatch_index']+1]
                if (supplement.checked(completed['record'])!=path or
                        completed['loops']!=record['loops'] or
                        completed['worker_sequence']!=record['worker_sequence']):
                    raise ValueError('Completed dispatch differs from its actual accepted output')
                if repetition=='calibration':calibration=record
                else:bench.validate_repetition(record,calibration,config);values.append(record)
                inputs[policy.relative(path)]=workers.digest(path)
            records[source,backend]=values;targets[f'{source}/{backend}']={'binding':supplement.bound(target_path),
                'build':supplement.bound(build_path),'reducer_runtime':target['reducer_runtime']}
        left,right=records['baseline',6],records['candidate',8];comparisons=[]
        for a,b in zip(left,right):
            if any(a[k]!=b[k] for k in ('points_sha256','parameter_card_sha256','shared_numerical_settings','answer_dimensions')):
                raise ValueError('Loop timing denominator physical/numerical drift')
            comparisons.append(policy.compare_rows({'original/6':a['rows'],'candidate_corrected/8':b['rows']}))
        performance=bench.runtime_statistics([r['seconds_per_point'] for r in left],
            [r['seconds_per_point'] for r in right],1.05,config['seed'])
        cases[cid]={'agreement':{'passed':all(c['passed'] for c in comparisons),'comparisons':comparisons},
            'performance':performance,'actual_pairs':7,'points':100,'born_verified_zero':True,'targets':targets,
            'status_counts':{f'{source}/{backend}':[r['status_counts'] for r in rows]
                             for (source,backend),rows in records.items()}}
    if set((report/'data').glob('*.json'))!=expected:raise ValueError('Injected or missing loop observations')
    if {p.name for p in (report/'targets').glob('*.json')} != {
            f'{cid}-{source}-{backend}.json' for cid in CASE_IDS for source,backend in TARGETS}:
        raise ValueError('Injected or missing loop runtime targets')
    correctness=all(c['agreement']['passed'] for c in cases.values())
    performance=all(c['performance']['passed'] for c in cases.values())
    return policy.seal({'schema':BINDINGS,'specification':supplement.bound(specification),
        'protocol':supplement.bound(report/'protocol.json'),'dispatch':supplement.bound(report/'dispatch.jsonl'),
        'process_ledger':supplement.bound(ownership_path),
        'candidate_cohort':spec['candidate_cohort'],'performance_denominator':'original/6',
        'cases':cases,'ninja_ninjars_correctness_passed':correctness,'performance_passed':performance,
        'status':'passed' if correctness and performance else 'failed' if not correctness or any(
            c['performance']['status']=='failed' for c in cases.values()) else 'unresolved',
        'new_cuttools_or_collier_evaluations':0,'inputs_sha256':dict(sorted(inputs.items()))})


def validate(filename):
    recorded=policy.verified(read(filename),BINDINGS)
    protocol=supplement.checked(recorded['protocol'])
    if snapshot(protocol.parent)!=recorded:raise ValueError('Stale or forged loop pair conclusion')
    return recorded


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare-spec','collect','validate'))
    parser.add_argument('--spec',type=Path,default=HERE/'specification-pair-held-v2.json')
    parser.add_argument('--handoff',type=Path)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--execute-numerical',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare-spec':result=prepare_spec(args.spec)
    elif args.action=='validate':result=validate(args.result)
    elif not args.execute_numerical or not args.handoff:parser.error('Collection requires root handoff and --execute-numerical')
    else:result=collect(args.spec,args.handoff)
    print(json.dumps(result,indent=2))
