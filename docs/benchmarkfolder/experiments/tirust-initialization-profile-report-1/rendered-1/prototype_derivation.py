#!/usr/bin/env python3
"""Pure private derivation of actual initialization and stock-CLI controls.

The final candidate adapter is admitted only after its actual qualified schema
is supplied. Control replay does not run a reducer, controller or compiler.
"""
from pathlib import Path, PurePosixPath
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import types
import contextlib
import tempfile
from decimal import Decimal

PHASE_SUMMARIZER='8648278079735c6f18c0cda499117aec66690ff791418da3a67a06908e0b359d'
ORIGINAL_DSO='85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0'
ORIGINAL_MG='cd0145922977fcfdc4e3cb3183ddc974c31a7897'
ORIGINAL_POINT='8880598b2d448930eedda40d9797fd137a5845c5289345d1e31cbcd22ff10c3b'
ORIGINAL_EXE='ffb951914a087327ae05ed63e90fa9ecfa3603137a254cca2781fc431791ffa3'
O2_EXE='c4b5a2bdc7ed555939239360e357fbcc09fbae06cee35e816f3133c5848648f2'
ORIGINAL_TIRUST='10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1'
PROTOTYPE_SOURCE='607e7f58842b50a07df7bbdbcbd00ee399f6baad0b670745b45cbf807de7ff07'
PROTOTYPE_DSO='470d46df9c8fcb76dc5448a04187dfc23a757481b3d4ef0384e13bd6e7e9e709'
FUNCTIONAL_SHA='ac70267103c6c740adb257991ad474c9794d26510889520917d848e67cfb4513'
QUIET_SHA='1a9ce52992e71699af1e560f37296ca2d216ce84900fd4c437c6b37efc744210'
ROUTE_REPLAYER_SHA='cbd4758cc34f84501b48b9082bcf2af7fb835b7e60e540b52e8a896abd8b8eaa'
METRIC_SOURCE_SHA='4e75170a0e688d06e91599962eb869c1c40dc9c5c1b24372869ec73a728dce8a'
STOCK_MG='cafb35d7987436661c3346e39c724d329846a1f2'
STOCK_INDEX='61ef3a2a1441a2e78ea87233d7d4a24ad6e4953e5a1fb71191a2962c0e8a4c39'
SOURCE_INDEX='e401cf5d4efb7708c277172eaa3be9f29bcadf54c5b3b484e7311109cf9a26cb'
SOURCE_ARCHIVE='408464bfc83c7d87dc45e52fa819c99f0f759e2cea7cbce1ae233ec36e25b2f1'
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def read(path):
    def unique(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    def invalid(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)


def identity(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def inside(root,name):
    require(isinstance(name,str) and name and '\\' not in name,'Invalid evidence path')
    part=PurePosixPath(name)
    require(not part.is_absolute() and '..' not in part.parts and str(part)==name,'Unsafe evidence path')
    path=Path(root)
    for piece in part.parts:
        path/=piece;require(not path.is_symlink(),'Symlink evidence input')
    return path


def packet(path,index_sha):
    path=Path(path);require(identity(path/'index.json')['sha256']==index_sha,'Packet seal changed')
    index=read(path/'index.json')
    keys=[key for key in ('files','payloads') if key in index]
    require(len(keys)==1,'Ambiguous packet inventory');files=index[keys[0]]
    require({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}==set(files)|{'index.json'},'Packet inventory changed')
    for name,expected in files.items():require(identity(inside(path,name))==expected,'Changed packet file '+name)
    return index


def initialization(folder,summarizer,point,preparation,expected_executable):
    result=read(folder/'result.json');require(result['status']=='ready' and result['shared_unchanged'] is True,'Initialization did not close')
    require(result['tirust_dso_sha256']==ORIGINAL_DSO,'Initialization source DSO changed')
    require(result['source_executable_sha256']==expected_executable==preparation['executable_sha256'], 'Control executable binding differs')
    require(identity(point)['sha256']==ORIGINAL_POINT==result['point_sha256']==preparation['points_sha256']
            and preparation['point_membership_one_based']==[1], 'Original one-point input binding differs')
    require(result['initialization']['points']==1 and result['initialization']['ndim']==1
            and result['initialization']['nso']==1 and result['initialization']['nborn']==1
            and (folder/'dimensions.dat').read_text().split()==['1','1','1'], 'Initialization dimensions changed')
    require(result['shared_after']==preparation['shared_state'] and preparation['instrumentation_is_benchmark'] is False,
            'Shared state/instrumentation scope differs')
    for field in ('mapped_tirust','first_cabi_mapped_tirust','loader_tirust'):
        require(set(result[field].values())=={ORIGINAL_DSO}, 'Actual runtime binding differs')
    require(identity(folder/'phases.jsonl.maps')['sha256']==result['first_cabi_maps_sha256'], 'Live map snapshot differs')
    trace=folder/'phases.jsonl';summary_path=folder/'phase-summary.json' 
    require(identity(trace)['sha256']==result['phase_events_sha256'] and identity(summary_path)['sha256']==result['phase_summary_sha256'],'Phase input binding changed')
    summary=summarizer.summarize(trace)
    require(summary==read(summary_path) and not summary['malformed_lines'] and not summary['incomplete_calls'],'Phase replay differs/incomplete')
    phases={}
    for group in summary['groups']:
        phase=phases.setdefault(group['operation'],{'calls':0,'wall_seconds':0.,'status_counts':{}})
        phase['calls']+=group['completed_calls'];phase['wall_seconds']+=group['wall_seconds']
        for status,count in group['status_counts'].items():phase['status_counts'][status]=phase['status_counts'].get(status,0)+count
    ready=result['initialization']['cold_seconds'];setup=phases['create']['wall_seconds']+phases['register']['wall_seconds']
    remaining=ready-setup-phases['dp']['wall_seconds']-phases['qp']['wall_seconds']
    require(remaining>=0 and summary['completed_calls']==18579,'Unexpected original attribution')
    raw=[line.split() for line in (folder/'warmup.dat').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    require(len(raw)==4 and {(int(r[0]),int(r[1]),int(r[2]),int(r[3])) for r in raw}=={(call,1,order,323) for call in (0,1) for order in (0,1)},'Expected same-point bootstrap/warmup QP323 pair')
    events=[read_line for read_line in map(json.loads,trace.read_text().splitlines())]
    pending={};calls=[]
    for row in events:
        key=(row['pid'],row['sequence'])
        if row['event']=='begin':pending[key]=row
        else:
            begin=pending.pop(key);calls.append({**begin,'end_ns':row['monotonic_ns'],'status':row['status']})
    require(not pending,'Open profile events')
    sequence=[tuple(c[k] for k in ('operation','n','rank','poles','finite','check','cache','input_fingerprint_fnv128','status')) for c in calls]
    blocks=[]
    for c in calls:
        if c['operation'] not in ('dp','qp'):continue
        if not blocks or blocks[-1]['operation']!=c['operation']:
            blocks.append({'operation':c['operation'],'start_ns':c['monotonic_ns'],'end_ns':c['end_ns']})
        else:blocks[-1]['end_ns']=c['end_ns']
    gaps=[(right['start_ns']-left['end_ns'])*1e-9 for left,right in zip(blocks,blocks[1:]) if left['operation']=='dp' and right['operation']=='qp']
    failures={str(check):sum(1 for c in calls if c['operation']=='dp' and c['check']==check and c['status']!=0) for check in (0,1)}
    return {'ready_seconds':ready,'setup_seconds':setup,'dp_seconds':phases['dp']['wall_seconds'],'qp_seconds':phases['qp']['wall_seconds'],
        'outside_dp_qp_setup_seconds':remaining,'all_completed_bridge_seconds':summary['completed_bridge_seconds'],
        'outside_bridge_in_event_interval_seconds':summary['interval_seconds']-summary['completed_bridge_seconds'],
        'event_interval_seconds':summary['interval_seconds'],'phases':phases,'dp_to_qp_gaps_seconds':gaps,
        'dp_refusals_primary_check':failures,'completed_abi_calls':summary['completed_calls'],
        'source_executable_sha256':result['source_executable_sha256'],'tirust_dso_sha256':result['tirust_dso_sha256'],
        'point_sha256':result['point_sha256'],'outer_calls':2,'distinct_physical_points':1,'outer_statuses':[323,323],
        'literal_warmup_rows':raw,'sequence_metadata_fnv_status':sequence,'performance_acceptance':False}


def stock_profiles(folder):
    packet(folder,STOCK_INDEX);summary=read(folder/'summary.json')
    require(summary['schema']=='madgraph-stock-profile-comparison-v1' and summary['process']=='u u~ > z [virt=QCD]'
            and summary['model']=='loop_sm' and summary['source_mg_revision']==STOCK_MG
            and summary['tirust_revision']==ORIGINAL_TIRUST and summary['tirust_dso_sha256']==ORIGINAL_DSO
            and summary['performance_acceptance'] is False,'Wrong stock source/scope')
    result=[]
    for backend in (8,9):
        base=folder/'backends'/str(backend);metadata_path=base/'SavedRunInfos/profile_timing_metadata.json';metadata=read(metadata_path)
        row=next(r for r in summary['rows'] if r['backend']==backend)
        require(identity(metadata_path)['sha256']==row['metadata_sha256'] and metadata['reduction_backends']==[backend],'Stock metadata binding changed')
        policy=metadata['selected_timing_helicity'];require(policy['filter_policy']=='all-helicities-explicit-poles' and policy['learned_filter_used'] is False,'Stock helicity scope changed')
        require(metadata['per_evaluation_statuses_available'] is False and row['per_evaluation_statuses_available'] is False,'Do not infer timed per-point precision')
        terminal=read(base/'run/terminal.json');require(terminal['status']=='profile-complete' and terminal['returncode']==0 and terminal['shared_unchanged'] and not terminal['source_changed'] and not terminal['cleanup']['survivors'],'Unclosed stock worker')
        timing=metadata['profile_timing_result'];stages={s['name']:s for s in metadata['stages']}
        for name,column in (('unpolarized_full','run_unpolarized_total'),('unpolarized_skip','run_unpolarized_coefs')):
            stage=stages[name];estimate=(stage['wall_seconds']-timing['Booting_time'])/stage['actual_npoints']
            require(math.isclose(estimate,timing[column],rel_tol=1e-14) and row[name+'_ms']==timing[column]*1000,'Stock wall-minus-boot estimator differs')
            card={k.lower():v for k,v in stage['effective_madloop_parameters'].items()}
            require(card['looppolemode']==0 and card['ctmoderun']==1 and card['tirustcomputefinite'] and card['useqpintegrandfortirust'],'Stock precision/pole policy changed')
        require(row['unpolarized_full_minus_skip_estimate_ms']==row['unpolarized_full_ms']-row['unpolarized_skip_ms']
                or math.isclose(row['unpolarized_full_minus_skip_estimate_ms'],row['unpolarized_full_ms']-row['unpolarized_skip_ms'],rel_tol=1e-14),'FULL-SKIP estimate changed')
        probe=read(base/'stability/raw-stability.json');require(len(probe['DP_stability'])==10 and len(probe['QP_stability'])==10,'Wrong separate stability-probe count')
        require(row['qp_probes']==sum(bool(v) for v in probe['QP_stability']) and row['max_dp_spread']==max(v['Accuracy'] for v in probe['DP_stability']),'Separate stability data differ')
        require(row['unstable_points']==len(probe['Unstable_PS_points']) and row['exceptional_points']==len(probe['Exceptional_PS_points']), 'Stability exceptional/unstable counts differ')
        require(row['retained_polarized_result_all_zero'] is True,'Unexpected polarized diagnostic; reassess scope')
        result.append({'backend':backend,'name':row['backend_label'],'full_ms':row['unpolarized_full_ms'],'skip_ms':row['unpolarized_skip_ms'],
            'full_minus_skip_estimate_ms':row['unpolarized_full_minus_skip_estimate_ms'],'full_points':stages['unpolarized_full']['actual_npoints'],
            'full_batch_wall_seconds':stages['unpolarized_full']['wall_seconds'],'boot_estimate_seconds':timing['Booting_time'],
            'initialization_seconds':row['initialization_seconds'],'stability_points':10,'qp_probes':row['qp_probes'],
            'unstable_points':row['unstable_points'],'exceptional_points':row['exceptional_points'],'maximum_within_backend_spread':row['max_dp_spread'],
            'timed_per_point_statuses_available':False,'source_mg_revision':STOCK_MG,'tirust_dso_sha256':ORIGINAL_DSO})
    require(result[1]['initialization_seconds'] is None,'Do not invent fresh TIRust initialization')
    ratio=result[1]['full_ms']/result[0]['full_ms'];require(ratio==summary['tirust_over_ninjars_full_ratio'],'Stock displayed ratio differs')
    return {'rows':result,'full_ratio_tirust_over_ninjars':ratio,'reliable_slowdown_inference':False,'paired':False,'confidence_interval':None,
        'estimator':'Python wall minus boot divided by stage points; FULL polls RSS every0.2s','performance_acceptance':False,
        'scope':'Single stockCLI control per backend; unequal short batches; all configured helicities; original source only'}


def derive_controls(root,manifest):
    root=Path(root);require(manifest['schema']=='madloop-init-profile-private-control-inputs-v1','Wrong control binding')
    for name,expected in manifest['files'].items():require(identity(inside(root,name))==expected,'Control input changed: '+name)
    code=inside(root,manifest['phase_summarizer']);require(identity(code)['sha256']==PHASE_SUMMARIZER,'Unreviewed phase replayer')
    summarizer=types.ModuleType('pinned_phase_replay');summarizer.__file__=str(code);exec(compile(code.read_bytes(),str(code),'exec'),summarizer.__dict__)
    for name in ('baseline','o2'):
        terminal=read(inside(root,manifest[name+'_terminal']))
        require(terminal['returncode']==0 and not terminal['cleanup']['survivors'], 'Control supervisor did not close')
    baseline=initialization(inside(root,manifest['baseline']),summarizer,inside(root,manifest['baseline_point']),read(inside(root,manifest['baseline_preparation'])),ORIGINAL_EXE)
    o2=initialization(inside(root,manifest['o2']),summarizer,inside(root,manifest['o2_point']),read(inside(root,manifest['o2_preparation'])),O2_EXE)
    preparation=read(inside(root,manifest['o2_preparation']))
    require(identity(inside(root,manifest['o2_closure']))['sha256']==preparation['optimized_fortran_build_closure_sha256']
            and identity(inside(root,manifest['o2_flags']))['sha256']==preparation['effective_compiler_flags_sha256'], 'O2 build closure/flags differ')
    comparison=read(inside(root,manifest['o2'])/'comparison.json')
    require(baseline['literal_warmup_rows']==o2['literal_warmup_rows'] and baseline['sequence_metadata_fnv_status']==o2['sequence_metadata_fnv_status'],'Original/O2 output or metadata/FNV/status sequence differs')
    ratio=o2['ready_seconds']/baseline['ready_seconds'];require(ratio==comparison['ready_ratio'] and comparison['literal_warmup_rows_identical'] is True and comparison['performance_acceptance'] is False,'O2 comparison changed')
    for value in (baseline,o2):value.pop('sequence_metadata_fnv_status');value.pop('literal_warmup_rows')
    source=inside(root,manifest['source_archive_packet']);index=packet(source,SOURCE_INDEX)
    original_spec=read(source/'measurements/specification.json');original_build=read(source/'measurements/compilation.json')
    require(original_spec['mg_revision']==ORIGINAL_MG and original_build['executable_sha256']==ORIGINAL_EXE, 'Published original source/executable differs')
    archive=inside(source,manifest['source_archive']);require(identity(archive)==index['files'][manifest['source_archive']] and identity(archive)['sha256']==SOURCE_ARCHIVE,'Existing source archive changed')
    return {'schema':'madloop-initialization-and-stock-controls-v1','baseline':baseline,'o2':o2,'o2_over_original_ready_ratio':ratio,
        'original_mg_revision':ORIGINAL_MG,'o2_promoted':False,'o2_benefit_qualified':False,'stock':stock_profiles(inside(root,manifest['stock_packet'])),
        'source_archive_reference':{'packet_index_sha256':SOURCE_INDEX,'path':manifest['source_archive_packet']+'/'+manifest['source_archive'],'identity':identity(archive)},
        'new_native_physics_evaluations':0,'performance_acceptance':False,
        'attribution_scope':'Instrumented ABI buckets; READY remainder includes budget/clear, boundary gaps, untraced work, scheduling and profiler/hash/log overhead; not clean kernel or pure Fortran timing'}


def literal_rows(path):
    names=('call','point','order','status','Born','finite','single','double','accuracy')
    return [dict(zip(names,line.split())) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def load_pure(path,name,expected):
    require(identity(path)['sha256']==expected,'Unreviewed pure replay source '+str(path))
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(Path(path).read_bytes(),str(path),'exec',optimize=0),module.__dict__)
    return module


def candidate_data(root,manifest,controls_manifest):
    root=Path(root);require(manifest['schema']=='madloop-adaptive-initialization-private-inputs-v1','Wrong candidate binding')
    for name,expected in manifest['files'].items():require(identity(inside(root,name))==expected,'Changed candidate input '+name)
    functional=inside(root,manifest['functional_dir']);quiet=inside(root,manifest['quiet_dir']);oracle=inside(root,manifest['oracle_root'])
    fs=read(functional/'functional-summary.json');fc=read(functional/'functional-comparison.json')
    require(identity(functional/'functional-summary.json')['sha256']==FUNCTIONAL_SHA and fs['schema']=='tirust-adaptive-one-point-functional-summary-v1'
            and fs['status']=='closed-functional-observation' and fs['candidate_DSO_sha256']==PROTOTYPE_DSO
            and fs['distinct_physical_points']==1 and fs['outer_amplitude_calls']==2 and fs['candidate_dispatch_status']==219
            and fs['functional_overlap_with_M1'] is True and fs['performance_eligible'] is False and fs['scientific_timing_claim'] is False,
            'Wrong functional source/scope')
    require(identity(functional/'functional-comparison.json')['sha256']==fs['comparison_sha256']
            and identity(inside(root,manifest['functional_source_index']))['sha256']==fs['native_observer_source_index_sha256'],
            'Functional source/comparison binding differs')
    compiled=read(oracle/'compiled.json');require(compiled['candidate_source_sha256']==PROTOTYPE_SOURCE
            and compiled['artifacts']['lib/libmadloop_tirust.so']==PROTOTYPE_DSO and compiled['upstream_revision']==ORIGINAL_TIRUST,
            'Prototype source/DSO binding differs')
    routes=load_pure(inside(root,manifest['route_replayer']),'actual_route_replay',ROUTE_REPLAYER_SHA).route_summary(functional/'phases.jsonl.routes.jsonl',functional/'phases.jsonl')
    require(routes==read(functional/'route-summary.json')==fc['route_precision_usage'] and routes['directions']==fs['directions']
            and routes['counter_observation_complete'] is True and not routes['malformed_lines']
            and routes['genuine_binary128_input_qp_cabi_calls']==fs['genuine_QP_input_calls']==0,'Actual route replay differs')
    require(routes['field_order_per_direction']==['Direct','MethodChoice','Tier106_DP_input','Tier256_DP_input','Tier512_DP_input','Refused','unreported_reduction_error','attempted106','attempted256','attempted512'],
            'Unreviewed20-counter layout')
    for direction in ('primary','check'):
        item=routes['directions'][direction];counts=item['counters']
        require(item['observed_dp_api_calls']==item['completed_full_dp_api_requests_reaching_counter']==2320
                and sum(counts[k] for k in routes['field_order_per_direction'][:7])==2320
                and counts['Refused']==counts['unreported_reduction_error']==0
                and item['attempted_tier_counts_are_lower_bounds'] is False,'Incomplete/error route accounting')
    result=read(functional/'result.json');release=read(inside(root,manifest['functional_release']));terminal=read(inside(root,manifest['functional_terminal']))
    require(result['status']=='ready' and result['source_executable_sha256']==ORIGINAL_EXE and result['point_sha256']==ORIGINAL_POINT
            and result['tirust_dso_sha256']==PROTOTYPE_DSO and result['shared_unchanged'] is True,'Functional runtime/input differs')
    for field in ('mapped_tirust','first_cabi_mapped_tirust','loader_tirust'):
        require(set(result[field].values())=={PROTOTYPE_DSO},'Functional live/loader DSO differs')
    require(release['closed'] is True and not release['owned_survivors'] and release['shared_unchanged'] is True
            and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Functional capture did not close')
    require(identity(functional/'result.json')['sha256']==release['result_sha256']==fc['candidate_result_sha256'],'Functional result identity differs')
    criterion={};exec(compile((HERE/'comparison-criterion.py').read_bytes(),str(HERE/'comparison-criterion.py'),'exec',optimize=0),criterion)
    compare=criterion['compare_rows'];candidate_rows=fc['candidate_amplitude_dispatch']['rows']
    require(len(candidate_rows)==2 and {(r['point'],r['order'],r['status']) for r in candidate_rows}=={(1,0,219),(1,1,219)},'Unexpected candidate output rows')
    references={}
    for key,observation in fc['comparisons'].items():
        for tolerance,field in ((3e-6,'criterion_3e_6'),(1e-8,'target_1e_8')):
            require(compare(observation['rows'],candidate_rows,tolerance)==observation[field],'Functional coefficient comparison differs')
        references[key]={'regression_pass':observation['criterion_3e_6']['passed'],
            'target_pass':observation['target_1e_8']['passed'],'maximum_raw_relative_error':observation['criterion_3e_6']['maximum_relative_error'],
            'target_failures':observation['target_1e_8']['failures']}
    require(all(r['regression_pass'] for r in references.values()) and references['1']['target_pass'] is False
            and all(references[k]['target_pass'] for k in ('genuine113bit_input_QP_baseline','6','7','8')),'Functional gates changed')
    quiet_summary=read(quiet/'observations/summary.json');qc=read(quiet/'observations/comparison.json');qr=read(quiet/'observations/result.json');prep=read(quiet/'preparation.json');qrelease=read(quiet/'native-release.json');qt=read(quiet/'supervision/terminal.json')
    require(identity(quiet/'observations/summary.json')['sha256']==QUIET_SHA and identity(quiet/'observations/comparison.json')['sha256']==quiet_summary['comparison_sha256']
            and identity(quiet/'source-index.json')['sha256']==quiet_summary['source_index_sha256']
            and identity(quiet/'native-release.json')['sha256']==quiet_summary['native_release_sha256'],'Quiet pair seal changed')
    require(qr['status']=='closed-pair' and qr['quiet_no_preload_no_counter_queries'] is True and qr['shared_and_sources_unchanged'] is True
            and qr['order']==['old','candidate'] and prep['same_point_sha256']==ORIGINAL_POINT and prep['CPU']==12 and prep['cpu_reserved'] is False,
            'Quiet pair inputs/scope differ')
    require(prep['lanes']['old']['settings']==prep['lanes']['candidate']['settings'],'Quiet numerical settings differ')
    quiet_rows={};seconds={};dates=[]
    for arm,dso,status in (('old',ORIGINAL_DSO,323),('candidate',PROTOTYPE_DSO,219)):
        lane=quiet/'observations'/arm;record=read(lane/'result.json');learning=read(lane/'learning.json')
        require(record['status']=='ready' and record['points_sha256']==ORIGINAL_POINT and record['executable_sha256']==ORIGINAL_EXE
                and record['dso_sha256']==dso and record['LD_PRELOAD'] is None and record['counter_queries'] is False and record['STOP_BYE_closed'] is True,
                'Quiet capture is not the same uninstrumented input/executable')
        require(set(record['loader_verified_dso'].values())==set(record['live_verified_dso'].values())=={dso},'Quiet actual DSO differs')
        require(record['ready']['points']==1 and record['ready']['ndim']==record['ready']['nso']==record['ready']['nborn']==1,'Quiet dimensions differ')
        require({r['status'] for r in learning['rows']}=={status} and len(learning['rows'])==2,'Quiet output status/count differs')
        quiet_rows[arm]=learning['rows'];seconds[arm]=record['ready']['cold_seconds'];dates.append(record['host_after']['utc'][:10])
    require(literal_rows(quiet/'observations/old/warmup.dat')==literal_rows(inside(root,controls_manifest['baseline'])/'warmup.dat')
            and literal_rows(quiet/'observations/candidate/warmup.dat')==literal_rows(functional/'warmup.dat'),'Quiet literal controls differ')
    require(seconds['old']==quiet_summary['old_READY_seconds'] and seconds['candidate']==quiet_summary['candidate_READY_seconds']
            and seconds['candidate']/seconds['old']==quiet_summary['candidate_over_old_ratio']
            and seconds['old']/seconds['candidate']==quiet_summary['old_over_candidate_speed_ratio']
            and qc['repetitions']==1 and qc['formal_95_percent_confidence_or_main_runtime_acceptance'] is False,'Quiet arithmetic/acceptance differs')
    require(qrelease['status']=='closed-pair' and not qrelease['owned_survivors'] and qrelease['shared_and_sources_unchanged'] is True
            and qrelease['no_preload_shim_or_counter_queries'] is True and qt['returncode']==0 and not qt['cleanup']['survivors'],'Quiet cleanup failed')
    for key,item in qc['comparisons'].items():
        reference=(quiet_rows['old'] if key=='clean_candidate_vs_clean_old_genuine_QP' else
            fc['comparisons']['genuine113bit_input_QP_baseline']['rows'] if key=='clean_old_vs_prior_genuine_QP' else
            candidate_rows if key=='clean_candidate_vs_prior_adaptive_functional' else item['reference_rows'])
        evaluated=quiet_rows['old'] if key=='clean_old_vs_prior_genuine_QP' else quiet_rows['candidate']
        for tolerance,field in ((3e-6,'criterion_3e_6'),(1e-8,'target_1e_8')):
            require(compare(reference,evaluated,tolerance)==item[field],'Quiet coefficient comparison differs')
    qualification=read(oracle/'tensor-functional-qualification.json');reference_comparison=read(oracle/'execution/functional-oracle/comparison.json')
    require(qualification['candidate_source_sha256']==PROTOTYPE_SOURCE and qualification['candidate_dso_sha256']==PROTOTYPE_DSO
            and qualification['same_engine_precision_reference'] is True and qualification['independent_physics_oracle'] is False
            and qualification['performance_eligible'] is False and not qualification['owned_survivors'],'Wrong reference qualification scope')
    require(identity(oracle/'execution/functional-oracle/comparison.json')['sha256']==qualification['comparison_sha256'],'Reference comparison identity differs')
    metric=load_pure(inside(root,manifest['oracle_metric_source']),'actual_tensor_metric',METRIC_SOURCE_SHA)
    metric.FIXTURES=inside(root,manifest['oracle_fixtures'])
    with tempfile.TemporaryDirectory(prefix='madloop-pure-reference-replay-') as tmp:
        output=Path(tmp)/'comparison.json'
        with contextlib.redirect_stdout(io.StringIO()):metric.compare(oracle/'execution/functional-oracle/oracle.log',oracle/'execution/functional-oracle/candidate-fixtures.log',output)
        require(read(output)==reference_comparison,'Same-input precision metric replay differs')
    slots=[row for case in reference_comparison['cases'] for row in case['slot_degree_norms']]
    misses=[row for row in reference_comparison['component_differences'] if Decimal(row['component_residual'])>Decimal('1e-8')]
    require(len(slots)==qualification['slot_degree_groups']==176 and all(row['pass_1e8'] and row['pass_3e6'] for row in slots)
            and len(misses)==qualification['component_relative_misses_1e8']==50 and reference_comparison['no_arbitrary_floor'] is True
            and reference_comparison['input_bits']==53 and reference_comparison['reference_working_bits']==512,
            'Reference vector/component qualification changed')
    raw_candidate=metric.load_rows(oracle/'execution/functional-oracle/candidate-fixtures.log')
    fixture_routes={name:sum(row['route_statistics'][offset]+row['route_statistics'][offset+10] for row in raw_candidate.values()) for name,offset in (('local106',2),('local256',3),('local512',4))}
    require(all(row['route_status']==0 and len(row['route_statistics'])==20 and sum(row['route_statistics'][i] for i in (2,3,4,12,13,14))==1 for row in raw_candidate.values())
            and sum(fixture_routes.values())==8,'Captured fixture accepted local routes differ')
    maximum=max(Decimal(row['residual']) for row in slots)
    require(maximum==Decimal(qualification['max_slot_degree_residual']),'Reference maximum differs')
    proof=read(functional.parents[1]/'preparation-snapshots/af0/genuine-QP-branch-preservation.json')
    require(proof['status']=='byte-identical' and proof['candidate_whole_bridge_sha256']==PROTOTYPE_SOURCE
            and proof['genuine_QP_input_precision_bits']==113 and proof['MPFR_work_bits']==256
            and proof['source_only_no_numerical_execution'] is True,'Source-only QP preservation differs')
    return {'actual_source_bound':True,'functional_qualified':True,'prototype_source_sha256':PROTOTYPE_SOURCE,'prototype_dso_sha256':PROTOTYPE_DSO,
        'production_artifact_qualified':False,'production_artifact':None,'measurement_role':'descriptive-clean-initialization',
        'timing_qualified':False,'descriptive_initialization_timing_admitted':True,'phases':None,
        'clean':{'old_ready_seconds':seconds['old'],'candidate_ready_seconds':seconds['candidate'],
            'candidate_over_old_ratio':seconds['candidate']/seconds['old'],'old_over_candidate_ratio':seconds['old']/seconds['candidate'],
            'ordered_pairs':1,'outer_calls_per_arm':2,'distinct_physical_points':1,'clock':'existing Fortran SYSTEM_CLOCK READY; launch/SETPARA excluded',
            'reserved_cpu':False,'CPU':12,'no_preload_or_queries':True,'date':dates[-1]},
        'functional':{'directions':routes['directions'],'field_order':routes['field_order_per_direction'],
            'genuine_qp_calls':0,'outer_statuses':[219,219],'outer_reported_accuracy':max(r['accuracy'] for r in candidate_rows),
            'performance_eligible':False,'overlapped_M1':True,'comparisons':references,'old_QP_vs_CT_preexisting_relative_error':fs['legacy_CT_vs_old_genuine_QP_finite_relative_error_before_candidate']},
        'reference':{'same_engine':True,'independent_physics_oracle':False,'input_bits':53,'working_bits':512,
            'fixtures':8,'fixture_routes':fixture_routes,'degree_groups':qualification['combined_degree_groups'],'slot_degree_groups':176,'maximum_slot_residual':str(maximum),
            'exactly_zero_slot_groups':qualification['both_zero_slot_groups'],'unexpected_one_zero_groups':qualification['unexpected_one_zero_slot_groups'],
            'component_count':len(reference_comparison['component_differences']),'component_relative_misses':misses,
            'component_miss_max_absolute':str(max(Decimal(row['complex_difference_norm']) for row in misses)),
            'component_miss_max_reference_norm':str(max((Decimal(row['reference_decimal'][0])**2+Decimal(row['reference_decimal'][1])**2).sqrt() for row in misses)),
            'no_floor':True},'genuine_qp_source_preserved':True,'genuine_qp_native_observed':False,'genuine_qp_proof':proof}


def derive(root):
    binding=read(HERE/'initialization-source-binding.json')
    require(binding['schema']=='madloop-initialization-profile-private-source-binding-v1' and binding['preview_only'] is True
            and binding['production_artifact_qualified'] is False,'Preview must not claim final production admission')
    controls_manifest=read(HERE/binding['controls_manifest']);manifest=read(HERE/binding['candidate_manifest'])
    controls=derive_controls(root,controls_manifest);candidate=candidate_data(root,manifest,controls_manifest)
    require(binding['date']==candidate['clean']['date'],'Report date differs from actual capture')
    data={'schema':'madloop-initialization-profile-report-data-v1','closed':True,'preview_only':True,
        'date':binding['date'],'source_label':'prototype607 / DSO470 · separate controls',
        'controls':controls,'candidate':candidate,'production_artifact_qualified':False,'performance_acceptance':False,
        'new_native_physics_evaluations':0,'evidence_note':'Private source-bound preview; actual functional/clean/reference receipts are inventoried. Final production source/artifact qualification is pending.'}
    selection={'schema':'madloop-initialization-profile-report-selection-v1','closed':True,'candidate_qualified':True,
        'production_artifact_qualified':False,'canonical_publication_admitted':False,'production_source_promotion':False,
        'performance_acceptance':False,'new_native_physics_evaluations':0,'style_sha256':identity(HERE/'style.typ')['sha256'],
        'packets':[],'private_inputs':{'controls':identity(HERE/binding['controls_manifest']),'candidate':identity(HERE/binding['candidate_manifest'])}}
    return data,selection


def routes_csv(candidate):
    out=io.StringIO(newline='');columns=('ABI_index','context','counter','kind','count','input_bits','arithmetic_bits','scope')
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    for block,direction in enumerate(('primary','check')):
        counts=candidate['functional']['directions'][direction]['counters']
        for i,name in enumerate(candidate['functional']['field_order']):
            kind='attempt' if i>=7 else 'unreported_error' if i==6 else 'final_refused' if i==5 else 'final_accepted'
            bits=(53,53,106,256,512,None,None,106,256,512)[i]
            writer.writerow({'ABI_index':block*10+i,'context':direction,'counter':name,'kind':kind,'count':counts[name],
                'input_bits':53,'arithmetic_bits':bits,'scope':'Context-lifetime completed full-DP bridge; excludes public pre-validation errors and Fortran cache hits; clear does not reset'})
    return out.getvalue().encode()


def clean_csv(candidate):
    out=io.StringIO(newline='');writer=csv.writer(out);writer.writerow(('arm','DSO_sha256','READY_seconds','outer_status','distinct_points','bootstrap_plus_warmup_calls','ordered_pairs','clock','formal_acceptance'))
    for name,dso,key,status in (('old',ORIGINAL_DSO,'old_ready_seconds',323),('prototype',PROTOTYPE_DSO,'candidate_ready_seconds',219)):
        writer.writerow((name,dso,format(candidate['clean'][key],'.17g'),status,1,2,1,candidate['clean']['clock'],False))
    return out.getvalue().encode()


def component_miss_csv(candidate):
    out=io.StringIO(newline='');columns=('fixture','slot','degree','component','monomial','component_residual','complex_difference_norm','reference_decimal','candidate_exact_decimal')
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    for row in candidate['reference']['component_relative_misses']:
        writer.writerow({key:json.dumps(row[key]) if isinstance(row[key],list) else row[key] for key in columns})
    return out.getvalue().encode()


def phase_csv(data):
    out=io.StringIO(newline='');writer=csv.writer(out);writer.writerow(('control','READY_seconds','create_register_seconds','DP_bridge_seconds','QP_bridge_seconds','READY_remainder_seconds','outer_calls','distinct_points','promoted'))
    for name,key in (('Original','baseline'),('FullO2control','o2')):
        row=data[key];writer.writerow((name,*[format(row[k],'.17g') for k in ('ready_seconds','setup_seconds','dp_seconds','qp_seconds','outside_dp_qp_setup_seconds')],2,1,False))
    return out.getvalue().encode()


def stock_csv(data):
    out=io.StringIO(newline='');columns=('backend','name','full_ms','skip_ms','full_minus_skip_estimate_ms','full_points','full_batch_wall_seconds','boot_estimate_seconds','initialization_seconds','stability_points','qp_probes','unstable_points','exceptional_points','maximum_within_backend_spread','timed_per_point_statuses_available','source_mg_revision','tirust_dso_sha256')
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    for row in data['stock']['rows']:writer.writerow(row)
    return out.getvalue().encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--check-preview',action='store_true')
    args=parser.parse_args()
    if args.preview or args.check_preview:
        require(not (args.preview and args.check_preview) and args.inputs is None,'Choose one preview operation')
        data,selection=derive(args.root.resolve())
        payloads={'initialization-profile-data.json':encoded(data),'initialization-profile-selection.json':encoded(selection),
            'phase-timings.csv':phase_csv(data['controls']),'stock-profile.csv':stock_csv(data['controls']),
            'candidate-routes.csv':routes_csv(data['candidate']),'clean-initialization.csv':clean_csv(data['candidate']),
            'reference-component-misses.csv':component_miss_csv(data['candidate'])}
        if args.check_preview:
            require(args.output is None,'Checking does not write report outputs')
            for name,value in payloads.items():require((HERE/name).read_bytes()==value,'Preview differs from pure replay: '+name)
            print(encoded({'verified':True,'production_artifact_qualified':False,'new_native_physics_evaluations':0}).decode(),end='')
            return
        require(args.output is not None and not args.output.exists(),'Fresh preview-data output required')
        args.output.mkdir(parents=True)
        for name,value in payloads.items():(args.output/name).write_bytes(value)
        print(encoded({'actual_preview_data_written':True,'production_artifact_qualified':False,'new_native_physics_evaluations':0}).decode(),end='')
        return
    require(args.inputs is not None and args.output is not None,'Controls require --inputs and a fresh --output')
    data=derive_controls(args.root.resolve(),read(args.inputs));require(not args.output.exists(),'Fresh output required');args.output.mkdir(parents=True)
    (args.output/'controls-data.json').write_bytes(encoded(data));(args.output/'phase-timings.csv').write_bytes(phase_csv(data));(args.output/'stock-profile.csv').write_bytes(stock_csv(data))
    print(encoded({'controls_replayed':True,'candidate_qualified':False,'PDF_created':False,'native_executed':False}).decode(),end='')

if __name__=='__main__':main()
