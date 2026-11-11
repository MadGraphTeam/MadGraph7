#!/usr/bin/env python3
"""Pure scope-preserving summary of parity-qualified retained profile records."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

STAGES={1:'initialization_and_budget',2:'kinematic_preprocessing',3:'allocation_and_input_decimal_encoding',
        4:'native_inclusive_QP_or_complete_DP_module_call',5:'output_decimal_decode_and_tensor_mapping',
        6:'contraction',7:'cache_clear'}
COUNTERS=('hits','misses','entries','bytes','not_stored','bypassed')


def read(path):return json.loads(Path(path).read_text())


def analyze(root):
    root=Path(root);modes=[read(root/f'mode-{i}/result.json') for i in range(3)]
    for i,result in enumerate(modes):
        if result['status']!='passed' or result['mode']!=i or result['warmup_points']!=100 or result['actual_anchor_calls']!=2:
            raise ValueError('A complete passing mode is required before analysis')
        if result['rows']!=modes[0]['rows'] or result['ieee_state']!=modes[0]['ieee_state']:
            raise ValueError('Numerical/actual IEEE parity missing')
    points=[]
    for point in ('1','16'):
        clock=modes[1]['profile'][point];rate=clock['clock_rate'];groups={}
        for row in clock['timers']:
            stage,ct,n,rank,precision=row['key'];key=(precision,ct,n,rank)
            target=groups.setdefault(key,{'precision':precision,'ctmode':ct,'n':n,'rank':rank,'stages':{}})
            if stage in target['stages']:raise ValueError('Duplicate timer scope')
            target['stages'][stage]={'calls':row['calls'],'ticks':row['ticks'],'seconds':row['ticks']/rate}
        totals={}
        for row in groups.values():
            key=(row['precision'],row['ctmode'])
            scope=totals.setdefault(key,{stage:{'calls':0,'ticks':0} for stage in STAGES})
            for stage,value in row['stages'].items():
                scope[stage]['calls']+=value['calls'];scope[stage]['ticks']+=value['ticks']
        scopes=[]
        for (precision,ct),stages in sorted(totals.items()):
            values=[{'stage':stage,'name':STAGES[stage],**v,'seconds':v['ticks']/rate}
                    for stage,v in sorted(stages.items()) if v['calls']]
            if any(not math.isfinite(v['seconds']) or v['seconds']<0 for v in values):raise ValueError('Invalid timer')
            scopes.append({'precision':precision,'ctmode':ct,'stages':values})
        trace=modes[2]['shadow'][point];by_topology={};call_scopes=defaultdict(int);repeat_cases=[]
        for call in trace['calls']:
            key=(call['precision'],call['ctmode'],call['n'],call['rank'])
            target=by_topology.setdefault(key,{'precision':key[0],'ctmode':key[1],'n':key[2],'rank':key[3],
                'tensor_calls':0,'valid_tensor_calls':0,'eligible_primary_full_qp_calls':0,'prior_exact_key_repeats':0,
                'native_counter_deltas':{name:0 for name in COUNTERS}})
            target['tensor_calls']+=1;target['valid_tensor_calls']+=int(call['validated_tensor'])
            target['eligible_primary_full_qp_calls']+=int(call['primary_full_qp_eligible'])
            target['prior_exact_key_repeats']+=int(call['natural_prior_valid_same_event_key'])
            call_scopes[(call['precision'],call['ctmode'],call['status'],call['validated_tensor'])]+=1
            snapshots=call['native_snapshots'];delta=None
            if snapshots:
                if len(snapshots)!=2 or [s['phase'] for s in snapshots]!=[1,2] or any(s['status']!=0 for s in snapshots):
                    raise ValueError('Native per-call before/after scope differs')
                delta=[b-a for a,b in zip(snapshots[0]['counts'],snapshots[1]['counts'])]
                if len(delta)!=6 or any(v<0 for v in delta):raise ValueError('Unexpected reset inside native call')
                for name,value in zip(COUNTERS,delta):target['native_counter_deltas'][name]+=value
            if call['natural_prior_valid_same_event_key']:
                repeat_cases.append({k:call[k] for k in ('call','epoch','ctmode','n','rank','precision','full','poles','cache_enabled')}
                    |{'native_counter_delta':dict(zip(COUNTERS,delta)) if delta is not None else None})
        for key,target in by_topology.items():
            target['timer_stages']={str(stage):value for stage,value in groups.get(key,{}).get('stages',{}).items()}
        qp={stage:sum(v[stage]['ticks'] for key,v in totals.items() if key[0]==2) for stage in range(1,7)}
        denominator=sum(qp.values())
        points.append({'point':int(point),'accepted_status':next(row['status'] for row in modes[0]['rows'] if row['point']==int(point)),
            'timer_scopes':scopes,'per_topology':[by_topology[k] for k in sorted(by_topology)],
            'tensor_call_status_scopes':[{'precision':k[0],'ctmode':k[1],'native_status':k[2],'validated_tensor':k[3],'calls':v}
                                       for k,v in sorted(call_scopes.items())],
            'eligible_primary_full_qp_calls':trace['eligible_calls'],
            'prior_exact_key_repeats':trace['potential_prior_exact_key_repeats'],'repeat_cases':repeat_cases,
            'qp_recorded_sequential_intervals':{'total_seconds':denominator/rate,
                'stages':[{'stage':stage,'name':STAGES[stage],'seconds':ticks/rate,
                           'fraction_of_recorded_qp_intervals':ticks/denominator if denominator else None} for stage,ticks in qp.items()],
                'scope':'Only served-QP sequential stages1..6; module decode and subsequent wrapper mapping share stage5. Excludes clear/host/numerator/probe overhead; not full-amplitude runtime.'},
            'clock_rate':rate})
    return {'schema':'private-qp-transport-profile-analysis-v1','status':'parity-qualified descriptive profile',
        'points':points,'warmup_points_per_mode':100,'anchor_calls_per_mode':2,'modes':3,
        'all_numerical_status_accuracy_words_equal':True,'observed_ieee_states_equal':True,
        'precision_labels':{'0':'lifecycle','1':'DP','2':'Fortran113-bit/MPFR256 QP'},
        'limitations':['One instrumented occurrence per point/mode on a shared host; no precision timing confidence interval.',
            'Raw clocks are not corrected for probe overhead. DP stage4 includes its full Fortran module call; QP stage4 is the native FFI only.',
            'QP stages1..6 are sequential in the served path; stage5 counts two sequential pieces per tensor. Fractions apply only to that recorded interval sum.',
            'Per-topology clocks aggregate hits and misses; repeated-call-specific elapsed costs are not measured.',
            'Exact-key repeats are opportunities, not a Fortran cache implementation, budget simulation or speedup estimate.',
            'Actual sticky IEEE pre/post states are observed; true-to-true does not identify newly raised exceptions.'],
        'optimization_applied':False,'performance_acceptance':False,'native_executions_by_analyzer':0}


if __name__=='__main__':
    print(json.dumps(analyze(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent),indent=2,sort_keys=True,allow_nan=False))
