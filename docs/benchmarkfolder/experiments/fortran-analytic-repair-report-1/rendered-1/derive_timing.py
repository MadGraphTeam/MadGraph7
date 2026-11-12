#!/usr/bin/env python3
"""Derive report data from frozen text evidence only. No scientific execution."""
import argparse
from collections import Counter
from decimal import Decimal,localcontext
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
CAPSULE_SHA='f8bc6ce06201665abc0da7d79bcb55697a215d99f2258e9ba623453ebc33e6e9'
REVIEW_SHA='9e10f47d4599235efef6fb7908c5a710282a9ffbb89cbe53793e0e2cfa780306'
STYLE_SHA='68e21877f61f92de75525e8d088b7de3fc00d8223b3aef393babc6015b59255e'
PRODUCTION='0fcd2cd873d61d134837e1d9e87afb13a47824fd'
SCIENTIFIC='84e8caf51078c3fb5817582e94e9a6a62f1ad22fc4bf7909f5034fce23dbc31c'
ROLES=('original','candidate')

def require(ok,message):
    if not ok:raise ValueError(message)
def identity(path):
    raw=Path(path).read_bytes();return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def encoded(value):return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
def decimal(token):
    value=Decimal(token.replace('D','E').replace('d','e'))
    require(value.is_finite(),'Nonfinite timing field');return value
def median_mad(values):
    center=median(values);return center,median([abs(v-center) for v in values])

class Evidence:
    def __init__(self,root):
        self.root=Path(root).resolve();raw=(self.root/'index.json').read_bytes()
        require(hashlib.sha256(raw).hexdigest()==CAPSULE_SHA,'Exact completed capsule required')
        self.index=json.loads(raw);self.used={}
    def raw(self,name):
        require(name in self.index['logical_files'],'Unbound logical text: '+name)
        row=self.index['logical_files'][name];relative=Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe stored path')
        path=self.root/relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.root),'Escaping evidence')
        stored=path.read_bytes();wanted=self.index['files'][row['path']]
        require(len(stored)==wanted['bytes'] and hashlib.sha256(stored).hexdigest()==wanted['sha256'],'Stored evidence changed')
        require(row['compression'] in ('none','gzip'),'Unsupported text storage')
        raw=gzip.decompress(stored) if row['compression']=='gzip' else stored
        require(len(raw)==row['raw_bytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256'],'Logical text changed')
        self.used[name]={'raw_sha256':row['raw_sha256'],'raw_bytes':row['raw_bytes']}
        return raw
    def json(self,name):return json.loads(self.raw(name))

def derive(capsule):
    ev=Evidence(capsule);review_name='reviews/fortran-orig-axial-qualification-timing-independent-review-2/receipt.json'
    require(ev.index['logical_files'][review_name]['raw_sha256']==REVIEW_SHA,'Closed independent timing review changed')
    review=ev.json(review_name)
    require(review['status']=='passed_descriptive_shared_host_N1_only'
        and review['all_recorded_pairs_and_summary_reproduced'] and not review['performance_acceptance']
        and not review['production_promotion'],'Independent timing review scope mismatch')
    terminal=ev.json('timing/terminal.json')
    require(terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Timing did not close')
    require(terminal['result_sha256']==ev.index['phase_pins']['timing'][1],'Timing result lineage mismatch')
    batches=[];pairs=[];times={role:[] for role in ROLES};accepted=Counter()
    with localcontext() as context:
        context.prec=50
        for index in range(1,8):
            recorded=ev.json(f'timing/pair-{index}.json')
            order=list(ROLES) if index%2 else list(reversed(ROLES))
            require(recorded['pair']==index and recorded['order']==order and recorded['comparison']['passed'],'Fixed pair/science gate changed')
            seconds={}
            for role in ROLES:
                base=f'timing/{role}/pair-{index}/';tokens=ev.raw(base+'timing.dat').decode().split()
                require(len(tokens)==2,'Malformed raw timing batch')
                cold,elapsed=map(decimal,tokens)
                require(cold>0 and elapsed>0,'Nonpositive duration must be retained as a failed study')
                status={}
                for line in ev.raw(base+'timed_status.dat').decode().splitlines():
                    fields=line.split();require(len(fields)==2,'Malformed status histogram')
                    code,count=map(int,fields);require(code not in status and count>0,'Duplicate/invalid status')
                    status[code]=count
                require(status=={219:12800},'Accepted status/call-count changed')
                require(float(elapsed)==recorded[role]['seconds'] and recorded[role]['own_comparison']['passed'],'Raw/closed batch mismatch')
                seconds[role]=elapsed;times[role].append(elapsed*Decimal(1_000_000)/Decimal(12800));accepted[role]+=12800
                batches.append({'pair':index,'role':role,'position':order.index(role)+1,
                    'seconds':float(elapsed),'raw_seconds_token':tokens[1],
                    'cold_first_call_seconds':float(cold),'microseconds_per_point':float(times[role][-1]),
                    'accepted_status_counts':{'219':12800},'raw_timing_member':base+'timing.dat',
                    'raw_status_member':base+'timed_status.dat'})
            ratio=seconds['candidate']/seconds['original']
            pairs.append({'pair':index,'order':order,'original_seconds':float(seconds['original']),
                'candidate_seconds':float(seconds['candidate']),'candidate_over_original':float(ratio),
                'candidate_over_original_decimal':str(ratio)})
        summaries={}
        for role in ROLES:
            center,mad=median_mad(times[role])
            summaries[role]={'batches':7,'median_microseconds_per_point':float(center),
                'unscaled_MAD_microseconds_per_point':float(mad),'accepted219_calls':accepted[role]}
        paired=median([Decimal(p['candidate_over_original_decimal']) for p in pairs])
        ratio_medians=median(times['candidate'])/median(times['original'])
        require(abs(float(paired)-0.952790310414709)<1e-14,'Paired median differs from closed review')
        require(abs(paired-ratio_medians)>Decimal('.001'),'Distinct ratio estimands accidentally merged')
    data={'schema':'madloop-tirust-analytic-repair-report-data-v1',
        'process':{'command':'u u~ > z [virt=QCD]','initial_particles':2,'final_particles':1,'model':'loop_sm',
            'reduction_backend':9,'accepted_precision':'DP','tensor_propagators':3,'tensor_rank':2},
        'comparison':{'original_label':'Original 0fcd control','production_revision':PRODUCTION,
            'candidate_label':'Unapplied same-flag Fortran candidate','candidate_source_sha256':SCIENTIFIC,
            'repair_unit_compiler_flags':['-w','-fPIC','-ffixed-line-length-132'],'same_flags':True,'source_promoted':False},
        'design':{'pairs':7,'alternating_order':'original/candidate on odd pairs; candidate/original on even pairs',
            'physical_points':100,'traversals_per_batch':128,'calls_per_batch':12800,'cpu':12,'cpu_exclusive':False,
            'MAD_definition':'median(abs(batch_us_per_point - median(batch_us_per_point))); unscaled, not SEM',
            'timing_scope':'Warmed complete amplitude; final100 words retained per batch, status histogram covers all calls'},
        'arms':summaries,'batches':batches,'pairs':pairs,
        'paired_median_candidate_over_original':float(paired),'ratio_of_arm_medians_candidate_over_original':float(ratio_medians),
        'minimum_batch_seconds':min(b['seconds'] for b in batches),'maximum_batch_seconds':max(b['seconds'] for b in batches),
        'total_accepted219_calls':sum(accepted.values()),
        'qualification':{'outer_wrapper_invocations':544,'fixtures':136,'precisions':['DP','QP'],'arms':2,
            'standalone_DP_output_word_differences':0,'standalone_QP_changed_fixture_results':94,'standalone_QP_changed_component_words':233,
            'full_amplitude_points':100,'returned_overall_and_sole_split_order_rows':200,'coefficient_fields_per_arm':800,'accuracy_fields_per_arm':200,
            'full_amplitude_coefficient_and_accuracy_word_differences':0,'independent_stability_checks_retained':True,
            'physical_model_MP_mass_precision':'Existing binary64 buffer lifted to binary128',
            'batch_final100_sets':14,'coefficients_recorded_for_every_timed_call':False},
        'evidence':{'capsule_index_sha256':CAPSULE_SHA,'timing_review_sha256':REVIEW_SHA,'raw_members':ev.used},
        'limits':{'formal_performance_acceptance':False,'broad_process_model_qualification':False,
            'actual_guard_branch_coverage':False,'independent_finite_truth':False,'trap_freedom':False,
            'production_promotion':False,'precise_local_acceptance':'Pending Hua-Sheng local paired timing campaign'},
        'new_native_physics_evaluations':0}
    require(identity(HERE/'style.typ')['sha256']==STYLE_SHA,'Preserved style changed')
    selection={'schema':'madloop-tirust-analytic-repair-report-selection-v1','kind':'documentary one-process Fortran timing supplement',
        'packets':[{'label':'Analytic Fortran repair qualification','path':'experiments/fortran-analytic-repair-qualification-1',
            'index_name':'index.json','index_sha256':CAPSULE_SHA}],
        'timing_independent_review_sha256':REVIEW_SHA,'production_revision':PRODUCTION,
        'source_promoted':False,'production_source_promotion':False,'performance_acceptance':False,
        'new_native_physics_evaluations':0,'style_sha256':STYLE_SHA,
        'data':{'path':'analytic-repair-data.json','sha256':hashlib.sha256(encoded(data).encode()).hexdigest()}}
    return {'analytic-repair-data.json':encoded(data),'analytic-repair-selection.json':encoded(selection)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',type=Path,required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for name,text in derive(args.capsule).items():
        path=HERE/name
        if args.check:require(path.read_text()==text,'Derived report input changed: '+name)
        else:
            with path.open('x') as stream:stream.write(text)
    print(encoded({'status':'checked' if args.check else 'derived','new_native_physics_evaluations':0,
        'capsule_index_sha256':CAPSULE_SHA}).strip())
if __name__=='__main__':main()
