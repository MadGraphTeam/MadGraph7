#!/usr/bin/env python3
"""Offline original/guarded mass-axis comparisons; fixed modes are unchecked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

if sys.flags.optimize:raise RuntimeError('Evidence validation requires assertions')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
NONMASS={'DP_M2L','HELPER_DP_PL','HELPER_QP_PL','REPAIRED_MP_PS','CURRENT_MP_P',
         'NUMERATOR_COEFS','ENTRY_PL','ORDERED_PL','NATIVE_SHIFTS','NATIVE_SCALE','NRS_NUMERATOR'}
POLICY={'ACTUAL_BACKEND','CHECK_PASS','CACHE_ENABLED'}
MASS={'ENTRY_M2_DP','ENTRY_M2_QP','QP_M2L','ORDERED_MASSES','NATIVE_MASSES'}
KINEMATIC={'KINEMATIC_S','NATIVE_INVARIANTS'}
OUTPUT={'GROUP_RESULT','GROUP_STABLE','RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE','NRS_RESULT','NRS_R1'}
POINT_OUTPUT={'AMPLITUDE','ACCURACY','PUBLIC_STATUS','FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'}


def require(test,message):
    if not test:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def semantic(rows):
    return [row[:2] if row[0] in ('INTEGER','LOGICAL') else row for row in rows]


def metadata(point):
    return {'begin_IEEE':point['begin_flags'],'end_IEEE':point['end_flags'],
            'point_status_flags':{key:rows for key,rows in point['data'].items() if key in POINT_OUTPUT and key not in ('AMPLITUDE','ACCURACY')},
            'groups':[{'key':group['key'],'begin_IEEE':group['flags'],'native':group['native'],'errors':group['errors'],
                       'scalar_status_flags':{key:rows for key,rows in group['data'].items()
                                              if rows and rows[0][0] in ('INTEGER','LOGICAL')}} for group in point['groups']]}


def validate_flags(capture):
    def flag(word):require(re.fullmatch('[TF]{5}',word) is not None,'Malformed host IEEE flags')
    for point in capture['points'].values():
        flag(point['begin_flags']);flag(point['end_flags'])
        for owner in [point,*point['groups']]:
            if 'flags' in owner:flag(owner['flags'])
            for rows in owner['data'].values():
                for row in rows:
                    if row[0] in ('INTEGER','LOGICAL'):flag(row[2])
            for native in owner.get('native',[]):flag(native['flags'])


def compare(left,right,paired_mass_axis):
    """Exact nonmass evidence is required; allowed differences remain explicit."""
    validate_flags(left);validate_flags(right)
    failures=[];changes=[];points=[]
    def changed(number,group,label,a,b,category):
        if a!=b:
            changes.append({'point':number,'group':group,'label':label,'category':category,
                            'left_records':len(a) if isinstance(a,list) else None,
                            'right_records':len(b) if isinstance(b,list) else None,
                            'left_sha256':value_hash(a),'right_sha256':value_hash(b)})
    for number in (1,2):
        a,b=left['points'][number],right['points'][number]
        require(set(a['data'])==set(b['data']),'Unrecognized point record-set difference')
        for label in sorted(a['data']):
            if label in POINT_OUTPUT:
                changed(number,0,label,a['data'][label],b['data'][label],'amplitude_accuracy_status_or_flags')
            elif semantic(a['data'][label])!=semantic(b['data'][label]):
                failures.append({'point':number,'group':0,'label':label,'reason':'nonmass_point_or_model_input'})
        require([g['key'] for g in a['groups']]==[g['key'] for g in b['groups']],'Group order/shape changed')
        for old,new in zip(a['groups'],b['groups'],strict=True):
            group=old['key'][1]
            for label in sorted(set(old['data'])|set(new['data'])):
                x,y=old['data'].get(label,[]),new['data'].get(label,[])
                argument=re.fullmatch(r'ARG_M([1-5])',label)
                mp_argument=re.fullmatch(r'MP_ARG_M([1-5])',label)
                if argument:require(int(argument[1])<=old['key'][4],'Invalid mass argument index')
                if mp_argument:require(int(mp_argument[1])<=old['key'][4],'Invalid MP mass argument index')
                if label in NONMASS or label in POLICY or argument:
                    if semantic(x)!=semantic(y):failures.append({'point':number,'group':group,'label':label,'reason':'nonmass_group_input_or_policy'})
                    changed(number,group,label,x,y,'intermediate_scalar_flag_timing' if semantic(x)==semantic(y) else 'unexpected_nonmass_input')
                elif label in MASS or mp_argument:
                    if not paired_mass_axis and semantic(x)!=semantic(y):
                        failures.append({'point':number,'group':group,'label':label,'reason':'same_mass_policy_input_changed'})
                    changed(number,group,label,x,y,'mass_argument_or_squared_mass')
                elif label in KINEMATIC:
                    if not paired_mass_axis and semantic(x)!=semantic(y):
                        failures.append({'point':number,'group':group,'label':label,'reason':'same_mass_policy_kinematics_changed'})
                    changed(number,group,label,x,y,'mass_dependent_kinematic_matrix_or_invariants')
                elif label in OUTPUT:
                    changed(number,group,label,x,y,'native_tensor_or_contracted_result')
                else:raise ValueError('Unclassified observer field '+label)
            changed(number,group,'NATIVE',old['native'],new['native'],'native_status_and_IEEE_flags')
            changed(number,group,'ERROR_BYTES',old['errors'],new['errors'],'native_failure_message')
            if paired_mass_axis:
                # Native entry sees the squared mass array in its own precision.
                require(old['data']['ENTRY_M2_DP']==old['data']['DP_M2L'],'Original entry differs from DP squared masses')
                require(new['data']['ENTRY_M2_QP']==new['data']['QP_M2L'],'Guarded entry differs from QP squared masses')
        points.append({'point':number,'left_event_metadata':metadata(a),'right_event_metadata':metadata(b),
            'left_native_refusals':[{'group':g['key'][1],'native':g['native'],'errors':g['errors']}
                                    for g in a['groups'] if any(n['status']!=0 for n in g['native'])],
            'right_native_refusals':[{'group':g['key'][1],'native':g['native'],'errors':g['errors']}
                                     for g in b['groups'] if any(n['status']!=0 for n in g['native'])]})
    categories={}
    for change in changes:categories[change['category']]=categories.get(change['category'],0)+1
    return {'exact_invariant_inputs_match':not failures,'invariant_failures':failures,
            'mass_transport_axis':paired_mass_axis,'changes':changes,'change_categories':categories,'points':points,
            'exact_all_observed_records_match':left==right,
            'crossarm_numeric_result_equality_required':False,
            'intermediate_IEEE_event_timing_required_equal':False}


def main():
    spec=json.loads((HERE/'child-prepared.json').read_text())
    for name in ('frozen_parser','frozen_support'):
        require(sha(spec[name]['path'])==spec[name]['sha256'],'Frozen analysis dependency changed')
    parser=load('original_qp_frozen_parser',spec['frozen_parser']['path'])
    support=load('original_qp_frozen_log_checks',spec['frozen_support']['path'])
    # This comparator is deliberately not a substitute for the additive
    # controller's new execution/runtime envelope. It also requires completion
    # of the prior combined campaign before reading its guarded-QP counterpart.
    prior=json.loads(Path(spec['required_prior_run']['path']).read_text())
    require(prior['state']=='complete' and prior['manifest_sha256']==spec['parent_prepared_sha256'],
            'Missing matching completed combined run')
    comparisons=[];journals={}
    for run in spec['runs']:
        backend=run['backend'];counterpart=spec['counterparts'][str(backend)]
        for p,digest in counterpart['input_sha256'].items():require(sha(p)==digest,'Guarded input changed')
        require(sha(counterpart['executable'])==counterpart['executable_sha256'],'Guarded executable changed')
        codes=support.check_run({**run,'expected_outputs':run['scientific_outputs']})
        guarded_run={**counterpart['run'],'mode':4,'point_count':2,'input_sha256':counterpart['input_sha256'],
                     'expected_outputs':['mass-observer.hex','observer-run.log','param.log']}
        guarded_codes=support.check_run(guarded_run)
        paths={'current_original':Path(run['cwd'])/'mass-observer.hex','current_guarded':Path(counterpart['journal'])}
        historical=spec['historical_oracles'][str(backend)]
        for arm in ('original','candidate'):
            entry=historical[arm];require(sha(entry['path'])==entry['sha256'],'Historical oracle changed')
            paths['historical_'+arm]=Path(entry['path'])
        capture={label:parser.parse(path,'original' if label.endswith('original') else 'candidate',backend)
                 for label,path in paths.items()}
        require(codes==[int(capture['current_original']['points'][p]['data']['PUBLIC_STATUS'][0][1]) for p in (1,2)],'Original log/status mismatch')
        require(guarded_codes==[int(capture['current_guarded']['points'][p]['data']['PUBLIC_STATUS'][0][1]) for p in (1,2)],'Guarded log/status mismatch')
        primary=compare(capture['current_original'],capture['current_guarded'],True)
        original_runtime=compare(capture['historical_original'],capture['current_original'],False)
        guarded_runtime=compare(capture['historical_candidate'],capture['current_guarded'],False)
        comparisons.append({'backend':backend,'same_current_runtime_mass_axis':primary,
            'secondary_original_historical49af_vs_currenteb3d':original_runtime,
            'secondary_guarded_historical49af_vs_currenteb3d':guarded_runtime})
        journals[str(backend)]={label:{'path':str(path),'sha256':sha(path)} for label,path in paths.items()}
    passed=all(item[key]['exact_invariant_inputs_match'] for item in comparisons for key in
        ('same_current_runtime_mass_axis','secondary_original_historical49af_vs_currenteb3d','secondary_guarded_historical49af_vs_currenteb3d'))
    result={'schema':'current-original-qp-offline-mass-axis-v1','comparison_passed':passed,
        'child_prepared_sha256':sha(HERE/'child-prepared.json'),'comparisons':comparisons,'journal_bindings':journals,
        'prior_completed_run_sha256':sha(spec['required_prior_run']['path']),
        'requires_additive_controller_execution_and_runtime_envelope':True,
        'physical_accuracy_or_host_stability_acceptance':False,'broader_graph_or_born_qualification':False,
        'performance_claim':False,'historical49af_is_same_runtime_control':False}
    print(json.dumps(result,indent=2,sort_keys=True))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
