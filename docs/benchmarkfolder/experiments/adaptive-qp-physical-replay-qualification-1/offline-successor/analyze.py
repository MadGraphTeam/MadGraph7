"""Offline invariants for one unchanged-input adaptive physical diagnostic."""
import hashlib
import json
from pathlib import Path

POINT_OUTPUTS={'AMPLITUDE','ACCURACY','PUBLIC_STATUS',
               'FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'}
GROUP_OUTPUTS={'RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE','GROUP_RESULT','GROUP_STABLE'}
RAW={'RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE'}
EXPECTED_REFUSALS=[5,6,23,32,33,34,35,40,41,42,43,51,52,55,56,62,63,78,87,88,
                   92,93,105,108,109,115,142,143,147,148,149,151,172,176,179,180,183]


def canonical(data):
    # Event-time sticky IEEE flags are observations, not input values.
    return {k:None if v is None else [r[:2] if r[0] in ('LOGICAL','INTEGER') else r for r in v]
            for k,v in data.items()}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compare(old,new,old_stdout,new_stdout):
    """Require input parity and served-output preservation; report recoveries."""
    failures=[];served=[];transitions=[];flags=[];point_outputs=[];point_fpe=[];born=[]
    original_refusals={};calls=0
    if set(old['points'])!={1,2} or set(new['points'])!={1,2}:
        raise ValueError('Exactly the same two observed physical points are required')
    for number in (1,2):
        left,right=old['points'][number],new['points'][number]
        a={k:v for k,v in left['data'].items() if k not in POINT_OUTPUTS}
        b={k:v for k,v in right['data'].items() if k not in POINT_OUTPUTS}
        if canonical(a)!=canonical(b):failures.append([number,'point_input_or_model_bits'])
        # ANS(0:3,0:1) is stored in Fortran order. Each split-order block's
        # coefficient0 is Born; retain those physics checks independently of
        # changing loop sums after formerly omitted reductions are served.
        left_amplitude=left['data']['AMPLITUDE'];right_amplitude=right['data']['AMPLITUDE']
        if len(left_amplitude)!=8 or len(right_amplitude)!=8:
            failures.append([number,'expected_two_four_coefficient_amplitude_blocks'])
        else:
            same=all(left_amplitude[i]==right_amplitude[i] for i in (0,4))
            born.append({'point':number,'flat_indices':[1,5],
                         'old':[left_amplitude[i] for i in (0,4)],
                         'new':[right_amplitude[i] for i in (0,4)],'bits_preserved':same})
            if not same:failures.append([number,'Born_output_bits_changed'])
        for label in ('FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'):
            if left['data'][label]!=right['data'][label]:
                point_fpe.append({'point':number,'label':label,
                                  'old':left['data'][label],'new':right['data'][label],
                                  'classification':'Observed host FPE state; not physical/model/native input'})
        if [g['key'] for g in left['groups']]!=[g['key'] for g in right['groups']]:
            failures.append([number,'group_sequence']);continue
        original_refusals[number]=[]
        for l,r in zip(left['groups'],right['groups'],strict=True):
            group=l['key'][1];calls+=1
            a={k:v for k,v in l['data'].items() if k not in GROUP_OUTPUTS}
            b={k:v for k,v in r['data'].items() if k not in GROUP_OUTPUTS}
            if canonical(a)!=canonical(b):failures.append([number,group,'native_input_bits_or_cache_check_policy'])
            if len(l['native'])!=1 or len(r['native'])!=1:
                failures.append([number,group,'one_native_call_required']);continue
            ln,rn=l['native'][0],r['native'][0]
            if ln['backend']!=9 or rn['backend']!=9:failures.append([number,group,'explicit_backend9'])
            if ln['status']==0:
                labels=GROUP_OUTPUTS
                same=rn['status']==0 and canonical({k:l['data'].get(k) for k in labels})==canonical({k:r['data'].get(k) for k in labels})
                served.append({'point':number,'group':group,'input_sha256':digest(canonical(a)),
                               'old_served_output_bits_preserved':same})
                if not same:failures.append([number,group,'previously_served_output_or_status_changed'])
            else:
                original_refusals[number].append(group)
                if ln['status']!=1 or len(l['errors'])!=1 or not l['errors'][0].startswith('TIRust QP: Unstable("ill-conditioned Gram matrix for '):
                    failures.append([number,group,'unexpected_original_refusal'])
                transitions.append({'point':number,'group':group,'key':l['key'],
                                    'input_sha256':digest(canonical(a)),
                                    'old_status':ln['status'],'new_status':rn['status'],
                                    'old_errors':l['errors'],'new_errors':r['errors'],
                                    'new_raw_slots':sorted(RAW & r['data'].keys()),
                                    'new_group_result':r['data'].get('GROUP_RESULT'),
                                    'new_group_stable':r['data'].get('GROUP_STABLE'),
                                    'actual_working_precision_bits':None,
                                    'independent_numerical_truth_available':False})
            if l['flags']!=r['flags'] or ln['flags']!=rn['flags']:
                flags.append({'point':number,'group':group,'old_group_flags':l['flags'],'new_group_flags':r['flags'],
                              'old_native_flags':ln['flags'],'new_native_flags':rn['flags']})
        point_outputs.append({'point':number,'old':{k:left['data'][k] for k in POINT_OUTPUTS},
                              'new':{k:right['data'][k] for k in POINT_OUTPUTS},
                              'old_begin_flags':left['begin_flags'],'new_begin_flags':right['begin_flags'],
                              'old_end_flags':left['end_flags'],'new_end_flags':right['end_flags']})
    if original_refusals!={1:[],2:EXPECTED_REFUSALS}:failures.append(['original_refusal_fixture_changed'])
    if calls!=400 or len(served)!=363 or len(transitions)!=37:failures.append(['original_call_counts_changed'])
    warnings=lambda text:[{'line':i,'text':line} for i,line in enumerate(text.splitlines(),1) if '##W03' in line]
    old_warnings,new_warnings=warnings(old_stdout),warnings(new_stdout)
    if len(old_warnings)!=74:failures.append(['original_W03_fixture_changed'])
    all_served=all(n['status']==0 for p in new['points'].values() for g in p['groups'] for n in g['native'])
    all_stable=all(g['data']['GROUP_STABLE'][0][1]=='T' for p in new['points'].values() for g in p['groups'])
    return {'schema':'adaptive-physical-same-input-diagnostic-v2-offline-classification-only','invariants_passed':not failures,
            'failures':failures,'observed_native_calls':calls,'previously_served_calls':served,
            'previously_refused_calls':transitions,'refusals_now_served':sum(t['new_status']==0 for t in transitions),
            'old_W03':old_warnings,'new_W03':new_warnings,'group_or_native_IEEE_flag_differences':flags,
            'point_outputs':point_outputs,'point_FPE_state_differences':point_fpe,
            'Born_bit_checks':born,'old_footer':old['footer'],'new_footer':new['footer'],
            'observed_sums_have_no_recorded_native_omissions':all_served and all_stable and not new_warnings,
            'precision_transport':'Same captured genuine binary128 inputs; no new per-call precision telemetry',
            'actual_per_call_working_precision_available':False,'fixed_CTMode4_is_checked':False,
            'physical_accuracy_qualified':False,'stability_or_performance_qualified':False,
            'independent_truth_gate':'Same-input independent reducer or precision-convergence evidence remains required.'}
