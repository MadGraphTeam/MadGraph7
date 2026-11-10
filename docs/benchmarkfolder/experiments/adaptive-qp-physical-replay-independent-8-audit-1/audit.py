"""Read-only comparison of retained fixed-QP backend8 and adaptive9 text."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from fractions import Fraction

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
POINT_OBSERVATIONS={'AMPLITUDE','ACCURACY','PUBLIC_STATUS','FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'}
PHYSICAL_GROUP_LABELS={'DP_M2L','QP_M2L','HELPER_DP_PL','HELPER_QP_PL','REPAIRED_MP_PS','CURRENT_MP_P',
                       'NUMERATOR_COEFS','ENTRY_PL','ENTRY_M2_QP','ORDERED_PL','ORDERED_MASSES'}
INTEGRAL_LABELS={'NATIVE_SHIFTS','KINEMATIC_S','NATIVE_INVARIANTS','NATIVE_MASSES','NATIVE_SCALE'}


def sha(data):return hashlib.sha256(data).hexdigest()


def canonical(data):
    return {label:[row[:2] if row[0] in ('LOGICAL','INTEGER') else row for row in rows] for label,rows in data.items()}


def normalize_card(text,backend):
    lines=text.splitlines(True)
    indices=[i for i,line in enumerate(lines) if line.strip()=='#MLReductionLib'];assert len(indices)==1
    index=indices[0]+1;assert lines[index].strip()==str(backend)
    lines[index]='EXPLICIT_BACKEND_SELECTION\n';return ''.join(lines)


def analyze(blobs,parser):
    with tempfile.TemporaryDirectory(prefix='adaptive-independent-8-audit-') as directory:
        paths={}
        for label in ('guarded8','adaptive9'):
            paths[label]=Path(directory)/(label+'.hex');paths[label].write_text(blobs[label+'/mass-observer.hex'])
        left=parser.parse(paths['guarded8'],'candidate',8);right=parser.parse(paths['adaptive9'],'candidate',9)
    assert normalize_card(blobs['guarded8/MadLoop5_resources/MadLoopParams.dat'],8)==normalize_card(blobs['adaptive9/MadLoop5_resources/MadLoopParams.dat'],9)
    cards=[]
    for label in sorted(k.removeprefix('guarded8/') for k in blobs if k.startswith('guarded8/') and not k.endswith(('mass-observer.hex','stdout.log','observer-run.log','param.log','MadLoopParams.dat'))):
        assert blobs['guarded8/'+label]==blobs['adaptive9/'+label],label;cards.append(label)
    failures=[];integral_differences=[];residuals=[];point_observations=[];native_counts={'8':0,'9':0}
    for number in (1,2):
        l,r=left['points'][number],right['points'][number]
        physical_l={k:v for k,v in l['data'].items() if k not in POINT_OBSERVATIONS}
        physical_r={k:v for k,v in r['data'].items() if k not in POINT_OBSERVATIONS}
        if canonical(physical_l)!=canonical(physical_r):failures.append([number,'physical_point_model_bits'])
        assert [g['key'] for g in l['groups']]==[g['key'] for g in r['groups']]
        for lg,rg in zip(l['groups'],r['groups'],strict=True):
            labels=PHYSICAL_GROUP_LABELS|{x for x in lg['data'] if x.startswith(('ARG_M','MP_ARG_M'))}
            if canonical({k:lg['data'][k] for k in labels})!=canonical({k:rg['data'][k] for k in labels}):
                failures.append([number,lg['key'][1],'physical_helper_MP_mass_numerator_bits'])
            for backend,g in ((8,lg),(9,rg)):
                assert len(g['native'])==1 and g['native'][0]['backend']==backend and g['native'][0]['status']==0
                assert g['data']['GROUP_STABLE'][0][1]=='T';native_counts[str(backend)]+=1
            for label in sorted(INTEGRAL_LABELS):
                if lg['data'][label]!=rg['data'][label]:
                    integral_differences.append({'point':number,'group':lg['key'][1],'label':label,
                                                'backend8':lg['data'][label],'adaptive9':rg['data'][label]})
        for index,(a,b) in enumerate(zip(l['data']['AMPLITUDE'],r['data']['AMPLITUDE'],strict=True),1):
            x,y=parser.row_values(a)[0],parser.row_values(b)[0];assert x is not None and y is not None
            relative=abs(x-y)/max(abs(x),abs(y)) if x or y else 0
            born=index in (1,5)
            if born and a!=b:failures.append([number,index,'Born_bits'])
            residuals.append({'point':number,'flat_index':index,'coefficient':('Born','finite','single_pole','double_pole')[(index-1)%4],
                              'backend8_bits':a,'adaptive9_bits':b,'backend8':parser.decimal(x),'adaptive9':parser.decimal(y),
                              'absolute_difference':parser.decimal(abs(x-y)),
                              'symmetric_relative_difference':parser.decimal(relative),
                              'within_3e_6':relative<=Fraction(3,10**6),'within_1e_8':relative<=Fraction(1,10**8)})
        point_observations.append({'point':number,'backend8':{k:l['data'][k] for k in POINT_OBSERVATIONS},
                                   'adaptive9':{k:r['data'][k] for k in POINT_OBSERVATIONS},
                                   'backend8_IEEE_begin_end':[l['begin_flags'],l['end_flags']],
                                   'adaptive9_IEEE_begin_end':[r['begin_flags'],r['end_flags']]})
    warnings={key:[line for line in blobs[key+'/stdout.log'].splitlines() if '##W03' in line] for key in ('guarded8','adaptive9')}
    assert not any(warnings.values())
    assert len(integral_differences)==200 and all(x['point']==2 and x['label']=='NATIVE_SCALE' for x in integral_differences)
    assert all(row['within_1e_8'] for row in residuals)
    return {'schema':'adaptive-fixed-QP-independent8-diagnostic-audit-v1','physical_input_and_Born_parity_passed':not failures,
            'failures':failures,'identical_nonselection_card_resource_files':cards,
            'native_served_counts':native_counts,'W03':warnings,'residuals':residuals,
            'native_integral_input_bits_identical':False,'native_integral_input_differences':integral_differences,
            'scale_difference_explanation':'Backend8 computes REAL(MU_R**2,QP), squaring in DP before promotion; backend9 computes REAL(MU_R,QP)**2. Frozen adapter source records both expressions.',
            'point_observations':point_observations,'per_call_working_precision_available':False,
            'same_input_independent_truth_qualified':False,'physical_accuracy_qualified':False,
            'checked_stability_qualified':False,'performance_qualified':False,
            'scope':'Independent fixed-QP amplitude diagnostic with identical physical inputs; explicit native scale-encoding difference atpoint2. Neither status120 nor agreement is checked truth.'}


def main():
    index=json.loads((HERE/'evidence-index.json').read_text())
    for name,record in index['files'].items():
        data=(HERE/name).read_bytes();assert sha(data)==record['sha256'] and len(data)==record['bytes']
    for name,digest in index['antecedent_indices'].items():assert sha((ROOT/name).read_bytes())==digest
    compressed=(HERE/'text-inputs.json.gz').read_bytes();blobs=json.loads(gzip.decompress(compressed))
    inventory=json.loads((HERE/'text-inventory.json').read_text())
    for name,text in blobs.items():assert sha(text.encode())==inventory[name]['sha256'] and len(text.encode())==inventory[name]['bytes']
    with tempfile.TemporaryDirectory(prefix='adaptive-audit-parser-') as directory:
        path=Path(directory)/'portable/replay/source/frozen/parser.py';path.parent.mkdir(parents=True);path.write_text(blobs['parser.py'])
        spec=importlib.util.spec_from_file_location('adaptive_independent_frozen_parser',path)
        parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
        result=analyze(blobs,parser)
    assert result==json.loads((HERE/'analysis.json').read_text())
    print(json.dumps({'verified':True,'physical_input_Born_parity':True,'native_scale_differences_explicit':200,
                      'native_executed':False,'same_integral_input_or_physics_truth_claim':False},sort_keys=True))


if __name__=='__main__':main()
