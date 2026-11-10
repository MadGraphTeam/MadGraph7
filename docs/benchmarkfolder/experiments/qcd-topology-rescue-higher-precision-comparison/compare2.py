#!/usr/bin/env python3
"""Compare retained private QCD traces offline; never execute a reducer."""
from pathlib import Path
from collections import Counter
from decimal import Decimal,localcontext
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
OLD=HERE.with_name('qcd-topology-rescue-native')
DECODER=HERE.with_name('qcd-loop-kinematics-observer-native')
OLD_SHA='c7d185c4991fdfacf5c76710d1790b7ccf31c5c2ec0f9270cc4f72ec3dfdfc32'
DECODER_SHA='6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac'
PROTO=HERE.with_name('qcd-topology-rescue-prototype')
PROTO_SHA='0af03d980acdac032abcb767713bf5906fa7fa53cf8b3c57da75b764281e3015'
PI=Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')
INPUT_TAGS={'EDGE_INITIAL','MASS2_DP','NUM_ORIGINAL','EDGE_ORDERED','MASS2_ORDERED',
            'CLOSURE_SUM','CLOSURE_ABS','CLOSURE_REF','SHIFTS','PAIR_DIFFSQ','PAIR_S_INITIAL',
            'PAIR_E_REFERENCE','PAIR_S_CLAMPED','PAIR_S_FINAL','S_COMPLEX','DISTANCES'}


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory,index_hash):
    require(sha(directory/'evidence-index.json')==index_hash,'Wrong frozen inventory.')
    entries=json.loads((directory/'evidence-index.json').read_text())['files']
    for name,entry in entries.items():
        p=directory/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'Unsafe inventory member.')
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Frozen member changed: '+name)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def one(events,tag):
    rows=[e for e in events if e['tag']==tag]
    require(len(rows)==1,'Expected exactly one '+tag)
    return rows[0]


def rational(x):
    return {'numerator':str(x.numerator),'denominator':str(x.denominator)}


def dec(x):
    return Decimal(x.numerator)/Decimal(x.denominator)


def metrics(a,b):
    diff=abs(a-b);den=abs(a)+abs(b)
    return {'left':rational(a),'right':rational(b),'absolute_error':str(dec(diff)),
            'symmetric_error':str(dec(2*diff/den)) if den else '0',
            'relative_to_right':str(dec(diff/abs(b))) if b else None,'exact':a==b}


def compute(directory,provisional):
    inventory(OLD,OLD_SHA);inventory(DECODER,DECODER_SHA);inventory(PROTO,PROTO_SHA)
    decoder=module('qcd_private_comparison_decoder',DECODER/'decode_trace.py')
    bindings={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for p in (OLD/'evidence-index.json',DECODER/'evidence-index.json',DECODER/'decode_trace.py'):
        bindings[str(p.relative_to(ROOT))]=sha(p)
    for p in (PROTO/'evidence-index.json',PROTO/'topology-design/TIRUST_interface.f',PROTO/'topology-design/CT_interface.f'):
        bindings[str(p.relative_to(ROOT))]=sha(p)
    tir_source=(PROTO/'topology-design/TIRUST_interface.f').read_text()
    nrs_source=(PROTO/'topology-design/CT_interface.f').read_text()
    require('SCALE=REAL(MU_R,KIND=KI_QTIR)**2' in tir_source and
            'NRS_SCALE=REAL(MU_R**2,KIND=KI_QNRS)' in nrs_source,'Scale source contract changed.')
    traces={};records={}
    for key,run in (('tirust512',directory),('tirust256',OLD/'attempt1/topology-c/backend-9'),
                    ('ninjars',OLD/'attempt1/topology-c/backend-8')):
        record=json.loads((run/'execution.json').read_text())
        require(record['native_executed'] and record['protected_inputs_unchanged'] and record['runtime_sha256_unchanged'],'Changed native inputs.')
        require(record['complete'] or (key=='tirust512' and provisional),'Incomplete native capture needs explicit provisional scope.')
        for name,field in (('trace.dat','trace_sha256'),('observed.log','log_sha256')):
            require(sha(run/name)==record[field],'Retained native output changed.')
        traces[key]=decoder.load_trace(run/'trace.dat',run/'observed.log');records[key]=record
        for name in ('execution.json','trace.dat','observed.log'):
            bindings[str((run/name).relative_to(ROOT))]=sha(run/name)
    require(not records['tirust512']['complete'] if provisional else records['tirust512']['complete'],'Scope does not match capture status.')
    def values(event,order):
        raw=decoder.decode_payload(event,order)
        if raw and 'real' in raw[0]:
            return [(decoder.finite_fraction(x['real']),decoder.finite_fraction(x['imag'])) for x in raw]
        return [decoder.finite_fraction(x) for x in raw]
    points={key:trace['points'][1] for key,trace in traces.items()}
    require(all(p['seq']==2 and p['point']==2 for p in points.values()),'Wrong stress point.')
    orders={key:trace['byte_order'] for key,trace in traces.items()}
    require(len(set(orders.values()))==1,'Byte order differs.')
    order=orders['tirust512']
    maps={key:{(c['ctmode'],c['group']):c for c in p['calls']} for key,p in points.items()}
    expected={(mode,group) for mode in (4,5) for group in range(1,117)}
    require(all(set(m)==expected and len(p['calls'])==232 for m,p in zip(maps.values(),points.values())),'Incomplete group/direction coverage.')
    def inputs(call):
        return [{k:e[k] for k in ('tag','i','j','value','words')} for e in call['events']
                if e['tag'] in INPUT_TAGS or e['tag'].startswith('CONTEXT_')]
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def raw_slots(key,call):
        if key=='ninjars':
            return values(one(call['events'],'NATIVE_CONTRACTED'),order)
        selected=[e for e in call['events'] if e['tag']=='CONTRACTION_RAW']
        require([e['i'] for e in selected]==[1,2,3],'Missing TIRust contraction slots.')
        return [values(e,order)[0] for e in selected]
    def converted_slots(key,call):
        if key=='ninjars':
            return values(one(call['events'],'DP_CONVERTED'),order)
        selected=[e for e in call['events'] if e['tag']=='DP_CONVERTED']
        require([e['i'] for e in selected]==[1,2,3],'Missing TIRust converted slots.')
        return [values(e,order)[0] for e in selected]
    group_rows=[]
    with localcontext() as ctx:
        ctx.prec=110
        for mode,group in sorted(expected):
            t=maps['tirust512'][(mode,group)];old=maps['tirust256'][(mode,group)];n=maps['ninjars'][(mode,group)]
            require(t['native_status']==0 and t['stable'] and n['native_status']==0 and n['stable'],'Unavailable matching stable contraction.')
            ti,oi,ni=inputs(t),inputs(old),inputs(n)
            ts=values(one(t['events'],'NATIVE_SCALE'),order)[0]
            os=values(one(old['events'],'NATIVE_SCALE'),order)[0]
            ns=values(one(n['events'],'NATIVE_SCALE'),order)[0]
            tt,nn=raw_slots('tirust512',t),raw_slots('ninjars',n)
            tc,nc=converted_slots('tirust512',t),converted_slots('ninjars',n)
            slots=[]
            for i,(a,b) in enumerate(zip(tt,nn)):
                slots.append({'slot':('finite','single','double')[i],
                              'real':metrics(a[0],b[0]),'imag':metrics(a[1],b[1]),
                              'actual_dp_projected_real':metrics(tc[i][0],nc[i][0]),
                              'actual_dp_projected_imag':metrics(tc[i][1],nc[i][1]),
                              'complex_absolute_error':str((dec(a[0]-b[0])**2+dec(a[1]-b[1])**2).sqrt())})
            group_rows.append({'ctmode':mode,'group':group,'n':t['n'],'rank':t['rank'],
                               'old256_status':old['native_status'],'old256_stable':bool(old['stable']),
                               'input_words_equal_to256':ti==oi,'input_words_equal_to_ninjars':ti==ni,
                               'input_digests':{'tirust512':digest(ti),'tirust256':digest(oi),'ninjars':digest(ni)},
                               'scale_tirust_mu2':rational(ts),'scale_tirust256_mu2':rational(os),'scale_ninjars_mu2':rational(ns),
                               'tirust_mu2_unchanged':ts==os,'native_mu2_equal':ts==ns,
                               'native_mu2_difference':rational(ts-ns),
                               'native_input_numerators_finite':t['numerical_summaries']['NUM_ORIGINAL']['all_finite'],
                               'contracted_laurent_slots':slots})
        host={};answers={}
        for key,p in points.items():
            ans=values(one(p['events'],'POINT_HOST_ANS'),order)
            require(len(ans)==8 and ans[:4]==ans[4:],'Split-order host values differ.')
            g=values(one(p['events'],'POINT_G_DP'),order)[0]
            prediction=dec(ans[0])*(-Decimal(26)/3)*dec(g)**2/(8*PI**2)
            host[key]={'host_code':p['host_code'],'ans':[rational(x) for x in ans[:4]],
                       'ans_decimal':[str(dec(x)) for x in ans[:4]],'g':rational(g),
                       'predicted_double_pole':str(prediction),
                       'double_pole_relative_error':str(abs(dec(ans[3])-prediction)/abs(prediction)),
                       'ieee_flags':{tag:one(p['events'],tag)['value'] for tag in ('POINT_INVALID','POINT_DIVZERO','POINT_OVERFLOW')},
                       'coordinate_words':{tag:one(p['events'],tag)['words'] for tag in
                                           ('POINT_P_USER','POINT_MP_BEFORE','POINT_MP_AFTER','POINT_MP_P')}}
            answers[key]=ans[:4]
        host_comparison={name:metrics(a,b) for name,a,b in zip(('born','finite','single','double'),answers['tirust512'],answers['ninjars'])}
        extremes={}
        for slot in ('finite','single','double'):
            for part in ('real','imag'):
                rows=[(g,s[part]) for g in group_rows for s in g['contracted_laurent_slots'] if s['slot']==slot]
                g,v=max(rows,key=lambda r:Decimal(r[1]['symmetric_error']))
                extremes[slot+':'+part]={'ctmode':g['ctmode'],'group':g['group'],'n':g['n'],'rank':g['rank'],**v}
    all_inputs=all(g['input_words_equal_to256'] and g['input_words_equal_to_ninjars'] and
                   g['tirust_mu2_unchanged'] and g['native_mu2_equal'] for g in group_rows)
    return {'schema':1,'native_executed_by_analyzer':False,'runtime_capture_complete':records['tirust512']['complete'],
            'provisional_raw_comparison':provisional,'physics_campaign_accepted':False,'production_promoted':False,
            'scope':'One private topology-C point,232 groups/directions and696 complex contracted Laurent slots; original113-bit inputs and historical promoted-DP squared-mass convention.',
            'scale_source_contract':{'tirust':'SCALE=REAL(MU_R,KIND=KI_QTIR)**2',
                                     'ninjars':'NRS_SCALE=REAL(MU_R**2,KIND=KI_QNRS)',
                                     'interpretation':'Both NATIVE_SCALE records are already mu squared; their different arithmetic constructions happen to agree exactly for this retained scale.'},
            'all_common_inputs_and_scale_semantics_exact':all_inputs,
            'same_original_and_effective_coordinate_words':len({digest(h['coordinate_words']) for h in host.values()})==1,
            'old256_refusals_now_served':sum(not g['old256_stable'] for g in group_rows),
            'group_calls':232,'complex_contracted_laurent_slots':696,'real_components':1392,
            'host':host,'host_512_vs_ninjars':host_comparison,'groups':group_rows,
            'worst_symmetric_component_per_slot_part':extremes,'bindings':bindings,
            'limits':['Working precision and the admission threshold changed jointly; their effects are not isolated.',
                      'NinjaRS trace contains contracted coefficients, not unit-numerator uncontracted tensors; no monomial-by-monomial truth is claimed.',
                      'Group component relative errors near zero must be read with absolute errors; the host amplitude criterion is separate.',
                      'The universal double pole is an independent normalization check; finite/single comparisons use the retained NinjaRS result, not analytic truth.',
                      'This uses changed effective C coordinates and a private topology policy; production/current campaign remain unchanged.',
                      'A successful host code and sticky IEEE flags are retained separately; no general rescue/precision policy is certified.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--provisional',action='store_true')
    args=parser.parse_args()
    require(not args.output.exists(),'Do not overwrite a comparison attempt.')
    result=compute(args.directory.resolve(),args.provisional)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'capture_complete':result['runtime_capture_complete'],
                      'same_inputs':result['all_common_inputs_and_scale_semantics_exact'],
                      'same_coordinates':result['same_original_and_effective_coordinate_words'],
                      'host_relative_errors':{k:v['relative_to_right'] for k,v in result['host_512_vs_ninjars'].items()},
                      'double_pole_error':result['host']['tirust512']['double_pole_relative_error'],
                      'groups':result['group_calls'],'slots':result['complex_contracted_laurent_slots']},sort_keys=True))


if __name__=='__main__':
    main()
