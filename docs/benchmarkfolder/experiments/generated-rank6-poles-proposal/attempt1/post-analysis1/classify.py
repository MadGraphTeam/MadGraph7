"""Classify retained native refusals and input differences, without rerunning."""
from collections import Counter
from decimal import Decimal as D
import json
from pathlib import Path
import runpy
import struct

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[4]

def classify():
    a=runpy.run_path(str(PARENT/'analyze.py'))
    schedule=json.loads((PARENT/'schedule.json').read_text())
    rows=[];native={};captures=[]
    for model in ('ew','qcd'):
        directory=ROOT/'.tirust-build/generated-rank6-poles-1'/model
        cap,cases=a['parse_records'](directory/'records.txt',model,[r for r in schedule if r['model']==model])
        rows+=cases;captures+=cap
        native.update(a['parse_observer'](directory/'observer.txt',[r['action'] for r in cap+cases]))
    refused=[]
    for row in rows:
        frame=native[row['action']]
        if row['backend']==9 and (frame['status'] or not row['stable']):
            error=bytes.fromhex(frame['error']).decode('utf8')
            refused.append({'action':row['action'],'fixture':row['fixture'],'model':row['model'],
                'public_precision':'DP' if row['ct']<4 else 'QP','ct':row['ct'],'finite':bool(row['finite']),
                'pole_mode':row['mode'],'native_kind':frame['kind'],'native_status':frame['status'],
                'host_stable':row['stable'],'native_error':error,
                'failure_stage':'native DP Gram conditioning gate' if frame['kind']==1 and frame['status']==1 and 'ill-conditioned Gram matrix' in error else 'unclassified'})
    differences=[]
    for fixture in range(101,108):
        trow=next(r for r in rows if r['fixture']==fixture and r['backend']==9 and r['ct']==1 and r['mode']==0 and r['finite'])
        crow=next(r for r in rows if r['fixture']==fixture and r['backend']==7 and r['mode']==0)
        t,c=native[trow['action']],native[crow['action']];n=trow['n']
        p,s,m=(a['native_values'](t,key) for key in ('p','s','m'))
        cp,cs,cm=([D.from_float(x[0]) for x in struct.iter_unpack('<d',bytes.fromhex(c[key]))] for key in ('cll_p','cll_s','cll_m'))
        pairs=[]
        for length in range(1,n):
            for start in range(n):
                if len(pairs)==n*(n-1)//2:break
                pairs.append((start,(start+length)%n))
        invariants=[{'cyclic_slot_1based':i+1,'pair_0based':list(pair),'tirust':str(s[pair[0]+pair[1]*n]),
                     'collier':str(value),'delta':str(s[pair[0]+pair[1]*n]-value)}
                    for i,(pair,value) in enumerate(zip(pairs,cs[::2])) if s[pair[0]+pair[1]*n]!=value]
        for mode in (1,2):
            other=next(r for r in rows if r['fixture']==fixture and r['backend']==9 and r['ct']==1 and r['mode']==mode and r['finite'])
            assert all(native[other['action']][key]==t[key] for key in ('p','s','m','mu'))
            other=next(r for r in rows if r['fixture']==fixture and r['backend']==7 and r['mode']==mode)
            assert all(native[other['action']][key]==c[key] for key in ('cll_p','cll_s','cll_m'))
        differences.append({'fixture':fixture,'dp_action':trow['action'],'collier_action':crow['action'],
            'first_tirust_shift':[str(v) for v in p[:4]],'collier_first_shift':'implicit exact zero',
            'other_momenta_equal':cp[::2]==p[4:] and all(v==0 for v in cp[1::2]),'masses_equal':cm==m,
            'invariant_differences':invariants,'identical_across_pole_modes':True})
    native_ieee=[{'action':row['action'],'kind':row['kind'],'status':row['status'],'flags_before':row['flags'][0],
                  'flags_after':row['flags'][1],'new_bad_mask':(row['flags'][1]&13)&~(row['flags'][0]&13)}
                 for row in native.values() if row['present']]
    return {'schema':1,'scope':'Offline retained tensor evidence only; no amplitude return-code interpretation or numerical rerun',
            'refused_rows':refused,'refusal_count':len(refused),'errors':dict(Counter(r['native_error'] for r in refused)),
            'public_qp_rows':sum(r['backend']==9 and r['ct']>=4 for r in rows),
            'public_qp_native_success':sum(r['backend']==9 and r['ct']>=4 and native[r['action']]['status']==0 and r['stable'] for r in rows),
            'input_differences':differences,'native_ieee':native_ieee,
            'interpretation':{'dp_refusals':'All45 are explicit DP Gram-conditioning refusals; successful fixed256 QP already supplies the next precision tier in these standalone diagnostics.',
                'unsupported':'No unsupported-rank/topology or ERROR status in these45 refusals.',
                'independent_checks':'The18 physical COLLIER comparisons remain unqualified: implicit/explicit first shift and cyclic invariant bytes differ. No mismatched inputs are repaired or counted as agreement.',
                'ieee':'Host frames inherit bad flags from earlier work. Native entry/exit flags are recorded separately; a new native divide-by-zero flag on failed DP full calls is distinguishable from inherited host state. Neither implies an accuracy verdict for a successful QP tensor.',
                'prospective_common_input':'A separate diagnostic could feed both reducers the exact captured COLLIER contract (implicit zero first shift, identical remaining vectors, cyclic S and complex masses), promote those DP words exactly for QP, and compare mapped tensors with existing normalization. This would test mapping/poles for those common inputs only; it would not validate production physical kinematic construction.',
                'prospective_ieee':'A separate fresh process loading saved tensor fixtures without prior amplitude/capture calls could record flags before model init, before adapter, at native entry and exit. Preserve original sticky-flag evidence; do not clear it retroactively.'}}

if __name__=='__main__':print(json.dumps(classify(),indent=2,sort_keys=True))
