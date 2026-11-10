"""Strict common-input comparison; no physical evidence is substituted."""
from decimal import Decimal as D,localcontext
import hashlib
import json
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent

def parser():
    source=(PARENT/'analyze.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='d3e6583162f3e3efe74bf933cda34e82eb0d8f606f752c475d15c6a3ac06eb4e'
    changes={
        '[1,0 if model=="ew" else 1,nc]':'[2,0,210]',
        'for action,k in ([(901,1),(902,2)] if model=="ew" else [(903,1)]):':'for action,k in []:'}
    for before,after in changes.items():
        assert source.count(before)==1;source=source.replace(before,after)
    ns={'__name__':'common_input_strict_parser','__file__':str(PARENT/'analyze.py')}
    exec(compile(source,str(PARENT/'analyze.py'),'exec'),ns)
    return ns

def _analyze(directory):
    directory=Path(directory);a=parser();require=a['require']
    fixtures={r['id']:r for r in json.loads((HERE/'fixtures.json').read_text())['fixtures']}
    schedule=json.loads((HERE/'schedule.json').read_text())
    lines=(directory/'records.txt').read_text().splitlines()
    require(lines[1].split()[0]=='INITIAL_IEEE' and lines[2].split()[0]=='MODEL_IEEE','Fresh-process IEEE observations missing')
    initial=list(map(int,lines[1].split()[1:]));model=list(map(int,lines[2].split()[1:]))
    require(len(initial)==len(model)==2,'IEEE observation schema')
    class Records:
        def read_text(self):return '\n'.join(lines[:1]+lines[3:])+'\n'
        def __str__(self):return str(directory/'records.txt')
    captures,cases=a['parse_records'](Records(),'ew',schedule)
    require(not captures and len(cases)==30,'Unexpected physical captures or call count')
    native=a['parse_observer'](directory/'observer.txt',[r['action'] for r in cases])
    open_items=[];checks=[];witnesses={}
    def issue(kind,action,detail):open_items.append({'kind':kind,'action':action,'detail':detail})
    def exact(left,right):
        return len(left)==len(right) and all(x==y and (x!=0 or x.is_signed()==y.is_signed()) for x,y in zip(left,right))
    if initial[0]&13:issue('initial_ieee',0,initial)
    if model[0]&13:issue('model_ieee',0,model)
    for row in cases:
        spec=fixtures[row['fixture']];frame=native[row['action']];n=spec['n']
        expected={'p':[[w for value in leg for w in a['promoted'](value)] for leg in spec['external_pl']],
                  'm':[[w for value in mass for w in a['promoted'](value)] for mass in spec['masses_squared']]}
        require(row['entry']==row['exit']==expected,'Synthetic input or unexpected adapter mutation')
        require(row['scale']=='4020000000000000','Scale not exactly8')
        p=[D(value) for leg in spec['shifts'] for value in leg]
        s=[D(value) for line in spec['invariants'] for value in line]
        m=[D.from_float(value) if isinstance(value,float) else D(value) for mass in spec['masses_squared'] for value in mass]
        if row['backend']==9:
            require(frame['present']==1,'TIRust returned before native interface')
            require([frame[k] for k in ('n','rank','mode','finite','check','cache')]==[n,6,row['mode'],row['finite'],0,0],'Wrong native options')
            require(frame['kind']==(1 if row['ct']==1 else 2),'Wrong native precision')
            require(exact(a['native_values'](frame,'p'),p) and exact(a['native_values'](frame,'s'),s) and exact(a['native_values'](frame,'m'),m)
                    and exact(a['native_values'](frame,'mu'),[D(64)]),'TIRust common effective inputs differ')
            if frame['kind']==1:
                for key,values in (('p',p),('s',s),('m',m),('mu',[D(64)])):
                    require(bytes.fromhex(frame[key])==b''.join(struct.pack('<d',float(v)) for v in values),'DP common input bytes differ')
            if frame['status'] or not row['stable']:issue('refused',row['action'],bytes.fromhex(frame['error']).decode('utf8'))
            if row['stable']:require(frame['status']==0,'Host accepted failed native tensor')
            bad=(frame['flags'][1]&13)&~(frame['flags'][0]&13)
            if bad:issue('new_native_ieee',row['action'],{'before':frame['flags'][0],'after':frame['flags'][1],'new_bad_mask':bad})
        else:
            decode=lambda key:[D.from_float(x[0]) for x in struct.iter_unpack('<d',bytes.fromhex(frame[key]))]
            cp,cs,cm=decode('cll_p'),decode('cll_s'),decode('cll_m');pairs=[]
            for length in range(1,n):
                for start in range(n):
                    if len(pairs)==n*(n-1)//2:break
                    pairs.append((start,(start+length)%n))
            require(exact(cp[::2],p[4:]) and exact(cp[1::2],[D(0)]*len(cp[1::2])),'COLLIER common shifts differ')
            require(exact(cs[::2],[D(spec['invariants'][i][j]) for i,j in pairs]) and exact(cs[1::2],[D(0)]*len(cs[1::2])),'COLLIER common invariants differ')
            require(exact(cm,m),'COLLIER common masses differ')
            if not row['stable']:issue('refused',row['action'],'COLLIER accuracy/error flag')
        if row['state'][0]&13 or row['state'][2]&13:issue('host_ieee',row['action'],row['state'])
        require(row['state'][1]==row['state'][3],'Changed rounding mode')
        if row['stable']:
            for coefficient in row['values']:
                for slot,component in enumerate(coefficient):
                    x,y=component['value']
                    if slot==0 and not row['finite']:require(x.is_nan() and y==0,'Partial finite marker missing')
                    else:require(x.is_finite() and y.is_finite(),'Nonfinite successful tensor')
    bykey={(r['fixture'],r['backend'],r['ct'],r['mode'],r['finite']):r for r in cases}
    def compare(left,right,slots,label,threshold,independent):
        if not left['stable'] or not right['stable']:
            checks.append({'kind':label,'actions':[left['action'],right['action']],'status':'refused'});return
        passed=True;target=True;worst=D(0)
        for lv,rv in zip(left['values'],right['values']):
            for slot in slots:
                for part,(x,y) in enumerate(zip(lv[slot]['value'],rv[slot]['value'])):
                    if independent:
                        symmetric=abs(x-y)/abs(x+y) if x+y else D(0) if x==y else D('Infinity')
                        finite=max(D('1e-99'),*[abs(c[0]['value'][part]) for c in (lv,rv) if c[0]['value'][part].is_finite()])
                        accepted=lambda bound:symmetric<bound or (slot>0 and max(abs(x),abs(y))/finite<bound)
                        passed=passed and accepted(threshold);target=target and accepted(D('1e-8'));worst=max(worst,symmetric)
                    else:
                        scaled=abs(x-y)/max(D(1),abs(x),abs(y));passed=passed and scaled<=threshold;worst=max(worst,scaled)
        checks.append({'kind':label,'actions':[left['action'],right['action']],'status':'passed' if passed else 'mismatched',
                       'maximum_residual':str(worst),'target_1e8_met':target if independent else None})
        if not passed:issue('mismatched',left['action'],label)
    for fixture in fixtures:
        for ct in (1,4):
            for mode in range(3):
                full=bykey[fixture,9,ct,mode,1];partial=bykey[fixture,9,ct,mode,0];collier=bykey[fixture,7,1,mode,1]
                compare(full,partial,(1,2),'full_partial',D('1e-10') if ct==1 else D('1e-24'),False)
                compare(full,collier,(0,1,2),'independent_collier',D('3e-6'),True)
                compare(partial,collier,(1,2),'partial_independent_collier',D('3e-6'),True)
            for finite in (0,1):
                both,uv,ir=(bykey[fixture,9,ct,mode,finite] for mode in range(3))
                if all(r['stable'] for r in (both,uv,ir)):
                    for b,u,v in zip(both['values'],uv['values'],ir['values']):
                        require(all(x==0 for x in u[2]['value']),'UV double pole nonzero')
                        for slot in (1,2):
                            for x,y,z in zip(b[slot]['value'],u[slot]['value'],v[slot]['value']):
                                if abs(x-y-z)> (D('1e-10') if ct==1 else D('1e-24'))*max(D(1),abs(x),abs(y+z)):
                                    issue('pole_reconstruction',both['action'],'UV+IR differs');break
                    if finite:
                        compare(both,uv,(0,),'selector_finite_exact',D(0),False)
                        compare(both,ir,(0,),'selector_finite_exact',D(0),False)
        witness=bykey[fixture,9,4,0 if fixture==108 else 1,1 if fixture==108 else 0]
        slot=0 if fixture==108 else 1
        nonzero=[i for i,c in enumerate(witness['values']) if witness['stable'] and any(abs(v)>0 for v in c[slot]['value'])]
        witnesses[str(fixture)]={'slot':'finite' if slot==0 else 'UV simple pole','nonzero_coefficient_indices':nonzero,'count':len(nonzero)}
        if len(nonzero)!=210:issue('mapping_witness_incomplete',witness['action'],{'nonzero':len(nonzero),'required':210})
    return {'schema':1,'status':'passed' if not open_items else 'open_acceptance_items','cases':30,'effective_inputs_exact':True,
            'initial_ieee':initial,'model_ieee':model,'checks':checks,'open_items':open_items,'mapping_witnesses':witnesses,
            'scope':'Independent exact common-input tensors only; no physical kinematic, amplitude or performance conclusion.'}

def analyze(directory):
    with localcontext() as context:
        context.prec=100
        return _analyze(directory)

if __name__=='__main__':
    import sys
    with localcontext() as context:
        context.prec=100
        print(json.dumps(analyze(Path(sys.argv[1])),indent=2,sort_keys=True))
