"""Twelve exact-input generated tensor diagnostics; no native/process calls."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import json,math,struct
from pathlib import Path
import legacy_parser as lp
from oracle import C,ZERO,triangle_poles,scaleless_bubble_poles,monomials,ieee_fraction
HERE=Path(__file__).resolve().parent

def require(x,message):
    if not x:raise ValueError(message)
def fraction(x):
    require(x.is_finite(),'Nonfinite evaluated coefficient')
    return F(x)
def errors(path,schedule):
    result={}
    lines=path.read_text().splitlines()
    require(len(lines)==len(schedule),'Incomplete host-error coverage')
    for line,spec in zip(lines,schedule):
        x=line.split();require(len(x)==515 and x[0]=='ERROR' and int(x[1])==spec['action'],'Host-error framing/order')
        length=int(x[2]);require(0<=length<=512,'Host error length')
        require(all(len(s)==2 and all(c in '0123456789ABCDEFabcdef' for c in s) for s in x[3:]),'Bad host-error hex')
        raw=bytes(int(s,16) for s in x[3:]);require(raw[length:]==b' '*(512-length),'Host-error padding changed')
        result[spec['action']]=raw[:length].decode('utf8',errors='replace')
    return result

def oracle_check(row):
    issues=[];worst=F(0);worst_sym=F(0);zero_max=F(0)
    tol=F(1,10**10) if row['ct']==1 else F(1,10**24)
    for i,alpha in enumerate(monomials(6)):
        uv,ir1,ir2=(triangle_poles(alpha) if row['fixture']==110 else scaleless_bubble_poles(alpha))
        want=(uv+ir1,ir2) if row['mode']==0 else (uv,ZERO) if row['mode']==1 else (ir1,ir2)
        unit=F(2)**(sum(alpha)+4-2*row['n'])
        for slot,expected in enumerate(want,1):
            value=row['values'][i][slot]['value']
            for part,(actual,truth) in enumerate(zip(value,(expected.re,expected.im))):
                got=ieee_fraction(row['values'][i][slot]['words'][part*2:part*2+2]);scaled=abs(got-truth)/max(unit,abs(got),abs(truth));worst=max(worst,scaled)
                accepted=scaled<=tol
                if truth:
                    denominator=abs(got+truth)
                    symmetric=abs(got-truth)/denominator if denominator else None
                    accepted=accepted and symmetric is not None and symmetric<F(1,10**8)
                    if symmetric is not None:worst_sym=max(worst_sym,symmetric)
                else:zero_max=max(zero_max,abs(got)/unit)
                if not accepted:issues.append({'index':i,'powers':alpha,'slot':slot,'part':part,'actual':str(got),'expected':str(truth),'scaled_error':str(scaled)})
    return {'passed':not issues,'failures':issues,'maximum_scaled_error':str(worst),'maximum_nonzero_symmetric_residual':str(worst_sym),'maximum_expected_zero_scaled_leakage':str(zero_max),'tolerance':str(tol)}

def analyze_rows(cases,native,host_errors,initial,model):
    issues=[];checked=[];by={}
    if initial[0]&13:issues.append({'kind':'initial_IEEE','state':initial})
    if model[0]&13:issues.append({'kind':'model_IEEE','state':model})
    require(initial[1]==model[1]==0,'Unsupported initial rounding mode')
    fixtures={110:json.loads((HERE/'future-triangle-fixture.json').read_text()),111:json.loads((HERE/'future-scaleless-fixture.json').read_text())}
    def geometry(fixture):
        p=[D(x) for leg in fixture['shifts'] for x in leg]
        s=[D(x) for leg in fixture['invariants'] for x in leg]
        m=[D(0)]*(2*fixture['n'])
        expected_entry={'p':[[w for value in leg for w in lp.promoted(value)] for leg in fixture['external_pl']],
                    'm':[[w for value in mass for w in lp.promoted(value)] for mass in fixture['masses_squared']]}
        return p,s,m,expected_entry
    for row in cases:
        a=row['action'];frame=native[a]
        fixture=fixtures[row['fixture']];p,s,m,expected_entry=geometry(fixture)
        by[row['fixture'],row['ct'],row['mode'],row['finite']]=row
        require(row['entry']==expected_entry and row['exit']==expected_entry,'Generated adapter changed tensor inputs')
        require(row['scale']=='4000000000000000','MU_R is not exact2')
        if row['state'][0]&13 or row['state'][2]&13:issues.append({'kind':'host_IEEE','action':a,'state':row['state']})
        require(row['state'][1]==row['state'][3]==0,'Changed/non-nearest host rounding')
        if not frame['present']:
            require(not row['stable'],'Host accepted absent native call')
            issues.append({'kind':'pre_ABI_refusal','action':a,'host_error':host_errors[a]})
            checked.append({'action':a,'native_present':False,'oracle':None});continue
        require([frame[k] for k in ('kind','n','rank','mode','finite','check','cache')]==[1 if row['ct']==1 else 2,row['n'],6,row['mode'],row['finite'],0,0],'Native options/precision/cache differ')
        for key,values in [('p',p),('s',s),('m',m),('mu',[D(4)])]:
            got=lp.native_values(frame,key)
            require(len(got)==len(values) and all(x==y for x,y in zip(got,values)),'Actual native '+key+' differs from oracle inputs')
            if frame['kind']==1:require(bytes.fromhex(frame[key])==b''.join(struct.pack('<d',float(v)) for v in values),'DP input bytes differ')
        if any(f&13 for f in frame['flags']):issues.append({'kind':'native_IEEE','action':a,'flags':frame['flags']})
        require(frame['rounding']==[0,0,0],'Changed native rounding')
        if frame['status'] or not row['stable']:
            require(not row['stable'],'Host accepted native failure')
            issues.append({'kind':'native_refusal','action':a,'native_status':frame['status'],'native_error':bytes.fromhex(frame['error']).decode('utf8',errors='replace'),'host_error':host_errors[a]})
            checked.append({'action':a,'native_present':True,'oracle':None});continue
        for coefficient in row['values']:
            for slot,value in enumerate(coefficient):
                real,imag=value['value']
                if slot==0 and not row['finite']:require(real.is_nan() and imag==0,'Missing partial finite NaN marker')
                else:require(real.is_finite() and imag.is_finite(),'Nonfinite successful output')
        if row['fixture']==111 and row['finite']:
            require(all(ieee_fraction(v['words'][:2])==0 and ieee_fraction(v['words'][2:])==0 for v in (c[0] for c in row['values'])),'Scaleless finite output nonzero')
        verdict=oracle_check(row)
        if not verdict['passed']:issues.append({'kind':'independent_pole_oracle','action':a,'failures':verdict['failures']})
        checked.append({'action':a,'native_present':True,'oracle':verdict})
    pairs=[]
    for fixture_id in [110,111]:
        for ct in [1,4]:
            for mode in range(3):
                full=by[fixture_id,ct,mode,1];partial=by[fixture_id,ct,mode,0]
                if not full['stable'] or not partial['stable']:
                    pairs.append({'fixture':fixture_id,'ctmode':ct,'pole_mode':mode,'status':'refused'});continue
                require(all(native[full['action']][k]==native[partial['action']][k] for k in ('p','s','m','mu')),'Full/partial ABI inputs not bit-identical')
                worst=F(0);words=True
                for i,alpha in enumerate(monomials(6)):
                    unit=F(2)**(sum(alpha)+4-2*full['n'])
                    for slot in [1,2]:
                        words=words and full['values'][i][slot]['words']==partial['values'][i][slot]['words']
                        for part in range(2):
                            u=ieee_fraction(full['values'][i][slot]['words'][part*2:part*2+2])
                            v=ieee_fraction(partial['values'][i][slot]['words'][part*2:part*2+2]);worst=max(worst,abs(u-v)/max(unit,abs(u),abs(v)))
                tol=F(1,10**10) if ct==1 else F(1,10**24)
                if worst>tol:issues.append({'kind':'full_partial_mismatch','ctmode':ct,'pole_mode':mode,'scaled_error':str(worst)})
                pairs.append({'fixture':fixture_id,'ctmode':ct,'pole_mode':mode,'status':'passed' if worst<=tol else 'mismatched','maximum_scaled_error':str(worst),'all_pole_words_equal':words})
            if all(by[fixture_id,ct,mode,1]['stable'] for mode in range(3)):
                first=by[fixture_id,ct,0,1]
                for mode in [1,2]:require(all(first['values'][i][0]['words']==by[fixture_id,ct,mode,1]['values'][i][0]['words'] for i in range(210)),'Finite output depends on pole selector')
    return {'schema':'rank6-IR-triangle-tensor-qualification-v1','status':'passed' if not issues else 'open_acceptance_items','cases':24,'components':210,'source_fixtures':[110,111],
      'initial_IEEE':initial,'model_IEEE':model,'checks':checked,'full_partial':pairs,'open_items':issues,
      'all210_exact_IR2_witnesses_nonzero':True,'all210_scaleless_UV1_IR1_witnesses_nonzero':True,'actual_ABI_inputs_match':all(c['native_present'] for c in checked),'cache_enabled':False,
      'physical_amplitude_qualified':False,'physical_stress_qualified':False,'performance_acceptance':False,
      'scope':'Synthetic generated rank6 diagnostic adapter; historical exporter identity, genuine QP API, no physical numerator or amplitude.'}

def analyze(directory):
    directory=Path(directory);schedule=json.loads((HERE/'schedule.json').read_text());lines=(directory/'records.txt').read_text().splitlines()
    require(len(lines)>=3 and lines[1].split()[0]=='INITIAL_IEEE' and lines[2].split()[0]=='MODEL_IEEE','Fresh IEEE observations missing')
    initial=list(map(int,lines[1].split()[1:]));model=list(map(int,lines[2].split()[1:]));require(len(initial)==len(model)==2,'IEEE schema')
    class Records:
        def read_text(self):return '\n'.join(lines[:1]+lines[3:])+'\n'
        def __str__(self):return str(directory/'records.txt')
    with localcontext() as c:
        c.prec=100
        captures,cases=lp.parse_records(Records(),'ew',schedule)
        require(not captures and len(cases)==24,'Unexpected capture or missing row')
        frames=lp.parse_observer(directory/'observer.txt',[r['action'] for r in schedule])
        return analyze_rows(cases,frames,errors(directory/'errors.txt',schedule),initial,model)
if __name__=='__main__':
    import sys
    print(json.dumps(analyze(Path(sys.argv[1])),indent=2,sort_keys=True))
