"""Pure offline parsing. Status120 remains unchecked; no reducer is loaded."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import re
import struct

BACKENDS=(1,6,8,9)
POINTS=(1,93,39)
MODEL=254*8
TAIL=MODEL+11*8
SIZES={(1,0):31,(1,1):42,(9,0):6,(10,0):22,(10,1):24,
       (20,0):25,(20,1):25,(21,0):45,(21,1):45,(30,0):42,(30,1):48,
       (31,0):32480,(32,0):0,(32,1):0,(34,0):51,(34,1):83,
       (36,0):51,(36,1):51,(40,0):31,(40,1):843,(110,0):280,
       (130,0):16,(130,1):20}
LEAN={9,130,140}

def require(ok,message):
    if not ok: raise ValueError(message)

def digest(data): return hashlib.sha256(data).hexdigest()
def ints(data):
    require(len(data)%8==0,'integer word alignment')
    return struct.unpack('<'+'q'*(len(data)//8),data)
def floats(data):
    require(len(data)%8==0,'DP word alignment')
    out=struct.unpack('<'+'d'*(len(data)//8),data)
    require(all(math.isfinite(x) for x in out),'nonfinite returned DP data')
    return out

def binary(data,width):
    """Exact finite IEEE binary64/binary128 rationals; no float128 narrowing."""
    size=width//8; fracbits,bias,ebits=(52,1023,11) if width==64 else (112,16383,15)
    require(width in (64,128) and len(data)%size==0,'binary precision/length')
    result=[]
    for start in range(0,len(data),size):
        value=int.from_bytes(data[start:start+size],'little')
        fraction=value&((1<<fracbits)-1); exponent=(value>>fracbits)&((1<<ebits)-1)
        require(exponent!=(1<<ebits)-1,'nonfinite native input')
        sign=-1 if value>>(width-1) else 1
        power=(exponent-bias-fracbits) if exponent else 1-bias-fracbits
        mantissa=fraction+(1<<fracbits if exponent else 0)
        result.append(sign*Fraction(mantissa)*(Fraction(2)**power) if mantissa else Fraction(0))
    return result

def rounded113(value):
    """MPFR's nearest-even 113-bit parse, for normal numbers in this fixture."""
    if not value: return value
    sign=-1 if value<0 else 1; value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**exponent: exponent-=1
    require(-16382<=exponent<=16383,'ASCII input outside qualified normal range')
    unit=Fraction(2)**(exponent-112)
    scaled=value/unit; q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2): q+=1
    return sign*q*unit

def ascii113(data):
    require(len(data)%64==0,'ASCII64 record length')
    out=[]
    for offset in range(0,len(data),64):
        raw=data[offset:offset+64].decode('ascii').strip(' \x00')
        require(bool(re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)[EeDd][+-]?\d+',raw)),'malformed decimal transport')
        value=Decimal(raw.replace('D','E').replace('d','e'))
        require(value.is_finite(),'nonfinite decimal transport')
        out.append(rounded113(Fraction(value)))
    return out

def parse_trace(text):
    lines=text.splitlines(); require(len(lines)>=3 and lines[-1]=='COMPLETE','truncated trace')
    header=lines[0].split(); require(len(header)==5 and header[0]=='QG1','trace header')
    observed,count,nbytes,health=map(int,header[1:])
    require(observed in (0,1) and health==0,'observer overflow/error')
    rows=[]
    for number,line in enumerate(lines[1:-1],1):
        columns=line.split(); require(len(columns)==11 and columns[0]=='E','trace row shape')
        seq,call,kind,part,group,so,rank,mode,size=map(int,columns[1:10])
        data=bytes.fromhex(columns[10]); require(seq==number and len(data)==size,'trace sequence/size')
        require(call in [-2,*range(9),101,102,103],'unexpected action')
        require(kind in {1,9,10,20,21,30,31,32,34,36,40,101,106,108,109,110,130,140},'unknown record kind')
        if kind<100 or kind in (110,130):
            require((kind,part) in SIZES,'unknown fixed record')
            require(size==8*SIZES[kind,part]+(0 if kind in LEAN else TAIL),'fixed record length')
        if kind<100: require((group,so,rank,mode)==(0,0,0,0),'unexpected host metadata')
        else: require(1<=group<=116 and so==1 and 0<=rank<=4 and mode in (4,5),'native group metadata')
        require(observed or kind in (1,9),'plain trace contains internal observations')
        rows.append(dict(call=call,kind=kind,part=part,group=group,so=so,rank=rank,mode=mode,data=data))
    require(len(rows)==count and sum(len(r['data']) for r in rows)==nbytes,'trace totals')
    return observed,rows

def payload(row): return row['data'] if row['kind'] in LEAN else row['data'][:-TAIL]
def model(row): return row['data'][-TAIL:-88]
def ieee(row):
    value=ints(row['data'][-88:]); require(all(x in (0,1) for x in value[:10]) and value[10] in range(4),'IEEE encoding')
    return value

def one(rows,kind,part):
    found=[r for r in rows if (r['kind'],r['part'])==(kind,part)]
    require(len(found)==1,f'expected one {kind}/{part}, got {len(found)}')
    return found[0]

def paired(rows,kind):
    found=[r for r in rows if r['kind']==kind]
    require(len(found)%2==0 and [r['part'] for r in found]==[0,1]*(len(found)//2),f'unpaired {kind}')
    return list(zip(found[::2],found[1::2]))

def native_calls(rows,backend,mode):
    tag=100+backend; started=None; output=[]; callback_pairs=paired(rows,130)
    for before,after in callback_pairs:
        require(payload(before)==payload(after)[:128],'CutTools changed callback Q')
        binary(payload(before),128); binary(payload(after)[128:],128)
    require(bool(callback_pairs)==(backend==1),'actual QP callback coverage')
    for row in rows:
        if row['kind'] not in (101,106,108,109): continue
        require(row['kind']==tag,'unexpected actual reducer')
        if row['part']==0:
            if started is not None: output.append(started)
            started={}
        require(started is not None and row['part'] not in started,'native part order/duplicate')
        started[row['part']]=row
    if started is not None: output.append(started)
    require(output,'no native reducer observations')
    summaries=[]
    for call in output:
        required={0,1,3,4,5,6,9}|({2} if backend in (1,6,8) else set())|({7} if backend in (8,9) else set())
        require(required<=set(call),'missing native inputs/status')
        n=struct.unpack('<i',call[0]['data'])[0]; rank=struct.unpack('<i',call[1]['data'])[0]
        require(1<=n<=5 and rank==call[0]['rank'],'native N/rank')
        require(len({(r['group'],r['so'],r['rank'],r['mode']) for r in call.values()})==1,'native identity drift')
        status=struct.unpack('<i',call[9]['data'])[0]; success=(status==1 if backend==1 else status==0)
        expected=required|({10,11,12} if backend==1 and success else {10,11} if backend==6 and success else {10} if success else set())
        require(set(call)==expected and list(call)==sorted(expected),'unexpected/missing/out-of-order native output')
        decode=(lambda b:binary(b,64)) if backend==1 else (lambda b:binary(b,128)) if backend==6 else ascii113
        fields={}
        for part,name,length in [(3,'shifts',4*n),(4,'squared_masses',2*n),(5,'scale',1)]:
            fields[name]=decode(call[part]['data']); require(len(fields[name])==length,'native array length')
        if backend==1:
            require(len(binary(call[6]['data'],64))==1,'CutTools root scale length')
        if backend!=1:
            fields['invariants']=decode(call[6]['data']); require(len(fields['invariants'])==n*n,'native invariant length')
        if backend in (6,8):
            co=decode(call[2]['data']); require(len(co)==2*math.comb(rank+4,4),'native coefficient length')
        if backend==1: require(struct.unpack('<i',call[2]['data'])[0]==mode,'CutTools requested mode')
        if backend==8: require(call[7]['data']==struct.pack('<i',0),'NinjaRS pole mode')
        if backend==9:
            flags=struct.unpack('<4i',call[7]['data']); require(flags[:3]==(0,1,int(mode==5)) and flags[3] in (0,1),'TIRust finite/check policy')
        if success:
            values=decode(call[10]['data'])
            require(len(values)==(6 if backend in (1,6) else 8 if backend==8 else 8*math.comb(rank+4,4)),'native result length')
            for part in (11,12):
                if part in call: require(len(decode(call[part]['data']))==2,'native remainder length')
        summaries.append({'group':call[0]['group'],'n':n,'rank':rank,'status':status,'success':success,
            'input_fraction_sha256':digest(json.dumps({k:[str(x) for x in v] for k,v in fields.items()},sort_keys=True).encode()),
            'scale_exact':str(fields['scale'][0]),
            'shifts_exact':[str(v) for v in fields['shifts']],
            'squared_masses_exact':[str(v) for v in fields['squared_masses']],
            'input_encoding':'DP binary64' if backend==1 else 'QP binary128' if backend==6 else 'ASCII parsed at 113 bits'})
    return summaries,len(callback_pairs)

def routing(rows,backend,mode,native):
    pairs=paired(rows,140)
    require(bool(pairs)==(backend!=1),'mode routing observation coverage')
    require(len(pairs)==len(native) if backend!=1 else not pairs,'routing/native count')
    for (before,after),actual in zip(pairs,native):
        require(before['group']==after['group']==actual['group'],'routing/native group')
        a,b=payload(before),payload(after); m,n=ints(a[:16])
        require(m==mode and 1<=n<=5 and len(a)==len(b)==16+96*n,'routing shape')
        require(a[:16]==b[:16],'routing metadata changed')
        pa=binary(a[16:16+64*n],128); pb=binary(b[16:16+64*n],128)
        ma=binary(a[16+64*n:],128); mb=binary(b[16+64*n:],128)
        expectedp=[-v for i in reversed(range(n)) for v in pa[4*i:4*i+4]] if mode==5 and n>2 else pa
        expectedm=ma[:2]+[v for i in reversed(range(1,n)) for v in ma[2*i:2*i+2]] if mode==5 and n>2 else ma
        require(pb==expectedp and mb==expectedm,'routing sign/permutation differs from frozen source')
        require([str(v) for v in mb]==actual['squared_masses_exact'],'native squared masses differ from routed actual masses')
        shifts=[]
        for j in range(n):
            count=n if j==0 else j
            for component in range(4):
                total=Fraction(0)
                for i in range(count): total=rounded113(total+pb[4*i+component])
                shifts.append(str(total))
        require(shifts==actual['shifts_exact'],'native shifts differ from observed routed QP prefixes')
    return len(pairs)

def observed_inputs(rows,backend,mode):
    for row in rows:
        if row['kind'] not in LEAN and row['kind']<100 or row['kind']==110: ieee(row)
    born=paired(rows,10); dp=paired(rows,20); qp=paired(rows,21)
    require(len(born)==len(dp)==len(qp)==1,'Born/DP/QP repair coverage')
    b0,b1=born[0]; d0,d1=dp[0]; q0,q1=qp[0]
    require(payload(b0)[:160]==payload(d1)[:160],'Born is not using observed repaired DP inputs')
    require(payload(b0)==payload(b1)[:176],'Born input mutation')
    c0,c1=one(rows,30,0),one(rows,30,1)
    require(payload(c0)[:320]==payload(c1)[:320]==payload(q1)[:320],'canonical repaired QP point differs')
    require(ints(payload(c0)[320:])[0]==mode,'actual MP construction mode')
    update=paired(rows,32); require(len(update)==1,'MP coupling update coverage')
    amp,coef=paired(rows,34),paired(rows,36)
    require(0<len(amp)==len(coef)<=48,'actual helicity coverage')
    require([payload(a) for a,b in amp]==[payload(a) for a,b in coef],'helicity/color construction paths differ')
    require(len({ints(payload(a)[320:])[5] for a,b in amp})==len(amp),'duplicate helicity')
    for a,b in amp+coef:
        require(payload(a)==payload(b)[:408] and payload(a)[:320]==payload(q1)[:320],'QP construction input mutation')
        require(model(a)==model(b)==model(update[0][1]),'model differs after coupling update')
    qco=payload(one(rows,31,0)); reductions=paired(rows,40)
    require(len(reductions)==1,'host reduction/contracted-result boundary coverage')
    r0,r1=reductions[0]
    require(payload(r0)==payload(r1)[:248],'host reduction call input mutation')
    stable=ints(payload(r1)[248+696*8:]); require(len(stable)==116 and all(x==1 for x in stable),'unstable/omitted host group')
    floats(payload(r1)[248:248+696*8])
    states=[r for r in rows if r['kind']==110]
    require(states and all(r['part']==0 for r in states),'native canonical state missing')
    for row in states:
        start=(row['group']-1)*2240
        require(payload(row)==qco[start:start+2240],'native numerator differs from canonical QP construction')
        require(model(row)==model(update[0][1]),'native model differs from QP construction')
    native,callbacks=native_calls(rows,backend,mode)
    require(len(states)==len(native),'native state/evaluation count')
    require(all(r['success'] for r in native),'native reducer refusal/failure')
    observed_groups={r['group'] for r in native}
    nonzero_groups={g+1 for g in range(116) if any(binary(qco[g*2240:(g+1)*2240],128))}
    require(nonzero_groups<=observed_groups,'missing native evaluation for nonzero QP coefficient groups: '+str(sorted(nonzero_groups-observed_groups)))
    order_count=routing(rows,backend,mode,native)
    # Include real Born and every actual common physical/model/construction input.
    # Exclude backend identifiers, reduction results and inherited IEEE state from
    # the common-input digest. Those are retained independently, not repaired.
    pieces=[payload(b0),payload(b1)[-16:],payload(d0),payload(d1),payload(q0),payload(q1),
            model(d0),model(d1),model(q0),model(q1),model(update[0][0]),model(update[0][1]),qco]
    pieces += [payload(a)+payload(b)[408:] for a,b in amp]
    return {'shared_input_sha256':digest(b''.join(pieces)),'native':native,
            'native_evaluations':len(native),'qp_numerator_callbacks':callbacks,
            'routing_calls':order_count,'helicities':len(amp),
            'observed_groups':sorted({r['group'] for r in native}),
            'host_group_array_extent':116,'nonzero_coefficient_groups':sorted(nonzero_groups),
            'zero_coefficient_groups_without_native_call':sorted(set(range(1,117))-observed_groups),
            'padded_qp_coefficient_slots':70,
            'native_rank_valid_coefficient_count':'binomial(rank+4,4)',
            'native_basis_identity_claimed':False,
            'unexposed':'Host-local QP counterterm/contraction temporaries before their DP return conversion.'}

def parse_worker(trace,stdout,backend,mode,observation,point_bytes):
    actual,rows=parse_trace(trace); require(actual==observation,'observation mode mismatch')
    complete=one(rows,9,0); completion=ints(payload(complete))
    require(completion[:3]==(backend,mode,observation) and completion[4:]==(113,128),'completion identity/precision')
    warm=completion[3]; require(1<=warm<=8,'warmup budget')
    calls=[*range(warm+1),101,102,103]
    require([r['call'] for r in rows if r['kind']==1 and r['part']==0]==calls,'public call sequence')
    require([r['call'] for r in rows if r['kind']==1 and r['part']==1]==calls,'public return sequence')
    require(rows[-1] is complete and complete['call']==-2,'completion placement')
    require(len(re.findall(r'\bQG_DONE\s+'+str(backend)+r'\s+'+str(mode)+r'\s+'+str(observation)+r'\s+'+str(warm)+r'\b',stdout))==1,'missing/duplicate successful stdout completion')
    warnings=[{'line':i,'text':line} for i,line in enumerate(stdout.splitlines(),1) if re.search(r'##W03\b',line,re.I)]
    require(not warnings,'W03 omitted-contribution warning: '+json.dumps(warnings))
    require(not re.search(r'QG_FAILED|(?:^|\n)\s*(?:ERROR )?STOP\b',stdout,re.I),'failed/early Fortran STOP')
    points=[]; public=[]
    for call in calls:
        group=[r for r in rows if r['call']==call]
        before,after=one(group,1,0),one(group,1,1)
        require(group[0] is before and group[-1] is after,'public envelope order')
        a,b=payload(before),payload(after); policy=ints(a[160:248]); policy_after=ints(b[160:248])
        point=POINTS[call-101] if call>=101 else 1
        require(policy[:6]==(backend,mode,point,call,113,128)==policy_after[:6],'policy identity')
        require(a[:160]==point_bytes[point],'physical point changed before public call')
        expected_policy=(mode,1,1,1,1) if call>=101 else (-1,1,1,1,1)
        require(policy[6:]==((-1,)*5 if call==0 else expected_policy) and policy_after[6:]==expected_policy,'actual policy mismatch')
        ieee(before);ieee(after)
        public.extend([before['data'].hex(),after['data'].hex()])
        values=floats(b[248:312]); accuracies=floats(b[312:328]); status=ints(b[328:336])[0]
        if call>=101:
            require(status==120 and accuracies==(-1.,-1.),'fixed diagnostic status/accuracy must remain unchecked120')
            details=observed_inputs(group,backend,mode) if observation else None
            points.append({'point':point,'status':status,'accuracy':list(accuracies),'host_checked':False,
                           'coefficients':list(values),'inputs':details,'ieee_before':ieee(before),'ieee_after':ieee(after)})
        else:
            require(all(r['kind']==1 for r in group),'unexpected bootstrap observations')
            if call:
                h,t,u=status//100,status%100//10,status%10
                aliases={1:9,6:2,8:8,9:3}
                require(t in (1,2,3) and ((h==2 and u==backend) or (h==3 and u==aliases[backend]) or (h==1 and t==3 and u==backend)),'unrecognized warmup status')
                require(call!=warm or t in (1,2),'initialization never reached checked steady state')
    return {'backend':backend,'mode':mode,'observed':observation,'warmup_calls':warm,
            'public_calls':len(calls),'diagnostic_calls':3,'host_checked':False,
            'warnings':warnings,'points':points,'public_raw':public,
            'trace_sha256':digest(trace.encode()),'stdout_sha256':digest(stdout.encode())}

def delta(a,b):
    difference=abs(a-b)
    return {'absolute':difference,
            'relative_to_left':difference/abs(a) if a else (0.0 if not difference else None),
            'symmetric':difference/abs(a+b) if a+b else (0.0 if not difference else None),
            'opposite_nonzero_sum':a+b==0 and difference!=0} 

def adjudicate(workers):
    require(len(workers)==16,'exact sixteen worker matrix required')
    keyed={(w['backend'],w['mode'],w['observed']):w for w in workers}
    require(set(keyed)=={(b,m,o) for b in BACKENDS for m in (4,5) for o in (0,1)},'duplicate/missing worker')
    for b in BACKENDS:
        for m in (4,5):
            a,z=keyed[b,m,0],keyed[b,m,1]
            require(a['public_raw']==z['public_raw'],'plain/observed input/output/status/IEEE mismatch')
    comparisons=[]
    arms=[(b,m) for b in BACKENDS for m in (4,5)]
    for left,right in combinations(arms,2):
        if left[0]!=right[0] and left[1]!=right[1]: continue
        for a,b in zip(keyed[*left,1]['points'],keyed[*right,1]['points']):
            equal=a['inputs']['shared_input_sha256']==b['inputs']['shared_input_sha256']
            comparisons.append({'left':list(left),'right':list(right),'point':a['point'],
                'shared_physical_born_repair_numerator_equal':equal,
                'native_basis_identity_claimed':False,'host_checked':False,
                'comparison_scope':'Same observed upstream physical/model/repair/Born/QP numerator inputs; backend-native bases and scalar construction are retained separately.' if equal else 'Inadmissible: shared effective inputs differ.',
                'coefficients':[delta(x,y) for x,y in zip(a['coefficients'],b['coefficients'])] if equal else None})
    return {'schema':'qcd-n3-genuine-qp-diagnostic-v1','workers':[{k:v for k,v in w.items() if k!='public_raw'} for w in workers],
            'diagnostic_physical_calls':48,'answer_order_rows':96,
            'all_public_calls':sum(w['public_calls'] for w in workers),
            'bootstrap_and_initialization_calls':sum(w['public_calls']-3 for w in workers),
            'public_call_budget':192,'host_checked':False,
            'timing_or_campaign_acceptance':False,'comparisons':comparisons,
            'all_shared_input_gates_pass':all(c['shared_physical_born_repair_numerator_equal'] for c in comparisons)}
