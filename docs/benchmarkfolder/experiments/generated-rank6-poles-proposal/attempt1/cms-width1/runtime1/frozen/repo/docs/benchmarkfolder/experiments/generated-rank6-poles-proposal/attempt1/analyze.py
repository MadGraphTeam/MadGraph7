#!/usr/bin/env python3
"""Strict retained-text analysis. No compiler, numerical library or subprocess.

Usage: python -B analyze.py EW_WORK_DIRECTORY QCD_WORK_DIRECTORY
Writes JSON to stdout only. Protocol/coverage violations raise ValueError;
numerical refusals/mismatches are retained as open items in the report.
"""
from decimal import Decimal as D, localcontext
import json
import math
from fractions import Fraction
from pathlib import Path
import re
import struct
import sys

HERE=Path(__file__).resolve().parent
HEX=re.compile(r"[0-9A-Fa-f]{16}\Z")


def require(value,message):
    if not value: raise ValueError(message)


def binary128(words):
    require(len(words)==2 and all(HEX.fullmatch(w) for w in words),"Invalid binary128 words")
    bits=int(words[0],16)+(int(words[1],16)<<64)
    sign=-1 if bits>>127 else 1
    exponent=(bits>>112)&32767
    fraction=bits&((1<<112)-1)
    if exponent==32767: return D("NaN") if fraction else D(sign)*D("Infinity")
    with localcontext() as ctx:
        ctx.prec=180
        mantissa=fraction if exponent==0 else fraction+(1<<112)
        return D(sign)*D(mantissa)*(D(2)**((exponent or 1)-16383-112))


def promoted(value):
    bits=struct.unpack("<Q",struct.pack("<d",value))[0]
    sign=bits>>63; exp=(bits>>52)&2047; frac=bits&((1<<52)-1)
    require(exp!=2047,"Nonfinite promoted fixture")
    if exp:
        q=(sign<<127)|((exp-1023+16383)<<112)|(frac<<60)
    elif not frac:
        q=sign<<127
    else:
        top=frac.bit_length()-1
        q=(sign<<127)|((top-1074+16383)<<112)|((frac-(1<<top))<<(112-top))
    return [f"{q&((1<<64)-1):016X}",f"{q>>64:016X}"]


def rounded113(value):
    """Round a decimal ABI token to the specified binary128 input carrier."""
    if not value.is_finite() or not value:return value
    fraction=Fraction(abs(value));a,b=fraction.numerator,fraction.denominator
    exponent=a.bit_length()-b.bit_length()
    if (a < (b<<exponent)) if exponent>=0 else ((a<<-exponent)<b):exponent-=1
    power=max(exponent-112,-16494)
    if power>=0:denominator=b<<power;numerator=a
    else:denominator=b;numerator=a<<-power
    integer,remainder=divmod(numerator,denominator)
    if 2*remainder>denominator or (2*remainder==denominator and integer&1):integer+=1
    with localcontext() as ctx:
        ctx.prec=180
        return D(-1 if value.is_signed() else 1)*D(integer)*(D(2)**power)


def parse_records(path,model,rows):
    lines=iter(enumerate(path.read_text().splitlines(),1))
    def take(tag,count=None):
        try: number,line=next(lines)
        except StopIteration: raise ValueError("Missing "+tag) from None
        tokens=line.split()
        require(tokens and tokens[0]==tag,f"Expected {tag} at {path}:{number}")
        require(count is None or len(tokens)==count+1,"Wrong field count for "+tag)
        return tokens[1:]
    def inputs(ptag,mtag,n):
        result={"p":[],"m":[]}
        for leg in range(1,n+1):
            for tag,key,count in ((ptag,"p",8),(mtag,"m",4)):
                fields=take(tag,count+1)
                require(int(fields[0])==leg and all(HEX.fullmatch(x) for x in fields[1:]),"Input row malformed")
                result[key].append([x.upper() for x in fields[1:]])
        return result
    def values(nc):
        output=[]
        for coefficient in range(nc):
            row=[]
            for slot in range(1,4):
                fields=take("V",8)
                require(list(map(int,fields[:2]))==[coefficient,slot],"Coefficient coverage/order")
                words=fields[2:6]
                exact=[binary128(words[:2]),binary128(words[2:])]
                printed=[D(x) for x in fields[6:]]
                for x,y in zip(exact,printed):
                    require((x.is_nan() and y.is_nan()) or (x.is_finite() and y.is_finite()
                      and abs(x-y)<=D('5e-35')*max(D(1),abs(x))),"Printed value disagrees with binary words")
                row.append({"words":[x.upper() for x in words],"value":exact})
            output.append(row)
        return output
    nc=70 if model=="ew" else 126
    require(list(map(int,take("HEADER",3)))==[1,0 if model=="ew" else 1,nc],"Header identity")
    captures=[]
    for action,k in ([(901,1),(902,2)] if model=="ew" else [(903,1)]):
        fields=list(map(int,take("CAPTURE",4)))
        require(fields[:2]==[action,k] and fields[2] in (0,1) and fields[3]==k,"Capture coverage")
        captures.append({"action":action,"stable":bool(fields[2]),"input":inputs("PI","MI",6),"values":values(nc)})
        require(list(map(int,take("CAPTURE_END",1)))==[action],"Capture footer")
    cases=[]
    for spec in rows:
        expected=[spec[x] for x in ("action","fixture","backend","ct","mode","finite","cache","budget","clear","n","rank")]
        expected.append(math.comb(spec["rank"]+4,4))
        require(list(map(int,take("CASE",12)))==expected,"Schedule case mismatch")
        scale=take("SCALE",1)[0]
        require(HEX.fullmatch(scale),"Scale bits")
        entry=inputs("PI","MI",spec["n"])
        pre=list(map(int,take("CACHE_PRE",15)))
        state=list(map(int,take("STATE",7)))
        require(state[4] in (0,1),"Logical status")
        exit_input=inputs("PO","MO",spec["n"])
        coefficients=values(expected[-1])
        post=list(map(int,take("CACHE_POST",15)))
        require(list(map(int,take("END",2)))==[spec["action"],0],"Case footer/health")
        require(min(pre+post)>=0,"Negative cache counters")
        cases.append({**spec,"scale":scale.upper(),"entry":entry,"exit":exit_input,"values":coefficients,
                      "pre":pre,"post":post,"state":state,"stable":bool(state[4])})
    final=list(map(int,take("FINAL",18)))
    require(final[0]==0 and final[1]>=0 and final[2]==-1,"Final context destruction")
    # Aggregate entries/bytes and QP entries/bytes are indices6/7 and11/12.
    require(all(final[3+i]==0 for i in (6,7,11,12)),"Final caches not empty")
    require(take("DONE",1)==["0"],"Missing healthy final marker")
    require(next(lines,None) is None,"Extra trailing records")
    return captures,cases


def parse_observer(path,expected):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"Duplicate observer JSON key")
            result[key]=value
        return result
    records=[json.loads(line,object_pairs_hook=unique) for line in path.read_text().splitlines()]
    require([r["action"] for r in records]==expected,"Observer complete action coverage/order")
    keys={"schema","action","present","kind","n","rank","mode","finite","check","cache","status",
          "flags","rounding","health","p","s","m","mu","out","error","cll_p","cll_s","cll_m"}
    for row in records:
        require(set(row)==keys and row["schema"]==1 and row["health"]==0,"Observer schema/health")
        require(all(type(row[k]) is int for k in keys-{"flags","rounding","p","s","m","mu","out","error","cll_p","cll_s","cll_m"}),"Observer metadata types")
        for k in ("p","s","m","mu","out","error","cll_p","cll_s","cll_m"):
            require(isinstance(row[k],str) and re.fullmatch(r"(?:[0-9a-f]{2})*",row[k]),"Observer byte encoding")
        for k in ("flags","rounding"):
            require(len(row[k])==3 and all(type(x) is int for x in row[k]),"Observer fenv schema")
        if row["present"]:
            require(row["kind"] in (1,2),"Native precision")
            require(1<=row['n']<=6 and 0<=row['rank']<=6 and row['status'] in (0,1,2,3),"Native dimensions/status")
            require(row["flags"][1]==row["flags"][2] and row["rounding"][1]==row["rounding"][2],"Observer changed fenv")
            n,rank=row["n"],row["rank"];nc=math.comb(rank+4,4)
            sizes=(32*n,8*n*n,16*n,8,64*nc) if row["kind"]==1 else (256*n,64*n*n,128*n,64,512*nc)
            for key,size in zip(("p","s","m","mu","out"),sizes):
                require(len(row[key])==2*(size if key!="out" or row["status"]==0 else 0),"Native buffer dimensions")
        else:
            require(row["kind"]==0 and all(not row[k] for k in ("p","s","m","mu","out","error")),"COLLIER unexpectedly called TIRust")
            require(all(row[k]==0 for k in ('mode','finite','check','cache','status')) and row['flags']==[0,0,0] and row['rounding']==[0,0,0],"Dirty absent-native metadata")
    return {r["action"]:r for r in records}


def native_values(frame,key):
    raw=bytes.fromhex(frame[key])
    if frame["kind"]==1:
        return [D.from_float(x[0]) for x in struct.iter_unpack("<d",raw)]
    return [rounded113(D(raw[i:i+64].decode("ascii").strip())) for i in range(0,len(raw),64)]


def input_contract(cases,captures):
    bycap={r["action"]:r["input"] for r in captures}
    bases={101:bycap[901],103:bycap[901],107:bycap[901],102:bycap[903],104:bycap[903]}
    for row in cases:
        expected=bases.get(row["fixture"])
        if expected is not None: require(row["entry"]==expected,"Fresh capture input mismatch")
        elif row["fixture"]==105:
            require(row["entry"]["p"]==bycap[902]["p"],"Complex fixture momentum altered")
            for j,(a,b) in enumerate(zip(row["entry"]["m"],bycap[902]["m"])):
                require(a==b if j!=2 else a[:2]==b[:2] and binary128(a[2:])<0,"Complex mass mutation differs")
        else:
            p=[[2.,0.,0.,2.],[-2.,-0.,-0.,-2.]]
            require(row["entry"]["p"]==[[w for v in leg for w in promoted(v)] for leg in p],"Scaleless fixture geometry")
            require(all(binary128(m[:2])==0 and binary128(m[2:])==0 for m in row["entry"]["m"]),"Scaleless masses")
        expected_exit={k:[v[:] for v in rows] for k,rows in row["entry"].items()}
        if row["ct"] in (2,5) and row["n"]>2:
            expected_exit["p"]=list(reversed(expected_exit["p"]))
            for leg in expected_exit["p"]:
                for j in (1,3,5,7): leg[j]=f"{int(leg[j],16)^(1<<63):016X}"
            expected_exit["m"]=[expected_exit["m"][0]]+list(reversed(expected_exit["m"][1:]))
        require(row["exit"]==expected_exit,"Unexpected input mutation/double reversal")
    for fixture in range(101,108):
        selected=[r for r in cases if r["fixture"]==fixture]
        require(len({r["scale"] for r in selected})==1,"Scale changed between modes")
    scales={r["fixture"]:struct.unpack('<d',bytes.fromhex(r["scale"])[::-1])[0] for r in cases}
    require(scales[101]==scales[102]==scales[103]==scales[104]==scales[105]==1000.,"Physical scale not frozen 1TeV")
    require(scales[106]==2. and scales[107]==2000.,"Changed scale control")


def residual(a,b):
    return abs(a-b)/max(D(1),abs(a),abs(b))


def analyze(workdirs):
    schedule=json.loads((HERE/"schedule.json").read_text())
    allcases=[];allcaptures=[];native={}
    for model,directory in zip(("ew","qcd"),workdirs):
        rows=[r for r in schedule if r["model"]==model]
        captures,cases=parse_records(directory/"records.txt",model,rows)
        native.update(parse_observer(directory/"observer.txt",[r["action"] for r in captures+cases]))
        allcases+=cases;allcaptures+=captures
    require(len(allcases)==123 and len(allcaptures)==3,"Complete fixed campaign")
    input_contract(allcases,allcaptures)
    monomials=json.loads((HERE/"materialized/ew/monomials.json").read_text())
    open_items=[];checks=[]
    def issue(kind,action,detail):open_items.append({"kind":kind,"action":action,"detail":detail})
    for capture in allcaptures:
        o=native[capture['action']]
        rank=5 if capture['action']==903 else 4
        if o['present']:
            require([o[k] for k in ('kind','n','rank','mode','finite','check','cache')]==[1,6,rank,0,1,0,0],"Capture native call changed")
            if capture['stable']:require(o['status']==0,"Capture accepted native failure")
        else:require(not capture['stable'],"Capture succeeded without a native call")
        if not capture['stable']:issue('capture_refused',capture['action'],bytes.fromhex(o['error']).decode('utf8','replace'))
    base={}
    for row in allcases:
        o=native[row["action"]]
        if row["backend"]==9:
            if not o['present']:
                require(not row['stable'] and not o['cll_p'],"Absent native call was accepted")
                issue('host_refused',row['action'],'Generated adapter returned before the C API')
            else:
                require([o[k] for k in ("n","rank","mode","finite","check","cache")]==
                    [row["n"],row["rank"],row["mode"],row["finite"],int(row["ct"] in (2,5)),int(row["cache"]==-1)],"C API options mismatch")
                require(o["kind"]==(1 if row["ct"]<4 else 2),"Native precision mismatch")
            if o['present'] and (o["status"]!=0 or not row["stable"]):issue("refused",row["action"],bytes.fromhex(o["error"]).decode("utf8","replace"))
            if row["stable"]:require(o["status"]==0,"Host accepted native failure")
        else:
            require(o["present"]==0 and len(o["cll_p"])==128*(row["n"]-1)
              and len(o["cll_s"])==16*row["n"]*(row["n"]-1) and len(o["cll_m"])==32*row["n"],"COLLIER input observation")
            if not row["stable"]:issue("refused",row["action"],"COLLIER error/accuracy flag")
        if row["state"][0]&13 or row["state"][2]&13:issue("ieee_flags",row["action"],row["state"])
        require(row["state"][1]==row["state"][3],"Reducer changed rounding mode")
        if row["stable"]:
            for coeff in row["values"]:
                for slot,component in enumerate(coeff):
                    a,b=component["value"]
                    if slot==0 and not row["finite"]:require(a.is_nan() and b==0,"Finite partial marker")
                    else:require(a.is_finite() and b.is_finite(),"Nonfinite successful output")
        if row["action"]<2000:base[(row["fixture"],row["backend"],row["ct"],row["mode"],row["finite"])]=row
    def compare(left,right,slots,bound,label):
        if not left["stable"] or not right["stable"]:
            checks.append({"kind":label,"actions":[left["action"],right["action"]],"status":"refused"});return
        worst=D(0);q=D(2 if left["fixture"]==106 else 1000)
        independent='independent_collier' in label
        regression=True;target=True;worst_relative=D(0);worst_symmetric=D(0);undefined=False
        for i,(lv,rv) in enumerate(zip(left["values"],right["values"])):
            power=2*left["n"]-4-sum(monomials[i]);factor=q**power
            for slot in slots:
                for part,(a,b) in enumerate(zip(lv[slot]["value"],rv[slot]["value"])):
                    worst=max(worst,residual(a*factor,b*factor))
                    magnitude=max(abs(a),abs(b))
                    symmetric=abs(a-b)/abs(a+b) if a+b else D(0) if magnitude==0 else None
                    relative=abs(a-b)/magnitude if magnitude else D(0)
                    worst_relative=max(worst_relative,relative)
                    if symmetric is None:undefined=True
                    else:worst_symmetric=max(worst_symmetric,symmetric)
                    finite_values=[abs(v) for v in (lv[0]['value'][part],rv[0]['value'][part]) if v.is_finite()]
                    finite_scale=max([D('1e-99')]+finite_values)
                    def accepted(threshold):
                        return (symmetric is not None and symmetric<threshold) or (slot>0 and magnitude/finite_scale<threshold)
                    regression=regression and accepted(bound)
                    target=target and accepted(D('1e-8'))
        passed=regression if independent else worst<=bound
        checks.append({"kind":label,"actions":[left["action"],right["action"]],"status":"passed" if passed else "mismatched",
          "max_scaled_residual":str(worst),"max_raw_relative":str(worst_relative),
          "max_symmetric":None if undefined else str(worst_symmetric),
          "criterion":"abs(a-b)/abs(a+b), with same-coefficient finite-relative small-pole exception" if independent else "dimensionless max(1,abs(a),abs(b)) floor",
          "target_1e8_met":target if independent else worst<=D('1e-8')})
        if not passed:issue("mismatched",left["action"],label+" "+str(worst))
    for fixture in range(101,108):
        for ct in (1,4):
            for mode in range(3):
                full=base[(fixture,9,ct,mode,1)];partial=base[(fixture,9,ct,mode,0)]
                # Full/partial paths receive bit-identical ABI input buffers.
                if native[full['action']]['present'] and native[partial['action']]['present']:
                    require(all(native[full['action']][k]==native[partial['action']][k] for k in ('p','s','m','mu')),"Full/partial effective inputs differ")
                compare(full,partial,(1,2),D('1e-10') if ct==1 else D('1e-24'),"full_partial")
            for finite in (0,1):
                both=base[(fixture,9,ct,0,finite)];uv=base[(fixture,9,ct,1,finite)];ir=base[(fixture,9,ct,2,finite)]
                if not all(row['stable'] for row in (both,uv,ir)):continue
                worst=D(0);q=D(2 if fixture==106 else 1000)
                for i,(bv,uvv,iv) in enumerate(zip(both['values'],uv['values'],ir['values'])):
                    factor=q**(2*both['n']-4-sum(monomials[i]))
                    require(all(v==0 for v in uvv[2]['value']),"Nonzero UV double pole")
                    for slot in (1,2):
                        for b,u,v in zip(bv[slot]['value'],uvv[slot]['value'],iv[slot]['value']):
                            worst=max(worst,residual(b*factor,(u+v)*factor))
                if worst>(D('1e-10') if ct==1 else D('1e-24')):issue('pole_reconstruction',both['action'],str(worst))
                if finite:
                    compare(both,uv,(0,),D(0),'selector_finite_exact')
                    compare(both,ir,(0,),D(0),'selector_finite_exact')
        for mode in range(3):
            tir=base[(fixture,9,1,mode,1)];cll=base[(fixture,7,1,mode,1)]
            t,c=native[tir["action"]],native[cll["action"]]
            if not t['present']:continue
            p=native_values(t,"p");s=native_values(t,"s");m=native_values(t,"m");n=tir["n"]
            cp=[D.from_float(x[0]) for x in struct.iter_unpack('<d',bytes.fromhex(c['cll_p']))]
            cs=[D.from_float(x[0]) for x in struct.iter_unpack('<d',bytes.fromhex(c['cll_s']))]
            cm=[D.from_float(x[0]) for x in struct.iter_unpack('<d',bytes.fromhex(c['cll_m']))]
            pairs=[]
            for length in range(1,n):
                for start in range(n):
                    if len(pairs)==n*(n-1)//2:break
                    pairs.append((start,(start+length)%n))
                if len(pairs)==n*(n-1)//2:break
            same=(all(x==0 for x in p[:4]) and cp[::2]==p[4:] and all(x==0 for x in cp[1::2])
                  and cm==m and cs[::2]==[s[i+j*n] for i,j in pairs] and all(x==0 for x in cs[1::2]))
            if not same:issue("input_policy_mismatch",tir["action"],"COLLIER effective inputs differ; coefficients are not counted as independent agreement")
            else:
                compare(tir,cll,(0,1,2),D('3e-6'),"independent_collier")
                compare(base[(fixture,9,1,mode,0)],cll,(1,2),D('3e-6'),"partial_independent_collier")
                # QP uses the same physical coordinates but may build slightly
                # different invariants. Do not claim exact-input QP/CLLIER parity
                # without checking those actual decimal ABI values too.
                qp=base[(fixture,9,4,mode,1)];qo=native[qp['action']]
                if qo['present'] and all(native_values(qo,k)==native_values(t,k) for k in ('p','s','m','mu')):
                    compare(qp,cll,(0,1,2),D('3e-6'),'qp_independent_collier')
                    compare(base[(fixture,9,4,mode,0)],cll,(1,2),D('3e-6'),'qp_partial_independent_collier')
                else:issue('input_policy_mismatch',qp['action'],'QP and COLLIER effective invariants are not identical')
    for ct in (1,4):
        for finite in (0,1):
            for mode in range(3):
                low=base[(103,9,ct,mode,finite)];high=base[(107,9,ct,mode,finite)]
                if not low['stable'] or not high['stable']:continue
                log4=D(4).ln();worst=D(0)
                for i,(lo,hi) in enumerate(zip(low['values'],high['values'])):
                    factor=D(1000)**(8-sum(monomials[i]))
                    for part in (0,1):
                        expect=lo[1]['value'][part]+log4*lo[2]['value'][part]
                        worst=max(worst,residual(expect*factor,hi[1]['value'][part]*factor),
                                  residual(lo[2]['value'][part]*factor,hi[2]['value'][part]*factor))
                if worst>(D('1e-10') if ct==1 else D('1e-24')):issue('scale_pole_mixing',high['action'],str(worst))
    # Analytical UV/IR residues are independent of either implementation.
    for ct in (1,4):
        for finite in (0,1):
            for mode in range(3):
                row=base[(106,9,ct,mode,finite)]
                if not row["stable"]:continue
                for i,exponents in enumerate(monomials):
                    rank=sum(exponents)
                    expect=D(0) if exponents[1]+exponents[2] else D(2)**rank/D(rank+1)
                    expected=expect if mode==1 else -expect if mode==2 else D(0)
                    actual=row['values'][i]
                    bound=D('1e-10') if ct==1 else D('1e-24')
                    if residual(actual[1]['value'][0],expected)>bound or actual[1]['value'][1]!=0 or any(v!=0 for v in actual[2]['value']):
                        issue('analytical_scaleless_mismatch',row['action'],i);break
                    if finite and any(v!=0 for v in actual[0]['value']):
                        issue('analytical_scaleless_finite',row['action'],i);break
    # Cache probes use independently retained uncached references.
    for row in allcases:
        if row['action']<2000:continue
        ct=1 if row['ct']<4 else 4
        if row['ct'] in (2,5):
            other=next(r for r in allcases if r['fixture']==104 and r['ct']==row['ct'] and r['cache']==0)
        else:other=base[(104,9,ct,row['mode'],row['finite'])]
        compare(row,other,(0,1,2) if row['finite'] else (1,2),D(0),'cache_exact')
        if row['stable'] and other['stable']:
            require(all(a[slot]['words']==b[slot]['words'] for a,b in zip(row['values'],other['values'])
                        for slot in ((0,1,2) if row['finite'] else (1,2))),"Cached output words differ")
        if row['budget']==0:require(all(row['post'][i]==0 for i in (6,7,11,12)),"Zero budget stored numerical values")
        if row['ct'] in (2,5) and row['stable']:
            require(row['post'][9]==row['pre'][9],"Check/reference consumed QP primary cache")
            if row['ct']==2:require(all(row['post'][i]==row['pre'][i] for i in (0,2,4,6,7)),"DP check/reference consumed primary cache")
        j=row['action']-(2000+ct*10)
        if j in (1,3) and row['stable']:
            require(row['post'][9]>row['pre'][9] if ct==4 else any(row['post'][i]>row['pre'][i] for i in (0,2,4)),"No native repeat cache-hit witness")
    for capture,fixture in ((next(c for c in allcaptures if c['action']==901),101),
                            (next(c for c in allcaptures if c['action']==903),102)):
        ref=base[(fixture,9,1,0,1)]
        if capture['stable'] and ref['stable']:
            require(capture['values']==ref['values'],"Captured actual tensor and standalone replay differ")
        else:issue('capture_refused',capture['action'],'Cannot establish successful actual-group full replay')
    witnesses={str(f):any(r['stable'] and r['fixture']==f and any(abs(z)>0 for v in r['values'] for slot in (1,2) for z in v[slot]['value'])
                         for r in allcases if r['backend']==9) for f in range(101,108)}
    if not witnesses['106']:issue('missing_nonzero_witness',106,'Separate scaleless UV/IR residues')
    if not any(witnesses[str(f)] for f in (101,102,103,104,105,107)):
        issue('missing_nonzero_witness',0,'Physical-geometry IR poles')
    return {"schema":1,"status":"passed" if not open_items else "open_acceptance_items",
            "scope":"generated tensor diagnostics; synthetic rank-six extension; not physical amplitude or timing qualification",
            "native_tirust_calls":sum(r['present'] for r in native.values()),"collier_adapter_calls":21,
            "checks":checks,"open_items":open_items,"nonzero_pole_witnesses":witnesses,
            "finite_work_omission":"current finite-mode dispatch plus separately bound core/source instrumentation; no dynamic finite-operation count"}


if __name__=="__main__":
    require(len(sys.argv)==3,"Expected EW and QCD retained work directories")
    with localcontext() as context:
        context.prec=100
        print(json.dumps(analyze([Path(x) for x in sys.argv[1:]]),indent=2,sort_keys=True))
