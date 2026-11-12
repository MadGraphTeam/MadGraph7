"""Exact-bit candidate capture and same-engine Float512 tensor comparison."""
from pathlib import Path
import argparse,ctypes,hashlib,json,math,struct,sys
from decimal import Decimal,localcontext

WORK=Path(__file__).resolve().parents[1]
DSO=WORK/"install/lib/libmadloop_tirust.so"
DSO_SHA="470d46df9c8fcb76dc5448a04187dfc23a757481b3d4ef0384e13bd6e7e9e709"
FIXTURES=WORK/"captured-failures.jsonl"
FIXTURE_SHA="4027cb7d650c535d2813926c4244724b5b5e6ced19819e1ff2aacc0cb29eeeb0"
SLOTS=("UV1","IR1","IR2","finite")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f64(bits):return struct.unpack("<d",int(bits,16).to_bytes(8,"little"))[0]
def bits(value):return f"{struct.unpack('<Q',struct.pack('<d',value))[0]:016x}"
def fixtures():
    assert sha(FIXTURES)==FIXTURE_SHA
    rows=[json.loads(s) for s in FIXTURES.read_text().splitlines()]
    assert len(rows)==8
    return rows
def basis(rank):
    return [(t,x,y,r-t-x-y) for r in range(rank+1) for t in range(r,-1,-1)
            for x in range(r-t,-1,-1) for y in range(r-t-x,-1,-1)]
def metric(candidate,reference):
    if len(candidate)!=len(reference) or not candidate:raise ValueError("Vector length mismatch")
    if any(not v.is_finite() for pairs in (candidate,reference) for z in pairs for v in z):raise ValueError("Nonfinite value")
    delta=sum(((c[0]-r[0])**2+(c[1]-r[1])**2 for c,r in zip(candidate,reference)),Decimal(0))
    cnorm=sum((re*re+im*im for re,im in candidate),Decimal(0))
    rnorm=sum((re*re+im*im for re,im in reference),Decimal(0))
    scale=max(cnorm,rnorm)
    rho=(delta/scale).sqrt() if scale else Decimal(0)
    return dict(residual=str(rho),delta_norm=str(delta.sqrt()),candidate_norm=str(cnorm.sqrt()),reference_norm=str(rnorm.sqrt()),
                exact_both_zero=not scale,one_zero=bool((cnorm==0)!=(rnorm==0)),
                pass_3e6=delta<=Decimal("3e-6")**2*scale,pass_1e8=delta<=Decimal("1e-8")**2*scale)
def capture():
    assert sha(DSO)==DSO_SHA
    class C64(ctypes.Structure):_fields_=[("re",ctypes.c_double),("im",ctypes.c_double)]
    lib=ctypes.CDLL(str(DSO))
    ci,sz,ptr=ctypes.c_int,ctypes.c_size_t,ctypes.c_void_p
    lib.ml_tirust_context_create.argtypes=[ctypes.c_uint32,ctypes.POINTER(ptr)];lib.ml_tirust_context_create.restype=ci
    lib.ml_tirust_context_destroy.argtypes=[ptr];lib.ml_tirust_context_destroy.restype=ci
    lib.ml_tirust_evaluate_dp.argtypes=[ptr,ci,ci,ctypes.POINTER(ctypes.c_double),sz,ctypes.POINTER(ctypes.c_double),sz,ctypes.POINTER(C64),sz,ctypes.c_double,ctypes.POINTER(C64),sz,ci,ci,ci,ci];lib.ml_tirust_evaluate_dp.restype=ci
    lib.ml_tirust_dp_route_statistics.argtypes=[ptr,ctypes.POINTER(ctypes.c_uint64),sz];lib.ml_tirust_dp_route_statistics.restype=ci
    lib.ml_tirust_cache_statistics.argtypes=[ptr,ctypes.POINTER(ctypes.c_uint64),sz];lib.ml_tirust_cache_statistics.restype=ci
    lib.ml_tirust_last_error.argtypes=[ctypes.c_void_p,sz];lib.ml_tirust_last_error.restype=sz
    mapped={line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if line.split()[-1].endswith("libmadloop_tirust.so")}
    assert mapped=={str(DSO.resolve())}
    print(json.dumps(dict(record="metadata",dso_sha256=DSO_SHA,mapped_dso=str(DSO.resolve()),fixture_sha256=FIXTURE_SHA,working_precision="reported-per-route",input_precision=53,performance_eligible=False)),flush=True)
    failed=0
    for index,row in enumerate(fixtures()):
        ctx=ptr();assert lib.ml_tirust_context_create(1,ctypes.byref(ctx))==0
        try:
            p=(ctypes.c_double*len(row["p_f64_bits"]))(*(f64(s) for s in row["p_f64_bits"]))
            s=(ctypes.c_double*len(row["s_f64_bits"]))(*(f64(s) for s in row["s_f64_bits"]))
            raw=row["m_f64_bits"];m=(C64*row["n"])(*(C64(f64(raw[i]),f64(raw[i+1])) for i in range(0,len(raw),2)))
            assert row["out_len"]==4*math.comb(row["rank"]+4,4)
            out=(C64*row["out_len"])(*(C64(float("nan"),float("nan")) for _ in range(row["out_len"])))
            status=lib.ml_tirust_evaluate_dp(ctx,row["n"],row["rank"],p,len(p),s,len(s),m,len(m),f64(row["mu2_f64_bits"][0]),out,len(out),row["poles"],row["finite"],row["check"],row["cache"])
            size=lib.ml_tirust_last_error(None,0);message=ctypes.create_string_buffer(size);lib.ml_tirust_last_error(message,size)
            routes=(ctypes.c_uint64*20)();cache=(ctypes.c_uint64*9)()
            route_status=lib.ml_tirust_dp_route_statistics(ctx,routes,20);cache_status=lib.ml_tirust_cache_statistics(ctx,cache,9)
            result=dict(fixture=index,n=row["n"],rank=row["rank"],check=bool(row["check"]),status=status,diagnostic_utf8_hex=message.value.hex(),
                        values=[[bits(z.re),bits(z.im)] for z in out],route_statistics=list(routes),route_status=route_status,cache_statistics=list(cache),cache_status=cache_status,
                        input_fingerprint_fnv128=row["input_fingerprint_fnv128"],fixture_sha256=FIXTURE_SHA)
            print(json.dumps(result,sort_keys=True),flush=True)
            if status or route_status or cache_status:failed+=1
        finally:assert lib.ml_tirust_context_destroy(ctx)==0
    assert sha(DSO)==DSO_SHA
    if failed:raise SystemExit(1)
def load_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")]
    fixtures_only=[r for r in rows if "fixture" in r]
    assert len(fixtures_only)==8 and {r["fixture"] for r in fixtures_only}==set(range(8))
    return {r["fixture"]:r for r in fixtures_only}
def compare(oracle_path,candidate_path,output):
    reference=load_rows(oracle_path);candidate=load_rows(candidate_path)
    cases=[];all_components=[]
    with localcontext() as ctx:
        ctx.prec=200
        for i,original in enumerate(fixtures()):
            ref=reference[i];cand=candidate[i]
            assert all(ref[k]==cand[k]==original[k] for k in ("n","rank"))
            assert bool(ref["check"])==bool(cand["check"])==bool(original["check"])
            assert ref["input_bits"]==53 and ref["working_bits"]==512 and cand["fixture_sha256"]==FIXTURE_SHA
            case=dict(fixture=i,n=ref["n"],rank=ref["rank"],check=ref["check"],reference_status=ref["status"],candidate_status=cand["status"],degree_norms=[],slot_degree_norms=[])
            if ref["status"]!="success" or cand["status"]!=0:
                case.update(status="unqualified-refusal",pass_3e6=False,pass_1e8=False);cases.append(case);continue
            monomials=basis(ref["rank"]);nc=len(monomials)
            assert len(ref["values"])==len(cand["values"])==nc*4
            rvec=[(Decimal(re),Decimal(im)) for re,im in ref["values"]]
            cvec=[(Decimal.from_float(f64(re)),Decimal.from_float(f64(im))) for re,im in cand["values"]]
            for flat,(c,r) in enumerate(zip(cvec,rvec)):
                slot,m=divmod(flat,nc);difference=(c[0]-r[0],c[1]-r[1])
                raw=metric([c],[r])
                all_components.append(dict(fixture=i,flat_index=flat,component=m,monomial=list(monomials[m]),degree=sum(monomials[m]),slot=SLOTS[slot],
                    candidate_f64_bits=cand["values"][flat],candidate_exact_decimal=[str(v) for v in c],reference_decimal=ref["values"][flat],
                    difference_decimal=[str(v) for v in difference],complex_difference_norm=raw["delta_norm"],component_residual=raw["residual"]))
            for degree in range(ref["rank"]+1):
                selected=[m for m,powers in enumerate(monomials) if sum(powers)==degree]
                ids=[slot*nc+m for slot in range(4) for m in selected]
                norms=metric([cvec[j] for j in ids],[rvec[j] for j in ids])
                case["degree_norms"].append(dict(degree=degree,**norms))
                for slot,label in enumerate(SLOTS):
                    ids=[slot*nc+m for m in selected]
                    norms=metric([cvec[j] for j in ids],[rvec[j] for j in ids])
                    case["slot_degree_norms"].append(dict(degree=degree,slot=label,analytic_uv_zero=(slot==0 and degree<2*ref["n"]-4),**norms))
            case["pass_3e6"]=all(x["pass_3e6"] for x in case["degree_norms"]+case["slot_degree_norms"])
            case["pass_1e8"]=all(x["pass_1e8"] for x in case["degree_norms"]+case["slot_degree_norms"])
            case["status"]="passed" if case["pass_1e8"] else "retained-target-miss"
            cases.append(case)
    summary=dict(schema="same-input-float512-tensor-comparison-1",same_engine_reference=True,independent_physics_oracle=False,performance_eligible=False,
        reference_working_bits=512,input_bits=53,reference_threshold="2^-400",normalization="fixed-frame packed complex Euclidean vectors, homogeneous degree, combined and per Laurent slot",
        no_arbitrary_floor=True,fixture_sha256=FIXTURE_SHA,oracle_log_sha256=sha(oracle_path),candidate_log_sha256=sha(candidate_path),
        fixtures=8,all_pass_3e6=all(c["pass_3e6"] for c in cases),all_pass_1e8=all(c["pass_1e8"] for c in cases),cases=cases,component_differences=all_components)
    Path(output).write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("cases","component_differences")},sort_keys=True),flush=True)
    if not summary["all_pass_1e8"]:raise SystemExit(1)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["capture","compare"]);parser.add_argument("--oracle");parser.add_argument("--candidate");parser.add_argument("--output");args=parser.parse_args()
    if args.action=="capture":capture()
    else:compare(args.oracle,args.candidate,args.output)
