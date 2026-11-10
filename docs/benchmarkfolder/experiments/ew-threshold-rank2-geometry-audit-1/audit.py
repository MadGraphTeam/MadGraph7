"""Pure rank-two contraction diagnostics of closed historical EW tensors.

Retained TIRust-S triangle coefficients are explicitly regathered in box
coordinates. They are not independently evaluated q-consistent pinches.
No reducer, native library, compiler, loader, input modification or dimensional
trace identity is used. This is not an integral oracle or acceptance test.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / "ew-threshold-adjudication"
REPLAY = BASE / "tensor-replay/higher-precision/replay-forward"
EXACT = BASE / "tensor-replay/input-consistency/audit.py"
METRIC = (1, -1, -1, -1)
ZERO = (F(0), F(0))
PINCHES = {1: (8, (2, 3, 4)), 2: (196, (3, 4, 1)),
           3: (4, (2, 1, 4)), 4: (16, (1, 2, 3))}
MAPPING = (1,2,3,4,5,6,7,10,8,11,13,9,12,14,15,
           16,17,20,26,18,21,27,23,29,32,19,22,28,24,30,33,25,31,34,35)
PAIRS = tuple((i,j) for i in range(4) for j in range(i,4))


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(a,b): return (a[0]+b[0],a[1]+b[1])
def sub(a,b): return (a[0]-b[0],a[1]-b[1])
def scale(t,a): return (t*a[0],t*a[1])
def total(values):
    answer = ZERO
    for value in values: answer = add(answer,value)
    return answer
def dot(a,b): return sum(METRIC[i]*a[i]*b[i] for i in range(4))
def det3(a):
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])


def solve(matrix, vector):
    n = len(matrix)
    require(n > 0 and len(vector)==n and all(len(row)==n for row in matrix), "Square solve shape")
    rows = [[F(x) for x in row]+[F(value)] for row,value in zip(matrix,vector)]
    for j in range(n):
        pivot = max(range(j,n),key=lambda i:abs(rows[i][j]))
        require(rows[pivot][j] != 0,"Singular exact solve; no fallback")
        rows[j],rows[pivot]=rows[pivot],rows[j]
        divisor=rows[j][j];rows[j]=[x/divisor for x in rows[j]]
        for i in range(n):
            if i==j: continue
            multiple=rows[i][j];rows[i]=[a-multiple*b for a,b in zip(rows[i],rows[j])]
    return [row[-1] for row in rows]


def invert(matrix):
    n=len(matrix);columns=[solve(matrix,[int(i==j) for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def decimal(x): return Decimal(x.numerator)/Decimal(x.denominator)
def text(x): return format(decimal(x),".55g")
def ctext(z): return [text(x) for x in z]
def norm(vector): return sum(decimal(x)**2 for z in vector for x in z).sqrt()
def ratio(a,b): return str(a/b) if b else None


def geometry(shifts, distances, masses):
    require(all(value[1]==0 for value in masses.values()),"Complex masses outside this bounded audit")
    origin=shifts[1]
    p=[[shifts[j][mu]-origin[mu] for mu in range(4)] for j in range(2,len(shifts)+1)]
    zq=[[2*dot(a,b) for b in p] for a in p]
    zs=[[distances[1,i]+distances[1,j]-distances[i,j]
         for j in range(2,len(shifts)+1)] for i in range(2,len(shifts)+1)]
    fq=[dot(p[j-2],p[j-2])+masses[1][0]-masses[j][0] for j in range(2,len(shifts)+1)]
    fs=[distances[1,j]+masses[1][0]-masses[j][0] for j in range(2,len(shifts)+1)]
    return origin,p,zq,zs,fq,fs


def untranslate(scalar, moment1, moment2, origin):
    j=[add(scale(-1,moment1[mu]),scale(origin[mu],scalar)) for mu in range(4)]
    k=[[add(sub(sub(moment2[mu][nu],scale(origin[mu],moment1[nu])),
                 scale(origin[nu],moment1[mu])),scale(origin[mu]*origin[nu],scalar))
         for nu in range(4)] for mu in range(4)]
    return j,k


def tensor_from_components(values):
    require(len(values)==10,"Exactly ten rank2 component values required")
    matrix=[[ZERO for _ in range(4)] for _ in range(4)]
    for (mu,nu),value in zip(PAIRS,values):matrix[mu][nu]=matrix[nu][mu]=value
    return matrix


def recover_vector_coefficients(moment,origin,scalar,p):
    require(len(p)==2,"Triangle must have two basis vectors")
    vector=[add(scale(-1,moment[mu]),scale(origin[mu],scalar)) for mu in range(4)]
    axes=max(itertools.combinations(range(4),2),key=lambda axes:abs(p[0][axes[0]]*p[1][axes[1]]-p[0][axes[1]]*p[1][axes[0]]))
    minor=[[p[a][mu] for a in range(2)] for mu in axes]
    parts=[solve(minor,[vector[mu][part] for mu in axes]) for part in range(2)]
    coefficients=list(zip(*parts))
    residual=[sub(total(scale(p[a][mu],coefficients[a]) for a in range(2)),vector[mu]) for mu in range(4)]
    return coefficients,residual,axes


def regather(coefficients,scalar,box_shifts,permutation):
    require(sorted(permutation)==sorted(set(permutation)) and len(permutation)==3 and all(i in box_shifts for i in permutation),"Bad pinch permutation")
    origin=box_shifts[permutation[0]]
    p=[[box_shifts[i][mu]-origin[mu] for mu in range(4)] for i in permutation[1:]]
    return [add(total(scale(p[a][mu],coefficients[a]) for a in range(2)),
                scale(box_shifts[1][mu]-origin[mu],scalar)) for mu in range(4)]


def affine_routing(box_shifts,triangle_shifts,permutation):
    candidates=[]
    for eta in (1,-1):
        translation=[box_shifts[permutation[0]][mu]-eta*triangle_shifts[1][mu] for mu in range(4)]
        defects=[[box_shifts[permutation[a-1]][mu]-eta*triangle_shifts[a][mu]-translation[mu]
                  for mu in range(4)] for a in (1,2,3)]
        maximum=max(abs(x) for row in defects for x in row)
        candidates.append((maximum,eta,translation,defects))
    maximum,eta,translation,defects=min(candidates,key=lambda item:item[0])
    return {"eta":eta,"translation_box_minus_eta_triangle":[text(x) for x in translation],
            "max_absolute_affine_defect":text(maximum),"exact_affine_routing":maximum==0,
            "affine_defects":[[text(x) for x in row] for row in defects]}


def decompose(k,p):
    """Extract metric piece using a normal, never the four-dimensional trace."""
    normal=[(-1)**mu*det3([[p[a][nu] for nu in range(4) if nu!=mu] for a in range(3)]) for mu in range(4)]
    n2=sum(METRIC[mu]*normal[mu]**2 for mu in range(4))
    require(n2!=0,"Singular box3-plane normal")
    metric_coefficient=scale(1/n2,total(scale(normal[mu]*normal[nu],k[mu][nu]) for mu in range(4) for nu in range(4)))
    gram=[[dot(a,b) for b in p] for a in p];inverse=invert(gram)
    dual=[[sum(inverse[a][b]*METRIC[mu]*p[b][mu] for b in range(3)) for mu in range(4)] for a in range(3)]
    remainder=[[sub(k[mu][nu],scale(METRIC[mu] if mu==nu else 0,metric_coefficient)) for nu in range(4)] for mu in range(4)]
    b=[[total(scale(dual[a][mu]*dual[c][nu],remainder[mu][nu]) for mu in range(4) for nu in range(4)) for c in range(3)] for a in range(3)]
    reconstructed=[[add(scale(METRIC[mu] if mu==nu else 0,metric_coefficient),
                         total(scale(p[a][mu]*p[c][nu],b[a][c]) for a in range(3) for c in range(3))) for nu in range(4)] for mu in range(4)]
    residual=[sub(reconstructed[mu][nu],k[mu][nu]) for mu,nu in PAIRS]
    antisymmetric=[sub(b[a][c],b[c][a]) for a in range(3) for c in range(a+1,3)]
    return metric_coefficient,b,residual,antisymmetric


def contractions(k,jmoment,p,zs,fs,fq,metric_coefficient,b,kind):
    output=[]
    for j in range(3):
        if kind=="exact_shift_cartesian":
            contraction=[total(scale(2*METRIC[mu]*p[j][mu],k[mu][nu]) for mu in range(4)) for nu in range(4)]
            f=fq[j]
        else:
            require(kind=="formal_supplied_S","Unknown contraction geometry")
            contraction=[add(scale(2*p[j][nu],metric_coefficient),
                              total(scale(p[a][nu]*zs[j][c],b[c][a]) for a in range(3) for c in range(3))) for nu in range(4)]
            f=fs[j]
        scalar=[scale(f,x) for x in jmoment]
        output.append((contraction,scalar,[add(a,b) for a,b in zip(contraction,scalar)]))
    return output


def bind_pinch(box,triangle,permutation,number):
    require(triangle["status"]==0 and triangle["n"]==3 and triangle["rank"]>=1,"Missing successful rank1 triangle")
    require(number(triangle["scale"])==number(box["scale"]),"Pinch scale mismatch")
    bm={int(i):(number(re),number(im)) for i,re,im in box["masses"]};tm={int(i):(number(re),number(im)) for i,re,im in triangle["masses"]}
    bs={(int(i),int(j)):number(x) for i,j,x in box["invariants"]};ts={(int(i),int(j)):number(x) for i,j,x in triangle["invariants"]}
    require(all(tm[i]==bm[permutation[i-1]] for i in (1,2,3)),"Pinch mass/permutation mismatch")
    require(all(ts[i,j]==bs[permutation[i-1],permutation[j-1]] for i in (1,2,3) for j in (1,2,3)),"Pinch invariant/permutation mismatch")


def decode(record,number):
    shifts={int(i):list(map(number,row)) for i,*row in record["shifts"]}
    masses={int(i):(number(re),number(im)) for i,re,im in record["masses"]}
    distances={(int(i),int(j)):number(x) for i,j,x in record["invariants"]}
    require(all(value[1]==0 for value in masses.values()),"This bounded audit requires retained real squared masses")
    return shifts,masses,distances


def raw_slots(record,number):
    raw={(int(i),int(k)):(number(re),number(im)) for i,k,re,im in record["raw"]}
    require(len(raw)==len(record["raw"]),"Duplicate raw coefficient")
    return lambda i,slot:raw[i,4] if slot==0 else add(raw[i,1],raw[i,2]) if slot==1 else raw[i,3]


def massless_availability(trace,selected,number):
    box=next(r for r in selected["records"] if (r["point"],r["group"])==(2,24))
    results=[]
    for omitted in (1,2,3,4):
        matches=[]
        for gid,triangle in trace["2"]["groups"].items():
            if triangle["status"]!=0 or triangle["n"]!=3 or triangle["rank"]<1: continue
            for permutation in itertools.permutations(sorted(set(range(1,5))-{omitted})):
                try: bind_pinch(box,triangle,permutation,number)
                except ValueError: continue
                matches.append({"group":int(gid),"permutation":list(permutation)})
        results.append({"omitted":omitted,"matches":matches})
    require(not results[0]["matches"] and not results[2]["matches"] and len(results[3]["matches"])>1,
            "Saved massless availability changed; reassess explicit missing-data claim")
    return {"point":2,"group":24,"audited":False,"reason":"No retained successful rank1 pinch match for omitted1 or3; omit4 also ambiguous. No synthetic pinch is substituted.","pinches":results}


def evaluate():
    require(Path.cwd().resolve()==ROOT,"Run only from required checkout")
    spec=importlib.util.spec_from_file_location("rank2_exact_binary128",EXACT);exact=importlib.util.module_from_spec(spec);spec.loader.exec_module(exact)
    number=exact.binary128
    paths=[EXACT,BASE/"trace.json",BASE/"execution.json",BASE/"summary.json",BASE/"artifact-index.json",
           REPLAY/"selected-inputs.json",REPLAY/"selected-inputs.dat",REPLAY/"summary.json",REPLAY/"execution.json",REPLAY/"source-manifest.json",REPLAY/"output.log",REPLAY/"replay.f90",REPLAY/"analyze.py",
           HERE.parent/"ew-threshold-geometry-supplement/evidence-index.json",
           HERE.parent/"ew-threshold-geometry-supplement/source-excerpts.txt",
           BASE/"tensor-replay/higher-precision/source/lib.rs"]
    pins={str(path.relative_to(ROOT)):{"sha256":sha(path),"bytes":path.stat().st_size} for path in paths}
    trace=json.loads((BASE/"trace.json").read_text());selected=json.loads((REPLAY/"selected-inputs.json").read_text());summary=json.loads((REPLAY/"summary.json").read_text());execution=json.loads((REPLAY/"execution.json").read_text())
    trace_execution=json.loads((BASE/"execution.json").read_text())
    require(trace_execution["complete"] and trace_execution["final_values_identical_to_historical"],"Original amplitude trace incomplete/changed")
    require(execution["complete"] and execution["returncode"]==0 and execution["protected_inputs_unchanged"],"Replay execution incomplete")
    require(summary["raw_output_sha256"]==execution["output_sha256"]==sha(REPLAY/"output.log"),"Raw output binding")
    for field,path in (("execution_manifest_sha256",REPLAY/"execution.json"),("input_manifest_sha256",REPLAY/"selected-inputs.json"),("analyzer_sha256",REPLAY/"analyze.py")):
        require(summary[field]==sha(path),"Summary dependency changed")
    for name in ("trace","summary","execution"):
        require(selected["parent_"+name+"_sha256"]==sha(BASE/(name+".json")),"Parent trace binding")
    require(selected["source_manifest_sha256"]==execution["source_manifest_sha256"]==sha(REPLAY/"source-manifest.json"),"Replay source binding")
    require(selected["input_stream_sha256"]==execution["input_stream_sha256"]==sha(REPLAY/"selected-inputs.dat"),"Replay input binding")
    raw={};current=None
    for line in (REPLAY/"output.log").read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=="BEGIN":current=tuple(map(int,fields[1:3]))
        elif fields[0]=="COMPARE":
            key=(*current,int(fields[1]),int(fields[2]));require(key not in raw,"Duplicate raw comparison");raw[key]={"tir":fields[3:5],"nrs":fields[5:7]}
    results=[]
    with localcontext() as context:
        context.prec=80
        for point in (1,2):
            inp=next(r for r in selected["records"] if (r["point"],r["group"])==(point,5));out=next(r for r in summary["records"] if (r["point"],r["group"])==(point,5))
            require(inp["n"]==4 and inp["rank"]==3 and out["tir_status"]==0 and len(out["comparisons"])==105,"Expected saved rank3 successful box")
            for field in ("shifts","masses","invariants"):
                require([[str(x) for x in row] for row in inp[field]]==trace[str(point)]["groups"]["5"][field],"Saved box input changed")
            require(inp["scale"]==trace[str(point)]["groups"]["5"]["scale"],"Saved box scale changed")
            qb,mb,sb=decode(inp,number);origin,p,zq,zs,fq,fs=geometry(qb,sb,mb)
            observations={(r["unit"],r["slot"]):r for r in out["comparisons"]}
            require(len(observations)==105,"Duplicate unit/slot")
            for row in out["comparisons"]:
                require(raw[point,5,row["unit"],row["slot"]]=={b:row[b] for b in ("tir","nrs")},"Summarized tensor bytes changed")
            for unit in range(35):
                native=out["NRS_units"][str(unit)]
                require(native["status"]==0 and native["mapping"]==MAPPING[unit],"Native unit mapping/status")
                require(native["degree"]==(0 if unit==0 else 1 if unit<5 else 2 if unit<15 else 3),"Native unit degree")
            pinched={};bindings=[]
            for omitted,(gid,permutation) in PINCHES.items():
                triangle=trace[str(point)]["groups"][str(gid)];bind_pinch(inp,triangle,permutation,number)
                qt,mt,st=decode(triangle,number);to,tp,*_=geometry(qt,st,mt);slots=raw_slots(triangle,number)
                per_slot=[];pinched[omitted]={}
                for slot in range(3):
                    scalar=slots(1,slot);moment=[slots(i,slot) for i in range(2,6)]
                    c,residual,axes=recover_vector_coefficients(moment,to,scalar,tp)
                    pinched[omitted][slot]=regather(c,scalar,qb,permutation)
                    per_slot.append({"slot":slot,"scalar":ctext(scalar),"coefficient_axes":list(axes),"S_based_coefficients":[ctext(x) for x in c],"native_vector_reconstruction_residual":[ctext(x) for x in residual],"native_vector_reconstruction_L2":str(norm(residual))})
                bindings.append({"omitted":omitted,"group":gid,"permutation":list(permutation),"status":triangle["status"],"stored_rank":triangle["rank"],"exact_S_masses_scale_match":True,"routing":affine_routing(qb,qt,permutation),"slots":per_slot})
            identities=[];decompositions=[]
            raw_to_unit={v:k for k,v in enumerate(MAPPING)}
            for backend in ("tir","nrs"):
                for slot in range(3):
                    scalar=tuple(map(number,observations[0,slot][backend]));m1=[tuple(map(number,observations[i,slot][backend])) for i in range(1,5)]
                    m2=tensor_from_components([tuple(map(number,observations[raw_to_unit[raw_index],slot][backend])) for raw_index in range(6,16)])
                    jm,k=untranslate(scalar,m1,m2,origin);a,b,reconstruction,antisym=decompose(k,p)
                    decompositions.append({"backend":backend,"slot":slot,"metric_coefficient":ctext(a),"reconstruction_residual_by_unique_component":[ctext(x) for x in reconstruction],"reconstruction_L2":str(norm(reconstruction)),"momentum_matrix_antisymmetric_residual":[ctext(x) for x in antisym]})
                    for kind in ("exact_shift_cartesian","formal_supplied_S"):
                        for index,(contraction,scalar_term,lhs) in enumerate(contractions(k,jm,p,zs,fs,fq,a,b,kind)):
                            j=index+2;vj=pinched[j][slot];v1=pinched[1][slot];rhs=[sub(x,y) for x,y in zip(vj,v1)];residual=[sub(x,y) for x,y in zip(lhs,rhs)]
                            absolute=norm(residual);rhs_scale=norm(rhs);term_scale=max(map(norm,(contraction,scalar_term,vj,v1)))
                            identities.append({"backend":backend,"slot":slot,"slot_name":("finite","single","double")[slot],"geometry":kind,"j":j,"rhs_policy":"Same retained TIRust-S triangle coefficients, explicitly regathered in box-pinch coordinates; no q recurrence conversion or independent pinch oracle.","residual_vector":[ctext(x) for x in residual],"absolute_residual_L2":str(absolute),"rhs_scale":str(rhs_scale),"rhs_exactly_zero":all(x==ZERO for x in rhs),"largest_uncancelled_term_L2":str(term_scale),"relative_to_rhs":ratio(absolute,rhs_scale),"relative_to_largest_uncancelled_term":ratio(absolute,term_scale)})
            maxima=[]
            for backend,kind,slot in itertools.product(("tir","nrs"),("exact_shift_cartesian","formal_supplied_S"),range(3)):
                rows=[r for r in identities if (r["backend"],r["geometry"],r["slot"])==(backend,kind,slot)]
                witness=max(rows,key=lambda r:Decimal(r["absolute_residual_L2"]))
                defined=[r for r in rows if r["relative_to_largest_uncancelled_term"] is not None]
                maxima.append({"backend":backend,"geometry":kind,"slot":slot,"maximum_absolute_L2":witness["absolute_residual_L2"],"absolute_witness_j":witness["j"],"maximum_relative_to_uncancelled_term":max((r["relative_to_largest_uncancelled_term"] for r in defined),key=Decimal,default=None),"zero_rhs_rows":sum(r["rhs_exactly_zero"] for r in rows)})
            results.append({"point":point,"group":5,"historical_box_status":inp["historical_status"],"private512_box_status":out["tir_status"],"pinch_bindings":bindings,"rank2_decompositions":decompositions,"identity_rows":identities,"maxima":maxima})
        missing=massless_availability(trace,selected,number)
    for path,metadata in pins.items():require(sha(ROOT/path)==metadata["sha256"],"Closed input mutated during audit")
    return {"schema":"closed-ew-rank2-contraction-diagnostic-1","scope":"Historical conditional geometry/recurrence residuals only, not independent accuracy or current stress-failure adjudication.","native_execution":False,"production_change":False,"independent_integral_oracle":False,"scientific_acceptance":False,"full_rank2_resolution":False,"transverse_metric_freedom":"Three external Gram contractions leave a transverse tensor component; contraction residuals alone do not determine the full tensor.","arithmetic":"Exact rational operations on decoded retained binary128 values;80decimal digits for norms,55significant digits for displayed rational values.","normalization":"Convention::MadLoop; TensorWeights::Components. No offdiagonal factor2, odd rank minus sign exactly once, masses already squared. Finite raw4; merged single raw1+raw2; double raw3. No real projection or2/(16*pi^2). NRS finite already contains R1.","pinch_provenance":"Historical256-bit TIRust triangles; compared box TIRust private512 and NinjaRS256, all returned at113bits. Regather preserves triangleS coefficients; no bubble RHS or NRS triangles available as independent vectors.","limits":["Neither geometry is declared uniquely correct; no input or native integral is changed.","Tiny signed-affine routing defects are retained, not treated as exact denominator equivalence.","Residuals include retained tensor rounding, conditioning and pinch provenance; they cannot isolate a box error.","FormalS metric contraction stays2*A*p_j; only momentum-pair contractions use suppliedS.","Metric coefficient comes from an orthogonal normal, not a four-dimensional trace; no dimensional trace or Laurent-mixing inference is used.","No full tensor geometry prediction is attempted; no Gram guard is relaxed.","Current EW status420 is an availability/refusal failure; this historical group5 audit does not prove the current failed group/cause."],"sources":pins,"sources_unchanged":True,"records":results,"unavailable_control":missing}


if __name__=="__main__":
    # The only output mutation is in this new, task-owned additive namespace.
    destination=HERE/"summary.json";require(not destination.exists(),"Refusing to overwrite closed summary")
    result=evaluate();destination.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"summary_sha256":sha(destination),"cases":len(result["records"]),"native_execution":False}))
