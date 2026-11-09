#!/usr/bin/env python3
"""Check native Born records and exact-input propagator arithmetic; no native calls."""
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib,json
ROOT=Path.cwd().resolve();R=ROOT/'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):(R/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
e=json.loads((R/'execution.json').read_text());assert e['complete'] and e['protected_inputs_unchanged']
for p,d in e['source_and_link_inputs_after'].items():assert sha(ROOT/p)==d,p
for f,d in e['logs_sha256'].items():assert sha(R/f)==d,f
rows=[x.split() for x in (R/'output.log').read_text().splitlines()]
dp={int(x[1]):x[2:] for x in rows if x[0]=='DP_BORN'}
qp={(int(x[1]),int(x[2])):x[3:] for x in rows if x[0]=='QP_BORN'}
den={(int(x[1]),int(x[2])):x[3:] for x in rows if x[0]=='DENOM'}
assert len(dp)==2 and len(qp)==6 and len(den)==6
assert dp[2][0]=='NaN' and dp[2][1:] == ['T','F','F']
for x in qp.values():assert x[1:]==['F']*4 and Decimal(x[0]).is_finite()
points=ROOT/'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat'
a=[[float(s) for s in line.split()] for line in points.read_text().splitlines()[6:11]]
qf=[a[0][i]-a[3][i] for i in range(4)]
qe=[Fraction.from_float(a[0][i])-Fraction.from_float(a[3][i]) for i in range(4)]
sq=lambda x:x[0]*x[0]-x[1]*x[1]-x[2]*x[2]-x[3]*x[3]
with localcontext() as c:
 c.prec=100
 dec=lambda f:Decimal(f.numerator)/Decimal(f.denominator)
 exact=dec(sq(qe))
 relative=lambda a,b:abs(Decimal(a)-Decimal(b))/abs(Decimal(b))
 out={'schema':1,'classification':'Avoidable double-precision arithmetic failure, not an exactly singular invariant in the supplied binary64 inputs. Native standalone QP Born resolves it; automatic full-loop acceptance not yet qualified.',
      'exact_input_arithmetic':{'q0_DP_hex':qf[0].hex(),'q3_DP_hex':qf[3].hex(),'q0_equals_q3_DP':qf[0]==qf[3], 'exact_q0_minus_q3':str(dec(qe[0]-qe[3])), 'exact_q_squared':str(exact),'exact_square_after_DP_momentum_formation':str(dec(sq([Fraction.from_float(x) for x in qf]))),'left_associative_DP_square':sq(qf),'factored_DP_square':(qf[0]-qf[3])*(qf[0]+qf[3])-qf[1]*qf[1]-qf[2]*qf[2]},
      'native':{'DP_Born_ordinary':dp[1],'DP_Born_stress':dp[2], 'QP_Born':[{'point':k[0],'variant':k[1],'value':v[0],'bad_term_invalid_divzero_overflow':v[1:]} for k,v in qp.items()], 'QP_stress_denominator':den[2,0][1], 'QP_stress_denominator_relative_error_vs_exact_input':str(relative(den[2,0][1],exact)), 'ordinary_QP_vs_DP_relative_error':str(relative(qp[1,0][0],dp[1][0])), 'stress_gluon_exchange_relative_error':str(relative(qp[2,1][0],qp[2,0][0])), 'stress_axis_permutation_relative_error':str(relative(qp[2,2][0],qp[2,0][0]))},
      'scope_limits':['No reducer orSymbolica wasloaded; allloopamplitudes/acceptancestatusremainhistorical.','QPmodelparametersareexactliftsofDPvalues, notindependentlyrecomputedUFOcouplings.','Sourceandscalararithmeticidentifya zero propagatordenominator; original fullwavefunctionfirstfailure wasnotinstrumented.','All48helicities andoneSO/unpolarized weights aretested; genericpolarized/multisplit-orderfallbacknotimplemented.'],
      'source_links':{'born_call':'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg/born_matrix.f:567','denominator':'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/Source/DHELAS/FFV1_2.f:20','host_born_assignment':'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg/loop_matrix.f:807','host_reference':'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg/loop_matrix.f:814','QP_repair_not_used':'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg/loop_matrix.f:2129'},
      'bindings':{'execution.json':sha(R/'execution.json'),'preparation.json':sha(R/'preparation.json'),'output.log':sha(R/'output.log'),str(points.relative_to(ROOT)):sha(points)}}
 assert relative(den[2,0][1],exact)<Decimal('1e-14')
 assert relative(qp[1,0][0],dp[1][0])<Decimal('1e-14')
 assert qp[2,0][0]==qp[2,1][0]
 assert relative(qp[2,2][0],qp[2,0][0])<Decimal('1e-14')
 dump('analysis.json',out)
files={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(R.rglob('*')) if p.is_file() and p.name!='evidence-index.json'}
dump('evidence-index.json',{'schema':1,'scope':'Born-only diagnostic and source arithmetic; not full-loop acceptance.','files':files})
print('PASS: DPNaN reproduced; six finite QPBorns; unchangedinputs, denominatororacle, symmetrychecks; indexed',len(files),'files')
