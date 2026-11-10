"""Pure clamp arithmetic on two CLOSED EW group5 binary128 snapshots."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication'
INPUTS={
 'trace.json':'03fbfd3795a468661162c0e574a7de6878065efbab79ca9d0f80e743f29e787f',
 'tensor-replay/selected-inputs.json':'6e13ab1cbfbd88e746c83bb725a547343330375acea741ce1caef2fd16822305',
 'tensor-replay/selected-inputs.dat':'a357490c15bc615869b3b11677788efb74d1529814056108aa2b57e2c06244e9',
 'ct-qp-transport/sources/CT_interface.f.original':'ac240d6ffc2f7e1a15a72e61a325a53d0acf776f65b9329e4e9518cd85f3da0a',
 'MadLoopParams.dat':'e145689b4e1b9fb5aeb464c9962be726dc1a302711a0a5889cc1a80f9b3558a1',
}
CHANNEL={(1,2):(3,),(1,3):(3,4),(1,4):(3,4,5),(2,3):(4,),(2,4):(4,5),(3,4):(5,)}
def need(ok,message):
 if not ok:raise ValueError(message)
def pow2(e):return F(2)**e
def rnd(value,bits=113):
 value=F(value)
 if not value:return value
 sign=-1 if value<0 else 1;v=abs(value);e=v.numerator.bit_length()-v.denominator.bit_length()
 if v<pow2(e):e-=1
 need(-16382<=e<=16383,'Outside normal qualified binary128 range')
 unit=pow2(e-(bits-1));scaled=v/unit;q,r=divmod(scaled.numerator,scaled.denominator)
 if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2):q+=1
 return sign*q*unit

def binary(text):return rnd(F(Decimal(text)))
def text(x):
 with localcontext() as ctx:
  ctx.prec=70
  return str(Decimal(x.numerator)/Decimal(x.denominator))
def dot(v):return v[0]**2-sum(x*x for x in v[1:])
def minksq(v):
 out=rnd(v[0]*v[0])
 for value in v[1:]:out=rnd(out-rnd(value*value))
 return out

def calculate():
 need(rnd(1+pow2(-113))==1 and rnd(1+3*pow2(-113))==1+pow2(-111),'Ties-to-even check failed')
 need(rnd(-1-3*pow2(-113))==-1-pow2(-111),'Negative ties-to-even check failed')
 for name,h in INPUTS.items():need(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==h,'Historical input changed: '+name)
 card=(BASE/'MadLoopParams.dat').read_text();need('#OSThres\n1.000000d-08' in card,'Frozen threshold changed')
 source=(BASE/'ct-qp-transport/sources/CT_interface.f.original').read_text().upper()
 need('ABS(DIFFSQ/REF_NORMALIZATION)' in source and 'REF_NORMALIZATION = (REF_NORMALIZATION/2.0E0_16)**2' in source,'Frozen clamp source changed')
 threshold=F.from_float(1e-8)
 trace=json.loads((BASE/'trace.json').read_text());selected=json.loads((BASE/'tensor-replay/selected-inputs.json').read_text())
 cases=[]
 for point in (1,2):
  raw=trace[str(point)];record=next(r for r in selected['records'] if r['point']==point and r['group']==5)
  original=raw['groups']['5'];need(all(original[k]==record[k] for k in ('n','rank','scale')) and all([[str(x) for x in row] for row in original[k]]==[[str(x) for x in row] for row in record[k]] for k in ('shifts','masses','invariants')),'Selected/captured request mismatch')
  q={int(row[0]):list(map(binary,row[1:])) for row in record['shifts']}
  masses={int(row[0]):binary(row[1]) for row in record['masses']};need(all(binary(row[2])==0 for row in record['masses']),'Complex mass outside this EW fixture')
  supplied={(int(i),int(j)):binary(x) for i,j,x in record['invariants']}
  external={int(row[0]):list(map(binary,row[1:])) for row in raw['MP_P']}
  dp_external={int(row[0]):list(map(binary,row[1:])) for row in raw['P']}
  mp_mass=binary(raw['mt'][1]);mp_mass_sq=rnd(mp_mass*mp_mass)
  shells=[]
  for leg in range(1,6):
   target=mp_mass_sq if leg in (3,4) else F(0)
   shells.append({'leg':leg,'kind':'top/antitop' if leg in (3,4) else 'massless',
    'exact_fixed_DP_input_shell_residual':text(dot(dp_external[leg])-target),
    'exact_fixed_repaired_MP_input_shell_residual':text(dot(external[leg])-target),
    'rounded113_repaired_shell_residual':text(minksq(external[leg])-target)})
  rows=[]
  for i in range(1,5):
   for j in range(i+1,5):
    exact_delta=[a-b for a,b in zip(q[i],q[j])];delta=[rnd(x) for x in exact_delta]
    ds=minksq(delta);ref=rnd(rnd((abs(q[i][0])+abs(q[j][0]))/2)**2)
    ratio=rnd(abs(ds/ref)) if ref else None
    smat=rnd(rnd(ds-masses[i])-masses[j]);hits=[]
    for k in (i,j):
     if masses[k] and rnd(abs(rnd(ds-masses[k])/masses[k]))<threshold:
      smat=-masses[j if k==i else i];hits.append('mass_'+str(k))
    if ref and ratio<threshold:smat=-rnd(masses[i]+masses[j]);hits.append('massless_zero')
    predicted=rnd(rnd(smat+masses[i])+masses[j])
    euclidean=sum(x*x for x in exact_delta);maximum=max(abs(x) for x in exact_delta)**2
    channel=CHANNEL[i,j];expected=[sum(external[leg][mu] for leg in channel) for mu in range(4)]
    defects=[a-b for a,b in zip(exact_delta,expected)]
    rows.append({'pair':[i,j],'shifts':[[text(x) for x in q[k]] for k in (i,j)],
     'delta_vector_exact':[text(x) for x in exact_delta],
     'exact_DIFFSQ_of_fixed_binary_inputs':text(dot(exact_delta)),
     'source_order_rounded113_DIFFSQ':text(ds),'original_energy_denominator_rounded113':text(ref),
     'current_massless_clamp_ratio':text(ratio) if ratio is not None else None,
     'delta_euclidean_norm_squared_exact':text(euclidean),'delta_max_component_squared_exact':text(maximum),
     'absolute_exact_DiffSq_over_delta_euclidean':text(abs(dot(exact_delta))/euclidean) if euclidean else None,
     'absolute_exact_DiffSq_over_delta_max_component_squared':text(abs(dot(exact_delta))/maximum) if maximum else None,
     'massless_clamp_fires':bool(ref and ratio<threshold),'ordered_clamp_branches':hits,
     'mass_i_comparison_abs_rounded113':text(abs(rnd(ds-masses[i]))),
     'mass_j_comparison_abs_rounded113':text(abs(rnd(ds-masses[j]))),
     'difference_euclidean_ratio_below_existing_threshold':bool(euclidean and abs(dot(exact_delta))/euclidean<threshold),
     'centered_pair_energy_denominator_exact':text(exact_delta[0]**2/4),
     'centered_pair_energy_denominator_scope':'Exact common time-component translation to center this pair; no new rounded or evaluated input.',
     'source_prediction_invariant':text(predicted),'supplied_invariant':text(supplied[i,j]),
     'source_prediction_matches_captured_invariant':predicted==supplied[i,j],
     'physical_channel_correspondence_legs':list(channel),
     'physical_channel_correspondence':' + '.join('p'+str(k) for k in channel),
     'delta_minus_external_sum_max_abs':text(max(map(abs,defects))),
     'delta_minus_external_sum_exact':[text(x) for x in defects],
     'channel_bit_exact':not any(defects),
     'provenance_scope':'Numerical correspondence to captured repaired external MP_P; no archived symbolic group routing is assumed exact when defects are nonzero.'})
  need(all(r['source_prediction_matches_captured_invariant'] for r in rows),'Source arithmetic prediction differs from capture')
  cases.append({'point':point,'group':5,'n':4,'rank':3,'native_status':record['historical_status'],
   'native_error':record['historical_error'],'external_leg_shells':shells,
   'loop_mass_square_minus_repaired_model_mass_square':text(masses[1]-mp_mass_sq),'pairs':rows})
 return {'schema':'closed-ew-clamp-classification-v1','source_files':{str((BASE/k).relative_to(ROOT)):v for k,v in INPUTS.items()},
  'source_order_arithmetic':'Decode36-significant-digit round-trip values to normal binary128, then round113 nearest-even after every real subtraction/product/division/addition in the displayed helper; no reducer called.',
  'OSThres_DP_value_promoted_exact':text(threshold),'native_executed':False,'production_change':False,
  'scope':'New scale/classification diagnostics of closed points1/2 group5 only; not an error oracle or proposed replacement cutoff.',
  'cases':cases,
  'limits':['No threshold, input, scalar call or backend guard changes.',
   'Euclidean and max-component difference scales are diagnostic alternatives only, not a qualified on-shell classifier or Lorentz-invariant policy.',
   'Captured external-channel numerical defects remain explicit; generated symbolic topology provenance was not reconstructed.',
   'Exact rational shell residuals describe the captured binary inputs, not mathematical exact kinematics.']}

def main():
 value=calculate();raw=json.dumps(value,sort_keys=True,indent=2)+'\n'
 if '--write' in sys.argv:(HERE/'summary.json').write_text(raw)
 elif '--check' in sys.argv:need((HERE/'summary.json').read_text()==raw,'Summary differs')
 else:print(raw,end='')
 if '--write' in sys.argv or '--check' in sys.argv:
  print(json.dumps([{'point':r['point'],'group':5,'clamps':[(p['pair'],p['ordered_clamp_branches']) for p in r['pairs']],'all_source_predictions_match':all(p['source_prediction_matches_captured_invariant'] for p in r['pairs'])} for r in value['cases']],sort_keys=True))
if __name__=='__main__':main()
