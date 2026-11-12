"""Pure route-counter replay. DP-input tensor rescues never count as genuine QP."""
from pathlib import Path
import json
FIELDS=('Direct','MethodChoice','Tier106_DP_input','Tier256_DP_input','Tier512_DP_input','Refused','unreported_reduction_error','attempted106','attempted256','attempted512')
def route_summary(path,events_path):
 snapshots=[];malformed=[]
 for index,line in enumerate(Path(path).read_text().splitlines(),1):
  try:snapshots.append(json.loads(line))
  except json.JSONDecodeError:malformed.append(index)
 valid=[s for s in snapshots if s.get('getter_available') and s.get('query_status')==0 and len(s.get('counts',[]))==20 and all(isinstance(x,int) and x>=0 for x in s['counts'])]
 monotone=all(all(b>=a for a,b in zip(x['counts'],y['counts'])) for x,y in zip(valid,valid[1:]))
 events=[json.loads(line) for line in Path(events_path).read_text().splitlines()]
 calls={'primary':0,'check':0};qp_calls=0
 for row in events:
  if row.get('operation')=='dp':calls['check' if row['check'] else 'primary']+=1
  elif row.get('operation')=='qp':qp_calls+=1
 result={'schema':'tirust-prototype-dp-route-observer-v1','snapshots':len(snapshots),'valid_snapshots':len(valid),'malformed_lines':malformed,'query_statuses':sorted({s.get('query_status') for s in snapshots}),'all_getters_available':all(s.get('getter_available') for s in snapshots),'counter_monotonic':monotone,'field_order_per_direction':list(FIELDS),'dp_cabi_calls':calls,'genuine_binary128_input_qp_cabi_calls':qp_calls,'precision_meaning':{'DP_API':'binary64 kinematics and DP numerator, potentially using local tensor precision tiers','106':'double-double tensor rescue from binary64 inputs','256':'256bit tensor rescue from binary64 inputs','512':'512bit tensor rescue from binary64 inputs','QP_API':'genuine113bit Fortran inputs transported losslessly; original256bit branch'},'native_dp_only_cannot_be_inferred_from_amplitude_dispatch_status':True,'scientific_timing_claim':False,'performance_eligible':False}
 if valid:
  last=valid[-1];result['final_snapshot_reason']=last['reason'];result['final_snapshot_sequence']=last['source_sequence'];result['directions']={}
  for index,name in enumerate(('primary','check')):
   values=last['counts'][index*10:(index+1)*10];counters=dict(zip(FIELDS,values));total=sum(values[:7]);result['directions'][name]={'counters':counters,'completed_full_dp_api_requests_reaching_counter':total,'observed_dp_api_calls':calls[name],'counter_total_matches_dp_api_calls':total==calls[name],'attempted_tier_counts_are_lower_bounds':values[6]>0,'accepted_dp_input_tensor_rescues':sum(values[2:5])}
  result['any_local_dp_input_tensor_rescue']=any(x['accepted_dp_input_tensor_rescues'] for x in result['directions'].values())
 result['counter_observation_complete']=bool(valid) and not malformed and len(valid)==len(snapshots) and monotone and all(x['counter_total_matches_dp_api_calls'] for x in result.get('directions',{}).values())
 return result
