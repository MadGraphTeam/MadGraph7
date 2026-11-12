"""Pure replay of seven warmed timing cycles at one diagnosed physical point."""
from pathlib import Path
import collections,hashlib,json,math,statistics
IDS=(1,6,7,8,9);NAMES={1:'CutTools',6:'Ninja',7:'COLLIER',8:'NinjaRS',9:'TIRust'}
ROUTES=('Direct','MethodChoice','Tier106','Tier256','Tier512','Refused','unreported/reduction-error','attempted106','attempted256','attempted512')
def derive(root):
 root=Path(root);inputs={}
 def read(name):
  path=root/name;raw=path.read_bytes();inputs[str(Path(name).as_posix())]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 spec=read('specification.json');build=read('build.json');data=read('observations/result.json');terminal=read('supervision-run/terminal.json');after=read('observations/shared-after.json')
 assert terminal['status']=='observations-collected' and terminal['returncode']==0 and terminal['cleanup']['survivors']==[]
 assert after==build['shared'] and not (root/'observations/shared-integrity-failure.json').exists() and not list((root/'observations').glob('cleanup-failure-*.json'))
 assert hashlib.sha256((root/'first-point.dat').read_bytes()).hexdigest()==spec['point_sha256'] and spec['points']==1 and spec['membership_one_based']==[1]
 assert len(data['cycles'])==7 and [r['cycle'] for r in data['cycles']]==list(range(1,8));backends={}
 for backend in IDS:
  key=str(backend);record=data['lanes'][key];samples=[];usage=collections.Counter();route_sum=[0]*20
  for row in data['cycles']:
   assert row['order']==spec['cycles'][row['cycle']-1]['order'];receipt=read('observations/cycle-'+str(row['cycle'])+'.json');assert row==receipt
   entry=row['observations'].get(key)
   if entry is None:continue
   base='observations/lanes/'+key+'/raw/cycle-'+str(row['cycle']);obs=read(base+'/observation.json');routes=read(base+'/routes.json');comparison=read(base+'/comparison.json')
   assert obs==entry['observation'] and routes==entry['routes'] and comparison['passed'] in (True,False)
   raw=(root/(base+'/tirust_routes.dat')).read_bytes();inputs[base+'/tirust_routes.dat']=hashlib.sha256(raw).hexdigest();values=list(map(int,raw.split()));assert len(values)==66
   assert values[:6]==routes['header']==[1,row['cycle']+1,entry['loops'],1,0,0] and values[6:26]==routes['before'] and values[26:46]==routes['after'] and values[46:]==routes['delta']
   assert all(a>=b>=0 and d==a-b for a,b,d in zip(routes['after'],routes['before'],routes['delta']))
   assert sum(obs['backend_precision_counts'].values())==entry['loops'] and sum(obs['timed_status_counts'].values())==entry['loops']
   assert math.isclose(obs['seconds_per_point'],obs['seconds']/entry['loops'],rel_tol=1e-12) and len(obs['rows'])==2
   assert all(r['point']==1 and r['actual_backend']==backend and len(r['coefficients'])==4 for r in obs['rows'])
   usage.update(obs['backend_precision_counts']);route_sum=[a+b for a,b in zip(route_sum,routes['delta'])]
   samples.append({'cycle':row['cycle'],'position':entry['position'],'loops':entry['loops'],'evaluations':entry['loops'],'seconds':obs['seconds'],'seconds_per_point':obs['seconds_per_point'],'dispatch_precision_counts':obs['backend_precision_counts'],'route_delta':routes['delta'],'route_blocks':routes['blocks'],'raw':base,'comparison':comparison})
  checks=[]
  for item in data['checks'][key]:
   parts=Path(item['path']).parts;relative=Path('observations',*parts[parts.index('lanes'):]);check=read(relative);assert check==item['result'];checks.append({'path':str(relative),'result':check})
  regress=all(v['result']['passed'] for v in checks) if checks else None;target=all(v['result']['ordinary_target_passed'] for v in checks) if checks else None
  complete=record['status']=='complete' and len(samples)==7 and regress is True
  if complete:
   assert read('observations/lanes/'+key+'/STOP-BYE.json')['closed'] is True
   for phase in ('before','after'):
    maps=read('observations/lanes/'+key+'/maps-'+phase+'.json');assert set(maps['signed_tirust_dso'].values())=={build['dso_sha256']}
  times=[s['seconds_per_point'] for s in samples];median=statistics.median(times) if complete else None
  blocks={name:dict(zip(ROUTES,route_sum[i*10:(i+1)*10])) for i,name in enumerate(('primary','independent-check'))}
  local_tiers=any(route_sum[i]>0 for i in (2,3,4,7,8,9,12,13,14,17,18,19));unreported=any(route_sum[i]>0 for i in (6,16));qp=usage.get(key+':QP',0)
  if backend==9:
   note='DP-input dispatch with observed local tensor precision tiers' if local_tiers else 'DP-input dispatch; native-only tensor evaluation is not inferred from the amplitude status'
   if qp:note+='; genuine binary128-input QP amplitude dispatch also observed'
   precision={'dispatch_counts':dict(usage),'local_tensor_tiers_observed':local_tiers,'genuine_qp_amplitudes':qp,'native_dp_only_qualified':False if local_tiers or qp else None,'route_blocks':blocks,'attempt_totals_lower_bounds':unreported,'note':note,'fortran_coefficient_cache_hits_outside_route_scope':True}
  else:
   assert route_sum==[0]*20;precision={'dispatch_counts':dict(usage),'note':'Accepted MadLoop amplitude dispatch; TIRust route counters are unused','tirust_route_counters_zero':True}
  backends[key]={'backend':backend,'name':NAMES[backend],'status':record['status'],'timing_available':complete,'sample_count':len(samples),'samples':samples,'median_seconds_per_point':median,'mad_seconds_per_point':statistics.median(abs(t-median) for t in times) if complete else None,'minimum_seconds_per_point':min(times) if complete else None,'maximum_seconds_per_point':max(times) if complete else None,'initialization':record.get('initialization'),'regression_pass':regress,'ordinary_target_pass':target,'checks':checks,'precision':precision,'aggregate_dispatch_precision_counts':dict(usage),'aggregate_route_delta':route_sum}
 for key,lane in backends.items():
  ratios=[]
  if lane['timing_available'] and backends['8']['timing_available']:
   reference={s['cycle']:s['seconds_per_point'] for s in backends['8']['samples']};ratios=[{'cycle':s['cycle'],'ratio':s['seconds_per_point']/reference[s['cycle']]} for s in lane['samples']]
  lane['paired_cycle_ratios_to_8']=ratios;lane['median_paired_ratio_to_8']=statistics.median(r['ratio'] for r in ratios) if ratios else None
 return {'schema':'madloop-zggg-diagnosed-first-point-timing-summary-v1','scope':spec['scope'],'physical_points':1,'membership_one_based':[1],'point_sha256':spec['point_sha256'],'cycles':7,'case':spec['case'],'original_packet_index_sha256':spec['original_packet_index_sha256'],'original100_sha256':spec['original100_sha256'],'original10_sha256':spec['original10_sha256'],'final_prefix':build['prefix'],'final_dso_sha256':build['dso_sha256'],'qualification':build['qualification'],'executable_sha256':build['executable_sha256'],'compiler_flags':build['compiler_flags'],'position_counts':spec['position_counts'],'pairwise_precedence_counts':spec['pairwise_precedence_counts'],'backends':backends,'dispersion':'unscaled MAD of seven per-point warmed times','ratio_method':'median of matched within-cycle ratios to NinjaRS8 at the same single physical point','initialization_clock':'Fortran SYSTEM_CLOCK around bootstrap plus one complete steady same-point warm-up; excludes launch/SETPARA','route_clock_policy':'getter before T0/after T1, all route writes outside the timed numerical loop','shared_unchanged':True,'owned_cleanup_survivors':[],'physics_regression_pass':data['physics_regression_pass'],'performance_acceptance':False,'confidence_intervals':False,'cpu_reserved':False,'input_sha256':dict(sorted(inputs.items()))}
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',nargs='?',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();result=derive(args.root);path=args.root/'summary.json';path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(path)
