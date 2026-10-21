"""Independent retained-byte timing/uncertainty/custody audit; no engines."""
from pathlib import Path
import collections, datetime, hashlib, json, math, random, statistics, struct
R=Path.cwd();B=R/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing';D=B/'runs/run-v1'
FILES={}
def req(x,w):
 if not x:raise ValueError(w)
def read(p):
 p=Path(p);p=p if p.is_absolute() else R/p;raw=p.read_bytes();FILES[str(p.relative_to(R))]=hashlib.sha256(raw).hexdigest();return raw
def load(p):return json.loads(read(p))
def check(b):
 raw=read(b['path']);req(hashlib.sha256(raw).hexdigest()==b['sha256'] and ('bytes' not in b or len(raw)==b['bytes']),'bound bytes');return raw
s=load(D/'input-spec.json');v=load(D/'result.json');protocol=load(D/'protocol.json');terminal=load(B/'launch-v1/result.json');launch=load(B/'launch-v1/launch.json');summary=load(B/'terminal-summary-v1.json')
cids=['qcd-1','ew-1','qcd-2','ew-2','supplement-gg-h','supplement-gg-hh'];req(list(v['cases'])==cids and s['configuration']['points_per_case']==10,'exact6/N10');req(s['repetitions']==7 and s['minimum_seconds']==.5,'sevenpair calibration target');req(v['old_full8_runtime_gate_required_for_completion'] is False,'superseded heavy gate')
req(terminal['status']=='complete' and terminal['returncode']==0 and terminal['controller_absent'] is True and terminal['controller_group_absent'] is True and terminal['failure'] is None and not terminal['interruptions'],'recorded controller terminal')
req(terminal['controller_identity']==launch['controller_identity'] and terminal['terminal_custody']['spawned_groups']==36 and terminal['terminal_custody']['all_reaped_exit0_and_absent'] is True,'recorded36 childgroups')
check(terminal['terminal_custody']['ledger']);check(terminal['terminal_custody']['status']);check(launch['specification']);check(launch['root_handoff']);check(launch['retained_handoff']);req(launch['root_handoff']['sha256']==launch['retained_handoff']['sha256'],'exactroot handoff')
req(launch['worker_cpu']==207 and set(launch['thread_environment'].values())=={'1'} and len(launch['thread_environment'])==4 and launch['automatic_retry'] is False,'thread/cpu/no retry')
starts={};reaps={}
for e in read(D/'process-ledger.jsonl').decode().splitlines():
 e=json.loads(e);k=tuple(e[x] for x in ['pid','pgid','start_ticks']);req(all(type(x)is int and x>0 for x in k) and k[0]==k[1],'exactowned identity')
 if e['event']=='spawn':req(k not in starts,'unique spawn');starts[k]=e
 else:req(e['event']=='reaped' and k in starts and k not in reaps and e['returncode']==0 and e['reaped'] is True,'successful matching reap');reaps[k]=e
req(len(starts)==36 and starts.keys()==reaps.keys(),'complete36 custody')
events=[json.loads(x) for x in read(D/'dispatch.jsonl').decode().splitlines()];req([x['index'] for x in events]==list(range(len(events))),'dispatch indices')
contexts=0
for e in events:
 if e['phase'] not in ('calibration','measurement'):continue
 c=json.loads(check(e['source_checkpoint']));inventory=json.loads(check(c['observed_inventory']));req(all(c[k]==e[k] for k in ['index','case','phase','event','source','backend','repetition']),'source context ties event');req(inventory['production_sources']==s['production_sources'] and c['numerical_revision']==s['candidate_cohort']['numerical_revision'] and c['generator_identity_sha256']==s['candidate_cohort']['generator_identity_sha256'],'observed frozen source');contexts+=1
req(contexts==192,'192 before/after guards')
case_results={};all_durations=[];rawbits=coeffs=0;measurement_status=collections.Counter();calibration_status=collections.Counter()
for cid in cids:
 series={};records={}
 for role,tool in [('baseline',6),('candidate',8)]:
  a=[]
  for rep in ['calibration',*range(7)]:
   rec=load(D/'data'/f'{cid}-{role}-{tool}-{rep}.json');req(rec['cpu']==207 and rec['backend']==tool and rec['case']==cid and rec['repetition']==rep,'actualselected record');req(rec['points_sha256']==s['points'][cid]['sha256'] and rec['paired_baseline_candidate'] is True,'matchedpointpair')
   raw={name:check(binding) for name,binding in rec['raw_capture'].items()};times=[float(x) for x in raw['timing.dat'].split()];req(times==[rec['cold_seconds'],rec['seconds']] and rec['seconds_per_point']==times[1]/(10*rec['loops']),'raw time normalization')
   rawrows=[x.split() for x in raw['values.dat'].splitlines()];req(len(rawrows)==len(rec['rows'])==20,'10points/2orders')
   for fields,row in zip(rawrows,rec['rows']):
    req(list(map(int,fields[:3]))==[row['point'],row['order'],row['status']],'raw rowidentity')
    actual=[float(x.replace(b'D',b'E')) for x in fields[3:]]
    for x,y in zip(actual,row['coefficients']+[row['accuracy']]):req(struct.pack('>d',x)==struct.pack('>d',y),'raw numericbits');rawbits+=1
   timed={str(int(x)):int(y) for x,y in [line.split() for line in raw['timed_status.dat'].splitlines()]};req(timed==rec['timed_status_counts'] and sum(timed.values())==10*rec['loops'],'all actual eval statuses');req(set(timed)=={str(210+tool)} and set(rec['status_counts'])=={str(210+tool)},'actualDP status')
   if rep=='calibration':req(rec['loops']==1,'cal not timing repetition');calibration_status.update(timed)
   else:a.append(rec['seconds_per_point']);all_durations.append(rec['seconds']);measurement_status.update(timed)
   records[role,rep]=rec
  series[role]=a
 stats=v['cases'][cid]['timing'];rng=random.Random(s['configuration']['seed']);boot=[]
 for _ in range(10000):
  ix=rng.choices(range(7),k=7);boot.append(statistics.median(series['candidate'][j] for j in ix)/statistics.median(series['baseline'][j] for j in ix))
 boot.sort();ci=[boot[249],boot[9749]];ratio=statistics.median(series['candidate'])/statistics.median(series['baseline']);req(stats['runtime_ratio']==ratio and stats['ratio_confidence_interval_95']==ci and stats['bootstrap']=={'method':'paired percentile','resamples':10000,'seed':s['configuration']['seed']},'independent paired bootstrap')
 expected='passed' if ci[1]<=1.05 else 'failed' if ci[0]>1.05 else 'unresolved';req(stats['status']==expected and stats['passed']==(expected=='passed'),'uncertainty gate')
 for role in ['baseline','candidate']:
  x=series[role];z=stats['measurements'][role];med=statistics.median(x);req(z['samples']==x and z['median']==med and z['minimum']==min(x) and z['maximum']==max(x) and z['median_absolute_deviation']==statistics.median(abs(t-med) for t in x),'dispersion/sample math');req(math.isclose(z['stdev'],statistics.stdev(x),rel_tol=1e-15),'sample standard deviation')
 for rep in range(7):
  left,right=records['baseline',rep],records['candidate',rep];req(all(left[k]==right[k] for k in ['points_sha256','parameter_card_sha256','shared_numerical_settings','answer_dimensions']),'same physics/settings');scale={}
  for row in left['rows']+right['rows']:scale[row['point']]=max(scale.get(row['point'],0.),*map(abs,row['coefficients']))
  for a,b in zip(left['rows'],right['rows']):
   req((a['point'],a['order'])==(b['point'],b['order']),'pair order')
   for x,y in zip(a['coefficients'],b['coefficients']):req(abs(x-y)<=1e-7*max(abs(x),abs(y))+1e-10*scale[a['point']],'paircoeff agreement');coeffs+=1
 req(v['cases'][cid]['actual_pairs']==7 and v['cases'][cid]['actual_points']==10,'actual casecounts');case_results[cid]={'ratio':ratio,'paired_bootstrap_interval95':ci,'reference1p05_status':expected,'actual_pairs':7,'unique_points':10}
req(len(all_durations)==84 and min(all_durations)==summary['actual_measured_command_seconds_range'][0] and max(all_durations)==summary['actual_measured_command_seconds_range'][1] and sum(all_durations)==summary['total_actual_measured_seconds'],'84 actual duration observations')
req(summary['unique_process_points']==60 and summary['whole_goal_achieved_claimed'] is False and summary['minimum_seconds_configuration']==.5,'scoped summary')
result={'schema':'independent-light-six-actual-timing-audit-v1','status':'PASS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reader':{'path':str(Path(__file__).relative_to(R)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'cases':case_results,'unique_process_points':60,'actual_points_per_case':10,'calibration_commands':12,'actual_measured_commands':84,'actual_recorded_owned_reaps':36,'recorded_controller_absent':True,'before_after_source_checkpoints':contexts,'raw_float64_bit_checks':rawbits,'actual_pair_coefficient_checks':coeffs,'actual_measured_duration_range_seconds':[min(all_durations),max(all_durations)],'actual_measured_seconds_sum':sum(all_durations),'duration_semantics':'0.5s is calibration-derived target, not an achieved minimum; loops selected once with no retry. Actual short observations and seven-pair empirical confidence intervals retained.','timed_status_counts':dict(measurement_status),'calibration_status_counts':dict(calibration_status),'inputs_sha256':dict(sorted(FILES.items())),'scope':'Six light processes/first10 frozen points only; all source/runtime/raw/dispatch validated separately by frozen retained authority; no withdrawn full8 timing or whole-goal claim','numerical_calls':0,'compiler_invocations':0,'tests_run':0,'process_queries':0}
out=Path(__file__).with_name('light-six-actual-timing-audit-v1.json');req(not out.exists(),'preserve audit');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:x for k,x in result.items() if k!='inputs_sha256'},indent=2));print('sha256',hashlib.sha256(out.read_bytes()).hexdigest())
