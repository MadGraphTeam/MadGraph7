"""Pure replay of C ABI phase events; no numerical library imports."""
from pathlib import Path
import collections,json,sys

def summarize(path):
 rows=[]; malformed=[]
 for index,line in enumerate(Path(path).read_text().splitlines(),1):
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: malformed.append(index)
 pending={};groups={};first=None;last=None;completed=0;calls=[]
 for row in rows:
  identity=(row['pid'],row['sequence'])
  if row['event']=='begin':
   if identity in pending: raise ValueError('Duplicate open call')
   pending[identity]=row
   first=row['monotonic_ns'] if first is None else min(first,row['monotonic_ns'])
  else:
   begin=pending.pop(identity)
   calls.append({**begin,**{k:row[k] for k in ('status','wall_ns','thread_cpu_ns')}})
   key=tuple(begin[k] for k in ('operation','n','rank','poles','finite','check','cache'))
   group=groups.setdefault(key,dict(zip(('operation','n','rank','poles','finite','check','cache'),key))|{'completed_calls':0,'status_counts':{},'wall_seconds':0.,'thread_cpu_seconds':0.,'max_call_seconds':0.})
   group['completed_calls']+=1;completed+=1
   status=str(row['status']);group['status_counts'][status]=group['status_counts'].get(status,0)+1
   seconds=row['wall_ns']*1e-9;group['wall_seconds']+=seconds;group['thread_cpu_seconds']+=row['thread_cpu_ns']*1e-9
   group['max_call_seconds']=max(group['max_call_seconds'],seconds)
   last=row['monotonic_ns'] if last is None else max(last,row['monotonic_ns'])
 runs=[];run=None
 for call in calls:
  if call['operation'] not in ('dp','qp'):
   run=None;continue
  if run is None or run['operation']!=call['operation']:
   run={'operation':call['operation'],'first_sequence':call['sequence'],'calls':0,'wall_seconds':0.,'first_failed_sequence':None,'wall_after_first_failed_return_seconds':0.,'calls_after_first_failed_return':0};runs.append(run)
  run['last_sequence']=call['sequence'];run['calls']+=1;run['wall_seconds']+=call['wall_ns']*1e-9
  if run['first_failed_sequence'] is not None:
   run['wall_after_first_failed_return_seconds']+=call['wall_ns']*1e-9;run['calls_after_first_failed_return']+=1
  elif call['status']!=0:run['first_failed_sequence']=call['sequence']
 segments=[];segment=None
 for call in calls:
  if call['operation']=='clear' or segment is None:
   segment={'first_sequence':call['sequence'],'clear_sequence':call['sequence'] if call['operation']=='clear' else None,'first_dp_failure_sequence':None,'first_qp_sequence':None,'dp_calls_after_first_failed_return_before_qp':0,'dp_wall_after_first_failed_return_before_qp_seconds':0.};segments.append(segment)
  if call['operation']=='qp' and segment['first_qp_sequence'] is None:segment['first_qp_sequence']=call['sequence']
  if call['operation']=='dp' and segment['first_qp_sequence'] is None:
   if segment['first_dp_failure_sequence'] is not None:
    segment['dp_calls_after_first_failed_return_before_qp']+=1;segment['dp_wall_after_first_failed_return_before_qp_seconds']+=call['wall_ns']*1e-9
   elif call['status']!=0:segment['first_dp_failure_sequence']=call['sequence']
 duplicates={}
 for call in calls:
  fingerprint=call.get('input_fingerprint_fnv128')
  if not fingerprint:continue
  key=(call['operation'],fingerprint)
  item=duplicates.setdefault(key,{'operation':call['operation'],'input_fingerprint_fnv128':fingerprint,'n':call['n'],'rank':call['rank'],'check':call['check'],'cache':call['cache'],'finite':call['finite'],'calls':0,'status_counts':{},'wall_seconds':0.,'repeated_call_wall_seconds':0.,'sequences':[]})
  if item['calls']:item['repeated_call_wall_seconds']+=call['wall_ns']*1e-9
  item['calls']+=1;item['wall_seconds']+=call['wall_ns']*1e-9;item['sequences'].append(call['sequence'])
  status=str(call['status']);item['status_counts'][status]=item['status_counts'].get(status,0)+1
 repeated=[d for d in duplicates.values() if d['calls']>1]
 return {'cache_clear_segments':segments,'fingerprint_scheme':'FNV-1a128 with length-delimited raw input sections, raw integer modes/dimensions and output length; duplicate candidates only, never a numerical cache key','duplicate_candidate_groups':sorted(repeated,key=lambda x:-x['repeated_call_wall_seconds']),'reduction_runs':runs,'scheme':'tirust-private-cabi-phase-profile-v1','events':len(rows),'completed_calls':completed,'malformed_lines':malformed,'interval_seconds':(last-first)*1e-9 if last is not None else None,'completed_bridge_seconds':sum(g['wall_seconds'] for g in groups.values()),'completed_bridge_thread_cpu_seconds':sum(g['thread_cpu_seconds'] for g in groups.values()),'groups':sorted(groups.values(),key=lambda x:-x['wall_seconds']),'incomplete_calls':list(pending.values()),'interpretation':'Instrumented diagnostic, not a performance benchmark. Begin records retain calls interrupted by timeout; completed durations exclude the tail of those unfinished calls.'}

if __name__=='__main__':
 value=summarize(sys.argv[1]);text=json.dumps(value,indent=2)+'\n'
 if len(sys.argv)>2:Path(sys.argv[2]).write_text(text)
 else:print(text,end='')
