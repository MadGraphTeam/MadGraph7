#!/usr/bin/env python3
"""One-shot final core generation supervisor. No retries or numerical calls."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path.cwd()
HERE=ROOT/'.benchmark-work/policy-v2/final-core-generation-launch-v1'
HEAD='b0ea105a04afa34304d437d2956e4d4672360670'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();f=raw[raw.rindex(')')+2:].split()
  return {'pid':pid,'ppid':int(f[1]),'pgid':int(f[2]),'state':f[0],'start_ticks':int(f[19]),'cpu':int(f[36]),'cpu_seconds':(int(f[11])+int(f[12]))/os.sysconf('SC_CLK_TCK'),'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
 except (FileNotFoundError,ProcessLookupError):return None
def descendants(root):
 ps={}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdecimal():continue
  d=snap(int(proc.name))
  if d:ps[d['pid']]=d
 selected={root};changed=True
 while changed:
  added={pid for pid,d in ps.items() if d['ppid'] in selected}
  changed=not added.issubset(selected);selected|=added
 return [ps[pid] for pid in sorted(selected) if pid in ps]
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==HEAD
command=['bash','docs/benchmarkfolder/policy-v2/final-core-generation-v1/commands.sh']
env=os.environ.copy();env['GENERATION_CPUS']='180,181,182,183,184,185,186,187'
log=HERE/'controller.log'
with log.open('xb') as stream:
 child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={'schema':'one-shot-final-core-generation-launch-v1','utc':stamp(),'root_explicitly_authorized':True,'frozen_head':HEAD,'supervisor':snap(os.getpid()),'child':snap(child.pid),'argv':command,'generation_cpus':env['GENERATION_CPUS'],'freeze_sha256':digest(ROOT/'docs/benchmarkfolder/policy-v2/final-core-generation-v1/freeze.json'),'log':str(log.relative_to(ROOT)),'retry_policy':'None; preserve any failed attempt; root reviews before a new namespace.'}
 (HERE/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
 with (HERE/'process-history.jsonl').open('x') as history:
  while child.poll() is None:
   history.write(json.dumps({'utc':stamp(),'processes':descendants(child.pid)})+'\n');history.flush();time.sleep(5)
 result={'schema':'one-shot-final-core-generation-result-v1','utc':stamp(),'child_pid':child.pid,'exit_code':child.returncode,'loop_amplitude_evaluations':0,'automatic_retries':0,'head_after':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'log_sha256':digest(log)}
 (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 sys.exit(child.returncode)
