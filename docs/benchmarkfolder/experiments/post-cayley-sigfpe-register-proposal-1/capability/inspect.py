#!/usr/bin/env python3
"""Inspect an installed debugger without any target, inferior or attachment."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
HERE=ROOT/'.tirust-build/post-cayley-sigfpe-register-proposal-1/capability'
GDB=Path('/nix/store/2xrxnqpkwlr4smxwhg5x34s2nbs2rpi6-gdb-17.2/bin/gdb')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 assert not (HERE/'execution.json').exists()
 commands=[]
 before=sha(GDB.read_bytes())
 for name,tail in [('version',['--version']),('configuration',['--configuration']),('target-free',['-nx','-nh','--batch','--quiet','-x',str(HERE/'target-free.gdb')])]:
  argv=[str(GDB),*tail]
  started=time.monotonic()
  process=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=30)
  for stream,raw in [('stdout',process.stdout),('stderr',process.stderr)]:
   raw.decode('utf8');(HERE/(name+'.'+stream+'.log')).write_bytes(raw)
  commands.append({'label':name,'argv':argv,'cwd':str(ROOT),'returncode':process.returncode,
    'process_reaped':True,'seconds':time.monotonic()-started,'stdout_sha256':sha(process.stdout),'stderr_sha256':sha(process.stderr),
    'tool_path':str(GDB),'tool_resolved_path':str(GDB.resolve()),'tool_sha256':before})
  assert process.returncode==0,(name,process.returncode,process.stderr.decode())
 after=sha(GDB.read_bytes());assert after==before
 output=(HERE/'target-free.stdout.log').read_text()
 match=re.search(r'^TARGET_FREE_CAPABILITY_JSON (.*)$',output,re.M);assert match
 observed=json.loads(match[1]);assert observed['selected_inferior_pid']==0 and observed['no_target_file_loaded']
 table={}
 for line in output.splitlines():
  words=line.split()
  if words and words[0] in {'rip','xmm0','xmm1','mxcsr'}:table[words[0]]=line
 assert set(table)=={'rip','xmm0','xmm1','mxcsr'}
 assert re.search(r'^SIGFPE\s+Yes\s+Yes\s+No\s+',output,re.M)
 assert 'main' in output and '<PENDING>' in output
 result={'schema':'target-free-installed-gdb-capability-v1','commands':commands,'tool_hash_before':before,'tool_hash_after':after,
   'script_sha256':sha((HERE/'target-free.gdb').read_bytes()),'observed':observed,'x86_64_register_descriptors':table,
   'SIGFPE_stop_print_nopass_configuration_accepted':True,'pending_main_breakpoint_creation_accepted':True,
   'siginfo_without_inferior':re.search(r'^TARGET_FREE_SIGINFO (.*)$',output,re.M)[1],
   'artifact_target_loaded':False,'inferior_run_start_or_attach_executed':False,'native_replay_or_library_load_executed':False,
   'actual_register_access_siginfo_fields_main_hit_and_ptrace_qualified':False,
   'scope':'Target-free debugger startup/configuration/Python/architecture metadata only; actual target debugging remains untested and requires separate root admission.'}
 (HERE/'execution.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'gdb':str(GDB),'gdb_sha256':before,'version':observed['gdb_version'],'python_version':observed['python_version'],
                   'batch_Python_available':True,'inferior_pid':observed['selected_inferior_pid'],'target_loaded':False,'register_descriptors':sorted(table)},indent=2))
if __name__=='__main__':main()
