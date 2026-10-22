#!/usr/bin/env python3
"""Root-authorized remaining CLI cells2..71; retain completed pilot0/1."""
from pathlib import Path
import sys,os,json,hashlib,datetime,time,subprocess
ROOT=Path.cwd()
assert (ROOT/'tools/ninjars_profile_cli_v3.py').is_file() and (ROOT/'bin/madgraph').is_file(), 'Incorrect repository root'
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_profile_v2 as ownership
import ninjars_benchmark_workers as workers

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ident(pid):
    fields=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),session=int(fields[3]))
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def main():
    report=Path(__file__).resolve().parent;run=report.parent
    spec=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/profile-cli-expanded-plan-v4/specification.json'
    handoff=run/'root-handoff-continuation.json';operational=run/'root-continuation-operational-scope.json';tool=ROOT/'tools/ninjars_profile_cli_v3.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='464b441fe04d3cb5b3da7708897fa831e32c55eb'
    assert sha(spec)=='43179488a3fac72a88a45f95717c82b11f71e5a91b1697af7e084b32795a5492'
    assert sha(handoff)=='88e8031ff216b146ae7bd8d29cbad9ec1be564bab0475e4d0504bae975b98b73'
    assert sha(operational)=='8fa0adc874581580f02d8209d37491af96f8def6b482ecab6e6ce7b2a05f16a0'
    assert sha(tool)=='b0681842604b651eac164448f0ca5b19cc61ae02c16704ceb4e9e926516c8ff7'
    scope=json.loads(operational.read_text());assert scope['authorized_indices']==list(range(2,72)) and scope['global_timing_quiet_confirmed'] is True
    pilot=run/'pilot-launch-v1/terminal-cleanup-and-validation.json'
    assert sha(pilot)=='3706d9dc9060265f11d9bb09423b8ee80a387cf094e06b7a807cfa19d4533928'
    assert scope['completed_pilot_indices']==[0,1] and scope['pilot_terminal_validation_sha256']==sha(pilot)
    specification=json.loads(spec.read_text());assert len(specification['schedule'])==72
    assert os.sched_getaffinity(0)=={207} and all(os.environ.get(n)=='1' for n in ownership.pair.THREADS)
    record={'scope':'Exactly remaining declared cells2..71; completed pilot0/1 preserved','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'producer_sha256':sha(tool),'specification_sha256':sha(spec),'root_handoff_sha256':sha(handoff),'root_operational_sha256':sha(operational),'supervisor':ident(os.getpid()),'commands':[],'completed_indices':[],'status':'incomplete'}
    write(report/'launch.json',record);print(json.dumps(record),flush=True)
    try:
        with ownership.owned_runtime(report):
            for index in range(2,72):
                declared=specification['schedule'][index];started=time.monotonic()
                command=['taskset','-c','207',sys.executable,'-B','-u',str(tool),'--spec',str(spec),'--index',str(index),'--handoff',str(handoff),'--execute-numerical']
                with (report/('index-%d.stdout.log'%index)).open('w') as stdout,(report/('index-%d.stderr.log'%index)).open('w') as stderr:
                    child=workers.subprocess.Popen(command,cwd=ROOT,env=dict(os.environ),stdout=stdout,stderr=stderr,start_new_session=True)
                    item={'index':index,'declared_item':declared,'command':command,'process':ident(child.pid),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['commands'].append(item);write(report/'launch.json',record);print(json.dumps({'started':item}),flush=True)
                    item['returncode']=child.wait(timeout=1860);item['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();item['elapsed_seconds']=time.monotonic()-started;write(report/'launch.json',record)
                    if item['returncode']!=0:raise RuntimeError('Continuation CLI cell failed: '+str(index))
                    result=run/declared['case']/str(declared['backend'])/str(declared['repetition'])/'result.json';assert json.loads(result.read_text())['complete'] is True
                    record['completed_indices'].append(index);write(report/'launch.json',record);print(json.dumps({'completed':index,'returncode':item['returncode']}),flush=True)
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(report/'result.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
