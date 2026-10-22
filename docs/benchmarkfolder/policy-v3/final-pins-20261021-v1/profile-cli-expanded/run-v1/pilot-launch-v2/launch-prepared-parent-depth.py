#!/usr/bin/env python3
"""Root-authorized two-cell pilot only; no whole-series queue."""
from pathlib import Path
import sys,os,json,hashlib,datetime,time,subprocess
ROOT=Path(__file__).resolve().parents[7]
assert (ROOT/'tools/ninjars_profile_cli_v3.py').is_file(), 'Incorrect repository root'
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
    spec=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/profile-cli-expanded-v1/specification.json'
    handoff=run/'root-handoff-pilot.json';operational=run/'root-pilot-operational-scope.json';tool=ROOT/'tools/ninjars_profile_cli_v3.py'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='17cbbf00a3c50fab7e2102fd53b02a194049a1c9'
    assert sha(spec)=='e24874d9ed1b5e694998c5ee08fb51d1721d55f2fa5bb3e405619d4fae3533ac'
    assert sha(handoff)=='678a4e0a012c707b52a75d318ab5ccc1555793decc1c1c6c086a6ff727cf2db1'
    assert sha(operational)=='1cbfc302aac964d2bd999c0f6795a9866257c12e41b8e520801d1e16fbd43ea7'
    assert sha(tool)=='8aed579e7bb3130651cf4fb585c5b701d60b35e2de2c7d79960301396882b20d'
    scope=json.loads(operational.read_text());assert scope['authorized_indices']==[0,1] and scope['global_timing_quiet_confirmed'] is True
    assert os.sched_getaffinity(0)=={207} and all(os.environ.get(n)=='1' for n in ownership.pair.THREADS)
    record={'scope':'Exactly QCD1 pilot indices0/1 only; no automatic remaining70','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'producer_sha256':sha(tool),'specification_sha256':sha(spec),'root_handoff_sha256':sha(handoff),'root_operational_sha256':sha(operational),'supervisor':ident(os.getpid()),'commands':[],'completed_indices':[],'status':'incomplete'}
    write(report/'launch.json',record);print(json.dumps(record),flush=True)
    try:
        with ownership.owned_runtime(report):
            for index in (0,1):
                command=['taskset','-c','207',sys.executable,'-B','-u',str(tool),'--spec',str(spec),'--index',str(index),'--handoff',str(handoff),'--execute-numerical']
                with (report/('index-%d.stdout.log'%index)).open('w') as stdout,(report/('index-%d.stderr.log'%index)).open('w') as stderr:
                    child=workers.subprocess.Popen(command,cwd=ROOT,env=dict(os.environ),stdout=stdout,stderr=stderr,start_new_session=True)
                    item={'index':index,'command':command,'process':ident(child.pid),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['commands'].append(item);write(report/'launch.json',record);print(json.dumps({'started':item}),flush=True)
                    item['returncode']=child.wait(timeout=1860);item['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(report/'launch.json',record)
                    if item['returncode']!=0:raise RuntimeError('Pilot CLI cell failed: '+str(index))
                    result=run/'qcd-1'/str(6 if index==0 else 8)/'0/result.json';assert json.loads(result.read_text())['complete'] is True
                    record['completed_indices'].append(index);write(report/'launch.json',record);print(json.dumps({'completed':index,'returncode':item['returncode']}),flush=True)
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(report/'result.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
