"""Replay final NinjaRS rank-n+2 inputs through the unchanged public C ABI."""
from pathlib import Path
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
DATA = ROOT / '.ninjars-build/upstream-final-20261021/ninjars/crates/ninjars-symbolica/tests/rank_n2_data'
sys.path.insert(0,str(ROOT / 'tools'))
import ninjars_precision_v2 as precision


def bound(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


requests, references, inputs = [], [], {}
for n in [2,3,4]:
    case_name=f'rn2-int-complex-n{n}-r{n+2}'
    case_path=DATA/(case_name+'.json')
    case=json.loads(case_path.read_text())
    inputs[case_path.name]=bound(case_path)
    reference_sets={}
    for prefix in ['red','hpref']:
        path=DATA/(prefix+'-'+case_name+'.json')
        data=json.loads(path.read_text())
        inputs[path.name]=bound(path)
        reference_sets[prefix]={row['id']:row for row in data['lanes']}
    for lane in case['lanes']:
        assert not lane.get('mu2_tensors')
        for mode in [0,1,2]:
            flatten=lambda key:[value for row in lane[key] for value in row]
            row={'record':len(requests),'case':case_name,'lane':lane['id'],'mode':mode,
                 'n':n,'rank':n+2,'coefficients':flatten('tensor'),'s_matrix':flatten('s_matrix'),
                 'momenta':flatten('shifts'),'masses_squared':flatten('masses_squared'),
                 'scale_squared':[lane['mu_squared']]}
            if mode==2:
                for key in ['coefficients','s_matrix','momenta','masses_squared','scale_squared']:
                    row[key]=[format(float(value),'.39e') for value in row[key]]
            requests.append(row)
            references.append({prefix:rows[lane['id']] for prefix,rows in reference_sets.items()})
assert len(requests)==27
(REPORT/'high-rank-requests.json').write_text(json.dumps(requests,indent=2)+'\n')
(REPORT/'high-rank-references.json').write_text(json.dumps(references,indent=2)+'\n')
(REPORT/'high-rank-requests.txt').write_text(precision.encode(requests))
command=['taskset','-c','190',str(REPORT/'physical-driver')]
start=time.monotonic()
with (REPORT/'high-rank-requests.txt').open() as inp,(REPORT/'high-rank.stdout').open('x') as out,(REPORT/'high-rank.stderr').open('x') as err:
    p=subprocess.Popen(command,cwd=ROOT,stdin=inp,stdout=out,stderr=err,start_new_session=True)
    ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split(')',1)[1].split()[19])
    (REPORT/'high-rank-active.json').write_text(json.dumps({'pid':p.pid,'process_group_id':p.pid,'start_ticks':ticks,'command':command},indent=2)+'\n')
    print('HIGH_RANK_PID',p.pid,'START_TICKS',ticks,flush=True)
    try:
        code=p.wait(timeout=1800)
    except BaseException:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
outputs=[json.loads(line.split('NINJARS_INSTALLED ',1)[1]) for line in (REPORT/'high-rank.stdout').read_text().splitlines() if line.startswith('NINJARS_INSTALLED ')]
checks,failures=[],[]
for request,reference,actual in zip(requests,references,outputs):
    good=actual['record']==request['record'] and actual['mode']==request['mode'] and actual['status']==0 and len(actual['output'])==8
    errors={}
    if good:
        for name,row in reference.items():
            values=row.get('laurent_decimal',row)
            errors[name]=[]
            for index,key in enumerate(['eps0','eps_m1','eps_m2']):
                x=complex(*map(float,actual['output'][2*index:2*index+2]))
                y=complex(*map(float,values[key]))
                error=abs(x-y)/max(1.,abs(y))
                errors[name].append(error)
                good &= math.isfinite(error) and error<=1e-9
    check={'record':request['record'],'case':request['case'],'lane':request['lane'],
           'mode':request['mode'],'status':actual['status'],'complete_laurent_scaled_errors':errors,
           'tolerance':1e-9,'passed':good,'rational_compared':False}
    checks.append(check)
    if not good:failures.append(check)
complete=len(outputs)==len(requests) and code==0
proof={'schema':'final-release-high-rank-private-abi-v1','inputs':inputs,'requests':bound(REPORT/'high-rank-requests.json'),
       'references':bound(REPORT/'high-rank-references.json'),'input_stream':bound(REPORT/'high-rank-requests.txt'),
       'stdout':bound(REPORT/'high-rank.stdout'),'stderr':bound(REPORT/'high-rank.stderr'),
       'driver':bound(REPORT/'physical-driver'),'driver_source':bound(REPORT/'physical-driver.c'),
       'archive':bound(ROOT/'.ninjars-build/final-dependency-candidate-20261021/bridge/libninjars.a'),
       'controller':bound(Path(__file__)),'checks':checks,'failures':failures,'exit_code':code,
       'complete':complete,'passed':complete and not failures,'wall_seconds':time.monotonic()-start,
       'call_count':len(outputs),'topologies':[2,3,4],'ranks':[4,5,6],'lanes':9,'modes':[0,1,2],
       'explicit_mu_squared_numerators':False,'installed_archive':False,'performance_claim':False,
       'rational_comparison':'Complete finite/single/double Laurent coefficients only; independent references do not specify the engine-basis rational split.'}
(REPORT/'high-rank-proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:proof[key] for key in ['exit_code','complete','passed','call_count','wall_seconds']}),flush=True)
sys.exit(0 if proof['passed'] else 1)
