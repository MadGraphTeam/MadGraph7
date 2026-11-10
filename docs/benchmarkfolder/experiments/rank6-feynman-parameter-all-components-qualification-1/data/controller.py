import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
root=Path.cwd(); out=root/'.tirust-build/rank6-fp-all-components-runtime-1'
proposal=root/'docs/benchmarkfolder/experiments/rank6-feynman-parameter-all-components-proposal-1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
index=proposal/'source-index.json'
assert sha(index)=='128df03a4790e25ea49f7bacee5a0a6d5339035ad5e6b73b62df7a7a2dabef27'
bindings={str(index):sha(index),str(Path(__file__).resolve()):sha(Path(__file__))}
d=json.loads(index.read_text())
for key in ('files','inputs'):
 for r in d[key]:
  p=(proposal if key=='files' else root)/r['path']
  assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
  bindings[str(p)]=r['sha256']
supervisor=root/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'
assert sha(supervisor)=='ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
bindings[str(supervisor)]=sha(supervisor)
spec=importlib.util.spec_from_file_location('fp_owned_supervisor',supervisor)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
command=['prlimit','--as=4294967296','--core=0','--','taskset','-c','10',sys.executable,'-B',str(proposal/'all_components.py'),'run','--orders','8','16','18','--chunk-size','4096','--seconds','180','--output',str(out/'vector-result.json'),'--admit-physics']
env=os.environ.copy()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'): env[k]='1'
record={'schema':'independent-all210-parameter-integration-runtime-v1','root_admitted':True,'command':command,'cpu':10,'cpu_exclusive':False,'address_space_limit_bytes':4294967296,'aggregate_worker_wall_cap_seconds':180,'automatic_retries':0,'native_reducer_calls':0,'source_before':bindings,'progress':{},'complete':False,'started_unix':time.time()}
started=time.monotonic()
try:
 with (out/'stdout.txt').open('w') as log:
  code=module.run_plain_capped(command,log,cwd=str(root),env=env,progress=record['progress'],seconds=180)
 record['returncode']=code
 record['source_after']={name:sha(Path(name)) for name in bindings}
 assert record['source_after']==bindings,'Source drift'
 assert code==0,code
 result=json.loads((out/'vector-result.json').read_text())
 assert result['status'].startswith('completed')
 assert [x['n'] for x in result['orders']]==[8,16,18]
 assert result['actual_cpu_affinity']==[10]
 record['complete']=True
except BaseException as error:
 record['error']=f'{type(error).__name__}: {error}'
 raise
finally:
 record['elapsed_seconds']=time.monotonic()-started
 record['finished_unix']=time.time()
 record['outputs']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (out/'stdout.txt',out/'vector-result.json') if p.exists()}
 (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'complete':True,'elapsed_seconds':record['elapsed_seconds'],'outputs':record['outputs']},indent=2))
