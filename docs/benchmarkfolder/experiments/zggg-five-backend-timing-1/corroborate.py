"""Pure point4 finite-coefficient cross-check; never supplies a timing qualification."""
from pathlib import Path
import collections,hashlib,json,sys
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent;REPORT=HERE/'measurements'
sys.path.insert(0,str(ROOT/'tools'));from tirust_benchmark_worker import validate_status

def derive(directory):
 directory=Path(directory);records={};inputs={}
 for b in (1,6,7,8):
  path=directory/'lanes'/str(b)/'learning.json';value=json.loads(path.read_text());row=next(r for r in value['rows'] if r['point']==4 and r['order']==0)
  records[str(b)]={'finite':row['coefficients'][1],'status':row['status'],'precision':row['precision'],'source_phase':'last complete steady warm-up traversal'};inputs[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
 path=directory/'lanes/9/failed-startup-raw/warmup.dat'
 if not path.exists():path=directory/'lanes/9/startup/warmup.dat'
 partial=None
 if path.exists():
  lines=[l.split() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
  steady=[f for f in lines if int(f[0])>0 and int(f[2])==0]
  usage=collections.Counter();statuses=collections.Counter()
  for f in steady:
   code=int(f[3]);decoded=validate_status(code,9);usage['9:'+decoded['precision']]+=1;statuses[str(code)]+=1
  partial={'phase':'partial steady warm-up','calls':len(steady),'points_one_based':[int(f[1]) for f in steady],'backend_precision_counts':dict(usage),'status_counts':dict(statuses),'complete_ten_point_traversal':False,'timing_available':False}
  candidates=[f for f in lines if int(f[1])==4 and int(f[2])==0 and int(f[0])>0]
  if candidates:
   row=candidates[-1];status=int(row[3]);decoded=validate_status(status,9)
   records['9']={'finite':float(row[5]),'status':status,'precision':decoded['precision'],'call':int(row[0]),'source_phase':'partial steady warm-up trace; no complete READY/timing qualification inferred'}
  inputs[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
 ct=records['1']['finite'];cluster=[v['finite'] for k,v in records.items() if k!='1'];scale=max(map(abs,cluster))
 return {'schema':'madloop-zggg-point4-finite-corroboration-v1','point':4,'order':0,'coefficient':'finite','records':records,'tirust_partial_warmup':partial,'cluster_backends':sorted(int(k) for k in records if k!='1'),'cluster_relative_spread':(max(cluster)-min(cluster))/scale,'cuttools_raw_relative_residuals':{k:abs(ct-v['finite'])/max(abs(ct),abs(v['finite'])) for k,v in records.items() if k!='1'},'inference':'Independent reducers 6/7/8 and available TIRust 9 partial QP output cluster more closely than their difference from the CutTools 1 DP value. This does not identify a mathematical truth or automatically assign fault. The separate 3e-6 regression criterion remains the qualification gate.','new_native_evaluations':0,'performance_acceptance':False,'input_sha256':inputs}
if __name__=='__main__':
 result=derive(REPORT);(REPORT/'point4-corroboration.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
