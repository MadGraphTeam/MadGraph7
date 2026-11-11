from pathlib import Path
import hashlib,json,re,sys
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src/madgraph/loop'))
from invariant_environment_source import Library,fixed as source_statements
from bindings import verify_parents
verify_parents(live=True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import madgraph.loop
madgraph.loop.__path__.insert(0,str(HERE/'src/madgraph/loop'))
madgraph.loop.__path__.insert(1,str(ROOT/'.tirust-build/generic-invariant-provenance-proposal-1/src/madgraph/loop'))
from madgraph.loop.invariant_window import momentum_window
from madgraph.loop.invariant_source import statements,call,blocks
from madgraph.loop.invariant_provenance import validate_momentum_writes,kind
INPUTS={}
def sha(b):return hashlib.sha256(b).hexdigest()
def read(path,want):
 b=(ROOT/path).read_bytes();assert sha(b)==want,(path,sha(b),want);INPUTS[str(path)]={'sha256':want,'bytes':len(b)};return b
pins=json.loads((HERE/'parent-pins.json').read_bytes())
for p,h in pins.items():read(p,h)
idx=json.loads(read('.tirust-build/kinematic-provenance-current-executor-1/results/derived/index.json',pins['.tirust-build/kinematic-provenance-current-executor-1/results/derived/index.json']))
b=read('.tirust-build/kinematic-provenance-current-executor-1/results/derived/cms-source-bindings.json',idx['files']['cms-source-bindings.json']['sha256']);cms=json.loads(b)
allcases={}
root='.tirust-build/kinematic-provenance-current-executor-1/work/exports/cms/'
allcases['gmu']={n.removeprefix(root):read(n,m['sha256']) for n,m in cms['files'].items()}
p='.benchmark-work/tirust/campaigns/entry-latch-cache-guard-1/outputs/candidate/qcd_n3_uux_zgg/compilation.json'
d=json.loads(read(p,pins[p]));root=str(Path(p).parent)+'/'
allcases['loop_sm']={n:read(root+n,h) for n,h in d['sources'].items()}
results={}
for case,raw in allcases.items():
 selected={n:b for n,b in raw.items() if n.endswith('.inc') or (n.endswith('.f') and ('/DHELAS/' in n or '/MODEL/' in n and Path(n).name not in ('testprog.f','printout.f') or re.search(r'/P[^/]+/(?:loop_matrix|mp_compute_loop_coefs|polynomial|(?:mp_)?(?:helas_calls_(?:ampb|uvct)|coef_construction)_[0-9]+)\.f$',n)))}
 lib=Library(selected)
 roots=[k for k,v in lib.units.items() if re.fullmatch(r'ML5_[0-9_]+(?:MP_)?(?:HELAS_CALLS_(?:AMPB|UVCT)|COEF_CONSTRUCTION)_[0-9]+',k)]
 result=lib.closure(roots)
 windows={}
 for width,filename in ((64,'loop_matrix.f'),(128,'mp_compute_loop_coefs.f')):
  caller=next(b for n,b in selected.items() if n.endswith('/'+filename))
  order=[]
  for _,stmt,_ in statements(caller.decode()):
   c=call(stmt)
   if c and c[0] in roots and (('MP_' in c[0])==(width==128)):
    assert c[0] not in order;order.append(c[0])
  batches=[(Path(lib.units[k].file).name,selected[lib.units[k].file].decode()) for k in order]
  routines={k:v for n,b in selected.items() if '/DHELAS/' in n and n.endswith('.f') for k,v in blocks(b.decode()).items()}
  receipts=[]
  for fn,body in batches:
   for line,stmt,_ in statements(body):
    c=call(stmt)
    if not c or not c[1] or not re.fullmatch(r'W\([1-9][0-9]*\)',c[1][-1]):continue
    block=routines[c[0]];validate_momentum_writes(block)
    for formal,actual in zip(block['arguments'],c[1]):
     if re.fullmatch(r'W\([1-9][0-9]*\)',actual):kind(block,formal,width)
    receipts.append({'caller':fn,'line':line,'call':stmt})
  windows[str(width)]=momentum_window(selected,batches,receipts)
  windows[str(width)]['receipt_scope']='all actual same-kind W-output source calls; full HELAS graph DAG separately exercised by fresh generation tests'
 model_functions={k:lib.closure([k]) for k,v in lib.units.items() if v.kind=='function' and v.file=='Source/MODEL/model_functions.f'}
 initialization={k:lib.closure([k]) for k in ('COUP','UPDATE_AS_PARAM','MP_UPDATE_AS_PARAM') if k in lib.units}
 results[case]={'momentum_windows':windows,'model_functions':model_functions,'initialization':initialization,'source_files':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in sorted(selected.items())},'producer':result}
 print(case,len(selected),'units',len(lib.units),'roots',len(roots),'complete',result['complete'],'known',len(result['procedures']))
 for x in result['unknown'][:35]:print(x)
 print('windows',[(k,v['complete'],v['reason']) for k,v in windows.items()])
 print('model functions',sum(x['complete'] for x in model_functions.values()),len(model_functions),'initialization',[(k,x['complete']) for k,x in initialization.items()])
(HERE/'closed-results.work.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')

(HERE/'input-bindings.work.json').write_text(json.dumps(INPUTS,sort_keys=True,indent=2)+'\n')
