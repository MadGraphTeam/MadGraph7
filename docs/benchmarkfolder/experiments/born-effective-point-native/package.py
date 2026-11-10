#!/usr/bin/env python3
"""Capture reviewed private sources and closed native evidence as portable text."""
import gzip,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
WORK=ROOT/'.tirust-build/born-effective-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert Path.cwd()==ROOT and not(HERE/'sources.json.gz').exists()
 files={};blobs={};binary={};links={};copied={}
 def add(p,role):
  p=Path(p);name=str(p.relative_to(ROOT))
  if '__pycache__' in p.parts or p.suffix in ('.pyc','.pkl','.pickle'):return
  if not p.is_file():return
  if p.is_symlink():links[name]={'target':str(p.readlink()),'resolved_path':str(p.resolve()),'sha256':sha(p)}
  data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
  native=data.startswith((b'\x7fELF',b'!<arch>\n')) or p.suffix.lower() in ('.o','.a','.so','.mod','.smod','.exe') or p.name in ('runtime_probe','check')
  try:text=data.decode('utf-8') if not native and b'\0' not in data else None
  except UnicodeDecodeError:text=None
  record={'sha256':digest,'bytes':len(data),'role':role,'mode':oct(p.stat().st_mode&0o777)}
  if text is None:binary[name]=record;return
  if name in files:assert files[name]['sha256']==digest;return
  files[name]=record;blobs.setdefault(digest,text)
 def tree(base,role):
  for p in sorted(base.rglob('*')):add(p,role)
 def indexed(base,index):
  d=json.loads((base/index).read_text());add(base/index,'frozen source index')
  for name,entry in d['files'].items():
   p=base/name;assert sha(p)==entry['sha256'],p;add(p,'reviewed private implementation')
 indexed(ROOT/'.tirust-build/born-effective-implementation','review-index.json')
 indexed(ROOT/'.tirust-build/born-effective-exporter-addendum','source-index.json')
 indexed(ROOT/'.tirust-build/general-kinematics-consumer','source-index.json')
 mass=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification'
 add(mass/'source-manifest.json','paired MP mass/caller seam');tree(mass/'src','paired MP mass/caller source')
 basepins=json.loads((ROOT/'.tirust-build/born-effective-implementation/review/source-inputs.json').read_text())
 for name,entry in basepins['sources'].items():
  p=ROOT/name;assert sha(p)==entry['working_sha256'];add(p,'canonical generation source matching frozen review')
 # All16 complete generated source trees, actual private UFO sources, cards,
 # commands, original failures and build logs. Native bytes are never embedded.
 tree(WORK/'attempt1','complete generated exports and preparation/build evidence')
 for dirname in ('driver-hex-addendum','driver-hex-addendum-attempt2','driver-hex-addendum-attempt3','legacy-controller','legacy-controller2','legacy-controller3','legacy-controller4'):
  tree(WORK/dirname,'driver/source/controller lineage and execution evidence')
 for name in ('docs/benchmarkfolder/inputs/loop_qcd_qed_sm_Gmu.json','tools/tirust_env.sh','tools/tirust_benchmark_points.py','docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py','docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-native/audit_failure.py','docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'):
  add(ROOT/name,'pinned input/tool')
 # Runtime/compiler dependencies are hashes only, including dereferenced links.
 build=json.loads((WORK/'attempt1/build.json').read_text());assert build['state']=='complete' and not build['changed_inputs']
 for name,h in json.loads((WORK/'attempt1/build-inputs.json').read_text()).items():
  p=ROOT/name;assert sha(p)==h,name
  if name not in files:add(p,'bound build source or dependency')
 for name,h in build['artifacts'].items():assert sha(ROOT/name)==h;binary.setdefault(name,{'sha256':h,'bytes':(ROOT/name).stat().st_size,'role':'compiled artifact (hash only)','mode':oct((ROOT/name).stat().st_mode&0o777)})
 for p in (ROOT/'.tirust-build/born-effective-build/attempt7/materialized.json',ROOT/'.tirust-build/born-effective-build/attempt7/build.json'):
  add(p,'prior model/source/artifact donor provenance')
 for name in ('loop_sm','loop_qcd_qed_sm_Gmu'):
  tree(WORK/'attempt1/models'/name,'actual copied UFO source used for generation')
 # Human-readable key records are also copied directly, with original identities.
 direct={
  'build/complete.json':WORK/'attempt1/build.json','build/initial-inputs.json':WORK/'attempt1/build-inputs.json',
  'build/materialized.json':WORK/'attempt1/materialized.json','build/exact-driver.json':WORK/'driver-hex-addendum-attempt3/build.json',
  'ordinary/summary.json':WORK/'legacy-controller4/execution-ordinary/summary.json',
  'fixed-qp/summary.json':WORK/'legacy-controller4/execution-fixed-qp/summary.json',
  'fixed-qp/exact-summary.json':WORK/'legacy-controller4/execution-fixed-qp/exact-summary.json',
  'failure-permission/summary.json':WORK/'legacy-controller2/execution-ordinary/summary.json',
  'failure-layout/summary.json':WORK/'legacy-controller3/execution-ordinary/summary.json',
  'runtime-specifications.json':WORK/'attempt1/runtime-prepared.json','legacy-prepared.json':WORK/'legacy-controller4/prepared.json'}
 for name,p in direct.items():
  dst=HERE/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);copied[name]={'original':str(p.relative_to(ROOT)),'sha256':sha(p)}
 for stage in ('ordinary','fixed-qp'):
  base=WORK/'legacy-controller4'/('execution-'+stage)
  assert json.loads((base/'summary.json').read_text())['state']=='complete'
  for p in sorted(base.rglob('*')):
   if not p.is_file() or p.name=='summary.json':continue
   if p.suffix not in ('.json','.log','.txt','.dat'):continue
   # Avoid duplicating large mapping snapshots/cards; every byte is in bundle.
   if p.name not in ('parsed.json','output.log','execution.json','runtime.json','mapping.json','ldd.txt','ldd-execution.json'):continue
   target=HERE/stage/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);copied[str(target.relative_to(HERE))]={'original':str(p.relative_to(ROOT)),'sha256':sha(p)}
 bundle={'schema':'deduplicated-utf8-source-evidence-v1','files':files,'blobs':blobs,'scope':'Exact UTF8 text bytes keyed by root-relative original path. No binaries, pickles or symlink objects embedded.'}
 encoded=json.dumps(bundle,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();(HERE/'sources.json.gz').write_bytes(gzip.compress(encoded,compresslevel=9,mtime=0))
 dump(HERE/'source-inventory.json',{'files':files,'blob_count':len(blobs),'bundle_sha256':sha(HERE/'sources.json.gz'),'uncompressed_bytes':len(encoded)})
 dump(HERE/'native-identities.json',{'files':binary,'scope':'Hash/size/mode provenance only; no native file bytes distributed.'})
 dump(HERE/'link-identities.json',{'links':links,'scope':'Original alias/link provenance only; source text was dereferenced and native targets are hash-only. No alias directories copied.'})
 dump(HERE/'direct-copy-identities.json',copied)
 print(json.dumps({'text_paths':len(files),'unique_text_blobs':len(blobs),'native_hash_only':len(binary),'original_links':len(links),'compressed_bytes':(HERE/'sources.json.gz').stat().st_size,'direct_copies':len(copied)}))
if __name__=='__main__':main()
