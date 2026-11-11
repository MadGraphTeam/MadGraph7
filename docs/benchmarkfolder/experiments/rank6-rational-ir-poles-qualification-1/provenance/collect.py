"""Text/gzip retention only. No binary open, native call, compiler or Git action."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent;OUT=HERE/'package'
SOURCES={
 'scientific':('.tirust-build/rank6-ir-triangle-controller-1','4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0'),
 'validation':('.tirust-build/rank6-ir-triangle-validation-controller-1','73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640'),
 'run':('.tirust-build/rank6-ir-triangle-run-controller-1','1d1642c77d698afd8f0a4f8fa3c0db361733fa6ca6d9b1c1e1e44cbdd8be30cf'),
 'oracle':('.tirust-build/rank6-rational-pole-oracle-proposal-1','9f866cc001a47cda221d6be4183ed135cb61af2ad6b48536012bce00b6e7587d'),
}
DATASETS={
 'failed-build':'.tirust-build/rank6-ir-triangle-work-1/build',
 'validation':'.tirust-build/rank6-ir-triangle-validation-work-1/validate',
 'run':'.tirust-build/rank6-ir-triangle-run-work-1/run',
}
def identity(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def save(path,data):
 path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
def text_storage(prefix,rel,data):
 data.decode('utf8');assert b'\x00' not in data
 encoding='gzip' if len(data)>65536 else 'identity'
 name=prefix+'/'+rel+('.gz' if encoding=='gzip' else '')
 stored=gzip.compress(data,mtime=0) if encoding=='gzip' else data
 save(name,stored)
 return {'stored':name,'encoding':encoding,**identity(data),'stored_identity':identity(stored)}
def main():
 retention={'schema':'rank6-ir-pole-closed-retention-v1','sources':{},'datasets':{},'context':{},'independent_audit':{},'excluded_binaries':{},'scope':'Two exact-input historical CAP6 rank6 tensor fixtures; cache0, no amplitude/rescue/trap/performance qualification.'}
 for label,(folder,pin) in SOURCES.items():
  source=ROOT/folder;index=source/'source-index.json';raw=index.read_bytes();assert identity(raw)['sha256']==pin
  meta=json.loads(raw);assert {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}==set(meta['files'])|{'source-index.json'}
  for rel,want in meta['files'].items():
   data=(source/rel).read_bytes();assert identity(data)==want
   if rel.endswith('.gz'):gzip.decompress(data).decode('utf8')
   else:data.decode('utf8');assert b'\x00' not in data
   save('source/'+label+'/'+rel,data)
  save('source/'+label+'/source-index.json',raw)
  retention['sources'][label]={'root':'source/'+label,'index_sha256':pin,'payload_files':len(meta['files']),'original_root':folder}
 runpins=json.loads((ROOT/SOURCES['run'][0]/'input-pins.json').read_text())
 for label,folder in DATASETS.items():
  source=ROOT/folder;items={};excluded={}
  for p in sorted(source.rglob('*')):
   if not p.is_file():continue
   rel=str(p.relative_to(source))
   if label=='failed-build' and (p.name=='check' or p.suffix in ('.o','.mod')):
    expected=runpins['protected_files'][str(p)]
    excluded[rel]=expected;retention['excluded_binaries'][str(p)]={'dataset':label,'relative':rel,**expected,'retained':'identity only; binary never opened/copied by collection'}
    continue
   data=p.read_bytes();expected=runpins['protected_files'].get(str(p))
   if expected is not None:assert identity(data)==expected
   items[rel]=text_storage('data/'+label,rel,data)
  retention['datasets'][label]={'original_root':folder,'files':items,'excluded_binaries':excluded}
 ancestor=ROOT/'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1'
 paths=[ancestor/'build1/archive.json',ancestor/'build1/build.json',ancestor/'build-plan.json',ancestor/'materialized-index.json',ancestor/'review-index.json',ancestor/'runtime-preparation.json',ancestor/'observer.c']
 paths += [p for p in sorted((ancestor/'materialized/ew').rglob('*')) if p.is_file()]
 for p in paths:
  rel=str(p.relative_to(ancestor));data=p.read_bytes();retention['context'][rel]=text_storage('context',rel,data)
 audit=ROOT/'.tirust-build/rank6-ir-triangle-independent-audit-1'
 assert identity((audit/'receipt.json').read_bytes())['sha256']=='bc771c770aaf3e0280727fd4e8ffd4930ae0a923788b229768f6f180b1095171'
 for p in sorted(audit.iterdir()):
  if p.is_file():retention['independent_audit'][p.name]=text_storage('independent-audit',p.name,p.read_bytes())
 save('retention.json',(json.dumps(retention,indent=2,sort_keys=True)+'\n').encode())
 print(json.dumps({'source_packages':len(SOURCES),'datasets':{k:len(v['files']) for k,v in retention['datasets'].items()},'excluded_binaries':len(retention['excluded_binaries']),'context_files':len(retention['context'])}))
if __name__=='__main__':main()
