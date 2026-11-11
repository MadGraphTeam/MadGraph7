"""Retain explicit closed UTF-8 text only; never inspect object/executable bytes."""
from pathlib import Path
import gzip,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCES={
 'factory':('.tirust-build/kinematic-invariant-caller-metadata-1','aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047'),
 'kernel':('.tirust-build/kinematic-invariant-general-proposal-1','1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964'),
 'proposal':('.tirust-build/caller-bound-native-proposal-1','8c3639c9412831e369f86bbd4b657818f39aad1aa2f521126e49f544a655a837'),
 'conditional':('.tirust-build/kinematic-invariant-caller-trace-conditional-1','a51a685abffb384e9503f2a25703f4fb09a08ba13e07c164081eae52993e081e')}
DEPENDENCIES={
 'current-kinematic-trace-build-qualification-1':'89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
 'current-kinematic-trace-runtime-qualification-1':'730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4'}
def pin(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 assert not(HERE/'evidence-index.json').exists()
 origins={};compressed={}
 def retain(name,path,expected=None):
  assert not path.is_symlink();b=path.read_bytes();b.decode('utf8');assert b'\0' not in b
  if expected:assert pin(b)==expected,name
  origins[name]={'path':str(path.relative_to(ROOT)),**pin(b)}
  if len(b)>16384:
   dest='compressed/'+name+'.gz';data=gzip.compress(b,mtime=0)
   compressed[name]={'path':dest,'raw_bytes':len(b),'raw_sha256':pin(b)['sha256']}
  else:dest=name;data=b
  p=HERE/dest;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data)
 for label,(base,sha) in SOURCES.items():
  p=ROOT/base;raw=(p/'index.json').read_bytes();assert pin(raw)['sha256']==sha
  retain(label+'/index.json',p/'index.json')
  for name,v in json.loads(raw)['files'].items():retain(label+'/'+name,p/name,v)
 native=ROOT/'.tirust-build/caller-bound-native-qualification-1'
 for name in ['build/result.json','build/01-c.log','build/02-factory.log','build/03-driver.log','build/04-link.log',
              'build/caller_bounds.f90','build/cms_bounds_data.inc','build/environment_token.c','build/driver.f90',
              'run/result.json','run/native.log','run/inputs.txt','run/tokens.txt']:
  retain('execution/'+name,native/name)
 for name,sha in DEPENDENCIES.items():assert pin((HERE.parent/name/'evidence-index.json').read_bytes())['sha256']==sha
 # Authenticated source metadata already retained in sibling build; no original source/binary reads.
 build=HERE.parent/'current-kinematic-trace-build-qualification-1'
 for name in ('results/derived/index.json','results/derived/cms-source-bindings.json'):
  p=build/name;assert p.is_file();retain('provenance/'+Path(name).name,p)
 (HERE/'capture-receipt.json').write_text(json.dumps({'origins':origins,'compressed_text':compressed,'dependencies':DEPENDENCIES,
   'external_binary_reads':0,'original_269_source_reads':0},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
