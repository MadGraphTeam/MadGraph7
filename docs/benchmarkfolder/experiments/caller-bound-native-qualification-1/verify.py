"""Portable text-only replay; opens this capsule and two named sibling capsules only."""
import argparse,gzip,hashlib,json,math,re,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
SEALED={'factory':'aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047',
 'kernel':'1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964',
 'proposal':'8c3639c9412831e369f86bbd4b657818f39aad1aa2f521126e49f544a655a837',
 'conditional':'a51a685abffb384e9503f2a25703f4fb09a08ba13e07c164081eae52993e081e'}
DEPENDENCIES={'current-kinematic-trace-build-qualification-1':'89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
 'current-kinematic-trace-runtime-qualification-1':'730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4'}
def pin(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(ok,message):
 if not ok:raise ValueError(message)
def safe(name):
 p=Path(name);require(str(p)==name and not p.is_absolute() and name not in ('','.') and '..' not in p.parts,'Unsafe relative path');return p
class Store:
 def __init__(self,root,index_name='evidence-index.json',expected=None,full=False):
  self.root=root.resolve();self.used={};self.index_name=index_name
  raw=self.file(index_name)
  if expected:require(pin(raw)['sha256']==expected,'Sibling index differs')
  self.index=json.loads(raw)
  if full:
   found=set()
   for p in self.root.rglob('*'):
    require(not p.is_symlink(),'Symlink in capsule')
    if p.is_file():found.add(str(p.relative_to(self.root)))
   require(found==set(self.index['files'])|{index_name},'Capsule file inventory differs')
   for name in self.index['files']:self.stored(name)
 def file(self,name):
  p=safe(name);target=self.root
  for part in p.parts:
   target=target/part;require(not target.is_symlink(),'Symlink input')
  return target.read_bytes()
 def stored(self,name):
  require(name in self.index['files'],'Unindexed stored input: '+name)
  b=self.file(name);require(pin(b)==self.index['files'][name],'Stored input differs: '+name)
  self.used[name]=pin(b)
  t=gzip.decompress(b) if name.endswith('.gz') else b
  require(b'\0' not in t and not t.startswith((b'\x7fELF',b'!<arch>')),'Nontext input');t.decode('utf8')
  return b
 def read(self,name):
  safe(name)
  if name in self.index.get('compressed_text',{}):
   p=self.index['compressed_text'][name];b=gzip.decompress(self.stored(p['path']))
   require(pin(b)=={'bytes':p['raw_bytes'],'sha256':p['raw_sha256']},'Raw input differs');return b
  return self.stored(name)
 def document(self,name):return json.loads(self.read(name))

class MemoryPath:
 def __init__(self,store,name,original):self.store,self.name,self.original=store,name,original
 def __truediv__(self,name):return MemoryPath(self.store,self.name+'/'+str(name),self.original+'/'+str(name))
 def read_bytes(self):return self.store.read(self.name)
 def read_text(self,*a,**k):return self.read_bytes().decode(k.get('encoding','utf8'))
 def relative_to(self,ignored):return self.original
 def __str__(self):return self.original

def module(name,raw):
 m=types.ModuleType(name);m.__file__='/portable-only/retained/source/'+name+'.py';sys.modules[name]=m
 exec(compile(raw,m.__file__,'exec'),m.__dict__);return m

def check_tables(text,dag):
 arrays={}
 pattern=r'(integer|logical), parameter :: (\w+)\(([^)]+)\) = reshape\(\[ &(.*?)\], \[([^]]+)\]\)'
 for kind,name,shape,body,closing in re.findall(pattern,text,re.S):
  dims=list(map(int,shape.split(',')));require(dims==list(map(int,closing.split(','))),'Shape declaration differs')
  items=[s.strip() for s in body.replace('&','').split(',')]
  values=[s=='.true.' for s in items] if kind=='logical' else list(map(int,items))
  require(len(values)==math.prod(dims),'Array element count');arrays[name]=(dims,values)
 require(set(arrays)=={'route_n','route_nw','route_we','route_pairing','route_supported','route_ops','route_mult','route_class'},'Array inventory')
 def at(name,*coords):
  dims,vals=arrays[name];require(len(dims)==len(coords),'Array coordinate rank');offset=0;stride=1
  for c,d in zip(coords,dims):require(0<=c<d,'Array coordinate');offset+=c*stride;stride*=d
  return vals[offset]
 checks=0
 for r in dag['routes']:
  rid=r['route']-1;n=r['n'];require(at('route_n',rid)==n and at('route_nw',rid)==r['nw'],'Route size')
  for j in range(4):
   require(at('route_we',j,rid)==(r['we'][j] if j<r['nw'] else 0),'WE orientation')
   require(at('route_pairing',j,rid)==(r['pairing'][j] if j<n else 0),'PAIRING orientation')
  for slot,mode in enumerate(('1','2','4','5')):
   m=r['modes'][mode];require(at('route_supported',rid,slot)==m['supported'],'Supported orientation')
   for j in range(4):
    require(at('route_ops',j,rid,slot)==(m['chords'][j]['expanded_add_sub_operations'] if j<n else 0),'Operation orientation')
    for i in range(4):
     require(at('route_mult',i,j,rid,slot)==(m['chords'][j]['unsigned_leaf_multiplicities'][i] if j<n else 0),'Multiplicity orientation')
     require(at('route_class',i,j,rid,slot)==(m['pair_class'][i][j] if i<n and j<n else 0),'Class orientation')
   checks+=1
 return checks

def verify(root=HERE,index_name='evidence-index.json'):
 own=Store(root,index_name,full=True);receipt=own.document('capture-receipt.json')
 require(receipt['dependencies']==DEPENDENCIES,'Dependency declaration')
 siblings={n:Store(root.parent/n,expected=sha) for n,sha in DEPENDENCIES.items()}
 build_sibling=siblings['current-kinematic-trace-build-qualification-1'];runtime=siblings['current-kinematic-trace-runtime-qualification-1']
 packets={}
 for label,sha in SEALED.items():
  raw=own.read(label+'/index.json');require(pin(raw)['sha256']==sha,'Historical source seal differs');d=json.loads(raw);packets[label]=d
  for name,v in d['files'].items():require(pin(own.read(label+'/'+name))==v,'Historical source payload differs')
 for name,v in receipt['origins'].items():require(pin(own.read(name))=={'bytes':v['bytes'],'sha256':v['sha256']},'Captured origin identity')
 for name in ['index.json','cms-source-bindings.json']:
  require(own.read('provenance/'+name)==build_sibling.read('results/derived/'+name),'Source-binding sibling mismatch')
 bindings=own.document('provenance/cms-source-bindings.json');require(len(bindings['files'])==269,'Historical source count')
 generation_raw=build_sibling.read('results/generation.json');generation=json.loads(generation_raw)
 require(generation['raw_sources']==bindings['files'] and generation['generation_only'] and generation['amplitude_evaluations']==0,'CLI/source inventory provenance')
 require(pin(generation_raw)==own.document('provenance/index.json')['generated_source_receipt'],'Generation receipt binding')
 template=build_sibling.read('helpers/.tirust-build/kinematic-provenance-current-proposal-1/proposal/candidate-cms.mg5').decode()
 script=template.replace('.tirust-build/kinematic-provenance-current-proposal-1/work/models/loop_qcd_qed_sm_Gmu','.tirust-build/kinematic-provenance-current-executor-1/work/models/loop_qcd_qed_sm_Gmu').replace('.tirust-build/kinematic-provenance-current-proposal-1/work/exports/candidate/cms_ew_n2_epem_mupmum','.tirust-build/kinematic-provenance-current-executor-1/work/exports/cms')
 require(pin(script.encode())==generation['command'],'Reconstructed CLI command differs')
 generator_source=build_sibling.read('source/run.py').decode();require('def generate():' in generator_source and 'script=script.replace' in generator_source,'Generator recipe provenance')
 dag=own.document('factory/dag-metadata.json');require(dag['source_files_verified']==269 and len(dag['routes'])==26,'DAG historical source scope')
 emit=module('portable_emit',own.read('factory/emit_data.py'))
 require(emit.emit(dag).encode()==own.read('factory/cms_bounds_data.inc'),'Exact table emission differs')
 orientation=check_tables(own.read('factory/cms_bounds_data.inc').decode(),dag)
 # Execute original pure corpus functions with the filesystem access seam bound to retained text.
 prep=module('portable_prepare',own.read('proposal/prepare.py'));prep.FACTORY=MemoryPath(own,'factory','.tirust-build/kinematic-invariant-caller-metadata-1')
 prep.verify_factory=lambda:packets['factory']
 prep.load=lambda name,path:module('portable_'+name,path.read_bytes())
 cases,expected,tokens=prep.corpus();require(json.loads(json.dumps(cases))==own.document('proposal/cases.json'),'Corpus changed')
 require(expected==own.document('proposal/expected.json') and tokens==own.document('proposal/tokens.json'),'Reference changed')
 a,b=prep.render(cases,tokens);require(a.encode()==own.read('proposal/inputs.txt') and b.encode()==own.read('proposal/tokens.txt'),'Wire corpus changed')
 for name in ['inputs.txt','tokens.txt']:require(own.read('execution/run/'+name)==own.read('proposal/'+name),'Actual native input differs')
 for name in ['caller_bounds.f90','cms_bounds_data.inc','environment_token.c']:
  require(own.read('execution/build/'+name)==own.read('factory/'+name),'Compiled module input differs')
 require(own.read('execution/build/driver.f90')==own.read('proposal/driver.f90'),'Compiled driver differs')
 build=own.document('execution/build/result.json');run=own.document('execution/run/result.json')
 require(build['executable']==run['executable'],'Recorded executable identity differs')
 paths=packets['proposal']['compiler_paths'];ff=['-O2','-Wall','-Wextra','-fcheck=all','-fno-fast-math'];prefix=[paths['prlimit'],'--as=4294967296','--']
 cmds=[prefix+[paths['gcc'],'-O2','-Wall','-Wextra','-std=c11','-fno-fast-math','-c','environment_token.c','-o','environment_token.o'],
       prefix+[paths['gfortran'],*ff,'-c','caller_bounds.f90','-o','caller_bounds.o'],
       prefix+[paths['gfortran'],*ff,'-c','driver.f90','-o','driver.o'],
       prefix+[paths['gfortran'],*ff,'caller_bounds.o','environment_token.o','driver.o','-o','probe']]
 require([r['argv'] for r in build['commands']]==cmds,'Build recipe differs')
 last_exit=0;native_raw=own.read('execution/run/native.log')
 for stage,record,labels,cap in [('build',build,['01-c','02-factory','03-driver','04-link'],60),('run',run,['native'],10)]:
  require(record['passed'] is True and record['stage']==stage and record['tensor_calls']==0 and record['budget_seconds']==cap and 0<=record['elapsed_seconds']<cap,'Stage status/budget')
  require(record['source_index_sha256']==SEALED['proposal'] and len(record['commands'])==len(labels),'Execution source/count')
  for cmd,label in zip(record['commands'],labels):
   pr=cmd['progress'];require(cmd['returncode']==pr['returncode']==0 and pr['process_spawned'] is True and pr['process_reaped'] is True,'Command ownership/exit')
   require(last_exit<=pr['spawn_unix']<=pr['exit_observed_unix'] and 0<=pr['process_wall_seconds']<cap,'Serial command chronology');last_exit=pr['exit_observed_unix']
   require(pin(own.read('execution/'+stage+'/'+label+'.log'))==cmd['log'],'Actual log differs')
 require(run['commands'][0]['argv']==prefix+[build['commands'][0]['cwd']+'/probe'],'Run target differs')
 parser=module('portable_analyzer',own.read('proposal/analyze.py'));qualified=parser.parse(native_raw,expected,tokens)
 require(qualified==run['qualification'],'Raw native adjudication differs')
 # The decoder and conditional algorithm retain their exact source. Only file reads are remapped.
 conditional=module('portable_conditional',own.read('conditional/analyze.py'))
 conditional.FAC=MemoryPath(own,'factory','.tirust-build/kinematic-invariant-caller-metadata-1')
 conditional.GEN=MemoryPath(own,'kernel','.tirust-build/kinematic-invariant-general-proposal-1')
 conditional.CAP=MemoryPath(runtime,'','docs/benchmarkfolder/experiments/current-kinematic-trace-runtime-qualification-1')
 # CAP has an empty logical prefix; normalize only the joined logical relative name.
 class RootPath(MemoryPath):
  def __truediv__(self,n):return MemoryPath(self.store,str(n),self.original+'/'+str(n))
 conditional.CAP=RootPath(runtime,'',conditional.CAP.original)
 conditional.module=lambda name,path:module('conditional_'+name,path.read_bytes())
 # Original conditional read() opens the stored gzip path, so read it as stored bytes.
 original_read=runtime.read
 def runtime_read(name):
  if name=='evidence-index.json':return runtime.file(name)
  if name.startswith('compressed/'):return runtime.stored(name)
  return original_read(name)
 runtime.read=runtime_read
 conditional.PINS={};replayed=conditional.main()
 require(replayed==own.document('conditional/results.json'),'Conditional replay changed')
 native_lines=native_raw.decode().splitlines();counts={tag:sum(line.startswith(tag+' ') for line in native_lines) for tag in ('C','B','P','E','T')}
 require(counts['C']==counts['E']==174 and counts['T']==11,'Raw protocol counts')
 require(counts['B']==sum(c['n'] for c in cases) and counts['P']==sum(c['n']**2 for c in cases),'Raw array counts')
 summary={'schema':'portable-caller-bound-native-review-v1','passed':True,'compiled_cases':174,'base_route_modes':orientation,'invalid_boundary_controls':70,'environment_token_controls':11,
  'raw_protocol_bytes':len(native_raw),'raw_protocol_counts':counts,'actual_environment_token':qualified['actual_environment_token'],
  'reference_mismatches':qualified['reference_mismatches'],'observed_ieee_flag_changes':qualified['observed_ieee_flag_changes'],'observed_flag_patterns':qualified['observed_flag_patterns'],
  'closed_build_commands':4,'closed_standalone_workers':1,'recorded_executable':build['executable'],'table_orientation_verified':True,
  'conditional_trace_counts':replayed['counts'],'conditional_causal_pair24_cases':4,'historically_verified_source_hashes':269,
  'CLI_script_reconstructed_sha256':generation['command']['sha256'],'portable_fresh_269_source_reparse':False,'full_amplitude_qualified':False,'full_amplitude_environment_path_qualified':False,'tensor_calls_in_standalone_test':0,
  'external_binary_reads_by_portable_review':0,'new_native_calls_by_portable_review':0,
  'scope':'Integer caller factory/compiler/read-only host environment readiness; retained trace eligibility remains conditional on valid entry-to-use environment and genuine same-kind producer.'}
 if 'qualification-summary.json' in own.index['files']:require(own.document('qualification-summary.json')==summary,'Summary differs from derived result')
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',default='evidence-index.json');args=p.parse_args();print(json.dumps(verify(index_name=args.index),indent=2,sort_keys=True))
