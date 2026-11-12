"""Pure artifact copy/compression/seal after native closure; no numerical programs."""
from pathlib import Path
import gzip,hashlib,json,shutil
ROOT=Path.cwd();HERE=ROOT/'.tirust-build/zggg-five-backend-timing-1';PACKET=HERE/'closed-packet'
assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
def identity(path):return {'bytes':Path(path).stat().st_size,'sha256':digest(path)}
def dump(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
release=read(HERE/'native-release.json');terminal=read(HERE/'numerical-terminal.json');summary=read(HERE/'measurements/summary.json')
assert release['closed'] and release['shared_unchanged'] and release['mutex_free_at_check'] and not release['matched_owned_processes_alive']
assert terminal['returncode']==0 and terminal['cleanup']['survivors']==[] and summary['completed_cycle_receipts']==list(range(1,11))
assert summary['closure']['shared_unchanged'] and summary['closure']['cleanup_failures']==[]
PACKET.mkdir(exist_ok=False)
shutil.copytree(HERE/'measurements',PACKET/'measurements')
for name in ('numerical.py','run_numerical.py','summarize.py','corroborate.py','seal.py','native-release.json','numerical-started.json','numerical-terminal.json','numerical-console.log','specification.json','preparation.json','source-build-receipt.json','compilation-pending.json','shared-before.json','shared-after.json','generate.mg5','generation.log','generation.started.json','generation.terminal.json','compile-MODEL.log','compile-MODEL.started.json','compile-MODEL.terminal.json','compile-DHELAS.log','compile-DHELAS.started.json','compile-DHELAS.terminal.json','compile-process.log','compile-process.started.json','compile-process.terminal.json','prepare.py'):
 path=HERE/name
 if path.is_file():copy(path,PACKET/name)
for name in ('failed-start-1','failed-compile-1'):
 shutil.copytree(HERE/name,PACKET/'failed-preparation-attempts'/name)
archives={}
for name in ('mg-source.tar','mg-additional-source.tar'):
 original=HERE/name;dest=PACKET/'source-archives'/(name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True)
 with original.open('rb') as src,dest.open('xb') as stream:
  with gzip.GzipFile(filename='',mode='wb',fileobj=stream,compresslevel=6,mtime=0) as compressed:shutil.copyfileobj(src,compressed,1024*1024)
 h=hashlib.sha256();size=0
 with gzip.open(dest,'rb') as stream:
  while data:=stream.read(1024*1024):h.update(data);size+=len(data)
 original_identity=identity(original);assert {'bytes':size,'sha256':h.hexdigest()}==original_identity
 archives[name]={'original_private_path':str(original),'original':original_identity,'retained_path':str(dest.relative_to(PACKET)),'compressed':identity(dest),'gzip_mtime':0,'gzip_filename_header':'','decompressed_identity_verified':True}
dump(PACKET/'source-archive-receipt.json',archives)
pending=read(HERE/'compilation-pending.json');output=HERE/'output'
for name,checksum in pending['sources'].items():
 original=output/name;assert digest(original)==checksum;copy(original,PACKET/'generated-source'/name)
for name in ('tirust_benchmark.py','tirust_benchmark_worker.py','tirust_benchmark_worker.f','tirust_benchmark_points.py','tirust_benchmark_warmup.py','tirust_benchmark_compiler.py','tirust_benchmark_campaign.py'):
 copy(ROOT/'tools'/name,PACKET/'protocol-sources'/name)
assert digest(PACKET/'protocol-sources/tirust_benchmark_worker.f')==pending['worker_template_sha256']
for backend in (1,6,7,8,9):
 cards=HERE/'lanes'/str(backend)/'output/Cards';dest=PACKET/'frozen-cards'/str(backend)
 shutil.copytree(cards,dest,symlinks=False)
 assert digest(dest/'param_card.dat')==pending['parameter_card_sha256']
metadata=ROOT/'.tirust-build/tirust-oem-signed-prefix-1/HEPTools/tirust/lib'
for name in ('tirust-config.json','tirust-native-libs.txt','tirust-link-before.txt','tirust-dependency-provenance.json'):
 copy(metadata/name,PACKET/'dependency-metadata'/name)
readme='''# Original zggg five-backend shared-host pilot\n\nThis immutable packet retains the original signed-source trial. The actual CLI process is `u u~ > z g g g [virt=QCD]` in `loop_sm`. The complete 100-point fixture and its unselected first-ten prefix are retained. Four backends completed ten balanced timed cycles; TIRust reached its 600-second READY deadline and has no qualified warmed timing. The original timeout and all partial outputs remain evidence for later diagnostic/repair work.\n\n`summarize.py` derives `measurements/summary.json` using only retained JSON files. `corroborate.py` derives the point4 finite diagnostic from retained complete/partial warm-up outputs. The report owner authenticates this packet index before running the pure derivation; no native programs are required for replay.\n\nActual generation commands, compiler flags/source inventories, cards, exact source archives, original worker/controller, loaded-library witnesses for the four READY lanes, separate input/loader binding for the failed TIRust lane, every coefficient check and timed sample, and owned cleanup/mutex-release receipts are retained. Source archives use deterministic gzip, with original/decompressed identities in source-archive-receipt.json.\n\nThe numerical reproduction scripts retain the original host paths and qualified Nix toolchain. A local rerun must explicitly rebind paths/tools and installed dependency prefixes, retain the same point/card streams and policy, and produce a fresh evidence namespace. It must not overwrite this packet or reuse these measurements as qualification for changed source. The 100-point main campaign and precise local performance acceptance remain separate TODOs.\n'''
(PACKET/'README.md').write_text(readme)
files={p.relative_to(PACKET).as_posix():identity(p) for p in sorted(PACKET.rglob('*')) if p.is_file()}
assert not any(p.is_symlink() for p in PACKET.rglob('*'))
index={'schema':'madloop-zggg-five-backend-closed-packet-v1','closed':True,'source_mg_revision':summary['mg_revision'],'tirust_revision':summary['tirust_revision'],'case':'qcd_n4_uux_zggg','performance_acceptance':False,'native_release_sha256':digest(PACKET/'native-release.json'),'status':'four complete timing lanes; original TIRust READY timeout retained','files':files}
dump(PACKET/'index.json',index)
print(json.dumps({'packet':str(PACKET),'index_sha256':digest(PACKET/'index.json'),'payload_files':len(files),'payload_bytes':sum(v['bytes'] for v in files.values()),'measurements':'measurements','summarizer':'summarize.py','controller':'numerical.py'},indent=2))
