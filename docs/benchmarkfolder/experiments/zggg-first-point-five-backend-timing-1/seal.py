"""Small file-only packet for the diagnosed first-point five-backend pilot."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess
ROOT=Path.cwd();HERE=ROOT/'.tirust-build/zggg-first-point-five-backend-timing-1';PACKET=HERE/'closed-packet'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def copy(src,dest):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
release=read(HERE/'native-release.json');summary=read(HERE/'summary.json');build=read(HERE/'build.json');source=read(HERE/'committed-source-binding.json')
assert release['closed'] and release['mutex_free_at_check'] and release['reported_survivors']==[] and release['shared_before_after_current_equal']
assert summary['physics_regression_pass'] and all(lane['timing_available'] for lane in summary['backends'].values())
PACKET.mkdir(exist_ok=False)
for name in ('pilot.py','supervise.py','summarize.py','seal.py','specification.json','first-point.dat','worker-route-observer.f','driver-observer.diff','driver-observer-proof.json','static-final-binding.json','committed-source-binding.json','timing-admission.json','build.json','build.log','summary.json','native-release.json'):
 copy(HERE/name,PACKET/name)
for name in ('observations','supervision-build','supervision-run'):shutil.copytree(HERE/name,PACKET/name)
for backend in (1,6,7,8,9):shutil.copytree(HERE/'lanes'/str(backend)/'output/Cards',PACKET/'frozen-cards'/str(backend),symlinks=False)
final=Path(build['prefix']).parents[1]
for name in ('published.json','qualified.json','compiled.json','live-abi.json','qualification.log','source-prepared.json','build-prepared.json','activation-delivery-receipt.json'):
 copy(final/name,PACKET/'final-qualification'/name)
for name in ('tirust-config.json','tirust-native-libs.txt','tirust-link-before.txt','tirust-dependency-provenance.json'):
 copy(Path(build['prefix'])/'lib'/name,PACKET/'dependency-metadata'/name)
parity=ROOT/'.tirust-build/tirust-route-statistics-benchmark-native-qualification-1/result.json';copy(parity,PACKET/'tooling-native-parity.json')
for name,checksum in source['committed_tooling_files'].items():
 raw=subprocess.check_output(['git','show',source['head']+':'+name]);assert hashlib.sha256(raw).hexdigest()==checksum;dest=PACKET/'committed-tooling'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
(PACKET/'README.md').write_text('''# Five reducers at the diagnosed first zggg point\n\nThis packet concerns one deliberately selected diagnostic fixture, not an average over physical points. It retains the exact first saved point from the original100-point fixture, the same original O0 generated process and numerical object/flag/card identities, seven balanced matched cycle orders, all35 timed batches, all45 full coefficient checks, the final qualified d427 library binding and observed20-counter primary/check route deltas.\n\nThe historical ten-point/100-point trial remains separate and unchanged, including its original TIRust600-second READY timeout. Its closed source packet is referenced by index SHAe401cf5d4efb7708c277172eaa3be9f29bcadf54c5b3b484e7311109cf9a26cb. Generated process source is inherited from that packet at MG revisioncd0145922977fcfdc4e3cb3183ddc974c31a7897; the native source commit is8b74cdbb3e106f2859514bb4ca95d9fe5d70b370 and tooling source77eb57b82d4858ff302fa9870778e01cc86d7326. No earlier timing or numerical observation is promoted as validation of the final library.\n\nAll timed getters run before T0 and after T1; snapshot writes occur after T1. The numerical loop and entire cold/warm-up body are byte-identical to the original rendered worker after removing observer additions. TIRust's accepted DP-input dispatch uses observed106-bit DD and256-bit local tensor tiers. These are not genuine binary128-input amplitudes, and no native-DP-only claim is made. The genuine QP dispatch count is separately retained. Fortran coefficient-cache hits are outside the bridge counter scope.\n\nMedian/MAD and matched-cycle ratios are descriptive for this one point on a shared host. No CPU was reserved and no main-campaign runtime acceptance, process-average claim or confidence bound is inferred. The strict1e-8 target misses remain in every retained comparison; all3e-6 regression checks pass.\n\nPure replay: import summarize.py and call derive(packet_root), then compare to summary.json. This needs no installed libraries, native program or external source. Native reproduction retains original-host paths, exact qualification/payload hashes and original cards/fixture; a local rerun must explicitly rebind tools/paths and use a fresh namespace.\n''')
files={p.relative_to(PACKET).as_posix():identity(p) for p in sorted(PACKET.rglob('*')) if p.is_file()};assert not any(p.is_symlink() for p in PACKET.rglob('*'))
index={'schema':'madloop-zggg-diagnosed-onepoint-closed-packet-v1','closed':True,'case':'qcd_n4_uux_zggg','point_sha256':summary['point_sha256'],'tooling_head':source['head'],'final_dso_sha256':summary['final_dso_sha256'],'source_packet_predecessor_index_sha256':summary['original_packet_index_sha256'],'performance_acceptance':False,'files':files};dump(PACKET/'index.json',index)
spec=importlib.util.spec_from_file_location('relocated_onepoint_summary',PACKET/'summarize.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);assert module.derive(PACKET)==summary
print(json.dumps({'packet':str(PACKET),'index_sha256':sha(PACKET/'index.json'),'payload_files':len(files),'payload_bytes':sum(v['bytes'] for v in files.values()),'pure_relocated_summary_replay':'passed','new_native_evaluations':0},indent=2))
