"""Pure archival/replay inputs for two closed stock CLI controls; no native calls."""
from pathlib import Path
import csv,hashlib,json,pickle,re,shutil
ROOT=Path.cwd();assert ROOT==Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent;PACKET=HERE/'closed-packet';PACKET.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def cp(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ('prepare.py','run_cli.py','observe_result.py','seal.py','verify.py','prepared.json','source-index.json','delivery-receipt.json','README.md'):
 cp(HERE/name,PACKET/'preparation'/name)
rows=[];stability={};bind={}
for backend,label in ((8,'NinjaRS'),(9,'TIRust')):
 run=HERE/'runs'/f'n1-{backend}-1';terminal=read(run/'terminal.json')
 assert terminal['status']=='profile-complete' and terminal['returncode']==0
 assert terminal['shared_unchanged'] and not terminal['source_changed'] and not terminal['cleanup']['survivors']
 export=Path(terminal['saved_export']);metadata_path=export/'SavedRunInfos/profile_timing_metadata.json';metadata=read(metadata_path)
 assert metadata['reduction_backends']==[backend] and metadata['backend_labels']==[label]
 target=PACKET/'backends'/str(backend)
 for name in ('admission.json','started.json','terminal.json','summary.json','console.log','profile.log','commands.mg5','shared-before.json'):
  cp(run/name,target/'run'/name)
 cp(metadata_path,target/'SavedRunInfos/profile_timing_metadata.json')
 retained=[]
 for stage in metadata['stages']:
  for record in stage['source_snapshots'].values():
   source=Path(record['path']);assert sha(source)==record['sha256']
   relative=source.relative_to(export);cp(source,target/relative);retained.append(relative.as_posix())
 for name in ('param_card.dat','MadLoopParams.dat'):
  cp(export/'Cards'/name,target/'cards-after-stability'/name)
 for name in ('Source/make_opts','SubProcesses/MadLoop_makefile_definitions'):
  cp(export/name,target/name)
 for source in export.parent.glob('stability_*.log'):cp(source,target/'stability'/source.name)
 pickle_paths=list(export.glob('SubProcesses/P*_*'+f'/SavedStabilityRun_{label}_0.pkl'));assert len(pickle_paths)==1
 with pickle_paths[0].open('rb') as stream:saved=pickle.load(stream)
 assert len(saved['DP_stability'])==10
 cp(pickle_paths[0],target/'stability'/pickle_paths[0].name)
 dump(target/'stability/raw-stability.json',saved);stability[str(backend)]=saved
 summary=read(run/'summary.json')
 polarized=next(stage for stage in metadata['stages'] if stage['name']=='polarized_full')
 observed=dict(re.findall(r'(?m)^\s*(BORN|FIN|1EPS|2EPS)\s+([^\s]+)',polarized['raw_outputs']['result.dat'].get('text','')))
 summary['retained_polarized_result_all_zero']=len(observed)==4 and all(float(value)==0. for value in observed.values())
 summary['polarized_scope']='first configured helicity, not qualified as contributing; only all-helicity timings used in comparison'
 summary.update(stability_points=10,qp_probes=sum(bool(value) for value in saved['QP_stability']),
                max_dp_spread=max(value['Accuracy'] for value in saved['DP_stability']),
                unstable_points=len(saved['Unstable_PS_points']),exceptional_points=len(saved['Exceptional_PS_points']))
 rows.append(summary)
 bind[str(backend)]={'original_export_directory':str(export),'retained_export_root':str(target.relative_to(PACKET)),
    'original_run_directory':str(run),'retained_run_root':str((target/'run').relative_to(PACKET)),
    'retained_source_snapshots':retained,'note':'Raw metadata paths remain original observations; resolve export-relative snapshot paths within retained_export_root. Runtime binaries are identified, not copied.'}
ratio=rows[1]['unpolarized_full_ms']/rows[0]['unpolarized_full_ms']
dump(PACKET/'path-bindings.json',bind)
dump(PACKET/'summary.json',{'schema':'madgraph-stock-profile-comparison-v1','process':'u u~ > z [virt=QCD]','model':'loop_sm',
 'source_mg_revision':read(HERE/'prepared.json')['base_mg_revision'],'tirust_revision':'10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1',
 'tirust_dso_sha256':rows[1]['tirust_dso_sha256'],'rows':rows,'tirust_over_ninjars_full_ratio':ratio,
 'ratio_scope':'single stock CLI run per backend, unequal short batches, descriptive unpaired shared-host control',
 'precision_scope':'stock CTModeRun1 DP-interface timing with local tensor-tier policy; timed tensor routes/statuses unavailable; separate ten-point within-backend stability probes',
 'pole_scope':'full finite plus UV and IR, LoopPoleMode0, all configured helicities, no learned filters',
 'polarized_scope':'retained last polarized FULL results are zero for both backends; do not use as a contributing-helicity comparison',
 'initialization_scope':'NinjaRS fresh filter-free two-point bootstrap; TIRust reused export, Initialization=null; boot and estimate stages retained separately',
 'performance_acceptance':False,'automatic_whole_amplitude_qp_timing':False,'route_statistics':read(HERE/'prepared.json')['route_statistics'],'timing_clocks':read(HERE/'prepared.json')['timing_clocks']})
with (PACKET/'timings.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=['backend','backend_label','unpolarized_full_ms','unpolarized_skip_ms','unpolarized_full_minus_skip_estimate_ms','polarized_full_ms','polarized_skip_ms','initialization_seconds','boot_seconds','stability_points','qp_probes','max_dp_spread'])
 writer.writeheader();writer.writerows({key:row[key] for key in writer.fieldnames} for row in rows)
with (PACKET/'stability-spreads.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['backend','point_one_based','dp_relative_spread','qp_relative_spread'])
 for backend,saved in stability.items():
  for index,(dp,qp) in enumerate(zip(saved['DP_stability'],saved['QP_stability']),1):writer.writerow([backend,index,dp['Accuracy'],qp.get('Accuracy','')])
with (PACKET/'stability-cdf.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['relative_spread_threshold','ninjars_fraction_worse','tirust_fraction_worse'])
 for index in range(141):
  threshold=10**(-17+index*.05)
  writer.writerow([threshold,*[sum(value['Accuracy']>threshold for value in stability[str(backend)]['DP_stability'])/10 for backend in (8,9)]])
(PACKET/'README.md').write_text('''# Closed stock MadGraph profile control\n\nActual CLI `check profile` commands compare backend8 NinjaRS and backend9 TIRust for QCD `u u~ > z`. Both completed with the same signed TIRust installation, the source revision in preparation/prepared.json, full finite plus UV/IR mode0 and all configured helicities. The second command reuses a private clone of the first export. Both owned cleanups have no survivors; source and shared inputs remain unchanged.\n\nThe table retains stock DP-interface FULL and SKIP timings with local tensor tiers enabled. The retained polarized FULL result is zero for the first configured helicity in both backends; that selection was not qualified as contributing, so only the all-helicity timings are used for comparison. FULL−SKIP is an estimated reduction time. The corrected unpolarized FULL clock records monotonic child completion independently of RSS polling. Other stages retain stock subprocess wall durations. The two commands choose different batch lengths; their single-run ratio is descriptive, without a paired interval or performance acceptance. Stock timing has no per-point return-code or tensor-route stream. Installed route capability is recorded, but missing timed counts remain unknown, not zero or proof of all-native DP. Separate ten-point stability samples are within-backend direction/rotation spreads, not errors against an independent reference. The normal stability samples do not retain every original momentum.\n\nNinjaRS records a fresh two-point filter-free initialization. TIRust reuses the generated export and has no separate Initialization measurement (null, never zero); its estimate and boot stages are retained. These timings belong to the final production DSOd427 and captured source revision only; they do not transfer to future changes or the earlier coarse stock controls.\n\n`summary.json`, `timings.csv` and stability CSVs derive only from retained stock metadata/pickles. The full CLI/profile/stability logs, commands, cards, flags, stage driver/matrix snapshots and owned terminal receipts are retained. `path-bindings.json` maps original absolute source-snapshot observations to packet-relative copies; source and runtime library identities remain unchanged. Existing dynamic/static libraries are identified by hashes rather than repackaged. Source commit and reproduction commands are in preparation/. A rerun must use a new namespace and explicitly rebind local tools and dependency prefixes.\n''')
files={p.relative_to(PACKET).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(PACKET.rglob('*')) if p.is_file()}
dump(PACKET/'index.json',{'schema':'madgraph-final-stock-profile-closed-v1','source_mg_revision':read(HERE/'prepared.json')['base_mg_revision'],'payloads':files,'performance_acceptance':False,'source_controller_sha256':sha(HERE/'run_cli.py'),'pure_seal_source_sha256':sha(__file__)})
print(json.dumps({'files':len(files),'bytes':sum(value['bytes'] for value in files.values()),'index_sha256':sha(PACKET/'index.json'),'ratio':ratio}))
