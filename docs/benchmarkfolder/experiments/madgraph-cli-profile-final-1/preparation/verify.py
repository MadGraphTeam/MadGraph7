"""Pure relocated verification of the closed stock-profile control packet."""
from pathlib import Path
import csv,hashlib,json,math,pickle,sys
HERE=Path(__file__).resolve().parent
PACKET=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else (HERE.parent if (HERE.parent/'index.json').is_file() else HERE/'closed-packet')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
index=read(PACKET/'index.json');payloads=index['payloads']
assert {p.relative_to(PACKET).as_posix() for p in PACKET.rglob('*') if p.is_file() and p.name!='index.json'}==set(payloads)
for name,identity in payloads.items():
 path=PACKET/name;assert path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256'],name
summary=read(PACKET/'summary.json');bindings=read(PACKET/'path-bindings.json');rows={row['backend']:row for row in summary['rows']}
for backend in (8,9):
 root=PACKET/'backends'/str(backend);m=read(root/'SavedRunInfos/profile_timing_metadata.json');r=rows[backend]
 assert m['reduction_backends']==[backend]
 assert sha(root/'SavedRunInfos/profile_timing_metadata.json')==r['metadata_sha256']
 assert m['selected_timing_helicity']['filter_policy']=='all-helicities-explicit-poles'
 assert m['selected_timing_helicity']['learned_filter_used'] is False
 t=read(root/'run/terminal.json');assert t['status']=='profile-complete' and t['returncode']==0 and t['shared_unchanged'] and not t['source_changed'] and not t['cleanup']['survivors']
 result=m.get('profile_timing_result',m['timing_result'])
 for field,raw in [('unpolarized_full_ms','run_unpolarized_total'),('unpolarized_skip_ms','run_unpolarized_coefs'),('polarized_full_ms','run_polarized_total'),('polarized_skip_ms','run_polarized_coefs')]:assert r[field]==result[raw]*1000
 assert r['unpolarized_full_minus_skip_estimate_ms']==(result['run_unpolarized_total']-result['run_unpolarized_coefs'])*1000
 assert r['initialization_seconds']==result.get('Initialization')
 for stage in m['stages']:
  card={key.lower():value for key,value in stage['effective_madloop_parameters'].items()}
  assert card['looppolemode']==0 and card['ctmoderun']==1 and card['tirustcomputefinite'] and card['useqpintegrandfortirust']
  for record in stage['source_snapshots'].values():
   relative=Path(record['path']).relative_to(bindings[str(backend)]['original_export_directory']);assert sha(root/relative)==record['sha256']
 st=read(root/'stability/raw-stability.json');assert len(st['DP_stability'])==10 and len(st['QP_stability'])==10
 assert r['qp_probes']==sum(bool(value) for value in st['QP_stability'])
 assert r['max_dp_spread']==max(value['Accuracy'] for value in st['DP_stability'])
 assert m['selected_timing_helicity']['count_kind']=='configured' and m['selected_timing_helicity']['polarized_contribution_verified'] is False
 assert r['route_statistics']['installed_capability'] is True and r['route_statistics']['timed_counts'] is None
 assert r['unpolarized_last_result']['scope'].startswith('retained end-of-batch')
assert summary['tirust_over_ninjars_full_ratio']==rows[9]['unpolarized_full_ms']/rows[8]['unpolarized_full_ms']
assert summary['performance_acceptance'] is False
assert summary['route_statistics']['timed_counts'] is None
assert 'monotonic' in summary['timing_clocks']['unpolarized_full']
with (PACKET/'stability-cdf.csv').open() as stream:
 for row in csv.DictReader(stream):
  threshold=float(row['relative_spread_threshold'])
  for backend,column in ((8,'ninjars_fraction_worse'),(9,'tirust_fraction_worse')):
   values=read(PACKET/'backends'/str(backend)/'stability/raw-stability.json')['DP_stability']
   assert float(row[column])==sum(value['Accuracy']>threshold for value in values)/len(values)
print(json.dumps({'status':'verified','payloads':len(payloads),'index_sha256':sha(PACKET/'index.json'),'stock_profiles':2,'stability_points_per_backend':10,'native_executed':False,'tirust_over_ninjars_full_ratio':summary['tirust_over_ninjars_full_ratio']}))
