"""Pure relocation-safe replay of this one stock-profile evidence packet."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,re
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def verify(packet):
 packet=Path(packet).resolve();index=read(packet/'index.json')
 assert not any(p.is_symlink() for p in packet.rglob('*'))
 assert {p.relative_to(packet).as_posix() for p in packet.rglob('*') if p.is_file()}==set(index['files'])|{'index.json'}
 for name,digest in index['files'].items():
  p=packet/name;assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert p.is_file() and not p.is_symlink() and sha(p)==digest,name
 release=read(packet/'native-release.json');assert release['all_five_profile_children_closed'] and release['source_files_unchanged'] and release['shared_unchanged_all_profiles'] and release['mutex_nonblocking_acquired'] and not release['owned_survivors']
 combined=read(packet/'summary.json');assert [r['backend'] for r in combined['rows']]==[1,6,7,8,9]
 for row in combined['rows']:
  lane=packet/'runs'/str(row['backend']);meta=read(lane/'profile_timing_metadata.json')
  assert row==read(lane/'summary.json') and meta['reduction_backends']==[row['backend']]
  assert meta['runtime_files']['Source/make_opts']['sha256']==sha(packet/'generated-makefiles/Source/make_opts')
  result=meta.get('profile_timing_result',meta['timing_result']);stages={s['name']:s for s in meta['stages']}
  for field,key in [('unpolarized_full_ms','run_unpolarized_total'),('unpolarized_skip_ms','run_unpolarized_coefs')]:assert row[field]==1000*result[key]
  assert row['unpolarized_full_minus_skip_estimate_ms']==1000*(result['run_unpolarized_total']-result['run_unpolarized_coefs'])
  assert math.isclose(row['unpolarized_full_ms'],1000*(stages['unpolarized_full']['wall_seconds']-result['Booting_time'])/stages['unpolarized_full']['actual_npoints'],rel_tol=1e-12)
  assert row['initialization_seconds']==result.get('Initialization')
  full=stages['unpolarized_full'];skip=stages['unpolarized_skip']
  assert full['helicity_one_based_index']==skip['helicity_one_based_index']==-1
  assert full['read_ps'] is False and full['flags']['SKIPLOOPEVAL'] is False and full['flags']['BOOTANDSTOP'] is False
  assert skip['flags']['SKIPLOOPEVAL'] is True
  selection=meta['selected_timing_helicity'];assert selection['learned_filter_used'] and selection['count_kind']=='contributing'
  assert sum(int(t)>-10000 for t in selection['filter_tokens'])==selection['contributing_count']
  for s in stages.values():
   e={k.lower():v for k,v in s['effective_madloop_parameters'].items()}
   assert e['ctmoderun']==e['ctmodeinit']==1 and e['looppolemode']==-1 and e['helicityfilterlevel']==2 and e['useloopfilter']
   assert e['tirustcomputefinite'] and e['colliercomputeuvpoles'] and e['colliercomputeirpoles']
  log=(lane/'profile.log').read_text();match=re.search(r'Helicity sum time / PSpoint\s*=+\s*([0-9.eE+-]+)(m?s)',log);assert match
  rounded=float(match[1])*(1 if match[2]=='ms' else 1000)
  assert math.isclose(rounded,float(format(row['unpolarized_full_ms'],'.3g')),rel_tol=1e-12)
  terminal=read(lane/'terminal.json');assert terminal['status']=='child-complete' and terminal['shared_unchanged'] and not terminal['cleanup']['survivors']
 proof=read(packet/'build-proof.json');assert proof['all_generated_objects_verified_O3'] and proof['generated_object_count']==174 and proof['unsafe_math_flags'] is False
 assert all(r['optimization']=='-O3' and r['compile_command_verified'] for r in proof['objects'].values())
 assert combined['performance_acceptance'] is False and all(not r['per_evaluation_statuses_available'] for r in combined['rows'])
 spec=importlib.util.spec_from_file_location('o3_endpoint_replay',packet/'preparation/endpoint.py');ep=importlib.util.module_from_spec(spec);spec.loader.exec_module(ep)
 assert ep.endpoint_consistency(packet)==read(packet/'endpoint-consistency.json')
 return combined
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('packet',type=Path);a=p.parse_args();v=verify(a.packet)
 print(json.dumps({'verified':True,'rows':[{'backend':r['backend'],'label':r['backend_label'],'full_ms':r['unpolarized_full_ms']} for r in v['rows']]},indent=2))
