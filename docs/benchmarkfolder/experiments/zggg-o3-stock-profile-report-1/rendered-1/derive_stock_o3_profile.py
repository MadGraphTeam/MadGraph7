#!/usr/bin/env python3
"""Pure projection of the sealed actual stock-O3 profile packet."""
from pathlib import Path
import argparse,csv,hashlib,io,json,re,types
HERE=Path(__file__).resolve().parent
SOURCE='cda704e28d634e893fabc3ac358d9a27789f93aa'
DSO='d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052'
def require(v,m):
    if not v:raise ValueError(m)
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def identity(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def inside(root,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe evidence path')
    for part in p.parts:root=root/part;require(not root.is_symlink(),'Symlink evidence')
    return root

def derive(root):
    root=Path(root).resolve();binding=read(HERE/'stock-o3-binding.json');packet=inside(root,binding['packet'])
    require(identity(packet/'index.json')['sha256']==binding['packet_index_sha256'],'Packet seal changed')
    code=packet/'preparation/verify.py';require(identity(code)['sha256']==binding['verifier_sha256'],'Producer replayer changed')
    m=types.ModuleType('sealed_o3_profile_verify');m.__file__=str(code);exec(compile(code.read_bytes(),str(code),'exec'),m.__dict__)
    summary=m.verify(packet);require(summary['source_mg_revision']==SOURCE and summary['performance_acceptance'] is False,'Wrong source/scope')
    proof=read(packet/'build-proof.json');require(proof['all_generated_objects_verified_O3'] and proof['generated_object_count']==174 and not proof['shared_reducer_libraries_rebuilt'] and not proof['unsafe_math_flags'],'O3 proof incomplete')
    endpoint=read(packet/'endpoint-consistency.json');require(endpoint['all_PS_and_normalization_words_identical'] and endpoint['all_3e6_criterion_passed'] and endpoint['all_1e8_ordinary_target_passed'] and len(endpoint['comparisons'])==10,'Endpoint consistency changed')
    pmass=(packet/'generated-process/pmass.inc').read_text().split();require(pmass==['PMASS(1)=ZERO','PMASS(2)=ZERO','PMASS(3)=ABS(MDL_MZ)','PMASS(4)=ZERO','PMASS(5)=ZERO','PMASS(6)=ZERO'],'External-mass energy proof changed')
    reference=next(r['unpolarized_full_ms'] for r in summary['rows'] if r['backend']==8);rows=[]
    for r in summary['rows']:
        require(r['source_mg_revision']==SOURCE and r['tirust_dso_sha256']==DSO and not r['per_evaluation_statuses_available'],'Row source/precision scope changed')
        selection=r['selected_timing_helicity'];require(selection['learned_filter_used'] and selection['contributing_count']==48 and selection['total_count']==96,'Learned filter changed')
        require(r['stage_points']['unpolarized_full']==r['stage_points']['unpolarized_skip']==50,'Actual point counts changed')
        meta=read(packet/f"runs/{r['backend']}/profile_timing_metadata.json");full=next(s for s in meta['stages'] if s['name']=='unpolarized_full')
        require(full['read_ps'] is False and 'SQRTS=MAX(1000D0,2.0D0*TOTMASS)' in full['energy_assignment_lines'],'FULL energy path changed')
        mass=re.search(r'^\s*23\s+([0-9.eE+\-]+)\s+#\s*mz',full['parameter_card']['text'],re.M|re.I);require(mass and float(mass[1])==91.188,'Physical Z mass changed')
        rec=endpoint['records'][str(r['backend'])];require(rec['retained_returncode_words']==['110'] and rec['raw_output_freshness_verified'] is False,'Endpoint status/freshness scope changed')
        rows.append({'backend':r['backend'],'name':r['backend_label'],'full_ms':r['unpolarized_full_ms'],'skip_ms':r['unpolarized_skip_ms'],'full_minus_skip_estimate_ms':r['unpolarized_full_minus_skip_estimate_ms'],'full_points':r['stage_points']['unpolarized_full'],'full_wall_seconds':r['stage_wall_seconds']['unpolarized_full'],'skip_points':r['stage_points']['unpolarized_skip'],'skip_wall_seconds':r['stage_wall_seconds']['unpolarized_skip'],'boot_seconds':r['boot_seconds'],'initialization_seconds':r['initialization_seconds'],'ratio_to_8':r['unpolarized_full_ms']/reference,'status':'Completed; unchecked 110','precision_scope':'Configured DP; timed routes unknown','contributing_helicities':48,'configured_helicities':96})
    require([r['backend'] for r in rows]==[1,6,7,8,9],'Missing configured reducer')
    ratio=rows[-1]['ratio_to_8'];maximum=endpoint['maximum_relative_error']
    data={'schema':'madloop-stock-o3-zggg-report-data-v1','closed':True,'date':binding['date'],'source_mg_revision':SOURCE,'tirust_dso_sha256':DSO,'all_object_o3_proven':True,'performance_acceptance':False,'new_native_physics_evaluations':0,'rows':rows,
        'result_summary':f'TIRust FULL is {rows[-1]["full_ms"]:.3f} ms versus NinjaRS {reference:.3f} ms: ratio {ratio:.5f}, or {(ratio-1)*100:.1f}% higher. This remaining runtime gap is explicit; the ratio is descriptive, with no paired confidence interval.',
        'configuration_scope':'All runs retain 48 contributing helicities out of 96. Timed energy is 1 TeV, verified from the actual FULL driver, external masses and physical card. GNU Fortran 15.2.0; CPU 12 affinity on a shared, unreserved host. Observed O3 compiler commands cover 174 base objects; quiet stock recompiles inherit O3 make_opts/environment/recipe, without directly logged argv. The initial model dependency-race build and successful retry are both retained.',
        'clock_scope':'FULL is the log’s helicity-sum time per phase-space point: (monotonic child-completion wall minus stock boot time)/actual count. SKIP and boot retain Python wall clocks. Each FULL and SKIP stage contains 50 points. Stock/filter initialization uses a fresh three-point learning startup; compiler time is separate. It is not a library-only microbenchmark or the earlier diagnosed one-point initialization experiment.',
        'numerical_scope':f'One retained common FULL endpoint has identical six-momentum and normalization words across all five runs. Its ten Born/finite/pole pairwise comparisons pass 3e-6 and 1e-8 (maximum relative residual {maximum:.3e}). This does not qualify all 50 timed points. Endpoint return code 110 is unchecked and output freshness was not independently instrumented. The separate one-point stability phase is outside timing.',
        'precision_evidence_scope':'These are configured-backend timings, not accepted-backend or native-DP-only status proofs.',
        'endpoint_consistency':{'points':1,'comparisons':10,'maximum_relative_error':maximum,'all_3e6':True,'all_1e8':True,'all_PS_and_normalization_words_identical':True,'raw_output_freshness_verified':False,'retained_returncode':'110'},'generated_O3_objects':174,'predecessor_scientific_chain_replayed':False}
    selection={'schema':'madloop-stock-o3-zggg-selection-v1','append_proposal_admitted':True,'performance_acceptance':False,'new_native_physics_evaluations':0,'packets':[{'path':binding['packet'],'index_sha256':binding['packet_index_sha256']}],'source_binding':identity(HERE/'stock-o3-binding.json')}
    return data,selection

def csv_payloads(data):
    columns=tuple(data['rows'][0]);out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader();writer.writerows(data['rows']);return {'stock-o3-timings.csv':out.getvalue().encode()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();d,s=derive(a.root)
    for name,payload in {'stock-o3-data.json':encoded(d),'stock-o3-selection.json':encoded(s),**csv_payloads(d)}.items():
        if a.check:require((HERE/name).read_bytes()==payload,'Derived report differs: '+name)
        else:require(not (HERE/name).exists(),'Fresh derived output required');(HERE/name).write_bytes(payload)
    print(json.dumps({'verified':a.check,'rows':5,'native_evaluations':0,'old_science_replayed':False}))
if __name__=='__main__':main()
