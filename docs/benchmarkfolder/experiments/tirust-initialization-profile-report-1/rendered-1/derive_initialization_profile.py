#!/usr/bin/env python3
"""Pure replay of closed initialization, tensor, five-reducer and CLI evidence."""
from pathlib import Path
import argparse,contextlib,csv,hashlib,io,json,math,sys,types
HERE=Path(__file__).resolve().parent
DSO='d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052'
SOURCE='77eb57b82d4858ff302fa9870778e01cc86d7326'

def require(v,m):
    if not v:raise ValueError(m)
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def identity(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
    def unique(items):
        d={}
        for k,v in items:
            require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    def bad(v):raise ValueError('Nonfinite JSON '+v)
    return json.loads(Path(p).read_bytes(),object_pairs_hook=unique,parse_constant=bad)
def module(p,name,expected=None):
    if expected:require(identity(p)['sha256']==expected,'Changed pure replayer')
    m=types.ModuleType(name);m.__file__=str(p);exec(compile(Path(p).read_bytes(),str(p),'exec'),m.__dict__);return m

def stock(folder):
    # The retained verifier is pure; execute its checks without a subprocess or native call.
    code=folder/'preparation/verify.py';saved=sys.argv
    try:
        sys.argv=[str(code),str(folder)]
        with contextlib.redirect_stdout(io.StringIO()):module(code,'closed_profile_replay')
    finally:sys.argv=saved
    d=read(folder/'summary.json')
    require(d['source_mg_revision']==SOURCE and d['tirust_dso_sha256']==DSO,'Stock source differs')
    require(d['performance_acceptance'] is False and d['route_statistics']['timed_counts'] is None,'Stock scope changed')
    rows=[]
    for r in d['rows']:
        require(r['per_evaluation_statuses_available'] is False,'Stock precision scope differs')
        if r['backend']==9:require(r['initialization_seconds'] is None,'Reused stock initialization must stay unavailable')
        result=r['unpolarized_last_result'];require(result['finite'] and result['born_nonzero'] and result['finite_nonzero'],'Stock all-helicity result is zero/nonfinite')
        require(r['qp_probes']==r['unstable_points']==r['exceptional_points']==0 and r['stability_points']==10,'Stock stability scope differs')
        full=r['stage_wall_seconds']['unpolarized_full'];skip=r['stage_wall_seconds']['unpolarized_skip'];boot=r['boot_seconds'];n=r['stage_points']['unpolarized_full']
        require(n==r['stage_points']['unpolarized_skip'] and math.isclose(r['unpolarized_full_ms'],1000*(full-boot)/n,rel_tol=1e-12),'Stock clock/count arithmetic differs')
        require(math.isclose(r['unpolarized_skip_ms'],1000*(skip-boot)/n,rel_tol=1e-12),'Stock SKIP arithmetic differs')
        rows.append({'backend':r['backend'],'name':r['backend_label'],'full_ms':r['unpolarized_full_ms'],'skip_ms':r['unpolarized_skip_ms'],'full_minus_skip_estimate_ms':r['unpolarized_full_minus_skip_estimate_ms'],'full_points':n,'full_batch_wall_seconds':full,'skip_batch_wall_seconds':skip,'boot_seconds':boot,'initialization_seconds':r['initialization_seconds'],'maximum_within_backend_spread':r['max_dp_spread'],'timed_per_point_statuses_available':False,'timed_tensor_routes':None,'last_result_finite_nonzero':True})
    return {'rows':rows,'full_ratio_tirust_over_ninjars':d['tirust_over_ninjars_full_ratio'],'timing_clocks':d['timing_clocks'],'precision_scope':d['precision_scope'],'ratio_scope':d['ratio_scope'],'pole_scope':d['pole_scope'],'confidence_intervals':False,'performance_acceptance':False,'source_mg_revision':SOURCE,'dso_sha256':DSO}

def pilot(folder):
    d=module(folder/'summarize.py','closed_first_point_replay').derive(folder)
    require(d==read(folder/'summary.json'),'Pilot replay differs')
    require(d['physical_points']==1 and d['membership_one_based']==[1] and d['cycles']==7,'Pilot membership/cycles changed')
    require(d['final_dso_sha256']==DSO and d['physics_regression_pass'] is True and d['performance_acceptance'] is False and d['confidence_intervals'] is False,'Pilot scope changed')
    rows=[];checks=[];batches=[]
    for key in ('1','6','7','8','9'):
        r=d['backends'][key];require(r['status']=='complete' and r['sample_count']==7 and r['timing_available'] and r['regression_pass'],'Incomplete pilot lane')
        require(r['aggregate_dispatch_precision_counts']=={key+':DP':7} and len(r['checks'])==9,'Pilot usage/check count changed')
        require(r['ordinary_target_pass'] is (key=='1'),'Unexpected strict-target result')
        rows.append({'backend':int(key),'name':r['name'],'median_ms':1000*r['median_seconds_per_point'],'mad_ms':1000*r['mad_seconds_per_point'],'median_paired_ratio_to_8':r['median_paired_ratio_to_8'],'ready_seconds':r['initialization']['cold_seconds'],'regression_pass':True,'ordinary_target_pass':r['ordinary_target_pass'],'maximum_relative_error':max(c['result']['maximum_relative_error'] for c in r['checks']),'dispatch_precision_counts':r['aggregate_dispatch_precision_counts'],'samples':7})
        batches.extend(s['seconds'] for s in r['samples'])
        for c in r['checks']:checks.append({'backend':int(key),'path':c['path'],'regression_pass':c['result']['passed'],'ordinary_target_pass':c['result']['ordinary_target_passed'],'maximum_relative_error':c['result']['maximum_relative_error']})
    nine=d['backends']['9'];p=nine['precision'];require(p['genuine_qp_amplitudes']==0 and p['local_tensor_tiers_observed'] and p['native_dp_only_qualified'] is False,'Pilot tier scope changed')
    require(nine['aggregate_route_delta']==[7448,392,238,42,0,0,0,252,42,0,7518,350,182,70,0,0,0,182,70,0],'Pilot route totals differ')
    flags={};compilers=set()
    for scope,v in d['compiler_flags']['scopes'].items():
        flags[scope]=[list(v) for v in sorted({tuple(r['flags']) for r in v['recipes'].values()})]
        for r in v['recipes'].values():
            for c in r['compiler']:compilers.add(c['version'].split('\n')[0])
    hosts=[]
    for b in ('1','6','7','8','9'):
        for i in range(1,8):
            h=read(folder/f'observations/lanes/{b}/raw/cycle-{i}/host.json')
            for key in ('before','after'):hosts.append(h[key]['load_averages'][0])
    require(len(checks)==45 and len(batches)==35 and min(batches)>1.1,'Incomplete fixed pilot')
    return {'rows':rows,'checks':checks,'timed_batches':35,'numerical_checks':45,'cycles':7,'physical_points':1,'membership_one_based':[1],'batch_min_seconds':min(batches),'batch_max_seconds':max(batches),'cpu':12,'cpu_reserved':False,'host_load1_min':min(hosts),'host_load1_max':max(hosts),'compilers':sorted(compilers),'flag_sets':flags,'executable_sha256':d['executable_sha256'],'point_sha256':d['point_sha256'],'original100_sha256':d['original100_sha256'],'original10_sha256':d['original10_sha256'],'original_packet_index_sha256':d['original_packet_index_sha256'],'source_mg_revision':SOURCE,'dso_sha256':DSO,'precision':p,'route_delta':nine['aggregate_route_delta'],'confidence_intervals':False,'performance_acceptance':False}

def derive(root):
    root=Path(root).resolve();binding=read(HERE/'initialization-source-binding.json')
    require(binding['schema']=='madloop-initialization-profile-final-source-binding-v1' and binding['source_mg_revision']==SOURCE and binding['scientific_appendix_admitted'] is True,'Final binding absent')
    legacy=module(HERE/'prototype_derivation.py','retained_prototype_replay',binding['prototype_replayer_sha256'])
    for p in binding['packets']:legacy.packet(legacy.inside(root,p['path']),p['index_sha256'])
    cm=read(legacy.inside(root,binding['controls_manifest']));am=read(legacy.inside(root,binding['candidate_manifest']))
    controls=legacy.derive_controls(root,cm);candidate=legacy.candidate_data(root,am,cm)
    artifact_module=module(legacy.inside(root,binding['artifact_replayer']),'retained_final_artifact_replay',binding['artifact_replayer_sha256'])
    artifact=artifact_module.admit(root,read(legacy.inside(root,binding['artifact_manifest'])))
    require(artifact==read(legacy.inside(root,binding['artifact_admission'])) and artifact['final_artifact_qualified'] is True,'Final artifact chain replay differs')
    current_pilot=pilot(legacy.inside(root,binding['pilot']));current_stock=stock(legacy.inside(root,binding['stock']))
    data={'schema':'madloop-initialization-profile-report-data-v1','closed':True,'preview_only':False,'date':binding['date'],'source_label':'77eb57b8 / d427 · prototype controls separate','controls':controls,'candidate':candidate,'final_artifact':artifact,'pilot':current_pilot,'stock_current':current_stock,'production_artifact_qualified':True,'performance_acceptance':False,'new_native_physics_evaluations':0,'evidence_note':'Final-source pilot and corrected CLI are descriptive shared-host observations. Prototype470 initialization remains separately bound. All 66 fresh local numerical lanes and eight main runtime gates remain pending.'}
    selection={'schema':'madloop-initialization-profile-report-selection-v1','closed':True,'candidate_qualified':True,'production_artifact_qualified':True,'append_proposal_admitted':True,'canonical_publication_admitted':False,'production_source_promotion':False,'performance_acceptance':False,'new_native_physics_evaluations':0,'style_sha256':identity(HERE/'style.typ')['sha256'],'packets':binding['packets'],'source_binding':identity(HERE/'initialization-source-binding.json')}
    return data,selection

def simple_csv(rows,columns):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=columns);w.writeheader()
    for row in rows:w.writerow({k:json.dumps(row[k],sort_keys=True) if isinstance(row[k],(dict,list)) else row[k] for k in columns})
    return out.getvalue().encode()

def csv_payloads(data):
    legacy=module(HERE/'prototype_derivation.py','retained_csv_functions')
    rows=data['pilot']['rows'];p=data['pilot']['precision'];routes=[]
    for context,counts in p['route_blocks'].items():
        for counter,count in counts.items():routes.append({'context':context,'counter':counter,'count':count,'scope':'Timed bridge computations only; excludes Fortran coefficient-cache hits; 53-bit inputs/DP numerator; local tiers are not genuine Fortran QP'})
    return {'phase-timings.csv':legacy.phase_csv(data['controls']),'stock-profile.csv':legacy.stock_csv(data['controls']),'candidate-routes.csv':legacy.routes_csv(data['candidate']),'clean-initialization.csv':legacy.clean_csv(data['candidate']),'reference-component-misses.csv':legacy.component_miss_csv(data['candidate']),'first-point-timings.csv':simple_csv(rows,tuple(rows[0])),'first-point-routes.csv':simple_csv(routes,('context','counter','count','scope')),'first-point-checks.csv':simple_csv(data['pilot']['checks'],tuple(data['pilot']['checks'][0])),'stock-profile-current.csv':simple_csv(data['stock_current']['rows'],tuple(data['stock_current']['rows'][0]))}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();d,s=derive(a.root)
    payloads={'initialization-profile-data.json':encoded(d),'initialization-profile-selection.json':encoded(s),**csv_payloads(d)}
    for n,b in payloads.items():
        if a.check:require((HERE/n).read_bytes()==b,'Derived output differs: '+n)
        else:
            require(not (HERE/n).exists(),'Fresh output required: '+n);(HERE/n).write_bytes(b)
    print(json.dumps({'verified':a.check,'derived':not a.check,'native_evaluations':0,'performance_acceptance':False,'pilot_batches':35,'pilot_checks':45}))
if __name__=='__main__':main()
