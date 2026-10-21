"""Independent stdlib-only arithmetic/selection audit of the fixed light8 data."""
from pathlib import Path
import collections, datetime, hashlib, json, math
ROOT=Path.cwd()
BASE=ROOT/'.benchmark-work/paper-dp-stability-20261021-v1/lightweight-final-eight-qualified-v2'
CASES=['qcd-1','ew-1','qcd-2','ew-2','qcd-3','ew-3','supplement-gg-h','supplement-gg-hh']
INPUTS={}
def require(x,why):
    if not x: raise ValueError(why)
def record(path):
    path=Path(path); path=path if path.is_absolute() else ROOT/path
    raw=path.read_bytes(); INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return raw
def bound(info):
    raw=record(info['path']);require(hashlib.sha256(raw).hexdigest()==info['sha256'],'binding hash')
    require('bytes' not in info or len(raw)==info['bytes'],'binding size');return raw
def load(path):return json.loads(record(path))
def parse(run):
    rows={};meta={};header=None
    for line in bound(run['raw']).decode().splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='DPSD_HEADER':header=list(map(int,f[1:]))
        elif f[0]=='OBS':meta[int(f[1]),int(f[2])]=list(map(int,f[3:]))
        elif f[0]=='ROW':
            key=tuple(map(int,f[1:4]));require(key not in rows,'duplicate row')
            rows[key]=[float(x.replace('D','E')) for x in f[4:]]
    require(header==[1,100,run['nso'],run['nso'],run['helicity'],run['tool'],1],'raw header/selector')
    require(set(rows)=={(p,t,o) for p in range(1,101) for t in (1,2,3) for o in range(run['nso']+1)},'raw coverage')
    require(all(math.isfinite(x) for row in rows.values() for x in row),'finite raw')
    return rows,meta
m=load(BASE/'manifest.json');s=load(BASE/'summary.json');d=load(BASE/'cross-backend-discrepancy.json')
require(m['requested_tools']==s['requested_tools']==d['requested_tools']==[6,8],'selected tools')
require(list(s['cases'])==CASES and list(d['cases'])==CASES,'selected cases')
require(s['complete'] is True and s['physical_qualified'] is True and s['observations_complete'] is True,'qualified scope')
require(s['qualification']=='diagnostic only; does not replace correctness or timing gates','diagnostic scope')
require(m['rescue_enabled'] is False and s['rescue_enabled'] is False,'forced DP scope')
require(len(m['runs'])==16 and {(r['case'],r['tool']) for r in m['runs']}=={(c,t) for c in CASES for t in (6,8)},'exact16 runs')
random_manifest=None;parsed={};score_count=tail_count=scalar_count=0;worst=(0.,None);metadata={}
for run in m['runs']:
    cid,tool=run['case'],run['tool'];require(type(run['N']) is int and run['N']==100,'N100')
    require(run['measurement_authority']['kind']=='random-dp-v1','random authority')
    rm=load(Path(run['measurement_authority']['manifest']['path']));bound(run['measurement_authority']['manifest'])
    require(random_manifest is None or rm==random_manifest,'same random point authority');random_manifest=rm
    definition=rm['cases'][cid];definition_points=definition['points']
    require(run['artifacts']['points']==definition_points,'exact actual random points')
    points=bound(definition_points);require(int(points.splitlines()[0])==100,'actual point count')
    entry=s['cases'][cid]['tools'][str(tool)]
    require(entry['N']==100 and entry['physical_qualified'] is True and entry['complete'] is True,'qualified entry')
    require(entry['raw']==run['raw'] and entry['point_stream']==definition_points,'entry original raw/points')
    require(entry['seed']==definition['seed'] and entry['cuts']==definition['cuts'],'sample seed/cuts')
    select=entry['helicity_selection'];table=[list(map(int,line.split())) for line in bound(select['table']).decode().splitlines()]
    require(select['argument']==run['helicity']==entry['helicity'],'helicity selector')
    if cid.startswith('supplement-'):
        physical=[-1,-1]+[0]*(run['nexternal']-2);hits=[i+1 for i,row in enumerate(table) if row==physical]
        require(hits==[select['one_based_index']] and select['argument']==hits[0] and select['physical_tuple']==physical and select['policy']=='loop-induced-all-minus','physical loop tuple')
    else:
        require(select['argument']==-1 and select['one_based_index'] is None and select['physical_tuple'] is None and select['policy']=='full-helicity-color-sum','Born full helicity')
    if run['nexternal']==3:
        require(entry['sample_kind']=='random-boosted-on-shell-descriptive' and entry['cuts']['pt_min_gev'] is None and entry['cuts']['delta_r_min'] is None,'descriptive2to1')
    else:require(entry['center_of_mass_energy_gev']==1000.,'1TeV multiparticle')
    rows,meta=parse(run);parsed[cid,tool]=(run,rows,meta)
    for name,transforms in [('combined',(1,2,3)),('direction',(1,2)),('lorentz',(1,3))]:
        group=entry['contrasts'][name];require(len(group['point_estimators'])==100,'estimator count');deltas=[]
        for p in range(1,101):
            vals=[rows[p,t,0][1] for t in transforms];scale=max(map(abs,vals))
            normalized=[v/scale for v in vals] if scale else vals
            mean=math.fsum(normalized)/len(vals);spread=max(normalized)-min(normalized)
            require(scale>0 and mean!=0,'actual finite estimator support');delta=spread/abs(mean)
            expected={'point':p,'classification':'finite','delta':delta,'unbounded':False,'values':vals,'normalized_mean':mean,'normalized_spread':spread}
            require(group['point_estimators'][p-1]==expected,'independent estimator arithmetic')
            require(entry['scores'][name][p-1]==expected,'score alias');deltas.append(delta);score_count+=1
        require(group['classification_counts']=={'finite':100} and group['failure_categories']=={},'finite classifications')
        require(len(group['thresholds'])==len(m['thresholds']),'threshold count')
        for t,tail in zip(m['thresholds'],group['thresholds']):
            k=sum(v>t for v in deltas);n=100;z=1.959963984540054;den=1+z*z/n
            center=(k/n+z*z/(2*n))/den;half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/den
            expected={'threshold':t,'k':k,'N':n,'fraction':k/n,'interval95':[max(0.,center-half),min(1.,center+half)],'zero_event_upper95':-math.expm1(math.log(.05)/n) if k==0 else None}
            require(tail==expected,'tail/binomial interval arithmetic');tail_count+=1
        if name=='combined':require(entry['point_estimators']==group['point_estimators'] and entry['thresholds']==group['thresholds'],'combined alias')
    metadata[cid,str(tool)]={'seed':entry['seed'],'cuts':entry['cuts'],'sample_kind':entry['sample_kind'],'center_of_mass_energy_gev':entry['center_of_mass_energy_gev'],'helicity':select}
for cid in CASES:
    a,ar,am=parsed[cid,6];b,br,bm=parsed[cid,8];case=d['cases'][cid]
    require(a['artifacts']['points']==b['artifacts']['points'] and a['helicity']==b['helicity'],'matched points and selector')
    require(case['N']==100 and case['helicity']==a['helicity'] and case['points']==a['artifacts']['points'],'crossdiag scope')
    require(case['point_failures']==0 and case['finite_coefficient_tolerance_failures']==0,'recorded no failures')
    for p,row in enumerate(case['points_data'],1):
        scale=max(abs(v) for raw in (ar,br) for o in range(a['nso']+1) for v in raw[p,1,o][:4])
        require(row['point']==p and row['scale']==scale and row['failure_categories']=={'6':None,'8':None} and row['status_codes']=={'6':110,'8':110} and row['physical_qualified'] is True,'crossdiag point')
        expected=[]
        for o in range(a['nso']+1):
            for i,label in enumerate(['born','finite','single_pole','double_pole']):
                x,y=ar[p,1,o][i],br[p,1,o][i];difference=abs(x-y);pair=max(abs(x),abs(y));tol=1e-7*pair+1e-10*scale
                expected.append({'order':o,'coefficient':label,'values':{'6':x,'8':y},'absolute_difference':difference,'absolute_difference_overflow':False,'pair_scale':pair,'relative_difference':difference/pair if pair else 0.,'point_scale_normalized_difference':difference/scale if scale else 0.,'tolerance':tol,'passes_tolerance':difference<=tol})
                require(difference<=tol,'crossbackend coefficient');fraction=difference/tol if tol else 0.
                if fraction>worst[0]:worst=(fraction,{'case':cid,'point':p,'order':o,'coefficient':label})
                scalar_count+=1
        require(row['coefficients']==expected,'independent full crossdiag arithmetic')
require(score_count==4800 and scalar_count==6400,'complete estimator/coeff count')
result={'schema':'independent-light-eight-arithmetic-audit-v1','status':'PASS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reader':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'cases':CASES,'requested_tools':[6,8],'points_per_case_per_tool':100,'finite_combined_scores':1600,'all_contrast_estimators_recomputed':score_count,'tail_threshold_intervals_recomputed':tail_count,'crossbackend_coefficients_recomputed':scalar_count,'coefficient_tolerance_failures':0,'maximum_tolerance_fraction':worst[0],'worst_comparison':worst[1],'metadata_selection':{c:{str(t):metadata[c,str(t)] for t in (6,8)} for c in CASES},'inputs_sha256':dict(sorted(INPUTS.items())),'scope':'Native-DP diagnostic self consistency; six Born full sums plus ggH/HH physical all-minus; boosted2to1 descriptive, not extensive independent invariants; no rescued correctness/timing replacement','numerical_calls':0,'compiler_invocations':0,'tests_run':0,'performance_qualification':False}
out=Path(__file__).with_name('light-eight-arithmetic-audit-v1.json');require(not out.exists(),'preserve old audit');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['metadata_selection','inputs_sha256']},indent=2));print('result_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
