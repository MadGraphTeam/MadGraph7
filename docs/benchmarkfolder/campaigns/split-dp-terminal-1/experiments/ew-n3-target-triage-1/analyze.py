#!/usr/bin/env python3
"""Reparse closed EW n3 streams from the immutable completed9 checkpoint."""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
REPORT = ROOT/'docs/benchmarkfolder/campaigns/split-dp-terminal-1'
CASE = 'ew_n3_epem_ttxa'
CURRENT = REPORT/('experiments/numerical/full100-warmup-'+CASE+'-1')
CHECKPOINT = REPORT/'numerical-checkpoints/completed9-1/index.json'
CHECKPOINT_SHA = 'c598014c21cd25be50bd8a059e46f209d870862b53ed1e7f76f51607f6b9ef86'
MANIFEST_SHA = '1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31'
LANES = [('baseline',1), *[('candidate',b) for b in (1,6,7,8,9)]]
COLUMNS = ['Born','finite','single pole','double pole']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes':path.stat().st_size,'sha256':sha(path)}


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def bind_inputs():
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    assert not any(p.is_symlink() for p in [CHECKPOINT,*CHECKPOINT.parents] if p.is_relative_to(REPORT))
    checkpoint=read(CHECKPOINT)
    assert checkpoint['case_count']==9 and checkpoint['lane_count']==54
    assert checkpoint['performance_acceptance_eligible'] is False
    prefix=str(CURRENT.relative_to(REPORT))+'/'
    files={name:value for name,value in checkpoint['files'].items() if name.startswith(prefix)}
    required={REPORT/'cases.json',REPORT/'manifest.json',CHECKPOINT.parent/'summary.json',
              REPORT/('points/'+CASE+'.dat'),REPORT/('points/'+CASE+'.json')}
    required.update(CURRENT/(name+'.py') for name in
                    ('tirust_benchmark','tirust_benchmark_worker','tirust_benchmark_warmup'))
    required.add(CURRENT/'results.json');required.add(CURRENT/'specification.json')
    for source,backend in LANES:
        lane=CURRENT/f'{source}-{backend}'
        required.update(lane/name for name in ('result.json',source+'-inputs.json','inputs-after.json'))
        required.update(lane/'raw'/name for name in
                        ('warmup.dat','dimensions.dat','ready.json','observation.json'))
        required.update(REPORT/'inputs'/source/CASE/name for name in
                        ('generation.json','compilation.json','param_card.dat'))
    for path in required:
        name=str(path.relative_to(REPORT));files[name]=checkpoint['files'][name]
    for name,value in files.items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path=REPORT/relative
        assert not any(p.is_symlink() for p in [path,*path.parents] if p.is_relative_to(REPORT))
        assert info(path)==value,name
    summary=read(CHECKPOINT.parent/'summary.json')
    case=next(value for value in summary['cases'] if value['case']['id']==CASE)
    assert case['passed'] and len(case['lanes'])==6
    assert [(row['source'],row['backend']) for row in case['lanes']]==LANES
    return files,case


def retained_helpers():
    tree=ast.parse((CURRENT/'tirust_benchmark.py').read_text())
    node=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='compare_rows')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<retained compare_rows>','exec'),namespace)
    for name in ('tirust_benchmark_worker','tirust_benchmark_warmup'):
        spec=importlib.util.spec_from_file_location(name,CURRENT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return namespace['compare_rows'],sys.modules['tirust_benchmark_warmup'].parse_warmup


def missed(row):
    return not ((row['symmetric_relative_error'] is not None and row['symmetric_relative_error']<1e-8)
                or (row['pole_to_finite'] is not None and row['pole_to_finite']<1e-8))


def condensed(comparison):
    misses=[row for row in comparison['residuals'] if missed(row)]
    columns={}
    for index,name in enumerate(COLUMNS):
        rows=[row for row in comparison['residuals'] if row['column']==index]
        worst=max(rows,key=lambda row:row['symmetric_relative_error'] or 0)
        columns[name]={'maximum_symmetric_relative_error':worst['symmetric_relative_error'],
                       'worst_symmetric_residual':worst,
                       'target_missed_coefficient_comparisons':sum(missed(row) for row in rows)}
    return {key:comparison[key] for key in ('passed','ordinary_target_passed','maximum_relative_error')}|{
        'components':columns,'target_points':sorted({row['point'] for row in misses}),
        'target_missed_coefficient_comparisons':len(misses),'target_misses':misses}


def collect():
    files,closed=bind_inputs()
    compare,parse_warmup=retained_helpers()
    config=read(REPORT/'cases.json');assert config['relative_tolerance']==3e-6
    assert sha(REPORT/'manifest.json')==MANIFEST_SHA
    specification=read(CURRENT/'specification.json')
    assert specification['campaign_binding']=={'id':'split-dp-terminal-1','manifest_sha256':MANIFEST_SHA}
    assert specification['case']==closed['case']
    assert specification['configuration_sha256']==sha(REPORT/'cases.json')
    assert specification['measurement_phase']=='steady-warmup' and specification['performance_eligible'] is False
    point=REPORT/('points/'+CASE+'.dat');metadata=point.with_suffix('.json')
    assert len(read(metadata)['points'])==100
    results=read(CURRENT/'results.json')
    assert [(row['source'],row['backend']) for row in results['results']]==LANES
    current,usage,original={}, {}, {}
    for source,backend in LANES:
        key=f'{source}-{backend}';directory=CURRENT/key
        result=read(directory/'result.json')
        assert result==next(row for row in results['results'] if (row['source'],row['backend'])==(source,backend))
        assert result['status']=='complete' and result['performance_eligible'] is False
        inputs=read(directory/(source+'-inputs.json'));after=read(directory/'inputs-after.json')
        retained=REPORT/'inputs'/source/CASE
        assert inputs['generation']==read(retained/'generation.json')
        assert inputs['compilation']==read(retained/'compilation.json')
        assert inputs['param_card_sha256']==sha(retained/'param_card.dat')
        assert inputs['points_sha256']==sha(point)==after['points_sha256']
        assert inputs['points_metadata_sha256']==sha(metadata)
        assert inputs['numerical_cards']==after['numerical_cards']
        assert inputs['compilation']['executable_sha256']==after['executable_sha256']
        assert all(inputs['runtime'][kind]==after['runtime'][kind] for kind in ('libraries','archives','metadata'))
        assert inputs['settings']['mlreductionlib']==str(backend)
        expected={key.lower():value for key,value in config['madloop_parameters'].items()}
        if source=='candidate':expected.update({key.lower():value for key,value in config['candidate_parameters'].items()})
        assert all(inputs['settings'].get(key)==value for key,value in expected.items())
        observation=parse_warmup(directory/'raw',[backend],100,read(directory/'raw/ready.json'))
        assert observation==result['observation']==read(directory/'raw/observation.json')
        assert observation['seconds'] is None and observation['performance_eligible'] is False
        current[key]=observation['rows']
        usage[key]={'accepted_backend_precision_counts':observation['backend_precision_counts'],
                    'status_counts':observation['status_counts']}
        if source=='baseline':baseline=observation['rows']
        comparison=compare(baseline,observation['rows'],3e-6)
        assert comparison==result['comparison'] and comparison['passed']
        original[key]=condensed(comparison)
        saved=next(row for row in closed['lanes'] if (row['source'],row['backend'])==(source,backend))
        assert original[key]['ordinary_target_passed']==saved['ordinary_target_passed']
        assert original[key]['target_missed_coefficient_comparisons']==saved['ordinary_target_missed_coefficient_comparisons']
        assert usage[key]['accepted_backend_precision_counts']==saved['accepted_backend_precision_counts']
    backends=(1,6,7,8,9)
    pairwise={f'{a}-{b}':condensed(compare(current[f'candidate-{a}'],current[f'candidate-{b}'],3e-6))
              for a,b in itertools.combinations(backends,2)}
    target_points=sorted(set().union(*(set(row['target_points']) for row in [*original.values(),*pairwise.values()])))
    statuses={str(point):{str(backend):[{key:row[key] for key in ('order','status','actual_backend','precision','accuracy')}
                                       for row in current[f'candidate-{backend}'] if row['point']==point]
                           for backend in backends} for point in range(1,101)}
    target_results={str(point):{str(backend):[row for row in current[f'candidate-{backend}'] if row['point']==point]
                                 for backend in backends} for point in target_points}
    return {'schema_version':1,'case':CASE,'campaign':'split-dp-terminal-1',
        'scope':'Offline comparison of the six closed EW n3 full100 streams; original numerical gates unchanged',
        'checkpoint_sha256':CHECKPOINT_SHA,'bound_inputs':files,
        'metrics':{'gate':'abs(a-b)/abs(a+b)<3e-6, or pole magnitude/finite maximum<3e-6',
                   'ordinary_target':'same criterion at1e-8','raw_relative':'abs(a-b)/max(abs(a),abs(b))',
                   'columns':COLUMNS},
        'original_comparisons':original,'pairwise_all100':pairwise,'accepted_usage':usage,
        'all_point_statuses':statuses,'target_points':target_points,'target_point_results':target_results,
        'new_native_evaluations':0,'performance_eligible':False,'seconds':None,
        'fpe_flags_observed':False,'internal_retries_observed':False,
        'higher_precision_adjudication':None,'analytic_pole_adjudication':None,
        'truth_scope':'Agreement between reducers is not an independent truth certificate; no finite or pole truth assignment',
        'accepted_status_scope':'Final accepted backend/precision only; internal attempts are not observed'}


def main():
    assert not sys.flags.optimize
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if not args.freeze:
        index=read(HERE/'index.json')
        assert set(index['files'])=={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name!='index.json'}
        for name,value in index['files'].items():
            path=HERE/name;assert not path.is_symlink() and path.resolve().is_relative_to(HERE)
            assert info(path)==value
    analysis=collect()
    if args.freeze:
        write(HERE/'analysis.json',analysis)
        files={str(p.relative_to(HERE)):info(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='index.json'}
        write(HERE/'index.json',{'schema_version':1,'scope':analysis['scope'],'new_native_evaluations':0,
                                'performance_eligible':False,'files':files})
    else:assert analysis==read(HERE/'analysis.json')
    print(json.dumps({'status':'passed','target_points':analysis['target_points'],
        'index_sha256':sha(HERE/'index.json'),'analysis_sha256':sha(HERE/'analysis.json'),
        'new_native_evaluations':0},sort_keys=True))


if __name__=='__main__':main()
