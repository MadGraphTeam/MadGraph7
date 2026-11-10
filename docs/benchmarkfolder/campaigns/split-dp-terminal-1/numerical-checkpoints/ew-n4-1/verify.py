#!/usr/bin/env python3
"""Freeze/reparse the closed EW n4 numerical case; no native execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
SUPERVISOR = REPORT/'numerical-qualification-2'
MANIFEST = '1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31'
CASES = ['ew_n4_epem_mupmumaa']
LANES = [('baseline',1), *[('candidate',b) for b in (1,6,7,8,9)]]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes':path.stat().st_size,'sha256':sha(path)}


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def collect(closed):
    spec = read(SUPERVISOR/'specification.json')
    config = read(REPORT/'cases.json')
    assert sha(REPORT/'manifest.json') == MANIFEST == spec['manifest_sha256']
    assert sha(SUPERVISOR/'run.py') == spec['supervisor_sha256']
    assert sha(SUPERVISOR/'process_cleanup.py') == spec['process_cleanup_sha256']
    assert [x['case'] for x in spec['schedule']].count(CASES[0]) == 1
    # Import only after the saved specification proves the helper closure.
    for name,value in spec['protocols'].items():
        assert sha(ROOT/'tools'/name) == value
    sys.path.insert(0,str(ROOT/'tools'))
    from tirust_benchmark_warmup import parse_warmup
    from tirust_benchmark import compare_rows
    files = {HERE/'verify.py',HERE/'README.md',HERE/'closed-cases.json',REPORT/'manifest.json',
             REPORT/'cases.json',REPORT/'preparation-1/index.json',REPORT/'preparation-1/results.json',
             SUPERVISOR/'run.py',SUPERVISOR/'process_cleanup.py',SUPERVISOR/'specification.json',
             SUPERVISOR/'configuration-initial.json'}
    summaries=[]
    for case_id in CASES:
        case=next(c for c in config['cases'] if c['id']==case_id)
        state=next(v for v in closed if v['case']==case_id)
        assert state['status']=='passed' and state['returncode']==0 and state['finished_utc']
        schedule=next(v for v in spec['schedule'] if v['case']==case_id)
        directory=REPORT/'experiments/numerical'/schedule['label']
        case_spec, result=read(directory/'specification.json'),read(directory/'results.json')
        assert state['results_sha256']==sha(directory/'results.json')
        assert state['results_path']==str((directory/'results.json').relative_to(REPORT))
        log=SUPERVISOR/(case_id+'.log'); assert state['console_sha256']==sha(log)
        assert case_spec['campaign_binding']=={'id':'split-dp-terminal-1','manifest_sha256':MANIFEST}
        assert case_spec['case']==case and case_spec['configuration_sha256']==sha(REPORT/'cases.json')
        assert case_spec['measurement_phase']=='steady-warmup' and case_spec['performance_eligible'] is False
        assert case_spec['worker_timeout_seconds']==spec['worker_timeouts'][case_id]
        assert case_spec['protocols']==spec['protocols']
        for name,value in spec['protocols'].items():assert sha(directory/name)==value
        assert sha(directory/'cases.json')==sha(REPORT/'cases.json')
        assert sha(directory/'campaign-manifest.json')==MANIFEST
        assert result['case']==case and result['performance_eligible'] is False
        assert [(x['source'],x['backend']) for x in result['results']]==LANES
        files.add(log)
        for p in directory.rglob('*'):
            if p.is_file():files.add(p)
        point=REPORT/'points'/(case_id+'.dat'); meta=point.with_suffix('.json')
        files.update([point,meta]); assert len(read(meta)['points'])==100
        reference=None;lanes=[]
        for source,backend in LANES:
            lane=directory/f'{source}-{backend}';value=read(lane/'result.json')
            assert value==next(v for v in result['results'] if(v['source'],v['backend'])==(source,backend))
            assert value['status']=='complete' and value['performance_eligible'] is False
            assert not any(lane.rglob('failure.json'))
            inputs=read(lane/(source+'-inputs.json'));after=read(lane/'inputs-after.json')
            binding=spec['compiled_inputs'][source+'/'+case_id]
            retained=REPORT/'inputs'/source/case_id
            assert inputs['generation']==read(retained/'generation.json')
            assert inputs['compilation']==read(retained/'compilation.json')
            assert sha(retained/'generation.json')==binding['generation_sha256']
            assert sha(retained/'compilation.json')==binding['compilation_sha256']
            assert inputs['compilation']['executable_sha256']==binding['executable_sha256']==after['executable_sha256']
            assert inputs['points_sha256']==binding['points_sha256']==after['points_sha256']==sha(point)
            assert inputs['points_metadata_sha256']==binding['points_metadata_sha256']==sha(meta)
            assert inputs['param_card_sha256']==sha(retained/'param_card.dat')
            assert inputs['numerical_cards']==after['numerical_cards']
            assert inputs['worker_timeout_seconds']==case_spec['worker_timeout_seconds']
            assert inputs['settings']['mlreductionlib']==str(backend)
            expected={k.lower():v for k,v in config['madloop_parameters'].items()}
            if source=='candidate':expected.update({k.lower():v for k,v in config['candidate_parameters'].items()})
            assert all(inputs['settings'].get(k)==v for k,v in expected.items())
            assert all(inputs['runtime'][k]==after['runtime'][k] for k in ('libraries','archives','metadata'))
            assert inputs['compilation']['link_inputs_before']==inputs['compilation']['link_inputs_after']
            assert all(inputs['runtime'][k]==inputs['compilation']['link_inputs_before'][k] for k in ('archives','metadata'))
            files.update(retained/name for name in ('generation.json','compilation.json','param_card.dat'))
            raw=lane/'raw'; observation=parse_warmup(raw,[backend],100,read(raw/'ready.json'))
            assert observation==value['observation']==read(raw/'observation.json')
            assert observation['seconds'] is None and observation['performance_eligible'] is False
            assert len({v['point'] for v in observation['rows']})==100
            if reference is None:reference=observation
            comparison=compare_rows(reference['rows'],observation['rows'],config['relative_tolerance'])
            assert comparison==value['comparison'] and comparison['passed']
            lanes.append({'source':source,'backend':backend,'points':100,'passed':True,
                'ordinary_target_passed':comparison['ordinary_target_passed'],
                'ordinary_target_missed_coefficient_comparisons':sum(not ((x['symmetric_relative_error'] is not None and x['symmetric_relative_error']<1e-8) or (x['pole_to_finite'] is not None and x['pole_to_finite']<1e-8)) for x in comparison['residuals']),
                'status_counts':observation['status_counts'],
                'accepted_backend_precision_counts':observation['backend_precision_counts'],
                'fpe_flags':None,'internal_retry_counts':None,'seconds':None,'performance_eligible':False})
        summaries.append({'case':case,'passed':True,'lanes':lanes})
    inventory={}
    for path in sorted(files):
        assert path.is_file() and not path.is_symlink() and path.suffix!='.pyc'
        inventory[str(path.relative_to(REPORT))]=info(path)
    return {'schema':'madloop-tirust-completed-ew-n4-numerical-v1','case_count':1,'lane_count':6,
        'saved_points_per_lane':100,'case_ids':CASES,'cases':summaries,'manifest_sha256':MANIFEST,
        'scope':'Closed EW n4 numerical case only; other cases remain outside this checkpoint; no timing qualification',
        'full_matrix_complete':False,'performance_acceptance_eligible':False,'overall_goal_qualified':False,
        'fpe_flags_observed':False,'internal_retry_counts_observed':False},inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    assert sys.flags.optimize==0
    if args.freeze:
        assert not (HERE/'index.json').exists()
        status=read(SUPERVISOR/'status.json')
        closed=[next(v for v in status['cases'] if v['case']==case_id) for case_id in CASES]
        assert all(v['status']=='passed' for v in closed)
        write(HERE/'closed-cases.json',closed)
        summary,files=collect(closed);write(HERE/'summary.json',summary)
        files[str((HERE/'summary.json').relative_to(REPORT))]=info(HERE/'summary.json')
        write(HERE/'index.json',{'schema':'madloop-tirust-closed-numerical-checkpoint-v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),'root':'../..','files':files,
            'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),
            'case_count':1,'lane_count':6,'full_matrix_complete':False,'performance_acceptance_eligible':False})
    index=read(HERE/'index.json')
    for name,value in index['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        assert info(REPORT/path)==value
    summary,files=collect(read(HERE/'closed-cases.json'))
    assert summary==read(HERE/'summary.json')
    files[str((HERE/'summary.json').relative_to(REPORT))]=info(HERE/'summary.json')
    assert files==index['files'] and len(files)==index['file_count']
    assert sum(x['bytes'] for x in files.values())==index['total_bytes']
    assert not any('/full100-warmup-qcd_n4_' in n or n.endswith('numerical-qualification-2/status.json') for n in files)
    print(f'PASS {len(files)} retained files; one EW n4 case /6 lanes /100 points; no native calls or timing qualification')

if __name__=='__main__':main()
