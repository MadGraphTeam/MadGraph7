#!/usr/bin/env python3
"""Derive the timing appendix data/CSV by replaying selected closed capsules.

No native process, compiler, loader, benchmark, or network is invoked here.
The controller's immutable analyzer objects remain unchanged in the envelope.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import types

IDS=(1,6,7,8,9)
NAMES={1:'CutTools',6:'Ninja',7:'COLLIER',8:'NinjaRS',9:'TIRust'}
ORDER=('qcd_n1_uux_z','qcd_n2_gg_ttx','ew_n2_epem_mupmum','qcd_n4_uux_epem_mupmum')
LABELS={'qcd_n1_uux_z':'uū → Z · QCD','qcd_n2_gg_ttx':'gg → tt̄ · QCD',
        'ew_n2_epem_mupmum':'e⁺e⁻ → μ⁺μ⁻ · EW','qcd_n4_uux_epem_mupmum':'uū → e⁺e⁻μ⁺μ⁻ · QCD'}
FIRST='experiments/current-source-five-backend-timing-1'
FIRST_SHA='3dc2c2b46295fce266706fd05b9444ee7408e41e3164daff77054189e2d79966'
FOURTH='experiments/current-source-four-lepton-timing-1'
FOURTH_SHA='2373175c50a62f0c5e0664c2337b438ae147ae31b1f4ff958ece374f041a2e89'
SELECTIONS=((FIRST,FIRST_SHA,ORDER[:3]),(FOURTH,FOURTH_SHA,ORDER[3:]))
ANALYZER_SHA='30721543c6184ca9a7002b155b6f70bee567663ae15ba1b85c136b5007b51619'
CMS='experiments/normal-checked-cms20-qualification-1'
CMS_SHA='f83d036d054adc58a3202c8614b7b9dcec549309b30be30390887f838822f3fc'
CMS_NOTE_SHA='d448e426390d6704ad533c1648db6eb3b27aa07ff29d5131e60282ecc9cea659'


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def read(path):
    def unique(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    def invalid(value):raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_replay(root,relative,expected,case_ids):
    require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'Pending or invalid capsule pin')
    relative=Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe selected capsule path')
    folder=Path(root)/relative;index_path=folder/'index.json'
    require(sha(index_path)==expected,'Selected timing seal changed')
    index=read(index_path)
    # Authenticate the exact verifier source before executing its pure replay.
    for name in ('replay/verify.py','source/analyze.py'):
        record=index['files'][name]
        require(sha(folder/name)==record['sha256'] and (folder/name).stat().st_size==record['bytes'],
                'Changed retained replay/analyzer source')
    require(sha(folder/'source/analyze.py')==ANALYZER_SHA,'Unexpected analyzer revision')
    path=folder/'replay/verify.py';module=types.ModuleType('selected_timing_replay');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    replay=module.verify(index_path,expected)
    require(tuple(row['case_id'] for row in replay['cases'])==tuple(case_ids),'Selected case order differs')
    require(replay['native_executions']==0 and replay['performance_acceptance'] is False,
            'Unexpected replay execution/acceptance scope')
    prepared=read(folder/'preparation.json');specification=read(folder/'specification.json');cases=[]
    for result in replay['cases']:
        case_id=result['case_id'];require(result['status']=='complete','Incomplete selected timing case')
        summary=result['summary'];original=prepared['originals'][case_id]
        case=original['case'];require(case['id']==case_id,'Case metadata differs')
        points=read(folder/'points'/f'{case_id}.json')
        require(sha(folder/'points'/f'{case_id}.json')==original['points_metadata_sha256'],
                'Point metadata differs')
        require(points['sha256']==original['points_sha256'] and len(points['points'])==100
                and summary['points_per_traversal']==100,'Saved cohort differs')
        energies=sorted({point['sqrt_s'] for point in points['points']})
        if case_id=='qcd_n1_uux_z':
            require(len(energies)==1 and energies[0]==points['inputs']['masses'][0]
                    and len({point['rapidity'] for point in points['points']})>1,'Wrong on-shell boost cohort')
            physics={'kind':'on-shell Z boosts','sqrt_s_GeV':energies[0],
                     'nominal_configuration_energy_is_physical_sqrt_s':False}
        else:
            require(energies==[1000.0],'Wrong 1 TeV cohort')
            physics={'kind':'ordinary 1 TeV points','sqrt_s_GeV':1000.0,
                     'nominal_configuration_energy_is_physical_sqrt_s':True}
        before=read(folder/'cases'/case_id/'1/candidate-inputs.json')
        source=before['generation']['source_identity']
        checks={};actual_cards={}
        for backend in IDS:
            lane=summary['backends'][str(backend)]
            lane_inputs=read(folder/'cases'/case_id/str(backend)/'candidate-inputs.json')
            actual_cards[str(backend)]={'numerical_cards':lane_inputs['numerical_cards'],
                'physical_card_sha256':lane_inputs['param_card_sha256'],'effective_settings':lane_inputs['settings']}
            require(type(lane['numerical_regression_pass']) is bool and type(lane['ordinary_target_passed']) is bool,
                    'Missing replayed numerical gate')
            checks[str(backend)]={'regression_pass':lane['numerical_regression_pass'],
                'ordinary_target_pass':lane['ordinary_target_passed'],
                'source':'pure replay of result.numerical_checks; learning, two calibrations, ten cycle final traversals',
                'retained_checks':13,'reference_role':result['reference_role']}
        cases.append({'case_id':case_id,'label':LABELS[case_id],'status':'closed',
            'model':case['model'],'process':case['process'],'nfinal':case['nfinal'],
            'summary':summary,'initialization_seconds':{str(b):summary['initialization'][str(b)]['cold_seconds'] for b in IDS},
            'numerical_validation':{'backends':checks,'coefficient_rows_scope':'final 100-point traversal only',
                'timed_status_scope':'all output calls in ten timed cycles; excludes learning/calibration'},
            'source_packet':{'path':relative.as_posix(),'index_name':'index.json','index_sha256':expected},
            'provenance':{'generation_source_identity_revision':source['revision'],
                'registered_scientific_revision':original['shared']['source_identity']['revision'],
                'exporter_inventory_sha256':source['files_sha256'],
                'source_identity_role':'retained generation receipt; not inferred from inventory SHA',
                'generation_receipt_sha256':original['generation_sha256'],
                'compiler':before['compilation']['compiler'],
                'effective_compiler_flags':before['compilation']['effective_compiler_flags'],
                'points_sha256':original['points_sha256'],
                'points_metadata_sha256':original['points_metadata_sha256'],
                'physical_card_sha256':original['physical_card_sha256'],
                'executable_sha256':original['executable_sha256'],
                'original_export_numerical_cards':original['numerical_cards'],
                'actual_timing_cards_by_requested_backend':actual_cards,
                'runtime':original['runtime']},'cohort':physics,
            'loop_topology':result.get('loop_topology'),
            'common_settings':{key:value for key,value in before['settings'].items() if key!='mlreductionlib'},
            'host':{key:specification['machine'][key] for key in
                ('cpu_model','governor','selected_cpu','thread_environment','uname')}})
    return cases,{'path':relative.as_posix(),'index_name':'index.json','index_sha256':expected},replay


def accepted(lane,precision):
    return sum(count for actual,count in lane['aggregate_backend_precision_counts'].items() if actual.endswith(':'+precision))


def checked_cms_note(root):
    folder=Path(root)/CMS;index_path=folder/'evidence-index.json'
    require(sha(index_path)==CMS_SHA,'Checked-CMS20 evidence seal changed')
    index=read(index_path)
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
            ==set(index['files'])|{'evidence-index.json'},'Checked-CMS20 inventory changed')
    for name,record in index['files'].items():
        path=folder/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()),'Unsafe checked-CMS20 payload')
        require(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Checked-CMS20 payload changed')
    path=Path(__file__).resolve().parent/'checked-cms20-note.json'
    require(sha(path)==CMS_NOTE_SHA,'Reviewed narrow CMS note changed')
    note=read(path)
    require(note['closed'] is True and note['authenticated'] is True
            and note['evidence_index_sha256']==CMS_SHA,'Unclosed checked-CMS20 note')
    return note,{'path':CMS,'index_name':'evidence-index.json','index_sha256':CMS_SHA}


def csv_bytes(data):
    import io
    out=io.StringIO(newline='')
    columns=('case_id','model','process','backend_id','backend','points_per_traversal',
        'median_ms_per_point','unscaled_MAD_ms_per_point','median_within_cycle_ratio_to_8',
        'accepted_DP_output_calls','accepted_QP_output_calls','READY_seconds',
        'numerical_3e-6_pass','ordinary_1e-8_target_pass','source_packet_index_sha256')
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    for case in data['cases']:
        for backend in IDS:
            key=str(backend);lane=case['summary']['backends'][key];check=case['numerical_validation']['backends'][key]
            writer.writerow({'case_id':case['case_id'],'model':case['model'],'process':case['process'],
                'backend_id':backend,'backend':NAMES[backend],'points_per_traversal':case['summary']['points_per_traversal'],
                'median_ms_per_point':format(lane['median_seconds_per_point']*1000,'.17g'),
                'unscaled_MAD_ms_per_point':format(lane['mad_seconds_per_point']*1000,'.17g'),
                'median_within_cycle_ratio_to_8':format(lane['median_ratio_to_8'],'.17g'),
                'accepted_DP_output_calls':accepted(lane,'DP'),'accepted_QP_output_calls':accepted(lane,'QP'),
                'READY_seconds':format(case['initialization_seconds'][key],'.17g'),
                'numerical_3e-6_pass':check['regression_pass'],'ordinary_1e-8_target_pass':check['ordinary_target_pass'],
                'source_packet_index_sha256':case['source_packet']['index_sha256']})
    return out.getvalue().encode()


def prepare(root):
    require(all(len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
                for _,expected,_ in SELECTIONS),'All selected timing capsules must be closed and pinned before preparation')
    cases=[];evidence=[];replays=[]
    for relative,expected,case_ids in SELECTIONS:
        selected,packet,replay=selected_replay(root,relative,expected,case_ids)
        cases.extend(selected);evidence.append(packet);replays.append(replay)
    require(tuple(c['case_id'] for c in cases)==ORDER,'All four requested cases are required')
    cms,cms_packet=checked_cms_note(root);evidence.append(cms_packet)
    registered={c['provenance']['registered_scientific_revision'] for c in cases}
    require(len(registered)==1,'Registered scientific source identities differ')
    require(all(c['host']==cases[0]['host'] for c in cases),'Host metadata differs; require explicit report labels')
    require(all(c['common_settings']==cases[0]['common_settings'] for c in cases),'Common numerical policy differs')
    host=cases[0]['host'];compiler=cases[0]['provenance']['compiler']
    durations=[{'case_id':c['case_id'],'minimum_seconds':min(s['seconds'] for lane in c['summary']['backends'].values() for s in lane['samples']),
                'maximum_seconds':max(s['seconds'] for lane in c['summary']['backends'].values() for s in lane['samples'])} for c in cases]
    data={'schema':'madloop-five-backend-timing-report-data-v1','status':'closed','date':'2026-11-11',
        'source_label':'registered science '+next(iter(registered))[:10],
        'evidence_label':'Closed timing capsules; see the accompanying README and scientific-selection.json',
        'machine_note':compiler+'; '+host['cpu_model']+'; affinity CPU '+str(host['selected_cpu'])+
            '; one-thread environment on a shared, unreserved host.',
        'actual_timed_batch_durations':durations,
        'evidence':evidence,'cases':cases,'checked_cms20':cms,'performance_acceptance':False,'confidence_intervals':False,
        'new_native_physics_evaluations':0,
        'cohort_note':'Four independent saved 100-point cohorts are used without subsetting. All timed output calls and all final-traversal coefficient checks remain in their source capsules.'}
    return data,replays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'Fresh output directory required')
    data,replays=prepare(args.root.resolve())
    args.output.mkdir(parents=True)
    (args.output/'timing-report-data.json').write_bytes(encoded(data))
    (args.output/'timing-summary.csv').write_bytes(csv_bytes(data))
    (args.output/'scientific-selection.json').write_bytes(encoded({'closed':True,'case_ids':list(ORDER),'packets':data['evidence']}))
    (args.output/'timing-replay.json').write_bytes(encoded({'replays':replays,'new_native_physics_evaluations':0}))
    print(encoded({'closed':True,'cases':len(data['cases']),'CSV_rows':20,'new_native_physics_evaluations':0}).decode(),end='')


if __name__=='__main__':main()
