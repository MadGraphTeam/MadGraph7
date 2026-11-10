#!/usr/bin/env python3
"""Offline tighter-target analysis of two cases from one sealed public capsule."""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import tempfile

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[1]
PACKAGE=DOCS/'campaigns/invariant-symmetry-1/numerical-retention-1/sealed-1'
EXPECTED='8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be'
GUARD=DOCS/'campaigns/invariant-symmetry-1/numerical-retention-audit-1'
GUARD_INDEX='a8e11f69012d96a6013c0e0ee4dabd8e1f06dbf1327cbd7fba14f6b948eb59c9'
CASES=('qcd_n4_uux_zzgg','ew_n4_epem_mupmumaa')
COLS=('Born','finite','single_pole','double_pole')
BACKENDS=(1,6,7,8,9)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def symmetric(a,b): return abs(a-b)/abs(a+b) if a+b else (0.0 if a==b else None)
def passed(residual,tol):
    s=residual['symmetric_relative_error'];p=residual['pole_to_finite']
    return (s is not None and s<tol) or (p is not None and p<tol)
def bitwise_row(row):
    return {k:([x.hex() for x in v] if k=='coefficients' else v.hex() if isinstance(v,float) else v)
            for k,v in row.items() if k!='order'}
def rows(lane):
    allrows=lane['observation']['rows'];out={}
    assert len(allrows)==200
    for point in range(1,101):
        paired=[r for r in allrows if r['point']==point];assert len(paired)==2
        assert {r['order'] for r in paired}=={0,1}
        assert bitwise_row(paired[0])==bitwise_row(paired[1])
        out[point]=next(r for r in paired if r['order']==0)
    return out
def detail(point,column,left,right):
    a=left[point]['coefficients'][column];b=right[point]['coefficients'][column]
    return {'point':point,'coefficient':COLS[column],'left_value':a,'right_value':b,
      'left_hex':a.hex(),'right_hex':b.hex(),'signed_right_minus_left':b-a,'absolute_difference':abs(a-b),
      'symmetric_relative_error':symmetric(a,b),'left_precision':left[point]['precision'],
      'right_precision':right[point]['precision'],'left_status':left[point]['status'],'right_status':right[point]['status']}
def worst(column,left,right):
    return max((detail(p,column,left,right) for p in left),key=lambda r:math.inf if r['symmetric_relative_error'] is None else r['symmetric_relative_error'])
def components(values,tolerance):
    # Deterministic connected components are only proximity descriptions.
    remaining=set(values);groups=[]
    while remaining:
        group={min(remaining)};remaining-=group
        while True:
            added={b for b in remaining if any(symmetric(values[b],values[a]) is not None and symmetric(values[b],values[a])<tolerance for a in group)}
            if not added: break
            group|=added;remaining-=added
        groups.append(sorted(group))
    return groups

def analyze_case(case,result,compare):
    lanes={(x['source'],x['backend']):x for x in result['results']}
    assert set(lanes)=={('baseline',1),*(('candidate',b) for b in BACKENDS)}
    baseline=rows(lanes['baseline',1]);current={b:rows(lanes['candidate',b]) for b in BACKENDS}
    assert {p:bitwise_row(r) for p,r in baseline.items()}=={p:bitwise_row(r) for p,r in current[1].items()}
    distributions={}
    for b in BACKENDS:
        lane=lanes['candidate',b];actual=compare(lanes['baseline',1]['observation']['rows'],lane['observation']['rows'],3e-6)
        assert actual==lane['comparison'] and actual['passed'] is True
        by_coefficient={}
        for column,name in enumerate(COLS):
            residuals=[r for r in actual['residuals'] if r['order']==0 and r['column']==column]
            misses=[r for r in residuals if not passed(r,1e-8)]
            accepted={precision:sorted(p for p,r in current[b].items() if r['precision']==precision) for precision in ('DP','QP')}
            by_coefficient[name]={'physical_point_misses':len(misses),'public_answer_row_misses':2*len(misses),
              'points':[r['point'] for r in misses],
              'misses_by_candidate_accepted_precision':{precision:[r['point'] for r in misses if current[b][r['point']]['precision']==precision] for precision in ('DP','QP')},
              'misses_by_baseline_accepted_precision':{precision:[r['point'] for r in misses if baseline[r['point']]['precision']==precision] for precision in ('DP','QP')},
              'worst':worst(column,baseline,current[b])}
        distributions[str(b)]={'regression_pass':True,'ordinary_target_pass':actual['ordinary_target_passed'],
          'accepted_precision_point_counts':{p:len(ps) for p,ps in accepted.items()},'accepted_precision_points':accepted,
          'status_counts':dict(collections.Counter(str(r['status']) for r in current[b].values())),
          'coefficient_misses':by_coefficient}
    pairs={}
    for a,b in itertools.combinations((6,7,8,9),2):
        per_column={name:worst(col,current[a],current[b]) for col,name in enumerate(COLS)}
        finite=[detail(p,1,current[a],current[b]) for p in baseline]
        pairs[f'{a}-{b}']={'worst_by_coefficient':per_column,
          'finite_median_symmetric_relative_error':statistics.median(x['symmetric_relative_error'] for x in finite),
          'finite_points_at_or_above_1e8':[x['point'] for x in finite if x['symmetric_relative_error'] is None or x['symmetric_relative_error']>=1e-8],
          'finite_top5':sorted(finite,key=lambda x:x['symmetric_relative_error'],reverse=True)[:5]}
    cluster_counts=collections.Counter();cluster_points={}
    all_nonct=[];point_records=[]
    for point in baseline:
        values={b:current[b][point]['coefficients'][1] for b in (6,7,8,9)}
        groups=components(values,1e-8);label=' | '.join(','.join(map(str,g)) for g in groups)
        cluster_counts[label]+=1;cluster_points.setdefault(label,[]).append(point)
        ct={str(b):{'candidate_backend':b,**detail(point,1,baseline,current[b])} for b in (6,7,8,9)}
        pairdiffs=[{'pair':f'{a}-{b}',**detail(point,1,current[a],current[b])} for a,b in itertools.combinations((6,7,8,9),2)]
        nonct=max(pairdiffs,key=lambda x:x['symmetric_relative_error']);all_nonct.append(nonct)
        point_records.append({'point':point,'baseline_precision':baseline[point]['precision'],
          'candidate_precisions':{str(b):current[b][point]['precision'] for b in BACKENDS},
          'baseline_comparisons':ct,'max_nonct_pair':nonct,'nonct_proximity_components_at_1e8':groups})
    finite_miss_union=sorted(set().union(*(set(distributions[str(b)]['coefficient_misses']['finite']['points']) for b in (6,7,8,9))))
    common=set(distributions['6']['coefficient_misses']['finite']['points'])
    for b in (7,8,9): common&=set(distributions[str(b)]['coefficient_misses']['finite']['points'])
    selected=set(x['point'] for x in sorted(point_records,key=lambda x:max(y['symmetric_relative_error'] for y in x['baseline_comparisons'].values()),reverse=True)[:5])
    selected|=set(x['point'] for x in sorted(all_nonct,key=lambda x:x['symmetric_relative_error'],reverse=True)[:5])
    return {'case':case,'regression_all_lanes_pass':True,'original_candidate_CutTools_bit_and_status_identity':True,
      'point_count':100,'public_answer_orders':[0,1],'orders_duplicate_exactly':True,
      'miss_count_convention':'physical points; answer rows are exactly two identical orders per point',
      'backend_names':{'1':'CutTools','6':'Ninja','7':'COLLIER','8':'NinjaRS','9':'TIRust'},
      'baseline_accepted_precision_counts':dict(collections.Counter(r['precision'] for r in baseline.values())),
      'candidate_vs_baseline':distributions,'nonCT_pairs':pairs,
      'finite_miss_union_points':finite_miss_union,'finite_miss_common_to_all_nonCT':sorted(common),
      'nonCT_finite_proximity_components_at_1e8':{'counts':dict(cluster_counts),'point_ids':cluster_points,
        'definition':'connected components of pairwise distances below1e-8; transitivity does not imply every pair lies below threshold'},
      'worst_CT_to_nonCT':max((v for x in point_records for v in x['baseline_comparisons'].values()),key=lambda x:x['symmetric_relative_error']),
      'worst_nonCT_pair':max(all_nonct,key=lambda x:x['symmetric_relative_error']),
      'selected_largest_discrepancy_points':[x for x in point_records if x['point'] in selected]}

def calculate():
    indexraw=(PACKAGE/'index.json').read_bytes();assert sha(indexraw)==EXPECTED
    guardindex=(GUARD/'index.json').read_bytes();assert sha(guardindex)==GUARD_INDEX
    guardfiles=json.loads(guardindex)['files'];assert sha((GUARD/'audit.py').read_bytes())==guardfiles['audit.py']['sha256']
    guard=module('published_read_only_guard',GUARD/'audit.py')
    with tempfile.TemporaryDirectory(prefix='current-n4-offline-') as t:
        with guard.confined(PACKAGE,Path(t),'closed_n4_analysis'):
            ret=module('sealed_main_retainer',PACKAGE/'retain.py');index,store=ret.load(PACKAGE,EXPECTED)
            replay=ret.replay_completed(PACKAGE,EXPECTED)
            collector=ret.load_collector(PACKAGE)
            raw=(PACKAGE/'protocol-tirust_benchmark.py').read_bytes()
            compare=collector._functions(raw,('compare_rows',),{},'frozen-benchmark')['compare_rows']
            cases={};inputs={}
            for case in CASES:
                name=f'docs/benchmarkfolder/campaigns/invariant-symmetry-1/experiments/numerical/full100-warmup-{case}-1/results.json'
                inputs[name]=index['retained_files'][name];assert replay[case]['results_sha256']==inputs[name]['sha256']
                cases[case]=analyze_case(case,store.json(name),compare)
            assert all(x['operation']=='lstat' and x.get('path') in ('/portable-checkpoint','/portable-checkpoint/docs','/portable-checkpoint/docs/benchmarkfolder') for x in guard.STATE['blocked'])
    return {'schema':'closed-current-n4-tighter-target-analysis-v1','sealed_package_index_sha256':EXPECTED,
      'retained_result_inputs':inputs,'parser_source_sha256':sha(raw),'read_guard_index_sha256':GUARD_INDEX,
      'regression_threshold':3e-6,'tighter_target':1e-8,'metric':'abs(a-b)/abs(a+b); frozen negligible-pole criterion retained',
      'cases':cases,'limitations':['Pairwise agreement and clustering do not establish truth.',
        'Accepted DP/QP status is recorded; internal attempts and hardware IEEE state are unobserved.',
        'No new evaluation, native load, source change, policy change or case qualification was performed.',
        'No previous QCDn4 forced-QP implementation or private diagnostic result is an input.',
        'Remaining tighter-target investigation requires separately admitted independent higher-precision/common-input evidence.']}

def summarize(analysis):
    result={'sealed_package_index_sha256':EXPECTED,'regression_all_pass':True,'tighter_target':1e-8,
      'precision_meaning':'accepted host output; no internal-attempt or hardware-IEEE inference','cases':{}}
    for name,c in analysis['cases'].items():
        totals={};counts={};worst_per_column={}
        for b,lane in c['candidate_vs_baseline'].items():
            totals[b]=lane['accepted_precision_point_counts']
            counts[b]={coefficient:{'total':r['physical_point_misses'],
              'DP':len(r['misses_by_candidate_accepted_precision']['DP']),
              'QP':len(r['misses_by_candidate_accepted_precision']['QP'])}
              for coefficient,r in lane['coefficient_misses'].items()}
        for coefficient in COLS:
            worst_per_column[coefficient]=max(({'backend':int(b),**lane['coefficient_misses'][coefficient]['worst']}
              for b,lane in c['candidate_vs_baseline'].items()),key=lambda x:x['symmetric_relative_error'])
        badpairpoints=set().union(*(set(v['finite_points_at_or_above_1e8']) for v in c['nonCT_pairs'].values()))
        result['cases'][name]={'accepted_precision_counts':totals,'physical_point_misses_by_coefficient_and_precision':counts,
          'worst_vs_baseline_by_coefficient':worst_per_column,'worst_nonCT_finite_pair':c['worst_nonCT_pair'],
          'finite_miss_union_count':len(c['finite_miss_union_points']),
          'finite_common_miss_count':len(c['finite_miss_common_to_all_nonCT']),
          'all_nonCT_finite_pairs_below_1e8_points':100-len(badpairpoints),
          'nonCT_finite_pair_exception_points':sorted(badpairpoints),
          'nonCT_finite_pair_maxima':{pair:{'point':v['worst_by_coefficient']['finite']['point'],
             'symmetric_relative_error':v['worst_by_coefficient']['finite']['symmetric_relative_error']}
             for pair,v in c['nonCT_pairs'].items()}}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        index=json.loads((HERE/'index.json').read_bytes())
        assert index['sealed_package_index_sha256']==EXPECTED and index['read_guard_index_sha256']==GUARD_INDEX
        assert {p.name for p in HERE.iterdir() if p.is_file()}==set(index['files'])|{'index.json'}
        for name,entry in index['files'].items():
            assert Path(name).name==name and not (HERE/name).is_symlink()
            raw=(HERE/name).read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
    value=calculate();target=HERE/'analysis.json';summary=summarize(value)
    if args.verify:
        assert json.loads(target.read_bytes())==value
        assert json.loads((HERE/'summary.json').read_bytes())==summary
    else:
        with target.open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        with (HERE/'summary.json').open('x') as f: json.dump(summary,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print('PASS: exact sealed-input analysis of QCD n4 and EW n4; no native execution.')
if __name__=='__main__': main()
