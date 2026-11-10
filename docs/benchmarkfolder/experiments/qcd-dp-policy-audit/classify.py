#!/usr/bin/env python3
"""Pure retained-journal join; no numerical library, subprocess or native call."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PACKAGE=ROOT/'docs/benchmarkfolder/experiments/dp-rescue-cause-observer/numerical-qualification-1'
INDEX_SHA='9f934ced6cfe06d2f01252b8a86d6c5f50cd7bd29890fd2bf78b9e5db40769e2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def derive():
    assert sha(PACKAGE/'index.json')==INDEX_SHA
    index=json.loads((PACKAGE/'index.json').read_text())
    entry=index['inventory']['qcd_n3_uux_zgg/enabled/cause-native.jsonl']
    assert sha(PACKAGE/entry['payload'])==index['files'][entry['payload']]['sha256']
    h=hashlib.sha256();size=0;calls={};rows=[];options=Counter();tiers=0
    with gzip.open(PACKAGE/entry['payload'],'rb') as stream:
        for line in stream:
            h.update(line);size+=len(line);e=json.loads(line)
            if e['kind']!='event':continue
            d=e['data'];kind=d['kind'];number=e['native']
            if kind=='native_begin':
                assert number not in calls
                calls[number]={'native':number,'phase':e['phase'],'point':e['point'],'begin':d,'diagnostics':[]}
            elif number and kind=='diagnostic':calls[number]['diagnostics'].append(d['text'])
            elif number and kind=='execution':calls[number]['execution']=d
            elif number and kind=='options':
                options[json.dumps(d,sort_keys=True)]+=1
                assert d['method_choice'] and not d['partial_rescue'] and not d['precision_tiers']
                assert d['gate_bits'] is None and d['choice_gate_bits'] is None and d['master_policy']=='Default'
            elif number and kind=='tier_attempt':tiers+=1
            elif kind=='native_end':
                row=calls.pop(number)
                if row['phase']!=1 or row['begin']['input_bits']!=53:continue
                execution=row.get('execution',{})
                if not execution.get('expansion_rows'):continue
                assert execution['expansion_rows']==1 and row['diagnostics']
                text=row['diagnostics'][0]
                match=re.fullmatch(r'direct engine: ill-conditioned Gram matrix for \[([\d, ]+)\]: conditioning ([+\-\deE.]+)',text)
                assert match,text
                members=[int(x.strip()) for x in match[1].split(',')]
                assert len(set(members))==len(members) and all(0<=x<row['begin']['n'] for x in members)
                assert 0<float(match[2])<float('inf')
                top=len(members)==row['begin']['n']
                assert len(members) in (3,4)
                row.update(status=d['status'],error=d['error'],first_refused_members=members,
                           first_refusal_condition_decimal=match[2],top_subset=top,
                           partial_dd_static_shape_candidate=not top,
                           actual_partial_dd_attempt_observed=False)
                assert d['status'] in (0,1) and execution['expansion_successes']==int(d['status']==0)
                rows.append(row)
    assert not calls and h.hexdigest()==entry['raw_sha256'] and size==entry['raw_bytes']
    assert tiers==0 and len(rows)==454
    counts=Counter((r['status'],r['begin']['n'],r['begin']['rank'],len(r['first_refused_members']),r['top_subset']) for r in rows)
    failed=[r for r in rows if r['status']!=0]
    eligible_points=sorted({r['point'] for r in failed if r['partial_dd_static_shape_candidate']})
    excluded_points=sorted({r['point'] for r in failed if not r['partial_dd_static_shape_candidate']})
    return {'schema':'qcd-dp-policy-source-join-v1','raw_entry':entry,'raw_package_index_sha256':INDEX_SHA,
            'bootstrap_excluded':True,'new_native_execution':False,'performance_eligible':False,
            'method_choice_rows':len(rows),'method_choice_successes':sum(r['status']==0 for r in rows),
            'final_refusals':len(failed),'actual_whole_lane_tier_events':tiers,
            'partial_dd_static_shape_candidate_refusals':sum(r['partial_dd_static_shape_candidate'] for r in failed),
            'top_subset_excluded_refusals':sum(r['top_subset'] for r in failed),
            'candidate_refusal_points':eligible_points,'top_excluded_refusal_points':excluded_points,
            'candidate_points_without_a_recorded_top_refusal':sorted(set(eligible_points)-set(excluded_points)),
            'option_records':[{'record':json.loads(k),'count':v} for k,v in sorted(options.items())],
            'counts':[{'status':k[0],'n':k[1],'rank':k[2],'subset_size':k[3],'top_subset':k[4],'count':v} for k,v in sorted(counts.items())],
            'limitations':['First direct failure only; not a complete geometry walk or successful DD prediction.',
                          'Partial-DD is disabled; shape eligibility does not prove it will reach or pass a retry.',
                          'Counts are visits, not unique integrals, time, saved amplitude rescues, or higher-precision work.'],
            'rows':rows}

if __name__=='__main__':
    result=derive()
    if sys.argv[1:]==['--write']:(HERE/'classification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:assert result==json.loads((HERE/'classification.json').read_text())
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','option_records','counts','limitations','raw_entry')},sort_keys=True))
