#!/usr/bin/env python3
"""Independent exact-rational convergence/changed-history audit; no engine."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

POLICIES = ('Interpreted','NativeDenominators','FullNative')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(value):
    return {'numerator':str(value.numerator),'denominator':str(value.denominator),
        'approximate':float(value)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    args=parser.parse_args()
    experiment=args.experiment.resolve(strict=True)
    spec=importlib.util.spec_from_file_location('independent_replay_audit',Path(__file__).with_name('replay_audit.py'))
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    replay_path=experiment/'run-replay/raw.jsonl'
    replay=[json.loads(line) for line in replay_path.read_text().splitlines()]
    assert len(replay)==1393 and replay[-1]['complete'] and replay[-1]['kind']=='summary'
    replay.pop()
    selected={r['sequence'] for r in replay if any(a['raw']!=r['arms'][0]['raw']
        or a['output']!=r['arms'][0]['output'] or a['status']!=r['arms'][0]['status']
        for a in r['arms'][1:])}
    selected.add('physical_complex_mass_pentagon')
    stage=experiment/'run-convergence'
    result_path=stage/'result.json'
    result=json.loads(result_path.read_text())
    raw=stage/'raw.jsonl'
    assert result['complete'] and result['returncode']==0 and result['stage']=='convergence'
    assert result['files']['raw.jsonl']==sha(raw)
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    assert rows[-1]==result['summary']
    summary=rows.pop()
    assert summary['kind']=='summary' and summary['complete']
    assert summary['failed_arms']==0 and summary['requests_per_precision']==len(selected)
    fixture=json.loads((experiment/'source/tests/data/convergence.json').read_text())['input']
    fixture['masses']=fixture['masses_squared']
    fixture['scale']=fixture['scale_squared']
    by_key={}
    split_failures=[]
    changed_history=[]
    for row in rows:
        assert row['kind']=='convergence'
        key=row['sequence'],row['working_bits']
        assert key not in by_key and key[0] in selected and key[1] in (256,384)
        assert [a['policy'] for a in row['arms']]==list(POLICIES)
        expected_input=fixture if isinstance(key[0],str) else replay[key[0]]['input']
        assert row['input']==expected_input
        for arm in row['arms']:
            assert arm['working_bits']==key[1] and arm['status']==0 and arm['input_unchanged']
            assert len(arm['output'])==8 and arm['formatted_buffer_bytes']==8*64
            assert len(arm['raw'])==10
            values=[audit.dyadic(c) for c in arm['raw']]
            norm=max(map(abs,values[:6]))
            if any(abs(values[i]-values[i+6]-values[i+8])>Fraction(1,10**60)*norm for i in (0,1)):
                split_failures.append([key[0],key[1],arm['policy']])
            if key[1]==256 and not isinstance(key[0],str):
                reference=replay[key[0]]['arms'][POLICIES.index(arm['policy'])]
                if arm!=reference:changed_history.append([key[0],arm['policy']])
        by_key[key]=row
    assert set(by_key)=={(s,b) for s in selected for b in (256,384)}
    summaries={p:{'convergence_failed':0,'cross384_failed':0,'convergence_sign_changed':0,
        'cross384_sign_changed':0,'cross384_precision_changed':0,'cross384_nonzero_value':0,
        'max_convergence_ratio':Fraction(0),'max_cross384_ratio':Fraction(0)} for p in POLICIES}
    discrepancies=[]
    for sequence in sorted(selected,key=str):
        low,high=[by_key[sequence,b] for b in (256,384)]
        high_reference=high['arms'][0]
        for index,policy in enumerate(POLICIES):
            for kind,left,right in [('convergence',low['arms'][index],high['arms'][index]),
                ('cross384',high['arms'][index],high_reference)]:
                a,b=[[audit.dyadic(c) for c in x['raw']] for x in (left,right)]
                differences=[abs(x-y) for x,y in zip(a,b)]
                norm=max(map(abs,b[:6]))
                bound=Fraction(1,10**31)*norm
                failed=any(d>bound for d in differences[:6])
                sign=[i for i,(x,y) in enumerate(zip(left['raw'],right['raw'])) if x['negative']!=y['negative']]
                prec=[i for i,(x,y) in enumerate(zip(left['raw'],right['raw'])) if x['precision']!=y['precision']]
                s=summaries[policy]
                s[kind+'_failed']+=failed
                s[kind+'_sign_changed']+=bool(sign)
                if kind=='cross384':
                    s['cross384_precision_changed']+=bool(prec)
                    s['cross384_nonzero_value']+=any(differences)
                if norm:s['max_'+kind+'_ratio']=max(s['max_'+kind+'_ratio'],max(differences[:6])/norm)
                if failed or sign or (kind=='cross384' and prec):
                    discrepancies.append({'sequence':sequence,'policy':policy,'kind':kind,
                        'failed_value_screen':failed,'sign_changed_indices':sign,'precision_changed_indices':prec})
    for s in summaries.values():
        for key in ('max_convergence_ratio','max_cross384_ratio'):s[key]=ratio(s[key])
    print(json.dumps({'schema':'independent-native-convergence-audit-v1','complete':True,
        'engine_calls':0,'performance_claim':False,'audit_source_sha256':sha(Path(__file__)),
        'dyadic_audit_source_sha256':sha(Path(__file__).with_name('replay_audit.py')),
        'raw_sha256':sha(raw),'replay_sha256':sha(replay_path),'run_result_sha256':sha(result_path),
        'requests_per_precision':len(selected),'records':len(rows),
        'changed_history_256_arm_count':len(changed_history),'changed_history_256_arms':changed_history,
        'finite_split_failures':split_failures,'summaries':summaries,
        'automatic_sign_or_precision_difference_acceptance':False,'discrepancies':discrepancies},indent=2))


if __name__=='__main__':main()
