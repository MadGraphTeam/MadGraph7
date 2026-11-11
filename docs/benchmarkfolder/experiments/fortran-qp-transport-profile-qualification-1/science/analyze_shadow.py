#!/usr/bin/env python3
"""Read-only exact-key opportunity accounting; does not simulate a cache gain."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(prefix):
    prefix = Path(prefix)
    rows = {}
    for line in prefix.with_suffix('.calls.tsv').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 24:
            raise ValueError('Unexpected call record shape')
        numbers = list(map(int, fields[:-1]))
        number, kind, call, epoch = numbers[:4]
        if number in rows:
            raise ValueError('Repeated trace row')
        rows[number] = dict(kind=kind, call=call, epoch=epoch, key=tuple(numbers[4:8]),
            options=tuple(numbers[8:11]), budget=numbers[11], generation=numbers[12],
            status=numbers[13], valid=numbers[14], stats=numbers[15:21],
            offset=numbers[21], length=numbers[22], label=fields[-1], payload=bytearray())
    for line in prefix.with_suffix('.shadow.hex').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError('Unexpected shadow record shape')
        row = rows[int(fields[0])]
        if row['kind'] != 6 or int(fields[1]) != row['offset'] + len(row['payload']):
            raise ValueError('Out-of-order byte payload')
        row['payload'].extend(bytes.fromhex(fields[2]))
    calls, active = [], None
    for row in rows.values():
        if row['kind'] == 1:
            if active is not None:
                raise ValueError('Nested/incomplete tensor call')
            active = dict(begin=row, blobs={}, snapshots=[])
        elif row['kind'] == 6:
            if active is None or len(row['payload']) != row['length'] or row['label'] in active['blobs']:
                raise ValueError('Malformed tensor byte payload')
            active['blobs'][row['label']] = bytes(row['payload'])
        elif row['kind'] in (2, 3):
            if active is None:
                raise ValueError('Orphan native counter snapshot')
            active['snapshots'].append(row)
        elif row['kind'] == 4:
            if active is None or row['call'] != active['begin']['call']:
                raise ValueError('Orphan tensor finish')
            active['finish'] = row
            calls.append(active)
            active = None
        elif row['kind'] != 5:
            raise ValueError('Unknown event kind')
    if active is not None:
        raise ValueError('Unfinished tensor call')
    pre_labels = ('pre_pl','pre_m2l','pre_mur','pre_osthres')
    native_labels = ('encoded_p','encoded_s','encoded_m','encoded_scale')
    prior, relation, details = set(), {}, []
    for call in calls:
        b, f = call['begin'], call['finish']
        mode,n,rank,precision = b['key']
        poles,finite,cache = b['options']
        # Conservative initial proposal: primary full-QP only. Failed calls
        # clear our opportunity history; independent directions never seed it.
        eligible = precision == 2 and mode == 4 and bool(cache) and finite == 1 and f['status'] == 0 and f['valid'] == 1
        if f['valid'] != 1 or f['status'] != 0:
            prior.clear()
        repeat = False
        if eligible:
            if not all(k in call['blobs'] for k in pre_labels+native_labels):
                raise ValueError('Eligible QP call lacks complete exact keys')
            identity = (b['epoch'], b['generation'], b['key'], b['options'], b['budget'])
            pre = (identity, tuple(call['blobs'][k] for k in pre_labels))
            native = tuple(call['blobs'][k] for k in native_labels)
            if pre in relation and relation[pre] != native:
                raise ValueError('Equal proposed early key maps to different native geometry')
            relation[pre] = native
            repeat = pre in prior
            prior.add(pre)
        details.append({'call':b['call'],'epoch':b['epoch'],'ctmode':mode,'n':n,'rank':rank,
            'precision':precision,'full':bool(finite),'poles':poles,'cache_enabled':bool(cache),
            'status':f['status'],'validated_tensor':bool(f['valid']),'primary_full_qp_eligible':eligible,
            'natural_prior_valid_same_event_key':repeat,
            'native_snapshots':[{'phase':r['kind']-1,'status':r['status'],'counts':r['stats']} for r in call['snapshots']]})
    return {'schema':'tirust-qp-shadow-opportunities-v1','scope':'one observed point, no cross-event reuse',
        'calls':details,'tensor_calls':len(details),'eligible_calls':sum(x['primary_full_qp_eligible'] for x in details),
        'potential_prior_exact_key_repeats':sum(x['natural_prior_valid_same_event_key'] for x in details),
        'budget_simulated':False,'speedup_estimated':False,'performance_acceptance':False,
        'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [prefix.with_suffix('.calls.tsv'),prefix.with_suffix('.shadow.hex')]}}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('prefix',type=Path)
    args=parser.parse_args()
    print(json.dumps(inspect(args.prefix),indent=2))
