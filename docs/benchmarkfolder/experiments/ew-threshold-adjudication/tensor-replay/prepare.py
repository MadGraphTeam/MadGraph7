#!/usr/bin/env python3
"""Source/protocol preparation only: this script cannot compile or run native code."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[4]
PARENT=REPORT.parent
WORK=ROOT/'.tirust-build/ew-threshold-tensor-replay'
RUNTIME=ROOT/'.tirust-build/ew-threshold-tensor-replay-runtime'
PRIVATE_DSO=RUNTIME/'libmadloop_tirust.so'
EXPECTED_DSO='49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
EXPECTED_NRS='0edfa89d7d83161a6ce3fdf0bb1fbca928b73d2b5ec44211a3b2efa186a155d6'
MAPPING=[1,2,3,4,5,6,7,10,8,11,13,9,12,14,15,16,17,20,26,18,21,27,23,29,32,19,22,28,24,30,33,25,31,34,35]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def sources():
    interface=PARENT/'sources/TIRUST_interface.f'
    text=interface.read_text()
    mp=text.split('      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC',1)[1]
    data=re.findall(r'DATA \(TIRUST_MAP\(I\),I=\d+,\d+\)/([^/]+)/',mp)
    parsed=[int(v) for row in data for v in row.split(',')]
    assert parsed==MAPPING
    assert sorted(parsed)==list(range(1,36))
    for rank in range(4):
        count=(rank+1)*(rank+2)*(rank+3)*(rank+4)//24
        assert max(parsed[:count])==count
    assert digest(PRIVATE_DSO)==EXPECTED_DSO
    frozen=json.loads((REPORT/'runtime-snapshot.json').read_text())
    for name,record in frozen['files'].items():
        assert digest(RUNTIME/name)==record['sha256'],name
    assert digest(ROOT/'HEPTools/ninjars/lib/libninjars.a')==EXPECTED_NRS
    paths=[interface,REPORT/'replay.f90',REPORT/'prepare.py',REPORT/'analyze.py',REPORT/'README.md',
        RUNTIME/'mtirust.f90',RUNTIME/'libtirust.a',RUNTIME/'mtirust.mod',PRIVATE_DSO,
        RUNTIME/'mninjars.f90',ROOT/'HEPTools/ninjars/lib/libninjars.a',
        RUNTIME/'mninjars.mod',ROOT/'HEPTools/ninjars/lib/ninjars-link-before.txt',
        ROOT/'HEPTools/ninjars/lib/ninjars-native-libs.txt',
        ROOT/'HEPTools/ninjars/lib/ninjars-config.json',REPORT/'runtime-snapshot.json']
    snapshot={str(p.relative_to(ROOT)):digest(p) for p in paths}
    out=REPORT/'source-manifest.json'
    data={'scope':'Source/protocol only. No native compilation/evaluation authorized by this script.',
          'mapping_MadLoop_index_to_IREGI_one_based':MAPPING,'sources_and_link_inputs':snapshot,
          'historical_tirust_dso':str(PRIVATE_DSO.relative_to(ROOT)),
          'precision':'binary128 input transport; 256-bit MPFR working precision in both pinned bridges',
          'runtime_independence_limit':'Both scalar implementations have oneloopmaster ancestry with different pinned revisions.'}
    if out.exists():
        assert json.loads(out.read_text())==data,'Source manifest already exists with different content'
    else:
        dump(out,data)
    return data


def decimal_token(value):
    # Trace ES45.35E4 records are strings. Never pass through Python float.
    assert isinstance(value,str),type(value)
    assert Decimal(value).is_finite(),value
    return value


def extract(point,group):
    source=point['groups'][str(group)]
    n,rank=source['n'],source['rank']
    assert isinstance(n,int) and 1<=n<=6 and isinstance(rank,int) and 0<=rank<=3
    count=(rank+1)*(rank+2)*(rank+3)*(rank+4)//24
    shifts={int(r[0]):list(map(decimal_token,r[1:])) for r in source['shifts']}
    masses={int(r[0]):list(map(decimal_token,r[1:])) for r in source['masses']}
    invariants={(int(r[0]),int(r[1])):decimal_token(r[2]) for r in source['invariants']}
    assert len(source['shifts'])==len(shifts)==n and all(len(v)==4 for v in shifts.values())
    assert len(source['masses'])==len(masses)==n and all(len(v)==2 for v in masses.values())
    assert set(shifts)==set(masses)==set(range(1,n+1))
    assert len(source['invariants'])==len(invariants)==n*n
    assert set(invariants)=={(i,j) for i in range(1,n+1) for j in range(1,n+1)}
    calls=[c for c in point['calls'] if c['group']==group]
    assert calls and {c['status'] for c in calls}=={source['status']},'Ambiguous per-group native status'
    assert {c['mode'] for c in calls}=={4},'Replay scope is fixed-QP direct order'
    assert {(c['n'],c['rank']) for c in calls}=={(n,rank)},'Inconsistent per-group topology'
    raw={(int(r[0]),int(r[1])):list(map(decimal_token,r[2:])) for r in source['raw']}
    if source['status']==0:
        assert len(raw)==len(source['raw'])==4*count
        assert set(raw)=={(i,j) for i in range(1,count+1) for j in range(1,5)}
    else:
        assert not raw,'Failed native output must not be treated as a numerical zero'
    return {'point':point['index'],'group':group,'n':n,'rank':rank,'count':count,
            'historical_status':source['status'],'historical_error':source.get('error'),
            'scale':decimal_token(source['scale']),
            'shifts':[[i,*shifts[i]] for i in range(1,n+1)],
            'masses':[[i,*masses[i]] for i in range(1,n+1)],
            'invariants':[[i,j,invariants[i,j]] for i in range(1,n+1) for j in range(1,n+1)],
            'historical_raw_IREGI_UV1_IR1_IR2_finite':source['raw'],
            'unit_degrees':[next(d for d in range(4) if i<(d+1)*(d+2)*(d+3)*(d+4)//24)
                            for i in range(count)]}


def select(requested):
    source_manifest=sources()
    trace_path=PARENT/'trace.json'
    summary_path=PARENT/'summary.json'
    execution_path=PARENT/'execution.json'
    # These are intentionally absent while the parent native trace is pending.
    trace=json.loads(trace_path.read_text())
    summary=json.loads(summary_path.read_text())
    execution=json.loads(execution_path.read_text())
    assert execution['complete'] and execution['final_values_identical_to_historical']
    assert set(trace)=={'1','2'}
    failures={p['index']:p['failed_group_ids'] for p in summary['points']}
    if requested:
        selected=[tuple(map(int,v.split(':'))) for v in requested]
    else:
        assert failures[2],'No failed stress group available for the requested focused replay'
        group=min(failures[2])
        failed=trace['2']['groups'][str(group)]
        selected=[(2,group),(1,group)]
        controls=[g for g in trace['2']['groups'].values() if g['status']==0]
        if controls:
            controls.sort(key=lambda g:(g['n']!=failed['n'],g['rank']!=failed['rank'],g['group']))
            selected.append((2,controls[0]['group']))
    assert 1<=len(selected)<=3 and len(set(selected))==len(selected)
    records=[extract(trace[str(p)],g) for p,g in selected]
    lines=['EW_TENSOR_REPLAY_V1',str(len(records))]
    for case in records:
        lines.append(' '.join(str(case[k]) for k in ('point','group','n','rank','historical_status')))
        lines.append(case['scale'])
        lines.extend(' '.join(r[1:]) for r in case['shifts'])
        lines.extend(' '.join(r[1:]) for r in case['masses'])
        inv={(r[0],r[1]):r[2] for r in case['invariants']}
        lines.extend(' '.join(inv[i,j] for j in range(1,case['n']+1)) for i in range(1,case['n']+1))
    if WORK.exists() or (REPORT/'selected-inputs.dat').exists():
        raise ValueError('Refusing to overwrite tensor replay preparation')
    WORK.mkdir()
    (REPORT/'selected-inputs.dat').write_text('\n'.join(lines)+'\n')
    shutil.copy2(REPORT/'selected-inputs.dat',WORK/'selected-inputs.dat')
    shutil.copy2(REPORT/'replay.f90',WORK/'replay.f90')
    dump(REPORT/'selected-inputs.json',{
        'scope':'Prepared inputs only; no native job has run.',
        'source_manifest_sha256':digest(REPORT/'source-manifest.json'),
        'parent_trace_sha256':digest(trace_path),'parent_summary_sha256':digest(summary_path),
        'parent_execution_sha256':digest(execution_path),
        'input_stream_sha256':digest(REPORT/'selected-inputs.dat'),
        'common_scale_policy':'Both replay engines receive the exact captured TIRust scale; original NinjaRS scale used DP squaring.',
        'unit_numerator_policy':'NinjaRS coefficient (-1)^degree, no extra multiplicity/sign; TIR raw already Convention::MadLoop.',
        'expected_NinjaRS_calls':sum(c['count'] for c in records),
        'records':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('sources','select'))
    parser.add_argument('--group',action='append',help='Explicit unique point:group; at most three')
    args=parser.parse_args()
    if args.action=='sources':
        sources()
    else:
        select(args.group)
