#!/usr/bin/env python3
"""Read-only checksum and independent raw-row verification; no amplitudes."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(path):
    rows = []
    for line in path.read_text().splitlines():
        f = line.split()
        assert len(f) == 8
        rows.append(dict(point=int(f[0]), order=int(f[1]), status=int(f[2]),
                         coefficients=list(map(float, f[3:7])), accuracy=float(f[7])))
    assert [(r['point'],r['order']) for r in rows] == [(p,o) for p in range(1,101) for o in (0,1)]
    assert all(math.isfinite(x) for r in rows for x in r['coefficients']+[r['accuracy']])
    return rows


def comparisons(groups):
    assert len({tuple((r['point'],r['order']) for r in rows) for rows in groups.values()}) == 1
    scale = {}
    for rows in groups.values():
        for row in rows:
            scale[row['point']] = max(scale.get(row['point'], 0), *map(abs,row['coefficients']))
    maximum, count = 0., 0
    for left,right in itertools.combinations(groups, 2):
        for a,b in zip(groups[left], groups[right]):
            for x,y in zip(a['coefficients'], b['coefficients']):
                bound = 1e-7*max(abs(x),abs(y)) + 1e-10*scale[a['point']]
                error = abs(x-y)
                assert error <= bound, (left,right,a['point'],a['order'],x,y,bound)
                maximum = max(maximum, error/bound if bound else 0.)
                count += 1
    return maximum, count


def main():
    seal = json.loads((HERE/'SHA256SUMS.json').read_text())
    assert seal['schema'] == 'dp-factor-actual-qcd3-diagnostic-v1'
    assert seal['complete'] and len(seal['files']) == seal['file_count']
    for name, record in seal['files'].items():
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        path = HERE/p
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'],name
    all_groups, cohorts = {}, {}
    for backend in (6,8):
        report = HERE/f'backend-{backend}'
        result = json.loads((report/'result.json').read_text())
        assert result['complete'] and result['comparison']['passed'] and not result['acceptance_eligible']
        reference = json.loads((HERE/f'parent-source/reference-{backend}.json').read_text())
        assert sha(HERE/f'parent-source/reference-{backend}.json') == result['source_reference_sha256']
        groups = {}
        for variant in ('original','capture-only','factor-trace'):
            data = raw(report/variant/'values.dat')
            record = result['measurements'][variant]
            assert data == record['rows']
            assert record['shared_numerical_settings'] == reference['shared_numerical_settings']
            assert sha(report/variant/'values.dat') == record['values_sha256']
            expected = {6:{(2,6),(3,8)},8:{(2,8),(3,2)}}[backend]
            for row in data:
                code = row['status']
                assert (code//100,code%10) in expected and (code//10)%10 in (1,2)
                assert row['accuracy'] >= 0
            counts = dict(Counter(str(r['status']) for r in data if r['order']==0))
            assert counts == record['status_counts']
            timed = dict(tuple(map(int,line.split())) for line in (report/variant/'timed_status.dat').read_text().splitlines())
            assert sum(timed.values()) == 100 and all(v>0 for v in timed.values())
            assert {str(k):v for k,v in timed.items()} == record['timed_status_counts']
            if variant != 'original':
                trace = [list(map(int,line.split())) for line in (report/variant/'dp_factor_counts.dat').read_text().splitlines()]
                assert trace == record['guard_or_capture_counts']
                if variant == 'factor-trace':
                    assert all(r[1] == r[2]+r[3] and r[1]>0 for r in trace)
                else:
                    assert all(0<=r[2]<=r[1] and r[1]>0 for r in trace)
            groups[variant] = data
            all_groups[f'{backend}/{variant}'] = data
        assert groups['original'] == reference['rows'] == groups['capture-only']
        maximum,count = comparisons(groups)
        assert maximum == result['comparison']['max_tolerance_fraction']
        assert count == result['comparison']['comparisons']
        cohorts[str(backend)] = dict(comparisons=count,maximum_tolerance_fraction=maximum)
    maximum,count = comparisons(all_groups)
    print(json.dumps(dict(passed=True,bound_files=seal['file_count'],cohorts=cohorts,
                          all_six_stream_comparisons=count,maximum_tolerance_fraction=maximum)))


if __name__ == '__main__':
    main()
