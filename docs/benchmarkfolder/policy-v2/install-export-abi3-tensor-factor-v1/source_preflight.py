#!/usr/bin/env python3
"""Source-only preflight of held parser/driver changes; no export or evaluation."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,re
from pathlib import Path
import validate as v


def main():
    parent=v.ROOT/'.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1/outputs/candidate/qcd-3'
    v.RELOCATED=parent
    sites=v.factor_sites()
    original=parent/'SubProcesses/P0_uux_zgg/check_sa.original'
    transformed=v.diagnostic_driver(original.read_text())
    rows=[]
    for line in transformed.splitlines():
        if len(line)<7 or line[0] in 'cC*!#':continue
        body=re.sub(r'\s+','',line[6:]).upper()
        if not body:continue
        if line[5] not in ' 0':rows[-1]+=body
        else:rows.append(body)
    assert 'PARAMETER(READPS=.TRUE.)' in rows
    assert 'PARAMETER(NPSPOINTS=20)' in rows
    # The same five four-vectors are opened/read/closed inside each outer pass.
    start=rows.index('DOK=1,NPSPOINTS');end=rows.index('CLOSE(967)')
    segment=rows[start:end+1]
    assert segment.index('IF(READPS)THEN')<segment.index("OPEN(967,FILE='PS.INPUT',ERR=976,STATUS='OLD',ACTION='READ')")
    assert 'DOI=1,NEXTERNAL' in segment
    assert 'READ(967,*,END=978)P(0,I),P(1,I),P(2,I),P(3,I)' in segment
    assert sum('READ(967' in r for r in rows)==1
    assert 'DOJ=0,NSQUAREDSO_LOOP' in rows
    assert "WRITE(*,'(A,1X,I4,1X,4ES26.17E3)')'RELOCATION_ROW',J,(MATELEM(I,J),I=0,3)" in rows
    assert 'CALLML5_0_FORCE_STABILITY_CHECK(.TRUE.)' in rows
    points=v.ROOT/'docs/benchmarkfolder/points/qcd-3.dat'
    reference=v.read(v.REPORT/'reference-input.json');assert v.sha(points)==reference['points_sha256']
    lines=points.read_text().splitlines();assert lines[0]=='100'
    point='\n'.join(lines[1:6])+'\n';assert len(point.splitlines())==5
    assert all(len([float(x) for x in r.split()])==4 for r in point.splitlines())
    # Verify width policy is the frozen benchmark's actual implementation.
    text=(v.ROOT/'tools/ninjars_benchmark.py').read_text()
    assert 'decay.value = 0.0' in text and 'param.write(str(param_path))' in text
    result=dict(passed=True,source_only=True,build_launch=False,numerical_launch=False,
                factored_sites=sum(r['calls'] for r in sites.values()),batches=sites,
                original_driver_sha256=v.sha(original),diagnostic_driver_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                point_sha256=hashlib.sha256(point.encode()).hexdigest(),same_point_read_each_of_20_calls=True,
                all_physical_and_total_rows=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
