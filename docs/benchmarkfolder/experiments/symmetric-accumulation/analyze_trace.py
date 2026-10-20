#!/usr/bin/env python3
"""Bind the structural QCD3 trace to exact retained full-process outputs."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent/'qcd3-sharing-trace-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=json.loads((HERE/'result.json').read_text())
    proof=json.loads((HERE/'shared-trace-proof.json').read_text())
    assert result['complete'] and result['all_printed_rows_and_statuses_exact']
    assert (HERE/'original-0-values.dat').read_bytes()==(HERE/'shared-trace-0-values.dat').read_bytes()
    assert (HERE/'original-0-timed_status.dat').read_bytes()==(HERE/'shared-trace-0-timed_status.dat').read_bytes()
    values=[line.split() for line in (HERE/'original-0-values.dat').read_text().splitlines()]
    assert {int(row[0]) for row in values}==set(range(1,101))
    counts={int(code):int(count) for code,count in (
        line.split() for line in (HERE/'original-0-timed_status.dat').read_text().splitlines())}
    assert counts=={216:98,328:2}
    rows=[]
    for line in (HERE/'shared-trace-final.dat').read_text().splitlines():
        index,fast,fallback=map(int,line.split())
        helper=proof['helpers'][index-1]
        assert helper['index']==index
        rows.append(dict(index=index,rank=helper['rank'],terminal=helper['terminal'],
            vertex=helper['vertex'],fast=fast,fallback=fallback))
    assert len(rows)==len(proof['helpers'])
    summary=dict(complete=True,points=100,printed_rows=len(values),statuses=counts,
        all_printed_coefficients_accuracy_status_exact=True,rows=rows,
        sharing_calls=sum(row['fast'] for row in rows),
        fallback_calls=sum(row['fallback'] for row in rows),
        evidence_sha256={path.name:sha(path) for path in (
            HERE/'result.json',HERE/'shared-trace-proof.json',HERE/'shared-trace-final.dat',
            HERE/'original-0-values.dat',HERE/'shared-trace-0-values.dat',
            HERE/'original-0-timed_status.dat',HERE/'shared-trace-0-timed_status.dat')},
        scope='Structural process validation only. Concurrent, instrumented timing '
        'is not used to infer speedup. Every printed component and status is compared; '
        'this does not claim access to unprinted internal binary128 intermediates.')
    (HERE/'trace-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary['sharing_calls'],summary['fallback_calls'])


if __name__=='__main__':
    main()
