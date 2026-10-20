#!/usr/bin/env python3
"""HELD amplitude launch: root must first assign the sole MP slot."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare as prep
import parse_capture


def check_inputs():
    binding=prep.load(HERE/'bindings.json');prepared=prep.load(HERE/'prepared.json')
    assert prepared['complete'] and not prepared['numerical_launch']
    assert prep.bench.generated_source_manifest(prep.OUTPUT)==prepared['source_manifest']
    assert prep.runtime(prep.OUTPUT)==prepared['runtime']==binding['runtime']
    assert prep.sha(prep.OUTPUT/prep.SUB/'check')==prepared['executable_sha256']
    assert prep.sha(HERE/'points.dat')==binding['points_sha256']
    assert prep.sha(HERE/'MadLoopParams.dat')==binding['madloop_card_sha256']
    assert prep.sha(prep.OUTPUT/'Cards/param_card.dat')==binding['parameter_card_sha256']
    for p in [prep.OUTPUT/'Cards/MadLoopParams.dat',*(prep.OUTPUT/'SubProcesses').rglob('MadLoopParams.dat')]:
        assert prep.sha(p)==binding['madloop_card_sha256']
    directory=prep.OUTPUT/prep.SUB
    candidates=[directory/'param_card.dat']+[directory.joinpath(*(['..']*i),'Cards/param_card.dat') for i in range(6)]
    assert prep.sha(next(p for p in candidates if p.is_file()))==binding['parameter_card_sha256']
    assert prep.sha(prep.ROOT/binding['reference_result'])==binding['reference_result_sha256']
    return binding,prepared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-after-root-mp-handoff',action='store_true',required=True)
    parser.parse_args()
    os.sched_setaffinity(0,{2})
    assert not (HERE/'capture-result.json').exists() and not (HERE/'raw').exists()
    binding,prepared=check_inputs();directory=prep.OUTPUT/prep.SUB
    assert not (directory/'tensor-capture.hex').exists()
    for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat'):
        (directory/name).unlink(missing_ok=True)
    result={'complete':False,'acceptance_eligible':False,'cpu':2,'numerical_launch':True,
            'prepared_sha256':prep.sha(HERE/'prepared.json'),
            'bindings_sha256':prep.sha(HERE/'bindings.json'),
            'runner_sha256':prep.sha(Path(__file__)),
            'parser_sha256':prep.sha(HERE/'parse_capture.py')}
    raw=HERE/'raw';raw.mkdir()
    try:
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            SYMBOLICA_LOCK_DIR=str(prep.ROOT/'.ninjars-build/symbolica-locks'))
        command=[str(directory/'check'),str(HERE/'points.dat'),'1']
        start=time.monotonic()
        with (raw/'stdout.log').open('w') as stream:
            process=subprocess.run(command,cwd=directory,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        result.update(command=command,returncode=process.returncode,seconds_diagnostic=time.monotonic()-start)
        assert process.returncode==0
        rows=[]
        for line in (directory/'values.dat').read_text().splitlines():
            fields=line.split();assert len(fields)==8
            row={'point':int(fields[0]),'order':int(fields[1]),'status':int(fields[2]),
                 'coefficients':list(map(float,fields[3:7])),'accuracy':float(fields[7])}
            assert all(map(math.isfinite,row['coefficients']+[row['accuracy']]))
            rows.append(row)
        assert rows==binding['reference_rows'],'Capture changed full100 coefficients, statuses or accuracies'
        assert Counter(str(r['status']) for r in rows if r['order']==0)==Counter(binding['reference_status_counts'])
        assert [r['point'] for r in rows if r['order']==0 and r['status']==322]==[44,80]
        assert (directory/'dimensions.dat').read_text().split()==['1','1','1']
        timed=dict((code,int(count)) for code,count in (line.split() for line in (directory/'timed_status.dat').read_text().splitlines()))
        assert timed==binding['reference_status_counts']
        result['all_100_rows_statuses_accuracies_exact']=True
        result['timed_status_counts']=timed
        # Interpretation is gated by full-process invariance above.
        records=parse_capture.parse(directory/'tensor-capture.hex')
        matching=parse_capture.match(records)
        prep.write('matching.json',matching)
        result.update(record_count=len(records),matched_records=matching['matched_records'])
        check_inputs()
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat','tensor-capture.hex'):
            p=directory/name
            if p.is_file():shutil.copy2(p,raw/name)
        result['raw']={p.name:{'sha256':prep.sha(p),'bytes':p.stat().st_size} for p in raw.iterdir() if p.is_file()}
        prep.write('capture-result.json',result)


if __name__=='__main__':
    main()
