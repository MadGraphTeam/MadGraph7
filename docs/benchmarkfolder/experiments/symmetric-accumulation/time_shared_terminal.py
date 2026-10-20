#!/usr/bin/env python3
"""Exploratory guarded terminal-sharing kernel timing after exact validation."""
import json
import os
import shutil
import statistics
import subprocess

import check_shared_general as fixture
import shared
import time_gather


def main():
    os.sched_setaffinity(0,{2})
    check=fixture.check
    results=[]
    inventory=json.loads((check.HERE/'inventory.json').read_text())['qcd-3']
    shape=next(s for s in inventory['routines']['MP_FFV1L2_1']['shapes']
        if s['out_size']==s['in_size']==4)
    zero='\n'.join(f'b({j},{m},{k})=0' for j,m,k in shape['literal_zero_entries'])
    for group in shared.groups('MP_FFV1L2_1',True):
        reference=','.join(map(str,group[0]))
        for member in group[1:]:
            zero+='\nb('+','.join(map(str,member))+')=b('+reference+')'
    for rank in (2,3):
        key=f'MP_FFV1L2_1-rank{rank}-terminal'
        work=fixture.WORK/key
        record=next(r for r in json.loads((fixture.HERE/'exact.json').read_text())['results']
            if r['key']==key)
        assert record['runs']['check']['passed'] and record['runs']['traps']['passed']
        assert record['sources']['shared.f']==check.sha(work/'shared.f')
        loops=1000
        driver=(time_gather.DRIVER.replace('@KIND@','16').replace('@RANK@',str(rank))
            .replace('@LOOPS@',str(loops)).replace('@ZERO@',zero)
            .replace('_GATHER','_SHARED').replace('TEST_UPDATE_','TEST_TERMINAL_')
            .replace('(a,4,b,4,4,out)','(a,4,b,4,4,out,.true.)')
            .replace('implicit none','use, intrinsic :: ieee_exceptions\nimplicit none')
            .replace('  call cpu_time(t0)',
                '  call ieee_set_flag(ieee_inexact,.true.)\n  call cpu_time(t0)'))
        (work/'timing.f90').write_text(driver)
        command=['gfortran','-O3','-frounding-math','-fno-automatic',
            '-ffixed-line-length-132','constants.f','full.f','shared.f',
            'timing.f90','-o','timing']
        built=subprocess.run(command,cwd=work,capture_output=True,text=True)
        (work/'timing-build.log').write_text(built.stdout+built.stderr)
        built.check_returncode()
        ran=subprocess.run([str(work/'timing')],cwd=work,capture_output=True,text=True)
        (work/'timing.log').write_text(ran.stdout+ran.stderr)
        ran.check_returncode()
        rows=[line.split() for line in ran.stdout.splitlines()]
        assert len(rows)==14
        for density in (0,1):
            pairs=[(float(row[2]),float(row[3])) for row in rows if int(row[0])==density]
            assert len(pairs)==7
            ratio=statistics.median(t[1] for t in pairs)/statistics.median(t[0] for t in pairs)
            results.append(dict(rank=rank,density=density,loops=loops,pairs_seconds=pairs,
                ratio_of_medians=ratio,source_sha256=check.sha(work/'shared.f'),
                executable_sha256=check.sha(work/'timing')))
            print(rank,density,ratio,flush=True)
        for name in ('timing.f90','timing-build.log','timing.log'):
            shutil.copy2(work/name,fixture.HERE/key/name)
    for name in ('time_shared_terminal.py','time_gather.py'):
        shutil.copy2(check.HERE/name,fixture.HERE/name)
    (fixture.HERE/'terminal-timing.json').write_text(json.dumps(dict(
        acceptance_eligible=False,results=results,runner_sha256=check.sha(__file__),
        scope='Synthetic terminal QP kernel CPU time on CPU2 under concurrent work; '
        'seven alternating pairs, original private terminal routine as denominator. '
        'All guards/identity/trap dispatch and cache clearing timed; entry inexact set. '
        'FFV1L2_1 source mask/groups with synthetic values; no process speedup claim.'
    ),indent=2)+'\n')


if __name__=='__main__':
    main()
