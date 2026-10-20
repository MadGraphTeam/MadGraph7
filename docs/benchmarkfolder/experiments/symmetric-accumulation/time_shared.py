#!/usr/bin/env python3
"""Exploratory original-scatter product-cache timing, with all guards included."""
import json
import os
import shutil
import statistics
import subprocess

import check_shared as fixture
import time_gather


def main():
    os.sched_setaffinity(0,{2})
    check=fixture.check
    for name in ('time_shared.py','time_gather.py','check_shared.py'):
        shutil.copy2(check.HERE/name,fixture.HERE/name)
    inventory=json.loads((check.HERE/'inventory.json').read_text())['qcd-3']
    shape=next(s for s in inventory['routines']['MP_FFV1L2_1']['shapes']
               if s['out_size']==4 and s['in_size']==4)
    results=[]
    for rank in (2,3):
        work=fixture.WORK/f'rank{rank}'
        loops=300
        driver=(time_gather.DRIVER.replace('@KIND@','16').replace('@RANK@',str(rank))
            .replace('@LOOPS@',str(loops)).replace('_GATHER','_SHARED')
            .replace('@ZERO@','\n'.join(f'b({j},{m},{k})=0'
                       for j,m,k in shape['literal_zero_entries'])+'\n'+fixture.copy_groups()))
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
    (fixture.HERE/'timing.json').write_text(json.dumps(dict(
        acceptance_eligible=False,results=results,runner_sha256=check.sha(__file__),
        scope='Synthetic QP kernel CPU time on CPU2 under concurrent work; seven alternating '
        'pairs. All bounded/identity/trap dispatch and cache clearing are timed. '
        'Known FFV1L2_1 literal-zero and exact-expression groups, synthetic complex values; '
        'no process-level or acceptance claim.'),indent=2)+'\n')


if __name__=='__main__':
    main()
