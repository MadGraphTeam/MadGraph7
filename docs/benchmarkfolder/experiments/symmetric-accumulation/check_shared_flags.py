#!/usr/bin/env python3
"""Repeat exact arithmetic/IEEE controls with actual standalone and NLO flags."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import check_shared_general as fixture
from check_shared_traps import DRIVER as TRAP_DRIVER
import shared

check=fixture.check
HERE=check.HERE/'shared-product-production-flags'
WORK=check.ROOT/'.benchmark-work/experiments/shared-product-production-flags'


def main():
    os.sched_setaffinity(0,{3})
    HERE.mkdir(exist_ok=False)
    results=[]
    flags_by_mode=dict(standalone=['-O3','-w','-fPIC','-ffixed-line-length-132'],
                      nlo=['-O3','-fPIC','-ffixed-line-length-132','-fno-automatic'])
    for mode,flags in flags_by_mode.items():
      for rank in (2,4):
       for terminal in (False,True):
        key=f'{mode}-rank{rank}-'+('terminal' if terminal else 'full')
        work=WORK/key
        work.mkdir(parents=True,exist_ok=False)
        (work/'coef_specs.inc').write_text('      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
            '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
        (work/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n'
            '      PARAMETER(LOOPMAXCOEFS=210)\n')
        gen=check.FortranPolynomialRoutines(6,updater_max_rank=2,proc_prefix='TEST_',
            sub_prefix='TEST_',coef_format='complex*32')
        name='TEST_'+('TERMINAL' if terminal else 'UPDATE')+f'_WL_{rank}_1'
        source=shared.generate(gen,rank,terminal=terminal)
        full=gen.write_compact_wl_updater(rank,1)
        if terminal:
            full+='\n\n'+gen.write_compact_wl_updater(rank,1,terminal=True)
        for filename,text in [('constants.f',gen.write_polynomial_constant_module()),
            ('full.f',full),('shared.f',source),('checked-shared.f',fixture.instrument(source))]:
            with check.FortranWriter(str(work/filename)) as writer:
                writer.writelines(text)
        (work/'driver.f90').write_text(fixture.driver(name,shared.groups(
            terminal=terminal),terminal))
        trap=TRAP_DRIVER.replace('TEST_UPDATE_WL_2_1_SHARED',name+'_SHARED')
        if terminal:
            trap=trap.replace('b,4,4,out)','b,4,4,out,.false.)')
        (work/'traps.f90').write_text(trap)
        runs={}
        start=time.monotonic()
        for mainfile,executable in (('driver.f90','check'),('traps.f90','traps')):
            command=['gfortran',*flags,'constants.f','full.f','checked-shared.f',
                mainfile,'-o',executable]
            built=subprocess.run(command,cwd=work,capture_output=True,text=True)
            (work/(executable+'-build.log')).write_text(built.stdout+built.stderr)
            built.check_returncode()
            ran=subprocess.run([str(work/executable)],cwd=work,capture_output=True,text=True)
            (work/(executable+'.log')).write_text(ran.stdout+ran.stderr)
            runs[executable]=dict(command=command,passed=ran.returncode==0,
                output=ran.stdout+ran.stderr,executable_sha256=check.sha(work/executable))
            if ran.returncode:
                (HERE/'failure.json').write_text(json.dumps(dict(key=key,runs=runs),indent=2)+'\n')
                ran.check_returncode()
        record=dict(key=key,rank=rank,terminal=terminal,mode=mode,runs=runs,
            seconds=time.monotonic()-start,sources={p.name:check.sha(p) for p in work.iterdir()
                if p.suffix in ('.f','.f90','.inc')})
        results.append(record)
        destination=HERE/key
        destination.mkdir()
        for path in work.iterdir():
            if path.suffix in ('.f','.f90','.inc','.log'):
                shutil.copy2(path,destination/path.name)
        (HERE/'exact.json').write_text(json.dumps(dict(complete=False,results=results),indent=2)+'\n')
        print(key,runs['check']['output'].strip(),runs['traps']['output'].strip(),flush=True)
    dependencies=('check_shared_flags.py','check_shared_general.py','check_shared_traps.py',
                  'shared.py','gather.py','inventory.json')
    for name in dependencies:
        shutil.copy2(check.HERE/name,HERE/name)
    result=dict(complete=True,acceptance_eligible=False,cpu=3,results=results,
        comparisons=sum(12000 if r['terminal'] else 6000 for r in results),
        trap_controls=5*len(results),flags=flags_by_mode,
        source_sha256={name:check.sha(check.HERE/name) for name in dependencies},
        generator_sha256=check.sha(check.ROOT/'madgraph/various/q_polynomial.py'),
        scope='Representative full and terminal FFV1L2_1 maps at ranks2/4 under '
        'actual GNU standalone/NLO polynomial optimization flags, without '
        '-frounding-math or runtime bounds-check flags. Every output/input bit '
        'and IEEE flag compared over four rounding modes, exceptional/subnormal/'
        'signed-zero/cancellation/empty cases and both terminal entry flag states. '
        'Enabled traps must take the original fallback.')
    (HERE/'exact.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
