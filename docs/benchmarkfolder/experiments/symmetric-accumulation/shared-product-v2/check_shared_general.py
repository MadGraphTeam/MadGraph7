#!/usr/bin/env python3
"""Compile every source-derived sharing map, including terminal guards."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import check_gather as check
from check_shared_traps import DRIVER as TRAP_DRIVER
import shared

HERE=check.HERE/'shared-product-v2'
WORK=check.ROOT/'.benchmark-work/experiments/symmetric-product-sharing-v2'


def instrument(source):
    source=source.replace('  INTEGER I,J,K,L,M\n',
        '  INTEGER I,J,K,L,M\n  INTEGER FAST,FALLBACK\n'
        '  COMMON/SHARED_TRACE/FAST,FALLBACK\n',1)
    source=source.replace('PRODUCT_READY=.FALSE.',
        'FAST=FAST+1\nPRODUCT_READY=.FALSE.',1)
    # The only external calls in the private entry are IEEE queries and its
    # original public/private fallback. Count each original-routine call.
    import re
    return re.sub(r'(?m)^( +)(CALL TEST_(?:UPDATE|TERMINAL)_WL_\d+_1(?:_DENSE)?\()',
        r'\1FALLBACK=FALLBACK+1\n\1\2',source)


def driver(name,groups,terminal):
    text=check._SPARSE_DRIVER.replace('OLD_ROUTINE',name).replace(
        'NEW_ROUTINE',name+'_SHARED')
    text=text.replace('integer :: i,j,k,ls,ins,outs,trial,rounding,checks',
        'integer :: i,j,k,ls,ins,outs,trial,rounding,checks,fast,fallback,flagstate\n'
        'common/SHARED_TRACE/fast,fallback')
    text=text.replace('checks=0','checks=0\nfast=0\nfallback=0',1)
    copies=[]
    for group in groups:
        ref=','.join(map(str,group[0]))
        for entry in group[1:]:
            copies.append('b('+','.join(map(str,entry))+')=b('+ref+')')
    text=text.replace('    saved_a=transfer(a,saved_a)',
        'if(mod(trial,2)==1) then\n'+'\n'.join(copies)+
        '\nendif\n    saved_a=transfer(a,saved_a)')
    if terminal:
        text=text.replace('checks=0','checks=0\ndo flagstate=0,1',1)
        text=text.replace('call ieee_set_flag(ieee_all,.false.)',
            'call ieee_set_flag(ieee_all,.false.)\n'
            'if(flagstate==1) call ieee_set_flag(ieee_inexact,.true.)')
        text=text.replace('outs,expected)','outs,expected,flagstate==1)')
        text=text.replace('outs,actual)','outs,actual,flagstate==1)')
        text=text.replace('call ieee_set_rounding_mode(ieee_nearest)\nif(checks/=6000)',
            'enddo\ncall ieee_set_rounding_mode(ieee_nearest)\nif(checks/=12000)')
    text=text.replace("write(*,*) 'PASS',checks",
        "if(fast==0.or.fallback==0) stop 8\nwrite(*,*) 'PASS',checks,fast,fallback")
    return text


def main():
    os.sched_setaffinity(0,{2})
    HERE.mkdir(exist_ok=True)
    results=[]
    data=json.loads((check.HERE/'inventory.json').read_text())['qcd-3']['routines']
    vertices=[v for v,d in data.items() if any(
        s['in_size']==s['out_size']==4 for s in d['shapes'])]
    for name in ('gather.py','shared.py','check_shared_general.py','inventory.json'):
        shutil.copy2(check.HERE/name,HERE/name)
    for rank in (2,3):
      for terminal in (False,True):
       for vertex in vertices:
        groups=shared.groups(vertex,terminal)
        if not groups:
            continue
        key=f'{vertex}-rank{rank}-'+('terminal' if terminal else 'full')
        work=WORK/key
        work.mkdir(parents=True,exist_ok=False)
        (work/'coef_specs.inc').write_text('      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
            '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
        (work/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n'
            '      PARAMETER(LOOPMAXCOEFS=210)\n')
        gen=check.FortranPolynomialRoutines(6,updater_max_rank=2,proc_prefix='TEST_',
            sub_prefix='TEST_',coef_format='complex*32')
        name='TEST_'+('TERMINAL' if terminal else 'UPDATE')+f'_WL_{rank}_1'
        source=shared.generate(gen,rank,vertex,terminal)
        full=gen.write_compact_wl_updater(rank,1)
        if terminal:
            full+='\n\n'+gen.write_compact_wl_updater(rank,1,terminal=True)
        for filename,text in [('constants.f',gen.write_polynomial_constant_module()),
                              ('full.f',full),('shared.f',source),
                              ('checked-shared.f',instrument(source))]:
            with check.FortranWriter(str(work/filename)) as writer:
                writer.writelines(text)
        (work/'driver.f90').write_text(driver(name,groups,terminal))
        trap=TRAP_DRIVER.replace('TEST_UPDATE_WL_2_1_SHARED',name+'_SHARED')
        if terminal:
            trap=trap.replace('b,4,4,out)','b,4,4,out,.false.)')
        (work/'traps.f90').write_text(trap)
        runs={}
        start=time.monotonic()
        for mainfile,executable in (('driver.f90','check'),('traps.f90','traps')):
            command=['gfortran','-O3','-frounding-math','-fno-automatic','-fcheck=all',
                '-ffixed-line-length-132','constants.f','full.f','checked-shared.f',
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
        record=dict(key=key,rank=rank,terminal=terminal,vertex=vertex,
            seconds=time.monotonic()-start,groups=groups,runs=runs,
            sources={p.name:check.sha(p) for p in work.iterdir()
                if p.suffix in ('.f','.f90','.inc')})
        results.append(record)
        destination=HERE/key
        destination.mkdir()
        for p in work.iterdir():
            if p.suffix in ('.f','.f90','.inc','.log'):
                shutil.copy2(p,destination/p.name)
        (HERE/'exact.json').write_text(json.dumps(dict(complete=False,
            acceptance_eligible=False,results=results),indent=2)+'\n')
        print(key,runs['check']['output'].strip(),runs['traps']['output'].strip(),flush=True)
    result=dict(complete=True,acceptance_eligible=False,results=results,
        comparisons=sum(12000 if r['terminal'] else 6000 for r in results),
        trap_controls=5*len(results),runner_sha256=check.sha(__file__),
        generator_sha256=check.sha(check.ROOT/'madgraph/various/q_polynomial.py'),
        scope='Every available source-derived full/terminal group map at ranks 2 and 3. '
        'Full output/input bits and sticky flags against original public/private kernels; '
        'both entry-INEXACT states for terminal, four rounding modes, signed zero, '
        'cancellation, exceptional inputs and empty dimensions. Actual fast/fallback '
        'counters and enabled-halting fallback required. No process or timing claim.')
    (HERE/'exact.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
