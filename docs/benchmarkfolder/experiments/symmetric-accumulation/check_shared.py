#!/usr/bin/env python3
"""Exact QP oracle for lazy products shared only by bit-identical B entries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import check_gather as check
import shared

HERE=check.HERE/'shared-product-v1'
WORK=check.ROOT/'.benchmark-work/experiments/symmetric-product-sharing'


def copy_groups():
    lines=[]
    for group in shared.groups():
        reference=','.join(map(str,group[0]))
        for entry in group[1:]:
            lines.append('b('+','.join(map(str,entry))+')=b('+reference+')')
    return '\n'.join(lines)


def main():
    os.sched_setaffinity(0,{2})
    HERE.mkdir(exist_ok=True)
    for name in ('gather.py','shared.py','check_shared.py','inventory.json'):
        shutil.copy2(check.HERE/name,HERE/name)
    results=[]
    for rank in (2,3):
        work=WORK/f'rank{rank}'
        work.mkdir(parents=True,exist_ok=True)
        (work/'coef_specs.inc').write_text('      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
            '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
        (work/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n'
            '      PARAMETER(LOOPMAXCOEFS=210)\n')
        gen=check.FortranPolynomialRoutines(6,updater_max_rank=2,proc_prefix='TEST_',
            sub_prefix='TEST_',coef_format='complex*32')
        name=f'TEST_UPDATE_WL_{rank}_1'
        source=shared.generate(gen,rank)
        instrumented=source.replace('  INTEGER I,J,K,L,M\n',
            '  INTEGER I,J,K,L,M\n  INTEGER FAST,FALLBACK\n  COMMON/SHARED_TRACE/FAST,FALLBACK\n')
        instrumented=instrumented.replace('PRODUCT_READY=.FALSE.',
                                           'FAST=FAST+1\nPRODUCT_READY=.FALSE.',1)
        instrumented=instrumented.replace('  CALL '+name+'(',
                                           '  FALLBACK=FALLBACK+1\n  CALL '+name+'(')
        instrumented=instrumented.replace('  CALL '+name+'_DENSE(',
                                           '  FALLBACK=FALLBACK+1\n  CALL '+name+'_DENSE(')
        for filename,text in [('constants.f',gen.write_polynomial_constant_module()),
                              ('full.f',gen.write_compact_wl_updater(rank,1)),
                              ('shared.f',source),('checked-shared.f',instrumented)]:
            with check.FortranWriter(str(work/filename)) as writer:
                writer.writelines(text)
        driver=check._SPARSE_DRIVER.replace('OLD_ROUTINE',name).replace('NEW_ROUTINE',name+'_SHARED')
        driver=driver.replace('checks=0','checks=0\nfast=0\nfallback=0',1)
        driver=driver.replace('integer :: i,j,k,ls,ins,outs,trial,rounding,checks',
            'integer :: i,j,k,ls,ins,outs,trial,rounding,checks,fast,fallback\n'
            'common/SHARED_TRACE/fast,fallback')
        driver=driver.replace('    saved_a=transfer(a,saved_a)',
            'if(mod(trial,2)==1) then\n'+copy_groups()+'\nendif\n    saved_a=transfer(a,saved_a)')
        driver=driver.replace("write(*,*) 'PASS',checks",
            "if(fast==0.or.fallback==0) stop 8\nwrite(*,*) 'PASS',checks,fast,fallback")
        (work/'driver.f90').write_text(driver)
        command=['gfortran','-O3','-frounding-math','-fno-automatic','-fcheck=all',
            '-ffixed-line-length-132','constants.f','full.f','checked-shared.f',
            'driver.f90','-o','check']
        start=time.monotonic()
        built=subprocess.run(command,cwd=work,capture_output=True,text=True)
        (work/'build.log').write_text(built.stdout+built.stderr)
        built.check_returncode()
        ran=subprocess.run([str(work/'check')],cwd=work,capture_output=True,text=True)
        (work/'run.log').write_text(ran.stdout+ran.stderr)
        results.append(dict(rank=rank,command=command,passed=ran.returncode==0,
            output=ran.stdout+ran.stderr,seconds=time.monotonic()-start,
            sources={p.name:check.sha(p) for p in work.glob('*.f*')},
            executable_sha256=check.sha(work/'check')))
        (HERE/'exact.json').write_text(json.dumps(dict(complete=False,results=results),indent=2)+'\n')
        ran.check_returncode()
        print(rank,ran.stdout.strip(),flush=True)
    result=dict(complete=True,acceptance_eligible=False,comparisons=12000,results=results,
        groups=shared.groups(),runner_sha256=check.sha(__file__),
        scope='QP full-output/input bits and sticky flags. Odd trials impose source-proved '
        'same-K B equality; other trials test rejection. Instrumented counters require '
        'both sharing and fallback. Includes directed rounding, exceptional inputs, '
        'cancellation, signed zero and empty dimensions; no process/timing claim.')
    (HERE/'exact.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
