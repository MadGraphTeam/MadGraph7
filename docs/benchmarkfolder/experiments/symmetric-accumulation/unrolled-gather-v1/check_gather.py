#!/usr/bin/env python3
"""Compile isolated gather and compare full output/input bits and IEEE flags."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import gather

ROOT,HERE = gather.ROOT,Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from madgraph.iolibs.file_writers import FortranWriter
from tests.unit_tests.various.test_q_polynomial_compiled import _SPARSE_DRIVER

WORK = ROOT/'.benchmark-work/experiments/symmetric-gather'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0,{2})
    results = []
    for kind in (8,16):
        for rank in (2,3):
            work=WORK/f'kind{kind}-rank{rank}'
            work.mkdir(parents=True,exist_ok=True)
            (work/'coef_specs.inc').write_text('      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n'
                '      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n')
            (work/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n'
                '      PARAMETER(LOOPMAXCOEFS=210)\n')
            gen=FortranPolynomialRoutines(6,updater_max_rank=2,proc_prefix='TEST_',
                sub_prefix='TEST_',coef_format=f'complex*{2*kind}')
            for filename,source in [
                ('constants.f',gen.write_polynomial_constant_module()),
                ('full.f',gen.write_compact_wl_updater(rank,1)),
                ('gather.f',gather.generate(gen,rank))]:
                with FortranWriter(str(work/filename)) as writer:
                    writer.writelines(source)
            name=f'TEST_UPDATE_WL_{rank}_1'
            driver=_SPARSE_DRIVER.replace('OLD_ROUTINE',name).replace('NEW_ROUTINE',name+'_GATHER')
            if kind==8:
                driver=(driver.replace('complex(16)','complex(8)').replace('real(16)','real(8)')
                    .replace('kind=16','kind=8').replace('_16','_8')
                    .replace('1.e2000','1.e200').replace('1.e-2000','1.e-200')
                    .replace('4*4*210*4','2*4*210*4').replace('4*4*15*4','2*4*15*4'))
            (work/'driver.f90').write_text(driver)
            command=['gfortran','-O3','-frounding-math','-fno-automatic',
                '-fcheck=all','-ffixed-line-length-132','constants.f','full.f','gather.f',
                'driver.f90','-o','check']
            start=time.monotonic()
            built=subprocess.run(command,cwd=work,capture_output=True,text=True)
            (work/'build.log').write_text(built.stdout+built.stderr)
            if built.returncode:
                raise RuntimeError((work/'build.log').read_text())
            ran=subprocess.run([str(work/'check')],cwd=work,capture_output=True,text=True)
            (work/'run.log').write_text(ran.stdout+ran.stderr)
            record=dict(kind=kind,rank=rank,command=command,passed=ran.returncode==0,
                seconds=time.monotonic()-start,output=ran.stdout+ran.stderr,
                sources={p.name:sha(p) for p in work.glob('*.f*')},executable_sha256=sha(work/'check'))
            results.append(record)
            (HERE/'gather-exact.json').write_text(json.dumps(dict(results=results,
                complete=False,acceptance_eligible=False),indent=2)+'\n')
            if ran.returncode:
                raise RuntimeError(record['output'])
            print(kind,rank,record['output'].strip(),flush=True)
    result=dict(complete=True,acceptance_eligible=False,comparisons=24000,results=results,
        scope='Synthetic full updater DP/QP bits, inputs and IEEE sticky flags with disabled '
        'traps, including exceptional/empty/rounding cases. This is not a full-process or '
        'performance result. Prototype falls back to public updater when any trap is enabled.',
        prototype_sha256=sha(HERE/'gather.py'),driver_sha256=sha(__file__),
        generator_sha256=sha(ROOT/'madgraph/various/q_polynomial.py'))
    (HERE/'gather-exact.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
