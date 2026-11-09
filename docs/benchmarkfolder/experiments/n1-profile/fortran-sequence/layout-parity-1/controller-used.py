#!/usr/bin/env python3
"""Build/run native DP/QP array-layout parity separately from amplitude timing.

The dyadic massive fixture was already used by the pole-engine tests.  This
driver exercises the actual generated adapter and on-shell matrix helpers for
n=1..6, rank=0..2, primary/check order, both precisions, with bounds checking.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fortran'))
from prepare import clone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid name')
    library=args.library.resolve(strict=True)
    report=HERE/args.name
    work=ROOT/'.tirust-build/n1-profile/fortran-sequence'/args.name
    report.mkdir(exist_ok=False)
    work.mkdir(parents=True,exist_ok=False)
    meta=json.loads((HERE/'adapter-sequence-build/build-metadata.json').read_text())
    manifest={'purpose':__doc__,'performance_eligible':False,'driver_sha256':digest(HERE/'layout-driver.f'),
              'controller_sha256':digest(Path(__file__)),'library':str(library),
              'library_sha256':digest(library),'variants':{},'ran':False}
    for variant,source in [('original',Path(meta['source'])),('scratch',Path(meta['output']))]:
        output=work/variant
        clone(source,output)
        process=output/'SubProcesses/P0_uux_z'
        if variant=='scratch' and digest(process/'TIRUST_interface.f')!=meta['source_sha256']:
            raise RuntimeError('Scratch source changed')
        if variant=='original' and digest(process/'TIRUST_interface.f')!=meta['original_source_sha256']:
            raise RuntimeError('Original source changed')
        shutil.copy2(HERE/'layout-driver.f',process/'check_sa.f')
        dry=subprocess.check_output(['make','--no-print-directory','--dry-run','--always-make',
                                     'SHELL='+shutil.which('bash'),'FC='+os.environ['FC'],'check'],
                                    cwd=process,text=True,stderr=subprocess.STDOUT)
        rows=[shlex.split(line) for line in dry.splitlines() if line.strip()]
        commands=[]
        for filename in ('TIRUST_interface.f','check_sa.f'):
            command=next(row for row in rows if '-c' in row and filename in row)
            commands.append([command[0],'-fcheck=all','-fbacktrace',*command[1:]])
        commands.append(next(row for row in rows if '-o' in row and row[row.index('-o')+1]=='check'))
        with (report/(variant+'-build.log')).open('w') as log:
            for command in commands:
                log.write(shlex.join(command)+'\n');log.flush()
                subprocess.run(command,cwd=process,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest['variants'][variant]={'directory':str(process),'source_sha256':digest(process/'TIRUST_interface.f'),
                                      'executable_sha256':digest(process/'check'),'commands':commands}
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,report/'controller-used.py')
    shutil.copy2(HERE/'layout-driver.f',report/'driver-used.f')
    if not args.run:
        print('Built only; numerical execution requires --run in a new named output.')
        return
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=str(library.parent)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for variant,row in manifest['variants'].items():
            process=Path(row['directory'])
            with (report/(variant+'-run.log')).open('w') as log:
                result=subprocess.run(['taskset','-c','9','./check'],cwd=process,env=env,
                                      stdout=log,stderr=subprocess.STDOUT,timeout=180)
            row['returncode']=result.returncode
            if (process/'layout-values.dat').exists():
                shutil.copy2(process/'layout-values.dat',report/(variant+'-values.dat'))
            (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(variant+' native layout run failed')
    left=(report/'original-values.dat').read_bytes()
    right=(report/'scratch-values.dat').read_bytes()
    manifest.update(ran=True,exact_output_and_input_mutation_parity=left==right,
                    original_values_sha256=hashlib.sha256(left).hexdigest(),
                    scratch_values_sha256=hashlib.sha256(right).hexdigest())
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if left!=right:
        raise RuntimeError('Original/scratch native output bytes differ')
    print(json.dumps({'exact':True,'manifest':str(report/'manifest.json')},indent=2))


if __name__=='__main__':
    main()
