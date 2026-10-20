#!/usr/bin/env python3
"""Prepare/build the isolated old-runtime replay; never calls an engine."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import matrix

HERE=Path(__file__).resolve().parent
CAPTURE=HERE.parent
ROOT=HERE.parents[4]
WORK=ROOT/'.benchmark-work/experiments/ninjars-qp-numerator-dp-probe-v1/replay'
CAPTURE_WORK=WORK.parent/'qcd-3'
SEAL='03d63cee3b8c38ee1936f9662e816206f3b99f3a2ec55b8cc938c209ff8aefbd'
REVISION='f4bffc61ed67e5d083b80ff9901768d4d5779989'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    assert sha(CAPTURE/'manifest.json')==SEAL
    manifest=json.loads((CAPTURE/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=CAPTURE/name
        assert path.is_file() and not path.is_symlink()
        assert sha(path)==record['sha256'] and path.stat().st_size==record['bytes']
    runtime=json.loads((CAPTURE/'prepared.json').read_text())['runtime']
    consumed={'libninjars.a':runtime['libraries']['ninjars_lib/libninjars.a']}
    for name in ('mninjars.mod','ninjars-config.json','ninjars-link-before.txt','ninjars-native-libs.txt'):
        consumed[name]=runtime['metadata']['ninjars_lib/'+name]
    for name,digest in consumed.items():assert sha(CAPTURE_WORK/'lib/ninjars_lib'/name)==digest
    WORK.mkdir(parents=True,exist_ok=False)
    (WORK/'lib').mkdir()
    for name,digest in consumed.items():
        shutil.copyfile(CAPTURE_WORK/'lib/ninjars_lib'/name,WORK/'lib'/name)
        assert sha(WORK/'lib'/name)==digest
    # Retain the module/header source corresponding to the frozen numerical tree.
    for name in ('mninjars.f90','ninjars.h'):
        data=subprocess.check_output(['git','show',f'{REVISION}:vendor/ninjars/{name}'],cwd=ROOT)
        (HERE/name).write_bytes(data)
    inputs=matrix.input_text()
    (HERE/'inputs.hex').write_text(inputs)
    shutil.copyfile(HERE/'inputs.hex',WORK/'inputs.hex')
    shutil.copyfile(HERE/'driver.f90',WORK/'driver.f90')
    before=shlex.split((WORK/'lib/ninjars-link-before.txt').read_text())
    native=shlex.split((WORK/'lib/ninjars-native-libs.txt').read_text())
    assert before==['-lgcc_s']
    command=['taskset','-c','2','gfortran','-O2','-fno-fast-math','-frounding-math',
             '-I'+str(WORK/'lib'),str(WORK/'driver.f90'),'-L'+str(WORK/'lib'),
             *before,'-lninjars',*native,'-o',str(WORK/'replay')]
    result={'complete':False,'engine_calls':0,'capture_manifest_sha256':SEAL,
            'runtime_source_revision':REVISION,'runtime_assets':consumed,
            'protocol_sha256':sha(HERE/'PLAN.md'),'driver_sha256':sha(HERE/'driver.f90'),
            'matrix_sha256':sha(HERE/'matrix.py'),'preparer_sha256':sha(Path(__file__)),
            'inputs_sha256':sha(HERE/'inputs.hex'),'command':command,
            'compiler':subprocess.check_output(['gfortran','--version'],text=True),
            'module_source_sha256':sha(HERE/'mninjars.f90'),'header_sha256':sha(HERE/'ninjars.h')}
    start=time.monotonic()
    try:
        with (HERE/'compile.stdout').open('x') as out,(HERE/'compile.stderr').open('x') as err:
            completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=300)
        result['returncode']=completed.returncode
        assert completed.returncode==0
        for name,digest in consumed.items():assert sha(WORK/'lib'/name)==digest
        assert sha(WORK/'driver.f90')==result['driver_sha256']
        assert sha(WORK/'inputs.hex')==result['inputs_sha256']
        result['executable_sha256']=sha(WORK/'replay')
        result['complete']=True
    except BaseException as exc:
        result['failure']=repr(exc)
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        with (HERE/'prepared.json').open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
