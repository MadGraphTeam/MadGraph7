#!/usr/bin/env python3
"""Held, isolated compilation only; never executes either diagnostic binary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/'.benchmark-work/experiments/ninjars-mp-transport-audit-v1'
DEPS=ROOT/'.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build/release/deps'
NATIVE=DEPS.parent/'build/gmp-mpfr-sys-fbaf675a2d7bb3d7/out/lib'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    binding=json.loads((HERE/'scalar-binding.json').read_text())
    assert sha(ROOT/binding['source_path'])==binding['source_sha256']
    assert sha(HERE/'scalar.rs')==binding['scalar_sha256']
    assert sha(HERE/'scalar-fixtures.json')==binding['fixture_sha256']
    WORK.mkdir(parents=True,exist_ok=False)
    for name in ('scalar.rs','real16_read.f90'):
        shutil.copyfile(HERE/name,WORK/name)
    ninjars=DEPS/'libninjars-029811cf3d12c72d.rlib'
    fingerprints=DEPS.parent/'.fingerprint'
    ninjars_fingerprint=fingerprints/'ninjars-029811cf3d12c72d/lib-ninjars.json'
    dependency=json.loads(ninjars_fingerprint.read_text())
    symbolica_identity=next(row[3] for row in dependency['deps'] if row[1]=='symbolica')
    symbolica_fingerprints=[path for path in fingerprints.glob('symbolica-*/lib-symbolica')
                           if path.read_text().strip()==symbolica_identity.to_bytes(8,'little').hex()]
    assert len(symbolica_fingerprints)==1
    symbolica=DEPS/('lib'+symbolica_fingerprints[0].parent.name+'.rlib')
    assert symbolica.is_file() and ninjars.is_file() and NATIVE.is_dir()
    commands=[
        ['taskset','-c','2','rustc','--edition=2024','-O',str(WORK/'scalar.rs'),
         '--extern',f'ninjars={ninjars}','--extern',f'symbolica={symbolica}',
         '-L',f'dependency={DEPS}','-L',f'native={NATIVE}','-o',str(WORK/'scalar')],
        ['taskset','-c','2','gfortran','-O2','-fno-fast-math','-frounding-math','-fno-automatic',
         str(WORK/'real16_read.f90'),'-o',str(WORK/'real16-read')],
    ]
    inputs={str(path.relative_to(ROOT)):sha(path) for path in sorted(DEPS.iterdir())
            if path.is_file() and path.suffix in ('.rlib','.so')}
    inputs.update({str(path.relative_to(ROOT)):sha(path) for path in sorted(NATIVE.iterdir()) if path.is_file()})
    for path in (ninjars_fingerprint,symbolica_fingerprints[0]):
        inputs[str(path.relative_to(ROOT))]=sha(path)
    result={'compiled':False,'executed':False,'commands':commands,'input_libraries':inputs,
            'source_binding_sha256':sha(HERE/'scalar-binding.json'),
            'source_sha256':{name:sha(HERE/name) for name in ('scalar.rs','real16_read.f90')},
            'compiler_versions':{name:subprocess.check_output([name,'--version'],text=True)
                                 for name in ('rustc','gfortran')},'attempts':[]}
    try:
        for index,command in enumerate(commands,1):
            start=time.monotonic()
            with (HERE/f'compile-{index}.stdout').open('x') as out,(HERE/f'compile-{index}.stderr').open('x') as err:
                completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=300)
            result['attempts'].append({'returncode':completed.returncode,'elapsed_seconds':time.monotonic()-start})
            assert completed.returncode==0
        for name in inputs:
            assert sha(ROOT/name)==inputs[name]
        result['executables']={name:{'path':str((WORK/name).relative_to(ROOT)),'sha256':sha(WORK/name)}
                               for name in ('scalar','real16-read')}
        result['compiled']=True
    except BaseException as exc:
        result['failure']=repr(exc)
        raise
    finally:
        with (HERE/'scalar-build.json').open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
