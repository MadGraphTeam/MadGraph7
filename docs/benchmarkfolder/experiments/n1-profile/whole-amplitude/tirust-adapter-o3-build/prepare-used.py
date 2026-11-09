#!/usr/bin/env python3
"""Isolate a TIRUST_interface.f-only -O3 trial; compilation requires --build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'fortran'),str(HERE.parent/'fortran/protocol')]
from prepare import clone
from tirust_benchmark_compiler import _compile_command, COMPILER_ENVIRONMENT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid name')
    source=ROOT/'.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z'
    frozen=json.loads((ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json').read_text())
    process=source/'SubProcesses/P0_uux_z'
    if digest(process/'check')!=frozen['compilation']['executable_sha256']:
        raise RuntimeError('Original candidate executable changed')
    source_record=json.loads((HERE.parent/'fortran/preparation.json').read_text())
    if digest(process/'TIRUST_interface.f')!=source_record['source_sha256']['TIRUST_interface.f']:
        raise RuntimeError('Generated adapter source changed after attribution freeze')
    report=HERE/args.name
    destination=ROOT/'.tirust-build/n1-profile/whole-amplitude'/args.name
    report.mkdir(parents=True,exist_ok=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    clone(source,destination)
    process=destination/'SubProcesses/P0_uux_z'
    make=['make','--no-print-directory','--dry-run','--always-make',
          'SHELL='+shutil.which('bash'),'FC='+os.environ['FC'],'check']
    dry=subprocess.check_output(make,cwd=process,text=True,stderr=subprocess.STDOUT)
    (report/'original-expanded-recipes.log').write_text(dry)
    commands=[shlex.split(line) for line in dry.splitlines() if line.strip()]
    compiles=[row for row in commands if '-c' in row and 'TIRUST_interface.f' in row]
    links=[row for row in commands if '-o' in row and row[row.index('-o')+1]=='check']
    if len(compiles)!=1 or len(links)!=1:
        raise RuntimeError('Expected one original adapter compile and one final link recipe')
    original=compiles[0]
    parsed=_compile_command(shlex.join(original),process,dict(os.environ),{})
    expected_recipe=frozen['compilation']['effective_compiler_flags']['scopes']['process']['recipes']['TIRUST_interface.f']
    if parsed!=('TIRUST_interface.f',expected_recipe):
        raise RuntimeError('Original adapter compiler/flags differ from the frozen build')
    injected=set(COMPILER_ENVIRONMENT)|{k for k in os.environ if k.startswith(('NIX_CFLAGS_','NIX_LDFLAGS_'))}
    actual_environment={k:os.environ[k] for k in sorted(injected) if k in os.environ}
    if actual_environment!=frozen['compilation']['effective_compiler_flags']['environment']:
        raise RuntimeError('Injected compiler environment changed')
    archive_pins={}
    for path,expected in frozen['runtime']['archives'].items():
        item=Path(path)
        if item.is_relative_to(source):
            item=destination/item.relative_to(source)
        archive_pins[str(item)]=expected
        if digest(item)!=expected:
            raise RuntimeError('Static link input changed: '+str(item))
    if any(x.startswith('-O') for x in original):
        raise RuntimeError('Original adapter already has optimization flags; reassess trial')
    optimized=[original[0],'-O3',*original[1:]]
    before={p.name:digest(p) for p in process.glob('*.o')}
    manifest={'performance_eligible':False,'scope':'TIRUST_interface.f only; all common objects unchanged',
              'output':str(destination),'source':str(source),'prepare_sha256':digest(Path(__file__)),
              'source_sha256':digest(process/'TIRUST_interface.f'),'original_executable_sha256':digest(process/'check'),
              'original_compile':original,'candidate_compile':optimized,'link':links[0],
              'compiler_version':subprocess.check_output([original[0],'--version'],text=True),
              'frozen_recipe_verified':parsed[1],'compiler_environment':actual_environment,
              'static_archive_pins':archive_pins,'objects_before':before,'built':False}
    (report/'build-metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.build:
        print('Prepared only; no compiler or native numerical process launched.')
        return
    for name in ('TIRUST_interface.o','check'):
        if not (process/name).resolve().is_relative_to(destination):
            raise RuntimeError('Build output resolves outside isolated clone: '+name)
    with (report/'compile-link.log').open('w') as log:
        log.write('Reconstruct original executable before changing the adapter:\n'+shlex.join(links[0])+'\n');log.flush()
        subprocess.run(links[0],cwd=process,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest['reconstructed_original_executable_sha256']=digest(process/'check')
        (report/'build-metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if manifest['reconstructed_original_executable_sha256']!=manifest['original_executable_sha256']:
            raise RuntimeError('Untouched relink does not reconstruct frozen executable; no O3 trial built')
        for command in (optimized,links[0]):
            log.write(shlex.join(command)+'\n');log.flush()
            subprocess.run(command,cwd=process,stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(digest(Path(path))!=expected for path,expected in archive_pins.items()):
        raise RuntimeError('Static archive changed during adapter-only build')
    after={p.name:digest(p) for p in process.glob('*.o')}
    changes=[name for name in before if before[name]!=after[name]]
    if changes!=['TIRUST_interface.o']:
        raise RuntimeError('Objects changed outside the isolated adapter: '+repr(changes))
    manifest.update(built=True,objects_after=after,changed_objects=changes,
                    executable_sha256=digest(process/'check'),driver_sha256=digest(process/'check_sa.f'))
    (report/'build-metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(destination),'changed_objects':changes,
                      'executable_sha256':manifest['executable_sha256']},indent=2))


if __name__=='__main__':
    main()
