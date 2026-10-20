#!/usr/bin/env python3
"""Prepare uninstrumented isolated process clones; never evaluate amplitudes."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path[:0] = [str(ROOT/'tools'), str(BASE/'process-qcd3-v1')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import run as control

WORK = ROOT/'.benchmark-work/experiments/symmetric-factorization-dp/process-timing21-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    os.sched_setaffinity(0,{5})
    assert not WORK.exists() and not (HERE/'preparation.json').exists()
    completed = json.loads((BASE/'process-qcd3-v1/backend-6/result.json').read_text())
    assert completed['complete'] and completed['comparison']['passed']
    bound_preparation = json.loads((BASE/'process-qcd3-v1/preparation.json').read_text())
    bound_runtime = json.loads((BASE/'process-qcd3-v1/runtime.json').read_text())
    WORK.mkdir(parents=True)
    result = dict(complete=False, acceptance_eligible=False, amplitudes_evaluated=False,
        plan_sha256=sha(HERE/'PLAN.md'), script_sha256=sha(Path(__file__)),
        prerequisite_ninja6_sha256=sha(BASE/'process-qcd3-v1/backend-6/result.json'), builds={})
    dump(HERE/'preparation.json',result)
    source_original = control.WORK/'original'
    original_sub = control.directory('original')
    original_polynomial = (original_sub/'polynomial.f').read_bytes()
    original_driver = (original_sub/'check_sa.f').read_bytes()
    try:
        for variant, source in (('original',source_original),('factor',control.WORK/'factor-trace')):
            source_name = 'original' if variant == 'original' else 'factor-trace'
            source_sub = next(source.glob('SubProcesses/P*_*'))
            bound_build = bound_preparation['builds'][source_name]
            bound_manifest = json.loads((BASE/'process-qcd3-v1'/source_name/'compiled-sources.json').read_text())
            assert bench.generated_source_manifest(source) == bound_manifest
            assert bound_manifest['files_sha256'] == bound_build['generated_sources_sha256']
            assert sha(source_sub/'check') == bound_build['executable_sha256']
            runtime = bound_runtime['variants'][source_name]
            assert policy.library_hashes(source) == runtime['libraries']
            assert {n:policy.bound_file(p) for n,p in policy.runtime_metadata_files(source).items()} == runtime['metadata']
            output = WORK/variant
            subprocess.run(['cp','-a','--reflink=auto',str(source),str(output)],check=True)
            for path in output.rglob('*'):
                if path.is_symlink() and path.resolve().is_relative_to(source):
                    target=output/path.resolve().relative_to(source)
                    path.unlink()
                    path.symlink_to(os.path.relpath(target,path.parent))
            sub = next(output.glob('SubProcesses/P*_*'))
            removed=[]
            if variant=='factor':
                path=sub/'polynomial.f'
                contents=path.read_text()
                assert contents.encode().startswith(original_polynomial)
                lines=[]
                for line in contents.splitlines(True):
                    if 'DP_FACTOR_COUNTS' in line:
                        body=re.sub(r'\s+','',line[6:])
                        assert (body in ('INTEGER*8DP_FACTOR_COUNTS(3,4)',
                                         'COMMON/DP_FACTOR_TRACE/DP_FACTOR_COUNTS') or
                                re.fullmatch(r'DP_FACTOR_COUNTS\(([123]),([1-4])\)=DP_FACTOR_COUNTS\(\1,\2\)\+1',body)),body
                        assert line[5] in ' 0'
                        removed.append(line.rstrip())
                    else:
                        lines.append(line)
                assert len(removed)>32
                path.write_text(''.join(lines))
                assert 'DP_FACTOR_COUNTS' not in path.read_text()
                assert path.read_bytes().startswith(original_polynomial)
            (sub/'check_sa.f').write_bytes(original_driver)
            for path in sub.glob('mp_*.f'):
                assert path.read_bytes()==(original_sub/path.name).read_bytes()
            record_dir=HERE/variant
            record_dir.mkdir()
            for path in [*sub.glob('coef_construction_*.f'),sub/'polynomial.f',sub/'check_sa.f',
                         sub/'coef_specs.inc',sub/'loop_max_coefs.inc',sub/'makefile']:
                shutil.copy2(path,record_dir/path.name)
            manifest=bench.generated_source_manifest(output)
            dump(record_dir/'compiled-sources.json',manifest)
            built=control.command(['make','-j2','check'],sub,record_dir/'build.log')
            result['builds'][variant]=dict(**built,removed_trace_lines=removed,
                executable_sha256=sha(sub/'check'),generated_sources_sha256=manifest['files_sha256'],
                input_verified_before_copy=dict(source_variant=source_name,
                    executable_sha256=bound_build['executable_sha256'],
                    generated_sources_sha256=bound_manifest['files_sha256'],
                    library_and_module_binding_sha256=sha(BASE/'process-qcd3-v1/runtime.json')),
                libraries=policy.library_hashes(output),
                metadata={n:policy.bound_file(p) for n,p in policy.runtime_metadata_files(output).items()},
                link_inputs={n:sha(output/n) for n in policy.RUNTIME_LINK_INPUTS})
            assert result['builds'][variant]['libraries'] == runtime['libraries']
            assert {n:r['sha256'] for n,r in result['builds'][variant]['metadata'].items()} == {
                n:r['sha256'] for n,r in runtime['metadata'].items()}
            dump(HERE/'preparation.json',result)
            assert built['returncode']==0
            assert bench.generated_source_manifest(output)==manifest
            print(variant,'uninstrumented build complete',flush=True)
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        dump(HERE/'preparation.json',result)


if __name__=='__main__':
    main()
