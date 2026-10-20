#!/usr/bin/env python3
"""Build identical private workers differing only in their NinjaRS archive."""
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from build import HERE, ROOT, WORK, binding, sha, write
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy

ORIGIN = ROOT / '.benchmark-work/policy-v2/generation-abi3-tensor-factor-v1'
ORIGIN_REPORT = ROOT / 'docs/benchmarkfolder/policy-v2/generation-abi3-tensor-factor-v1'
CORRECTED = ROOT / '.benchmark-work/policy-v2/install-export-abi3-zero-sign-v1/source/HEPTools/ninjars-unavailable'

def relink_copy(source, target):
    subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(target)], check=True)
    for path in target.rglob('*'):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            replacement = target / path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(replacement, path.parent))

def metadata(output):
    return {n: {'sha256':sha(p),'bytes':p.stat().st_size}
        for n,p in policy.runtime_metadata_files(output).items()}

def main():
    assert Path.cwd() == ROOT
    os.sched_setaffinity(0, {234})
    compiled = json.loads((HERE / 'package.json').read_text())
    assert compiled['schema'] == 'allocation-process-packaged-archive-v1'
    allocation = ROOT / compiled['allocation_archive']['path']
    assert binding(allocation) == compiled['allocation_archive']
    assert binding(CORRECTED / 'lib/libninjars.a') == compiled['baseline_archive']
    config_path = ROOT / 'docs/benchmarkfolder/policy-v2/cases.json'
    config = json.loads(config_path.read_text())
    assert config['points_per_case'] == 100
    assert config['madloop_parameters'] == {'MLStabThres':1e-9,'COLLIERRequiredAccuracy':1e-12}
    cases = [next(c for c in config['cases'] if c['id'] == name) for name in ('qcd-3','ew-2')]
    report = HERE / 'processes-v2'
    report.mkdir()
    write(report / 'cases.json', config)
    record = {'schema':'allocation-process-preparation-v1','complete':False,
        'amplitudes_evaluated':False,'acceptance_eligible':False,'production_changed':False,
        'cpu':234,'jobs':1,'controller':binding(Path(__file__)),
        'build_result':binding(HERE / 'package.json'),
        'config':binding(config_path),'worker_protocol':workers.protocol_descriptor(),
        'compiler':subprocess.check_output(['gfortran','--version'],text=True),
        'builds':{}}
    write(report / 'started.json', record)
    try:
        for case in cases:
            name = case['id']
            original = ORIGIN / 'outputs/candidate' / name
            original_sub = next(original.glob('SubProcesses/P*_*'))
            origin_manifest = bench.generated_source_manifest(original)
            expected = json.loads((ORIGIN_REPORT / 'manifests' / f'{name}-generated.json').read_text())
            # The retained generated manifest precedes the common benchmark driver.
            compilation = json.loads((original / 'compilation.json').read_text())
            assert origin_manifest['files_sha256'] == compilation['generated_sources_sha256']
            assert sha(original_sub / 'check') == compilation['executable_sha256']
            points = ROOT / 'docs/benchmarkfolder/points' / f'{name}.dat'
            assert int(points.read_text().splitlines()[0]) == 100
            point_target = report / f'{name}-points.dat'
            shutil.copy2(points, point_target)
            record['builds'][name] = {}
            for arm, archive in [('original',CORRECTED / 'lib/libninjars.a'),('allocation',allocation)]:
                job = WORK / 'processes-v2' / arm
                output = job / 'outputs/candidate' / name
                output.parent.mkdir(parents=True, exist_ok=True)
                assert not output.exists()
                relink_copy(original, output)
                sub = next(output.glob('SubProcesses/P*_*'))
                assert bench.generated_source_manifest(output) == origin_manifest
                retained = report / name / arm
                retained.mkdir(parents=True)
                # Relative text is identical in both arms, with private archive bytes.
                private = output / 'lib/ninjars_lib'
                assert private.is_symlink()
                private.unlink()
                private.mkdir()
                for p in sorted((CORRECTED / 'lib').iterdir()):
                    shutil.copy2(archive if p.name == 'libninjars.a' else p, private / p.name)
                for p in sorted((CORRECTED / 'include').iterdir()):
                    shutil.copy2(p, private / p.name)
                include = output / 'lib/ninjars_include'
                include.unlink()
                include.symlink_to('ninjars_lib')
                definitions = output / 'SubProcesses/MadLoop_makefile_definitions'
                old = str(ROOT / 'HEPTools/ninjars/lib')
                contents = definitions.read_text()
                assert contents.count(old) == 4
                definitions.write_text(contents.replace(old, '$(LIBDIR)ninjars_lib'))
                with workers.isolated_harness(job, retained):
                    settings = bench.configure('candidate', case, 8, config=config)
                assert settings['mlstabthres'] == 1e-9 and settings['collierrequiredaccuracy'] == 1e-12
                assert settings['selected_backend_qp_integrand'] is True
                actual_card = workers.validate_param_card(sub, (output / 'Cards/param_card.dat').read_bytes())
                prefix = re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION', (sub / 'check_sa.f').read_text(), re.I).group(1)
                born = sub / 'nsqso_born.inc'
                nborn = int(re.search(r'NSQSO_BORN\s*=\s*(\d+)', born.read_text(), re.I).group(1)) if born.exists() else 0
                text = (ROOT / 'tools/ninjars_benchmark_worker.f').read_text().replace('@PREFIX@',prefix).replace('@NEXTERNAL@',str(case['nfinal']+2)).replace('@NBORN@',str(nborn))
                driver = sub / 'check_sa.f'
                driver.unlink()
                driver.write_text(text)
                (sub / 'check_sa.o').unlink(missing_ok=True)
                (sub / 'check').unlink()
                manifest = bench.generated_source_manifest(output)
                write(retained / 'compiled-sources.json', manifest)
                env = workers.environment(job)
                command = ['make','-j1','check']
                start = time.monotonic()
                with (retained / 'build.log').open('x') as log:
                    result = subprocess.run(command,cwd=sub,env=env,stdout=log,stderr=subprocess.STDOUT)
                assert result.returncode == 0
                assert bench.generated_source_manifest(output) == manifest
                for p in [driver, output / 'Source/make_opts', definitions, output / 'Cards/param_card.dat', output / 'Cards/MadLoopParams.dat']:
                    shutil.copy2(p, retained / p.name)
                build = {'output':str(output.relative_to(ROOT)), 'directory':str(sub.relative_to(ROOT)),
                    'case':case,'arm':arm,'backend':8,'command':command,
                    'compilation_seconds':time.monotonic()-start,'build_kind':'driver-relink; cached identical generated/model/DHELAS objects',
                    'origin_compilation':compilation,'origin_manifest_sha256':origin_manifest['files_sha256'],
                    'executable':binding(sub / 'check'),'driver':binding(driver),
                    'generated_sources_sha256':manifest['files_sha256'],
                    'libraries':policy.library_hashes(output),'runtime_metadata':metadata(output),
                    'header_sha256':sha(private / 'ninjars.h'),'module_source_sha256':sha(private / 'mninjars.f90'),
                    'link_inputs':{n:sha(output / n) for n in policy.RUNTIME_LINK_INPUTS},
                    'points':binding(point_target),'parameter_card':binding(output / 'Cards/param_card.dat'),
                    'madloop_card':binding(output / 'Cards/MadLoopParams.dat'),
                    'actual_parameter_card':binding(actual_card),'shared_numerical_settings':settings}
                record['builds'][name][arm] = build
                write(retained / 'build.json', build)
                print(name,arm,'worker compiled, no evaluation',flush=True)
            a,b = [record['builds'][name][arm] for arm in ['original','allocation']]
            for key in ['generated_sources_sha256','runtime_metadata','header_sha256','module_source_sha256','link_inputs','shared_numerical_settings']:
                assert a[key] == b[key], (name,key)
            for key in ['parameter_card','madloop_card']:
                assert a[key]['sha256'] == b[key]['sha256']
            differing = [n for n in a['libraries'] if a['libraries'][n] != b['libraries'][n]]
            assert differing == ['ninjars_lib/libninjars.a'], differing
            assert set(a['libraries']) == set(b['libraries'])
            assert a['libraries'][differing[0]] == compiled['baseline_archive']['sha256']
            assert b['libraries'][differing[0]] == compiled['allocation_archive']['sha256']
            assert bench.generated_source_manifest(original) == origin_manifest
        record['complete'] = True
    except BaseException as exc:
        record['failure'] = repr(exc)
        raise
    finally:
        write(report / 'preparation.json', record)
    print(json.dumps({'complete':True,'numerical_execution':False}),flush=True)

if __name__ == '__main__':
    main()
