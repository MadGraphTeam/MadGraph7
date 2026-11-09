#!/usr/bin/env python3
"""Compare actual rendered TIR routines using recording reduction stubs.

No native reduction library is linked.  This is a semantics/IEEE regression,
not an amplitude or performance measurement.  Generated sources and output are
retained below --work-dir; the JSON report goes to stdout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, directory, expect_success=True):
    result = subprocess.run([str(arg) for arg in args], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=45)
    if expect_success and result.returncode:
        raise RuntimeError('%s\n%s' % (args, result.stdout[-4000:].decode(errors='replace')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path,
                        default=ROOT / '.tirust-build/fortran-common-allocation-layout')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('gfortran')
    if not compiler:
        raise RuntimeError('gfortran is required')
    spec = importlib.util.spec_from_file_location('allocation_transform', HERE / 'transform.py')
    transform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform)
    template_path = ROOT / 'madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc'
    template = template_path.read_text()
    # Keep the actual common entry, both actual switches, and actual cache
    # helpers.  Only unrelated initialization/library-selection routines are
    # replaced by recording stubs.
    source = template[:template.index('\t  SUBROUTINE %(proc_prefix)sINITTIR')]
    source += template[template.index('\t  SUBROUTINE %(proc_prefix)sCLEAR_TIR_CACHE'):]
    driver_path = HERE / 'layout_driver.f'
    driver = driver_path.read_text()
    substitutions = dict(proc_prefix='TEST_', info_lines='', process_lines='',
                         nloop_groups=2, nSquaredSO=2, nexternal=3,
                         real_dp_format='REAL*8', complex_dp_format='COMPLEX*16',
                         mass_dp_format='COMPLEX*16', real_mp_format='REAL*16',
                         complex_mp_format='COMPLEX*32', include_vector='')
    configurations = [('cache2-interference', True, 2, False),
                      ('cache2-amplitude', True, 2, True),
                      ('cache0-interference', True, 0, False),
                      ('uncached-amplitude', False, 0, True)]
    builds = [('O0', ['-O0']), ('O3', ['-O3']),
              ('O3-static', ['-O3', '-fno-automatic'])]
    report = dict(schema='madloop-common-allocation-stub-parity-v1',
                  scope='Rendered common Fortran routines with recording stubs; no native integrals or performance claims.',
                  compiler=command([compiler, '--version'], ROOT).stdout.decode().splitlines()[0],
                  platform=platform.platform(), byteorder=sys.byteorder,
                  template_sha256=digest(template_path),
                  transformer_sha256=digest(HERE / 'transform.py'),
                  driver_sha256=digest(driver_path), program_sha256=digest(Path(__file__)),
                  configurations=[], comparisons=[], mutation_controls=[],
                  unsupported_configurations=[])
    for label, caching, cache_size, amplitude in configurations:
        directory = work / label
        directory.mkdir(exist_ok=True)
        context = dict(TIRCaching=caching, AmplitudeReduction=amplitude,
                       tirust_available=True, golem_available=True,
                       pjfry_available=True, iregi_available=True,
                       collier_available=True)
        with FortranWriter(str(directory / 'original.f')) as writer:
            writer.writelines(source % substitutions, context=context)
        original = (directory / 'original.f').read_text()
        for variant in transform.VARIANTS:
            (directory / (variant + '.f')).write_text(transform.transform(original, variant))
        (directory / 'loop_max_coefs.inc').write_text(
            '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=15)\n')
        (directory / 'tir_cache_size.inc').write_text(
            '      PARAMETER(TIR_CACHE_SIZE=%d)\n' % cache_size)
        (directory / 'coupl.inc').write_text(
            '      REAL*8 MU_R\n      COMMON/TEST_COUPL/MU_R\n')
        # These are precisely the parameters used by the extracted routine.
        # Keeping their types and COMMON layout explicit avoids initializing or
        # linking unrelated production parameter-reader code in this stub test.
        (directory / 'MadLoopParams.inc').write_text('''      INTEGER MLReductionLib(10),LoopPoleMode,IREGIMODE
      LOGICAL COLLIERUseInternalStabilityTest
      COMMON/TEST_PARAMS/MLReductionLib,LoopPoleMode,IREGIMODE,
     $ COLLIERUseInternalStabilityTest
''')
        rendered_driver = driver.replace('@CACHE@', '.TRUE.' if caching and cache_size else '.FALSE.')
        rendered_driver = rendered_driver.replace('@AMPLITUDE@', '.TRUE.' if amplitude else '.FALSE.')
        rendered_driver = rendered_driver.replace('@LOOPDIMS@', '0:14,2' if amplitude else '0:14,2,2')
        rendered_driver = rendered_driver.replace('@LOOPINIT@',
            'LOOPCOEFS(J,G)=DCMPLX(J+G,2*G-J)' if amplitude else
            'LOOPCOEFS(J,S,G)=DCMPLX(J+G+S,2*G-J+S)')
        with FortranWriter(str(directory / 'driver.f')) as writer:
            writer.writelines(rendered_driver)
        report['configurations'].append(dict(name=label, caching=caching,
            cache_size=cache_size, amplitude_reduction=amplitude,
            rendered_driver_sha256=digest(directory / 'driver.f'),
            include_sha256={name: digest(directory / name) for name in
                ('loop_max_coefs.inc', 'tir_cache_size.inc', 'coupl.inc', 'MadLoopParams.inc')}))
        if caching and cache_size == 0:
            # The unmodified generated source contains a constant MOD divisor
            # of zero in an unreachable ELSE arm.  GNU Fortran rejects it even
            # at O0.  Retain this baseline limitation instead of rewriting the
            # fixture or attributing it to an allocation variant.
            compile_command = [compiler, '-O0', '-fcheck=all,no-array-temps',
                '-ffixed-line-length-132', '-fbacktrace', '-frounding-math',
                'driver.f', 'original.f', '-o', directory / 'original-O0']
            compiled = command(compile_command, directory, expect_success=False)
            build_log = directory / 'original-O0.build.txt'
            build_log.write_bytes(compiled.stdout)
            if compiled.returncode:
                if b'shall not be zero' not in compiled.stdout:
                    raise RuntimeError('Unexpected zero-cache compile failure: ' + compiled.stdout.decode(errors='replace'))
                report['unsupported_configurations'].append(dict(configuration=label,
                    source_sha256=digest(directory / 'original.f'),
                    command=[str(x) for x in compile_command],
                    returncode=compiled.returncode, build_log_sha256=digest(build_log),
                    diagnostic=compiled.stdout.decode(errors='replace'),
                    outcome='unmodified baseline does not compile; variants not evaluated'))
                continue
        for build, optimization in builds:
            base_output = None
            for variant in ('original', *transform.VARIANTS):
                executable = directory / (variant + '-' + build)
                compile_command = [compiler, *optimization, '-fcheck=all,no-array-temps',
                    '-ffixed-line-length-132', '-fbacktrace', '-frounding-math',
                    'driver.f', variant + '.f', '-o', executable]
                compiled = command(compile_command, directory, expect_success=False)
                build_log = directory / (executable.name + '.build.txt')
                build_log.write_bytes(compiled.stdout)
                if compiled.returncode:
                    raise RuntimeError('Compile failed; retained %s\n%s' %
                        (build_log, compiled.stdout[-4000:].decode(errors='replace')))
                observed = command([executable], directory, expect_success=False)
                output_path = directory / (executable.name + '.out')
                output_path.write_bytes(observed.stdout)
                if observed.returncode:
                    raise RuntimeError('Driver failed; retained %s\n%s' %
                        (output_path, observed.stdout[-2000:].decode(errors='replace')))
                if not observed.stdout.rstrip().endswith(b'PASS'):
                    raise AssertionError('Driver did not finish: ' + str(output_path))
                if base_output is None:
                    base_output = observed.stdout
                elif observed.stdout != base_output:
                    raise AssertionError('Exact output parity failed: ' + str(output_path))
                report['comparisons'].append(dict(configuration=label, build=build,
                    variant=variant, command=[str(x) for x in compile_command],
                    source_sha256=digest(directory / (variant + '.f')),
                    output_sha256=digest(output_path), output_bytes=len(observed.stdout),
                    tir_records=sum(line.startswith(b'T ') for line in observed.stdout.splitlines()),
                    switch_records=sum(line.startswith(b'S ') for line in observed.stdout.splitlines()),
                    outcome='pass'))
        # Two deliberately broken copies demonstrate that the tests detect
        # missing cleanup under static locals and wrong public guard behavior.
        if label == 'cache2-interference':
            switch_source = (directory / 'switch-guard.f').read_text()
            mp_start = switch_source.index('      SUBROUTINE TEST_MP_SWITCH_ORDER(')
            mutation_sources = {
                'missing-cleanup': (directory / 'tir-alloc.f').read_text().replace(
                    '      IF(ALLOCATED(PCT)) DEALLOCATE(PCT,PDEN,M2LCT)\n', ''),
                'wrong-switch-guard': switch_source[:mp_start] + switch_source[mp_start:].replace(
                    'CTMODE.NE.2.AND.CTMODE.NE.5', 'CTMODE.NE.2.AND.CTMODE.NE.4', 1)}
            for name, mutated in mutation_sources.items():
                reference = 'tir-alloc' if name == 'missing-cleanup' else 'switch-guard'
                if mutated == (directory / (reference + '.f')).read_text():
                    raise AssertionError('Mutation did not change source: ' + name)
                (directory / (name + '.f')).write_text(mutated)
                executable = directory / name
                command([compiler, '-O3', '-fno-automatic', '-fcheck=all,no-array-temps',
                         '-ffixed-line-length-132', '-frounding-math', 'driver.f',
                         name + '.f', '-o', executable], directory)
                failed = command([executable], directory, expect_success=False)
                (directory / (name + '.out')).write_bytes(failed.stdout)
                if failed.returncode == 0:
                    raise AssertionError('Mutation unexpectedly passed: ' + name)
                expected = b'already allocated' if name == 'missing-cleanup' else b'input mutation'
                if expected not in failed.stdout:
                    raise AssertionError('Unexpected mutation failure: ' + failed.stdout.decode(errors='replace'))
                report['mutation_controls'].append(dict(name=name, returncode=failed.returncode,
                    source_sha256=digest(directory / (name + '.f')),
                    output_sha256=digest(directory / (name + '.out')),
                    diagnostic=[line.decode(errors='replace') for line in failed.stdout.splitlines()
                                if expected in line], outcome='detected'))
    report['outcome'] = 'pass'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
