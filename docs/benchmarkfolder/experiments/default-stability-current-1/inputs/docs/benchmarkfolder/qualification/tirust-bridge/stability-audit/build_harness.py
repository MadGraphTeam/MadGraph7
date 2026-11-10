#!/usr/bin/env python3
"""Build an isolated ABI around the current production MadLoop templates.

Run from the MadGraph checkout after sourcing tools/tirust_env.sh. This script
compiles only Fortran and never executes numerical code or rebuilds Rust.
Numerical consumers must acquire .tirust-build/numerical-tests.lock exclusively.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines

SOURCE = Path(__file__).resolve().parent
BUILD = ROOT / '.tirust-build/stability-audit'
TEMPLATES = ROOT / 'madgraph/iolibs/template_files/loop_optimized'
PARAMS = ROOT / 'Template/loop_material/StandAlone/SubProcesses'
BRIDGE = ROOT / '.tirust-build/bridge-qualification/libmadloop_tirust.so'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (BUILD / 'build.log').open('a') as stream:
        stream.write(json.dumps(arguments) + '\n' + result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f'Run from {ROOT}')
    if not BRIDGE.is_file():
        raise SystemExit(f'Missing qualified Rust bridge: {BRIDGE}')
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / 'build.log').write_text('')
    polynomial = FortranPolynomialRoutines(6)
    permutation = polynomial.get_TIRUST_mapping()
    mapping = '\n'.join('DATA (TIRUST_MAP(I),I=%d,%d)/%s/' % (
        start, min(start + 9, 209), ','.join(map(str, permutation[start:start + 10])))
        for start in range(0, 210, 10))
    replacements = dict(
        proc_prefix='AUDIT_', nloop_groups=1, nSquaredSO=1, maxrank=6,
        real_dp_format='REAL*8', complex_dp_format='COMPLEX*16',
        real_mp_format='REAL*16', complex_mp_format='COMPLEX*32',
        mass_dp_format='COMPLEX*16', include_vector='', tirust_coefmap=mapping,
        tirust_plan_registration='CALL TIRUST_REGISTER_PLAN(4,5,STATUS)')
    helpers = (TEMPLATES / 'CT_interface.inc').read_text()
    helpers = helpers[helpers.index('\t  SUBROUTINE %(proc_prefix)sBUILD_KINEMATIC_MATRIX'):]
    helpers = helpers[:helpers.index('## if(samurai_available)')]
    switch = (TEMPLATES / 'TIR_interface.inc').read_text()
    switch = switch[switch.index('      SUBROUTINE %(proc_prefix)sSWITCH_ORDER'):]
    switch = switch[:switch.index('\t  SUBROUTINE %(proc_prefix)sINITTIR')]
    adapter = (TEMPLATES / 'TIRUST_interface.inc').read_text() + helpers + switch
    with FortranWriter(str(BUILD / 'adapter.f')) as writer:
        writer.writelines(adapter % replacements, context={
            'tirust_available': True, 'tirust_supports_quad_prec': True,
            'AmplitudeReduction': True})
    accuracy = (TEMPLATES / 'loop_matrix_standalone.inc').read_text()
    accuracy = accuracy[accuracy.index('      SUBROUTINE %(proc_prefix)scompute_accuracy'):]
    accuracy = accuracy[:accuracy.index('      SUBROUTINE %(proc_prefix)sSET_N_EVALS')]
    with FortranWriter(str(BUILD / 'accuracy.f')) as writer:
        writer.writelines(accuracy % replacements)
    with FortranWriter(str(BUILD / 'driver.f')) as writer:
        writer.writelines((SOURCE / 'harness_driver.f').read_text())
    (BUILD / 'loop_max_coefs.inc').write_text(
        '      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n')
    (BUILD / 'coupl.inc').write_text(
        '      REAL*8 MU_R\n      COMMON/AUDIT_COUPL/MU_R\n')
    (BUILD / 'polynomial.f90').write_text(
        'module AUDIT_POLYNOMIAL_CONSTANTS\n'
        'implicit none\n'
        'integer,parameter :: NCOEF_R(0:6)=[1,5,15,35,70,126,210]\n'
        'end module\n')
    compiler = os.environ.get('FC', 'gfortran')
    common = [compiler, '-O3', '-fPIC', '-fcheck=all', '-ffixed-line-length-132',
              '-J' + str(BUILD), '-I' + str(BUILD), '-I' + str(PARAMS)]
    sources = [ROOT / 'vendor/tirust/mtirust.f90', BUILD / 'polynomial.f90',
               BUILD / 'adapter.f', BUILD / 'accuracy.f', BUILD / 'driver.f',
               PARAMS / 'MadLoopParamReader.f']
    objects = []
    for source in sources:
        output = BUILD / (source.stem + '.o')
        run(common + ['-c', str(source), '-o', str(output)])
        objects.append(str(output))
    compiler_runtime = run([compiler, '-print-libgcc-file-name']).strip()
    library = BUILD / 'libtirust_stability_audit.so'
    run([compiler, '-shared', '-Wl,-z,defs', *objects, compiler_runtime, '-lgcc_s',
         '-L' + str(BRIDGE.parent), '-lmadloop_tirust',
         '-Wl,-rpath,' + str(BRIDGE.parent), '-o', str(library)])
    provenance_inputs = [Path(__file__).resolve(), SOURCE / 'harness_driver.f',
                         ROOT / 'vendor/tirust/mtirust.f90',
                         TEMPLATES / 'TIRUST_interface.inc', TEMPLATES / 'CT_interface.inc',
                         TEMPLATES / 'TIR_interface.inc', TEMPLATES / 'loop_matrix_standalone.inc',
                         PARAMS / 'MadLoopParams.inc', PARAMS / 'MadLoopParamReader.f', BRIDGE,
                         *[BUILD / name for name in ('adapter.f', 'accuracy.f', 'driver.f')], library]
    metadata = {
        'compiler': run([compiler, '--version']).splitlines()[0],
        'library': str(library), 'bridge': str(BRIDGE),
        'coefficient_count': 126, 'propagators': 4, 'rank': 5,
        'max_generated_rank': 6, 'normalization': 'none: diagnostic tensor contraction',
        'runtime_lock': str(ROOT / '.tirust-build/numerical-tests.lock'),
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in provenance_inputs},
    }
    (BUILD / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(library)


if __name__ == '__main__':
    main()
