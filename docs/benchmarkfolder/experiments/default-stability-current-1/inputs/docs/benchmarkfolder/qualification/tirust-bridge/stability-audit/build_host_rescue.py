#!/usr/bin/env python3
"""Compile the production automatic-rescue block with an injected integral callback.

No numerical execution; run from the MadGraph root with tools/tirust_env.sh.
The host block is copied verbatim from its opening IF through label 9999 and
H/T/U finalization, with only ordinary template rendering. Physics evaluation
at labels 200/300 is replaced by the existing N4R5 diagnostic adapter callback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter

SOURCE = Path(__file__).resolve().parent
BUILD = ROOT / '.tirust-build/host-rescue-audit'
NATIVE = ROOT / '.tirust-build/stability-audit/libtirust_stability_audit.so'
TEMPLATE = ROOT / 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc'
PARAMS = ROOT / 'Template/loop_material/StandAlone/SubProcesses'
GENERATED = ROOT / '.tirust-build/adapter-export/release-external/SubProcesses/P0_uux_z/loop_matrix.f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    with (BUILD / 'build.log').open('a') as stream:
        stream.write(json.dumps(args) + '\n' + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def main():
    global BUILD, NATIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-library', type=Path, default=NATIVE)
    parser.add_argument('--build-dir', type=Path, default=BUILD)
    args = parser.parse_args()
    BUILD = args.build_dir.resolve()
    NATIVE = args.native_library.resolve()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f'Run from {ROOT}')
    if not NATIVE.is_file():
        raise SystemExit('Build the native stability harness first.')
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / 'build.log').write_text('')
    source = TEMPLATE.read_text()
    start = source.index('IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN')
    end = source.index('C Finally for the summed result in ANS', start)
    policy = source[start:end]
    (BUILD / 'production-policy.inc').write_text(policy)
    commons = (PARAMS / 'MadLoopCommons.inc').read_text()
    cstart = commons.index('      INTEGER FUNCTION SET_RET_CODE_U')
    cend = commons.index('      SUBROUTINE DETECT_LOOPLIB', cstart)
    retcode = commons[cstart:cend]
    (BUILD / 'production-retcode.inc').write_text(retcode)
    generated = GENERATED.read_text()
    library_constants = generated[generated.index('      INTEGER NLOOPLIB'):
                                  generated.index('      INTEGER MAXSTABILITYLENGTH')]
    eligibility = generated[generated.index('      LOGICAL LOOPLIBS_AVAILABLE'):
                            generated.index('C     PS CAN POSSIBLY')]
    istart = generated.index('        QP_TOOLS_AVAILABLE=.FALSE.')
    eligibility_init = generated[istart:generated.index('C       Setup the file paths', istart)]
    (BUILD / 'production-eligibility.inc').write_text(library_constants + eligibility + eligibility_init)
    eval_start = source.index('      SUBROUTINE %(proc_prefix)sSET_N_EVALS')
    eval_end = source.index('C THIS SUBROUTINE SIMPLY SET THE GLOBAL PS', eval_start)
    set_evals = source[eval_start:eval_end]
    driver = (SOURCE / 'host_rescue_driver.f').read_text().replace('@HOST_POLICY@', policy)
    driver = driver.replace('@LIBRARY_CONSTANTS@', library_constants)
    driver = driver.replace('@LIBRARY_ELIGIBILITY@', eligibility)
    driver = driver.replace('@ELIGIBILITY_INIT@', eligibility_init) + retcode + set_evals
    with FortranWriter(str(BUILD / 'host_rescue.f')) as writer:
        writer.writelines(driver % {'proc_prefix': 'AUDIT_'}, context={'ninjars_available': True})
    compiler = os.environ.get('FC', 'gfortran')
    obj = BUILD / 'host_rescue.o'
    run([compiler, '-O3', '-fPIC', '-fcheck=all', '-ffixed-line-length-132',
         '-I' + str(PARAMS), '-c', str(BUILD / 'host_rescue.f'), '-o', str(obj)])
    runtime = run([compiler, '-print-libgcc-file-name']).strip()
    library = BUILD / 'libtirust_host_rescue.so'
    run([compiler, '-shared', '-Wl,-z,defs', str(obj), runtime, '-lgcc_s',
         '-L' + str(NATIVE.parent), '-ltirust_stability_audit',
         '-Wl,-rpath,' + str(NATIVE.parent), '-o', str(library)])
    files = [Path(__file__).resolve(), SOURCE / 'host_rescue_driver.f', TEMPLATE,
             PARAMS / 'MadLoopCommons.inc', PARAMS / 'MadLoopParams.inc', NATIVE, GENERATED,
             BUILD / 'production-eligibility.inc',
             BUILD / 'production-policy.inc', BUILD / 'production-retcode.inc',
             BUILD / 'host_rescue.f', library]
    metadata = {
        'library': str(library), 'compiler': run([compiler, '--version']).splitlines()[0],
        'policy_source': str(TEMPLATE.relative_to(ROOT)),
        'policy_start_line': source[:start].count('\n') + 1,
        'policy_end_line': source[:end].count('\n'),
        'eligibility_source': str(GENERATED.relative_to(ROOT)),
        'eligibility_initialization': 'Verbatim generated availability filters and QP index construction',
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'physics_callback': 'Injected N4R5 production diagnostic adapter at labels 200 and 300',
        'rotations': 'Callbacks select caller-provided original, rotation1 and rotation2 fixture arrays',
        'limitations': ['The full SLOOPMATRIX physics/filter initialization pipeline is not executed.',
            'Real and imaginary projections are independent synthetic interference diagnostics.',
            'This executes the production automatic branch and native accuracy/return-code functions.'],
    }
    (BUILD / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(library)


if __name__ == '__main__':
    main()
