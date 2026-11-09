#!/usr/bin/env python3
"""Prepare isolated generated-code attribution; no compiler or numerical runs."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z'
DESTINATION = ROOT / '.tirust-build/n1-profile/fortran'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one instrumentation anchor: {old!r}')
    return text.replace(old, new)


def start():
    return '      CALL ML_PROFILE_START(ML_PT0)\n'


def end(stage, precision):
    return f'      CALL ML_PROFILE_END({stage},CTMODE,NLOOPLINE,RANK,{precision},ML_PT0)\n'


def instrument_adapter(text):
    for name, precision in [('ML5_0_TIRUST_TENSOR_DIAGNOSTIC', 1),
                            ('ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', 2)]:
        match = re.search(r'^      SUBROUTINE '+name+r'\(.*?^      END\n', text, re.M | re.S)
        if match is None:
            raise ValueError('Missing generated diagnostic: '+name)
        routine = match.group()
        routine = once(routine, '      USE MTIRUST\n', '      USE ML_PROFILE_STATE, ONLY: ML_PROFILE_START,ML_PROFILE_END\n      USE MTIRUST\n')
        routine = once(routine, '      IMPLICIT NONE\n', '      IMPLICIT NONE\n      INTEGER*8 ML_PT0\n')
        routine = once(routine, '      STABLE=.FALSE.\n', start()+'      STABLE=.FALSE.\n')
        routine = once(routine, '      PDEN_DUMMY=0\n', end(1,precision)+start()+'      PDEN_DUMMY=0\n')
        call = re.search(r'^      CALL ML5_0_(?:MP_)?BUILD_KINEMATIC_MATRIX\([^\n]*(?:\n     \$[^\n]*)*\n', routine, re.M)
        if call is None:
            raise ValueError('Missing kinematic matrix call')
        routine = routine[:call.start()]+end(2,precision)+start()+call.group()+end(3,precision)+start()+routine[call.end():]
        call = re.search(r'^      CALL TIRUST_TENSOR_EVALUATE\([^\n]*(?:\n     \$[^\n]*)*\n', routine, re.M)
        if call is None:
            raise ValueError('Missing bridge call')
        routine = routine[:call.start()]+end(4,precision)+start()+call.group()+end(5,precision)+start()+routine[call.end():]
        routine = once(routine, '      STABLE=.TRUE.\n', end(6,precision)+'      STABLE=.TRUE.\n')
        text = text[:match.start()]+routine+text[match.end():]
    return text


def instrument_contraction(text):
    first, rest = text.split('      END\n', 1)
    first = once(first, '      USE, INTRINSIC :: IEEE_ARITHMETIC',
                 '      USE ML_PROFILE_STATE, ONLY: ML_PROFILE_START,ML_PROFILE_END\n      USE, INTRINSIC :: IEEE_ARITHMETIC')
    marker = 'C     Reject unavailable OPP backends'
    first = once(first, marker, '      INTEGER*8 ML_PT0\n'+marker)
    first = once(first, '      DO I=1,3\n        RES(I)=(0.0D0,0.0D0)\n',
                 start()+'      DO I=1,3\n        RES(I)=(0.0D0,0.0D0)\n')
    marker = 'C     IF(MLReductionLib(I_LIB).EQ.2) THEN'
    first = once(first, marker, end(7,1)+marker)
    return first+'      END\n'+rest


def instrument_worker(text):
    text = once(text, '      IMPLICIT NONE\n', '      USE ML_PROFILE_STATE, ONLY: ML_PROFILE_RESET,ML_PROFILE_WRITE\n      IMPLICIT NONE\n')
    text = once(text, '      TIMED_CODES=0\n', '      TIMED_CODES=0\n      CALL ML_PROFILE_RESET()\n')
    return once(text, '      ELAPSED=DBLE(T1-T0)/DBLE(RATE)\n',
                '      ELAPSED=DBLE(T1-T0)/DBLE(RATE)\n      CALL ML_PROFILE_WRITE()\n')


def clone(source, destination):
    # Preserve existing compiled libraries and objects, but every mutable file
    # is copied, never hardlinked. Internal relative symlinks stay internal.
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('worker-raw', 'worker-protocol.fifo', '__pycache__'))
    for path in destination.rglob('*'):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            relative = path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(destination / relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-library', required=True, type=Path)
    parser.add_argument('--allow-pending-bridge', action='store_true',
                        help='Prepare/compile Fortran objects before the isolated DSO is linked')
    args = parser.parse_args()
    bridge = args.bridge_library.resolve()
    if not bridge.is_file() and not args.allow_pending_bridge:
        raise ValueError('Compile the isolated profiling bridge first')
    destination = DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'performance_eligible': False, 'source': str(SOURCE),
                'bridge': {'path': str(bridge), 'sha256': digest(bridge) if bridge.is_file() else None},
                'stages': {1:'initialization_and_budget', 2:'ordered_momenta_and_shifts',
                           3:'on_shell_kinematic_matrix', 4:'invariant_unpack_and_options',
                           5:'fortran_wrapper_and_rust', 6:'mapping_and_validation',
                           7:'contraction_normalization_final_checks'},
                'source_sha256': {}, 'instrumented_sha256': {}}
    for label in ('pristine', 'instrumented'):
        clone(SOURCE, destination / label)
    output = destination / 'instrumented'
    process = output / 'SubProcesses/P0_uux_z'
    changes = [('TIRUST_interface.f', instrument_adapter),
               ('TIR_interface.f', instrument_contraction), ('check_sa.f', instrument_worker)]
    for filename, transform in changes:
        path = process / filename
        original = path.read_text()
        changed = transform(original)
        path.write_text(changed)
        manifest['source_sha256'][filename] = digest(SOURCE / 'SubProcesses/P0_uux_z' / filename)
        manifest['instrumented_sha256'][filename] = digest(path)
        (destination / (filename+'.diff')).write_text(''.join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True), 'original/'+filename, 'instrumented/'+filename)))
    shutil.copy2(HERE / 'ml_profile.f90', process / 'ml_profile.f90')
    # The new DSO has the original SONAME; replace its search path explicitly.
    definitions = output / 'SubProcesses/MadLoop_makefile_definitions'
    contents = definitions.read_text()
    contents = once(contents, '-lmadloop_tirust', str(bridge))
    installed = ROOT / 'HEPTools/tirust/lib'
    contents = contents.replace('-Wl,-rpath,'+str(installed), '-Wl,-rpath,'+str(bridge.parent))
    definitions.write_text(contents)
    makefile = output / 'SubProcesses/makefile'
    makefile.write_text(makefile.read_text()+'''
# Isolated attribution only; original optimization flags are preserved.
PROCESS += ml_profile.o
TIRUST_interface.o TIR_interface.o check_sa.o: ml_profile.o
ml_profile.o: ml_profile.f90
\t$(FC) $(FFLAGS) -ffree-line-length-none -c $< -o $@
''')
    manifest['helper_sha256'] = digest(process / 'ml_profile.f90')
    manifest['worker_template_sha256'] = digest(ROOT / 'tools/tirust_benchmark_worker.f')
    manifest['compilation'] = json.loads((SOURCE / 'compilation.json').read_text())
    manifest['cards_sha256'] = {p.name:digest(p) for p in sorted((SOURCE/'Cards').glob('*.dat'))}
    manifest['prepare_sha256'] = digest(__file__)
    (destination / 'preparation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(destination / 'preparation.json')


if __name__ == '__main__':
    main()
