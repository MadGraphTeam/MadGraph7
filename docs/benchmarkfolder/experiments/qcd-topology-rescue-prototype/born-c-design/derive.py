#!/usr/bin/env python3
"""Prepare/verify a private C-coordinate handoff patch; no compiler/native work.

Run from the MadGraph root. --prepare creates the retained patch and bindings
once; default execution verifies them. transform_host(text) also accepts the
bare revision-2 host and preserves any existing QOBS blocks verbatim.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = Path('docs/benchmarkfolder/experiments/qcd-born-fallback-prototype')
OBSERVER = Path('docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer')
PROCESS = Path('.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg')
PINNED = {
    BASE / 'born_fallback.f': 'd06ec729509c616790d2960fd9d06a54f3aa0e2fc1408011641fd1896e7ba621',
    BASE / 'loop_matrix.f': '8c09cb06d2d087403ce0603e99dede30f22c4d7f450b5158043a7932047733b0',
    OBSERVER / 'sources/loop_matrix.f': '5545b6163693344446545e6bde30b16c7aa95bbae80a19ecdc9b696e6af9fb7e',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def once(text, old, new):
    require(text.count(old) == 1, f'expected one source anchor: {old!r}')
    return text.replace(old, new, 1)


def transform_wrapper(text):
    """Change only the wrapper prefix; retain the entire Born aggregate/kernel."""
    tail_marker = '      SUBROUTINE AUDIT_QP_MATRIX('
    tail = text[text.index(tail_marker):]
    text = once(text, '      SUBROUTINE ML5_0_BORN_WITH_RESCUE(P,HEL,FLAV_IDX,ANS,\n'
                     '     $ RESCUED,OK)\n',
                     '      SUBROUTINE ML5_0_BORN_WITH_C_RESCUE(P,HEL,FLAV_IDX,ANS,\n'
                     '     $ RESCUED,OK,C_READY,C,PC)\n')
    text = once(text, '      REAL*16 PQ(0:3,5),ANSQ(0:1)\n',
                     '      REAL*16 C(0:3,5),ANSQ(0:1)\n'
                     '      REAL*8 PC(0:3,5)\n'
                     '      REAL*16 MP_PS(0:3,5),MP_P(0:3,5)\n'
                     '      COMMON/ML5_0_MP_PSPOINT/MP_PS,MP_P\n')
    text = once(text, '      LOGICAL RESCUED,OK,RAISED(SIZE(IEEE_ALL))\n',
                     '      LOGICAL RESCUED,OK,C_READY,RAISED(SIZE(IEEE_ALL))\n')
    text = once(text, '      CALL IEEE_GET_STATUS(ENTRY_STATUS)\n',
                     '      C_READY=.FALSE.\n'
                     '      CALL IEEE_GET_STATUS(ENTRY_STATUS)\n')
    text = once(text, '      PQ=REAL(P,KIND=16)\n'
                     '      USERHEL=HEL\n'
                     '      CALL AUDIT_QP_SMATRIX_SPLITORDERS(PQ,FLAV_IDX,ANSQ)\n'
                     '      USERHEL=-1\n'
                     '      OK=ALL(IEEE_IS_FINITE(ANSQ))\n'
                     '      IF(OK) ANS=REAL(ANSQ,KIND=8)\n',
                     'C     Prepare exactly the coordinates returned by the actual host setter.\n'
                     'C     It may retain its input after a repair failure; this is not a\n'
                     'C     certificate that shell/conservation repair succeeded.\n'
                     '      CALL ML5_0_SET_MP_PS(P)\n'
                     '      C=MP_PS\n'
                     '      OK=ALL(IEEE_IS_FINITE(C))\n'
                     '      IF(OK) THEN\n'
                     'C       Check the DP trial copy within the same IEEE rescue interval.\n'
                     '        PC=REAL(C,KIND=8)\n'
                     '        OK=ALL(IEEE_IS_FINITE(PC))\n'
                     '      ENDIF\n'
                     '      CALL IEEE_GET_FLAG(IEEE_INVALID,INVALID)\n'
                     '      CALL IEEE_GET_FLAG(IEEE_DIVIDE_BY_ZERO,DIVZERO)\n'
                     '      CALL IEEE_GET_FLAG(IEEE_OVERFLOW,OVERFLOW)\n'
                     '      OK=OK.AND..NOT.(INVALID.OR.DIVZERO.OR.OVERFLOW)\n'
                     '      IF(OK) THEN\n'
                     '        USERHEL=HEL\n'
                     '        CALL AUDIT_QP_SMATRIX_SPLITORDERS(C,FLAV_IDX,ANSQ)\n'
                     '        USERHEL=-1\n'
                     '        OK=ALL(IEEE_IS_FINITE(ANSQ))\n'
                     '        IF(OK) ANS=REAL(ANSQ,KIND=8)\n'
                     '      ENDIF\n')
    text = once(text, '      IF(.NOT.OK) THEN\n        FAIL_COUNT=FAIL_COUNT+1\n',
                     'C     A current-call handoff, not a checked amplitude certificate.\n'
                     '      C_READY=OK\n'
                     '      IF(.NOT.OK) THEN\n        FAIL_COUNT=FAIL_COUNT+1\n')
    require(text[text.index(tail_marker):] == tail, 'Born aggregate/model/kernel changed')
    return text


def transform_host(text):
    """Apply invariant anchors to either frozen bare or observed revision-2 host."""
    setter_marker = '      SUBROUTINE ML5_0_SET_MP_PS(P)\n'
    setter_tail = text[text.index(setter_marker):]
    text = once(text, '      LOGICAL BORN_RESCUED,BORN_OK,BORN_QP_CHECKED\n',
                     '      LOGICAL BORN_RESCUED,BORN_OK,BORN_QP_CHECKED\n'
                     '      LOGICAL BORN_C_READY\n')
    text = once(text, '      REAL*8 BORNBUFF(0:NSQSO_BORN),TMPR\n',
                     '      REAL*8 BORNBUFF(0:NSQSO_BORN),TMPR\n'
                     '      REAL*16 BORN_C(0:3,NEXTERNAL)\n'
                     '      REAL*8 BORN_C_DP(0:3,NEXTERNAL)\n')
    text = once(text, '      CALL ML5_0_BORN_WITH_RESCUE(P_USER,USERHEL,IC,BORNBUFF,\n'
                     '     $ BORN_RESCUED,BORN_OK)\n',
                     '      CALL ML5_0_BORN_WITH_C_RESCUE(P_USER,USERHEL,IC,\n'
                     '     $ BORNBUFF,BORN_RESCUED,BORN_OK,BORN_C_READY,\n'
                     '     $ BORN_C,BORN_C_DP)\n')
    text = once(text, '      MP_DONE=.FALSE.\n      MP_DONE_ONCE=.FALSE.\n'
                     '      MP_PS_SET=.FALSE.\n',
                     '      MP_DONE=.FALSE.\n      MP_DONE_ONCE=.FALSE.\n'
                     '      MP_PS_SET=.FALSE.\n'
                     '      IF(BORN_C_READY) THEN\n'
                     'C       Adopt the current fallback output after the ordinary resets.\n'
                     'C       Preparing coordinates does not certify numerator computation.\n'
                     '        MP_PS=BORN_C\n'
                     '        MP_P=BORN_C\n'
                     '        MP_PS_SET=.TRUE.\n'
                     '      ENDIF\n')
    text = once(text, '      DO I=1,NEXTERNAL\n        DO J=0,3\n'
                     '          PS(J,I)=P_USER(J,I)\n        ENDDO\n      ENDDO\n',
                     '      IF(BORN_C_READY) THEN\n'
                     'C       Reuse the checked rounded C for DP trials, never for MP input.\n'
                     '        PS=BORN_C_DP\n'
                     '      ELSE\n'
                     '        DO I=1,NEXTERNAL\n          DO J=0,3\n'
                     '            PS(J,I)=P_USER(J,I)\n          ENDDO\n        ENDDO\n'
                     '      ENDIF\n')
    text = once(text, '      IF (IMPROVEPSPOINT.GE.0) THEN\n',
                     '      IF (IMPROVEPSPOINT.GE.0.AND..NOT.BORN_C_READY) THEN\n')
    text = once(text, '            BORN_QP_CHECKED=ALL(IEEE_IS_FINITE(ACC)).AND.\n',
                     '            BORN_QP_CHECKED=BORN_C_READY.AND.\n'
                     '     $        ALL(IEEE_IS_FINITE(ACC)).AND.\n')
    require(text[text.index(setter_marker):] == setter_tail, 'setter/QOBS/following routines changed')
    return text


def derive_sources():
    """Return transformed source strings; never write files or run native code."""
    require(Path.cwd().resolve() == ROOT, 'run from the MadGraph repository root')
    sources = {}
    for path, expected in PINNED.items():
        raw = path.read_bytes()
        require(sha(raw) == expected, f'frozen source mismatch: {path}')
        sources[path] = raw.decode('ascii')
    # Exercise the same invariant-anchor transformation on the bare B control.
    transform_host(sources[BASE / 'loop_matrix.f'])
    return sources, {
        'born_fallback.f': transform_wrapper(sources[BASE / 'born_fallback.f']),
        'loop_matrix.f': transform_host(sources[OBSERVER / 'sources/loop_matrix.f']),
    }


def artifacts():
    sources, derived = derive_sources()
    patch = ''
    for name, origin in (('born_fallback.f', BASE / 'born_fallback.f'),
                         ('loop_matrix.f', OBSERVER / 'sources/loop_matrix.f')):
        patch += ''.join(difflib.unified_diff(sources[origin].splitlines(True),
                    derived[name].splitlines(True), fromfile=str(origin), tofile='c-coordinate/' + name))
    dependencies = list(PINNED) + [BASE / 'evidence-index.json', OBSERVER / 'evidence-index.json',
        PROCESS / 'SubProcesses/P0_uux_zgg/mp_compute_loop_coefs.f',
        PROCESS / 'SubProcesses/P0_uux_zgg/improve_ps.f', PROCESS / 'Source/MODEL/couplings.f']
    bindings = {
        'schema': 1, 'source_only': True, 'compiler_executed': False, 'native_executed': False,
        'source_inputs': {str(path): {'bytes': len(path.read_bytes()), 'sha256': sha(path.read_bytes())}
                          for path in dependencies},
        'derived_sources': {name: {'bytes': len(text.encode('ascii')), 'sha256': sha(text.encode('ascii'))}
                            for name, text in derived.items()},
        'generator': {'bytes': len(Path(__file__).read_bytes()), 'sha256': sha(Path(__file__).read_bytes())},
        'patch': {'bytes': len(patch.encode('ascii')), 'sha256': sha(patch.encode('ascii'))},
        'coordinate_policy': 'Rare failed-DP Born uses actual SET_MP_PS output C; checked DP copy for trials; one setter call and no duplicate DP repair.',
        'parameter_policy': 'Unchanged promoted-DP parameters in QP Born; actual MP loop parameter differences remain unresolved.',
        'qualification': 'Source preparation only; no build, numerical acceptance, IEEE/runtime or performance claim.',
    }
    return patch, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    try:
        patch, bindings = artifacts()
        products = {'born-c-handoff.patch': patch,
                    'source-bindings.json': json.dumps(bindings, indent=2, sort_keys=True) + '\n'}
        if args.prepare:
            require(all(not (HERE / name).exists() for name in products), 'refusing to overwrite prepared artifacts')
            for name, text in products.items():
                with (HERE / name).open('x', encoding='ascii') as stream:
                    stream.write(text)
        else:
            for name, text in products.items():
                require((HERE / name).read_bytes() == text.encode('ascii'), f'prepared artifact mismatch: {name}')
    except (OSError, ValueError) as error:
        parser.exit(1, f'Born-C source derivation failed: {error}\n')
    print(json.dumps({'action': 'prepared' if args.prepare else 'verified', 'native_executed': False,
                      'patch_sha256': bindings['patch']['sha256'],
                      'derived_sources': bindings['derived_sources']}, sort_keys=True))


if __name__ == '__main__':
    main()
