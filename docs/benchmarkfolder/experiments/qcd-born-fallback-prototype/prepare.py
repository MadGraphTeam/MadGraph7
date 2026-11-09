#!/usr/bin/env python3
"""Prepare an isolated generated-export Born fallback; never build or execute."""
from pathlib import Path
import difflib
import hashlib
import json
import re

ROOT = Path.cwd().resolve()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
PROOF = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def main():
    if (REPORT / 'preparation.json').exists():
        raise SystemExit('Refusing to overwrite source-bound prototype preparation')
    proof = json.loads((PROOF / 'execution.json').read_text())
    assert proof['complete'] and proof['protected_inputs_unchanged']
    for path, expected in proof['source_and_link_inputs_after'].items():
        assert sha(ROOT / path) == expected, path
    original = (SP / 'born_matrix.f').read_text()
    begin = original.index('      SUBROUTINE ML5_0_SMATRIX_SPLITORDERS(')
    end = original.index('      SUBROUTINE ML5_0_MATRIX(')
    aggregate_original = original[begin:end]
    aggregate = aggregate_original.replace('SUBROUTINE ML5_0_SMATRIX_SPLITORDERS(',
                                           'SUBROUTINE AUDIT_QP_SMATRIX_SPLITORDERS(')
    aggregate = aggregate.replace('REAL*8', 'REAL*16')
    aggregate = re.sub(r'(?<=[0-9.])D([+-]?[0-9]+)', r'Q\1', aggregate)
    aggregate = aggregate.replace('CALL ML5_0_MATRIX(', 'CALL AUDIT_QP_MATRIX(')
    # The existing beam-polarization COMMON remains DOUBLE PRECISION.
    # Integer/logical selection and color COMMON blocks are shared unchanged.
    assert 'DOUBLE PRECISION BEAMPOL(2)' in aggregate
    assert 'CALL ML5_0_GET_FLAVOR(' in aggregate
    # The original Born object owns all initialized COMMON data.  The rare
    # fallback reads that state and deliberately never learns zero helicities.
    aggregate = replace_once(aggregate, '      DATA CHOSEN_SO_CONFIGS/.TRUE./\n', '')
    aggregate = re.sub(r'^      DATA \(NHEL\(I,.*\n', '', aggregate, flags=re.M)
    for line in ('      DATA USERHEL/-1/\n',
                 '      LOGICAL HELRESET\n',
                 '      COMMON/ML5_0_HELRESET/HELRESET\n',
                 '      DATA HELRESET/.TRUE./\n',
                 '      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/\n'):
        aggregate = replace_once(aggregate, line, '')
    cache_declarations = '''      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER ML5_0_GET_FLAVOR_INDEX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
'''
    aggregate = replace_once(aggregate, cache_declarations, '      INTEGER FLAV_IDX\n')
    start = aggregate.index('C     ----------\nC     Check if helreset mode is on')
    stop = aggregate.index('      DO IHEL=1,NCOMB\n', start)
    aggregate = aggregate[:start] + '''C     Sum every requested, selected helicity on each rare rescue.
C     No learning cache: failed DP evaluations cannot train QP selection.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS(:) = 0Q0
        RETURN
      ENDIF
      CALL ML5_0_GET_FLAVOR(FLAV_IDX, FLAVOR)
      JC=1
      ANS=0Q0
''' + aggregate[stop:]
    aggregate = replace_once(aggregate, '''          IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20
     $      .OR.USERHEL.NE.-1) THEN
            IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).NE.
     $       -1.AND.(.NOT.ML5_0_IS_BORN_HEL_SELECTED(IHEL))) THEN
''', '''            IF(POLARIZATIONS(0,0).NE.
     $       -1.AND.(.NOT.ML5_0_IS_BORN_HEL_SELECTED(IHEL))) THEN
''')
    aggregate = replace_once(aggregate, '''            IF (BUFF .NE. 0Q0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX))
     $        THEN
              GOODHEL(IHEL,FLAV_IDX)=.TRUE.
            ENDIF
          ENDIF
''', '')
    aggregate = aggregate.replace('SIGN(1Q0,BEAMPOL(JJ))', 'SIGN(1D0,BEAMPOL(JJ))')
    assert not any(s in aggregate for s in ('NTRY', 'GOODHEL', 'HELRESET'))
    qp = (PROOF / 'qp_born.f').read_text()
    matrix = '''      SUBROUTINE AUDIT_QP_MATRIX(P,NHEL,IC,FLAV_IDX,RES)
      IMPLICIT NONE
      REAL*16 P(0:3,5),RES(1)
      INTEGER NHEL(5),IC(5),FLAV_IDX
      COMPLEX*32 AMP(8),JAMP(2,1)
      INCLUDE 'coupl.inc'
      CALL AUDIT_QP_GET_AMP(P,NHEL,IC,FLAV_IDX,AMP,
     $ REAL(MDL_MZ,KIND=16),CMPLX(GC_4,KIND=16),
     $ CMPLX(GC_5,KIND=16),CMPLX(GC_22,KIND=16),
     $ CMPLX(GC_23,KIND=16))
      CALL AUDIT_QP_GET_JAMP(AMP,JAMP)
      CALL AUDIT_QP_GET_MATRIX(JAMP,RES)
      END

'''
    wrapper = (REPORT / 'born_rescue.f').read_text()
    combined = wrapper + matrix + aggregate + qp
    (REPORT / 'born_fallback.f').write_text(combined)
    (REPORT / 'aggregate-derivation.patch').write_text(''.join(difflib.unified_diff(
        aggregate_original.splitlines(True), aggregate.splitlines(True),
        fromfile='original-born-aggregation.f', tofile='qp-born-aggregation.f')))

    loop_original = (SP / 'loop_matrix.f').read_text()
    loop = replace_once(loop_original,
        '      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE\n',
        '      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,\n'
        '     $ IEEE_VALUE,IEEE_QUIET_NAN\n')
    loop = replace_once(loop,
        '      IMPLICIT NONE\n      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED\n',
        '      IMPLICIT NONE\n      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED\n'
        '      LOGICAL BORN_RESCUED,BORN_OK,BORN_QP_CHECKED\n')
    loop = replace_once(loop,
        '      CALL ML5_0_SMATRIXHEL_SPLITORDERS(P_USER,USERHEL,IC,BORNBUFF(0))',
        '      CALL ML5_0_BORN_WITH_RESCUE(P_USER,USERHEL,IC,BORNBUFF,\n'
        '     $ BORN_RESCUED,BORN_OK)\n'
        '      BORN_QP_CHECKED=.FALSE.')
    # Reject before advancing NTRY or publishing learned filters.  Only the
    # rotation snapshots are established here; later cleanup backups are not.
    failure_guard = '''      IF (.NOT.BORN_OK) THEN
        NROTATIONS_DP=NROTATIONS_DP_BU
        NROTATIONS_QP=NROTATIONS_QP_BU
        MP_DONE=.FALSE.
        MP_DONE_ONCE=.TRUE.
        MP_PS_SET=.FALSE.
        RET_CODE_H=4
        RET_CODE_T=2
        RET_CODE_U=0
        ANS=IEEE_VALUE(0D0,IEEE_QUIET_NAN)
        ACCURACY=1D0
        IF(AUTOMATIC_CACHE_CLEARING) CALL ML5_0_CLEAR_CACHES()
        CALL ML5_0_SET_COLLIER_GLOBAL_CACHE(.FALSE.)
        RETURN
      ENDIF

'''
    marker = '      BORN_QP_CHECKED=.FALSE.\n'
    loop = replace_once(loop, marker, marker + failure_guard)
    accuracy_call = ('        ELSEIF(.NOT.DOING_QP_EVALS)THEN\n'
                     '          CALL ML5_0_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)')
    loop = replace_once(loop, accuracy_call,
        accuracy_call + '\n'
        'C         A recovered Born requires the checked QP loop rescue path.\n'
        '          IF(BORN_RESCUED) FPE_IN_DP_REDUCTION=.TRUE.')
    marker = '''          RET_CODE_H=3
          RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.TRUE.
'''
    loop = replace_once(loop, marker, '''C         A current-call certificate, never inherited from COMMON status.
          IF(BORN_RESCUED) THEN
            BORN_QP_CHECKED=ALL(IEEE_IS_FINITE(ACC)).AND.
     $        ALL(IEEE_IS_FINITE(ANS)).AND.
     $        MAXVAL(ACC).LT.MLSTABTHRES.AND..NOT.FPE_IN_QP_REDUCTION
          ENDIF
''' + marker)
    final_marker = ('      CALL ML5_0_SET_COLLIER_GLOBAL_CACHE(.FALSE.)\n\n'
                    '      END\n\n      SUBROUTINE ML5_0_CLEAR_CACHES()')
    final = '''      CALL ML5_0_SET_COLLIER_GLOBAL_CACHE(.FALSE.)

C     A recovered Born requires a genuine checked QP loop in this call.
C     Keep intentionally unchecked H1 behavior; reject stale checked codes.
      IF(BORN_RESCUED.AND.(RET_CODE_H.EQ.2.OR.RET_CODE_H.EQ.3)
     $   .AND..NOT.BORN_QP_CHECKED) THEN
        RET_CODE_H=4
        RET_CODE_U=0
        ACCURACY=1D0
      ENDIF

      END

      SUBROUTINE ML5_0_CLEAR_CACHES()'''
    loop = replace_once(loop, final_marker, final)
    (REPORT / 'loop_matrix.f').write_text(loop)
    (REPORT / 'host-derivation.patch').write_text(''.join(difflib.unified_diff(
        loop_original.splitlines(True), loop.splitlines(True),
        fromfile='original/loop_matrix.f', tofile='prototype/loop_matrix.f')))
    files = [SP / 'born_matrix.f', SP / 'loop_matrix.f', PROOF / 'evidence-index.json',
             PROOF / 'execution.json', PROOF / 'qp_born.f']
    files += [p for p in REPORT.iterdir() if p.is_file()]
    result = {
        'schema': 1,
        'revision': 2,
        'supersedes': 'revision-1/evidence-index.json',
        'scope': 'Uncompiled private generated-export prototype; no production generator edits.',
        'files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'native_executed': False,
        'production_edited': False,
        'ordinary_path': 'Original DP Born call and arithmetic, plus IEEE status snapshot and finite-result guard.',
        'input_policy': 'Exact lift of P_USER and DP model parameters; no Born phase-space repair.',
        'trap_policy': 'No halting-mode changes; enabled caller traps retain original abort behavior.',
        'unchecked_policy': 'A rescued Born does not certify an unchecked loop; failed QP Born forces H4/U0/accuracy1.',
        'qualification_pending': ['Compilation', 'ordinary bit identity', 'QP Born failure rejection',
                                  'caller IEEE flag preservation', 'enabled trap boundary',
                                  'ordinary guard overhead', 'full checked amplitude and pole normalization'],
    }
    (REPORT / 'preparation.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('Prepared private source prototype only; all native qualification remains pending.')


if __name__ == '__main__':
    main()
