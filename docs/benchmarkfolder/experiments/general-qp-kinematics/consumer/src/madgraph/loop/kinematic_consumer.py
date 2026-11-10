"""Private QP8/9 identity consumer. No Born or coordinate ownership policy."""
import re
from madgraph.loop import kinematic_identity
from madgraph.models import check_param_card


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Nonunique consumer source anchor: ' + old)
    return text.replace(old, new, 1)


def adapter_template(template, filename):
    """Derive a third entry from the same mass-paired scientific template."""
    if filename == 'TIRUST_qp_tensor_interface.inc':
        template = replace_once(template, '(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)',
            '(GROUP_ID,KIN_POLICY,CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS,STABLE)')
        template = replace_once(template, 'INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS',
            'INTEGER GROUP_ID,KIN_POLICY,CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS')
    elif filename in ('NINJARS_qp_interface.inc', 'TIRUST_qp_loop_interface.inc'):
        template = replace_once(template, '(NLOOPLINE,PL,M2L,RANK,RES,STABLE)',
            '(GROUP_ID,CTMODE,KIN_POLICY,NLOOPLINE,PL,M2L,RANK,RES,STABLE)')
        template = replace_once(template, '      %(real_dp_format)s LSCALE\n', '')
        template = replace_once(template, '      COMMON/%(proc_prefix)sCT/LSCALE,CTMODE\n', '')
        if filename == 'NINJARS_qp_interface.inc':
            template = replace_once(template, '      INTEGER CTMODE\n',
                '      INTEGER GROUP_ID,CTMODE,KIN_POLICY\n')
        else:
            template = replace_once(template, '      INTEGER NLOOPLINE,RANK,CTMODE,I,J\n',
                '      INTEGER GROUP_ID,KIN_POLICY,NLOOPLINE,RANK,CTMODE,I,J\n')
            template = replace_once(template,
                'DIAGNOSTIC%(qp_mass_suffix)s(CTMODE,NLOOPLINE,RANK,PL,M2L,COEFS,STABLE)',
                'DIAGNOSTIC%(qp_mass_suffix)s(GROUP_ID,KIN_POLICY,CTMODE,NLOOPLINE,RANK,PL,M2L,COEFS,STABLE)')
    else:
        raise ValueError('Unsupported QP adapter template')
    if filename != 'TIRUST_qp_loop_interface.inc':
        template = replace_once(template,
            'MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES,S_MAT)',
            'MP_BUILD_KINEMATIC_MATRIX_KIN(GROUP_ID,CTMODE,KIN_POLICY,NLOOPLINE,P_S_MAT,MASSES,S_MAT)')
    return template


def connect_paired_helper(source):
    """Only the paired helper's already-selected QP8/9 branches change."""
    for backend in ('NINJARS', 'TIRUST'):
        old = 'MP_%s_LOOP_QPMASS(NLOOPLINE,' % backend
        if source.count(old) != 2:
            raise ValueError('Expected ordinary and induced helper branches')
        source = source.replace(old,
            'MP_%s_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,NLOOPLINE,' % backend)
    return source


def external_mass_rows(matrix_element):
    """Use existing physical real model masses; never CMASS or card reparsing."""
    process = matrix_element.get('processes')[0]
    model = process.get('model')
    parameters = {p.name: p.type for group in model.get('parameters').values() for p in group}
    result = []
    for leg in process.get('legs'):
        name = model.get_particle(leg.get('id')).get('mass')
        zero = name.lower() == 'zero'
        safe = bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name))
        supported = zero or (safe and parameters.get(name) == 'real' and not name.upper().startswith('CMASS_'))
        result.append({'leg': leg.get('number'), 'mass_symbol': name,
                       'mp_symbol': None if zero else check_param_card.ParamCard.mp_prefix + name,
                       'supported_real': supported, 'zero': zero})
    return result


def render(matrix_element, group_loops, prefix):
    table = kinematic_identity.build(matrix_element, group_loops)
    mass_rows = external_mass_rows(matrix_element)
    source = kinematic_identity.render(table, prefix)
    source += '''
      SUBROUTINE %(p)sMP_EXTERNAL_SHELL_MASS2(LEG,MASS2,VALID)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER LEG
      REAL*16 MASS2,MASS
      LOGICAL VALID
      INCLUDE 'mp_coupl.inc'
      MASS2=0.0E0_16
      VALID=.FALSE.
''' % {'p': prefix}
    for row in mass_rows:
        if not row['supported_real']:
            continue
        source += '      IF(LEG.EQ.%d) THEN\n' % row['leg']
        if row['zero']:
            source += '        VALID=.TRUE.\n        RETURN\n'
        else:
            source += ('        MASS=%s\n' % row['mp_symbol'] +
                '        IF(.NOT.IEEE_IS_FINITE(MASS)) RETURN\n'
                '        IF(MASS.LT.0.0E0_16) RETURN\n'
                '        IF(MASS.GE.SQRT(HUGE(MASS))) RETURN\n'
                '        MASS2=MASS**2\n'
                '        VALID=IEEE_IS_FINITE(MASS2)\n'
                '        RETURN\n')
        source += '      ENDIF\n'
    source += '''      END

      SUBROUTINE %(p)sMP_BUILD_KINEMATIC_MATRIX_KIN(G,CT,POLICY,
     $ N,P_LOOP,M2L,S_MAT)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER G,CT,POLICY,N,I,J,K,CLASS,LEG,SGN,CLOSURE
      REAL*16 P_LOOP(N,0:3),REF_NORMALIZATION,TARGET,TOL
      REAL*16 CLOSURE_SIZE,PAIR_SCALE,RESIDUAL
      COMPLEX*32 M2L(N),S_MAT(N,N),DIFFSQ
      LOGICAL KNOWN,DECLARED_ONSHELL,TARGET_VALID,TOL_OK
      LOGICAL P_FINITE
      INCLUDE 'MadLoopParams.inc'
C     Policy zero is explicit exact legacy compatibility, including its
C     internal-mass threshold guesses. No metadata lookup occurs there.
      IF(POLICY.EQ.0) THEN
        CALL %(p)sMP_BUILD_KINEMATIC_MATRIX(N,P_LOOP,M2L,S_MAT)
        RETURN
      ENDIF
C     No coordinate repair, projection, parameter update or flag masking.
C     First shift is the literal corner sum in these adapters, not zero.
      TOL=REAL(OSThres,KIND=16)
      TOL_OK=IEEE_IS_FINITE(TOL)
      IF(TOL_OK) TOL_OK=TOL.GT.0.0E0_16.AND.TOL.LT.1.0E0_16
      P_FINITE=ALL(IEEE_IS_FINITE(P_LOOP))
      CLOSURE_SIZE=0.0E0_16
      IF(N.GT.0.AND.TOL_OK.AND.POLICY.EQ.1.AND.P_FINITE) THEN
        CLOSURE_SIZE=MAXVAL(ABS(P_LOOP(1,0:3)))
      ENDIF
      DO I=1,N
        DO J=1,N
          IF(I.EQ.J) THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
C           Preserve the original computed-invariant arithmetic order.
            DIFFSQ=(CMPLX(P_LOOP(I,0),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,0),0.0E0_16,KIND=16))**2
            DO K=1,3
              DIFFSQ=DIFFSQ-(CMPLX(P_LOOP(I,K),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,K),0.0E0_16,KIND=16))**2
            ENDDO
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
            IF(POLICY.NE.1.OR..NOT.TOL_OK) CYCLE
            IF(.NOT.P_FINITE) CYCLE
            CALL %(p)sKINEMATIC_IDENTITY(G,CT,N,I,J,CLASS,LEG,
     $           SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
            IF(.NOT.KNOWN.OR..NOT.DECLARED_ONSHELL) CYCLE
            IF(CLASS.NE.1.AND.CLASS.NE.2) CYCLE
C           Normalize to this external pair, not a hard common shift.
            PAIR_SCALE=0.0E0_16
            DO K=0,3
              PAIR_SCALE=MAX(PAIR_SCALE,
     $                      ABS(P_LOOP(I,K)-P_LOOP(J,K)))
            ENDDO
            IF(.NOT.IEEE_IS_FINITE(PAIR_SCALE)) CYCLE
C           Complements additionally need closure at the leg's scale.
C           Reject ratios above one before division, avoiding overflow.
            IF(CLOSURE.NE.0) THEN
              IF(CLOSURE_SIZE.GT.PAIR_SCALE) CYCLE
              IF(PAIR_SCALE.EQ.0.0E0_16) THEN
                IF(CLOSURE_SIZE.NE.0.0E0_16) CYCLE
              ELSE
                IF(CLOSURE_SIZE/PAIR_SCALE.GT.TOL) CYCLE
              ENDIF
            ENDIF
            CALL %(p)sMP_EXTERNAL_SHELL_MASS2(LEG,TARGET,TARGET_VALID)
            IF(.NOT.TARGET_VALID) CYCLE
            IF(.NOT.IEEE_IS_FINITE(REAL(DIFFSQ,KIND=16))) CYCLE
            IF(.NOT.IEEE_IS_FINITE(AIMAG(DIFFSQ))) CYCLE
            IF(TARGET.NE.0.0E0_16) THEN
              RESIDUAL=ABS((DIFFSQ-TARGET)/TARGET)
            ELSE
C             A soft external leg must pass at its own four-vector
C             scale; unrelated hard shifts cannot dilute its defect.
C             Extreme scales conservatively retain computed invariants.
              IF(PAIR_SCALE.GE.SQRT(HUGE(PAIR_SCALE))) CYCLE
              IF(PAIR_SCALE.LT.SQRT(TINY(PAIR_SCALE))) CYCLE
              REF_NORMALIZATION=PAIR_SCALE**2
              IF(.NOT.IEEE_IS_FINITE(REF_NORMALIZATION)) CYCLE
              IF(REF_NORMALIZATION.LE.0.0E0_16) CYCLE
              RESIDUAL=ABS(DIFFSQ/REF_NORMALIZATION)
            ENDIF
            IF(IEEE_IS_FINITE(RESIDUAL)) THEN
              IF(RESIDUAL.LT.TOL) THEN
                S_MAT(I,J)=CMPLX(TARGET,0.0E0_16,KIND=16)
     $                     -M2L(I)-M2L(J)
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDDO
      END
''' % {'p': prefix}
    return source
