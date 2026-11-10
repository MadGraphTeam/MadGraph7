"""Patch private generated copies only, preserving every numerical operation."""
import re


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected one qualified source anchor: ' + old[:80])
    return text.replace(old, new, 1)


def instrument(tirust_text, tir_text, params):
    prefix = params['prefix']
    if not re.fullmatch('[A-Z][A-Z0-9_]*_', prefix):
        raise ValueError('Unexpected generated prefix')
    marker = '      SUBROUTINE ' + prefix + 'MP_TIRUST_TENSOR_DIAGNOSTIC'
    at = tirust_text.find(marker)
    if at < 0:
        raise ValueError('Qualified DP/QP split missing')
    dp, mp = tirust_text[:at], tirust_text[at:]
    dp = once(dp, '      STABLE=.FALSE.\n      TIRCOEFS=0\n',
              '      STABLE=.FALSE.\n      TIRCOEFS=0\n      CALL ML_TIRUST_CAPTURE_STAGE(10,0)\n')
    pattern = r'      IF\(RANK\.LT\.0\.OR\.RANK\.GT\.(\d+)\) RETURN\n'
    matches = list(re.finditer(pattern, dp))
    if len(matches) != 1 or int(matches[0].group(1)) != params['rank']:
        raise ValueError('Unexpected maximum-rank guard')
    dp = re.sub(pattern, lambda m: '      IF(RANK.LT.0.OR.RANK.GT.' + m.group(1) + ') THEN\n'
                '        CALL ML_TIRUST_CAPTURE_STAGE(11,RANK)\n        RETURN\n      ENDIF\n', dp)
    old = '      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))\n     $  RETURN\n'
    dp = once(dp, old, '      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE)) THEN\n'
              '        CALL ML_TIRUST_CAPTURE_STAGE(12,0)\n        RETURN\n      ENDIF\n')
    old = '      IF(STATUS.NE.TIRUST_SUCCESS) RETURN\n'
    if dp.count(old) != 3:
        raise ValueError('Unexpected status-guard population')
    for stage in (13, 14):
        dp = dp.replace(old, '      IF(STATUS.NE.TIRUST_SUCCESS) THEN\n'
            f'        CALL ML_TIRUST_CAPTURE_STATUS({stage},STATUS)\n'
            '        RETURN\n      ENDIF\n', 1)
    dp = once(dp, old, '      CALL ML_TIRUST_CAPTURE_STATUS(21,STATUS)\n' + old)
    old = '          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) RETURN\n'
    dp = once(dp, old, '          IF(ABS(P_TMP(I,0))/MAX(REF_P*1D-6,ABSP(I)).GT.1D-6) THEN\n'
              '            CALL ML_TIRUST_CAPTURE_STAGE(15,I)\n            RETURN\n          ENDIF\n')
    old = '      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS\n'
    dp = once(dp, old, '      CALL ML_TIRUST_CAPTURE_STAGE(20,0)\n' + old)
    old = '          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_TIR))\n     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) RETURN\n'
    dp = once(dp, old, '          IF(.NOT.IEEE_IS_FINITE(REAL(RAW(K,J),KIND=KI_TIR))\n'
              '     $     .OR..NOT.IEEE_IS_FINITE(AIMAG(RAW(K,J)))) THEN\n'
              '            CALL ML_TIRUST_CAPTURE_STAGE(22,10*I+J)\n            RETURN\n          ENDIF\n')
    dp = once(dp, '      STABLE=.TRUE.\n', '      STABLE=.TRUE.\n      CALL ML_TIRUST_CAPTURE_STAGE(30,0)\n')

    at = tir_text.find('      SUBROUTINE ' + prefix + 'SWITCH_ORDER')
    if at < 0:
        raise ValueError('Qualified TIR dispatch split missing')
    dispatch, tail = tir_text[:at], tir_text[at:]
    old = '        CALL ' + prefix + 'TIRUSTLOOP(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS\n     $   ,STABLE)\n'
    dispatch = once(dispatch, old,
        '        CALL ML_TIRUST_CAPTURE_GROUP(I_SQSO,I_LOOPGROUP,CTMODE,\n'
        '     $   NLOOPLINE,RANK,PL,M2L)\n' + old +
        '        CALL ML_TIRUST_CAPTURE_STAGE(40,MERGE(1,0,STABLE))\n')
    old = '          FPE_IN_DP_REDUCTION=.TRUE.\n          RES=0\n          RETURN\n'
    dispatch = once(dispatch, old, '          FPE_IN_DP_REDUCTION=.TRUE.\n          RES=0\n'
                    '          CALL ML_TIRUST_CAPTURE_END(STABLE)\n          RETURN\n')
    old = '          FPE_IN_DP_REDUCTION=.TRUE.\n          RES=0\n        ENDIF\n      ENDIF\n'
    dispatch = once(dispatch, old, '          FPE_IN_DP_REDUCTION=.TRUE.\n          RES=0\n        ENDIF\n'
                    '        CALL ML_TIRUST_CAPTURE_END(STABLE)\n      ENDIF\n')
    return {'TIRUST_interface.f': dp + mp, 'TIR_interface.f': dispatch + tail}


def render_driver(template, params):
    text = template
    for token, name in (('@PREFIX@', 'prefix'), ('@NEXTERNAL@', 'nexternal'), ('@NBORN@', 'nborn')):
        text = text.replace(token, str(params[name]))
    if '@' in text:
        raise ValueError('Unexpanded private driver token')
    case = params['case_id']; target = params['target_point']
    if (case, target, tuple(params['groups']), params['nprop'], params['rank']) not in (
        (1, 25, (59, 61), 5, 4), (2, 2, (408, 422), 6, 5)):
        raise ValueError('Unqualified capture registry')
    text = once(text, "      OPEN(73,FILE=TRIM(POINT_FILE),STATUS='OLD',ACTION='READ',\n",
        f'      CALL ML_TIRUST_CAPTURE_INIT({case})\n' +
        "      OPEN(73,FILE=TRIM(POINT_FILE),STATUS='OLD',ACTION='READ',\n")
    before, marker, after = text.partition('C     Each physical point supplies one three-method DP diagnostic.\n')
    if not marker:
        raise ValueError('Qualified primary-loop anchor missing')
    after = once(after, '      DO K=1,NPOINTS\n', f'      DO K=1,{target}\n')
    text = before + marker + after
    if text.count('3*NPOINTS') != 2:
        raise ValueError('Qualified completion records missing')
    text = text.replace('3*NPOINTS', '3*' + str(target))
    text = once(text, '      CALL STABILITY_FLAG_BEFORE(PHASE,CALL_INDEX,POINT_INDEX,\n',
        '      CALL ML_TIRUST_CAPTURE_POINT(PHASE,CALL_INDEX,POINT_INDEX,\n'
        '     $ METHOD,MODE)\n      CALL STABILITY_FLAG_BEFORE(PHASE,CALL_INDEX,POINT_INDEX,\n')
    return text
