#!/usr/bin/env python3
"""Exact source-only diagnostic transform; never compile or execute a worker."""
import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CASE = ROOT / '.benchmark-work/tirust/campaigns/entry-latch-cache-guard-1/outputs/candidate/qcd_n3_uux_zgg'
PROC = CASE / 'SubProcesses/P0_uux_zgg'
SOURCE = {
    'TIRUST_interface.f': (PROC / 'TIRUST_interface.f', '179e7b982a497176862bb56c9f6ddaa4c227ce593908053fbd4783f2fba8414d'),
    'TIR_interface.f': (PROC / 'TIR_interface.f', '2a1dffb507f359a9e0c0fe5ff4f597aa752306284e027a2d8c6231acb2b15241'),
    'check_sa.f': (PROC / 'check_sa.f', 'f9f1679df57dea235518a9f50570c4a2a1f975b32136114c71cf6222e2a62093'),
    'mtirust.f90': (ROOT / 'vendor/tirust/mtirust.f90', '3cb9cf98e82965e546365b293fdfd622cbae2ec09a7ae3d364de3ccf9b9bd56f'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one exact anchor: {old!r}')
    return text.replace(old, new)


def routine(text, name, transform):
    match = re.search(r'^      SUBROUTINE '+name+r'\(.*?^      END\n', text, re.M | re.S)
    if match is None:
        raise ValueError(name)
    return text[:match.start()] + transform(match.group()) + text[match.end():]


def fixed_call(text, prefix):
    match = re.search(r'^      CALL '+prefix+r'\([^\n]*(?:\n     \$[^\n]*)*\n', text, re.M)
    if match is None:
        raise ValueError(prefix)
    return match.group()


def start():
    return '      CALL QP_PROFILE_START(QP_PT0)\n'


def end(stage):
    return f'      CALL QP_PROFILE_END({stage},QP_PT0)\n'


def shadow(label, value):
    return f"      CALL QP_PROFILE_BLOB('{label}',\n     $ TRANSFER({value},[0_QP_PROFILE_BYTE]))\n"


def adapter(text):
    def diagnostic(body):
        body = once(body, '      USE MTIRUST\n', '      USE QP_PROFILE_STATE\n      USE MTIRUST\n')
        body = once(body, '      IMPLICIT NONE\n', '      IMPLICIT NONE\n      INTEGER*8 QP_PT0\n')
        body = once(body, '      CALL ML5_0_INITIALIZE_TIRUST(STATUS)\n', start()+'      CALL ML5_0_INITIALIZE_TIRUST(STATUS)\n')
        body = once(body, '      PDEN_DUMMY=0\n', end(1)+start()+'      PDEN_DUMMY=0\n')
        call = fixed_call(body, 'TIRUST_TENSOR_EVALUATE')
        trace = '      IF(QP_PROFILE_MODE.EQ.2) THEN\n'
        for label, value in [('post_shifts','SHIFTS'),('post_invariants','INVARIANTS'),('post_masses','MASSES'),('post_scale','SCALE')]:
            trace += shadow(label, value)
        trace += '      ENDIF\n'
        before = end(2)+trace
        # DP has no decimal transport; its complete module call is bucket four.
        is_mp = 'SUBROUTINE ML5_0_MP_' in body
        if not is_mp:
            before += start()
        after = '' if is_mp else end(4)
        after += '      QP_PROFILE_NATIVE_STATUS=STATUS\n'+start()
        body = once(body, call, before+call+after)
        body = once(body, '      STABLE=.TRUE.\n', end(5)+'      STABLE=.TRUE.\n')
        return body
    text = routine(text, 'ML5_0_TIRUST_TENSOR_DIAGNOSTIC', diagnostic)
    text = routine(text, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', diagnostic)

    def wrapper(body):
        mp = 'SUBROUTINE ML5_0_MP_' in body
        body = once(body, '      IMPLICIT NONE\n', '      USE QP_PROFILE_STATE\n'+
                    ('' if mp else '      USE MTIRUST, ONLY: TIRUST_CONTEXT_GENERATION\n')+
                    '      IMPLICIT NONE\n      INTEGER*8 QP_PT0\n')
        if mp:
            body = once(body, '      USE MTIRUST, ONLY: KI_QTIR\n',
                        '      USE MTIRUST, ONLY: KI_QTIR,TIRUST_CONTEXT_GENERATION\n')
        marker = '\n      RES=0\n' if mp else fixed_call(body, 'ML5_0_TIRUST_TENSOR_DIAGNOSTIC')
        pre = "      INCLUDE 'coupl.inc'\n      INCLUDE 'MadLoopParams.inc'\n"
        pre += '      CALL QP_PROFILE_BEGIN(CTMODE,NLOOPLINE,RANK,'+('2' if mp else '1')+',\n'
        pre += '     $ MAX(0,LOOPPOLEMODE),TIRUSTCOMPUTEFINITE,\n'
        pre += '     $ TIRUSTGLOBALCACHE.EQ.-1.OR.\n     $ NLOOPLINE.LE.TIRUSTGLOBALCACHE,\n'
        pre += '     $ INT(TIRUSTCACHEMAXBYTES,8),TIRUST_CONTEXT_GENERATION())\n'
        pre += '      IF(QP_PROFILE_MODE.EQ.2) THEN\n'
        for label,value in [('pre_pl','PL'),('pre_m2l','M2L'),('pre_mur','MU_R'),('pre_osthres','OSTHRES')]:
            pre += shadow(label,value)
        pre += '      ENDIF\n'
        body = once(body, marker, ('\n' if mp else '')+pre+marker)
        finish = '      CALL QP_PROFILE_FINISH(QP_PROFILE_NATIVE_STATUS,STABLE)\n'
        if mp:
            body = once(body, '        RETURN\n', finish+'        RETURN\n')
            body = once(body, '      RESULT=0\n', start()+'      RESULT=0\n')
            body = once(body, '      END\n', end(6)+finish+'      END\n')
        else:
            body = once(body, '      END\n', finish+'      END\n')
        return body
    text = routine(text, 'ML5_0_TIRUSTLOOP', wrapper)
    text = routine(text, 'ML5_0_MP_TIRUST_LOOP', wrapper)
    return text


def contraction(text):
    first, rest = text.split('      END\n', 1)
    first = once(first, '      USE, INTRINSIC :: IEEE_ARITHMETIC',
                 '      USE QP_PROFILE_STATE\n      USE, INTRINSIC :: IEEE_ARITHMETIC')
    first = once(first, '      INTEGER I_SQSO,I_LOOPGROUP,I_LIB\n',
                 '      INTEGER I_SQSO,I_LOOPGROUP,I_LIB\n      INTEGER*8 QP_PT0\n')
    begin = '      DO I=1,3\n        RES(I)=(0.0D0,0.0D0)\n'
    first = once(first, begin, '      IF(MLREDUCTIONLIB(I_LIB).EQ.9) THEN\n'+start()+'      ENDIF\n'+begin)
    # Includes normalization and public finite validation; nothing is removed.
    first += '      IF(MLREDUCTIONLIB(I_LIB).EQ.9) THEN\n'+end(6)+'      ENDIF\n'
    return first+'      END\n'+rest


def transport(text):
    text = once(text, '\nmodule mtirust\n', '\nmodule mtirust\n  use qp_profile_state\n')
    # Instrument native event clear in one place, including failed QP decoding.
    helper = '''  subroutine profile_clear_before()
    integer(c_int64_t) :: counts(6)
    integer :: result
    if (qp_profile_mode == 0) return
    counts = -1_c_int64_t
    result = -1
    if (c_associated(context)) result = qp_cache_statistics(context,counts,6_c_size_t)
    call qp_profile_stats(3,counts,result,tirust_context_generation())
  end subroutine

'''
    text = once(text, '  subroutine tirust_clear_cache(status)\n', helper+'  subroutine tirust_clear_cache(status)\n')
    text = once(text, '    integer :: result\n    call ensure_context(result)\n    if (result == TIRUST_SUCCESS) result = clear_cache(context)\n',
                '    integer :: result\n    integer(c_int64_t) :: qp_tick\n    call profile_clear_before()\n    call qp_profile_start(qp_tick)\n    call ensure_context(result)\n    if (result == TIRUST_SUCCESS) result = clear_cache(context)\n    call qp_profile_end(7,qp_tick)\n')
    begin = text.index('  subroutine tensor_mp(')
    finish = text.index('  end subroutine\n', begin)+len('  end subroutine\n')
    body = text[begin:finish]
    body = once(body, '    integer :: i,j,k,count,ios\n', '    integer :: i,j,k,count,ios,qp_stats_status\n    integer(c_int64_t) :: qp_tick,qp_counts(6)\n')
    body = once(body, '    allocate(p_text', '    call qp_profile_start(qp_tick)\n    allocate(p_text')
    marker = '    status = evaluate_mp('
    idx = body.index(marker)
    pre = '''    call qp_profile_end(3,qp_tick)
    if (qp_profile_mode /= 0) then
      call tirust_qp_cache_statistics(qp_counts,qp_stats_status)
      call qp_profile_stats(1,qp_counts,qp_stats_status,tirust_context_generation())
    endif
    if (qp_profile_mode == 2) then
      call qp_profile_blob('encoded_p',transfer(p_text,[0_qp_profile_byte]))
      call qp_profile_blob('encoded_s',transfer(s_text,[0_qp_profile_byte]))
      call qp_profile_blob('encoded_m',transfer(m_text,[0_qp_profile_byte]))
      call qp_profile_blob('encoded_scale',transfer(scale_text,[0_qp_profile_byte]))
    endif
    call qp_profile_start(qp_tick)
'''
    body = body[:idx]+pre+body[idx:]
    marker = '                         poles,finite,check,cache)\n'
    post = '''    call qp_profile_end(4,qp_tick)
    if (qp_profile_mode /= 0) then
      call tirust_qp_cache_statistics(qp_counts,qp_stats_status)
      call qp_profile_stats(2,qp_counts,qp_stats_status,tirust_context_generation())
    endif
    call qp_profile_start(qp_tick)
'''
    body = once(body, marker, marker+post)
    body = once(body, '        ios = clear_cache(context)\n', '        call profile_clear_before()\n        ios = clear_cache(context)\n')
    body = once(body, '  end subroutine\n', '    call qp_profile_end(5,qp_tick)\n  end subroutine\n')
    return text[:begin]+body+text[finish:]


def driver(text):
    text = once(text, '      PROGRAM BENCHMARK_MADLOOP_WORKER\n      IMPLICIT NONE\n',
                '      PROGRAM BENCHMARK_MADLOOP_WORKER\n      USE QP_PROFILE_STATE\n      IMPLICIT NONE\n')
    text = once(text, '      INTEGER PREVIOUS_SEQUENCE\n', '      INTEGER PREVIOUS_SEQUENCE,QP_POINT,QP_MODE\n      CHARACTER*80 QP_NAME\n')
    # All setup, bootstrap and full-100 learning above this branch remain exact.
    marker = "      IF(TRIM(COMMAND).NE.'RUN') GOTO 800\n"
    code = '''      IF(TRIM(COMMAND).EQ.'PROFILE') THEN
         IF(SEQUENCE.NE.1.OR.PREVIOUS_SEQUENCE.NE.0) GOTO 800
         IF(NLOOPS.LT.0.OR.NLOOPS.GT.2) GOTO 800
         QP_MODE=NLOOPS
         PREVIOUS_SEQUENCE=SEQUENCE
         OPEN(76,FILE='profile-values.dat',STATUS='NEW')
         DO QP_POINT=1,16,15
            CALL QP_PROFILE_RESET(QP_MODE,67108864_8)
            P=POINTS(:,:,QP_POINT)
            TOTAL=P(:,1)+P(:,2)
            MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
            CALL UPDATE_AS_PARAM()
            ANS=0D0
            ACC=-1D0
            CALL ML5_0_SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
            CALL BENCHMARK_TRACE_WARMUP(76,QP_POINT,QP_POINT,
     $           NDIM,NSO,CODE,ANS,ACC)
            WRITE(QP_NAME,'(A,I2.2)') 'point-',QP_POINT
            CALL QP_PROFILE_WRITE(TRIM(QP_NAME))
         ENDDO
         CLOSE(76)
         WRITE(74,'(A,2I12)') 'PROFILED ',SEQUENCE,QP_MODE
         FLUSH(74)
         GOTO 100
      ENDIF
C     Diagnostic-only driver never accepts a timing RUN.
      GOTO 800
'''
    text = once(text, marker, code)
    # Remove unreachable measured RUN body instead of retaining a hidden timer.
    begin = text.index('      IF(SEQUENCE.NE.PREVIOUS_SEQUENCE+1.OR.NLOOPS.LE.0)')
    finish = text.index(' 800  CONTINUE\n', begin)
    return text[:begin]+text[finish:]


def render():
    result = {}
    for name, (path, expected) in SOURCE.items():
        if expected is None or sha(path) != expected:
            raise ValueError(f'Frozen source mismatch/unpinned: {path}')
        result[name] = {'TIRUST_interface.f':adapter, 'TIR_interface.f':contraction,
                        'mtirust.f90':transport, 'check_sa.f':driver}[name](path.read_text())
    return result


def main():
    output = HERE/'source-preview'
    if output.exists():
        raise ValueError('Append-only source preview already exists')
    result = render()
    output.mkdir()
    for name, value in result.items():
        (output/name).write_text(value)
    patch = ''.join(''.join(difflib.unified_diff(SOURCE[name][0].read_text().splitlines(True), value.splitlines(True),
                         fromfile='original/'+name,tofile='private/'+name)) for name,value in result.items())
    (HERE/'instrumentation.diff').write_text(patch)
    print(json.dumps({'source_only':True,'compiled':False,'native_executions':0,
                      'files':{name:sha(output/name) for name in result}},indent=2))


if __name__ == '__main__':
    main()
