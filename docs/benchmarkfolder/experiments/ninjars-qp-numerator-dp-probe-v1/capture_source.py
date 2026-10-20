"""Generate integer-only capture instrumentation for the isolated QCD3 clone."""


def logger():
    text = '''
      SUBROUTINE NRS_PROBE_BEGIN(MODE,ID,SO,N,RANK,STAGE,SEQ)
      IMPLICIT NONE
      INTEGER MODE,ID,SO,N,RANK,STAGE,SEQ,IOS
      INTEGER PROBE_POINT,PROBE_UNIT,PROBE_SEQUENCE
      COMMON/NRS_PROBE_POINT/PROBE_POINT
      COMMON/NRS_PROBE_IO/PROBE_UNIT,PROBE_SEQUENCE
      SEQ=0
      IF(PROBE_POINT.NE.44.AND.PROBE_POINT.NE.80) RETURN
      IF(PROBE_UNIT.EQ.0) THEN
        OPEN(NEWUNIT=PROBE_UNIT,FILE='tensor-capture.hex',
     $   STATUS='NEW',ACTION='WRITE',IOSTAT=IOS)
        IF(IOS.NE.0) STOP 'PROBE OPEN FAILED'
        WRITE(PROBE_UNIT,'(A,3(1X,Z16.16))') 'LAYOUT',
     $   TRANSFER(1D0,0_8),TRANSFER(1.0e0_16,[0_8,0_8])
        WRITE(PROBE_UNIT,'(A,2(1X,Z16.16))') 'COMPLEX64',
     $   TRANSFER(CMPLX(1D0,2D0,KIND=8),[0_8,0_8])
        WRITE(PROBE_UNIT,'(A,4(1X,Z16.16))') 'COMPLEX128',
     $   TRANSFER(CMPLX(1.0e0_16,2.0e0_16,KIND=16),
     $   [0_8,0_8,0_8,0_8])
      ENDIF
      PROBE_SEQUENCE=PROBE_SEQUENCE+1
      IF(PROBE_SEQUENCE.GT.2500) STOP 'PROBE RECORD LIMIT'
      SEQ=PROBE_SEQUENCE
      WRITE(PROBE_UNIT,'(A,8I12)') 'BEGIN',SEQ,PROBE_POINT,
     $ MODE,ID,SO,N,RANK,STAGE
      END

      SUBROUTINE NRS_PROBE_CLOSE()
      IMPLICIT NONE
      INTEGER PROBE_UNIT,PROBE_SEQUENCE
      COMMON/NRS_PROBE_IO/PROBE_UNIT,PROBE_SEQUENCE
      IF(PROBE_UNIT.NE.0) CLOSE(PROBE_UNIT)
      PROBE_UNIT=0
      END

      BLOCK DATA NRS_PROBE_INIT
      IMPLICIT NONE
      INTEGER PROBE_POINT,PROBE_UNIT,PROBE_SEQUENCE
      COMMON/NRS_PROBE_POINT/PROBE_POINT
      COMMON/NRS_PROBE_IO/PROBE_UNIT,PROBE_SEQUENCE
      DATA PROBE_POINT,PROBE_UNIT,PROBE_SEQUENCE/0,0,0/
      END
'''
    for bits, real, complex_type, nword in ((64, 'REAL*8', 'COMPLEX*16', 1),
                                           (128, 'REAL*16', 'COMPLEX*32', 2)):
        mold = '0_8' if nword == 1 else '[0_8,0_8]'
        complex_mold = '[' + ','.join(['0_8'] * (2*nword)) + ']'
        text += f'''
      SUBROUTINE NRS_PROBE_{bits}(SEQ,N,NC,C,S,P,M,U,RES,R1,STATUS)
      IMPLICIT NONE
      INTEGER SEQ,N,NC,STATUS,I,J,PROBE_UNIT,PROBE_SEQUENCE
      COMMON/NRS_PROBE_IO/PROBE_UNIT,PROBE_SEQUENCE
      {complex_type} C(0:NC-1),M(N),RES(0:2),R1
      {real} S(N,N),P(0:3,N),U
      IF(SEQ.EQ.0) RETURN
      IF(PROBE_UNIT.EQ.0) STOP 'PROBE PAYLOAD WITHOUT BEGIN'
      WRITE(PROBE_UNIT,'(A,4I12)') 'PAYLOAD',SEQ,{bits},STATUS,NC
      DO I=0,NC-1
        WRITE(PROBE_UNIT,'(A,I6,{2*nword}(1X,Z16.16))') 'C',I,
     $   TRANSFER(C(I),{complex_mold})
      ENDDO
      DO J=1,N
        DO I=1,N
          WRITE(PROBE_UNIT,'(A,2I6,{nword}(1X,Z16.16))') 'S',I,J,
     $     TRANSFER(S(I,J),{mold})
        ENDDO
        DO I=0,3
          WRITE(PROBE_UNIT,'(A,2I6,{nword}(1X,Z16.16))') 'P',I,J,
     $     TRANSFER(P(I,J),{mold})
        ENDDO
        WRITE(PROBE_UNIT,'(A,I6,{2*nword}(1X,Z16.16))') 'M',J,
     $   TRANSFER(M(J),{complex_mold})
      ENDDO
      WRITE(PROBE_UNIT,'(A,{nword}(1X,Z16.16))') 'U',TRANSFER(U,{mold})
      DO I=0,2
        WRITE(PROBE_UNIT,'(A,I6,{2*nword}(1X,Z16.16))') 'R',I,
     $   TRANSFER(RES(I),{complex_mold})
      ENDDO
      WRITE(PROBE_UNIT,'(A,{2*nword}(1X,Z16.16))') 'R1',
     $ TRANSFER(R1,{complex_mold})
      WRITE(PROBE_UNIT,'(A,I12)') 'END',SEQ
      END
'''
    assert max(map(len, text.splitlines())) <= 132
    return text


def interface(source):
    chunks = []
    for name, bits in [('ML5_0_NINJARS_LOOP', 64), ('ML5_0_MP_NINJARS_LOOP', 128)]:
        start = source.index('      SUBROUTINE ' + name + '(')
        end = source.index('\n      END\n', start) + len('\n      END\n')
        body = source[start:end]
        declaration = '      INTEGER I,J,K,CURR_MAXCOEF,NRS_STATUS'
        assert body.count(declaration) == 1
        body = body.replace(declaration, declaration + '\n      INTEGER PROBE_SEQ,PROBE_STAGE')
        stage = '      PROBE_STAGE=3\n' if bits == 128 else (
            '      PROBE_STAGE=1\n      IF(NRS_RETRY_MODE) PROBE_STAGE=2\n')
        begin = stage + ('      CALL NRS_PROBE_BEGIN(CTMODE,ID,SQSOINDEX,\n'
                         '     $ NLOOPLINE,RANK,PROBE_STAGE,PROBE_SEQ)\n')
        assert body.count('      STABLE=.FALSE.') == 1
        body = body.replace('      STABLE=.FALSE.', begin + '      STABLE=.FALSE.')
        end_call = f'''      CALL NRS_PROBE_{bits}(PROBE_SEQ,NLOOPLINE,CURR_MAXCOEF,
     $ TENSORCOEFS,REAL_S_MAT,P_NRS,MASSES,NRS_SCALE,NRS_RES,R1,
     $ NRS_STATUS)
'''
        assert body.count('      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS') == 1
        body = body.replace('      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS',
                            end_call + '      STABLE=NRS_STATUS.EQ.NINJARS_SUCCESS')
        chunks.append((start, end, body))
    for start, end, body in reversed(chunks):
        source = source[:start] + body + source[end:]
    return source + logger()


def driver(source):
    needle = '      CHARACTER*1024 ARG'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n      INTEGER PROBE_POINT\n      COMMON/NRS_PROBE_POINT/PROBE_POINT')
    source = source.replace("      CALL SETPARA('param_card.dat')", "      PROBE_POINT=0\n      CALL SETPARA('param_card.dat')")
    needle = 'C     Save numerical outputs outside the timing interval.\n      DO K=1,NPOINTS'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n         PROBE_POINT=K')
    needle = 'C     Suppress any repeated Fortran diagnostics inside the timed region.'
    assert source.count(needle) == 1
    source = source.replace(needle, '      PROBE_POINT=0\n' + needle)
    assert source.endswith('      END\n')
    return source[:-len('      END\n')] + '      CALL NRS_PROBE_CLOSE()\n      END\n'
