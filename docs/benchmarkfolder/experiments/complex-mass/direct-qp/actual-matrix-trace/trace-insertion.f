
C     Diagnostic-only trace in an isolated generated copy.
      OPEN(901,FILE='cms_actual_mp_matrices.dat',
     $ STATUS='UNKNOWN',POSITION='APPEND')
      WRITE(901,'(A,I4,ES47.34E4)') 'BEGIN ',NLOOPLINE,
     $ REAL(OSTHRES,KIND=16)
      DO I=1,NLOOPLINE
        WRITE(901,'(A,I4,6ES47.34E4)') 'Q ',I,P_LOOP(I,:),
     $ REAL(M2L(I),KIND=16),AIMAG(M2L(I))
        DO J=1,NLOOPLINE
          WRITE(901,'(A,2I4,2ES47.34E4)') 'S ',I,J,
     $ REAL(S_MAT(I,J),KIND=16),AIMAG(S_MAT(I,J))
        ENDDO
      ENDDO
      WRITE(901,'(A)') 'END'
      CLOSE(901)
