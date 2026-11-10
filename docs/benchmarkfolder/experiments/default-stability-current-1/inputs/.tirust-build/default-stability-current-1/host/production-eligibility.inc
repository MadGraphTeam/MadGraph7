      INTEGER NLOOPLIB
      PARAMETER (NLOOPLIB=9)
C     CutTools, Ninja and NinjaRS can provide QP when installed with
C      support
      INTEGER QP_NLOOPLIB,TIRUST_QP_COUNT
      PARAMETER (TIRUST_QP_COUNT=1)
      PARAMETER (QP_NLOOPLIB=3+TIRUST_QP_COUNT)

      LOGICAL LOOPLIBS_AVAILABLE(NLOOPLIB)
      DATA LOOPLIBS_AVAILABLE/.TRUE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.
     $ ,.TRUE.,.TRUE.,.TRUE.,.TRUE./
      COMMON/ML5_0_LOOPLIBS_AV/ LOOPLIBS_AVAILABLE
C     A FLAG TO DENOTE WHETHER THE CORRESPONDING DIRECTION TESTS
C      AVAILABLE OR NOT IN THE LOOPLIBS
      LOGICAL LOOPLIBS_DIRECTEST(NLOOPLIB)
      DATA LOOPLIBS_DIRECTEST /9*.TRUE./
C     Specifying for which reduction tool quadruple precision is
C      available.
C     The index 0 is dummy and simply means that the corresponding
C      loop_library is not available
C     in which case neither is its quadruple precision version.
      LOGICAL LOOPLIBS_QPAVAILABLE(0:NLOOPLIB)
      DATA LOOPLIBS_QPAVAILABLE(0:5)/.FALSE.,.TRUE.,4*.FALSE./
      DATA LOOPLIBS_QPAVAILABLE(6)/.TRUE./
      DATA LOOPLIBS_QPAVAILABLE(7)/.FALSE./
      DATA LOOPLIBS_QPAVAILABLE(8)/.TRUE./
      DATA LOOPLIBS_QPAVAILABLE(9)/.TRUE./
        QP_TOOLS_AVAILABLE=.FALSE.
        INDEX_QP_TOOLS(1:QP_NLOOPLIB+1)=0
C       Explain an explicit NinjaRS-only request before dropping
C        unavailable tools.
        IF(MLREDUCTIONLIB(1).EQ.8.AND.MLREDUCTIONLIB(2)
     $   .EQ.0.AND..NOT.LOOPLIBS_AVAILABLE(8))THEN
          WRITE(*,*) 'NinjaRS (MLReductionLib=8) is unavailable in'
     $     //' this generated process.'
          WRITE(*,*) 'Run install ninjars in MadGraph, then regenerate'
     $     //' and rebuild this process.'
          STOP 'Requested NinjaRS backend is unavailable.'
        ENDIF
        IF(MLREDUCTIONLIB(1).EQ.9.AND.MLREDUCTIONLIB(2)
     $   .EQ.0.AND..NOT.LOOPLIBS_AVAILABLE(9))THEN
          WRITE(*,*) 'TIRust (MLReductionLib=9) is unavailable in this'
     $     //' generated process.'
          WRITE(*,*) 'Run install tirust, then regenerate and rebuild'
     $     //' this process.'
          WRITE(*,*) 'Requested TIRust backend is unavailable.'
          STOP 1
        ENDIF
C       SKIP THE ONES THAT NOT AVAILABLE
        J=1
        DO I=1,NLOOPLIB
          IF(MLREDUCTIONLIB(J).EQ.0)EXIT
          IF(.NOT.LOOPLIBS_AVAILABLE(MLREDUCTIONLIB(J)))THEN
            MLREDUCTIONLIB(J:NLOOPLIB-1)=MLREDUCTIONLIB(J+1:NLOOPLIB)
            MLREDUCTIONLIB(NLOOPLIB)=0
          ELSE
            J=J+1
          ENDIF
        ENDDO
        IF(MLREDUCTIONLIB(1).EQ.0)THEN
          STOP 'No available loop reduction lib is provided. Make sure'
     $     //' MLReductionLib is correct.'
        ENDIF
        J=0
        DO I=1,NLOOPLIB
          IF(LOOPLIBS_QPAVAILABLE(MLREDUCTIONLIB(I)))THEN
            J=J+1
            IF(.NOT.QP_TOOLS_AVAILABLE) THEN
              QP_TOOLS_AVAILABLE=.TRUE.
            ENDIF
            INDEX_QP_TOOLS(J)=I
          ENDIF
        ENDDO

