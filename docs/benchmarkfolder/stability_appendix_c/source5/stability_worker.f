C     Source-only Appendix C pilot. No automatic precision rescue.
C     Expand @PREFIX@, @NEXTERNAL@ and @NBORN@ before private relinking.
C     Usage: stability-check POINTS_FILE BACKEND
C     The controller freezes CTModeInit=1 and CTModeRun=1 before startup.
      PROGRAM STABILITY_MADLOOP_WORKER
      IMPLICIT NONE
      INTEGER NEXTERNAL,NSQSO_BORN,REQUIRED_POINTS,MAX_BOOTSTRAP
      PARAMETER(NEXTERNAL=@NEXTERNAL@,NSQSO_BORN=@NBORN@)
      PARAMETER(REQUIRED_POINTS=100,MAX_BOOTSTRAP=16)
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      INCLUDE 'loop_max_coefs.inc'
      INCLUDE 'stability-layout.inc'
      REAL*8, ALLOCATABLE :: POINTS(:,:,:),ANS(:,:),ACC(:)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NPOINTS,NDIM,NSO,I,K,METHOD,MODE,CODE,IOS
      INTEGER BACKEND,CALL_INDEX,BOOTSTRAP,POINT_INDEX
      LOGICAL READY
      CHARACTER*4096 POINT_FILE,ARG,EXTRA

C     No implicit point generation, extra arguments or precision forcing.
      IF(COMMAND_ARGUMENT_COUNT().NE.2) STOP 11
      CALL GET_COMMAND_ARGUMENT(1,POINT_FILE)
      CALL GET_COMMAND_ARGUMENT(2,ARG)
      READ(ARG,*,IOSTAT=IOS) BACKEND
      IF(IOS.NE.0) STOP 11
      IF(BACKEND.NE.1.AND.BACKEND.NE.6.AND.BACKEND.NE.7
     $ .AND.BACKEND.NE.8.AND.BACKEND.NE.9) STOP 11
      OPEN(73,FILE=TRIM(POINT_FILE),STATUS='OLD',ACTION='READ',
     $ IOSTAT=IOS)
      IF(IOS.NE.0) STOP 12
      READ(73,*,IOSTAT=IOS) NPOINTS
      IF(IOS.NE.0.OR.NPOINTS.NE.REQUIRED_POINTS) STOP 12
      ALLOCATE(POINTS(0:3,NEXTERNAL,NPOINTS))
      DO K=1,NPOINTS
         DO I=1,NEXTERNAL
            READ(73,*,IOSTAT=IOS) POINTS(:,I,K)
            IF(IOS.NE.0) STOP 12
         ENDDO
      ENDDO
C     Reject surplus nonblank records rather than silently changing input.
 10   READ(73,'(A)',IOSTAT=IOS) EXTRA
      IF(IOS.EQ.0) THEN
         IF(LEN_TRIM(EXTRA).NE.0) STOP 12
         GOTO 10
      ENDIF
      IF(IOS.GT.0) STOP 12
      CLOSE(73)

      CALL SETPARA('param_card.dat')
      CALL @PREFIX@GET_ANSWER_DIMENSION(NDIM)
      CALL @PREFIX@GET_NSQSO_LOOP(NSO)
      IF(NDIM.NE.MAX(NSO,NSQSO_BORN)) STOP 13
      IF(NSO.LT.0.OR.NSQSO_BORN.LT.0) STOP 13
      IF(NSO.NE.NSQUAREDSO.OR.NLOOPGROUPS.LE.0) STOP 13
      DO I=1,NLOOPGROUPS
         IF(STABILITY_NCOEFS(I).LE.0.OR.
     $        STABILITY_NCOEFS(I).GT.LOOPMAXCOEFS) STOP 13
      ENDDO
      ALLOCATE(ANS(0:3,0:NDIM),ACC(0:NSO))
      OPEN(75,FILE='stability-values.dat',STATUS='NEW',
     $ ACTION='WRITE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 14
      WRITE(75,'(A)')
     $ '# phase call point method order status Born finite single'//
     $ ' double accuracy'
      FLUSH(75)
      OPEN(76,FILE='stability-dimensions.dat',STATUS='NEW',
     $ ACTION='WRITE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 14
      WRITE(76,'(5I12)') 1,NPOINTS,NDIM,NSO,NSQSO_BORN
      CLOSE(76)
      OPEN(77,FILE='stability-groups.dat',STATUS='NEW',
     $ ACTION='WRITE',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 14
      WRITE(77,'(A,3I12)') '# GROUP_SCHEMA ',1,NSQUAREDSO,
     $ NLOOPGROUPS
      WRITE(77,'(A)') '# G phase call point method order group'//
     $ ' ncoefs nz_components nonfinite_coef_components stable'//
     $ ' nonfinite_res_components first_nz_words(2) res_words(6)'
      WRITE(77,'(A)') '# C phase call point method coverage active'//
     $ ' failed coef_bad_groups res_bad_groups fpe_dp fpe_qp'
      WRITE(77,'(A)') '# B phase call point method fpe_dp_before'//
     $ ' fpe_qp_before'
      FLUSH(77)

C     Positive modes are deliberately unchecked: normally code 110 and
C     ACC=-1. Do not ask the THRES wrapper to invent a checked result.
C     MadLoop reads its card inside the first call, so the controller's
C     immutable card must already request DP for initialization and run.
      CALL @PREFIX@FORCE_STABILITY_CHECK(.FALSE.)
      CTMODERUN=1
      CALL_INDEX=0
      READY=.FALSE.
      DO BOOTSTRAP=1,MAX_BOOTSTRAP
         POINT_INDEX=1+MOD(BOOTSTRAP-1,NPOINTS)
         P=POINTS(:,:,POINT_INDEX)
         CALL_INDEX=CALL_INDEX+1
         CALL STABILITY_EVALUATE(0,CALL_INDEX,POINT_INDEX,0,1,
     $        NEXTERNAL,NDIM,NSO,P,ANS,ACC,CODE)
C        All raw rows precede any rejection. T=3 is DP initialization;
C        T=1 is steady forced DP. A T=2/4 result is QP, not DP evidence.
         CALL STABILITY_CHECK_CARD(BACKEND,1)
         IF(CODE.EQ.110) THEN
            READY=.TRUE.
            EXIT
         ENDIF
         IF(CODE.NE.130) THEN
            WRITE(*,'(A,I12)') 'STABILITY_BOOTSTRAP_REJECT ',CODE
            FLUSH(6)
            CALL @PREFIX@EXIT_MADLOOP()
            CLOSE(75)
            CLOSE(77)
            STOP 15
         ENDIF
      ENDDO
      IF(.NOT.READY) THEN
         WRITE(*,'(A)') 'STABILITY_BOOTSTRAP_LIMIT'
         FLUSH(6)
         CALL @PREFIX@EXIT_MADLOOP()
         CLOSE(75)
         CLOSE(77)
         STOP 16
      ENDIF
      WRITE(75,'(A,I12)') '# BOOTSTRAP_COMPLETE ',BOOTSTRAP
      FLUSH(75)

C     Each physical point supplies one three-method DP diagnostic.
C     The signed permutation is an exact 90-degree rotation about z;
C     incoming beam directions and every four-vector's energy stay put.
      DO K=1,NPOINTS
         DO METHOD=1,3
            P=POINTS(:,:,K)
            MODE=1
            IF(METHOD.EQ.2) MODE=2
            IF(METHOD.EQ.3) THEN
               DO I=1,NEXTERNAL
                  P(1,I)=-POINTS(2,I,K)
                  P(2,I)=POINTS(1,I,K)
               ENDDO
            ENDIF
            CALL_INDEX=CALL_INDEX+1
            CALL STABILITY_EVALUATE(1,CALL_INDEX,K,METHOD,MODE,
     $           NEXTERNAL,NDIM,NSO,P,ANS,ACC,CODE)
            CALL STABILITY_CHECK_CARD(BACKEND,MODE)
            IF(CODE.NE.110) THEN
               WRITE(*,'(A,I12)') 'STABILITY_PRIMARY_REJECT ',CODE
               FLUSH(6)
               CALL @PREFIX@EXIT_MADLOOP()
               CLOSE(75)
               CLOSE(77)
               STOP 17
            ENDIF
         ENDDO
      ENDDO
      WRITE(75,'(A,2I12)') '# COMPLETE ',BOOTSTRAP,3*NPOINTS
      FLUSH(75)
      WRITE(77,'(A,2I12)') '# COMPLETE ',BOOTSTRAP,3*NPOINTS
      FLUSH(77)
      CALL @PREFIX@EXIT_MADLOOP()
      CLOSE(75)
      CLOSE(77)
      END

      SUBROUTINE STABILITY_EVALUATE(PHASE,CALL_INDEX,POINT_INDEX,
     $ METHOD,MODE,NEXTERNAL,NDIM,NSO,P,ANS,ACC,CODE)
      IMPLICIT NONE
      INTEGER PHASE,CALL_INDEX,POINT_INDEX,METHOD,MODE
      INTEGER NEXTERNAL,NDIM,NSO,CODE,J
      REAL*8 P(0:3,NEXTERNAL),ANS(0:3,0:NDIM),ACC(0:NSO)
      REAL*8 TOTAL(0:3),ESTIMATE
      INCLUDE 'coupl.inc'
      INCLUDE 'MadLoopParams.inc'
      CTMODERUN=MODE
      TOTAL=P(:,1)+P(:,2)
      MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))
      CALL UPDATE_AS_PARAM()
      ANS=0D0
      ACC=-1D0
      WRITE(*,'(A,5I12)') 'STABILITY_CALL_BEGIN ',PHASE,
     $ CALL_INDEX,POINT_INDEX,METHOD,MODE
      FLUSH(6)
      CALL STABILITY_FLAG_BEFORE(PHASE,CALL_INDEX,POINT_INDEX,
     $ METHOD)
      CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)
C     An unchecked 110 can hide a failed tensor contribution. Snapshot
C     the actual active-group state before accepting any metric. The
C     snapshot only copies bits and performs integer operations; it does
C     not clear native flags, perform FP classification or alter COMMON.
      CALL STABILITY_GROUP_SNAPSHOT(PHASE,CALL_INDEX,POINT_INDEX,
     $ METHOD,NSO)
      WRITE(*,'(A,6I12)') 'STABILITY_CALL_END ',PHASE,
     $ CALL_INDEX,POINT_INDEX,METHOD,MODE,CODE
      FLUSH(6)
      DO J=0,NDIM
C        Born-only orders have no individual loop accuracy estimate.
         ESTIMATE=ACC(0)
         IF(J.LE.NSO) ESTIMATE=ACC(J)
         WRITE(75,'(6I12,5ES26.17E3)') PHASE,CALL_INDEX,
     $        POINT_INDEX,METHOD,J,CODE,ANS(:,J),ESTIMATE
      ENDDO
      FLUSH(75)
      END

      SUBROUTINE STABILITY_FLAG_BEFORE(PHASE,CALL_INDEX,
     $ POINT_INDEX,METHOD)
      IMPLICIT NONE
      INTEGER PHASE,CALL_INDEX,POINT_INDEX,METHOD
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/@PREFIX@FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $ FPE_IN_QP_REDUCTION
C     These native logical flags have DATA initialization in the bound
C     matrix source. Forced positive modes may leave them sticky across
C     calls, so preserve the incoming state without resetting anything.
      WRITE(77,'(A,6I12)') 'B',PHASE,CALL_INDEX,POINT_INDEX,
     $ METHOD,MERGE(1,0,FPE_IN_DP_REDUCTION),
     $ MERGE(1,0,FPE_IN_QP_REDUCTION)
      FLUSH(77)
      END

      SUBROUTINE STABILITY_CHECK_CARD(BACKEND,MODE)
      IMPLICIT NONE
      INTEGER BACKEND,MODE
      INCLUDE 'MadLoopParams.inc'
C     Changing precision or backend is never a silent rescue in this run.
C     These checks supplement the controller's pre/post file-hash checks.
      IF(CTMODEINIT.NE.1.OR.CTMODERUN.NE.MODE.OR.
     $ MLReductionLib(1).NE.BACKEND.OR.MLReductionLib(2).NE.0) THEN
         WRITE(*,'(A,5I12)') 'STABILITY_CARD_REJECT ',
     $        CTMODEINIT,CTMODERUN,MODE,MLReductionLib(1),
     $        MLReductionLib(2)
         FLUSH(6)
         CLOSE(75)
         CLOSE(77)
         CALL @PREFIX@EXIT_MADLOOP()
         STOP 18
      ENDIF
      END

      SUBROUTINE STABILITY_GROUP_SNAPSHOT(PHASE,CALL_INDEX,
     $ POINT_INDEX,METHOD,NSO)
      USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: INT64
      IMPLICIT NONE
      INTEGER PHASE,CALL_INDEX,POINT_INDEX,METHOD,NSO
      INCLUDE 'loop_max_coefs.inc'
      INCLUDE 'stability-layout.inc'
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,
     $ NLOOPGROUPS)
      COMMON/@PREFIX@LCOEFS/LOOPCOEFS
      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/@PREFIX@LOOPRES/LOOPRES,S
      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
      LOGICAL MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
      LOGICAL CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/@PREFIX@SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,
     $ CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,
     $ MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
      INTEGER SQSO_TARGET
      COMMON/@PREFIX@SOCHOICE/SQSO_TARGET
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      COMMON/@PREFIX@FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $ FPE_IN_QP_REDUCTION
      INTEGER I,J,K,L,NZ,NF_COEF,NF_RES,STABLE,ACTIVE,FAILED
      INTEGER COEF_BAD,RES_BAD,COVERAGE,FPE_DP,FPE_QP
      INTEGER(INT64) WORDS(2),FIRST_NZ(2),RES_WORDS(6)
      INTEGER(INT64) MAG_MASK,EXP_MASK
      PARAMETER(MAG_MASK=INT(Z'7FFFFFFFFFFFFFFF',INT64))
      PARAMETER(EXP_MASK=INT(Z'7FF0000000000000',INT64))

C     Exact layout and complete DP coefficient construction are required
C     before touching either coefficient or group-result workspaces.
C     The pinned Born-process source initializes every coefficient slot
C     and LOOPRES before construction, then S before the reduction list.
C     The controller verifies those source facts and rank-valid extents.
      IF(STORAGE_SIZE(LOOPCOEFS(0,1,1)).NE.128) STOP 19
      COVERAGE=0
      ACTIVE=0
      FAILED=0
      COEF_BAD=0
      RES_BAD=0
      FPE_DP=MERGE(1,0,FPE_IN_DP_REDUCTION)
      FPE_QP=MERGE(1,0,FPE_IN_QP_REDUCTION)
      IF(NSO.NE.NSQUAREDSO.OR..NOT.LOOP_REQ_SO_DONE.OR.
     $ SQSO_TARGET.NE.-1.OR.FILTER_SO) GOTO 200
      COVERAGE=1
      DO I=1,NLOOPGROUPS
         DO J=1,NSQUAREDSO
            NZ=0
            NF_COEF=0
            FIRST_NZ=0_INT64
            DO K=0,STABILITY_NCOEFS(I)-1
               WORDS=TRANSFER(LOOPCOEFS(K,J,I),WORDS)
               DO L=1,2
                  IF(IAND(WORDS(L),MAG_MASK).NE.0_INT64) THEN
                     IF(NZ.EQ.0) FIRST_NZ=WORDS
                     NZ=NZ+1
                  ENDIF
                  IF(IAND(WORDS(L),EXP_MASK).EQ.EXP_MASK)
     $                 NF_COEF=NF_COEF+1
               ENDDO
            ENDDO
            STABLE=-1
            NF_RES=0
C           Inactive flags are not consulted. A nonfinite coefficient
C           is nonzero and therefore active; signed zeros are inactive.
            IF(NZ.GT.0) THEN
               ACTIVE=ACTIVE+1
               STABLE=MERGE(1,0,S(J,I))
               IF(STABLE.EQ.0) FAILED=FAILED+1
               IF(NF_COEF.GT.0) COEF_BAD=COEF_BAD+1
            ENDIF
            RES_WORDS=TRANSFER(LOOPRES(:,J,I),RES_WORDS)
            DO L=1,6
               IF(IAND(RES_WORDS(L),EXP_MASK).EQ.EXP_MASK)
     $              NF_RES=NF_RES+1
            ENDDO
            IF(NZ.GT.0.AND.NF_RES.GT.0) RES_BAD=RES_BAD+1
            WRITE(77,'(A,11I12,8(1X,Z16.16))') 'G',PHASE,
     $           CALL_INDEX,POINT_INDEX,METHOD,J,I,
     $           STABILITY_NCOEFS(I),NZ,NF_COEF,STABLE,NF_RES,
     $           FIRST_NZ,RES_WORDS
         ENDDO
      ENDDO
 200  WRITE(77,'(A,11I12)') 'C',PHASE,CALL_INDEX,POINT_INDEX,
     $ METHOD,COVERAGE,ACTIVE,FAILED,COEF_BAD,RES_BAD,FPE_DP,FPE_QP
      FLUSH(77)
C     Bad or unavailable groups remain raw evidence for the controller;
C     no skipped contribution is promoted to a valid finite estimate.
C     Continue the fixed point population instead of silently replacing it.
      END
