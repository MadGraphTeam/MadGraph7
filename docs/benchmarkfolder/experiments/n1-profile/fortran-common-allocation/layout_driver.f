C Recording stubs test the actual rendered common entry and switches.
C All floating-point exceptional operands are inspected through integer bits.
      MODULE AUDIT_STATE
      USE ISO_FORTRAN_ENV, ONLY: INT64
      USE, INTRINSIC :: IEEE_ARITHMETIC
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INTEGER CALLS(9),INIT_CALLS,CITE_CALLS,ACTION
      INTEGER EXPECT_N,EXPECT_R,EXPECT_LIB
      LOGICAL REFUSE,UV,IR
      REAL*8 EXPECT_PL(0:3,6),EXPECT_DEN(0:3,5)
      COMPLEX*16 EXPECT_M(6)
      TYPE(IEEE_FLAG_TYPE) FLAGS(5)
      INTEGER(INT64),PARAMETER :: SIGNBIT=INT(Z'8000000000000000',INT64)
      CONTAINS
      SUBROUTINE REQUIRE(TEST,MESSAGE)
      LOGICAL TEST
      CHARACTER*(*) MESSAGE
      IF(.NOT.TEST) THEN
        CALL IEEE_SET_HALTING_MODE(FLAGS,.FALSE.)
        CALL IEEE_SET_FLAG(FLAGS,.FALSE.)
        WRITE(*,*) 'ASSERTION: ',MESSAGE,' action ',ACTION
        STOP 71
      ENDIF
      END SUBROUTINE

      INTEGER FUNCTION FLAG_MASK(VALUES)
      LOGICAL VALUES(5)
      INTEGER I
      FLAG_MASK=0
      DO I=1,5
        IF(VALUES(I)) FLAG_MASK=IBSET(FLAG_MASK,I-1)
      ENDDO
      END FUNCTION

      INTEGER(INT64) FUNCTION HASH_BITS(VALUES)
      INTEGER(INT64) VALUES(:)
      INTEGER I
      HASH_BITS=0_INT64
      DO I=1,SIZE(VALUES)
        HASH_BITS=IEOR(ISHFTC(HASH_BITS,3),VALUES(I))
      ENDDO
      END FUNCTION

      SUBROUTINE CHECK_INPUT(LIB,N,R,P,D,M,HAS_P,HAS_D)
      INTEGER LIB,N,R,I,J
      REAL*8 P(0:3,N),D(0:3,N-1)
      COMPLEX*16 M(N)
      LOGICAL HAS_P,HAS_D
      CALL REQUIRE(LIB.EQ.EXPECT_LIB,'backend dispatch')
      CALL REQUIRE(N.EQ.EXPECT_N.AND.R.EQ.EXPECT_R,'backend dimensions')
      DO I=1,N
        CALL REQUIRE(ALL(TRANSFER(M(I),(/0_INT64,0_INT64/)).EQ.
     $    TRANSFER(EXPECT_M(I),(/0_INT64,0_INT64/))),'backend masses')
        IF(HAS_P) THEN
          DO J=0,3
            CALL REQUIRE(TRANSFER(P(J,I),0_INT64).EQ.
     $        TRANSFER(EXPECT_PL(J,I),0_INT64),'backend PL')
          ENDDO
        ENDIF
      ENDDO
      IF(HAS_D) THEN
        DO I=1,N-1
          DO J=0,3
            CALL REQUIRE(TRANSFER(D(J,I),0_INT64).EQ.
     $        TRANSFER(EXPECT_DEN(J,I),0_INT64),'backend PDEN')
          ENDDO
        ENDDO
      ENDIF
      CALLS(LIB)=CALLS(LIB)+1
      END SUBROUTINE

      SUBROUTINE FILL_COEFS(LIB,NC,C)
      INTEGER LIB,NC,I,J,BASE
      COMPLEX*16 C(0:NC-1,3)
      INCLUDE 'MadLoopParams.inc'
      BASE=100*LIB+1000*LoopPoleMode
      IF(UV) BASE=BASE+10000
      IF(IR) BASE=BASE+20000
      DO J=1,3
        DO I=0,NC-1
          C(I,J)=DCMPLX(BASE+10*J+I,LIB+2*J+I)
        ENDDO
      ENDDO
      END SUBROUTINE
      END MODULE

      PROGRAM LAYOUT
      USE AUDIT_STATE
      IMPLICIT NONE
      INTEGER I
      FLAGS=(/IEEE_INVALID,IEEE_DIVIDE_BY_ZERO,IEEE_OVERFLOW,
     $ IEEE_UNDERFLOW,IEEE_INEXACT/)
      ACTION=0
      DO I=1,5
        CALL REQUIRE(IEEE_SUPPORT_FLAG(FLAGS(I)),'IEEE flag support')
        CALL REQUIRE(IEEE_SUPPORT_HALTING(FLAGS(I)),'IEEE halt support')
      ENDDO
      CALL REQUIRE(STORAGE_SIZE(0.0_16).EQ.128,'binary128 storage')
      CALL RUN_TIR()
      CALL RUN_SWITCHES()
      WRITE(*,'(A)') 'PASS'
      END

      SUBROUTINE RUN_TIR()
      USE AUDIT_STATE
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      REAL*8 MU_R,LSCALE
      INTEGER CTMODE,LIBINDEX,ID,R,G,S,J,B,K,C,N,RANK
      INTEGER LIBRARIES(5),SIZES(4),RANKS(4),MODES(4)
      PARAMETER(LIBRARIES=(/2,3,4,7,9/))
      PARAMETER(SIZES=(/1,2,3,6/),RANKS=(/0,1,2,2/))
      PARAMETER(MODES=(/1,2,4,5/))
      COMPLEX*16 LOOPCOEFS(@LOOPDIMS@)
      LOGICAL CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT,COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,
     $ SAMURAIINIT,NINJAINIT,COLLIERINIT
      COMMON/TEST_COUPL/MU_R
      COMMON/TEST_CT/LSCALE,CTMODE
      COMMON/TEST_I_LIB/LIBINDEX
      COMMON/TEST_LCOEFS/LOOPCOEFS
      CALLS=0
      INIT_CALLS=0
      CITE_CALLS=0
      CTINIT=.TRUE.
      TIRINIT=.TRUE.
      GOLEMINIT=.TRUE.
      SAMURAIINIT=.TRUE.
      NINJAINIT=.TRUE.
      COLLIERINIT=.TRUE.
      REFUSE=.FALSE.
      UV=.TRUE.
      IR=.TRUE.
      LoopPoleMode=0
      IREGIMODE=1
      COLLIERUseInternalStabilityTest=.TRUE.
      MLReductionLib=0
      LIBINDEX=1
      MU_R=7D0
      LSCALE=7D0
      DO G=1,2
        DO S=1,2
          DO J=0,14
            @LOOPINIT@
          ENDDO
        ENDDO
      ENDDO
      DO B=1,5
        DO K=1,4
          N=SIZES(K)
          RANK=RANKS(K)
          DO C=1,4
            CTMODE=MODES(C)
            MLReductionLib(1)=LIBRARIES(B)
            LoopPoleMode=0
            UV=.TRUE.
            IR=.TRUE.
            CALL TEST_CLEAR_TIR_CACHE()
C Cold, same group with another split order, and another loop group.
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
            CALL TIR_CASE(N,RANK,1,2,
     $        .NOT.@CACHE@.OR.LIBRARIES(B).EQ.4,.FALSE.)
            CALL TIR_CASE(N,RANK,2,1,.TRUE.,.FALSE.)
C All three pole-configuration transitions must invalidate the cache.
            LoopPoleMode=1
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
            UV=.FALSE.
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
            IR=.FALSE.
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
C A backend transition and return may not reuse the earlier tensors.
            MLReductionLib(1)=9
            IF(LIBRARIES(B).EQ.9) MLReductionLib(1)=3
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
            MLReductionLib(1)=LIBRARIES(B)
            CALL TIR_CASE(N,RANK,1,1,.TRUE.,.FALSE.)
          ENDDO
        ENDDO
      ENDDO
C A refused backend-9 call cannot populate the cache.  These repeated
C returns also exercise an unallocated cleanup path with static locals.
      MLReductionLib(1)=9
      CTMODE=2
      LoopPoleMode=0
      UV=.TRUE.
      IR=.TRUE.
      CALL TEST_CLEAR_TIR_CACHE()
      REFUSE=.TRUE.
      CALL TIR_CASE(3,2,1,1,.TRUE.,.FALSE.)
      CALL TIR_CASE(3,2,1,1,.TRUE.,.FALSE.)
      REFUSE=.FALSE.
      CALL TIR_CASE(3,2,1,1,.TRUE.,.FALSE.)
      CALL TIR_CASE(3,2,1,2,.NOT.@CACHE@,.FALSE.)
C Backend 9 must not evaluate the exceptional PL/mass payloads through
C legacy momentum preparation; the recording stub only inspects bits.
      CALL TEST_CLEAR_TIR_CACHE()
      CALL TIR_CASE(3,2,1,1,.TRUE.,.TRUE.)
      CALL REQUIRE(INIT_CALLS.EQ.1,'one legacy TIR initialization')
      WRITE(*,'(A,11I8)') 'COUNTS ',CALLS,INIT_CALLS,CITE_CALLS
      END

      SUBROUTINE TIR_CASE(N,RANK,G,S,WILL_CALL,SIGNAL_INPUT)
      USE AUDIT_STATE
      IMPLICIT NONE
      INCLUDE 'MadLoopParams.inc'
      INTEGER N,RANK,G,S,I,J,LIB,BEFORE,CTMODE,LIBINDEX,DELTA
      INTEGER(INT64) P_BITS(0:3,8),M_BITS(2,8),RES_BITS(6),TEMP
      REAL*8 P(0:3,8),LSCALE,D(0:3,5),TOTAL
      COMPLEX*16 M(8),RES(3)
      LOGICAL WILL_CALL,SIGNAL_INPUT,STABLE,DP_FPE,QP_FPE,AFTER(5)
      LOGICAL INITIAL(5),FLAGS_BEFORE(5),REVERSE
      COMMON/TEST_CT/LSCALE,CTMODE
      COMMON/TEST_I_LIB/LIBINDEX
      COMMON/TEST_FPE_IN_REDUCTION/DP_FPE,QP_FPE
      ACTION=ACTION+1
      LIB=MLReductionLib(1)
      P=-321D0
      M=DCMPLX(-654D0,987D0)
      DO I=0,3
        TOTAL=0D0
        DO J=1,N-1
          P(I,J+1)=DBLE((I+1)*J)
          IF(I.EQ.3.AND.J.EQ.1) P(I,J+1)=-0D0
          TOTAL=TOTAL+P(I,J+1)
          D(I,J)=TOTAL
        ENDDO
        P(I,N+1)=-TOTAL
      ENDDO
      DO J=1,N
        M(J+1)=DCMPLX(J*J,-DBLE(J)/16D0)
      ENDDO
      IF(SIGNAL_INPUT) THEN
        TEMP=INT(Z'7FF0000000000001',INT64)
        P(0,2)=TRANSFER(TEMP,P(0,2))
        M(2)=TRANSFER((/TEMP,SIGNBIT/),M(2))
      ENDIF
      P_BITS=RESHAPE(TRANSFER(P,(/0_INT64/),32),SHAPE(P_BITS))
      M_BITS=RESHAPE(TRANSFER(M,(/0_INT64/),16),SHAPE(M_BITS))
      EXPECT_N=N
      EXPECT_R=RANK
      EXPECT_LIB=LIB
      EXPECT_PL(:,1:N)=P(:,2:N+1)
      EXPECT_DEN(:,1:N-1)=D(:,1:N-1)
      EXPECT_M(1:N)=M(2:N+1)
      REVERSE=WILL_CALL.AND.(CTMODE.EQ.2.OR.CTMODE.EQ.5)
     $ .AND.N.GT.2.AND.(LIB.EQ.2.OR.LIB.EQ.4.OR.LIB.EQ.7)
      IF(REVERSE) THEN
        DO J=1,N
          DO I=0,3
            TEMP=IEOR(TRANSFER(P(I,N+2-J),0_INT64),SIGNBIT)
            EXPECT_PL(I,J)=TRANSFER(TEMP,EXPECT_PL(I,J))
            P_BITS(I,J+1)=TEMP
          ENDDO
        ENDDO
        DO J=1,N-1
          EXPECT_DEN(:,J)=D(:,N-J)
        ENDDO
        DO J=2,N
          EXPECT_M(J)=M(N-J+3)
          M_BITS(:,J+1)=TRANSFER(M(N-J+3),(/0_INT64,0_INT64/))
        ENDDO
      ENDIF
      STABLE=.TRUE.
      DP_FPE=.FALSE.
      QP_FPE=.TRUE.
      BEFORE=SUM(CALLS)
      INITIAL=.FALSE.
      IF(MOD(ACTION,3).EQ.1) INITIAL(5)=.TRUE.
      IF(MOD(ACTION,3).EQ.2) INITIAL=.TRUE.
      CALL IEEE_SET_HALTING_MODE(FLAGS,.FALSE.)
      IF(SIGNAL_INPUT) THEN
        CALL IEEE_SET_HALTING_MODE(IEEE_INVALID,.TRUE.)
        INITIAL(1)=.FALSE.
      ENDIF
      CALL IEEE_SET_FLAG(FLAGS,INITIAL)
      CALL IEEE_GET_FLAG(FLAGS,FLAGS_BEFORE)
      CALL TEST_TIRLOOP(S,G,1,N,P(0,2),M(2),RANK,RES,STABLE)
      CALL IEEE_GET_FLAG(FLAGS,AFTER)
      CALL IEEE_SET_HALTING_MODE(FLAGS,.FALSE.)
      CALL IEEE_SET_FLAG(FLAGS,.FALSE.)
      DELTA=SUM(CALLS)-BEFORE
      CALL REQUIRE(DELTA.EQ.MERGE(1,0,WILL_CALL),'cache dispatch count')
      CALL REQUIRE(ALL(TRANSFER(P,(/0_INT64/),32).EQ.
     $ RESHAPE(P_BITS,(/32/))),'PL input mutation/canaries')
      CALL REQUIRE(ALL(TRANSFER(M,(/0_INT64/),16).EQ.
     $ RESHAPE(M_BITS,(/16/))),'mass input mutation/canaries')
      CALL REQUIRE(QP_FPE,'QP FPE sentinel preserved')
      CALL REQUIRE(STABLE.EQV.(.NOT.REFUSE),'returned stability')
      CALL REQUIRE(DP_FPE.EQV.REFUSE,'returned DP FPE')
      IF(REFUSE) CALL REQUIRE(ALL(RES.EQ.(0D0,0D0)),'refusal zero result')
      IF(LIB.EQ.4) THEN
        DO I=1,3
          CALL REQUIRE(RES(I).EQ.DCMPLX(400+I,-I),'Golem direct return')
        ENDDO
      ENDIF
      RES_BITS=TRANSFER(RES,RES_BITS)
      WRITE(*,'(A,15I7,1X,6Z16.16)') 'T ',ACTION,LIB,N,RANK,
     $ CTMODE,G,S,LoopPoleMode,MERGE(1,0,UV),MERGE(1,0,IR),
     $ DELTA,MERGE(1,0,STABLE),MERGE(1,0,DP_FPE),
     $ FLAG_MASK(FLAGS_BEFORE),FLAG_MASK(AFTER),RES_BITS
      END

      SUBROUTINE RUN_SWITCHES()
      USE AUDIT_STATE
      IMPLICIT NONE
      INTEGER NSET(4),I,C,F,H,P
      PARAMETER(NSET=(/1,2,3,6/))
      DO P=1,2
        DO I=1,4
          DO C=1,6
            DO F=0,2
              DO H=0,2
                CALL SWITCH_CASE(P,NSET(I),C,F,H)
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE SWITCH_CASE(PREC,N,CT,FLAG_STATE,HALT_STATE)
      USE AUDIT_STATE
      IMPLICIT NONE
      INTEGER PREC,N,CT,FLAG_STATE,HALT_STATE,I,J,K,L,POS
      INTEGER(INT64) PATTERN(9),QHIGH(9),QLOW(9)
      INTEGER(INT64) PB(0:3,8),DB(0:3,7),MB(2,8)
      INTEGER(INT64) QPB(2,0:3,8),QDB(2,0:3,7),QMB(4,8)
      INTEGER(INT64) PE(0:3,8),DE(0:3,7),ME(2,8)
      INTEGER(INT64) QPE(2,0:3,8),QDE(2,0:3,7),QME(4,8)
      INTEGER(INT64) HASH_P,HASH_D,HASH_M
      REAL*8 PL(0:3,8),DEN(0:3,7)
      COMPLEX*16 MASS(8)
      REAL*16 QPL(0:3,8),QDEN(0:3,7)
      COMPLEX*32 QMASS(8)
      LOGICAL INITIAL(5),HALTING(5),AFTER(5),HALT_AFTER(5),BEFORE(5)
      DATA PATTERN/Z'0000000000000000',Z'8000000000000000',
     $ Z'3FF0000000000000',Z'BFF0000000000000',Z'7FF0000000000001',
     $ Z'7FF8000000000001',Z'7FF0000000000000',Z'FFF0000000000000',
     $ Z'0000000000000001'/
      DATA QHIGH/Z'0000000000000000',Z'8000000000000000',
     $ Z'3FFF000000000000',Z'BFFF000000000000',Z'7FFF000000000000',
     $ Z'7FFF800000000000',Z'7FFF000000000000',Z'FFFF000000000000',
     $ Z'0000000000000000'/
      DATA QLOW/0,0,0,0,1,1,0,0,1/
      ACTION=ACTION+1
C Binary128 word order is checked explicitly before constructing operands.
      CALL REQUIRE(ALL(TRANSFER(1.0_16,(/0_INT64,0_INT64/)).EQ.
     $ (/0_INT64,INT(Z'3FFF000000000000',INT64)/)),
     $ 'binary128 little-endian words')
      DO J=1,8
        DO I=0,3
          POS=MOD(I+4*(J-1),9)+1
          PB(I,J)=PATTERN(POS)
          QPB(:,I,J)=(/QLOW(POS),QHIGH(POS)/)
          IF(J.LE.7) THEN
            POS=MOD(I+4*(J-1)+3,9)+1
            DB(I,J)=PATTERN(POS)
            QDB(:,I,J)=(/QLOW(POS),QHIGH(POS)/)
          ENDIF
        ENDDO
        DO K=1,2
          POS=MOD(2*J+K,9)+1
          MB(K,J)=PATTERN(POS)
          QMB(2*K-1:2*K,J)=(/QLOW(POS),QHIGH(POS)/)
        ENDDO
      ENDDO
      PE=PB
      DE=DB
      ME=MB
      QPE=QPB
      QDE=QDB
      QME=QMB
      IF((CT.EQ.2.OR.CT.EQ.5).AND.N.GT.2) THEN
        DO J=1,N
          PE(:,J+1)=IEOR(PB(:,N+2-J),SIGNBIT)
          QPE(:,:,J+1)=QPB(:,:,N+2-J)
          QPE(2,:,J+1)=IEOR(QPE(2,:,J+1),SIGNBIT)
        ENDDO
        DO J=1,N-1
          DE(:,J+1)=DB(:,N-J+1)
          QDE(:,:,J+1)=QDB(:,:,N-J+1)
        ENDDO
        DO J=2,N
          ME(:,J+1)=MB(:,N-J+3)
          QME(:,J+1)=QMB(:,N-J+3)
        ENDDO
      ENDIF
      PL=RESHAPE(TRANSFER(PB,(/0D0/),32),SHAPE(PL))
      DEN=RESHAPE(TRANSFER(DB,(/0D0/),28),SHAPE(DEN))
      MASS=TRANSFER(MB,MASS)
      QPL=RESHAPE(TRANSFER(QPB,(/0.0_16/),32),SHAPE(QPL))
      QDEN=RESHAPE(TRANSFER(QDB,(/0.0_16/),28),SHAPE(QDEN))
      QMASS=TRANSFER(QMB,QMASS)
      INITIAL=.FALSE.
      IF(FLAG_STATE.EQ.1) INITIAL(5)=.TRUE.
      IF(FLAG_STATE.EQ.2) INITIAL=.TRUE.
      HALTING=.FALSE.
      IF(HALT_STATE.EQ.1) HALTING(1)=.TRUE.
      IF(HALT_STATE.EQ.2) HALTING=.TRUE.
C Install traps first: gfortran's halting-mode setter clears flags.  Only
C masked exceptions may be raised as sticky flags without trapping during
C setup.  The state immediately before the tested call is retained too.
      INITIAL=INITIAL.AND.(.NOT.HALTING)
      CALL IEEE_SET_HALTING_MODE(FLAGS,.FALSE.)
      CALL IEEE_SET_HALTING_MODE(FLAGS,HALTING)
      CALL IEEE_SET_FLAG(FLAGS,INITIAL)
      CALL IEEE_GET_FLAG(FLAGS,BEFORE)
      IF(PREC.EQ.1) THEN
        CALL TEST_SWITCH_ORDER(CT,N,PL(0,2),DEN(0,2),MASS(2))
      ELSE
        CALL TEST_MP_SWITCH_ORDER(CT,N,QPL(0,2),QDEN(0,2),QMASS(2))
      ENDIF
      CALL IEEE_GET_FLAG(FLAGS,AFTER)
      CALL IEEE_GET_HALTING_MODE(FLAGS,HALT_AFTER)
      CALL IEEE_SET_HALTING_MODE(FLAGS,.FALSE.)
      CALL IEEE_SET_FLAG(FLAGS,.FALSE.)
      CALL REQUIRE(ALL(BEFORE.EQV.INITIAL),'switch IEEE flag setup')
      CALL REQUIRE(ALL(AFTER.EQV.BEFORE),'switch sticky IEEE flags')
      CALL REQUIRE(ALL(HALT_AFTER.EQV.HALTING),'switch IEEE halting')
      IF(PREC.EQ.1) THEN
        CALL REQUIRE(ALL(TRANSFER(PL,(/0_INT64/),32).EQ.
     $    RESHAPE(PE,(/32/))),'DP switch PL input mutation/canaries')
        CALL REQUIRE(ALL(TRANSFER(DEN,(/0_INT64/),28).EQ.
     $    RESHAPE(DE,(/28/))),'DP switch PDEN input mutation/canaries')
        CALL REQUIRE(ALL(TRANSFER(MASS,(/0_INT64/),16).EQ.
     $    RESHAPE(ME,(/16/))),'DP switch mass input mutation/canaries')
        HASH_P=HASH_BITS(TRANSFER(PL,(/0_INT64/),32))
        HASH_D=HASH_BITS(TRANSFER(DEN,(/0_INT64/),28))
        HASH_M=HASH_BITS(TRANSFER(MASS,(/0_INT64/),16))
      ELSE
        CALL REQUIRE(ALL(TRANSFER(QPL,(/0_INT64/),64).EQ.
     $    RESHAPE(QPE,(/64/))),'QP switch PL input mutation/canaries')
        CALL REQUIRE(ALL(TRANSFER(QDEN,(/0_INT64/),56).EQ.
     $    RESHAPE(QDE,(/56/))),'QP switch PDEN input mutation/canaries')
        CALL REQUIRE(ALL(TRANSFER(QMASS,(/0_INT64/),32).EQ.
     $    RESHAPE(QME,(/32/))),'QP switch mass input mutation/canaries')
        HASH_P=HASH_BITS(TRANSFER(QPL,(/0_INT64/),64))
        HASH_D=HASH_BITS(TRANSFER(QDEN,(/0_INT64/),56))
        HASH_M=HASH_BITS(TRANSFER(QMASS,(/0_INT64/),32))
      ENDIF
      WRITE(*,'(A,9I7,1X,3Z16.16)') 'S ',ACTION,PREC,N,CT,
     $ FLAG_STATE,HALT_STATE,FLAG_MASK(BEFORE),FLAG_MASK(AFTER),FLAG_MASK(HALT_AFTER),
     $ HASH_P,HASH_D,HASH_M
      END

      SUBROUTINE ML_LOOP_POLE_FLAGS(U,V)
      USE AUDIT_STATE, ONLY: UV,IR
      LOGICAL U,V
      U=UV
      V=IR
      END

      SUBROUTINE TEST_INITTIR()
      USE AUDIT_STATE, ONLY: INIT_CALLS
      INIT_CALLS=INIT_CALLS+1
      END

      SUBROUTINE CITE(KEY,DESCRIPTION)
      USE AUDIT_STATE, ONLY: CITE_CALLS
      CHARACTER*(*) KEY,DESCRIPTION
      CITE_CALLS=CITE_CALLS+1
      END

      SUBROUTINE PMLOOP(N,R,P,D,M,MU,C,STABLE)
      USE AUDIT_STATE
      INTEGER N,R,NC
      REAL*8 P(0:3,N),D(0:3,N-1),MU
      COMPLEX*16 M(N),C(0:(R+1)*(R+2)*(R+3)*(R+4)/24-1,3)
      LOGICAL STABLE
      CALL CHECK_INPUT(2,N,R,P,D,M,.TRUE.,.TRUE.)
      NC=(R+1)*(R+2)*(R+3)*(R+4)/24
      CALL FILL_COEFS(2,NC,C)
      STABLE=.TRUE.
      END

      SUBROUTINE IMLOOP(CT,MODE,N,MAXC,R,D,M,MU,C,STABLE)
      USE AUDIT_STATE
      INTEGER CT,MODE,N,MAXC,R
      REAL*8 D(0:3,N-1),MU
      COMPLEX*16 M(N),C(0:MAXC-1,3)
      LOGICAL STABLE
      CALL CHECK_INPUT(3,N,R,EXPECT_PL,D,M,.FALSE.,.TRUE.)
      CALL FILL_COEFS(3,MAXC,C)
      STABLE=.TRUE.
      END

      SUBROUTINE TEST_CONVERT_PJFRY_COEFFS(R,A,B)
      INTEGER R,NC,I,J
      COMPLEX*16 A(0:14,3),B(0:14,3)
      NC=(R+1)*(R+2)*(R+3)*(R+4)/24
      DO J=1,3
        DO I=0,NC-1
          B(I,J)=A(I,J)
        ENDDO
      ENDDO
      END

      SUBROUTINE TEST_CONVERT_IREGI_COEFFS(R,A,B)
      INTEGER R
      COMPLEX*16 A(0:14,3),B(0:14,3)
      CALL TEST_CONVERT_PJFRY_COEFFS(R,A,B)
      END

      SUBROUTINE TEST_COLLIERLOOP(CT,N,R,P,D,M,C,E)
      USE AUDIT_STATE
      INTEGER CT,N,R
      REAL*8 P(0:3,N),D(0:3,N-1)
      COMPLEX*16 M(N),C(0:14,3),E(0:14,3)
      CALL TEST_SWITCH_ORDER(CT,N,P,D,M)
      CALL CHECK_INPUT(7,N,R,P,D,M,.TRUE.,.TRUE.)
      CALL FILL_COEFS(7,15,C)
      E=DCMPLX(0.125D0,0.25D0)
      END

      SUBROUTINE TEST_GOLEMLOOP(N,P,M,R,C,STABLE)
      USE AUDIT_STATE
      INTEGER N,R,I
      REAL*8 P(0:3,N)
      COMPLEX*16 M(N),C(3)
      LOGICAL STABLE
      CALL CHECK_INPUT(4,N,R,P,EXPECT_DEN,M,.TRUE.,.FALSE.)
      DO I=1,3
        C(I)=DCMPLX(400+I,-I)
      ENDDO
      STABLE=.TRUE.
      END

      SUBROUTINE TEST_TIRUSTLOOP(CT,N,R,P,M,C,STABLE)
      USE AUDIT_STATE
      INTEGER CT,N,R
      REAL*8 P(0:3,N)
      COMPLEX*16 M(N),C(0:14,3)
      LOGICAL STABLE
      CALL CHECK_INPUT(9,N,R,P,EXPECT_DEN,M,.TRUE.,.FALSE.)
      CALL FILL_COEFS(9,15,C)
      STABLE=.NOT.REFUSE
      END
