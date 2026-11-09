      PROGRAM QUALIFY_BORN_FALLBACK
      USE, INTRINSIC :: IEEE_ARITHMETIC
      USE, INTRINSIC :: IEEE_EXCEPTIONS
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      REAL*8 INPUTS(0:3,5,2),P(0:3,5),A(0:1),B(0:1)
      REAL*8 BEAMPOL(2),START,FINISH,ORDINARY_SECONDS,WRAP_SECONDS
      COMPLEX*16 GC4_SAVED
      LOGICAL RESCUED,OK,FA(SIZE(IEEE_ALL)),FB(SIZE(IEEE_ALL))
      LOGICAL HALTING,HELRESET
      INTEGER I,J,K,H,N,REPEAT,RESCUES,FAILS
      INTEGER*8 BITS_A(2),BITS_B(2)
      INTEGER POLARIZATIONS(0:5,0:5)
      CHARACTER*1024 CARD,POINTS,MODE
      COMMON/TO_BEAMPOL/BEAMPOL
      COMMON/ML5_0_HELRESET/HELRESET
      COMMON/ML5_0_BORN_BEAM_POL/POLARIZATIONS
      COMMON/AUDIT_BORN_RESCUE_COUNTS/RESCUES,FAILS

      CALL GET_COMMAND_ARGUMENT(1,CARD)
      CALL GET_COMMAND_ARGUMENT(2,POINTS)
      CALL GET_COMMAND_ARGUMENT(3,MODE)
      CALL SETPARA(TRIM(CARD))
      BEAMPOL=0D0
      OPEN(17,FILE=TRIM(POINTS),STATUS='OLD')
      READ(17,*) N
      IF(N.NE.2) STOP 11
      DO K=1,2
        DO I=1,5
          READ(17,*) INPUTS(:,I,K)
        ENDDO
      ENDDO
      CLOSE(17)
      MU_R=SQRT(ABS((INPUTS(0,1,1)+INPUTS(0,2,1))**2-
     $  SUM((INPUTS(1:3,1,1)+INPUTS(1:3,2,1))**2)))
      CALL UPDATE_AS_PARAM()

      IF(INDEX(TRIM(MODE),'trap-').EQ.1) THEN
        IF(.NOT.IEEE_SUPPORT_HALTING(IEEE_INVALID)) STOP 12
        CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
        CALL IEEE_SET_HALTING_MODE(IEEE_INVALID,.TRUE.)
        CALL IEEE_GET_HALTING_MODE(IEEE_INVALID,HALTING)
        WRITE(*,'(A,1X,A,1X,L1)') 'TRAP_START',TRIM(MODE),HALTING
        FLUSH(6)
        IF(TRIM(MODE).EQ.'trap-original') THEN
          CALL ML5_0_SMATRIXHEL_SPLITORDERS(INPUTS(:,:,2),-1,1,A)
        ELSE IF(TRIM(MODE).EQ.'trap-fallback') THEN
          CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,2),-1,1,A,
     $      RESCUED,OK)
        ELSE
          STOP 13
        ENDIF
        WRITE(*,*) 'TRAP_RETURNED_UNEXPECTEDLY'
        STOP 14
      ENDIF

      IF(TRIM(MODE).EQ.'timing') THEN
C       A Born-only overhead diagnostic, not whole-amplitude acceptance.
        P=INPUTS(:,:,1)
        DO I=1,30
          CALL ML5_0_BORN_WITH_RESCUE(P,-1,1,A,RESCUED,OK)
          IF(RESCUED.OR..NOT.OK) STOP 21
        ENDDO
        DO REPEAT=1,7
          DO J=1,2
            CALL CPU_TIME(START)
            IF(MOD(REPEAT+J,2).EQ.0) THEN
              DO I=1,20000
                CALL ML5_0_SMATRIXHEL_SPLITORDERS(P,-1,1,A)
              ENDDO
              CALL CPU_TIME(FINISH)
              ORDINARY_SECONDS=FINISH-START
            ELSE
              DO I=1,20000
                CALL ML5_0_BORN_WITH_RESCUE(P,-1,1,B,RESCUED,OK)
              ENDDO
              CALL CPU_TIME(FINISH)
              WRAP_SECONDS=FINISH-START
              IF(RESCUED.OR..NOT.OK) STOP 22
            ENDIF
          ENDDO
          BITS_A=TRANSFER(A,BITS_A)
          BITS_B=TRANSFER(B,BITS_B)
          IF(ANY(BITS_A.NE.BITS_B)) STOP 23
          WRITE(*,'(A,1X,I2,2(1X,ES24.15E3))')
     $      'BORN_ONLY_TIMING',REPEAT,ORDINARY_SECONDS,WRAP_SECONDS
        ENDDO
        STOP
      ENDIF

C     Ordinary all-helicity and selected-helicity calls retain exact bits
C     and the same exception flags, with original COMMON selection state.
      DO J=0,2
        BEAMPOL=0D0
        POLARIZATIONS=-1
        IF(J.EQ.1) BEAMPOL=(/1.35D0,-1.75D0/)
        IF(J.EQ.2) THEN
          POLARIZATIONS(0,0)=0
          POLARIZATIONS(1,0)=1
          POLARIZATIONS(1,1)=1
        ENDIF
        DO H=-1,48
          IF(H.EQ.0) CYCLE
          HELRESET=.TRUE.
          CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
          CALL ML5_0_SMATRIXHEL_SPLITORDERS(INPUTS(:,:,1),H,1,A)
          CALL IEEE_GET_FLAG(IEEE_ALL,FA)
          HELRESET=.TRUE.
          CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
          CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,1),H,1,B,
     $      RESCUED,OK)
          CALL IEEE_GET_FLAG(IEEE_ALL,FB)
          BITS_A=TRANSFER(A,BITS_A)
          BITS_B=TRANSFER(B,BITS_B)
          IF(ANY(BITS_A.NE.BITS_B).OR.RESCUED.OR..NOT.OK) STOP 31
          IF(ANY(FA.NEQV.FB)) STOP 32
          WRITE(*,'(A,2(1X,I3),2(1X,Z16.16),*(1X,L1))')
     $      'ORDINARY_PARITY',J,H,BITS_B,FB
        ENDDO
      ENDDO
      BEAMPOL=0D0
      POLARIZATIONS=-1

C     Repeated rescues must not create a separate helicity-learning cache.
      DO I=1,25
        CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
        CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,2),-1,1,B,
     $    RESCUED,OK)
        CALL IEEE_GET_FLAG(IEEE_ALL,FB)
        IF(.NOT.RESCUED.OR..NOT.OK.OR..NOT.HELRESET) STOP 41
        WRITE(*,'(A,1X,I3,2(1X,ES46.35E4),*(1X,L1))')
     $    'STRESS_RESCUE',I,B,FB
      ENDDO

C     A caller's pre-existing invalid flag is retained, but cannot make
C     an otherwise successful QP rescue look like a new arithmetic failure.
      CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
      CALL IEEE_SET_FLAG(IEEE_INVALID,.TRUE.)
      CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,2),-1,1,B,RESCUED,OK)
      CALL IEEE_GET_FLAG(IEEE_INVALID,HALTING)
      IF(.NOT.RESCUED.OR..NOT.OK.OR..NOT.HALTING) STOP 42
      WRITE(*,'(A,2(1X,ES46.35E4))') 'ENTRY_FLAG_PRESERVED',B

C     Explicit invalid input is a rejection probe, not the physical fixture.
      P=INPUTS(:,:,1)
      P(0,1)=IEEE_VALUE(0D0,IEEE_QUIET_NAN)
      CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
      CALL ML5_0_BORN_WITH_RESCUE(P,-1,1,B,RESCUED,OK)
      IF(.NOT.RESCUED.OR.OK.OR.ANY(IEEE_IS_FINITE(B))) STOP 51
      WRITE(*,*) 'INVALID_QP_REJECTED'

C     Finite QP arithmetic whose final Born overflows DP also must fail.
      GC4_SAVED=GC_4
      GC_4=DCMPLX(1D100,0D0)
      CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
      CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,1),-1,1,B,RESCUED,OK)
      GC_4=GC4_SAVED
      IF(.NOT.RESCUED.OR.OK.OR.ANY(IEEE_IS_FINITE(B))) STOP 52
      WRITE(*,*) 'QP_CONVERSION_OVERFLOW_REJECTED'
      CALL IEEE_SET_FLAG(IEEE_ALL,.FALSE.)
      CALL ML5_0_BORN_WITH_RESCUE(INPUTS(:,:,1),-1,1,B,RESCUED,OK)
      IF(RESCUED.OR..NOT.OK) STOP 53
      WRITE(*,'(A,2(1X,I5),2(1X,ES46.35E4))')
     $  'TREE_QUALIFICATION_PASS',RESCUES,FAILS,B
      END
