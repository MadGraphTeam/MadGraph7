      SUBROUTINE ML5_0_HELAS_CALLS_AMPB_3(P,NHEL,H,IC)
C     
C     Modules
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=6)
      INTEGER    NCOMB
      PARAMETER (NCOMB=96)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=54)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=2871, NLOOPGROUPS=1160, NCTAMPS=2982)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=5853)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=227,NLOOPWAVEFUNCS=4000)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      REAL*16     MP__ZERO
      PARAMETER (MP__ZERO=0.0E0_16)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=1, NSQUAREDSO=1, NAMPSO=2)
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      COMPLEX*16 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)

      LOGICAL DUMMYFALSE
      DATA DUMMYFALSE/.FALSE./
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      INTEGER I_SO
      COMMON/ML5_0_I_SO/I_SO
      INTEGER I_LIB
      COMMON/ML5_0_I_LIB/I_LIB

      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/ML5_0_AMPS/AMP
      TYPE(ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_W/W
      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,
     $ -1:NLOOPWAVEFUNCS)
      COMPLEX*16 PL(0:3,-1:NLOOPWAVEFUNCS)
      COMMON/ML5_0_WL/WL,PL

      COMPLEX*16 AMPL(3,NCTAMPS)
      COMMON/ML5_0_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

C     Counter-term amplitude(s) for loop diagram number 2394
      CALL R2_GGZ_0(W(14),W(212),W(3),R2_GGZUP,AMPL(1,2857))
C     Counter-term amplitude(s) for loop diagram number 2404
      CALL R2_GG_1_R2_GG_3_0(W(14),W(161),R2_GGQ,R2_GGT,AMPL(1,2858))
C     Counter-term amplitude(s) for loop diagram number 2405
      CALL VVV1_0(W(4),W(6),W(161),R2_3GQ,AMPL(1,2859))
C     Counter-term amplitude(s) for loop diagram number 2407
      CALL R2_GGZ_0(W(4),W(218),W(3),R2_GGZUP,AMPL(1,2860))
C     Counter-term amplitude(s) for loop diagram number 2408
      CALL R2_GGVV_0(W(4),W(5),W(217),W(3),R2_GGGZVECUP,AMPL(1,2861))
      CALL R2_GGGVA_0(W(4),W(5),W(217),W(3),R2_GGGZAXIALUP,AMPL(1,2862)
     $ )
C     Counter-term amplitude(s) for loop diagram number 2409
      CALL R2_GGZ_0(W(5),W(219),W(3),R2_GGZUP,AMPL(1,2863))
C     Counter-term amplitude(s) for loop diagram number 2411
      CALL R2_GGZ_0(W(5),W(220),W(3),R2_GGZUP,AMPL(1,2864))
C     Counter-term amplitude(s) for loop diagram number 2412
      CALL R2_GGZ_0(W(4),W(221),W(3),R2_GGZUP,AMPL(1,2865))
C     Counter-term amplitude(s) for loop diagram number 2413
      CALL R2_GGZ_0(W(16),W(217),W(3),R2_GGZUP,AMPL(1,2866))
C     Counter-term amplitude(s) for loop diagram number 2423
      CALL R2_GG_1_R2_GG_3_0(W(16),W(163),R2_GGQ,R2_GGT,AMPL(1,2867))
C     Counter-term amplitude(s) for loop diagram number 2424
      CALL VVV1_0(W(4),W(5),W(163),R2_3GQ,AMPL(1,2868))
C     Counter-term amplitude(s) for loop diagram number 2426
      CALL R2_GGZ_0(W(6),W(222),W(3),R2_GGZUP,AMPL(1,2869))
C     Counter-term amplitude(s) for loop diagram number 2427
      CALL R2_GGZ_0(W(6),W(223),W(3),R2_GGZUP,AMPL(1,2870))
C     Counter-term amplitude(s) for loop diagram number 2430
      CALL R2_GGZ_0(W(5),W(224),W(3),R2_GGZUP,AMPL(1,2871))
C     Counter-term amplitude(s) for loop diagram number 2431
      CALL R2_GGZ_0(W(5),W(225),W(3),R2_GGZUP,AMPL(1,2872))
C     Counter-term amplitude(s) for loop diagram number 2434
      CALL R2_GGZ_0(W(4),W(226),W(3),R2_GGZUP,AMPL(1,2873))
C     Counter-term amplitude(s) for loop diagram number 2435
      CALL R2_GGZ_0(W(4),W(227),W(3),R2_GGZUP,AMPL(1,2874))
C     At this point, all CT amps needed for (QCD=8), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

