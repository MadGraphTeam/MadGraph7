      SUBROUTINE ML5_0_MP_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NCOMB
      PARAMETER (NCOMB=48)

      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=8)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=224, NLOOPGROUPS=116, NCTAMPS=278)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=502)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35,NLOOPWAVEFUNCS=388)
      REAL*16     ZERO
      PARAMETER (ZERO=0.0E0_16)
      COMPLEX*32     IZERO
      PARAMETER (IZERO=CMPLX(0.0E0_16,0.0E0_16,KIND=16))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=1, NSQUAREDSO=1, NAMPSO=2)
C     
C     ARGUMENTS
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      COMPLEX*32 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'mp_coupl_same_name.inc'

      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      COMPLEX*32 AMP(NBORNAMPS)
      COMMON/ML5_0_MP_AMPS/AMP
      TYPE(MP_ALOHA) W(NWAVEFUNCS)
      COMMON/ML5_0_MP_W/W

      COMPLEX*32 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,
     $ -1:NLOOPWAVEFUNCS)
      COMPLEX*32 PL(0:3,-1:NLOOPWAVEFUNCS)
      COMMON/ML5_0_MP_WL/WL,PL

      COMPLEX*32 AMPL(3,NCTAMPS)
      COMMON/ML5_0_MP_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.MP_CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL MP_IXXXXX(P(0,1),ZERO,NHEL(1),+1, FLAVOR(1),W(1))
      CALL MP_OXXXXX(P(0,2),ZERO,NHEL(2),-1, FLAVOR(2),W(2))
      CALL MP_VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL MP_VXXXXX(P(0,4),ZERO,NHEL(4),+1,W(4))
      CALL MP_VXXXXX(P(0,5),ZERO,NHEL(5),+1,W(5))
      CALL MP_FFV2_5_2(W(1),W(3),GC_22,GC_23,ZERO,ZERO,W(6))
      CALL MP_FFV1_1(W(2),W(4),GC_5,ZERO,ZERO,W(7))
C     Amplitude(s) for born diagram with ID 1
      CALL MP_FFV1_0(W(6),W(7),W(5),GC_5,AMP(1))
      CALL MP_FFV1_1(W(2),W(5),GC_5,ZERO,ZERO,W(8))
C     Amplitude(s) for born diagram with ID 2
      CALL MP_FFV1_0(W(6),W(8),W(4),GC_5,AMP(2))
      CALL MP_VVV1P0_1(W(4),W(5),GC_4,ZERO,ZERO,W(9))
C     Amplitude(s) for born diagram with ID 3
      CALL MP_FFV1_0(W(6),W(2),W(9),GC_5,AMP(3))
      CALL MP_FFV1_2(W(1),W(4),GC_5,ZERO,ZERO,W(10))
      CALL MP_FFV2_5_1(W(2),W(3),GC_22,GC_23,ZERO,ZERO,W(11))
C     Amplitude(s) for born diagram with ID 4
      CALL MP_FFV1_0(W(10),W(11),W(5),GC_5,AMP(4))
C     Amplitude(s) for born diagram with ID 5
      CALL MP_FFV2_5_0(W(10),W(8),W(3),GC_22,GC_23,AMP(5))
      CALL MP_FFV1_2(W(1),W(5),GC_5,ZERO,ZERO,W(12))
C     Amplitude(s) for born diagram with ID 6
      CALL MP_FFV1_0(W(12),W(11),W(4),GC_5,AMP(6))
C     Amplitude(s) for born diagram with ID 7
      CALL MP_FFV2_5_0(W(12),W(7),W(3),GC_22,GC_23,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL MP_FFV1_0(W(1),W(11),W(9),GC_5,AMP(8))
      CALL MP_FFV1_2(W(6),W(5),GC_5,ZERO,ZERO,W(13))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL MP_R2_QQ_1_0(W(13),W(7),R2_QQQ,AMPL(1,1))
      CALL MP_FFV1_1(W(7),W(5),GC_5,ZERO,ZERO,W(14))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL MP_R2_QQ_1_0(W(6),W(14),R2_QQQ,AMPL(1,2))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL MP_FFV1_0(W(6),W(7),W(5),R2_GQQ,AMPL(1,3))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,4))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,5))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,6))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,7))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,8))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,9))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQG_1EPS,AMPL(2,10))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQB,AMPL(1,11))
      CALL MP_FFV1_0(W(6),W(7),W(5),UV_GQQT,AMPL(1,12))
      CALL MP_FFV1_2(W(6),W(4),GC_5,ZERO,ZERO,W(15))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL MP_R2_QQ_1_0(W(15),W(8),R2_QQQ,AMPL(1,13))
      CALL MP_FFV1_1(W(8),W(4),GC_5,ZERO,ZERO,W(16))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL MP_R2_QQ_1_0(W(6),W(16),R2_QQQ,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 15
      CALL MP_FFV1_0(W(6),W(8),W(4),R2_GQQ,AMPL(1,15))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,16))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,17))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,18))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,19))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,20))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,21))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQG_1EPS,AMPL(2,22))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQB,AMPL(1,23))
      CALL MP_FFV1_0(W(6),W(8),W(4),UV_GQQT,AMPL(1,24))
C     Counter-term amplitude(s) for loop diagram number 17
      CALL MP_FFV1_0(W(15),W(2),W(5),R2_GQQ,AMPL(1,25))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,26))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,27))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,28))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,29))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,30))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,31))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQG_1EPS,AMPL(2,32))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQB,AMPL(1,33))
      CALL MP_FFV1_0(W(15),W(2),W(5),UV_GQQT,AMPL(1,34))
C     Counter-term amplitude(s) for loop diagram number 18
      CALL MP_FFV1_0(W(13),W(2),W(4),R2_GQQ,AMPL(1,35))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,36))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,37))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,38))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,39))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,40))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,41))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQG_1EPS,AMPL(2,42))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQB,AMPL(1,43))
      CALL MP_FFV1_0(W(13),W(2),W(4),UV_GQQT,AMPL(1,44))
C     Counter-term amplitude(s) for loop diagram number 21
      CALL MP_FFV1_0(W(6),W(2),W(9),R2_GQQ,AMPL(1,45))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,46))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,47))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,48))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,49))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,50))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,51))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQG_1EPS,AMPL(2,52))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQB,AMPL(1,53))
      CALL MP_FFV1_0(W(6),W(2),W(9),UV_GQQT,AMPL(1,54))
      CALL MP_FFV1P0_3(W(6),W(2),GC_5,ZERO,ZERO,W(17))
C     Counter-term amplitude(s) for loop diagram number 30
      CALL MP_R2_GG_1_R2_GG_2_0(W(9),W(17),R2_GGG_1,R2_GGG_2,AMPL(1,55)
     $ )
      CALL MP_FFV1_1(W(2),W(9),GC_5,ZERO,ZERO,W(18))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL MP_R2_QQ_1_0(W(6),W(18),R2_QQQ,AMPL(1,56))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GG,AMPL(1,57))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,58))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,59))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,60))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,61))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,62))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,63))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GG_1EPS,AMPL(2,64))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GB,AMPL(1,65))
      CALL MP_VVV1_0(W(4),W(5),W(17),UV_3GT,AMPL(1,66))
      CALL MP_FFV1_2(W(10),W(5),GC_5,ZERO,ZERO,W(19))
C     Counter-term amplitude(s) for loop diagram number 36
      CALL MP_R2_QQ_1_0(W(19),W(11),R2_QQQ,AMPL(1,67))
      CALL MP_FFV1_1(W(11),W(5),GC_5,ZERO,ZERO,W(20))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL MP_R2_QQ_1_0(W(10),W(20),R2_QQQ,AMPL(1,68))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL MP_FFV1_0(W(10),W(11),W(5),R2_GQQ,AMPL(1,69))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,70))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,71))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,72))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,73))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,74))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,75))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQG_1EPS,AMPL(2,76))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQB,AMPL(1,77))
      CALL MP_FFV1_0(W(10),W(11),W(5),UV_GQQT,AMPL(1,78))
      CALL MP_FFV2_5_2(W(10),W(3),GC_22,GC_23,ZERO,ZERO,W(21))
C     Counter-term amplitude(s) for loop diagram number 40
      CALL MP_R2_QQ_1_0(W(21),W(8),R2_QQQ,AMPL(1,79))
      CALL MP_FFV2_5_1(W(8),W(3),GC_22,GC_23,ZERO,ZERO,W(22))
C     Counter-term amplitude(s) for loop diagram number 41
      CALL MP_R2_QQ_1_0(W(10),W(22),R2_QQQ,AMPL(1,80))
C     Counter-term amplitude(s) for loop diagram number 42
      CALL MP_FFV2_5_0(W(10),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,81))
C     Counter-term amplitude(s) for loop diagram number 43
      CALL MP_FFV1_0(W(21),W(2),W(5),R2_GQQ,AMPL(1,82))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,83))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,84))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,85))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,86))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,87))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,88))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQG_1EPS,AMPL(2,89))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQB,AMPL(1,90))
      CALL MP_FFV1_0(W(21),W(2),W(5),UV_GQQT,AMPL(1,91))
C     Counter-term amplitude(s) for loop diagram number 44
      CALL MP_FFV2_5_0(W(19),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,92))
      CALL MP_FFV1_2(W(12),W(4),GC_5,ZERO,ZERO,W(23))
C     Counter-term amplitude(s) for loop diagram number 49
      CALL MP_R2_QQ_1_0(W(23),W(11),R2_QQQ,AMPL(1,93))
      CALL MP_FFV1_1(W(11),W(4),GC_5,ZERO,ZERO,W(24))
C     Counter-term amplitude(s) for loop diagram number 50
      CALL MP_R2_QQ_1_0(W(12),W(24),R2_QQQ,AMPL(1,94))
C     Counter-term amplitude(s) for loop diagram number 51
      CALL MP_FFV1_0(W(12),W(11),W(4),R2_GQQ,AMPL(1,95))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,96))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,97))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,98))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,99))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,100))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,101))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQG_1EPS,AMPL(2,102))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQB,AMPL(1,103))
      CALL MP_FFV1_0(W(12),W(11),W(4),UV_GQQT,AMPL(1,104))
      CALL MP_FFV2_5_2(W(12),W(3),GC_22,GC_23,ZERO,ZERO,W(25))
C     Counter-term amplitude(s) for loop diagram number 53
      CALL MP_R2_QQ_1_0(W(25),W(7),R2_QQQ,AMPL(1,105))
      CALL MP_FFV2_5_1(W(7),W(3),GC_22,GC_23,ZERO,ZERO,W(26))
C     Counter-term amplitude(s) for loop diagram number 54
      CALL MP_R2_QQ_1_0(W(12),W(26),R2_QQQ,AMPL(1,106))
C     Counter-term amplitude(s) for loop diagram number 55
      CALL MP_FFV2_5_0(W(12),W(7),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,107))
C     Counter-term amplitude(s) for loop diagram number 56
      CALL MP_FFV1_0(W(25),W(2),W(4),R2_GQQ,AMPL(1,108))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,109))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,110))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,111))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,112))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,113))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,114))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQG_1EPS,AMPL(2,115))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQB,AMPL(1,116))
      CALL MP_FFV1_0(W(25),W(2),W(4),UV_GQQT,AMPL(1,117))
C     Counter-term amplitude(s) for loop diagram number 57
      CALL MP_FFV2_5_0(W(23),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,118))
C     Counter-term amplitude(s) for loop diagram number 62
      CALL MP_FFV1_0(W(1),W(20),W(4),R2_GQQ,AMPL(1,119))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,120))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,121))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,122))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,123))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,124))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,125))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQG_1EPS,AMPL(2,126))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQB,AMPL(1,127))
      CALL MP_FFV1_0(W(1),W(20),W(4),UV_GQQT,AMPL(1,128))
C     Counter-term amplitude(s) for loop diagram number 64
      CALL MP_FFV1_0(W(1),W(24),W(5),R2_GQQ,AMPL(1,129))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,130))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,131))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,132))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,133))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,134))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,135))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQG_1EPS,AMPL(2,136))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQB,AMPL(1,137))
      CALL MP_FFV1_0(W(1),W(24),W(5),UV_GQQT,AMPL(1,138))
C     Counter-term amplitude(s) for loop diagram number 66
      CALL MP_FFV1_0(W(1),W(11),W(9),R2_GQQ,AMPL(1,139))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,140))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,141))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,142))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,143))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,144))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,145))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQG_1EPS,AMPL(2,146))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQB,AMPL(1,147))
      CALL MP_FFV1_0(W(1),W(11),W(9),UV_GQQT,AMPL(1,148))
C     Counter-term amplitude(s) for loop diagram number 75
      CALL MP_FFV2_5_0(W(1),W(14),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,149))
C     Counter-term amplitude(s) for loop diagram number 77
      CALL MP_FFV1_0(W(1),W(26),W(5),R2_GQQ,AMPL(1,150))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,151))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,152))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,153))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,154))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,155))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,156))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQG_1EPS,AMPL(2,157))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQB,AMPL(1,158))
      CALL MP_FFV1_0(W(1),W(26),W(5),UV_GQQT,AMPL(1,159))
C     Counter-term amplitude(s) for loop diagram number 81
      CALL MP_FFV2_5_0(W(1),W(16),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,160))
C     Counter-term amplitude(s) for loop diagram number 83
      CALL MP_FFV1_0(W(1),W(22),W(4),R2_GQQ,AMPL(1,161))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,162))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,163))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,164))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,165))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,166))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,167))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQG_1EPS,AMPL(2,168))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQB,AMPL(1,169))
      CALL MP_FFV1_0(W(1),W(22),W(4),UV_GQQT,AMPL(1,170))
C     Counter-term amplitude(s) for loop diagram number 95
      CALL MP_FFV2_5_0(W(1),W(18),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,171))
      CALL MP_FFV1P0_3(W(1),W(11),GC_5,ZERO,ZERO,W(27))
C     Counter-term amplitude(s) for loop diagram number 104
      CALL MP_R2_GG_1_R2_GG_2_0(W(9),W(27),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,172))
      CALL MP_FFV1_2(W(1),W(9),GC_5,ZERO,ZERO,W(28))
C     Counter-term amplitude(s) for loop diagram number 105
      CALL MP_R2_QQ_1_0(W(28),W(11),R2_QQQ,AMPL(1,173))
C     Counter-term amplitude(s) for loop diagram number 106
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GG,AMPL(1,174))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,175))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,176))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,177))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,178))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,179))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,180))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GG_1EPS,AMPL(2,181))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GB,AMPL(1,182))
      CALL MP_VVV1_0(W(4),W(5),W(27),UV_3GT,AMPL(1,183))
C     Counter-term amplitude(s) for loop diagram number 110
      CALL MP_FFV2_5_0(W(28),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,184))
      CALL MP_FFV1P0_3(W(1),W(2),GC_5,ZERO,ZERO,W(29))
      CALL MP_VVV1P0_1(W(4),W(29),GC_4,ZERO,ZERO,W(30))
C     Counter-term amplitude(s) for loop diagram number 117
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,185))
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,186))
      CALL MP_VVV1P0_1(W(5),W(29),GC_4,ZERO,ZERO,W(31))
C     Counter-term amplitude(s) for loop diagram number 118
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,187))
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,188))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,189)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,190))
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,191)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,192))
C     Counter-term amplitude(s) for loop diagram number 121
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,193))
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,194))
C     Counter-term amplitude(s) for loop diagram number 129
      CALL MP_R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,195))
      CALL MP_R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,196))
      CALL MP_R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,197))
      CALL MP_R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,198))
C     Counter-term amplitude(s) for loop diagram number 130
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,199))
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,200))
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,201))
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,202))
      CALL MP_FFV1P0_3(W(10),W(2),GC_5,ZERO,ZERO,W(32))
C     Counter-term amplitude(s) for loop diagram number 132
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,203))
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,204))
      CALL MP_FFV1P0_3(W(12),W(2),GC_5,ZERO,ZERO,W(33))
C     Counter-term amplitude(s) for loop diagram number 134
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,205))
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,206))
C     Counter-term amplitude(s) for loop diagram number 136
      CALL MP_R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,207))
      CALL MP_R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,208))
      CALL MP_R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,209))
      CALL MP_R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,210))
C     Counter-term amplitude(s) for loop diagram number 137
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,211))
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,212))
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,213))
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,214))
      CALL MP_FFV1P0_3(W(1),W(7),GC_5,ZERO,ZERO,W(34))
C     Counter-term amplitude(s) for loop diagram number 139
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,215))
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,216))
      CALL MP_FFV1P0_3(W(1),W(8),GC_5,ZERO,ZERO,W(35))
C     Counter-term amplitude(s) for loop diagram number 141
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,217))
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,218))
C     Counter-term amplitude(s) for loop diagram number 143
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,219))
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,220))
C     Counter-term amplitude(s) for loop diagram number 144
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,221))
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,222))
C     Counter-term amplitude(s) for loop diagram number 145
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,223))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,224))
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,225))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,226))
C     Counter-term amplitude(s) for loop diagram number 147
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,227))
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,228))
C     Counter-term amplitude(s) for loop diagram number 155
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,229))
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,230))
C     Counter-term amplitude(s) for loop diagram number 157
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,231))
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,232))
C     Counter-term amplitude(s) for loop diagram number 159
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,233))
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,234))
C     Counter-term amplitude(s) for loop diagram number 161
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,235))
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,236))
C     Counter-term amplitude(s) for loop diagram number 163
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,237))
C     Counter-term amplitude(s) for loop diagram number 164
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,238))
C     Counter-term amplitude(s) for loop diagram number 165
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,239)
     $ )
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,240))
C     Counter-term amplitude(s) for loop diagram number 167
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,241))
C     Counter-term amplitude(s) for loop diagram number 175
      CALL MP_R2_GG_1_R2_GG_3_0(W(9),W(17),R2_GGQ,R2_GGB,AMPL(1,242))
C     Counter-term amplitude(s) for loop diagram number 176
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,243))
C     Counter-term amplitude(s) for loop diagram number 178
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,244))
C     Counter-term amplitude(s) for loop diagram number 180
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,245))
C     Counter-term amplitude(s) for loop diagram number 182
      CALL MP_R2_GG_1_R2_GG_3_0(W(9),W(27),R2_GGQ,R2_GGB,AMPL(1,246))
C     Counter-term amplitude(s) for loop diagram number 183
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,247))
C     Counter-term amplitude(s) for loop diagram number 185
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,248))
C     Counter-term amplitude(s) for loop diagram number 187
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,249))
C     Counter-term amplitude(s) for loop diagram number 189
      CALL MP_R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,250))
C     Counter-term amplitude(s) for loop diagram number 190
      CALL MP_R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,251))
C     Counter-term amplitude(s) for loop diagram number 191
      CALL MP_R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,252))
      CALL MP_R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1
     $ ,253))
C     Counter-term amplitude(s) for loop diagram number 193
      CALL MP_R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,254))
C     Counter-term amplitude(s) for loop diagram number 201
      CALL MP_R2_GG_1_R2_GG_3_0(W(9),W(17),R2_GGQ,R2_GGT,AMPL(1,255))
C     Counter-term amplitude(s) for loop diagram number 202
      CALL MP_VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,256))
C     Counter-term amplitude(s) for loop diagram number 204
      CALL MP_R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,257))
C     Counter-term amplitude(s) for loop diagram number 206
      CALL MP_R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,258))
C     Counter-term amplitude(s) for loop diagram number 208
      CALL MP_R2_GG_1_R2_GG_3_0(W(9),W(27),R2_GGQ,R2_GGT,AMPL(1,259))
C     Counter-term amplitude(s) for loop diagram number 209
      CALL MP_VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,260))
C     Counter-term amplitude(s) for loop diagram number 211
      CALL MP_R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,261))
C     Counter-term amplitude(s) for loop diagram number 213
      CALL MP_R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,262))
C     At this point, all CT amps needed for (QCD=6), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      MP_CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

