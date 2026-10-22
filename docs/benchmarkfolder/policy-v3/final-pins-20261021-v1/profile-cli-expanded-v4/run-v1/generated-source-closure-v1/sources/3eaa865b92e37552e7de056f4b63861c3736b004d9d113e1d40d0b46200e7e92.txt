      SUBROUTINE ML5_0_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
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

      CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1, FLAVOR(1),W(1))
      CALL OXXXXX(P(0,2),ZERO,NHEL(2),-1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1,W(4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1,W(5))
      CALL FFV2_5_2(W(1),W(3),GC_22,GC_23,ZERO,ZERO,W(6))
      CALL FFV1_1(W(2),W(4),GC_5,ZERO,ZERO,W(7))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(6),W(7),W(5),GC_5,AMP(1))
      CALL FFV1_1(W(2),W(5),GC_5,ZERO,ZERO,W(8))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(6),W(8),W(4),GC_5,AMP(2))
      CALL VVV1P0_1(W(4),W(5),GC_4,ZERO,ZERO,W(9))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV1_0(W(6),W(2),W(9),GC_5,AMP(3))
      CALL FFV1_2(W(1),W(4),GC_5,ZERO,ZERO,W(10))
      CALL FFV2_5_1(W(2),W(3),GC_22,GC_23,ZERO,ZERO,W(11))
C     Amplitude(s) for born diagram with ID 4
      CALL FFV1_0(W(10),W(11),W(5),GC_5,AMP(4))
C     Amplitude(s) for born diagram with ID 5
      CALL FFV2_5_0(W(10),W(8),W(3),GC_22,GC_23,AMP(5))
      CALL FFV1_2(W(1),W(5),GC_5,ZERO,ZERO,W(12))
C     Amplitude(s) for born diagram with ID 6
      CALL FFV1_0(W(12),W(11),W(4),GC_5,AMP(6))
C     Amplitude(s) for born diagram with ID 7
      CALL FFV2_5_0(W(12),W(7),W(3),GC_22,GC_23,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL FFV1_0(W(1),W(11),W(9),GC_5,AMP(8))
      CALL FFV1_2(W(6),W(5),GC_5,ZERO,ZERO,W(13))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL R2_QQ_1_0(W(13),W(7),R2_QQQ,AMPL(1,1))
      CALL FFV1_1(W(7),W(5),GC_5,ZERO,ZERO,W(14))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL R2_QQ_1_0(W(6),W(14),R2_QQQ,AMPL(1,2))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL FFV1_0(W(6),W(7),W(5),R2_GQQ,AMPL(1,3))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,4))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,5))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,6))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,7))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,8))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB_1EPS,AMPL(2,9))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQG_1EPS,AMPL(2,10))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQB,AMPL(1,11))
      CALL FFV1_0(W(6),W(7),W(5),UV_GQQT,AMPL(1,12))
      CALL FFV1_2(W(6),W(4),GC_5,ZERO,ZERO,W(15))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL R2_QQ_1_0(W(15),W(8),R2_QQQ,AMPL(1,13))
      CALL FFV1_1(W(8),W(4),GC_5,ZERO,ZERO,W(16))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL R2_QQ_1_0(W(6),W(16),R2_QQQ,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 15
      CALL FFV1_0(W(6),W(8),W(4),R2_GQQ,AMPL(1,15))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,16))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,17))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,18))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,19))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,20))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB_1EPS,AMPL(2,21))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQG_1EPS,AMPL(2,22))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQB,AMPL(1,23))
      CALL FFV1_0(W(6),W(8),W(4),UV_GQQT,AMPL(1,24))
C     Counter-term amplitude(s) for loop diagram number 17
      CALL FFV1_0(W(15),W(2),W(5),R2_GQQ,AMPL(1,25))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,26))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,27))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,28))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,29))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,30))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB_1EPS,AMPL(2,31))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQG_1EPS,AMPL(2,32))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQB,AMPL(1,33))
      CALL FFV1_0(W(15),W(2),W(5),UV_GQQT,AMPL(1,34))
C     Counter-term amplitude(s) for loop diagram number 18
      CALL FFV1_0(W(13),W(2),W(4),R2_GQQ,AMPL(1,35))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,36))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,37))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,38))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,39))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,40))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB_1EPS,AMPL(2,41))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQG_1EPS,AMPL(2,42))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQB,AMPL(1,43))
      CALL FFV1_0(W(13),W(2),W(4),UV_GQQT,AMPL(1,44))
C     Counter-term amplitude(s) for loop diagram number 21
      CALL FFV1_0(W(6),W(2),W(9),R2_GQQ,AMPL(1,45))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,46))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,47))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,48))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,49))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,50))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB_1EPS,AMPL(2,51))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQG_1EPS,AMPL(2,52))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQB,AMPL(1,53))
      CALL FFV1_0(W(6),W(2),W(9),UV_GQQT,AMPL(1,54))
      CALL FFV1P0_3(W(6),W(2),GC_5,ZERO,ZERO,W(17))
C     Counter-term amplitude(s) for loop diagram number 30
      CALL R2_GG_1_R2_GG_2_0(W(9),W(17),R2_GGG_1,R2_GGG_2,AMPL(1,55))
      CALL FFV1_1(W(2),W(9),GC_5,ZERO,ZERO,W(18))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL R2_QQ_1_0(W(6),W(18),R2_QQQ,AMPL(1,56))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL VVV1_0(W(4),W(5),W(17),R2_3GG,AMPL(1,57))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,58))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,59))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,60))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,61))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,62))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB_1EPS,AMPL(2,63))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GG_1EPS,AMPL(2,64))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GB,AMPL(1,65))
      CALL VVV1_0(W(4),W(5),W(17),UV_3GT,AMPL(1,66))
      CALL FFV1_2(W(10),W(5),GC_5,ZERO,ZERO,W(19))
C     Counter-term amplitude(s) for loop diagram number 36
      CALL R2_QQ_1_0(W(19),W(11),R2_QQQ,AMPL(1,67))
      CALL FFV1_1(W(11),W(5),GC_5,ZERO,ZERO,W(20))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL R2_QQ_1_0(W(10),W(20),R2_QQQ,AMPL(1,68))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL FFV1_0(W(10),W(11),W(5),R2_GQQ,AMPL(1,69))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,70))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,71))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,72))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,73))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,74))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB_1EPS,AMPL(2,75))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQG_1EPS,AMPL(2,76))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQB,AMPL(1,77))
      CALL FFV1_0(W(10),W(11),W(5),UV_GQQT,AMPL(1,78))
      CALL FFV2_5_2(W(10),W(3),GC_22,GC_23,ZERO,ZERO,W(21))
C     Counter-term amplitude(s) for loop diagram number 40
      CALL R2_QQ_1_0(W(21),W(8),R2_QQQ,AMPL(1,79))
      CALL FFV2_5_1(W(8),W(3),GC_22,GC_23,ZERO,ZERO,W(22))
C     Counter-term amplitude(s) for loop diagram number 41
      CALL R2_QQ_1_0(W(10),W(22),R2_QQQ,AMPL(1,80))
C     Counter-term amplitude(s) for loop diagram number 42
      CALL FFV2_5_0(W(10),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,81))
C     Counter-term amplitude(s) for loop diagram number 43
      CALL FFV1_0(W(21),W(2),W(5),R2_GQQ,AMPL(1,82))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,83))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,84))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,85))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,86))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,87))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB_1EPS,AMPL(2,88))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQG_1EPS,AMPL(2,89))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQB,AMPL(1,90))
      CALL FFV1_0(W(21),W(2),W(5),UV_GQQT,AMPL(1,91))
C     Counter-term amplitude(s) for loop diagram number 44
      CALL FFV2_5_0(W(19),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,92))
      CALL FFV1_2(W(12),W(4),GC_5,ZERO,ZERO,W(23))
C     Counter-term amplitude(s) for loop diagram number 49
      CALL R2_QQ_1_0(W(23),W(11),R2_QQQ,AMPL(1,93))
      CALL FFV1_1(W(11),W(4),GC_5,ZERO,ZERO,W(24))
C     Counter-term amplitude(s) for loop diagram number 50
      CALL R2_QQ_1_0(W(12),W(24),R2_QQQ,AMPL(1,94))
C     Counter-term amplitude(s) for loop diagram number 51
      CALL FFV1_0(W(12),W(11),W(4),R2_GQQ,AMPL(1,95))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,96))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,97))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,98))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,99))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,100))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB_1EPS,AMPL(2,101))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQG_1EPS,AMPL(2,102))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQB,AMPL(1,103))
      CALL FFV1_0(W(12),W(11),W(4),UV_GQQT,AMPL(1,104))
      CALL FFV2_5_2(W(12),W(3),GC_22,GC_23,ZERO,ZERO,W(25))
C     Counter-term amplitude(s) for loop diagram number 53
      CALL R2_QQ_1_0(W(25),W(7),R2_QQQ,AMPL(1,105))
      CALL FFV2_5_1(W(7),W(3),GC_22,GC_23,ZERO,ZERO,W(26))
C     Counter-term amplitude(s) for loop diagram number 54
      CALL R2_QQ_1_0(W(12),W(26),R2_QQQ,AMPL(1,106))
C     Counter-term amplitude(s) for loop diagram number 55
      CALL FFV2_5_0(W(12),W(7),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,107))
C     Counter-term amplitude(s) for loop diagram number 56
      CALL FFV1_0(W(25),W(2),W(4),R2_GQQ,AMPL(1,108))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,109))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,110))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,111))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,112))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,113))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB_1EPS,AMPL(2,114))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQG_1EPS,AMPL(2,115))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQB,AMPL(1,116))
      CALL FFV1_0(W(25),W(2),W(4),UV_GQQT,AMPL(1,117))
C     Counter-term amplitude(s) for loop diagram number 57
      CALL FFV2_5_0(W(23),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,118))
C     Counter-term amplitude(s) for loop diagram number 62
      CALL FFV1_0(W(1),W(20),W(4),R2_GQQ,AMPL(1,119))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,120))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,121))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,122))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,123))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,124))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB_1EPS,AMPL(2,125))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQG_1EPS,AMPL(2,126))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQB,AMPL(1,127))
      CALL FFV1_0(W(1),W(20),W(4),UV_GQQT,AMPL(1,128))
C     Counter-term amplitude(s) for loop diagram number 64
      CALL FFV1_0(W(1),W(24),W(5),R2_GQQ,AMPL(1,129))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,130))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,131))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,132))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,133))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,134))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB_1EPS,AMPL(2,135))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQG_1EPS,AMPL(2,136))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQB,AMPL(1,137))
      CALL FFV1_0(W(1),W(24),W(5),UV_GQQT,AMPL(1,138))
C     Counter-term amplitude(s) for loop diagram number 66
      CALL FFV1_0(W(1),W(11),W(9),R2_GQQ,AMPL(1,139))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,140))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,141))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,142))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,143))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,144))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB_1EPS,AMPL(2,145))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQG_1EPS,AMPL(2,146))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQB,AMPL(1,147))
      CALL FFV1_0(W(1),W(11),W(9),UV_GQQT,AMPL(1,148))
C     Counter-term amplitude(s) for loop diagram number 75
      CALL FFV2_5_0(W(1),W(14),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,149))
C     Counter-term amplitude(s) for loop diagram number 77
      CALL FFV1_0(W(1),W(26),W(5),R2_GQQ,AMPL(1,150))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,151))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,152))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,153))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,154))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,155))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB_1EPS,AMPL(2,156))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQG_1EPS,AMPL(2,157))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQB,AMPL(1,158))
      CALL FFV1_0(W(1),W(26),W(5),UV_GQQT,AMPL(1,159))
C     Counter-term amplitude(s) for loop diagram number 81
      CALL FFV2_5_0(W(1),W(16),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,160))
C     Counter-term amplitude(s) for loop diagram number 83
      CALL FFV1_0(W(1),W(22),W(4),R2_GQQ,AMPL(1,161))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,162))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,163))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,164))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,165))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,166))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB_1EPS,AMPL(2,167))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQG_1EPS,AMPL(2,168))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQB,AMPL(1,169))
      CALL FFV1_0(W(1),W(22),W(4),UV_GQQT,AMPL(1,170))
C     Counter-term amplitude(s) for loop diagram number 95
      CALL FFV2_5_0(W(1),W(18),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,171))
      CALL FFV1P0_3(W(1),W(11),GC_5,ZERO,ZERO,W(27))
C     Counter-term amplitude(s) for loop diagram number 104
      CALL R2_GG_1_R2_GG_2_0(W(9),W(27),R2_GGG_1,R2_GGG_2,AMPL(1,172))
      CALL FFV1_2(W(1),W(9),GC_5,ZERO,ZERO,W(28))
C     Counter-term amplitude(s) for loop diagram number 105
      CALL R2_QQ_1_0(W(28),W(11),R2_QQQ,AMPL(1,173))
C     Counter-term amplitude(s) for loop diagram number 106
      CALL VVV1_0(W(4),W(5),W(27),R2_3GG,AMPL(1,174))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,175))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,176))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,177))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,178))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,179))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB_1EPS,AMPL(2,180))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GG_1EPS,AMPL(2,181))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GB,AMPL(1,182))
      CALL VVV1_0(W(4),W(5),W(27),UV_3GT,AMPL(1,183))
C     Counter-term amplitude(s) for loop diagram number 110
      CALL FFV2_5_0(W(28),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,184))
      CALL FFV1P0_3(W(1),W(2),GC_5,ZERO,ZERO,W(29))
      CALL VVV1P0_1(W(4),W(29),GC_4,ZERO,ZERO,W(30))
C     Counter-term amplitude(s) for loop diagram number 117
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,185))
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,186))
      CALL VVV1P0_1(W(5),W(29),GC_4,ZERO,ZERO,W(31))
C     Counter-term amplitude(s) for loop diagram number 118
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,187))
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,188))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,189))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1,190)
     $ )
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,191))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1,192)
     $ )
C     Counter-term amplitude(s) for loop diagram number 121
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,193))
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,194))
C     Counter-term amplitude(s) for loop diagram number 129
      CALL R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,195))
      CALL R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,196))
      CALL R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,197))
      CALL R2_GG_1_0(W(9),W(17),R2_GGQ,AMPL(1,198))
C     Counter-term amplitude(s) for loop diagram number 130
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,199))
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,200))
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,201))
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,202))
      CALL FFV1P0_3(W(10),W(2),GC_5,ZERO,ZERO,W(32))
C     Counter-term amplitude(s) for loop diagram number 132
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,203))
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,204))
      CALL FFV1P0_3(W(12),W(2),GC_5,ZERO,ZERO,W(33))
C     Counter-term amplitude(s) for loop diagram number 134
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,205))
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,206))
C     Counter-term amplitude(s) for loop diagram number 136
      CALL R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,207))
      CALL R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,208))
      CALL R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,209))
      CALL R2_GG_1_0(W(9),W(27),R2_GGQ,AMPL(1,210))
C     Counter-term amplitude(s) for loop diagram number 137
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,211))
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,212))
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,213))
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,214))
      CALL FFV1P0_3(W(1),W(7),GC_5,ZERO,ZERO,W(34))
C     Counter-term amplitude(s) for loop diagram number 139
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,215))
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,216))
      CALL FFV1P0_3(W(1),W(8),GC_5,ZERO,ZERO,W(35))
C     Counter-term amplitude(s) for loop diagram number 141
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,217))
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,218))
C     Counter-term amplitude(s) for loop diagram number 143
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,219))
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,220))
C     Counter-term amplitude(s) for loop diagram number 144
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,221))
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,222))
C     Counter-term amplitude(s) for loop diagram number 145
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,223))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1,224))
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,225))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1,226))
C     Counter-term amplitude(s) for loop diagram number 147
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,227))
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,228))
C     Counter-term amplitude(s) for loop diagram number 155
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,229))
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,230))
C     Counter-term amplitude(s) for loop diagram number 157
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,231))
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,232))
C     Counter-term amplitude(s) for loop diagram number 159
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,233))
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,234))
C     Counter-term amplitude(s) for loop diagram number 161
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,235))
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,236))
C     Counter-term amplitude(s) for loop diagram number 163
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZDOWN,AMPL(1,237))
C     Counter-term amplitude(s) for loop diagram number 164
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZDOWN,AMPL(1,238))
C     Counter-term amplitude(s) for loop diagram number 165
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECDOWN,AMPL(1,239))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALDOWN,AMPL(1,240)
     $ )
C     Counter-term amplitude(s) for loop diagram number 167
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZDOWN,AMPL(1,241))
C     Counter-term amplitude(s) for loop diagram number 175
      CALL R2_GG_1_R2_GG_3_0(W(9),W(17),R2_GGQ,R2_GGB,AMPL(1,242))
C     Counter-term amplitude(s) for loop diagram number 176
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,243))
C     Counter-term amplitude(s) for loop diagram number 178
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZDOWN,AMPL(1,244))
C     Counter-term amplitude(s) for loop diagram number 180
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZDOWN,AMPL(1,245))
C     Counter-term amplitude(s) for loop diagram number 182
      CALL R2_GG_1_R2_GG_3_0(W(9),W(27),R2_GGQ,R2_GGB,AMPL(1,246))
C     Counter-term amplitude(s) for loop diagram number 183
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,247))
C     Counter-term amplitude(s) for loop diagram number 185
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZDOWN,AMPL(1,248))
C     Counter-term amplitude(s) for loop diagram number 187
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZDOWN,AMPL(1,249))
C     Counter-term amplitude(s) for loop diagram number 189
      CALL R2_GGZ_0(W(5),W(30),W(3),R2_GGZUP,AMPL(1,250))
C     Counter-term amplitude(s) for loop diagram number 190
      CALL R2_GGZ_0(W(4),W(31),W(3),R2_GGZUP,AMPL(1,251))
C     Counter-term amplitude(s) for loop diagram number 191
      CALL R2_GGVV_0(W(4),W(5),W(29),W(3),R2_GGGZVECUP,AMPL(1,252))
      CALL R2_GGGVA_0(W(4),W(5),W(29),W(3),R2_GGGZAXIALUP,AMPL(1,253))
C     Counter-term amplitude(s) for loop diagram number 193
      CALL R2_GGZ_0(W(29),W(9),W(3),R2_GGZUP,AMPL(1,254))
C     Counter-term amplitude(s) for loop diagram number 201
      CALL R2_GG_1_R2_GG_3_0(W(9),W(17),R2_GGQ,R2_GGT,AMPL(1,255))
C     Counter-term amplitude(s) for loop diagram number 202
      CALL VVV1_0(W(4),W(5),W(17),R2_3GQ,AMPL(1,256))
C     Counter-term amplitude(s) for loop diagram number 204
      CALL R2_GGZ_0(W(5),W(32),W(3),R2_GGZUP,AMPL(1,257))
C     Counter-term amplitude(s) for loop diagram number 206
      CALL R2_GGZ_0(W(4),W(33),W(3),R2_GGZUP,AMPL(1,258))
C     Counter-term amplitude(s) for loop diagram number 208
      CALL R2_GG_1_R2_GG_3_0(W(9),W(27),R2_GGQ,R2_GGT,AMPL(1,259))
C     Counter-term amplitude(s) for loop diagram number 209
      CALL VVV1_0(W(4),W(5),W(27),R2_3GQ,AMPL(1,260))
C     Counter-term amplitude(s) for loop diagram number 211
      CALL R2_GGZ_0(W(5),W(34),W(3),R2_GGZUP,AMPL(1,261))
C     Counter-term amplitude(s) for loop diagram number 213
      CALL R2_GGZ_0(W(4),W(35),W(3),R2_GGZUP,AMPL(1,262))
C     At this point, all CT amps needed for (QCD=6), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

