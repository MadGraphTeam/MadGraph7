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
      PARAMETER (NEXTERNAL=6)
      INTEGER    NCOMB
      PARAMETER (NCOMB=144)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=30)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=1378, NLOOPGROUPS=745, NCTAMPS=1202)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=2580)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=147,NLOOPWAVEFUNCS=2146)
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
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1,W(5))
      CALL VXXXXX(P(0,6),ZERO,NHEL(6),+1,W(6))
      CALL FFV2_5_2(W(1),W(3),GC_22,GC_23,ZERO,ZERO,W(7))
      CALL FFV2_5_1(W(2),W(4),GC_22,GC_23,ZERO,ZERO,W(8))
      CALL FFV1_2(W(7),W(5),GC_5,ZERO,ZERO,W(9))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(9),W(8),W(6),GC_5,AMP(1))
      CALL FFV1_2(W(7),W(6),GC_5,ZERO,ZERO,W(10))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(10),W(8),W(5),GC_5,AMP(2))
      CALL VVV1P0_1(W(5),W(6),GC_4,ZERO,ZERO,W(11))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV1_0(W(7),W(8),W(11),GC_5,AMP(3))
      CALL FFV1_1(W(2),W(5),GC_5,ZERO,ZERO,W(12))
      CALL FFV2_5_2(W(7),W(4),GC_22,GC_23,ZERO,ZERO,W(13))
C     Amplitude(s) for born diagram with ID 4
      CALL FFV1_0(W(13),W(12),W(6),GC_5,AMP(4))
C     Amplitude(s) for born diagram with ID 5
      CALL FFV2_5_0(W(10),W(12),W(4),GC_22,GC_23,AMP(5))
      CALL FFV1_1(W(2),W(6),GC_5,ZERO,ZERO,W(14))
C     Amplitude(s) for born diagram with ID 6
      CALL FFV1_0(W(13),W(14),W(5),GC_5,AMP(6))
C     Amplitude(s) for born diagram with ID 7
      CALL FFV2_5_0(W(9),W(14),W(4),GC_22,GC_23,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL FFV1_0(W(13),W(2),W(11),GC_5,AMP(8))
      CALL FFV2_5_2(W(1),W(4),GC_22,GC_23,ZERO,ZERO,W(15))
      CALL FFV2_5_1(W(2),W(3),GC_22,GC_23,ZERO,ZERO,W(16))
      CALL FFV1_2(W(15),W(5),GC_5,ZERO,ZERO,W(17))
C     Amplitude(s) for born diagram with ID 9
      CALL FFV1_0(W(17),W(16),W(6),GC_5,AMP(9))
      CALL FFV1_2(W(15),W(6),GC_5,ZERO,ZERO,W(18))
C     Amplitude(s) for born diagram with ID 10
      CALL FFV1_0(W(18),W(16),W(5),GC_5,AMP(10))
C     Amplitude(s) for born diagram with ID 11
      CALL FFV1_0(W(15),W(16),W(11),GC_5,AMP(11))
      CALL FFV2_5_2(W(15),W(3),GC_22,GC_23,ZERO,ZERO,W(19))
C     Amplitude(s) for born diagram with ID 12
      CALL FFV1_0(W(19),W(12),W(6),GC_5,AMP(12))
C     Amplitude(s) for born diagram with ID 13
      CALL FFV2_5_0(W(18),W(12),W(3),GC_22,GC_23,AMP(13))
C     Amplitude(s) for born diagram with ID 14
      CALL FFV1_0(W(19),W(14),W(5),GC_5,AMP(14))
C     Amplitude(s) for born diagram with ID 15
      CALL FFV2_5_0(W(17),W(14),W(3),GC_22,GC_23,AMP(15))
C     Amplitude(s) for born diagram with ID 16
      CALL FFV1_0(W(19),W(2),W(11),GC_5,AMP(16))
      CALL FFV1_2(W(1),W(5),GC_5,ZERO,ZERO,W(20))
      CALL FFV2_5_2(W(20),W(4),GC_22,GC_23,ZERO,ZERO,W(21))
C     Amplitude(s) for born diagram with ID 17
      CALL FFV1_0(W(21),W(16),W(6),GC_5,AMP(17))
      CALL FFV1_2(W(20),W(6),GC_5,ZERO,ZERO,W(22))
C     Amplitude(s) for born diagram with ID 18
      CALL FFV2_5_0(W(22),W(16),W(4),GC_22,GC_23,AMP(18))
      CALL FFV2_5_2(W(20),W(3),GC_22,GC_23,ZERO,ZERO,W(23))
C     Amplitude(s) for born diagram with ID 19
      CALL FFV1_0(W(23),W(8),W(6),GC_5,AMP(19))
C     Amplitude(s) for born diagram with ID 20
      CALL FFV2_5_0(W(22),W(8),W(3),GC_22,GC_23,AMP(20))
C     Amplitude(s) for born diagram with ID 21
      CALL FFV2_5_0(W(23),W(14),W(4),GC_22,GC_23,AMP(21))
C     Amplitude(s) for born diagram with ID 22
      CALL FFV2_5_0(W(21),W(14),W(3),GC_22,GC_23,AMP(22))
      CALL FFV1_2(W(1),W(6),GC_5,ZERO,ZERO,W(24))
      CALL FFV2_5_2(W(24),W(4),GC_22,GC_23,ZERO,ZERO,W(25))
C     Amplitude(s) for born diagram with ID 23
      CALL FFV1_0(W(25),W(16),W(5),GC_5,AMP(23))
      CALL FFV1_2(W(24),W(5),GC_5,ZERO,ZERO,W(26))
C     Amplitude(s) for born diagram with ID 24
      CALL FFV2_5_0(W(26),W(16),W(4),GC_22,GC_23,AMP(24))
      CALL FFV2_5_2(W(24),W(3),GC_22,GC_23,ZERO,ZERO,W(27))
C     Amplitude(s) for born diagram with ID 25
      CALL FFV1_0(W(27),W(8),W(5),GC_5,AMP(25))
C     Amplitude(s) for born diagram with ID 26
      CALL FFV2_5_0(W(26),W(8),W(3),GC_22,GC_23,AMP(26))
C     Amplitude(s) for born diagram with ID 27
      CALL FFV2_5_0(W(27),W(12),W(4),GC_22,GC_23,AMP(27))
C     Amplitude(s) for born diagram with ID 28
      CALL FFV2_5_0(W(25),W(12),W(3),GC_22,GC_23,AMP(28))
      CALL FFV1_2(W(1),W(11),GC_5,ZERO,ZERO,W(28))
C     Amplitude(s) for born diagram with ID 29
      CALL FFV2_5_0(W(28),W(16),W(4),GC_22,GC_23,AMP(29))
C     Amplitude(s) for born diagram with ID 30
      CALL FFV2_5_0(W(28),W(8),W(3),GC_22,GC_23,AMP(30))
      CALL FFV1_1(W(8),W(6),GC_5,ZERO,ZERO,W(29))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL R2_QQ_1_0(W(9),W(29),R2_QQQ,AMPL(1,1))
      CALL FFV1_2(W(9),W(6),GC_5,ZERO,ZERO,W(30))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL R2_QQ_1_0(W(30),W(8),R2_QQQ,AMPL(1,2))
C     Counter-term amplitude(s) for loop diagram number 33
      CALL FFV1_0(W(9),W(8),W(6),R2_GQQ,AMPL(1,3))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,4))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,5))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,6))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,7))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,8))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,9))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQG_1EPS,AMPL(2,10))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQB,AMPL(1,11))
      CALL FFV1_0(W(9),W(8),W(6),UV_GQQT,AMPL(1,12))
      CALL FFV1_1(W(8),W(5),GC_5,ZERO,ZERO,W(31))
C     Counter-term amplitude(s) for loop diagram number 34
      CALL R2_QQ_1_0(W(10),W(31),R2_QQQ,AMPL(1,13))
      CALL FFV1_2(W(10),W(5),GC_5,ZERO,ZERO,W(32))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL R2_QQ_1_0(W(32),W(8),R2_QQQ,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 36
      CALL FFV1_0(W(10),W(8),W(5),R2_GQQ,AMPL(1,15))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,16))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,17))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,18))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,19))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,20))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,21))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQG_1EPS,AMPL(2,22))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQB,AMPL(1,23))
      CALL FFV1_0(W(10),W(8),W(5),UV_GQQT,AMPL(1,24))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL FFV1_0(W(7),W(31),W(6),R2_GQQ,AMPL(1,25))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,26))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,27))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,28))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,29))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,30))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB_1EPS,AMPL(2,31))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQG_1EPS,AMPL(2,32))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQB,AMPL(1,33))
      CALL FFV1_0(W(7),W(31),W(6),UV_GQQT,AMPL(1,34))
      CALL FFV1_1(W(31),W(6),GC_5,ZERO,ZERO,W(33))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL R2_QQ_1_0(W(7),W(33),R2_QQQ,AMPL(1,35))
C     Counter-term amplitude(s) for loop diagram number 39
      CALL FFV1_0(W(7),W(29),W(5),R2_GQQ,AMPL(1,36))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,37))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,38))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,39))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,40))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,41))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB_1EPS,AMPL(2,42))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQG_1EPS,AMPL(2,43))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQB,AMPL(1,44))
      CALL FFV1_0(W(7),W(29),W(5),UV_GQQT,AMPL(1,45))
      CALL FFV1_1(W(29),W(5),GC_5,ZERO,ZERO,W(34))
C     Counter-term amplitude(s) for loop diagram number 40
      CALL R2_QQ_1_0(W(7),W(34),R2_QQQ,AMPL(1,46))
      CALL FFV1P0_3(W(7),W(8),GC_5,ZERO,ZERO,W(35))
C     Counter-term amplitude(s) for loop diagram number 43
      CALL R2_GG_1_R2_GG_2_0(W(11),W(35),R2_GGG_1,R2_GGG_2,AMPL(1,47))
      CALL FFV1_2(W(7),W(11),GC_5,ZERO,ZERO,W(36))
C     Counter-term amplitude(s) for loop diagram number 44
      CALL R2_QQ_1_0(W(36),W(8),R2_QQQ,AMPL(1,48))
      CALL FFV1_1(W(8),W(11),GC_5,ZERO,ZERO,W(37))
C     Counter-term amplitude(s) for loop diagram number 45
      CALL R2_QQ_1_0(W(7),W(37),R2_QQQ,AMPL(1,49))
C     Counter-term amplitude(s) for loop diagram number 46
      CALL FFV1_0(W(7),W(8),W(11),R2_GQQ,AMPL(1,50))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,51))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,52))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,53))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,54))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,55))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,56))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQG_1EPS,AMPL(2,57))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQB,AMPL(1,58))
      CALL FFV1_0(W(7),W(8),W(11),UV_GQQT,AMPL(1,59))
C     Counter-term amplitude(s) for loop diagram number 48
      CALL VVV1_0(W(5),W(6),W(35),R2_3GG,AMPL(1,60))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,61))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,62))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,63))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,64))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,65))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB_1EPS,AMPL(2,66))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GG_1EPS,AMPL(2,67))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GB,AMPL(1,68))
      CALL VVV1_0(W(5),W(6),W(35),UV_3GT,AMPL(1,69))
      CALL FFV1_1(W(12),W(6),GC_5,ZERO,ZERO,W(38))
C     Counter-term amplitude(s) for loop diagram number 61
      CALL R2_QQ_1_0(W(13),W(38),R2_QQQ,AMPL(1,70))
      CALL FFV1_2(W(13),W(6),GC_5,ZERO,ZERO,W(39))
C     Counter-term amplitude(s) for loop diagram number 62
      CALL R2_QQ_1_0(W(39),W(12),R2_QQQ,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 63
      CALL FFV1_0(W(13),W(12),W(6),R2_GQQ,AMPL(1,72))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,73))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,74))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,75))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,76))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,77))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,78))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQG_1EPS,AMPL(2,79))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQB,AMPL(1,80))
      CALL FFV1_0(W(13),W(12),W(6),UV_GQQT,AMPL(1,81))
      CALL FFV2_5_1(W(12),W(4),GC_22,GC_23,ZERO,ZERO,W(40))
C     Counter-term amplitude(s) for loop diagram number 64
      CALL R2_QQ_1_0(W(10),W(40),R2_QQQ,AMPL(1,82))
      CALL FFV2_5_2(W(10),W(4),GC_22,GC_23,ZERO,ZERO,W(41))
C     Counter-term amplitude(s) for loop diagram number 65
      CALL R2_QQ_1_0(W(41),W(12),R2_QQQ,AMPL(1,83))
C     Counter-term amplitude(s) for loop diagram number 66
      CALL FFV2_5_0(W(10),W(12),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,84))
C     Counter-term amplitude(s) for loop diagram number 67
      CALL FFV1_0(W(7),W(40),W(6),R2_GQQ,AMPL(1,85))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,86))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,87))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,88))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,89))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,90))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,91))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQG_1EPS,AMPL(2,92))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQB,AMPL(1,93))
      CALL FFV1_0(W(7),W(40),W(6),UV_GQQT,AMPL(1,94))
      CALL FFV1_1(W(40),W(6),GC_5,ZERO,ZERO,W(42))
C     Counter-term amplitude(s) for loop diagram number 68
      CALL R2_QQ_1_0(W(7),W(42),R2_QQQ,AMPL(1,95))
C     Counter-term amplitude(s) for loop diagram number 69
      CALL FFV2_5_0(W(7),W(38),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,96))
      CALL FFV2_5_1(W(38),W(4),GC_22,GC_23,ZERO,ZERO,W(43))
C     Counter-term amplitude(s) for loop diagram number 70
      CALL R2_QQ_1_0(W(7),W(43),R2_QQQ,AMPL(1,97))
      CALL FFV1_1(W(14),W(5),GC_5,ZERO,ZERO,W(44))
C     Counter-term amplitude(s) for loop diagram number 76
      CALL R2_QQ_1_0(W(13),W(44),R2_QQQ,AMPL(1,98))
      CALL FFV1_2(W(13),W(5),GC_5,ZERO,ZERO,W(45))
C     Counter-term amplitude(s) for loop diagram number 77
      CALL R2_QQ_1_0(W(45),W(14),R2_QQQ,AMPL(1,99))
C     Counter-term amplitude(s) for loop diagram number 78
      CALL FFV1_0(W(13),W(14),W(5),R2_GQQ,AMPL(1,100))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,101))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,102))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,103))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,104))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,105))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB_1EPS,AMPL(2,106))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQG_1EPS,AMPL(2,107))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQB,AMPL(1,108))
      CALL FFV1_0(W(13),W(14),W(5),UV_GQQT,AMPL(1,109))
      CALL FFV2_5_1(W(14),W(4),GC_22,GC_23,ZERO,ZERO,W(46))
C     Counter-term amplitude(s) for loop diagram number 79
      CALL R2_QQ_1_0(W(9),W(46),R2_QQQ,AMPL(1,110))
      CALL FFV2_5_2(W(9),W(4),GC_22,GC_23,ZERO,ZERO,W(47))
C     Counter-term amplitude(s) for loop diagram number 80
      CALL R2_QQ_1_0(W(47),W(14),R2_QQQ,AMPL(1,111))
C     Counter-term amplitude(s) for loop diagram number 81
      CALL FFV2_5_0(W(9),W(14),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,112))
C     Counter-term amplitude(s) for loop diagram number 82
      CALL FFV1_0(W(7),W(46),W(5),R2_GQQ,AMPL(1,113))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,114))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,115))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,116))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,117))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,118))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB_1EPS,AMPL(2,119))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQG_1EPS,AMPL(2,120))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQB,AMPL(1,121))
      CALL FFV1_0(W(7),W(46),W(5),UV_GQQT,AMPL(1,122))
      CALL FFV1_1(W(46),W(5),GC_5,ZERO,ZERO,W(48))
C     Counter-term amplitude(s) for loop diagram number 83
      CALL R2_QQ_1_0(W(7),W(48),R2_QQQ,AMPL(1,123))
C     Counter-term amplitude(s) for loop diagram number 84
      CALL FFV2_5_0(W(7),W(44),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,124))
      CALL FFV2_5_1(W(44),W(4),GC_22,GC_23,ZERO,ZERO,W(49))
C     Counter-term amplitude(s) for loop diagram number 85
      CALL R2_QQ_1_0(W(7),W(49),R2_QQQ,AMPL(1,125))
C     Counter-term amplitude(s) for loop diagram number 91
      CALL FFV1_0(W(39),W(2),W(5),R2_GQQ,AMPL(1,126))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,127))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,128))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,129))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,130))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,131))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB_1EPS,AMPL(2,132))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQG_1EPS,AMPL(2,133))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQB,AMPL(1,134))
      CALL FFV1_0(W(39),W(2),W(5),UV_GQQT,AMPL(1,135))
C     Counter-term amplitude(s) for loop diagram number 93
      CALL FFV1_0(W(45),W(2),W(6),R2_GQQ,AMPL(1,136))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,137))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,138))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,139))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,140))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,141))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB_1EPS,AMPL(2,142))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQG_1EPS,AMPL(2,143))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQB,AMPL(1,144))
      CALL FFV1_0(W(45),W(2),W(6),UV_GQQT,AMPL(1,145))
C     Counter-term amplitude(s) for loop diagram number 95
      CALL FFV2_5_0(W(30),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,146))
C     Counter-term amplitude(s) for loop diagram number 97
      CALL FFV1_0(W(47),W(2),W(6),R2_GQQ,AMPL(1,147))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,148))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,149))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,150))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,151))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,152))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB_1EPS,AMPL(2,153))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQG_1EPS,AMPL(2,154))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQB,AMPL(1,155))
      CALL FFV1_0(W(47),W(2),W(6),UV_GQQT,AMPL(1,156))
C     Counter-term amplitude(s) for loop diagram number 99
      CALL FFV2_5_0(W(32),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,157))
C     Counter-term amplitude(s) for loop diagram number 101
      CALL FFV1_0(W(41),W(2),W(5),R2_GQQ,AMPL(1,158))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,159))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,160))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,161))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,162))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,163))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB_1EPS,AMPL(2,164))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQG_1EPS,AMPL(2,165))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQB,AMPL(1,166))
      CALL FFV1_0(W(41),W(2),W(5),UV_GQQT,AMPL(1,167))
C     Counter-term amplitude(s) for loop diagram number 109
      CALL FFV1_0(W(13),W(2),W(11),R2_GQQ,AMPL(1,168))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,169))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,170))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,171))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,172))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,173))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,174))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQG_1EPS,AMPL(2,175))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQB,AMPL(1,176))
      CALL FFV1_0(W(13),W(2),W(11),UV_GQQT,AMPL(1,177))
C     Counter-term amplitude(s) for loop diagram number 111
      CALL FFV2_5_0(W(36),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,178))
      CALL FFV1_1(W(2),W(11),GC_5,ZERO,ZERO,W(50))
C     Counter-term amplitude(s) for loop diagram number 133
      CALL R2_QQ_1_0(W(13),W(50),R2_QQQ,AMPL(1,179))
      CALL FFV1P0_3(W(13),W(2),GC_5,ZERO,ZERO,W(51))
C     Counter-term amplitude(s) for loop diagram number 134
      CALL R2_GG_1_R2_GG_2_0(W(11),W(51),R2_GGG_1,R2_GGG_2,AMPL(1,180))
C     Counter-term amplitude(s) for loop diagram number 135
      CALL FFV2_5_0(W(7),W(50),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,181))
      CALL FFV2_5_1(W(50),W(4),GC_22,GC_23,ZERO,ZERO,W(52))
C     Counter-term amplitude(s) for loop diagram number 136
      CALL R2_QQ_1_0(W(7),W(52),R2_QQQ,AMPL(1,182))
C     Counter-term amplitude(s) for loop diagram number 137
      CALL VVV1_0(W(5),W(6),W(51),R2_3GG,AMPL(1,183))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,184))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,185))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,186))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,187))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,188))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB_1EPS,AMPL(2,189))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GG_1EPS,AMPL(2,190))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GB,AMPL(1,191))
      CALL VVV1_0(W(5),W(6),W(51),UV_3GT,AMPL(1,192))
      CALL FFV1_1(W(16),W(6),GC_5,ZERO,ZERO,W(53))
C     Counter-term amplitude(s) for loop diagram number 141
      CALL R2_QQ_1_0(W(17),W(53),R2_QQQ,AMPL(1,193))
      CALL FFV1_2(W(17),W(6),GC_5,ZERO,ZERO,W(54))
C     Counter-term amplitude(s) for loop diagram number 142
      CALL R2_QQ_1_0(W(54),W(16),R2_QQQ,AMPL(1,194))
C     Counter-term amplitude(s) for loop diagram number 143
      CALL FFV1_0(W(17),W(16),W(6),R2_GQQ,AMPL(1,195))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,196))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,197))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,198))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,199))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,200))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB_1EPS,AMPL(2,201))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQG_1EPS,AMPL(2,202))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQB,AMPL(1,203))
      CALL FFV1_0(W(17),W(16),W(6),UV_GQQT,AMPL(1,204))
      CALL FFV1_1(W(16),W(5),GC_5,ZERO,ZERO,W(55))
C     Counter-term amplitude(s) for loop diagram number 144
      CALL R2_QQ_1_0(W(18),W(55),R2_QQQ,AMPL(1,205))
      CALL FFV1_2(W(18),W(5),GC_5,ZERO,ZERO,W(56))
C     Counter-term amplitude(s) for loop diagram number 145
      CALL R2_QQ_1_0(W(56),W(16),R2_QQQ,AMPL(1,206))
C     Counter-term amplitude(s) for loop diagram number 146
      CALL FFV1_0(W(18),W(16),W(5),R2_GQQ,AMPL(1,207))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,208))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,209))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,210))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,211))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,212))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB_1EPS,AMPL(2,213))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQG_1EPS,AMPL(2,214))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQB,AMPL(1,215))
      CALL FFV1_0(W(18),W(16),W(5),UV_GQQT,AMPL(1,216))
C     Counter-term amplitude(s) for loop diagram number 147
      CALL FFV1_0(W(15),W(55),W(6),R2_GQQ,AMPL(1,217))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,218))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,219))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,220))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,221))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,222))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB_1EPS,AMPL(2,223))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQG_1EPS,AMPL(2,224))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQB,AMPL(1,225))
      CALL FFV1_0(W(15),W(55),W(6),UV_GQQT,AMPL(1,226))
      CALL FFV1_1(W(55),W(6),GC_5,ZERO,ZERO,W(57))
C     Counter-term amplitude(s) for loop diagram number 148
      CALL R2_QQ_1_0(W(15),W(57),R2_QQQ,AMPL(1,227))
C     Counter-term amplitude(s) for loop diagram number 149
      CALL FFV1_0(W(15),W(53),W(5),R2_GQQ,AMPL(1,228))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,229))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,230))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,231))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,232))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,233))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB_1EPS,AMPL(2,234))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQG_1EPS,AMPL(2,235))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQB,AMPL(1,236))
      CALL FFV1_0(W(15),W(53),W(5),UV_GQQT,AMPL(1,237))
      CALL FFV1_1(W(53),W(5),GC_5,ZERO,ZERO,W(58))
C     Counter-term amplitude(s) for loop diagram number 150
      CALL R2_QQ_1_0(W(15),W(58),R2_QQQ,AMPL(1,238))
      CALL FFV1P0_3(W(15),W(16),GC_5,ZERO,ZERO,W(59))
C     Counter-term amplitude(s) for loop diagram number 153
      CALL R2_GG_1_R2_GG_2_0(W(11),W(59),R2_GGG_1,R2_GGG_2,AMPL(1,239))
      CALL FFV1_2(W(15),W(11),GC_5,ZERO,ZERO,W(60))
C     Counter-term amplitude(s) for loop diagram number 154
      CALL R2_QQ_1_0(W(60),W(16),R2_QQQ,AMPL(1,240))
      CALL FFV1_1(W(16),W(11),GC_5,ZERO,ZERO,W(61))
C     Counter-term amplitude(s) for loop diagram number 155
      CALL R2_QQ_1_0(W(15),W(61),R2_QQQ,AMPL(1,241))
C     Counter-term amplitude(s) for loop diagram number 156
      CALL FFV1_0(W(15),W(16),W(11),R2_GQQ,AMPL(1,242))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,243))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,244))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,245))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,246))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,247))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB_1EPS,AMPL(2,248))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQG_1EPS,AMPL(2,249))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQB,AMPL(1,250))
      CALL FFV1_0(W(15),W(16),W(11),UV_GQQT,AMPL(1,251))
C     Counter-term amplitude(s) for loop diagram number 158
      CALL VVV1_0(W(5),W(6),W(59),R2_3GG,AMPL(1,252))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,253))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,254))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,255))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,256))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,257))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB_1EPS,AMPL(2,258))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GG_1EPS,AMPL(2,259))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GB,AMPL(1,260))
      CALL VVV1_0(W(5),W(6),W(59),UV_3GT,AMPL(1,261))
C     Counter-term amplitude(s) for loop diagram number 171
      CALL R2_QQ_1_0(W(19),W(38),R2_QQQ,AMPL(1,262))
      CALL FFV1_2(W(19),W(6),GC_5,ZERO,ZERO,W(62))
C     Counter-term amplitude(s) for loop diagram number 172
      CALL R2_QQ_1_0(W(62),W(12),R2_QQQ,AMPL(1,263))
C     Counter-term amplitude(s) for loop diagram number 173
      CALL FFV1_0(W(19),W(12),W(6),R2_GQQ,AMPL(1,264))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,265))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,266))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,267))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,268))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,269))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB_1EPS,AMPL(2,270))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQG_1EPS,AMPL(2,271))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQB,AMPL(1,272))
      CALL FFV1_0(W(19),W(12),W(6),UV_GQQT,AMPL(1,273))
      CALL FFV2_5_1(W(12),W(3),GC_22,GC_23,ZERO,ZERO,W(63))
C     Counter-term amplitude(s) for loop diagram number 174
      CALL R2_QQ_1_0(W(18),W(63),R2_QQQ,AMPL(1,274))
      CALL FFV2_5_2(W(18),W(3),GC_22,GC_23,ZERO,ZERO,W(64))
C     Counter-term amplitude(s) for loop diagram number 175
      CALL R2_QQ_1_0(W(64),W(12),R2_QQQ,AMPL(1,275))
C     Counter-term amplitude(s) for loop diagram number 176
      CALL FFV2_5_0(W(18),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,276))
C     Counter-term amplitude(s) for loop diagram number 177
      CALL FFV1_0(W(15),W(63),W(6),R2_GQQ,AMPL(1,277))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,278))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,279))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,280))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,281))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,282))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB_1EPS,AMPL(2,283))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQG_1EPS,AMPL(2,284))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQB,AMPL(1,285))
      CALL FFV1_0(W(15),W(63),W(6),UV_GQQT,AMPL(1,286))
      CALL FFV1_1(W(63),W(6),GC_5,ZERO,ZERO,W(65))
C     Counter-term amplitude(s) for loop diagram number 178
      CALL R2_QQ_1_0(W(15),W(65),R2_QQQ,AMPL(1,287))
C     Counter-term amplitude(s) for loop diagram number 179
      CALL FFV2_5_0(W(15),W(38),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,288))
      CALL FFV2_5_1(W(38),W(3),GC_22,GC_23,ZERO,ZERO,W(66))
C     Counter-term amplitude(s) for loop diagram number 180
      CALL R2_QQ_1_0(W(15),W(66),R2_QQQ,AMPL(1,289))
C     Counter-term amplitude(s) for loop diagram number 186
      CALL R2_QQ_1_0(W(19),W(44),R2_QQQ,AMPL(1,290))
      CALL FFV1_2(W(19),W(5),GC_5,ZERO,ZERO,W(67))
C     Counter-term amplitude(s) for loop diagram number 187
      CALL R2_QQ_1_0(W(67),W(14),R2_QQQ,AMPL(1,291))
C     Counter-term amplitude(s) for loop diagram number 188
      CALL FFV1_0(W(19),W(14),W(5),R2_GQQ,AMPL(1,292))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,293))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,294))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,295))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,296))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,297))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB_1EPS,AMPL(2,298))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQG_1EPS,AMPL(2,299))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQB,AMPL(1,300))
      CALL FFV1_0(W(19),W(14),W(5),UV_GQQT,AMPL(1,301))
      CALL FFV2_5_1(W(14),W(3),GC_22,GC_23,ZERO,ZERO,W(68))
C     Counter-term amplitude(s) for loop diagram number 189
      CALL R2_QQ_1_0(W(17),W(68),R2_QQQ,AMPL(1,302))
      CALL FFV2_5_2(W(17),W(3),GC_22,GC_23,ZERO,ZERO,W(69))
C     Counter-term amplitude(s) for loop diagram number 190
      CALL R2_QQ_1_0(W(69),W(14),R2_QQQ,AMPL(1,303))
C     Counter-term amplitude(s) for loop diagram number 191
      CALL FFV2_5_0(W(17),W(14),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,304))
C     Counter-term amplitude(s) for loop diagram number 192
      CALL FFV1_0(W(15),W(68),W(5),R2_GQQ,AMPL(1,305))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,306))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,307))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,308))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,309))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,310))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB_1EPS,AMPL(2,311))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQG_1EPS,AMPL(2,312))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQB,AMPL(1,313))
      CALL FFV1_0(W(15),W(68),W(5),UV_GQQT,AMPL(1,314))
      CALL FFV1_1(W(68),W(5),GC_5,ZERO,ZERO,W(70))
C     Counter-term amplitude(s) for loop diagram number 193
      CALL R2_QQ_1_0(W(15),W(70),R2_QQQ,AMPL(1,315))
C     Counter-term amplitude(s) for loop diagram number 194
      CALL FFV2_5_0(W(15),W(44),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,316))
      CALL FFV2_5_1(W(44),W(3),GC_22,GC_23,ZERO,ZERO,W(71))
C     Counter-term amplitude(s) for loop diagram number 195
      CALL R2_QQ_1_0(W(15),W(71),R2_QQQ,AMPL(1,317))
C     Counter-term amplitude(s) for loop diagram number 201
      CALL FFV1_0(W(62),W(2),W(5),R2_GQQ,AMPL(1,318))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,319))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,320))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,321))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,322))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,323))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB_1EPS,AMPL(2,324))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQG_1EPS,AMPL(2,325))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQB,AMPL(1,326))
      CALL FFV1_0(W(62),W(2),W(5),UV_GQQT,AMPL(1,327))
C     Counter-term amplitude(s) for loop diagram number 203
      CALL FFV1_0(W(67),W(2),W(6),R2_GQQ,AMPL(1,328))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,329))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,330))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,331))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,332))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,333))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB_1EPS,AMPL(2,334))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQG_1EPS,AMPL(2,335))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQB,AMPL(1,336))
      CALL FFV1_0(W(67),W(2),W(6),UV_GQQT,AMPL(1,337))
C     Counter-term amplitude(s) for loop diagram number 205
      CALL FFV2_5_0(W(54),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,338))
C     Counter-term amplitude(s) for loop diagram number 207
      CALL FFV1_0(W(69),W(2),W(6),R2_GQQ,AMPL(1,339))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,340))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,341))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,342))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,343))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,344))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB_1EPS,AMPL(2,345))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQG_1EPS,AMPL(2,346))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQB,AMPL(1,347))
      CALL FFV1_0(W(69),W(2),W(6),UV_GQQT,AMPL(1,348))
C     Counter-term amplitude(s) for loop diagram number 209
      CALL FFV2_5_0(W(56),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,349))
C     Counter-term amplitude(s) for loop diagram number 211
      CALL FFV1_0(W(64),W(2),W(5),R2_GQQ,AMPL(1,350))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,351))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,352))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,353))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,354))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,355))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB_1EPS,AMPL(2,356))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQG_1EPS,AMPL(2,357))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQB,AMPL(1,358))
      CALL FFV1_0(W(64),W(2),W(5),UV_GQQT,AMPL(1,359))
C     Counter-term amplitude(s) for loop diagram number 219
      CALL FFV1_0(W(19),W(2),W(11),R2_GQQ,AMPL(1,360))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,361))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,362))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,363))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,364))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,365))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB_1EPS,AMPL(2,366))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQG_1EPS,AMPL(2,367))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQB,AMPL(1,368))
      CALL FFV1_0(W(19),W(2),W(11),UV_GQQT,AMPL(1,369))
C     Counter-term amplitude(s) for loop diagram number 221
      CALL FFV2_5_0(W(60),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,370))
C     Counter-term amplitude(s) for loop diagram number 243
      CALL R2_QQ_1_0(W(19),W(50),R2_QQQ,AMPL(1,371))
      CALL FFV1P0_3(W(19),W(2),GC_5,ZERO,ZERO,W(72))
C     Counter-term amplitude(s) for loop diagram number 244
      CALL R2_GG_1_R2_GG_2_0(W(11),W(72),R2_GGG_1,R2_GGG_2,AMPL(1,372))
C     Counter-term amplitude(s) for loop diagram number 245
      CALL FFV2_5_0(W(15),W(50),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,373))
      CALL FFV2_5_1(W(50),W(3),GC_22,GC_23,ZERO,ZERO,W(73))
C     Counter-term amplitude(s) for loop diagram number 246
      CALL R2_QQ_1_0(W(15),W(73),R2_QQQ,AMPL(1,374))
C     Counter-term amplitude(s) for loop diagram number 247
      CALL VVV1_0(W(5),W(6),W(72),R2_3GG,AMPL(1,375))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,376))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,377))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,378))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,379))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,380))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB_1EPS,AMPL(2,381))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GG_1EPS,AMPL(2,382))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GB,AMPL(1,383))
      CALL VVV1_0(W(5),W(6),W(72),UV_3GT,AMPL(1,384))
C     Counter-term amplitude(s) for loop diagram number 251
      CALL R2_QQ_1_0(W(21),W(53),R2_QQQ,AMPL(1,385))
      CALL FFV1_2(W(21),W(6),GC_5,ZERO,ZERO,W(74))
C     Counter-term amplitude(s) for loop diagram number 252
      CALL R2_QQ_1_0(W(74),W(16),R2_QQQ,AMPL(1,386))
C     Counter-term amplitude(s) for loop diagram number 253
      CALL FFV1_0(W(21),W(16),W(6),R2_GQQ,AMPL(1,387))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,388))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,389))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,390))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,391))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,392))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB_1EPS,AMPL(2,393))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQG_1EPS,AMPL(2,394))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQB,AMPL(1,395))
      CALL FFV1_0(W(21),W(16),W(6),UV_GQQT,AMPL(1,396))
      CALL FFV2_5_1(W(16),W(4),GC_22,GC_23,ZERO,ZERO,W(75))
C     Counter-term amplitude(s) for loop diagram number 254
      CALL R2_QQ_1_0(W(22),W(75),R2_QQQ,AMPL(1,397))
      CALL FFV2_5_2(W(22),W(4),GC_22,GC_23,ZERO,ZERO,W(76))
C     Counter-term amplitude(s) for loop diagram number 255
      CALL R2_QQ_1_0(W(76),W(16),R2_QQQ,AMPL(1,398))
C     Counter-term amplitude(s) for loop diagram number 256
      CALL FFV2_5_0(W(22),W(16),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,399))
C     Counter-term amplitude(s) for loop diagram number 257
      CALL FFV1_0(W(20),W(75),W(6),R2_GQQ,AMPL(1,400))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,401))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,402))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,403))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,404))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,405))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB_1EPS,AMPL(2,406))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQG_1EPS,AMPL(2,407))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQB,AMPL(1,408))
      CALL FFV1_0(W(20),W(75),W(6),UV_GQQT,AMPL(1,409))
      CALL FFV1_1(W(75),W(6),GC_5,ZERO,ZERO,W(77))
C     Counter-term amplitude(s) for loop diagram number 258
      CALL R2_QQ_1_0(W(20),W(77),R2_QQQ,AMPL(1,410))
C     Counter-term amplitude(s) for loop diagram number 259
      CALL FFV2_5_0(W(20),W(53),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,411))
      CALL FFV2_5_1(W(53),W(4),GC_22,GC_23,ZERO,ZERO,W(78))
C     Counter-term amplitude(s) for loop diagram number 260
      CALL R2_QQ_1_0(W(20),W(78),R2_QQQ,AMPL(1,412))
C     Counter-term amplitude(s) for loop diagram number 266
      CALL R2_QQ_1_0(W(23),W(29),R2_QQQ,AMPL(1,413))
      CALL FFV1_2(W(23),W(6),GC_5,ZERO,ZERO,W(79))
C     Counter-term amplitude(s) for loop diagram number 267
      CALL R2_QQ_1_0(W(79),W(8),R2_QQQ,AMPL(1,414))
C     Counter-term amplitude(s) for loop diagram number 268
      CALL FFV1_0(W(23),W(8),W(6),R2_GQQ,AMPL(1,415))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,416))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,417))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,418))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,419))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,420))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB_1EPS,AMPL(2,421))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQG_1EPS,AMPL(2,422))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQB,AMPL(1,423))
      CALL FFV1_0(W(23),W(8),W(6),UV_GQQT,AMPL(1,424))
      CALL FFV2_5_1(W(8),W(3),GC_22,GC_23,ZERO,ZERO,W(80))
C     Counter-term amplitude(s) for loop diagram number 269
      CALL R2_QQ_1_0(W(22),W(80),R2_QQQ,AMPL(1,425))
      CALL FFV2_5_2(W(22),W(3),GC_22,GC_23,ZERO,ZERO,W(81))
C     Counter-term amplitude(s) for loop diagram number 270
      CALL R2_QQ_1_0(W(81),W(8),R2_QQQ,AMPL(1,426))
C     Counter-term amplitude(s) for loop diagram number 271
      CALL FFV2_5_0(W(22),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,427))
C     Counter-term amplitude(s) for loop diagram number 272
      CALL FFV1_0(W(20),W(80),W(6),R2_GQQ,AMPL(1,428))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,429))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,430))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,431))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,432))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,433))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB_1EPS,AMPL(2,434))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQG_1EPS,AMPL(2,435))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQB,AMPL(1,436))
      CALL FFV1_0(W(20),W(80),W(6),UV_GQQT,AMPL(1,437))
      CALL FFV1_1(W(80),W(6),GC_5,ZERO,ZERO,W(82))
C     Counter-term amplitude(s) for loop diagram number 273
      CALL R2_QQ_1_0(W(20),W(82),R2_QQQ,AMPL(1,438))
C     Counter-term amplitude(s) for loop diagram number 274
      CALL FFV2_5_0(W(20),W(29),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,439))
      CALL FFV2_5_1(W(29),W(3),GC_22,GC_23,ZERO,ZERO,W(83))
C     Counter-term amplitude(s) for loop diagram number 275
      CALL R2_QQ_1_0(W(20),W(83),R2_QQQ,AMPL(1,440))
C     Counter-term amplitude(s) for loop diagram number 281
      CALL R2_QQ_1_0(W(23),W(46),R2_QQQ,AMPL(1,441))
      CALL FFV2_5_2(W(23),W(4),GC_22,GC_23,ZERO,ZERO,W(84))
C     Counter-term amplitude(s) for loop diagram number 282
      CALL R2_QQ_1_0(W(84),W(14),R2_QQQ,AMPL(1,442))
C     Counter-term amplitude(s) for loop diagram number 283
      CALL FFV2_5_0(W(23),W(14),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,443))
C     Counter-term amplitude(s) for loop diagram number 284
      CALL R2_QQ_1_0(W(21),W(68),R2_QQQ,AMPL(1,444))
      CALL FFV2_5_2(W(21),W(3),GC_22,GC_23,ZERO,ZERO,W(85))
C     Counter-term amplitude(s) for loop diagram number 285
      CALL R2_QQ_1_0(W(85),W(14),R2_QQQ,AMPL(1,445))
C     Counter-term amplitude(s) for loop diagram number 286
      CALL FFV2_5_0(W(21),W(14),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,446))
C     Counter-term amplitude(s) for loop diagram number 287
      CALL FFV2_5_0(W(20),W(68),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,447))
      CALL FFV2_5_1(W(68),W(4),GC_22,GC_23,ZERO,ZERO,W(86))
C     Counter-term amplitude(s) for loop diagram number 288
      CALL R2_QQ_1_0(W(20),W(86),R2_QQQ,AMPL(1,448))
C     Counter-term amplitude(s) for loop diagram number 289
      CALL FFV2_5_0(W(20),W(46),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,449))
      CALL FFV2_5_1(W(46),W(3),GC_22,GC_23,ZERO,ZERO,W(87))
C     Counter-term amplitude(s) for loop diagram number 290
      CALL R2_QQ_1_0(W(20),W(87),R2_QQQ,AMPL(1,450))
C     Counter-term amplitude(s) for loop diagram number 293
      CALL FFV2_5_0(W(79),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,451))
C     Counter-term amplitude(s) for loop diagram number 295
      CALL FFV1_0(W(84),W(2),W(6),R2_GQQ,AMPL(1,452))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,453))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,454))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,455))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,456))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,457))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB_1EPS,AMPL(2,458))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQG_1EPS,AMPL(2,459))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQB,AMPL(1,460))
      CALL FFV1_0(W(84),W(2),W(6),UV_GQQT,AMPL(1,461))
C     Counter-term amplitude(s) for loop diagram number 297
      CALL FFV2_5_0(W(74),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,462))
C     Counter-term amplitude(s) for loop diagram number 299
      CALL FFV1_0(W(85),W(2),W(6),R2_GQQ,AMPL(1,463))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,464))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,465))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,466))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,467))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,468))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB_1EPS,AMPL(2,469))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQG_1EPS,AMPL(2,470))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQB,AMPL(1,471))
      CALL FFV1_0(W(85),W(2),W(6),UV_GQQT,AMPL(1,472))
C     Counter-term amplitude(s) for loop diagram number 301
      CALL FFV2_5_0(W(76),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,473))
C     Counter-term amplitude(s) for loop diagram number 303
      CALL FFV2_5_0(W(81),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,474))
C     Counter-term amplitude(s) for loop diagram number 317
      CALL R2_QQ_1_0(W(25),W(55),R2_QQQ,AMPL(1,475))
      CALL FFV1_2(W(25),W(5),GC_5,ZERO,ZERO,W(88))
C     Counter-term amplitude(s) for loop diagram number 318
      CALL R2_QQ_1_0(W(88),W(16),R2_QQQ,AMPL(1,476))
C     Counter-term amplitude(s) for loop diagram number 319
      CALL FFV1_0(W(25),W(16),W(5),R2_GQQ,AMPL(1,477))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,478))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,479))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,480))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,481))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,482))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB_1EPS,AMPL(2,483))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQG_1EPS,AMPL(2,484))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQB,AMPL(1,485))
      CALL FFV1_0(W(25),W(16),W(5),UV_GQQT,AMPL(1,486))
C     Counter-term amplitude(s) for loop diagram number 320
      CALL R2_QQ_1_0(W(26),W(75),R2_QQQ,AMPL(1,487))
      CALL FFV2_5_2(W(26),W(4),GC_22,GC_23,ZERO,ZERO,W(89))
C     Counter-term amplitude(s) for loop diagram number 321
      CALL R2_QQ_1_0(W(89),W(16),R2_QQQ,AMPL(1,488))
C     Counter-term amplitude(s) for loop diagram number 322
      CALL FFV2_5_0(W(26),W(16),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,489))
C     Counter-term amplitude(s) for loop diagram number 323
      CALL FFV1_0(W(24),W(75),W(5),R2_GQQ,AMPL(1,490))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,491))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,492))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,493))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,494))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,495))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB_1EPS,AMPL(2,496))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQG_1EPS,AMPL(2,497))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQB,AMPL(1,498))
      CALL FFV1_0(W(24),W(75),W(5),UV_GQQT,AMPL(1,499))
      CALL FFV1_1(W(75),W(5),GC_5,ZERO,ZERO,W(90))
C     Counter-term amplitude(s) for loop diagram number 324
      CALL R2_QQ_1_0(W(24),W(90),R2_QQQ,AMPL(1,500))
C     Counter-term amplitude(s) for loop diagram number 325
      CALL FFV2_5_0(W(24),W(55),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,501))
      CALL FFV2_5_1(W(55),W(4),GC_22,GC_23,ZERO,ZERO,W(91))
C     Counter-term amplitude(s) for loop diagram number 326
      CALL R2_QQ_1_0(W(24),W(91),R2_QQQ,AMPL(1,502))
C     Counter-term amplitude(s) for loop diagram number 332
      CALL R2_QQ_1_0(W(27),W(31),R2_QQQ,AMPL(1,503))
      CALL FFV1_2(W(27),W(5),GC_5,ZERO,ZERO,W(92))
C     Counter-term amplitude(s) for loop diagram number 333
      CALL R2_QQ_1_0(W(92),W(8),R2_QQQ,AMPL(1,504))
C     Counter-term amplitude(s) for loop diagram number 334
      CALL FFV1_0(W(27),W(8),W(5),R2_GQQ,AMPL(1,505))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,506))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,507))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,508))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,509))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,510))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB_1EPS,AMPL(2,511))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQG_1EPS,AMPL(2,512))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQB,AMPL(1,513))
      CALL FFV1_0(W(27),W(8),W(5),UV_GQQT,AMPL(1,514))
C     Counter-term amplitude(s) for loop diagram number 335
      CALL R2_QQ_1_0(W(26),W(80),R2_QQQ,AMPL(1,515))
      CALL FFV2_5_2(W(26),W(3),GC_22,GC_23,ZERO,ZERO,W(93))
C     Counter-term amplitude(s) for loop diagram number 336
      CALL R2_QQ_1_0(W(93),W(8),R2_QQQ,AMPL(1,516))
C     Counter-term amplitude(s) for loop diagram number 337
      CALL FFV2_5_0(W(26),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,517))
C     Counter-term amplitude(s) for loop diagram number 338
      CALL FFV1_0(W(24),W(80),W(5),R2_GQQ,AMPL(1,518))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,519))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,520))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,521))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,522))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,523))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB_1EPS,AMPL(2,524))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQG_1EPS,AMPL(2,525))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQB,AMPL(1,526))
      CALL FFV1_0(W(24),W(80),W(5),UV_GQQT,AMPL(1,527))
      CALL FFV1_1(W(80),W(5),GC_5,ZERO,ZERO,W(94))
C     Counter-term amplitude(s) for loop diagram number 339
      CALL R2_QQ_1_0(W(24),W(94),R2_QQQ,AMPL(1,528))
C     Counter-term amplitude(s) for loop diagram number 340
      CALL FFV2_5_0(W(24),W(31),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,529))
      CALL FFV2_5_1(W(31),W(3),GC_22,GC_23,ZERO,ZERO,W(95))
C     Counter-term amplitude(s) for loop diagram number 341
      CALL R2_QQ_1_0(W(24),W(95),R2_QQQ,AMPL(1,530))
C     Counter-term amplitude(s) for loop diagram number 347
      CALL R2_QQ_1_0(W(27),W(40),R2_QQQ,AMPL(1,531))
      CALL FFV2_5_2(W(27),W(4),GC_22,GC_23,ZERO,ZERO,W(96))
C     Counter-term amplitude(s) for loop diagram number 348
      CALL R2_QQ_1_0(W(96),W(12),R2_QQQ,AMPL(1,532))
C     Counter-term amplitude(s) for loop diagram number 349
      CALL FFV2_5_0(W(27),W(12),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,533))
C     Counter-term amplitude(s) for loop diagram number 350
      CALL R2_QQ_1_0(W(25),W(63),R2_QQQ,AMPL(1,534))
      CALL FFV2_5_2(W(25),W(3),GC_22,GC_23,ZERO,ZERO,W(97))
C     Counter-term amplitude(s) for loop diagram number 351
      CALL R2_QQ_1_0(W(97),W(12),R2_QQQ,AMPL(1,535))
C     Counter-term amplitude(s) for loop diagram number 352
      CALL FFV2_5_0(W(25),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,536))
C     Counter-term amplitude(s) for loop diagram number 353
      CALL FFV2_5_0(W(24),W(63),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,537))
      CALL FFV2_5_1(W(63),W(4),GC_22,GC_23,ZERO,ZERO,W(98))
C     Counter-term amplitude(s) for loop diagram number 354
      CALL R2_QQ_1_0(W(24),W(98),R2_QQQ,AMPL(1,538))
C     Counter-term amplitude(s) for loop diagram number 355
      CALL FFV2_5_0(W(24),W(40),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,539))
      CALL FFV2_5_1(W(40),W(3),GC_22,GC_23,ZERO,ZERO,W(99))
C     Counter-term amplitude(s) for loop diagram number 356
      CALL R2_QQ_1_0(W(24),W(99),R2_QQQ,AMPL(1,540))
C     Counter-term amplitude(s) for loop diagram number 359
      CALL FFV2_5_0(W(92),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,541))
C     Counter-term amplitude(s) for loop diagram number 361
      CALL FFV1_0(W(96),W(2),W(5),R2_GQQ,AMPL(1,542))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,543))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,544))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,545))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,546))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,547))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB_1EPS,AMPL(2,548))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQG_1EPS,AMPL(2,549))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQB,AMPL(1,550))
      CALL FFV1_0(W(96),W(2),W(5),UV_GQQT,AMPL(1,551))
C     Counter-term amplitude(s) for loop diagram number 363
      CALL FFV2_5_0(W(88),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,552))
C     Counter-term amplitude(s) for loop diagram number 365
      CALL FFV1_0(W(97),W(2),W(5),R2_GQQ,AMPL(1,553))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,554))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,555))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,556))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,557))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,558))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,559))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQG_1EPS,AMPL(2,560))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQB,AMPL(1,561))
      CALL FFV1_0(W(97),W(2),W(5),UV_GQQT,AMPL(1,562))
C     Counter-term amplitude(s) for loop diagram number 367
      CALL FFV2_5_0(W(89),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,563))
C     Counter-term amplitude(s) for loop diagram number 369
      CALL FFV2_5_0(W(93),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,564))
C     Counter-term amplitude(s) for loop diagram number 384
      CALL FFV2_5_0(W(1),W(57),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,565))
C     Counter-term amplitude(s) for loop diagram number 386
      CALL FFV2_5_0(W(1),W(58),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,566))
C     Counter-term amplitude(s) for loop diagram number 390
      CALL FFV1_0(W(1),W(77),W(5),R2_GQQ,AMPL(1,567))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,568))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,569))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,570))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,571))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,572))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB_1EPS,AMPL(2,573))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQG_1EPS,AMPL(2,574))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQB,AMPL(1,575))
      CALL FFV1_0(W(1),W(77),W(5),UV_GQQT,AMPL(1,576))
C     Counter-term amplitude(s) for loop diagram number 392
      CALL FFV1_0(W(1),W(78),W(5),R2_GQQ,AMPL(1,577))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,578))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,579))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,580))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,581))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,582))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB_1EPS,AMPL(2,583))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQG_1EPS,AMPL(2,584))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQB,AMPL(1,585))
      CALL FFV1_0(W(1),W(78),W(5),UV_GQQT,AMPL(1,586))
C     Counter-term amplitude(s) for loop diagram number 396
      CALL FFV1_0(W(1),W(90),W(6),R2_GQQ,AMPL(1,587))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,588))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,589))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,590))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,591))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,592))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB_1EPS,AMPL(2,593))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQG_1EPS,AMPL(2,594))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQB,AMPL(1,595))
      CALL FFV1_0(W(1),W(90),W(6),UV_GQQT,AMPL(1,596))
C     Counter-term amplitude(s) for loop diagram number 398
      CALL FFV1_0(W(1),W(91),W(6),R2_GQQ,AMPL(1,597))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,598))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,599))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,600))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,601))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,602))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB_1EPS,AMPL(2,603))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQG_1EPS,AMPL(2,604))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQB,AMPL(1,605))
      CALL FFV1_0(W(1),W(91),W(6),UV_GQQT,AMPL(1,606))
C     Counter-term amplitude(s) for loop diagram number 401
      CALL FFV2_5_0(W(1),W(61),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,607))
C     Counter-term amplitude(s) for loop diagram number 404
      CALL FFV1_0(W(1),W(75),W(11),R2_GQQ,AMPL(1,608))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,609))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,610))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,611))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,612))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,613))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB_1EPS,AMPL(2,614))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQG_1EPS,AMPL(2,615))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQB,AMPL(1,616))
      CALL FFV1_0(W(1),W(75),W(11),UV_GQQT,AMPL(1,617))
C     Counter-term amplitude(s) for loop diagram number 426
      CALL FFV2_5_0(W(1),W(33),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,618))
C     Counter-term amplitude(s) for loop diagram number 428
      CALL FFV2_5_0(W(1),W(34),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,619))
C     Counter-term amplitude(s) for loop diagram number 432
      CALL FFV1_0(W(1),W(82),W(5),R2_GQQ,AMPL(1,620))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,621))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,622))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,623))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,624))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,625))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB_1EPS,AMPL(2,626))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQG_1EPS,AMPL(2,627))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQB,AMPL(1,628))
      CALL FFV1_0(W(1),W(82),W(5),UV_GQQT,AMPL(1,629))
C     Counter-term amplitude(s) for loop diagram number 434
      CALL FFV1_0(W(1),W(83),W(5),R2_GQQ,AMPL(1,630))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,631))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,632))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,633))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,634))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,635))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB_1EPS,AMPL(2,636))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQG_1EPS,AMPL(2,637))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQB,AMPL(1,638))
      CALL FFV1_0(W(1),W(83),W(5),UV_GQQT,AMPL(1,639))
C     Counter-term amplitude(s) for loop diagram number 438
      CALL FFV1_0(W(1),W(94),W(6),R2_GQQ,AMPL(1,640))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,641))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,642))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,643))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,644))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,645))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB_1EPS,AMPL(2,646))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQG_1EPS,AMPL(2,647))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQB,AMPL(1,648))
      CALL FFV1_0(W(1),W(94),W(6),UV_GQQT,AMPL(1,649))
C     Counter-term amplitude(s) for loop diagram number 440
      CALL FFV1_0(W(1),W(95),W(6),R2_GQQ,AMPL(1,650))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,651))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,652))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,653))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,654))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,655))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB_1EPS,AMPL(2,656))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQG_1EPS,AMPL(2,657))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQB,AMPL(1,658))
      CALL FFV1_0(W(1),W(95),W(6),UV_GQQT,AMPL(1,659))
C     Counter-term amplitude(s) for loop diagram number 443
      CALL FFV2_5_0(W(1),W(37),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,660))
C     Counter-term amplitude(s) for loop diagram number 446
      CALL FFV1_0(W(1),W(80),W(11),R2_GQQ,AMPL(1,661))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,662))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,663))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,664))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,665))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,666))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB_1EPS,AMPL(2,667))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQG_1EPS,AMPL(2,668))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQB,AMPL(1,669))
      CALL FFV1_0(W(1),W(80),W(11),UV_GQQT,AMPL(1,670))
C     Counter-term amplitude(s) for loop diagram number 468
      CALL FFV2_5_0(W(1),W(42),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,671))
C     Counter-term amplitude(s) for loop diagram number 470
      CALL FFV2_5_0(W(1),W(43),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,672))
C     Counter-term amplitude(s) for loop diagram number 474
      CALL FFV2_5_0(W(1),W(65),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,673))
C     Counter-term amplitude(s) for loop diagram number 476
      CALL FFV2_5_0(W(1),W(66),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,674))
C     Counter-term amplitude(s) for loop diagram number 480
      CALL FFV1_0(W(1),W(98),W(6),R2_GQQ,AMPL(1,675))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,676))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,677))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,678))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,679))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,680))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB_1EPS,AMPL(2,681))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQG_1EPS,AMPL(2,682))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQB,AMPL(1,683))
      CALL FFV1_0(W(1),W(98),W(6),UV_GQQT,AMPL(1,684))
C     Counter-term amplitude(s) for loop diagram number 482
      CALL FFV1_0(W(1),W(99),W(6),R2_GQQ,AMPL(1,685))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,686))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,687))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,688))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,689))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,690))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB_1EPS,AMPL(2,691))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQG_1EPS,AMPL(2,692))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQB,AMPL(1,693))
      CALL FFV1_0(W(1),W(99),W(6),UV_GQQT,AMPL(1,694))
C     Counter-term amplitude(s) for loop diagram number 492
      CALL FFV2_5_0(W(1),W(48),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,695))
C     Counter-term amplitude(s) for loop diagram number 494
      CALL FFV2_5_0(W(1),W(49),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,696))
C     Counter-term amplitude(s) for loop diagram number 498
      CALL FFV2_5_0(W(1),W(70),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,697))
C     Counter-term amplitude(s) for loop diagram number 500
      CALL FFV2_5_0(W(1),W(71),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,698))
C     Counter-term amplitude(s) for loop diagram number 504
      CALL FFV1_0(W(1),W(86),W(5),R2_GQQ,AMPL(1,699))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,700))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,701))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,702))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,703))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,704))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB_1EPS,AMPL(2,705))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQG_1EPS,AMPL(2,706))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQB,AMPL(1,707))
      CALL FFV1_0(W(1),W(86),W(5),UV_GQQT,AMPL(1,708))
C     Counter-term amplitude(s) for loop diagram number 506
      CALL FFV1_0(W(1),W(87),W(5),R2_GQQ,AMPL(1,709))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,710))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,711))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,712))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,713))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,714))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB_1EPS,AMPL(2,715))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQG_1EPS,AMPL(2,716))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQB,AMPL(1,717))
      CALL FFV1_0(W(1),W(87),W(5),UV_GQQT,AMPL(1,718))
C     Counter-term amplitude(s) for loop diagram number 546
      CALL FFV2_5_0(W(1),W(52),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,719))
C     Counter-term amplitude(s) for loop diagram number 549
      CALL FFV2_5_0(W(1),W(73),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,720))
C     Counter-term amplitude(s) for loop diagram number 569
      CALL R2_QQ_1_0(W(28),W(75),R2_QQQ,AMPL(1,721))
      CALL FFV2_5_2(W(28),W(4),GC_22,GC_23,ZERO,ZERO,W(100))
C     Counter-term amplitude(s) for loop diagram number 570
      CALL R2_QQ_1_0(W(100),W(16),R2_QQQ,AMPL(1,722))
C     Counter-term amplitude(s) for loop diagram number 571
      CALL FFV2_5_0(W(28),W(16),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,723))
      CALL FFV1P0_3(W(1),W(75),GC_5,ZERO,ZERO,W(101))
C     Counter-term amplitude(s) for loop diagram number 572
      CALL R2_GG_1_R2_GG_2_0(W(11),W(101),R2_GGG_1,R2_GGG_2,AMPL(1,724)
     $ )
C     Counter-term amplitude(s) for loop diagram number 573
      CALL VVV1_0(W(5),W(6),W(101),R2_3GG,AMPL(1,725))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,726))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,727))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,728))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,729))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,730))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB_1EPS,AMPL(2,731))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GG_1EPS,AMPL(2,732))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GB,AMPL(1,733))
      CALL VVV1_0(W(5),W(6),W(101),UV_3GT,AMPL(1,734))
C     Counter-term amplitude(s) for loop diagram number 577
      CALL R2_QQ_1_0(W(28),W(80),R2_QQQ,AMPL(1,735))
      CALL FFV2_5_2(W(28),W(3),GC_22,GC_23,ZERO,ZERO,W(102))
C     Counter-term amplitude(s) for loop diagram number 578
      CALL R2_QQ_1_0(W(102),W(8),R2_QQQ,AMPL(1,736))
C     Counter-term amplitude(s) for loop diagram number 579
      CALL FFV2_5_0(W(28),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,737))
      CALL FFV1P0_3(W(1),W(80),GC_5,ZERO,ZERO,W(103))
C     Counter-term amplitude(s) for loop diagram number 580
      CALL R2_GG_1_R2_GG_2_0(W(11),W(103),R2_GGG_1,R2_GGG_2,AMPL(1,738)
     $ )
C     Counter-term amplitude(s) for loop diagram number 581
      CALL VVV1_0(W(5),W(6),W(103),R2_3GG,AMPL(1,739))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,740))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,741))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,742))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,743))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,744))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB_1EPS,AMPL(2,745))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GG_1EPS,AMPL(2,746))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GB,AMPL(1,747))
      CALL VVV1_0(W(5),W(6),W(103),UV_3GT,AMPL(1,748))
C     Counter-term amplitude(s) for loop diagram number 585
      CALL FFV2_5_0(W(100),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,749))
C     Counter-term amplitude(s) for loop diagram number 587
      CALL FFV2_5_0(W(102),W(2),W(4),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,750))
      CALL FFV1P0_3(W(1),W(2),GC_5,ZERO,ZERO,W(104))
      CALL VVV1P0_1(W(5),W(104),GC_4,ZERO,ZERO,W(105))
C     Counter-term amplitude(s) for loop diagram number 607
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZDOWN,AMPL(1,751))
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZDOWN,AMPL(1,752))
      CALL VVV1P0_1(W(6),W(104),GC_4,ZERO,ZERO,W(106))
C     Counter-term amplitude(s) for loop diagram number 609
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZDOWN,AMPL(1,753))
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZDOWN,AMPL(1,754))
C     Counter-term amplitude(s) for loop diagram number 627
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZDOWN,AMPL(1,755))
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZDOWN,AMPL(1,756))
C     Counter-term amplitude(s) for loop diagram number 649
      CALL R2_GG_1_0(W(11),W(35),R2_GGQ,AMPL(1,757))
      CALL R2_GG_1_0(W(11),W(35),R2_GGQ,AMPL(1,758))
      CALL R2_GG_1_0(W(11),W(35),R2_GGQ,AMPL(1,759))
      CALL R2_GG_1_0(W(11),W(35),R2_GGQ,AMPL(1,760))
C     Counter-term amplitude(s) for loop diagram number 650
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,761))
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,762))
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,763))
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,764))
      CALL FFV1P0_3(W(7),W(12),GC_5,ZERO,ZERO,W(107))
C     Counter-term amplitude(s) for loop diagram number 652
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZDOWN,AMPL(1,765))
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZDOWN,AMPL(1,766))
      CALL FFV1P0_3(W(7),W(14),GC_5,ZERO,ZERO,W(108))
C     Counter-term amplitude(s) for loop diagram number 654
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZDOWN,AMPL(1,767))
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZDOWN,AMPL(1,768))
      CALL FFV1P0_3(W(7),W(2),GC_5,ZERO,ZERO,W(109))
      CALL VVV1P0_1(W(6),W(109),GC_4,ZERO,ZERO,W(110))
C     Counter-term amplitude(s) for loop diagram number 656
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZDOWN,AMPL(1,769))
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZDOWN,AMPL(1,770))
C     Counter-term amplitude(s) for loop diagram number 657
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECDOWN,AMPL(1,771))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,772))
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECDOWN,AMPL(1,773))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,774))
      CALL VVV1P0_1(W(5),W(109),GC_4,ZERO,ZERO,W(111))
C     Counter-term amplitude(s) for loop diagram number 658
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZDOWN,AMPL(1,775))
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZDOWN,AMPL(1,776))
      CALL FFV1P0_3(W(9),W(2),GC_5,ZERO,ZERO,W(112))
C     Counter-term amplitude(s) for loop diagram number 660
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZDOWN,AMPL(1,777))
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZDOWN,AMPL(1,778))
      CALL FFV1P0_3(W(10),W(2),GC_5,ZERO,ZERO,W(113))
C     Counter-term amplitude(s) for loop diagram number 661
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZDOWN,AMPL(1,779))
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZDOWN,AMPL(1,780))
C     Counter-term amplitude(s) for loop diagram number 662
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZDOWN,AMPL(1,781))
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZDOWN,AMPL(1,782))
C     Counter-term amplitude(s) for loop diagram number 672
      CALL R2_GG_1_0(W(11),W(51),R2_GGQ,AMPL(1,783))
      CALL R2_GG_1_0(W(11),W(51),R2_GGQ,AMPL(1,784))
      CALL R2_GG_1_0(W(11),W(51),R2_GGQ,AMPL(1,785))
      CALL R2_GG_1_0(W(11),W(51),R2_GGQ,AMPL(1,786))
C     Counter-term amplitude(s) for loop diagram number 673
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,787))
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,788))
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,789))
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,790))
C     Counter-term amplitude(s) for loop diagram number 675
      CALL R2_GG_1_0(W(11),W(59),R2_GGQ,AMPL(1,791))
      CALL R2_GG_1_0(W(11),W(59),R2_GGQ,AMPL(1,792))
      CALL R2_GG_1_0(W(11),W(59),R2_GGQ,AMPL(1,793))
      CALL R2_GG_1_0(W(11),W(59),R2_GGQ,AMPL(1,794))
C     Counter-term amplitude(s) for loop diagram number 676
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,795))
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,796))
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,797))
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,798))
      CALL FFV1P0_3(W(15),W(12),GC_5,ZERO,ZERO,W(114))
C     Counter-term amplitude(s) for loop diagram number 678
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZDOWN,AMPL(1,799))
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZDOWN,AMPL(1,800))
      CALL FFV1P0_3(W(15),W(14),GC_5,ZERO,ZERO,W(115))
C     Counter-term amplitude(s) for loop diagram number 680
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZDOWN,AMPL(1,801))
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZDOWN,AMPL(1,802))
      CALL FFV1P0_3(W(15),W(2),GC_5,ZERO,ZERO,W(116))
      CALL VVV1P0_1(W(6),W(116),GC_4,ZERO,ZERO,W(117))
C     Counter-term amplitude(s) for loop diagram number 682
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZDOWN,AMPL(1,803))
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZDOWN,AMPL(1,804))
C     Counter-term amplitude(s) for loop diagram number 683
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECDOWN,AMPL(1,805))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,806))
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECDOWN,AMPL(1,807))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,808))
      CALL VVV1P0_1(W(5),W(116),GC_4,ZERO,ZERO,W(118))
C     Counter-term amplitude(s) for loop diagram number 684
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZDOWN,AMPL(1,809))
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZDOWN,AMPL(1,810))
      CALL FFV1P0_3(W(17),W(2),GC_5,ZERO,ZERO,W(119))
C     Counter-term amplitude(s) for loop diagram number 686
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZDOWN,AMPL(1,811))
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZDOWN,AMPL(1,812))
      CALL FFV1P0_3(W(18),W(2),GC_5,ZERO,ZERO,W(120))
C     Counter-term amplitude(s) for loop diagram number 687
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZDOWN,AMPL(1,813))
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZDOWN,AMPL(1,814))
C     Counter-term amplitude(s) for loop diagram number 688
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZDOWN,AMPL(1,815))
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZDOWN,AMPL(1,816))
C     Counter-term amplitude(s) for loop diagram number 698
      CALL R2_GG_1_0(W(11),W(72),R2_GGQ,AMPL(1,817))
      CALL R2_GG_1_0(W(11),W(72),R2_GGQ,AMPL(1,818))
      CALL R2_GG_1_0(W(11),W(72),R2_GGQ,AMPL(1,819))
      CALL R2_GG_1_0(W(11),W(72),R2_GGQ,AMPL(1,820))
C     Counter-term amplitude(s) for loop diagram number 699
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,821))
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,822))
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,823))
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,824))
      CALL FFV1P0_3(W(20),W(16),GC_5,ZERO,ZERO,W(121))
C     Counter-term amplitude(s) for loop diagram number 701
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZDOWN,AMPL(1,825))
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZDOWN,AMPL(1,826))
      CALL FFV1P0_3(W(20),W(8),GC_5,ZERO,ZERO,W(122))
C     Counter-term amplitude(s) for loop diagram number 703
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZDOWN,AMPL(1,827))
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZDOWN,AMPL(1,828))
      CALL FFV1P0_3(W(20),W(2),GC_5,ZERO,ZERO,W(123))
C     Counter-term amplitude(s) for loop diagram number 705
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZDOWN,AMPL(1,829))
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZDOWN,AMPL(1,830))
      CALL FFV1P0_3(W(21),W(2),GC_5,ZERO,ZERO,W(124))
C     Counter-term amplitude(s) for loop diagram number 707
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZDOWN,AMPL(1,831))
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZDOWN,AMPL(1,832))
      CALL FFV1P0_3(W(23),W(2),GC_5,ZERO,ZERO,W(125))
C     Counter-term amplitude(s) for loop diagram number 713
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZDOWN,AMPL(1,833))
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZDOWN,AMPL(1,834))
      CALL FFV1P0_3(W(24),W(16),GC_5,ZERO,ZERO,W(126))
C     Counter-term amplitude(s) for loop diagram number 715
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZDOWN,AMPL(1,835))
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZDOWN,AMPL(1,836))
      CALL FFV1P0_3(W(24),W(8),GC_5,ZERO,ZERO,W(127))
C     Counter-term amplitude(s) for loop diagram number 717
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZDOWN,AMPL(1,837))
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZDOWN,AMPL(1,838))
      CALL FFV1P0_3(W(24),W(2),GC_5,ZERO,ZERO,W(128))
C     Counter-term amplitude(s) for loop diagram number 719
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZDOWN,AMPL(1,839))
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZDOWN,AMPL(1,840))
      CALL FFV1P0_3(W(25),W(2),GC_5,ZERO,ZERO,W(129))
C     Counter-term amplitude(s) for loop diagram number 721
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZDOWN,AMPL(1,841))
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZDOWN,AMPL(1,842))
      CALL FFV1P0_3(W(27),W(2),GC_5,ZERO,ZERO,W(130))
C     Counter-term amplitude(s) for loop diagram number 727
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZDOWN,AMPL(1,843))
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZDOWN,AMPL(1,844))
      CALL FFV1P0_3(W(1),W(16),GC_5,ZERO,ZERO,W(131))
      CALL VVV1P0_1(W(6),W(131),GC_4,ZERO,ZERO,W(132))
C     Counter-term amplitude(s) for loop diagram number 729
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZDOWN,AMPL(1,845))
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZDOWN,AMPL(1,846))
C     Counter-term amplitude(s) for loop diagram number 730
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECDOWN,AMPL(1,847))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,848))
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECDOWN,AMPL(1,849))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,850))
      CALL VVV1P0_1(W(5),W(131),GC_4,ZERO,ZERO,W(133))
C     Counter-term amplitude(s) for loop diagram number 731
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZDOWN,AMPL(1,851))
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZDOWN,AMPL(1,852))
      CALL FFV1P0_3(W(1),W(55),GC_5,ZERO,ZERO,W(134))
C     Counter-term amplitude(s) for loop diagram number 733
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZDOWN,AMPL(1,853))
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZDOWN,AMPL(1,854))
      CALL FFV1P0_3(W(1),W(53),GC_5,ZERO,ZERO,W(135))
C     Counter-term amplitude(s) for loop diagram number 734
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZDOWN,AMPL(1,855))
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZDOWN,AMPL(1,856))
C     Counter-term amplitude(s) for loop diagram number 735
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZDOWN,AMPL(1,857))
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZDOWN,AMPL(1,858))
C     Counter-term amplitude(s) for loop diagram number 745
      CALL R2_GG_1_0(W(11),W(101),R2_GGQ,AMPL(1,859))
      CALL R2_GG_1_0(W(11),W(101),R2_GGQ,AMPL(1,860))
      CALL R2_GG_1_0(W(11),W(101),R2_GGQ,AMPL(1,861))
      CALL R2_GG_1_0(W(11),W(101),R2_GGQ,AMPL(1,862))
C     Counter-term amplitude(s) for loop diagram number 746
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,863))
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,864))
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,865))
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,866))
      CALL FFV1P0_3(W(1),W(8),GC_5,ZERO,ZERO,W(136))
      CALL VVV1P0_1(W(6),W(136),GC_4,ZERO,ZERO,W(137))
C     Counter-term amplitude(s) for loop diagram number 748
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZDOWN,AMPL(1,867))
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZDOWN,AMPL(1,868))
C     Counter-term amplitude(s) for loop diagram number 749
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECDOWN,AMPL(1,869))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,870))
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECDOWN,AMPL(1,871))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,872))
      CALL VVV1P0_1(W(5),W(136),GC_4,ZERO,ZERO,W(138))
C     Counter-term amplitude(s) for loop diagram number 750
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZDOWN,AMPL(1,873))
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZDOWN,AMPL(1,874))
      CALL FFV1P0_3(W(1),W(31),GC_5,ZERO,ZERO,W(139))
C     Counter-term amplitude(s) for loop diagram number 752
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZDOWN,AMPL(1,875))
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZDOWN,AMPL(1,876))
      CALL FFV1P0_3(W(1),W(29),GC_5,ZERO,ZERO,W(140))
C     Counter-term amplitude(s) for loop diagram number 753
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZDOWN,AMPL(1,877))
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZDOWN,AMPL(1,878))
C     Counter-term amplitude(s) for loop diagram number 754
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZDOWN,AMPL(1,879))
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZDOWN,AMPL(1,880))
C     Counter-term amplitude(s) for loop diagram number 764
      CALL R2_GG_1_0(W(11),W(103),R2_GGQ,AMPL(1,881))
      CALL R2_GG_1_0(W(11),W(103),R2_GGQ,AMPL(1,882))
      CALL R2_GG_1_0(W(11),W(103),R2_GGQ,AMPL(1,883))
      CALL R2_GG_1_0(W(11),W(103),R2_GGQ,AMPL(1,884))
C     Counter-term amplitude(s) for loop diagram number 765
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,885))
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,886))
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,887))
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,888))
      CALL FFV1P0_3(W(1),W(12),GC_5,ZERO,ZERO,W(141))
C     Counter-term amplitude(s) for loop diagram number 767
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZDOWN,AMPL(1,889))
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZDOWN,AMPL(1,890))
      CALL FFV1P0_3(W(1),W(40),GC_5,ZERO,ZERO,W(142))
C     Counter-term amplitude(s) for loop diagram number 769
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZDOWN,AMPL(1,891))
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZDOWN,AMPL(1,892))
      CALL FFV1P0_3(W(1),W(63),GC_5,ZERO,ZERO,W(143))
C     Counter-term amplitude(s) for loop diagram number 775
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZDOWN,AMPL(1,893))
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZDOWN,AMPL(1,894))
      CALL FFV1P0_3(W(1),W(14),GC_5,ZERO,ZERO,W(144))
C     Counter-term amplitude(s) for loop diagram number 777
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZDOWN,AMPL(1,895))
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZDOWN,AMPL(1,896))
      CALL FFV1P0_3(W(1),W(46),GC_5,ZERO,ZERO,W(145))
C     Counter-term amplitude(s) for loop diagram number 779
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZDOWN,AMPL(1,897))
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZDOWN,AMPL(1,898))
      CALL FFV1P0_3(W(1),W(68),GC_5,ZERO,ZERO,W(146))
C     Counter-term amplitude(s) for loop diagram number 785
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZDOWN,AMPL(1,899))
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZDOWN,AMPL(1,900))
C     Counter-term amplitude(s) for loop diagram number 787
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZUP,AMPL(1,901))
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZUP,AMPL(1,902))
C     Counter-term amplitude(s) for loop diagram number 789
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZUP,AMPL(1,903))
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZUP,AMPL(1,904))
C     Counter-term amplitude(s) for loop diagram number 807
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZUP,AMPL(1,905))
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZUP,AMPL(1,906))
C     Counter-term amplitude(s) for loop diagram number 829
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZUP,AMPL(1,907))
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZUP,AMPL(1,908))
C     Counter-term amplitude(s) for loop diagram number 831
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZUP,AMPL(1,909))
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZUP,AMPL(1,910))
C     Counter-term amplitude(s) for loop diagram number 833
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZUP,AMPL(1,911))
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZUP,AMPL(1,912))
C     Counter-term amplitude(s) for loop diagram number 834
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECUP,AMPL(1,913))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALUP,AMPL(1,914))
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECUP,AMPL(1,915))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALUP,AMPL(1,916))
C     Counter-term amplitude(s) for loop diagram number 835
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZUP,AMPL(1,917))
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZUP,AMPL(1,918))
C     Counter-term amplitude(s) for loop diagram number 837
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZUP,AMPL(1,919))
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZUP,AMPL(1,920))
C     Counter-term amplitude(s) for loop diagram number 838
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZUP,AMPL(1,921))
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZUP,AMPL(1,922))
C     Counter-term amplitude(s) for loop diagram number 839
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZUP,AMPL(1,923))
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZUP,AMPL(1,924))
C     Counter-term amplitude(s) for loop diagram number 849
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZUP,AMPL(1,925))
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZUP,AMPL(1,926))
C     Counter-term amplitude(s) for loop diagram number 851
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZUP,AMPL(1,927))
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZUP,AMPL(1,928))
C     Counter-term amplitude(s) for loop diagram number 853
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZUP,AMPL(1,929))
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZUP,AMPL(1,930))
C     Counter-term amplitude(s) for loop diagram number 854
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECUP,AMPL(1,931))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALUP,AMPL(1,932))
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECUP,AMPL(1,933))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALUP,AMPL(1,934))
C     Counter-term amplitude(s) for loop diagram number 855
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZUP,AMPL(1,935))
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZUP,AMPL(1,936))
C     Counter-term amplitude(s) for loop diagram number 857
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZUP,AMPL(1,937))
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZUP,AMPL(1,938))
C     Counter-term amplitude(s) for loop diagram number 858
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZUP,AMPL(1,939))
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZUP,AMPL(1,940))
C     Counter-term amplitude(s) for loop diagram number 859
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZUP,AMPL(1,941))
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZUP,AMPL(1,942))
C     Counter-term amplitude(s) for loop diagram number 869
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZUP,AMPL(1,943))
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZUP,AMPL(1,944))
C     Counter-term amplitude(s) for loop diagram number 871
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZUP,AMPL(1,945))
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZUP,AMPL(1,946))
C     Counter-term amplitude(s) for loop diagram number 873
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZUP,AMPL(1,947))
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZUP,AMPL(1,948))
C     Counter-term amplitude(s) for loop diagram number 875
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZUP,AMPL(1,949))
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZUP,AMPL(1,950))
C     Counter-term amplitude(s) for loop diagram number 881
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZUP,AMPL(1,951))
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZUP,AMPL(1,952))
C     Counter-term amplitude(s) for loop diagram number 883
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZUP,AMPL(1,953))
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZUP,AMPL(1,954))
C     Counter-term amplitude(s) for loop diagram number 885
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZUP,AMPL(1,955))
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZUP,AMPL(1,956))
C     Counter-term amplitude(s) for loop diagram number 887
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZUP,AMPL(1,957))
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZUP,AMPL(1,958))
C     Counter-term amplitude(s) for loop diagram number 889
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZUP,AMPL(1,959))
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZUP,AMPL(1,960))
C     Counter-term amplitude(s) for loop diagram number 895
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZUP,AMPL(1,961))
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZUP,AMPL(1,962))
C     Counter-term amplitude(s) for loop diagram number 897
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZUP,AMPL(1,963))
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZUP,AMPL(1,964))
C     Counter-term amplitude(s) for loop diagram number 898
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECUP,AMPL(1,965))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALUP,AMPL(1,966))
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECUP,AMPL(1,967))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALUP,AMPL(1,968))
C     Counter-term amplitude(s) for loop diagram number 899
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZUP,AMPL(1,969))
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZUP,AMPL(1,970))
C     Counter-term amplitude(s) for loop diagram number 901
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZUP,AMPL(1,971))
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZUP,AMPL(1,972))
C     Counter-term amplitude(s) for loop diagram number 902
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZUP,AMPL(1,973))
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZUP,AMPL(1,974))
C     Counter-term amplitude(s) for loop diagram number 903
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZUP,AMPL(1,975))
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZUP,AMPL(1,976))
C     Counter-term amplitude(s) for loop diagram number 913
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZUP,AMPL(1,977))
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZUP,AMPL(1,978))
C     Counter-term amplitude(s) for loop diagram number 914
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECUP,AMPL(1,979))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALUP,AMPL(1,980))
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECUP,AMPL(1,981))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALUP,AMPL(1,982))
C     Counter-term amplitude(s) for loop diagram number 915
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZUP,AMPL(1,983))
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZUP,AMPL(1,984))
C     Counter-term amplitude(s) for loop diagram number 917
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZUP,AMPL(1,985))
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZUP,AMPL(1,986))
C     Counter-term amplitude(s) for loop diagram number 918
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZUP,AMPL(1,987))
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZUP,AMPL(1,988))
C     Counter-term amplitude(s) for loop diagram number 919
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZUP,AMPL(1,989))
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZUP,AMPL(1,990))
C     Counter-term amplitude(s) for loop diagram number 929
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZUP,AMPL(1,991))
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZUP,AMPL(1,992))
C     Counter-term amplitude(s) for loop diagram number 931
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZUP,AMPL(1,993))
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZUP,AMPL(1,994))
C     Counter-term amplitude(s) for loop diagram number 937
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZUP,AMPL(1,995))
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZUP,AMPL(1,996))
C     Counter-term amplitude(s) for loop diagram number 939
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZUP,AMPL(1,997))
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZUP,AMPL(1,998))
C     Counter-term amplitude(s) for loop diagram number 941
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZUP,AMPL(1,999))
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZUP,AMPL(1,1000))
C     Counter-term amplitude(s) for loop diagram number 947
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZUP,AMPL(1,1001))
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZUP,AMPL(1,1002))
      CALL VVS1_3(W(3),W(4),GC_32,MDL_MH,MDL_WH,W(147))
C     Counter-term amplitude(s) for loop diagram number 949
      CALL VVS1_0(W(6),W(105),W(147),R2_GGHB,AMPL(1,1003))
C     Counter-term amplitude(s) for loop diagram number 951
      CALL VVS1_0(W(5),W(106),W(147),R2_GGHB,AMPL(1,1004))
C     Counter-term amplitude(s) for loop diagram number 957
      CALL VVS1_0(W(104),W(11),W(147),R2_GGHB,AMPL(1,1005))
C     Counter-term amplitude(s) for loop diagram number 961
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZDOWN,AMPL(1,1006))
C     Counter-term amplitude(s) for loop diagram number 963
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZDOWN,AMPL(1,1007))
C     Counter-term amplitude(s) for loop diagram number 981
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZDOWN,AMPL(1,1008))
C     Counter-term amplitude(s) for loop diagram number 1003
      CALL R2_GG_1_R2_GG_3_0(W(11),W(35),R2_GGQ,R2_GGB,AMPL(1,1009))
C     Counter-term amplitude(s) for loop diagram number 1004
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,1010))
C     Counter-term amplitude(s) for loop diagram number 1006
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZDOWN,AMPL(1,1011))
C     Counter-term amplitude(s) for loop diagram number 1008
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZDOWN,AMPL(1,1012))
C     Counter-term amplitude(s) for loop diagram number 1010
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZDOWN,AMPL(1,1013))
C     Counter-term amplitude(s) for loop diagram number 1011
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECDOWN,AMPL(1,1014))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1015))
C     Counter-term amplitude(s) for loop diagram number 1012
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZDOWN,AMPL(1,1016))
C     Counter-term amplitude(s) for loop diagram number 1014
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZDOWN,AMPL(1,1017))
C     Counter-term amplitude(s) for loop diagram number 1015
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZDOWN,AMPL(1,1018))
C     Counter-term amplitude(s) for loop diagram number 1016
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZDOWN,AMPL(1,1019))
C     Counter-term amplitude(s) for loop diagram number 1026
      CALL R2_GG_1_R2_GG_3_0(W(11),W(51),R2_GGQ,R2_GGB,AMPL(1,1020))
C     Counter-term amplitude(s) for loop diagram number 1027
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,1021))
C     Counter-term amplitude(s) for loop diagram number 1029
      CALL R2_GG_1_R2_GG_3_0(W(11),W(59),R2_GGQ,R2_GGB,AMPL(1,1022))
C     Counter-term amplitude(s) for loop diagram number 1030
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,1023))
C     Counter-term amplitude(s) for loop diagram number 1032
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZDOWN,AMPL(1,1024))
C     Counter-term amplitude(s) for loop diagram number 1034
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZDOWN,AMPL(1,1025))
C     Counter-term amplitude(s) for loop diagram number 1036
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZDOWN,AMPL(1,1026))
C     Counter-term amplitude(s) for loop diagram number 1037
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECDOWN,AMPL(1,1027))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1028))
C     Counter-term amplitude(s) for loop diagram number 1038
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZDOWN,AMPL(1,1029))
C     Counter-term amplitude(s) for loop diagram number 1040
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZDOWN,AMPL(1,1030))
C     Counter-term amplitude(s) for loop diagram number 1041
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZDOWN,AMPL(1,1031))
C     Counter-term amplitude(s) for loop diagram number 1042
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZDOWN,AMPL(1,1032))
C     Counter-term amplitude(s) for loop diagram number 1052
      CALL R2_GG_1_R2_GG_3_0(W(11),W(72),R2_GGQ,R2_GGB,AMPL(1,1033))
C     Counter-term amplitude(s) for loop diagram number 1053
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,1034))
C     Counter-term amplitude(s) for loop diagram number 1055
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZDOWN,AMPL(1,1035))
C     Counter-term amplitude(s) for loop diagram number 1057
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZDOWN,AMPL(1,1036))
C     Counter-term amplitude(s) for loop diagram number 1059
      CALL VVS1_0(W(6),W(123),W(147),R2_GGHB,AMPL(1,1037))
C     Counter-term amplitude(s) for loop diagram number 1061
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZDOWN,AMPL(1,1038))
C     Counter-term amplitude(s) for loop diagram number 1063
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZDOWN,AMPL(1,1039))
C     Counter-term amplitude(s) for loop diagram number 1069
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZDOWN,AMPL(1,1040))
C     Counter-term amplitude(s) for loop diagram number 1071
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZDOWN,AMPL(1,1041))
C     Counter-term amplitude(s) for loop diagram number 1073
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZDOWN,AMPL(1,1042))
C     Counter-term amplitude(s) for loop diagram number 1075
      CALL VVS1_0(W(5),W(128),W(147),R2_GGHB,AMPL(1,1043))
C     Counter-term amplitude(s) for loop diagram number 1077
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZDOWN,AMPL(1,1044))
C     Counter-term amplitude(s) for loop diagram number 1079
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZDOWN,AMPL(1,1045))
C     Counter-term amplitude(s) for loop diagram number 1085
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZDOWN,AMPL(1,1046))
C     Counter-term amplitude(s) for loop diagram number 1087
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZDOWN,AMPL(1,1047))
C     Counter-term amplitude(s) for loop diagram number 1088
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECDOWN,AMPL(1,1048))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1049))
C     Counter-term amplitude(s) for loop diagram number 1089
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZDOWN,AMPL(1,1050))
C     Counter-term amplitude(s) for loop diagram number 1091
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZDOWN,AMPL(1,1051))
C     Counter-term amplitude(s) for loop diagram number 1092
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZDOWN,AMPL(1,1052))
C     Counter-term amplitude(s) for loop diagram number 1093
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZDOWN,AMPL(1,1053))
C     Counter-term amplitude(s) for loop diagram number 1103
      CALL R2_GG_1_R2_GG_3_0(W(11),W(101),R2_GGQ,R2_GGB,AMPL(1,1054))
C     Counter-term amplitude(s) for loop diagram number 1104
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,1055))
C     Counter-term amplitude(s) for loop diagram number 1106
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZDOWN,AMPL(1,1056))
C     Counter-term amplitude(s) for loop diagram number 1107
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECDOWN,AMPL(1,1057))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALDOWN,AMPL(1
     $ ,1058))
C     Counter-term amplitude(s) for loop diagram number 1108
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZDOWN,AMPL(1,1059))
C     Counter-term amplitude(s) for loop diagram number 1110
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZDOWN,AMPL(1,1060))
C     Counter-term amplitude(s) for loop diagram number 1111
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZDOWN,AMPL(1,1061))
C     Counter-term amplitude(s) for loop diagram number 1112
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZDOWN,AMPL(1,1062))
C     Counter-term amplitude(s) for loop diagram number 1122
      CALL R2_GG_1_R2_GG_3_0(W(11),W(103),R2_GGQ,R2_GGB,AMPL(1,1063))
C     Counter-term amplitude(s) for loop diagram number 1123
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,1064))
C     Counter-term amplitude(s) for loop diagram number 1125
      CALL VVS1_0(W(6),W(141),W(147),R2_GGHB,AMPL(1,1065))
C     Counter-term amplitude(s) for loop diagram number 1127
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZDOWN,AMPL(1,1066))
C     Counter-term amplitude(s) for loop diagram number 1129
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZDOWN,AMPL(1,1067))
C     Counter-term amplitude(s) for loop diagram number 1135
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZDOWN,AMPL(1,1068))
C     Counter-term amplitude(s) for loop diagram number 1137
      CALL VVS1_0(W(5),W(144),W(147),R2_GGHB,AMPL(1,1069))
C     Counter-term amplitude(s) for loop diagram number 1139
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZDOWN,AMPL(1,1070))
C     Counter-term amplitude(s) for loop diagram number 1141
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZDOWN,AMPL(1,1071))
C     Counter-term amplitude(s) for loop diagram number 1147
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZDOWN,AMPL(1,1072))
C     Counter-term amplitude(s) for loop diagram number 1149
      CALL VVS1_0(W(6),W(105),W(147),R2_GGHT,AMPL(1,1073))
C     Counter-term amplitude(s) for loop diagram number 1151
      CALL VVS1_0(W(5),W(106),W(147),R2_GGHT,AMPL(1,1074))
C     Counter-term amplitude(s) for loop diagram number 1157
      CALL VVS1_0(W(104),W(11),W(147),R2_GGHT,AMPL(1,1075))
C     Counter-term amplitude(s) for loop diagram number 1161
      CALL R2_GGVV_0(W(6),W(105),W(3),W(4),R2_GGZZUP,AMPL(1,1076))
C     Counter-term amplitude(s) for loop diagram number 1163
      CALL R2_GGVV_0(W(5),W(106),W(3),W(4),R2_GGZZUP,AMPL(1,1077))
C     Counter-term amplitude(s) for loop diagram number 1181
      CALL R2_GGVV_0(W(11),W(104),W(3),W(4),R2_GGZZUP,AMPL(1,1078))
C     Counter-term amplitude(s) for loop diagram number 1203
      CALL R2_GG_1_R2_GG_3_0(W(11),W(35),R2_GGQ,R2_GGT,AMPL(1,1079))
C     Counter-term amplitude(s) for loop diagram number 1204
      CALL VVV1_0(W(5),W(6),W(35),R2_3GQ,AMPL(1,1080))
C     Counter-term amplitude(s) for loop diagram number 1206
      CALL R2_GGZ_0(W(6),W(107),W(4),R2_GGZUP,AMPL(1,1081))
C     Counter-term amplitude(s) for loop diagram number 1208
      CALL R2_GGZ_0(W(5),W(108),W(4),R2_GGZUP,AMPL(1,1082))
C     Counter-term amplitude(s) for loop diagram number 1210
      CALL R2_GGZ_0(W(5),W(110),W(4),R2_GGZUP,AMPL(1,1083))
C     Counter-term amplitude(s) for loop diagram number 1211
      CALL R2_GGVV_0(W(5),W(6),W(109),W(4),R2_GGGZVECUP,AMPL(1,1084))
      CALL R2_GGGVA_0(W(5),W(6),W(109),W(4),R2_GGGZAXIALUP,AMPL(1,1085)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1212
      CALL R2_GGZ_0(W(6),W(111),W(4),R2_GGZUP,AMPL(1,1086))
C     Counter-term amplitude(s) for loop diagram number 1214
      CALL R2_GGZ_0(W(6),W(112),W(4),R2_GGZUP,AMPL(1,1087))
C     Counter-term amplitude(s) for loop diagram number 1215
      CALL R2_GGZ_0(W(5),W(113),W(4),R2_GGZUP,AMPL(1,1088))
C     Counter-term amplitude(s) for loop diagram number 1216
      CALL R2_GGZ_0(W(11),W(109),W(4),R2_GGZUP,AMPL(1,1089))
C     Counter-term amplitude(s) for loop diagram number 1226
      CALL R2_GG_1_R2_GG_3_0(W(11),W(51),R2_GGQ,R2_GGT,AMPL(1,1090))
C     Counter-term amplitude(s) for loop diagram number 1227
      CALL VVV1_0(W(5),W(6),W(51),R2_3GQ,AMPL(1,1091))
C     Counter-term amplitude(s) for loop diagram number 1229
      CALL R2_GG_1_R2_GG_3_0(W(11),W(59),R2_GGQ,R2_GGT,AMPL(1,1092))
C     Counter-term amplitude(s) for loop diagram number 1230
      CALL VVV1_0(W(5),W(6),W(59),R2_3GQ,AMPL(1,1093))
C     Counter-term amplitude(s) for loop diagram number 1232
      CALL R2_GGZ_0(W(6),W(114),W(3),R2_GGZUP,AMPL(1,1094))
C     Counter-term amplitude(s) for loop diagram number 1234
      CALL R2_GGZ_0(W(5),W(115),W(3),R2_GGZUP,AMPL(1,1095))
C     Counter-term amplitude(s) for loop diagram number 1236
      CALL R2_GGZ_0(W(5),W(117),W(3),R2_GGZUP,AMPL(1,1096))
C     Counter-term amplitude(s) for loop diagram number 1237
      CALL R2_GGVV_0(W(5),W(6),W(116),W(3),R2_GGGZVECUP,AMPL(1,1097))
      CALL R2_GGGVA_0(W(5),W(6),W(116),W(3),R2_GGGZAXIALUP,AMPL(1,1098)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1238
      CALL R2_GGZ_0(W(6),W(118),W(3),R2_GGZUP,AMPL(1,1099))
C     Counter-term amplitude(s) for loop diagram number 1240
      CALL R2_GGZ_0(W(6),W(119),W(3),R2_GGZUP,AMPL(1,1100))
C     Counter-term amplitude(s) for loop diagram number 1241
      CALL R2_GGZ_0(W(5),W(120),W(3),R2_GGZUP,AMPL(1,1101))
C     Counter-term amplitude(s) for loop diagram number 1242
      CALL R2_GGZ_0(W(11),W(116),W(3),R2_GGZUP,AMPL(1,1102))
C     Counter-term amplitude(s) for loop diagram number 1252
      CALL R2_GG_1_R2_GG_3_0(W(11),W(72),R2_GGQ,R2_GGT,AMPL(1,1103))
C     Counter-term amplitude(s) for loop diagram number 1253
      CALL VVV1_0(W(5),W(6),W(72),R2_3GQ,AMPL(1,1104))
C     Counter-term amplitude(s) for loop diagram number 1255
      CALL R2_GGZ_0(W(6),W(121),W(4),R2_GGZUP,AMPL(1,1105))
C     Counter-term amplitude(s) for loop diagram number 1257
      CALL R2_GGZ_0(W(6),W(122),W(3),R2_GGZUP,AMPL(1,1106))
C     Counter-term amplitude(s) for loop diagram number 1259
      CALL VVS1_0(W(6),W(123),W(147),R2_GGHT,AMPL(1,1107))
C     Counter-term amplitude(s) for loop diagram number 1261
      CALL R2_GGVV_0(W(6),W(123),W(3),W(4),R2_GGZZUP,AMPL(1,1108))
C     Counter-term amplitude(s) for loop diagram number 1263
      CALL R2_GGZ_0(W(6),W(124),W(3),R2_GGZUP,AMPL(1,1109))
C     Counter-term amplitude(s) for loop diagram number 1269
      CALL R2_GGZ_0(W(6),W(125),W(4),R2_GGZUP,AMPL(1,1110))
C     Counter-term amplitude(s) for loop diagram number 1271
      CALL R2_GGZ_0(W(5),W(126),W(4),R2_GGZUP,AMPL(1,1111))
C     Counter-term amplitude(s) for loop diagram number 1273
      CALL R2_GGZ_0(W(5),W(127),W(3),R2_GGZUP,AMPL(1,1112))
C     Counter-term amplitude(s) for loop diagram number 1275
      CALL VVS1_0(W(5),W(128),W(147),R2_GGHT,AMPL(1,1113))
C     Counter-term amplitude(s) for loop diagram number 1277
      CALL R2_GGVV_0(W(5),W(128),W(3),W(4),R2_GGZZUP,AMPL(1,1114))
C     Counter-term amplitude(s) for loop diagram number 1279
      CALL R2_GGZ_0(W(5),W(129),W(3),R2_GGZUP,AMPL(1,1115))
C     Counter-term amplitude(s) for loop diagram number 1285
      CALL R2_GGZ_0(W(5),W(130),W(4),R2_GGZUP,AMPL(1,1116))
C     Counter-term amplitude(s) for loop diagram number 1287
      CALL R2_GGZ_0(W(5),W(132),W(4),R2_GGZUP,AMPL(1,1117))
C     Counter-term amplitude(s) for loop diagram number 1288
      CALL R2_GGVV_0(W(5),W(6),W(131),W(4),R2_GGGZVECUP,AMPL(1,1118))
      CALL R2_GGGVA_0(W(5),W(6),W(131),W(4),R2_GGGZAXIALUP,AMPL(1,1119)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1289
      CALL R2_GGZ_0(W(6),W(133),W(4),R2_GGZUP,AMPL(1,1120))
C     Counter-term amplitude(s) for loop diagram number 1291
      CALL R2_GGZ_0(W(6),W(134),W(4),R2_GGZUP,AMPL(1,1121))
C     Counter-term amplitude(s) for loop diagram number 1292
      CALL R2_GGZ_0(W(5),W(135),W(4),R2_GGZUP,AMPL(1,1122))
C     Counter-term amplitude(s) for loop diagram number 1293
      CALL R2_GGZ_0(W(11),W(131),W(4),R2_GGZUP,AMPL(1,1123))
C     Counter-term amplitude(s) for loop diagram number 1303
      CALL R2_GG_1_R2_GG_3_0(W(11),W(101),R2_GGQ,R2_GGT,AMPL(1,1124))
C     Counter-term amplitude(s) for loop diagram number 1304
      CALL VVV1_0(W(5),W(6),W(101),R2_3GQ,AMPL(1,1125))
C     Counter-term amplitude(s) for loop diagram number 1306
      CALL R2_GGZ_0(W(5),W(137),W(3),R2_GGZUP,AMPL(1,1126))
C     Counter-term amplitude(s) for loop diagram number 1307
      CALL R2_GGVV_0(W(5),W(6),W(136),W(3),R2_GGGZVECUP,AMPL(1,1127))
      CALL R2_GGGVA_0(W(5),W(6),W(136),W(3),R2_GGGZAXIALUP,AMPL(1,1128)
     $ )
C     Counter-term amplitude(s) for loop diagram number 1308
      CALL R2_GGZ_0(W(6),W(138),W(3),R2_GGZUP,AMPL(1,1129))
C     Counter-term amplitude(s) for loop diagram number 1310
      CALL R2_GGZ_0(W(6),W(139),W(3),R2_GGZUP,AMPL(1,1130))
C     Counter-term amplitude(s) for loop diagram number 1311
      CALL R2_GGZ_0(W(5),W(140),W(3),R2_GGZUP,AMPL(1,1131))
C     Counter-term amplitude(s) for loop diagram number 1312
      CALL R2_GGZ_0(W(11),W(136),W(3),R2_GGZUP,AMPL(1,1132))
C     Counter-term amplitude(s) for loop diagram number 1322
      CALL R2_GG_1_R2_GG_3_0(W(11),W(103),R2_GGQ,R2_GGT,AMPL(1,1133))
C     Counter-term amplitude(s) for loop diagram number 1323
      CALL VVV1_0(W(5),W(6),W(103),R2_3GQ,AMPL(1,1134))
C     Counter-term amplitude(s) for loop diagram number 1325
      CALL VVS1_0(W(6),W(141),W(147),R2_GGHT,AMPL(1,1135))
C     Counter-term amplitude(s) for loop diagram number 1327
      CALL R2_GGVV_0(W(6),W(141),W(3),W(4),R2_GGZZUP,AMPL(1,1136))
C     Counter-term amplitude(s) for loop diagram number 1329
      CALL R2_GGZ_0(W(6),W(142),W(3),R2_GGZUP,AMPL(1,1137))
C     Counter-term amplitude(s) for loop diagram number 1335
      CALL R2_GGZ_0(W(6),W(143),W(4),R2_GGZUP,AMPL(1,1138))
C     Counter-term amplitude(s) for loop diagram number 1337
      CALL VVS1_0(W(5),W(144),W(147),R2_GGHT,AMPL(1,1139))
C     Counter-term amplitude(s) for loop diagram number 1339
      CALL R2_GGVV_0(W(5),W(144),W(3),W(4),R2_GGZZUP,AMPL(1,1140))
C     Counter-term amplitude(s) for loop diagram number 1341
      CALL R2_GGZ_0(W(5),W(145),W(3),R2_GGZUP,AMPL(1,1141))
C     Counter-term amplitude(s) for loop diagram number 1347
      CALL R2_GGZ_0(W(5),W(146),W(4),R2_GGZUP,AMPL(1,1142))
C     At this point, all CT amps needed for (QCD=6), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

