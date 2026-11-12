      SUBROUTINE ML5_0_MP_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
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
      CALL MP_VXXXXX(P(0,6),ZERO,NHEL(6),+1,W(6))
      CALL MP_FFV2_5_2(W(1),W(3),GC_22,GC_23,ZERO,ZERO,W(7))
      CALL MP_FFV1_1(W(2),W(4),GC_5,ZERO,ZERO,W(8))
      CALL MP_FFV1_2(W(7),W(5),GC_5,ZERO,ZERO,W(9))
C     Amplitude(s) for born diagram with ID 1
      CALL MP_FFV1_0(W(9),W(8),W(6),GC_5,AMP(1))
      CALL MP_FFV1_2(W(7),W(6),GC_5,ZERO,ZERO,W(10))
C     Amplitude(s) for born diagram with ID 2
      CALL MP_FFV1_0(W(10),W(8),W(5),GC_5,AMP(2))
      CALL MP_VVV1P0_1(W(5),W(6),GC_4,ZERO,ZERO,W(11))
C     Amplitude(s) for born diagram with ID 3
      CALL MP_FFV1_0(W(7),W(8),W(11),GC_5,AMP(3))
      CALL MP_FFV1_1(W(2),W(5),GC_5,ZERO,ZERO,W(12))
      CALL MP_FFV1_2(W(7),W(4),GC_5,ZERO,ZERO,W(13))
C     Amplitude(s) for born diagram with ID 4
      CALL MP_FFV1_0(W(13),W(12),W(6),GC_5,AMP(4))
C     Amplitude(s) for born diagram with ID 5
      CALL MP_FFV1_0(W(10),W(12),W(4),GC_5,AMP(5))
      CALL MP_VVV1P0_1(W(4),W(6),GC_4,ZERO,ZERO,W(14))
C     Amplitude(s) for born diagram with ID 6
      CALL MP_FFV1_0(W(7),W(12),W(14),GC_5,AMP(6))
      CALL MP_FFV1_1(W(2),W(6),GC_5,ZERO,ZERO,W(15))
C     Amplitude(s) for born diagram with ID 7
      CALL MP_FFV1_0(W(13),W(15),W(5),GC_5,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL MP_FFV1_0(W(9),W(15),W(4),GC_5,AMP(8))
      CALL MP_VVV1P0_1(W(4),W(5),GC_4,ZERO,ZERO,W(16))
C     Amplitude(s) for born diagram with ID 9
      CALL MP_FFV1_0(W(7),W(15),W(16),GC_5,AMP(9))
      CALL MP_FFV1P0_3(W(7),W(2),GC_5,ZERO,ZERO,W(17))
C     Amplitude(s) for born diagram with ID 10
      CALL MP_VVV1_0(W(16),W(6),W(17),GC_4,AMP(10))
C     Amplitude(s) for born diagram with ID 11
      CALL MP_FFV1_0(W(10),W(2),W(16),GC_5,AMP(11))
C     Amplitude(s) for born diagram with ID 12
      CALL MP_VVV1_0(W(14),W(5),W(17),GC_4,AMP(12))
C     Amplitude(s) for born diagram with ID 13
      CALL MP_FFV1_0(W(9),W(2),W(14),GC_5,AMP(13))
C     Amplitude(s) for born diagram with ID 14
      CALL MP_VVV1_0(W(4),W(11),W(17),GC_4,AMP(14))
C     Amplitude(s) for born diagram with ID 15
      CALL MP_FFV1_0(W(13),W(2),W(11),GC_5,AMP(15))
      CALL MP_VVVV1P0_1(W(4),W(5),W(6),GC_6,ZERO,ZERO,W(18))
      CALL MP_VVVV3P0_1(W(4),W(5),W(6),GC_6,ZERO,ZERO,W(19))
      CALL MP_VVVV4P0_1(W(4),W(5),W(6),GC_6,ZERO,ZERO,W(20))
C     Amplitude(s) for born diagram with ID 16
      CALL MP_FFV1_0(W(7),W(2),W(18),GC_5,AMP(16))
      CALL MP_FFV1_0(W(7),W(2),W(19),GC_5,AMP(17))
      CALL MP_FFV1_0(W(7),W(2),W(20),GC_5,AMP(18))
      CALL MP_FFV1_2(W(1),W(4),GC_5,ZERO,ZERO,W(21))
      CALL MP_FFV2_5_1(W(2),W(3),GC_22,GC_23,ZERO,ZERO,W(22))
      CALL MP_FFV1_2(W(21),W(5),GC_5,ZERO,ZERO,W(23))
C     Amplitude(s) for born diagram with ID 17
      CALL MP_FFV1_0(W(23),W(22),W(6),GC_5,AMP(19))
      CALL MP_FFV1_2(W(21),W(6),GC_5,ZERO,ZERO,W(24))
C     Amplitude(s) for born diagram with ID 18
      CALL MP_FFV1_0(W(24),W(22),W(5),GC_5,AMP(20))
C     Amplitude(s) for born diagram with ID 19
      CALL MP_FFV1_0(W(21),W(22),W(11),GC_5,AMP(21))
      CALL MP_FFV2_5_2(W(21),W(3),GC_22,GC_23,ZERO,ZERO,W(25))
C     Amplitude(s) for born diagram with ID 20
      CALL MP_FFV1_0(W(25),W(12),W(6),GC_5,AMP(22))
C     Amplitude(s) for born diagram with ID 21
      CALL MP_FFV2_5_0(W(24),W(12),W(3),GC_22,GC_23,AMP(23))
C     Amplitude(s) for born diagram with ID 22
      CALL MP_FFV1_0(W(25),W(15),W(5),GC_5,AMP(24))
C     Amplitude(s) for born diagram with ID 23
      CALL MP_FFV2_5_0(W(23),W(15),W(3),GC_22,GC_23,AMP(25))
C     Amplitude(s) for born diagram with ID 24
      CALL MP_FFV1_0(W(25),W(2),W(11),GC_5,AMP(26))
      CALL MP_FFV1_2(W(1),W(5),GC_5,ZERO,ZERO,W(26))
      CALL MP_FFV1_2(W(26),W(4),GC_5,ZERO,ZERO,W(27))
C     Amplitude(s) for born diagram with ID 25
      CALL MP_FFV1_0(W(27),W(22),W(6),GC_5,AMP(27))
      CALL MP_FFV1_2(W(26),W(6),GC_5,ZERO,ZERO,W(28))
C     Amplitude(s) for born diagram with ID 26
      CALL MP_FFV1_0(W(28),W(22),W(4),GC_5,AMP(28))
C     Amplitude(s) for born diagram with ID 27
      CALL MP_FFV1_0(W(26),W(22),W(14),GC_5,AMP(29))
      CALL MP_FFV2_5_2(W(26),W(3),GC_22,GC_23,ZERO,ZERO,W(29))
C     Amplitude(s) for born diagram with ID 28
      CALL MP_FFV1_0(W(29),W(8),W(6),GC_5,AMP(30))
C     Amplitude(s) for born diagram with ID 29
      CALL MP_FFV2_5_0(W(28),W(8),W(3),GC_22,GC_23,AMP(31))
C     Amplitude(s) for born diagram with ID 30
      CALL MP_FFV1_0(W(29),W(15),W(4),GC_5,AMP(32))
C     Amplitude(s) for born diagram with ID 31
      CALL MP_FFV2_5_0(W(27),W(15),W(3),GC_22,GC_23,AMP(33))
C     Amplitude(s) for born diagram with ID 32
      CALL MP_FFV1_0(W(29),W(2),W(14),GC_5,AMP(34))
      CALL MP_FFV1_2(W(1),W(6),GC_5,ZERO,ZERO,W(30))
      CALL MP_FFV1_2(W(30),W(4),GC_5,ZERO,ZERO,W(31))
C     Amplitude(s) for born diagram with ID 33
      CALL MP_FFV1_0(W(31),W(22),W(5),GC_5,AMP(35))
      CALL MP_FFV1_2(W(30),W(5),GC_5,ZERO,ZERO,W(32))
C     Amplitude(s) for born diagram with ID 34
      CALL MP_FFV1_0(W(32),W(22),W(4),GC_5,AMP(36))
C     Amplitude(s) for born diagram with ID 35
      CALL MP_FFV1_0(W(30),W(22),W(16),GC_5,AMP(37))
      CALL MP_FFV2_5_2(W(30),W(3),GC_22,GC_23,ZERO,ZERO,W(33))
C     Amplitude(s) for born diagram with ID 36
      CALL MP_FFV1_0(W(33),W(8),W(5),GC_5,AMP(38))
C     Amplitude(s) for born diagram with ID 37
      CALL MP_FFV2_5_0(W(32),W(8),W(3),GC_22,GC_23,AMP(39))
C     Amplitude(s) for born diagram with ID 38
      CALL MP_FFV1_0(W(33),W(12),W(4),GC_5,AMP(40))
C     Amplitude(s) for born diagram with ID 39
      CALL MP_FFV2_5_0(W(31),W(12),W(3),GC_22,GC_23,AMP(41))
C     Amplitude(s) for born diagram with ID 40
      CALL MP_FFV1_0(W(33),W(2),W(16),GC_5,AMP(42))
      CALL MP_FFV1P0_3(W(1),W(22),GC_5,ZERO,ZERO,W(34))
C     Amplitude(s) for born diagram with ID 41
      CALL MP_VVV1_0(W(16),W(6),W(34),GC_4,AMP(43))
      CALL MP_FFV1_2(W(1),W(16),GC_5,ZERO,ZERO,W(35))
C     Amplitude(s) for born diagram with ID 42
      CALL MP_FFV1_0(W(35),W(22),W(6),GC_5,AMP(44))
C     Amplitude(s) for born diagram with ID 43
      CALL MP_VVV1_0(W(14),W(5),W(34),GC_4,AMP(45))
      CALL MP_FFV1_2(W(1),W(14),GC_5,ZERO,ZERO,W(36))
C     Amplitude(s) for born diagram with ID 44
      CALL MP_FFV1_0(W(36),W(22),W(5),GC_5,AMP(46))
C     Amplitude(s) for born diagram with ID 45
      CALL MP_VVV1_0(W(4),W(11),W(34),GC_4,AMP(47))
      CALL MP_FFV1_2(W(1),W(11),GC_5,ZERO,ZERO,W(37))
C     Amplitude(s) for born diagram with ID 46
      CALL MP_FFV1_0(W(37),W(22),W(4),GC_5,AMP(48))
C     Amplitude(s) for born diagram with ID 47
      CALL MP_FFV1_0(W(1),W(22),W(18),GC_5,AMP(49))
      CALL MP_FFV1_0(W(1),W(22),W(19),GC_5,AMP(50))
      CALL MP_FFV1_0(W(1),W(22),W(20),GC_5,AMP(51))
C     Amplitude(s) for born diagram with ID 48
      CALL MP_FFV2_5_0(W(37),W(8),W(3),GC_22,GC_23,AMP(52))
C     Amplitude(s) for born diagram with ID 49
      CALL MP_FFV2_5_0(W(36),W(12),W(3),GC_22,GC_23,AMP(53))
C     Amplitude(s) for born diagram with ID 50
      CALL MP_FFV2_5_0(W(35),W(15),W(3),GC_22,GC_23,AMP(54))
      CALL MP_FFV1_1(W(8),W(6),GC_5,ZERO,ZERO,W(38))
C     Counter-term amplitude(s) for loop diagram number 51
      CALL MP_R2_QQ_1_0(W(9),W(38),R2_QQQ,AMPL(1,1))
      CALL MP_FFV1_2(W(9),W(6),GC_5,ZERO,ZERO,W(39))
C     Counter-term amplitude(s) for loop diagram number 52
      CALL MP_R2_QQ_1_0(W(39),W(8),R2_QQQ,AMPL(1,2))
C     Counter-term amplitude(s) for loop diagram number 53
      CALL MP_FFV1_0(W(9),W(8),W(6),R2_GQQ,AMPL(1,3))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,4))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,5))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,6))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,7))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,8))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB_1EPS,AMPL(2,9))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQG_1EPS,AMPL(2,10))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQB,AMPL(1,11))
      CALL MP_FFV1_0(W(9),W(8),W(6),UV_GQQT,AMPL(1,12))
      CALL MP_FFV1_1(W(8),W(5),GC_5,ZERO,ZERO,W(40))
C     Counter-term amplitude(s) for loop diagram number 54
      CALL MP_R2_QQ_1_0(W(10),W(40),R2_QQQ,AMPL(1,13))
      CALL MP_FFV1_2(W(10),W(5),GC_5,ZERO,ZERO,W(41))
C     Counter-term amplitude(s) for loop diagram number 55
      CALL MP_R2_QQ_1_0(W(41),W(8),R2_QQQ,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 56
      CALL MP_FFV1_0(W(10),W(8),W(5),R2_GQQ,AMPL(1,15))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,16))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,17))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,18))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,19))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,20))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB_1EPS,AMPL(2,21))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQG_1EPS,AMPL(2,22))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQB,AMPL(1,23))
      CALL MP_FFV1_0(W(10),W(8),W(5),UV_GQQT,AMPL(1,24))
C     Counter-term amplitude(s) for loop diagram number 57
      CALL MP_FFV1_0(W(7),W(40),W(6),R2_GQQ,AMPL(1,25))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,26))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,27))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,28))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,29))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,30))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB_1EPS,AMPL(2,31))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQG_1EPS,AMPL(2,32))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQB,AMPL(1,33))
      CALL MP_FFV1_0(W(7),W(40),W(6),UV_GQQT,AMPL(1,34))
      CALL MP_FFV1_1(W(40),W(6),GC_5,ZERO,ZERO,W(42))
C     Counter-term amplitude(s) for loop diagram number 58
      CALL MP_R2_QQ_1_0(W(7),W(42),R2_QQQ,AMPL(1,35))
C     Counter-term amplitude(s) for loop diagram number 59
      CALL MP_FFV1_0(W(7),W(38),W(5),R2_GQQ,AMPL(1,36))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,37))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,38))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,39))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,40))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,41))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB_1EPS,AMPL(2,42))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQG_1EPS,AMPL(2,43))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQB,AMPL(1,44))
      CALL MP_FFV1_0(W(7),W(38),W(5),UV_GQQT,AMPL(1,45))
      CALL MP_FFV1_1(W(38),W(5),GC_5,ZERO,ZERO,W(43))
C     Counter-term amplitude(s) for loop diagram number 60
      CALL MP_R2_QQ_1_0(W(7),W(43),R2_QQQ,AMPL(1,46))
      CALL MP_FFV1P0_3(W(7),W(8),GC_5,ZERO,ZERO,W(44))
C     Counter-term amplitude(s) for loop diagram number 63
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(44),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,47))
      CALL MP_FFV1_2(W(7),W(11),GC_5,ZERO,ZERO,W(45))
C     Counter-term amplitude(s) for loop diagram number 64
      CALL MP_R2_QQ_1_0(W(45),W(8),R2_QQQ,AMPL(1,48))
      CALL MP_FFV1_1(W(8),W(11),GC_5,ZERO,ZERO,W(46))
C     Counter-term amplitude(s) for loop diagram number 65
      CALL MP_R2_QQ_1_0(W(7),W(46),R2_QQQ,AMPL(1,49))
C     Counter-term amplitude(s) for loop diagram number 66
      CALL MP_FFV1_0(W(7),W(8),W(11),R2_GQQ,AMPL(1,50))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,51))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,52))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,53))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,54))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,55))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB_1EPS,AMPL(2,56))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQG_1EPS,AMPL(2,57))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQB,AMPL(1,58))
      CALL MP_FFV1_0(W(7),W(8),W(11),UV_GQQT,AMPL(1,59))
C     Counter-term amplitude(s) for loop diagram number 68
      CALL MP_VVV1_0(W(5),W(6),W(44),R2_3GG,AMPL(1,60))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,61))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,62))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,63))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,64))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,65))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB_1EPS,AMPL(2,66))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GG_1EPS,AMPL(2,67))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GB,AMPL(1,68))
      CALL MP_VVV1_0(W(5),W(6),W(44),UV_3GT,AMPL(1,69))
      CALL MP_FFV1_1(W(12),W(6),GC_5,ZERO,ZERO,W(47))
C     Counter-term amplitude(s) for loop diagram number 81
      CALL MP_R2_QQ_1_0(W(13),W(47),R2_QQQ,AMPL(1,70))
      CALL MP_FFV1_2(W(13),W(6),GC_5,ZERO,ZERO,W(48))
C     Counter-term amplitude(s) for loop diagram number 82
      CALL MP_R2_QQ_1_0(W(48),W(12),R2_QQQ,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 83
      CALL MP_FFV1_0(W(13),W(12),W(6),R2_GQQ,AMPL(1,72))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,73))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,74))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,75))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,76))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,77))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB_1EPS,AMPL(2,78))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQG_1EPS,AMPL(2,79))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQB,AMPL(1,80))
      CALL MP_FFV1_0(W(13),W(12),W(6),UV_GQQT,AMPL(1,81))
      CALL MP_FFV1_1(W(12),W(4),GC_5,ZERO,ZERO,W(49))
C     Counter-term amplitude(s) for loop diagram number 84
      CALL MP_R2_QQ_1_0(W(10),W(49),R2_QQQ,AMPL(1,82))
      CALL MP_FFV1_2(W(10),W(4),GC_5,ZERO,ZERO,W(50))
C     Counter-term amplitude(s) for loop diagram number 85
      CALL MP_R2_QQ_1_0(W(50),W(12),R2_QQQ,AMPL(1,83))
C     Counter-term amplitude(s) for loop diagram number 86
      CALL MP_FFV1_0(W(10),W(12),W(4),R2_GQQ,AMPL(1,84))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,85))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,86))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,87))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,88))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,89))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB_1EPS,AMPL(2,90))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQG_1EPS,AMPL(2,91))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQB,AMPL(1,92))
      CALL MP_FFV1_0(W(10),W(12),W(4),UV_GQQT,AMPL(1,93))
C     Counter-term amplitude(s) for loop diagram number 87
      CALL MP_FFV1_0(W(7),W(49),W(6),R2_GQQ,AMPL(1,94))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,95))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,96))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,97))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,98))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,99))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB_1EPS,AMPL(2,100))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQG_1EPS,AMPL(2,101))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQB,AMPL(1,102))
      CALL MP_FFV1_0(W(7),W(49),W(6),UV_GQQT,AMPL(1,103))
      CALL MP_FFV1_1(W(49),W(6),GC_5,ZERO,ZERO,W(51))
C     Counter-term amplitude(s) for loop diagram number 88
      CALL MP_R2_QQ_1_0(W(7),W(51),R2_QQQ,AMPL(1,104))
C     Counter-term amplitude(s) for loop diagram number 89
      CALL MP_FFV1_0(W(7),W(47),W(4),R2_GQQ,AMPL(1,105))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,106))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,107))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,108))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,109))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,110))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB_1EPS,AMPL(2,111))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQG_1EPS,AMPL(2,112))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQB,AMPL(1,113))
      CALL MP_FFV1_0(W(7),W(47),W(4),UV_GQQT,AMPL(1,114))
      CALL MP_FFV1_1(W(47),W(4),GC_5,ZERO,ZERO,W(52))
C     Counter-term amplitude(s) for loop diagram number 90
      CALL MP_R2_QQ_1_0(W(7),W(52),R2_QQQ,AMPL(1,115))
      CALL MP_FFV1P0_3(W(7),W(12),GC_5,ZERO,ZERO,W(53))
C     Counter-term amplitude(s) for loop diagram number 93
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(53),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,116))
      CALL MP_FFV1_2(W(7),W(14),GC_5,ZERO,ZERO,W(54))
C     Counter-term amplitude(s) for loop diagram number 94
      CALL MP_R2_QQ_1_0(W(54),W(12),R2_QQQ,AMPL(1,117))
      CALL MP_FFV1_1(W(12),W(14),GC_5,ZERO,ZERO,W(55))
C     Counter-term amplitude(s) for loop diagram number 95
      CALL MP_R2_QQ_1_0(W(7),W(55),R2_QQQ,AMPL(1,118))
C     Counter-term amplitude(s) for loop diagram number 96
      CALL MP_FFV1_0(W(7),W(12),W(14),R2_GQQ,AMPL(1,119))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,120))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,121))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,122))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,123))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,124))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB_1EPS,AMPL(2,125))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQG_1EPS,AMPL(2,126))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQB,AMPL(1,127))
      CALL MP_FFV1_0(W(7),W(12),W(14),UV_GQQT,AMPL(1,128))
C     Counter-term amplitude(s) for loop diagram number 98
      CALL MP_VVV1_0(W(4),W(6),W(53),R2_3GG,AMPL(1,129))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,130))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,131))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,132))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,133))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,134))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB_1EPS,AMPL(2,135))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GG_1EPS,AMPL(2,136))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GB,AMPL(1,137))
      CALL MP_VVV1_0(W(4),W(6),W(53),UV_3GT,AMPL(1,138))
      CALL MP_FFV1_1(W(15),W(5),GC_5,ZERO,ZERO,W(56))
C     Counter-term amplitude(s) for loop diagram number 111
      CALL MP_R2_QQ_1_0(W(13),W(56),R2_QQQ,AMPL(1,139))
      CALL MP_FFV1_2(W(13),W(5),GC_5,ZERO,ZERO,W(57))
C     Counter-term amplitude(s) for loop diagram number 112
      CALL MP_R2_QQ_1_0(W(57),W(15),R2_QQQ,AMPL(1,140))
C     Counter-term amplitude(s) for loop diagram number 113
      CALL MP_FFV1_0(W(13),W(15),W(5),R2_GQQ,AMPL(1,141))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,142))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,143))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,144))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,145))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,146))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB_1EPS,AMPL(2,147))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQG_1EPS,AMPL(2,148))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQB,AMPL(1,149))
      CALL MP_FFV1_0(W(13),W(15),W(5),UV_GQQT,AMPL(1,150))
      CALL MP_FFV1_1(W(15),W(4),GC_5,ZERO,ZERO,W(58))
C     Counter-term amplitude(s) for loop diagram number 114
      CALL MP_R2_QQ_1_0(W(9),W(58),R2_QQQ,AMPL(1,151))
      CALL MP_FFV1_2(W(9),W(4),GC_5,ZERO,ZERO,W(59))
C     Counter-term amplitude(s) for loop diagram number 115
      CALL MP_R2_QQ_1_0(W(59),W(15),R2_QQQ,AMPL(1,152))
C     Counter-term amplitude(s) for loop diagram number 116
      CALL MP_FFV1_0(W(9),W(15),W(4),R2_GQQ,AMPL(1,153))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,154))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,155))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,156))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,157))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,158))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB_1EPS,AMPL(2,159))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQG_1EPS,AMPL(2,160))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQB,AMPL(1,161))
      CALL MP_FFV1_0(W(9),W(15),W(4),UV_GQQT,AMPL(1,162))
C     Counter-term amplitude(s) for loop diagram number 117
      CALL MP_FFV1_0(W(7),W(58),W(5),R2_GQQ,AMPL(1,163))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,164))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,165))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,166))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,167))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,168))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB_1EPS,AMPL(2,169))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQG_1EPS,AMPL(2,170))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQB,AMPL(1,171))
      CALL MP_FFV1_0(W(7),W(58),W(5),UV_GQQT,AMPL(1,172))
      CALL MP_FFV1_1(W(58),W(5),GC_5,ZERO,ZERO,W(60))
C     Counter-term amplitude(s) for loop diagram number 118
      CALL MP_R2_QQ_1_0(W(7),W(60),R2_QQQ,AMPL(1,173))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL MP_FFV1_0(W(7),W(56),W(4),R2_GQQ,AMPL(1,174))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,175))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,176))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,177))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,178))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,179))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB_1EPS,AMPL(2,180))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQG_1EPS,AMPL(2,181))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQB,AMPL(1,182))
      CALL MP_FFV1_0(W(7),W(56),W(4),UV_GQQT,AMPL(1,183))
      CALL MP_FFV1_1(W(56),W(4),GC_5,ZERO,ZERO,W(61))
C     Counter-term amplitude(s) for loop diagram number 120
      CALL MP_R2_QQ_1_0(W(7),W(61),R2_QQQ,AMPL(1,184))
      CALL MP_FFV1P0_3(W(7),W(15),GC_5,ZERO,ZERO,W(62))
C     Counter-term amplitude(s) for loop diagram number 123
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(62),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,185))
      CALL MP_FFV1_2(W(7),W(16),GC_5,ZERO,ZERO,W(63))
C     Counter-term amplitude(s) for loop diagram number 124
      CALL MP_R2_QQ_1_0(W(63),W(15),R2_QQQ,AMPL(1,186))
      CALL MP_FFV1_1(W(15),W(16),GC_5,ZERO,ZERO,W(64))
C     Counter-term amplitude(s) for loop diagram number 125
      CALL MP_R2_QQ_1_0(W(7),W(64),R2_QQQ,AMPL(1,187))
C     Counter-term amplitude(s) for loop diagram number 126
      CALL MP_FFV1_0(W(7),W(15),W(16),R2_GQQ,AMPL(1,188))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,189))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,190))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,191))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,192))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,193))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB_1EPS,AMPL(2,194))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQG_1EPS,AMPL(2,195))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQB,AMPL(1,196))
      CALL MP_FFV1_0(W(7),W(15),W(16),UV_GQQT,AMPL(1,197))
C     Counter-term amplitude(s) for loop diagram number 128
      CALL MP_VVV1_0(W(4),W(5),W(62),R2_3GG,AMPL(1,198))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,199))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,200))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,201))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,202))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,203))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB_1EPS,AMPL(2,204))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GG_1EPS,AMPL(2,205))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GB,AMPL(1,206))
      CALL MP_VVV1_0(W(4),W(5),W(62),UV_3GT,AMPL(1,207))
C     Counter-term amplitude(s) for loop diagram number 141
      CALL MP_FFV1_0(W(48),W(2),W(5),R2_GQQ,AMPL(1,208))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,209))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,210))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,211))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,212))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,213))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB_1EPS,AMPL(2,214))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQG_1EPS,AMPL(2,215))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQB,AMPL(1,216))
      CALL MP_FFV1_0(W(48),W(2),W(5),UV_GQQT,AMPL(1,217))
C     Counter-term amplitude(s) for loop diagram number 143
      CALL MP_FFV1_0(W(57),W(2),W(6),R2_GQQ,AMPL(1,218))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,219))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,220))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,221))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,222))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,223))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB_1EPS,AMPL(2,224))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQG_1EPS,AMPL(2,225))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQB,AMPL(1,226))
      CALL MP_FFV1_0(W(57),W(2),W(6),UV_GQQT,AMPL(1,227))
C     Counter-term amplitude(s) for loop diagram number 145
      CALL MP_FFV1_0(W(39),W(2),W(4),R2_GQQ,AMPL(1,228))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,229))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,230))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,231))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,232))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,233))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB_1EPS,AMPL(2,234))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQG_1EPS,AMPL(2,235))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQB,AMPL(1,236))
      CALL MP_FFV1_0(W(39),W(2),W(4),UV_GQQT,AMPL(1,237))
C     Counter-term amplitude(s) for loop diagram number 147
      CALL MP_FFV1_0(W(59),W(2),W(6),R2_GQQ,AMPL(1,238))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,239))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,240))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,241))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,242))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,243))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB_1EPS,AMPL(2,244))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQG_1EPS,AMPL(2,245))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQB,AMPL(1,246))
      CALL MP_FFV1_0(W(59),W(2),W(6),UV_GQQT,AMPL(1,247))
C     Counter-term amplitude(s) for loop diagram number 149
      CALL MP_FFV1_0(W(41),W(2),W(4),R2_GQQ,AMPL(1,248))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,249))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,250))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,251))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,252))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,253))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB_1EPS,AMPL(2,254))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQG_1EPS,AMPL(2,255))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQB,AMPL(1,256))
      CALL MP_FFV1_0(W(41),W(2),W(4),UV_GQQT,AMPL(1,257))
C     Counter-term amplitude(s) for loop diagram number 151
      CALL MP_FFV1_0(W(50),W(2),W(5),R2_GQQ,AMPL(1,258))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,259))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,260))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,261))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,262))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,263))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB_1EPS,AMPL(2,264))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQG_1EPS,AMPL(2,265))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQB,AMPL(1,266))
      CALL MP_FFV1_0(W(50),W(2),W(5),UV_GQQT,AMPL(1,267))
      CALL MP_VVV1P0_1(W(6),W(16),GC_4,ZERO,ZERO,W(65))
C     Counter-term amplitude(s) for loop diagram number 159
      CALL MP_FFV1_0(W(7),W(2),W(65),R2_GQQ,AMPL(1,268))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,269))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,270))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,271))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,272))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,273))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB_1EPS,AMPL(2,274))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQG_1EPS,AMPL(2,275))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQB,AMPL(1,276))
      CALL MP_FFV1_0(W(7),W(2),W(65),UV_GQQT,AMPL(1,277))
C     Counter-term amplitude(s) for loop diagram number 162
      CALL MP_FFV1_0(W(63),W(2),W(6),R2_GQQ,AMPL(1,278))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,279))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,280))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,281))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,282))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,283))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB_1EPS,AMPL(2,284))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQG_1EPS,AMPL(2,285))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQB,AMPL(1,286))
      CALL MP_FFV1_0(W(63),W(2),W(6),UV_GQQT,AMPL(1,287))
C     Counter-term amplitude(s) for loop diagram number 163
      CALL MP_FFV1_0(W(10),W(2),W(16),R2_GQQ,AMPL(1,288))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,289))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,290))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,291))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,292))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,293))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB_1EPS,AMPL(2,294))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQG_1EPS,AMPL(2,295))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQB,AMPL(1,296))
      CALL MP_FFV1_0(W(10),W(2),W(16),UV_GQQT,AMPL(1,297))
      CALL MP_VVV1P0_1(W(5),W(14),GC_4,ZERO,ZERO,W(66))
C     Counter-term amplitude(s) for loop diagram number 172
      CALL MP_FFV1_0(W(7),W(2),W(66),R2_GQQ,AMPL(1,298))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,299))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,300))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,301))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,302))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,303))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB_1EPS,AMPL(2,304))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQG_1EPS,AMPL(2,305))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQB,AMPL(1,306))
      CALL MP_FFV1_0(W(7),W(2),W(66),UV_GQQT,AMPL(1,307))
C     Counter-term amplitude(s) for loop diagram number 175
      CALL MP_FFV1_0(W(54),W(2),W(5),R2_GQQ,AMPL(1,308))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,309))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,310))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,311))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,312))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,313))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB_1EPS,AMPL(2,314))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQG_1EPS,AMPL(2,315))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQB,AMPL(1,316))
      CALL MP_FFV1_0(W(54),W(2),W(5),UV_GQQT,AMPL(1,317))
C     Counter-term amplitude(s) for loop diagram number 176
      CALL MP_FFV1_0(W(9),W(2),W(14),R2_GQQ,AMPL(1,318))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,319))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,320))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,321))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,322))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,323))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB_1EPS,AMPL(2,324))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQG_1EPS,AMPL(2,325))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQB,AMPL(1,326))
      CALL MP_FFV1_0(W(9),W(2),W(14),UV_GQQT,AMPL(1,327))
C     Counter-term amplitude(s) for loop diagram number 200
      CALL MP_FFV1_0(W(45),W(2),W(4),R2_GQQ,AMPL(1,328))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,329))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,330))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,331))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,332))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,333))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB_1EPS,AMPL(2,334))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQG_1EPS,AMPL(2,335))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQB,AMPL(1,336))
      CALL MP_FFV1_0(W(45),W(2),W(4),UV_GQQT,AMPL(1,337))
      CALL MP_VVV1P0_1(W(4),W(11),GC_4,ZERO,ZERO,W(67))
C     Counter-term amplitude(s) for loop diagram number 201
      CALL MP_FFV1_0(W(7),W(2),W(67),R2_GQQ,AMPL(1,338))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,339))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,340))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,341))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,342))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,343))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB_1EPS,AMPL(2,344))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQG_1EPS,AMPL(2,345))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQB,AMPL(1,346))
      CALL MP_FFV1_0(W(7),W(2),W(67),UV_GQQT,AMPL(1,347))
C     Counter-term amplitude(s) for loop diagram number 204
      CALL MP_FFV1_0(W(13),W(2),W(11),R2_GQQ,AMPL(1,348))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,349))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,350))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,351))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,352))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,353))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB_1EPS,AMPL(2,354))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQG_1EPS,AMPL(2,355))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQB,AMPL(1,356))
      CALL MP_FFV1_0(W(13),W(2),W(11),UV_GQQT,AMPL(1,357))
C     Counter-term amplitude(s) for loop diagram number 237
      CALL MP_FFV1_0(W(7),W(2),W(18),R2_GQQ,AMPL(1,358))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,359))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,360))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,361))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,362))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,363))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB_1EPS,AMPL(2,364))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQG_1EPS,AMPL(2,365))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQB,AMPL(1,366))
      CALL MP_FFV1_0(W(7),W(2),W(18),UV_GQQT,AMPL(1,367))
      CALL MP_FFV1_0(W(7),W(2),W(19),R2_GQQ,AMPL(1,368))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,369))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,370))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,371))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,372))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,373))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB_1EPS,AMPL(2,374))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQG_1EPS,AMPL(2,375))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQB,AMPL(1,376))
      CALL MP_FFV1_0(W(7),W(2),W(19),UV_GQQT,AMPL(1,377))
      CALL MP_FFV1_0(W(7),W(2),W(20),R2_GQQ,AMPL(1,378))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,379))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,380))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,381))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,382))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,383))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB_1EPS,AMPL(2,384))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQG_1EPS,AMPL(2,385))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQB,AMPL(1,386))
      CALL MP_FFV1_0(W(7),W(2),W(20),UV_GQQT,AMPL(1,387))
C     Counter-term amplitude(s) for loop diagram number 248
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(65),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,388))
C     Counter-term amplitude(s) for loop diagram number 249
      CALL MP_VVV1_0(W(16),W(6),W(17),R2_3GG,AMPL(1,389))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,390))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,391))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,392))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,393))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,394))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB_1EPS,AMPL(2,395))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GG_1EPS,AMPL(2,396))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GB,AMPL(1,397))
      CALL MP_VVV1_0(W(16),W(6),W(17),UV_3GT,AMPL(1,398))
      CALL MP_VVV1P0_1(W(6),W(17),GC_4,ZERO,ZERO,W(68))
C     Counter-term amplitude(s) for loop diagram number 250
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(68),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,399))
      CALL MP_FFV1_1(W(2),W(16),GC_5,ZERO,ZERO,W(69))
C     Counter-term amplitude(s) for loop diagram number 253
      CALL MP_R2_QQ_1_0(W(10),W(69),R2_QQQ,AMPL(1,400))
      CALL MP_FFV1P0_3(W(10),W(2),GC_5,ZERO,ZERO,W(70))
C     Counter-term amplitude(s) for loop diagram number 254
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(70),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,401))
C     Counter-term amplitude(s) for loop diagram number 255
      CALL MP_FFV1_0(W(7),W(69),W(6),R2_GQQ,AMPL(1,402))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,403))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,404))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,405))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,406))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,407))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB_1EPS,AMPL(2,408))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQG_1EPS,AMPL(2,409))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQB,AMPL(1,410))
      CALL MP_FFV1_0(W(7),W(69),W(6),UV_GQQT,AMPL(1,411))
      CALL MP_FFV1_1(W(69),W(6),GC_5,ZERO,ZERO,W(71))
C     Counter-term amplitude(s) for loop diagram number 256
      CALL MP_R2_QQ_1_0(W(7),W(71),R2_QQQ,AMPL(1,412))
      CALL MP_FFV1_1(W(2),W(65),GC_5,ZERO,ZERO,W(72))
C     Counter-term amplitude(s) for loop diagram number 257
      CALL MP_R2_QQ_1_0(W(7),W(72),R2_QQQ,AMPL(1,413))
C     Counter-term amplitude(s) for loop diagram number 260
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(66),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,414))
C     Counter-term amplitude(s) for loop diagram number 261
      CALL MP_VVV1_0(W(14),W(5),W(17),R2_3GG,AMPL(1,415))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,416))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,417))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,418))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,419))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,420))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB_1EPS,AMPL(2,421))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GG_1EPS,AMPL(2,422))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GB,AMPL(1,423))
      CALL MP_VVV1_0(W(14),W(5),W(17),UV_3GT,AMPL(1,424))
      CALL MP_VVV1P0_1(W(5),W(17),GC_4,ZERO,ZERO,W(73))
C     Counter-term amplitude(s) for loop diagram number 262
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(73),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,425))
      CALL MP_FFV1_1(W(2),W(14),GC_5,ZERO,ZERO,W(74))
C     Counter-term amplitude(s) for loop diagram number 265
      CALL MP_R2_QQ_1_0(W(9),W(74),R2_QQQ,AMPL(1,426))
      CALL MP_FFV1P0_3(W(9),W(2),GC_5,ZERO,ZERO,W(75))
C     Counter-term amplitude(s) for loop diagram number 266
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(75),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,427))
C     Counter-term amplitude(s) for loop diagram number 267
      CALL MP_FFV1_0(W(7),W(74),W(5),R2_GQQ,AMPL(1,428))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,429))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,430))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,431))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,432))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,433))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB_1EPS,AMPL(2,434))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQG_1EPS,AMPL(2,435))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQB,AMPL(1,436))
      CALL MP_FFV1_0(W(7),W(74),W(5),UV_GQQT,AMPL(1,437))
      CALL MP_FFV1_1(W(74),W(5),GC_5,ZERO,ZERO,W(76))
C     Counter-term amplitude(s) for loop diagram number 268
      CALL MP_R2_QQ_1_0(W(7),W(76),R2_QQQ,AMPL(1,438))
      CALL MP_FFV1_1(W(2),W(66),GC_5,ZERO,ZERO,W(77))
C     Counter-term amplitude(s) for loop diagram number 269
      CALL MP_R2_QQ_1_0(W(7),W(77),R2_QQQ,AMPL(1,439))
C     Counter-term amplitude(s) for loop diagram number 272
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_44,AMPL(1
     $ ,440))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_59,AMPL(1
     $ ,441))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_137_44,AMPL(1
     $ ,442))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_59,AMPL(1
     $ ,443))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_59,AMPL(1
     $ ,444))
      CALL MP_R2RGA_VVVV1_0(W(4),W(5),W(6),W(17),R2GC_145_59,AMPL(1
     $ ,445))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_0(W(4),W(5),W(6),W(17)
     $ ,R2GC_141_51,R2GC_142_53,AMPL(1,446))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_144_57,R2GC_138_46,R2GC_143_55,AMPL(1,447))
      CALL MP_R2RGA_VVVV2_R2RGA_VVVV3_R2RGA_VVVV5_0(W(4),W(5),W(6)
     $ ,W(17),R2GC_138_46,R2GC_144_57,R2GC_143_55,AMPL(1,448))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_139_47,AMPL(1
     $ ,449))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_49,AMPL(1
     $ ,450))
      CALL MP_R2RGA_VVVV10_0(W(4),W(5),W(6),W(17),R2GC_140_49,AMPL(1
     $ ,451))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,452))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,453))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,454))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,455))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,456))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,457))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,458))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,459))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,460))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,461))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,462))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,463))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,464))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,465))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,466))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,467))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,468))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB_1EPS,AMPL(2,469))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GG_1EPS,AMPL(2,470))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GG_1EPS,AMPL(2,471))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GG_1EPS,AMPL(2,472))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GB,AMPL(1,473))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GB,AMPL(1,474))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GB,AMPL(1,475))
      CALL MP_VVVV1_0(W(4),W(5),W(6),W(17),UV_4GGT,AMPL(1,476))
      CALL MP_VVVV3_0(W(4),W(5),W(6),W(17),UV_4GGT,AMPL(1,477))
      CALL MP_VVVV4_0(W(4),W(5),W(6),W(17),UV_4GGT,AMPL(1,478))
C     Counter-term amplitude(s) for loop diagram number 273
      CALL MP_VVV1_0(W(4),W(5),W(68),R2_3GG,AMPL(1,479))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,480))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,481))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,482))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,483))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,484))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB_1EPS,AMPL(2,485))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GG_1EPS,AMPL(2,486))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GB,AMPL(1,487))
      CALL MP_VVV1_0(W(4),W(5),W(68),UV_3GT,AMPL(1,488))
C     Counter-term amplitude(s) for loop diagram number 276
      CALL MP_VVV1_0(W(4),W(6),W(73),R2_3GG,AMPL(1,489))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,490))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,491))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,492))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,493))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,494))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB_1EPS,AMPL(2,495))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GG_1EPS,AMPL(2,496))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GB,AMPL(1,497))
      CALL MP_VVV1_0(W(4),W(6),W(73),UV_3GT,AMPL(1,498))
C     Counter-term amplitude(s) for loop diagram number 281
      CALL MP_VVV1_0(W(4),W(6),W(75),R2_3GG,AMPL(1,499))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,500))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,501))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,502))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,503))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,504))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB_1EPS,AMPL(2,505))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GG_1EPS,AMPL(2,506))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GB,AMPL(1,507))
      CALL MP_VVV1_0(W(4),W(6),W(75),UV_3GT,AMPL(1,508))
C     Counter-term amplitude(s) for loop diagram number 283
      CALL MP_VVV1_0(W(4),W(5),W(70),R2_3GG,AMPL(1,509))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,510))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,511))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,512))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,513))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,514))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB_1EPS,AMPL(2,515))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GG_1EPS,AMPL(2,516))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GB,AMPL(1,517))
      CALL MP_VVV1_0(W(4),W(5),W(70),UV_3GT,AMPL(1,518))
C     Counter-term amplitude(s) for loop diagram number 285
      CALL MP_VVV1_0(W(4),W(11),W(17),R2_3GG,AMPL(1,519))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,520))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,521))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,522))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,523))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,524))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB_1EPS,AMPL(2,525))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GG_1EPS,AMPL(2,526))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GB,AMPL(1,527))
      CALL MP_VVV1_0(W(4),W(11),W(17),UV_3GT,AMPL(1,528))
      CALL MP_FFV1_1(W(2),W(11),GC_5,ZERO,ZERO,W(78))
C     Counter-term amplitude(s) for loop diagram number 287
      CALL MP_FFV1_0(W(7),W(78),W(4),R2_GQQ,AMPL(1,529))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,530))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,531))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,532))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,533))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,534))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB_1EPS,AMPL(2,535))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQG_1EPS,AMPL(2,536))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQB,AMPL(1,537))
      CALL MP_FFV1_0(W(7),W(78),W(4),UV_GQQT,AMPL(1,538))
C     Counter-term amplitude(s) for loop diagram number 289
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(67),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,539))
      CALL MP_VVV1P0_1(W(4),W(17),GC_4,ZERO,ZERO,W(79))
C     Counter-term amplitude(s) for loop diagram number 291
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(79),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,540))
C     Counter-term amplitude(s) for loop diagram number 293
      CALL MP_R2_QQ_1_0(W(13),W(78),R2_QQQ,AMPL(1,541))
      CALL MP_FFV1P0_3(W(13),W(2),GC_5,ZERO,ZERO,W(80))
C     Counter-term amplitude(s) for loop diagram number 294
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(80),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,542))
      CALL MP_FFV1_1(W(78),W(4),GC_5,ZERO,ZERO,W(81))
C     Counter-term amplitude(s) for loop diagram number 296
      CALL MP_R2_QQ_1_0(W(7),W(81),R2_QQQ,AMPL(1,543))
      CALL MP_FFV1_1(W(2),W(67),GC_5,ZERO,ZERO,W(82))
C     Counter-term amplitude(s) for loop diagram number 297
      CALL MP_R2_QQ_1_0(W(7),W(82),R2_QQQ,AMPL(1,544))
C     Counter-term amplitude(s) for loop diagram number 299
      CALL MP_VVV1_0(W(5),W(6),W(79),R2_3GG,AMPL(1,545))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,546))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,547))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,548))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,549))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,550))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB_1EPS,AMPL(2,551))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GG_1EPS,AMPL(2,552))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GB,AMPL(1,553))
      CALL MP_VVV1_0(W(5),W(6),W(79),UV_3GT,AMPL(1,554))
C     Counter-term amplitude(s) for loop diagram number 303
      CALL MP_VVV1_0(W(5),W(6),W(80),R2_3GG,AMPL(1,555))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,556))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,557))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,558))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,559))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,560))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB_1EPS,AMPL(2,561))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GG_1EPS,AMPL(2,562))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GB,AMPL(1,563))
      CALL MP_VVV1_0(W(5),W(6),W(80),UV_3GT,AMPL(1,564))
C     Counter-term amplitude(s) for loop diagram number 311
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(18),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,565))
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(19),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,566))
      CALL MP_R2_GG_1_R2_GG_2_0(W(17),W(20),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,567))
      CALL MP_FFV1_1(W(2),W(18),GC_5,ZERO,ZERO,W(83))
      CALL MP_FFV1_1(W(2),W(19),GC_5,ZERO,ZERO,W(84))
      CALL MP_FFV1_1(W(2),W(20),GC_5,ZERO,ZERO,W(85))
C     Counter-term amplitude(s) for loop diagram number 312
      CALL MP_R2_QQ_1_0(W(7),W(83),R2_QQQ,AMPL(1,568))
      CALL MP_R2_QQ_1_0(W(7),W(84),R2_QQQ,AMPL(1,569))
      CALL MP_R2_QQ_1_0(W(7),W(85),R2_QQQ,AMPL(1,570))
      CALL MP_FFV1_1(W(22),W(6),GC_5,ZERO,ZERO,W(86))
C     Counter-term amplitude(s) for loop diagram number 322
      CALL MP_R2_QQ_1_0(W(23),W(86),R2_QQQ,AMPL(1,571))
      CALL MP_FFV1_2(W(23),W(6),GC_5,ZERO,ZERO,W(87))
C     Counter-term amplitude(s) for loop diagram number 323
      CALL MP_R2_QQ_1_0(W(87),W(22),R2_QQQ,AMPL(1,572))
C     Counter-term amplitude(s) for loop diagram number 324
      CALL MP_FFV1_0(W(23),W(22),W(6),R2_GQQ,AMPL(1,573))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,574))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,575))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,576))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,577))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,578))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB_1EPS,AMPL(2,579))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQG_1EPS,AMPL(2,580))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQB,AMPL(1,581))
      CALL MP_FFV1_0(W(23),W(22),W(6),UV_GQQT,AMPL(1,582))
      CALL MP_FFV1_1(W(22),W(5),GC_5,ZERO,ZERO,W(88))
C     Counter-term amplitude(s) for loop diagram number 325
      CALL MP_R2_QQ_1_0(W(24),W(88),R2_QQQ,AMPL(1,583))
      CALL MP_FFV1_2(W(24),W(5),GC_5,ZERO,ZERO,W(89))
C     Counter-term amplitude(s) for loop diagram number 326
      CALL MP_R2_QQ_1_0(W(89),W(22),R2_QQQ,AMPL(1,584))
C     Counter-term amplitude(s) for loop diagram number 327
      CALL MP_FFV1_0(W(24),W(22),W(5),R2_GQQ,AMPL(1,585))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,586))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,587))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,588))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,589))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,590))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB_1EPS,AMPL(2,591))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQG_1EPS,AMPL(2,592))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQB,AMPL(1,593))
      CALL MP_FFV1_0(W(24),W(22),W(5),UV_GQQT,AMPL(1,594))
C     Counter-term amplitude(s) for loop diagram number 328
      CALL MP_FFV1_0(W(21),W(88),W(6),R2_GQQ,AMPL(1,595))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,596))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,597))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,598))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,599))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,600))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB_1EPS,AMPL(2,601))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQG_1EPS,AMPL(2,602))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQB,AMPL(1,603))
      CALL MP_FFV1_0(W(21),W(88),W(6),UV_GQQT,AMPL(1,604))
      CALL MP_FFV1_1(W(88),W(6),GC_5,ZERO,ZERO,W(90))
C     Counter-term amplitude(s) for loop diagram number 329
      CALL MP_R2_QQ_1_0(W(21),W(90),R2_QQQ,AMPL(1,605))
C     Counter-term amplitude(s) for loop diagram number 330
      CALL MP_FFV1_0(W(21),W(86),W(5),R2_GQQ,AMPL(1,606))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,607))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,608))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,609))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,610))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,611))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB_1EPS,AMPL(2,612))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQG_1EPS,AMPL(2,613))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQB,AMPL(1,614))
      CALL MP_FFV1_0(W(21),W(86),W(5),UV_GQQT,AMPL(1,615))
      CALL MP_FFV1_1(W(86),W(5),GC_5,ZERO,ZERO,W(91))
C     Counter-term amplitude(s) for loop diagram number 331
      CALL MP_R2_QQ_1_0(W(21),W(91),R2_QQQ,AMPL(1,616))
      CALL MP_FFV1P0_3(W(21),W(22),GC_5,ZERO,ZERO,W(92))
C     Counter-term amplitude(s) for loop diagram number 334
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(92),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,617))
      CALL MP_FFV1_2(W(21),W(11),GC_5,ZERO,ZERO,W(93))
C     Counter-term amplitude(s) for loop diagram number 335
      CALL MP_R2_QQ_1_0(W(93),W(22),R2_QQQ,AMPL(1,618))
      CALL MP_FFV1_1(W(22),W(11),GC_5,ZERO,ZERO,W(94))
C     Counter-term amplitude(s) for loop diagram number 336
      CALL MP_R2_QQ_1_0(W(21),W(94),R2_QQQ,AMPL(1,619))
C     Counter-term amplitude(s) for loop diagram number 337
      CALL MP_FFV1_0(W(21),W(22),W(11),R2_GQQ,AMPL(1,620))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,621))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,622))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,623))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,624))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,625))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB_1EPS,AMPL(2,626))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQG_1EPS,AMPL(2,627))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQB,AMPL(1,628))
      CALL MP_FFV1_0(W(21),W(22),W(11),UV_GQQT,AMPL(1,629))
C     Counter-term amplitude(s) for loop diagram number 339
      CALL MP_VVV1_0(W(5),W(6),W(92),R2_3GG,AMPL(1,630))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,631))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,632))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,633))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,634))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,635))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB_1EPS,AMPL(2,636))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GG_1EPS,AMPL(2,637))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GB,AMPL(1,638))
      CALL MP_VVV1_0(W(5),W(6),W(92),UV_3GT,AMPL(1,639))
C     Counter-term amplitude(s) for loop diagram number 352
      CALL MP_R2_QQ_1_0(W(25),W(47),R2_QQQ,AMPL(1,640))
      CALL MP_FFV1_2(W(25),W(6),GC_5,ZERO,ZERO,W(95))
C     Counter-term amplitude(s) for loop diagram number 353
      CALL MP_R2_QQ_1_0(W(95),W(12),R2_QQQ,AMPL(1,641))
C     Counter-term amplitude(s) for loop diagram number 354
      CALL MP_FFV1_0(W(25),W(12),W(6),R2_GQQ,AMPL(1,642))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,643))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,644))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,645))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,646))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,647))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB_1EPS,AMPL(2,648))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQG_1EPS,AMPL(2,649))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQB,AMPL(1,650))
      CALL MP_FFV1_0(W(25),W(12),W(6),UV_GQQT,AMPL(1,651))
      CALL MP_FFV2_5_1(W(12),W(3),GC_22,GC_23,ZERO,ZERO,W(96))
C     Counter-term amplitude(s) for loop diagram number 355
      CALL MP_R2_QQ_1_0(W(24),W(96),R2_QQQ,AMPL(1,652))
      CALL MP_FFV2_5_2(W(24),W(3),GC_22,GC_23,ZERO,ZERO,W(97))
C     Counter-term amplitude(s) for loop diagram number 356
      CALL MP_R2_QQ_1_0(W(97),W(12),R2_QQQ,AMPL(1,653))
C     Counter-term amplitude(s) for loop diagram number 357
      CALL MP_FFV2_5_0(W(24),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,654)
     $ )
C     Counter-term amplitude(s) for loop diagram number 358
      CALL MP_FFV1_0(W(21),W(96),W(6),R2_GQQ,AMPL(1,655))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,656))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,657))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,658))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,659))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,660))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB_1EPS,AMPL(2,661))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQG_1EPS,AMPL(2,662))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQB,AMPL(1,663))
      CALL MP_FFV1_0(W(21),W(96),W(6),UV_GQQT,AMPL(1,664))
      CALL MP_FFV1_1(W(96),W(6),GC_5,ZERO,ZERO,W(98))
C     Counter-term amplitude(s) for loop diagram number 359
      CALL MP_R2_QQ_1_0(W(21),W(98),R2_QQQ,AMPL(1,665))
C     Counter-term amplitude(s) for loop diagram number 360
      CALL MP_FFV2_5_0(W(21),W(47),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,666)
     $ )
      CALL MP_FFV2_5_1(W(47),W(3),GC_22,GC_23,ZERO,ZERO,W(99))
C     Counter-term amplitude(s) for loop diagram number 361
      CALL MP_R2_QQ_1_0(W(21),W(99),R2_QQQ,AMPL(1,667))
C     Counter-term amplitude(s) for loop diagram number 367
      CALL MP_R2_QQ_1_0(W(25),W(56),R2_QQQ,AMPL(1,668))
      CALL MP_FFV1_2(W(25),W(5),GC_5,ZERO,ZERO,W(100))
C     Counter-term amplitude(s) for loop diagram number 368
      CALL MP_R2_QQ_1_0(W(100),W(15),R2_QQQ,AMPL(1,669))
C     Counter-term amplitude(s) for loop diagram number 369
      CALL MP_FFV1_0(W(25),W(15),W(5),R2_GQQ,AMPL(1,670))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,671))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,672))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,673))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,674))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,675))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB_1EPS,AMPL(2,676))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQG_1EPS,AMPL(2,677))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQB,AMPL(1,678))
      CALL MP_FFV1_0(W(25),W(15),W(5),UV_GQQT,AMPL(1,679))
      CALL MP_FFV2_5_1(W(15),W(3),GC_22,GC_23,ZERO,ZERO,W(101))
C     Counter-term amplitude(s) for loop diagram number 370
      CALL MP_R2_QQ_1_0(W(23),W(101),R2_QQQ,AMPL(1,680))
      CALL MP_FFV2_5_2(W(23),W(3),GC_22,GC_23,ZERO,ZERO,W(102))
C     Counter-term amplitude(s) for loop diagram number 371
      CALL MP_R2_QQ_1_0(W(102),W(15),R2_QQQ,AMPL(1,681))
C     Counter-term amplitude(s) for loop diagram number 372
      CALL MP_FFV2_5_0(W(23),W(15),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,682)
     $ )
C     Counter-term amplitude(s) for loop diagram number 373
      CALL MP_FFV1_0(W(21),W(101),W(5),R2_GQQ,AMPL(1,683))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,684))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,685))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,686))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,687))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,688))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB_1EPS,AMPL(2,689))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQG_1EPS,AMPL(2,690))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQB,AMPL(1,691))
      CALL MP_FFV1_0(W(21),W(101),W(5),UV_GQQT,AMPL(1,692))
      CALL MP_FFV1_1(W(101),W(5),GC_5,ZERO,ZERO,W(103))
C     Counter-term amplitude(s) for loop diagram number 374
      CALL MP_R2_QQ_1_0(W(21),W(103),R2_QQQ,AMPL(1,693))
C     Counter-term amplitude(s) for loop diagram number 375
      CALL MP_FFV2_5_0(W(21),W(56),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,694)
     $ )
      CALL MP_FFV2_5_1(W(56),W(3),GC_22,GC_23,ZERO,ZERO,W(104))
C     Counter-term amplitude(s) for loop diagram number 376
      CALL MP_R2_QQ_1_0(W(21),W(104),R2_QQQ,AMPL(1,695))
C     Counter-term amplitude(s) for loop diagram number 382
      CALL MP_FFV1_0(W(95),W(2),W(5),R2_GQQ,AMPL(1,696))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,697))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,698))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,699))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,700))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,701))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB_1EPS,AMPL(2,702))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQG_1EPS,AMPL(2,703))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQB,AMPL(1,704))
      CALL MP_FFV1_0(W(95),W(2),W(5),UV_GQQT,AMPL(1,705))
C     Counter-term amplitude(s) for loop diagram number 384
      CALL MP_FFV1_0(W(100),W(2),W(6),R2_GQQ,AMPL(1,706))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,707))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,708))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,709))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,710))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,711))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB_1EPS,AMPL(2,712))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQG_1EPS,AMPL(2,713))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQB,AMPL(1,714))
      CALL MP_FFV1_0(W(100),W(2),W(6),UV_GQQT,AMPL(1,715))
C     Counter-term amplitude(s) for loop diagram number 386
      CALL MP_FFV2_5_0(W(87),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,716))
C     Counter-term amplitude(s) for loop diagram number 388
      CALL MP_FFV1_0(W(102),W(2),W(6),R2_GQQ,AMPL(1,717))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,718))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,719))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,720))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,721))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,722))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB_1EPS,AMPL(2,723))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQG_1EPS,AMPL(2,724))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQB,AMPL(1,725))
      CALL MP_FFV1_0(W(102),W(2),W(6),UV_GQQT,AMPL(1,726))
C     Counter-term amplitude(s) for loop diagram number 390
      CALL MP_FFV2_5_0(W(89),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,727))
C     Counter-term amplitude(s) for loop diagram number 392
      CALL MP_FFV1_0(W(97),W(2),W(5),R2_GQQ,AMPL(1,728))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,729))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,730))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,731))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,732))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,733))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB_1EPS,AMPL(2,734))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQG_1EPS,AMPL(2,735))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQB,AMPL(1,736))
      CALL MP_FFV1_0(W(97),W(2),W(5),UV_GQQT,AMPL(1,737))
C     Counter-term amplitude(s) for loop diagram number 400
      CALL MP_FFV1_0(W(25),W(2),W(11),R2_GQQ,AMPL(1,738))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,739))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,740))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,741))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,742))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,743))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB_1EPS,AMPL(2,744))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQG_1EPS,AMPL(2,745))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQB,AMPL(1,746))
      CALL MP_FFV1_0(W(25),W(2),W(11),UV_GQQT,AMPL(1,747))
C     Counter-term amplitude(s) for loop diagram number 402
      CALL MP_FFV2_5_0(W(93),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,748))
C     Counter-term amplitude(s) for loop diagram number 424
      CALL MP_R2_QQ_1_0(W(25),W(78),R2_QQQ,AMPL(1,749))
      CALL MP_FFV1P0_3(W(25),W(2),GC_5,ZERO,ZERO,W(105))
C     Counter-term amplitude(s) for loop diagram number 425
      CALL MP_R2_GG_1_R2_GG_2_0(W(11),W(105),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,750))
C     Counter-term amplitude(s) for loop diagram number 426
      CALL MP_FFV2_5_0(W(21),W(78),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,751)
     $ )
      CALL MP_FFV2_5_1(W(78),W(3),GC_22,GC_23,ZERO,ZERO,W(106))
C     Counter-term amplitude(s) for loop diagram number 427
      CALL MP_R2_QQ_1_0(W(21),W(106),R2_QQQ,AMPL(1,752))
C     Counter-term amplitude(s) for loop diagram number 428
      CALL MP_VVV1_0(W(5),W(6),W(105),R2_3GG,AMPL(1,753))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,754))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,755))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,756))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,757))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,758))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB_1EPS,AMPL(2,759))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GG_1EPS,AMPL(2,760))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GB,AMPL(1,761))
      CALL MP_VVV1_0(W(5),W(6),W(105),UV_3GT,AMPL(1,762))
C     Counter-term amplitude(s) for loop diagram number 432
      CALL MP_R2_QQ_1_0(W(27),W(86),R2_QQQ,AMPL(1,763))
      CALL MP_FFV1_2(W(27),W(6),GC_5,ZERO,ZERO,W(107))
C     Counter-term amplitude(s) for loop diagram number 433
      CALL MP_R2_QQ_1_0(W(107),W(22),R2_QQQ,AMPL(1,764))
C     Counter-term amplitude(s) for loop diagram number 434
      CALL MP_FFV1_0(W(27),W(22),W(6),R2_GQQ,AMPL(1,765))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,766))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,767))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,768))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,769))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,770))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB_1EPS,AMPL(2,771))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQG_1EPS,AMPL(2,772))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQB,AMPL(1,773))
      CALL MP_FFV1_0(W(27),W(22),W(6),UV_GQQT,AMPL(1,774))
      CALL MP_FFV1_1(W(22),W(4),GC_5,ZERO,ZERO,W(108))
C     Counter-term amplitude(s) for loop diagram number 435
      CALL MP_R2_QQ_1_0(W(28),W(108),R2_QQQ,AMPL(1,775))
      CALL MP_FFV1_2(W(28),W(4),GC_5,ZERO,ZERO,W(109))
C     Counter-term amplitude(s) for loop diagram number 436
      CALL MP_R2_QQ_1_0(W(109),W(22),R2_QQQ,AMPL(1,776))
C     Counter-term amplitude(s) for loop diagram number 437
      CALL MP_FFV1_0(W(28),W(22),W(4),R2_GQQ,AMPL(1,777))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,778))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,779))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,780))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,781))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,782))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB_1EPS,AMPL(2,783))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQG_1EPS,AMPL(2,784))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQB,AMPL(1,785))
      CALL MP_FFV1_0(W(28),W(22),W(4),UV_GQQT,AMPL(1,786))
C     Counter-term amplitude(s) for loop diagram number 438
      CALL MP_FFV1_0(W(26),W(108),W(6),R2_GQQ,AMPL(1,787))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,788))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,789))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,790))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,791))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,792))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB_1EPS,AMPL(2,793))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQG_1EPS,AMPL(2,794))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQB,AMPL(1,795))
      CALL MP_FFV1_0(W(26),W(108),W(6),UV_GQQT,AMPL(1,796))
      CALL MP_FFV1_1(W(108),W(6),GC_5,ZERO,ZERO,W(110))
C     Counter-term amplitude(s) for loop diagram number 439
      CALL MP_R2_QQ_1_0(W(26),W(110),R2_QQQ,AMPL(1,797))
C     Counter-term amplitude(s) for loop diagram number 440
      CALL MP_FFV1_0(W(26),W(86),W(4),R2_GQQ,AMPL(1,798))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,799))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,800))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,801))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,802))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,803))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB_1EPS,AMPL(2,804))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQG_1EPS,AMPL(2,805))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQB,AMPL(1,806))
      CALL MP_FFV1_0(W(26),W(86),W(4),UV_GQQT,AMPL(1,807))
      CALL MP_FFV1_1(W(86),W(4),GC_5,ZERO,ZERO,W(111))
C     Counter-term amplitude(s) for loop diagram number 441
      CALL MP_R2_QQ_1_0(W(26),W(111),R2_QQQ,AMPL(1,808))
      CALL MP_FFV1P0_3(W(26),W(22),GC_5,ZERO,ZERO,W(112))
C     Counter-term amplitude(s) for loop diagram number 444
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(112),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,809))
      CALL MP_FFV1_2(W(26),W(14),GC_5,ZERO,ZERO,W(113))
C     Counter-term amplitude(s) for loop diagram number 445
      CALL MP_R2_QQ_1_0(W(113),W(22),R2_QQQ,AMPL(1,810))
      CALL MP_FFV1_1(W(22),W(14),GC_5,ZERO,ZERO,W(114))
C     Counter-term amplitude(s) for loop diagram number 446
      CALL MP_R2_QQ_1_0(W(26),W(114),R2_QQQ,AMPL(1,811))
C     Counter-term amplitude(s) for loop diagram number 447
      CALL MP_FFV1_0(W(26),W(22),W(14),R2_GQQ,AMPL(1,812))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,813))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,814))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,815))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,816))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,817))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB_1EPS,AMPL(2,818))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQG_1EPS,AMPL(2,819))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQB,AMPL(1,820))
      CALL MP_FFV1_0(W(26),W(22),W(14),UV_GQQT,AMPL(1,821))
C     Counter-term amplitude(s) for loop diagram number 449
      CALL MP_VVV1_0(W(4),W(6),W(112),R2_3GG,AMPL(1,822))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,823))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,824))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,825))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,826))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,827))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB_1EPS,AMPL(2,828))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GG_1EPS,AMPL(2,829))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GB,AMPL(1,830))
      CALL MP_VVV1_0(W(4),W(6),W(112),UV_3GT,AMPL(1,831))
C     Counter-term amplitude(s) for loop diagram number 462
      CALL MP_R2_QQ_1_0(W(29),W(38),R2_QQQ,AMPL(1,832))
      CALL MP_FFV1_2(W(29),W(6),GC_5,ZERO,ZERO,W(115))
C     Counter-term amplitude(s) for loop diagram number 463
      CALL MP_R2_QQ_1_0(W(115),W(8),R2_QQQ,AMPL(1,833))
C     Counter-term amplitude(s) for loop diagram number 464
      CALL MP_FFV1_0(W(29),W(8),W(6),R2_GQQ,AMPL(1,834))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,835))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,836))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,837))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,838))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,839))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB_1EPS,AMPL(2,840))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQG_1EPS,AMPL(2,841))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQB,AMPL(1,842))
      CALL MP_FFV1_0(W(29),W(8),W(6),UV_GQQT,AMPL(1,843))
      CALL MP_FFV2_5_1(W(8),W(3),GC_22,GC_23,ZERO,ZERO,W(116))
C     Counter-term amplitude(s) for loop diagram number 465
      CALL MP_R2_QQ_1_0(W(28),W(116),R2_QQQ,AMPL(1,844))
      CALL MP_FFV2_5_2(W(28),W(3),GC_22,GC_23,ZERO,ZERO,W(117))
C     Counter-term amplitude(s) for loop diagram number 466
      CALL MP_R2_QQ_1_0(W(117),W(8),R2_QQQ,AMPL(1,845))
C     Counter-term amplitude(s) for loop diagram number 467
      CALL MP_FFV2_5_0(W(28),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,846))
C     Counter-term amplitude(s) for loop diagram number 468
      CALL MP_FFV1_0(W(26),W(116),W(6),R2_GQQ,AMPL(1,847))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,848))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,849))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,850))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,851))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,852))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB_1EPS,AMPL(2,853))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQG_1EPS,AMPL(2,854))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQB,AMPL(1,855))
      CALL MP_FFV1_0(W(26),W(116),W(6),UV_GQQT,AMPL(1,856))
      CALL MP_FFV1_1(W(116),W(6),GC_5,ZERO,ZERO,W(118))
C     Counter-term amplitude(s) for loop diagram number 469
      CALL MP_R2_QQ_1_0(W(26),W(118),R2_QQQ,AMPL(1,857))
C     Counter-term amplitude(s) for loop diagram number 470
      CALL MP_FFV2_5_0(W(26),W(38),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,858)
     $ )
      CALL MP_FFV2_5_1(W(38),W(3),GC_22,GC_23,ZERO,ZERO,W(119))
C     Counter-term amplitude(s) for loop diagram number 471
      CALL MP_R2_QQ_1_0(W(26),W(119),R2_QQQ,AMPL(1,859))
C     Counter-term amplitude(s) for loop diagram number 477
      CALL MP_R2_QQ_1_0(W(29),W(58),R2_QQQ,AMPL(1,860))
      CALL MP_FFV1_2(W(29),W(4),GC_5,ZERO,ZERO,W(120))
C     Counter-term amplitude(s) for loop diagram number 478
      CALL MP_R2_QQ_1_0(W(120),W(15),R2_QQQ,AMPL(1,861))
C     Counter-term amplitude(s) for loop diagram number 479
      CALL MP_FFV1_0(W(29),W(15),W(4),R2_GQQ,AMPL(1,862))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,863))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,864))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,865))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,866))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,867))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB_1EPS,AMPL(2,868))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQG_1EPS,AMPL(2,869))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQB,AMPL(1,870))
      CALL MP_FFV1_0(W(29),W(15),W(4),UV_GQQT,AMPL(1,871))
C     Counter-term amplitude(s) for loop diagram number 480
      CALL MP_R2_QQ_1_0(W(27),W(101),R2_QQQ,AMPL(1,872))
      CALL MP_FFV2_5_2(W(27),W(3),GC_22,GC_23,ZERO,ZERO,W(121))
C     Counter-term amplitude(s) for loop diagram number 481
      CALL MP_R2_QQ_1_0(W(121),W(15),R2_QQQ,AMPL(1,873))
C     Counter-term amplitude(s) for loop diagram number 482
      CALL MP_FFV2_5_0(W(27),W(15),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,874)
     $ )
C     Counter-term amplitude(s) for loop diagram number 483
      CALL MP_FFV1_0(W(26),W(101),W(4),R2_GQQ,AMPL(1,875))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,876))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,877))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,878))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,879))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,880))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB_1EPS,AMPL(2,881))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQG_1EPS,AMPL(2,882))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQB,AMPL(1,883))
      CALL MP_FFV1_0(W(26),W(101),W(4),UV_GQQT,AMPL(1,884))
      CALL MP_FFV1_1(W(101),W(4),GC_5,ZERO,ZERO,W(122))
C     Counter-term amplitude(s) for loop diagram number 484
      CALL MP_R2_QQ_1_0(W(26),W(122),R2_QQQ,AMPL(1,885))
C     Counter-term amplitude(s) for loop diagram number 485
      CALL MP_FFV2_5_0(W(26),W(58),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,886)
     $ )
      CALL MP_FFV2_5_1(W(58),W(3),GC_22,GC_23,ZERO,ZERO,W(123))
C     Counter-term amplitude(s) for loop diagram number 486
      CALL MP_R2_QQ_1_0(W(26),W(123),R2_QQQ,AMPL(1,887))
C     Counter-term amplitude(s) for loop diagram number 492
      CALL MP_FFV1_0(W(115),W(2),W(4),R2_GQQ,AMPL(1,888))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,889))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,890))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,891))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,892))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,893))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB_1EPS,AMPL(2,894))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQG_1EPS,AMPL(2,895))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQB,AMPL(1,896))
      CALL MP_FFV1_0(W(115),W(2),W(4),UV_GQQT,AMPL(1,897))
C     Counter-term amplitude(s) for loop diagram number 494
      CALL MP_FFV1_0(W(120),W(2),W(6),R2_GQQ,AMPL(1,898))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,899))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,900))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,901))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,902))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,903))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB_1EPS,AMPL(2,904))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQG_1EPS,AMPL(2,905))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQB,AMPL(1,906))
      CALL MP_FFV1_0(W(120),W(2),W(6),UV_GQQT,AMPL(1,907))
C     Counter-term amplitude(s) for loop diagram number 496
      CALL MP_FFV2_5_0(W(107),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,908)
     $ )
C     Counter-term amplitude(s) for loop diagram number 498
      CALL MP_FFV1_0(W(121),W(2),W(6),R2_GQQ,AMPL(1,909))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,910))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,911))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,912))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,913))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,914))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB_1EPS,AMPL(2,915))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQG_1EPS,AMPL(2,916))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQB,AMPL(1,917))
      CALL MP_FFV1_0(W(121),W(2),W(6),UV_GQQT,AMPL(1,918))
C     Counter-term amplitude(s) for loop diagram number 500
      CALL MP_FFV2_5_0(W(109),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,919)
     $ )
C     Counter-term amplitude(s) for loop diagram number 502
      CALL MP_FFV1_0(W(117),W(2),W(4),R2_GQQ,AMPL(1,920))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,921))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,922))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,923))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,924))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,925))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB_1EPS,AMPL(2,926))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQG_1EPS,AMPL(2,927))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQB,AMPL(1,928))
      CALL MP_FFV1_0(W(117),W(2),W(4),UV_GQQT,AMPL(1,929))
C     Counter-term amplitude(s) for loop diagram number 510
      CALL MP_FFV1_0(W(29),W(2),W(14),R2_GQQ,AMPL(1,930))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,931))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,932))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,933))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,934))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,935))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB_1EPS,AMPL(2,936))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQG_1EPS,AMPL(2,937))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQB,AMPL(1,938))
      CALL MP_FFV1_0(W(29),W(2),W(14),UV_GQQT,AMPL(1,939))
C     Counter-term amplitude(s) for loop diagram number 512
      CALL MP_FFV2_5_0(W(113),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,940)
     $ )
C     Counter-term amplitude(s) for loop diagram number 534
      CALL MP_R2_QQ_1_0(W(29),W(74),R2_QQQ,AMPL(1,941))
      CALL MP_FFV1P0_3(W(29),W(2),GC_5,ZERO,ZERO,W(124))
C     Counter-term amplitude(s) for loop diagram number 535
      CALL MP_R2_GG_1_R2_GG_2_0(W(14),W(124),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,942))
C     Counter-term amplitude(s) for loop diagram number 536
      CALL MP_FFV2_5_0(W(26),W(74),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,943)
     $ )
      CALL MP_FFV2_5_1(W(74),W(3),GC_22,GC_23,ZERO,ZERO,W(125))
C     Counter-term amplitude(s) for loop diagram number 537
      CALL MP_R2_QQ_1_0(W(26),W(125),R2_QQQ,AMPL(1,944))
C     Counter-term amplitude(s) for loop diagram number 538
      CALL MP_VVV1_0(W(4),W(6),W(124),R2_3GG,AMPL(1,945))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,946))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,947))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,948))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,949))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,950))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB_1EPS,AMPL(2,951))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GG_1EPS,AMPL(2,952))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GB,AMPL(1,953))
      CALL MP_VVV1_0(W(4),W(6),W(124),UV_3GT,AMPL(1,954))
C     Counter-term amplitude(s) for loop diagram number 542
      CALL MP_R2_QQ_1_0(W(31),W(88),R2_QQQ,AMPL(1,955))
      CALL MP_FFV1_2(W(31),W(5),GC_5,ZERO,ZERO,W(126))
C     Counter-term amplitude(s) for loop diagram number 543
      CALL MP_R2_QQ_1_0(W(126),W(22),R2_QQQ,AMPL(1,956))
C     Counter-term amplitude(s) for loop diagram number 544
      CALL MP_FFV1_0(W(31),W(22),W(5),R2_GQQ,AMPL(1,957))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,958))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,959))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,960))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,961))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,962))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB_1EPS,AMPL(2,963))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQG_1EPS,AMPL(2,964))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQB,AMPL(1,965))
      CALL MP_FFV1_0(W(31),W(22),W(5),UV_GQQT,AMPL(1,966))
C     Counter-term amplitude(s) for loop diagram number 545
      CALL MP_R2_QQ_1_0(W(32),W(108),R2_QQQ,AMPL(1,967))
      CALL MP_FFV1_2(W(32),W(4),GC_5,ZERO,ZERO,W(127))
C     Counter-term amplitude(s) for loop diagram number 546
      CALL MP_R2_QQ_1_0(W(127),W(22),R2_QQQ,AMPL(1,968))
C     Counter-term amplitude(s) for loop diagram number 547
      CALL MP_FFV1_0(W(32),W(22),W(4),R2_GQQ,AMPL(1,969))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,970))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,971))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,972))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,973))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,974))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB_1EPS,AMPL(2,975))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQG_1EPS,AMPL(2,976))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQB,AMPL(1,977))
      CALL MP_FFV1_0(W(32),W(22),W(4),UV_GQQT,AMPL(1,978))
C     Counter-term amplitude(s) for loop diagram number 548
      CALL MP_FFV1_0(W(30),W(108),W(5),R2_GQQ,AMPL(1,979))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,980))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,981))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,982))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,983))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,984))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB_1EPS,AMPL(2,985))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQG_1EPS,AMPL(2,986))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQB,AMPL(1,987))
      CALL MP_FFV1_0(W(30),W(108),W(5),UV_GQQT,AMPL(1,988))
      CALL MP_FFV1_1(W(108),W(5),GC_5,ZERO,ZERO,W(128))
C     Counter-term amplitude(s) for loop diagram number 549
      CALL MP_R2_QQ_1_0(W(30),W(128),R2_QQQ,AMPL(1,989))
C     Counter-term amplitude(s) for loop diagram number 550
      CALL MP_FFV1_0(W(30),W(88),W(4),R2_GQQ,AMPL(1,990))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,991))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,992))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,993))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,994))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,995))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB_1EPS,AMPL(2,996))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQG_1EPS,AMPL(2,997))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQB,AMPL(1,998))
      CALL MP_FFV1_0(W(30),W(88),W(4),UV_GQQT,AMPL(1,999))
      CALL MP_FFV1_1(W(88),W(4),GC_5,ZERO,ZERO,W(129))
C     Counter-term amplitude(s) for loop diagram number 551
      CALL MP_R2_QQ_1_0(W(30),W(129),R2_QQQ,AMPL(1,1000))
      CALL MP_FFV1P0_3(W(30),W(22),GC_5,ZERO,ZERO,W(130))
C     Counter-term amplitude(s) for loop diagram number 554
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(130),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1001))
      CALL MP_FFV1_2(W(30),W(16),GC_5,ZERO,ZERO,W(131))
C     Counter-term amplitude(s) for loop diagram number 555
      CALL MP_R2_QQ_1_0(W(131),W(22),R2_QQQ,AMPL(1,1002))
      CALL MP_FFV1_1(W(22),W(16),GC_5,ZERO,ZERO,W(132))
C     Counter-term amplitude(s) for loop diagram number 556
      CALL MP_R2_QQ_1_0(W(30),W(132),R2_QQQ,AMPL(1,1003))
C     Counter-term amplitude(s) for loop diagram number 557
      CALL MP_FFV1_0(W(30),W(22),W(16),R2_GQQ,AMPL(1,1004))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1005))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1006))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1007))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1008))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1009))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB_1EPS,AMPL(2,1010))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQG_1EPS,AMPL(2,1011))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQB,AMPL(1,1012))
      CALL MP_FFV1_0(W(30),W(22),W(16),UV_GQQT,AMPL(1,1013))
C     Counter-term amplitude(s) for loop diagram number 559
      CALL MP_VVV1_0(W(4),W(5),W(130),R2_3GG,AMPL(1,1014))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1015))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1016))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1017))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1018))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1019))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB_1EPS,AMPL(2,1020))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GG_1EPS,AMPL(2,1021))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GB,AMPL(1,1022))
      CALL MP_VVV1_0(W(4),W(5),W(130),UV_3GT,AMPL(1,1023))
C     Counter-term amplitude(s) for loop diagram number 572
      CALL MP_R2_QQ_1_0(W(33),W(40),R2_QQQ,AMPL(1,1024))
      CALL MP_FFV1_2(W(33),W(5),GC_5,ZERO,ZERO,W(133))
C     Counter-term amplitude(s) for loop diagram number 573
      CALL MP_R2_QQ_1_0(W(133),W(8),R2_QQQ,AMPL(1,1025))
C     Counter-term amplitude(s) for loop diagram number 574
      CALL MP_FFV1_0(W(33),W(8),W(5),R2_GQQ,AMPL(1,1026))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1027))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1028))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1029))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1030))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1031))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB_1EPS,AMPL(2,1032))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQG_1EPS,AMPL(2,1033))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQB,AMPL(1,1034))
      CALL MP_FFV1_0(W(33),W(8),W(5),UV_GQQT,AMPL(1,1035))
C     Counter-term amplitude(s) for loop diagram number 575
      CALL MP_R2_QQ_1_0(W(32),W(116),R2_QQQ,AMPL(1,1036))
      CALL MP_FFV2_5_2(W(32),W(3),GC_22,GC_23,ZERO,ZERO,W(134))
C     Counter-term amplitude(s) for loop diagram number 576
      CALL MP_R2_QQ_1_0(W(134),W(8),R2_QQQ,AMPL(1,1037))
C     Counter-term amplitude(s) for loop diagram number 577
      CALL MP_FFV2_5_0(W(32),W(8),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1038)
     $ )
C     Counter-term amplitude(s) for loop diagram number 578
      CALL MP_FFV1_0(W(30),W(116),W(5),R2_GQQ,AMPL(1,1039))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1040))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1041))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1042))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1043))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1044))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB_1EPS,AMPL(2,1045))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQG_1EPS,AMPL(2,1046))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQB,AMPL(1,1047))
      CALL MP_FFV1_0(W(30),W(116),W(5),UV_GQQT,AMPL(1,1048))
      CALL MP_FFV1_1(W(116),W(5),GC_5,ZERO,ZERO,W(135))
C     Counter-term amplitude(s) for loop diagram number 579
      CALL MP_R2_QQ_1_0(W(30),W(135),R2_QQQ,AMPL(1,1049))
C     Counter-term amplitude(s) for loop diagram number 580
      CALL MP_FFV2_5_0(W(30),W(40),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1050))
      CALL MP_FFV2_5_1(W(40),W(3),GC_22,GC_23,ZERO,ZERO,W(136))
C     Counter-term amplitude(s) for loop diagram number 581
      CALL MP_R2_QQ_1_0(W(30),W(136),R2_QQQ,AMPL(1,1051))
C     Counter-term amplitude(s) for loop diagram number 587
      CALL MP_R2_QQ_1_0(W(33),W(49),R2_QQQ,AMPL(1,1052))
      CALL MP_FFV1_2(W(33),W(4),GC_5,ZERO,ZERO,W(137))
C     Counter-term amplitude(s) for loop diagram number 588
      CALL MP_R2_QQ_1_0(W(137),W(12),R2_QQQ,AMPL(1,1053))
C     Counter-term amplitude(s) for loop diagram number 589
      CALL MP_FFV1_0(W(33),W(12),W(4),R2_GQQ,AMPL(1,1054))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1055))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1056))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1057))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1058))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1059))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB_1EPS,AMPL(2,1060))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQG_1EPS,AMPL(2,1061))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQB,AMPL(1,1062))
      CALL MP_FFV1_0(W(33),W(12),W(4),UV_GQQT,AMPL(1,1063))
C     Counter-term amplitude(s) for loop diagram number 590
      CALL MP_R2_QQ_1_0(W(31),W(96),R2_QQQ,AMPL(1,1064))
      CALL MP_FFV2_5_2(W(31),W(3),GC_22,GC_23,ZERO,ZERO,W(138))
C     Counter-term amplitude(s) for loop diagram number 591
      CALL MP_R2_QQ_1_0(W(138),W(12),R2_QQQ,AMPL(1,1065))
C     Counter-term amplitude(s) for loop diagram number 592
      CALL MP_FFV2_5_0(W(31),W(12),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1066))
C     Counter-term amplitude(s) for loop diagram number 593
      CALL MP_FFV1_0(W(30),W(96),W(4),R2_GQQ,AMPL(1,1067))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1068))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1069))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1070))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1071))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1072))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB_1EPS,AMPL(2,1073))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQG_1EPS,AMPL(2,1074))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQB,AMPL(1,1075))
      CALL MP_FFV1_0(W(30),W(96),W(4),UV_GQQT,AMPL(1,1076))
      CALL MP_FFV1_1(W(96),W(4),GC_5,ZERO,ZERO,W(139))
C     Counter-term amplitude(s) for loop diagram number 594
      CALL MP_R2_QQ_1_0(W(30),W(139),R2_QQQ,AMPL(1,1077))
C     Counter-term amplitude(s) for loop diagram number 595
      CALL MP_FFV2_5_0(W(30),W(49),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1078))
      CALL MP_FFV2_5_1(W(49),W(3),GC_22,GC_23,ZERO,ZERO,W(140))
C     Counter-term amplitude(s) for loop diagram number 596
      CALL MP_R2_QQ_1_0(W(30),W(140),R2_QQQ,AMPL(1,1079))
C     Counter-term amplitude(s) for loop diagram number 602
      CALL MP_FFV1_0(W(133),W(2),W(4),R2_GQQ,AMPL(1,1080))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1081))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1082))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1083))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1084))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1085))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1086))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQG_1EPS,AMPL(2,1087))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQB,AMPL(1,1088))
      CALL MP_FFV1_0(W(133),W(2),W(4),UV_GQQT,AMPL(1,1089))
C     Counter-term amplitude(s) for loop diagram number 604
      CALL MP_FFV1_0(W(137),W(2),W(5),R2_GQQ,AMPL(1,1090))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1091))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1092))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1093))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1094))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1095))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1096))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQG_1EPS,AMPL(2,1097))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQB,AMPL(1,1098))
      CALL MP_FFV1_0(W(137),W(2),W(5),UV_GQQT,AMPL(1,1099))
C     Counter-term amplitude(s) for loop diagram number 606
      CALL MP_FFV2_5_0(W(126),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1100))
C     Counter-term amplitude(s) for loop diagram number 608
      CALL MP_FFV1_0(W(138),W(2),W(5),R2_GQQ,AMPL(1,1101))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1102))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1103))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1104))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1105))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1106))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB_1EPS,AMPL(2,1107))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQG_1EPS,AMPL(2,1108))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQB,AMPL(1,1109))
      CALL MP_FFV1_0(W(138),W(2),W(5),UV_GQQT,AMPL(1,1110))
C     Counter-term amplitude(s) for loop diagram number 610
      CALL MP_FFV2_5_0(W(127),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1111))
C     Counter-term amplitude(s) for loop diagram number 612
      CALL MP_FFV1_0(W(134),W(2),W(4),R2_GQQ,AMPL(1,1112))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1113))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1114))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1115))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1116))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1117))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB_1EPS,AMPL(2,1118))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQG_1EPS,AMPL(2,1119))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQB,AMPL(1,1120))
      CALL MP_FFV1_0(W(134),W(2),W(4),UV_GQQT,AMPL(1,1121))
C     Counter-term amplitude(s) for loop diagram number 620
      CALL MP_FFV1_0(W(33),W(2),W(16),R2_GQQ,AMPL(1,1122))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1123))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1124))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1125))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1126))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1127))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB_1EPS,AMPL(2,1128))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQG_1EPS,AMPL(2,1129))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQB,AMPL(1,1130))
      CALL MP_FFV1_0(W(33),W(2),W(16),UV_GQQT,AMPL(1,1131))
C     Counter-term amplitude(s) for loop diagram number 622
      CALL MP_FFV2_5_0(W(131),W(2),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1132))
C     Counter-term amplitude(s) for loop diagram number 644
      CALL MP_R2_QQ_1_0(W(33),W(69),R2_QQQ,AMPL(1,1133))
      CALL MP_FFV1P0_3(W(33),W(2),GC_5,ZERO,ZERO,W(141))
C     Counter-term amplitude(s) for loop diagram number 645
      CALL MP_R2_GG_1_R2_GG_2_0(W(16),W(141),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,1134))
C     Counter-term amplitude(s) for loop diagram number 646
      CALL MP_FFV2_5_0(W(30),W(69),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1
     $ ,1135))
      CALL MP_FFV2_5_1(W(69),W(3),GC_22,GC_23,ZERO,ZERO,W(142))
C     Counter-term amplitude(s) for loop diagram number 647
      CALL MP_R2_QQ_1_0(W(30),W(142),R2_QQQ,AMPL(1,1136))
C     Counter-term amplitude(s) for loop diagram number 648
      CALL MP_VVV1_0(W(4),W(5),W(141),R2_3GG,AMPL(1,1137))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1138))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1139))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1140))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1141))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1142))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB_1EPS,AMPL(2,1143))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GG_1EPS,AMPL(2,1144))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GB,AMPL(1,1145))
      CALL MP_VVV1_0(W(4),W(5),W(141),UV_3GT,AMPL(1,1146))
C     Counter-term amplitude(s) for loop diagram number 653
      CALL MP_FFV1_0(W(1),W(90),W(4),R2_GQQ,AMPL(1,1147))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1148))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1149))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1150))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1151))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1152))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB_1EPS,AMPL(2,1153))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQG_1EPS,AMPL(2,1154))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQB,AMPL(1,1155))
      CALL MP_FFV1_0(W(1),W(90),W(4),UV_GQQT,AMPL(1,1156))
C     Counter-term amplitude(s) for loop diagram number 655
      CALL MP_FFV1_0(W(1),W(91),W(4),R2_GQQ,AMPL(1,1157))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1158))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1159))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1160))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1161))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1162))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB_1EPS,AMPL(2,1163))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQG_1EPS,AMPL(2,1164))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQB,AMPL(1,1165))
      CALL MP_FFV1_0(W(1),W(91),W(4),UV_GQQT,AMPL(1,1166))
C     Counter-term amplitude(s) for loop diagram number 659
      CALL MP_FFV1_0(W(1),W(110),W(5),R2_GQQ,AMPL(1,1167))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1168))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1169))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1170))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1171))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1172))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB_1EPS,AMPL(2,1173))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQG_1EPS,AMPL(2,1174))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQB,AMPL(1,1175))
      CALL MP_FFV1_0(W(1),W(110),W(5),UV_GQQT,AMPL(1,1176))
C     Counter-term amplitude(s) for loop diagram number 661
      CALL MP_FFV1_0(W(1),W(111),W(5),R2_GQQ,AMPL(1,1177))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1178))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1179))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1180))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1181))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1182))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB_1EPS,AMPL(2,1183))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQG_1EPS,AMPL(2,1184))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQB,AMPL(1,1185))
      CALL MP_FFV1_0(W(1),W(111),W(5),UV_GQQT,AMPL(1,1186))
C     Counter-term amplitude(s) for loop diagram number 665
      CALL MP_FFV1_0(W(1),W(128),W(6),R2_GQQ,AMPL(1,1187))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1188))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1189))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1190))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1191))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1192))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB_1EPS,AMPL(2,1193))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQG_1EPS,AMPL(2,1194))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQB,AMPL(1,1195))
      CALL MP_FFV1_0(W(1),W(128),W(6),UV_GQQT,AMPL(1,1196))
C     Counter-term amplitude(s) for loop diagram number 667
      CALL MP_FFV1_0(W(1),W(129),W(6),R2_GQQ,AMPL(1,1197))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1198))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1199))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1200))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1201))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1202))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB_1EPS,AMPL(2,1203))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQG_1EPS,AMPL(2,1204))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQB,AMPL(1,1205))
      CALL MP_FFV1_0(W(1),W(129),W(6),UV_GQQT,AMPL(1,1206))
C     Counter-term amplitude(s) for loop diagram number 670
      CALL MP_FFV1_0(W(1),W(22),W(65),R2_GQQ,AMPL(1,1207))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1208))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1209))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1210))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1211))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1212))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB_1EPS,AMPL(2,1213))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQG_1EPS,AMPL(2,1214))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQB,AMPL(1,1215))
      CALL MP_FFV1_0(W(1),W(22),W(65),UV_GQQT,AMPL(1,1216))
C     Counter-term amplitude(s) for loop diagram number 673
      CALL MP_FFV1_0(W(1),W(86),W(16),R2_GQQ,AMPL(1,1217))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1218))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1219))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1220))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1221))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1222))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB_1EPS,AMPL(2,1223))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQG_1EPS,AMPL(2,1224))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQB,AMPL(1,1225))
      CALL MP_FFV1_0(W(1),W(86),W(16),UV_GQQT,AMPL(1,1226))
C     Counter-term amplitude(s) for loop diagram number 675
      CALL MP_FFV1_0(W(1),W(132),W(6),R2_GQQ,AMPL(1,1227))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1228))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1229))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1230))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1231))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1232))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB_1EPS,AMPL(2,1233))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQG_1EPS,AMPL(2,1234))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQB,AMPL(1,1235))
      CALL MP_FFV1_0(W(1),W(132),W(6),UV_GQQT,AMPL(1,1236))
C     Counter-term amplitude(s) for loop diagram number 683
      CALL MP_FFV1_0(W(1),W(22),W(66),R2_GQQ,AMPL(1,1237))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1238))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1239))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1240))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1241))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1242))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB_1EPS,AMPL(2,1243))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQG_1EPS,AMPL(2,1244))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQB,AMPL(1,1245))
      CALL MP_FFV1_0(W(1),W(22),W(66),UV_GQQT,AMPL(1,1246))
C     Counter-term amplitude(s) for loop diagram number 686
      CALL MP_FFV1_0(W(1),W(88),W(14),R2_GQQ,AMPL(1,1247))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1248))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1249))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1250))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1251))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1252))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB_1EPS,AMPL(2,1253))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQG_1EPS,AMPL(2,1254))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQB,AMPL(1,1255))
      CALL MP_FFV1_0(W(1),W(88),W(14),UV_GQQT,AMPL(1,1256))
C     Counter-term amplitude(s) for loop diagram number 688
      CALL MP_FFV1_0(W(1),W(114),W(5),R2_GQQ,AMPL(1,1257))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1258))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1259))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1260))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1261))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1262))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB_1EPS,AMPL(2,1263))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQG_1EPS,AMPL(2,1264))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQB,AMPL(1,1265))
      CALL MP_FFV1_0(W(1),W(114),W(5),UV_GQQT,AMPL(1,1266))
C     Counter-term amplitude(s) for loop diagram number 710
      CALL MP_FFV1_0(W(1),W(94),W(4),R2_GQQ,AMPL(1,1267))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1268))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1269))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1270))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1271))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1272))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB_1EPS,AMPL(2,1273))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQG_1EPS,AMPL(2,1274))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQB,AMPL(1,1275))
      CALL MP_FFV1_0(W(1),W(94),W(4),UV_GQQT,AMPL(1,1276))
C     Counter-term amplitude(s) for loop diagram number 712
      CALL MP_FFV1_0(W(1),W(22),W(67),R2_GQQ,AMPL(1,1277))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1278))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1279))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1280))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1281))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1282))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB_1EPS,AMPL(2,1283))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQG_1EPS,AMPL(2,1284))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQB,AMPL(1,1285))
      CALL MP_FFV1_0(W(1),W(22),W(67),UV_GQQT,AMPL(1,1286))
C     Counter-term amplitude(s) for loop diagram number 718
      CALL MP_FFV1_0(W(1),W(108),W(11),R2_GQQ,AMPL(1,1287))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1288))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1289))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1290))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1291))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1292))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB_1EPS,AMPL(2,1293))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQG_1EPS,AMPL(2,1294))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQB,AMPL(1,1295))
      CALL MP_FFV1_0(W(1),W(108),W(11),UV_GQQT,AMPL(1,1296))
C     Counter-term amplitude(s) for loop diagram number 748
      CALL MP_FFV1_0(W(1),W(22),W(18),R2_GQQ,AMPL(1,1297))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1298))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1299))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1300))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1301))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1302))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB_1EPS,AMPL(2,1303))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQG_1EPS,AMPL(2,1304))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQB,AMPL(1,1305))
      CALL MP_FFV1_0(W(1),W(22),W(18),UV_GQQT,AMPL(1,1306))
      CALL MP_FFV1_0(W(1),W(22),W(19),R2_GQQ,AMPL(1,1307))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1308))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1309))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1310))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1311))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1312))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB_1EPS,AMPL(2,1313))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQG_1EPS,AMPL(2,1314))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQB,AMPL(1,1315))
      CALL MP_FFV1_0(W(1),W(22),W(19),UV_GQQT,AMPL(1,1316))
      CALL MP_FFV1_0(W(1),W(22),W(20),R2_GQQ,AMPL(1,1317))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1318))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1319))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1320))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1321))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1322))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB_1EPS,AMPL(2,1323))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQG_1EPS,AMPL(2,1324))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQB,AMPL(1,1325))
      CALL MP_FFV1_0(W(1),W(22),W(20),UV_GQQT,AMPL(1,1326))
C     Counter-term amplitude(s) for loop diagram number 760
      CALL MP_FFV2_5_0(W(1),W(42),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1327)
     $ )
C     Counter-term amplitude(s) for loop diagram number 762
      CALL MP_FFV2_5_0(W(1),W(43),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1328)
     $ )
C     Counter-term amplitude(s) for loop diagram number 766
      CALL MP_FFV1_0(W(1),W(118),W(5),R2_GQQ,AMPL(1,1329))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1330))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1331))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1332))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1333))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1334))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB_1EPS,AMPL(2,1335))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQG_1EPS,AMPL(2,1336))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQB,AMPL(1,1337))
      CALL MP_FFV1_0(W(1),W(118),W(5),UV_GQQT,AMPL(1,1338))
C     Counter-term amplitude(s) for loop diagram number 768
      CALL MP_FFV1_0(W(1),W(119),W(5),R2_GQQ,AMPL(1,1339))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1340))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1341))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1342))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1343))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1344))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB_1EPS,AMPL(2,1345))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQG_1EPS,AMPL(2,1346))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQB,AMPL(1,1347))
      CALL MP_FFV1_0(W(1),W(119),W(5),UV_GQQT,AMPL(1,1348))
C     Counter-term amplitude(s) for loop diagram number 772
      CALL MP_FFV1_0(W(1),W(135),W(6),R2_GQQ,AMPL(1,1349))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1350))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1351))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1352))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1353))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1354))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB_1EPS,AMPL(2,1355))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQG_1EPS,AMPL(2,1356))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQB,AMPL(1,1357))
      CALL MP_FFV1_0(W(1),W(135),W(6),UV_GQQT,AMPL(1,1358))
C     Counter-term amplitude(s) for loop diagram number 774
      CALL MP_FFV1_0(W(1),W(136),W(6),R2_GQQ,AMPL(1,1359))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1360))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1361))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1362))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1363))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1364))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB_1EPS,AMPL(2,1365))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQG_1EPS,AMPL(2,1366))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQB,AMPL(1,1367))
      CALL MP_FFV1_0(W(1),W(136),W(6),UV_GQQT,AMPL(1,1368))
C     Counter-term amplitude(s) for loop diagram number 777
      CALL MP_FFV2_5_0(W(1),W(46),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1369)
     $ )
C     Counter-term amplitude(s) for loop diagram number 780
      CALL MP_FFV1_0(W(1),W(116),W(11),R2_GQQ,AMPL(1,1370))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1371))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1372))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1373))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1374))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1375))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB_1EPS,AMPL(2,1376))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQG_1EPS,AMPL(2,1377))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQB,AMPL(1,1378))
      CALL MP_FFV1_0(W(1),W(116),W(11),UV_GQQT,AMPL(1,1379))
C     Counter-term amplitude(s) for loop diagram number 802
      CALL MP_FFV2_5_0(W(1),W(51),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1380)
     $ )
C     Counter-term amplitude(s) for loop diagram number 804
      CALL MP_FFV2_5_0(W(1),W(52),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1381)
     $ )
C     Counter-term amplitude(s) for loop diagram number 808
      CALL MP_FFV1_0(W(1),W(98),W(4),R2_GQQ,AMPL(1,1382))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1383))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1384))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1385))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1386))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1387))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB_1EPS,AMPL(2,1388))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQG_1EPS,AMPL(2,1389))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQB,AMPL(1,1390))
      CALL MP_FFV1_0(W(1),W(98),W(4),UV_GQQT,AMPL(1,1391))
C     Counter-term amplitude(s) for loop diagram number 810
      CALL MP_FFV1_0(W(1),W(99),W(4),R2_GQQ,AMPL(1,1392))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1393))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1394))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1395))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1396))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1397))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB_1EPS,AMPL(2,1398))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQG_1EPS,AMPL(2,1399))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQB,AMPL(1,1400))
      CALL MP_FFV1_0(W(1),W(99),W(4),UV_GQQT,AMPL(1,1401))
C     Counter-term amplitude(s) for loop diagram number 814
      CALL MP_FFV1_0(W(1),W(139),W(6),R2_GQQ,AMPL(1,1402))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1403))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1404))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1405))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1406))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1407))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB_1EPS,AMPL(2,1408))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQG_1EPS,AMPL(2,1409))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQB,AMPL(1,1410))
      CALL MP_FFV1_0(W(1),W(139),W(6),UV_GQQT,AMPL(1,1411))
C     Counter-term amplitude(s) for loop diagram number 816
      CALL MP_FFV1_0(W(1),W(140),W(6),R2_GQQ,AMPL(1,1412))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1413))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1414))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1415))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1416))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1417))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB_1EPS,AMPL(2,1418))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQG_1EPS,AMPL(2,1419))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQB,AMPL(1,1420))
      CALL MP_FFV1_0(W(1),W(140),W(6),UV_GQQT,AMPL(1,1421))
C     Counter-term amplitude(s) for loop diagram number 819
      CALL MP_FFV2_5_0(W(1),W(55),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1422)
     $ )
C     Counter-term amplitude(s) for loop diagram number 822
      CALL MP_FFV1_0(W(1),W(96),W(14),R2_GQQ,AMPL(1,1423))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1424))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1425))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1426))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1427))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1428))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB_1EPS,AMPL(2,1429))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQG_1EPS,AMPL(2,1430))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQB,AMPL(1,1431))
      CALL MP_FFV1_0(W(1),W(96),W(14),UV_GQQT,AMPL(1,1432))
C     Counter-term amplitude(s) for loop diagram number 844
      CALL MP_FFV2_5_0(W(1),W(60),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1433)
     $ )
C     Counter-term amplitude(s) for loop diagram number 846
      CALL MP_FFV2_5_0(W(1),W(61),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1434)
     $ )
C     Counter-term amplitude(s) for loop diagram number 850
      CALL MP_FFV1_0(W(1),W(103),W(4),R2_GQQ,AMPL(1,1435))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1436))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1437))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1438))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1439))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1440))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB_1EPS,AMPL(2,1441))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQG_1EPS,AMPL(2,1442))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQB,AMPL(1,1443))
      CALL MP_FFV1_0(W(1),W(103),W(4),UV_GQQT,AMPL(1,1444))
C     Counter-term amplitude(s) for loop diagram number 852
      CALL MP_FFV1_0(W(1),W(104),W(4),R2_GQQ,AMPL(1,1445))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1446))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1447))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1448))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1449))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1450))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB_1EPS,AMPL(2,1451))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQG_1EPS,AMPL(2,1452))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQB,AMPL(1,1453))
      CALL MP_FFV1_0(W(1),W(104),W(4),UV_GQQT,AMPL(1,1454))
C     Counter-term amplitude(s) for loop diagram number 856
      CALL MP_FFV1_0(W(1),W(122),W(5),R2_GQQ,AMPL(1,1455))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1456))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1457))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1458))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1459))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1460))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB_1EPS,AMPL(2,1461))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQG_1EPS,AMPL(2,1462))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQB,AMPL(1,1463))
      CALL MP_FFV1_0(W(1),W(122),W(5),UV_GQQT,AMPL(1,1464))
C     Counter-term amplitude(s) for loop diagram number 858
      CALL MP_FFV1_0(W(1),W(123),W(5),R2_GQQ,AMPL(1,1465))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1466))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1467))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1468))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1469))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1470))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB_1EPS,AMPL(2,1471))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQG_1EPS,AMPL(2,1472))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQB,AMPL(1,1473))
      CALL MP_FFV1_0(W(1),W(123),W(5),UV_GQQT,AMPL(1,1474))
C     Counter-term amplitude(s) for loop diagram number 861
      CALL MP_FFV2_5_0(W(1),W(64),W(3),R2_UUZ_V2,R2_DDZ_V3,AMPL(1,1475)
     $ )
C     Counter-term amplitude(s) for loop diagram number 864
      CALL MP_FFV1_0(W(1),W(101),W(16),R2_GQQ,AMPL(1,1476))

      GOTO 1001
 2000 CONTINUE
      MP_CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

