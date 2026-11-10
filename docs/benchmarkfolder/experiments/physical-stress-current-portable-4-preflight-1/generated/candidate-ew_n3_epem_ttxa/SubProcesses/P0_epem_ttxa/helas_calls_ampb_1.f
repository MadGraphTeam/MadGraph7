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
      PARAMETER (NCOMB=32)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=8)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=921, NLOOPGROUPS=200, NCTAMPS=354)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=1275)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=35,NLOOPWAVEFUNCS=1362)
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

      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1, FLAVOR(3),W(3))
      CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1, FLAVOR(4),W(4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1,W(5))
      CALL FFV1P0_3(W(2),W(1),GC_3,ZERO,ZERO,W(6))
      CALL FFV1_1(W(3),W(5),GC_2,MDL_MT,MDL_WT,W(7))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(4),W(7),W(6),GC_2,AMP(1))
      CALL FFV2_4P0_3(W(2),W(1),GC_68,GC_77,MDL_MZ,MDL_WZ,W(8))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV2_5_0(W(4),W(7),W(8),GC_69,GC_76,AMP(2))
      CALL FFV1_2(W(4),W(5),GC_2,MDL_MT,MDL_WT,W(9))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV1_0(W(9),W(3),W(6),GC_2,AMP(3))
C     Amplitude(s) for born diagram with ID 4
      CALL FFV2_5_0(W(9),W(3),W(8),GC_69,GC_76,AMP(4))
      CALL FFV1_1(W(1),W(5),GC_3,ZERO,ZERO,W(10))
      CALL FFV1P0_3(W(4),W(3),GC_2,ZERO,ZERO,W(11))
C     Amplitude(s) for born diagram with ID 5
      CALL FFV1_0(W(2),W(10),W(11),GC_3,AMP(5))
      CALL FFV2_5P0_3(W(4),W(3),GC_69,GC_76,MDL_MZ,MDL_WZ,W(12))
C     Amplitude(s) for born diagram with ID 6
      CALL FFV2_4_0(W(2),W(10),W(12),GC_68,GC_77,AMP(6))
      CALL FFV1_2(W(2),W(5),GC_3,ZERO,ZERO,W(13))
C     Amplitude(s) for born diagram with ID 7
      CALL FFV1_0(W(13),W(1),W(11),GC_3,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL FFV2_4_0(W(13),W(1),W(12),GC_68,GC_77,AMP(8))
      CALL FFV1_2(W(4),W(6),GC_2,MDL_MT,MDL_WT,W(14))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(14),W(7)
     $ ,C_UVTTXMASS3EW_1EPS,C_UVTTXMASS1EW_1EPS,C_UVTTXMASS2EW_1EPS
     $ ,AMPL(2,1))
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(14),W(7)
     $ ,C_UVTTXMASS3EW,C_UVTTXMASS1EW,C_UVTTXMASS2EW,AMPL(1,2))
      CALL FFV2_5_2(W(4),W(8),GC_69,GC_76,MDL_MT,MDL_WT,W(15))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(15),W(7)
     $ ,C_UVTTXMASS3EW_1EPS,C_UVTTXMASS1EW_1EPS,C_UVTTXMASS2EW_1EPS
     $ ,AMPL(2,3))
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(15),W(7)
     $ ,C_UVTTXMASS3EW,C_UVTTXMASS1EW,C_UVTTXMASS2EW,AMPL(1,4))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(7),W(6),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,5))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(7),W(6),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,6))
C     Counter-term amplitude(s) for loop diagram number 12
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(7),W(8),C_UVTXTZ1EW_1EPS
     $ ,C_UVTXTZ2EW_1EPS,AMPL(2,7))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(7),W(8),C_UVTXTZ1EW
     $ ,C_UVTXTZ2EW,AMPL(1,8))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL L_VEXVEA21_L_VEXVEA22_0(W(14),W(3),W(5),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,9))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(14),W(3),W(5),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,10))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL L_VEXVEA21_L_VEXVEA22_0(W(15),W(3),W(5),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,11))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(15),W(3),W(5),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,12))
C     Counter-term amplitude(s) for loop diagram number 15
      CALL L_VEXVEA21_L_VEXVEA22_0(W(9),W(3),W(6),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,13))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(9),W(3),W(6),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 16
      CALL L_VEXVEA21_L_VEXVEA22_0(W(9),W(3),W(8),C_UVTXTZ1EW_1EPS
     $ ,C_UVTXTZ2EW_1EPS,AMPL(2,15))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(9),W(3),W(8),C_UVTXTZ1EW
     $ ,C_UVTXTZ2EW,AMPL(1,16))
      CALL FFV1_1(W(3),W(6),GC_2,MDL_MT,MDL_WT,W(16))
C     Counter-term amplitude(s) for loop diagram number 21
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(9),W(16)
     $ ,C_UVTTXMASS3EW_1EPS,C_UVTTXMASS1EW_1EPS,C_UVTTXMASS2EW_1EPS
     $ ,AMPL(2,17))
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(9),W(16)
     $ ,C_UVTTXMASS3EW,C_UVTTXMASS1EW,C_UVTTXMASS2EW,AMPL(1,18))
      CALL FFV2_5_1(W(3),W(8),GC_69,GC_76,MDL_MT,MDL_WT,W(17))
C     Counter-term amplitude(s) for loop diagram number 22
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(9),W(17)
     $ ,C_UVTTXMASS3EW_1EPS,C_UVTTXMASS1EW_1EPS,C_UVTTXMASS2EW_1EPS
     $ ,AMPL(2,19))
      CALL L_VEVEXMASS12_L_VEVEXMASS8_L_VEVEXMASS9_0(W(9),W(17)
     $ ,C_UVTTXMASS3EW,C_UVTTXMASS1EW,C_UVTTXMASS2EW,AMPL(1,20))
C     Counter-term amplitude(s) for loop diagram number 23
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(16),W(5),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,21))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(16),W(5),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,22))
C     Counter-term amplitude(s) for loop diagram number 24
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(17),W(5),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,23))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(17),W(5),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,24))
C     Counter-term amplitude(s) for loop diagram number 25
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(10),W(11),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,25))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(10),W(11),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,26))
C     Counter-term amplitude(s) for loop diagram number 26
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(10),W(12),C_UVEPEMZ1EW_1EPS
     $ ,C_UVEPEMZ2EW_1EPS,AMPL(2,27))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(10),W(12),C_UVEPEMZ1EW
     $ ,C_UVEPEMZ2EW,AMPL(1,28))
      CALL FFV1_2(W(2),W(11),GC_3,ZERO,ZERO,W(18))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(18),W(10)
     $ ,C_UVEMEPMASS1EW_1EPS,C_UVEMEPMASS2EW_1EPS,AMPL(2,29))
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(18),W(10),C_UVEMEPMASS1EW
     $ ,C_UVEMEPMASS2EW,AMPL(1,30))
      CALL FFV2_4_2(W(2),W(12),GC_68,GC_77,ZERO,ZERO,W(19))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(19),W(10)
     $ ,C_UVEMEPMASS1EW_1EPS,C_UVEMEPMASS2EW_1EPS,AMPL(2,31))
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(19),W(10),C_UVEMEPMASS1EW
     $ ,C_UVEMEPMASS2EW,AMPL(1,32))
      CALL FFV1P0_3(W(2),W(10),GC_3,ZERO,ZERO,W(20))
C     Counter-term amplitude(s) for loop diagram number 34
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(20),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,33))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(20),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,34))
      CALL FFV2_4P0_3(W(2),W(10),GC_68,GC_77,MDL_MZ,MDL_WZ,W(21))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(21),C_UVTXTZ1EW_1EPS
     $ ,C_UVTXTZ2EW_1EPS,AMPL(2,35))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(21),C_UVTXTZ1EW
     $ ,C_UVTXTZ2EW,AMPL(1,36))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL L_VEXVEA21_L_VEXVEA22_0(W(13),W(1),W(11),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,37))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(13),W(1),W(11),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,38))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL L_VEXVEA21_L_VEXVEA22_0(W(13),W(1),W(12),C_UVEPEMZ1EW_1EPS
     $ ,C_UVEPEMZ2EW_1EPS,AMPL(2,39))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(13),W(1),W(12),C_UVEPEMZ1EW
     $ ,C_UVEPEMZ2EW,AMPL(1,40))
      CALL FFV1P0_3(W(4),W(7),GC_2,ZERO,ZERO,W(22))
C     Counter-term amplitude(s) for loop diagram number 47
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(22),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,41))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(22),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,42))
      CALL FFV2_5P0_3(W(4),W(7),GC_69,GC_76,MDL_MZ,MDL_WZ,W(23))
C     Counter-term amplitude(s) for loop diagram number 48
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(23),C_UVEPEMZ1EW_1EPS
     $ ,C_UVEPEMZ2EW_1EPS,AMPL(2,43))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(23),C_UVEPEMZ1EW
     $ ,C_UVEPEMZ2EW,AMPL(1,44))
      CALL FFV1P0_3(W(9),W(3),GC_2,ZERO,ZERO,W(24))
C     Counter-term amplitude(s) for loop diagram number 49
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(24),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,45))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(24),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,46))
      CALL FFV2_5P0_3(W(9),W(3),GC_69,GC_76,MDL_MZ,MDL_WZ,W(25))
C     Counter-term amplitude(s) for loop diagram number 50
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(25),C_UVEPEMZ1EW_1EPS
     $ ,C_UVEPEMZ2EW_1EPS,AMPL(2,47))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(1),W(25),C_UVEPEMZ1EW
     $ ,C_UVEPEMZ2EW,AMPL(1,48))
C     Counter-term amplitude(s) for loop diagram number 51
      CALL L_VEXVEA21_L_VEXVEA22_0(W(18),W(1),W(5),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,49))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(18),W(1),W(5),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,50))
C     Counter-term amplitude(s) for loop diagram number 52
      CALL L_VEXVEA21_L_VEXVEA22_0(W(19),W(1),W(5),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,51))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(19),W(1),W(5),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,52))
      CALL FFV1_1(W(1),W(11),GC_3,ZERO,ZERO,W(26))
C     Counter-term amplitude(s) for loop diagram number 69
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(13),W(26)
     $ ,C_UVEMEPMASS1EW_1EPS,C_UVEMEPMASS2EW_1EPS,AMPL(2,53))
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(13),W(26),C_UVEMEPMASS1EW
     $ ,C_UVEMEPMASS2EW,AMPL(1,54))
      CALL FFV2_4_1(W(1),W(12),GC_68,GC_77,ZERO,ZERO,W(27))
C     Counter-term amplitude(s) for loop diagram number 70
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(13),W(27)
     $ ,C_UVEMEPMASS1EW_1EPS,C_UVEMEPMASS2EW_1EPS,AMPL(2,55))
      CALL L_VEVEXMASS8_L_VEVEXMASS9_0(W(13),W(27),C_UVEMEPMASS1EW
     $ ,C_UVEMEPMASS2EW,AMPL(1,56))
      CALL FFV1P0_3(W(13),W(1),GC_3,ZERO,ZERO,W(28))
C     Counter-term amplitude(s) for loop diagram number 72
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(28),C_UVTXTA1EW_1EPS
     $ ,C_UVTXTA2EW_1EPS,AMPL(2,57))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(28),C_UVTXTA1EW
     $ ,C_UVTXTA2EW,AMPL(1,58))
      CALL FFV2_4P0_3(W(13),W(1),GC_68,GC_77,MDL_MZ,MDL_WZ,W(29))
C     Counter-term amplitude(s) for loop diagram number 73
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(29),C_UVTXTZ1EW_1EPS
     $ ,C_UVTXTZ2EW_1EPS,AMPL(2,59))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(4),W(3),W(29),C_UVTXTZ1EW
     $ ,C_UVTXTZ2EW,AMPL(1,60))
C     Counter-term amplitude(s) for loop diagram number 75
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(26),W(5),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,61))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(26),W(5),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,62))
C     Counter-term amplitude(s) for loop diagram number 76
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(27),W(5),C_UVEPEMA1EW_1EPS
     $ ,C_UVEPEMA2EW_1EPS,AMPL(2,63))
      CALL L_VEXVEA21_L_VEXVEA22_0(W(2),W(27),W(5),C_UVEPEMA1EW
     $ ,C_UVEPEMA2EW,AMPL(1,64))
C     Counter-term amplitude(s) for loop diagram number 85
      CALL R2_QQ_2_R2_QQ_3_R2_QQ_4_0(W(14),W(7),R2_TTC0,R2_UUCM
     $ ,R2_QQCP0,AMPL(1,65))
C     Counter-term amplitude(s) for loop diagram number 86
      CALL R2_QQ_2_R2_QQ_3_R2_QQ_4_0(W(15),W(7),R2_TTC0,R2_UUCM
     $ ,R2_QQCP0,AMPL(1,66))
      CALL FFS2_3(W(4),W(7),GC_117,MDL_MZ,MDL_WZ,W(30))
C     Counter-term amplitude(s) for loop diagram number 88
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(30),W(8),C_UVG0ZMASS1EW_1EPS
     $ ,C_UVG0ZMASS2EW_1EPS,AMPL(2,67))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(30),W(8),C_UVG0ZMASS1EW
     $ ,C_UVG0ZMASS2EW,AMPL(1,68))
C     Counter-term amplitude(s) for loop diagram number 90
      CALL FFV2_6_0(W(4),W(7),W(6),R2_TTA2CP,R2_TTA2CM,AMPL(1,69))
C     Counter-term amplitude(s) for loop diagram number 91
      CALL FFV2_6_0(W(4),W(7),W(8),R2_TTZ2CP,R2_TTZ2CM,AMPL(1,70))
C     Counter-term amplitude(s) for loop diagram number 93
      CALL FFV2_6_0(W(14),W(3),W(5),R2_TTA2CP,R2_TTA2CM,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 94
      CALL FFV2_6_0(W(15),W(3),W(5),R2_TTA2CP,R2_TTA2CM,AMPL(1,72))
C     Counter-term amplitude(s) for loop diagram number 96
      CALL FFV2_6_0(W(9),W(3),W(6),R2_TTA2CP,R2_TTA2CM,AMPL(1,73))
C     Counter-term amplitude(s) for loop diagram number 97
      CALL FFV2_6_0(W(9),W(3),W(8),R2_TTZ2CP,R2_TTZ2CM,AMPL(1,74))
C     Counter-term amplitude(s) for loop diagram number 103
      CALL R2_QQ_2_R2_QQ_3_R2_QQ_4_0(W(9),W(16),R2_TTC0,R2_UUCM
     $ ,R2_QQCP0,AMPL(1,75))
C     Counter-term amplitude(s) for loop diagram number 104
      CALL R2_QQ_2_R2_QQ_3_R2_QQ_4_0(W(9),W(17),R2_TTC0,R2_UUCM
     $ ,R2_QQCP0,AMPL(1,76))
      CALL FFS2_3(W(9),W(3),GC_117,MDL_MZ,MDL_WZ,W(31))
C     Counter-term amplitude(s) for loop diagram number 106
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(31),W(8),C_UVG0ZMASS1EW_1EPS
     $ ,C_UVG0ZMASS2EW_1EPS,AMPL(2,77))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(31),W(8),C_UVG0ZMASS1EW
     $ ,C_UVG0ZMASS2EW,AMPL(1,78))
C     Counter-term amplitude(s) for loop diagram number 108
      CALL FFV2_6_0(W(4),W(16),W(5),R2_TTA2CP,R2_TTA2CM,AMPL(1,79))
C     Counter-term amplitude(s) for loop diagram number 109
      CALL FFV2_6_0(W(4),W(17),W(5),R2_TTA2CP,R2_TTA2CM,AMPL(1,80))
C     Counter-term amplitude(s) for loop diagram number 111
      CALL FFV2_6_0(W(2),W(10),W(11),R2_EEACP,R2_EEACM,AMPL(1,81))
C     Counter-term amplitude(s) for loop diagram number 112
      CALL FFV2_6_0(W(2),W(10),W(12),R2_EEZCP,R2_EEZCM,AMPL(1,82))
      CALL FFS4_3(W(4),W(3),GC_116,MDL_MH,MDL_WH,W(32))
C     Counter-term amplitude(s) for loop diagram number 116
      CALL R2_QQ_3_R2_QQ_4_0(W(18),W(10),R2_LLCM,R2_LLCP0,AMPL(1,83))
C     Counter-term amplitude(s) for loop diagram number 118
      CALL R2_QQ_3_R2_QQ_4_0(W(19),W(10),R2_LLCM,R2_LLCP0,AMPL(1,84))
      CALL FFS2_3(W(4),W(3),GC_117,MDL_MZ,MDL_WZ,W(33))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(21),C_UVG0ZMASS1EW_1EPS
     $ ,C_UVG0ZMASS2EW_1EPS,AMPL(2,85))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(21),C_UVG0ZMASS1EW
     $ ,C_UVG0ZMASS2EW,AMPL(1,86))
C     Counter-term amplitude(s) for loop diagram number 121
      CALL FFV2_6_0(W(4),W(3),W(20),R2_TTA2CP,R2_TTA2CM,AMPL(1,87))
C     Counter-term amplitude(s) for loop diagram number 122
      CALL FFV2_6_0(W(4),W(3),W(21),R2_TTZ2CP,R2_TTZ2CM,AMPL(1,88))
C     Counter-term amplitude(s) for loop diagram number 124
      CALL FFV2_6_0(W(13),W(1),W(11),R2_EEACP,R2_EEACM,AMPL(1,89))
C     Counter-term amplitude(s) for loop diagram number 125
      CALL FFV2_6_0(W(13),W(1),W(12),R2_EEZCP,R2_EEZCM,AMPL(1,90))
C     Counter-term amplitude(s) for loop diagram number 133
      CALL FFV2_6_0(W(2),W(1),W(22),R2_EEACP,R2_EEACM,AMPL(1,91))
C     Counter-term amplitude(s) for loop diagram number 134
      CALL FFV2_6_0(W(2),W(1),W(23),R2_EEZCP,R2_EEZCM,AMPL(1,92))
C     Counter-term amplitude(s) for loop diagram number 135
      CALL FFV2_6_0(W(2),W(1),W(24),R2_EEACP,R2_EEACM,AMPL(1,93))
C     Counter-term amplitude(s) for loop diagram number 136
      CALL FFV2_6_0(W(2),W(1),W(25),R2_EEZCP,R2_EEZCM,AMPL(1,94))
C     Counter-term amplitude(s) for loop diagram number 137
      CALL FFV2_6_0(W(18),W(1),W(5),R2_EEACP,R2_EEACM,AMPL(1,95))
C     Counter-term amplitude(s) for loop diagram number 138
      CALL FFV2_6_0(W(19),W(1),W(5),R2_EEACP,R2_EEACM,AMPL(1,96))
      CALL FFS4_3(W(4),W(7),GC_116,MDL_MH,MDL_WH,W(34))
      CALL FFS4_3(W(9),W(3),GC_116,MDL_MH,MDL_WH,W(35))
C     Counter-term amplitude(s) for loop diagram number 150
      CALL R2_QQ_3_R2_QQ_4_0(W(13),W(26),R2_LLCM,R2_LLCP0,AMPL(1,97))
C     Counter-term amplitude(s) for loop diagram number 152
      CALL R2_QQ_3_R2_QQ_4_0(W(13),W(27),R2_LLCM,R2_LLCP0,AMPL(1,98))
C     Counter-term amplitude(s) for loop diagram number 153
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(29),C_UVG0ZMASS1EW_1EPS
     $ ,C_UVG0ZMASS2EW_1EPS,AMPL(2,99))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(29),C_UVG0ZMASS1EW
     $ ,C_UVG0ZMASS2EW,AMPL(1,100))
C     Counter-term amplitude(s) for loop diagram number 155
      CALL FFV2_6_0(W(4),W(3),W(28),R2_TTA2CP,R2_TTA2CM,AMPL(1,101))
C     Counter-term amplitude(s) for loop diagram number 156
      CALL FFV2_6_0(W(4),W(3),W(29),R2_TTZ2CP,R2_TTZ2CM,AMPL(1,102))
C     Counter-term amplitude(s) for loop diagram number 158
      CALL FFV2_6_0(W(2),W(26),W(5),R2_EEACP,R2_EEACM,AMPL(1,103))
C     Counter-term amplitude(s) for loop diagram number 159
      CALL FFV2_6_0(W(2),W(27),W(5),R2_EEACP,R2_EEACM,AMPL(1,104))
C     Counter-term amplitude(s) for loop diagram number 165
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(8),W(23),R2_ZZBOSON1
     $ ,R2_ZZBOSON2,R2_ZZBOSON3,AMPL(1,105))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(8),W(23)
     $ ,C_UVZMASS1EW_1EPS,C_UVZMASS2EW_1EPS,C_UVZMASS3EW_1EPS,AMPL(2
     $ ,106))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(8),W(23)
     $ ,C_UVZMASS1EW,C_UVZMASS2EW,C_UVZMASS3EW,AMPL(1,107))
C     Counter-term amplitude(s) for loop diagram number 184
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(8),W(25),R2_ZZBOSON1
     $ ,R2_ZZBOSON2,R2_ZZBOSON3,AMPL(1,108))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(8),W(25)
     $ ,C_UVZMASS1EW_1EPS,C_UVZMASS2EW_1EPS,C_UVZMASS3EW_1EPS,AMPL(2
     $ ,109))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(8),W(25)
     $ ,C_UVZMASS1EW,C_UVZMASS2EW,C_UVZMASS3EW,AMPL(1,110))
C     Counter-term amplitude(s) for loop diagram number 189
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(12),W(21),R2_ZZBOSON1
     $ ,R2_ZZBOSON2,R2_ZZBOSON3,AMPL(1,111))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(12),W(21)
     $ ,C_UVZMASS1EW_1EPS,C_UVZMASS2EW_1EPS,C_UVZMASS3EW_1EPS,AMPL(2
     $ ,112))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(12),W(21)
     $ ,C_UVZMASS1EW,C_UVZMASS2EW,C_UVZMASS3EW,AMPL(1,113))
C     Counter-term amplitude(s) for loop diagram number 196
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(12),W(29),R2_ZZBOSON1
     $ ,R2_ZZBOSON2,R2_ZZBOSON3,AMPL(1,114))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(12),W(29)
     $ ,C_UVZMASS1EW_1EPS,C_UVZMASS2EW_1EPS,C_UVZMASS3EW_1EPS,AMPL(2
     $ ,115))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(12),W(29)
     $ ,C_UVZMASS1EW,C_UVZMASS2EW,C_UVZMASS3EW,AMPL(1,116))
C     Counter-term amplitude(s) for loop diagram number 252
      CALL VVS1_0(W(5),W(6),W(32),R2_AAH,AMPL(1,117))
C     Counter-term amplitude(s) for loop diagram number 289
      CALL VVS1_0(W(5),W(8),W(32),R2_AZH,AMPL(1,118))
      CALL L_HWPWM20_0(W(32),W(8),W(5),C_UVHZA1EW_1EPS,AMPL(2,119))
      CALL L_HWPWM20_0(W(32),W(8),W(5),C_UVHZA1EW,AMPL(1,120))
C     Counter-term amplitude(s) for loop diagram number 356
      CALL R2_QQ_4_0(W(14),W(7),R2_QQCPCS,AMPL(1,121))
C     Counter-term amplitude(s) for loop diagram number 357
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(6),W(22),R2_AABOSON1
     $ ,R2_AABOSON2,R2_AABOSON3,AMPL(1,122))
C     Counter-term amplitude(s) for loop diagram number 358
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(6),W(23),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,123))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(6),W(23)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,124))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(6),W(23)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,125))
C     Counter-term amplitude(s) for loop diagram number 362
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(30),W(6),C_UVAG0MASS1EW_1EPS
     $ ,C_UVAG0MASS2EW_1EPS,AMPL(2,126))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(30),W(6),C_UVAG0MASS1EW
     $ ,C_UVAG0MASS2EW,AMPL(1,127))
C     Counter-term amplitude(s) for loop diagram number 364
      CALL FFV2_6_0(W(4),W(7),W(6),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,128))
C     Counter-term amplitude(s) for loop diagram number 372
      CALL R2_QQ_4_0(W(15),W(7),R2_QQCPCS,AMPL(1,129))
C     Counter-term amplitude(s) for loop diagram number 373
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(22),W(8),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,130))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(22),W(8)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,131))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(22),W(8)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,132))
C     Counter-term amplitude(s) for loop diagram number 380
      CALL FFV2_6_0(W(4),W(7),W(8),R2_CCZ2CP_S,R2_TTZ2CM_B,AMPL(1,133))
C     Counter-term amplitude(s) for loop diagram number 392
      CALL FFV2_6_0(W(14),W(3),W(5),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,134)
     $ )
C     Counter-term amplitude(s) for loop diagram number 400
      CALL FFV2_6_0(W(15),W(3),W(5),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,135)
     $ )
C     Counter-term amplitude(s) for loop diagram number 408
      CALL FFV2_6_0(W(9),W(3),W(6),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,136))
C     Counter-term amplitude(s) for loop diagram number 411
      CALL FFV2_6_0(W(9),W(3),W(8),R2_CCZ2CP_S,R2_TTZ2CM_B,AMPL(1,137))
C     Counter-term amplitude(s) for loop diagram number 434
      CALL R2_QQ_4_0(W(9),W(16),R2_QQCPCS,AMPL(1,138))
C     Counter-term amplitude(s) for loop diagram number 435
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(6),W(24),R2_AABOSON1
     $ ,R2_AABOSON2,R2_AABOSON3,AMPL(1,139))
C     Counter-term amplitude(s) for loop diagram number 436
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(6),W(25),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,140))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(6),W(25)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,141))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(6),W(25)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,142))
C     Counter-term amplitude(s) for loop diagram number 440
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(31),W(6),C_UVAG0MASS1EW_1EPS
     $ ,C_UVAG0MASS2EW_1EPS,AMPL(2,143))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(31),W(6),C_UVAG0MASS1EW
     $ ,C_UVAG0MASS2EW,AMPL(1,144))
C     Counter-term amplitude(s) for loop diagram number 449
      CALL R2_QQ_4_0(W(9),W(17),R2_QQCPCS,AMPL(1,145))
C     Counter-term amplitude(s) for loop diagram number 450
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(24),W(8),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,146))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(24),W(8)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,147))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(24),W(8)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,148))
C     Counter-term amplitude(s) for loop diagram number 464
      CALL FFV2_6_0(W(4),W(16),W(5),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,149)
     $ )
C     Counter-term amplitude(s) for loop diagram number 468
      CALL FFV2_6_0(W(4),W(17),W(5),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,150)
     $ )
C     Counter-term amplitude(s) for loop diagram number 490
      CALL FFV2_0(W(2),W(10),W(11),R2_LLACP,AMPL(1,151))
C     Counter-term amplitude(s) for loop diagram number 491
      CALL FFV2_0(W(2),W(10),W(12),R2_LLZCP,AMPL(1,152))
C     Counter-term amplitude(s) for loop diagram number 495
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(11),W(20),R2_AABOSON1
     $ ,R2_AABOSON2,R2_AABOSON3,AMPL(1,153))
C     Counter-term amplitude(s) for loop diagram number 497
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(11),W(21),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,154))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(11),W(21)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,155))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(11),W(21)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,156))
C     Counter-term amplitude(s) for loop diagram number 503
      CALL R2_QQ_4_0(W(18),W(10),R2_QQCPCS,AMPL(1,157))
C     Counter-term amplitude(s) for loop diagram number 504
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(20),W(12),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,158))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(20),W(12)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,159))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(20),W(12)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,160))
C     Counter-term amplitude(s) for loop diagram number 512
      CALL R2_QQ_4_0(W(19),W(10),R2_QQCPCS,AMPL(1,161))
C     Counter-term amplitude(s) for loop diagram number 513
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(20),C_UVAG0MASS1EW_1EPS
     $ ,C_UVAG0MASS2EW_1EPS,AMPL(2,162))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(20),C_UVAG0MASS1EW
     $ ,C_UVAG0MASS2EW,AMPL(1,163))
C     Counter-term amplitude(s) for loop diagram number 523
      CALL FFV2_6_0(W(4),W(3),W(20),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,164)
     $ )
C     Counter-term amplitude(s) for loop diagram number 525
      CALL FFV2_6_0(W(4),W(3),W(21),R2_CCZ2CP_S,R2_TTZ2CM_B,AMPL(1,165)
     $ )
C     Counter-term amplitude(s) for loop diagram number 531
      CALL FFV2_0(W(13),W(1),W(11),R2_LLACP,AMPL(1,166))
C     Counter-term amplitude(s) for loop diagram number 532
      CALL FFV2_0(W(13),W(1),W(12),R2_LLZCP,AMPL(1,167))
C     Counter-term amplitude(s) for loop diagram number 536
      CALL FFV2_0(W(2),W(1),W(23),R2_LLZCP,AMPL(1,168))
C     Counter-term amplitude(s) for loop diagram number 537
      CALL FFV2_0(W(2),W(1),W(25),R2_LLZCP,AMPL(1,169))
C     Counter-term amplitude(s) for loop diagram number 548
      CALL FFV2_0(W(18),W(1),W(5),R2_LLACP,AMPL(1,170))
C     Counter-term amplitude(s) for loop diagram number 552
      CALL FFV2_0(W(19),W(1),W(5),R2_LLACP,AMPL(1,171))
C     Counter-term amplitude(s) for loop diagram number 557
      CALL FFV2_0(W(2),W(1),W(22),R2_LLACP,AMPL(1,172))
C     Counter-term amplitude(s) for loop diagram number 565
      CALL FFV2_0(W(2),W(1),W(24),R2_LLACP,AMPL(1,173))
C     Counter-term amplitude(s) for loop diagram number 570
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(11),W(28),R2_AABOSON1
     $ ,R2_AABOSON2,R2_AABOSON3,AMPL(1,174))
C     Counter-term amplitude(s) for loop diagram number 572
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(11),W(29),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,175))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(11),W(29)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,176))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(11),W(29)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,177))
C     Counter-term amplitude(s) for loop diagram number 578
      CALL R2_QQ_4_0(W(13),W(26),R2_QQCPCS,AMPL(1,178))
C     Counter-term amplitude(s) for loop diagram number 579
      CALL R2_GG_1_R2_GG_2_R2_GG_3_0(W(28),W(12),R2_AZBOSON1
     $ ,R2_AZBOSON2,R2_AZBOSON3,AMPL(1,179))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(28),W(12)
     $ ,C_UVAZMASS1EW_1EPS,C_UVAZMASS2EW_1EPS,C_UVAZMASS3EW_1EPS
     $ ,AMPL(2,180))
      CALL L_WMWPMASS1_L_WMWPMASS2_L_WMWPMASS3_0(W(28),W(12)
     $ ,C_UVAZMASS1EW,C_UVAZMASS2EW,C_UVAZMASS3EW,AMPL(1,181))
C     Counter-term amplitude(s) for loop diagram number 587
      CALL R2_QQ_4_0(W(13),W(27),R2_QQCPCS,AMPL(1,182))
C     Counter-term amplitude(s) for loop diagram number 588
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(28),C_UVAG0MASS1EW_1EPS
     $ ,C_UVAG0MASS2EW_1EPS,AMPL(2,183))
      CALL L_GPWMMASS4_L_GPWMMASS5_0(W(33),W(28),C_UVAG0MASS1EW
     $ ,C_UVAG0MASS2EW,AMPL(1,184))
C     Counter-term amplitude(s) for loop diagram number 598
      CALL FFV2_6_0(W(4),W(3),W(28),R2_CCA2CP_S,R2_TTA2CM_B,AMPL(1,185)
     $ )
C     Counter-term amplitude(s) for loop diagram number 600
      CALL FFV2_6_0(W(4),W(3),W(29),R2_CCZ2CP_S,R2_TTZ2CM_B,AMPL(1,186)
     $ )
C     Counter-term amplitude(s) for loop diagram number 606
      CALL FFV2_0(W(2),W(26),W(5),R2_LLACP,AMPL(1,187))
C     Counter-term amplitude(s) for loop diagram number 607
      CALL FFV2_0(W(2),W(27),W(5),R2_LLACP,AMPL(1,188))
C     Counter-term amplitude(s) for loop diagram number 818
      CALL R2_GG_1_0(W(6),W(22),R2_AAD,AMPL(1,189))
      CALL R2_GG_1_0(W(6),W(22),R2_AAD,AMPL(1,190))
      CALL R2_GG_1_0(W(6),W(22),R2_AAD,AMPL(1,191))
C     Counter-term amplitude(s) for loop diagram number 819
      CALL R2_GG_1_0(W(6),W(23),R2_AZD,AMPL(1,192))
      CALL R2_GG_1_0(W(6),W(23),R2_AZD,AMPL(1,193))
      CALL R2_GG_1_0(W(6),W(23),R2_AZD,AMPL(1,194))
C     Counter-term amplitude(s) for loop diagram number 820
      CALL R2_GG_1_0(W(22),W(8),R2_AZD,AMPL(1,195))
      CALL R2_GG_1_0(W(22),W(8),R2_AZD,AMPL(1,196))
      CALL R2_GG_1_0(W(22),W(8),R2_AZD,AMPL(1,197))
C     Counter-term amplitude(s) for loop diagram number 821
      CALL R2_GG_1_0(W(8),W(23),R2_ZZD,AMPL(1,198))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZD,AMPL(1,199))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZD,AMPL(1,200))
C     Counter-term amplitude(s) for loop diagram number 822
      CALL R2_GG_1_0(W(6),W(24),R2_AAD,AMPL(1,201))
      CALL R2_GG_1_0(W(6),W(24),R2_AAD,AMPL(1,202))
      CALL R2_GG_1_0(W(6),W(24),R2_AAD,AMPL(1,203))
C     Counter-term amplitude(s) for loop diagram number 823
      CALL R2_GG_1_0(W(6),W(25),R2_AZD,AMPL(1,204))
      CALL R2_GG_1_0(W(6),W(25),R2_AZD,AMPL(1,205))
      CALL R2_GG_1_0(W(6),W(25),R2_AZD,AMPL(1,206))
C     Counter-term amplitude(s) for loop diagram number 824
      CALL R2_GG_1_0(W(24),W(8),R2_AZD,AMPL(1,207))
      CALL R2_GG_1_0(W(24),W(8),R2_AZD,AMPL(1,208))
      CALL R2_GG_1_0(W(24),W(8),R2_AZD,AMPL(1,209))
C     Counter-term amplitude(s) for loop diagram number 825
      CALL R2_GG_1_0(W(8),W(25),R2_ZZD,AMPL(1,210))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZD,AMPL(1,211))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZD,AMPL(1,212))
C     Counter-term amplitude(s) for loop diagram number 826
      CALL R2_GG_1_0(W(11),W(20),R2_AAD,AMPL(1,213))
      CALL R2_GG_1_0(W(11),W(20),R2_AAD,AMPL(1,214))
      CALL R2_GG_1_0(W(11),W(20),R2_AAD,AMPL(1,215))
C     Counter-term amplitude(s) for loop diagram number 827
      CALL R2_GG_1_0(W(11),W(21),R2_AZD,AMPL(1,216))
      CALL R2_GG_1_0(W(11),W(21),R2_AZD,AMPL(1,217))
      CALL R2_GG_1_0(W(11),W(21),R2_AZD,AMPL(1,218))
C     Counter-term amplitude(s) for loop diagram number 828
      CALL R2_GG_1_0(W(20),W(12),R2_AZD,AMPL(1,219))
      CALL R2_GG_1_0(W(20),W(12),R2_AZD,AMPL(1,220))
      CALL R2_GG_1_0(W(20),W(12),R2_AZD,AMPL(1,221))
C     Counter-term amplitude(s) for loop diagram number 829
      CALL R2_GG_1_0(W(12),W(21),R2_ZZD,AMPL(1,222))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZD,AMPL(1,223))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZD,AMPL(1,224))
C     Counter-term amplitude(s) for loop diagram number 830
      CALL R2_GG_1_0(W(11),W(28),R2_AAD,AMPL(1,225))
      CALL R2_GG_1_0(W(11),W(28),R2_AAD,AMPL(1,226))
      CALL R2_GG_1_0(W(11),W(28),R2_AAD,AMPL(1,227))
C     Counter-term amplitude(s) for loop diagram number 831
      CALL R2_GG_1_0(W(11),W(29),R2_AZD,AMPL(1,228))
      CALL R2_GG_1_0(W(11),W(29),R2_AZD,AMPL(1,229))
      CALL R2_GG_1_0(W(11),W(29),R2_AZD,AMPL(1,230))
C     Counter-term amplitude(s) for loop diagram number 832
      CALL R2_GG_1_0(W(28),W(12),R2_AZD,AMPL(1,231))
      CALL R2_GG_1_0(W(28),W(12),R2_AZD,AMPL(1,232))
      CALL R2_GG_1_0(W(28),W(12),R2_AZD,AMPL(1,233))
C     Counter-term amplitude(s) for loop diagram number 833
      CALL R2_GG_1_0(W(12),W(29),R2_ZZD,AMPL(1,234))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZD,AMPL(1,235))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZD,AMPL(1,236))
C     Counter-term amplitude(s) for loop diagram number 840
      CALL R2_GG_1_0(W(6),W(22),R2_AAU,AMPL(1,237))
      CALL R2_GG_1_0(W(6),W(22),R2_AAU,AMPL(1,238))
C     Counter-term amplitude(s) for loop diagram number 841
      CALL R2_GG_1_0(W(6),W(23),R2_AZU,AMPL(1,239))
      CALL R2_GG_1_0(W(6),W(23),R2_AZU,AMPL(1,240))
C     Counter-term amplitude(s) for loop diagram number 842
      CALL R2_GG_1_0(W(22),W(8),R2_AZU,AMPL(1,241))
      CALL R2_GG_1_0(W(22),W(8),R2_AZU,AMPL(1,242))
C     Counter-term amplitude(s) for loop diagram number 843
      CALL R2_GG_1_0(W(8),W(23),R2_ZZU,AMPL(1,243))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZU,AMPL(1,244))
C     Counter-term amplitude(s) for loop diagram number 844
      CALL R2_GG_1_0(W(6),W(24),R2_AAU,AMPL(1,245))
      CALL R2_GG_1_0(W(6),W(24),R2_AAU,AMPL(1,246))
C     Counter-term amplitude(s) for loop diagram number 845
      CALL R2_GG_1_0(W(6),W(25),R2_AZU,AMPL(1,247))
      CALL R2_GG_1_0(W(6),W(25),R2_AZU,AMPL(1,248))
C     Counter-term amplitude(s) for loop diagram number 846
      CALL R2_GG_1_0(W(24),W(8),R2_AZU,AMPL(1,249))
      CALL R2_GG_1_0(W(24),W(8),R2_AZU,AMPL(1,250))
C     Counter-term amplitude(s) for loop diagram number 847
      CALL R2_GG_1_0(W(8),W(25),R2_ZZU,AMPL(1,251))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZU,AMPL(1,252))
C     Counter-term amplitude(s) for loop diagram number 848
      CALL R2_GG_1_0(W(11),W(20),R2_AAU,AMPL(1,253))
      CALL R2_GG_1_0(W(11),W(20),R2_AAU,AMPL(1,254))
C     Counter-term amplitude(s) for loop diagram number 849
      CALL R2_GG_1_0(W(11),W(21),R2_AZU,AMPL(1,255))
      CALL R2_GG_1_0(W(11),W(21),R2_AZU,AMPL(1,256))
C     Counter-term amplitude(s) for loop diagram number 850
      CALL R2_GG_1_0(W(20),W(12),R2_AZU,AMPL(1,257))
      CALL R2_GG_1_0(W(20),W(12),R2_AZU,AMPL(1,258))
C     Counter-term amplitude(s) for loop diagram number 851
      CALL R2_GG_1_0(W(12),W(21),R2_ZZU,AMPL(1,259))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZU,AMPL(1,260))
C     Counter-term amplitude(s) for loop diagram number 852
      CALL R2_GG_1_0(W(11),W(28),R2_AAU,AMPL(1,261))
      CALL R2_GG_1_0(W(11),W(28),R2_AAU,AMPL(1,262))
C     Counter-term amplitude(s) for loop diagram number 853
      CALL R2_GG_1_0(W(11),W(29),R2_AZU,AMPL(1,263))
      CALL R2_GG_1_0(W(11),W(29),R2_AZU,AMPL(1,264))
C     Counter-term amplitude(s) for loop diagram number 854
      CALL R2_GG_1_0(W(28),W(12),R2_AZU,AMPL(1,265))
      CALL R2_GG_1_0(W(28),W(12),R2_AZU,AMPL(1,266))
C     Counter-term amplitude(s) for loop diagram number 855
      CALL R2_GG_1_0(W(12),W(29),R2_ZZU,AMPL(1,267))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZU,AMPL(1,268))
C     Counter-term amplitude(s) for loop diagram number 860
      CALL VVS1_0(W(5),W(6),W(32),R2_AAHT,AMPL(1,269))
C     Counter-term amplitude(s) for loop diagram number 868
      CALL VVS1_0(W(5),W(8),W(32),R2_AZHT,AMPL(1,270))
C     Counter-term amplitude(s) for loop diagram number 870
      CALL R2_GG_1_R2_GG_3_0(W(6),W(22),R2_AAU,R2_AAT3,AMPL(1,271))
C     Counter-term amplitude(s) for loop diagram number 871
      CALL R2_GG_1_R2_GG_3_0(W(6),W(23),R2_AZU,R2_AZT3,AMPL(1,272))
C     Counter-term amplitude(s) for loop diagram number 874
      CALL R2_GG_1_R2_GG_3_0(W(22),W(8),R2_AZU,R2_AZT3,AMPL(1,273))
C     Counter-term amplitude(s) for loop diagram number 875
      CALL R2_GG_1_R2_GG_3_0(W(8),W(23),R2_ZZU,R2_ZZT3,AMPL(1,274))
C     Counter-term amplitude(s) for loop diagram number 878
      CALL R2_GG_1_R2_GG_3_0(W(6),W(24),R2_AAU,R2_AAT3,AMPL(1,275))
C     Counter-term amplitude(s) for loop diagram number 879
      CALL R2_GG_1_R2_GG_3_0(W(6),W(25),R2_AZU,R2_AZT3,AMPL(1,276))
C     Counter-term amplitude(s) for loop diagram number 882
      CALL R2_GG_1_R2_GG_3_0(W(24),W(8),R2_AZU,R2_AZT3,AMPL(1,277))
C     Counter-term amplitude(s) for loop diagram number 883
      CALL R2_GG_1_R2_GG_3_0(W(8),W(25),R2_ZZU,R2_ZZT3,AMPL(1,278))
C     Counter-term amplitude(s) for loop diagram number 886
      CALL R2_GG_1_R2_GG_3_0(W(11),W(20),R2_AAU,R2_AAT3,AMPL(1,279))
C     Counter-term amplitude(s) for loop diagram number 887
      CALL R2_GG_1_R2_GG_3_0(W(11),W(21),R2_AZU,R2_AZT3,AMPL(1,280))
C     Counter-term amplitude(s) for loop diagram number 888
      CALL R2_GG_1_R2_GG_3_0(W(20),W(12),R2_AZU,R2_AZT3,AMPL(1,281))
C     Counter-term amplitude(s) for loop diagram number 889
      CALL R2_GG_1_R2_GG_3_0(W(12),W(21),R2_ZZU,R2_ZZT3,AMPL(1,282))
C     Counter-term amplitude(s) for loop diagram number 894
      CALL R2_GG_1_R2_GG_3_0(W(11),W(28),R2_AAU,R2_AAT3,AMPL(1,283))
C     Counter-term amplitude(s) for loop diagram number 895
      CALL R2_GG_1_R2_GG_3_0(W(11),W(29),R2_AZU,R2_AZT3,AMPL(1,284))
C     Counter-term amplitude(s) for loop diagram number 896
      CALL R2_GG_1_R2_GG_3_0(W(28),W(12),R2_AZU,R2_AZT3,AMPL(1,285))
C     Counter-term amplitude(s) for loop diagram number 897
      CALL R2_GG_1_R2_GG_3_0(W(12),W(29),R2_ZZU,R2_ZZT3,AMPL(1,286))
C     Counter-term amplitude(s) for loop diagram number 910
      CALL R2_GG_1_0(W(6),W(22),R2_AAL,AMPL(1,287))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(6),W(22),C_UVAMASS1EW_1EPS
     $ ,C_UVAMASS2EW_1EPS,AMPL(2,288))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(6),W(22),C_UVAMASS1EW
     $ ,C_UVAMASS2EW,AMPL(1,289))
      CALL R2_GG_1_0(W(6),W(22),R2_AAL,AMPL(1,290))
      CALL R2_GG_1_0(W(6),W(22),R2_AAL,AMPL(1,291))
C     Counter-term amplitude(s) for loop diagram number 911
      CALL R2_GG_1_0(W(6),W(23),R2_AZL,AMPL(1,292))
      CALL R2_GG_1_0(W(6),W(23),R2_AZL,AMPL(1,293))
      CALL R2_GG_1_0(W(6),W(23),R2_AZL,AMPL(1,294))
C     Counter-term amplitude(s) for loop diagram number 912
      CALL R2_GG_1_0(W(22),W(8),R2_AZL,AMPL(1,295))
      CALL R2_GG_1_0(W(22),W(8),R2_AZL,AMPL(1,296))
      CALL R2_GG_1_0(W(22),W(8),R2_AZL,AMPL(1,297))
C     Counter-term amplitude(s) for loop diagram number 913
      CALL R2_GG_1_0(W(8),W(23),R2_ZZL,AMPL(1,298))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZL,AMPL(1,299))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZL,AMPL(1,300))
C     Counter-term amplitude(s) for loop diagram number 914
      CALL R2_GG_1_0(W(6),W(24),R2_AAL,AMPL(1,301))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(6),W(24),C_UVAMASS1EW_1EPS
     $ ,C_UVAMASS2EW_1EPS,AMPL(2,302))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(6),W(24),C_UVAMASS1EW
     $ ,C_UVAMASS2EW,AMPL(1,303))
      CALL R2_GG_1_0(W(6),W(24),R2_AAL,AMPL(1,304))
      CALL R2_GG_1_0(W(6),W(24),R2_AAL,AMPL(1,305))
C     Counter-term amplitude(s) for loop diagram number 915
      CALL R2_GG_1_0(W(6),W(25),R2_AZL,AMPL(1,306))
      CALL R2_GG_1_0(W(6),W(25),R2_AZL,AMPL(1,307))
      CALL R2_GG_1_0(W(6),W(25),R2_AZL,AMPL(1,308))
C     Counter-term amplitude(s) for loop diagram number 916
      CALL R2_GG_1_0(W(24),W(8),R2_AZL,AMPL(1,309))
      CALL R2_GG_1_0(W(24),W(8),R2_AZL,AMPL(1,310))
      CALL R2_GG_1_0(W(24),W(8),R2_AZL,AMPL(1,311))
C     Counter-term amplitude(s) for loop diagram number 917
      CALL R2_GG_1_0(W(8),W(25),R2_ZZL,AMPL(1,312))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZL,AMPL(1,313))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZL,AMPL(1,314))
C     Counter-term amplitude(s) for loop diagram number 918
      CALL R2_GG_1_0(W(11),W(20),R2_AAL,AMPL(1,315))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(11),W(20),C_UVAMASS1EW_1EPS
     $ ,C_UVAMASS2EW_1EPS,AMPL(2,316))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(11),W(20),C_UVAMASS1EW
     $ ,C_UVAMASS2EW,AMPL(1,317))
      CALL R2_GG_1_0(W(11),W(20),R2_AAL,AMPL(1,318))
      CALL R2_GG_1_0(W(11),W(20),R2_AAL,AMPL(1,319))
C     Counter-term amplitude(s) for loop diagram number 919
      CALL R2_GG_1_0(W(11),W(21),R2_AZL,AMPL(1,320))
      CALL R2_GG_1_0(W(11),W(21),R2_AZL,AMPL(1,321))
      CALL R2_GG_1_0(W(11),W(21),R2_AZL,AMPL(1,322))
C     Counter-term amplitude(s) for loop diagram number 920
      CALL R2_GG_1_0(W(20),W(12),R2_AZL,AMPL(1,323))
      CALL R2_GG_1_0(W(20),W(12),R2_AZL,AMPL(1,324))
      CALL R2_GG_1_0(W(20),W(12),R2_AZL,AMPL(1,325))
C     Counter-term amplitude(s) for loop diagram number 921
      CALL R2_GG_1_0(W(12),W(21),R2_ZZL,AMPL(1,326))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZL,AMPL(1,327))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZL,AMPL(1,328))
C     Counter-term amplitude(s) for loop diagram number 922
      CALL R2_GG_1_0(W(11),W(28),R2_AAL,AMPL(1,329))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(11),W(28),C_UVAMASS1EW_1EPS
     $ ,C_UVAMASS2EW_1EPS,AMPL(2,330))
      CALL L_WMWPMASS1_L_WMWPMASS3_0(W(11),W(28),C_UVAMASS1EW
     $ ,C_UVAMASS2EW,AMPL(1,331))
      CALL R2_GG_1_0(W(11),W(28),R2_AAL,AMPL(1,332))
      CALL R2_GG_1_0(W(11),W(28),R2_AAL,AMPL(1,333))
C     Counter-term amplitude(s) for loop diagram number 923
      CALL R2_GG_1_0(W(11),W(29),R2_AZL,AMPL(1,334))
      CALL R2_GG_1_0(W(11),W(29),R2_AZL,AMPL(1,335))
      CALL R2_GG_1_0(W(11),W(29),R2_AZL,AMPL(1,336))
C     Counter-term amplitude(s) for loop diagram number 924
      CALL R2_GG_1_0(W(28),W(12),R2_AZL,AMPL(1,337))
      CALL R2_GG_1_0(W(28),W(12),R2_AZL,AMPL(1,338))
      CALL R2_GG_1_0(W(28),W(12),R2_AZL,AMPL(1,339))
C     Counter-term amplitude(s) for loop diagram number 925
      CALL R2_GG_1_0(W(12),W(29),R2_ZZL,AMPL(1,340))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZL,AMPL(1,341))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZL,AMPL(1,342))
C     Counter-term amplitude(s) for loop diagram number 926
      CALL R2_GG_1_0(W(8),W(23),R2_ZZV,AMPL(1,343))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZV,AMPL(1,344))
      CALL R2_GG_1_0(W(8),W(23),R2_ZZV,AMPL(1,345))
C     Counter-term amplitude(s) for loop diagram number 927
      CALL R2_GG_1_0(W(8),W(25),R2_ZZV,AMPL(1,346))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZV,AMPL(1,347))
      CALL R2_GG_1_0(W(8),W(25),R2_ZZV,AMPL(1,348))
C     Counter-term amplitude(s) for loop diagram number 928
      CALL R2_GG_1_0(W(12),W(21),R2_ZZV,AMPL(1,349))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZV,AMPL(1,350))
      CALL R2_GG_1_0(W(12),W(21),R2_ZZV,AMPL(1,351))
C     Counter-term amplitude(s) for loop diagram number 929
      CALL R2_GG_1_0(W(12),W(29),R2_ZZV,AMPL(1,352))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZV,AMPL(1,353))
      CALL R2_GG_1_0(W(12),W(29),R2_ZZV,AMPL(1,354))
C     At this point, all CT amps needed for (QED=8), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

