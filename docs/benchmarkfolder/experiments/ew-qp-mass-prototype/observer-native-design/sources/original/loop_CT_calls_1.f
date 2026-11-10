      SUBROUTINE ML5_0_LOOP_CT_CALLS_1(P,NHEL,H,IC)
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
      IF (FILTER_SO.AND.CTCALL_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

C     CutTools call for loop numbers 1,2
      CALL ML5_0_LOOP_2(7,14,DCMPLX(ZERO),DCMPLX(MDL_MT),1,I_SO,1)
C     CutTools call for loop numbers 3,4
      CALL ML5_0_LOOP_3(4,6,7,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),2,I_SO,2)
C     CutTools call for loop numbers 5,6
      CALL ML5_0_LOOP_3(3,5,14,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),2,I_SO,3)
C     CutTools call for loop numbers 7,8
      CALL ML5_0_LOOP_3(3,6,9,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),2,I_SO,4)
C     CutTools call for loop numbers 9,11
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,5)
C     CutTools call for loop numbers 10,12
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,6)
C     CutTools call for loop numbers 13,14
      CALL ML5_0_LOOP_2(9,16,DCMPLX(ZERO),DCMPLX(MDL_MT),1,I_SO,7)
C     CutTools call for loop numbers 15,16
      CALL ML5_0_LOOP_3(4,5,16,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),2,I_SO,8)
C     CutTools call for loop numbers 17,18
      CALL ML5_0_LOOP_3(2,11,10,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,9)
C     CutTools call for loop numbers 19
      CALL ML5_0_LOOP_4(2,3,4,10,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,10)
C     CutTools call for loop numbers 20
      CALL ML5_0_LOOP_4(2,3,4,10,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,11)
C     CutTools call for loop numbers 21
      CALL ML5_0_LOOP_4(2,4,3,10,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,12)
C     CutTools call for loop numbers 22
      CALL ML5_0_LOOP_4(2,4,3,10,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,13)
C     CutTools call for loop numbers 23,24
      CALL ML5_0_LOOP_2(10,18,DCMPLX(ZERO),DCMPLX(ZERO),1,I_SO,14)
C     CutTools call for loop numbers 25
      CALL ML5_0_LOOP_4(2,4,3,10,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,15)
C     CutTools call for loop numbers 26,27
      CALL ML5_0_LOOP_3(3,4,20,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,16)
C     CutTools call for loop numbers 28
      CALL ML5_0_LOOP_4(2,3,4,10,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,17)
C     CutTools call for loop numbers 29,30
      CALL ML5_0_LOOP_3(1,11,13,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,18)
C     CutTools call for loop numbers 31
      CALL ML5_0_LOOP_4(1,3,4,13,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,19)
C     CutTools call for loop numbers 32
      CALL ML5_0_LOOP_4(1,3,4,13,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,20)
C     CutTools call for loop numbers 33
      CALL ML5_0_LOOP_4(1,4,3,13,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,21)
C     CutTools call for loop numbers 34
      CALL ML5_0_LOOP_4(1,4,3,13,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,22)
C     CutTools call for loop numbers 35,37
      CALL ML5_0_LOOP_4(1,2,5,11,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),3,I_SO,23)
C     CutTools call for loop numbers 36,38
      CALL ML5_0_LOOP_4(1,2,11,5,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),3,I_SO,24)
C     CutTools call for loop numbers 39,40
      CALL ML5_0_LOOP_3(1,2,22,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,25)
C     CutTools call for loop numbers 41,42
      CALL ML5_0_LOOP_3(1,2,24,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,26)
C     CutTools call for loop numbers 43,44
      CALL ML5_0_LOOP_3(1,5,18,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,27)
C     CutTools call for loop numbers 45
      CALL ML5_0_LOOP_4(1,2,4,7,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,28)
C     CutTools call for loop numbers 46
      CALL ML5_0_LOOP_4(1,2,4,7,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,29)
C     CutTools call for loop numbers 47
      CALL ML5_0_LOOP_4(1,2,7,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,30)
C     CutTools call for loop numbers 48
      CALL ML5_0_LOOP_4(1,2,7,4,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,31)
C     CutTools call for loop numbers 49
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,32)
C     CutTools call for loop numbers 50
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,33)
C     CutTools call for loop numbers 51
      CALL ML5_0_LOOP_5(1,3,4,2,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,34)
C     CutTools call for loop numbers 52
      CALL ML5_0_LOOP_5(1,3,4,2,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,35)
C     CutTools call for loop numbers 53
      CALL ML5_0_LOOP_4(1,2,9,3,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,36)
C     CutTools call for loop numbers 54
      CALL ML5_0_LOOP_4(1,2,9,3,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,37)
C     CutTools call for loop numbers 55
      CALL ML5_0_LOOP_4(1,2,3,9,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,38)
C     CutTools call for loop numbers 56
      CALL ML5_0_LOOP_4(1,2,3,9,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),2,I_SO,39)
C     CutTools call for loop numbers 57
      CALL ML5_0_LOOP_5(1,2,3,5,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,40)
C     CutTools call for loop numbers 58
      CALL ML5_0_LOOP_5(1,2,3,5,4,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,41)
C     CutTools call for loop numbers 59
      CALL ML5_0_LOOP_5(1,4,3,2,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,42)
C     CutTools call for loop numbers 60
      CALL ML5_0_LOOP_5(1,4,3,2,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,43)
C     CutTools call for loop numbers 61,62
      CALL ML5_0_LOOP_2(13,26,DCMPLX(ZERO),DCMPLX(ZERO),1,I_SO,44)
C     CutTools call for loop numbers 63
      CALL ML5_0_LOOP_4(1,4,3,13,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,45)
C     CutTools call for loop numbers 64,65
      CALL ML5_0_LOOP_3(3,4,28,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,46)
C     CutTools call for loop numbers 66
      CALL ML5_0_LOOP_4(1,3,4,13,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,47)
C     CutTools call for loop numbers 67,68
      CALL ML5_0_LOOP_3(2,5,26,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,48)
C     CutTools call for loop numbers 69
      CALL ML5_0_LOOP_4(1,2,7,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,49)
C     CutTools call for loop numbers 70
      CALL ML5_0_LOOP_4(1,2,4,7,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,50)
C     CutTools call for loop numbers 71
      CALL ML5_0_LOOP_5(1,2,3,5,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MZ),3,I_SO,51)
C     CutTools call for loop numbers 72
      CALL ML5_0_LOOP_5(1,4,3,2,5,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,52)
C     CutTools call for loop numbers 73
      CALL ML5_0_LOOP_4(1,2,3,9,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,53)
C     CutTools call for loop numbers 74
      CALL ML5_0_LOOP_4(1,2,9,3,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,54)
C     CutTools call for loop numbers 75
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MZ),3,I_SO,55)
C     CutTools call for loop numbers 76
      CALL ML5_0_LOOP_5(1,3,4,2,5,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,56)
C     CutTools call for loop numbers 77,78,194,195
      CALL ML5_0_LOOP_2(7,14,DCMPLX(MDL_MZ),DCMPLX(MDL_MT),1,I_SO,57)
C     CutTools call for loop numbers 80,79
      CALL ML5_0_LOOP_2(8,30,DCMPLX(MDL_MH),DCMPLX(MDL_MZ),1,I_SO,58)
C     CutTools call for loop numbers 82,83,197,198
      CALL ML5_0_LOOP_3(4,6,7,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),2,I_SO,59)
C     CutTools call for loop numbers 85,86,199,200
      CALL ML5_0_LOOP_3(3,5,14,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),2,I_SO,60)
C     CutTools call for loop numbers 88,89,201,202
      CALL ML5_0_LOOP_3(3,6,9,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),2,I_SO,61)
C     CutTools call for loop numbers 91,93,203,205
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,62)
C     CutTools call for loop numbers 92,94,204,206
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,63)
C     CutTools call for loop numbers 95,96,207,208
      CALL ML5_0_LOOP_2(9,16,DCMPLX(MDL_MZ),DCMPLX(MDL_MT),1,I_SO,64)
C     CutTools call for loop numbers 98,97
      CALL ML5_0_LOOP_2(8,31,DCMPLX(MDL_MH),DCMPLX(MDL_MZ),1,I_SO,65)
C     CutTools call for loop numbers 100,101,210,211
      CALL ML5_0_LOOP_3(4,5,16,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),2,I_SO,66)
C     CutTools call for loop numbers 103,104
      CALL ML5_0_LOOP_3(2,11,10,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MZ),2,I_SO,67)
C     CutTools call for loop numbers 105
      CALL ML5_0_LOOP_3(2,32,10,DCMPLX(MDL_MZ),DCMPLX(MDL_MZ)
     $ ,DCMPLX(ZERO),1,I_SO,68)
C     CutTools call for loop numbers 106
      CALL ML5_0_LOOP_4(2,3,4,10,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,69)
C     CutTools call for loop numbers 107
      CALL ML5_0_LOOP_4(2,4,3,10,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,70)
C     CutTools call for loop numbers 108,110
      CALL ML5_0_LOOP_2(10,18,DCMPLX(MDL_MZ),DCMPLX(ZERO),1,I_SO,71)
C     CutTools call for loop numbers 111,109
      CALL ML5_0_LOOP_2(33,21,DCMPLX(MDL_MH),DCMPLX(MDL_MZ),1,I_SO,72)
C     CutTools call for loop numbers 113,114,213,214
      CALL ML5_0_LOOP_3(3,4,20,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,73)
C     CutTools call for loop numbers 116,117
      CALL ML5_0_LOOP_3(1,11,13,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MZ),2,I_SO,74)
C     CutTools call for loop numbers 118
      CALL ML5_0_LOOP_3(1,32,13,DCMPLX(MDL_MZ),DCMPLX(MDL_MZ)
     $ ,DCMPLX(ZERO),1,I_SO,75)
C     CutTools call for loop numbers 119
      CALL ML5_0_LOOP_4(1,3,4,13,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,76)
C     CutTools call for loop numbers 120
      CALL ML5_0_LOOP_4(1,4,3,13,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),2,I_SO,77)
C     CutTools call for loop numbers 121,123
      CALL ML5_0_LOOP_4(1,2,5,11,DCMPLX(MDL_MZ),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,78)
C     CutTools call for loop numbers 122,124
      CALL ML5_0_LOOP_4(1,2,11,5,DCMPLX(MDL_MZ),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,79)
C     CutTools call for loop numbers 125,126
      CALL ML5_0_LOOP_3(1,2,22,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,80)
C     CutTools call for loop numbers 127,128
      CALL ML5_0_LOOP_3(1,2,24,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,81)
C     CutTools call for loop numbers 129,130
      CALL ML5_0_LOOP_3(1,5,18,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,2,I_SO,82)
C     CutTools call for loop numbers 131
      CALL ML5_0_LOOP_4(1,5,2,32,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MZ),DCMPLX(MDL_MZ),2,I_SO,83)
C     CutTools call for loop numbers 132
      CALL ML5_0_LOOP_3(1,2,34,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MZ),1,I_SO,84)
C     CutTools call for loop numbers 133
      CALL ML5_0_LOOP_4(1,2,4,7,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,85)
C     CutTools call for loop numbers 134
      CALL ML5_0_LOOP_4(1,2,7,4,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,86)
C     CutTools call for loop numbers 135
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MZ),3,I_SO,87)
C     CutTools call for loop numbers 136
      CALL ML5_0_LOOP_5(1,3,4,2,5,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,88)
C     CutTools call for loop numbers 137
      CALL ML5_0_LOOP_4(1,2,9,3,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,89)
C     CutTools call for loop numbers 138
      CALL ML5_0_LOOP_3(1,2,35,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MZ),1,I_SO,90)
C     CutTools call for loop numbers 139
      CALL ML5_0_LOOP_4(1,2,3,9,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MZ),2,I_SO,91)
C     CutTools call for loop numbers 140
      CALL ML5_0_LOOP_5(1,2,3,5,4,DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),DCMPLX(MDL_MZ),3,I_SO,92)
C     CutTools call for loop numbers 141
      CALL ML5_0_LOOP_5(1,4,3,2,5,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,93)
C     CutTools call for loop numbers 142,144
      CALL ML5_0_LOOP_2(13,26,DCMPLX(MDL_MZ),DCMPLX(ZERO),1,I_SO,94)
C     CutTools call for loop numbers 145,143
      CALL ML5_0_LOOP_2(33,29,DCMPLX(MDL_MH),DCMPLX(MDL_MZ),1,I_SO,95)
C     CutTools call for loop numbers 147,148,216,217
      CALL ML5_0_LOOP_3(3,4,28,DCMPLX(MDL_MZ),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,96)
C     CutTools call for loop numbers 150,151
      CALL ML5_0_LOOP_3(2,5,26,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MZ)
     $ ,2,I_SO,97)
C     CutTools call for loop numbers 152,153
      CALL ML5_0_LOOP_2(7,14,DCMPLX(MDL_MH),DCMPLX(MDL_MT),1,I_SO,98)
C     CutTools call for loop numbers 154,155
      CALL ML5_0_LOOP_2(8,23,DCMPLX(MDL_MZ),DCMPLX(MDL_MH),2,I_SO,99)
C     CutTools call for loop numbers 156,84
      CALL ML5_0_LOOP_3(4,8,7,DCMPLX(MDL_MZ),DCMPLX(MDL_MH)
     $ ,DCMPLX(MDL_MT),2,I_SO,100)
C     CutTools call for loop numbers 157
      CALL ML5_0_LOOP_1_2(2,8,23,DCMPLX(MDL_MH),0,I_SO,101)
C     CutTools call for loop numbers 158,159
      CALL ML5_0_LOOP_3(4,6,7,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MH),2,I_SO,102)
C     CutTools call for loop numbers 160,81
      CALL ML5_0_LOOP_3(4,8,7,DCMPLX(MDL_MH),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),2,I_SO,103)
C     CutTools call for loop numbers 161,162
      CALL ML5_0_LOOP_3(3,5,14,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MH),2,I_SO,104)
C     CutTools call for loop numbers 163,102
      CALL ML5_0_LOOP_4(3,5,4,8,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MZ),DCMPLX(MDL_MH),3,I_SO,105)
C     CutTools call for loop numbers 164,165
      CALL ML5_0_LOOP_3(3,6,9,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MH),2,I_SO,106)
C     CutTools call for loop numbers 166,99
      CALL ML5_0_LOOP_3(3,8,9,DCMPLX(MDL_MH),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MT),2,I_SO,107)
C     CutTools call for loop numbers 167,169
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(MDL_MH),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,108)
C     CutTools call for loop numbers 168,170
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(MDL_MH),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,109)
C     CutTools call for loop numbers 171,172
      CALL ML5_0_LOOP_2(9,16,DCMPLX(MDL_MH),DCMPLX(MDL_MT),1,I_SO,110)
C     CutTools call for loop numbers 173,174
      CALL ML5_0_LOOP_2(8,25,DCMPLX(MDL_MZ),DCMPLX(MDL_MH),2,I_SO,111)
C     CutTools call for loop numbers 175,90
      CALL ML5_0_LOOP_3(3,8,9,DCMPLX(MDL_MZ),DCMPLX(MDL_MH)
     $ ,DCMPLX(MDL_MT),2,I_SO,112)
C     CutTools call for loop numbers 176
      CALL ML5_0_LOOP_1_2(2,8,25,DCMPLX(MDL_MH),0,I_SO,113)
C     CutTools call for loop numbers 177,178
      CALL ML5_0_LOOP_3(4,5,16,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MH),2,I_SO,114)
C     CutTools call for loop numbers 179,87
      CALL ML5_0_LOOP_4(3,5,4,8,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MH),DCMPLX(MDL_MZ),3,I_SO,115)
C     CutTools call for loop numbers 180,182
      CALL ML5_0_LOOP_2(12,21,DCMPLX(MDL_MZ),DCMPLX(MDL_MH),2,I_SO,116)
C     CutTools call for loop numbers 181
      CALL ML5_0_LOOP_1_2(2,12,21,DCMPLX(MDL_MH),0,I_SO,117)
C     CutTools call for loop numbers 183,115
      CALL ML5_0_LOOP_3(3,4,21,DCMPLX(MDL_MT),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MH),2,I_SO,118)
C     CutTools call for loop numbers 184,185
      CALL ML5_0_LOOP_3(3,4,20,DCMPLX(MDL_MH),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,119)
C     CutTools call for loop numbers 186,112
      CALL ML5_0_LOOP_3(3,4,21,DCMPLX(MDL_MT),DCMPLX(MDL_MH)
     $ ,DCMPLX(MDL_MZ),2,I_SO,120)
C     CutTools call for loop numbers 187,189
      CALL ML5_0_LOOP_2(12,29,DCMPLX(MDL_MZ),DCMPLX(MDL_MH),2,I_SO,121)
C     CutTools call for loop numbers 188
      CALL ML5_0_LOOP_1_2(2,12,29,DCMPLX(MDL_MH),0,I_SO,122)
C     CutTools call for loop numbers 190,149
      CALL ML5_0_LOOP_3(3,4,29,DCMPLX(MDL_MT),DCMPLX(MDL_MZ)
     $ ,DCMPLX(MDL_MH),2,I_SO,123)
C     CutTools call for loop numbers 191,192
      CALL ML5_0_LOOP_3(3,4,28,DCMPLX(MDL_MH),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),2,I_SO,124)
C     CutTools call for loop numbers 193,146
      CALL ML5_0_LOOP_3(3,4,29,DCMPLX(MDL_MT),DCMPLX(MDL_MH)
     $ ,DCMPLX(MDL_MZ),2,I_SO,125)
C     CutTools call for loop numbers 196
      CALL ML5_0_LOOP_1_2(2,8,23,DCMPLX(MDL_MZ),0,I_SO,126)
C     CutTools call for loop numbers 209
      CALL ML5_0_LOOP_1_2(2,8,25,DCMPLX(MDL_MZ),0,I_SO,127)
C     CutTools call for loop numbers 212
      CALL ML5_0_LOOP_1_2(2,12,21,DCMPLX(MDL_MZ),0,I_SO,128)
C     CutTools call for loop numbers 215
      CALL ML5_0_LOOP_1_2(2,12,29,DCMPLX(MDL_MZ),0,I_SO,129)
C     CutTools call for loop numbers
C      218,223,228,233,255,260,265,270,600,602,604,606,611,613,615,617,
C     708,709,710,711,716,717,718,719,756,757,758,759,764,765,766,767,2
C     38,239,241,242,244,245,246,249,250,251,275,276,278,279,281,282,28
C     3,286,287,288,302,303,305,306,307,309,321,322,324,325,326,328,335
C     ,337,344,346,608,609,619,620,712,713,714,715,720,721,722,723,760,
C     761,762,763,768,769,770,771,219,220,221,224,225,226,229,230,231,2
C     34,235,236,256,257,258,261,262,263,266,267,268,271,272,273,293,29
C     4,295,296,298,299,300,301,312,313,314,315,317,318,319,320,330,331
C     ,332,333,339,340,341,342,247,252,284,289,310,329,338,347
      CALL ML5_0_LOOP_3(5,6,11,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),3,I_SO,130)
C     CutTools call for loop numbers
C      222,232,259,269,601,605,612,616,240,243,248,253,277,280,285,290
      CALL ML5_0_LOOP_2_3(1,2,6,5,11,DCMPLX(MDL_MW),DCMPLX(MDL_MW),1
     $ ,I_SO,131)
C     CutTools call for loop numbers 264,274,614,618,291,621
      CALL ML5_0_LOOP_2_3(1,2,11,5,8,DCMPLX(MDL_MW),DCMPLX(MDL_MW),1
     $ ,I_SO,132)
C     CutTools call for loop numbers
C      292,297,316,622,623,625,304,308,323,327,334,336,343,345
      CALL ML5_0_LOOP_2_3(1,2,5,6,11,DCMPLX(MDL_MW),DCMPLX(MDL_MW),1
     $ ,I_SO,133)
C     CutTools call for loop numbers 311,624
      CALL ML5_0_LOOP_2_3(1,2,5,11,8,DCMPLX(MDL_MW),DCMPLX(MDL_MW),1
     $ ,I_SO,134)
C     CutTools call for loop numbers 348,364,626,633
      CALL ML5_0_LOOP_2(7,14,DCMPLX(MDL_MW),DCMPLX(ZERO),1,I_SO,135)
C     CutTools call for loop numbers
C      349,350,365,366,627,628,634,635,724,725,728,729,772,773,776,777,
C     351,354,355,360,361,367,370,371,376,377,629,636,726,727,730,731,7
C     74,775,778,779,352,353,358,359,368,369,374,375
      CALL ML5_0_LOOP_2(6,22,DCMPLX(MDL_MW),DCMPLX(MDL_MW),2,I_SO,136)
C     CutTools call for loop numbers 356,372,630,637,357,373,381,383
      CALL ML5_0_LOOP_3(4,6,7,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,137)
C     CutTools call for loop numbers 362,363,379,631,632,639
      CALL ML5_0_LOOP_1_2(2,6,22,DCMPLX(MDL_MW),0,I_SO,138)
C     CutTools call for loop numbers 378,638
      CALL ML5_0_LOOP_1_2(2,22,8,DCMPLX(MDL_MW),0,I_SO,139)
C     CutTools call for loop numbers 380,382,640,641
      CALL ML5_0_LOOP_3(4,6,7,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,2,I_SO,140)
C     CutTools call for loop numbers 384,392,642,646
      CALL ML5_0_LOOP_3(3,5,14,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,2,I_SO,141)
C     CutTools call for loop numbers
C      385,393,643,647,386,388,394,396,465,468,470,473,387,395,475,477
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,142)
C     CutTools call for loop numbers 389,397,644,648,390,398,459,463
      CALL ML5_0_LOOP_4(3,5,4,6,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,143)
C     CutTools call for loop numbers 399,649
      CALL ML5_0_LOOP_3_4(1,1,2,3,4,5,8,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),1,I_SO,144)
C     CutTools call for loop numbers 400,403,650,652
      CALL ML5_0_LOOP_3(3,6,9,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,2,I_SO,145)
C     CutTools call for loop numbers 401,404,651,653,402,405,438,453
      CALL ML5_0_LOOP_3(3,6,9,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,146)
C     CutTools call for loop numbers 406,408,654,656
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(MDL_MW),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,147)
C     CutTools call for loop numbers 407,409,655,657
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(MDL_MW),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,148)
C     CutTools call for loop numbers 410,418,658,661,414,422,467,472
      CALL ML5_0_LOOP_4(3,5,4,6,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,149)
C     CutTools call for loop numbers 411,419,659,662,415,423,478,480
      CALL ML5_0_LOOP_3(3,5,14,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,150)
C     CutTools call for loop numbers
C      412,420,660,663,413,416,421,424,457,458,461,462,417,425,479,481
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,151)
C     CutTools call for loop numbers 426,441,664,670
      CALL ML5_0_LOOP_2(9,16,DCMPLX(MDL_MW),DCMPLX(ZERO),1,I_SO,152)
C     CutTools call for loop numbers
C      427,428,442,443,665,666,671,672,732,733,736,737,780,781,784,785,
C     429,432,433,436,437,444,447,448,451,452,667,673,734,735,738,739,7
C     82,783,786,787,430,431,434,435,445,446,449,450
      CALL ML5_0_LOOP_2(6,24,DCMPLX(MDL_MW),DCMPLX(MDL_MW),2,I_SO,153)
C     CutTools call for loop numbers 439,440,455,668,669,675
      CALL ML5_0_LOOP_1_2(2,6,24,DCMPLX(MDL_MW),0,I_SO,154)
C     CutTools call for loop numbers 454,674
      CALL ML5_0_LOOP_1_2(2,24,8,DCMPLX(MDL_MW),0,I_SO,155)
C     CutTools call for loop numbers 456,460,676,677
      CALL ML5_0_LOOP_3(4,5,16,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,2,I_SO,156)
C     CutTools call for loop numbers 464,469,678,679,466,471,474,476
      CALL ML5_0_LOOP_3(4,5,16,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,157)
C     CutTools call for loop numbers 482,483,485
      CALL ML5_0_LOOP_3(2,11,10,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,158)
C     CutTools call for loop numbers 484
      CALL ML5_0_LOOP_3(2,12,10,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MW),2,I_SO,159)
C     CutTools call for loop numbers 486
      CALL ML5_0_LOOP_4(2,4,3,10,DCMPLX(MDL_MW),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MW),DCMPLX(ZERO),2,I_SO,160)
C     CutTools call for loop numbers
C      487,489,496,498,680,681,684,685,740,741,742,743,788,789,790,791,
C     227,237,505,506,507,508,509,510,511,512,513,514,603,607,688,689,7
C     44,745,746,747,792,793,794,795,254,488,490,491,492,497,499,500,50
C     1,610
      CALL ML5_0_LOOP_2(11,20,DCMPLX(MDL_MW),DCMPLX(MDL_MW),2,I_SO,161)
C     CutTools call for loop numbers 493,494,503,682,683,687
      CALL ML5_0_LOOP_1_2(2,11,20,DCMPLX(MDL_MW),0,I_SO,162)
C     CutTools call for loop numbers 495,504
      CALL ML5_0_LOOP_2(10,18,DCMPLX(MDL_MW),DCMPLX(ZERO),1,I_SO,163)
C     CutTools call for loop numbers 502,686
      CALL ML5_0_LOOP_1_2(2,20,12,DCMPLX(MDL_MW),0,I_SO,164)
C     CutTools call for loop numbers
C      515,517,690,691,391,516,518,521,522,645
      CALL ML5_0_LOOP_3(3,4,20,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),2,I_SO,165)
C     CutTools call for loop numbers 519,520,692,693
      CALL ML5_0_LOOP_3(3,4,20,DCMPLX(MDL_MW),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,166)
C     CutTools call for loop numbers 523,524,526
      CALL ML5_0_LOOP_3(1,11,13,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,167)
C     CutTools call for loop numbers 525
      CALL ML5_0_LOOP_3(1,12,13,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MW),2,I_SO,168)
C     CutTools call for loop numbers 527
      CALL ML5_0_LOOP_4(1,3,4,13,DCMPLX(MDL_MW),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MW),DCMPLX(ZERO),2,I_SO,169)
C     CutTools call for loop numbers 528
      CALL ML5_0_LOOP_3(1,2,23,DCMPLX(MDL_MW),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,170)
C     CutTools call for loop numbers 529
      CALL ML5_0_LOOP_3(1,2,25,DCMPLX(MDL_MW),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,171)
C     CutTools call for loop numbers 530,533,536,537,538,531,534
      CALL ML5_0_LOOP_4(1,2,5,11,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,172)
C     CutTools call for loop numbers 532,535
      CALL ML5_0_LOOP_3_4(1,1,2,1,2,5,11,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),1,I_SO,173)
C     CutTools call for loop numbers 539,542,546,547,548,541,545
      CALL ML5_0_LOOP_4(1,2,11,5,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,174)
C     CutTools call for loop numbers 540,544
      CALL ML5_0_LOOP_3(1,5,18,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,175)
C     CutTools call for loop numbers 543
      CALL ML5_0_LOOP_4(1,5,2,12,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),3,I_SO,176)
C     CutTools call for loop numbers 549,550,551
      CALL ML5_0_LOOP_3(1,2,22,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),2,I_SO,177)
C     CutTools call for loop numbers 552
      CALL ML5_0_LOOP_4(1,2,4,7,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MW),2,I_SO,178)
C     CutTools call for loop numbers 553,554
      CALL ML5_0_LOOP_5(1,2,5,4,3,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),DCMPLX(ZERO),DCMPLX(MDL_MW),3,I_SO,179)
C     CutTools call for loop numbers 555
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MW),3,I_SO,180)
C     CutTools call for loop numbers 556
      CALL ML5_0_LOOP_4(1,2,9,3,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MW),2,I_SO,181)
C     CutTools call for loop numbers 557,558,559
      CALL ML5_0_LOOP_3(1,2,24,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),2,I_SO,182)
C     CutTools call for loop numbers 560,561
      CALL ML5_0_LOOP_5(1,2,4,3,5,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MW),DCMPLX(MDL_MW),3,I_SO,183)
C     CutTools call for loop numbers
C      562,564,571,573,694,695,698,699,748,749,750,751,796,797,798,799,
C     580,581,582,583,584,585,586,587,588,589,702,703,752,753,754,755,8
C     00,801,802,803,563,565,566,567,572,574,575,576
      CALL ML5_0_LOOP_2(11,28,DCMPLX(MDL_MW),DCMPLX(MDL_MW),2,I_SO,184)
C     CutTools call for loop numbers 568,569,578,696,697,701
      CALL ML5_0_LOOP_1_2(2,11,28,DCMPLX(MDL_MW),0,I_SO,185)
C     CutTools call for loop numbers 570,579
      CALL ML5_0_LOOP_2(13,26,DCMPLX(MDL_MW),DCMPLX(ZERO),1,I_SO,186)
C     CutTools call for loop numbers 577,700
      CALL ML5_0_LOOP_1_2(2,28,12,DCMPLX(MDL_MW),0,I_SO,187)
C     CutTools call for loop numbers 590,592,704,705,591,593,596,597
      CALL ML5_0_LOOP_3(3,4,28,DCMPLX(ZERO),DCMPLX(MDL_MW)
     $ ,DCMPLX(MDL_MW),2,I_SO,188)
C     CutTools call for loop numbers 594,595,706,707
      CALL ML5_0_LOOP_3(3,4,28,DCMPLX(MDL_MW),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,2,I_SO,189)
C     CutTools call for loop numbers 598,599
      CALL ML5_0_LOOP_3(2,5,26,DCMPLX(MDL_MW),DCMPLX(MDL_MW)
     $ ,DCMPLX(ZERO),2,I_SO,190)
C     CutTools call for loop numbers
C      804,805,806,807,808,809,826,827,828,829,830,831,894,895,896,897,
C     898,899,900,901
      CALL ML5_0_LOOP_3(5,6,12,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,3,I_SO,191)
C     CutTools call for loop numbers
C      810,811,812,813,832,833,834,835,902,903,904,905,918
      CALL ML5_0_LOOP_2(6,22,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,192)
C     CutTools call for loop numbers
C      814,815,816,817,836,837,838,839,906,907,908,909,919
      CALL ML5_0_LOOP_2(6,24,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,193)
C     CutTools call for loop numbers
C      818,819,820,821,840,841,842,843,910,911,912,913,920
      CALL ML5_0_LOOP_2(11,20,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,194)
C     CutTools call for loop numbers
C      822,823,824,825,844,845,846,847,914,915,916,917,921
      CALL ML5_0_LOOP_2(11,28,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,195)
C     CutTools call for loop numbers
C      848,849,850,851,852,853,854,855,856,857,858,859,860,861
      CALL ML5_0_LOOP_3(5,6,12,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,196)
C     CutTools call for loop numbers 862,863,864,865,866,867,868,869
      CALL ML5_0_LOOP_2(6,22,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO,197)
C     CutTools call for loop numbers 870,871,872,873,874,875,876,877
      CALL ML5_0_LOOP_2(6,24,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO,198)
C     CutTools call for loop numbers 878,879,880,881,882,883,884,885
      CALL ML5_0_LOOP_2(11,20,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO,199)
C     CutTools call for loop numbers 886,887,888,889,890,891,892,893
      CALL ML5_0_LOOP_2(11,28,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO,200)
C     At this point, all reductions needed for (QED=8), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 5000

      GOTO 1001
 5000 CONTINUE
      CTCALL_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

