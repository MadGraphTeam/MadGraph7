      SUBROUTINE OFF_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L
     $ ,S_MAT)
C     
C     Helper function that compute the loop kinematic matrix with
C     proper thresholds
C     NLOOPLINE : Number of loop lines
C     P_LOOP    : List of external momenta running in the loop, i.e.
C     q_i in the denominator (l_i+q_i)**2-m_i**2
C     M2L       : List of complex-valued masses running in the loop.
C     S_MAT(N,N): Kinematic matrix output.
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE
      REAL*16 P_LOOP(NLOOPLINE,0:3)
      COMPLEX*32 M2L(NLOOPLINE)
      COMPLEX*32 S_MAT(NLOOPLINE,NLOOPLINE)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      COMPLEX*32 DIFFSQ
      REAL*16 REF_NORMALIZATION

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NLOOPLINE
        DO J=1,NLOOPLINE

          IF(I.EQ.J)THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
            DIFFSQ = (CMPLX(P_LOOP(I,0),0.0E0_16,KIND=16)
     $       -CMPLX(P_LOOP(J,0),0.0E0_16,KIND=16))**2
            DO K=1,3
              DIFFSQ = DIFFSQ - (CMPLX(P_LOOP(I,K),0.0E0_16,KIND=16)
     $         -CMPLX(P_LOOP(J,K),0.0E0_16,KIND=16))**2
            ENDDO
C           Default value of the kinematic matrix
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
C           And we now test various thresholds. Normally, at most one
C           applies.
            IF(ABS(M2L(I)).NE.0.0E0_16)THEN
              IF(ABS((DIFFSQ-M2L(I))/M2L(I)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(J)
              ENDIF
            ENDIF
            IF(ABS(M2L(J)).NE.0.0E0_16)THEN
              IF(ABS((DIFFSQ-M2L(J))/M2L(J)).LT.OSTHRES)THEN
                S_MAT(I,J)=-M2L(I)
              ENDIF
            ENDIF
C           Choose what seems the most appropriate way to compare
C           massless onshellness.
            REF_NORMALIZATION=0.0E0_16
C           Here, we chose to base the threshold only on the energy
C           component
            DO K=0,0
              REF_NORMALIZATION = REF_NORMALIZATION + ABS(P_LOOP(I,K))
     $          + ABS(P_LOOP(J,K))
            ENDDO
            REF_NORMALIZATION = (REF_NORMALIZATION/2.0E0_16)**2
            IF(REF_NORMALIZATION.NE.0.0E0_16)THEN
              IF(ABS(DIFFSQ/REF_NORMALIZATION).LT.OSTHRES)THEN
                S_MAT(I,J)=-(M2L(I)+M2L(J))
              ENDIF
            ENDIF
          ENDIF

        ENDDO
      ENDDO

      END




      SUBROUTINE OFF_KINEMATIC_IDENTITY(G,CT,N,I,J,
     $ CLASS,LEG,SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
      IMPLICIT NONE
      INTEGER G,CT,N,I,J,CLASS,LEG,SGN,CLOSURE
      LOGICAL DECLARED_ONSHELL,KNOWN
      INTEGER A,B,K,DIR,NP,ORI,L
      INTEGER GOFF(116),GNUM(116),GVALID(116)
      INTEGER CLASSES(954),LEGS(954),SIGNS(954)
      INTEGER CLOSURES(954),SHELLS(954)
C     Identity only: declared on-shell is not a numerical certificate.
      DATA (GOFF(L),L=1,8) /
     $ 0,2,4,10,12,14,20,26/
      DATA (GOFF(L),L=9,16) /
     $ 32,44,56,62,74,80,82,88/
      DATA (GOFF(L),L=17,24) /
     $ 90,92,94,96,98,104,106,108/
      DATA (GOFF(L),L=25,32) /
     $ 114,120,126,138,150,162,164,166/
      DATA (GOFF(L),L=33,40) /
     $ 172,174,176,182,188,194,206,218/
      DATA (GOFF(L),L=41,48) /
     $ 230,236,248,254,266,272,284,290/
      DATA (GOFF(L),L=49,56) /
     $ 296,308,314,326,338,344,356,362/
      DATA (GOFF(L),L=57,64) /
     $ 374,386,406,426,446,466,486,506/
      DATA (GOFF(L),L=65,72) /
     $ 518,530,542,562,582,602,622,642/
      DATA (GOFF(L),L=73,80) /
     $ 662,674,676,682,684,686,688,694/
      DATA (GOFF(L),L=81,88) /
     $ 700,712,724,730,742,748,754,760/
      DATA (GOFF(L),L=89,96) /
     $ 766,772,778,790,802,808,820,822/
      DATA (GOFF(L),L=97,104) /
     $ 828,834,840,842,848,854,860,866/
      DATA (GOFF(L),L=105,112) /
     $ 872,884,896,902,914,916,922,928/
      DATA (GOFF(L),L=113,116) /
     $ 934,936,942,948/
      DATA (GNUM(L),L=1,8) /
     $ 2,2,3,2,2,3,3,3/
      DATA (GNUM(L),L=9,16) /
     $ 4,4,3,4,3,2,3,2/
      DATA (GNUM(L),L=17,24) /
     $ 2,2,2,2,3,2,2,3/
      DATA (GNUM(L),L=25,32) /
     $ 3,3,4,4,4,2,2,3/
      DATA (GNUM(L),L=33,40) /
     $ 2,2,3,3,3,4,4,4/
      DATA (GNUM(L),L=41,48) /
     $ 3,4,3,4,3,4,3,3/
      DATA (GNUM(L),L=49,56) /
     $ 4,3,4,4,3,4,3,4/
      DATA (GNUM(L),L=57,64) /
     $ 4,5,5,5,5,5,5,4/
      DATA (GNUM(L),L=65,72) /
     $ 4,4,5,5,5,5,5,5/
      DATA (GNUM(L),L=73,80) /
     $ 4,2,3,2,2,2,3,3/
      DATA (GNUM(L),L=81,88) /
     $ 4,4,3,4,3,3,3,3/
      DATA (GNUM(L),L=89,96) /
     $ 3,3,4,4,3,4,2,3/
      DATA (GNUM(L),L=97,104) /
     $ 3,3,2,3,3,3,3,3/
      DATA (GNUM(L),L=105,112) /
     $ 4,4,3,4,2,3,3,3/
      DATA (GNUM(L),L=113,116) /
     $ 2,3,3,3/
      DATA (GVALID(L),L=1,8) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=9,16) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=17,24) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=25,32) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=33,40) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=41,48) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=49,56) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=57,64) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=65,72) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=73,80) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=81,88) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=89,96) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=97,104) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=105,112) /
     $ 1,1,1,1,1,1,1,1/
      DATA (GVALID(L),L=113,116) /
     $ 1,1,1,1/
      DATA (CLASSES(L),L=1,8) /
     $ 3,3,3,3,2,3,3,3/
      DATA (CLASSES(L),L=9,16) /
     $ 1,3,3,3,3,3,2,3/
      DATA (CLASSES(L),L=17,24) /
     $ 3,3,1,3,2,3,1,3/
      DATA (CLASSES(L),L=25,32) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=33,40) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=41,48) /
     $ 1,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=49,56) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=57,64) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=65,72) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=73,80) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=81,88) /
     $ 3,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=89,96) /
     $ 2,2,2,2,3,3,3,3/
      DATA (CLASSES(L),L=97,104) /
     $ 3,3,2,3,3,3,1,3/
      DATA (CLASSES(L),L=105,112) /
     $ 3,3,3,3,2,3,3,3/
      DATA (CLASSES(L),L=113,120) /
     $ 1,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=121,128) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=129,136) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=137,144) /
     $ 3,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=145,152) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=153,160) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=161,168) /
     $ 3,1,3,3,3,3,2,3/
      DATA (CLASSES(L),L=169,176) /
     $ 3,3,1,3,3,3,3,3/
      DATA (CLASSES(L),L=177,184) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=185,192) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=193,200) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=201,208) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=209,216) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=217,224) /
     $ 3,1,2,3,1,1,3,3/
      DATA (CLASSES(L),L=225,232) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=233,240) /
     $ 1,3,1,1,2,3,3,1/
      DATA (CLASSES(L),L=241,248) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=249,256) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=257,264) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=265,272) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=273,280) /
     $ 2,3,1,1,3,3,2,3/
      DATA (CLASSES(L),L=281,288) /
     $ 1,3,3,1,2,3,3,3/
      DATA (CLASSES(L),L=289,296) /
     $ 1,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=297,304) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=305,312) /
     $ 1,1,3,1,2,3,1,3/
      DATA (CLASSES(L),L=313,320) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=321,328) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=329,336) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=337,344) /
     $ 3,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=345,352) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=353,360) /
     $ 1,1,3,1,2,3,1,3/
      DATA (CLASSES(L),L=361,368) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=369,376) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=377,384) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=385,392) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=393,400) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=401,408) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=409,416) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=417,424) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=425,432) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=433,440) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=441,448) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=449,456) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=457,464) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=465,472) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=473,480) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=481,488) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=489,496) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=497,504) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=505,512) /
     $ 3,1,2,3,1,1,3,3/
      DATA (CLASSES(L),L=513,520) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=521,528) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=529,536) /
     $ 3,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=537,544) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=545,552) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=553,560) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=561,568) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=569,576) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=577,584) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=585,592) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=593,600) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=601,608) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=609,616) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=617,624) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=625,632) /
     $ 3,1,1,3,3,1,3,1/
      DATA (CLASSES(L),L=633,640) /
     $ 2,3,3,1,1,3,3,1/
      DATA (CLASSES(L),L=641,648) /
     $ 3,1,2,3,3,1,1,3/
      DATA (CLASSES(L),L=649,656) /
     $ 3,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=657,664) /
     $ 1,3,3,1,3,1,2,3/
      DATA (CLASSES(L),L=665,672) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=673,680) /
     $ 3,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=681,688) /
     $ 1,1,2,2,2,2,3,3/
      DATA (CLASSES(L),L=689,696) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=697,704) /
     $ 1,3,1,1,2,3,3,1/
      DATA (CLASSES(L),L=705,712) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=713,720) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=721,728) /
     $ 1,1,3,1,2,3,3,3/
      DATA (CLASSES(L),L=729,736) /
     $ 1,3,2,3,1,1,3,3/
      DATA (CLASSES(L),L=737,744) /
     $ 2,3,1,3,3,1,2,3/
      DATA (CLASSES(L),L=745,752) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=753,760) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=761,768) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=769,776) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=777,784) /
     $ 1,1,2,3,3,1,3,1/
      DATA (CLASSES(L),L=785,792) /
     $ 3,3,1,1,3,1,2,3/
      DATA (CLASSES(L),L=793,800) /
     $ 3,1,3,1,3,3,1,1/
      DATA (CLASSES(L),L=801,808) /
     $ 3,1,2,3,3,3,1,3/
      DATA (CLASSES(L),L=809,816) /
     $ 2,3,1,1,3,3,2,3/
      DATA (CLASSES(L),L=817,824) /
     $ 1,3,3,1,3,3,2,3/
      DATA (CLASSES(L),L=825,832) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=833,840) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=841,848) /
     $ 3,3,2,3,1,3,1,1/
      DATA (CLASSES(L),L=849,856) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=857,864) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=865,872) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=873,880) /
     $ 2,3,3,1,3,1,3,3/
      DATA (CLASSES(L),L=881,888) /
     $ 1,1,3,1,2,3,3,1/
      DATA (CLASSES(L),L=889,896) /
     $ 3,1,3,3,1,1,3,1/
      DATA (CLASSES(L),L=897,904) /
     $ 2,3,3,3,1,3,2,3/
      DATA (CLASSES(L),L=905,912) /
     $ 1,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=913,920) /
     $ 3,1,3,3,2,3,1,3/
      DATA (CLASSES(L),L=921,928) /
     $ 1,1,2,3,1,3,1,1/
      DATA (CLASSES(L),L=929,936) /
     $ 2,3,1,3,1,1,3,3/
      DATA (CLASSES(L),L=937,944) /
     $ 2,3,1,3,1,1,2,3/
      DATA (CLASSES(L),L=945,952) /
     $ 1,3,1,1,2,3,1,3/
      DATA (CLASSES(L),L=953,954) /
     $ 1,1/
      DATA (LEGS(L),L=1,8) /
     $ 0,0,0,0,5,0,0,0/
      DATA (LEGS(L),L=9,16) /
     $ 5,0,0,0,0,0,4,0/
      DATA (LEGS(L),L=17,24) /
     $ 0,0,4,0,2,0,5,0/
      DATA (LEGS(L),L=25,32) /
     $ 2,5,2,0,4,0,2,4/
      DATA (LEGS(L),L=33,40) /
     $ 2,0,0,4,0,5,0,0/
      DATA (LEGS(L),L=41,48) /
     $ 2,5,0,4,2,0,0,5/
      DATA (LEGS(L),L=49,56) /
     $ 0,4,0,0,2,4,0,5/
      DATA (LEGS(L),L=57,64) /
     $ 2,0,0,0,2,0,2,0/
      DATA (LEGS(L),L=65,72) /
     $ 5,4,0,0,5,0,2,0/
      DATA (LEGS(L),L=73,80) /
     $ 0,4,2,0,0,0,2,0/
      DATA (LEGS(L),L=81,88) /
     $ 0,0,4,0,5,0,4,5/
      DATA (LEGS(L),L=89,96) /
     $ 4,4,5,5,0,0,0,0/
      DATA (LEGS(L),L=97,104) /
     $ 0,0,5,0,0,0,5,0/
      DATA (LEGS(L),L=105,112) /
     $ 0,0,0,0,3,0,0,0/
      DATA (LEGS(L),L=113,120) /
     $ 3,0,2,0,5,0,2,5/
      DATA (LEGS(L),L=121,128) /
     $ 2,0,3,0,2,3,2,0/
      DATA (LEGS(L),L=129,136) /
     $ 0,3,0,5,0,0,2,5/
      DATA (LEGS(L),L=137,144) /
     $ 0,3,2,0,0,5,0,3/
      DATA (LEGS(L),L=145,152) /
     $ 0,0,2,3,0,5,2,0/
      DATA (LEGS(L),L=153,160) /
     $ 5,3,0,0,5,0,2,0/
      DATA (LEGS(L),L=161,168) /
     $ 0,3,0,0,0,0,4,0/
      DATA (LEGS(L),L=169,176) /
     $ 0,0,4,0,0,0,0,0/
      DATA (LEGS(L),L=177,184) /
     $ 3,0,0,0,3,0,2,0/
      DATA (LEGS(L),L=185,192) /
     $ 4,0,2,4,2,0,3,0/
      DATA (LEGS(L),L=193,200) /
     $ 2,3,2,0,0,3,0,4/
      DATA (LEGS(L),L=201,208) /
     $ 0,0,2,4,0,3,2,0/
      DATA (LEGS(L),L=209,216) /
     $ 0,4,0,3,0,0,2,3/
      DATA (LEGS(L),L=217,224) /
     $ 0,4,2,0,4,3,0,0/
      DATA (LEGS(L),L=225,232) /
     $ 4,0,2,0,0,3,1,0/
      DATA (LEGS(L),L=233,240) /
     $ 4,0,1,4,1,0,0,4/
      DATA (LEGS(L),L=241,248) /
     $ 0,5,0,0,1,5,0,4/
      DATA (LEGS(L),L=249,256) /
     $ 1,0,5,0,1,5,1,0/
      DATA (LEGS(L),L=257,264) /
     $ 0,5,0,4,0,0,1,4/
      DATA (LEGS(L),L=265,272) /
     $ 0,5,1,0,0,0,1,0/
      DATA (LEGS(L),L=273,280) /
     $ 1,0,5,4,0,0,5,0/
      DATA (LEGS(L),L=281,288) /
     $ 1,0,0,4,1,0,0,0/
      DATA (LEGS(L),L=289,296) /
     $ 1,0,1,0,3,0,1,3/
      DATA (LEGS(L),L=297,304) /
     $ 1,0,0,3,0,5,0,0/
      DATA (LEGS(L),L=305,312) /
     $ 1,5,0,3,1,0,5,0/
      DATA (LEGS(L),L=313,320) /
     $ 1,5,1,0,0,5,0,3/
      DATA (LEGS(L),L=321,328) /
     $ 0,0,1,3,0,5,1,0/
      DATA (LEGS(L),L=329,336) /
     $ 5,3,0,0,5,0,1,0/
      DATA (LEGS(L),L=337,344) /
     $ 0,3,1,0,3,0,1,3/
      DATA (LEGS(L),L=345,352) /
     $ 1,0,0,3,0,4,0,0/
      DATA (LEGS(L),L=353,360) /
     $ 1,4,0,3,1,0,4,0/
      DATA (LEGS(L),L=361,368) /
     $ 1,4,1,0,0,4,0,3/
      DATA (LEGS(L),L=369,376) /
     $ 0,0,1,3,0,4,1,0/
      DATA (LEGS(L),L=377,384) /
     $ 4,3,0,0,4,0,1,0/
      DATA (LEGS(L),L=385,392) /
     $ 0,3,1,0,0,3,2,0/
      DATA (LEGS(L),L=393,400) /
     $ 0,4,0,5,3,0,0,1/
      DATA (LEGS(L),L=401,408) /
     $ 5,0,0,4,0,2,1,0/
      DATA (LEGS(L),L=409,416) /
     $ 0,3,2,0,0,5,0,4/
      DATA (LEGS(L),L=417,424) /
     $ 3,0,0,1,4,0,0,5/
      DATA (LEGS(L),L=425,432) /
     $ 0,2,1,0,0,4,2,0/
      DATA (LEGS(L),L=433,440) /
     $ 0,3,0,5,4,0,0,1/
      DATA (LEGS(L),L=441,448) /
     $ 5,0,0,3,0,2,1,0/
      DATA (LEGS(L),L=449,456) /
     $ 0,4,2,0,0,5,0,3/
      DATA (LEGS(L),L=457,464) /
     $ 4,0,0,1,3,0,0,5/
      DATA (LEGS(L),L=465,472) /
     $ 0,2,1,0,0,5,2,0/
      DATA (LEGS(L),L=473,480) /
     $ 0,3,0,4,5,0,0,1/
      DATA (LEGS(L),L=481,488) /
     $ 4,0,0,3,0,2,1,0/
      DATA (LEGS(L),L=489,496) /
     $ 0,5,2,0,0,4,0,3/
      DATA (LEGS(L),L=497,504) /
     $ 5,0,0,1,3,0,0,4/
      DATA (LEGS(L),L=505,512) /
     $ 0,2,1,0,3,2,0,0/
      DATA (LEGS(L),L=513,520) /
     $ 3,0,1,0,0,2,1,0/
      DATA (LEGS(L),L=521,528) /
     $ 0,2,0,3,0,0,1,3/
      DATA (LEGS(L),L=529,536) /
     $ 0,2,1,0,0,3,0,2/
      DATA (LEGS(L),L=537,544) /
     $ 0,0,1,2,0,3,1,0/
      DATA (LEGS(L),L=545,552) /
     $ 0,4,3,0,0,5,0,2/
      DATA (LEGS(L),L=553,560) /
     $ 4,0,0,1,2,0,0,5/
      DATA (LEGS(L),L=561,568) /
     $ 0,3,1,0,0,4,3,0/
      DATA (LEGS(L),L=569,576) /
     $ 0,2,0,5,4,0,0,1/
      DATA (LEGS(L),L=577,584) /
     $ 5,0,0,2,0,3,1,0/
      DATA (LEGS(L),L=585,592) /
     $ 0,5,4,0,0,2,0,3/
      DATA (LEGS(L),L=593,600) /
     $ 5,0,0,1,3,0,0,2/
      DATA (LEGS(L),L=601,608) /
     $ 0,4,1,0,0,5,3,0/
      DATA (LEGS(L),L=609,616) /
     $ 0,4,0,2,5,0,0,1/
      DATA (LEGS(L),L=617,624) /
     $ 2,0,0,4,0,3,1,0/
      DATA (LEGS(L),L=625,632) /
     $ 0,5,3,0,0,2,0,4/
      DATA (LEGS(L),L=633,640) /
     $ 5,0,0,1,4,0,0,2/
      DATA (LEGS(L),L=641,648) /
     $ 0,3,1,0,0,5,4,0/
      DATA (LEGS(L),L=649,656) /
     $ 0,3,0,2,5,0,0,1/
      DATA (LEGS(L),L=657,664) /
     $ 2,0,0,3,0,4,1,0/
      DATA (LEGS(L),L=665,672) /
     $ 0,3,0,2,0,0,1,2/
      DATA (LEGS(L),L=673,680) /
     $ 0,3,0,0,4,0,5,0/
      DATA (LEGS(L),L=681,688) /
     $ 4,5,4,4,5,5,0,0/
      DATA (LEGS(L),L=689,696) /
     $ 3,0,5,0,3,5,3,0/
      DATA (LEGS(L),L=697,704) /
     $ 4,0,3,4,3,0,0,4/
      DATA (LEGS(L),L=705,712) /
     $ 0,5,0,0,3,5,0,4/
      DATA (LEGS(L),L=713,720) /
     $ 3,0,0,5,0,4,0,0/
      DATA (LEGS(L),L=721,728) /
     $ 3,4,0,5,3,0,0,0/
      DATA (LEGS(L),L=729,736) /
     $ 3,0,3,0,5,4,0,0/
      DATA (LEGS(L),L=737,744) /
     $ 5,0,3,0,0,4,3,0/
      DATA (LEGS(L),L=745,752) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=753,760) /
     $ 3,4,3,0,5,0,3,5/
      DATA (LEGS(L),L=761,768) /
     $ 3,0,4,0,3,4,3,0/
      DATA (LEGS(L),L=769,776) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=777,784) /
     $ 3,4,3,0,0,4,0,5/
      DATA (LEGS(L),L=785,792) /
     $ 0,0,3,5,0,4,3,0/
      DATA (LEGS(L),L=793,800) /
     $ 0,5,0,4,0,0,3,4/
      DATA (LEGS(L),L=801,808) /
     $ 0,5,3,0,0,0,3,0/
      DATA (LEGS(L),L=809,816) /
     $ 3,0,5,4,0,0,5,0/
      DATA (LEGS(L),L=817,824) /
     $ 3,0,0,4,0,0,4,0/
      DATA (LEGS(L),L=825,832) /
     $ 5,0,4,5,3,0,5,0/
      DATA (LEGS(L),L=833,840) /
     $ 3,5,3,0,4,0,3,4/
      DATA (LEGS(L),L=841,848) /
     $ 0,0,4,0,5,0,4,5/
      DATA (LEGS(L),L=849,856) /
     $ 3,0,5,0,3,5,3,0/
      DATA (LEGS(L),L=857,864) /
     $ 4,0,3,4,3,0,5,0/
      DATA (LEGS(L),L=865,872) /
     $ 3,5,3,0,4,0,3,4/
      DATA (LEGS(L),L=873,880) /
     $ 3,0,0,4,0,5,0,0/
      DATA (LEGS(L),L=881,888) /
     $ 3,5,0,4,3,0,0,5/
      DATA (LEGS(L),L=889,896) /
     $ 0,4,0,0,3,4,0,5/
      DATA (LEGS(L),L=897,904) /
     $ 3,0,0,0,3,0,3,0/
      DATA (LEGS(L),L=905,912) /
     $ 5,4,0,0,5,0,3,0/
      DATA (LEGS(L),L=913,920) /
     $ 0,4,0,0,4,0,5,0/
      DATA (LEGS(L),L=921,928) /
     $ 4,5,3,0,5,0,3,5/
      DATA (LEGS(L),L=929,936) /
     $ 3,0,4,0,3,4,0,0/
      DATA (LEGS(L),L=937,944) /
     $ 4,0,5,0,4,5,3,0/
      DATA (LEGS(L),L=945,952) /
     $ 5,0,3,5,3,0,4,0/
      DATA (LEGS(L),L=953,954) /
     $ 3,4/
      DATA (SIGNS(L),L=1,8) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (SIGNS(L),L=9,16) /
     $ -1,0,0,0,0,0,-1,0/
      DATA (SIGNS(L),L=17,24) /
     $ 0,0,-1,0,-1,0,-1,0/
      DATA (SIGNS(L),L=25,32) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=33,40) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=41,48) /
     $ -1,1,0,1,-1,0,0,-1/
      DATA (SIGNS(L),L=49,56) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=57,64) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=65,72) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=73,80) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=81,88) /
     $ 0,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=89,96) /
     $ -1,-1,-1,-1,0,0,0,0/
      DATA (SIGNS(L),L=97,104) /
     $ 0,0,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=105,112) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (SIGNS(L),L=113,120) /
     $ -1,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=121,128) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=129,136) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=137,144) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=145,152) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=153,160) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=161,168) /
     $ 0,1,0,0,0,0,-1,0/
      DATA (SIGNS(L),L=169,176) /
     $ 0,0,-1,0,0,0,0,0/
      DATA (SIGNS(L),L=177,184) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=185,192) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=193,200) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=201,208) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=209,216) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=217,224) /
     $ 0,1,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=225,232) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=233,240) /
     $ -1,0,-1,1,-1,0,0,-1/
      DATA (SIGNS(L),L=241,248) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=249,256) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=257,264) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=265,272) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=273,280) /
     $ -1,0,1,-1,0,0,1,0/
      DATA (SIGNS(L),L=281,288) /
     $ -1,0,0,1,-1,0,0,0/
      DATA (SIGNS(L),L=289,296) /
     $ -1,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=297,304) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=305,312) /
     $ -1,1,0,1,-1,0,-1,0/
      DATA (SIGNS(L),L=313,320) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=321,328) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=329,336) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=337,344) /
     $ 0,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=345,352) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=353,360) /
     $ -1,1,0,1,-1,0,-1,0/
      DATA (SIGNS(L),L=361,368) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=369,376) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=377,384) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=385,392) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=393,400) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=401,408) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=409,416) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=417,424) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=425,432) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=433,440) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=441,448) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=449,456) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=457,464) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=465,472) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=473,480) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=481,488) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=489,496) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=497,504) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=505,512) /
     $ 0,1,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=513,520) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=521,528) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=529,536) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=537,544) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=545,552) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=553,560) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=561,568) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=569,576) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=577,584) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=585,592) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=593,600) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=601,608) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=609,616) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=617,624) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=625,632) /
     $ 0,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=633,640) /
     $ 1,0,0,-1,1,0,0,1/
      DATA (SIGNS(L),L=641,648) /
     $ 0,1,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=649,656) /
     $ 0,-1,0,-1,1,0,0,-1/
      DATA (SIGNS(L),L=657,664) /
     $ 1,0,0,1,0,1,-1,0/
      DATA (SIGNS(L),L=665,672) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=673,680) /
     $ 0,1,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=681,688) /
     $ -1,1,-1,-1,-1,-1,0,0/
      DATA (SIGNS(L),L=689,696) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=697,704) /
     $ -1,0,-1,1,-1,0,0,-1/
      DATA (SIGNS(L),L=705,712) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=713,720) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=721,728) /
     $ -1,1,0,1,-1,0,0,0/
      DATA (SIGNS(L),L=729,736) /
     $ -1,0,-1,0,1,-1,0,0/
      DATA (SIGNS(L),L=737,744) /
     $ 1,0,-1,0,0,1,-1,0/
      DATA (SIGNS(L),L=745,752) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=753,760) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=761,768) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=769,776) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=777,784) /
     $ -1,1,-1,0,0,-1,0,-1/
      DATA (SIGNS(L),L=785,792) /
     $ 0,0,-1,1,0,1,-1,0/
      DATA (SIGNS(L),L=793,800) /
     $ 0,-1,0,-1,0,0,-1,1/
      DATA (SIGNS(L),L=801,808) /
     $ 0,1,-1,0,0,0,-1,0/
      DATA (SIGNS(L),L=809,816) /
     $ -1,0,1,-1,0,0,1,0/
      DATA (SIGNS(L),L=817,824) /
     $ -1,0,0,1,0,0,-1,0/
      DATA (SIGNS(L),L=825,832) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=833,840) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=841,848) /
     $ 0,0,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=849,856) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=857,864) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=865,872) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=873,880) /
     $ -1,0,0,-1,0,-1,0,0/
      DATA (SIGNS(L),L=881,888) /
     $ -1,1,0,1,-1,0,0,-1/
      DATA (SIGNS(L),L=889,896) /
     $ 0,-1,0,0,-1,1,0,1/
      DATA (SIGNS(L),L=897,904) /
     $ -1,0,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=905,912) /
     $ 1,-1,0,0,1,0,-1,0/
      DATA (SIGNS(L),L=913,920) /
     $ 0,1,0,0,-1,0,-1,0/
      DATA (SIGNS(L),L=921,928) /
     $ -1,1,-1,0,-1,0,-1,1/
      DATA (SIGNS(L),L=929,936) /
     $ -1,0,-1,0,-1,1,0,0/
      DATA (SIGNS(L),L=937,944) /
     $ -1,0,-1,0,-1,1,-1,0/
      DATA (SIGNS(L),L=945,952) /
     $ -1,0,-1,1,-1,0,-1,0/
      DATA (SIGNS(L),L=953,954) /
     $ -1,1/
      DATA (CLOSURES(L),L=1,8) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=9,16) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=17,24) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=25,32) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=33,40) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=41,48) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=49,56) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=57,64) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=65,72) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=73,80) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=81,88) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=89,96) /
     $ 1,1,1,1,0,0,0,0/
      DATA (CLOSURES(L),L=97,104) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=105,112) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=113,120) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=121,128) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=129,136) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=137,144) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=145,152) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=153,160) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=161,168) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=169,176) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=177,184) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=185,192) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=193,200) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=201,208) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=209,216) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=217,224) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=225,232) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=233,240) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=241,248) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=249,256) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=257,264) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=265,272) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=273,280) /
     $ 1,0,0,0,0,0,-1,0/
      DATA (CLOSURES(L),L=281,288) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=289,296) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=297,304) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=305,312) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=313,320) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=321,328) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=329,336) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=337,344) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=345,352) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=353,360) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=361,368) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=369,376) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=377,384) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=385,392) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=393,400) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=401,408) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=409,416) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=417,424) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=425,432) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=433,440) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=441,448) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=449,456) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=457,464) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=465,472) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=473,480) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=481,488) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=489,496) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=497,504) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=505,512) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=513,520) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=521,528) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=529,536) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=537,544) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=545,552) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=553,560) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=561,568) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=569,576) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=577,584) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=585,592) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=593,600) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=601,608) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=609,616) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=617,624) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=625,632) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=633,640) /
     $ -1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=641,648) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=649,656) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=657,664) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=665,672) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=673,680) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=681,688) /
     $ 0,0,1,1,1,1,0,0/
      DATA (CLOSURES(L),L=689,696) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=697,704) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=705,712) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=713,720) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=721,728) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=729,736) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=737,744) /
     $ -1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=745,752) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=753,760) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=761,768) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=769,776) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=777,784) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=785,792) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=793,800) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=801,808) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=809,816) /
     $ 1,0,0,0,0,0,-1,0/
      DATA (CLOSURES(L),L=817,824) /
     $ 0,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=825,832) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=833,840) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=841,848) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=849,856) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=857,864) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=865,872) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=873,880) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=881,888) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=889,896) /
     $ 0,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=897,904) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=905,912) /
     $ 0,0,0,0,-1,0,0,0/
      DATA (CLOSURES(L),L=913,920) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=921,928) /
     $ 0,0,1,0,0,0,0,0/
      DATA (CLOSURES(L),L=929,936) /
     $ 1,0,0,0,0,0,0,0/
      DATA (CLOSURES(L),L=937,944) /
     $ 1,0,0,0,0,0,1,0/
      DATA (CLOSURES(L),L=945,952) /
     $ 0,0,0,0,1,0,0,0/
      DATA (CLOSURES(L),L=953,954) /
     $ 0,0/
      DATA (SHELLS(L),L=1,8) /
     $ 0,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=9,16) /
     $ 1,0,0,0,0,0,0,0/
      DATA (SHELLS(L),L=17,24) /
     $ 0,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=25,32) /
     $ 1,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=33,40) /
     $ 1,0,0,0,0,1,0,0/
      DATA (SHELLS(L),L=41,48) /
     $ 1,1,0,0,1,0,0,1/
      DATA (SHELLS(L),L=49,56) /
     $ 0,0,0,0,1,0,0,1/
      DATA (SHELLS(L),L=57,64) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=65,72) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=73,80) /
     $ 0,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=81,88) /
     $ 0,0,0,0,1,0,0,1/
      DATA (SHELLS(L),L=89,96) /
     $ 0,0,1,1,0,0,0,0/
      DATA (SHELLS(L),L=97,104) /
     $ 0,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=105,112) /
     $ 0,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=113,120) /
     $ 1,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=121,128) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=129,136) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=137,144) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=145,152) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=153,160) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=161,168) /
     $ 0,1,0,0,0,0,0,0/
      DATA (SHELLS(L),L=169,176) /
     $ 0,0,0,0,0,0,0,0/
      DATA (SHELLS(L),L=177,184) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=185,192) /
     $ 0,0,1,0,1,0,1,0/
      DATA (SHELLS(L),L=193,200) /
     $ 1,1,1,0,0,1,0,0/
      DATA (SHELLS(L),L=201,208) /
     $ 0,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=209,216) /
     $ 0,0,0,1,0,0,1,1/
      DATA (SHELLS(L),L=217,224) /
     $ 0,0,1,0,0,1,0,0/
      DATA (SHELLS(L),L=225,232) /
     $ 0,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=233,240) /
     $ 0,0,1,0,1,0,0,0/
      DATA (SHELLS(L),L=241,248) /
     $ 0,1,0,0,1,1,0,0/
      DATA (SHELLS(L),L=249,256) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=257,264) /
     $ 0,1,0,0,0,0,1,0/
      DATA (SHELLS(L),L=265,272) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=273,280) /
     $ 1,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=281,288) /
     $ 1,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=289,296) /
     $ 1,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=297,304) /
     $ 1,0,0,1,0,1,0,0/
      DATA (SHELLS(L),L=305,312) /
     $ 1,1,0,1,1,0,1,0/
      DATA (SHELLS(L),L=313,320) /
     $ 1,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=321,328) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=329,336) /
     $ 1,1,0,0,1,0,1,0/
      DATA (SHELLS(L),L=337,344) /
     $ 0,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=345,352) /
     $ 1,0,0,1,0,0,0,0/
      DATA (SHELLS(L),L=353,360) /
     $ 1,0,0,1,1,0,0,0/
      DATA (SHELLS(L),L=361,368) /
     $ 1,0,1,0,0,0,0,1/
      DATA (SHELLS(L),L=369,376) /
     $ 0,0,1,1,0,0,1,0/
      DATA (SHELLS(L),L=377,384) /
     $ 0,1,0,0,0,0,1,0/
      DATA (SHELLS(L),L=385,392) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=393,400) /
     $ 0,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=401,408) /
     $ 1,0,0,0,0,1,1,0/
      DATA (SHELLS(L),L=409,416) /
     $ 0,1,1,0,0,1,0,0/
      DATA (SHELLS(L),L=417,424) /
     $ 1,0,0,1,0,0,0,1/
      DATA (SHELLS(L),L=425,432) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=433,440) /
     $ 0,1,0,1,0,0,0,1/
      DATA (SHELLS(L),L=441,448) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=449,456) /
     $ 0,0,1,0,0,1,0,1/
      DATA (SHELLS(L),L=457,464) /
     $ 0,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=465,472) /
     $ 0,1,1,0,0,1,1,0/
      DATA (SHELLS(L),L=473,480) /
     $ 0,1,0,0,1,0,0,1/
      DATA (SHELLS(L),L=481,488) /
     $ 0,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=489,496) /
     $ 0,1,1,0,0,0,0,1/
      DATA (SHELLS(L),L=497,504) /
     $ 1,0,0,1,1,0,0,0/
      DATA (SHELLS(L),L=505,512) /
     $ 0,1,1,0,1,1,0,0/
      DATA (SHELLS(L),L=513,520) /
     $ 1,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=521,528) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=529,536) /
     $ 0,1,1,0,0,1,0,1/
      DATA (SHELLS(L),L=537,544) /
     $ 0,0,1,1,0,1,1,0/
      DATA (SHELLS(L),L=545,552) /
     $ 0,0,1,0,0,1,0,1/
      DATA (SHELLS(L),L=553,560) /
     $ 0,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=561,568) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=569,576) /
     $ 0,1,0,1,0,0,0,1/
      DATA (SHELLS(L),L=577,584) /
     $ 1,0,0,1,0,1,1,0/
      DATA (SHELLS(L),L=585,592) /
     $ 0,1,0,0,0,1,0,1/
      DATA (SHELLS(L),L=593,600) /
     $ 1,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=601,608) /
     $ 0,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=609,616) /
     $ 0,0,0,1,1,0,0,1/
      DATA (SHELLS(L),L=617,624) /
     $ 1,0,0,0,0,1,1,0/
      DATA (SHELLS(L),L=625,632) /
     $ 0,1,1,0,0,1,0,0/
      DATA (SHELLS(L),L=633,640) /
     $ 1,0,0,1,0,0,0,1/
      DATA (SHELLS(L),L=641,648) /
     $ 0,1,1,0,0,1,0,0/
      DATA (SHELLS(L),L=649,656) /
     $ 0,1,0,1,1,0,0,1/
      DATA (SHELLS(L),L=657,664) /
     $ 1,0,0,1,0,0,1,0/
      DATA (SHELLS(L),L=665,672) /
     $ 0,1,0,1,0,0,1,1/
      DATA (SHELLS(L),L=673,680) /
     $ 0,1,0,0,0,0,1,0/
      DATA (SHELLS(L),L=681,688) /
     $ 0,1,0,0,1,1,0,0/
      DATA (SHELLS(L),L=689,696) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=697,704) /
     $ 0,0,1,0,1,0,0,0/
      DATA (SHELLS(L),L=705,712) /
     $ 0,1,0,0,1,1,0,0/
      DATA (SHELLS(L),L=713,720) /
     $ 1,0,0,1,0,0,0,0/
      DATA (SHELLS(L),L=721,728) /
     $ 1,0,0,1,1,0,0,0/
      DATA (SHELLS(L),L=729,736) /
     $ 1,0,1,0,1,0,0,0/
      DATA (SHELLS(L),L=737,744) /
     $ 1,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=745,752) /
     $ 1,0,1,1,1,0,0,0/
      DATA (SHELLS(L),L=753,760) /
     $ 1,0,1,0,1,0,1,1/
      DATA (SHELLS(L),L=761,768) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=769,776) /
     $ 1,0,1,1,1,0,0,0/
      DATA (SHELLS(L),L=777,784) /
     $ 1,0,1,0,0,0,0,1/
      DATA (SHELLS(L),L=785,792) /
     $ 0,0,1,1,0,0,1,0/
      DATA (SHELLS(L),L=793,800) /
     $ 0,1,0,0,0,0,1,0/
      DATA (SHELLS(L),L=801,808) /
     $ 0,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=809,816) /
     $ 1,0,1,0,0,0,1,0/
      DATA (SHELLS(L),L=817,824) /
     $ 1,0,0,0,0,0,0,0/
      DATA (SHELLS(L),L=825,832) /
     $ 1,0,0,1,1,0,1,0/
      DATA (SHELLS(L),L=833,840) /
     $ 1,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=841,848) /
     $ 0,0,0,0,1,0,0,1/
      DATA (SHELLS(L),L=849,856) /
     $ 1,0,1,0,1,1,1,0/
      DATA (SHELLS(L),L=857,864) /
     $ 0,0,1,0,1,0,1,0/
      DATA (SHELLS(L),L=865,872) /
     $ 1,1,1,0,0,0,1,0/
      DATA (SHELLS(L),L=873,880) /
     $ 1,0,0,0,0,1,0,0/
      DATA (SHELLS(L),L=881,888) /
     $ 1,1,0,0,1,0,0,1/
      DATA (SHELLS(L),L=889,896) /
     $ 0,0,0,0,1,0,0,1/
      DATA (SHELLS(L),L=897,904) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=905,912) /
     $ 1,0,0,0,1,0,1,0/
      DATA (SHELLS(L),L=913,920) /
     $ 0,0,0,0,0,0,1,0/
      DATA (SHELLS(L),L=921,928) /
     $ 0,1,1,0,1,0,1,1/
      DATA (SHELLS(L),L=929,936) /
     $ 1,0,0,0,1,0,0,0/
      DATA (SHELLS(L),L=937,944) /
     $ 0,0,1,0,0,1,1,0/
      DATA (SHELLS(L),L=945,952) /
     $ 1,0,1,1,1,0,0,0/
      DATA (SHELLS(L),L=953,954) /
     $ 1,0/
      CLASS=0
      LEG=0
      SGN=0
      CLOSURE=0
      DECLARED_ONSHELL=.FALSE.
      KNOWN=.FALSE.
      IF(G.LT.1.OR.G.GT.116) RETURN
      IF(GVALID(G).NE.1.OR.N.NE.GNUM(G)) RETURN
      IF(I.LT.1.OR.J.LT.1.OR.I.GT.N.OR.J.GT.N) RETURN
      IF(I.EQ.J) RETURN
      DIR=0
      IF(CT.EQ.2.OR.CT.EQ.5) DIR=1
      IF(CT.NE.1.AND.CT.NE.2.AND.CT.NE.4.AND.CT.NE.5)
     $ RETURN
      A=MIN(I,J)
      B=MAX(I,J)
      ORI=1
      IF(I.GT.J) ORI=-1
      NP=N*(N-1)/2
      K=GOFF(G)+DIR*NP+(A-1)*N-(A-1)*A/2+B-A
      CLASS=CLASSES(K)
      LEG=LEGS(K)
      SGN=ORI*SIGNS(K)
      CLOSURE=ORI*CLOSURES(K)
      DECLARED_ONSHELL=SHELLS(K).EQ.1
      KNOWN=.TRUE.
      END

      SUBROUTINE OFF_MP_EXTERNAL_SHELL_MASS2(LEG,MASS2,VALID)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER LEG
      REAL*16 MASS2,MASS
      LOGICAL VALID
      INCLUDE 'mp_coupl.inc'
      MASS2=0.0E0_16
      VALID=.FALSE.
      IF(LEG.EQ.1) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.2) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.3) THEN
        MASS=MP__MDL_MZ
        IF(.NOT.IEEE_IS_FINITE(MASS)) RETURN
        IF(MASS.LT.0.0E0_16) RETURN
        IF(MASS.GE.SQRT(HUGE(MASS))) RETURN
        MASS2=MASS**2
        VALID=IEEE_IS_FINITE(MASS2)
        RETURN
      ENDIF
      IF(LEG.EQ.4) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      IF(LEG.EQ.5) THEN
        VALID=.TRUE.
        RETURN
      ENDIF
      END

      SUBROUTINE OFF_MP_BUILD_KINEMATIC_MATRIX_KIN(G,CT,POLICY,
     $ N,P_LOOP,M2L,S_MAT)
      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
      IMPLICIT NONE
      INTEGER G,CT,POLICY,N,I,J,K,CLASS,LEG,SGN,CLOSURE
      REAL*16 P_LOOP(N,0:3),REF_NORMALIZATION,TARGET,TOL
      REAL*16 CLOSURE_SIZE,PAIR_SCALE,RESIDUAL
      COMPLEX*32 M2L(N),S_MAT(N,N),DIFFSQ
      LOGICAL KNOWN,DECLARED_ONSHELL,TARGET_VALID,TOL_OK
      LOGICAL P_FINITE
      INCLUDE 'MadLoopParams.inc'
C     Policy zero is explicit exact legacy compatibility, including its
C     internal-mass threshold guesses. No metadata lookup occurs there.
      IF(POLICY.EQ.0) THEN
        CALL OFF_MP_BUILD_KINEMATIC_MATRIX(N,P_LOOP,M2L,S_MAT)
        RETURN
      ENDIF
C     No coordinate repair, projection, parameter update or flag
C      masking.
C     First shift is the literal corner sum in these adapters, not
C      zero.
      TOL=REAL(OSTHRES,KIND=16)
      TOL_OK=IEEE_IS_FINITE(TOL)
      IF(TOL_OK) TOL_OK=TOL.GT.0.0E0_16.AND.TOL.LT.1.0E0_16
      P_FINITE=ALL(IEEE_IS_FINITE(P_LOOP))
      CLOSURE_SIZE=0.0E0_16
      IF(N.GT.0.AND.TOL_OK.AND.POLICY.EQ.1.AND.P_FINITE) THEN
        CLOSURE_SIZE=MAXVAL(ABS(P_LOOP(1,0:3)))
      ENDIF
      DO I=1,N
        DO J=1,N
          IF(I.EQ.J) THEN
            S_MAT(I,J)=-(M2L(I)+M2L(J))
          ELSE
C           Preserve the original computed-invariant arithmetic order.
            DIFFSQ=(CMPLX(P_LOOP(I,0),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,0),0.0E0_16,KIND=16))**2
            DO K=1,3
              DIFFSQ=DIFFSQ-(CMPLX(P_LOOP(I,K),0.0E0_16,KIND=16)-
     $               CMPLX(P_LOOP(J,K),0.0E0_16,KIND=16))**2
            ENDDO
            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)
            IF(POLICY.NE.1.OR..NOT.TOL_OK) CYCLE
            IF(.NOT.P_FINITE) CYCLE
            CALL OFF_KINEMATIC_IDENTITY(G,CT,N,I,J,CLASS,LEG,
     $           SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)
            IF(.NOT.KNOWN.OR..NOT.DECLARED_ONSHELL) CYCLE
            IF(CLASS.NE.1.AND.CLASS.NE.2) CYCLE
C           Normalize to this external pair, not a hard common shift.
            PAIR_SCALE=0.0E0_16
            DO K=0,3
              PAIR_SCALE=MAX(PAIR_SCALE,
     $                      ABS(P_LOOP(I,K)-P_LOOP(J,K)))
            ENDDO
            IF(.NOT.IEEE_IS_FINITE(PAIR_SCALE)) CYCLE
C           Complements additionally need closure at the leg's scale.
C           Reject ratios above one before division, avoiding overflow.
            IF(CLOSURE.NE.0) THEN
              IF(CLOSURE_SIZE.GT.PAIR_SCALE) CYCLE
              IF(PAIR_SCALE.EQ.0.0E0_16) THEN
                IF(CLOSURE_SIZE.NE.0.0E0_16) CYCLE
              ELSE
                IF(CLOSURE_SIZE/PAIR_SCALE.GT.TOL) CYCLE
              ENDIF
            ENDIF
            CALL OFF_MP_EXTERNAL_SHELL_MASS2(LEG,TARGET,TARGET_VALID)
            IF(.NOT.TARGET_VALID) CYCLE
            IF(.NOT.IEEE_IS_FINITE(REAL(DIFFSQ,KIND=16))) CYCLE
            IF(.NOT.IEEE_IS_FINITE(AIMAG(DIFFSQ))) CYCLE
            IF(TARGET.NE.0.0E0_16) THEN
              RESIDUAL=ABS((DIFFSQ-TARGET)/TARGET)
            ELSE
C             A soft external leg must pass at its own four-vector
C             scale; unrelated hard shifts cannot dilute its defect.
C             Extreme scales conservatively retain computed invariants.
              IF(PAIR_SCALE.GE.SQRT(HUGE(PAIR_SCALE))) CYCLE
              IF(PAIR_SCALE.LT.SQRT(TINY(PAIR_SCALE))) CYCLE
              REF_NORMALIZATION=PAIR_SCALE**2
              IF(.NOT.IEEE_IS_FINITE(REF_NORMALIZATION)) CYCLE
              IF(REF_NORMALIZATION.LE.0.0E0_16) CYCLE
              RESIDUAL=ABS(DIFFSQ/REF_NORMALIZATION)
            ENDIF
            IF(IEEE_IS_FINITE(RESIDUAL)) THEN
              IF(RESIDUAL.LT.TOL) THEN
                S_MAT(I,J)=CMPLX(TARGET,0.0E0_16,KIND=16)
     $                     -M2L(I)-M2L(J)
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDDO
      END

