      subroutine sborn_hel(p,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest),save::request
      type(BornResult),save::result
      double precision helicities(12)
      common/c_born_hel/helicities
      double precision helicity_orders(1,12)
      common/c_born_hel_split/helicity_orders
      request%helicities=.true.
      call mc_born_local(p,request,result)
      helicities=result%helicities
      helicity_orders=result%helicity_orders
      ans=sum(helicities)
      end
      SUBROUTINE PICKHELICITYMC(P,GOODHEL,HEL,IHEL_OUT,VOL,LOC_SAVEAMP)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)
      INTEGER GOODHEL(MAX_BHEL),HEL(0:MAX_BHEL)
      INTEGER IHEL_OUT
      DOUBLE PRECISION VOL
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
      common/c_born_hel_split/wgt_hel
      DOUBLE PRECISION SUM_HEL(NSQAMPSO)
      INTEGER I, IHEL
      INTEGER N_NONZERO_ORD
      DOUBLE PRECISION SUM_ALL
      DOUBLE PRECISION ACCUM, TARGET
      DOUBLE PRECISION BORN_WGT_RECOMP_DIRECT
      DOUBLE COMPLEX LOC_SAVEAMP(1,MAX_BHEL)
      DOUBLE PRECISION RAN2
      call sborn_hel(p,born_wgt_recomp_direct)
      N_NONZERO_ORD = 0
      SUM_ALL = 0D0
      DO I = 1, NSQAMPSO
        SUM_HEL(I) = 0D0
        DO IHEL = 1, HEL(0)
          IF (WGT_HEL(I, HEL(IHEL)).LT.0D0) THEN
            WRITE(*,*) 'Helicities from squared diagrams must be > 0  !'
            STOP 1
          ENDIF
          SUM_HEL(I)=SUM_HEL(I) + WGT_HEL(I, HEL(IHEL))      *DBLE(GOODHEL(IHEL))
        ENDDO
        IF (SUM_HEL(I).GT.0D0) THEN
          N_NONZERO_ORD = N_NONZERO_ORD + 1
          SUM_ALL = SUM_ALL + SUM_HEL(I)
        ENDIF
      ENDDO
      TARGET=RAN2()
      IHEL=1
      ACCUM=0D0
      DO I = 1, NSQAMPSO
        IF (SUM_HEL(I).EQ.0D0) CYCLE
        ACCUM=ACCUM+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)*DBLE(GOODHEL(IHEL))    /N_NONZERO_ORD
      ENDDO
      DO WHILE (ACCUM.LT.TARGET)
        IHEL=IHEL+1
        DO I = 1, NSQAMPSO
          IF (SUM_HEL(I).EQ.0D0) CYCLE
          ACCUM=ACCUM+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)      *DBLE(GOODHEL(IHEL))/N_NONZERO_ORD
        ENDDO
      ENDDO
      VOL=0D0
      DO I = 1, NSQAMPSO
        IF (SUM_HEL(I).EQ.0D0) CYCLE
        VOL=VOL+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)*DBLE(GOODHEL(IHEL))    /N_NONZERO_ORD
      ENDDO
      IHEL_OUT=IHEL
      RETURN
      END
      SUBROUTINE sborn_hel_buffered(P,ANS_SUMMED,LOC_SAVEAMP,WGT_HEL)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'born_nhel.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      REAL*8 P(0:3,NEXTERNAL-1), ANS_SUMMED
      DOUBLE COMPLEX LOC_SAVEAMP(1,MAX_BHEL)
      INTEGER I, J
      REAL*8 ANS(0:NSQAMPSO)
      DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
      INTEGER GETORDPOWFROMINDEX_B
      CALL SBORN_HEL_SPLITORDERS(P,ANS,LOC_SAVEAMP,WGT_HEL)
      ANS_SUMMED = 0D0
      DO I = 1, NSQAMPSO
        ANS_SUMMED = ANS_SUMMED + ANS(I)
      ENDDO
      RETURN
      END
      SUBROUTINE SBORN_HEL_SPLITORDERS(P1,ANS,LOC_SAVEAMP,WGT_HEL)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  12 )
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*8)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   1)
      REAL*8 P1(0:3,NEXTERNAL-1),ANS(0:NSQAMPSO)
      DOUBLE COMPLEX LOC_SAVEAMP(NGRAPHS,MAX_BHEL)
      INTEGER IHEL,IDEN,I,J
      DOUBLE PRECISION T(NSQAMPSO)
      INTEGER IDEN_VALUES(8)
      DATA IDEN_VALUES /36, 36, 36, 36, 36, 36, 36, 36/
      LOGICAL GOODHEL(NCOMB,8)
      COMMON /C_GOODHEL/ GOODHEL
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
      IDEN=IDEN_VALUES(NFKSPROCESS)
      DO I=0,NSQAMPSO
        ANS(I) = 0D0
      ENDDO
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL,NFKSPROCESS)) THEN
          CALL BORN_HEL_SPLITORDERS(P1,IHEL,T,LOC_SAVEAMP)
          DO I=1,NSQAMPSO
            WGT_HEL(I, IHEL) = T(I) / DBLE(IDEN)
            ANS(I)=ANS(I)+T(I)
          ENDDO
        ENDIF
      ENDDO
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
      SUBROUTINE BORN_HEL_SPLITORDERS(P,HELL,ANS,LOC_SAVEAMP)
      IMPLICIT NONE
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=1, NSQAMPSO=1)
      INTEGER     NGRAPHS
      PARAMETER ( NGRAPHS = 1 )
      INTEGER NCOLOR
      PARAMETER (NCOLOR=1)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER HELL
      REAL*8 ANS(NSQAMPSO)
      DOUBLE COMPLEX LOC_SAVEAMP(NGRAPHS,MAX_BHEL)
      INTEGER I,J,M,N
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(0)
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER SQSOINDEXB
      DATA DENOM/1/
      DATA (CF(I),I=  1,  1) /3/
      DO I=1,NGRAPHS
        AMP(I)=LOC_SAVEAMP(I,HELL)
      ENDDO
      JAMP(:,:) = (0D0,0D0)
      JAMP(1,1) = AMP(1)
      ANS(:) = 0D0
      DO M = 1, NAMPSO
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX +1
            ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
          ANS(SQSOINDEXB(M,M))=ANS(SQSOINDEXB(M,M))+ZTEMP      *DCONJG(JAMP(I,M))
        ENDDO
      ENDDO
      ANS(:) = ANS(:)/DENOM
      END
