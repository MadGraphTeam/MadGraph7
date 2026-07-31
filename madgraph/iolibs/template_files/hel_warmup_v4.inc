      PROGRAM %(proc_prefix)sHEL_WARMUP
C**************************************************************************
C     Helicity-recycling warm-up driver (standalone --hel_recycling).
C     Links against matrix_orig.f (the un-recycled per-helicity MATRIX with
C     init_mode instrumentation) and evaluates it over several RAMBO phase-
C     space points and every flavor index, recording:
C       * which helicity combinations ever contribute (good helicities), and
C       * which amplitudes are zero (globally, or per helicity),
C     then prints them in the exact format hel_recycle.py / gen_ximprove.py
C     parse ('Matrix Element/Good Helicity:', 'Amplitude/ZEROAMP:',
C     'HEL/ZEROAMP:'). The generation step re-runs hel_recycle with this
C     information to drop the dead work from the final matrix.f.
C**************************************************************************
      use model_object
      IMPLICIT NONE
      INCLUDE "coupl.inc"
      INCLUDE "nexternal.inc"
      INCLUDE "ngraphs.inc"
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER    NPS
      PARAMETER (NPS=16)
C     Phase-space points per CROSSED configuration (the uncrossed one keeps the
C     full NPS): a crossing only adds rows to the union, so a coarser scan is
C     enough and keeps the probe cheap when NCROSS is large.
      INTEGER    NPSCROSS
      PARAMETER (NPSCROSS=4)
C     Crossing codes to scan. 1 (identity only) when the process was generated
C     without crossing symmetry.
      INTEGER    NCROSS
      PARAMETER (NCROSS=%(hr_warmup_ncross)s)
      REAL*8     LIMHEL
      PARAMETER (LIMHEL=1D-12)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C     LOCAL
      INTEGER I, IHEL, ITRY, IFLV, CROSS, NPSUSE, FLAV_EXT
      REAL*8 P(0:3,NEXTERNAL), PMASS(NEXTERNAL), TOTALMASS
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER JC(NEXTERNAL), ICUSE(NEXTERNAL)
      INTEGER NHELDUM(NEXTERNAL), NHELOUT(NEXTERNAL), DUMFLAV
      REAL*8 SQRTS, T, ANS, TSARR(NCOMB)
      LOGICAL GOODHEL(NCOMB)
C     EXTERNAL
      DOUBLE PRECISION %(proc_prefix)sMATRIX
      EXTERNAL %(proc_prefix)sMATRIX
%(hr_warmup_cross_decl)s
C     SHARED WITH matrix_orig.f
      INTEGER NHEL(NEXTERNAL, NCOMB)
      COMMON/%(proc_prefix)sHEL_TABLE/NHEL
      LOGICAL INIT_MODE
      COMMON/%(proc_prefix)sto_determine_zero_hel/INIT_MODE
      INTEGER %(proc_prefix)sCUR_IHEL
      COMMON/%(proc_prefix)sto_cur_ihel/%(proc_prefix)sCUR_IHEL
C-----
C BEGIN CODE
C-----
      CALL SETPARA('param_card.dat')
      INCLUDE "pmass.inc"
      TOTALMASS = 0D0
      DO I=1,NEXTERNAL
        TOTALMASS = TOTALMASS + PMASS(I)
      ENDDO

      INIT_MODE = .TRUE.
      CALL %(proc_prefix)sRESET_ZEROAMP()
      DO I=1,NCOMB
        GOODHEL(I) = .FALSE.
      ENDDO

C     Loop over every flavor index: a helicity that contributes for ANY flavor
C     must be kept, since one recycled matrix.f serves them all. Same for the
C     crossings: the recycled table bakes its helicity rows, and under a
C     crossing those rows are used as the CROSSED configurations, so a row that
C     any crossing needs must survive. Measuring each crossing here is what
C     makes the recycled matrix.f a valid union over all of them.
      DO IFLV=1, NFLAV
        DO CROSS=0, NCROSS-1
          FLAV_EXT = CROSS*NFLAV + IFLV
%(hr_warmup_cross_skip)s
          NPSUSE = NPS
          IF (CROSS.NE.0) NPSUSE = NPSCROSS
          DO ITRY=1, NPSUSE
            IF (NINCOMING.EQ.1) THEN
              SQRTS = PMASS(1)
            ELSE
              SQRTS = 500D0 + 250D0*ITRY
              IF (SQRTS.LE.2D0*TOTALMASS) SQRTS = 2.1D0*TOTALMASS + 100D0*ITRY
            ENDIF
            CALL GET_MOMENTA(SQRTS, PMASS, P)
            DO I=1, NEXTERNAL
              JC(I) = +1
            ENDDO
            PUSE(:,:) = P(:,:)
            ICUSE(:) = JC(:)
%(hr_warmup_cross_apply)s
            ANS = 0D0
            DO IHEL=1, NCOMB
              %(proc_prefix)sCUR_IHEL = IHEL
              T = %(proc_prefix)sMATRIX(PUSE, NHEL(1,IHEL), ICUSE, IFLV)
              TSARR(IHEL) = T
              ANS = ANS + T
            ENDDO
            DO IHEL=1, NCOMB
              IF (DABS(TSARR(IHEL)) .GT. ANS*LIMHEL/NCOMB) GOODHEL(IHEL)=.TRUE.
            ENDDO
          ENDDO
        ENDDO
      ENDDO

      DO IHEL=1, NCOMB
        IF (GOODHEL(IHEL)) THEN
          WRITE(*,*) 'Matrix Element/Good Helicity: 1 ', IHEL
        ENDIF
      ENDDO
      CALL %(proc_prefix)sPRINT_ZERO_AMP()
      END


      SUBROUTINE %(proc_prefix)sRESET_ZEROAMP()
      IMPLICIT NONE
      INTEGER    NCOMB, NGRAPHS
      PARAMETER (NCOMB=%(ncomb)d, NGRAPHS=%(ngraphs)d)
      LOGICAL %(proc_prefix)sZEROAMP(NCOMB, NGRAPHS)
      COMMON/%(proc_prefix)sto_zeroamp/%(proc_prefix)sZEROAMP
      %(proc_prefix)sZEROAMP(:,:) = .TRUE.
      END


      SUBROUTINE %(proc_prefix)sPRINT_ZERO_AMP()
      IMPLICIT NONE
      INTEGER    NCOMB, NGRAPHS
      PARAMETER (NCOMB=%(ncomb)d, NGRAPHS=%(ngraphs)d)
      LOGICAL %(proc_prefix)sZEROAMP(NCOMB, NGRAPHS)
      COMMON/%(proc_prefix)sto_zeroamp/%(proc_prefix)sZEROAMP
      INTEGER I, J
      LOGICAL ALL_FALSE
      DO I=1, NGRAPHS
        ALL_FALSE = .TRUE.
        DO J=1, NCOMB
          IF (.NOT.%(proc_prefix)sZEROAMP(J,I)) THEN
            ALL_FALSE = .FALSE.
            GOTO 20
          ENDIF
        ENDDO
 20     CONTINUE
        IF (ALL_FALSE) THEN
          WRITE(*,*) 'Amplitude/ZEROAMP:', 1, I
        ELSE
          DO J=1, NCOMB
            IF (%(proc_prefix)sZEROAMP(J,I)) THEN
              WRITE(*,*) 'HEL/ZEROAMP:', 1, J, I
            ENDIF
          ENDDO
        ENDIF
      ENDDO
      END


      DOUBLE PRECISION FUNCTION DOT(P1,P2)
C     4-Vector Dot product
      IMPLICIT NONE
      DOUBLE PRECISION P1(0:3),P2(0:3)
      DOT=P1(0)*P2(0)-P1(1)*P2(1)-P1(2)*P2(2)-P1(3)*P2(3)
      END


      SUBROUTINE GET_MOMENTA(ENERGY,PMASS,P)
C---- auxiliary function to change convention between MadGraph5_aMC@NLO and rambo
C---- four momenta.
      IMPLICIT NONE
      INCLUDE "nexternal.inc"
C     ARGUMENTS
      REAL*8 ENERGY,PMASS(NEXTERNAL),P(0:3,NEXTERNAL),PRAMBO(4,10),WGT
C     LOCAL
      INTEGER I
      REAL*8 etot2,mom,m1,m2,e1,e2
      ETOT2=energy**2
      if(nincoming.eq.2) then
         m1=pmass(1)
         m2=pmass(2)
         mom=(Etot2**2 - 2*Etot2*m1**2 + m1**4 -
     &        2*Etot2*m2**2 - 2*m1**2*m2**2 + m2**4)/(4.*Etot2)
         mom=dsqrt(mom)
         e1=DSQRT(mom**2+m1**2)
         e2=DSQRT(mom**2+m2**2)
        P(0,1)=e1
        P(1,1)=0d0
        P(2,1)=0d0
        P(3,1)=mom
        P(0,2)=e2
        P(1,2)=0d0
        P(2,2)=0d0
        P(3,2)=-mom
        call rambo(nexternal-2,energy,pmass(nincoming+1),prambo,WGT)
        DO I=3, NEXTERNAL
           P(0,I)=PRAMBO(4,I-2)
           P(1,I)=PRAMBO(1,I-2)
           P(2,I)=PRAMBO(2,I-2)
           P(3,I)=PRAMBO(3,I-2)
        ENDDO
      elseif(nincoming.eq.1) then
         P(0,1)=energy
         P(1,1)=0d0
         P(2,1)=0d0
         P(3,1)=0d0
         call rambo(nexternal-1,energy,pmass(2),prambo,WGT)
         DO I=2, NEXTERNAL
            P(0,I)=PRAMBO(4,I-1)
            P(1,I)=PRAMBO(1,I-1)
            P(2,I)=PRAMBO(2,I-1)
            P(3,I)=PRAMBO(3,I-1)
         ENDDO
      endif
      RETURN
      END


      SUBROUTINE RAMBO(N,ET,XM,P,WT)
C***********************************************************************
C                       RAMBO
C    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)
C    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR
C    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING
C***********************************************************************
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE "nexternal.inc"
      DIMENSION XM(NEXTERNAL-NINCOMING),P(4,NEXTERNAL-NINCOMING)
      DIMENSION Q(4,NEXTERNAL-NINCOMING),Z(NEXTERNAL-NINCOMING),R(4),
     .   B(3),P2(NEXTERNAL-NINCOMING),XM2(NEXTERNAL-NINCOMING),
     .   E(NEXTERNAL-NINCOMING),V(NEXTERNAL-NINCOMING),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
      SAVE TWOPI, PO2LOG, Z
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,NEXTERNAL-NINCOMING
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,NEXTERNAL-NINCOMING
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP
  201 DO 202 I=1,N
         r1=rn(1)
      C=2.*r1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RN(2)
      r1=rn(3)
      r2=rn(4)
      Q(4,I)=-LOG(r1*r2)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
  202 Q(1,I)=Q(4,I)*S*SIN(F)
      DO 203 I=1,4
  203 R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
  204 R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
  205 B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(4,I)=X*(G*Q(4,I)+BQ)
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
  209 IF(NM.NE.0) GOTO 210
      RETURN
  210 XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
  301 P2(I)=P(4,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
  302 F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
  303 G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      GOTO 305
  304 X=X-F0/(X*G0)
      GOTO 302
  305 DO 307 I=1,N
      V(I)=X*P(4,I)
      DO 306 K=1,3
  306 P(K,I)=X*P(K,I)
  307 P(4,I)=E(I)
      RETURN
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',
     . ' SMALLER THAN TOTAL ENERGY =',D15.6)
      END


      FUNCTION RN(IDUMMY)
      REAL*8 RN,RAN
      SAVE INIT
      DATA INIT /1/
      IF (INIT.EQ.1) THEN
        INIT=0
        CALL RMARIN(1802,9373)
      END IF
  10  CALL RANMAR(RAN)
      IF (RAN.LT.1D-16) GOTO 10
      RN=RAN
      END


      SUBROUTINE RANMAR(RVEC)
C     Universal random number generator proposed by Marsaglia and Zaman.
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
      UNI = RANU(IRANMR) - RANU(JRANMR)
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RANU(IRANMR) = UNI
      IRANMR = IRANMR - 1
      JRANMR = JRANMR - 1
      IF(IRANMR .EQ. 0) IRANMR = 97
      IF(JRANMR .EQ. 0) JRANMR = 97
      RANC = RANC - RANCD
      IF(RANC .LT. 0D0) RANC = RANC + RANCM
      UNI = UNI - RANC
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RVEC = UNI
      END


      SUBROUTINE RMARIN(IJ,KL)
C     Initializing routine for RANMAR.
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
      I = MOD( IJ/177 , 177 ) + 2
      J = MOD( IJ     , 177 ) + 2
      K = MOD( KL/169 , 178 ) + 1
      L = MOD( KL     , 169 )
      DO 300 II = 1 , 97
        S =  0D0
        T = .5D0
        DO 200 JJ = 1 , 24
          M = MOD( MOD(I*J,179)*K , 179 )
          I = J
          J = K
          K = M
          L = MOD( 53*L+1 , 169 )
          IF(MOD(L*M,64) .GE. 32) S = S + T
          T = .5D0*T
  200   CONTINUE
        RANU(II) = S
  300 CONTINUE
      RANC  =   362436D0 / 16777216D0
      RANCD =  7654321D0 / 16777216D0
      RANCM = 16777213D0 / 16777216D0
      IRANMR = 97
      JRANMR = 33
      END
