C     Exact spectator sum of the linear interference, retaining all
C      colours.
C     The physical Born is kept. Longitudinal corrections are mapped
C      from the
C     Goldstone amplitudes with HELAS equivalence phases. The second
C      output is
C     a spectator-resolved Goldstone-normalised projection for
C      comparison.
C     Only the first nvar variants are evaluated; the others stay zero.

      SUBROUTINE SDK_DENSITY_INTERFERENCE(P,NRES,POS,DIMS,NVAR,BORN,
     $     correction,relative,zero_gold,status)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER NRES,POS(10),DIMS(10),NVAR,ZERO_GOLD,STATUS
      DOUBLE PRECISION P(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX BORN(36,36),CORRECTION(36,36,5)
      DOUBLE COMPLEX RELATIVE(36,36,5)
      INTEGER NCOMB
      PARAMETER (NCOMB=36)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),I
      DATA (NHEL(I,   1),I=1,4) /-1, 1,-1,-1/
      DATA (NHEL(I,   2),I=1,4) /-1, 1,-1, 0/
      DATA (NHEL(I,   3),I=1,4) /-1, 1,-1, 1/
      DATA (NHEL(I,   4),I=1,4) /-1, 1, 0,-1/
      DATA (NHEL(I,   5),I=1,4) /-1, 1, 0, 0/
      DATA (NHEL(I,   6),I=1,4) /-1, 1, 0, 1/
      DATA (NHEL(I,   7),I=1,4) /-1, 1, 1,-1/
      DATA (NHEL(I,   8),I=1,4) /-1, 1, 1, 0/
      DATA (NHEL(I,   9),I=1,4) /-1, 1, 1, 1/
      DATA (NHEL(I,  10),I=1,4) /-1,-1,-1,-1/
      DATA (NHEL(I,  11),I=1,4) /-1,-1,-1, 0/
      DATA (NHEL(I,  12),I=1,4) /-1,-1,-1, 1/
      DATA (NHEL(I,  13),I=1,4) /-1,-1, 0,-1/
      DATA (NHEL(I,  14),I=1,4) /-1,-1, 0, 0/
      DATA (NHEL(I,  15),I=1,4) /-1,-1, 0, 1/
      DATA (NHEL(I,  16),I=1,4) /-1,-1, 1,-1/
      DATA (NHEL(I,  17),I=1,4) /-1,-1, 1, 0/
      DATA (NHEL(I,  18),I=1,4) /-1,-1, 1, 1/
      DATA (NHEL(I,  19),I=1,4) / 1, 1,-1,-1/
      DATA (NHEL(I,  20),I=1,4) / 1, 1,-1, 0/
      DATA (NHEL(I,  21),I=1,4) / 1, 1,-1, 1/
      DATA (NHEL(I,  22),I=1,4) / 1, 1, 0,-1/
      DATA (NHEL(I,  23),I=1,4) / 1, 1, 0, 0/
      DATA (NHEL(I,  24),I=1,4) / 1, 1, 0, 1/
      DATA (NHEL(I,  25),I=1,4) / 1, 1, 1,-1/
      DATA (NHEL(I,  26),I=1,4) / 1, 1, 1, 0/
      DATA (NHEL(I,  27),I=1,4) / 1, 1, 1, 1/
      DATA (NHEL(I,  28),I=1,4) / 1,-1,-1,-1/
      DATA (NHEL(I,  29),I=1,4) / 1,-1,-1, 0/
      DATA (NHEL(I,  30),I=1,4) / 1,-1,-1, 1/
      DATA (NHEL(I,  31),I=1,4) / 1,-1, 0,-1/
      DATA (NHEL(I,  32),I=1,4) / 1,-1, 0, 0/
      DATA (NHEL(I,  33),I=1,4) / 1,-1, 0, 1/
      DATA (NHEL(I,  34),I=1,4) / 1,-1, 1,-1/
      DATA (NHEL(I,  35),I=1,4) / 1,-1, 1, 0/
      DATA (NHEL(I,  36),I=1,4) / 1,-1, 1, 1/
      INTEGER ISPEC,H,HP,R,K,D,IV,S,IP,IB,BASE(36),SOURCE_BASE(NS)
      DATA SOURCE_BASE /1,1,1,1,1,1,1,1,1,1/
      INTEGER HELS(NEXTERNAL-1,36),IFLIST(NEXTERNAL-1)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(1)
      DATA IDEN_VALUES / 72 /
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1,NEXTERNAL-1,5),NORM(36)
      DOUBLE COMPLEX AMPS(NC,NO,NS,36),DERIV(NC,NO,6,36)
      DOUBLE COMPLEX COEFF(NS),PARCOEFF(6,5),PHASE(36)
      DOUBLE COMPLEX DELTA(36,5),MIXED(36,36,5),BROW(36,36)
      DOUBLE COMPLEX XS(36,NS),YS(NS),XD(36,6),YD(6)
      DOUBLE COMPLEX SDK_DENSITY_INTER,TERM
      EXTERNAL SDK_DENSITY_INTER
      LOGICAL OK
      INTEGER SUD_MOD,SAVED_MOD
      COMMON/TO_SUD_MOD/SUD_MOD
      LOGICAL S_TO_RIJ,RIJ_GE_MW,SAVED_STOR,SAVED_RIJ
      COMMON/TO_S_TO_RIJ/S_TO_RIJ
      COMMON/RIJ_GE_MW/RIJ_GE_MW

      BORN=(0D0,0D0)
      CORRECTION=(0D0,0D0)
      RELATIVE=(0D0,0D0)
      ZERO_GOLD=0
C     One to five variants, distinct resonance positions with two or
C      three
C     spin states each, and at most 36 resonance spin states in total.
      STATUS=1
      IF (NVAR.LT.1.OR.NVAR.GT.5) RETURN
      IF (NRES.LT.1.OR.NRES.GT.10) RETURN
      D=1
      DO R=1,NRES
        IF (DIMS(R).LT.2.OR.DIMS(R).GT.3) RETURN
        IF (POS(R).LT.1.OR.POS(R).GT.NEXTERNAL-1) RETURN
        IF (ANY(POS(1:R-1).EQ.POS(R))) RETURN
        IF (D.GT.36/DIMS(R)) RETURN
        D=D*DIMS(R)
      ENDDO
      STATUS=0
      SAVED_MOD=SUD_MOD
      SAVED_STOR=S_TO_RIJ
      SAVED_RIJ=RIJ_GE_MW
      IFLIST=(/-1,-1,1,1/)
C     Each variant builds its invariants with its own floor.
      DO IV=1,NVAR
        CALL SDK_DENSITY_VARIANT(IV)
        CALL SDK_GET_INVARIANTS(P,IFLIST,INVARIANTS(1,1,IV))
        CALL SDK_DENSITY_PARCOEFF(INVARIANTS(1,1,IV),PARCOEFF(1,IV))
      ENDDO

C     One representative of each spectator tuple. No good-helicity
C      filtering:
C     a zero Born component can have a nonzero mixed-helicity
C      interference.
C     Every amplitude read below is evaluated first for this tuple.
      DO ISPEC=1,NCOMB
        IF (ANY(NHEL(POS(1:NRES),ISPEC).NE.-1)) CYCLE
        DO H=1,D
          HELS(:,H)=NHEL(:,ISPEC)
          K=H-1
          DO R=1,NRES
            IP=MOD(K,DIMS(R))
            K=K/DIMS(R)
            IF (DIMS(R).EQ.2) THEN
              HELS(POS(R),H)=2*IP-1
            ELSE
              HELS(POS(R),H)=IP-1
            ENDIF
          ENDDO
          CALL SDK_DENSITY_BASE(HELS(1,H),BASE(H),PHASE(H))
          CALL SDK_DENSITY_EVAL(1,P,HELS(1,H),AMPS(1,1,1,H))
          IB=BASE(H)
          IF (IB.EQ.0) CYCLE
          DO S=2,NS
            IF (SOURCE_BASE(S).EQ.IB)
     $        call sdk_density_eval(s,p,hels(1,h),amps(1,1,s,h))
          ENDDO
        ENDDO
        CALL SDK_DENSITY_DERIVATIVES(P,D,BASE,HELS,DERIV,OK)
        IF (.NOT.OK) THEN
          STATUS=3
          GOTO 900
        ENDIF
        DO H=1,D
          DO HP=1,D
            BROW(H,HP)=SDK_DENSITY_INTER(1,1,AMPS(1,1,1,HP),
     $                                  amps(1,1,1,h))
          ENDDO
          IB=BASE(H)
          IF (IB.EQ.0) THEN
            NORM(H)=0D0
          ELSE
            NORM(H)=DBLE(SDK_DENSITY_INTER(IB,IB,
     $                           amps(1,1,ib,h),amps(1,1,ib,h)))
          ENDIF
          IF (NORM(H).LE.0D0.AND.DBLE(BROW(H,H)).GT.0D0)
     $        zero_gold=zero_gold+1
        ENDDO
        MIXED(1:D,1:D,:)=(0D0,0D0)
        DELTA(1:D,:)=(0D0,0D0)
        DO H=1,D
          IB=BASE(H)
          IF (IB.EQ.0) CYCLE
C         The colour contractions do not depend on the variant.
C          coeff(s) can be
C         nonzero only for the sources of this base.
          DO S=1,NS
            IF (SOURCE_BASE(S).NE.IB) CYCLE
            YS(S)=SDK_DENSITY_INTER(IB,S,AMPS(1,1,IB,H),AMPS(1,1,S,H))
            DO HP=1,D
              XS(HP,S)=SDK_DENSITY_INTER(1,S,AMPS(1,1,1,HP),
     $                                   amps(1,1,s,h))
            ENDDO
          ENDDO
          DO IP=1,6
            YD(IP)=SDK_DENSITY_INTER(IB,IB,AMPS(1,1,IB,H),
     $                               deriv(1,1,ip,h))
            DO HP=1,D
              XD(HP,IP)=SDK_DENSITY_INTER(1,IB,AMPS(1,1,1,HP),
     $                                    deriv(1,1,ip,h))
            ENDDO
          ENDDO
          DO IV=1,NVAR
            CALL SDK_DENSITY_VARIANT(IV)
            CALL SDK_DENSITY_COEFF(IB,HELS(1,H),INVARIANTS(1,1,IV),
     $                             coeff)
            TERM=(0D0,0D0)
            DO S=1,NS
              IF (COEFF(S).EQ.(0D0,0D0)) CYCLE
              TERM=TERM+COEFF(S)*YS(S)
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*COEFF(S)*
     $            xs(hp,s)
              ENDDO
            ENDDO
            DO IP=1,6
              TERM=TERM+PARCOEFF(IP,IV)*YD(IP)
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*
     $            parcoeff(ip,iv)*xd(hp,ip)
              ENDDO
            ENDDO
            IF (NORM(H).GT.0D0) DELTA(H,IV)=TERM/NORM(H)
          ENDDO
        ENDDO
        DO IV=1,NVAR
          DO H=1,D
            DO HP=1,D
              CORRECTION(H,HP,IV)=CORRECTION(H,HP,IV)+
     $             mixed(h,hp,iv)+dconjg(mixed(hp,h,iv))
              RELATIVE(H,HP,IV)=RELATIVE(H,HP,IV)+BROW(H,HP)*
     $             (delta(h,iv)+dconjg(delta(hp,iv)))
            ENDDO
          ENDDO
        ENDDO
        BORN(1:D,1:D)=BORN(1:D,1:D)+BROW(1:D,1:D)
      ENDDO
      BORN(1:D,1:D)=BORN(1:D,1:D)/DBLE(IDEN_VALUES(NFKSPROCESS))
      CORRECTION(1:D,1:D,:)=CORRECTION(1:D,1:D,:)/
     $  dble(iden_values(nfksprocess))
      RELATIVE(1:D,1:D,:)=RELATIVE(1:D,1:D,:)/
     $  dble(iden_values(nfksprocess))
 900  CONTINUE
      SUD_MOD=SAVED_MOD
      S_TO_RIJ=SAVED_STOR
      RIJ_GE_MW=SAVED_RIJ
      END

      SUBROUTINE SDK_DENSITY_VARIANT(IV)
      IMPLICIT NONE
      INTEGER IV,SUD_MOD
      LOGICAL S_TO_RIJ,RIJ_GE_MW
      COMMON/TO_SUD_MOD/SUD_MOD
      COMMON/TO_S_TO_RIJ/S_TO_RIJ
      COMMON/RIJ_GE_MW/RIJ_GE_MW
C     Same order as ewsudakov_py results(2:6): QED, weak, s/r off,
C     s/r off and invariant floor off, invariant floor off.
      SUD_MOD=1
      IF (IV.EQ.1) SUD_MOD=0
      S_TO_RIJ=IV.NE.3.AND.IV.NE.4
      RIJ_GE_MW=IV.LT.4
      END

      SUBROUTINE SDK_DENSITY_BASE(HELS,IBASE,PHASE)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER HELS(NEXTERNAL-1),IBASE
      DOUBLE COMPLEX PHASE
      IBASE=1
      PHASE=(1D0,0D0)
      IF (HELS(3).EQ.0) THEN
        IBASE=0
        PHASE=PHASE*(0D0,-1D0)
      ENDIF
      IF (HELS(4).EQ.0) THEN
        IBASE=0
        PHASE=PHASE*(0D0,-1D0)
      ENDIF
      END

      SUBROUTINE SDK_DENSITY_EVAL(ISOURCE,P,HELS,JAMP)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER ISOURCE,HELS(NEXTERNAL-1)
      DOUBLE PRECISION P(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX JAMP(NC,NO)
      IF (ISOURCE.EQ.1) THEN
        CALL SDK_DENSITY_RAW_1(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.2) THEN
        CALL SDK_DENSITY_RAW_2(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.3) THEN
        CALL SDK_DENSITY_RAW_3(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.4) THEN
        CALL SDK_DENSITY_RAW_4(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.5) THEN
        CALL SDK_DENSITY_RAW_5(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.6) THEN
        CALL SDK_DENSITY_RAW_6(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.7) THEN
        CALL SDK_DENSITY_RAW_7(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.8) THEN
        CALL SDK_DENSITY_RAW_8(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.9) THEN
        CALL SDK_DENSITY_RAW_9(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.10) THEN
        CALL SDK_DENSITY_RAW_10(P,HELS,JAMP)
      ENDIF
      END

      SUBROUTINE SDK_DENSITY_COEFF(IBASE,HELS,INVARIANTS,COEFF)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'ewsudakov_haslo.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER IBASE,HELS(NEXTERNAL-1),IFLIST(NEXTERNAL-1)
      INTEGER PDGLIST(NEXTERNAL-1)
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1,NEXTERNAL-1)
      DOUBLE COMPLEX COEFF(NS)
      DOUBLE COMPLEX GET_LSC_DIAG,GET_SSC_N_DIAG,GET_XXC_DIAG
      DOUBLE COMPLEX GET_LSC_NONDIAG,GET_XXC_NONDIAG
      DOUBLE COMPLEX GET_SSC_N_NONDIAG_1,GET_SSC_N_NONDIAG_2,GET_SSC_C
      EXTERNAL GET_LSC_DIAG,GET_SSC_N_DIAG,GET_XXC_DIAG
      EXTERNAL GET_LSC_NONDIAG,GET_XXC_NONDIAG
      EXTERNAL GET_SSC_N_NONDIAG_1,GET_SSC_N_NONDIAG_2,GET_SSC_C
      COEFF=(0D0,0D0)
      IF (.NOT.HAS_LO1) RETURN
      IFLIST=(/-1,-1,1,1/)
      IF (IBASE.EQ.1) THEN
        PDGLIST=(/-5,5,23,23/)
        COEFF(1)=GET_LSC_DIAG(PDGLIST,HELS,IFLIST,INVARIANTS)
        COEFF(1)=COEFF(1)+GET_SSC_N_DIAG(PDGLIST,HELS,IFLIST
     $   ,INVARIANTS)
        COEFF(1)=COEFF(1)+GET_XXC_DIAG(PDGLIST,HELS,IFLIST,INVARIANTS)
        COEFF(2)=GET_LSC_NONDIAG(PDGLIST,HELS,IFLIST,INVARIANTS,3,23
     $   ,22)+GET_XXC_NONDIAG(PDGLIST,HELS,IFLIST,INVARIANTS,3,23,22)
        COEFF(3)=GET_LSC_NONDIAG(PDGLIST,HELS,IFLIST,INVARIANTS,4,23
     $   ,22)+GET_XXC_NONDIAG(PDGLIST,HELS,IFLIST,INVARIANTS,4,23,22)
        COEFF(4)=GET_SSC_C(1,2,PDGLIST,-6,6,HELS,IFLIST,INVARIANTS)
        COEFF(5)=GET_SSC_C(1,3,PDGLIST,-6,-24,HELS,IFLIST,INVARIANTS)
        COEFF(6)=GET_SSC_C(1,4,PDGLIST,-6,-24,HELS,IFLIST,INVARIANTS)
        COEFF(7)=GET_SSC_C(2,3,PDGLIST,6,24,HELS,IFLIST,INVARIANTS)
        COEFF(8)=GET_SSC_C(2,4,PDGLIST,6,24,HELS,IFLIST,INVARIANTS)
        COEFF(9)=GET_SSC_C(3,4,PDGLIST,24,-24,HELS,IFLIST,INVARIANTS)
        COEFF(10)=GET_SSC_C(3,4,PDGLIST,-24,24,HELS,IFLIST,INVARIANTS)
      ENDIF
      END

      DOUBLE COMPLEX FUNCTION SDK_DENSITY_INTER(IBASE,ISOURCE,A,B)
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER IBASE,ISOURCE,I,J,M,N,MASK(NO,NO)
      DOUBLE COMPLEX A(NC,NO),B(NC,NO),METRIC(NC,NC,NS,NB)
      DATA MASK /1/
      DATA (METRIC(I,1,1,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,2,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,3,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,4,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,5,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,6,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,7,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,8,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,9,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,10,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      SDK_DENSITY_INTER=(0D0,0D0)
      DO M=1,NO
        DO N=1,NO
          IF (MASK(M,N).EQ.0) CYCLE
          DO I=1,NC
            DO J=1,NC
              SDK_DENSITY_INTER=SDK_DENSITY_INTER+
     $         dconjg(metric(i,j,isource,ibase)*a(i,n))*b(j,m)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END

      SUBROUTINE SDK_DENSITY_DERIVATIVES(P,D,BASE,HELS,DERIV,OK)
C     Central differences of the base amplitudes in the six input
C      parameters.
C     Each shift updates the couplings once for all resonance
C      helicities.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER D,BASE(36),HELS(NEXTERNAL-1,36),IP,H
      DOUBLE PRECISION P(0:3,NEXTERNAL-1),ORIGINAL(6),STEP(6)
      DOUBLE COMPLEX DERIV(NC,NO,6,36),UP(NC,NO,36),DN(NC,NO)
      LOGICAL OK
      ORIGINAL=(/MDL_GF,MDL_MZ,MDL_MW,MDL_MT,MDL_MH,
     $           mdl_ntadpole/)
      STEP=ABS(ORIGINAL)*1D-5
      OK=MDL_MT.EQ.MDL_YMT.AND.MDL_NTADPOLE.EQ.1D0.AND.
     $   all(step.ne.0d0)
      IF (.NOT.OK) RETURN
      DO IP=1,6
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP)+STEP(IP))
        CALL COUP()
        DO H=1,D
          IF (BASE(H).NE.0)
     $      call sdk_density_eval(base(h),p,hels(1,h),up(1,1,h))
        ENDDO
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP)-STEP(IP))
        CALL COUP()
        DO H=1,D
          IF (BASE(H).EQ.0) CYCLE
          CALL SDK_DENSITY_EVAL(BASE(H),P,HELS(1,H),DN)
          DERIV(:,:,IP,H)=(UP(:,:,H)-DN)/(2D0*STEP(IP))
        ENDDO
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP))
      ENDDO
      CALL COUP()
      END

      SUBROUTINE SDK_DENSITY_SET_PARAMETER(IP,VALUE)
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER IP
      DOUBLE PRECISION VALUE
      IF (IP.EQ.1) MDL_GF=VALUE
      IF (IP.EQ.2) MDL_MZ=VALUE
      IF (IP.EQ.3) MDL_MW=VALUE
      IF (IP.EQ.4) THEN
        MDL_MT=VALUE
        MDL_YMT=VALUE
      ENDIF
      IF (IP.EQ.5) MDL_MH=VALUE
      IF (IP.EQ.6) MDL_NTADPOLE=VALUE
      END

      SUBROUTINE SDK_DENSITY_PARCOEFF(INVARIANTS,COEFF)
C     Reuse the scalar kernel's renormalisation constants. It acts
C      linearly
C     on derivatives of |M|^2 and divides by two; probe with unit
C      derivatives
C     and undo that division for derivatives of the complex amplitude.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'ewsudakov_haslo.inc'
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1,NEXTERNAL-1)
      DOUBLE COMPLEX COEFF(6),DER(AMP_SPLIT_SIZE,6)
      DOUBLE COMPLEX DER2(AMP_SPLIT_SIZE,6),ANS(AMP_SPLIT_SIZE)
      COMMON/TO_AMP_SPLIT_EWSUD_DER/DER
      COMMON/TO_AMP_SPLIT_EWSUD_DER2/DER2
      COMMON/TO_AMP_SPLIT_EWSUD/ANS
      DOUBLE COMPLEX SAVED(AMP_SPLIT_SIZE,6),SAVED2(AMP_SPLIT_SIZE,6)
      DOUBLE COMPLEX SAVEDANS(AMP_SPLIT_SIZE)
      INTEGER IP
      COEFF=(0D0,0D0)
      IF (.NOT.HAS_LO1) RETURN
      SAVED=DER
      SAVED2=DER2
      SAVEDANS=ANS
      DER2=(0D0,0D0)
      DO IP=1,6
        DER=(0D0,0D0)
        DER(LO1_POS,IP)=(1D0,0D0)
        CALL GET_PAR_REN_GMU(INVARIANTS)
        COEFF(IP)=2D0*ANS(LO1_POS)
      ENDDO
      DER=SAVED
      DER2=SAVED2
      ANS=SAVEDANS
      END

      SUBROUTINE SDK_DENSITY_RAW_1(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(2),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL FFV2_3_1(W(1),W(3),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV2_3_0(W(2),W(5),W(4),GC_68,GC_76,AMP(1))
      CALL FFV2_3_1(W(1),W(4),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 2
      CALL FFV2_3_0(W(2),W(5),W(3),GC_68,GC_76,AMP(2))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_2(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(2),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,0/),(/23,0/),(/22,0/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),ZERO,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL FFV1_1(W(1),W(3),GC_1,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV2_3_0(W(2),W(5),W(4),GC_68,GC_76,AMP(1))
      CALL FFV2_3_1(W(1),W(4),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(2),W(5),W(3),GC_1,AMP(2))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_3(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(2),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/4,0/),(/23,0/),(/22,0/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1,W(4))
      CALL FFV2_3_1(W(1),W(3),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(2),W(5),W(4),GC_1,AMP(1))
      CALL FFV1_1(W(1),W(4),GC_1,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 2
      CALL FFV2_3_0(W(2),W(5),W(3),GC_68,GC_76,AMP(2))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_4(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(3),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,2/),(/-5,5/),(/-6,6/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),MDL_MT,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),MDL_MT,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL FFS4_3(W(2),W(1),GC_116,MDL_MH,MDL_WH,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVS1_0(W(3),W(4),W(5),GC_99,AMP(1))
      CALL FFV2_5_1(W(1),W(3),GC_69,GC_76,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 2
      CALL FFV2_5_0(W(2),W(5),W(4),GC_69,GC_76,AMP(2))
      CALL FFV2_5_1(W(1),W(4),GC_69,GC_76,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_5_0(W(2),W(5),W(3),GC_69,GC_76,AMP(3))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_5(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(4),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,3/),(/-5,23/),(/-6,-24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),MDL_MT,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MW,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL FFV2P0_3(W(2),W(1),GC_124,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(3),W(5),W(4),GC_71,AMP(1))
      CALL FFS1_3(W(2),W(1),GC_39,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVS1_0(W(3),W(4),W(5),GC_85,AMP(2))
      CALL FFV2_1(W(1),W(3),GC_124,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_3_0(W(2),W(5),W(4),GC_68,GC_76,AMP(3))
      CALL FFV2_5_1(W(1),W(4),GC_69,GC_76,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 4
      CALL FFV2_0(W(2),W(5),W(3),GC_124,AMP(4))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)+(
     $ -1.000000000000000D+00)*AMP(4)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_6(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(4),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,4/),(/-5,23/),(/-6,-24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),MDL_MT,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MW,NHEL(4),+1,W(4))
      CALL FFV2P0_3(W(2),W(1),GC_124,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(4),W(5),W(3),GC_71,AMP(1))
      CALL FFS1_3(W(2),W(1),GC_39,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVS1_0(W(4),W(3),W(5),GC_85,AMP(2))
      CALL FFV2_5_1(W(1),W(3),GC_69,GC_76,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_0(W(2),W(5),W(4),GC_124,AMP(3))
      CALL FFV2_1(W(1),W(4),GC_124,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 4
      CALL FFV2_3_0(W(2),W(5),W(3),GC_68,GC_76,AMP(4))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)+(
     $ -1.000000000000000D+00)*AMP(4)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_7(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(4),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/2,3/),(/5,23/),(/6,24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),MDL_MT,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MW,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MZ,NHEL(4),+1,W(4))
      CALL FFV2P0_3(W(2),W(1),GC_124,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(5),W(3),W(4),GC_71,AMP(1))
      CALL FFS3_3(W(2),W(1),GC_30,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVS1_0(W(3),W(4),W(5),GC_84,AMP(2))
      CALL FFV2_1(W(1),W(3),GC_124,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_5_0(W(2),W(5),W(4),GC_69,GC_76,AMP(3))
      CALL FFV2_3_1(W(1),W(4),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 4
      CALL FFV2_0(W(2),W(5),W(3),GC_124,AMP(4))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)+(
     $ -1.000000000000000D+00)*AMP(4)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_8(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(4),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/2,4/),(/5,23/),(/6,24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),MDL_MT,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MZ,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MW,NHEL(4),+1,W(4))
      CALL FFV2P0_3(W(2),W(1),GC_124,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(5),W(4),W(3),GC_71,AMP(1))
      CALL FFS3_3(W(2),W(1),GC_30,MDL_MW,MDL_WW,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVS1_0(W(4),W(3),W(5),GC_84,AMP(2))
      CALL FFV2_3_1(W(1),W(3),GC_68,GC_76,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_0(W(2),W(5),W(4),GC_124,AMP(3))
      CALL FFV2_1(W(1),W(4),GC_124,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 4
      CALL FFV2_5_0(W(2),W(5),W(3),GC_69,GC_76,AMP(4))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)+(
     $ -1.000000000000000D+00)*AMP(4)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_9(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(3),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,4/),(/23,23/),(/24,-24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MW,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MW,NHEL(4),+1,W(4))
      CALL FFV1P0_3(W(2),W(1),GC_1,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(5),W(4),W(3),GC_4,AMP(1))
      CALL FFV2_3P0_3(W(2),W(1),GC_68,GC_76,MDL_MZ,MDL_WZ,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVV1_0(W(4),W(3),W(5),GC_71,AMP(2))
      CALL FFV2_1(W(1),W(3),GC_124,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_0(W(2),W(5),W(4),GC_124,AMP(3))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)
      RESULT(1:1,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_10(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      DOUBLE PRECISION PIN(0:3,NEXTERNAL-1),P(0:3,NEXTERNAL-1)
      DOUBLE PRECISION PRESH(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1),IC(NEXTERNAL-1)
      DATA IC /4*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /4*1/
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE COMPLEX IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      DOUBLE COMPLEX RESULT(NC,NO)
      DOUBLE COMPLEX AMP(3),JAMP(1,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,4/),(/23,23/),(/-24,24/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),MDL_MW,NHEL(3),+1,W(3))
      CALL VXXXXX(P(0,4),MDL_MW,NHEL(4),+1,W(4))
      CALL FFV1P0_3(W(2),W(1),GC_1,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(5),W(3),W(4),GC_4,AMP(1))
      CALL FFV2_3P0_3(W(2),W(1),GC_68,GC_76,MDL_MZ,MDL_WZ,W(5))
C     Amplitude(s) for diagram number 2
      CALL VVV1_0(W(3),W(4),W(5),GC_71,AMP(2))
      CALL FFV2_1(W(1),W(4),GC_124,MDL_MT,MDL_WT,W(5))
C     Amplitude(s) for diagram number 3
      CALL FFV2_0(W(2),W(5),W(3),GC_124,AMP(3))
C     JAMPs contributing to orders QCD=0 QED=2
      JAMP(1,1) = (-1.000000000000000D+00)*AMP(1)+(-1.000000000000000D
     $ +00)*AMP(2)+(-1.000000000000000D+00)*AMP(3)
      RESULT(1:1,1)=JAMP(:,1)
      END

