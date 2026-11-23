C     Exact spectator sum of the linear interference, retaining all
C      colours.
C     The physical Born is kept. Longitudinal corrections are mapped
C      from the
C     Goldstone amplitudes with HELAS equivalence phases. With lrel=1
C      the
C     second output is a spectator-resolved Goldstone-normalised
C      projection
C     for comparison, and zero_gold counts the states where it is
C      undefined;
C     with lrel=0 both stay zero. Only the first nvar variants are
C      evaluated;
C     the others stay zero.

      SUBROUTINE SDK_DENSITY_INTERFERENCE(P,NRES,POS,DIMS,NVAR,LREL,
     $     born,correction,relative,zero_gold,status)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER NRES,POS(10),DIMS(10),NVAR,LREL,ZERO_GOLD,STATUS
      DOUBLE PRECISION P(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX BORN(36,36),CORRECTION(36,36,5)
      DOUBLE COMPLEX RELATIVE(36,36,5)
      INTEGER NCOMB
      PARAMETER (NCOMB=16)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),I
      DATA (NHEL(I,   1),I=1,4) /-1, 1,-1, 1/
      DATA (NHEL(I,   2),I=1,4) /-1, 1,-1,-1/
      DATA (NHEL(I,   3),I=1,4) /-1, 1, 1, 1/
      DATA (NHEL(I,   4),I=1,4) /-1, 1, 1,-1/
      DATA (NHEL(I,   5),I=1,4) /-1,-1,-1, 1/
      DATA (NHEL(I,   6),I=1,4) /-1,-1,-1,-1/
      DATA (NHEL(I,   7),I=1,4) /-1,-1, 1, 1/
      DATA (NHEL(I,   8),I=1,4) /-1,-1, 1,-1/
      DATA (NHEL(I,   9),I=1,4) / 1, 1,-1, 1/
      DATA (NHEL(I,  10),I=1,4) / 1, 1,-1,-1/
      DATA (NHEL(I,  11),I=1,4) / 1, 1, 1, 1/
      DATA (NHEL(I,  12),I=1,4) / 1, 1, 1,-1/
      DATA (NHEL(I,  13),I=1,4) / 1,-1,-1, 1/
      DATA (NHEL(I,  14),I=1,4) / 1,-1,-1,-1/
      DATA (NHEL(I,  15),I=1,4) / 1,-1, 1, 1/
      DATA (NHEL(I,  16),I=1,4) / 1,-1, 1,-1/
      INTEGER ISPEC,H,HP,R,K,D,IV,S,IP,IB,BASE(36),SOURCE_BASE(NS)
      DATA SOURCE_BASE /1,1,1/
      LOGICAL POLARISED(NEXTERNAL-1)
      DATA POLARISED /.FALSE.,.FALSE.,.FALSE.,.FALSE./
      INTEGER HELS(NEXTERNAL-1,36),IFLIST(NEXTERNAL-1)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(1)
      DATA IDEN_VALUES / 36 /
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1,NEXTERNAL-1,5),NORM(36)
      DOUBLE COMPLEX AMPS(NC,NO,NS,36),DERIV(NC,NO,6,36)
      DOUBLE COMPLEX UB(NC,NO,NS,36)
      DOUBLE COMPLEX COEFF(NS),PARCOEFF(6,5),PHASE(36)
      DOUBLE COMPLEX DELTA(36,5),MIXED(36,36,5),BROW(36,36)
      DOUBLE COMPLEX XS(36,NS),YS(NS),XD(36,6),YD(6)
      DOUBLE COMPLEX SDK_DENSITY_INTER,SDK_DENSITY_DOT,TERM
      EXTERNAL SDK_DENSITY_INTER,SDK_DENSITY_DOT
      LOGICAL DEP(6,NB),USED(NS),OK
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
      IF (LREL.LT.0.OR.LREL.GT.1) RETURN
      IF (NRES.LT.1.OR.NRES.GT.10) RETURN
      D=1
      DO R=1,NRES
        IF (DIMS(R).LT.2.OR.DIMS(R).GT.3) RETURN
        IF (POS(R).LT.1.OR.POS(R).GT.NEXTERNAL-1) RETURN
        IF (ANY(POS(1:R-1).EQ.POS(R))) RETURN
        IF (D.GT.36/DIMS(R)) RETURN
        D=D*DIMS(R)
      ENDDO
C     The helicity table restricts a polarised leg, so it cannot be
C      retained.
      STATUS=4
      IF (ANY(POLARISED(POS(1:NRES)))) RETURN
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
C     The input parameters that each base amplitude depends on.
      CALL SDK_DENSITY_DEPENDENCE(DEP,OK)
      IF (.NOT.OK) THEN
        STATUS=3
        GOTO 900
      ENDIF

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
        ENDDO
C       Every term has a Born amplitude of this tuple on its
C        conjugated side.
        IF (ALL(AMPS(:,:,1,1:D).EQ.(0D0,0D0))) CYCLE
        USED=.FALSE.
        USED(1)=.TRUE.
        DO H=1,D
          IB=BASE(H)
          IF (IB.EQ.0) CYCLE
          DO S=2,NS
            IF (SOURCE_BASE(S).NE.IB) CYCLE
            USED(S)=.TRUE.
            CALL SDK_DENSITY_EVAL(S,P,HELS(1,H),AMPS(1,1,S,H))
          ENDDO
        ENDDO
        CALL SDK_DENSITY_DERIVATIVES(P,D,BASE,HELS,DEP,DERIV)
C       The conjugated Born side of the colour contractions, once per
C        tuple.
        DO HP=1,D
          DO S=1,NS
            IF (USED(S))
     $        call sdk_density_left(1,s,amps(1,1,1,hp),ub(1,1,s,hp))
          ENDDO
        ENDDO
        DO H=1,D
          DO HP=1,D
            BROW(H,HP)=SDK_DENSITY_DOT(UB(1,1,1,HP),AMPS(1,1,1,H))
          ENDDO
          IF (LREL.EQ.0) CYCLE
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
C         nonzero only for the sources of this base, a derivative only
C          for the
C         parameters the base depends on.
          DO S=1,NS
            IF (SOURCE_BASE(S).NE.IB) CYCLE
            IF (LREL.EQ.1) YS(S)=SDK_DENSITY_INTER(IB,S,AMPS(1,1,IB,H),
     $                                             amps(1,1,s,h))
            DO HP=1,D
              XS(HP,S)=SDK_DENSITY_DOT(UB(1,1,S,HP),AMPS(1,1,S,H))
            ENDDO
          ENDDO
          DO IP=1,6
            IF (.NOT.DEP(IP,IB)) CYCLE
            IF (LREL.EQ.1) YD(IP)=SDK_DENSITY_INTER(IB,IB,
     $                               amps(1,1,ib,h),deriv(1,1,ip,h))
            DO HP=1,D
              XD(HP,IP)=SDK_DENSITY_DOT(UB(1,1,IB,HP),DERIV(1,1,IP,H))
            ENDDO
          ENDDO
          DO IV=1,NVAR
            CALL SDK_DENSITY_VARIANT(IV)
            CALL SDK_DENSITY_COEFF(IB,HELS(1,H),INVARIANTS(1,1,IV),
     $                             coeff)
            TERM=(0D0,0D0)
            DO S=1,NS
              IF (COEFF(S).EQ.(0D0,0D0)) CYCLE
              IF (LREL.EQ.1) TERM=TERM+COEFF(S)*YS(S)
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*COEFF(S)*
     $            xs(hp,s)
              ENDDO
            ENDDO
            DO IP=1,6
              IF (.NOT.DEP(IP,IB)) CYCLE
              IF (LREL.EQ.1) TERM=TERM+PARCOEFF(IP,IV)*YD(IP)
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*
     $            parcoeff(ip,iv)*xd(hp,ip)
              ENDDO
            ENDDO
            IF (LREL.EQ.1) THEN
              IF (NORM(H).GT.0D0) DELTA(H,IV)=TERM/NORM(H)
            ENDIF
          ENDDO
        ENDDO
        DO IV=1,NVAR
          DO H=1,D
            DO HP=1,D
              CORRECTION(H,HP,IV)=CORRECTION(H,HP,IV)+
     $             mixed(h,hp,iv)+dconjg(mixed(hp,h,iv))
            ENDDO
          ENDDO
        ENDDO
        IF (LREL.EQ.1) THEN
          DO IV=1,NVAR
            DO H=1,D
              DO HP=1,D
                RELATIVE(H,HP,IV)=RELATIVE(H,HP,IV)+BROW(H,HP)*
     $               (delta(h,iv)+dconjg(delta(hp,iv)))
              ENDDO
            ENDDO
          ENDDO
        ENDIF
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
      END

      SUBROUTINE SDK_DENSITY_EVAL(ISOURCE,P,HELS,JAMP)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER ISOURCE,HELS(NEXTERNAL-1)
      DOUBLE PRECISION P(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX JAMP(NC,NO)
      IF (ISOURCE.EQ.1) THEN
        CALL SDK_DENSITY_RAW_1(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.2) THEN
        CALL SDK_DENSITY_RAW_2(P,HELS,JAMP)
      ELSE IF (ISOURCE.EQ.3) THEN
        CALL SDK_DENSITY_RAW_3(P,HELS,JAMP)
      ENDIF
      END

      SUBROUTINE SDK_DENSITY_COEFF(IBASE,HELS,INVARIANTS,COEFF)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'ewsudakov_haslo.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

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
        PDGLIST=(/-1,1,6,-6/)
        COEFF(1)=GET_LSC_DIAG(PDGLIST,HELS,IFLIST,INVARIANTS)
        COEFF(1)=COEFF(1)+GET_SSC_N_DIAG(PDGLIST,HELS,IFLIST
     $   ,INVARIANTS)
        COEFF(1)=COEFF(1)+GET_XXC_DIAG(PDGLIST,HELS,IFLIST,INVARIANTS)
        COEFF(2)=GET_SSC_C(1,2,PDGLIST,-2,2,HELS,IFLIST,INVARIANTS)
        COEFF(3)=GET_SSC_C(3,4,PDGLIST,5,-5,HELS,IFLIST,INVARIANTS)
      ENDIF
      END

      SUBROUTINE SDK_DENSITY_LEFT(IBASE,ISOURCE,A,U)
C     u(j,n)=sum_i conj(metric(i,j)*a(i,n)), the conjugated side of the
C     colour contraction of base ibase with source isource.
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER IBASE,ISOURCE,I,J,N
      DOUBLE COMPLEX A(NC,NO),U(NC,NO),METRIC(NC,NC,NS,NB)
      DATA (METRIC(I,1,1,1),I=1,NC)/(9.00000000000000000D+00
     $ ,0.00000000000000000D+00),(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,2,1,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00),(9.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,2,1),I=1,NC)/(9.00000000000000000D+00
     $ ,0.00000000000000000D+00),(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,2,2,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00),(9.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,1,3,1),I=1,NC)/(9.00000000000000000D+00
     $ ,0.00000000000000000D+00),(3.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DATA (METRIC(I,2,3,1),I=1,NC)/(3.00000000000000000D+00
     $ ,0.00000000000000000D+00),(9.00000000000000000D+00
     $ ,0.00000000000000000D+00)/
      DO N=1,NO
        DO J=1,NC
          U(J,N)=DCONJG(METRIC(1,J,ISOURCE,IBASE)*A(1,N))
          DO I=2,NC
            U(J,N)=U(J,N)+DCONJG(METRIC(I,J,ISOURCE,IBASE)*A(I,N))
          ENDDO
        ENDDO
      ENDDO
      END

      DOUBLE COMPLEX FUNCTION SDK_DENSITY_DOT(U,B)
C     Contraction of a conjugated side u of sdk_density_left with b,
C      over
C     the coupling-order pairs of the target order.
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER J,M,N,MASK(NO,NO)
      DOUBLE COMPLEX U(NC,NO),B(NC,NO)
      DATA MASK /1/
      SDK_DENSITY_DOT=(0D0,0D0)
      DO M=1,NO
        DO N=1,NO
          IF (MASK(M,N).EQ.0) CYCLE
          DO J=1,NC
            SDK_DENSITY_DOT=SDK_DENSITY_DOT+U(J,N)*B(J,M)
          ENDDO
        ENDDO
      ENDDO
      END

      DOUBLE COMPLEX FUNCTION SDK_DENSITY_INTER(IBASE,ISOURCE,A,B)
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER IBASE,ISOURCE
      DOUBLE COMPLEX A(NC,NO),B(NC,NO),U(NC,NO)
      DOUBLE COMPLEX SDK_DENSITY_DOT
      EXTERNAL SDK_DENSITY_DOT
      CALL SDK_DENSITY_LEFT(IBASE,ISOURCE,A,U)
      SDK_DENSITY_INTER=SDK_DENSITY_DOT(U,B)
      END

      SUBROUTINE SDK_DENSITY_DEPENDENCE(DEP,OK)
C     dep(ip,b): the amplitudes of base b depend on input parameter
C      ip. A
C     shift that leaves its couplings, masses and widths bitwise
C      unchanged
C     leaves its amplitudes unchanged, so that derivative is exactly
C      zero.
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER NV
      PARAMETER (NV=2)
      LOGICAL DEP(6,NB),OK
      INTEGER IP,B,K
      DOUBLE PRECISION ORIGINAL(6),STEP(6)
      DOUBLE COMPLEX CENTRAL(NV,NB),SHIFTED(NV)
      ORIGINAL=(/MDL_GF,MDL_MZ,MDL_MW,MDL_MT,MDL_MH,
     $           mdl_ntadpole/)
      STEP=ABS(ORIGINAL)*1D-5
      OK=MDL_MT.EQ.MDL_YMT.AND.MDL_NTADPOLE.EQ.1D0.AND.
     $   all(step.ne.0d0)
      IF (.NOT.OK) RETURN
      DO B=1,NB
        CALL SDK_DENSITY_INPUTS(B,CENTRAL(1,B))
      ENDDO
      DEP=.FALSE.
      DO IP=1,6
        DO K=-1,1,2
          CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP)+K*STEP(IP))
          CALL COUP()
          DO B=1,NB
            CALL SDK_DENSITY_INPUTS(B,SHIFTED)
            IF (ANY(SHIFTED.NE.CENTRAL(:,B))) DEP(IP,B)=.TRUE.
          ENDDO
        ENDDO
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP))
      ENDDO
      CALL COUP()
      END

      SUBROUTINE SDK_DENSITY_INPUTS(IBASE,VALUES)
C     The couplings, masses and widths that the amplitudes of base
C      ibase use.
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INTEGER IBASE
      DOUBLE COMPLEX VALUES(2)
      VALUES=(0D0,0D0)
      IF (IBASE.EQ.1) THEN
        VALUES(1)=MDL_MT
        VALUES(2)=GC_11
      ENDIF
      END

      SUBROUTINE SDK_DENSITY_DERIVATIVES(P,D,BASE,HELS,DEP,DERIV)
C     Central differences of the base amplitudes in the input
C      parameters
C     they depend on. Each shift updates the couplings once for all
C     resonance helicities.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

      INTEGER D,BASE(36),HELS(NEXTERNAL-1,36),IP,H
      DOUBLE PRECISION P(0:3,NEXTERNAL-1),ORIGINAL(6),STEP(6)
      DOUBLE COMPLEX DERIV(NC,NO,6,36),UP(NC,NO,36),DN(NC,NO)
      LOGICAL DEP(6,NB),NEED
      ORIGINAL=(/MDL_GF,MDL_MZ,MDL_MW,MDL_MT,MDL_MH,
     $           mdl_ntadpole/)
      STEP=ABS(ORIGINAL)*1D-5
      DO IP=1,6
        NEED=.FALSE.
        DO H=1,D
          IF (BASE(H).NE.0) NEED=NEED.OR.DEP(IP,BASE(H))
        ENDDO
        IF (.NOT.NEED) CYCLE
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP)+STEP(IP))
        CALL COUP()
        DO H=1,D
          IF (BASE(H).EQ.0) CYCLE
          IF (DEP(IP,BASE(H)))
     $      call sdk_density_eval(base(h),p,hels(1,h),up(1,1,h))
        ENDDO
        CALL SDK_DENSITY_SET_PARAMETER(IP,ORIGINAL(IP)-STEP(IP))
        CALL COUP()
        DO H=1,D
          IF (BASE(H).EQ.0) CYCLE
          IF (.NOT.DEP(IP,BASE(H))) CYCLE
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
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

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
      DOUBLE COMPLEX AMP(1),JAMP(2,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      JAMP=(0D0,0D0)
      P=PIN
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1, FLAVOR(3),W(3))
      CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1, FLAVOR(4),W(4))
      CALL FFV1P0_3(W(2),W(1),GC_11,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(4),W(3),W(5),GC_11,AMP(1))
C     JAMPs contributing to orders QCD=2 QED=0
      JAMP(1,1) = (-1.666666666666667D-01)*AMP(1)
      JAMP(2,1) = (5.000000000000000D-01)*AMP(1)
      RESULT(1:2,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_2(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

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
      DOUBLE COMPLEX AMP(1),JAMP(2,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      JAMP=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,2/),(/-1,1/),(/-2,2/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1, FLAVOR(3),W(3))
      CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1, FLAVOR(4),W(4))
      CALL FFV1P0_3(W(2),W(1),GC_11,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(4),W(3),W(5),GC_11,AMP(1))
C     JAMPs contributing to orders QCD=2 QED=0
      JAMP(1,1) = (-1.666666666666667D-01)*AMP(1)
      JAMP(2,1) = (5.000000000000000D-01)*AMP(1)
      RESULT(1:2,1)=JAMP(:,1)
      END

      SUBROUTINE SDK_DENSITY_RAW_3(PIN,NHEL,RESULT)
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=2,NO=1,NS=3,NB=1)

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
      DOUBLE COMPLEX AMP(1),JAMP(2,1),TMP_JAMP(0)
      TYPE(ALOHA) W(5)
      LOGICAL OK
      RESULT=(0D0,0D0)
      JAMP=(0D0,0D0)
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,4/),(/6,-6/),(/5,-5/),OK)
      IF (.NOT.OK) RETURN
      P=PRESH
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1, FLAVOR(1),W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1, FLAVOR(3),W(3))
      CALL IXXXXX(P(0,4),ZERO,NHEL(4),-1, FLAVOR(4),W(4))
      CALL FFV1P0_3(W(2),W(1),GC_11,ZERO,ZERO,W(5))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(4),W(3),W(5),GC_11,AMP(1))
C     JAMPs contributing to orders QCD=2 QED=0
      JAMP(1,1) = (-1.666666666666667D-01)*AMP(1)
      JAMP(2,1) = (5.000000000000000D-01)*AMP(1)
      RESULT(1:2,1)=JAMP(:,1)
      END

