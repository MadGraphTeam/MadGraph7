C     EW Sudakov corrections per resonance-helicity configuration h for
C     the spin-density reweighting
C     B^EWSL[h,h'] = B[h,h'] * (1 + delta_h + delta*_h'),
C     with the amplitude-level delta_h averaged over spectator
C      helicities
C     (exact for h = h' only). The density matrix B[h,h'] is computed
C     separately (GET_DENSITY of the standalone density-matrix output).

      SUBROUTINE DENSITY_MATRIX_WRAPPER(P, NRES, RES_INDICES, RES_DIMS,
     $     density_born, density_delta, born_diag)
C     -----------------------------------------------------------------
C     ---------
C     Arguments:
C     p(0:3, nexternal-1)  - external momenta (restored on exit)
C     nres                 - number of resonances
C     res_indices(nres)    - resonance leg positions (1-based)
C     res_dims(nres)       - spin dimension per resonance (2=fermion,
C      3=vector)
C     density_born(*)      - output: zeros, D*(D+1)/2 entries
C     density_delta(*)     - output: sum of ME*delta, D entries
C     born_diag(*)         - output: Born, D entries
C     D = product of res_dims. density_delta and born_diag are summed
C      over
C     spectator helicities and both carry 1/IDEN;
C     delta_h = density_delta(h)/born_diag(h). ME is the Born, or the
C     Goldstone-boson ME for configurations with longitudinal vector
C      legs;
C     for those delta_h includes the ratio of the Goldstone ME to the
C      Born.
C     -----------------------------------------------------------------
C     ---------
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'

C     Arguments
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)
      INTEGER NRES
      INTEGER RES_INDICES(NRES)
      INTEGER RES_DIMS(NRES)
      DOUBLE COMPLEX DENSITY_BORN(*)
      DOUBLE COMPLEX DENSITY_DELTA(*)
      DOUBLE PRECISION BORN_DIAG(*)

C     Local variables
      INTEGER I, K, N
      INTEGER IH
      INTEGER TOTAL_DIM
      INTEGER H_INDICES(10)  ! Individual resonance hel indices (0-based)
      INTEGER HEL_VALUES(10)  ! Actual helicity values (-1, 0, +1)

      INTEGER ALLOW_HEL(300)  ! D*nres entries, size checked by the caller

      DOUBLE COMPLEX DELTA_TOTAL_THIS_HEL
      DOUBLE PRECISION P_SAVE(0:3, NEXTERNAL-1)
      INTEGER ILEG, IMU
      LOGICAL SAVED_FILTER_HEL

      DOUBLE COMPLEX CZERO
      PARAMETER (CZERO = (0D0, 0D0))

C     Results of sudakov_wrapper
      INCLUDE 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_LSC/AMP_SPLIT_EWSUD_LSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_SSC/AMP_SPLIT_EWSUD_SSC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_XXC/AMP_SPLIT_EWSUD_XXC
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE PRECISION AMP_BORN_SUMHEL
      COMMON /TO_AMP_BORN_SUMHEL/AMP_BORN_SUMHEL

C     Resonance-helicity selection for sudakov_wrapper
      INTEGER EWSUD_NRES
      INTEGER EWSUD_RES_POS(10)
      INTEGER EWSUD_RES_HEL(10)
      COMMON/TO_EWSUD_RES_SEL/EWSUD_NRES, EWSUD_RES_POS, EWSUD_RES_HEL

      LOGICAL SUD_FILTER_HEL
      COMMON /TO_FILTER_HEL/ SUD_FILTER_HEL

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN


C     -----------------------------------------------------------------
C     ---------
C     BEGIN CODE
C     -----------------------------------------------------------------
C     ---------

      TOTAL_DIM = 1
      DO I = 1, NRES
        TOTAL_DIM = TOTAL_DIM * RES_DIMS(I)
      ENDDO

      DO I = 1, TOTAL_DIM * (TOTAL_DIM + 1) / 2
        DENSITY_BORN(I) = CZERO
      ENDDO
      DO I = 1, TOTAL_DIM
        DENSITY_DELTA(I) = CZERO
        BORN_DIAG(I) = 0D0
      ENDDO

C     allow_hel((ih-1)*nres+n) = helicity of resonance n in
C      configuration
C     ih = 1..D, with ih-1 = h_1 + res_dims(1)*(h_2 +
C      res_dims(2)*(...))
C     for 0-based indices h_n (first resonance runs fastest).

      DO IH = 1, TOTAL_DIM
        K = IH - 1
        DO N = 1, NRES
          H_INDICES(N) = MOD(K, RES_DIMS(N))
          K = K / RES_DIMS(N)
        ENDDO

        DO N = 1, NRES
          IF (RES_DIMS(N) .EQ. 2) THEN
C           Fermion: 0 -> -1, 1 -> +1
            HEL_VALUES(N) = 2 * H_INDICES(N) - 1
          ELSE IF (RES_DIMS(N) .EQ. 3) THEN
C           Vector: 0 -> -1, 1 -> 0, 2 -> +1
            HEL_VALUES(N) = H_INDICES(N) - 1
          ELSE
            HEL_VALUES(N) = H_INDICES(N)
          ENDIF
        ENDDO

        DO N = 1, NRES
          ALLOW_HEL((IH-1)*NRES + N) = HEL_VALUES(N)
        ENDDO
      ENDDO

C     For each configuration ih, sudakov_wrapper restricted to the
C     resonance helicities allow_hel(ih) sums over spectator
C      helicities.

      SAVED_FILTER_HEL = SUD_FILTER_HEL
      SUD_FILTER_HEL = .FALSE.

C     Save momenta (PAR numerical derivatives may modify p)
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P_SAVE(IMU, ILEG) = P(IMU, ILEG)
        ENDDO
      ENDDO

      DO IH = 1, TOTAL_DIM
        DO ILEG = 1, NEXTERNAL-1
          DO IMU = 0, 3
            P(IMU, ILEG) = P_SAVE(IMU, ILEG)
          ENDDO
        ENDDO

        EWSUD_NRES = NRES
        DO N = 1, NRES
          EWSUD_RES_POS(N) = RES_INDICES(N)
          EWSUD_RES_HEL(N) = ALLOW_HEL((IH-1)*NRES + N)
        ENDDO

        CALCULATEDBORN = .FALSE.

        CALL SUDAKOV_WRAPPER(P)

        DELTA_TOTAL_THIS_HEL = AMP_SPLIT_EWSUD_LSC(1)
     $       + amp_split_ewsud_ssc(1)
     $       + amp_split_ewsud_xxc(1)
     $       + amp_split_ewsud_par(1)

        DENSITY_DELTA(IH) = DELTA_TOTAL_THIS_HEL
        BORN_DIAG(IH) = AMP_BORN_SUMHEL
      ENDDO

      EWSUD_NRES = 0

      SUD_FILTER_HEL = SAVED_FILTER_HEL
      CALCULATEDBORN = .FALSE.
      DO ILEG = 1, NEXTERNAL-1
        DO IMU = 0, 3
          P(IMU, ILEG) = P_SAVE(IMU, ILEG)
        ENDDO
      ENDDO

      RETURN
      END

C     Exact spectator sum of the linear interference, retaining all
C      colours.
C     The physical Born is kept. Longitudinal corrections are mapped
C      from the
C     Goldstone amplitudes with HELAS equivalence phases. The second
C      output is
C     a spectator-resolved Goldstone-normalised projection for
C      comparison.

      SUBROUTINE SDK_DENSITY_INTERFERENCE(P,NRES,POS,DIMS,BORN,
     $     correction,relative,zero_gold,status)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER NRES,POS(10),DIMS(10),ZERO_GOLD,STATUS
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
      INTEGER ISPEC,H,HP,R,K,D,IV,S,IP,IB,BASE(36)
      INTEGER HELS(NEXTERNAL-1,36),IFLIST(NEXTERNAL-1)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(1)
      DATA IDEN_VALUES / 72 /
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1,NEXTERNAL-1,5),NORM(36)
      DOUBLE COMPLEX AMPS(NC,NO,NS,36),DERIV(NC,NO,6,36)
      DOUBLE COMPLEX COEFF(NS),PARCOEFF(6,5),PHASE(36)
      DOUBLE COMPLEX DELTA(36,5),MIXED(36,36,5),BROW(36,36)
      DOUBLE COMPLEX SDK_DENSITY_INTER,TERM
      EXTERNAL SDK_DENSITY_INTER
      LOGICAL OK,SELECTED
      INTEGER SUD_MOD,SAVED_MOD
      COMMON/TO_SUD_MOD/SUD_MOD
      LOGICAL S_TO_RIJ,RIJ_GE_MW,SAVED_STOR,SAVED_RIJ
      COMMON/TO_S_TO_RIJ/S_TO_RIJ
      COMMON/RIJ_GE_MW/RIJ_GE_MW

      BORN=(0D0,0D0)
      CORRECTION=(0D0,0D0)
      RELATIVE=(0D0,0D0)
      ZERO_GOLD=0
      STATUS=0
C     The underlying Sudakov charge/reshuffling functions do not
C      implement
C     massive vector beams. Do not return their undefined numerical
C      results.
      IF (.NOT..TRUE.) THEN
        STATUS=4
        RETURN
      ENDIF
      D=1
      IF (NRES.LT.1.OR.NRES.GT.10) THEN
        STATUS=1
        RETURN
      ENDIF
      DO R=1,NRES
        IF (DIMS(R).LT.2.OR.DIMS(R).GT.3.OR.POS(R).LT.1.OR.
     $      pos(r).gt.nexternal-1) then
        STATUS=1
        RETURN
        ENDIF
        DO K=1,R-1
          IF (POS(K).EQ.POS(R)) THEN
            STATUS=1
            RETURN
          ENDIF
        ENDDO
        IF (D.GT.36/DIMS(R)) THEN
          STATUS=1
          RETURN
        ENDIF
        D=D*DIMS(R)
      ENDDO
      SAVED_MOD=SUD_MOD
      SAVED_STOR=S_TO_RIJ
      SAVED_RIJ=RIJ_GE_MW
      IFLIST=(/-1,-1,1,1/)
C     Each variant builds its invariants with its own floor.
      DO IV=1,5
        CALL SDK_DENSITY_VARIANT(IV)
        CALL SDK_GET_INVARIANTS(P,IFLIST,INVARIANTS(1,1,IV))
        CALL SDK_DENSITY_PARCOEFF(INVARIANTS(1,1,IV),PARCOEFF(1,IV))
      ENDDO

C     One representative of each spectator tuple. No good-helicity
C      filtering:
C     a zero Born component can have a nonzero mixed-helicity
C      interference.
      DO ISPEC=1,NCOMB
        SELECTED=.TRUE.
        DO R=1,NRES
          IF (NHEL(POS(R),ISPEC).NE.-1) SELECTED=.FALSE.
        ENDDO
        IF (.NOT.SELECTED) CYCLE
        AMPS=(0D0,0D0)
        DERIV=(0D0,0D0)
        MIXED=(0D0,0D0)
        DELTA=(0D0,0D0)
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
          CALL SDK_DENSITY_EVAL(1,P,HELS(1,H),AMPS(1,1,1,H),OK)
          IF (.NOT.OK) THEN
            STATUS=2
            GOTO 900
          ENDIF
          IB=BASE(H)
          IF (IB.EQ.0) CYCLE
          DO S=2,NS
            CALL SDK_DENSITY_NEEDED(IB,S,SELECTED)
            IF (.NOT.SELECTED) CYCLE
            CALL SDK_DENSITY_EVAL(S,P,HELS(1,H),AMPS(1,1,S,H),OK)
            IF (.NOT.OK) THEN
              STATUS=2
              GOTO 900
            ENDIF
          ENDDO
          CALL SDK_DENSITY_DERIVATIVES(IB,P,HELS(1,H),
     $         deriv(1,1,1,h),ok)
          IF (.NOT.OK) THEN
            STATUS=3
            GOTO 900
          ENDIF
        ENDDO
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
        DO IV=1,5
          CALL SDK_DENSITY_VARIANT(IV)
          DO H=1,D
            IB=BASE(H)
            IF (IB.EQ.0) CYCLE
            CALL SDK_DENSITY_COEFF(IB,HELS(1,H),INVARIANTS(1,1,IV),
     $                             coeff)
            TERM=(0D0,0D0)
            DO S=1,NS
              IF (COEFF(S).EQ.(0D0,0D0)) CYCLE
              TERM=TERM+COEFF(S)*SDK_DENSITY_INTER(IB,S,
     $                    amps(1,1,ib,h),amps(1,1,s,h))
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*COEFF(S)*
     $              sdk_density_inter(1,s,amps(1,1,1,hp),
     $                                    amps(1,1,s,h))
              ENDDO
            ENDDO
            DO IP=1,6
              TERM=TERM+PARCOEFF(IP,IV)*SDK_DENSITY_INTER(IB,IB,
     $                        amps(1,1,ib,h),deriv(1,1,ip,h))
              DO HP=1,D
                MIXED(H,HP,IV)=MIXED(H,HP,IV)+PHASE(H)*
     $             parcoeff(ip,iv)*sdk_density_inter(1,ib,
     $                          amps(1,1,1,hp),deriv(1,1,ip,h))
              ENDDO
            ENDDO
            IF (NORM(H).GT.0D0) DELTA(H,IV)=TERM/NORM(H)
          ENDDO
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
      BORN=BORN/DBLE(IDEN_VALUES(NFKSPROCESS))
      CORRECTION=CORRECTION/DBLE(IDEN_VALUES(NFKSPROCESS))
      RELATIVE=RELATIVE/DBLE(IDEN_VALUES(NFKSPROCESS))
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

      SUBROUTINE SDK_DENSITY_NEEDED(IBASE,ISOURCE,NEEDED)
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER IBASE,ISOURCE,SOURCE_BASE(NS)
      LOGICAL NEEDED
      DATA SOURCE_BASE /1,1,1,1,1,1,1,1,1,1/
      NEEDED=SOURCE_BASE(ISOURCE).EQ.IBASE
      END

      SUBROUTINE SDK_DENSITY_EVAL(ISOURCE,P,HELS,JAMP,OK)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER ISOURCE,HELS(NEXTERNAL-1)
      DOUBLE PRECISION P(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX JAMP(NC,NO)
      LOGICAL OK
      JAMP=(0D0,0D0)
      OK=.FALSE.
      IF (ISOURCE.EQ.1) THEN
        CALL SDK_DENSITY_RAW_1(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.2) THEN
        CALL SDK_DENSITY_RAW_2(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.3) THEN
        CALL SDK_DENSITY_RAW_3(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.4) THEN
        CALL SDK_DENSITY_RAW_4(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.5) THEN
        CALL SDK_DENSITY_RAW_5(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.6) THEN
        CALL SDK_DENSITY_RAW_6(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.7) THEN
        CALL SDK_DENSITY_RAW_7(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.8) THEN
        CALL SDK_DENSITY_RAW_8(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.9) THEN
        CALL SDK_DENSITY_RAW_9(P,HELS,JAMP,OK)
      ELSE IF (ISOURCE.EQ.10) THEN
        CALL SDK_DENSITY_RAW_10(P,HELS,JAMP,OK)
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

      SUBROUTINE SDK_DENSITY_DERIVATIVES(IBASE,P,HELS,DERIV,OK)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER IBASE,HELS(NEXTERNAL-1),IP
      DOUBLE PRECISION P(0:3,NEXTERNAL-1),VALUE,STEP,ORIGINAL
      DOUBLE COMPLEX DERIV(NC,NO,6),UP(NC,NO),DN(NC,NO)
      LOGICAL OK,OKUP,OKDN
      DERIV=(0D0,0D0)
      OK=.FALSE.
      IF (MDL_MT.NE.MDL_YMT.OR.MDL_NTADPOLE.NE.1D0) RETURN
      DO IP=1,6
        CALL SDK_DENSITY_PARAMETER(IP,VALUE,.FALSE.)
        ORIGINAL=VALUE
        STEP=ABS(VALUE)*1D-5
        IF (STEP.EQ.0D0) RETURN
        VALUE=VALUE+STEP
        CALL SDK_DENSITY_PARAMETER(IP,VALUE,.TRUE.)
        CALL COUP()
        CALL SDK_DENSITY_EVAL(IBASE,P,HELS,UP,OKUP)
        VALUE=ORIGINAL-STEP
        CALL SDK_DENSITY_PARAMETER(IP,VALUE,.TRUE.)
        CALL COUP()
        CALL SDK_DENSITY_EVAL(IBASE,P,HELS,DN,OKDN)
        VALUE=ORIGINAL
        CALL SDK_DENSITY_PARAMETER(IP,VALUE,.TRUE.)
        CALL COUP()
        IF (.NOT.OKUP.OR..NOT.OKDN) RETURN
        DERIV(:,:,IP)=(UP-DN)/(2D0*STEP)
      ENDDO
      OK=.TRUE.
      END

      SUBROUTINE SDK_DENSITY_PARAMETER(IP,VALUE,WRITE_VALUE)
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'
      INTEGER IP
      DOUBLE PRECISION VALUE,PARAMS(6)
      LOGICAL WRITE_VALUE
      IF (WRITE_VALUE) THEN
        IF (IP.EQ.1) MDL_GF=VALUE
        IF (IP.EQ.2) MDL_MZ=VALUE
        IF (IP.EQ.3) MDL_MW=VALUE
        IF (IP.EQ.4) THEN
          MDL_MT=VALUE
          MDL_YMT=VALUE
        ENDIF
        IF (IP.EQ.5) MDL_MH=VALUE
        IF (IP.EQ.6) MDL_NTADPOLE=VALUE
      ELSE
        PARAMS=(/MDL_GF,MDL_MZ,MDL_MW,MDL_MT,MDL_MH,
     $            mdl_ntadpole/)
        VALUE=PARAMS(IP)
      ENDIF
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

      SUBROUTINE SDK_DENSITY_DIMENSIONS(NCOLOR,NORDER,NSOURCE,NBASE)
      IMPLICIT NONE
      INTEGER NC,NO,NS,NB
      PARAMETER (NC=1,NO=1,NS=10,NB=1)

      INTEGER NCOLOR,NORDER,NSOURCE,NBASE
      NCOLOR=NC
      NORDER=NO
      NSOURCE=NS
      NBASE=NB
      END

      SUBROUTINE SDK_DENSITY_RAW_1(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
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

      SUBROUTINE SDK_DENSITY_RAW_2(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,0/),(/23,0/),(/22,0/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_3(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/4,0/),(/23,0/),(/22,0/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_4(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,2/),(/-5,5/),(/-6,6/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_5(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,3/),(/-5,23/),(/-6,-24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_6(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/1,4/),(/-5,23/),(/-6,-24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_7(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/2,3/),(/5,23/),(/6,24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_8(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/2,4/),(/5,23/),(/6,24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_9(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,4/),(/23,23/),(/24,-24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

      SUBROUTINE SDK_DENSITY_RAW_10(PIN,NHEL,RESULT,OK)
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
      OK=.TRUE.
      P=PIN
      CALL RESHUFFLE_MOMENTA(P,PRESH,(/3,4/),(/23,23/),(/-24,24/),OK)
      IF (.NOT.OK) THEN
        OK=.TRUE.
        RETURN
      ENDIF
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

