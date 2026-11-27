      SUBROUTINE %(proc_prefix)sGET_DENSITY_IDX(P, POS, N_CHANGING,
     & ALLOW_HEL, N_COMB, FLAV_IDX, ALPHAS, MU_R2, INTER_SUM)
C     GET_DENSITY for an extended FLAV_IDX = K*NFLAV + flav,
C     summed over the chosen squared split orders as
C     GET_DENSITY sums them. Same contract as the
C     GET_DENSITY_IDX of the split-order-free template: P and
C     POS follow the caller's legs (the slot order of the
C     process FLAV_IDX names, GET_PDG_FOR_FLAVOR) and the
C     crossing is applied ONCE here, to the momenta, the NSF
C     flags and the whole NHEL table (APPLY_CROSSING_TABLE).
C     Only written with crossing on: it is how the f2py PDG
C     dispatch evaluates a folded crossed subprocess.
      use model_object
      implicit none
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NB_NHEL
      PARAMETER (NB_NHEL=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(nincoming)d)
C     ARGUMENTS
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE PRECISION ALPHAS, MU_R2
      DOUBLE COMPLEX INTER_SUM(*)
C     LOCAL
      INTEGER I, J, IHEL, IPART, JJ, NINTER
      INTEGER THISNHEL(NEXTERNAL)
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:,:)
      DOUBLE PRECISION PI, POLFACT, RESCALE
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER NHELUSE(NEXTERNAL,NB_NHEL)
      INTEGER IC(NEXTERNAL), ICUSE(NEXTERNAL)
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL), CPOS(NEXTERNAL)
      INTEGER FLAV_USE, DUMFLAV
C     GLOBAL
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
      DOUBLE PRECISION BEAMPOL(2)
      COMMON/TO_BEAMPOL/BEAMPOL
      INTEGER NHEL(NEXTERNAL,NB_NHEL)
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
      include 'coupl.inc'

      NINTER = N_COMB*(N_COMB+1)/2
      DO I = 1, NINTER
        INTER_SUM(I) = (0d0, 0d0)
      ENDDO
      IF (ALPHAS.NE.0d0) THEN
        PI = 3.141592653589793d0
        G = 2*DSQRT(ALPHAS*PI)
        CALL UPDATE_AS_PARAM()
      ENDIF
C     Unresolved flavor or impossible crossing: zero, as
C     GET_DENSITY_IDX does.
      IF (FLAV_IDX.LT.1) RETURN
      CALL %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE, RESCALE)
      IF (RESCALE.EQ.0D0) RETURN
C     A crossed flavor with a polarised beam: refused, as in
C     SMATRIX (the beam weight would be read off a base slot).
      IF (FLAV_IDX.NE.FLAV_USE.AND.NINITIAL.EQ.2.AND.
     &   (ABS(BEAMPOL(1)).GT.1D0.OR.ABS(BEAMPOL(2)).GT.1D0)) THEN
        WRITE(*,*) 'ERROR: beam polarisation is not supported for a',
     &    ' crossed flavor index (FLAV_IDX=',FLAV_IDX,')'
        STOP 1
      ENDIF
      CALL %(proc_prefix)sFILL_NHEL()
      IC(:) = 1
      CALL %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX, NB_NHEL, P,
     & NHEL, IC, PUSE, NHELUSE, ICUSE, DUMFLAV)
C     Base slot K holds caller leg PERM(K): the base slot of
C     POS is the K with PERM(K)=POS.
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX, PERM, SGN, DUMFLAV)
      DO IPART = 1, N_CHANGING
        DO I = 1, NEXTERNAL
          IF (PERM(I).EQ.POS(IPART)) CPOS(IPART) = I
        ENDDO
      ENDDO
      ALLOCATE(TMP_INTER(NSQAMPSO, NINTER))
      DO IHEL = 1, NB_NHEL
        THISNHEL(:) = NHELUSE(:, IHEL)
        DO IPART = 1, N_CHANGING
          IF (THISNHEL(CPOS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10
        ENDDO
        TMP_INTER(:,:) = (0d0, 0d0)
        CALL %(proc_prefix)sGET_ALL_INTER_SO_CROSSED(PUSE, THISNHEL,
     &   ICUSE, CPOS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_USE, RESCALE,
     &   TMP_INTER)
C       Polarised beams: as GET_DENSITY (uncrossed only, a
C       crossed flavor was refused above).
        POLFACT = 1D0
        IF (NINITIAL.EQ.2) THEN
          DO JJ = 1, NINITIAL
            IF (ABS(BEAMPOL(JJ)).LE.1D0) CYCLE
            IF (THISNHEL(JJ).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
              POLFACT = POLFACT*ABS(BEAMPOL(JJ))
            ELSE
              POLFACT = POLFACT*(2D0-ABS(BEAMPOL(JJ)))
            ENDIF
          ENDDO
        ENDIF
        DO J = 1, NSQAMPSO
          IF (CHOSEN_SO_CONFIGS(J)) THEN
            DO I = 1, NINTER
              INTER_SUM(I) = INTER_SUM(I) + POLFACT*TMP_INTER(J,I)
            ENDDO
          ENDIF
        ENDDO
 10   ENDDO
      DEALLOCATE(TMP_INTER)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_ALL_INTER_SUMSO_IDX(P, NHEL, POS,
     & N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, INTER)
C     GET_ALL_INTER_SUMSO for an extended FLAV_IDX: the
C     interference terms of one helicity configuration of the
C     legs not in POS, summed over the chosen squared split
C     orders. P, POS and NHEL all follow the caller's legs
C     (see GET_DENSITY_IDX); base slot XK takes NHEL(PERM(XK)),
C     with no sign, as GET_ALL_INTER_IDX of the
C     split-order-free template. NHEL is not written to.
      implicit none
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C     ARGUMENTS
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE COMPLEX INTER(*)
C     LOCAL
      INTEGER I, J, IPART, NINTER
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:,:)
      DOUBLE PRECISION RESCALE
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER NHELUSE(NEXTERNAL)
      INTEGER IC(NEXTERNAL), ICUSE(NEXTERNAL)
      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL), CPOS(NEXTERNAL)
      INTEGER FLAV_USE, DUMFLAV
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS

      NINTER = N_COMB*(N_COMB+1)/2
      DO I = 1, NINTER
        INTER(I) = (0d0, 0d0)
      ENDDO
      IF (FLAV_IDX.LT.1) RETURN
      CALL %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE, RESCALE)
      IF (RESCALE.EQ.0D0) RETURN
      IC(:) = 1
      CALL %(proc_prefix)sAPPLY_CROSSING(FLAV_IDX, P, NHEL, IC, PUSE,
     & NHELUSE, ICUSE, DUMFLAV)
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX, PERM, SGN, DUMFLAV)
      DO I = 1, NEXTERNAL
        NHELUSE(I) = NHEL(PERM(I))
      ENDDO
      DO IPART = 1, N_CHANGING
        DO I = 1, NEXTERNAL
          IF (PERM(I).EQ.POS(IPART)) CPOS(IPART) = I
        ENDDO
      ENDDO
      ALLOCATE(TMP_INTER(NSQAMPSO, NINTER))
      TMP_INTER(:,:) = (0d0, 0d0)
      CALL %(proc_prefix)sGET_ALL_INTER_SO_CROSSED(PUSE, NHELUSE, ICUSE,
     & CPOS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_USE, RESCALE,
     & TMP_INTER)
      DO I = 1, NINTER
        DO J = 1, NSQAMPSO
          IF (CHOSEN_SO_CONFIGS(J)) INTER(I) = INTER(I) + TMP_INTER(J,I)
        ENDDO
      ENDDO
      DEALLOCATE(TMP_INTER)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_ALL_INTER_SO_CROSSED(P, NHEL, IC,
     & POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAV_IDX, RESCALE, INTER)
C     Inner worker of the two routines above, the split-order
C     GET_ALL_INTER on arrays that are ALREADY crossed: P, NHEL
C     and IC in the base slots, POS in the base slot numbering,
C     FLAV_IDX reduced to [1,NFLAV] and RESCALE the flavor /
C     crossing dependent normalisation (GET_INTER_RESCALE).
C     INTER(NSQAMPSO,*) holds every squared split order. NHEL
C     is overwritten at the POS slots.
      implicit none
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
C     ARGUMENTS
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAV_IDX
      DOUBLE PRECISION RESCALE
      DOUBLE COMPLEX INTER(NSQAMPSO,*)
C     LOCAL
      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX, ALLOCATABLE :: JAMP(:,:,:)
      INTEGER I, J, N, SOL

      ALLOCATE(JAMP(NCOLOR, NAMPSO, N_COMB))
      DO I = 1, N_COMB
        DO N = 1, N_CHANGING
          NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
        ENDDO
        CALL %(proc_prefix)sGET_AMP(P, NHEL, IC, FLAV_IDX, AMP)
        CALL %(proc_prefix)sGET_JAMP(AMP, JAMP(1,1,I))
      ENDDO
      SOL = 0
      DO I = 1, N_COMB
        DO J = I, N_COMB
          SOL = SOL + 1
          CALL %(proc_prefix)sGET_INTER(JAMP(1,1,I), JAMP(1,1,J),
     &     INTER(1,SOL))
          INTER(:,SOL) = INTER(:,SOL)*RESCALE
        ENDDO
      ENDDO
      DEALLOCATE(JAMP)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_INTER_RESCALE(FLAV_IDX, FLAV_USE,
     & RESCALE)
C     Split an extended FLAV_IDX and return the factor by which
C     GET_INTER's output must be multiplied (GET_INTER only
C     sees JAMPs and normalises with the bare static IDEN), as
C     in the split-order-free template. RESCALE=0 marks a
C     crossing that cannot be applied (zero matrix element).
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER FLAV_IDX, FLAV_USE
      DOUBLE PRECISION RESCALE
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sBROKEN_SYM
%(inter_rescale_decl)s

      FLAV_USE = MOD(FLAV_IDX-1, NFLAV) + 1
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_USE, FLAVOR)
%(inter_rescale_body)s

      RETURN
      END
