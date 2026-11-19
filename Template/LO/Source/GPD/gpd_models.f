C ======================================================================
C     GPD models of the probed proton.
C
C     GPD_EVAL(IPART, X, XI, T, MU, G): G = (H, E, Ht, Et) of parton IPART
C     (21 gluon, 1..5 quark flavours) at momentum fraction X in (-1,1),
C     skewness XI, momentum transfer T (GeV^2, <= 0) and scale MU (GeV).
C     Conventions of Diehl, Phys. Rept. 388 (2003): H^g(x,0,0) = x g(x),
C     H^q(x,0,0) = q(x) for x > 0 and -qbar(-x) for x < 0; H^g, E^g even
C     and Ht^g, Et^g odd in x. The quark channel gpd_q carries both the
C     quark (x > -xi) and the antiquark (x < xi) of flavour q.
C
C     Models (GPD_MODEL):
C       0  toy   analytic functions for the tests (not physics)
C       3  toy2  H = 1 - x^2, E = Ht = Et = 0 (closed-form convolutions)
C       1  dd    Radyushkin double distributions on the forward PDFs at MU
C                (Source/GPD/gpd_dd.f; madevent only)
C       2  grid  table read from GPD_GRIDFILE (placeholder for user GPDs,
C                e.g. exported from PARTONS), see GPD_GRID_READ
C     t dependence (GPD_TMODEL): 0 none, 1 exp(B t) at amplitude level,
C       2 Regge exp(B t) |x|^(-alpha' t) (exact inside the dd model,
C       evaluated at |beta| -> max(|x|,xi) for the other models).
C ======================================================================
      SUBROUTINE GPD_INIT()
C     Defaults, then the standalone card Cards/gpd_card.dat if present
C     (madevent sets GPD_INITIALISED from the run card before any call).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      IF (GPD_INITIALISED.EQ.1) RETURN
      GPD_MODEL = 0
      GPD_BG = 2D0
      GPD_BS = 2D0
      GPD_BV = 1D0
      GPD_TMODEL = 0
      GPD_TSLOPEG = 0D0
      GPD_TSLOPEQ = 0D0
      GPD_ALPHAPG = 0D0
      GPD_ALPHAPQ = 0D0
      GPD_XMODE = 0
      GPD_XTOL = 1D-8
      GPD_EPSRES = 5D-2
      GPD_DPTOL = 1D-3
      GPD_GRIDFILE = ''
      GPD_BEAM = 2
      USEMC = 0
      GPD_NMASK = 0
      CALL GPD_READ_CARD()
      GPD_INITIALISED = 1
      END

      SUBROUTINE GPD_READ_CARD()
C     'value = name' lines of gpd_card.dat (standalone outputs).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      CHARACTER*256 LINE, VAL, NAME
      INTEGER IEQ, IOS, LUN
      LOGICAL EX
      DATA LUN/87/
      INQUIRE(FILE='../../Cards/gpd_card.dat', EXIST=EX)
      IF (EX) THEN
        OPEN(LUN, FILE='../../Cards/gpd_card.dat', STATUS='OLD')
      ELSE
        INQUIRE(FILE='../../../Cards/gpd_card.dat', EXIST=EX)
        IF (EX) THEN
          OPEN(LUN, FILE='../../../Cards/gpd_card.dat', STATUS='OLD')
        ELSE
          INQUIRE(FILE='gpd_card.dat', EXIST=EX)
          IF (.NOT.EX) RETURN
          OPEN(LUN, FILE='gpd_card.dat', STATUS='OLD')
        ENDIF
      ENDIF
      DO
        READ(LUN, '(A)', IOSTAT=IOS) LINE
        IF (IOS.NE.0) EXIT
        IF (INDEX(LINE,'#').GT.0) LINE = LINE(1:INDEX(LINE,'#')-1)
        IF (INDEX(LINE,'!').GT.0) LINE = LINE(1:INDEX(LINE,'!')-1)
        IEQ = INDEX(LINE, '=')
        IF (IEQ.EQ.0) CYCLE
        VAL = ADJUSTL(LINE(1:IEQ-1))
        NAME = ADJUSTL(LINE(IEQ+1:))
        IF (VAL(1:1).EQ."'") VAL = VAL(2:INDEX(VAL(2:),"'"))
        IF (NAME.EQ.'gpd_model') THEN
          IF (INDEX(VAL,'toy2').GT.0) THEN
            GPD_MODEL = 3
          ELSE IF (INDEX(VAL,'toy').GT.0) THEN
            GPD_MODEL = 0
          ELSE IF (INDEX(VAL,'dd').GT.0) THEN
            GPD_MODEL = 1
          ELSE IF (INDEX(VAL,'grid').GT.0) THEN
            GPD_MODEL = 2
          ENDIF
        ELSE IF (NAME.EQ.'gpd_grid') THEN
          GPD_GRIDFILE = VAL
        ELSE IF (NAME.EQ.'gpd_tmodel') THEN
          READ(VAL,*) GPD_TMODEL
        ELSE IF (NAME.EQ.'gpd_tslope_g') THEN
          READ(VAL,*) GPD_TSLOPEG
        ELSE IF (NAME.EQ.'gpd_tslope_q') THEN
          READ(VAL,*) GPD_TSLOPEQ
        ELSE IF (NAME.EQ.'gpd_alphap_g') THEN
          READ(VAL,*) GPD_ALPHAPG
        ELSE IF (NAME.EQ.'gpd_alphap_q') THEN
          READ(VAL,*) GPD_ALPHAPQ
        ELSE IF (NAME.EQ.'gpd_b_g') THEN
          READ(VAL,*) GPD_BG
        ELSE IF (NAME.EQ.'gpd_b_sea') THEN
          READ(VAL,*) GPD_BS
        ELSE IF (NAME.EQ.'gpd_b_val') THEN
          READ(VAL,*) GPD_BV
        ELSE IF (NAME.EQ.'gpd_x_integration') THEN
          IF (INDEX(VAL,'mc').GT.0) THEN
            GPD_XMODE = 1
          ELSE
            GPD_XMODE = 0
          ENDIF
        ELSE IF (NAME.EQ.'gpd_xtol') THEN
          READ(VAL,*) GPD_XTOL
        ELSE IF (NAME.EQ.'gpd_epsres') THEN
          READ(VAL,*) GPD_EPSRES
        ELSE IF (NAME.EQ.'gpd_dptol') THEN
          READ(VAL,*) GPD_DPTOL
        ENDIF
      ENDDO
      CLOSE(LUN)
      END

      SUBROUTINE GPD_EVAL(IPART, X, XI, T, MU, G)
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, MU, G(4)
      DOUBLE PRECISION FT
      CALL GPD_INIT()
      G = 0D0
      IF (ABS(X).GE.1D0) RETURN
      IF (GPD_MODEL.EQ.0) THEN
        CALL GPD_TOY(IPART, X, XI, G)
      ELSE IF (GPD_MODEL.EQ.1) THEN
        CALL GPD_DD(IPART, X, XI, T, MU, G)
        RETURN
      ELSE IF (GPD_MODEL.EQ.2) THEN
        CALL GPD_GRID(IPART, X, XI, MU, G)
      ELSE IF (GPD_MODEL.EQ.3) THEN
C       toy2: H = 1 - x^2 for every parton, E = Ht = Et = 0 (closed forms)
        G(1) = 1D0 - X*X
      ENDIF
      CALL GPD_TFACTOR(IPART, X, XI, T, FT)
      G = G*FT
      END

      SUBROUTINE GPD_TFACTOR(IPART, X, XI, T, FT)
C     t dependence of the toy and grid models (the dd model has its own).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, FT, B, AP, XB
      IF (IPART.EQ.21) THEN
        B = GPD_TSLOPEG
        AP = GPD_ALPHAPG
      ELSE
        B = GPD_TSLOPEQ
        AP = GPD_ALPHAPQ
      ENDIF
      FT = 1D0
      IF (GPD_TMODEL.EQ.1) THEN
        FT = EXP(B*T)
      ELSE IF (GPD_TMODEL.EQ.2) THEN
        XB = MAX(ABS(X), XI, 1D-6)
        FT = EXP(B*T)*XB**(-AP*T)
      ENDIF
      END

      SUBROUTINE GPD_TOY(IPART, X, XI, G)
C     Analytic toy GPDs with the symmetries of the true ones, a kink at
C     x = +-xi (as double distributions have) and support |x| < 1.
      IMPLICIT NONE
      INTEGER IPART
      DOUBLE PRECISION X, XI, G(4), S, K
      S = (1D0-X*X)**2
      K = ABS(X*X-XI*XI)
      IF (IPART.EQ.21) THEN
        G(1) = S*(1D0 + K)/(1D0+XI)
        G(2) = 0.3D0*G(1)
        G(3) = X*S
        G(4) = 0.2D0*X*S
      ELSE
        G(1) = (1D0-X*X)*(1D0+0.5D0*X)*(1D0 + K)
        G(2) = 0.4D0*G(1)
        G(3) = 0.6D0*(1D0-X*X)*(1D0+0.2D0*X)
        G(4) = (1D0-X*X)/(XI+0.1D0)
      ENDIF
      END

C ----------------------------------------------------------------------
C     The grid model. File format (plain text, '#' starts a comment):
C       nx nxi
C       x(1) ... x(nx)            ascending, in (-1,1)
C       xi(1) ... xi(nxi)         ascending, in (0,1)
C       then blocks:  pdg func  followed by nx*nxi values, x running fastest
C       (pdg 21 or 1..5, func H E Ht Et), at the scale and t = 0 of the
C       table; the t dependence is GPD_TMODEL's. Bilinear interpolation.
      MODULE GPD_GRID_MOD
      IMPLICIT NONE
      INTEGER NX, NXI, LOADED
      DOUBLE PRECISION, ALLOCATABLE :: XG(:), XIG(:), TAB(:,:,:,:)
      DATA LOADED/0/
      END MODULE GPD_GRID_MOD

      SUBROUTINE GPD_GRID_READ()
      USE GPD_GRID_MOD
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER LUN, IOS, IPDG, IFN, I, J
      CHARACTER*8 FNAME
      DATA LUN/88/
      IF (LOADED.EQ.1) RETURN
      OPEN(LUN, FILE=GPD_GRIDFILE, STATUS='OLD', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
        WRITE(*,*) 'GPD: cannot open the grid ', TRIM(GPD_GRIDFILE)
        STOP 1
      ENDIF
      CALL GPD_SKIPCOMMENTS(LUN)
      READ(LUN,*) NX, NXI
      ALLOCATE(XG(NX), XIG(NXI), TAB(NX,NXI,4,0:6))
      TAB = 0D0
      CALL GPD_SKIPCOMMENTS(LUN)
      READ(LUN,*) (XG(I), I=1,NX)
      CALL GPD_SKIPCOMMENTS(LUN)
      READ(LUN,*) (XIG(I), I=1,NXI)
      DO
        CALL GPD_SKIPCOMMENTS(LUN)
        READ(LUN, *, IOSTAT=IOS) IPDG, FNAME
        IF (IOS.NE.0) EXIT
        IF (FNAME.EQ.'H') THEN
          IFN = 1
        ELSE IF (FNAME.EQ.'E') THEN
          IFN = 2
        ELSE IF (FNAME.EQ.'Ht') THEN
          IFN = 3
        ELSE
          IFN = 4
        ENDIF
        IF (IPDG.EQ.21) IPDG = 0
        READ(LUN,*) ((TAB(I,J,IFN,IPDG), I=1,NX), J=1,NXI)
      ENDDO
      CLOSE(LUN)
      LOADED = 1
      END

      SUBROUTINE GPD_SKIPCOMMENTS(LUN)
      IMPLICIT NONE
      INTEGER LUN, IOS
      CHARACTER*512 LINE
      DO
        READ(LUN,'(A)',IOSTAT=IOS) LINE
        IF (IOS.NE.0) RETURN
        LINE = ADJUSTL(LINE)
        IF (LINE(1:1).NE.'#' .AND. LEN_TRIM(LINE).GT.0) THEN
          BACKSPACE(LUN)
          RETURN
        ENDIF
      ENDDO
      END

      SUBROUTINE GPD_GRID(IPART, X, XI, MU, G)
      USE GPD_GRID_MOD
      IMPLICIT NONE
      INTEGER IPART, IP, I, J, K
      DOUBLE PRECISION X, XI, MU, G(4), U, V
      CALL GPD_GRID_READ()
      IP = IPART
      IF (IPART.EQ.21) IP = 0
      G = 0D0
      IF (X.LE.XG(1) .OR. X.GE.XG(NX)) RETURN
      I = 1
      DO WHILE (I.LT.NX-1 .AND. XG(I+1).LT.X)
        I = I + 1
      ENDDO
      J = 1
      DO WHILE (J.LT.NXI-1 .AND. XIG(J+1).LT.XI)
        J = J + 1
      ENDDO
      U = (X - XG(I))/(XG(I+1)-XG(I))
      V = (XI - XIG(J))/(XIG(J+1)-XIG(J))
      V = MAX(0D0, MIN(1D0, V))
      DO K = 1, 4
        G(K) = (1D0-U)*(1D0-V)*TAB(I,J,K,IP) + U*(1D0-V)*TAB(I+1,J,K,IP)
     &       + (1D0-U)*V*TAB(I,J+1,K,IP) + U*V*TAB(I+1,J+1,K,IP)
      ENDDO
      END
