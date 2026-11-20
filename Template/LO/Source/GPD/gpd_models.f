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
C       4  toy3  smooth, all four GPDs non-zero (i0-extrapolated references)
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
      GPD_TRANGE = 2D0
      GPD_BMAP = 4D0
      GPD_SURVIVAL = 1D0
      GPD_BEAM = 2
      USEMC = 0
      GPD_NMASK = 0
      GPD_NUCL = 0
      GPD_TA = 1
      GPD_TZ = 1
      GPD_ISO = 0
      GPD_INCOH = 0
      GPD_TRANGEA = 0.5D0
      GPD_BMAPA = -1D0
      GPD_NPDF = ''
      GPD_TNAME = 'p'
      CALL GPD_READ_CARD()
      GPD_INITIALISED = 1
C     a nucleus as the target of a standalone output (gpd_target): probed
C     coherently; madevent sets the target of each beam (GPD_SET_TARGET)
      IF (GPD_TNAME.NE.'p' .AND. GPD_TNAME.NE.'P'
     &    .AND. GPD_TNAME.NE.'') THEN
        CALL GPD_TARGET_BYNAME(GPD_TNAME)
        GPD_NUCL = 1
        WRITE(*,*) 'GPD: target ', TRIM(GPD_TNAME), ' (A, Z =',
     &    GPD_TA, GPD_TZ, '), coherent, |M|^2 of the nucleus'
      ENDIF
C     the x quadrature cannot do better than ~1e-10 in double precision
C     (cancellations between graphs near the poles)
      IF (GPD_XTOL.LT.1D-10) THEN
        WRITE(*,*) 'GPD: gpd_xtol', GPD_XTOL, ' raised to 1e-10'
        GPD_XTOL = 1D-10
      ENDIF
      WRITE(*,'(A,I2,A,I2,A,I2,A,2F8.4)') ' GPD: model', GPD_MODEL,
     &  '  x integration (0 quadrature, 1 mc)', GPD_XMODE,
     &  '  t model', GPD_TMODEL, '  slopes g, q', GPD_TSLOPEG,
     &  GPD_TSLOPEQ
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
          IF (INDEX(VAL,'toy3').GT.0) THEN
            GPD_MODEL = 4
          ELSE IF (INDEX(VAL,'toy2').GT.0) THEN
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
        ELSE IF (NAME.EQ.'gpd_t_range') THEN
          READ(VAL,*) GPD_TRANGE
        ELSE IF (NAME.EQ.'gpd_t_bmap') THEN
          READ(VAL,*) GPD_BMAP
        ELSE IF (NAME.EQ.'gpd_survival') THEN
          READ(VAL,*) GPD_SURVIVAL
        ELSE IF (NAME.EQ.'gpd_target') THEN
          GPD_TNAME = VAL
        ELSE IF (NAME.EQ.'gpd_nuclear') THEN
          IF (INDEX(VAL,'incoh').GT.0) THEN
            GPD_INCOH = 1
          ELSE
            GPD_INCOH = 0
          ENDIF
        ELSE IF (NAME.EQ.'gpd_npdf') THEN
          GPD_NPDF = VAL
        ELSE IF (NAME.EQ.'gpd_t_range_A') THEN
          READ(VAL,*) GPD_TRANGEA
        ELSE IF (NAME.EQ.'gpd_t_bmap_A') THEN
          READ(VAL,*) GPD_BMAPA
        ENDIF
      ENDDO
      CLOSE(LUN)
      END

      SUBROUTINE GPD_EVAL(IPART, X, XI, T, MU, G)
C     The GPDs of the target (GPD_NUCL): a nucleon (per nucleon of a
C     nucleus, with isospin), times F_A(t) and without E, Ht, Et for a
C     nucleus probed coherently (spin 0). X, XI per nucleon.
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART, ISW
      DOUBLE PRECISION X, XI, T, MU, G(4), GS(4), WP, WN
      DOUBLE PRECISION GPD_FORMFACTOR
      CALL GPD_INIT()
      CALL GPD_EVAL_NUCLEON(IPART, X, XI, T, MU, G)
      IF (GPD_NUCL.EQ.0) RETURN
C     isospin: u <-> d between proton and neutron (the dd model takes it
C     from the per-nucleon PDFs, Source/GPD/gpd_dd.f)
      IF (GPD_MODEL.NE.1 .AND. (IPART.EQ.1 .OR. IPART.EQ.2)) THEN
        IF (GPD_ISO.EQ.1) THEN
          WP = 1D0
          WN = 0D0
        ELSE IF (GPD_ISO.EQ.-1) THEN
          WP = 0D0
          WN = 1D0
        ELSE
          WP = DBLE(GPD_TZ)/GPD_TA
          WN = 1D0 - WP
        ENDIF
        IF (WN.GT.0D0) THEN
          ISW = 3 - IPART
          CALL GPD_EVAL_NUCLEON(ISW, X, XI, T, MU, GS)
          G = WP*G + WN*GS
        ENDIF
      ENDIF
      IF (GPD_NUCL.EQ.1) THEN
        G(1) = G(1)*GPD_FORMFACTOR(T)
        G(2) = 0D0
        G(3) = 0D0
        G(4) = 0D0
      ENDIF
      END

      SUBROUTINE GPD_EVAL_NUCLEON(IPART, X, XI, T, MU, G)
C     The GPDs of a proton (the models of GPD_MODEL).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IPART
      DOUBLE PRECISION X, XI, T, MU, G(4)
      DOUBLE PRECISION FT
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
      ELSE IF (GPD_MODEL.EQ.4) THEN
C       toy3: smooth (no kink at -+xi), all four non-zero, H and E even,
C       Ht and Et odd in x: references by i0 regularisation and extrapolation
        G(1) = (1D0 - X*X)**2*(1D0 + 0.5D0*X*X)
        G(2) = 0.3D0*(1D0 - X*X)**2
        G(3) = X*(1D0 - X*X)**2
        G(4) = 0.2D0*X*(1D0 - X*X)**2
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


C ----------------------------------------------------------------------
C     Nuclear targets: the Woods-Saxon (3pF) data of gamma-UPC
C     (OpticalGlauber_Geometry.f90, De Vries et al.), rho(r) ~
C     (1 + w r^2/R^2)/(1 + exp((r - R)/a)), and the form factor
C       F_A(q) = (4 pi/(q A)) int dr r sin(q r) rho(r),  q = sqrt(-t).
      SUBROUTINE GPD_WS_TABLE(K, NAME, IA, IZ, R, AA, W)
      IMPLICIT NONE
      INTEGER K, IA, IZ, NT
      PARAMETER (NT=42)
      CHARACTER*(*) NAME
      DOUBLE PRECISION R, AA, W
      CHARACTER*5 NAMES(NT)
      INTEGER TA(NT), TZ(NT)
      DOUBLE PRECISION TR(NT), TAA(NT), TW(NT)
      DATA NAMES/'H2   ','Li7  ','Be9  ','B10  ','B11  ','C13  ',
     &  'C14  ','N14  ','N15  ','O16  ','Ne20 ','Mg24 ','Mg25 ','Al27 ',
     &  'Si28 ','Si29 ','Si30 ','P31  ','Cl35 ','Cl37 ','Ar40 ','K39  ',
     &  'Ca40 ','Ca48 ','Fe56 ','Ni58 ','Ni60 ','Ni61 ','Ni62 ','Ni64 ',
     &  'Cu63 ','Kr78 ','Ag110','Sb122','Xe129','Xe132','Nd142','Er166',
     &  'W186 ','Au197','Pb207','Pb208'/
      DATA TA/2,7,9,10,11,13,14,14,15,16,20,24,25,27,28,29,30,31,35,37,
     &  40,39,40,48,56,58,60,61,62,64,63,78,110,122,129,132,142,166,186,
     &  197,207,208/
      DATA TZ/1,3,4,5,5,6,6,7,7,8,10,12,12,13,14,14,14,15,17,17,18,19,
     &  20,20,26,28,28,28,28,28,29,36,47,51,54,54,60,68,74,79,82,82/
      DATA TR/0.01D0,1.77D0,1.791D0,1.71D0,1.69D0,1.635D0,1.73D0,
     &  2.570D0,2.334D0,2.608D0,2.791D0,3.108D0,3.22D0,3.07D0,3.340D0,
     &  3.338D0,3.338D0,3.369D0,3.476D0,3.554D0,3.766D0,3.743D0,
     &  3.766D0,3.7369D0,4.074D0,4.3092D0,4.4891D0,4.4024D0,4.4425D0,
     &  4.5211D0,4.214D0,4.5D0,5.33D0,5.32D0,5.36D0,5.4D0,5.6135D0,
     &  5.98D0,6.58D0,6.38D0,6.62D0,6.624D0/
      DATA TAA/0.5882D0,0.327D0,0.611D0,0.837D0,0.811D0,1.403D0,1.38D0,
     &  0.5052D0,0.498D0,0.513D0,0.698D0,0.607D0,0.58D0,0.519D0,0.580D0,
     &  0.547D0,0.547D0,0.582D0,0.599D0,0.588D0,0.586D0,0.595D0,0.586D0,
     &  0.5245D0,0.536D0,0.5169D0,0.5369D0,0.5401D0,0.5386D0,0.5278D0,
     &  0.586D0,0.5D0,0.535D0,0.57D0,0.59D0,0.61D0,0.5868D0,0.446D0,
     &  0.480D0,0.535D0,0.546D0,0.549D0/
      DATA TW/0D0,0D0,0D0,0D0,0D0,0D0,0D0,-0.180D0,0.139D0,-0.051D0,
     &  -0.168D0,-0.163D0,-0.236D0,0D0,-0.233D0,-0.203D0,-0.203D0,
     &  -0.173D0,-0.10D0,-0.13D0,-0.161D0,-0.201D0,-0.161D0,-0.030D0,
     &  0D0,-0.1308D0,-0.2668D0,-0.1983D0,-0.2090D0,-0.2284D0,0D0,0D0,
     &  0D0,0D0,0D0,0D0,0.096D0,0.19D0,0D0,0D0,0D0,0D0/
C     K = 0 on input: look up NAME, or (IA, IZ) if NAME is blank; K is the
C     row found (0 if none)
      DO K = 1, NT
        IF (LEN_TRIM(NAME).GT.0) THEN
          IF (TRIM(NAMES(K)).EQ.TRIM(NAME)) GOTO 10
        ELSE
          IF (TA(K).EQ.IA .AND. TZ(K).EQ.IZ) GOTO 10
        ENDIF
      ENDDO
      K = 0
      RETURN
 10   IA = TA(K)
      IZ = TZ(K)
      R = TR(K)
      AA = TAA(K)
      W = TW(K)
      END

      SUBROUTINE GPD_SET_TARGET(IA, IZ)
C     The target nucleus (A, Z) of the beam probed by the GPDs (madevent).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER IA, IZ, K, JA, JZ
      CHARACTER*1 BLANK
      IF (IA.EQ.GPD_TA .AND. IZ.EQ.GPD_TZ) RETURN
      GPD_TA = IA
      GPD_TZ = IZ
      IF (IA.LE.1) RETURN
      BLANK = ' '
      JA = IA
      JZ = IZ
      CALL GPD_WS_TABLE(K, BLANK, JA, JZ, GPD_WSR, GPD_WSAA, GPD_WSW)
      IF (K.EQ.0) THEN
        WRITE(*,*) 'GPD: no Woods-Saxon data for the nucleus A, Z =',
     &    IA, IZ, ' (Source/GPD/gpd_models.f, GPD_WS_TABLE)'
        STOP 1
      ENDIF
      END

      SUBROUTINE GPD_TARGET_BYNAME(NAME)
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      CHARACTER*(*) NAME
      INTEGER K, IA, IZ
      IA = 0
      IZ = 0
      CALL GPD_WS_TABLE(K, NAME, IA, IZ, GPD_WSR, GPD_WSAA, GPD_WSW)
      IF (K.EQ.0) THEN
        WRITE(*,*) 'GPD: unknown target ', TRIM(NAME), ' (gpd_target:',
     &    ' p or a nucleus such as Pb208, Au197, Xe129, Ca40, O16)'
        STOP 1
      ENDIF
      GPD_TA = IA
      GPD_TZ = IZ
      END

      DOUBLE PRECISION FUNCTION GPD_WS_MOMENT(NPOW, Q)
C     int_0^Rmax dr r^NPOW rho(r) (NPOW >= 0, sin(q r)/(q r) kernel for
C     Q > 0 with NPOW = 2), unnormalised rho, Gauss-Legendre panels.
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER NPOW, IP, K, NP
      DOUBLE PRECISION Q, R, H, F, RMAX, XG(16), WG(16), RHO
      LOGICAL INIT
      DATA INIT/.TRUE./
      SAVE INIT, XG, WG
      IF (INIT) THEN
        CALL GPD_GL16(XG, WG)
        INIT = .FALSE.
      ENDIF
      RMAX = GPD_WSR + 30D0*GPD_WSAA
      NP = 60
      H = RMAX/NP
      F = 0D0
      DO IP = 1, NP
        DO K = 1, 16
          R = H*(IP - 0.5D0 + 0.5D0*XG(K))
          RHO = (1D0 + GPD_WSW*R*R/GPD_WSR**2)
     &        /(1D0 + EXP(MIN((R - GPD_WSR)/GPD_WSAA, 700D0)))
          IF (Q.GT.0D0) THEN
            F = F + 0.5D0*H*WG(K)*R*SIN(Q*R)/Q*RHO
          ELSE
            F = F + 0.5D0*H*WG(K)*R**NPOW*RHO
          ENDIF
        ENDDO
      ENDDO
      GPD_WS_MOMENT = F
      END

      DOUBLE PRECISION FUNCTION GPD_FORMFACTOR(T)
C     F_A(sqrt(-t)) of the target nucleus (1 at t = 0), cached in t.
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      DOUBLE PRECISION T, Q, GPD_WS_MOMENT, TLAST, FLAST, NORM
      DOUBLE PRECISION RLAST, ALAST, WLAST
      PARAMETER (Q=0D0)
      DOUBLE PRECISION QF
      DATA TLAST/1D99/, RLAST/-1D0/, ALAST/-1D0/, WLAST/-1D0/
      SAVE TLAST, FLAST, NORM, RLAST, ALAST, WLAST
      IF (GPD_WSR.NE.RLAST .OR. GPD_WSAA.NE.ALAST
     &    .OR. GPD_WSW.NE.WLAST) THEN
        NORM = GPD_WS_MOMENT(2, Q)
        RLAST = GPD_WSR
        ALAST = GPD_WSAA
        WLAST = GPD_WSW
        TLAST = 1D99
      ENDIF
      IF (T.NE.TLAST) THEN
C       q in fm^-1 (hbar c = 0.1973 GeV fm, as gamma-UPC)
        QF = SQRT(MAX(-T, 0D0))/0.1973D0
        IF (QF.LT.1D-9) THEN
          FLAST = 1D0
        ELSE
          FLAST = GPD_WS_MOMENT(1, QF)/NORM
        ENDIF
        TLAST = T
      ENDIF
      GPD_FORMFACTOR = FLAST
      END

      DOUBLE PRECISION FUNCTION GPD_WS_R2()
C     <r^2> of the target nucleus (GeV^-2)
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      DOUBLE PRECISION GPD_WS_MOMENT, Q
      PARAMETER (Q=0D0)
      GPD_WS_R2 = GPD_WS_MOMENT(4, Q)/GPD_WS_MOMENT(2, Q)/0.1973D0**2
      END

      SUBROUTINE GPD_GL16(XG, WG)
C     16-point Gauss-Legendre nodes and weights on [-1, 1]
      IMPLICIT NONE
      DOUBLE PRECISION XG(16), WG(16), Z, Z1, P1, P2, P3, PP, PI
      INTEGER I, J
      PARAMETER (PI=3.14159265358979323846D0)
      DO I = 1, 8
        Z = COS(PI*(I - 0.25D0)/16.5D0)
 20     P1 = 1D0
        P2 = 0D0
        DO J = 1, 16
          P3 = P2
          P2 = P1
          P1 = ((2D0*J - 1D0)*Z*P2 - (J - 1D0)*P3)/J
        ENDDO
        PP = 16D0*(Z*P1 - P2)/(Z*Z - 1D0)
        Z1 = Z
        Z = Z1 - P1/PP
        IF (ABS(Z - Z1).GT.1D-15) GOTO 20
        XG(I) = -Z
        XG(17 - I) = Z
        WG(I) = 2D0/((1D0 - Z*Z)*PP*PP)
        WG(17 - I) = WG(I)
      ENDDO
      END
