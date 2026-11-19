C     Settings of the GPD library (generalised parton distributions of the
C     probed proton), filled by GPD_INIT from the run card (madevent) or
C     from Cards/gpd_card.dat (standalone).
C
C     GPD_MODEL : 0 toy (analytic, tests), 1 dd (double-distribution Ansatz
C                 on the forward PDFs, madevent only), 2 grid (user table)
C     GPD_BG, GPD_BS, GPD_BV : DD profile powers b of gluon, sea, valence
C     GPD_TMODEL: 0 no t dependence, 1 exp(B t), 2 Regge exp(B t) |beta|^(-alpha' t)
C     GPD_TSLOPEG, GPD_TSLOPEQ : B (GeV^-2) of gluons and quarks
C     GPD_ALPHAPG, GPD_ALPHAPQ : alpha' (GeV^-2) of gluons and quarks
C     GPD_XMODE : 0 per-point adaptive quadrature in x, 1 Monte Carlo (two replicas)
C     GPD_XTOL  : relative tolerance of the quadrature
C     GPD_EPSRES: step of the numerical residues (relative to the pole spacing)
C     GPD_DPTOL : relative tolerance of the double-pole detector
C     GPD_GRIDFILE : file of the grid model
      INTEGER GPD_MODEL, GPD_TMODEL, GPD_XMODE, GPD_INITIALISED
      DOUBLE PRECISION GPD_BG, GPD_BS, GPD_BV
      DOUBLE PRECISION GPD_TSLOPEG, GPD_TSLOPEQ, GPD_ALPHAPG, GPD_ALPHAPQ
      DOUBLE PRECISION GPD_XTOL, GPD_EPSRES, GPD_DPTOL
      CHARACTER*256 GPD_GRIDFILE
      COMMON/GPD_SETTINGS/GPD_BG, GPD_BS, GPD_BV, GPD_TSLOPEG, GPD_TSLOPEQ,
     &     GPD_ALPHAPG, GPD_ALPHAPQ, GPD_XTOL, GPD_EPSRES, GPD_DPTOL,
     &     GPD_MODEL, GPD_TMODEL, GPD_XMODE, GPD_INITIALISED, GPD_GRIDFILE
C     The two Monte Carlo replicas x, x' (madevent, gpd_x_integration = mc):
C     USEMC = 1 when RMC1, RMC2 in (0,1) are set for the current event.
      DOUBLE PRECISION RMC1, RMC2
      INTEGER USEMC
      COMMON/GPD_XMC_POINT/RMC1, RMC2, USEMC
C     Residues per pole group: while GPD_NMASK > 0, the kernel keeps only the
C     diagrams of the x-carrying propagators GPD_MASKIDS(1:GPD_NMASK) of the
C     process (gpd_driver.f): coincident poles with opposite i0 signs (e.g.
C     gamma q -> gamma gamma q) have separate residues.
      INTEGER GPD_NMASK, GPD_MASKIDS(400)
      COMMON/GPD_MASK/GPD_NMASK, GPD_MASKIDS
