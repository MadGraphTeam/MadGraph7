C     Bound states of the process (loop_onia.py, onia_loop.f)
      INTEGER ONIA_NONIA, ONIA_NJZT, ONIA_NOTHERS, ONIA_IDEN
      PARAMETER (ONIA_NONIA=1, ONIA_NJZT=1, ONIA_NOTHERS=4,
     $  ONIA_IDEN=4)
C     physical leg of each constituent leg, and the other constituent
      INTEGER ONIA_MAPPING(4), ONIA_PARTNER(4)
C     per bound state: particle and antiparticle legs, S, J, colour
      INTEGER ONIA_Q1(1), ONIA_Q2(1), ONIA_S(1), ONIA_J(1), ONIA_C(1)
C     J_z of each bound state in each polarisation configuration
      INTEGER ONIA_JZ(1,1)
C     the constituent helicity configurations (HelConfigs.dat) and
C     the index of the helicities of their other legs
      INTEGER ONIA_HELC(4,16), ONIA_OTHERS(16)
      DATA ONIA_MAPPING/1,2,3,3/
      DATA ONIA_PARTNER/0,0,4,3/
      DATA ONIA_Q1/3/
      DATA ONIA_Q2/4/
      DATA ONIA_S/0/
      DATA ONIA_J/0/
      DATA ONIA_C/1/
      DATA ONIA_JZ/0/
      DATA ONIA_HELC/-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,
     $ -1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,1
     $ ,1,1,-1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,-1/
      DATA ONIA_OTHERS/1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4/
C     the colour flows of the projected Born amplitudes (ONIA_BF, one
C     basis: ONIA_BORN_FLOWS) and the colour matrix between them
      INTEGER ONIA_NFLOW
      PARAMETER (ONIA_NFLOW=1)
      INTEGER ONIA_BF_NB, ONIA_BF_NMAP
      PARAMETER (ONIA_BF_NB=1, ONIA_BF_NMAP=2)
      INTEGER ONIA_BF_NF(1), ONIA_BF_MPTR(2)
      INTEGER ONIA_BF_MA(2), ONIA_BF_MF(2)
      REAL*8 ONIA_BF_MC_RE(2), ONIA_BF_MC_IM(2)
      DATA ONIA_BF_NF/1/
      DATA ONIA_BF_MPTR/1,3/
      DATA ONIA_BF_MA/1,2/
      DATA ONIA_BF_MF/1,1/
      DATA ONIA_BF_MC_RE/-3.00000000000000000D+00,
     $ -3.00000000000000000D+00/
      DATA ONIA_BF_MC_IM/0.00000000000000000D+00,0.00000000000000000D
     $ +00/
      REAL*8 ONIA_FCM_RE(1,1), ONIA_FCM_IM(1,1)
      DATA ONIA_FCM_RE/1.00000000000000000D+00/
      DATA ONIA_FCM_IM/0.00000000000000000D+00/
C     some loop holds both constituents of a bound state
      LOGICAL ONIA_THRESHOLD_LOOPS
      PARAMETER (ONIA_THRESHOLD_LOOPS=.TRUE.)
C     the P-wave bound state (0 if none), and L of each bound state
      INTEGER ONIA_IPW, ONIA_L(1)
      PARAMETER (ONIA_IPW=0)
      DATA ONIA_L/0/
