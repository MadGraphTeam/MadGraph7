C     Bound states of the process (loop_onia.py, onia_loop.f)
      INTEGER ONIA_NONIA, ONIA_NJZT, ONIA_NOTHERS, ONIA_IDEN
      PARAMETER (ONIA_NONIA=2, ONIA_NJZT=3, ONIA_NOTHERS=4,
     $  ONIA_IDEN=4)
C     physical leg of each constituent leg, and the other constituent
      INTEGER ONIA_MAPPING(6), ONIA_PARTNER(6)
C     per bound state: particle and antiparticle legs, S, J, colour
      INTEGER ONIA_Q1(2), ONIA_Q2(2), ONIA_S(2), ONIA_J(2), ONIA_C(2)
C     J_z of each bound state in each polarisation configuration
      INTEGER ONIA_JZ(2,3)
C     the constituent helicity configurations (HelConfigs.dat) and
C     the index of the helicities of their other legs
      INTEGER ONIA_HELC(6,64), ONIA_OTHERS(64)
      DATA ONIA_MAPPING/1,2,3,3,4,4/
      DATA ONIA_PARTNER/0,0,4,3,6,5/
      DATA ONIA_Q1/3,5/
      DATA ONIA_Q2/4,6/
      DATA ONIA_S/1,0/
      DATA ONIA_J/0,1/
      DATA ONIA_C/1,1/
      DATA ONIA_JZ/0,-1,0,0,0,1/
      DATA ONIA_HELC/-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1
     $ ,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1
     $ ,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,1,1,1
     $ ,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,1,-1,1,
     $ -1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1
     $ ,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1
     $ ,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1
     $ ,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,
     $ -1,1,-1,1,1,-1,1,-1,1,1,1,-1,1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1
     $ ,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1
     $ ,1,1,1,-1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,-1
     $ ,1,1,1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,
     $ -1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,-1
     $ ,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1
     $ ,1,1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,
     $ -1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1/
      DATA ONIA_OTHERS/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2
     $ ,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4
     $ ,4,4,4,4,4,4,4,4,4/
C     the colour flows of the projected Born amplitudes (ONIA_BF, one
C     basis: ONIA_BORN_FLOWS) and the colour matrix between them
      INTEGER ONIA_NFLOW
      PARAMETER (ONIA_NFLOW=1)
      INTEGER ONIA_BF_NB, ONIA_BF_NMAP
      PARAMETER (ONIA_BF_NB=1, ONIA_BF_NMAP=16)
      INTEGER ONIA_BF_NF(1), ONIA_BF_MPTR(2)
      INTEGER ONIA_BF_MA(16), ONIA_BF_MF(16)
      REAL*8 ONIA_BF_MC_RE(16), ONIA_BF_MC_IM(16)
      DATA ONIA_BF_NF/1/
      DATA ONIA_BF_MPTR/1,17/
      DATA ONIA_BF_MA/5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12/
      DATA ONIA_BF_MF/1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/
      DATA ONIA_BF_MC_RE/4.50000000000000000D+00,-5.00000000000000000D
     $ -01,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01
     $ ,4.50000000000000000D+00,-5.00000000000000000D-01/
      DATA ONIA_BF_MC_IM/0.00000000000000000D+00,0.00000000000000000D
     $ +00,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00/
      REAL*8 ONIA_FCM_RE(1,1), ONIA_FCM_IM(1,1)
      DATA ONIA_FCM_RE/1.00000000000000000D+00/
      DATA ONIA_FCM_IM/0.00000000000000000D+00/
C     some loop holds both constituents of a bound state
      LOGICAL ONIA_THRESHOLD_LOOPS
      PARAMETER (ONIA_THRESHOLD_LOOPS=.TRUE.)
C     the P-wave bound state (0 if none), and L of each bound state
      INTEGER ONIA_IPW, ONIA_L(2)
      PARAMETER (ONIA_IPW=1)
C     the second P-wave bound state (dual component 2)
      INTEGER ONIA_IPW2
      PARAMETER (ONIA_IPW2=2)
      DATA ONIA_L/1,1/
