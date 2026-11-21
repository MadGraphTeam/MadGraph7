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
C     colour factors between the projected Born amplitudes
      REAL*8 ONIA_CFB_RE(2,2), ONIA_CFB_IM(2,2)
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
      DATA ONIA_CFB_RE/9.00000000000000000D+00,9.00000000000000000D+00
     $ ,9.00000000000000000D+00,9.00000000000000000D+00/
      DATA ONIA_CFB_IM/0.00000000000000000D+00,0.00000000000000000D+00
     $ ,0.00000000000000000D+00,0.00000000000000000D+00/
