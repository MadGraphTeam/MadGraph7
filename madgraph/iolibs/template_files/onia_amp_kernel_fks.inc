
      SUBROUTINE %(kname)s(P,NHEL,AMP_OUT)
C
%(info_lines)s
C
C Bound states at NLO (FKS, arXiv:2402.19221; madgraph/fks/fks_onia.py):
C the projected amplitudes AMP_OUT(NGRAPHS) of the diagrams, for the
C physical momenta P and helicities NHEL (one entry per bound state, its
C J_z). The constituents of an onium of momentum K have k_i = m_i/(m1+m2) K
C (q = 0, no reshuffling at NLO: the onium mass is m1+m2). Each diagram is
C contracted with the spin projector over the constituent helicities and
C carries the norm of the colour projector (1/sqrt(3) or sqrt(2)); the
C colour projection itself is in the colour basis. The LDME and averages of
C the Fock states are the factor %(kname)sFAC(), applied to the squared
C amplitudes.
%(process_lines)s
C
      USE ALOHA_OBJECT
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(onia_nexternal)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
%(onia_data)s
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      COMPLEX*16 AMP_OUT(NGRAPHS)
C
C LOCAL VARIABLES
C
      INTEGER I, J, K, QQ, Q1, Q2
      REAL*8 FRAC, NORM
      COMPLEX*16 AMP(NGRAPHS), PROJ
      type(aloha) W(NWAVEFUNCS)
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA)
      REAL*8 PMASS_ONIA(NEXTERNAL+NONIA), PMASS_RESH(NEXTERNAL)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER FLAVOR_ONIA(NEXTERNAL+NONIA)
      COMPLEX*16 PROJ_SAVED(NONIA,-1:1,-1:1)
      INTEGER HELQ(2)
      DATA HELQ/-1, 1/
      INCLUDE 'coupl.inc'
C ----------
C BEGIN CODE
C ----------
%(pmassonia_lines)s
      PMASS_RESH = 0D0
      DO I = 1, NEXTERNAL+NONIA
        PMASS_RESH(MAPPING(I)) = PMASS_RESH(MAPPING(I))+PMASS_ONIA(I)
      ENDDO
      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).NE.0) THEN
          J = FINDLOC(PAIRS,-PAIRS(I),DIM=1)
          FRAC = PMASS_ONIA(I)/(PMASS_ONIA(I)+PMASS_ONIA(J))
        ELSE
          FRAC = 1D0
        ENDIF
        P_ONIA(:,I) = FRAC*P(:,MAPPING(I))
        NHEL_ONIA(I) = NHEL(MAPPING(I))
        IC_ONIA(I) = 1
        FLAVOR_ONIA(I) = 1
      ENDDO
C     the spin projectors of each onium for its J_z, and the colour norm
      NORM = 1D0
      DO I = 1, NONIA
        QQ = ONIA(I)
        Q1 = FINDLOC(PAIRS, I, DIM=1)
        Q2 = FINDLOC(PAIRS,-I, DIM=1)
        DO J = 1, 2
          DO K = 1, 2
            CALL ONIA_PROJ(P_ONIA(:,Q1),PMASS_ONIA(Q1),HELQ(J),
     &        P_ONIA(:,Q2),PMASS_ONIA(Q2),HELQ(K),P(:,QQ),PMASS_RESH(QQ),
     &        NHEL(QQ),S_ONIA(I),PROJ)
            PROJ_SAVED(I,HELQ(J),HELQ(K)) = PROJ
          ENDDO
        ENDDO
        IF (C_ONIA(I).EQ.1) THEN
          NORM = NORM/3D0
        ELSE IF (C_ONIA(I).EQ.8) THEN
          NORM = NORM*2D0
        ENDIF
      ENDDO
C     the open amplitudes for every constituent helicity, projected
      AMP_OUT(:) = (0D0,0D0)
      DO J = 1, 4**NONIA
        PROJ = SQRT(NORM)
        DO I = 1, NONIA
          Q1 = FINDLOC(PAIRS, I, DIM=1)
          Q2 = FINDLOC(PAIRS,-I, DIM=1)
          NHEL_ONIA(Q1) = HELQ(IBITS(J-1,2*I-2,1)+1)
          NHEL_ONIA(Q2) = HELQ(IBITS(J-1,2*I-1,1)+1)
          PROJ = PROJ*PROJ_SAVED(I,NHEL_ONIA(Q1),NHEL_ONIA(Q2))
        ENDDO
        IF (PROJ.NE.(0D0,0D0)) THEN
%(helas_calls)s
          AMP_OUT(:) = AMP_OUT(:)+AMP(:)*PROJ
        ENDIF
      ENDDO
      END


      DOUBLE PRECISION FUNCTION %(kname)sFAC()
C
C The LDME factor of the bound states of %(kname)s: for each onium
C <O_n> (m1+m2)/(2 m1 m2)/(N_[C] (2J+1)), N_[1] = 6, N_[8] = 8
C (arXiv:2402.19221, eq. 2.9), with the LDMEs of the onia card.
C
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(onia_nexternal)d)
%(onia_data)s
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      INTEGER I, Q1, Q2
      REAL*8 LDME, PMASS_ONIA(NEXTERNAL+NONIA)
      LOGICAL FIRSTTIME
      DATA FIRSTTIME/.TRUE./
      REAL*8 SAVED
      SAVE FIRSTTIME, SAVED
      INCLUDE 'coupl.inc'
%(ldme_decl)s
      INTEGER NLDME
      PARAMETER (NLDME=%(nldme)d)
      INTEGER LDME_PDG(NLDME+1)
      DATA LDME_PDG/%(ldme_pdgs)s/
      DOUBLE PRECISION LDME_VAL(NLDME+1)
      IF (FIRSTTIME) THEN
        CALL %(kname)sREADCARD(NLDME, LDME_PDG, LDME_VAL)
%(ldme_assign)s
%(pmassonia_lines)s
        LDME = 1D0
        DO I = 1, NONIA
          Q1 = FINDLOC(PAIRS, I, DIM=1)
          Q2 = FINDLOC(PAIRS,-I, DIM=1)
          LDME = LDME*(PMASS_ONIA(Q1)+PMASS_ONIA(Q2))
     &      /(2D0*PMASS_ONIA(Q1)*PMASS_ONIA(Q2))
          IF (C_ONIA(I).EQ.1) THEN
            LDME = LDME/6D0/(2D0*J_ONIA(I)+1D0)
          ELSE IF (C_ONIA(I).EQ.8) THEN
            LDME = LDME/8D0/(2D0*J_ONIA(I)+1D0)
          ENDIF
        ENDDO
C       perturbative (leptonium) LDMEs
        %(ldme_perturbative)s
        SAVED = LDME*%(ldme_product)s
        FIRSTTIME = .FALSE.
      ENDIF
      %(kname)sFAC = SAVED
      END


      SUBROUTINE %(kname)sREADCARD(N, PDG, VAL)
C
C The LDMEs of Block ldme of the onia card for the codes PDG(1:N), read
C directly (no COMMON: the native Born providers keep private copies of
C the COMMON blocks of their sources).
C
      IMPLICIT NONE
      INTEGER N, PDG(*)
      DOUBLE PRECISION VAL(*)
      CHARACTER*256 LINE
      CHARACTER*16 WORD, BNAME
      CHARACTER*40 FILES(4)
      DATA FILES /'onia_card.dat', '../onia_card.dat',
     &  '../../Cards/onia_card.dat', '../../../Cards/onia_card.dat'/
      INTEGER I, K, IUNIT, IOS, ID
      DOUBLE PRECISION X
      LOGICAL INBLOCK, FOUND(64)
      IF (N.EQ.0) RETURN
      IUNIT = 97
      DO K = 1, 4
        OPEN(UNIT=IUNIT, FILE=FILES(K), STATUS='OLD', IOSTAT=IOS)
        IF (IOS.EQ.0) GOTO 10
      ENDDO
      WRITE(*,*) 'Error: the onia card onia_card.dat was not found'
      STOP 1
 10   CONTINUE
      INBLOCK = .FALSE.
      DO I = 1, N
        FOUND(I) = .FALSE.
      ENDDO
 20   CONTINUE
      READ(IUNIT,'(A)',IOSTAT=IOS) LINE
      IF (IOS.NE.0) GOTO 30
      IF (INDEX(ADJUSTL(LINE),'#').EQ.1) GOTO 20
      WORD = ' '
      BNAME = ' '
      READ(LINE,*,IOSTAT=IOS) WORD, BNAME
      IF (WORD.EQ.'Block' .OR. WORD.EQ.'BLOCK' .OR. WORD.EQ.'block') THEN
        INBLOCK = BNAME.EQ.'ldme' .OR. BNAME.EQ.'LDME' .OR. BNAME.EQ.'Ldme'
        GOTO 20
      ENDIF
      IF (INBLOCK) THEN
        READ(LINE,*,IOSTAT=IOS) ID, X
        IF (IOS.EQ.0) THEN
          DO I = 1, N
            IF (PDG(I).EQ.ID) THEN
              VAL(I) = X
              FOUND(I) = .TRUE.
            ENDIF
          ENDDO
        ENDIF
      ENDIF
      GOTO 20
 30   CONTINUE
      CLOSE(IUNIT)
      DO I = 1, N
        IF (.NOT.FOUND(I)) THEN
          WRITE(*,*) 'Error: no LDME for ', PDG(I), ' in Block ldme'
     &      //' of the onia card'
          STOP 1
        ENDIF
      ENDDO
      END
