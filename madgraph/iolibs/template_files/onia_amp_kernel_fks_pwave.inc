
      SUBROUTINE %(kname)s(P,NHEL,AMP_OUT)
C
%(info_lines)s
C
C Bound states at NLO (FKS, arXiv:2402.19221; madgraph/fks/fks_onia.py),
C with P waves: the projected amplitudes AMP_OUT(NGRAPHS) of the diagrams
C for the physical momenta P and helicities NHEL (J_z for an onium). The
C constituents of an onium of momentum K and mass M = m1+m2 have
C k1 = m1/M K + q, k2 = m2/M K - q. A P wave is the derivative
C eps*(L_z).d/dq at q = 0 of the spin-projected amplitude, combined with
C the spin S_z = J_z - L_z by Clebsch-Gordan coefficients (as the dual
C numbers of the LO templates). Here the derivative is taken numerically
C along three real directions e_a orthogonal to K (eps* = sum_a c_a e_a),
C with the seven-point stencil (error O(h**6), h = 1d-2 M): the plain HELAS
C library serves S and P waves, and the native Born providers of the NLO
C runtime need no dual-number library. The LDME and averages are the
C factor %(kname)sFAC().
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
C     the step h = HREL*M of the derivative: its error is O((h/M)**6)
C     (q.p/K.p <= |q|/M for any momentum p, so h/M is the expansion
C     parameter), while its round-off grows as M/h near the soft limit of
C     a real emission (on-shell constituents below). The five-point
C     stencil, O((h/M)**4) ~ 1d-6, was seen by the FKS limit tests.
      REAL*8     HREL
      PARAMETER (HREL=1D-2)
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
      INTEGER I, J, K, KP, QQ, Q1, Q2, MU, A, IPT, NPT, ISZ, NSZ
      INTEGER SZ(NONIA), NSZ_ONIA(NONIA), SZ_LIST(3,NONIA)
      INTEGER DIR(NONIA), STEP(NONIA)
      REAL*8 FRAC, NORM, CG, KDOTN, WEIGHT, QSQ
      REAL*8 Q(0:3,NONIA), E3(0:3,3,NONIA), H(NONIA)
      REAL*8 STENCIL(6), SWEIGHT(6)
      DATA STENCIL /-3D0, -2D0, -1D0, 1D0, 2D0, 3D0/
      DATA SWEIGHT /-1D0, 9D0, -45D0, 45D0, -9D0, 1D0/
      COMPLEX*16 AMP(NGRAPHS), PROJ
      type(aloha) W(NWAVEFUNCS), EPS_L
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA), ONIA_MS(NEXTERNAL+NONIA)
      REAL*8 PMASS_ONIA(NEXTERNAL+NONIA), PMASS_RESH(NEXTERNAL)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER FLAVOR_ONIA(NEXTERNAL+NONIA)
C     the S_z combinations with a non-zero Clebsch-Gordan weight
      INTEGER    NSZMAX
      PARAMETER (NSZMAX=3**NONIA)
      INTEGER NGOOD, IGS, SZS(NONIA,NSZMAX)
      REAL*8 CGS(NSZMAX)
      COMPLEX*16 CAS(3,NONIA,NSZMAX), COEFS(NSZMAX), PRJ
      COMPLEX*16 PROJ_SAVED(NONIA,-1:1,-1:1,NSZMAX)
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
        NHEL_ONIA(I) = NHEL(MAPPING(I))
        IC_ONIA(I) = 1
        FLAVOR_ONIA(I) = 1
      ENDDO
C     for each onium: the S_z it can have, the colour norm, and for a P
C     wave the real directions orthogonal to K (rest-frame axes boosted to
C     the frame of P)
      NORM = 1D0
      DO I = 1, NONIA
        QQ = ONIA(I)
        IF (C_ONIA(I).EQ.1) THEN
          NORM = NORM/3D0
        ELSE IF (C_ONIA(I).EQ.8) THEN
          NORM = NORM*2D0
        ENDIF
        NSZ_ONIA(I) = 0
        DO K = -S_ONIA(I), S_ONIA(I)
          IF (ABS(NHEL(QQ)-K).LE.L_ONIA(I)) THEN
            NSZ_ONIA(I) = NSZ_ONIA(I)+1
            SZ_LIST(NSZ_ONIA(I),I) = K
          ENDIF
        ENDDO
        H(I) = HREL*PMASS_RESH(QQ)
        IF (L_ONIA(I).EQ.1) THEN
          DO A = 1, 3
            KDOTN = P(A,QQ)
            E3(0,A,I) = KDOTN/PMASS_RESH(QQ)
            DO MU = 1, 3
              E3(MU,A,I) = P(MU,QQ)*KDOTN/(PMASS_RESH(QQ)*(P(0,QQ)
     &          +PMASS_RESH(QQ)))
            ENDDO
            E3(A,A,I) = E3(A,A,I)+1D0
          ENDDO
        ENDIF
      ENDDO
      AMP_OUT(:) = (0D0,0D0)
C     the combinations of S_z of the onia (ISZ codes them) with a non-zero
C     Clebsch-Gordan weight, and for each P wave eps*(L_z) = sum_a CA(a) e_a,
C     L_z = J_z - S_z. The diagrams do not depend on S_z: they are evaluated
C     once per stencil point and constituent helicities, with the projectors
C     summed over S_z.
      NSZ = 1
      DO I = 1, NONIA
        NSZ = NSZ*NSZ_ONIA(I)
      ENDDO
      NGOOD = 0
      DO ISZ = 1, NSZ
        K = ISZ-1
        CG = 1D0
        DO I = 1, NONIA
          SZ(I) = SZ_LIST(MOD(K,NSZ_ONIA(I))+1,I)
          K = K/NSZ_ONIA(I)
          CALL %(kname)sCG(L_ONIA(I),NHEL(ONIA(I))-SZ(I),S_ONIA(I),
     &      SZ(I),J_ONIA(I),NHEL(ONIA(I)),WEIGHT)
          CG = CG*WEIGHT
        ENDDO
        IF (CG.EQ.0D0) CYCLE
        NGOOD = NGOOD+1
        CGS(NGOOD) = CG
        DO I = 1, NONIA
          SZS(I,NGOOD) = SZ(I)
          IF (L_ONIA(I).NE.1) CYCLE
          QQ = ONIA(I)
          CALL VXXXXX(P(0,QQ),PMASS_RESH(QQ),NHEL(QQ)-SZ(I),1,EPS_L)
          DO A = 1, 3
            CAS(A,I,NGOOD) = -(EPS_L%%W(1)*E3(0,A,I)
     &        -EPS_L%%W(2)*E3(1,A,I)-EPS_L%%W(3)*E3(2,A,I)
     &        -EPS_L%%W(4)*E3(3,A,I))
          ENDDO
        ENDDO
      ENDDO
      IF (NGOOD.EQ.0) RETURN
C     the derivatives: the directions and stencil points of every P wave
      NPT = 18**NPWAVE
      DO IPT = 1, NPT
        K = IPT-1
        DO I = 1, NONIA
          Q(:,I) = 0D0
          IF (L_ONIA(I).NE.1) CYCLE
          STEP(I) = MOD(K,6)+1
          K = K/6
          DIR(I) = MOD(K,3)+1
          K = K/3
          Q(:,I) = STENCIL(STEP(I))*H(I)*E3(:,DIR(I),I)
        ENDDO
        DO IGS = 1, NGOOD
          COEFS(IGS) = DCMPLX(CGS(IGS)*SQRT(NORM), 0D0)
          DO I = 1, NONIA
            IF (L_ONIA(I).NE.1) CYCLE
            COEFS(IGS) = COEFS(IGS)*CAS(DIR(I),I,IGS)
     &        *SWEIGHT(STEP(I))/(60D0*H(I))
          ENDDO
        ENDDO
C       the constituent momenta and spin projectors at these q:
C       k_i = m_i/M K +- q, which conserves momentum exactly, with
C       k_i**2 = m_i**2 + q**2. The constituents are given this mass,
C       ONIA_MS(i) = sqrt(m_i**2 + q**2), in their wave functions, in the
C       spin projector and in the propagators of a constituent line next
C       to the emission of one gluon (get_onia_kernel), so that such a
C       propagator is 2 k_i.k_g exactly near the soft limit of a real
C       emission. The shift is even in q and leaves the derivative
C       unchanged. (On-shell constituents with k_1 + k_2 /= K would break
C       momentum conservation at O(h**2) instead, and with it the
C       propagators near the soft limit built from the other side of a
C       diagram.)
        DO I = 1, NEXTERNAL+NONIA
          ONIA_MS(I) = PMASS_ONIA(I)
          IF (PAIRS(I).NE.0) THEN
            J = FINDLOC(PAIRS,-PAIRS(I),DIM=1)
            KP = ABS(PAIRS(I))
            QSQ = Q(0,KP)**2-Q(1,KP)**2-Q(2,KP)**2-Q(3,KP)**2
            FRAC = PMASS_ONIA(I)/(PMASS_ONIA(I)+PMASS_ONIA(J))
            ONIA_MS(I) = SQRT(PMASS_ONIA(I)**2+QSQ)
            P_ONIA(:,I) = FRAC*P(:,MAPPING(I))
     &        +SIGN(1,PAIRS(I))*Q(:,KP)
          ELSE
            P_ONIA(:,I) = P(:,MAPPING(I))
          ENDIF
        ENDDO
        DO IGS = 1, NGOOD
          DO I = 1, NONIA
            QQ = ONIA(I)
            Q1 = FINDLOC(PAIRS, I, DIM=1)
            Q2 = FINDLOC(PAIRS,-I, DIM=1)
            DO J = 1, 2
              DO A = 1, 2
                CALL ONIA_PROJ(P_ONIA(:,Q1),ONIA_MS(Q1),HELQ(J),
     &            P_ONIA(:,Q2),ONIA_MS(Q2),HELQ(A),P(:,QQ),
     &            PMASS_RESH(QQ),SZS(I,IGS),S_ONIA(I),PROJ)
                PROJ_SAVED(I,HELQ(J),HELQ(A),IGS) = PROJ
              ENDDO
            ENDDO
          ENDDO
        ENDDO
        DO J = 1, 4**NONIA
          DO I = 1, NONIA
            Q1 = FINDLOC(PAIRS, I, DIM=1)
            Q2 = FINDLOC(PAIRS,-I, DIM=1)
            NHEL_ONIA(Q1) = HELQ(IBITS(J-1,2*I-2,1)+1)
            NHEL_ONIA(Q2) = HELQ(IBITS(J-1,2*I-1,1)+1)
          ENDDO
          PROJ = (0D0,0D0)
          DO IGS = 1, NGOOD
            PRJ = COEFS(IGS)
            DO I = 1, NONIA
              Q1 = FINDLOC(PAIRS, I, DIM=1)
              Q2 = FINDLOC(PAIRS,-I, DIM=1)
              PRJ = PRJ*PROJ_SAVED(I,NHEL_ONIA(Q1),NHEL_ONIA(Q2),IGS)
            ENDDO
            PROJ = PROJ+PRJ
          ENDDO
          IF (PROJ.NE.(0D0,0D0)) THEN
%(helas_calls)s
            AMP_OUT(:) = AMP_OUT(:)+AMP(:)*PROJ
          ENDIF
        ENDDO
      ENDDO
      END


      SUBROUTINE %(kname)sCG(L, LZ, S, SZ, J, JZ, CG)
C     <L L_z; S S_z | J J_z> for L, S <= 1 (as CLEBSCH_GORDAN of the
C     dual-number HELAS library, which the NLO output does not build)
      IMPLICIT NONE
      INTEGER L, LZ, S, SZ, J, JZ
      DOUBLE PRECISION CG
      IF (JZ.NE.LZ+SZ .OR. ABS(LZ).GT.L .OR. ABS(SZ).GT.S) THEN
        CG = 0D0
      ELSE IF (L.EQ.0 .OR. S.EQ.0) THEN
        CG = 1D0
      ELSE IF (J.EQ.2) THEN
        IF (LZ.EQ.0 .AND. SZ.EQ.0) THEN
          CG = DSQRT(2D0/3D0)
        ELSE IF (LZ.EQ.0 .OR. SZ.EQ.0) THEN
          CG = 1D0/DSQRT(2D0)
        ELSE IF (LZ.EQ.SZ) THEN
          CG = 1D0
        ELSE
          CG = 1D0/DSQRT(6D0)
        ENDIF
      ELSE IF (J.EQ.1) THEN
        IF (LZ.EQ.1 .OR. SZ.EQ.-1) THEN
          CG = 1D0/DSQRT(2D0)
        ELSE IF (LZ.EQ.-1 .OR. SZ.EQ.1) THEN
          CG = -1D0/DSQRT(2D0)
        ELSE
          CG = 0D0
        ENDIF
      ELSE IF (J.EQ.0) THEN
        IF (LZ.EQ.0 .AND. SZ.EQ.0) THEN
          CG = -1D0/DSQRT(3D0)
        ELSE
          CG = 1D0/DSQRT(3D0)
        ENDIF
      ELSE
        CG = 0D0
      ENDIF
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
