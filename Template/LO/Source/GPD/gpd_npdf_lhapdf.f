      SUBROUTINE GPD_NPDF_XF(TEMPLATE, IA, IZ, IPDG, X, MU, XF)
C     x f(x, MU) of parton IPDG (21 gluon) of the per-nucleon nuclear PDF
C     set TEMPLATE with %A, %Z replaced by IA, IZ (e.g.
C     nCTEQ15HQ_FullNuc_%A_%Z), from LHAPDF in its own set slot (11..14,
C     MadGraph7 uses slot 1), central member.
      IMPLICIT NONE
      CHARACTER*(*) TEMPLATE
      INTEGER IA, IZ, IPDG
      DOUBLE PRECISION X, MU, XF
      INTEGER NSLOT
      PARAMETER (NSLOT=4)
      CHARACTER*256 NAMES(NSLOT), NAME
      INTEGER NUSED, K, NSET, PID, I
      DATA NUSED/0/
      SAVE NAMES, NUSED
      CHARACTER*8 SA, SZ
      WRITE(SA, '(I0)') IA
      WRITE(SZ, '(I0)') IZ
      NAME = ''
      I = 1
      K = 1
      DO WHILE (I.LE.LEN_TRIM(TEMPLATE))
        IF (TEMPLATE(I:MIN(I+1,LEN(TEMPLATE))).EQ.'%A') THEN
          NAME(K:) = TRIM(SA)
          K = K + LEN_TRIM(SA)
          I = I + 2
        ELSE IF (TEMPLATE(I:MIN(I+1,LEN(TEMPLATE))).EQ.'%Z') THEN
          NAME(K:) = TRIM(SZ)
          K = K + LEN_TRIM(SZ)
          I = I + 2
        ELSE
          NAME(K:K) = TEMPLATE(I:I)
          K = K + 1
          I = I + 1
        ENDIF
      ENDDO
      NSET = 0
      DO K = 1, NUSED
        IF (NAMES(K).EQ.NAME) NSET = 10 + K
      ENDDO
      IF (NSET.EQ.0) THEN
        IF (NUSED.GE.NSLOT) THEN
          WRITE(*,*) 'GPD: too many nuclear PDF sets'
          STOP 1
        ENDIF
        NUSED = NUSED + 1
        NAMES(NUSED) = NAME
        NSET = 10 + NUSED
        WRITE(*,*) 'GPD: nuclear PDFs ', TRIM(NAME), ' (LHAPDF slot',
     &    NSET, ')'
        CALL LHAPDF_INITPDFSET_BYNAME(NSET, TRIM(NAME))
      ENDIF
      PID = IPDG
      CALL LHAPDF_XFXQ(NSET, 0, PID, X, MU, XF)
      END
