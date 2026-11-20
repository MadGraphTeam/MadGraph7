      SUBROUTINE smatrix_real_local(P, RET_AMP_SPLIT, WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION WGT
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      IF (NFKSPROCESS.EQ.1) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.2) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.9) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.10) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.11) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.12) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.13) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.14) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.15) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.16) THEN
        CALL SMATRIX1_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.17) THEN
        CALL SMATRIX2_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.18) THEN
        CALL SMATRIX2_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.19) THEN
        CALL SMATRIX2_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.20) THEN
        CALL SMATRIX2_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.21) THEN
        CALL SMATRIX3_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.22) THEN
        CALL SMATRIX3_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.23) THEN
        CALL SMATRIX3_AMP(P, RET_AMP_SPLIT, WGT)
      ELSEIF (NFKSPROCESS.EQ.24) THEN
        CALL SMATRIX3_AMP(P, RET_AMP_SPLIT, WGT)
      ELSE
        WRITE(*,*) 'ERROR: invalid n in real_matrix :', NFKSPROCESS
        STOP
      ENDIF
      RETURN
      END

      RECURSIVE SUBROUTINE SMATRIX_REAL_VEC(P, RET_AMP_SPLIT, WGT,
     $  IVEC, NFKSPROCESS, REAL_FLAV_IDX)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION WGT
      INTEGER IVEC
      INTEGER NFKSPROCESS, REAL_FLAV_IDX
      IF (NFKSPROCESS.EQ.1) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.2) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.9) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.10) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.11) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.12) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.13) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.14) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.15) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.16) THEN
        CALL SMATRIX1_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.17) THEN
        CALL SMATRIX2_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.18) THEN
        CALL SMATRIX2_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.19) THEN
        CALL SMATRIX2_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.20) THEN
        CALL SMATRIX2_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.21) THEN
        CALL SMATRIX3_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.22) THEN
        CALL SMATRIX3_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.23) THEN
        CALL SMATRIX3_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSEIF (NFKSPROCESS.EQ.24) THEN
        CALL SMATRIX3_AMP_VEC(P, RET_AMP_SPLIT, WGT, IVEC,
     $                   real_flav_idx, nfksprocess)
      ELSE
        WRITE(*,*) 'ERROR: invalid n in real_matrix_vec :', NFKSPROCESS
        STOP
      ENDIF
      RETURN
      END


      RECURSIVE SUBROUTINE SMATRIX_REAL_VEC_BATCH(P,G_STRONG,
     $            ret_amp_split,wgt,active,coup_index,vector_size,
     $            nfksprocess,real_flav_idx)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INTEGER VECTOR_SIZE,NFKSPROCESS,REAL_FLAV_IDX,IVEC
      INTEGER COUP_INDEX(VECTOR_SIZE)
      LOGICAL ACTIVE(VECTOR_SIZE)
      DOUBLE PRECISION P(0:3,NEXTERNAL,VECTOR_SIZE)
      DOUBLE PRECISION G_STRONG(VECTOR_SIZE)
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE,VECTOR_SIZE)
      DOUBLE PRECISION WGT(VECTOR_SIZE)
      RET_AMP_SPLIT(:,:) = 0D0
      WGT(:) = 0D0
      DO IVEC = 1,VECTOR_SIZE
        IF (ACTIVE(IVEC)) THEN
          CALL SMATRIX_REAL_VEC(P(:,:,IVEC),
     $                   ret_amp_split(:,ivec),wgt(ivec),
     $                   coup_index(ivec),nfksprocess,real_flav_idx)
        ENDIF
      ENDDO
      RETURN
      END


      LOGICAL FUNCTION MC_PROVIDER_CAN_BATCH(ROW,FLAV)
      IMPLICIT NONE
      INTEGER ROW,FLAV
      MC_PROVIDER_CAN_BATCH=.FALSE.
      END

      SUBROUTINE MC_EVALUATE_PROVIDER_BATCH(P,G,EW,ROW,FLAV,N,
     $             wgt,amps,ok)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INTEGER ROW,FLAV,N
      DOUBLE PRECISION P(0:3,NEXTERNAL,N),G(N),EW
      DOUBLE PRECISION WGT(N),AMPS(AMP_SPLIT_SIZE,N)
      LOGICAL OK
      OK=.FALSE.
      WGT=0D0
      AMPS=0D0
      END

      LOGICAL FUNCTION MG7_NLO_REAL_IS_FORTRAN()
      IMPLICIT NONE
      MG7_NLO_REAL_IS_FORTRAN=.TRUE.
      END

