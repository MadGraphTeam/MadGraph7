      SUBROUTINE SMATRIX%(proc_prefix)s(P,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (NLO_ORDERS)
C  
C  
%(process_lines)s
C
C
C CONSTANTS
C 
      implicit none
      include 'orders.inc'
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
C
C VARIABLES
C
      INTEGER I,J
      REAL*8 ANS(0:NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO), FIRSTTIME
      data keep_order / NSQAMPSO*.TRUE. /
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision ans_max, tiny
      parameter (tiny = 1d-12)
      DOUBLE PRECISION DUMMY_AMP_SPLIT(AMP_SPLIT_SIZE)
      double precision wgt_ME_born,wgt_ME_real
      common/c_wgt_ME_tree/wgt_ME_born,wgt_ME_real
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX%(proc_prefix)s
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

CALL SMATRIX%(proc_prefix)s_AMP(P,DUMMY_AMP_SPLIT,ANS_SUMMED)
AMP_SPLIT(:)=DUMMY_AMP_SPLIT(:)
WGT_ME_REAL=ANS_SUMMED

END


      SUBROUTINE SMATRIX%(proc_prefix)s_AMP(P,RET_AMP_SPLIT,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (NLO_ORDERS)
C  
C  
%(process_lines)s
C
C
C CONSTANTS
C 
      implicit none
      include 'orders.inc'
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
C
C VARIABLES
C
      INTEGER I,J
      REAL*8 ANS(0:NSQAMPSO)
      LOGICAL KEEP_ORDER%(proc_prefix)s(NSQAMPSO), FIRSTTIME
      COMMON /TO_KEEP_ORDER%(proc_prefix)s/keep_order%(proc_prefix)s
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision ans_max, tiny
      parameter (tiny = 1d-12)
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX%(proc_prefix)s
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

C look for orders which match the nlo order constraint 

if (firsttime) then
  keep_order%(proc_prefix)s(:) = .true.
  call firsttime%(proc_prefix)s_setup(firsttime,keep_order%(proc_prefix)s)
endif

CALL SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
ans_summed = 0d0
ans_max = 0d0

C reset the amp_split array
ret_amp_split(1:amp_split_size) = 0d0

do i = 1, nsqampso
 ans_max = max(dabs(ans(i)),ans_max)
enddo

do i = 1, nsqampso
 if (keep_order%(proc_prefix)s(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(I)
C   keep track of the separate pieces correspoinding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX%(proc_prefix)s(j, i)
    enddo
    if (abs(ans(i)).gt.ans_max*tiny) ret_amp_split(orders_to_amp_split_pos(amp_orders)) = ans(i)
 endif
enddo

C avoid fake non-zeros
if (dabs(ans_summed).lt.tiny*ans_max) ans_summed=0d0

C AMP_SPLIT(1:AMP_SPLIT_SIZE) = RET_AMP_SPLIT(1:AMP_SPLIT_SIZE)

return
END




      RECURSIVE SUBROUTINE SMATRIX%(proc_prefix)s_AMP_VEC(P,
     $     RET_AMP_SPLIT,ANS_SUMMED,IVEC,FLAV_IDX)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (NLO_ORDERS)
C  
C  
%(process_lines)s
C
C
C CONSTANTS
C 
      implicit none
      include 'orders.inc'
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      INTEGER IVEC,FLAV_IDX
C
C VARIABLES
C
      INTEGER I,J
      REAL*8 ANS(0:NSQAMPSO)
      LOGICAL KEEP_ORDER%(proc_prefix)s(NSQAMPSO)
      COMMON /TO_KEEP_ORDER%(proc_prefix)s/keep_order%(proc_prefix)s 
      integer amp_orders(nsplitorders)
      double precision ans_max, tiny
      parameter (tiny = 1d-12)
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX%(proc_prefix)s
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

C look for orders which match the nlo order constraint 

C if (firsttime) then
C   call firsttime%(proc_prefix)s_setup(firsttime,keep_order)
C endif

CALL SMATRIX%(proc_prefix)s_SPLITORDERS_VEC(P,ANS,IVEC,FLAV_IDX)
ans_summed = 0d0
ans_max = 0d0

C reset the amp_split array
ret_amp_split(1:amp_split_size) = 0d0

do i = 1, nsqampso
 ans_max = max(dabs(ans(i)),ans_max)
enddo

do i = 1, nsqampso
 if (keep_order%(proc_prefix)s(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(I)
C   keep track of the separate pieces correspoinding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX%(proc_prefix)s(j, i)
    enddo
    if (abs(ans(i)).gt.ans_max*tiny) ret_amp_split(orders_to_amp_split_pos(amp_orders)) = ans(i)
 endif
enddo

C avoid fake non-zeros
if (dabs(ans_summed).lt.tiny*ans_max) ans_summed=0d0

return
END




      SUBROUTINE FIRSTTIME%(proc_prefix)s_SETUP(FIRSTTIME,KEEP_ORDER)
C
C     Sets up the orders that match the NLO order constraint
C
implicit none
include 'orders.inc'
integer i,J
INTEGER NSQAMPSO
PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
LOGICAL KEEP_ORDER(NSQAMPSO), FIRSTTIME
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX%(proc_prefix)s

 do i = 1, nsqampso
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX%(proc_prefix)s(j, i) .gt. nlo_orders(j)) then
    keep_order(i) = .false.
    EXIT 
   endif
  enddo
  if (keep_order(i)) then
   write(*,*) 'REAL %(proc_prefix)s: keeping split order ', i
  else
   write(*,*) 'REAL %(proc_prefix)s: not keeping split order ', i
  endif
 enddo
 firsttime = .false.

 end


      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'nexternal.inc'
      INCLUDE 'nFKSconfigs.inc'
      INTEGER     NCOMB         
       PARAMETER ( NCOMB=%(ncomb)d)
      INTEGER HEL_CONFIGS
      PARAMETER (HEL_CONFIGS=%(hel_configs)s)
      INTEGER THEL
      PARAMETER (THEL=NCOMB*HEL_CONFIGS)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INCLUDE 'fks_info.inc'
      INTEGER IHEL,IDEN,I,J,FLAV_IDX,HEL_INDEX
      INTEGER T_IDENT(NCOMB,HEL_CONFIGS)
      REAL*8 T(0:NSQAMPSO)
      REAL*8 T_SAVE(NCOMB,0:NSQAMPSO,HEL_CONFIGS)
      SAVE T_SAVE,T_IDENT
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL(NCOMB,HEL_CONFIGS)
      DATA GOODHEL/THEL*.FALSE./
      INTEGER NTRY(HEL_CONFIGS)
      DATA NTRY/HEL_CONFIGS*0/
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      FLAV_IDX=REAL_FLAVOR_INDEX_D(NFKSPROCESS)
      %(hel_index_setup)s
%(flavor_den_factor_setup)s
      NTRY(HEL_INDEX)=NTRY(HEL_INDEX)+1
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL,HEL_INDEX) .OR.
     $      NTRY(HEL_INDEX) .LT. 2) THEN
          IF (NTRY(HEL_INDEX).LT.2) THEN
C for the first ps-point, check for helicities that give
C identical matrix elements
            call MATRIX_%(proc_prefix)s(P,NHEL(1,IHEL),FLAV_IDX,T)
            DO I=0,NSQAMPSO
                T_SAVE(IHEL,I,HEL_INDEX)=T(I)
            ENDDO
            T_IDENT(IHEL,HEL_INDEX)=-1
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0,HEL_INDEX).EQ.0D0) CYCLE
                do j = 0, nsqampso
                  IF (ABS(T(j)/T_SAVE(I,j,HEL_INDEX)-1D0)
     $                .GT. 1D-12) goto 444
                enddo
                T_IDENT(IHEL,HEL_INDEX) = I
 444            continue
            ENDDO
          ELSE
            IF (T_IDENT(IHEL,HEL_INDEX).GT.0) THEN
C if two helicity states are identical, dont recompute
            DO I=0,NSQAMPSO
              T(I)=T_SAVE(T_IDENT(IHEL,HEL_INDEX),I,HEL_INDEX)
              T_SAVE(IHEL,I,HEL_INDEX)=T(I)
            ENDDO
            ELSE
              CALL MATRIX_%(proc_prefix)s(P,NHEL(1,IHEL),FLAV_IDX,T)
              DO I=0,NSQAMPSO
                T_SAVE(IHEL,I,HEL_INDEX)=T(I)
              ENDDO
            ENDIF
          ENDIF
C add to the sum of helicities
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0 .AND.
     $        .NOT. GOODHEL(IHEL,HEL_INDEX)) THEN
            GOODHEL(IHEL,HEL_INDEX)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
       

      RECURSIVE SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS_VEC(P,
     $     ANS,IVEC,FLAV_IDX)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'nexternal.inc'
      INTEGER     NCOMB         
      PARAMETER ( NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER IVEC,FLAV_IDX
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,I,J,T_IDENT%(proc_prefix)s(NCOMB)
      REAL*8 T(0:NSQAMPSO),T_SAVE(NCOMB,0:NSQAMPSO)
C      SAVE T_SAVE,T_IDENT
      common /to_t_ident%(proc_prefix)s/t_ident%(proc_prefix)s
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL%(proc_prefix)s(NCOMB), DO_TRY
      DATA DO_TRY/.TRUE./
      COMMON /TO_GOODHEL%(proc_prefix)s/GOODHEL%(proc_prefix)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      T(:) = 0d0
      T_SAVE(:,:) = 0d0 
%(flavor_den_factor_setup)s
C      IF (DO_TRY) THEN
C        CALL SMATRIX%(proc_prefix)s_SPLITORDERS_NTRY_VEC(P,ANS,DO_TRY,GOODHEL%(proc_prefix)s,T_IDENT%(proc_prefix)s,IVEC)
C      ELSE
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
        T(:) = 0d0
        CALL MATRIX_%(proc_prefix)s_VEC(P,NHEL(1,IHEL),FLAV_IDX,T,
     $       IVEC)
C add to the sum of helicities
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
      ENDDO
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
C      ENDIF
      END


      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS_NTRY(P,ANS,DO_TRY,GOODHEL,T_IDENT)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'nexternal.inc'
      include 'fks_info.inc'
      INTEGER     NCOMB         
      PARAMETER ( NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,I,J,T_IDENT(NCOMB),FLAV_IDX,NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      REAL*8 T(0:NSQAMPSO),T_SAVE(NCOMB,0:NSQAMPSO)
      SAVE T_SAVE
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL(NCOMB), DO_TRY
C      DATA GOODHEL/NCOMB*.FALSE./
      INTEGER MAX_TRY
      DATA MAX_TRY/20/
      INTEGER NTRY
      DATA NTRY/0/
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      FLAV_IDX=REAL_FLAVOR_INDEX_D(NFKSPROCESS)
%(flavor_den_factor_setup)s
      T_SAVE(:,:) = 0d0
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
            T(:) = 0d0
C for the first ps-point, check for helicities that give
C identical matrix elements
            call MATRIX_%(proc_prefix)s(P,NHEL(1,IHEL),FLAV_IDX,T)
            DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
            ENDDO
C            T_IDENT(IHEL)=-1
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0).EQ.0D0) CYCLE
                do j = 0, nsqampso
                  IF (ABS(T(j)/T_SAVE(I,j)-1D0) .GT. 1D-12) goto 444
                enddo
                T_IDENT(IHEL) = I
 444            continue
            ENDDO
C add to the sum of helicities
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
      ENDDO
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO

      IF (NTRY.GE.MAX_TRY) THEN
            DO_TRY = .FALSE.
      ENDIF

      END
       

      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS_NTRY_VEC(P,ANS,
     $     DO_TRY,GOODHEL,T_IDENT,IVEC,FLAV_IDX)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'nexternal.inc'
      INTEGER     NCOMB         
      PARAMETER ( NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER IVEC,FLAV_IDX
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,I,J,T_IDENT(NCOMB)
      REAL*8 T(0:NSQAMPSO),T_SAVE(NCOMB,0:NSQAMPSO)
      SAVE T_SAVE
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL(NCOMB), DO_TRY
C      DATA GOODHEL/NCOMB*.FALSE./
      INTEGER MAX_TRY
      DATA MAX_TRY/20/
      INTEGER NTRY
      DATA NTRY/0/
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
%(flavor_den_factor_setup)s
      T_SAVE(:,:) = 0d0
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
            T(:) = 0d0
C for the first ps-point, check for helicities that give
C identical matrix elements
            call MATRIX_%(proc_prefix)s_VEC(P,NHEL(1,IHEL),FLAV_IDX,T,
     $           IVEC)
            DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
            ENDDO
C            T_IDENT(IHEL)=-1
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0).EQ.0D0) CYCLE
                do j = 0, nsqampso
                  IF (ABS(T(j)/T_SAVE(I,j)-1D0) .GT. 1D-12) goto 444
                enddo
                T_IDENT(IHEL) = I
 444            continue
            ENDDO
C add to the sum of helicities
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
      ENDDO
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO

      IF (NTRY.GE.MAX_TRY) THEN
            DO_TRY = .FALSE.
      ENDIF

      END
       

      SUBROUTINE MATRIX_%(proc_prefix)s(P,NHEL,FLAV_IDX,RES)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      USE ALOHA_OBJECT
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      include 'nexternal.inc'
      include 'coupl.inc'
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL),FLAV_IDX
      REAL*8 RES(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
%(flavor_mask_decl)s
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_prefix)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      jamp(:,:) = (0d0,0d0)
      FLAVOR(:) = 1
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      RES(:) = 0d0
      DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX+1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
            RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N)) + ZTEMP*DCONJG(JAMP(I,N))
          ENDDO
      ENDDO

      ENDDO
      RES(:) = RES(:)/DENOM
      DO I=1,NSQAMPSO
         RES(0)=RES(0)+RES(I)
      ENDDO

      END


      RECURSIVE SUBROUTINE MATRIX_%(proc_prefix)s_VEC(P,NHEL,FLAV_IDX,
     $     RES,IVEC)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      USE COUPLINGS
      USE ALOHA_OBJECT
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      include 'nexternal.inc'
      include 'coupl.inc'
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL),FLAV_IDX
      REAL*8 RES(0:NSQAMPSO)
      INTEGER IVEC
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
%(flavor_mask_decl)s
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_prefix)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      jamp(:,:) = (0d0,0d0)
      FLAVOR(:) = 1
%(flavor_mask_setup)s
%(helas_calls_vec)s
%(jamp_lines)s

      RES(:) = 0d0
      DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX+1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
            RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N)) + ZTEMP*DCONJG(JAMP(I,N))
          ENDDO
      ENDDO

      ENDDO
      RES(:) = RES(:)/DENOM
      DO I=1,NSQAMPSO
         RES(0)=RES(0)+RES(I)
      ENDDO

      END


C
C Helper functions to deal with the split orders.
C

      RECURSIVE INTEGER FUNCTION SQSOINDEX%(proc_prefix)s(AMPORDERA,AMPORDERB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or 
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices of the jamps, it return the corresponding
C index in the squared order canonical ordering.
C
C CONSTANTS
C      
      implicit none
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER AMPORDERA, AMPORDERB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSPLITORDERS)
      INTEGER AMPSPLITORDERS(NAMPSO,NSPLITORDERS)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER SQSOINDEX_FROM_ORDERS%(proc_prefix)s
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEX%(proc_prefix)s=SQSOINDEX_FROM_ORDERS%(proc_prefix)s(SQORDERS)
     END



      RECURSIVE INTEGER FUNCTION SQSOINDEX_FROM_ORDERS%(proc_prefix)s(ORDERS)
C
C From a list of values for the split orders, this function returns the
c corresponding index in the squared orders canonical ordering.
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER ORDERS(NSPLITORDERS)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      DO I=1,NSQAMPSO
	    DO J=1,NSPLITORDERS
		  IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		SQSOINDEX_FROM_ORDERS%(proc_prefix)s = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      RECURSIVE INTEGER FUNCTION GETORDPOWFROMINDEX%(proc_prefix)s(IORDER, INDX)
C
C Return the power of the IORDER-th order appearing at position INDX
C in the split-orders output
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER IORDER, INDX
C
C LOCAL VARIABLES
C
      integer i
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
      WRITE(*,*) "INVALID IORDER %(proc_prefix)s", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX %(proc_prefix)s", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX%(proc_prefix)s=SQSPLITORDERS(INDX, IORDER)
      END



      RECURSIVE SUBROUTINE GET_NSQSO_REAL%(proc_prefix)s(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END
