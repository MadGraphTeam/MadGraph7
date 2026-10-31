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
      include 'orders.inc'
      data keep_order / NSQAMPSO*.TRUE. /
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision ans_max, tiny
      parameter (tiny = 1d-12)
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
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
endif

CALL SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
ans_summed = 0d0
ans_max = 0d0

C reset the amp_split array
amp_split(1:amp_split_size) = 0d0

do i = 1, nsqampso
 ans_max = max(dabs(ans(i)),ans_max)
enddo

do i = 1, nsqampso
 if (keep_order(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(I)
C   keep track of the separate pieces corresponding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX%(proc_prefix)s(j, i)
    enddo
    if (abs(ans(i)).gt.ans_max*tiny) amp_split(orders_to_amp_split_pos(amp_orders)) = ans(i)
 endif
enddo

C avoid fake non-zeros
if (dabs(ans_summed).lt.tiny*ans_max) ans_summed=0d0

wgt_ME_real = ans_summed

END



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
      INTEGER IHEL,IDEN,I,J,T_IDENT(NCOMB)
      REAL*8 T(0:NSQAMPSO),T_SAVE(NCOMB,0:NSQAMPSO)
      SAVE T_SAVE,T_IDENT
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL(NCOMB)
      DATA GOODHEL/NCOMB*.FALSE./
      INTEGER NTRY
      DATA NTRY/0/
      include 'real_matrix_weights.inc'
%(den_factor_line)s
## if(density) {
      include 'density_info.inc'
## }
C ----------
C BEGIN CODE
C ----------
## if(density) {
C     Projected on the density legs: Tr(W rho) instead of the helicity
C     sum. Before NTRY is incremented, so that the helicity filter is
C     still set up by the first unprojected call.
      CALL DENSITY_W_INIT()
      IF (DENS_W_ON) THEN
        CALL SMATRIX%(proc_prefix)s_W(P,ANS)
        RETURN
      ENDIF
## }
      NTRY=NTRY+1
      if (get_real_diagram_weights) real_amp2(:)=0d0
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          IF (NTRY.LT.2) THEN
C for the first ps-point, check for helicities that give
C identical matrix elements
            call MATRIX_%(proc_prefix)s(P ,NHEL(1,IHEL),T)            
            DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
            ENDDO
            T_IDENT(IHEL)=-1
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0).EQ.0D0) CYCLE
                do j = 0, nsqampso
                  IF (ABS(T(j)/T_SAVE(I,j)-1D0) .GT. 1D-12) goto 444
                enddo
                T_IDENT(IHEL) = I
 444            continue
            ENDDO
          ELSE
            IF (T_IDENT(IHEL).GT.0) THEN
C if two helicity states are identical, dont recompute
            DO I=0,NSQAMPSO
              T(I)=T_SAVE(T_IDENT(IHEL),I)
              T_SAVE(IHEL,I)=T(I)
            ENDDO
            ELSE
              CALL MATRIX_%(proc_prefix)s(P ,NHEL(1,IHEL),T)            
              DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
              ENDDO
            ENDIF
          ENDIF
C add to the sum of helicities
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
C the diagram weights sum over the helicities computed after the
C first point (good ones, not identical to an earlier one), also
C at the first point, so that they depend on P only
      IF (NTRY.LT.2 .AND. get_real_diagram_weights) THEN
        real_amp2(:)=0d0
        DO IHEL=1,NCOMB
          IF (GOODHEL(IHEL) .AND. T_IDENT(IHEL).LE.0) THEN
            CALL MATRIX_%(proc_prefix)s(P ,NHEL(1,IHEL),T)
          ENDIF
        ENDDO
      ENDIF
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
       

      SUBROUTINE MATRIX_%(proc_prefix)s(P,NHEL,RES)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
c The amplitudes are computed by GET_AMP, the JAMPs by GET_JAMP and the
c colour sum by GET_MATRIX.
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      include 'nexternal.inc'
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      REAL*8 RES(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      include 'real_matrix_weights.inc'
C ----------
C BEGIN CODE
C ----------
      CALL GET_AMP_%(proc_prefix)s(P,NHEL,AMP)
      if (get_real_diagram_weights) then
%(real_amp2_lines)s
      endif
      CALL GET_JAMP_%(proc_prefix)s(AMP,JAMP)
      CALL GET_MATRIX_%(proc_prefix)s(JAMP,RES)
      END


      SUBROUTINE GET_AMP_%(proc_prefix)s(P,NHEL,AMP)
C
%(info_lines)s
C Returns the amplitudes AMP(NGRAPHS) for the helicity configuration NHEL
C (the HELAS calls)
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
      INTEGER NHEL(NEXTERNAL)
      COMPLEX*16 AMP(NGRAPHS)
C
C LOCAL VARIABLES
C
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      type(aloha) W(NWAVEFUNCS)
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
      END


      SUBROUTINE GET_JAMP_%(proc_prefix)s(AMP,JAMP)
C
%(info_lines)s
C Returns the JAMPs (colour-ordered amplitudes) for each amplitude split
C order
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
C
C LOCAL VARIABLES
C
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
C ----------
C BEGIN CODE
C ----------
      jamp(:,:) = (0d0,0d0)
%(jamp_lines)s
      END


      SUBROUTINE GET_MATRIX_%(proc_prefix)s(JAMP,RES)
C
C The colour sum conj(JAMP).CF.JAMP/DENOM of one helicity, for each squared
C split order, and their sum in RES(0)
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)
      REAL*8 RES(0:NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP
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

## if(density) {

      SUBROUTINE GET_INTER_%(proc_prefix)s(JAMPA,JAMPB,RES)
C
C The colour contraction sum_ij S_ij JAMPA(i) conj(JAMPB(j)) of the JAMPs
C of two helicity configurations, for each squared split order; with
C JAMPA = JAMPB its real part is GET_MATRIX
C
      IMPLICIT NONE
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      COMPLEX*16 JAMPA(NCOLOR,NAMPSO), JAMPB(NCOLOR,NAMPSO)
      COMPLEX*16 RES(NSQAMPSO)
      INTEGER I,J,M,N
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP
      INTEGER SQSOINDEX%(proc_prefix)s
%(color_data_lines)s
      RES(:) = (0d0,0d0)
      DO M = 1, NAMPSO
        DO N = 1, NAMPSO
          ZTEMP = (0d0,0d0)
          CF_INDEX = 0
          DO I = 1, NCOLOR
            DO J = I, NCOLOR
              CF_INDEX = CF_INDEX+1
              ZTEMP = ZTEMP + CF(CF_INDEX)*(JAMPA(J,M)*DCONJG(JAMPB(I,N))+JAMPA(I,M)*DCONJG(JAMPB(J,N)))
            ENDDO
          ENDDO
          RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N)) + ZTEMP
        ENDDO
      ENDDO
      RES(:) = RES(:)/(2d0*DENOM)
      END


      SUBROUTINE SMATRIX%(proc_prefix)s_W(P,ANS)
C
C The real emission projected on the density legs of the Born
C (density_info.inc): Tr(W rho) for the projector W, per squared split
C order (ANS(0) is their sum), averaged as SMATRIX. The density legs are
C at the Born positions POS_B; in this real emission they are shifted by
C the FKS parton i_fks of the current FKS configuration (the mapping of
C get_frame_mask_real in me_frame_boost.f).
C
      IMPLICIT NONE
      include 'nexternal.inc'
      include 'density_info.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      INTEGER     NCOMB         
      PARAMETER ( NCOMB=%(ncomb)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER IHEL,JHEL,IDEN,I,K,C,C1,C2,RPOS(N_CHANGING_B)
      INTEGER HELIDX(N_COMB_B)
      LOGICAL ISFIRST, SAME
      COMPLEX*16 AMP(NGRAPHS), JALL(NCOLOR,NAMPSO,NCOMB)
      COMPLEX*16 RES(NSQAMPSO), ZANS(NSQAMPSO)
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
%(den_factor_line)s
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DO K=1,N_CHANGING_B
        RPOS(K)=POS_B(K)
        IF (POS_B(K).GE.FKS_I_D(NFKSPROCESS)) RPOS(K)=POS_B(K)+1
      ENDDO
      DO IHEL=1,NCOMB
        CALL GET_AMP_%(proc_prefix)s(P,NHEL(1,IHEL),AMP)
        CALL GET_JAMP_%(proc_prefix)s(AMP,JALL(1,1,IHEL))
      ENDDO
      ZANS(:)=(0d0,0d0)
      DO IHEL=1,NCOMB
C       one pass per helicity of the other legs: the configurations whose
C       density legs are in the first combination
        ISFIRST=.TRUE.
        DO K=1,N_CHANGING_B
          IF (NHEL(RPOS(K),IHEL).NE.ALLOW_HEL_B(K)) ISFIRST=.FALSE.
        ENDDO
        IF (.NOT.ISFIRST) CYCLE
        DO C=1,N_COMB_B
          HELIDX(C)=0
          DO JHEL=1,NCOMB
            SAME=.TRUE.
            DO I=1,NEXTERNAL
              IF (ANY(RPOS(:).EQ.I)) CYCLE
              IF (NHEL(I,JHEL).NE.NHEL(I,IHEL)) SAME=.FALSE.
            ENDDO
            DO K=1,N_CHANGING_B
              IF (NHEL(RPOS(K),JHEL).NE.ALLOW_HEL_B((C-1)*N_CHANGING_B+K)) SAME=.FALSE.
            ENDDO
            IF (SAME) HELIDX(C)=JHEL
          ENDDO
          IF (HELIDX(C).EQ.0) THEN
            WRITE(*,*) 'ERROR: real helicity not found for combination', C
            STOP 1
          ENDIF
        ENDDO
        DO C1=1,N_COMB_B
          DO C2=1,N_COMB_B
            IF (DENS_W(C2,C1).EQ.(0d0,0d0)) CYCLE
            CALL GET_INTER_%(proc_prefix)s(JALL(1,1,HELIDX(C1)),JALL(1,1,HELIDX(C2)),RES)
            ZANS(:)=ZANS(:)+DENS_W(C2,C1)*RES(:)
          ENDDO
        ENDDO
      ENDDO
      ANS(0)=0d0
      DO I=1,NSQAMPSO
        ANS(I)=DBLE(ZANS(I))/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
## }

C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEX%(proc_prefix)s(AMPORDERA,AMPORDERB)
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



      INTEGER FUNCTION SQSOINDEX_FROM_ORDERS%(proc_prefix)s(ORDERS)
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



      INTEGER FUNCTION GETORDPOWFROMINDEX%(proc_prefix)s(IORDER, INDX)
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



      SUBROUTINE GET_NSQSO_REAL%(proc_prefix)s(NSQSO)
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
