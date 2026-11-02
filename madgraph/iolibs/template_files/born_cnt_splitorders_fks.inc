      SUBROUTINE SBORN_CNT%(proc_prefix)s(P,ANS_CNT)
C
C     Return the counterterms which are different from the born. in particular this is needed
C     for the g/a > qqbar splitting
C  
C CONSTANTS
C 
      implicit none
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL-1)
C
C VARIABLES
C
      INTEGER I,J,K
      include 'orders.inc'
      include 'coupl.inc'
      COMPLEX*16 ANS(2,0:NSQAMPSO), ANS_CNT(2, NSPLITORDERS)
      LOGICAL  KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO), FIRSTTIME
      save keep_order_cnt 
      integer ord_subtract
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
       DOUBLE PRECISION TINY
       PARAMETER (TINY = 1D-12)
       DOUBLE PRECISION MAX_VAL
## if(density) {
      include "density_info.inc"
## }
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B_CNT%(proc_prefix)s
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C
C look for orders which match the born order constraint 

if (firsttime) then
 do i = 1, nsqampso

C this is for the orders of the counterterms
  do j = 1, nsplitorders
  keep_order_cnt(j,i) = .true.
   do k = 1, nsplitorders
    if (j.eq.k) then
     ord_subtract=2
    else
     ord_subtract=0
    endif
    if(GETORDPOWFROMINDEX_B_CNT%(proc_prefix)s(k, i) .gt. nlo_orders(k)-ord_subtract) then
     keep_order_cnt(j,i) = .false.
     EXIT 
    endif
   enddo
  enddo
 enddo

 do j = 1, nsplitorders
  !write(*,*) 'counterterm S.O', j, ordernames(j)
  do i = 1, nsqampso
   if (keep_order_cnt(j,i)) then
    write(*,*) 'CNT%(proc_prefix)s: keeping split order', i
   else
    write(*,*) 'CNT%(proc_prefix)s: not keeping split order', i
   endif
  enddo
 enddo
 firsttime = .false.
endif

## if(density) {
C the projector W of the density matrix is not implemented for the
C extra counterterms
      IF (DENS_W_ON) THEN
        WRITE(*,*) 'ERROR: the density-matrix projector W is not'
        WRITE(*,*) 'available for the extra counterterms (born_cnt)'
        STOP 1
      ENDIF
## }
CALL SBORN_CNT%(proc_prefix)s_SPLITORDERS(P,ANS)

MAX_VAL = 0D0
 
C reset the amp_split array
amp_split_cnt(1:amp_split_size,1:2,1:nsplitorders) = dcmplx(0d0,0d0)

DO I = 1, NSQAMPSO
  MAX_VAL = MAX(MAX_VAL, ABS(ANS(1,I)))
ENDDO
 
C quantities for the counterterms
do j = 1, nsplitorders
 ans_cnt(1:2,j) = dcmplx(0d0, 0d0)
 do i = 1, nsqampso
  if (keep_order_cnt(j,i)) then 
    IF(ABS(ANS(1,I)).GT.MAX_VAL*TINY) ANS_CNT(1,J) = ANS_CNT(1,J) + ANS(1,I)
    IF(ABS(ANS(1,I)).GT.MAX_VAL*TINY) ANS_CNT(2,J) = ANS_CNT(2,J) + ANS(2,I)
C   keep track of the separate pieces also for counterterms
   do k = 1, nsplitorders
      amp_orders(k) = GETORDPOWFROMINDEX_B_CNT%(proc_prefix)s(k, i)
C take into account the fact that the counterterm for a given split order
C will be multiplied by the corresponding squared coupling
      if (k.eq.j) amp_orders(k) = amp_orders(k) + 2
   enddo
C this is to avoid fake non-zero contributions
   IF (ABS(ANS(1,I)).GT.MAX_VAL*TINY) AMP_SPLIT_CNT(ORDERS_TO_AMP_SPLIT_POS(AMP_ORDERS),1,J) = ANS(1,I)
   IF (ABS(ANS(2,I)).GT.MAX_VAL*TINY) AMP_SPLIT_CNT(ORDERS_TO_AMP_SPLIT_POS(AMP_ORDERS),2,J) = ANS(2,I)
  endif
 enddo
C this is to avoid fake non-zero contributions 
 IF (ABS(ANS_CNT(1,J)).LT.MAX_VAL*TINY) ANS_CNT(1,J)=(0D0,0D0)
 IF (ABS(ANS_CNT(2,J)).LT.MAX_VAL*TINY) ANS_CNT(2,J)=(0D0,0D0)
enddo

return
END



SUBROUTINE SBORN_CNT%(proc_prefix)s_SPLITORDERS(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C  
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include "nexternal.inc"
      include "born_nhel.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(2,0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,i,glu_ij
      REAL*8 borns(2,NSQAMPSO)
      INTEGER NTRY
      DATA NTRY / 0 /
      COMPLEX*16 T(2,NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
%(den_factor_lines)s
%(ij_lines)s
C  
C GLOBAL VARIABLES
C  
C MZ uncomment and change these lines if we'll decide to
C use also the amp from the counterterms for the multichannelling
C      Double Precision amp2(%(ngraphs)d), jamp2(0:%(ncolor)d,0:NAMPSO)
C      common/to_amps/  amp2,       jamp2
C
C MZ      DATA jamp2(0,0) /   %(ncolor)d/          
      LOGICAL GOODHEL(NCOMB)
      save goodhel
      double precision hel_fac
      integer get_hel,skip
      common/cBorn_CNT%(proc_prefix)s/hel_fac,get_hel,skip
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
      NTRY=NTRY+1
      if (NTRY.lt.2) then
         skip=1			 
         do while(nhel(glu_ij ,skip).ne.1)
            skip=skip+1
         enddo
         skip=skip-1
      endif
c MZ      DO JJ=1,NGRAPHS
c          amp2(jj)=0d0
c      ENDDO
c      DO I=0,NAMPSO
c      DO JJ=1,int(jamp2(0,0))
c          jamp2(jj,I)=0d0
c      ENDDO
c MZ      ENDDO
      DO I=0,NSQAMPSO
          ANS(1,I) = 0D0
          ANS(2,I) = 0D0
      ENDDO
      hel_fac=1d0
      DO IHEL=1,NCOMB
        if (nhel(glu_ij,ihel).le.0) then
          IF ((GOODHEL(IHEL) .OR. GOODHEL(IHEL+SKIP) .OR. NTRY .LT. 2) ) THEN
            
            CALL BORN_CNT%(proc_prefix)s(P1,NHEL(1,IHEL),IHEL,T,borns)
            DO I=1,NSQAMPSO
                ANS(1,I)=ANS(1,I)+T(1,I)
                ANS(2,I)=ANS(2,I)+T(2,I)
            ENDDO
            if ( borns(1,1).ne.0d0 .AND. .NOT. GOODHEL(IHEL) ) then
              GOODHEL(IHEL)=.TRUE.
            endif
            if ( borns(2,1).ne.0d0 .AND. .NOT. GOODHEL(IHEL+SKIP) ) then
              GOODHEL(IHEL+SKIP)=.TRUE.
            endif
	  ENDIF
        ENDIF
      ENDDO
      DO I=1,NSQAMPSO
      ANS(1,I)=ANS(1,I)/DBLE(IDEN)
      ANS(2,I)=ANS(2,I)/DBLE(IDEN)
      ANS(1,0)=ANS(1,0)+ANS(1,I)
      ANS(2,0)=ANS(2,0)+ANS(2,I)
      ENDDO
      END
       
       
      SUBROUTINE BORN_CNT%(proc_prefix)s(P,NHEL,HELL,ANS,borns)
C  
%(info_lines)s
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)
C The amplitudes are computed by GET_AMP_BORN_CNT, the JAMPs by
C GET_JAMP_BORN_CNT; GET_MATRIX_BORN_CNT is the colour sum of one
C helicity, GET_INTER_BORN_CNT the colour contraction of the JAMPs of the
C two helicities of the FKS mother (the spin-correlated Born, ANS(2,:)).

%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d) 
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      include "nexternal.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1),borns(2,NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1), HELL
      COMPLEX*16 ANS(2,NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,ihel,back_hel,glu_ij,iflip
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), jamph(2, ncolor,nampso)
      COMPLEX*16 RES(NSQAMPSO)
C  
C GLOBAL VARIABLES
C  
      integer nfksprocess
      common/c_nfksprocess/nfksprocess

%(ij_lines)s
C ----------
C BEGIN CODE
C ----------
      glu_ij = ij_values(nfksprocess)
      jamph(:,:,:) = (0d0,0d0)
      DO I = 1, NSQAMPSO
        ANS(1,I)=0D0
        ANS(2,I)=0D0
        borns(1,I)=0d0
        borns(2,I)=0d0
      ENDDO
      back_hel = nhel(glu_ij)
      do ihel=-1,1,2
        if (ihel.eq.-1.or.nhel(glu_ij).ne.0) then
          if (nhel(glu_ij).ne.0) nhel(glu_ij) = ihel
          iflip = 2-(1-ihel)/2
          CALL GET_AMP_BORN_CNT%(proc_prefix)s(P,NHEL,AMP)
          CALL GET_JAMP_BORN_CNT%(proc_prefix)s(AMP,JAMP)
          CALL GET_MATRIX_BORN_CNT%(proc_prefix)s(JAMP,RES)
          DO I = 1, NSQAMPSO
            BORNS(iflip,I) = DBLE(RES(I))
          ENDDO
          jamph(iflip,:,:) = jamp(:,:)
        endif
      Enddo
      do i = 1, nsqampso
      ans(1,i) = borns(1,i) + borns(2,i) 
      enddo
      CALL GET_INTER_BORN_CNT%(proc_prefix)s(JAMPH(2,:,:),JAMPH(1,:,:),RES)
      ANS(2,:) = RES(:)
      nhel(glu_ij) = back_hel
      END


      SUBROUTINE GET_AMP_BORN_CNT%(proc_prefix)s(P,NHEL,AMP)
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
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   %(ngraphs)d,NEIGEN=  1) 
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "coupl.inc"
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      COMPLEX*16 AMP(NGRAPHS)
C
C LOCAL VARIABLES
C
      INTEGER IC(NEXTERNAL-1),nmo
      parameter (nmo=nexternal-1)
      data ic /nmo*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /NMO*1/
      type(aloha) W(NWAVEFUNCS)
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
      END


      SUBROUTINE GET_JAMP_BORN_CNT%(proc_prefix)s(AMP,JAMP)
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
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
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


      SUBROUTINE GET_MATRIX_BORN_CNT%(proc_prefix)s(JAMP,RES)
C
C conj(JAMP).CF.JAMP / DENOM for each squared split order (one helicity);
C only the real part of RES is meaningful
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
C
C ARGUMENTS
C
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      INTEGER  CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP
C
C FUNCTION
C
      INTEGER SQSOINDEXB_CNT%(proc_prefix)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      RES(:) = (0d0,0d0)
      DO M = 1, NAMPSO
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX +1
            ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
          DO N = 1, NAMPSO
            RES(SQSOINDEXB_CNT%(proc_prefix)s(M,N))=RES(SQSOINDEXB_CNT%(proc_prefix)s(M,N))+ZTEMP*DCONJG(JAMP(I,N))
          ENDDO
        ENDDO
      ENDDO
      RES(:)=RES(:)/DENOM
      END


      SUBROUTINE GET_INTER_BORN_CNT%(proc_prefix)s(JAMPA,JAMPB,RES)
C
C The colour contraction of two sets of JAMPs, for each squared split order:
C   RES = sum CF_ij (JAMPA(j,M) conj(JAMPB(i,N)) + JAMPA(i,M) conj(JAMPB(j,N)))
C         / (2 DENOM)
C i.e. sum_ij S_ij JAMPA(i) conj(JAMPB(j)) for the symmetric colour matrix S
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
C
C ARGUMENTS
C
      COMPLEX*16 JAMPA(NCOLOR,NAMPSO), JAMPB(NCOLOR,NAMPSO), RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      INTEGER  CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
C
C FUNCTION
C
      INTEGER SQSOINDEXB_CNT%(proc_prefix)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      RES(:) = (0d0,0d0)
      DO M = 1, NAMPSO
      CF_INDEX = 0 
      DO I = 1, NCOLOR
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          DO N = 1, NAMPSO
             RES(SQSOINDEXB_CNT%(proc_prefix)s(M,N))= RES(SQSOINDEXB_CNT%(proc_prefix)s(M,N)) + CF(CF_INDEX)*JAMPA(J,M)*DCONJG(JAMPB(I,N))+CF(CF_INDEX)*JAMPA(I,M)*DCONJG(JAMPB(J,N))
          ENDDO
        ENDDO
      ENDDO
      ENDDO
      RES(:) = RES(:) / (2d0*DENOM)
      END
       


C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEXB_CNT%(proc_prefix)s(AMPORDERA,AMPORDERB)
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
      INTEGER SQSOINDEXB_CNT%(proc_prefix)s_FROM_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEXB_CNT%(proc_prefix)s=SQSOINDEXB_CNT%(proc_prefix)s_FROM_ORDERS(SQORDERS)
	  END



      INTEGER FUNCTION SQSOINDEXB_CNT%(proc_prefix)s_FROM_ORDERS(ORDERS)
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
		SQSOINDEXB_CNT%(proc_prefix)s_FROM_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindexb_CNT%(proc_prefix)s_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX_B_CNT%(proc_prefix)s(IORDER, INDX)
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
      WRITE(*,*) "INVALID IORDER B", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX B", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX_B_CNT%(proc_prefix)s=SQSPLITORDERS(INDX, IORDER)
      END


      SUBROUTINE GET_NSQSO_BORN_CNT%(proc_prefix)s(NSQSO)
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


