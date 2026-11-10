      SUBROUTINE %(proc_prefix)scompute_accuracy(fulllist, length, acc, estimate)
      implicit none
C  
C PARAMETERS 
C
      integer maxstabilitylength
	  common/%(proc_prefix)sstability_tests/maxstabilitylength
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C
      real*8 fulllist(3,0:NSQUAREDSO,maxstabilitylength)
      integer length
      real*8 acc(0:NSQUAREDSO), estimate(0:3,0:NSQUAREDSO)
C  
C LOCAL VARIABLES 
C
      logical mask(maxstabilitylength)
	  logical mask3(3)
	  data mask3/.TRUE.,.TRUE.,.TRUE./
      integer i,j,k
      real*8 avg
      real*8 diff
	  real*8 accuracies(3)
	  real*8 list(maxstabilitylength)

C
C GLOBAL VARIABLES
C
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS
	  integer I_LIB
	  common/%(proc_prefix)sI_LIB/I_LIB
	  include 'MadLoopParams.inc'

C ----------
C BEGIN CODE
C ----------
      do i=1,length
        mask(i)=.TRUE.
      enddo
      do i=length+1,maxstabilitylength
        mask(i)=.FALSE.
C For some architectures, it is necessary to initialize all the elements of fulllist(i,j)
C Beware that if the length provided is incorrect, then this can corrup the fulllist given in argument.
        do j=0,NSQUAREDSO
          do k=1,3
		        fulllist(k,j,i)=0.0d0
		      enddo
		    enddo
      enddo

      do k=0,NSQUAREDSO

        do i=1,3
          do j=1,maxstabilitylength
            list(j)=fulllist(i,k,j)
          enddo
          diff=maxval(list,1,mask)-minval(list,1,mask)
          avg=(maxval(list,1,mask)+minval(list,1,mask))/2.0d0
          estimate(i,k)=avg
          if (avg.eq.0.0d0) then
            accuracies(i)=diff
          else
            accuracies(i)=diff/abs(avg)
          endif
        enddo

C       The technique below is too sensitive, typically to
C       unstablities in very small poles
C       acc(k)=MAXVAL(ACCURACIES,1,MASK3)
C       The following is used instead
        acc(k) = 0.0d0
        AVG = 0.0d0
        DO I=1,3
          acc(k) = acc(k) + ACCURACIES(I)*ABS(ESTIMATE(I,K))
          AVG = AVG + ESTIMATE(I,K)
        ENDDO
        if (avg.ne.0.0d0) then
              acc(k)  = acc(k) / ( ABS(AVG) / 3.0d0)
        endif
        
C       When using COLLIER with the internal stability test, the first evaluation is typically more reliable so we do not want to use the average but rather the first evaluation.
        IF (MLREDUCTIONLIB(I_LIB).eq.7.and.COLLIERUseInternalStabilityTest) THEN
          DO I=1,3
            estimate(i,k) = fulllist(i,k,1)
          ENDDO
        ENDIF

c       Make sure to hard-set to zero accuracies of coupling orders not included
        if (K.ne.0) THEN
          IF (.NOT.CHOSEN_SO_CONFIGS(K)) THEN
            acc(k) = 0.0d0
          ENDIF
		    ENDIF

c       If NaN are present in the evaluation, automatically set the accuracy to 1.0d99.
        DO I=1,3
          DO J=1,MAXSTABILITYLENGTH
            IF (isnan(FULLLIST(I,K,J))) then
              ACC(K) = 1.0d99
            ENDIF
          ENDDO
        ENDDO

	    enddo

      end

