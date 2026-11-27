      subroutine sborn_sf(p,m,n,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest),save::request
      type(BornResult),save::result
      integer m,n
      logical need_color_links,need_charge_links
      common/c_need_links/need_color_links,need_charge_links
      double precision amp_split_soft(amp_split_size)
      common/to_amp_split_soft/amp_split_soft
      request%colour=need_color_links
      request%charge=need_charge_links
      request%m=m
      request%n=n
      call mc_born_local(p,request,result)
      ans=result%correlation
      amp_split_soft=result%soft
      amp_split_cnt=result%split_counterterms
      end
      SUBROUTINE sborn_sf_amp(P_BORN,M,N,WGT,ANS_CNT,RET_AMP_SPLIT_CNT  ,LOC_SAVEAMP,AMP_SPLIT_SOFT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'genps.inc'
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1), WGT
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO = 1)
      DOUBLE COMPLEX WGT_BORN(2,0:NSQAMPSO)
      DOUBLE PRECISION WGT_COL
      DOUBLE PRECISION CHARGEPROD
      DOUBLE PRECISION SOFT_PREFAC
      INTEGER I,M,N
      INCLUDE 'orders.inc'
      COMPLEX*16 ANS_CNT(2, NSPLITORDERS)
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      COMMON /C_KEEP_ORDER_CNT/ KEEP_ORDER_CNT
      DOUBLE PRECISION CHARGES_BORN(NEXTERNAL-1)
      COMMON /C_CHARGES_BORN/CHARGES_BORN
      LOGICAL NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      COMMON /C_NEED_LINKS/NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      DOUBLE PRECISION PMASS(NEXTERNAL), ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE PRECISION AMP_SPLIT_SOFT(AMP_SPLIT_SIZE)
      DOUBLE COMPLEX RET_AMP_SPLIT_CNT(AMP_SPLIT_SIZE,2,NSPLITORDERS)
      DOUBLE COMPLEX LOC_SAVEAMP(NGRAPHS,MAX_BHEL)
      CHARGEPROD = 0D0
      IF (NEED_COLOR_LINKS.AND.NEED_CHARGE_LINKS) THEN
        WRITE(*,*) 'ERROR IN sborn_sf_amp, both color and charged links'    //' are needed'
        STOP
      ENDIF
      IF (NEED_COLOR_LINKS) THEN
        IF ((M.EQ.1 .AND. N.EQ.2).OR.(M.EQ.2 .AND. N.EQ.1)) THEN
          CALL SB_SF_001(P_BORN,WGT_COL,RET_AMP_SPLIT_CNT,LOC_SAVEAMP)
        ENDIF
        WGT = WGT_COL * G**2
        AMP_SPLIT_SOFT(1:AMP_SPLIT_SIZE) =     DBLE(RET_AMP_SPLIT_CNT(1:AMP_SPLIT_SIZE,1,QCD_POS)) * G**2
      ELSE IF (NEED_CHARGE_LINKS) THEN
        CHARGEPROD = CHARGES_BORN(M) * CHARGES_BORN(N)
        IF ((M.LE.NINCOMING.AND.N.GT.NINCOMING) .OR.     (N.LE.NINCOMING.AND.M.GT.NINCOMING)) CHARGEPROD = -     CHARGEPROD
        IF (M.EQ.N) CHARGEPROD = CHARGEPROD / 2D0
        WGT = DBLE(ANS_CNT(1, QED_POS)) * CHARGEPROD * DBLE(GAL(1))**2
        AMP_SPLIT_SOFT(1:AMP_SPLIT_SIZE) =     DBLE(RET_AMP_SPLIT_CNT(1:AMP_SPLIT_SIZE,1,QED_POS)) *     CHARGEPROD * DBLE(GAL(1))**2
      ENDIF
      RETURN
      END
      subroutine sborn_sf_store(p_born,m,n,wgt,ans_cnt,ret_amp_split_cnt,loc_saveamp,amp_split_soft)
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      include 'orders.inc'
      integer m,n
      double precision p_born(0:3,nexternal-1),wgt
      double complex ans_cnt(2,nsplitorders)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex loc_saveamp(ngraphs,max_bhel)
      double precision amp_split_soft(amp_split_size)
      double precision provider_soft(amp_split_size)
      common/to_amp_split_soft/provider_soft
      call sborn_sf_frame(p_born,m,n,wgt)
      amp_split_soft=provider_soft
      end
