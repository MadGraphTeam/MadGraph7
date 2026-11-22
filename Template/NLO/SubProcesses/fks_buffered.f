c Explicit-buffer fixed-order entrypoints retained from the feature branch.
c Scalar MC/native matching uses the canonical kernels in fks_singular.f.

      subroutine compute_born_buffered(ret_amp_split,wgt_c)
      use fks_phase_space_data, only: xi_i_hat_cnt
      use fks_phase_space_data, only: xi_i_hat_ev
      use fks_phase_space_data, only: xiimax_cnt
c This subroutine computes the Born matrix elements and adds its value
c to the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'genps.inc'
      integer i
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'born_leshouche.inc'
      integer orders(nsplitorders)
      integer iamp
      integer nFKSprocess,born_id1,born_id2
      common/c_nFKSprocess/nFKSprocess

      double precision ret_amp2(ngraphs), ret_jamp2(0:ncolor)
      complex*16 ret_ans_cnt(2,nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)

      include 'born_nhel.inc'
      double complex ret_saveamp(ngraphs,max_bhel)

      double precision wgt_c
      double precision wgt1
      double precision p_born(0:3,nexternal-1)
C      common /pborn/   p_born
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision g22
      integer get_orders_tag
      if (f_b.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return
      if(timing_sampled)call timing_enter(timing_born)
      do iamp=1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=ret_amp_split(iamp)*f_b/g**(qcd_power)
c     For UPC processes, we only need to fill the Born contribution for
c     photon-photon initial state
        if (HAS_PHYSICAL_FKS_CLASSES) then
          born_id1=BORN_PDG_TYPE_D(nFKSprocess,1)
          born_id2=BORN_PDG_TYPE_D(nFKSprocess,2)
        else
          born_id1=idup(1,1)
          born_id2=idup(2,1)
        endif
        if ((abs(lpp(1)).eq.2 .and. abs(lpp(2)).eq.2)
     &   .and. .not. (born_id1.eq.22 .and. born_id2.eq.22)) then
          cycle
        endif
        call add_wgt_buffered(2,orders,wgt1,0d0,0d0,wgt_c,0d0)
      enddo

      if(timing_sampled)call timing_leave(timing_born)
      return
      end



      subroutine compute_ewsudakov_buffered(p_born,wgt_c,ret_saveamp)
      use fks_phase_space_data, only: xi_i_hat_cnt
      use fks_phase_space_data, only: xi_i_hat_ev
      use fks_phase_space_data, only: xiimax_cnt
c This subroutine computes the NLO EW corrections in the Sudakov
c   approximation
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      include 'has_ewsudakov.inc'
      integer orders_qcd(nsplitorders), orders_ew(nsplitorders)
      integer iamp

      include 'genps.inc'
      include 'born_nhel.inc'
      double precision ret_amp2(ngraphs), ret_jamp2(0:ncolor)
      complex*16 ret_ans_cnt(2,nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
      double complex loc_saveamp(ngraphs,max_bhel)
      double precision saved_born

      double precision wgt_c
      double precision wgt1
      double precision p_born(0:3,nexternal-1)
C      common /pborn/   p_born
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision g22
      integer get_orders_tag

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double complex amp_split_ewsud_par(amp_split_size)
      common /to_amp_ewsud_par/amp_split_ewsud_par
      double complex amp_split_ewsud_qcd(amp_split_size)
      common /to_amp_ewsud_qcd/amp_split_ewsud_qcd
      double complex amp_split_ewsud_parqcd(amp_split_size)
      common /to_amp_ewsud_parqcd/amp_split_ewsud_parqcd
      ! sudakov mode
      integer sud_mod
      common /to_sud_mod/ sud_mod
      include 'ewsudakov_haslo.inc' 

      if (.not.has_ewsudakov) return

      if (f_b.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return
      if(timing_sampled)call timing_enter(timing_ewsudakov)
c The Born of sborn_amp_frame may come from the native provider, which
c returns no per-helicity amplitudes; the Sudakov kernel needs them.
      call sborn_amp_saved_frame(p_born,ret_amp2,ret_jamp2,ret_amp_split,
     $     ret_amp_split_cnt,saved_born,ret_ans_cnt,ret_saveamp)

      if (cpower_pos.gt.0) then
          write(*,*)'Error, cannot compute EW sudakov with Cpower >0'
          stop 1
      endif

      ! sud_mod = 0
      do sud_mod = 0,1

       loc_saveamp(:,:) = ret_saveamp(:,:)
       call sudakov_wrapper_frame(p_born,ret_saveamp)
       do iamp=1, amp_split_size
        if (amp_split_ewsud_lsc(iamp).eq.0d0.and.
     $      amp_split_ewsud_ssc(iamp).eq.0d0.and.
     $      amp_split_ewsud_xxc(iamp).eq.0d0.and.
     $      amp_split_ewsud_par(iamp).eq.0d0.and.
     $      AMP_SPLIT_EWSUD_QCD(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders_ew)
        orders_qcd(:) = orders_ew(:)
        ! we have two arrays of orders, one for the contributions
        ! of EW origin (from LO1) and one for those of QCD origin
        ! (from LO2)

        !!!! first the contribution of EW origin
        ! increase the EW-coupling of 2, since until here
        ! the EW sudakov amp_split has the same positions of 
        ! those for the Born
        if (has_lo1) then
          orders_ew(qed_pos)=orders_ew(qed_pos)+2
          QCD_power=orders_ew(qcd_pos)
          wgtcpower=0d0
          !!!!if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
          orders_tag=get_orders_tag(orders_ew)
          wgt1=(amp_split_ewsud_lsc(iamp)+
     $        amp_split_ewsud_ssc(iamp)+
     $        amp_split_ewsud_xxc(iamp)+
     $        amp_split_ewsud_par(iamp))
     $         *f_b/g**(qcd_power)
          wgt1=wgt1*2d0 ! missing factor in the sudakov correction
          ! the type will be 20+the value of the sudakov mode
          call add_wgt_buffered(20+sud_mod,orders_ew,wgt1,0d0,0d0,wgt_c,0d0)
        endif

        !!!! then the contribution of QCD origin
        ! increase the QCD-coupling of 2, since until here
        ! the EW sudakov amp_split has the same positions of 
        ! those for the Born, and for QCD this is LO2
        if (has_lo2) then
          orders_qcd(qcd_pos)=orders_qcd(qcd_pos)+2
          QCD_power=orders_qcd(qcd_pos)
          !!wgtcpower=0d0
          !!if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
          orders_tag=get_orders_tag(orders_qcd)
          wgt1=(amp_split_ewsud_qcd(iamp)+
     $        amp_split_ewsud_parqcd(iamp))
     $         *f_b/g**(qcd_power)
          wgt1=wgt1*2d0 ! missing factor in the sudakov correction
          ! the type will be 20+the value of the sudakov mode
          call add_wgt_buffered(20+sud_mod,orders_qcd,wgt1,0d0,0d0,wgt_c,0d0)
        endif
       enddo
      enddo
      if(timing_sampled)call timing_leave(timing_ewsudakov)
      return
      end




      subroutine compute_6to5flav_cnt_buffered(amp_split_6to5f,amp_split_6to5f_muf,amp_split_6to5f_mur
     &                               ,ret_amp_split)
      use fks_phase_space_data, only: p_born
C This is the counterterm for the 6f->5f scheme change 
C of parton distributions (e.g. NNPDF2.3). 
C It is called in this function such that if is included
C in the LO cross section
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      include 'coupl.inc' 
      include 'q_es.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'orders.inc'

      include 'born_nhel.inc'
      double complex ret_saveamp(ngraphs,max_bhel)

      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_6to5f(amp_split_size),
     &                 amp_split_6to5f_muf(amp_split_size),
     &                 amp_split_6to5f_mur(amp_split_size)
      integer orders_to_amp_split_pos
      integer niglu
      save niglu
      
      double precision ret_amp2(ngraphs), ret_jamp2(0:ncolor)
      complex*16 ret_ans_cnt(2,nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)

      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     $     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer i, j, k
      integer nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision tf, pi
      parameter (tf=0.5d0)
      parameter (pi=3.1415926535897932385d0)
      integer alphasbpow
      double precision wgtborn, alphas
      ! switch on/off here
      logical include_6to5_cnt 
      data include_6to5_cnt /.false./ 

c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]

      ! set everything to 0
      amp_split_6to5f(1:amp_split_size) = 0d0
      amp_split_6to5f_muf(1:amp_split_size) = 0d0
      amp_split_6to5f_mur(1:amp_split_size) = 0d0

      ! skip if we don't want this piece or if the scale is
      ! below mt
      if (.not.include_6to5_cnt.or.scale.lt.mdl_mt) return

C the contribution is the following (if mu > mt):
C      Add a term -alphas n TF/3pi log (muR^2/mt^2) sigma(0) 
C      where n is the power of alphas for the Born xsec sigma(0)
C      Add a term −alphas TF/3pi log (mt^2/muF^2) sigma(0) for each
C      gluon in the initial state

      niglu=0
      do i = 1, nincoming
          if (HAS_PHYSICAL_FKS_CLASSES) then
              if (BORN_PDG_TYPE_D(nFKSprocess,i).eq.21) niglu=niglu+1
          elseif (idup(i,1).eq.21) then
              niglu=niglu+1
          endif
      enddo

      alphas = g**2/4d0/pi
      do iamp = 1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        alphasbpow = orders(qcd_pos)/2
        if (niglu.ne.0 .or. alphasbpow.ne.0) then
          ! this contribution will end up with one extra power
          ! of alpha_s. Check that we are including the corresponding
          ! order in the computation, otherwise skip the contribution
          orders(qcd_pos) = orders(qcd_pos) + 2
          if (orders(qcd_pos).gt.nlo_orders(qcd_pos)) cycle

          amp_split_6to5f_muf(orders_to_amp_split_pos(orders)) = 
     &     alphas / 3d0 / pi * TF * dble(niglu) * ret_amp_split(iamp)  

          amp_split_6to5f_mur(orders_to_amp_split_pos(orders)) = 
     &    - alphas / 3d0 / pi * TF * dble(alphasbpow) * ret_amp_split(iamp) 

          amp_split_6to5f(orders_to_amp_split_pos(orders)) = 
     &    dlog(qes2/mdl_mt**2) * 
     &     (alphas / 3d0 / pi * TF * dble(niglu)   
     &    - alphas / 3d0 / pi * TF * dble(alphasbpow)) * ret_amp_split(iamp)
        endif
      enddo

      return
      end



      subroutine compute_alpha_cnt_buffered(amp_split_alpha,amp_split_alpha_muf,amp_split_alpha_mur,
     &                             ret_amp_split)
      use fks_phase_space_data, only: p_born
C This is the counterterm for the change of scheme
C in the UV renormalisation for alpha in (leptonic) PDFs
C wrt the hard matrix element. Relevant for lepton collisions. 
C It is called in this function such that if is included
C in the LO cross section
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc' 
      include 'q_es.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'orders.inc'
      include 'born_nhel.inc'
      
      double precision ret_amp2(ngraphs), ret_jamp2(0:ncolor)
      complex*16 ret_ans_cnt(2,nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
      
      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_alpha(amp_split_size),
     &                 amp_split_alpha_muf(amp_split_size),
     &                 amp_split_alpha_mur(amp_split_size)
      integer orders_to_amp_split_pos
      integer i, j, k
      logical firsttime
      data firsttime /.true./
      double precision tf, nc, pi
      parameter (tf=0.5d0)
      parameter (nc=3d0)
      parameter (pi=3.1415926535897932385d0)
      integer alphabpow
      double precision wgtborn, alpha
      double precision ch_lep, ch_up, ch_dn
      parameter (ch_lep=1d0)
      parameter (ch_up=2d0/3d0)
      parameter (ch_dn=1d0/3d0)
      ! the number of families
      integer n_lep, n_up, n_dn
      double precision sumcharge, beta0, const
      double precision w_thresh
      integer jsign

c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]

      ! set everything to 0
      amp_split_alpha(1:amp_split_size) = 0d0
      amp_split_alpha_muf(1:amp_split_size) = 0d0
      amp_split_alpha_mur(1:amp_split_size) = 0d0

      if (firsttime) then
        write(*,*) 'Parameters related to the different treatment of'
     ^     // ' alpha in ME and PDFs'
        write(*,*) 'alphascheme', alphascheme
        write(*,*) 'nlep_run   ', nlep_run
        write(*,*) 'nupq_run   ', nupq_run
        write(*,*) 'ndnq_run   ', ndnq_run
        write(*,*) 'w_run      ', w_run
        firsttime = .false.
      endif

      ! skip if we don't want this piece 
      if (alphascheme.eq.0) then
         ! do nothing, assumes same UV scheme in alpha for
         ! PDF's and model
         return
      else if (abs(alphascheme).eq.1) then
         ! the sign of alphascheme controls the sign of the counterterm.
         ! negative sign used for testing (in the test process, the
         ! virtuals are computed in the MSbar scheme, unlike in the
         ! processes automatically generated by MadGraph7
         jsign = sign(1,alphascheme)
         ! assumes alpha(MZ) for model, MSbar for PDFs
         ! the number of flavours depends on mur.
         ! here we will treat all leptons as massless, 
         ! blocking the code if mur < 5 gev
         if (nlep_run.eq.-1) then
            n_lep = 3
         else
            n_lep = nlep_run
         endif
         if (nupq_run.eq.-1) then
            n_up = 2
         else
            n_up = nupq_run
         endif
         if (ndnq_run.eq.-1) then
            n_dn = 3
         else
            n_dn = ndnq_run
         endif
         sumcharge = n_lep*ch_lep**2 + nc*(n_up*ch_up**2+n_dn*ch_dn**2)
         if (firsttime) then
            write(*,*) 'compute_alpha_cnt_buffered with nlep, nup, ndn, wrun', 
     #        n_lep, n_up, n_dn, w_run
            write(*,*) '  sum_{lep,up,dn} n*q^2*nc ', sumcharge 
            firsttime=.false.
          endif
         ! the factor has the form 
         ! alpha/3pi* (cons + beta0 log(mur^2/mz^2) * bpow
         ! where bpow is the power of alpha
         const = 5d0/3d0 * sumcharge - 
     #       w_run * (1d0/2d0 + 21d0/4d0 * dlog(mdl_mz**2/mdl_mw**2))
         ! in practice we have only the case mur > mw or mur<mw
         if (scale.gt.mdl_mw) then
            beta0 = sumcharge - 21d0/4d0 * w_run
            w_thresh = 0d0
         else if (scale.gt.5d0) then
            beta0 = sumcharge 
            w_thresh = - 21d0/4d0 * w_run
         else
            ! we hardcode 5d0 instead of MB as in the model the bottom
            ! is massless
            write(*,*) 'MUR too low, bottom threshold not implemented'
            stop 1
         endif
         alpha = dble(gal(1))**2/4d0/pi
         do iamp = 1, amp_split_size
           if (ret_amp_split(iamp).eq.0d0) cycle
           call amp_split_pos_to_orders(iamp, orders)
           alphabpow = orders(qed_pos)/2
           if (alphabpow.ne.0) then
             ! this contribution will end up with one extra power
             ! of alpha
             orders(qed_pos) = orders(qed_pos) + 2
             amp_split_alpha_muf(orders_to_amp_split_pos(orders)) = 0d0 
             amp_split_alpha_mur(orders_to_amp_split_pos(orders)) = 
     &           - jsign * alpha / 3d0 / pi * alphabpow * ret_amp_split(iamp) *
     &              beta0
             amp_split_alpha(orders_to_amp_split_pos(orders)) = 
     &           - jsign * alpha / 3d0 / pi * alphabpow * ret_amp_split(iamp) * 
     &            (beta0 * dlog(qes2/mdl_mz**2) + 
     &             w_thresh * dlog(mdl_mw**2/mdl_mz**2) + const)
           endif
         enddo
      else 
         write(*,*) 'change of scheme factors for gmu not implemented'
         stop 1
      endif

      return
      end





      subroutine compute_nbody_noborn_buffered(p,bsv_wgt,virt_wgt,born_wgt
     &           ,amp_split_virt,amp_split_born_for_virt,amp_split_avv
     &           ,amp_split_wgtnstmp,amp_split_wgtwnstmpmuf,amp_split_wgtwnstmpmur
     &           ,born_split,wgt_born)
      use fks_phase_space_data, only: xi_i_hat_cnt
      use fks_phase_space_data, only: xi_i_hat_ev
      use fks_phase_space_data, only: xiimax_cnt
c This subroutine computes the soft-virtual matrix elements and adds its
c value to the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp, i
      double precision born_split(amp_split_size)
      double precision amp_split_virt(amp_split_size),
     &     amp_split_born_for_virt(amp_split_size),
     &     amp_split_avv(amp_split_size)
      double precision amp_split_wgtnstmp(amp_split_size),
     $                 amp_split_wgtwnstmpmuf(amp_split_size),
     $                 amp_split_wgtwnstmpmur(amp_split_size)

      ! stuff for the 6->5 flav scheme
      double precision amp_split_6to5f(amp_split_size),
     &                 amp_split_6to5f_muf(amp_split_size),
     &                 amp_split_6to5f_mur(amp_split_size)

      ! stuff for the alpha UV-scheme in lepton collisions
      double precision amp_split_alpha(amp_split_size),
     &                 amp_split_alpha_muf(amp_split_size),
     &                 amp_split_alpha_mur(amp_split_size)

      double precision wgt6f1,wgt6f2,wgt6f3
      double precision wgtal1,wgtal2,wgtal3

      double precision wgt1,wgt2,wgt3,bsv_wgt,virt_wgt,born_wgt,pi,g2
     &     ,g22,wgt4,wgt_born
      parameter (pi=3.1415926535897932385d0)
      double precision p(0:3,nexternal)
   !    double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
   !   $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
   !    common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      integer get_orders_tag
      character*4      abrv
      common /to_abrv/ abrv
      integer iamp_test
      if (f_nb.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return
      if(timing_sampled)call timing_enter(timing_fks_integrated)
   !    call bornsoftvirtual_buffered(p1_cnt(0,1,0),bsv_wgt,virt_wgt,born_wgt
   !   $     ,amp_split_virt,amp_split_born_for_virt,amp_split_avv
   !   $     ,amp_split_wgtnstmp,amp_split_wgtwnstmpmuf,amp_split_wgtwnstmpmur)
      iamp_test=0
      do iamp=1, amp_split_size
        if (amp_split_wgtnstmp(iamp).eq.0d0.and.
     $      amp_split_wgtwnstmpmur(iamp).eq.0d0.and.
     $      amp_split_wgtwnstmpmuf(iamp).eq.0d0.and.
     $      amp_split_avv(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        wgt1=amp_split_wgtnstmp(iamp)*f_nb/g22
        wgt2=amp_split_wgtwnstmpmur(iamp)*f_nb/g22
        wgt3=amp_split_wgtwnstmpmuf(iamp)*f_nb/g22
        wgt4=amp_split_avv(iamp)*f_nb/g22
        if (ickkw.eq.3 .and. fxfx_exp_rewgt.ne.0d0
     &       .and. abrv.ne.'born') then
! This assumes a single Born order, which must always be the case for
! FxFx. Explicitly check this just to be sure.
           iamp_test=iamp_test+1
           if(iamp_test.ne.1) then
              write (*,*) "There should only be one possible"/
     $             /" Born order for FxFx"
              stop 1
           endif
           g2=g**(QCD_power-2)
           wgt1=wgt1 - fxfx_exp_rewgt*born_wgt*f_nb/g2/(4d0*pi)
        endif
        call add_wgt_buffered(3,orders,wgt1,wgt2,wgt3,wgt_born,0d0)
        call add_wgt_buffered(15,orders,wgt4,0d0,0d0,wgt_born,0d0)
      enddo
c Special for the soft-virtual needed for the virt-tricks. The
c *_wgt_mint variable should be directly passed to the mint-integrator
c and not be part of the plots nor computation of the cross section.
      virt_wgt_mint(0)=virt_wgt_mint(0)+virt_wgt*f_nb
      born_wgt_mint(0)=born_wgt_mint(0)+born_wgt*f_b
      do iamp=1, amp_split_size
        if (amp_split_virt(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=amp_split_virt(iamp)*f_nb
        virt_wgt_mint(iamp)=virt_wgt_mint(iamp)
     $       +wgt1
        born_wgt_mint(iamp)=born_wgt_mint(iamp)
     $       +amp_split_born_for_virt(iamp)*f_nb
        wgt1=wgt1/g**(QCD_power)
        call add_wgt_buffered(14,orders,wgt1,0d0,0d0,wgt_born,0d0)
      enddo

C This is the counterterm for the 6f->5f scheme change 
C of parton distributions (e.g. NNPDF2.3). 
      call compute_6to5flav_cnt_buffered(amp_split_6to5f,amp_split_6to5f_muf,amp_split_6to5f_mur,born_split)
      do iamp=1, amp_split_size
        if (amp_split_6to5f(iamp).eq.0d0.and.
     $      amp_split_6to5f_mur(iamp).eq.0d0.and.
     $      amp_split_6to5f_muf(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        g22=g**(QCD_power)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt6f1=amp_split_6to5f(iamp)*f_nb/g**(qcd_power)
        wgt6f2=amp_split_6to5f_mur(iamp)*f_nb/g**(qcd_power)
        wgt6f3=amp_split_6to5f_muf(iamp)*f_nb/g**(qcd_power)
        call add_wgt_buffered(3,orders,wgt6f1,wgt6f2,wgt6f3,wgt_born,0d0)
      enddo

C This is the counterterm for the change of scheme
C in the UV renormalisation for alpha in (leptonic) PDFs
C wrt the hard matrix element. Relevant for lepton collisions. 
      call compute_alpha_cnt_buffered(amp_split_alpha,amp_split_alpha_muf,amp_split_alpha_mur,born_split)
      do iamp=1, amp_split_size
        if (amp_split_alpha(iamp).eq.0d0.and.
     $      amp_split_alpha_mur(iamp).eq.0d0.and.
     $      amp_split_alpha_muf(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        g22=g**(QCD_power)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgtal1=amp_split_alpha(iamp)*f_nb/g**(qcd_power)
        wgtal2=amp_split_alpha_mur(iamp)*f_nb/g**(qcd_power)
        wgtal3=amp_split_alpha_muf(iamp)*f_nb/g**(qcd_power)
        call add_wgt_buffered(3,orders,wgtal1,wgtal2,wgtal3,wgt_born,0d0)
      enddo

      if(timing_sampled)call timing_leave(timing_fks_integrated)
      return
      end


      subroutine compute_real_emission_buffered(p,sudakov_damp,ret_amp_split,fx_ev)
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: y_ij_fks_ev
c This subroutine computes the real-emission matrix elements and adds
c its value to the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      use mint_module, only: fixed_order
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      include 'genps.inc'

      double precision ret_amp_split(amp_split_size)

      include 'run.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision s_ev,fks_Sij,p(0:3,nexternal),wgt1,fx_ev
     $     ,sudakov_damp,real_channel_soft_fraction
      external real_channel_soft_fraction
      external fks_Sij
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      integer get_orders_tag
      if (f_r.eq.0d0) return
      s_ev = fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      if (s_ev.le.0.d0) return
      if(timing_sampled)call timing_enter(timing_real)
C      call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx_ev,ret_amp_split)
      do iamp=1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=ret_amp_split(iamp)*s_ev*f_r/g**(qcd_power)
c MadFKS section 6.3: only the real event is damped. Local subtraction
c terms already have f=1 in their singular limits. Keep this separate
c from Sudakov damping, whose complement uses n-body kinematics.
        if (fixed_order.and.real_channels_x0.gt.0d0.and.
     $      real_channels_x0.lt.1d0) then
           wgt1=wgt1*real_channel_soft_fraction(p)
        endif
        if (sudakov_damp.gt.0d0) then
          call add_wgt_buffered(1,orders,wgt1*sudakov_damp,0d0,0d0,0d0,fx_ev)
        endif
        if (sudakov_damp.lt.1d0) then
          call add_wgt_buffered(11,orders,wgt1*(1d0-sudakov_damp),0d0,0d0,0d0,fx_ev)
        endif
      enddo
      if(timing_sampled)call timing_leave(timing_real)
      return
      end


      subroutine compute_soft_counter_term_buffered(replace_MC_subt,ret_amp_split,fx_s)
      use fks_phase_space_data, only: jac_cnt
      use fks_phase_space_data, only: p1_cnt
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: xi_i_hat_cnt
      use fks_phase_space_data, only: xi_i_hat_ev
      use fks_phase_space_data, only: xiimax_cnt
      use fks_phase_space_data, only: y_ij_fks_ev
c This subroutine computes the soft counter term and adds its value to
c the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'

      double precision ret_amp_split(amp_split_size)

      integer orders(nsplitorders)
      integer iamp
      double precision wgt1,s_s,fks_Sij,fx_s,zero,replace_MC_subt,g22
      parameter (zero=0d0)
      external fks_Sij
      double precision wgt_cnt(-2:2),pswgt_cnt(-2:2)
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      integer get_orders_tag
      if (f_s.eq.0d0 .and. f_s_MC_S.eq.0d0 .and. f_s_MC_H.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0).gt.xiScut_used .and. replace_MC_subt.eq.0d0)
     $     return
      s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
      if (s_s.le.0d0) return
      if(timing_sampled)call timing_enter(timing_fks_local)
C      call sreal(p1_cnt(0,1,0),0d0,y_ij_fks_ev,fx_s,ret_amp_split)

      do iamp=1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=ret_amp_split(iamp)*s_s/g22*replace_MC_subt
          call add_wgt_buffered(8,orders,-wgt1*f_s_MC_H,0d0,0d0,0d0,fx_s)
          wgt1=wgt1*f_s_MC_S
        else
          wgt1=0d0
        endif
        if (xi_i_fks_ev.le.xiScut_used) then
          wgt1=wgt1-ret_amp_split(iamp)*s_s*f_s/g22
        endif
        if (wgt1.ne.0d0) call add_wgt_buffered(4,orders,wgt1,0d0,0d0,0d0,fx_s)
      enddo

      if(timing_sampled)call timing_leave(timing_fks_local)
      return
      end


      subroutine compute_collinear_counter_term_buffered(replace_MC_subt,ret_amp_split,fx_c,
     &    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,
     &    amp_split_wgtdegrem_muF,
     &    amp_split_wgtpsch_p, amp_split_wgtpsch_l,
     &    amp_split_wgtpsch_d)
      use fks_phase_space_data, only: jac_cnt
      use fks_phase_space_data, only: p1_cnt
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: xi_i_fks_cnt
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: y_ij_fks_ev
c This subroutine computes the collinear counter term and adds its value
c to the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'fks_powers.inc'
      include 'timing_variables.inc'
      include 'orders.inc'

      double precision ret_amp_split(amp_split_size)

      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
   !    common /to_amp_split_deg/amp_split_wgtdegrem_xi,
   !   $                         amp_split_wgtdegrem_lxi,
   !   $                         amp_split_wgtdegrem_muF
      ! amp_split for the PDF scheme
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
   !    common /to_amp_split_dis/amp_split_wgtpsch_p,
   !   $                         amp_split_wgtpsch_l,
   !   $                         amp_split_wgtpsch_d
      double precision zero,one,s_c,fks_Sij,fx_c,deg_xi_c,deg_lxi_c,wgt1
     &     ,wgt3,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision wgt_cnt(-2:2),pswgt_cnt(-2:2)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision pmass(nexternal)
      integer get_orders_tag
      include 'pmass.inc'
      if (f_c.eq.0d0 .and. f_dc.eq.0d0 .and. f_c_MC_S.eq.0d0 .and.
     $     f_c_MC_H.eq.0d0)return
      if ( (y_ij_fks_ev.le.1d0-deltaS .and. replace_MC_subt.eq.0d0) .or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(1),one)
      if (s_c.le.0d0) return
      if(timing_sampled)call timing_enter(timing_fks_local)
      ! sreal_deg_buffered should be called **BEFORE** sreal 
      ! in order not to overwrtie the amp_split array
C      call sreal_deg_buffered(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,deg_xi_c
C     $     ,deg_lxi_c)
C      call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx_c,ret_amp_split)

      do iamp=1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_xi(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_lxi(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_p(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_l(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_d(iamp).eq.0d0) cycle

        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=ret_amp_split(iamp)*s_c/g22*replace_MC_subt
          call add_wgt_buffered(9,orders,-wgt1*f_c_MC_H,0d0,0d0,0d0,fx_c)
          wgt1=wgt1*f_c_MC_S
        else
          wgt1=0d0
        endif
        if (y_ij_fks_ev.gt.1d0-deltaS) then
          wgt1=wgt1-ret_amp_split(iamp)*s_c*f_c/g22
          wgt1=wgt1+
     $         (amp_split_wgtdegrem_xi(iamp)+amp_split_wgtpsch_p(iamp)+
     $         (amp_split_wgtdegrem_lxi(iamp)+amp_split_wgtpsch_l(iamp))
     $           *log(xi_i_fks_cnt(1)))*f_dc/g22
          wgt3=amp_split_wgtdegrem_muF(iamp)*f_dc/g22
        else
          wgt3=0d0
        endif
        if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt_buffered(5,orders,wgt1,0d0,wgt3,0d0,fx_c)
      enddo

      if(timing_sampled)call timing_leave(timing_fks_local)
      return
      end


      subroutine compute_soft_collinear_counter_term_buffered(replace_MC_subt, ret_amp_split,fx_sc,
     #    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,
     #    amp_split_wgtdegrem_muF,
     #    amp_split_wgtpsch_p, amp_split_wgtpsch_l,
     #    amp_split_wgtpsch_d)
      use fks_phase_space_data, only: jac_cnt
      use fks_phase_space_data, only: p1_cnt
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: xi_i_fks_cnt
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: xi_i_hat_cnt
      use fks_phase_space_data, only: xi_i_hat_ev
      use fks_phase_space_data, only: xiimax_cnt
      use fks_phase_space_data, only: y_ij_fks_ev
c This subroutine computes the soft-collinear counter term and adds its
c value to the list of weights using the add_wgt_buffered subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'fks_powers.inc'
      include 'timing_variables.inc'
      include 'orders.inc'

      double precision ret_amp_split(amp_split_size)

      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
   !    common /to_amp_split_deg/amp_split_wgtdegrem_xi,
   !   $                         amp_split_wgtdegrem_lxi,
   !   $                         amp_split_wgtdegrem_muF
      ! amp_split for the PDF scheme
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
   !    common /to_amp_split_dis/amp_split_wgtpsch_p,
   !   $                         amp_split_wgtpsch_l,
   !   $                         amp_split_wgtpsch_d
      double precision zero,one,s_sc,fks_Sij,fx_sc,wgt1,wgt3,deg_xi_sc
     $     ,deg_lxi_sc,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision wgt_cnt(-2:2),pswgt_cnt(-2:2)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      ! PDF scheme prefactors
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      double precision pmass(nexternal)
      integer get_orders_tag
      include 'pmass.inc'
      if (f_sc.eq.0d0 .and. f_dsc(1).eq.0d0 .and. f_dsc(2).eq.0d0 .and.
     $     f_dsc(3).eq.0d0 .and. f_dsc(4).eq.0d0 .and. f_sc_MC_S.eq.0d0
     $     .and. f_sc_MC_H.eq.0d0) return
      if ( ((xi_i_hat_ev*xiimax_cnt(1).ge.xiScut_used .or. y_ij_fks_ev.le.1d0
     $     -deltaS) .and. replace_MC_subt.eq.0d0).or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
      if (s_sc.le.0d0) return
      if(timing_sampled)call timing_enter(timing_fks_local)
      ! sreal_deg_buffered should be called **BEFORE** sreal 
      ! in order not to overwrtie the amp_split array
C      call sreal_deg_buffered(p1_cnt(0,1,2),zero,one, deg_xi_sc,deg_lxi_sc)
C      call sreal(p1_cnt(0,1,2),zero,one,fx_sc,ret_amp_split)

      do iamp=1, amp_split_size
        if (ret_amp_split(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_xi(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_lxi(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_p(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_l(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_d(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=-ret_amp_split(iamp)*s_sc/g22*replace_MC_subt
          call add_wgt_buffered(10,orders,-wgt1*f_sc_MC_H,0d0,0d0,0d0,fx_sc)
          wgt1=wgt1*f_sc_MC_S
        else
          wgt1=0d0
        endif
        if (xi_i_fks_cnt(1).lt.xiScut_used .and. 
     $      y_ij_fks_ev.gt.1d0-deltaS) then
          wgt1=wgt1+ret_amp_split(iamp)*s_sc*f_sc/g22
          wgt1=wgt1+
     $         (-(amp_split_wgtdegrem_xi(iamp)+amp_split_wgtpsch_p(iamp)+
     $           (amp_split_wgtdegrem_lxi(iamp)+amp_split_wgtpsch_l(iamp))
     $              *log(xi_i_fks_cnt(1)))*f_dsc(1)-
     $           (amp_split_wgtdegrem_xi(iamp)*f_dsc(2)+
     $            amp_split_wgtdegrem_lxi(iamp)*f_dsc(3))+
     $            amp_split_wgtpsch_d(iamp)*f_pdfsch_d+
     $            amp_split_wgtpsch_p(iamp)*f_pdfsch_p+
     $            amp_split_wgtpsch_l(iamp)*f_pdfsch_l)/g22
          wgt3=-amp_split_wgtdegrem_muF(iamp)*f_dsc(4)/g22
        else
          wgt3=0d0
        endif
        if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt_buffered(6,orders,wgt1,0d0,wgt3,0d0,fx_sc)
      enddo

      if(timing_sampled)call timing_leave(timing_fks_local)
      return
      end


      subroutine include_multichannel_enhance_buffered(imode,born_amp2,ev_amp2,norad_amp2)
      use fks_phase_space_data, only: jac_cnt
      use fks_phase_space_data, only: p1_cnt
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: p_born_coll
      use fks_phase_space_data, only: p_born_ev
      use fks_phase_space_data, only: p_born_norad
      use fks_phase_space_data, only: p_ev
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'orders.inc'
      include 'nFKSconfigs.inc'
      include 'timing_variables.inc'
      include 'born_nhel.inc'
      double precision xnoborn_cnt,xtot,wgt_c,enhance,enhance_real
     $     ,pas(0:3,nexternal)
      data xnoborn_cnt /0d0/
      integer inoborn_cnt,i,imode
      data inoborn_cnt /0/
      double precision p_born_used(0:3,nexternal-1)
      double precision wgt_cnt(-2:2),pswgt_cnt(-2:2)
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      integer            this_config
      common/to_mconfigs/this_config
      Double Precision amp2(ngraphs), jamp2(0:ncolor)
      double precision born_amp2(ngraphs), ev_amp2(ngraphs)
     $     , norad_amp2(ngraphs)
      complex*16 dummy_ans(2,nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
C      common/to_amps/  amp2,          jamp2
      double precision   diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal),granny_is_res
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical use_evpr
      common /to_use_evpr/use_evpr


c Compute the multi-channel enhancement factor 'enhance'.
      enhance=1.d0
      amp2(:) = 0d0
      jamp2(:) = 0d0
      if (p_born(0,1).gt.0d0) then
         amp2(:) = born_amp2(:)
      elseif(p_born(0,1).lt.0d0)then
         enhance=0d0
      endif
      if (enhance.eq.0d0)then
         xnoborn_cnt=xnoborn_cnt+1.d0
         if(log10(xnoborn_cnt).gt.inoborn_cnt)then
            write (*,*) 'WARNING: no Born momenta more than 10**',
     $           inoborn_cnt,'times'
            inoborn_cnt=inoborn_cnt+1
         endif
      else
         xtot=0d0
         if (mapconfig(0,0).eq.0) then
            write (*,*) 'Fatal error in compute_prefactor_nbody:'/
     &           /' no Born diagrams ',mapconfig,
     &           '. Check bornfromreal.inc'
            write (*,*) 'Is fks_singular compiled correctly?'
            stop 1
         endif
         do i=1, mapconfig(0,0)
            xtot=xtot+amp2(mapconfig(i,0))
         enddo
         if (xtot.ne.0d0) then
            enhance=amp2(mapconfig(this_config,0))/xtot
            enhance=enhance*diagramsymmetryfactor
         else
            enhance=0d0
         endif
      endif

      amp2(:) = 0d0
      jamp2(:) = 0d0

c In the case there is the special phase-space mapping for resonances,
C or when not doing event projection
c use the Born computed with those as the mapping.
      enhance_real=1.d0
      if (imode.eq.2) then
         ! if (granny_is_res) p_born_used(:,:) = p_born_ev(:,:) 
         ! if (.not.use_evpr) p_born_used(:,:) = p_born_norad(:,:) 
         if (use_evpr) then
            if(p_born_ev(0,1).gt.0d0) then
               amp2(:) = ev_amp2(:)
               call set_alphas(p_ev)
            else 
               enhance_real=enhance
            endif
         else 
            if(p_born_norad(0,1).gt.0d0) then
               amp2(:) = norad_amp2(:)
               call set_alphas(p_ev)
            else 
               enhance_real=enhance
            endif
         endif
c Compute the multi-channel enhancement factor 'enhance_real'.
         if (enhance_real.eq.0d0)then
            xnoborn_cnt=xnoborn_cnt+1.d0
            if(log10(xnoborn_cnt).gt.inoborn_cnt)then
               write (*,*) 'WARNING: no Born momenta more than 10**',
     $              inoborn_cnt,'times'
               inoborn_cnt=inoborn_cnt+1
            endif
         else
            xtot=0d0
            if (mapconfig(0,0).eq.0) then
               write (*,*) 'Fatal error in compute_prefactor_n1body,'/
     &              /' no Born diagrams ',mapconfig
     &              ,'. Check bornfromreal.inc'
               write (*,*) 'Is fks_singular compiled correctly?'
               stop 1
            endif
            do i=1, mapconfig(0,0)
               xtot=xtot+amp2(mapconfig(i,0))
            enddo
            if (xtot.ne.0d0) then
               enhance_real=amp2(mapconfig(this_config,0))/xtot
               enhance_real=enhance_real*diagramsymmetryfactor
            else
               enhance_real=0d0
            endif
         endif
      else
         enhance_real=enhance
      endif

      if (imode.eq.1) then
         f_b=      f_b      *enhance
         f_nb=     f_nb     *enhance
      elseif(imode.eq.2) then
         f_r=      f_r      *enhance_real
      elseif(imode.eq.4) then
         f_MC_S=   f_MC_S   *enhance
         f_MC_H=   f_MC_H   *enhance
      elseif(imode.eq.3) then
         f_s=      f_s      *enhance
         f_s_MC_S= f_s_MC_S *enhance
         f_S_MC_H= f_S_MC_H *enhance
         f_c=      f_c      *enhance
         f_c_MC_S= f_c_MC_S *enhance
         f_c_MC_H= f_c_MC_H *enhance
         f_dc=     f_dc     *enhance
         f_sc=     f_sc     *enhance
         f_sc_MC_S=f_sc_MC_S*enhance
         f_sc_MC_H=f_sc_MC_H*enhance
         f_dsc(1)= f_dsc(1) *enhance
         f_dsc(2)= f_dsc(2) *enhance
         f_dsc(3)= f_dsc(3) *enhance
         f_dsc(4)= f_dsc(4) *enhance
         f_pdfsch_d=  f_pdfsch_d  *enhance
         f_pdfsch_p=  f_pdfsch_p  *enhance
         f_pdfsch_l=  f_pdfsch_l  *enhance
      endif

      return
      end
      


      subroutine bornsoftvirtual_buffered(p,p_born,bsv_wgt,virt_wgt,born_wgt
     &           ,amp_split_virt,amp_split_born_for_virt,amp_split_avv
     &           ,amp_split_wgtnstmp,amp_split_wgtwnstmpmuf,amp_split_wgtwnstmpmur
     &           ,born_wgt_arg,born_cnt,born_split_cnt,born_saveamp,ret_amp_split
     &           ,virt_cache)
      use fks_phase_space_data, only: shat
      use fks_phase_space_data, only: sqrtshat
      use fks_phase_space_data, only: ybst_til_tocm
      use fks_phase_space_data, only: ybst_til_tolab
      use extra_weights
      use mint_module
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      include 'q_es.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge
      include "run.inc"
      include "fks_powers.inc"
      double precision p(0:3,nexternal),bsv_wgt,born_wgt,avv_wgt
      double precision pp(0:3,nexternal)
      
      double precision wgt1
      double precision born_wgt_arg,wgt_born
      double precision rwgt,Q,Ej,wgt,contr,eikIreg,m1l_W_finite_CDR
      double precision aso2pi, aeo2pi
      double precision shattmp,dot
      integer i,j,aj,m,n,k,iord,ipos_ord

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision xicut_used
      common /cxicut_used/xicut_used


      double precision pi
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      common/fks_colors/c,gamma,gammap,gamma_ph,gammap_ph
      double precision c_used, gamma_used, gammap_used
      double precision p_born(0:3,nexternal-1)
C      common/pborn/p_born
      double precision double,single,xmu2
      logical ComputePoles,fksprefact
      parameter (ComputePoles=.false.)
      parameter (fksprefact=.true.)

      double precision beta0,ren_group_coeff
      common/cbeta0/beta0,ren_group_coeff

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      double precision virt_fraction_inc
      data virt_fraction_inc /1d0/

      integer include_virt
      double precision vol3

c For tests of virtuals
      double precision xnormsv
      common/cxnormsv/xnormsv
      double precision vrat

      double precision virt_wgt,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

      double precision virtual_over_born
      common/c_vob/virtual_over_born

c timing statistics
      include "timing_variables.inc"

c For the MINT folding
      integer fold,ifold_counter
      common /cfl/fold,ifold_counter

      double precision pmass(nexternal),zero,tiny
      parameter (zero=0d0)
      parameter (tiny=1d-6)
      include 'orders.inc'
      include 'born_nhel.inc'
      logical firsttime
      data firsttime / .true. /
      logical need_color_links_used, need_charge_links_used
      data need_color_links_used / .false. /
      data need_charge_links_used / .false. /
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      double precision amp2(ngraphs), jamp2(0:ncolor)
      complex*16 ans_cnt(2, nsplitorders)
      complex*16 born_cnt(2, nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex born_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
      double complex born_saveamp(ngraphs,max_bhel)
      double precision oneo8pi2
      parameter(oneo8pi2 = 1d0/(8d0*pi**2))
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      INTEGER nFKSprocess, nFKSprocess_save, nFKSprocess_col, nFKSprocess_chg
      COMMON/c_nFKSprocess/nFKSprocess
      data nFKSprocess_col / 0 /
      data nFKSprocess_chg / 0 /
      integer nFKSprocess_col_used,nFKSprocess_chg_used,born_class
      logical need_color_links_bsv,need_charge_links_bsv
      double precision bsv_wgt_mufoqes, bsv_wgt_mufomur
      double precision contr_mufoqes, contr_mufomur
C to keep track of the various split orders
      integer iamp
      integer orders(nsplitorders)
      double precision soft_prefactor
      double precision amp_split_link(amp_split_size), wgt_link
      double precision amp_split_born(amp_split_size)
      double precision amp_split_bsv(amp_split_size)
      double precision amp_split_soft(amp_split_size)
      double precision amp_split_finite_ML(amp_split_size)
C Caller owns one cache per MC lane, retained across folds. FO always uses fold0.
      double precision virt_cache(0:amp_split_size)
      double precision amp_split_virt(amp_split_size),
     &      amp_split_born_for_virt(amp_split_size),
     &      amp_split_avv(amp_split_size)
      double precision amp_split_wgtnstmp(amp_split_size),
     $                 amp_split_wgtwnstmpmuf(amp_split_size),
     $                 amp_split_wgtwnstmpmur(amp_split_size)
      double precision coupl_wgtwnstmpmuf

      double precision amp_tot

      include "pmass.inc"
      
      if (firsttime) then
C check if any real emission need cahrge/color links
         nFKSprocess_save = nFKSprocess
         do nFKSprocess = 1, FKS_INTEGRATED
            call fks_inc_chooser()
            need_color_links_used = need_color_links_used .or. need_color_links
            need_charge_links_used = need_charge_links_used .or. need_charge_links
C keep track of which FKS configuration actually needs color/charge
C links
            if (need_color_links.and.nFKSprocess_col.eq.0)
     1          nFKSprocess_col = nFKSprocess
            if (need_charge_links.and.nFKSprocess_chg.eq.0)
     1          nFKSprocess_chg = nFKSprocess
         enddo
         if (need_charge_links_used) then
             write(*,*) 'Charge-linked born are used'
         else
             write(*,*) 'Charge-linked born are not used'
         endif
         if (need_color_links_used) then
             write(*,*) 'Color-linked born are used'
         else
             write(*,*) 'Color-linked born are not used'
         endif
         firsttime = .false.
         nFKSprocess = nFKSprocess_save
         call fks_inc_chooser()
      endif

c The integrated soft terms must use color/charge-linked Borns from the
c same physical underlying-Born class as the contribution being evaluated.
c The legacy output has one luminosity row for all of its physical flavours,
c so preserve its process-wide representatives there.
      if (HAS_PHYSICAL_FKS_CLASSES) then
         born_class=BORN_FLAVOR_INDEX_D(nFKSprocess)
         need_color_links_bsv=.false.
         need_charge_links_bsv=.false.
         nFKSprocess_col_used=0
         nFKSprocess_chg_used=0
         do i=1,FKS_INTEGRATED
            if (BORN_FLAVOR_INDEX_D(i).ne.born_class) cycle
            if (NEED_COLOR_LINKS_D(i).and.
     $           .not.need_color_links_bsv) then
               need_color_links_bsv=.true.
               nFKSprocess_col_used=i
            endif
            if (NEED_CHARGE_LINKS_D(i).and.
     $           .not.need_charge_links_bsv) then
               need_charge_links_bsv=.true.
               nFKSprocess_chg_used=i
            endif
         enddo
      else
         need_color_links_bsv=need_color_links_used
         need_charge_links_bsv=need_charge_links_used
         nFKSprocess_col_used=nFKSprocess_col
         nFKSprocess_chg_used=nFKSprocess_chg
      endif
         

      aso2pi=g**2/(8*pi**2)
      aeo2pi=dble(gal(1))**2/(8*pi**2)

      amp_split_bsv(1:amp_split_size)=0d0
      amp_split_virt(1:amp_split_size)=0d0
      amp_split_avv(1:amp_split_size)=0d0

      if (.not.(need_color_links_bsv.or.need_charge_links_bsv)) then
C just return 0
         bsv_wgt=0d0
         virt_wgt=0d0
         born_wgt=0d0
         goto 999
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
         write(*,*)'Error in bornsoftvirtual_buffered: inconsistent shat'
         write(*,*)shattmp,shat
         stop
      endif
      ans_cnt(:,:) = born_cnt(:,:)
      ret_amp_split_cnt(:,:,:) = born_split_cnt(:,:,:)
      ret_saveamp(:,:) = born_saveamp(:,:)
      wgt1 = born_wgt_arg

c Born contribution:
      bsv_wgt=wgt1
      born_wgt=wgt1
      virt_wgt=0d0
      avv_wgt=0d0 
      amp_split_born(1:amp_split_size)=ret_amp_split(1:amp_split_size)
      amp_split_bsv(1:amp_split_size)=ret_amp_split(1:amp_split_size)

      if (abrv.eq.'born') goto 549
      if (abrv.eq.'virt') goto 547

c Q contribution eq 5.5 and 5.6 of FKS
C loop over QCD/QED (iord=1,2 respectively)
      do iord= 1,2
         Q=0d0
C skip what we don't need
         ipos_ord = qcd_pos
         if (iord.eq.2) ipos_ord = qed_pos
         if (.not.split_type_used(ipos_ord)) cycle
         do i=1 ,nexternal
            if (i.ne.i_fks .and. pmass(i).eq.ZERO) then
c set the various color factors according to the 
c type of the leg
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               else
                  aj=-1
               endif
               Ej=p(0,i)
               
               if (iord.eq.1) then
C     set colour factors
                  if (aj.eq.-1) cycle
                  c_used = c(aj)
                  gamma_used = gamma(aj)
                  gammap_used = gammap(aj)
               else if (iord.eq.2) then
C     skip particles which are not photons or charged
                  if (particle_charge(i).eq.0d0.and.pdg_type(i).ne.22)
     $                 cycle
C     set charge factors
                  if (pdg_type(i).eq.22.and..not.particle_tag(i)) then
                     c_used = 0d0
                     gamma_used = gamma_ph
                     gammap_used = gammap_ph
                  else
                     c_used = particle_charge(i)**2
                     gamma_used = 3d0/2d0 * particle_charge(i)**2
                     gammap_used = (13d0/2d0 - 2d0 * pi**2 / 3d0) *
     $                    particle_charge(i)**2
                  endif
               endif

               if (i.gt.nincoming) then 
C Q terms for final state partons
                  if(abrv.ne.'virt')then
c 1+2+3+4
                     Q = Q+gammap_used
     &                    -dlog(shat*deltaO/2d0/QES2)*( gamma_used-
     &                    2d0*c_used*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &                    +2d0*c_used*( dlog(2d0*Ej/sqrtshat)**2
     &                    -dlog(xicut_used)**2 )
     &                    -2d0*gamma_used*dlog(2d0*Ej/sqrtshat)
                  else
                     write(*,*)'Error in bornsoftvirtual_buffered'
                     write(*,*)'abrv in Q:',abrv
                     stop
                  endif

               else
C Q terms for initial state partons
                  if(abrv.ne.'virt')then
c 1+2+3+4
                     Q=Q-dlog(q2fact(i)/QES2)*(
     &                    gamma_used+2d0*c_used*dlog(xicut_used))
                  else
                     write(*,*)'Error in bornsoftvirtual_buffered'
                     write(*,*)'abrv in Q:',abrv
                     stop
                  endif
               endif
            endif
         enddo
C end of the external particle loop
         if (iord.eq.1) then 
            bsv_wgt = bsv_wgt+aso2pi*Q*dble(ans_cnt(1,qcd_pos))
            amp_split_bsv(1:amp_split_size)=
     $           amp_split_bsv(1:amp_split_size)+aso2pi*Q
     $           *dble(ret_amp_split_cnt(1:amp_split_size,1,qcd_pos))
         endif
         if (iord.eq.2) then
            bsv_wgt = bsv_wgt+aeo2pi*Q*dble(ans_cnt(1,qed_pos))
            amp_split_bsv(1:amp_split_size)=
     $           amp_split_bsv(1:amp_split_size)+aeo2pi*Q
     $           *dble(ret_amp_split_cnt(1:amp_split_size,1,qed_pos))
         endif
      enddo

c     If doing MC over helicities, must sum over the two
c     helicity contributions for the Q-terms of collinear limit.
 547  continue
      if (abrv.eq.'virt') goto 548
c
c I(reg) terms, eq 5.5 of FKS
      nFKSprocess_save = nFKSprocess
      do iord = 1, nsplitorders
         if (iord.eq.qcd_pos) then
            if (.not. need_color_links_bsv) cycle
            need_color_links=need_color_links_bsv
            need_charge_links=.false.
            nFKSprocess=nFKSprocess_col_used
         else if (iord.eq.qed_pos) then
            if (.not. need_charge_links_bsv) cycle
            need_charge_links=need_charge_links_bsv
            need_color_links=.false.
            nFKSprocess=nFKSprocess_chg_used
         else
            cycle
         endif
C setup the fks i/j info
         call fks_inc_chooser()
C The Born counterterm values (ans_cnt and their split orders) depend on
C the FKS configuration, and the links are evaluated for nFKSprocess_col_used
C or nFKSprocess_chg_used, not for the configuration whose Born was passed
C in. Recompute them here, as the original code did with sborn(p_born,wgt1);
C reusing the passed born_cnt made the integrated soft term differ from MG5
C at almost every point (W+j, mixed QCD/QED). Temporaries keep the caller's
C ret_amp_split and the Born weight untouched.
         calculatedBorn = .false.
         call sborn_amp_frame(p_born,amp2,jamp2,amp_split_link,ret_amp_split_cnt,
     &        wgt_link,ans_cnt,ret_saveamp)
         contr=0d0
         do i=1,fks_j_from_i(i_fks,0)
            do j=1,i
               m=fks_j_from_i(i_fks,i)
               n=fks_j_from_i(i_fks,j)
               if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)).and.
     &              n.ne.i_fks.and.m.ne.i_fks) then
c To be sure that color-correlated Borns work well, we need to have
c *always* a call to sborn(p_born,wgt) just before. This is okay,
c because there is a call above in this subroutine
C wgt includes the gs/w^2
                  call sborn_sf_store(p_born,m,n,wgt,ans_cnt,ret_amp_split_cnt
     &                               ,ret_saveamp,amp_split_soft) 
                  if (wgt.ne.0d0) then
                     call eikonal_Ireg(p,m,n,xicut_used,eikIreg)
                     contr=contr+wgt*eikIreg
                     do k=1,amp_split_size
                        amp_split_bsv(k) = amp_split_bsv(k) - 2d0 *
     $                       eikIreg * oneo8pi2 * amp_split_soft(k)
                     enddo
                  endif
               endif
            enddo
         enddo

C WARNING: THE FACTOR -2 BELOW COMPENSATES FOR THE MISSING -2 IN THE
C COLOUR LINKED BORN -- SEE ALSO SBORNSOFT().
C If the colour-linked Borns were normalized as reported in the paper
c we should set
c   bsv_wgt=bsv_wgt+ao2pi*contr  <-- DO NOT USE THIS LINE
c
         bsv_wgt=bsv_wgt-2*oneo8pi2*contr
      enddo

C set back the fks i/j info as prior to enter this function
      nFKSprocess = nFKSprocess_save
      call fks_inc_chooser()
C and the Born values of that configuration (overwritten above)
      ans_cnt(:,:) = born_cnt(:,:)
      ret_saveamp(:,:) = born_saveamp(:,:)
      calculatedBorn = .false.

 548  continue
c Finite part of one-loop corrections
c convert to Binoth Les Houches Accord standards
      virt_wgt=0d0

      ret_amp_split_cnt(:,:,:) = born_split_cnt(:,:,:)
      ! use the amp_split_cnt as the born to approximate the virtual
      ! check which one of the two (QCD, QED) is !=0
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     THIS IS DANGEROUS: if these are not always the same for all
c     events, the whole virt_trics doesn't work and gives wrong results!
c     CHECK THIS.
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      do iamp=1, amp_split_size
         amp_split_virt(iamp)=0d0
         amp_split_born_for_virt(iamp)=0d0
         if (dble(ret_amp_split_cnt(iamp,1,qcd_pos)).ne.0d0) then
            amp_split_born_for_virt(iamp)=dble(ret_amp_split_cnt(iamp,1
     $           ,qcd_pos))
         else if (dble(ret_amp_split_cnt(iamp,1,qed_pos)).ne.0d0) then
            amp_split_born_for_virt(iamp)=dble(ret_amp_split_cnt(iamp,1
     $           ,qed_pos))
         endif
      enddo

      if (fold.eq.0) then
         if ((ran2().le.virtual_fraction(ichan) .and.
     $        abrv(1:3).ne.'nov').or.abrv(1:4).eq.'virt') then
            if(timing_sampled)call timing_enter(timing_virtual)
            Call binothlha_amp_frame(p_born,born_wgt,virt_wgt,
     $           ret_amp_split,ret_saveamp,amp_split_finite_ML,
     $           ret_amp_split_cnt)
            do iamp=1,amp_split_size
               amp_split_virt(iamp)=amp_split_finite_ML(iamp)
            enddo
            virtual_over_born=0d0
            if (born_wgt.ne.0d0) virtual_over_born=virt_wgt/born_wgt
            if (ickkw.ne.-1) then
               virt_wgt = 0d0
               do iamp=1,amp_split_size
                  if (amp_split_virt(iamp).eq.0d0) cycle
                  if (use_poly_virtual) then
                     amp_split_virt(iamp)=amp_split_virt(iamp)-
     $                    polyfit(iamp)
     $                    *amp_split_born_for_virt(iamp)
                  else
                     amp_split_virt(iamp)=amp_split_virt(iamp)-
     $                    average_virtual(iamp,ichan)
     $                     *amp_split_born_for_virt(iamp)
                  endif
                  virt_wgt = virt_wgt + amp_split_virt(iamp)
               enddo
            endif
            if (abrv.ne.'virt') then
               virt_wgt=virt_wgt/virtual_fraction(ichan)
               do iamp=1,amp_split_size
                  amp_split_virt(iamp)=amp_split_virt(iamp)
     &                 /virtual_fraction(ichan)
               enddo
            endif
            if(timing_sampled)call timing_leave(timing_virtual)
         endif
         virt_cache(0)=virt_wgt
         virt_cache(1:amp_split_size)=
     $        amp_split_virt(1:amp_split_size)
      elseif(fold.eq.1) then
         virt_wgt=virt_cache(0)
         amp_split_virt(1:amp_split_size)=
     $        virt_cache(1:amp_split_size)
      endif
      if (abrv(1:4).ne.'virt' .and. ickkw.ne.-1) then
         if (use_poly_virtual) then
            avv_wgt=polyfit(0)*born_wgt
            do iamp=1, amp_split_size
               if (amp_split_born_for_virt(iamp).eq.0d0) cycle
               amp_split_avv(iamp)= polyfit(iamp)
     $              *amp_split_born_for_virt(iamp)
            enddo
         else
            avv_wgt=average_virtual(0,ichan)*born_wgt
            do iamp=1, amp_split_size
               if (amp_split_born_for_virt(iamp).eq.0d0) cycle
               amp_split_avv(iamp)= average_virtual(iamp,ichan)
     $              *amp_split_born_for_virt(iamp)
            enddo
         endif
      endif

c eq.(MadFKS.C.13)
      if(abrv.ne.'virt')then
         bsv_wgt_mufoqes=0d0
         do iamp=1,amp_split_size
            if (dble(ret_amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
            call amp_split_pos_to_orders(iamp, orders)
            wgtcpower=0d0
            if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
            contr_mufoqes=2*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $           +ren_group_coeff*wgtcpower)*log(q2fact(1)/QES2)*aso2pi
     $           *dble(ret_amp_split_cnt(iamp,1,qcd_pos))
            amp_split_bsv(iamp) = amp_split_bsv(iamp)+contr_mufoqes
            bsv_wgt_mufoqes=bsv_wgt_mufoqes+contr_mufoqes
         enddo
         bsv_wgt=bsv_wgt+bsv_wgt_mufoqes
      endif

c  eq.(MadFKS.C.14)
      if(abrv(1:2).ne.'vi')then
         bsv_wgt_mufomur=0d0
         do iamp=1,amp_split_size
            if (dble(ret_amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
            call amp_split_pos_to_orders(iamp, orders)
            wgtcpower=0d0
            if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
            contr_mufomur=-2*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $           +ren_group_coeff*wgtcpower)*log(q2fact(1)/scale**2)
     $           *aso2pi*dble(ret_amp_split_cnt(iamp,1,qcd_pos))
            amp_split_bsv(iamp) = amp_split_bsv(iamp)+contr_mufomur
            bsv_wgt_mufomur=bsv_wgt_mufomur+contr_mufomur
         enddo
         bsv_wgt=bsv_wgt+bsv_wgt_mufomur
      endif

 549  continue

      wgtwnstmpmuf=0.d0
      wgtnstmp=0d0
      wgtwnstmpmur=0.d0
      amp_split_wgtnstmp(1:amp_split_size)=0d0
      amp_split_wgtwnstmpmuf(1:amp_split_size)=0d0
      amp_split_wgtwnstmpmur(1:amp_split_size)=0d0

      if(abrv.ne.'born' .and. abrv.ne.'grid')then
         if(abrv(1:2).eq.'vi')then
            wgtwnstmpmur=0.d0
         else
C loop over QCD/QED (iord=1,2 respectively)
            do iord= 1,2
C skip what we don't need
               ipos_ord = qcd_pos
               if (iord.eq.2) ipos_ord = qed_pos
               if (.not.split_type_used(ipos_ord)) cycle
               do i=1,nincoming
                  if (particle_type(i).eq.8) then
                     aj=0
                  elseif(abs(particle_type(i)).eq.3) then
                     aj=1
                  else
                     aj=-1
                  endif
                  if (iord.eq.1) then
C     set colour factors
                     if (aj.eq.-1) cycle
                     c_used = c(aj)
                     gamma_used = gamma(aj)
                     gammap_used = gammap(aj)
                  else if (iord.eq.2) then
C     skip particles which are not photons or charged
                     if (particle_charge(i).eq.0d0.and.pdg_type(i).ne.22)
     $                    cycle
C     set charge factors
                     if (pdg_type(i).eq.22.and..not.particle_tag(i)) then
                        c_used = 0d0
                        gamma_used = gamma_ph
                        gammap_used = gammap_ph
                     else
                        c_used = particle_charge(i)**2
                        gamma_used = 3d0/2d0 * particle_charge(i)**2
                        gammap_used = (13d0/2d0 - 2d0 * pi**2 / 3d0) *
     $                       particle_charge(i)**2
                     endif
                  endif
                  do iamp=1,amp_split_size
                     if (dble(ret_amp_split_cnt(iamp,1,ipos_ord)).eq.0d0)
     $                    cycle
                     if (iord.eq.1) then
                        coupl_wgtwnstmpmuf=aso2pi
                     else if (iord.eq.2) then
                        coupl_wgtwnstmpmuf=aeo2pi
                     endif
                     amp_split_wgtwnstmpmuf(iamp)
     $                    =amp_split_wgtwnstmpmuf(iamp)-(gamma_used+2d0
     $                    *c_used*dlog(xicut_used))
     $                    *dble(ret_amp_split_cnt(iamp,1,ipos_ord))
     $                    *coupl_wgtwnstmpmuf
                  enddo
               enddo            !end loop i=1,nincoming
            enddo               !end loop iord=1,2
            do iamp=1,amp_split_size
               if (dble(ret_amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
               call amp_split_pos_to_orders(iamp, orders)
               wgtcpower=0d0
               if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
               amp_split_wgtwnstmpmur(iamp)=dble(ret_amp_split_cnt(iamp,1
     $              ,qcd_pos))*2d0*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $              +ren_group_coeff*wgtcpower)*aso2pi
            enddo
         endif
c bsv_wgt here always contains the Born; must subtract it, since 
c we need the pure NLO terms only
         amp_split_wgtnstmp(1:amp_split_size) =
     $        amp_split_bsv(1:amp_split_size)
     $        -amp_split_born(1:amp_split_size)-log(q2fact(1)/QES2)
     $        *amp_split_wgtwnstmpmuf(1:amp_split_size)-log(scale**2
     $        /QES2)*amp_split_wgtwnstmpmur(1:amp_split_size)
      endif

      ret_amp_split(1:amp_split_size)=amp_split_bsv(1:amp_split_size)

      if (abrv(1:2).eq.'vi') then
         bsv_wgt=bsv_wgt-born_wgt
         born_wgt=0d0
      endif

      if (ComputePoles) then

         print*,"           "
         write(*,123)((p(i,j),i=0,3),j=1,nexternal)
         xmu2=q2fact(1)
         call getpoles(p,xmu2,double,single,fksprefact)
         print*,"BORN",born_wgt!/conv
         print*,"DOUBLE",double
         print*,"SINGLE",single
c         print*,"LOOP",virt_wgt!/born_wgt/ao2pi*2d0
c         print*,"LOOP2",(virtcor+born_wgt*4d0/3d0-double*pi**2/6d0)
c         stop
 123     format(4(1x,d22.16))
      endif

 999  continue
      return
      end




      subroutine sreal_store_buffered(pp,xi_i_fks,y_ij_fks,wgt,ret_amp_split,real_amp_split,
     &                 born_split,born_cnt,born_split_cnt,born_saveamp,
     &                 coll_split,coll_cnt,coll_split_cnt,coll_saveamp)
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: shat
      use fks_phase_space_data, only: sqrtshat
      use fks_phase_space_data, only: ybst_til_tocm
      use fks_phase_space_data, only: ybst_til_tolab
c Wrapper for the n+1 contribution. Returns the n+1 matrix element
c squared reduced by the FKS damping factor xi**2*(1-y).
c Close to the soft or collinear limits it calls the corresponding
c Born and multiplies with the AP splitting function or eikonal factors.
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include 'orders.inc'
      include 'genps.inc'
      include 'born_nhel.inc'

      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks

      double precision ret_amp_split(amp_split_size)

      double precision real_amp_split(amp_split_size)

      double precision shattmp,dot
      integer i,j

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks


      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision zero,tiny
      parameter (zero=0d0)
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links

      double precision amp2(ngraphs), jamp2(0:ncolor)
      double precision born_split(amp_split_size)
      complex*16 born_cnt(2, nsplitorders)
      double complex born_split_cnt(amp_split_size,2,nsplitorders)
      double complex born_saveamp(ngraphs,max_bhel)
      double precision coll_split(amp_split_size)
      complex*16 coll_cnt(2, nsplitorders)
      double complex coll_split_cnt(amp_split_size,2,nsplitorders)
      double complex coll_saveamp(ngraphs,max_bhel)

      double precision pmass(nexternal)
      include "pmass.inc"



      if (softtest.or.colltest) then
         tiny=1d-12
      else
         tiny=1d-6
      endif

      if(pp(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
        wgt=0.d0
        return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(pp(0,1),pp(0,2))
      else
         shattmp=pp(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      ret_amp_split(:) = 0d0

      if (1d0-y_ij_fks.lt.tiny)then
         ret_amp_split(:) = born_split(:)
         if (pmass(j_fks).eq.zero.and.j_fks.le.nincoming)then
            call sborncol_isr_store_buffered(pp,xi_i_fks,y_ij_fks,wgt
     &                       ,ret_amp_split,born_cnt,born_split_cnt
     &                       ,coll_split,coll_cnt,coll_split_cnt)
         elseif (pmass(j_fks).eq.zero.and.j_fks.ge.nincoming+1)then
            call sborncol_fsr_buffered(pp,xi_i_fks,y_ij_fks,wgt,ret_amp_split,born_cnt,born_split_cnt)
         else
            wgt=0d0
            ret_amp_split(1:amp_split_size) = 0d0
         endif
      elseif (xi_i_fks.lt.tiny)then
         if (need_color_links.or.need_charge_links)then
c has soft singularities
            ret_amp_split(:) = born_split(:)
            call sbornsoft_store_buffered(pp,xi_i_fks,y_ij_fks,wgt,ret_amp_split,
     $                           born_cnt,born_saveamp,born_split_cnt)
         else
            wgt=0d0
            ret_amp_split(1:amp_split_size) = 0d0
         endif
      else
C         call smatrix_real(pp,ret_amp_split,wgt)
         ret_amp_split(:) = real_amp_split(:)
         wgt=wgt*xi_i_fks**2*(1d0-y_ij_fks)
         ret_amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)*xi_i_fks**2*(1d0-y_ij_fks)
      endif

c      amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)

      return
      end




      subroutine sborncol_fsr_buffered(p,xi_i_fks,y_ij_fks,wgt,ret_amp_split,
     &                        ans_cnt, ret_amp_split_cnt)
      use fks_phase_space_data, only: resonance_momentum,
     $     resonance_mass2,resonance_recoil
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: shat
      use fks_phase_space_data, only: sqrtshat
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: xij_aor
      use fks_phase_space_data, only: xij_kperp
      use fks_phase_space_data, only: y_ij_fks_ev
      use fks_phase_space_data, only: ybst_til_tocm
      use fks_phase_space_data, only: ybst_til_tolab
      USE ALOHA_OBJECT
      implicit none
      include "nexternal.inc"
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
      double precision radiation_mass2,dot
      external dot
C  

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks




      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      integer i,j,imother_fks,iord
C ap and Q contain the QCD(1) and QED(2) Altarelli-Parisi kernel
      double precision t,z,ap(2),E_j_fks,E_i_fks,Q(2),cphi_mother,
     # sphi_mother,pi(0:3),pj(0:3),wgt_born
      TYPE(ALOHA) W1,W2,W3,W4
      double complex Wij_angle,Wij_recta
      double complex azifact
      logical me_boosted

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      integer j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     $     i_type,j_type,m_type,j_pdg

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      include 'orders.inc'
      include 'born_nhel.inc'
      double precision amp_split_local(amp_split_size)
      double complex amp_split_cnt_local(amp_split_size,2,nsplitorders)
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      double precision amp2(ngraphs), jamp2(0:ncolor)
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
c      common /c_born_cnt/ ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt

      double precision iden_comp
      double precision iden_comp_FKS(fks_configs)
      common/c_iden_comp/iden_comp,iden_comp_FKS
C  
      amp_split_local(1:amp_split_size) = 0d0
      
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_fsr_buffered"
         wgt=0.d0
         return
      endif

      radiation_mass2=shat
      E_j_fks = p(0,j_fks)
      E_i_fks = p(0,i_fks)
      if (resonance_recoil) then
         radiation_mass2=resonance_mass2
         E_i_fks=dot(resonance_momentum,p(0,i_fks))/sqrt(resonance_mass2)
         E_j_fks=dot(resonance_momentum,p(0,j_fks))/sqrt(resonance_mass2)
      endif
      z = 1d0 - E_i_fks/(E_i_fks+E_j_fks)
      t = z * radiation_mass2/4d0
      if (iextra_cnt.gt.0)
     1    call extra_cnt_frame(p_born, iextra_cnt, ans_extra_cnt)
      call AP_reduced(j_type,i_type,ch_j,ch_i,t,z,ap)
      call Qterms_reduced_timelike(j_type,i_type,ch_j,ch_i,t,z,Q)
      wgt=0d0
      do iord = 1, nsplitorders
         if (.not.split_type(iord) .or. (iord.ne.qed_pos .and.
     $        iord.ne.qcd_pos)) cycle
C check if any extra_cnt is needed
         if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
               wgt1(1) = ans_cnt(1,iord)
               wgt1(2) = ans_cnt(2,iord)
            elseif (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt_frame(p_born, iextra_cnt, ans_extra_cnt)
               wgt1(1) = ans_extra_cnt(1,iord)
               wgt1(2) = ans_extra_cnt(2,iord)
            else
               write(*,*) 'ERROR in sborncol_fsr_buffered', iord
               stop
            endif
         else
            wgt1(1) = ans_cnt(1,iord)
            wgt1(2) = ans_cnt(2,iord)
         endif
c The split orders of the counterterm Born: those of the Born (argument),
c or, for the extra counterterm (isplitorder_cnt), those that extra_cnt has
c just written to the amp_split_cnt common (as the summed wgt1 uses
c ans_extra_cnt). They are copied, so the caller's array is never rescaled.
         if (iextra_cnt.gt.0 .and. iord.eq.isplitorder_cnt) then
            amp_split_cnt_local(1:amp_split_size,1:2,iord)=
     $           amp_split_cnt(1:amp_split_size,1:2,iord)
         else
            amp_split_cnt_local(1:amp_split_size,1:2,iord)=
     $           ret_amp_split_cnt(1:amp_split_size,1:2,iord)
         endif
         if ((abs(j_type).eq.3 .and.i_type.eq.8) .or.
     #       (dabs(ch_j).ne.0d0 .and.ch_i.eq.0d0)) then
            Q(1)=0d0
            Q(2)=0d0
            wgt1(2)=0d0
         elseif (m_type.eq.8.or.ch_m.eq.0d0) then
c Insert <ij>/[ij] which is not included by sborn()
            imother_fks=min(i_fks,j_fks)
            call azifact_me_frame(p_born,imother_fks,p_i_fks_ev,
     &        p(0,j_fks),y_ij_fks,azifact,me_boosted)
            if (me_boosted) then
c Same as the ISR case, up to the conjugation that distinguishes a timelike
c from a spacelike splitting: psi is measured in the helicity basis of the
c boosted mother, so the getaziangles factor of the legacy branch below is
c already accounted for and must not be applied again.
               wgt1(2) = -azifact * wgt1(2)
               amp_split_cnt_local(1:amp_split_size,2,iord) = -azifact
     $              *amp_split_cnt_local(1:amp_split_size,2,iord)
            else
            if (1d0-y_ij_fks.lt.vtiny)then
               azifact=xij_aor
            else
               do i=0,3
                  if (xi_i_fks.lt.1d-8) then
                  pi(i)=p_i_fks_ev(i)
               else
                  pi(i)=p(i,i_fks)
               endif
                  pj(i)=p(i,j_fks)
               enddo
               CALL IXXXXX(pi ,ZERO ,+1,+1,1,W1)        
               CALL OXXXXX(pj ,ZERO ,-1,+1,1,W2)        
               CALL IXXXXX(pi ,ZERO ,-1,+1,1,W3)        
               CALL OXXXXX(pj ,ZERO ,+1,+1,1,W4)        
               Wij_angle=(0d0,0d0)
               Wij_recta=(0d0,0d0)
               do i=1,4
                  Wij_angle = Wij_angle + W1%W(i)*W2%W(i)
                  Wij_recta = Wij_recta + W3%W(i)*W4%W(i)
               enddo
               azifact=Wij_angle/Wij_recta
            endif
c Insert the extra factor due to Madgraph convention for polarization vectors
            call getaziangles(p_born(0,imother_fks),
     #                       cphi_mother,sphi_mother)
            wgt1(2) = -(cphi_mother-ximag*sphi_mother)**2 *
     #             wgt1(2) * azifact
            amp_split_cnt_local(1:amp_split_size,2,iord) =
     $           -(cphi_mother-ximag*sphi_mother)**2
     $           *amp_split_cnt_local(1:amp_split_size,2,iord) * azifact
            endif
         else
            write(*,*) 'FATAL ERROR in sborncol_fsr_buffered',i_type,j_type,i_fks
     $           ,j_fks
            stop 1
         endif
         if (iord.eq.qcd_pos) then
            wgt=wgt+dble(wgt1(1)*ap(1)+wgt1(2)*Q(1))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)*AP(1)
     $           +amp_split_cnt_local(1:amp_split_size,2,iord)*Q(1))
         endif
         if (iord.eq.qed_pos) then
            wgt=wgt+dble(wgt1(1)*ap(2)+wgt1(2)*Q(2))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)*AP(2)
     $           +amp_split_cnt_local(1:amp_split_size,2,iord)*Q(2))
         endif
      enddo
      wgt=wgt*iden_comp
      ret_amp_split(1:amp_split_size) = amp_split_local(1:amp_split_size)
     $     *iden_comp
c      amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)
      return
      end




      subroutine sborncol_isr_store_buffered(p,xi_i_fks,y_ij_fks,wgt,ret_amp_split,
     &                        ans_cnt, ret_amp_split_cnt,
     &                        coll_split, coll_cnt, coll_split_cnt)
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: p_born_coll
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: shat
      use fks_phase_space_data, only: sqrtshat
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: xij_aor
      use fks_phase_space_data, only: xij_kperp
      use fks_phase_space_data, only: y_ij_fks_ev
      use fks_phase_space_data, only: ybst_til_tocm
      use fks_phase_space_data, only: ybst_til_tolab
      USE ALOHA_OBJECT
      implicit none
      include "nexternal.inc"
      include 'nFKSconfigs.inc'
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  


      double precision p_born_used(0:3,nexternal-1)

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks




      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      integer j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     $     i_type,j_type,m_type,j_pdg

      integer i, j, iord
C ap and Q contain the QCD(1) and QED(2) Altarelli-Parisi kernel
      double precision t,z,ap(2),Q(2),cphi_mother,sphi_mother,
     $ pi(0:3),pj(0:3),wgt_born
      TYPE(ALOHA) W1,W2,W3,W4
      double complex Wij_angle,Wij_recta
      double complex azifact
      logical me_boosted

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      include 'orders.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      double precision amp_split_local(amp_split_size)
      double complex amp_split_cnt_local(amp_split_size,2,nsplitorders)
      integer iamp
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      double precision amp2(ngraphs), jamp2(0:ncolor)
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      complex*16 coll_cnt(2, nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double precision coll_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex coll_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
c      common /c_born_cnt/ ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt

      double precision iden_comp
      double precision iden_comp_FKS(fks_configs)
      common/c_iden_comp/iden_comp,iden_comp_FKS
      logical use_evpr
      common /to_use_evpr/use_evpr
C  
      amp_split_local(1:amp_split_size) = 0d0

C in the case of the collinear CT, use p_born_coll
C  (when not doing event projection). 
C For the soft-collinear one, use p_born
      if (xi_i_fks.gt.0d0.and..not.use_evpr) then
          p_born_used(:,:) = p_born_coll(:,:)
          ret_amp_split(:) = coll_split(:)
          ans_cnt(:,:) = coll_cnt(:,:)
          ret_amp_split_cnt(:,:,:) = coll_split_cnt(:,:,:)
      else ! if (xi_i_fks.eq.0d0) then
          p_born_used(:,:) = p_born(:,:)
      endif

      if(p_born_used(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_isr"
         wgt=0.d0
         return
      endif

      z = 1d0 - xi_i_fks
c sreal return {\cal M} of FKS except for the partonic flux 1/(2*s).
c Thus, an extra factor z (implicit in the flux of the reduced Born
c in FKS) has to be inserted here
      t = z*shat/4d0
      call AP_reduced(m_type,i_type,ch_m,ch_i,t,z,ap)
      call Qterms_reduced_spacelike(m_type,i_type,ch_m,ch_i,t,z,Q)
      wgt=0d0
      do iord = 1, nsplitorders
         if (.not.split_type(iord) .or. (iord.ne.qed_pos .and.
     $        iord.ne.qcd_pos)) cycle
C check if any extra_cnt is needed
         if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
            ! this is the contribution from the born ME
               wgt1(1:2) = ans_cnt(1:2,iord)
            else if (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt_frame(p_born_used, iextra_cnt, ans_extra_cnt)
               wgt1(1:2) = ans_extra_cnt(1:2,iord)
            else
               write(*,*) 'ERROR in sborncol_isr', iord
               stop
            endif
         else
            wgt1(1:2) = ans_cnt(1:2,iord)
        endif
c The split orders of the counterterm Born: those of the Born (argument),
c or, for the extra counterterm (isplitorder_cnt), those that extra_cnt has
c just written to the amp_split_cnt common (as the summed wgt1 uses
c ans_extra_cnt). They are copied, so the caller's array is never rescaled.
        if (iextra_cnt.gt.0 .and. iord.eq.isplitorder_cnt) then
           amp_split_cnt_local(1:amp_split_size,1:2,iord)=
     $          amp_split_cnt(1:amp_split_size,1:2,iord)
        else
           amp_split_cnt_local(1:amp_split_size,1:2,iord)=
     $          ret_amp_split_cnt(1:amp_split_size,1:2,iord)
        endif
        if (abs(m_type).eq.3.or.ch_m.ne.0d0) then
           Q(1)=0d0
           Q(2)=0d0
           wgt1(2)=dcmplx(0d0,0d0)
           amp_split_cnt_local(1:amp_split_size,2,iord)=dcmplx(0d0,0d0)
        else
c Insert <ij>/[ij] which is not included by sborn()
           call azifact_me_frame(p_born_used,j_fks,p_i_fks_ev,
     &        p(0,j_fks),y_ij_fks,azifact,me_boosted)
           if (me_boosted) then
C Match the scalar ISR contraction in the boosted mother's helicity basis.
              wgt1(2)=-dconjg(azifact)*wgt1(2)
              amp_split_cnt_local(1:amp_split_size,2,iord)=
     $            -dconjg(azifact)*amp_split_cnt_local(1:amp_split_size,2,iord)
           else
           if (1d0-y_ij_fks.lt.vtiny)then
              azifact=xij_aor
           else
              do i=0,3
                 if (xi_i_fks.lt.1d-8) then
                  pi(i)=p_i_fks_ev(i)
               else
                  pi(i)=p(i,i_fks)
               endif
                 pj(i)=p(i,j_fks)
              enddo
              if(j_fks.eq.2 .and. nincoming.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed
                 pi(1)=-pi(1)
                 pi(3)=-pi(3)
                 pj(1)=-pj(1)
                 pj(3)=-pj(3)
              endif
c Same spinors as in sborncol_isr (the ALOHA-object interface).
              CALL IXXXXX(pi ,ZERO ,+1,+1,1,W1)
              CALL OXXXXX(pj ,ZERO ,-1,+1,1,W2)
              CALL IXXXXX(pi ,ZERO ,-1,+1,1,W3)
              CALL OXXXXX(pj ,ZERO ,+1,+1,1,W4)
              Wij_angle=(0d0,0d0)
              Wij_recta=(0d0,0d0)
              do i=1,4
                 Wij_angle = Wij_angle + W1%W(i)*W2%W(i)
                 Wij_recta = Wij_recta + W3%W(i)*W4%W(i)
              enddo
              azifact=Wij_angle/Wij_recta
           endif
c Insert the extra factor due to Madgraph convention for polarization vectors
           cphi_mother=1.d0
           sphi_mother=0.d0
           wgt1(2) = -(cphi_mother+ximag*sphi_mother)**2 * wgt1(2) *
     $          dconjg(azifact)
           amp_split_cnt_local(1:amp_split_size,2,iord) = -(cphi_mother
     $          +ximag*sphi_mother)**2
     $          *amp_split_cnt_local(1:amp_split_size,2,iord) *
     $          dconjg(azifact)
           endif
        endif
        if (iord.eq.qcd_pos) then
            wgt=wgt+dble(wgt1(1)*ap(1)+wgt1(2)*Q(1))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)
     $           *AP(1)+amp_split_cnt_local(1:amp_split_size,2,iord)
     $           *Q(1))
         endif
         if (iord.eq.qed_pos) then
            wgt=wgt+dble(wgt1(1)*ap(2)+wgt1(2)*Q(2))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)
     $           *AP(2)+amp_split_cnt_local(1:amp_split_size,2,iord)
     $           *Q(2))
         endif
      enddo
      wgt=wgt*iden_comp
      ret_amp_split(1:amp_split_size) = amp_split_local(1:amp_split_size)
     $     *iden_comp
c      amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)
      return
      end



      subroutine sbornsoft_store_buffered(pp,xi_i_fks,y_ij_fks,wgt,ret_amp_split,
     $                     ans_cnt, ret_saveamp, ret_amp_split_cnt)
      use fks_phase_space_data, only: p_born
c ans_cnt: see sbornsoft.
      implicit none

      include "nexternal.inc"
      include 'nFKSconfigs.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "coupl.inc"
      include 'orders.inc'
      include 'genps.inc'
      include 'born_nhel.inc'

      complex*16 ans_cnt(2,nsplitorders)
      double precision amp2(ngraphs), jamp2(0:ncolor)
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE COMPLEX RET_AMP_SPLIT_CNT(AMP_SPLIT_SIZE,2,NSPLITORDERS)
      double complex local_split_cnt(AMP_SPLIT_SIZE,2,NSPLITORDERS)
      double complex ret_saveamp(ngraphs,max_bhel)

      integer m,n

      double precision softcontr,pp(0:3,nexternal),wgt,eik,xi_i_fks
     &     ,y_ij_fks
      double precision wgt1
      integer i,j,k 


      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision zero,pmass(nexternal)
      parameter(zero=0d0)

      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      integer ipos_ord
      double precision soft_prefactor
      double precision amp_split_soft(amp_split_size)

      double precision iden_comp
      double precision iden_comp_FKS(fks_configs)
      common/c_iden_comp/iden_comp,iden_comp_FKS
      include "pmass.inc"
C
C Reset the amp_split array
      ret_amp_split(1:amp_split_size) = 0d0
      local_split_cnt(:,:,:) = ret_amp_split_cnt(:,:,:)

      softcontr=0d0
      do i=1,fks_j_from_i(i_fks,0)
         do j=1,i
            m=fks_j_from_i(i_fks,i)
            n=fks_j_from_i(i_fks,j)
            if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)) .and.
     &           n.ne.i_fks.and.m.ne.i_fks) then
C wgt includes the gs/w^2
               call sborn_sf_store(p_born,m,n,wgt
     &                            ,ans_cnt,local_split_cnt,ret_saveamp,amp_split_soft)
               if (wgt.ne.0d0) then
                  call eikonal_reduced(pp,m,n,i_fks,j_fks,
     &                                 xi_i_fks,y_ij_fks,eik)
                  softcontr=softcontr+wgt*eik*iden_comp
                  ! update the amp_split array
                  if (need_color_links) ipos_ord = qcd_pos
                  if (need_charge_links) ipos_ord = qed_pos
                  ret_amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)
     $                - 2d0 * eik * amp_split_soft(1:amp_split_size)*iden_comp
               endif
            endif
         enddo
      enddo
      wgt=softcontr
c Add minus sign to compensate the minus in the color factor
c of the color-linked Borns (b_sf_0??.f)
c Factor two to fix the limits.
      wgt=-2d0*wgt
c      amp_split(1:amp_split_size) = ret_amp_split(1:amp_split_size)
      return
      end




      subroutine sreal_deg_buffered(p,xi_i_fks,y_ij_fks,
     #                     collrem_xi,collrem_lxi,
     #    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,
     #    amp_split_wgtdegrem_muF,
     #    amp_split_wgtpsch_p, amp_split_wgtpsch_l,
     #    amp_split_wgtpsch_d,
     #    ans_cnt,ret_amp_split_cnt,
     #    coll_ans_cnt,coll_split_cnt)
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: p_born_coll
      use fks_phase_space_data, only: shat
      use fks_phase_space_data, only: sqrtshat
      use fks_phase_space_data, only: ybst_til_tocm
      use fks_phase_space_data, only: ybst_til_tolab
      use extra_weights
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include "coupl.inc"
      include 'q_es.inc'
      include "run.inc"
      include "orders.inc"
      include "born_nhel.inc"

      integer iord, iap
      double precision p(0:3,nexternal),collrem_xi,collrem_lxi
      double precision xi_i_fks,y_ij_fks
      double precision collrem_xi_tmp, collrem_lxi_tmp

      double precision wgt_born


      double precision p_born_used(0:3,nexternal-1)

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks


      double precision delta_used
      common /cdelta_used/delta_used

      double precision rwgt,shattmp,dot,born_wgt,oo2pi,z,t,ap(2),
     # apprime(2),xkkernp(2),xkkernd(2),xkkernl(2),xnorm
      external dot

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i, ch_j, ch_m
      integer j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     $     i_type,j_type,m_type,j_pdg
      double precision amp2(ngraphs), jamp2(0:ncolor)
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      complex*16 coll_ans_cnt(2, nsplitorders), used_cnt(2, nsplitorders)
      double precision ret_amp_split(amp_split_size)
      double complex ret_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex coll_split_cnt(amp_split_size,2,nsplitorders)
      double complex used_split_cnt(amp_split_size,2,nsplitorders)
      double complex ret_saveamp(ngraphs,max_bhel)
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      
      double precision one,pi
      parameter (one=1.d0)
      parameter (pi=3.1415926535897932385d0)
      double precision iden_comp
      double precision iden_comp_FKS(fks_configs)
      common/c_iden_comp/iden_comp,iden_comp_FKS
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical check_born_amps, born_checks_amps
      common/c_born_amps_check/check_born_amps,born_checks_amps
      logical discard_born

C keep track of each split orders
      integer iamp
      double precision amp_split_collrem_xi(amp_split_size), 
     $                 amp_split_collrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
      double precision prefact_xi

      logical use_evpr
      common /to_use_evpr/use_evpr

      logical firsttime_pdf
      data firsttime_pdf /.true./

      ! This is needed for the PDFscheme variable 
      include "../../Source/PDF/pdf.inc"
      ! PDFscheme = : 
      ! 0 -> msbar
      ! 1 -> dis (hadronic)
      ! 2 -> eta (leptonic)
      ! 3 -> beta (leptonic)
      ! 4 -> mixed (leptonic)
      ! 5 -> nobeta (leptonic)
      ! 6 -> delta (leptonic)
      ! 7 -> UPC
      if(firsttime_pdf) then
        write(*,*) 'PDFscheme' , pdfscheme
        firsttime_pdf = .false.
      endif

      amp_split_collrem_xi(1:amp_split_size) = 0d0
      amp_split_collrem_lxi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_xi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_lxi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_muF(1:amp_split_size) = 0d0
      amp_split_wgtpsch_p(1:amp_split_size) = 0d0
      amp_split_wgtpsch_l(1:amp_split_size) = 0d0
      amp_split_wgtpsch_d(1:amp_split_size) = 0d0

C in the case of the collinear CT, use p_born_coll
C  (when not doing event projection). 
C For the soft-collinear one, use p_born
      if (xi_i_fks.gt.0d0.and..not.use_evpr) then
          p_born_used(:,:) = p_born_coll(:,:)
          used_split_cnt(:,:,:) = coll_split_cnt(:,:,:)
          used_cnt(:,:) = coll_ans_cnt(:,:)
      else ! if (xi_i_fks.eq.0d0) then
          p_born_used(:,:) = p_born(:,:)
          used_split_cnt(:,:,:) = ret_amp_split_cnt(:,:,:)
          used_cnt(:,:) = ans_cnt(:,:)
      endif

      if(j_fks.gt.nincoming)then
c Do not include this contribution for final-state branchings
         collrem_xi=0.d0
         collrem_lxi=0.d0
         if(doreweight)then
           wgtdegrem_xi=0.d0
           wgtdegrem_lxi=0.d0
           wgtdegrem_muF=0.d0
         endif
         return
      endif

      if(p_born_used(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sreal_deg_buffered"
         collrem_xi=0.d0
         collrem_lxi=0.d0
         if(doreweight)then
           wgtdegrem_xi=0.d0
           wgtdegrem_lxi=0.d0
           wgtdegrem_muF=0.d0
         endif
         return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

c A factor gS^2 is included in the Altarelli-Parisi kernels
      oo2pi=one/(8d0*PI**2)

      z = 1d0 - xi_i_fks
      t = one
      call AP_reduced(m_type,i_type,ch_m,ch_i,t,z,ap)
      call AP_reduced_prime(m_type,i_type,ch_m,ch_i,t,z,apprime)

      ! call the PDF-scheme kernels here 
      !   p-> [1/(1-z)]_+  
      !   l-> [log(1-z)/(1-z)]_+  
      !   d-> delta(1-z)
      call xkplus(PDFscheme,m_type,i_type,ch_m,ch_i,z,xkkernp)
      call xkdelta(PDFscheme,m_type,i_type,ch_m,ch_i,xkkernd)
      call xklog(PDFscheme,m_type,i_type,ch_m,ch_i,z,xkkernl)

      collrem_xi=0.d0
      collrem_lxi=0.d0
      do iord = 1, nsplitorders
        if (.not.split_type(iord).or.(iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle

C check if any extra_cnt is needed
        if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
            ! this is the contribution from the born ME
               wgt1(1) = used_cnt(1,iord)
               wgt1(2) = used_cnt(2,iord)
            else if (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt_frame(p_born_used,iextra_cnt,ans_extra_cnt)
               wgt1(1) = ans_extra_cnt(1,iord)
               wgt1(2) = ans_extra_cnt(2,iord)
c its split orders, which extra_cnt writes to the amp_split_cnt common
               used_split_cnt(1:amp_split_size,1:2,iord) =
     $              amp_split_cnt(1:amp_split_size,1:2,iord)
            else
               write(*,*) 'ERROR in sreal_deg_buffered', iord
               stop
            endif
        else
           wgt1(1) = used_cnt(1,iord)
           wgt1(2) = used_cnt(2,iord)
        endif
        
        if (iord.eq.qcd_pos) iap = 1
        if (iord.eq.qed_pos) iap = 2
        collrem_xi_tmp=ap(iap)*log(shat*delta_used/(2*q2fact(j_fks))) -
     #           apprime(iap) 
        collrem_lxi_tmp=2*ap(iap)

c The partonic flux 1/(2*s) is inserted in genps. Thus, an extra 
c factor z (implicit in the flux of the reduced Born in FKS) 
c has to be inserted here
        xnorm=1.d0/z *iden_comp

        collrem_xi=collrem_xi + oo2pi*dble(wgt1(1))*collrem_xi_tmp*
     &       xnorm
        collrem_lxi=collrem_lxi + oo2pi*dble(wgt1(1))*collrem_lxi_tmp*
     &       xnorm

        amp_split_collrem_xi(1:amp_split_size) = amp_split_collrem_xi(1:amp_split_size)+ 
     &   dble(used_split_cnt(1:amp_split_size,1,iord))*oo2pi*collrem_xi_tmp*xnorm
        amp_split_collrem_lxi(1:amp_split_size) = amp_split_collrem_lxi(1:amp_split_size)+
     &   dble(used_split_cnt(1:amp_split_size,1,iord))*oo2pi*collrem_lxi_tmp*xnorm

        prefact_xi=ap(iap)*log(shat*delta_used/(2*QES2)) -
     &               apprime(iap)
        amp_split_wgtdegrem_xi(1:amp_split_size) = amp_split_wgtdegrem_xi(1:amp_split_size)+
     &   oo2pi*dble(used_split_cnt(1:amp_split_size,1,iord))*prefact_xi*xnorm
        amp_split_wgtdegrem_lxi(1:amp_split_size) = amp_split_collrem_lxi(1:amp_split_size)
        amp_split_wgtdegrem_muF(1:amp_split_size) = amp_split_wgtdegrem_muF(1:amp_split_size)-
     &   oo2pi*dble(used_split_cnt(1:amp_split_size,1,iord))*ap(iap)*xnorm
        ! amp split for the PDF scheme
        if (PDFscheme.ne.0) then
          amp_split_wgtpsch_p(1:amp_split_size) = amp_split_wgtpsch_p(1:amp_split_size) - 
     $     dble(used_split_cnt(1:amp_split_size,1,iord))*xkkernp(iap)*oo2pi*xnorm
          amp_split_wgtpsch_l(1:amp_split_size) = amp_split_wgtpsch_l(1:amp_split_size) - 
     $     dble(used_split_cnt(1:amp_split_size,1,iord))*xkkernl(iap)*oo2pi*xnorm
          amp_split_wgtpsch_d(1:amp_split_size) = amp_split_wgtpsch_d(1:amp_split_size) - 
     $     dble(used_split_cnt(1:amp_split_size,1,iord))*xkkernd(iap)*oo2pi*xnorm
        endif

      enddo
C      calculatedborn=.false.

      return
      end



      subroutine add_wgt_buffered(type,orders,w1,w2,w3,wb,wr)
      implicit none
      include 'orders.inc'
      integer type,orders(nsplitorders)
      double precision w1,w2,w3,wb,wr,wgt_ME_born,wgt_ME_real
      common/c_wgt_ME_tree/wgt_ME_born,wgt_ME_real
      wgt_ME_born=wb
      wgt_ME_real=wr
      call add_wgt(type,orders,w1,w2,w3)
      end


      subroutine refresh_born_cache()
c Before a Born evaluation whose momenta or couplings may have changed: let
c the Born reuse its cached amplitudes only after checking them (identical
c momenta, and identical couplings or couplings differing only through G,
c in which case the amplitudes are rescaled; see SBORN). The check needs
c the couplings to be recorded by born.f (born_checks_amps) and a cache
c filled for the current FKS configuration, since the summed helicities
c depend on it. Otherwise discard the cached amplitudes, as was always done.
      implicit none
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical check_born_amps, born_checks_amps
      common/c_born_amps_check/check_born_amps,born_checks_amps
      logical born_cache_ok
      integer born_cache_nfks
      common/c_born_cache/born_cache_ok,born_cache_nfks
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      if (born_checks_amps .and. born_cache_ok .and.
     &     nfksprocess.eq.born_cache_nfks) then
         check_born_amps=.true.
      else
         calculatedBorn=.false.
      endif
      end
