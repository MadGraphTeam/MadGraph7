      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      use process_module
      implicit none
C
C     CONSTANTS
C
      include 'nexternal.inc'
      include 'genps.inc'
C
C     LOCAL
C
      integer i,j,kchan
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
      
c     Vegas stuff
      integer         nndim
      common/tosigint/nndim

      real*8 sigint
      external sigint

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      double precision xratmax
      integer i_momcmp_count
      common/ccheckcnt/xratmax,i_momcmp_count

      character*4      abrv
      common /to_abrv/ abrv
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt

      logical Hevents
      common/SHevents/Hevents
      character*10 dum
c statistics for MadLoop      
      INTEGER NINJARS_DP_USAGE,NINJARS_QP_USAGE
      COMMON/NINJARSSTATS/NINJARS_DP_USAGE,NINJARS_QP_USAGE
      INTEGER TIRUST_DP_USAGE,TIRUST_QP_USAGE
      COMMON/TIRUSTSTATS/TIRUST_DP_USAGE,TIRUST_QP_USAGE
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      include 'pineappl_common.inc'
      integer orders(nsplitorders)

c timing statistics
      double precision job_started,job_finished,tTot

c PineAPPL
      logical pineappl
      common /for_pineappl/ pineappl

      logical useitmax
      common/cuseitmax/useitmax
      include 'born_nhel.inc'
C-----
C  BEGIN CODE
C-----
c Write the process PID in the log.txt files (i.e., to the screen)
      write (*,*) getpid()
      
      useitmax=.false. ! to be overwritten in open_output_files.f if need be
c
c     Setup the timing variable
c
      call timing_reset()
      call cpu_time(job_started)
      fixed_order=.true.
      nlo_ps=.false.

c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      min_virt_fraction_mint=min_virt_fraction
      do kchan=1,maxchannels
         do i=0,n_ave_virt
            average_virtual(i,kchan)=0d0
         enddo
         virtual_fraction(kchan)=max(virt_fraction,min_virt_fraction)
      enddo
      n_ord_virt=amp_split_size
      
c
c     Read process number
c
      ntot=0
      nsun=0
      nsps=0
      nups=0
      neps=0
      n100=0
      nddp=0
      nqdp=0
      nini=0
      n10=0
      do i=0,9
        n1(i)=0
      enddo
      NINJARS_DP_USAGE=0
      NINJARS_QP_USAGE=0
      TIRUST_DP_USAGE=0
      TIRUST_QP_USAGE=0
      
      call setrun                !Sets up run parameters
      if (ickkw.eq.-1) then
         write(*,*) 'VETO XSec mode is not supported by current main'
         stop 1
      endif
      call configure_fxfx_sudakov(ickkw,nlo_ps,shower_mc,.true.)
      call print_fks_isr_mapping(nlo_ps,shower_mc)
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of parameters
      call run_printout          !Prints out a summary of the run settings
      call fill_configurations_common
      call check_amp_split 
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncalls0,itmax,imode)
      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      nndim=ndim
c Don't proceed if muF1#muF2 (we need to work out the relevant formulae
c at the NLO)
      if( ( fixed_fac_scale .and.
     #       (muF1_over_ref*muF1_ref_fixed) .ne.
     #       (muF2_over_ref*muF2_ref_fixed) ) .or.
     #    ( (.not.fixed_fac_scale) .and.
     #      muF1_over_ref.ne.muF2_over_ref ) )then
        write(*,*)'NLO computations require muF1=muF2'
        stop
      endif
      write(*,*) "about to integrate ", ndim,ncalls0,itmax
c PineAPPL
      if (imode.eq.0) pineappl=.False. ! overwrite when starting completely fresh
      if(pineappl) then
         write(6,*) "Initializing PineAPPL ..."
c     Set flavor map, starting from all possible
c     parton lumi configurations defined in initial_states_map.dat
         call setup_flavourmap
c     Fill the number of combined matrix elements for given initial state luminosity
         call find_iproc_map
         write(6,*) "   ... done."
      endif
      if (abrv(1:4).eq.'virt') then
         only_virt=.true.
      else
         only_virt=.false.
      endif
c     Prepare the MINT folding
      do j=1,ndimmax
         if (j.le.ndim) then
            ifold(j)=1
         else
            ifold(j)=0
         endif
      enddo
      ifold_energy=ndim-2
      ifold_yij=ndim-1
      ifold_phi=ndim
c

      ! initialise the global, but process dependent, information in the process module.
      call init_process_module_global(shower_mc,abrv,nexternal,nincoming
     $     ,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow,ickkw)
      ! Also put all the n-body process dependent stuff here. It does
      ! not depend on PS point or FKS config, so all global information.
      call init_process_module_nbody_wrapper()


      
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.
      call addfil(dum)
      if (imode.eq.-1.or.imode.eq.0) then
         if(imode.eq.0)then
c Don't safe the reweight information when just setting up the grids.
            doreweight=.false.
            do_rwgt_scale=.false.
            do_rwgt_pdf=.false.
         else
            doreweight=do_rwgt_scale.or.do_rwgt_pdf.or.store_rwgt_info
         endif
c
         write (*,*) 'imode is ',imode

         if (ickkw.eq.-1) then
            min_virt_fraction=1d0
            do kchan=1,nchans
               virtual_fraction(kchan)=1d0
            enddo
         endif
c A grouped real contribution can survive failed Born phase-space
c generation. Book histograms before either contribution is evaluated.
c PineAPPL needs the coupling orders before the analysis books its grids.
         if (pineappl) then
            appl_amp_split_size=amp_split_size
            do j=1,amp_split_size
               call amp_split_pos_to_orders(j,orders)
               appl_qcdpower(j)=orders(qcd_pos)
               appl_qedpower(j)=orders(qed_pos)
            enddo
         endif
         call initplot
         call mint(sigint)
         call topout
         call deallocate_weight_lines
c
      else
         write (*,*) 'Unknown imode',imode
         stop
      endif

      if (ntot.ne.0) then
         write(*,*) "Statistics from MadLoop:"
         write(*,*)
     &        "  Total points tried:                              ",ntot
         write(*,*)
     &        "  Stability unknown:                               ",nsun
         write(*,*)
     &        "  Stable PS point:                                 ",nsps
         write(*,*)
     &        "  Unstable PS point (and rescued):                 ",nups
         write(*,*)
     &        "  Exceptional PS point (unstable and not rescued): ",neps
         write(*,*)
     &        "  Double precision used:                           ",nddp
         write(*,*)
     &        "  Quadruple precision used:                        ",nqdp
         write(*,*)
     &        "  Initialization phase-space points:               ",nini
         write(*,*)
     &        "  Unknown return code (100):                       ",n100
         write(*,*)
     &        "  Unknown return code (10):                        ",n10
         write(*,*)
     &        "  Unit return code distribution (1):               "
         do j=0,9
           if (n1(j).ne.0) then
              write(*,*) "#Unit ",j," = ",n1(j)
           endif
         enddo
         write(*,*) '#ReturnCodeVersion = 2'
         write(*,*) '#NinjaRS DP = ',NINJARS_DP_USAGE
         write(*,*) '#NinjaRS QP = ',NINJARS_QP_USAGE
         write(*,*) '#TIRust DP = ',TIRUST_DP_USAGE
         write(*,*) '#TIRust precision note: DP denotes input'
     &        //' precision; tensor routes can use higher precision.'
         write(*,*) '#TIRust QP = ',TIRUST_QP_USAGE
      endif

      call cpu_time(job_finished)
      tTot=job_finished-job_started
      call timing_report(tTot)

      open (unit=12, file='res.dat',status='unknown')
      do kchan=0,nchans
         write (12,*)ans(1,kchan),unc(1,kchan),ans(2,kchan),unc(2,kchan)
     $        ,itmax,ncalls0,tTot
      enddo
      close(12)


      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
      endif

      end


      double precision function sigint(xx,vegas_wgt,ifl,f)
      use fks_phase_space_data, only: jac_cnt
      use fks_phase_space_data, only: p1_cnt
      use fks_phase_space_data, only: p_born
      use fks_phase_space_data, only: p_born_coll
      use fks_phase_space_data, only: p_born_ev
      use fks_phase_space_data, only: p_born_norad
      use fks_phase_space_data, only: p_i_fks_cnt
      use fks_phase_space_data, only: p_i_fks_ev
      use fks_phase_space_data, only: xi_i_fks_cnt
      use fks_phase_space_data, only: xi_i_fks_ev
      use fks_phase_space_data, only: y_ij_fks_ev
      use fks_phase_space_data, only: p1_cnt,p_born
      use fks_phase_space, only: fks_phase_space_point,
     $     generate_born_contribution,generate_real_phase_space
      use weight_lines
      use extra_weights
      use mint_module
      implicit none
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      double precision xx(ndimmax),vegas_wgt,f(nintegrals),jac,p(0:3
     $     ,nexternal),rwgt,vol,sig,x(99),MC_int_wgt
     $     ,f_class(nintegrals),sig_no_nbody
      double precision bsv_virt_cache(0:amp_split_size)
      integer ifl,nFKS_born,nFKS_picked,iFKS,nFKS_min,iamp,igroup
     $     ,nFKS_max,izero,ione,itwo,mohdr,i,j,iran_picked,nFKS_sector
     $     ,iflav_config,born_class,dum
      parameter (izero=0,ione=1,itwo=2,mohdr=-100)
      logical passcuts,passcuts_nbody,passcuts_n1body,sum,firsttime
      data firsttime/.true./
      external passcuts
      integer             ini_fin_fks(maxchannels)
      common/fks_channels/ini_fin_fks
      integer real_channel_type
      common /c_real_channel_type/ real_channel_type
      data sum /.false./
      integer         nndim
      common/tosigint/nndim
      logical       nbody
      common/cnbody/nbody
      double precision wgt_cnt(-2:2),pswgt_cnt(-2:2)
      

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      character*4      abrv
      common /to_abrv/ abrv
c PineAPPL
      logical pineappl
      common /for_pineappl/ pineappl
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      integer ini_fin_fks_map(0:2,0:fks_configs)
      integer physical_fks_group_count(0:2)
      integer physical_fks_group_size(fks_configs,0:2)
      integer physical_fks_group_offset(fks_configs,0:2)
      integer physical_fks_members(fks_configs,0:2)
      save ini_fin_fks_map,physical_fks_group_count
      save physical_fks_group_size,physical_fks_group_offset
      save physical_fks_members
      include 'has_ewsudakov.inc'

C Born variables
      double precision born_amp2(ngraphs), born_jamp2(0:ncolor)
      double precision ev_amp2(ngraphs),norad_amp2(ngraphs)
      complex*16 born_ans_cnt(2,nsplitorders)
      double precision born_amp_split(amp_split_size)
      double precision born_bsv_output(amp_split_size)
      double complex born_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex born_saveamp(ngraphs,max_bhel)
      double precision wgt_born

C Storage for born-like collinears
      double precision coll_amp2(ngraphs), coll_jamp2(0:ncolor)
      complex*16 coll_ans_cnt(2,nsplitorders)
      double precision coll_amp_split(amp_split_size)
      double complex coll_amp_split_cnt(amp_split_size,2,nsplitorders)
      double complex coll_saveamp(ngraphs,max_bhel)
      double precision wgt_coll

C Virtual variables
      double precision amp_split_virt(amp_split_size),
     &     amp_split_born_for_virt(amp_split_size),
     &     amp_split_avv(amp_split_size)
      double precision amp_split_wgtnstmp(amp_split_size),
     $                 amp_split_wgtwnstmpmuf(amp_split_size),
     $                 amp_split_wgtwnstmpmur(amp_split_size)
      double precision bsv_wgt,virt_wgt,born_wgt

C Real variables
      double precision zero, one
      double precision projection_scale(0:3,nexternal)
      parameter (zero=0d0,one=1d0)
      double precision real_amp_split(amp_split_size),
     $                 ret_amp_split(amp_split_size)
      double precision fx_ev, fx_s, fx_c, fx_sc
      double precision deg_xi_c, deg_lxi_c, deg_xi_sc, deg_lxi_sc
C Real deg amplitudes
      double precision amp_split_collrem_xi(amp_split_size), 
     $                 amp_split_collrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
      double precision p_born_used(0:3,nexternal-1)

      logical storeCalculatedBorn
      integer iextra_cnt,isplitorder_born,isplitorder_cnt
      common /c_extra_cnt/iextra_cnt,isplitorder_born,isplitorder_cnt

      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess

      logical use_evpr, passcuts_coll
      common /to_use_evpr/use_evpr
      type(fks_phase_space_point) phase_space

c Share the sampled Born within this evaluation when sectors permit it.
      phase_space%born%valid=.false.
      if (new_point .and. ifl.ne.2) then
         pass_cuts_check=.false.
      endif
      call vs_begin_point
      call reset_folded_virtual
      if (firsttime) then
         firsttime=.false.
         call setup_ini_fin_fks_map(ini_fin_fks_map)
         if (HAS_PHYSICAL_FKS_CLASSES) then
            call setup_physical_fks_map(physical_fks_group_count,
     $           physical_fks_group_size,physical_fks_group_offset,
     $           physical_fks_members)
         endif
         write (*,*) 'initial-final FKS maps:'
         write (*,*) 0 ,':',ini_fin_fks_map(0,:)
         write (*,*) 1 ,':',ini_fin_fks_map(1,:)
         write (*,*) 2 ,':',ini_fin_fks_map(2,:)
      endif
      if (ifl.ne.0) then
         write (*,*) 'ERROR ifl not equal to zero in sigint',ifl
         stop 1
      endif
      bsv_virt_cache(:)=0d0
      if (pineappl .and. sum) then
         write (*,*) 'WARNING: PineAPPL only possible '/
     &        /'with MC over FKS directories',pineappl,sum
         write (*,*) 'Switching to MC over FKS directories'
         sum=.false.
      endif
      sigint=0d0
      icontr=0
      do iamp=0,amp_split_size
         virt_wgt_mint(iamp)=0d0
         born_wgt_mint(iamp)=0d0
      enddo
      virtual_over_born=0d0
c      wgt_me_born=0d0
c      wgt_me_real=0d0
      if (ickkw.eq.3) call set_FxFx_scale(0,p)
      call update_vegas_x(xx,x)
      if (real_channel_type.eq.1) then
c The finite hard remainder has its own real-diagram phase space and
c sums only the FKS sectors owned by this subprocess and real ME.
         nbody=.false.
         call compute_hard_real_channels(nndim,iconfig,x,p,vegas_wgt)
         goto 12
      endif
      nFKS_sector=ini_fin_fks(ichan)

c Sample grouped fixed-order contributions by physical FKS topology. Sum all
c physical Born flavours there at the same phase-space point, so
c that flavour is stratified rather than Monte Carlo sampled.
      if (HAS_PHYSICAL_FKS_CLASSES) then
         call get_MC_integer(max(nFKS_sector,1),
     $        physical_fks_group_count(nFKS_sector),iran_picked,vol)
         if (sum) then
            nFKS_min=1
            nFKS_max=physical_fks_group_count(nFKS_sector)
            MC_int_wgt=1d0
         else
            nFKS_min=iran_picked
            nFKS_max=iran_picked
            MC_int_wgt=1d0/vol
         endif
         f(1:nintegrals)=0d0
         sig_no_nbody=0d0

         do iflav_config=1,NBORN_FLAVOR_CONFIGS
            icontr=0
            vs_vcomp=.false.
            vs_ratio_set=.false.
            do iamp=0,amp_split_size
               virt_wgt_mint(iamp)=0d0
               born_wgt_mint(iamp)=0d0
            enddo
            virtual_over_born=0d0
            wgt_me_born=0d0
            wgt_me_real=0d0
            if (ickkw.eq.3) call set_FxFx_scale(0,p)

            nFKS_born=BORN_FKS_CONFIG_D(iflav_config)
            born_class=BORN_FLAVOR_INDEX_D(nFKS_born)

c The n-body contribution is independent of which topology was selected.
c Evaluate each physical Born class once without a topology correction.
            if (abrv.ne.'real') then
               nbody=.true.
               call refresh_born_cache()
               call update_fks_dir_FO(nFKS_born)
               if (nFKS_sector.eq.0) then
                  jac=1d0
               else
                  jac=0.5d0
               endif
               if (nbody) then
                  call generate_born_contribution(nndim,iconfig,jac,x,
     $                 phase_space)
               else
                  call generate_real_phase_space(nndim,iconfig,jac,x,
     $                 phase_space)
               endif
               p=phase_space%p
               if (p_born(0,1).ge.0d0) then
                  call compute_prefactors_nbody(vegas_wgt)
                  call set_cms_stuff(izero)
                  if (ickkw.eq.3)
     $                 call set_FxFx_scale(1,p1_cnt(0,1,0))
                  passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
                  if (passcuts_nbody) then
                     pass_cuts_check=.true.
                     call set_alphaS(p1_cnt(0,1,0))
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      call include_multichannel_enhance_buffered(1,born_amp2,
     $                     ev_amp2,norad_amp2)
                      if (abrv(1:2).ne.'vi') then
                         call compute_born_buffered(born_amp_split,wgt_born)
                         if(abrv.ne.'born'.and.abrv.ne.'bovi')
     $                       call compute_ewsudakov_buffered(p_born,wgt_born,
     $                       born_saveamp)
                      endif
                      if (abrv.ne.'born'.and.abrv.ne.'bosk') then
                      born_bsv_output=born_amp_split
                         call bornsoftvirtual_buffered(p1_cnt(0,1,0),p_born,
     $                    bsv_wgt,virt_wgt,born_wgt,
     $                    amp_split_virt,amp_split_born_for_virt,
     $                    amp_split_avv,amp_split_wgtnstmp,
     $                    amp_split_wgtwnstmpmuf,
     $                    amp_split_wgtwnstmpmur,wgt_born,
     $                    born_ans_cnt,born_amp_split_cnt,
     $                    born_saveamp,born_bsv_output,bsv_virt_cache)
                         call compute_nbody_noborn_buffered(p1_cnt(0,1,0),
     $                    bsv_wgt,virt_wgt,born_wgt,
     $                    amp_split_virt,amp_split_born_for_virt,
     $                    amp_split_avv,amp_split_wgtnstmp,
     $                    amp_split_wgtwnstmpmuf,
     $                    amp_split_wgtwnstmpmur,born_amp_split,
     $                    wgt_born)
                      endif
                  endif
               endif
            endif

c Sum mode visits every topology exactly. Otherwise the n+1-body contribution
c belongs to the sampled topology and receives inverse topology probability
c through the phase-space Jacobian.
            do igroup=nFKS_min,nFKS_max
               do j=1,physical_fks_group_size(igroup,nFKS_sector)
                  i=physical_fks_group_offset(igroup,nFKS_sector)+j
                  iFKS=physical_fks_members(i,nFKS_sector)
                  if (BORN_FLAVOR_INDEX_D(iFKS).ne.born_class) cycle
                  if (abrv(1:4).eq.'born' .or.
     $             abrv(1:4).eq.'bovi' .or.
     $             abrv(1:4).eq.'bosk' .or.
     $               abrv(1:2).eq.'vi') cycle
                  nbody=.false.
                  call refresh_born_cache()
c MZ this is a temporary fix for processes without soft singularities
c associated to the initial state. Do not extend this to event generation.
                  wgt_me_born=0d0
                  wgt_me_real=0d0
                  jac=MC_int_wgt
                  call update_fks_dir_FO(iFKS)
                  if (nbody) then
                  call generate_born_contribution(nndim,iconfig,jac,x,
     $                 phase_space)
               else
                  call generate_real_phase_space(nndim,iconfig,jac,x,
     $                 phase_space)
               endif
               p=phase_space%p
                  if (p_born(0,1).ge.0d0) then
                  call compute_prefactors_n1body(vegas_wgt,jac)
                  call set_cms_stuff(izero)
                  if (ickkw.eq.3)
     $                 call set_FxFx_scale(2,p1_cnt(0,1,0))
                  passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
                  call set_cms_stuff(ione)
                  passcuts_coll=(use_evpr.and.passcuts_nbody).or.
     $                 passcuts(p1_cnt(0,1,1),rwgt)
                  call set_cms_stuff(mohdr)
                  if (ickkw.eq.3)
     $                 call set_FxFx_scale(3,p)
                  passcuts_n1body=passcuts(p,rwgt)
                  if (passcuts_nbody .and. abrv.ne.'real') then
                     pass_cuts_check=.true.
                     call set_cms_stuff(izero)
                     call set_alphaS(p1_cnt(0,1,0))
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,real_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      call include_multichannel_enhance_buffered(3,
     $                     born_amp2,ev_amp2,norad_amp2)
                      call sreal_store_buffered(p1_cnt(0,1,0),0d0,
     $                     y_ij_fks_ev,fx_s,ret_amp_split,
     $                     real_amp_split,born_amp_split,born_ans_cnt,
     $                     born_amp_split_cnt,born_saveamp,
     $                     coll_amp_split,coll_ans_cnt,
     $                     coll_amp_split_cnt,coll_saveamp)
                      call compute_soft_counter_term_buffered(0d0,
     $                     ret_amp_split,fx_s)
                      call set_cms_stuff(itwo)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)

                      call refresh_born_cache()
                      call sborn_amp_frame(p_born_coll,coll_amp2,
     $                     coll_jamp2,coll_amp_split,
     $                     coll_amp_split_cnt,wgt_coll,coll_ans_cnt,
     $                     coll_saveamp)
                      call sreal_deg_buffered(p1_cnt(0,1,2),zero,one,
     $                     deg_xi_sc,deg_lxi_sc,
     $                     amp_split_wgtdegrem_xi,
     $                     amp_split_wgtdegrem_lxi,
     $                     amp_split_wgtdegrem_muF,
     $                     amp_split_wgtpsch_p,
     $                     amp_split_wgtpsch_l,
     $                     amp_split_wgtpsch_d,born_ans_cnt,
     $                     born_amp_split_cnt,coll_ans_cnt,
     $                     coll_amp_split_cnt)
                      p_born_used(:,:)=p_born(:,:)
c Ordinary sreal_deg_buffered consumes copies of these Born arrays. An extra
c counterterm hook may modify model/cache state, so retain its refresh.
                      if (iextra_cnt.gt.0) then
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born_coll,coll_amp2,
     $                     coll_jamp2,coll_amp_split,
     $                     coll_amp_split_cnt,wgt_coll,coll_ans_cnt,
     $                     coll_saveamp)
                      endif
                      call sreal_store_buffered(p1_cnt(0,1,2),zero,one,
     $                     fx_sc,ret_amp_split,real_amp_split,
     $                     born_amp_split,born_ans_cnt,born_amp_split_cnt,
     $                     born_saveamp,coll_amp_split,coll_ans_cnt,
     $                     coll_amp_split_cnt,coll_saveamp)
                      call compute_soft_collinear_counter_term_buffered(0d0,
     $                     ret_amp_split,fx_sc,
     $                     amp_split_wgtdegrem_xi,
     $                     amp_split_wgtdegrem_lxi,
     $                     amp_split_wgtdegrem_muF,
     $                     amp_split_wgtpsch_p,
     $                     amp_split_wgtpsch_l,
     $                     amp_split_wgtpsch_d)
                  endif
                  if (passcuts_coll .and. abrv.ne.'real') then
                     call set_alphaS(p1_cnt(0,1,1))
                     call set_cms_stuff(ione)
                      storeCalculatedBorn=calculatedBorn
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born_coll,coll_amp2,
     $                     coll_jamp2,coll_amp_split,
     $                     coll_amp_split_cnt,wgt_coll,coll_ans_cnt,
     $                     coll_saveamp)
                      call sreal_deg_buffered(p1_cnt(0,1,1),xi_i_fks_cnt(1),
     $                     one,deg_xi_c,deg_lxi_c,
     $                     amp_split_wgtdegrem_xi,
     $                     amp_split_wgtdegrem_lxi,
     $                     amp_split_wgtdegrem_muF,
     $                     amp_split_wgtpsch_p,
     $                     amp_split_wgtpsch_l,
     $                     amp_split_wgtpsch_d,born_ans_cnt,
     $                     born_amp_split_cnt,coll_ans_cnt,
     $                     coll_amp_split_cnt)
                      p_born_used(:,:)=p_born(:,:)
c Ordinary sreal_deg_buffered consumes copies of these Born arrays. An extra
c counterterm hook may modify model/cache state, so retain its refresh.
                      if (iextra_cnt.gt.0) then
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born_coll,coll_amp2,
     $                     coll_jamp2,coll_amp_split,
     $                     coll_amp_split_cnt,wgt_coll,coll_ans_cnt,
     $                     coll_saveamp)
                      endif
                      call sreal_store_buffered(p1_cnt(0,1,1),
     $                     xi_i_fks_cnt(1),one,fx_c,ret_amp_split,
     $                     real_amp_split,born_amp_split,born_ans_cnt,
     $                     born_amp_split_cnt,born_saveamp,
     $                     coll_amp_split,coll_ans_cnt,
     $                     coll_amp_split_cnt,coll_saveamp)
                      call compute_collinear_counter_term_buffered(0d0,
     $                     ret_amp_split,fx_c,
     $                     amp_split_wgtdegrem_xi,
     $                     amp_split_wgtdegrem_lxi,
     $                     amp_split_wgtdegrem_muF,
     $                     amp_split_wgtpsch_p,
     $                     amp_split_wgtpsch_l,
     $                     amp_split_wgtpsch_d)
                  endif
                  if (passcuts_n1body) then
                     pass_cuts_check=.true.
                      call set_cms_stuff(mohdr)
                      call set_alphaS(p)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born,born_amp2,
     $                     born_jamp2,born_amp_split,
     $                     born_amp_split_cnt,wgt_born,born_ans_cnt,
     $                     born_saveamp)
                      ev_amp2(:)=0d0
                      norad_amp2(:)=0d0
                      if (p_born_ev(0,1).gt.0d0) then
                         call refresh_born_cache()
                      projection_scale=0d0
                      projection_scale(:,1:nexternal-1)=p_born_ev
                      call set_alphaS(projection_scale)
                         call sborn_amp_frame(p_born_ev,ev_amp2,
     $                        coll_jamp2,coll_amp_split,
     $                        coll_amp_split_cnt,wgt_coll,
     $                        coll_ans_cnt,coll_saveamp)
                      call set_alphaS(p)
                      endif
                      if (p_born_norad(0,1).gt.0d0) then
                         call refresh_born_cache()
                      projection_scale=0d0
                      projection_scale(:,1:nexternal-1)=p_born_norad
                      call set_alphaS(projection_scale)
                         call sborn_amp_frame(p_born_norad,norad_amp2,
     $                        coll_jamp2,coll_amp_split,
     $                        coll_amp_split_cnt,wgt_coll,
     $                        coll_ans_cnt,coll_saveamp)
                      call set_alphaS(p)
                      endif
                      call include_multichannel_enhance_buffered(2,born_amp2,
     $                     ev_amp2,norad_amp2)
                      storeCalculatedBorn=calculatedBorn
                      p_born_used(:,:)=p_born(:,:)
                      call refresh_born_cache()
                      call sborn_amp_frame(p_born_coll,coll_amp2,
     $                     coll_jamp2,coll_amp_split,
     $                     coll_amp_split_cnt,wgt_coll,coll_ans_cnt,
     $                     coll_saveamp)
c sreal_store_buffered takes the real matrix element as an input (event
c generation evaluates it in amplitudes_vec): evaluate it here.
                      call smatrix_real_amp_frame(p,real_amp_split,fx_ev)
                      call sreal_store_buffered(p,xi_i_fks_ev,y_ij_fks_ev,
     $                     fx_ev,ret_amp_split,real_amp_split,
     $                     born_amp_split,born_ans_cnt,born_amp_split_cnt,
     $                     born_saveamp,coll_amp_split,coll_ans_cnt,
     $                     coll_amp_split_cnt,coll_saveamp)
                      call compute_real_emission_buffered(p,1d0,
     $                     ret_amp_split,fx_ev)
                  endif
                  endif
               enddo
            enddo

            call finalize_fixed_order_born_class(vegas_wgt,f_class,sig)
            sig_no_nbody=sig_no_nbody+sig
            do i=2,nintegrals
               if (i.ne.4.and.i.ne.5) f(i)=f(i)+f_class(i)
            enddo
         enddo

         f(1)=abs(f(2))
         f(5)=abs(f(3))
         if (sum) then
            call fill_MC_integer(max(nFKS_sector,1),iran_picked,
     $           abs(sig_no_nbody))
         else
            call fill_MC_integer(max(nFKS_sector,1),iran_picked,
     $           abs(sig_no_nbody)*vol)
         endif
         return
      endif

      call get_MC_integer(max(nFKS_sector,1)
     $     ,ini_fin_fks_map(nFKS_sector,0),iran_picked,vol)
      nFKS_picked=ini_fin_fks_map(nFKS_sector,iran_picked)
c The nbody contributions
      if (abrv.eq.'real') goto 11
      nbody=.true.
      call get_born_nFKSprocess(nFKS_picked,nFKS_born)
      call update_fks_dir_FO(nFKS_born)
      if (ini_fin_fks(ichan).eq.0) then
         jac=1d0
      else
         jac=0.5d0
      endif
      call generate_born_contribution(nndim,iconfig,jac,x,
     $     phase_space)
      p=phase_space%p
      if (p_born(0,1).lt.0d0) goto 12
      call compute_prefactors_nbody(vegas_wgt)
      call set_cms_stuff(izero)
      if (ickkw.eq.3) call set_FxFx_scale(1,p1_cnt(0,1,0))
      passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
      if (passcuts_nbody) then
         pass_cuts_check=.true.
         call set_alphaS(p1_cnt(0,1,0))
         call sborn_amp_frame(p_born,born_amp2,born_jamp2,
     $        born_amp_split,born_amp_split_cnt,wgt_born,
     $        born_ans_cnt,born_saveamp)
         call include_multichannel_enhance_buffered(1,born_amp2,ev_amp2,
     $        norad_amp2)
         if (abrv(1:2).ne.'vi') then
            call compute_born_buffered(born_amp_split,wgt_born)
            if(abrv.ne.'born'.and.abrv.ne.'bovi')
     $           call compute_ewsudakov_buffered(p_born,wgt_born,born_saveamp)
         endif
         if (abrv.ne.'born'.and.abrv.ne.'bosk') then
                      born_bsv_output=born_amp_split
            call bornsoftvirtual_buffered(p1_cnt(0,1,0),p_born,bsv_wgt,
     $     virt_wgt,born_wgt
     $     ,amp_split_virt,amp_split_born_for_virt,amp_split_avv
     $     ,amp_split_wgtnstmp,amp_split_wgtwnstmpmuf,
     $     amp_split_wgtwnstmpmur,wgt_born,born_ans_cnt,
     $     born_amp_split_cnt,born_saveamp,born_bsv_output,bsv_virt_cache)
            call compute_nbody_noborn_buffered(p1_cnt(0,1,0),bsv_wgt,virt_wgt,born_wgt
     $     ,amp_split_virt,amp_split_born_for_virt,amp_split_avv
     $     ,amp_split_wgtnstmp,amp_split_wgtwnstmpmuf,amp_split_wgtwnstmpmur
     $     ,born_amp_split,wgt_born)
         endif
      endif

 11   continue
c The n+1-body contributions (including counter terms)
      if ( abrv(1:4).eq.'born' .or.
     $     abrv(1:4).eq.'bovi' .or.
     $     abrv(1:4).eq.'bosk' .or.
     $     abrv(1:2).eq.'vi' ) goto 12
      nbody=.false.
      if (sum) then
         nFKS_min=1
         nFKS_max=ini_fin_fks_map(ini_fin_fks(ichan),0)
         MC_int_wgt=1d0
      else
         nFKS_min=iran_picked
         nFKS_max=iran_picked
         MC_int_wgt=1d0/vol
      endif

      do i=nFKS_min,nFKS_max
         iFKS=ini_fin_fks_map(ini_fin_fks(ichan),i)
         ! MZ this is a temporary fix for processes without
         ! soft singularities associated to the initial state
         ! DO NOT extend this fix to event generation
c         wgt_me_born=0d0
c         wgt_me_real=0d0
         jac=MC_int_wgt
         call update_fks_dir_FO(iFKS)
         call generate_real_phase_space(nndim,iconfig,jac,x,
     $        phase_space)
         p=phase_space%p
         if (p_born(0,1).lt.0d0) cycle
         call init_process_module_n1body_wrapper(dum)
         call compute_prefactors_n1body(vegas_wgt,jac)
         call set_cms_stuff(izero)
         if (ickkw.eq.3) call set_FxFx_scale(2,p1_cnt(0,1,0))
         passcuts_nbody =passcuts(p1_cnt(0,1,0),rwgt)
         ! needed for the mapping without event projection
         call set_cms_stuff(ione)
         passcuts_coll =(use_evpr.and.passcuts_nbody).or.passcuts(p1_cnt(0,1,1),rwgt)
         call set_cms_stuff(mohdr)
         if (ickkw.eq.3) call set_FxFx_scale(3,p)
         passcuts_n1body=passcuts(p,rwgt)
         if (passcuts_nbody .and. abrv.ne.'real') then
            pass_cuts_check=.true.
            call set_cms_stuff(izero)
c Clustering the real event above overwrites the active FxFx scales.
c Counterevents keep the scales of their underlying Born state.
            if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
            call set_alphaS(p1_cnt(0,1,0))
            call sborn_amp_frame(p_born,born_amp2,born_jamp2,real_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
            call include_multichannel_enhance_buffered(3,born_amp2,ev_amp2,
     $           norad_amp2)
            call sreal_store_buffered(p1_cnt(0,1,0),0d0,y_ij_fks_ev,fx_s,
     $              ret_amp_split,real_amp_split,born_amp_split,
     $              born_ans_cnt,born_amp_split_cnt,born_saveamp
     $              ,coll_amp_split,coll_ans_cnt,coll_amp_split_cnt,coll_saveamp)
            call compute_soft_counter_term_buffered(0d0,ret_amp_split,fx_s)
            call set_cms_stuff(itwo)
            call refresh_born_cache()
            call sborn_amp_frame(p_born,born_amp2,born_jamp2,born_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
            call refresh_born_cache()
            call sborn_amp_frame(p_born_coll,coll_amp2,coll_jamp2,coll_amp_split
     $                    ,coll_amp_split_cnt,wgt_coll,coll_ans_cnt,coll_saveamp)
            call sreal_deg_buffered(p1_cnt(0,1,2),zero,one, deg_xi_sc,deg_lxi_sc,
     $    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,amp_split_wgtdegrem_muF,
     $    amp_split_wgtpsch_p, amp_split_wgtpsch_l, amp_split_wgtpsch_d,
     $    born_ans_cnt,born_amp_split_cnt,coll_ans_cnt,coll_amp_split_cnt)
               p_born_used(:,:) = p_born(:,:)
c Ordinary sreal_deg_buffered consumes copies of these Born arrays. An extra
c counterterm hook may modify model/cache state, so retain its refresh.
               if (iextra_cnt.gt.0) then
               call refresh_born_cache()
               call sborn_amp_frame(p_born,born_amp2,born_jamp2,born_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
               call refresh_born_cache()
               call sborn_amp_frame(p_born_coll,coll_amp2,coll_jamp2,coll_amp_split
     $                    ,coll_amp_split_cnt,wgt_coll,coll_ans_cnt,coll_saveamp)
               endif
               call sreal_store_buffered(p1_cnt(0,1,2),zero,one,fx_sc,
     $              ret_amp_split,real_amp_split,born_amp_split,
     $              born_ans_cnt,born_amp_split_cnt,born_saveamp
     $              ,coll_amp_split,coll_ans_cnt,coll_amp_split_cnt,coll_saveamp)
            call compute_soft_collinear_counter_term_buffered(0d0,ret_amp_split,fx_sc,
     &    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,
     &    amp_split_wgtdegrem_muF,
     &    amp_split_wgtpsch_p, amp_split_wgtpsch_l,
     &    amp_split_wgtpsch_d)
         endif
         if (passcuts_coll .and. abrv.ne.'real') then
           if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
           call set_alphaS(p1_cnt(0,1,1))
           call set_cms_stuff(ione)
               storeCalculatedBorn=calculatedBorn
               call refresh_born_cache()
               call sborn_amp_frame(p_born,born_amp2,born_jamp2,born_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
               call refresh_born_cache()
               call sborn_amp_frame(p_born_coll,coll_amp2,coll_jamp2,coll_amp_split
     $                    ,coll_amp_split_cnt,wgt_coll,coll_ans_cnt,coll_saveamp)
               call sreal_deg_buffered(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,deg_xi_c,deg_lxi_c,
     $    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,amp_split_wgtdegrem_muF,
     $    amp_split_wgtpsch_p, amp_split_wgtpsch_l, amp_split_wgtpsch_d,
     $    born_ans_cnt,born_amp_split_cnt,coll_ans_cnt,coll_amp_split_cnt)
               p_born_used(:,:) = p_born(:,:)
c Ordinary sreal_deg_buffered consumes copies of these Born arrays. An extra
c counterterm hook may modify model/cache state, so retain its refresh.
               if (iextra_cnt.gt.0) then
               call refresh_born_cache()
               call sborn_amp_frame(p_born,born_amp2,born_jamp2,born_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
               call refresh_born_cache()
               call sborn_amp_frame(p_born_coll,coll_amp2,coll_jamp2,coll_amp_split
     $                    ,coll_amp_split_cnt,wgt_coll,coll_ans_cnt,coll_saveamp)
               endif
               call sreal_store_buffered(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx_c,
     $              ret_amp_split,real_amp_split,born_amp_split,
     $              born_ans_cnt,born_amp_split_cnt,born_saveamp
     $              ,coll_amp_split,coll_ans_cnt,coll_amp_split_cnt,coll_saveamp)
            call compute_collinear_counter_term_buffered(0d0,ret_amp_split,fx_c,
     &    amp_split_wgtdegrem_xi,amp_split_wgtdegrem_lxi,
     &    amp_split_wgtdegrem_muF,
     &    amp_split_wgtpsch_p, amp_split_wgtpsch_l,
     &    amp_split_wgtpsch_d)
         endif
         if (passcuts_n1body) then
            pass_cuts_check=.true.
            call set_cms_stuff(mohdr)
            if (ickkw.eq.3) call set_FxFx_scale(-3,p)
            call set_alphaS(p)
            call refresh_born_cache()
            call sborn_amp_frame(p_born,born_amp2,born_jamp2,
     $           born_amp_split,born_amp_split_cnt,wgt_born,
     $           born_ans_cnt,born_saveamp)
   !          storeCalculatedBorn=calculatedBorn
   !          p_born_used(:,:) = p_born(:,:)
   !          calculatedBorn=.false.
   !          call sborn_amp(p_born_used,born_amp2,born_jamp2,born_amp_split
   !   $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
   !          calculatedBorn=storeCalculatedBorn
            ev_amp2(:)=0d0
            norad_amp2(:)=0d0
            if (p_born_ev(0,1).gt.0d0) then
               call refresh_born_cache()
                      projection_scale=0d0
                      projection_scale(:,1:nexternal-1)=p_born_ev
                      call set_alphaS(projection_scale)
               call sborn_amp_frame(p_born_ev,ev_amp2,coll_jamp2,
     $              coll_amp_split,coll_amp_split_cnt,wgt_coll,
     $              coll_ans_cnt,coll_saveamp)
                      call set_alphaS(p)
            endif
            if (p_born_norad(0,1).gt.0d0) then
               call refresh_born_cache()
                      projection_scale=0d0
                      projection_scale(:,1:nexternal-1)=p_born_norad
                      call set_alphaS(projection_scale)
               call sborn_amp_frame(p_born_norad,norad_amp2,
     $              coll_jamp2,coll_amp_split,coll_amp_split_cnt,
     $              wgt_coll,coll_ans_cnt,coll_saveamp)
                      call set_alphaS(p)
            endif
            call include_multichannel_enhance_buffered(2,born_amp2,ev_amp2,
     $           norad_amp2)
               storeCalculatedBorn=calculatedBorn
               p_born_used(:,:) = p_born(:,:)
               call refresh_born_cache()
               call sborn_amp_frame(p_born_coll,coll_amp2,coll_jamp2,coll_amp_split
     $                    ,coll_amp_split_cnt,wgt_coll,coll_ans_cnt,coll_saveamp)
               call refresh_born_cache()
               call sborn_amp_frame(p_born,born_amp2,born_jamp2,born_amp_split
     $                    ,born_amp_split_cnt,wgt_born,born_ans_cnt,born_saveamp)
c sreal_store_buffered takes the real matrix element as an input: evaluate it.
               call smatrix_real_amp_frame(p,real_amp_split,fx_ev)
               call sreal_store_buffered(p,xi_i_fks_ev,y_ij_fks_ev,fx_ev,
     $              ret_amp_split,real_amp_split,born_amp_split,
     $              born_ans_cnt,born_amp_split_cnt,born_saveamp
     $              ,coll_amp_split,coll_ans_cnt,coll_amp_split_cnt,coll_saveamp)
            call compute_real_emission_buffered(p,1d0,ret_amp_split,fx_ev)
         endif
      enddo
      
 12   continue
c Include PDFs and alpha_S and reweight to include the uncertainties
      call include_PDF_and_alphas

c Include the bias weight specified in the bias_weight_function
      call include_bias_wgt
c Fit the virtual surrogate after the PDF and bias factors are included.
      call virt_surrogate_fit_point

      if (doreweight) then
         if (do_rwgt_scale) call reweight_scale
         if (do_rwgt_pdf) call reweight_pdf
      endif
      
      if (pineappl) then
         if (sum) then
            write (*,*) 'ERROR: PineAPPL only possible '/
     &           /'with MC over FKS directories',pineappl,sum
            stop 1
         endif
         call fill_pineappl_weights(vegas_wgt)
      endif

c Importance sampling for FKS configurations
      if (real_channel_type.eq.1) then
c There is no sampled FKS integer in the hard contribution.
         continue
      elseif (sum) then
         call get_wgt_nbody(sig)
         call fill_MC_integer(max(ini_fin_fks(ichan),1),iran_picked
     $        ,abs(sig))
      else
         call get_wgt_no_nbody(sig)
         call fill_MC_integer(max(ini_fin_fks(ichan),1),iran_picked
     $        ,abs(sig)*vol)
      endif

c Finalize PS point
      call fill_plots
      call fill_mint_function(f)
      return
      end

      subroutine finalize_fixed_order_born_class(vegas_wgt,f,sig)
c Apply PDFs, reweighting and analysis to one physical Born class. Keeping
c this local is needed because virtual-trick weights have no flavour index
c and must receive exactly one luminosity.
      use weight_lines
      use extra_weights
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      double precision vegas_wgt,f(nintegrals),sig
      logical pineappl
      common /for_pineappl/ pineappl

      call include_PDF_and_alphas
      call include_bias_wgt

      if (doreweight) then
         if (do_rwgt_scale) call reweight_scale
         if (do_rwgt_pdf) call reweight_pdf
      endif

      if (pineappl) call fill_pineappl_weights(vegas_wgt)
      call get_wgt_no_nbody(sig)
      call fill_plots
      call fill_mint_function(f)
      return
      end

      subroutine update_fks_dir_FO(nFKS)
c Fixed-order variant of update_fks_dir: unlike the version in
c driver_vec.f (used by the multi-event interface), setfksfactor is
c called with .false. here.
      implicit none
      include 'run.inc'
      integer nFKS
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      nFKSprocess=nFKS
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setcuts
      call setfksfactor(.false.)
      return
      end
      
      subroutine setup_ini_fin_FKS_map(ini_fin_FKS_map)
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer ini_fin_FKS_map(0:2,0:fks_configs),iFKS
      ini_fin_FKS_map(0,0)=0
      ini_fin_FKS_map(1,0)=0
      ini_fin_FKS_map(2,0)=0
      do iFKS=1,fks_integrated
         ini_fin_FKS_map(0,0)=ini_fin_FKS_map(0,0)+1
         ini_fin_FKS_map(0,ini_fin_FKS_map(0,0))=iFKS
         if (fks_j_d(iFKS).le.nincoming .and.
     $       fks_j_d(iFKS).gt.0) then
            ini_fin_FKS_map(2,0)=ini_fin_FKS_map(2,0)+1
            ini_fin_FKS_map(2,ini_fin_FKS_map(2,0))=iFKS
         elseif (fks_j_d(iFKS).gt.nincoming .and.
     $           fks_j_d(iFKS).le.nexternal) then
            ini_fin_FKS_map(1,0)=ini_fin_FKS_map(1,0)+1
            ini_fin_FKS_map(1,ini_fin_FKS_map(1,0))=iFKS
         else
            write (*,*) 'ERROR in setup_ini_fin_FKS_map',fks_j_d(iFKS)
     $           ,nincoming,iFKS
            stop 1
         endif
      enddo
      return
      end

      subroutine setup_physical_fks_map(group_count,group_size,
     $     group_offset,members)
c Collapse physical FKS rows onto the topology that generated them. Keep a
c linear compressed list per initial/final sector: each group's offset and
c size select its contiguous range in members. This avoids storage quadratic
c in the number of physical FKS rows.
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer group_count(0:2),group_size(fks_configs,0:2)
      integer group_offset(fks_configs,0:2),members(fks_configs,0:2)
      integer sector,iFKS,igroup,imember,existing_fks,born_class,iborn
      integer topology,topology_group(fks_configs)
      integer group_fill(fks_configs)
      logical belongs,born_class_exists(fks_configs)

      group_count=0
      group_size=0
      group_offset=0
      members=0
      born_class_exists=.false.
      do iborn=1,NBORN_FLAVOR_CONFIGS
         existing_fks=BORN_FKS_CONFIG_D(iborn)
         born_class=BORN_FLAVOR_INDEX_D(existing_fks)
         if (born_class.lt.1.or.born_class.gt.fks_configs) then
            write (*,*) 'Invalid Born flavor class',born_class,
     $           existing_fks
            stop 1
         endif
         born_class_exists(born_class)=.true.
      enddo
      do sector=0,2
         topology_group=0
         do iFKS=1,FKS_INTEGRATED
            belongs=sector.eq.0 .or.
     $           (sector.eq.1.and.FKS_J_D(iFKS).gt.nincoming) .or.
     $           (sector.eq.2.and.FKS_J_D(iFKS).le.nincoming)
            if (.not.belongs) cycle

            topology=FKS_TOPOLOGY_D(iFKS)
            if (topology.lt.1.or.topology.gt.fks_configs) then
               write (*,*) 'Invalid FKS topology',topology,iFKS
               stop 1
            endif
            igroup=topology_group(topology)
            if (igroup.eq.0) then
               group_count(sector)=group_count(sector)+1
               igroup=group_count(sector)
               topology_group(topology)=igroup
            endif

            born_class=BORN_FLAVOR_INDEX_D(iFKS)
            if (born_class.lt.1.or.born_class.gt.fks_configs) then
               write (*,*) 'Invalid Born flavor class for FKS row',
     $              iFKS,born_class
               stop 1
            endif
            if (.not.born_class_exists(born_class)) then
               write (*,*) 'No Born representative for FKS row',
     $              iFKS,born_class
               stop 1
            endif
            group_size(igroup,sector)=group_size(igroup,sector)+1
         enddo

         if (group_count(sector).eq.0) cycle
         do igroup=2,group_count(sector)
            group_offset(igroup,sector)=
     $           group_offset(igroup-1,sector)+
     $           group_size(igroup-1,sector)
         enddo
         group_fill=0
         do iFKS=1,FKS_INTEGRATED
            belongs=sector.eq.0 .or.
     $           (sector.eq.1.and.FKS_J_D(iFKS).gt.nincoming) .or.
     $           (sector.eq.2.and.FKS_J_D(iFKS).le.nincoming)
            if (.not.belongs) cycle
            igroup=topology_group(FKS_TOPOLOGY_D(iFKS))
            group_fill(igroup)=group_fill(igroup)+1
            imember=group_offset(igroup,sector)+group_fill(igroup)
            members(imember,sector)=iFKS
         enddo

         write (*,*) 'Physical FKS topology map for sector',sector
         do igroup=1,group_count(sector)
            write (*,*) igroup,'-->',
     $           (members(group_offset(igroup,sector)+imember,sector),
     $            imember=1,group_size(igroup,sector))
         enddo
      enddo
      return
      end


      subroutine update_vegas_x(xx,x)
      use mint_module
      implicit none
      integer i
      double precision xx(ndimmax),x(99),ran2
      external ran2
      integer         nndim
      common/tosigint/nndim
      character*4      abrv
      common /to_abrv/ abrv
      do i=1,99
         if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') then
            if(i.le.nndim-3)then
               x(i)=xx(i)
            elseif(i.le.nndim) then
               x(i)=ran2()      ! Choose them flat when not including real-emision
            else
               x(i)=0.d0
            endif
         else
            if(i.le.nndim)then
               x(i)=xx(i)
            else
               x(i)=0.d0
            endif
         endif
      enddo
      return
      end

c
      subroutine get_user_params(ncall,nitmax,irestart)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      use mint_module
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'orders.inc'
      include 'real_channels.inc'
c
c     Arguments
c
      integer ncall,nitmax
c
c     Local
c
      integer i,kchan
      double precision dconfig(maxchannels)
c
c     Global
c
      integer ini_fin_fks(maxchannels)
      common /fks_channels/ini_fin_fks
      integer real_channel_type
      common /c_real_channel_type/ real_channel_type
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody

      include 'has_ewsudakov.inc'

c
c To convert diagram number to configuration
c
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
c
c Vegas stuff
c
      integer irestart

      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      integer random_offset_split
      common /c_random_offset_split/ random_offset_split
      logical done
      character*4096 buffer
c-----
c  Begin Code
c-----
      unwgt=.false.
      open (unit=83,file='input_app.txt',status='old')
      done=.false.
      nchans=0
      real_channel_type=0
      do while (.not. done)
         read(83,'(a)',err=222,end=222) buffer
         if (buffer(1:7).eq.'NPOINTS') then
            buffer=buffer(10:100)
            read(buffer,*) ncall
            write (*,*) 'Number of phase-space points per iteration:',ncall
         elseif(buffer(1:11).eq.'NITERATIONS') then
            read(buffer(14:),*) nitmax
            write (*,*) 'Maximum number of iterations is:',nitmax
         elseif(buffer(1:8).eq.'ACCURACY') then
            read(buffer(11:),*) accuracy
            write (*,*) 'Desired accuracy is:',accuracy
         elseif(buffer(1:10).eq.'ADAPT_GRID') then
            read(buffer(13:),*) use_cut
            write (*,*) 'Using adaptive grids:',use_cut
         elseif(buffer(1:12).eq.'MULTICHANNEL') then
            read(buffer(15:),*) i
            if (i.eq.1) then
               multi_channel=.true.
               write (*,*) 'Using Multi-channel integration'
            else
               multi_channel=.false.
               write (*,*) 'Not using Multi-channel integration'
            endif
         elseif(buffer(1:12).eq.'SUM_HELICITY') then
            read(buffer(15:),*) i
            if (nincoming.eq.1) then
               write (*,*) 'Sum over helicities in the virtuals'/
     $              /' for decay process'
               mc_hel=0
c Several virtual rows without a helicity filter for each of them in
c MadLoop (see HELFILTER_PER_VIRTUAL_FLAVOR): sum the helicities.
            elseif (i.eq.0.or.(N_VIRTUAL_FLAVOR_CONFIGS.gt.1.and.
     $          .not.HELFILTER_PER_VIRTUAL_FLAVOR)) then
               mc_hel=0
               write (*,*) 'Explicitly summing over helicities'/
     $              /' for the virtuals'
            else
               mc_hel=1
               write(*,*) 'Do MC over helicities for the virtuals'
            endif
            isum_hel=0
         elseif(buffer(1:6).eq.'NCHANS') then
            read(buffer(9:),*) nchans
            write (*,*) 'Number of channels to integrate together:'
     $           ,nchans
            if (nchans.gt.maxchannels) then
               write (*,*) 'Too many integration channels to be '/
     $              /'integrated together. Increase maxchannels',nchans
     $              ,maxchannels
               stop 1
            endif
         elseif(buffer(1:12).eq.'CHANNEL_TYPE') then
            read(buffer(15:),*) real_channel_type
            if (real_channel_type.ne.0.and.real_channel_type.ne.1) then
               write (*,*) 'Invalid CHANNEL_TYPE',real_channel_type
               stop 1
            endif
            if (real_channel_type.eq.1.and.
     $           .not.(real_channels_x0.gt.0d0.and.
     $                 real_channels_x0.lt.1d0)) then
               write (*,*) 'Hard channels require 0 < real_channels_x0 < 1'
               stop 1
            endif
         elseif(buffer(1:7).eq.'CHANNEL') then
            if (nchans.le.0) then
               write (*,*) '"NCHANS" missing in input files'/
     $              /' (still zero)',nchans
               stop
            endif
            read(buffer(10:),*) (dconfig(kchan),kchan=1,nchans)
            do kchan=1,nchans
               iconfigs(kchan) = int(dconfig(kchan))
               if (real_channel_type.eq.1) then
                  if (dconfig(kchan).ne.dble(iconfigs(kchan)).or.
     $                iconfigs(kchan).lt.1.or.
     $                iconfigs(kchan).gt.n_real_channels) then
                     write (*,*) 'Invalid hard channel',dconfig(kchan)
                     stop 1
                  endif
                  ini_fin_fks(kchan)=0
                  cycle
               endif
               if ( nint(dconfig(kchan)*10)-iconfigs(kchan)*10.eq.0 )
     $              then
                  ini_fin_fks(kchan)=0
               elseif ( nint(dconfig(kchan)*10)-iconfigs(kchan)*10.eq.1
     $                 ) then
                  ini_fin_fks(kchan)=1
               elseif ( nint(dconfig(kchan)*10)-iconfigs(kchan)*10.eq.2
     $                 ) then
                  ini_fin_fks(kchan)=2
               else
                  write (*,*) 'ERROR: invalid configuration number',dconfig
                  stop 1
               endif
               do i=1,mapconfig(0,0)
                  if (iconfigs(kchan).eq.mapconfig(i,0)) then
                     iconfigs(kchan)=i
                     exit
                  endif
               enddo
            enddo
            write(*,*) 'Running Configuration Number(s): '
     $           ,(iconfigs(kchan),kchan=1,nchans)
            write(*,*) 'initial-or-final',(ini_fin_fks(kchan),kchan=1,nchans)
         elseif(buffer(1:5).eq.'SPLIT') then
            read(buffer(8:),*) random_offset_split
            write (*,*) 'Splitting channel:',random_offset_split
         elseif(buffer(1:8).eq.'WGT_MULT') then
            read(buffer(11:),*) wgt_mult
            write (*,*) 'Weight multiplier:',wgt_mult
         elseif(buffer(1:8).eq.'RUN_MODE') then
            read(buffer(11:),*) abrvinput
            if(abrvinput(5:5).eq.'0')then
               nbody=.true.
            else
               nbody=.false.
            endif
            abrv=abrvinput(1:4)
            write (*,*) "doing the ",abrv," of this channel"
            if(nbody)then
               write (*,*) "integration Born/virtual with Sfunction=1"
            else
               write (*,*) "Normal integration (Sfunction != 1)"
            endif
         elseif(buffer(1:7).eq.'RESTART') then
            read(buffer(10:),*) irestart
            if (irestart.eq.0) then
               write (*,*) 'RESTART: Fresh run'
            elseif(irestart.eq.-1) then
               write (*,*) 'RESTART: Use old grids, but refil plots'
            elseif(irestart.eq.1) then
               write (*,*) 'RESTART: continue with existing run'
            else
               write (*,*) 'RESTART:',irestart
            endif
         endif
         cycle
 222     done=.true.
      enddo
      close(83)
      if (real_channel_type.eq.1) then
         if (ickkw.ne.0.or.abrv.eq.'born'.or.abrv.eq.'bovi'.or.
     $       abrv.eq.'bosk'.or.abrv(1:2).eq.'vi') then
            write (*,*) 'Hard channels require an ordinary real NLO run'
            stop 1
         endif
      endif

      if (fks_integrated.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21.and.abrv.ne.'born') then
C Two cases can occur
C   1) the process has been generated with the LOonly flav
C   2) the process has only virtual corrections, e.g. z > v v [QED]
C   3) the process has been generated in the sudakov Approximation
C the different cases can be distinguished by looking at the values
C  of AMP_SPLIT_SIZE, AMP_SPLIT_SIZE_BORN (if they are ==, it is 1))
C  and with the flag has_ewsudakov
           if (amp_split_size.eq.amp_split_size_born) then
             write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'Setting abrv to "born".'
             abrv='born'
             if (ickkw.eq.3) then
               write (*,*) 'FxFx merging not possible with'/
     $              /' [LOonly=QCD] processes'
               stop 1
             endif
           else if (has_ewsudakov) then
             write (*,*) 'Process with sudakov approximation for EWcorr'/
     $           /'Setting abrv to "bosk".'
             abrv='bosk'
           else
             write (*,*) 'Process only with virtual corrections'/
     $           /'Setting abrv to "bovi".'
             abrv='bovi'
           endif
         endif
      endif
c
c     Here I want to set up with B.W. we map and which we don't
c
      lbw(0)=0
      end
c
      subroutine init_process_module_nbody_wrapper()
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      integer colour(1:nexternal-1),i,get_color
      external get_color
      double precision mass(1:nexternal-1),get_mass_from_id
      external get_mass_from_id
      logical valid_dipole(1:nexternal-1,1:nexternal-1,1:max_bcol)
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      do i=1,nexternal-1
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      call init_process_module_nbody(nexternal-1,mass,colour
     $     ,max_bcol,valid_dipole)
      
      end

      
      subroutine init_process_module_n1body_wrapper(dum)
c Keep the calling convention of the MC wrapper, which selects a Born flow.
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer colour(1:nexternal),i,get_color,dum
      double precision mass(1:nexternal),get_mass_from_id
      external get_color
      external get_mass_from_id
      logical valid_dipole(1:nexternal,1:nexternal)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     dummy(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,dummy,niprocs
      do i=1,nexternal
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      call init_process_module_n1body(nexternal,mass,colour
     $     ,maxflow,valid_dipole)
      end

      subroutine compute_hard_real_channels(ndim,id,x,p,vegas_wgt)
c Finite real remainder of MadFKS (6.28). The full real phase space is
c partitioned by real diagram squares. The FKS sum is local to this
c Born subprocess: the same real process can belong to several Borns.
      use fks_phase_space, only: generate_momenta_real_channels
      use fks_phase_space_data, only: xi_i_fks_ev,y_ij_fks_ev,
     $     p_i_fks_ev,p_i_fks_cnt,sqrtshat,shat
      use extra_weights
      use mint_module, only: pass_cuts_check
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'real_channels.inc'
      include 'real_matrix_weights.inc'
      integer ndim,id,ime,iflav,sector,i,j,k,iamp,orders(nsplitorders)
      integer ndiagrams,get_orders_tag
      double precision x(99),p(0:3,nexternal),vegas_wgt,jac
      double precision total,denom,fraction,softvar,hard,sij,wgt1
      double precision rwgt,real_channel_invariant,fks_Sij,raw_split(amp_split_size)
      double precision vi,vj
      integer ids(nexternal),ids_me(nexternal)
      logical passcuts
      external real_channel_invariant,fks_Sij,passcuts
      integer i_fks,j_fks
      common /fks_indices/ i_fks,j_fks
      integer fks_j_from_i(nexternal,0:nexternal)
     $     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/ fks_j_from_i,particle_type,pdg_type
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     $                 fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common /numberofparticles/ fkssymmetryfactor,fkssymmetryfactorBorn,
     $                 fkssymmetryfactorDeg,ngluons,nquarks,nphotons

      call initialize_weight_analysis
      ime=real_channel_me(id)
      iflav=REAL_FLAVOR_INDEX_D(real_channel_fks(id))
      call update_fks_dir_FO(real_channel_fks(id))
      call generate_momenta_real_channels(ndim,real_channel_config(id),
     $     real_channel_fks(id),x,p,jac)
      if (jac.le.0d0.or.p(0,1).le.0d0) return
      call set_cms_stuff(-100)
      if (.not.passcuts(p,rwgt)) return
      pass_cuts_check=.true.
c Exact soft/collinear endpoints have zero measure. Avoid evaluating
c a divergent raw matrix element there, including singular pairs owned
c by a different Born subprocess.
      do i=nincoming+1,nexternal
         if (p(0,i).le.0d0) return
         do j=1,fks_j_from_i(i,0)
            k=fks_j_from_i(i,j)
            if (real_channel_invariant(p,i,k).le.0d0) return
         enddo
      enddo
      call set_alphaS(p)
c Evaluate the real ME in the me_frame, as sreal does (polarised
c processes). The frame mask depends on i_fks: the diagram weights use
c the frame of the representative sector of this ME, which is the same
c for all its channels, so the partition is unchanged.
      get_real_diagram_weights=.true.
      call smatrix_real_frame(p,total)
      get_real_diagram_weights=.false.
      raw_split(1:amp_split_size)=amp_split(1:amp_split_size)
      call get_frame_mask_real(real_channel_fks(id),ids_me)
      denom=0d0
      ndiagrams=0
      do i=1,n_real_channels
         if (real_channel_me(i).ne.ime) cycle
         if (REAL_FLAVOR_INDEX_D(real_channel_fks(i)).ne.iflav) cycle
         ndiagrams=ndiagrams+1
         denom=denom+max(0d0,real_amp2(real_channel_diagram(i)))
      enddo
      if (denom.gt.0d0) then
         fraction=max(0d0,real_amp2(real_channel_diagram(id)))/denom
      else
c A zero diagram sum can occur at an amplitude zero. A uniform
c partition also covers nonzero interference with vanishing weights.
         fraction=1d0/dble(ndiagrams)
      endif
      if (fraction.eq.0d0) return
      do sector=1,FKS_INTEGRATED
         if (real_sector_me(sector).ne.ime) cycle
         if (REAL_FLAVOR_INDEX_D(sector).ne.iflav) cycle
         call update_fks_dir_FO(sector)
         xi_i_fks_ev=2d0*p(0,i_fks)/sqrtshat
         vi=sqrt(sum(p(1:3,i_fks)**2))
         vj=sqrt(sum(p(1:3,j_fks)**2))
         if (vi.eq.0d0.or.vj.eq.0d0) cycle
         y_ij_fks_ev=sum(p(1:3,i_fks)*p(1:3,j_fks))/(vi*vj)
         y_ij_fks_ev=max(-1d0,min(1d0,y_ij_fks_ev))
         p_i_fks_ev(:)=p(:,i_fks)/xi_i_fks_ev
         do k=-2,2
            p_i_fks_cnt(:,k)=p(:,i_fks)/xi_i_fks_ev
         enddo
         softvar=max(0d0,2d0*real_channel_invariant(p,i_fks,j_fks)/shat)
         hard=softvar**2/(real_channels_x0**2+softvar**2)
         if (hard.eq.0d0) cycle
         sij=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
         if (sij.le.0d0) cycle
c A sector whose me_frame differs needs its own evaluation, as in the
c Born channels. Without me_frame all the masks agree: no extra call.
         call get_frame_mask_real(sector,ids)
         if (any(ids.ne.ids_me)) then
            call smatrix_real_frame(p,total)
            raw_split(1:amp_split_size)=amp_split(1:amp_split_size)
            ids_me=ids
         endif
         do iamp=1,amp_split_size
            if (raw_split(iamp).eq.0d0) cycle
            call amp_split_pos_to_orders(iamp,orders)
            QCD_power=orders(qcd_pos)
            wgtcpower=0d0
            if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
            orders_tag=get_orders_tag(orders)
            amp_pos=iamp
            wgt1=raw_split(iamp)/g**QCD_power*jac*vegas_wgt*
     $           fkssymmetryfactor*sij*hard*fraction
            call add_wgt_buffered(1,orders,wgt1,0d0,0d0,0d0,total)
         enddo
      enddo
      return
      end
