      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mc_batch_trace, only: mc_real_profile_open,mc_real_profile_close
      use mc_decision_ledger, only: decision_open,decision_close,
     $     decision_context,current_draw,stage_output,stage_setup
      use iso_fortran_env, only: int64
      use mint_module
      use mc_native_cohort, only: evaluate_native_cohort,
     $     native_context_base_bytes
      use mc_context_memory, only: configure_context_memory,
     $     report_context_memory
      use mc_event_queues, only: event_queue_base_bytes
      use mc_shower_state, only: shower_heap_bound
      use mc_born_support, only: born_model_dimensions
      use mc_born_types, only: born_set_cache_lanes
      use mc_native_context, only: real_cache_size
      use mc_native_generation, only: write_requested_event,
     $     finish_native_generation
      use FKSParams
      use process_module
      use scale_module
      use weight_lines, only: rejected_born_flow_points,
     $     rejected_mc_kinematics_points,rejected_nonfinite_points
      implicit none
C
C     CONSTANTS
C
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,nndim,nevts,p_label,nr_compact,nc_compact
      integer(int64) bytes,compact_bytes,nleg,nfolds
      integer(int64),external :: mc_live_model_bytes
      include 'real_matrix_weights.inc'
      include 'timing_variables.inc'

      integer lunlhe
      parameter (lunlhe=98)
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
c Vegas stuff
      common/tosigint/nndim

      real*8 sigintF
      external sigintF

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      double precision xratmax
      integer i_momcmp_count
      common/ccheckcnt/xratmax,i_momcmp_count

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      include 'orders.inc'

      double precision weight,event_weight
      character*7 event_norm
      common /event_normalisation/event_norm
      integer ixi_i,iphi_i,iy_ij,vn
      logical putonshell
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision ran2,x(ndimmax)
      external ran2
      
      integer ifile,ievents
      double precision inter,absint,uncer
      common /to_write_header_init/inter,absint,uncer,ifile,ievents

      logical SHsep
      logical Hevents
      common/SHevents/Hevents
c statistics for MadLoop      
      INTEGER NINJARS_DP_USAGE,NINJARS_QP_USAGE
      COMMON/NINJARSSTATS/NINJARS_DP_USAGE,NINJARS_QP_USAGE
      INTEGER TIRUST_DP_USAGE,TIRUST_QP_USAGE
      COMMON/TIRUSTSTATS/TIRUST_DP_USAGE,TIRUST_QP_USAGE
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

c timing statistics
      double precision job_started,job_finished,tTot
c general MadFKS parameters
      include 'born_nhel.inc'
C-----
C  BEGIN CODE
C-----  
c Write the process PID in the log.txt files (i.e., to the screen)
      write (*,*) getpid()

      call timing_reset()
      call decision_open()
      call cpu_time(job_started)
      fixed_order=.false.
      nlo_ps=.true.
      if (nincoming.ne.2) then
         write (*,*) 'Decay processes not supported for'/
     &        /' event generation'
         stop 1
      endif

c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      min_virt_fraction_mint=min_virt_fraction
      do i=0,n_ave_virt
         average_virtual(i,1)=0d0
      enddo
      virtual_fraction(1)=virt_fraction
      n_ord_virt=amp_split_size
      
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
      call get_user_params(ncalls0,itmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
      call mc_real_profile_open(imode)
c Only do the reweighting when actually generating the events
      if (imode.eq.2) then
         doreweight=do_rwgt_scale.or.do_rwgt_pdf.or.store_rwgt_info
      else
         doreweight=.false.
         do_rwgt_scale=.false.
         do_rwgt_pdf=.false.
      endif
      if (abrv(1:4).eq.'virt') then
         only_virt=.true.
      else
         only_virt=.false.
      endif

      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      nndim=ndim
      call born_spread_configure(born_spreading.and..not.only_virt
     $     .and.(abrv.eq.'all'.or.abrv.eq.'novi'),nexternal,
     $     nincoming,fks_configs,ndim)
      if (born_spread_active.and.imode.gt.0)
     $     call born_spread_load_table
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
      write(*,*) "about to integrate ", ndim,ncalls0,itmax,iconfig
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.
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
      ifold(ifold_energy)=ixi_i
      ifold(ifold_phi)=iphi_i
      ifold(ifold_yij)=iy_ij

      ! initialise the global, but process dependent, information in the process module.
      call init_process_module_global(shower_mc,abrv,nexternal,nincoming
     $     ,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow,ickkw)
      ! Also put all the n-body process dependent stuff here. It does
      ! not depend on PS point or FKS config, so all global information.
      call init_process_module_nbody_wrapper()
      call init_scale_module(nexternal,shower_scale_factor,fks_configs
     $     ,product(ifold(1:ndim)))
      if(native_mc_capacity.gt.1)then
c The cohort interleaves its lanes between real-emission waves. Give
c each lane its own Born cache slots so a resumed lane finds its Born.
         call born_set_cache_lanes(native_mc_capacity)
         nleg=nexternal
         nfolds=product(int(ifold(1:ndim),int64))
         call born_model_dimensions(nr_compact,nc_compact)
         compact_bytes=8_int64*nr_compact+16_int64*nc_compact
c Fixed descriptors and payloads sized from generated dimensions. The
c active contribution store, optional Born results and variable event
c bytes are reserved separately at each growth, before allocation.
         bytes=native_context_base_bytes()+mc_live_model_bytes()
     $       +shower_heap_bound()
         if(imode.eq.2)bytes=bytes+event_queue_base_bytes()
c Compact models in the provider/request-copy and the real cache, with
c per-diagram output for every entry including the last active result.
         bytes=bytes+(real_cache_size+2_int64)*compact_bytes
     $       +(real_cache_size+1_int64)*8*size(real_amp2,kind=8)
c Provider packing buffers, saved coordinate/fold/sector records,
c history and flow cursors, plus resumed timer stack.
         bytes=bytes+8*(4*nleg+2+amp_split_size)+12
     $       +8*nfolds*ndim+2*8*nfolds+4*nleg*2
     $       +4_int64*(timing_stack_size+1+fks_configs+2*lmaxconfigs)
     $       +12_int64*max_bcol
     $       +8_int64*(fks_configs+2_int64)*ndelH*ndelH
         call configure_context_memory(native_mc_capacity,bytes)
      endif
         
      
c*************************************************************
c     setting of the grids
c*************************************************************
      if (imode.eq.-1.or.imode.eq.0) then
         write (*,*) 'imode is ',imode
         if (native_mc_capacity.eq.1) then
            call mint(sigintF)
         else
            call mint(sigintF,evaluate_native_cohort,native_mc_capacity)
         endif
         call deallocate_weight_lines
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1,1),unc(2,1),0d0,0,0,0,0,0d0,0d0,ans(2,1)
         close(58)
c*************************************************************
c     computation of upper bounding envelope
c*************************************************************
      elseif(imode.eq.1) then
         write (*,*) 'imode is ',imode
         if (native_mc_capacity.eq.1) then
            call mint(sigintF)
         else
            call mint(sigintF,evaluate_native_cohort,native_mc_capacity)
         endif
         call deallocate_weight_lines
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1,1)+ans(5,1),unc(2,1),0d0,0,0,0,0,0d0,0d0
     $        ,ans(2,1) 
         close(58)
c*************************************************************
c     event generation
c*************************************************************
      elseif(imode.eq.2) then
c Mass-shell stuff. This is MC-dependent
         call fill_MC_mshell()
         putonshell=.true.
         unwgt=.true.
         write(*,*) 'Generating ', nevts, ' events'
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
            call mc_real_profile_close()
            call decision_close()
            stop
         endif
         ncalls0=nevts ! Update ncall with the number found in 'nevts'

c     to restore grids:

         call read_grids_from_file

c determine how many events for the virtual and how many for the no-virt
         ncall_virt=int(ans(5,1)/(ans(1,1)+ans(5,1)) * ncalls0)
         ncall_novi=ncalls0-ncall_virt

         write (*,*) "Generating virt :: novi approx.",ncall_virt
     $        ,ncall_novi

         open(unit=lunlhe,file='events.lhe',status='unknown')

c fill the information for the write_header_init common block
         ifile=lunlhe
         ievents=ncalls0
         inter=ans(2,1)
         absint=ans(1,1)+ans(5,1)
         uncer=unc(2,1)
         weight=event_weight

         if (abrv(1:3).ne.'all' .and. abrv(1:4).ne.'born' .and.
     $        abrv(1:4).ne.'virt') then
            write (*,*) 'CANNOT GENERATE EVENTS FOR ABRV',abrv
            stop 1
         endif

         write (*,*) 'imode is ',imode
         vn=-1
         call gen(sigintF,0,vn,x)
         do j=1,ncalls0
            call decision_context(0_int64,j,stage_setup)
            if (abrv(1:4).eq.'born') then
               vn=3
            else
               if (ran2().lt.ans(5,1)/(ans(1,1)+ans(5,1)) .or. only_virt) then
                  abrv='virt'
                  if (only_virt) then
                     vn=2
                  else
                     vn=1
                  endif
               else
                  abrv='novi'
                  vn=2
               endif
            endif
c Randomly pick the contribution that will be written in the event file
            if (native_mc_capacity.eq.1) then
               call gen(sigintF,1,vn,x)
               call decision_context(current_draw,0,stage_output)
               call select_native_event(event_weight,x,weight)
               call finalize_event(x,weight,lunlhe,putonshell,p_label)
               call trace_mc_draw('WRITE')
            else
               call write_requested_event(abrv,vn,native_mc_capacity,
     $              event_weight,putonshell,p_label,lunlhe)
            endif
         enddo
         if (native_mc_capacity.gt.1) call finish_native_generation()
         call deallocate_weight_lines
         vn=-1
         call gen(sigintF,3,vn,x) ! print counters generation efficiencies
         write (lunlhe,'(a)') "</LesHouchesEvents>"
         close(lunlhe)
      endif

      if(rejected_born_flow_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for invalid Born flows:',
     $     rejected_born_flow_points
      if(rejected_mc_kinematics_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for invalid MC kinematics:',
     $     rejected_mc_kinematics_points
      if(rejected_nonfinite_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for nonfinite weights:',
     $     rejected_nonfinite_points
      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
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
      call report_native_draws()
      call report_context_memory()

      open (unit=12, file='res.dat',status='unknown')
      if (imode.eq.0) then
         write (12,*)ans(1,1),unc(1,1),ans(2,1),unc(2,1),itmax,ncalls0,tTot
      else
         write (12,*)ans(1,1)+ans(5,1),sqrt(unc(1,1)**2+unc(5,1)**2),ans(2,1)
     $        ,unc(2,1),itmax,ncalls0,tTot
      endif
      close(12)

      call mc_real_profile_close()
      call decision_close()
      return
      end


      subroutine select_native_event(event_weight,x,weight)
      use mint_module, only: ndim,ndimmax,max_fold
      implicit none
      double precision event_weight,x(ndimmax),weight,inv_bias
      double precision x_save(ndimmax,max_fold)
      integer sector,ifold_picked
      character*7 event_norm
      common /event_normalisation/event_norm
      common /c_vegas_x_fold/x_save,ifold_picked
      call pick_unweight_contr(sector,ifold_picked)
      call update_fks_dir(sector)
      weight=event_weight
      if (event_norm(1:4).eq.'bias') then
         call include_inverse_bias_wgt(inv_bias)
         weight=event_weight*inv_bias
      endif
      call fill_rwgt_lines
      x(:ndim)=x_save(:ndim,ifold_picked)
      end

      subroutine get_user_params(ncall,nitmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      use mint_module
      use iso_fortran_env, only: input_unit
      implicit none
c
c     Constants
c
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'run.inc'
c
c     Arguments
c
      integer ncall,nitmax
c
c     Local
c
      integer i
      double precision dconfig
c
c     Global
c
      integer             ini_fin_fks
      common/fks_channels/ini_fin_fks
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody
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
c MC counterterm stuff
c
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      
      logical SHsep
      logical Hevents
      common/SHevents/Hevents

      character*7 event_norm
      common /event_normalisation/event_norm
c Les Houches init block (for the <init> info)
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
      double precision dum1,dum2,dum3
      integer p_label,nevts,nevents
      double precision event_weight
c
c MINT stuff
c
      integer ixi_i,iphi_i,iy_ij

c-----
c  Begin Code
c-----
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,nitmax
      write(*,*) 'Number of events and iterations ',ncall,nitmax

      write(*,'(a)') 'Enter desired fractional accuracy: '
      read(*,*) accuracy
      write(*,*) 'Desired fractional accuracy: ',accuracy

      write(*,*)'Enter alpha, beta for G_soft'
      write(*,*)'  Enter alpha<0 to set G_soft=1 (no ME soft)'
      read(*,*)alsf,besf
      write (*,*) 'for G_soft: alpha=',alsf,', beta=',besf 

      write(*,*)'Enter alpha, beta for G_azi'
      write(*,*)'  Enter alpha>0 to set G_azi=0 (no azi corr)'
      read(*,*)alazi,beazi
      write (*,*) 'for G_azi: alpha=',alazi,', beta=',beazi
      i=2
      if (i.eq.0) then
         Hevents=.true.
         write (*,*) 'Doing the H-events'
         SHsep=.true.
      elseif (i.eq.1) then
         Hevents=.false.
         write (*,*) 'Doing the S-events'
         SHsep=.true.
      elseif (i.eq.2) then
         Hevents=.true.
         write (*,*) 'Doing the S and H events together'
         SHsep=.false.
      endif

c These should be ignored (but kept for 'historical reasons')      
      use_cut=2


      write(*,*) 'Suppress amplitude (0 no, 1 yes)? '
      read(*,*) i
      if (i .eq. 1) then
         multi_channel = .true.
         write(*,*) 'Using suppressed amplitude.'
      else
         multi_channel = .false.
         write(*,*) 'Using full amplitude.'
      endif

      write(*,*) 'Exact helicity sum (0 yes, n = number/event)? '
      read(*,*) i
      if (nincoming.eq.1) then
         write (*,*) 'Sum over helicities in the virtuals'/
     $        /' for decay process'
         mc_hel=0
      elseif (i.eq.0.or.(N_VIRTUAL_FLAVOR_CONFIGS.gt.1.and.
     $     .not.HELFILTER_PER_VIRTUAL_FLAVOR)) then
         mc_hel=0
         write (*,*) 'Explicitly summing over helicities'/
     $        /' for the virtuals'
      else
         mc_hel=1
         write(*,*) 'Do MC over helicities for the virtuals'
      endif
      isum_hel = 0

      write(*,'(a)') 'Enter Configuration Number: '
      read(*,*) dconfig
      iconfig = int(dconfig)
      if ( nint(dconfig*10) - iconfig*10 .eq.0 ) then
         ini_fin_fks=0
      elseif ( nint(dconfig*10) -iconfig*10 .eq.1 ) then
         ini_fin_fks=1
      elseif ( nint(dconfig*10) -iconfig*10 .eq.2 ) then
         ini_fin_fks=2
      else
         write (*,*) 'ERROR: invalid configuration number',dconfig
         stop 1
      endif
      do i=1,mapconfig(0,0)
         if (iconfig.eq.mapconfig(i,0)) then
            iconfig=i
            exit
         endif
      enddo
      write(*,*) 'Running Configuration Number: ',iconfig,ini_fin_fks
      nchans=1
      iconfigs(1)=iconfig
      wgt_mult=1d0

      write (*,'(a)') 'Enter running mode for MINT:'
      write (*,'(a)') '0 to set-up grids, 1 to integrate,'//
     &     ' 2 to generate events'
      read (*,*) imode
      write (*,*) 'MINT running mode:',imode
      if (imode.eq.2)then
         write (*,*) 'Generating events, doing only one iteration'
         nitmax=1
      endif

      write (*,'(a)') 'Set the three folding parameters for MINT'
      write (*,'(a)') 'xi_i, y_ij, phi_i'
      read (*,*) ixi_i,iy_ij,iphi_i
      write (*,*)ixi_i,iy_ij,iphi_i


      abrvinput='     '
      write (*,*) "'all ', 'born', 'real', 'virt', 'novi' or 'grid'?"
      write (*,*) "Enter 'born0' or 'virt0' to perform"
      write (*,*) " a pure n-body integration (no S functions)"
      read(*,*) abrvinput
      if(abrvinput(5:5).eq.'0')then
         write (*,*) 'This option is no longer supported:',abrvinput
         stop
        nbody=.true.
      else
        nbody=.false.
      endif
      abrv=abrvinput(1:4)
      if (fks_integrated.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
            write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'Setting abrv to "born".'
            abrv='born'
c$$$            if (ickkw.eq.3) then
c$$$               write (*,*) 'FxFx merging not possible with'/
c$$$     $              /' [LOonly=QCD] processes'
c$$$               stop 1
c$$$            endif
         endif
      endif
      if(nbody.and.abrv.ne.'born'.and.abrv.ne.'virt'
     &     .and. abrv.ne.'grid')then
        write(*,*)'Error in driver: inconsistent input',abrvinput
        stop
      endif

      write (*,*) "doing the ",abrv," of this channel"
      if(nbody)then
        write (*,*) "integration Born/virtual with Sfunction=1"
      else
        write (*,*) "Normal integration (Sfunction != 1)"
      endif
c
      lbw(0)=0

c     Current job files carry an explicit vector size after the run mode.
c     Older native event jobs continue directly with the integer pair.
      call read_mc_job_header(input_unit,imode,native_mc_capacity,
     &     p_label,nevts)
      if (imode.eq.2) then
         read(*,*) nevents,event_weight,dum1,dum2,dum3
         read(*,*) NPRUP
         do i=1,NPRUP
            read(*,*)LPRUP(i),dum1,dum2,XSECUP(i),XERRUP(i)
         enddo
      endif
      if (event_norm(1:5).eq.'unity'.or.event_norm(1:3).eq.'sum') then
         IDWTUP=-3
         if (event_norm(1:5).eq.'unity') then
            XMAXUP(1:NPRUP)=1d0
         elseif(event_norm(1:3).eq.'sum') then
            XMAXUP(1:NPRUP)=XMAXUP(1:NPRUP)/nevents
         endif
      else
         IDWTUP=-4
         if (event_norm(1:4).eq.'bias') then
            XMAXUP(1:NPRUP)=-1d0
         else
            XMAXUP(1:NPRUP)=event_weight
         endif
      endif
      if (event_norm(1:5).eq.'unity') then
         event_weight=1d0
      elseif(event_norm(1:3).eq.'sum') then
         event_weight=event_weight/dble(nevents)
      endif
      end



      function sigintF(xx,vegas_wgt,ifl,f)
      use mc_native_points, only: native_mc_point,mc_wait_real
      use mint_module, only: ndimmax,nintegrals
      implicit none
      double precision sigintF,xx(ndimmax),vegas_wgt,f(nintegrals)
      integer ifl
      type(native_mc_point),save :: point
      point%batched=.false.
      do
         call sigintF_advance(point,xx,vegas_wgt,ifl,f)
         if(point%status.ne.mc_wait_real)exit
         call resolve_native_real_scalar(point%emission%request)
      enddo
      sigintF=0d0
      end

      subroutine sigintF_advance(point,xx,vegas_wgt,ifl,f)
      use mc_native_points, only: native_mc_point,mc_wait_real,
     $     mc_fold_complete
      use mc_point_state, only: real_waiting,complete_native_real
      use mc_native_context, only: mc_prepare_provider_request,
     $     mc_consume_provider_result
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use fks_phase_space_data, only: p_born,p1_cnt
      use fks_phase_space, only: fks_phase_space_point,
     $     generate_born_contribution,generate_real_phase_space
      use mc_native_context, only: mc_begin_real_point,mc_end_real_point
      use FKSParams, only: MCSubtractionAtFixedFlow
      use weight_lines
      use mint_module
      use mc_counterterms, only: fksfather,gfactsf,gfactcl,
     $     mc_shower_scale_mass
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'mc_histories.inc'
      logical firsttime,passcuts,passcuts_native_born,
     $     passcuts_nbody,passcuts_n1body
      type(native_mc_point),intent(inout) :: point
      logical weights_valid,born_cuts_for_flow,
     $     calibrating_spread,folded
      save folded
      double precision raw_total,raw_split(amp_split_size)
      integer i,ifl,proc_map(0:fks_configs,0:fks_configs),nFKS_picked_nbody,nFKS_in,nFKS_out,izero,mohdr,iFKS,sum,
     $     first_native_H,
     $     born_group_count,igroup
      double precision xx(ndimmax),vegas_wgt,f(nintegrals),jac,p(0:3,nexternal),rwgt,x_local(99),probne,
     $     n1body_wgt,p_lab(0:3,nexternal),p_cms(0:3,nexternal),jacPS
      save proc_map,sum
      integer             ini_fin_fks
      common/fks_channels/ini_fin_fks
      external passcuts,passcuts_native_born
      parameter (izero=0,mohdr=-100)
      data firsttime/.true./
c fold is unused; it keeps the /cfl/ layout shared with other files.
      integer     fold,ifold_counter
      common /cfl/fold,ifold_counter
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      logical       nbody
      common/cnbody/nbody
      integer         nndim
      common/tosigint/nndim
      character*4      abrv
      common /to_abrv/ abrv
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      integer ifold_picked
      double precision x_save(ndimmax,max_fold)
      common /c_vegas_x_fold/x_save,ifold_picked
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common /colour_connections/ icolup_s,icolup_h
      integer fks_father
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision wgt1
      logical have_sum_born
      double precision sum_born(0:3,nexternal-1),sum_xbjrk(2)
      double precision p_born_flow(0:3,nexternal),born_ref_scale
      interface
         subroutine advance_native_histories(h,flow,emission,first_native,p,p_lab,
     $        p_cms,jacPS,vegas_wgt,sampling_wgt,born_flow_factor,
     $        outer_point,weights_valid)
            use mc_native_points, only: native_history_cursor,native_flow_cursor
            use mc_point_state, only: native_real_emission
            import fks_phase_space_point,nexternal
            type(native_history_cursor),intent(inout) :: h
            type(native_flow_cursor),intent(inout) :: flow
            type(native_real_emission),intent(inout) :: emission
            integer first_native
            double precision p(0:3,nexternal),
     $           p_lab(0:3,nexternal),p_cms(0:3,nexternal),jacPS,
     $           vegas_wgt,sampling_wgt,born_flow_factor
            type(fks_phase_space_point),intent(in) :: outer_point
            logical,intent(out) :: weights_valid
         end subroutine
      end interface
c
      if(point%status.ne.mc_wait_real)then
c Each fold owns its Born sample; compatible sectors can reuse it.
      if(ifl.ne.2)then
         point%outer%born%valid=.false.
         point%has_chart=.false.
      endif
      have_sum_born=.false.
      calibrating_spread=born_spread_phase.eq.1.or.
     $                   born_spread_phase.eq.2
      if (new_point .and. ifl.ne.2) then
         pass_cuts_check=.false.
      endif
c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
         folded=any(ifold(1:ndim).gt.1)
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
         call setup_proc_map(sum,proc_map,ini_fin_fks)
c For the S-events, we can combine processes when they give identical
c processes at the Born. Make sure we check that we get indeed identical
c IRPOC's
         call find_iproc_map()
c For FxFx matching with pythia8, set the correct attributes
c for the <event> tag in the LHEF file. "npNLO" are the number of Born
c partons in this multiplicity when running the code at NLO accuracy
c ("npLO" is -1 in that case). When running LO only, invert "npLO" and
c "npNLO".
         call setup_event_attributes
      endif

      if (.not.allocated(point%sample%partner))allocate(point%sample%partner(fks_configs))
      if(ifl.ne.0)then
         ifold_counter=point%sample%fold_number
         born_flow_picked=point%sample%born_flow
      endif
      if (ifl.eq.0) then
         point%sequence=0
         ifold_counter=1
         point%sample%rejected=.false.
         nonfinite_contribution=.false.
         nonfinite_type=0
         nonfinite_sector=0
         nonfinite_history=0
      elseif(ifl.eq.1) then
         ifold_counter=ifold_counter+1
      endif

! A failed colour draw or MC map invalidates the entire folded point.
! Keep advancing fold counters, but never retain a later fold by itself.
      if(point%sample%rejected)goto 900
      endif
      if (ifl.eq.0 .or. ifl.eq.1) then
         if(point%status.eq.mc_wait_real)then
            if(point%resume_ifl.ne.ifl)error stop 'Wrong MC fold at real resume'
            if(point%emission%request%status.eq.real_waiting)then
               call mc_consume_provider_result(point%provider,raw_total,
     $              raw_split,point%draw_id,point%sequence,
     $              point%sample%fold_number)
               call complete_native_real(point%emission%request,raw_total,raw_split)
            endif
            call compute_real_emission_finish(point%emission)
            point%status=mc_fold_complete
            weights_valid=.not.nonfinite_contribution
            p=point%outer%p
            p_lab=point%outer%p_lab
            p_cms=point%outer%p_cms
            x_local=point%x_local
            jacPS=point%jacPS
            first_native_H=point%first_H
            have_sum_born=point%shared_born
            if(have_sum_born)then
               sum_born=point%sum_born
               sum_xbjrk=point%sum_xbjrk
            endif
            calibrating_spread=born_spread_phase.eq.1.or.born_spread_phase.eq.2
            passcuts_nbody=point%cuts_born
            passcuts_n1body=point%cuts_real
            if(point%history%waiting)goto 222
            goto 218
         endif
         if (ifl.eq.0) then
            icontr=0
            virt_wgt_mint(0:amp_split_size)=0d0
            born_wgt_mint(0:amp_split_size)=0d0
            virtual_over_born=0d0
            call vs_begin_point
            call reset_folded_virtual
            born_flow_picked=0
         endif
         MCcntcalled=0
         wgt_me_real=0d0
         wgt_me_born=0d0
         if (ickkw.eq.3) call set_FxFx_scale(0,p)
         call update_vegas_x(xx,x_local)
         do i=1,nndim
            x_save(i,ifold_counter)=x_local(i)
         enddo
         call born_spread_set_point(x_local(ndim-2)**2,
     $        x_local(ndim-1)**2)
         born_spread_bin_fold(ifold_counter)=born_spread_current_bin
         born_spread_sector_fold(ifold_counter)=1
         if (ifl.eq.0)
     &        call get_MC_integer(1,proc_map(0,0),point%sample%group,point%sample%group_probability)

c The nbody contributions
         if (abrv.eq.'real') goto 11
         nbody=.true.
c Pick the first one because that's the one with the soft singularity
         nFKS_picked_nbody=proc_map(point%sample%group,1)
         if (sum.eq.0) then
c For sum=0, determine nFKSprocess so that the soft limit gives a non-zero Born
            nFKS_in=nFKS_picked_nbody
            call get_born_nFKSprocess(nFKS_in,nFKS_out)
            nFKS_picked_nbody=nFKS_out
         endif
         call update_fks_dir(nFKS_picked_nbody)
c Native QED sectors can change the Born photon/fermion identity.
c Refresh model charges and IDs before constructing recoil scales.
         if (qed_matching) call init_process_module_nbody_wrapper()
c Keep the Born sector even when later real-emission maps change nFKS.
         born_spread_sector_fold(ifold_counter)=nFKS_picked_nbody
         if (ini_fin_fks.eq.0) then
            jac=1d0
         else
            jac=0.5d0
         endif
c Also the Born needs to be included in the Importance Sampling over the
c FKS configurations (for the shower scale) (multiply by
c 1/proc_map(0,0)*point%sample%group_probability)
         born_group_count=proc_map(0,0)
         if (HAS_PHYSICAL_FKS_CLASSES) then
            born_group_count=0
            do igroup=1,proc_map(0,0)
               if (BORN_FLAVOR_INDEX_D(proc_map(igroup,1)).eq.
     $             BORN_FLAVOR_INDEX_D(nFKS_picked_nbody))
     $             born_group_count=born_group_count+1
            enddo
         endif
         jac=jac/(born_group_count*point%sample%group_probability)
         call generate_born_contribution(nndim,iconfig,jac,x_local,
     $        point%outer)
         point%has_chart=.true.
         p=point%outer%p
         p_lab=point%outer%p_lab
         p_cms=point%outer%p_cms
         if (p_born(0,1).lt.0d0) goto 12
         call check_event_projection(.false.)
         if (FKSExplicitSum) then
            sum_born=point%outer%born%p
            sum_xbjrk=point%outer%born%xbjrk
            have_sum_born=.true.
         endif
         call compute_prefactors_nbody(vegas_wgt)
         call set_cms_stuff(izero)
         if (ickkw.eq.3) call set_FxFx_scale(1,p1_cnt(0,1,0))
         passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
            
         if (passcuts_nbody) then
            pass_cuts_check=.true.
            call set_born_spread_point(x_local(ndim-2),
     $           x_local(ndim-1),ifold_counter)
            call set_alphaS(p1_cnt(0,1,0))
            call include_multichannel_enhance(1)
            if (abrv.eq.'born') then
               ! Doing only the Born contribution.
               call compute_born
               if (ifl.eq.0) then
                  call get_born_flow(born_flow_picked,point%sample%flow_probability)
                  if(born_flow_picked.eq.0)goto 900
               endif
               call Bornonly_shower_scale(p_born,born_flow_picked)
               call save_shower_scale_nbody(nFKS_picked_nbody,
     $              ifold_counter,fksfather)
            elseif (abrv(1:2).eq.'vi') then
               ! Doing only the Virtual contribution (could be because
               ! we are generating a virtual event).
               call compute_nbody_noborn
               if (ifl.eq.0) then
                  call get_born_flow(born_flow_picked,point%sample%flow_probability)
                  if(born_flow_picked.eq.0)goto 900
               endif
               call compute_shower_scale_nbody(p_born,born_flow_picked,
     $              mc_shower_scale_mass())
               call save_shower_scale_nbody(nFKS_picked_nbody,
     $              ifold_counter,fksfather)
            else
               ! Normal: all contributions included. Determine the
               ! shower scale when looping over FKS configurations.
               call compute_born
               if (abrv.ne.'bovi') call compute_ewsudakov
               call compute_nbody_noborn
               ! only for ifl==0, since we want the same flow for each fold.
               if (ifl.eq.0) then
                  call get_born_flow(born_flow_picked,point%sample%flow_probability)
                  if(born_flow_picked.eq.0)goto 900
               endif
               ! We need to fill emsca_S(iFKS_born) with a value that
               ! will be used if we are in the dead-zone. If we are not
               ! in the dead-zone, this will not be used (or
               ! overwritten).
               call compute_shower_scale_nbody(p_born,born_flow_picked,
     $              mc_shower_scale_mass())
               call save_shower_scale_nbody(nFKS_picked_nbody,
     $              ifold_counter,fksfather)
            endif
         elseif (ifl.eq.0) then
            call sborn_frame(p_born,wgt1)
            call get_born_flow(born_flow_picked,point%sample%flow_probability)
            if(born_flow_picked.eq.0)goto 900
! A second massive solution has no soft endpoint, but another fold or
! sector can have one at the same Born point. Check
! the physical Born cuts before marking its shared flow negative.
! Without folding or an explicit FKS sum no other fold or sector can
! give an S event at this Born, so H events keep their dipole scale.
! Born/virtual weights still require the reference sector's endpoint.
            born_cuts_for_flow=.false.
            if(p1_cnt(0,1,0).lt.0d0.and.(folded.or.FKSExplicitSum)
     $           .and.point%outer%born%event_projection)then
               p_born_flow(:,1:i_fks-1)=p_born(:,1:i_fks-1)
               p_born_flow(:,i_fks)=0d0
               p_born_flow(:,i_fks+1:nexternal)=
     $              p_born(:,i_fks:nexternal-1)
! FxFx cuts need the physical Born's clustering labels and scales too.
               born_cuts_for_flow=passcuts_native_born(p_born_flow,
     $              rwgt,born_ref_scale)
            endif
            if(.not.born_cuts_for_flow)
     $           born_flow_picked=-born_flow_picked
         endif
         
 11      continue
c The n+1-body contributions (including counter terms)
         if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') goto 12
         nbody=.false.
         point%sector_index=1
 200     continue
         if(point%sector_index.gt.proc_map(point%sample%group,0))goto 240
            wgt_me_real=0d0
            wgt_me_born=0d0
            iFKS=proc_map(point%sample%group,point%sector_index)
            call update_fks_dir(iFKS)
            if (qed_matching) call init_process_module_nbody_wrapper()
            if (FKSExplicitSum.and.have_sum_born) then
c The preceding sector may have rejected its radiation and invalidated
c the active Born. Starting scales still belong to the shared sample.
               p_born=sum_born
            endif
            fks_father=min(i_fks,j_fks)
            if (born_flow_picked.gt.0) then
!     Consider all flows for the shower scale assignment (with
!     assignements only needed for the dipoles where the fks-mother is
!     one end of the dipole line)
               call compute_shower_scale_nbody(p_born,-fks_father,
     $              mc_shower_scale_mass())
!     assign emsca_S: we know flow (from driver_mintMC) and the
!     father. Therefore the partner is fixed (except when father is a gluon
!     (then there is a two-fold ambiguity)). Determine the partner:

!     Determine the partner of the father (keep it the same for each fold)
               if (ifl.eq.0) call determine_partner(born_flow_picked
     $              ,fksfather,point%sample%partner(iFKS))
!     The shower scale to be used in the event file (if it's an S-event and
!     fks_picked will be iFKS):
               call save_shower_scale_nbody(iFKS,ifold_counter,
     $              fksfather,point%sample%partner(iFKS))
            endif
               
            probne=1d0
            gfactsf=1.d0
            gfactcl=1.d0
            MCcntcalled=0
            icolup_s(1,1)=-1    ! set colour connection to -1: i.e., complete_xmcsubt has not been called
! Apply the outer-sector sampling weight to the counterevents as well.
            jacPS=1d0/point%sample%group_probability
            call generate_real_phase_space(nndim,iconfig,jacPS,
     $           x_local,point%outer)
            point%has_chart=.true.
            p=point%outer%p
            p_lab=point%outer%p_lab
            p_cms=point%outer%p_cms
            jac=jacPS
            jacPS=jacPS*point%sample%group_probability
c Every contribution has to have a viable set of Born momenta (even if
c counter-event momenta do not exist).
            if (p_born(0,1).lt.0d0) goto 230
            call check_event_projection(.true.)
            if (FKSExplicitSum) then
c All S records in this fold must describe the same underlying event.
c Check the sampled Born before native H histories change active data.
               if (have_sum_born) then
                  if (maxval(abs(point%outer%born%p-sum_born)).gt.
     $                1d-10*max(1d0,maxval(abs(sum_born))).or.
     $                maxval(abs(point%outer%born%xbjrk-sum_xbjrk))
     $                .gt.1d-10) then
                     write(*,*) 'Incompatible Born points in ',
     $                    'FKSExplicitSum',iFKS
                     stop 1
                  endif
               else
                  sum_born=point%outer%born%p
                  sum_xbjrk=point%outer%born%xbjrk
                  have_sum_born=.true.
               endif
            endif

! fill the valid_dipole array and fill the H-event shower scale array            
            call init_process_module_n1body_wrapper(born_flow_picked)
            call compute_shower_scale_n1body(p,i_fks,j_fks,
     $           mc_shower_scale_mass())
! The shower scale to be used in the event file (if it's an H-event and
! fks_picked will be iFKS). If the emission is hard, we take the dipole
! scale as upper boundary for the next emissions. On the other hand, to
! make sure that (soft/collinear) real-emission still "cancels the
! divergence in the virtual" these configurations should have a shower
! scale related to the underlying S-event.
            if (born_flow_picked.gt.0) then
               ! TODO: maybe we should also go here when the n-body does not pass the cuts. CHECK.
               if (.not.mcatnlo_delta) then
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =max(shower_scale_n1body(i_fks,j_fks),
     $                 shower_scale_nbody(fks_father
     $                 ,point%sample%partner(iFKS)))
               else
! in the case of MC@NLO-delta, an H-event contribution is by definition
! 'hard', and we should use the corresponding dipole scale for
! subsequent showering.
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(1:ndelH,1:ndelH)
               endif
            else ! we have no Born, so take the n+1-body dipole(s) as
                 ! starting scale.
               if (.not.mcatnlo_delta) then
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(i_fks,j_fks)
               else
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(1:ndelH,1:ndelH)
               endif
            endif
c Compute the n1-body prefactors
            call compute_prefactors_n1body(vegas_wgt,jac)
! This flow was drawn with q_c=p_c at the common outer Born point.
            if (MCSubtractionAtFixedFlow)
     $           call include_born_flow_weight(point%sample%flow_probability,
     $           point%sample%flow_probability)
c Include the FxFx Sudakovs into the prefactors
            if (ickkw.eq.3) then
               call set_FxFx_scale(0,p) ! reset the FxFx scales
               call set_cms_stuff(izero)
               call set_FxFx_scale(2,p1_cnt(0,1,0))
               call set_cms_stuff(mohdr)
               if (p(0,1).gt.0d0) call set_FxFx_scale(3,p)
            endif
c check if event or counter-event passes cuts
            call set_cms_stuff(izero)
            if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
            passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
            if (passcuts_nbody .and. (born_flow_picked .le.0)) then
               write (*,*) 'something funny is going on: passing cuts,'
     $              //' but no born_flow assigned.'
               write (*,*) born_flow_picked
               stop 1
            endif
            
            call set_cms_stuff(mohdr)
            if (ickkw.eq.3) call set_FxFx_scale(-3,p)
            passcuts_n1body=passcuts(p,rwgt)
! Another native Born projection can pass cuts even if this one does
! not. The complete H sum must still be evaluated in that case.
            if (.not.(passcuts_nbody.or.passcuts_n1body) .and.
     $           abrv.eq.'real') goto 230
            first_native_H=icontr+1
! Share full real amplitudes at this physical point. Each native
! history retains its own counterevents and regulated factors; FxFx H
! weights and scales below are taken from the common outer real state.
            call mc_begin_real_point(p)
! Calibration fits S weights only. Retain their (1-P) real term and G
! replacements, but omit H records and the complete inner H-history sum.
! Outside calibration, suppress outer H only when it will be rebuilt.
            mc_S_only=calibrating_spread.or.
     $           (MCExplicitKLSum.and.abrv.ne.'real'.and.
     $           p(0,1).gt.0d0.and.jacPS.gt.0d0.and.MC_HIST_COUNT.gt.0)
 218        continue
            call advance_native_flows(point%flow,point%emission,p,p_cms,
     $           passcuts_nbody,passcuts_n1body,probne,weights_valid)
            if(point%emission%request%status.eq.real_waiting)goto 225

            mc_S_only=.false.
            if(.not.weights_valid)then
               call mc_end_real_point()
               goto 900
            endif
            if(.not.MCExplicitKLSum.or.calibrating_spread.or.
     $           abrv.eq.'real')goto 224
 222        continue
               call advance_native_histories(point%history,point%flow,
     $              point%emission,first_native_H,p,p_lab,
     $              p_cms,jacPS,vegas_wgt,1d0/point%sample%group_probability,point%sample%flow_probability,
     $              point%outer,weights_valid)
               if(point%emission%request%status.eq.real_waiting)goto 225
               if(.not.weights_valid)then
                  call mc_end_real_point()
                  goto 900
               endif
 224        call mc_end_real_point()
            goto 230
 225        continue
            point%sequence=point%sequence+1
            point%status=mc_wait_real
            point%resume_ifl=ifl
            point%has_chart=.true.
            point%x_local=x_local
            point%jacPS=jacPS
            point%first_H=first_native_H
            point%cuts_born=passcuts_nbody
            point%cuts_real=passcuts_n1body
            point%shared_born=have_sum_born
            if(have_sum_born)then
               point%sum_born=sum_born
               point%sum_xbjrk=sum_xbjrk
            endif
            point%sample%fold_number=ifold_counter
            point%sample%born_flow=born_flow_picked
            if(point%batched)call mc_prepare_provider_request(
     $                 point%provider,point%emission%request%p,
     $                 point%draw_id,point%sequence,ifold_counter)
            return
 230        point%sector_index=point%sector_index+1
            goto 200
 240     continue
         call apply_born_spread_weight(ifold_counter)
 12      continue
! Also catch overflow in history/flow rescaling after add_wgt returned.
         if (icontr.gt.0) then
            if (.not.all(ieee_is_finite(wgt(:,1:icontr))))
     $           nonfinite_contribution=.true.
         endif
         if(nonfinite_contribution)goto 900
      elseif(ifl.eq.2) then
         if (ifold_counter .ne.
     $       ifold(ifold_energy)*ifold(ifold_yij)*ifold(ifold_phi)) then
            write (*,*) "ERROR in folding parameters (driver_mintMC.f)"
     $           ,ifold_counter,ifold_energy,ifold_yij,ifold_phi
            write (*,*) ifold(:)
            stop 1
         endif
c Special check: in rare cases there can be S-event contributions,
c without a single FKS configuration that contains a soft singularity
c passing cuts (this only happens if n-body configuration does not pass
c the cuts and DELTA (from complete_xmcsubt) is not equal to 1). Need to
c add a bogus contribution corresponding to an FKS configuration that
c contains a soft singularity to make sure that the code continues
c correctly.
         ! TODO: HOW CAN THIS HAPPEN? delta can only be not equal to one if n-body passes the cuts...
         call special_check_SoftSing(proc_map(point%sample%group,1))
c Include PDFs and alpha_S and reweight to include the uncertainties
         call include_PDF_and_alphas
c Include the weight from the bias_function
         call include_bias_wgt
! PDF, coupling and bias factors are applied after all folds. Check the
! central and flavour weights before combining events or fitting grids.
         do i=1,icontr
            if (.not.ieee_is_finite(wgts(1,i)).or.
     $          .not.all(ieee_is_finite(parton_iproc(1:niproc(i),i))))
     $           nonfinite_contribution=.true.
         enddo
         if(nonfinite_contribution)goto 900
c Sum the contributions that can be summed before taking the ABS value
         call sum_identical_contributions
         call fill_mint_function_NLOPS(f,n1body_wgt)
         if(nonfinite_contribution)goto 900
         call fill_MC_integer(1,point%sample%group,n1body_wgt*point%sample%group_probability)
c Use the final, finite PDF/bias weights of the complete point for the fit.
         call virt_surrogate_fit_point
      endif

      point%sample%fold_number=ifold_counter
      point%sample%born_flow=born_flow_picked
      return
 900  continue
! Nothing from this point may enter integration, unweighting, or the
! Born-spreading fit, including records made before the failure.
      if(nonfinite_contribution.and..not.point%sample%rejected)then
         rejected_nonfinite_points=rejected_nonfinite_points+1
         if(rejected_nonfinite_points.le.5.or.
     $        mod(rejected_nonfinite_points,100).eq.0)then
            write(*,*) 'Rejecting phase-space point: nonfinite weight',
     $           '; count =',rejected_nonfinite_points
            write(*,*) 'Contribution type, sector, native history:',
     $           nonfinite_type,nonfinite_sector,nonfinite_history
         endif
         if(rejected_nonfinite_points.eq.1)then
            write(*,*) 'Rejected fold and integration coordinates:',
     $           ifold_counter,x_save(1:nndim,ifold_counter)
         endif
      endif
      point%sample%rejected=.true.
      call vs_begin_point
      call mc_set_history(0)
      icontr=0
      born_flow_picked=0
      point%sample%flow_probability=0d0
      virt_wgt_mint=0d0
      born_wgt_mint=0d0
      virtual_over_born=0d0
      pass_cuts_check=.false.
      f=0d0
      if(ifl.eq.2)call fill_MC_integer(1,point%sample%group,0d0)
      point%sample%fold_number=ifold_counter
      point%sample%born_flow=born_flow_picked
      return
      contains
      subroutine check_event_projection(sector_sum)
c Folds must share their Born point, also in Born-only and virtual-only
c runs. FKSExplicitSum only matters where sectors are summed: Born-only
c and virtual-only runs never reach the real-emission sector loop.
      logical,intent(in) :: sector_sum
      if (point%outer%born%event_projection) return
      if (folded) then
         write(*,*) 'Folding requires event projection'
         stop 1
      elseif (sector_sum.and.FKSExplicitSum) then
         write(*,*) 'FKSExplicitSum requires event projection'
         stop 1
      endif
      end subroutine check_event_projection
      end

      subroutine advance_native_histories(h,flow,emission,first_native,p,p_lab,p_cms,
     $     jacPS,vegas_wgt,sampling_wgt,born_flow_factor,outer_point,
     $     weights_valid)
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use mc_native_points, only: native_history_cursor,native_flow_cursor
      use mc_point_state, only: native_real_emission,real_waiting
      use fks_phase_space_data,only: p_born,xi_i_fks_ev,y_ij_fks_ev,xinorm_ev,p1_cnt,ybst_til_tolab
      use FKSParams, only: MCSubtractionAtFixedFlow
! At a fixed real point form Hhat_a = S_a sum_b P_b (S_b R - M_b).
! The ordinary S records have already been made and are not changed.
! Each M_b includes its native G replacement, luminosities and Born map.
! Each inner history uses its OWN Born colour basis. Fixed-flow mode
! samples a flow with 1/q_b,c; summed mode enumerates that basis.
! The outer event colour is only the event owner, not an inner proposal.
! Thus the colour-sampled summand is
! P_b,c*(p_b,c*S_b*R-M_b,c)/q_b,c, with M_b,c already flow-weighted.
! Summed mode instead accumulates P_b,c*(p_b,c*S_b*R-M_b,c) over c.
      use fks_phase_space, only: fks_phase_space_point,
     $     initialize_fks_phase_space,capture_fks_phase_space,
     $     restore_fks_phase_space,generate_native_momenta
      use mc_native_context, only: set_native_history,
     $     native_mapping
      use weight_lines, only: icontr,H_event,wgt,event_nFKS,momenta,
     $     momenta_m,y_bst,need_match,mc_H_only,
     $     rejected_mc_kinematics_points
      use mint_module, only: iconfig
      use process_module, only: ndelH
      use fks_phase_space_helpers, only: apply_momentum_permutation,
     $     boost_n1_to_lab
      use mc_counterterms, only: fksfather,gfactsf,gfactcl,gfactazi,
     $     prepare_mc_kinematics,mc_shower_scale_mass
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      include 'run.inc'
      include 'fks_symmetry.inc'
      include 'mc_histories.inc'
      type(fks_phase_space_point),intent(in) :: outer_point
      logical,intent(out) :: weights_valid
      type(native_history_cursor),intent(inout) :: h
      type(native_flow_cursor),intent(inout) :: flow
      type(native_real_emission),intent(inout) :: emission
      integer first_native,last_native,ict,i,j
      integer this_config
      common /to_mconfigs/this_config
      double precision p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),jacPS,vegas_wgt,
     $     sampling_wgt,born_flow_factor,native_measure,probne_native,
     $     flow_factor_native,rwgt,outer_channel,mc_outer_channel_weight,
     $     native_born_ref_scale
      double precision factors(9),mc_factors(8),pdf_factors(3)
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      logical passcuts,passcuts_native_born,
     $     native_valid
      character(len=48) rejection_reason
      double precision fks_Sij
      external fks_Sij,passcuts,passcuts_native_born
      double precision born_weight
      external mc_outer_channel_weight
      integer nFKSprocess,i_fks,j_fks
      common/c_nFKSprocess/nFKSprocess
      common/fks_indices/i_fks,j_fks
      double precision pmass(nexternal)
      common/to_mass/pmass
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      integer fold,ifold_counter,MCcntcalled
      common/cfl/fold,ifold_counter
      common/c_MCcntcalled/MCcntcalled
      integer need_matching_S(nexternal),need_matching_H(nexternal),
     $     need_matching_cuts(nexternal)
      common/c_need_matching/need_matching_S,need_matching_H,
     $     need_matching_cuts
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common/colour_connections/icolup_s,icolup_h

      weights_valid=.true.
      if(h%waiting)goto 430
      if (p(0,1).le.0d0 .or. jacPS.le.0d0) return
      if (MC_HIST_COUNT.eq.0) return
      call set_cms_stuff(-100)
      h%owner=nFKSprocess
      if (.not.MC_HIST_COMPLETE(h%owner)) then
         write (*,*) 'Incomplete MC history sum for FKS sector',h%owner
         write (*,*) 'Required global histories are unresolved or',
     $        ' ambiguous; see Source/BornSupport/registry.json'
         stop 1
      endif
      h%outer_config=iconfig
      outer_channel=mc_outer_channel_weight()
      h%outer_boost=ybst_til_tolab
      h%owner_match=need_matching_H
      h%matching(:,1)=need_matching_S
      h%matching(:,2)=need_matching_H
      h%matching(:,3)=need_matching_cuts
      h%sector_weight=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      last_native=icontr
      do ict=first_native,last_native
         if (H_event(ict)) wgt(:,ict)=0d0
      enddo
      if (h%sector_weight.eq.0d0) return

! K_a is the full outer real measure, including its sampling probability
! and statistical compensation, and the outer matrix-element channel weight.
! This compensation can exceed the strict MC orbit for tagged photons.
      h%outer_measure=xinorm_ev*xi_i_fks_ev*jacPS*vegas_wgt*
     $     sampling_wgt*fkssymmetryfactor*outer_channel
      if (h%outer_measure.le.0d0) return
      if (fkssymmetryfactor.ne.
     $     dble(FKS_FAC_I_D(h%owner)*FKS_FAC_J_D(h%owner))) then
         write (*,*) 'Incorrect outer MC H orbit factor',h%owner
         stop 1
      endif
      if (MC_HIST_OWN(h%owner).lt.MC_HIST_FIRST(h%owner) .or.
     $     MC_HIST_OWN(h%owner).gt.MC_HIST_LAST(h%owner)) then
         write (*,*) 'Missing native entry in MC history table',h%owner
         stop 1
      endif
      h%gfun=[gfactsf,gfactcl,gfactazi]
      h%flow=born_flow_picked
      h%called=MCcntcalled
      h%colours_s=icolup_s
      h%colours_h=icolup_h
      h%hard_scale=shower_scale_hard
      h%nbody_scales(:,:,1)=shower_scale_nbody
      h%nbody_scales(:,:,2)=shower_scale_nbody_min
      h%nbody_scales(:,:,3)=shower_scale_nbody_max
      h%n1body_scales=shower_scale_n1body
      h%emsca=emsca_H(:,ifold_counter,:,:)
      h%factors=factors
      h%mc_factors=mc_factors
      h%pdf_factors=pdf_factors
! Matching refreshes some active phase-space data, including the FSR
! momentum factors. Preserve their current values with the original
! generated momenta and measure before visiting the native histories.
      h%saved_point=outer_point
      call capture_fks_phase_space(h%saved_point)
      mc_H_only=.true.
      native_mapping=.true.

! The exporter supplies unique ordered histories, including both gg
! orientations, with native Born/order identities and checked label maps.
! Every row has unit multiplicity. Recompute the owner too, using the
! same measure conversion and channel-free density as all other rows.
      if(.not.allocated(h%recoil_configs))allocate(h%recoil_configs(lmaxconfigs),h%group_of(lmaxconfigs))
      h%ihist=MC_HIST_FIRST(h%owner)
 410  if(h%ihist.gt.MC_HIST_LAST(h%owner))goto 900
         h%sector=MC_HIST_NATIVE(h%ihist)
         h%ii=MC_HIST_I(h%ihist)
         h%jj=MC_HIST_J(h%ihist)
         iconfig=1
         call update_fks_dir(h%sector)
         call set_native_history(h%ihist)
         call init_process_module_nbody_wrapper()
         call update_coltype_and_charge(h%sector,i_fks,j_fks)
         if (fkssymmetryfactor.ne.
     $        dble(FKS_FAC_I_D(h%sector)*FKS_FAC_J_D(h%sector)) .or.
     $        MC_HIST_PERM(i_fks,h%ihist).ne.h%ii .or.
     $        MC_HIST_PERM(j_fks,h%ihist).ne.h%jj) then
            write (*,*) 'Incorrect native MC H orbit',h%owner,h%ihist
            stop 1
         endif
         call apply_momentum_permutation(MC_HIST_PERM(:,h%ihist),
     $        p_lab,h%p_flipped,native_valid)
         if (.not.native_valid) then
            rejection_reason='invalid native MC momentum permutation'
            goto 890
         endif
! Sum the recoil histories as well as the FKS histories. A native FKS
! sector may have both production and decay Born diagrams; its own
! diagram partition supplies their weights at their Born projections.
         call native_recoil_groups(h%nrecoil,h%recoil_configs,
     $        h%group_of)
         h%recoil=1
 420     if(h%recoil.gt.h%nrecoil)goto 450
            iconfig=h%recoil_configs(h%recoil)
            call select_native_recoil(iconfig)
! Only the radiation projection and counter/real measure ratios are
! needed here. The common Born sampling factor cancels from the H density.
            call generate_native_momenta(h%p_flipped,h%pn,h%pn_lab,h%pn_cms,
     $           h%jac_native,native_valid)
            this_config=iconfig
            if (.not.native_valid) then
               rejection_reason='invalid native MC H radiation projection'
               goto 890
            endif
            native_measure=xinorm_ev*xi_i_fks_ev*h%jac_native*
     $           fkssymmetryfactor
            if (.not.ieee_is_finite(native_measure).or.
     $           native_measure.le.0d0) then
               rejection_reason='invalid native MC H measure'
               goto 890
            endif

! Divide out the native real measure of the COMPLETE generated H weight
! and insert K_a. This is not an extra Jacobian on a raw MC density:
! its native K_b cancels exactly. The counter/real measure ratios inside
! the G replacement, however, must be retained. The inner orbit factor
! cancels too, since the labelled histories are explicitly enumerated.
            h%factor=h%sector_weight*h%outer_measure/native_measure
            MCcntcalled=0
            call prepare_mc_kinematics(h%pn_cms,i_fks,j_fks,
     $           xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
            call compute_prefactors_n1body(1d0,h%jac_native)
            if(h%nrecoil.gt.1)
     $           call include_native_recoil_weights(h%recoil,
     $           h%group_of)
            if (ickkw.eq.3) then
! All kl terms describe the same n+1-body H event. Reuse its outer ij
! FxFx Sudakov and scales; only the particle labels need permuting.
! The native Born clustering is needed separately for subtraction cuts.
               need_matching_H=h%owner_match(MC_HIST_PERM(:,h%ihist))
               call set_FxFx_scale(4,h%pn)
            endif
            call set_cms_stuff(0)
            if (ickkw.eq.3) call set_FxFx_scale(-3,h%pn)
! Sample in this history's own Born basis. Reusing the outer label
! would require a flow map and support at a different Born point.
! q_b,c=p_b,c here; no additional outer 1/q_a,c belongs on this term.
            call set_alphaS(p1_cnt(0,1,0))
            call sborn_frame(p_born,born_weight)
            call get_born_flow(born_flow_picked,flow_factor_native)
            if(born_flow_picked.eq.0)then
               weights_valid=.false.
               goto 900
            endif
            if (MCSubtractionAtFixedFlow)
     $           call include_born_flow_weight(flow_factor_native,
     $           flow_factor_native)
            call init_process_module_n1body_wrapper(born_flow_picked)
            if (ickkw.eq.3) then
! Keep the common H event's dipole scales with its labels permuted.
! The native Born starting scale and damping bounds are set below from
! the same Born clustering that supplies the subtraction cuts.
               do j=1,nexternal
                  do i=1,nexternal
                     shower_scale_n1body(i,j)=h%n1body_scales(
     $                    MC_HIST_PERM(i,h%ihist),MC_HIST_PERM(j,h%ihist))
                  enddo
               enddo
            else
               call compute_shower_scale_nbody(p_born,-fksfather,
     $              mc_shower_scale_mass())
               call compute_shower_scale_n1body(h%pn,i_fks,j_fks,
     $              mc_shower_scale_mass())
            endif
            h%cuts_born=passcuts_native_born(p1_cnt(0,1,0),rwgt,
     $           native_born_ref_scale)
            if (ickkw.eq.3)call set_FxFx_shower_scale_nbody(
     $           native_born_ref_scale)
            call set_cms_stuff(-100)
            if (ickkw.eq.3) call set_FxFx_scale(-3,h%pn)
            h%cuts_real=passcuts(h%pn,rwgt)
            h%first_alt=icontr+1
 430        continue
            call advance_native_flows(flow,emission,h%pn,h%pn_cms,h%cuts_born,
     $           h%cuts_real,probne_native,weights_valid)
            if(emission%request%status.eq.real_waiting)then
               h%waiting=.true.
               return
            endif
            h%waiting=.false.
            if(.not.weights_valid)goto 900
            do ict=h%first_alt,icontr
               if (.not.H_event(ict)) then
                  write (*,*) 'S event entered the inner MC H sum',h%ihist
                  stop 1
               endif
               wgt(:,ict)=wgt(:,ict)*h%factor
! Keep BOTH native momentum sets for ME reweighting, expressed in the
! outer frame. Only event kinematics and shower ownership are outer.
               event_nFKS(ict)=h%owner
               call boost_n1_to_lab(momenta_m(:,:,1,ict),h%pn_cms,
     $              y_bst(ict)-h%outer_boost)
               momenta_m(:,:,1,ict)=h%pn_cms
               call boost_n1_to_lab(momenta_m(:,:,2,ict),h%pn_cms,
     $              y_bst(ict)-h%outer_boost)
               momenta_m(:,:,2,ict)=h%pn_cms
               momenta(:,:,ict)=p
               y_bst(ict)=h%outer_boost
               need_match(:,ict)=h%owner_match
            enddo
            h%recoil=h%recoil+1
            goto 420
 450     h%ihist=h%ihist+1
         goto 410
 890  continue
! Drop the COMPLETE folded point, including earlier S and H records.
! Do not keep a partial history sum when a native map is unreliable.
      weights_valid=.false.
      rejected_mc_kinematics_points=rejected_mc_kinematics_points+1
      if(rejected_mc_kinematics_points.le.5.or.
     $     mod(rejected_mc_kinematics_points,100).eq.0)then
         write(*,*) 'Rejecting phase-space point: ',
     $        trim(rejection_reason),'; count =',
     $        rejected_mc_kinematics_points
         write(*,*) 'Outer sector, history, native sector, i, j:',
     $        h%owner,h%ihist,h%sector,h%ii,h%jj
      endif
      if(rejected_mc_kinematics_points.eq.1)then
         write(*,*) 'Rejected outer lab momenta (E, px, py, pz):'
         do i=1,nexternal
            write(*,'(i4,4es26.17e3)') i,p_lab(:,i)
         enddo
      endif
 900  continue
! Also restore the outer state after a failed native colour draw/map. The
! caller then discards every contribution from this integration point.
! Restore the outer channel and its generated phase-space point.
! The snapshot includes all endpoint and recoil data, so native
! histories need not be followed by another radiation generation.
      call set_native_history(0)
      native_mapping=.false.
      iconfig=h%outer_config
      call update_fks_dir(h%owner)
      call init_process_module_nbody_wrapper()
      call update_coltype_and_charge(h%owner,i_fks,j_fks)
      call initialize_fks_phase_space(iconfig)
      call restore_fks_phase_space(h%saved_point)
      h%jac_native=h%saved_point%weight
      if (h%jac_native.le.0d0 .or. p_born(0,1).le.0d0) then
         write (*,*) 'Could not restore outer FKS point after MC H sum'
         stop 1
      endif
      born_flow_picked=h%flow
      call init_process_module_n1body_wrapper(born_flow_picked)
      shower_scale_hard=h%hard_scale
      shower_scale_nbody=h%nbody_scales(:,:,1)
      shower_scale_nbody_min=h%nbody_scales(:,:,2)
      shower_scale_nbody_max=h%nbody_scales(:,:,3)
      shower_scale_n1body=h%n1body_scales
      emsca_H(:,ifold_counter,:,:)=h%emsca
      icolup_s=h%colours_s
      icolup_h=h%colours_h
      MCcntcalled=h%called
      call prepare_mc_kinematics(p_cms,i_fks,j_fks,
     $     xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
      gfactsf=h%gfun(1)
      gfactcl=h%gfun(2)
      gfactazi=h%gfun(3)
      need_matching_S=h%matching(:,1)
      need_matching_H=h%matching(:,2)
      if (ickkw.eq.3) then
! The outer Born/real FxFx cache was never replaced by a native one.
! Restore prefactors without another clustering or Sudakov evaluation.
         factors=h%factors
         mc_factors=h%mc_factors
         pdf_factors=h%pdf_factors
         call set_FxFx_scale(-3,p)
      else
         call compute_prefactors_n1body(vegas_wgt,h%jac_native)
         if (MCSubtractionAtFixedFlow)
     $        call include_born_flow_weight(born_flow_factor,
     $        born_flow_factor)
      endif
      call set_cms_stuff(-100)
      call set_alphaS(p)
      need_matching_cuts=h%matching(:,3)
      mc_H_only=.false.
      end

      logical function passcuts_native_born(p,rwgt,born_ref_scale)
! MC subtraction is cut on its own underlying Born process. In FxFx,
! Born momenta alone are not enough: passcuts also uses the clustering
! scales and the labels distinguishing QCD jets from EW decay products.
! The real-event labels can omit the zero-momentum FKS slot and leave
! an unresolved native Born jet uncut. Obtain the native Born cut data
! without replacing the outer Born/real caches or their weight factors.
! Return its minimum clustering scale for the native shower bounds.
      implicit none
      include 'timing_variables.inc'
      include 'nexternal.inc'
      include 'run.inc'
      double precision p(0:3,nexternal),rwgt,sudakov,expanded_sudakov,
     $     fac_save(2),ren_save(0:nexternal)
      double precision,intent(out) :: born_ref_scale
      double precision FxFx_fac_scale(2),FxFx_ren_scales(0:nexternal)
      integer nFxFx_ren_scales,nscales_save
      common/c_FxFx_scales/FxFx_fac_scale,FxFx_ren_scales,
     $     nFxFx_ren_scales
      integer need_matching_S(nexternal),need_matching_H(nexternal),
     $     need_matching_cuts(nexternal),born_matching(nexternal),
     $     matching_save(nexternal),i_fks,j_fks
      common/c_need_matching/need_matching_S,need_matching_H,
     $     need_matching_cuts
      common/fks_indices/i_fks,j_fks
      logical passcuts
      external passcuts

      if(timing_sampled)call timing_enter(timing_cuts)
      born_ref_scale=-1d0
      if(ickkw.ne.3)then
         passcuts_native_born=passcuts(p,rwgt)
         goto 9909
      endif
      fac_save=FxFx_fac_scale
      ren_save=FxFx_ren_scales
      nscales_save=nFxFx_ren_scales
      matching_save=need_matching_cuts
! Clustering-only mode returns the Born labels/scales without computing
! a Sudakov factor. It does not change the set_FxFx_scale caches.
      call cluster_and_reweight(0,sudakov,expanded_sudakov,
     $     nFxFx_ren_scales,FxFx_ren_scales,FxFx_fac_scale(1),
     $     born_matching,.true.)
      FxFx_fac_scale(2)=FxFx_fac_scale(1)
      born_ref_scale=minval(FxFx_ren_scales(0:nFxFx_ren_scales))
! Jet counting expects the zero-momentum FKS slot to remain present.
      need_matching_cuts=[born_matching(1:i_fks-1),1,
     $     born_matching(i_fks:nexternal-1)]
      passcuts_native_born=passcuts(p,rwgt)
      FxFx_fac_scale=fac_save
      FxFx_ren_scales=ren_save
      nFxFx_ren_scales=nscales_save
      need_matching_cuts=matching_save
 9909 continue
      if(timing_sampled)call timing_leave(timing_cuts)
      return
      end

      subroutine init_process_module_nbody_wrapper()
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      integer colour(1:nexternal-1),i,j,k,get_color
      external get_color
      double precision mass(1:nexternal-1),get_mass_from_id
      external get_mass_from_id
      logical valid_dipole(1:nexternal-1,1:nexternal-1,1:max_bcol)
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      double precision particle_charge_born(nexternal-1)
      common /c_charges_born/particle_charge_born

      do i=1,nexternal-1
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      do k=1,max_bcol
         do j=1,nexternal-1
            if (icolup(1,j,k).eq.0 .and. icolup(2,j,k).eq.0) cycle
            do i=1,nexternal-1
               if (i.eq.j) cycle
               if (icolup(1,i,k).eq.0 .and. icolup(2,i,k).eq.0) cycle
               if ( (abs(icolup(1,i,k)).eq.abs(icolup(1,j,k)).and.icolup(1,i,k).ne.0) .or.
     &              (abs(icolup(1,i,k)).eq.abs(icolup(2,j,k)).and.icolup(1,i,k).ne.0) .or.
     &              (abs(icolup(2,i,k)).eq.abs(icolup(1,j,k)).and.icolup(2,i,k).ne.0) .or.
     &              (abs(icolup(2,i,k)).eq.abs(icolup(2,j,k)).and.icolup(2,i,k).ne.0) ) then
                  valid_dipole(i,j,k)=.true.
               endif
            enddo
         enddo
      enddo
      call init_process_module_nbody(nexternal-1,mass,colour
     $     ,max_bcol,valid_dipole,idup(:,1),particle_charge_born)
      
      end

      
      subroutine compute_NLOPS_flow_weights(p,p_cms,passcuts_nbody,
     $     passcuts_n1body,probne,weights_valid)
      use mc_point_state, only: native_flow_cursor
      use mc_point_state, only: native_real_emission,real_waiting
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),p_cms(0:3,nexternal),probne
      logical passcuts_nbody,passcuts_n1body,weights_valid
      type(native_flow_cursor) state
      type(native_real_emission) emission
      do
         call advance_native_flows(state,emission,p,p_cms,passcuts_nbody,
     $        passcuts_n1body,probne,weights_valid)
         if(emission%request%status.ne.real_waiting)exit
         call resolve_native_real_scalar(emission%request)
         call compute_real_emission_finish(emission)
      enddo
      end

      subroutine advance_native_flows(state,emission,p,p_cms,passcuts_nbody,
     $     passcuts_n1body,probne,weights_valid)
      use mc_point_state, only: native_flow_cursor
      use mc_point_state, only: native_real_emission,real_waiting
! Sum complete native contributions before unweighting, including the
! flow-dependent G replacement and Delta/real H-S split. The sampled
! outer flow remains the event colour and shower-scale owner. Native
! inner histories use their own Born colour flows.
      use FKSParams, only: MCSubtractionAtFixedFlow
      use fks_phase_space_data, only: p_born,p1_cnt
      use scale_module, only: born_flow_picked,shower_scale_n1body,
     $     emsca_H,compute_shower_scale_n1body
      use weight_lines, only: mc_H_only,nonfinite_contribution
      use process_module, only: ndelH,valid_dipole_n1
      use mc_counterterms, only: mc_shower_scale_mass,
     $     gfactsf,gfactcl,gfactazi
      implicit none
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'run.inc'
      type(native_flow_cursor),intent(inout) :: state
      type(native_real_emission),intent(inout) :: emission
      double precision p(0:3,nexternal),p_cms(0:3,nexternal),probne,
     $     born_weight
      logical passcuts_nbody,passcuts_n1body,weights_valid
      integer num_leading_cflows
      logical is_leading_cflow(max_bcol)
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows
! Keep the prefactors BEFORE multichannel enhancement; the single-flow
! evaluator multiplies these in place on each call.
      double precision factors(9),mc_factors(8),pdf_factors(3)
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      integer nFKSprocess,i_fks,j_fks,fold,ifold_counter,
     $     MCcntcalled
      common/c_nFKSprocess/nFKSprocess
      common/fks_indices/i_fks,j_fks
      common/cfl/fold,ifold_counter
      common/c_MCcntcalled/MCcntcalled
      character*4 abrv
      common/to_abrv/abrv
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common/colour_connections/icolup_s,icolup_h

      weights_valid=.true.
      if(state%stage.eq.1)goto 310
      if(state%stage.eq.2)goto 330
      if (MCSubtractionAtFixedFlow.or..not.passcuts_nbody.or.
     $     abrv.eq.'real') then
         call prepare_native_NLOPS_weights(p,p_cms,passcuts_nbody,
     $        passcuts_n1body,state%probability,emission)
         if(emission%request%status.eq.real_waiting)then
            state%stage=1
            return
         endif
         goto 310
      endif
      goto 315
 310  weights_valid=.not.nonfinite_contribution
      probne=state%probability
      state%stage=0
      return
 315  continue

! The physical partition p_c belongs to this history's underlying Born
! point and coupling scale. get_mbar supplies its OWN flow fraction in
! the raw MC Born at the real coupling scale; do not multiply it twice.
      call set_cms_stuff(0)
      if (ickkw.eq.3) then
         if (mc_H_only) then
            call set_FxFx_scale(-3,p)
         else
            call set_FxFx_scale(-2,p1_cnt(0,1,0))
         endif
      endif
      call set_alphaS(p1_cnt(0,1,0))
      call sborn_frame(p_born,born_weight)
      if(.not.allocated(state%flow_weights))
     $     allocate(state%flow_weights(max_bcol))
      call get_born_flow_weights(state%flow_weights,state%sumborn)
      state%leading=is_leading_cflow
      if (state%sumborn.le.0d0) then
         weights_valid=.false.
         probne=1d0
         return
      endif
      state%flow_owner=born_flow_picked
      state%factors=factors
      state%mc_factors=mc_factors
      state%pdf_factors=pdf_factors
      state%dipoles=valid_dipole_n1
      state%scales=shower_scale_n1body
      state%emsca_base=emsca_H(nFKSprocess,ifold_counter,:,:)
      state%emsca_owner=state%emsca_base
      state%colours_s=icolup_s
      state%colours_h=icolup_h
      state%called_owner=MCcntcalled
      state%gfun=[gfactsf,gfactcl,gfactazi]
      state%probability=0d0
      state%flow=1
 320  if(state%flow.gt.max_bcol)goto 350
         if (.not.state%leading(state%flow)) goto 340
! Include even zero-p_c flows: their MC Born fraction can differ at
! the real coupling scale in processes with mixed Born orders.
         state%flow_fraction=state%flow_weights(state%flow)/state%sumborn
         born_flow_picked=state%flow
         factors=state%factors
         mc_factors=state%mc_factors
         pdf_factors=state%pdf_factors
         call include_born_flow_weight(state%flow_fraction,1d0)
         MCcntcalled=0
! Reuse the event's colour-insertion variate and the already sampled
! all-flow Born dipole scales. Internal flows must not overwrite the
! saved H-event colours or draw a new shower starting scale.
         call init_process_module_n1body_flow(state%flow,.true.)
! Native FxFx H histories keep the outer ij shower scales, also when
! summing their Born flows. Recomputing here would recluster per flow.
         if (.not.(ickkw.eq.3.and.mc_H_only))
     $        call compute_shower_scale_n1body(p,i_fks,j_fks,
     $        mc_shower_scale_mass())
         emsca_H(nFKSprocess,ifold_counter,:,:)=state%emsca_base
         call prepare_native_NLOPS_weights(p,p_cms,passcuts_nbody,
     $        passcuts_n1body,state%flow_probability,emission)
         if(emission%request%status.eq.real_waiting)then
            state%stage=2
            return
         endif
 330     continue
         if(nonfinite_contribution)then
            weights_valid=.false.
            goto 350
         endif
         state%probability=state%probability+state%flow_fraction*state%flow_probability
         if (state%flow.eq.state%flow_owner) then
! Delta replaces the H scale with this owner's stopping scales. Retain
! that replacement, not the pre-Delta scale or the last flow's scale.
            state%emsca_owner=emsca_H(nFKSprocess,ifold_counter,:,:)
            state%colours_s=icolup_s
            state%colours_h=icolup_h
            state%called_owner=MCcntcalled
            state%gfun=[gfactsf,gfactcl,gfactazi]
         endif
 340     state%flow=state%flow+1
         goto 320
 350  continue
      state%stage=0
      probne=state%probability
      born_flow_picked=state%flow_owner
      factors=state%factors
      mc_factors=state%mc_factors
      pdf_factors=state%pdf_factors
      valid_dipole_n1=state%dipoles
      shower_scale_n1body=state%scales
      emsca_H(nFKSprocess,ifold_counter,:,:)=state%emsca_owner
      icolup_s=state%colours_s
      icolup_h=state%colours_h
      MCcntcalled=state%called_owner
      gfactsf=state%gfun(1)
      gfactcl=state%gfun(2)
      gfactazi=state%gfun(3)
      end

      subroutine init_process_module_n1body_wrapper(bornflow)
      implicit none
      integer bornflow
      call init_process_module_n1body_flow(bornflow,.false.)
      end


      subroutine init_process_module_n1body_flow(bornflow,
     $     preserve_event_owner)
      use process_module
      use weight_lines, only: mc_H_only
      use scale_module, only: event_colour_H
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer colour(1:nexternal),i,j,get_color,bornflow
      logical preserve_event_owner
      double precision mass(1:nexternal),get_mass_from_id
      external get_color
      external get_mass_from_id
      logical valid_dipole(1:nexternal,1:nexternal)
      integer icolup(1:2,1:nexternal)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     dummy(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,dummy,niprocs
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge
      integer nFKSprocess,fold,ifold_counter
      common/c_nFKSprocess/nFKSprocess
      common/cfl/fold,ifold_counter

      if (bornflow.ne.0) then
         ! take ABS because bornflow is negative if n-body did not pass the cuts
! Keep the same colour-insertion variate throughout the native H sum
! and its outer-state restoration; do not redraw the event's colour.
         call fill_icolor_H(abs(bornflow),jpart,
     $        .not.(mc_H_only.or.preserve_event_owner))
      else
         write (*,*) 'Born-flow not set in n1body_wrapper'
         stop 1
      endif
      do i=1,nexternal
         ICOLUP(1,i)=jpart(4,i)
         ICOLUP(2,i)=jpart(5,i)
      enddo
! Keep the outer colour assignment with its sector and fold. Inner
! histories and their restoration must not overwrite event ownership.
      if (.not.(mc_H_only.or.preserve_event_owner)) then
         event_colour_H(:,:,nFKSprocess,ifold_counter)=ICOLUP
      endif
      
      do i=1,nexternal
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      do j=1,nexternal
         if (icolup(1,j).eq.0 .and. icolup(2,j).eq.0) cycle
         do i=1,nexternal
            if (i.eq.j) cycle
            if (icolup(1,i).eq.0 .and. icolup(2,i).eq.0) cycle
            if ( (abs(icolup(1,i)).eq.abs(icolup(1,j)).and.icolup(1,i).ne.0) .or.
     &           (abs(icolup(1,i)).eq.abs(icolup(2,j)).and.icolup(1,i).ne.0) .or.
     &           (abs(icolup(2,i)).eq.abs(icolup(1,j)).and.icolup(2,i).ne.0) .or.
     &           (abs(icolup(2,i)).eq.abs(icolup(2,j)).and.icolup(2,i).ne.0) ) then
               valid_dipole(i,j)=.true.
            endif
         enddo
      enddo
      
      call init_process_module_n1body(nexternal,mass,colour
     $     ,maxflow,valid_dipole,idup(:,1),particle_charge)
      
      end
      

      subroutine setup_proc_map(sum,proc_map,ini_fin_fks)
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
      use FKSParams, only: FKSExplicitSum
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      double precision lum,dlum
      external dlum
      logical found_ini1,found_ini2,found_fnl
      integer proc_map(0:fks_configs,0:fks_configs)
     $     ,j_fks_proc(fks_configs),i_fks_pdg_proc(fks_configs)
     $     ,j_fks_pdg_proc(fks_configs),i,sum,j,ini_fin_fks
      integer all_fks(fks_configs),n_all
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      sum=3
      do nFKSprocess=1,fks_integrated
         call fks_inc_chooser()
c Set Bjorken x's to some random value before calling the dlum() function
         xbk(1)=0.5d0
         xbk(2)=0.5d0
         lum=dlum()  ! updates IPROC
      enddo
      write (*,*) 'Total number of FKS directories is', fks_configs
c Physical rows group only with their own underlying Born identity.
      if (sum.eq.3.and.HAS_PHYSICAL_FKS_CLASSES) then
         call setup_physical_proc_map(proc_map,ini_fin_fks)
         goto 20
      endif
c For sum over identical FKS pairs, need to find the identical structures
      if (sum.eq.3) then
c MC over FKS pairs that have soft singularity
         proc_map(0,0)=0
         do i=1,fks_integrated
            proc_map(i,0)=0
            i_fks_pdg_proc(i)=0
            j_fks_pdg_proc(i)=0
            j_fks_proc(i)=0
         enddo
c First find all the nFKSprocesses that have a soft singularity and put
c them in the process map
         do nFKSprocess=1,fks_integrated
            call fks_inc_chooser()
            if (ini_fin_fks.eq.1 .and. j_fks.le.nincoming) cycle
            if (ini_fin_fks.eq.2 .and. j_fks.gt.nincoming) cycle
            if (need_color_links.or.need_charge_links) then
               proc_map(0,0)=proc_map(0,0)+1
               proc_map(proc_map(0,0),0)=proc_map(proc_map(0,0),0)+1
               proc_map(proc_map(0,0),proc_map(proc_map(0,0),0))
     $              =nFKSprocess
               i_fks_pdg_proc(proc_map(0,0))=PDG_type(i_fks)
               j_fks_pdg_proc(proc_map(0,0))=PDG_type(j_fks)
               j_fks_proc(proc_map(0,0))=j_fks
            endif
         enddo
c Check to make sure that there is at most two initial and one final
c state all gluon
         found_ini1=.false.
         found_ini2=.false.
         found_fnl=.false.
         do i=1,proc_map(0,0)
            if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     &       .and. j_fks_proc(i).eq.1 .and. .not.found_ini1) then
               found_ini1=.true.
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.1.and. found_ini1) then
               write (*,*)'Initial state 1 g->gg already'/
     $              /' found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.2.and. .not.found_ini2) then
               found_ini2=.true.
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.2.and. found_ini2) then
               write (*,*)'Initial state 2 g->gg already'/
     $              /' found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_pdg_proc(i).eq.21 .and.
     $              j_fks_proc(i).gt.nincoming .and. .not.found_fnl)
     $              then
               found_fnl=.true.
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_pdg_proc(i).eq.21 .and.
     $              j_fks_proc(i).gt.nincoming .and. found_fnl) then
               write (*,*)
     &              'Final state g->gg already found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            endif
         enddo
c Loop again, and identify the nFKSprocesses that do not have a soft
c singularity and put them together with the corresponding gluon to
c gluons splitting
         do nFKSprocess=1,fks_integrated
            call fks_inc_chooser()
            if (ini_fin_fks.eq.1 .and. j_fks.le.nincoming) cycle
            if (ini_fin_fks.eq.2 .and. j_fks.gt.nincoming) cycle
            if (.not.(need_color_links.or.need_charge_links)) then
               if (j_fks.eq.1 .and. found_ini1) then
                  do i=1,proc_map(0,0)
                     if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $                    .and. j_fks_proc(i).eq.1) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               elseif (j_fks.eq.2 .and. found_ini2) then
                  do i=1,proc_map(0,0)
                     if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $                   .and. j_fks_proc(i).eq.2) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               elseif (j_fks.gt.nincoming .and. found_fnl) then
                  do i=1,proc_map(0,0)
                     if (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $                    j_fks_pdg_proc(i).eq.21.and.
     $                    j_fks_proc(i).gt.nincoming) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               else
                  write (*,*) 'Driver_mintMC: inconsistent process'
                  write (*,*) 'This process has nFKSprocesses'/
     $                 /' without soft singularities, but not a'/
     $                 /' corresponding g->gg splitting that has a'/
     $                 /' soft singularity.',found_ini1,found_ini2
     $                 ,found_fnl
                  do i=1,proc_map(0,0)
                     write (*,*) i,'-->',proc_map(i,0),':',
     &                    (proc_map(i,j),j=1,proc_map(i,0))
                  enddo
                  stop
               endif
            endif
         enddo
      else
         write (*,*) 'sum not known in driver_mintMC.f',sum
         stop
      endif
 20   continue
      if (FKSExplicitSum.and.HAS_PHYSICAL_FKS_CLASSES.and.
     $     NBORN_FLAVOR_CONFIGS.gt.1) then
         write(*,*) 'FKSExplicitSum requires one physical Born class'
         stop 1
      endif
      if (FKSExplicitSum) then
c Collapse the sampled groups into one explicit list. Retain the first
c soft sector as the Born/event owner and respect ini_fin_fks filtering.
c With one group get_MC_integer returns unit volume: Born normalization
c 1/(proc_map(0,0)*vol1) is one and each real sector also has unit weight.
         n_all=0
         do i=1,proc_map(0,0)
            do j=1,proc_map(i,0)
               n_all=n_all+1
               all_fks(n_all)=proc_map(i,j)
            enddo
         enddo
         if (n_all.eq.0) then
            write (*,*) 'No FKS configurations for explicit sum'
            stop 1
         endif
         proc_map=0
         proc_map(0,0)=1
         proc_map(1,0)=n_all
         proc_map(1,1:n_all)=all_fks(1:n_all)
         sum=1
      endif
      write (*,*) 'FKS process map (sum=',sum,') :'
      do i=1,proc_map(0,0)
         write (*,*) i,'-->',proc_map(i,0),':',
     &        (proc_map(i,j),j=1,proc_map(i,0))
      enddo
      return
      end
c


      subroutine setup_event_attributes
c For FxFx matching with pythia8, set the correct attributes
c for the <event> tag in the LHEF file. "npNLO" are the number of Born
c partons in this multiplicity when running the code at NLO accuracy
c ("npLO" is -1 in that case). When running LO only, invert "npLO" and
c "npNLO".
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      integer i
      integer                 nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      character*4      abrv
      common /to_abrv/ abrv
      if ((shower_mc.eq.'PYTHIA8' .or. shower_mc.eq.'HERWIG7') .and.
     $     ickkw.eq.3)then
         nattr=2
         nFKSprocess=1          ! just pick one
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         npNLO=0
         npLO=-1
         do i=nincoming+1,nexternal
c     include all quarks (except top quark) and the gluon.
            if(abs(idup(i,1)).le.5 .or. abs(idup(i,1)).eq.21)
     &           npNLO=npNLO+1
         enddo
         npNLO=npNLO-1
         if (npNLO.gt.99) then
            write (*,*) 'Too many partons',npNLO
            stop
         endif
         if (abrv.eq.'born') then
            npLO=npNLO
            npNLO=-1
         endif
      else
         nattr=0
      endif
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
         if (abrv.eq.'born') then
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



      subroutine special_check_SoftSing(isoft)
      use weight_lines
      use extra_weights, only: QCD_power,orders_tag,amp_pos,wgtcpower,
     $     cpower_pos
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      integer i_soft,isoft,i,i_S
      integer orders(nsplitorders)
      integer get_orders_tag
      logical found_S
      i_soft=0
      i_S=0
      found_S=.false.
      do i=1,icontr
         if (H_event(i)) then
            cycle
         else
            found_S=.true.
            if (i_S.eq.0) i_S=i
         endif
         if (abs(pdg_type_d(nFKS(i),fks_i_d(nFKS(i)))).eq.21) then
            i_soft=i
            exit
         endif
      enddo
      if (found_S .and. i_soft.eq.0) then
c Add an artificial, negligible Born contribution in a sector with a soft
c singularity. It takes the coupling orders of an existing S contribution,
c which passed the coupling-order selection of add_wgt.
         call update_fks_dir(isoft)
         call amp_split_pos_to_orders(amppos(i_S),orders)
         QCD_power=orders(qcd_pos)
         wgtcpower=0d0
         if (cpower_pos.gt.0) wgtcpower=dble(orders(max(1,cpower_pos)))
         orders_tag=get_orders_tag(orders)
         amp_pos=amppos(i_S)
         call add_wgt(2,orders,1d-199,0d0,0d0)
      endif
      return
      end

      subroutine setup_physical_proc_map(proc_map,ini_fin_fks)
c Build event-generation groups for one-row physical FKS classes. Each group
c starts with a soft-capable class and can only receive non-soft classes with
c the same physical Born row, initial/final emitter sector and primary split.
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      integer proc_map(0:fks_configs,0:fks_configs),ini_fin_fks
      integer i,j,k,nFKSprocess,soft_fks,primary_split
      logical found,same_sector,same_split

      proc_map(0,0)=0
      do i=1,FKS_INTEGRATED
         proc_map(i,0)=0
      enddo

c First create one group for every physical class with a soft singularity.
      do nFKSprocess=1,FKS_INTEGRATED
         if (ini_fin_fks.eq.1 .and.
     $       FKS_J_D(nFKSprocess).le.nincoming) cycle
         if (ini_fin_fks.eq.2 .and.
     $       FKS_J_D(nFKSprocess).gt.nincoming) cycle
         if (NEED_COLOR_LINKS_D(nFKSprocess).or.
     $       NEED_CHARGE_LINKS_D(nFKSprocess)) then
            proc_map(0,0)=proc_map(0,0)+1
            proc_map(proc_map(0,0),0)=1
            proc_map(proc_map(0,0),1)=nFKSprocess
         endif
      enddo

c Attach every purely collinear class to its own physical soft class.
      do nFKSprocess=1,FKS_INTEGRATED
         if (ini_fin_fks.eq.1 .and.
     $       FKS_J_D(nFKSprocess).le.nincoming) cycle
         if (ini_fin_fks.eq.2 .and.
     $       FKS_J_D(nFKSprocess).gt.nincoming) cycle
         if (NEED_COLOR_LINKS_D(nFKSprocess).or.
     $       NEED_CHARGE_LINKS_D(nFKSprocess)) cycle
         found=.false.
         do i=1,proc_map(0,0)
            soft_fks=proc_map(i,1)
            same_sector=(FKS_J_D(nFKSprocess).eq.1 .and.
     $                   FKS_J_D(soft_fks).eq.1) .or.
     $                  (FKS_J_D(nFKSprocess).eq.2 .and.
     $                   FKS_J_D(soft_fks).eq.2) .or.
     $                  (FKS_J_D(nFKSprocess).gt.nincoming .and.
     $                   FKS_J_D(soft_fks).gt.nincoming)
            if (.not.same_sector) cycle
            if (BORN_FLAVOR_INDEX_D(nFKSprocess).ne.
     $          BORN_FLAVOR_INDEX_D(soft_fks)) cycle
            if (EXTRA_CNT_D(nFKSprocess).gt.0) then
c A g/a -> q qbar class also advertises the alternate splitting used by
c its extra counterterm.  Its process-map anchor is nevertheless selected
c by the primary underlying-Born splitting.
               primary_split=ISPLITORDER_BORN_D(nFKSprocess)
               same_split=primary_split.gt.0 .and.
     $              SPLIT_TYPE_D(soft_fks,primary_split)
            else
               same_split=.true.
               do j=1,nsplitorders
                  if (SPLIT_TYPE_D(nFKSprocess,j).neqv.
     $                SPLIT_TYPE_D(soft_fks,j)) same_split=.false.
               enddo
            endif
            if (.not.same_split) cycle
            k=proc_map(i,0)+1
            proc_map(i,0)=k
            proc_map(i,k)=nFKSprocess
            found=.true.
            exit
         enddo
         if (.not.found) then
            write (*,*) 'No physical soft FKS class for',nFKSprocess,
     $           BORN_FLAVOR_INDEX_D(nFKSprocess),
     $           FKS_TOPOLOGY_D(nFKSprocess)
            stop 1
         endif
      enddo
      if (proc_map(0,0).eq.0) then
         write (*,*) 'No physical FKS class with a soft singularity'
         stop 1
      endif
      return
      end
