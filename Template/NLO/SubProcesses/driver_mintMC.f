      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'nexternal.inc'
      include 'genps.inc'
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,k,l,l1,l2,nndim,nevts,p_label

      integer lunlhe
      parameter (lunlhe=98)
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
c
c     Properly initialize PY8 controls
c
      include 'pythia8_control.inc'
      include 'pythia8_control_setup.inc'
c Vegas stuff
      common/tosigint/nndim

      real*8 sigintF
      external sigintF

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      include 'orders.inc'

      double precision weight,event_weight,inv_bias
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
      character*10 dum
      integer iFKS_picked
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

c timing statistics
      include "timing_variables.inc"
      real*4 tOther, tTot
c general MadFKS parameters
      integer ifold_picked
      double precision x_save(ndimmax,max_fold)
      common /c_vegas_x_fold/x_save,ifold_picked
      double precision deravg,derstd,dermax,xi_i_fks_ev_der_max
     &     ,y_ij_fks_ev_der_max
      integer ntot_granny,derntot,ncase(0:6)
      common /c_granny_counters/ ntot_granny,ncase,derntot,deravg,derstd
     &     ,dermax,xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
      integer                     n_MC_subt_diverge
      common/counter_subt_diverge/n_MC_subt_diverge
C-----
C  BEGIN CODE
C-----  
c Write the process PID in the log.txt files (i.e., to the screen)
      write (*,*) getpid()

      call cpu_time(tBefore)
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
      n_MC_subt_diverge=0
      ntot_granny=0
      derntot=0
      do i=0,6
         ncase(i)=0
      enddo
      
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

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      call fill_configurations_common
      call check_amp_split 
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncalls0,itmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
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
c Don''t proceed if muF1#muF2 (we need to work out the relevant formulae
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

c*************************************************************
c     setting of the grids
c*************************************************************
      if (imode.eq.-1.or.imode.eq.0) then
         write (*,*) 'imode is ',imode
         call mint(sigintF)
         call deallocate_weight_lines
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1,1),unc(2,1),0d0,0,0,0,0,0d0,0d0,ans(2,1)
         close(58)
c*************************************************************
c     computation of upper bounding envelope
c*************************************************************
      elseif(imode.eq.1) then
         write (*,*) 'imode is ',imode
         call mint(sigintF)
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
         if (ickkw.eq.-1) putonshell=.false.
         unwgt=.true.
         write(*,*) 'Generating ', nevts, ' events'
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
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
            if (abrv(1:4).eq.'born') then
               vn=3
               call gen(sigintF,1,vn,x)
            else
               if (ran2().lt.ans(5,1)/(ans(1,1)+ans(5,1)) .or. only_virt) then
                  abrv='virt'
                  if (only_virt) then
                     vn=2
                     call gen(sigintF,1,vn,x)
                  else
                     vn=1
                     call gen(sigintF,1,vn,x)
                  endif
               else
                  abrv='novi'
                  vn=2
                  call gen(sigintF,1,vn,x)
               endif
            endif
c Randomly pick the contribution that will be written in the event file
            call pick_unweight_contr(iFKS_picked,ifold_picked)
            call update_fks_dir(iFKS_picked)
            if (event_norm(1:4).eq.'bias') then
               call include_inverse_bias_wgt(inv_bias)
               weight=event_weight*inv_bias
            endif
            call fill_rwgt_lines
            call finalize_event(x_save(1,ifold_picked),weight,lunlhe
     $           ,putonshell,p_label)
         enddo
         call deallocate_weight_lines
         vn=-1
         call gen(sigintF,3,vn,x) ! print counters generation efficiencies
         write (lunlhe,'(a)') "</LesHouchesEvents>"
         close(lunlhe)
      endif

      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
      endif

      if (ntot.ne.0) then
         write(*,*) "Satistics from MadLoop:"
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
      endif

      write (*,*) 'counters for the granny resonances'
      write (*,*) 'ntot     ',ntot_granny
      if (ntot_granny.gt.0) then
         do i=0,6
            write (*,*) '% icase ',i,' : ',ncase(i)/dble(ntot_granny)
         enddo
         write (*,*) 'average,std dev. and max of derivative:',deravg
     &        ,sqrt(abs(derstd-deravg**2)),dermax
         write (*,*)
     &        'and xi_i_fks and y_ij_fks corresponding to max of der.',
     &        xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
      endif
      write (*,*) 'counter for the diverging MC subtraction',n_MC_subt_diverge
      call cpu_time(tAfter)
      tTot = tAfter-tBefore
      tOther = tTot - (tBorn+tGenPS+tReal+tCount+tIS+tFxFx+tf_nb+tf_all
     $     +t_as+tr_s+tr_pdf+t_plot+t_cuts+t_MC_subt+t_isum+t_p_unw
     $     +t_write+t_coupl)
      write(*,*) 'Time spent in Born : ',tBorn
      write(*,*) 'Time spent in PS_Generation : ',tGenPS
      write(*,*) 'Time spent in Reals_evaluation: ',tReal
      write(*,*) 'Time spent in MCsubtraction : ',t_MC_subt
      write(*,*) 'Time spent in Counter_terms : ',tCount
      write(*,*) 'Time spent in Integrated_CT : ',tIS-tOLP
      write(*,*) 'Time spent in Virtuals : ',tOLP      
      write(*,*) 'Time spent in FxFx_cluster : ',tFxFx
      write(*,*) 'Time spent in Nbody_prefactor : ',tf_nb
      write(*,*) 'Time spent in N1body_prefactor : ',tf_all
      write(*,*) 'Time spent in Adding_alphas_pdf : ',t_as
      write(*,*) 'Time spent in Reweight_scale : ',tr_s
      write(*,*) 'Time spent in Reweight_pdf : ',tr_pdf
      write(*,*) 'Time spent in Filling_plots : ',t_plot
      write(*,*) 'Time spent in Applying_cuts : ',t_cuts
      write(*,*) 'Time spent in Sum_ident_contr : ',t_isum
      write(*,*) 'Time spent in Pick_unwgt : ',t_p_unw
      write(*,*) 'Time spent in Write_events : ',t_write
      write(*,*) 'Time spent in AlphaS_dependencies : ',t_coupl
      write(*,*) 'Time spent in Other_tasks : ',tOther
      write(*,*) 'Time spent in Total : ',tTot

      open (unit=12, file='res.dat',status='unknown')
      if (imode.eq.0) then
         write (12,*)ans(1,1),unc(1,1),ans(2,1),unc(2,1),itmax,ncalls0,tTot
      else
         write (12,*)ans(1,1)+ans(5,1),sqrt(unc(1,1)**2+unc(5,1)**2),ans(2,1)
     $        ,unc(2,1),itmax,ncalls0,tTot
      endif
      close(12)

      return
 999  write (*,*) 'nevts file not found'
      stop
      end


      block data timing
c timing statistics
      include "timing_variables.inc"
      data tOLP/0.0/
      data tGenPS/0.0/
      data tBorn/0.0/
      data tIS/0.0/
      data tReal/0.0/
      data tCount/0.0/
      data tFxFx/0.0/
      data tf_nb/0.0/
      data tf_all/0.0/
      data t_as/0.0/
      data tr_s/0.0/
      data tr_pdf/0.0/
      data t_plot/0.0/
      data t_cuts/0.0/
      data t_MC_subt/0.0/
      data t_isum/0.0/
      data t_p_unw/0.0/
      data t_write/0.0/
      data t_coupl/0.0/
      end


      subroutine get_user_params(ncall,nitmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      use mint_module
      implicit none
c
c     Constants
c
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
c
c     Arguments
c
      integer ncall,nitmax
c
c     Local
c
      integer i, j
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
      elseif (i.eq.0) then
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
      if (fks_configs.eq.1) then
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

      if (imode.eq.2) then
         read(*,*) p_label,nevts
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


