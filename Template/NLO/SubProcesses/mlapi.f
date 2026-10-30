c ML API for MC@NLO.
c
c Evaluates the MC@NLO S and H contributions of MadGraph at phase-space
c points chosen by the caller, so that an external sampler (MadNIS) or
c control variate can take the place of MINT.  A point is
c
c   x(1:ndim)  the unit-cube vector MINT would pass to sigintF.  The last
c              three entries are the FKS radiation variables (xi_i, y_ij,
c              phi_i), the others fix the Born configuration.
c   ichan      the channel, i.e. the Born diagram number as listed in
c              channels.txt.
c   igroup     the group of FKS configurations (1..ngroups), i.e. the
c              row of proc_map that sigintF would otherwise pick at
c              random with get_MC_integer.
c
c The weights are those of a single fold of sigintF with vegas_wgt=1 and
c the group fixed (vol1=1), so that for every P* directory
c
c   sigma = sum_channels sum_groups int d^ndim x w(x, ichan, igroup).
c
c The virtual corrections are computed exactly at every point (no
c average virtual, virtual_fraction=1), with an exact helicity sum.
c
c Entry points, all arguments by reference and arrays in Fortran order.
c They read their input files from the current directory, which must be
c the P* directory.  mlapi_server.f serves them to other programs over
c named pipes (make madevent_mlapi).
c
c   mlapi_init(info)          once, before anything else.
c                             info(1) ndim, (2) nexternal, (3) ngroups,
c                             (4) nconfigs, (5) maxev, (6) fks_configs,
c                             (7) nincoming, (8) the largest group size.
c   mlapi_group(igroup, members)
c                             members(0) is the number of FKS
c                             configurations in group igroup and
c                             members(1:) are their nFKSprocess numbers.
c   mlapi_config(iconf, idiag)
c                             the Born diagram number of channel index
c                             iconf (1..nconfigs).
c   mlapi_eval(n, x, ichan, igroup, totals, nev, ev_int, ev_dbl, ev_mom,
c              parts)
c                             n points, see the routine for the layout.

      subroutine mlapi_init(info)
      use mint_module
      use FKSParams
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'orders.inc'
      integer info(8)
      integer i,j,sum,maxsize
      integer nndim
      common/tosigint/nndim
      character*4 abrv
      common /to_abrv/ abrv
      character*16 abrv_env
      integer len_env,stat_env
      integer soft_phi_n
      common /c_soft_phi/soft_phi_n
      logical nbody
      common/cnbody/nbody
      integer ini_fin_fks
      common/fks_channels/ini_fin_fks
      integer isum_hel
      logical multi_channel
      common/to_matrix/isum_hel, multi_channel
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      logical Hevents
      common/SHevents/Hevents
      integer use_cut
      common /to_weight/use_cut
      integer lbw(0:nexternal)
      common /to_BW/ lbw
      double precision evtsgn
      logical unwgt
      common /c_unwgt/evtsgn,unwgt
      integer n_MC_subt_diverge
      common/counter_subt_diverge/n_MC_subt_diverge
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10
     $     ,n1
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      logical mlapi_mode
      integer mlapi_group
      common /c_mlapi/ mlapi_mode,mlapi_group
      logical born_only_S_scale
      common /c_born_only_S_scale/born_only_S_scale
      integer proc_map(0:fks_configs,0:fks_configs)
      common /c_mlapi_map/ proc_map

c What the main program of driver_mintMC.f sets up before MINT
      fixed_order=.false.
      nlo_ps=.true.
      if (nincoming.ne.2) then
         write (*,*) 'mlapi: decay processes are not supported'
         stop 1
      endif
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      min_virt_fraction_mint=min_virt_fraction
      n_ord_virt=amp_split_size
      n_MC_subt_diverge=0
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
      n1(0:9)=0
      call setrun
      call setpara('param_card.dat')
      call setcuts
      call fill_configurations_common
      call check_amp_split

c What get_user_params reads from input_app.txt, with the values the
c run interface writes for event generation
      alsf=1d0
      besf=-0.1d0
      alazi=-1d0
      beazi=-0.1d0
      Hevents=.true.
      use_cut=2
      multi_channel=.true.
c exact helicity sum in the virtuals
      mc_hel=0
      isum_hel=0
      ini_fin_fks=0
      nchans=1
      ichan=1
      iconfig=1
      iconfigs(1)=iconfig
      wgt_mult=1d0
      imode=0
      doreweight=.false.
      do_rwgt_scale=.false.
      do_rwgt_pdf=.false.
      abrv='all '
c Which contributions to compute can be set from the environment,
c MLAPI_ABRV (default 'all').  'novi' skips the virtual (fks_singular.f,
c bornsoftvirtual), for quantities that do not depend on it, e.g. the
c n+1-body counter terms of the S events.
      call get_environment_variable('MLAPI_ABRV',abrv_env,len_env
     &     ,stat_env)
      if (stat_env.eq.0 .and. len_env.gt.0) then
         abrv=abrv_env(1:4)
         write (*,*) 'mlapi_init: abrv set to ',abrv
      endif
c Average of the S part of the soft counter term over MLAPI_SOFT_PHI
c azimuths of the soft parton (fks_singular.f, compute_soft_counter_term;
c default 1: none)
      call get_environment_variable('MLAPI_SOFT_PHI',abrv_env,len_env
     &     ,stat_env)
      if (stat_env.eq.0 .and. len_env.gt.0) then
         read (abrv_env(1:len_env),*) soft_phi_n
         write (*,*) 'mlapi_init: soft counter term averaged over'
     &        ,soft_phi_n,' azimuths'
      endif
      nbody=.false.
      only_virt=.false.
      lbw(0)=0
      unwgt=.false.

c The integration variables and a single fold
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      nndim=ndim
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

c No virtual tricks: the exact virtual at every point
      use_poly_virtual=.false.
      average_virtual(0:n_ave_virt,1:maxchannels)=0d0
      virtual_fraction(1:maxchannels)=1d0

c The caller picks the group of FKS configurations
      mlapi_mode=.true.
c S-event starting scales from the Born kinematics alone (fks_singular.f,
c set_shower_scale), so that they do not depend on how the radiation
c variables are chosen
      born_only_S_scale=.true.
      mlapi_group=1
      call setup_proc_map(sum,proc_map,ini_fin_fks)
      maxsize=0
      do i=1,proc_map(0,0)
         maxsize=max(maxsize,proc_map(i,0))
      enddo

      info(1)=ndim
      info(2)=nexternal
      info(3)=proc_map(0,0)
      info(4)=mapconfig(0,0)
      info(5)=fks_configs+1
      info(6)=fks_configs
      info(7)=nincoming
      info(8)=maxsize
      return
      end


      subroutine mlapi_group(igroup,members)
      implicit none
      include 'nFKSconfigs.inc'
      integer igroup,members(0:fks_configs),i
      integer proc_map(0:fks_configs,0:fks_configs)
      common /c_mlapi_map/ proc_map
      members(0:fks_configs)=0
      if (igroup.lt.1 .or. igroup.gt.proc_map(0,0)) return
      do i=0,proc_map(igroup,0)
         members(i)=proc_map(igroup,i)
      enddo
      return
      end


      subroutine mlapi_config(iconf,idiag)
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer iconf,idiag
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      idiag=0
      if (iconf.ge.1 .and. iconf.le.mapconfig(0,0))
     $     idiag=mapconfig(iconf,0)
      return
      end


      subroutine mlapi_eval(n,x,ichan_in,igroup,totals,nev,ev_int,ev_dbl
     $     ,ev_mom,parts)
c Evaluates n points.  For point k:
c   x(1:ndim,k)             the unit-cube vector
c   ichan_in(k), igroup(k)  the channel (Born diagram number) and group
c Returns
c   totals(1,k)  sum over events of |weight|, summed over subprocesses
c                with the ABS taken per subprocess (MINT's integral 1)
c   totals(2,k)  the signed sum (MINT's integral 2)
c   totals(3,k)  the signed S-event weight
c   totals(4,k)  the signed sum of the H-event weights
c   nev(k)       the number of events (distinct kinematics) at the point;
c                -1 if the channel or group is not valid
c   for every event e = 1..nev(k):
c   ev_int(1,e,k)  1 for an H event, 0 for an S event
c   ev_int(2,e,k)  its FKS configuration nFKSprocess
c   ev_int(3,e,k)  i_fks of that configuration (for an S event the
c                  momentum of i_fks is zero and must be dropped)
c   ev_dbl(1,e,k)  the signed weight
c   ev_dbl(2,e,k)  the weight with the ABS taken per subprocess
c   ev_dbl(3,e,k)  the shower starting scale
c   ev_mom(0:3,1:nexternal,e,k)  the momenta in the partonic c.m. frame
c   parts(t,k)   the signed weights before any sum, by contribution type
c                t = itype for t=1..15 (see add_wgt in fks_singular.f)
c                and t=16 for the EW Sudakov types (itype>=20)
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer n,maxev,nparts
      parameter (maxev=fks_configs+1,nparts=16)
      double precision x(ndim,n),totals(4,n),ev_dbl(3,maxev,n)
     $     ,ev_mom(0:3,nexternal,maxev,n),parts(nparts,n)
      integer ichan_in(n),igroup(n),nev(n),ev_int(3,maxev,n)
      integer k,i,j,ne,idx,iconf
      double precision xx(ndimmax),f(nintegrals),dummy,sw,aw
      double precision sigintF
      external sigintF
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      logical mlapi_mode
      integer mlapi_group
      common /c_mlapi/ mlapi_mode,mlapi_group
      integer proc_map(0:fks_configs,0:fks_configs)
      common /c_mlapi_map/ proc_map

      do k=1,n
         totals(1:4,k)=0d0
         parts(1:nparts,k)=0d0
         nev(k)=0
c the channel index of this Born diagram
         iconf=0
         do i=1,mapconfig(0,0)
            if (mapconfig(i,0).eq.ichan_in(k)) then
               iconf=i
               exit
            endif
         enddo
         if (iconf.eq.0 .or. igroup(k).lt.1 .or.
     $        igroup(k).gt.proc_map(0,0)) then
            nev(k)=-1
            cycle
         endif
         iconfig=iconf
         iconfigs(1)=iconf
         ichan=1
         mlapi_group=igroup(k)
         xx(1:ndimmax)=0d0
         xx(1:ndim)=x(1:ndim,k)
c one fold, then the finalisation MINT does with ifl=2
         new_point=.true.
         dummy=sigintF(xx,1d0,0,f)
         new_point=.false.
         dummy=sigintF(xx,1d0,2,f)
         totals(1,k)=f(1)
         totals(2,k)=f(2)
         ne=0
         do i=1,icontr
            idx=itype(i)
            if (idx.ge.20) idx=16
            if (idx.ge.1 .and. idx.le.nparts)
     $           parts(idx,k)=parts(idx,k)+wgts(1,i)
            if (icontr_sum(0,i).eq.0) cycle
            ne=ne+1
            if (ne.gt.maxev) then
               write (*,*) 'mlapi: more events than maxev',ne,maxev
               stop 1
            endif
            sw=0d0
            aw=0d0
            do j=1,niproc(i)
               sw=sw+unwgt(j,i)
               aw=aw+abs(unwgt(j,i))
            enddo
            if (H_event(i)) then
               ev_int(1,ne,k)=1
               totals(4,k)=totals(4,k)+sw
            else
               ev_int(1,ne,k)=0
               totals(3,k)=totals(3,k)+sw
            endif
            ev_int(2,ne,k)=nFKS(i)
            ev_int(3,ne,k)=fks_i_d(nFKS(i))
            ev_dbl(1,ne,k)=sw
            ev_dbl(2,ne,k)=aw
            ev_dbl(3,ne,k)=shower_scale(i)
            ev_mom(0:3,1:nexternal,ne,k)=momenta(0:3,1:nexternal,i)
         enddo
         nev(k)=ne
      enddo
      return
      end


      block data mlapi_pythia8
c The Pythia8 controls the main program of driver_mintMC.f initialises
      include 'pythia8_control.inc'
      include 'pythia8_control_setup.inc'
      end
