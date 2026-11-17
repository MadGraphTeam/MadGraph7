!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! MINT Integrator Package
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Original version by Paolo Nason (for POWHEG (BOX))
! Modified by Rikkert Frederix (for MadGraph7)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!      subroutine mint(fun,ndim,ncalls0,itmax,imode,
! ndim=number of dimensions
! ncalls0=# of calls per iteration
! itmax =# of iterations
! fun(xx,www,ifirst): returns the function to be integrated multiplied by www;
!                     xx(1:ndim) are the variables of integration
!                     ifirst=0: normal behaviour
! imode: integer flag
!
! imode=-1:
! same as imode=0 as far as this routine is concerned, except for the
! fact that a grid is read at the beginning (rather than initialized).
! The return value of imode will be zero.
!
! imode=0:
! When called with imode=0 the routine integrates the absolute value of
! the function and sets up a grid xgrid(0:50,ndim) such that in each
! ndim-1 dimensional slice (i.e. xgrid(m-1,n)<xx(n)<xgrid(m,n)) the
! contribution of the integral is the same the array xgrid is setup at
! this stage; ans and err are the integral and its error
!
! imode=1 (in fact #0)
! When called with imode=1, the routine performs the integral of the
! function fun using the grid xgrid. If some number in the array ifold,
! (say, ifold(n)) is different from 1, it must be a divisor of 50, and
! the 50 intervals xgrid(0:50,n) are grouped into ifold(n) groups, each
! group containing 50/ifold(n) nearby intervals. For example, if
! ifold(1)=5, the 50 intervals for the first dimension are divided in 5
! groups of 10. The integral is then performed by folding on top of each
! other these 5 groups. Suppose, for example, that we choose a random
! point in xx(1) = xgrid(2,1)+x*(xgrid(3,1)-xgrid(2,1)), in the group of
! the first 5 interval.  we sum the contribution of this point to the
! contributions of points
! xgrid(2+m*10,1)+x*(xgrid(3+m*10,1)-xgrid(2+m*10,1)), with m=1,...,4.
! In the sequence of calls to the function fun, the call for the first
! point is performed with ifirst=0, and that for all subsequent points
! with ifirst=1, so that the function can avoid to compute quantities
! that only depend upon dimensions that have ifold=1, and do not change
! in each group of folded call. The values returned by fun in a sequence
! of folded calls with ifirst=0 and ifirst=1 are not used. The function
! itself must accumulate the values, and must return them when called
! with ifirst=2.
! 
! Added the possibility to keep track of more than one integral:
!
! nintegrals=1 : the function that is used to update the grids. This is
! the ABS cross section. If imode.eq.1, this does not contain the
! virtual corrections because for them a separate maximum is kept using (5).
! nintegrals=2 : the actual cross section. This includes virtual corrections.
! nintegrals=3 : the cross section from the M_Virt/M_Born ratio alone:
! this defines the average virtual that is added to each phase-space
! point
! nintegrals=4 : the cross section of the actual virtual minus the
! average virtual. This is used to determine the fraction of phase-space
! points for which we include the virtual.
! nintegrals=5 : abs of 3
! nintegrals=6 : born
! nintegrals>6 : virtual and born order by order
!

module mint_module
  ! Keep IEEE imports local to the procedures using them. Module-level
  ! imports make gfortran save/restore FPU state in unrelated module users.
  use FKSParams ! contains use_poly_virtual
  use virt_surrogate, only: vs_begin_point,vs_vcomp,vs_ratio_set,vs_ratio, &
       vs_reset,vs_solve,vs_write_grids,vs_read_grids
  ! Integration points and side evaluations (point_evaluation).
  use mint_evaluation
  ! FKSSectorLearning; its hooks into the grids and the integrand are below.
  use sector_learning
  implicit none
  integer, parameter, private :: nintervals=32    ! max number of intervals in the integration grids
  integer, parameter, public  :: ndimmax=60       ! max number of dimensions of the integral
  integer, parameter, public  :: n_ave_virt=16    ! max number of grids to set up to approx virtual
  integer, parameter, public  :: nintegrals=38    ! number of integrals to keep track of
  integer, parameter, private :: nintervals_virt=8! max number of intervals in the grids for the approx virtual
  integer, parameter, private :: min_inter=4      ! minimal number of intervals
  integer, parameter, private :: min_it0=4        ! minimal number of iterations in the mint step 0 phase
  integer, parameter, private :: min_it1=5        ! minimal number of iterations in the mint step 1 phase
  integer, parameter, public :: max_fold=512     ! 8*8*8 is max folding for the three variables
  integer, parameter, private :: max_points=100000! maximum number of points to trow per iteration if not enough non-zero points can be found.
  integer, parameter, public  :: maxchannels=20 ! set as least as large as in amcatnlo_run_interface
  integer, parameter, public :: born_spread_nxi=40,born_spread_ny=40
  integer, parameter, private :: born_spread_nbins=born_spread_nxi*born_spread_ny
  integer, parameter, private :: born_spread_train_points=800000
  integer, parameter, private :: born_spread_validation_points=200000
  ! Total exploration share, divided equally over all integration channels.
  double precision, parameter, private :: channel_exploration=0.05d0
  ! Note that the number of intervals in the integration grids, 'nintervals', cannot be arbitrarily large.
  ! It should be equal to
  !     nintervals = min_inter * 2^n,
  ! where 'n' is an integer smaller than or equal to min(min_it0,min_it1).
  !
  ! The number of integrals should be equal to
  !     nintegrals=6+2*n_ave_virt
  !

! public variables 
  integer, public :: ncalls0,ndim,itmax,imode,n_ord_virt,nchans,iconfig,ichan,ifold_energy,ifold_yij,ifold_phi
  integer, dimension(ndimmax), public :: ifold
  integer, dimension(maxchannels), public :: iconfigs
  double precision, public :: accuracy,min_virt_fraction_mint,wgt_mult
  double precision, dimension(0:n_ave_virt,maxchannels), public :: average_virtual
  double precision, dimension(0:n_ave_virt), public :: virt_wgt_mint,born_wgt_mint,polyfit
  double precision, dimension(maxchannels), public :: virtual_fraction
  double precision, dimension(nintegrals,0:maxchannels), public :: ans,unc
  logical, public :: born_spread_active=.false.,born_spread_ready=.false.

  ! The per-point state a side evaluation between two integration points
  ! leaves as the last point set it (save_point_state, restore_point_state).
  type,private :: mint_point_state
     integer :: ifold(ndimmax),ncell(ndimmax),icell(ndimmax),ichan,iconfig,sl_current
     double precision :: f(nintegrals),rand(ndimmax),vol_chan,hwu_values(2)
     double precision :: average_virtual(0:n_ave_virt,maxchannels),polyfit(0:n_ave_virt)
     logical :: even_rn,pass_cuts_check,new_point
     type(evaluation_mode) :: evaluation
  end type mint_point_state
  ! The learning probes' own ranmar state (swap_probe_random_numbers).
  double precision,private :: probe_ranu(97)=0d0,probe_ranc=0d0,probe_rancd=0d0,probe_rancm=0d0
  integer,private :: probe_iranmr=0,probe_jranmr=0
  logical,private :: probe_rng_ready=.false.
  integer, public :: born_spread_phase=0,born_spread_current_bin=1
  integer, public :: born_spread_bin_fold(max_fold)=1
  integer, public :: born_spread_current_sector=1
  integer, public :: born_spread_sector_fold(max_fold)=1
  integer, private :: born_spread_nexternal=0,born_spread_nincoming=0
  integer, private :: born_spread_nfks=0,born_spread_ndim=0
  integer, private :: born_spread_restart_ncalls=0
  integer, private :: born_spread_training_count=0
  integer, private :: born_spread_validation_count=0
  double precision, public :: born_spread_x=0d0,born_spread_y=0d0
  ! Each sampled Born FKS sector carries its own unit normalization.
  double precision, allocatable, public :: born_spread_factor(:,:,:)
  double precision, private :: born_spread_mean_delta=0d0
  double precision, private :: born_spread_m2_delta=0d0
  double precision, private :: born_spread_base_total=0d0
  double precision, private :: born_spread_spread_total=0d0
  double precision, private :: born_spread_point_base=0d0
  double precision, private :: born_spread_point_spread=0d0
  double precision, allocatable, private :: born_spread_deriv0(:,:)
  ! Flatten (bin,sector) here so one sort groups all sector fits.
  integer, dimension(:), allocatable, private :: born_spread_break_bin
  double precision, dimension(:), allocatable, private :: born_spread_break
  double precision, dimension(:), allocatable, private :: born_spread_jump
  integer, private :: born_spread_nbreak=0,born_spread_capacity=0
  logical :: only_virt,new_point,pass_cuts_check

! private variables
  character(len=13), parameter, dimension(nintegrals), private :: title=(/ &
                                                   'ABS integral ', & !  1
                                                   'Integral     ', & !  2
                                                   'Virtual      ', & !  3
                                                   'Virtual ratio', & !  4
                                                   'ABS virtual  ', & !  5
                                                   'Born         ', & !  6
                                                   'V  1         ', & !  7
                                                   'B  1         ', & !  8
                                                   'V  2         ', & !  9
                                                   'B  2         ', & ! 10
                                                   'V  3         ', & ! 11
                                                   'B  3         ', & ! 12
                                                   'V  4         ', & ! 13
                                                   'B  4         ', & ! 14
                                                   'V  5         ', & ! 15
                                                   'B  5         ', & ! 16
                                                   'V  6         ', & ! 17
                                                   'B  6         ', & ! 18
                                                   'V  7         ', & ! 19
                                                   'B  7         ', & ! 20
                                                   'V  8         ', & ! 21
                                                   'B  8         ', & ! 22
                                                   'V  9         ', & ! 23
                                                   'B  9         ', & ! 24
                                                   'V 10         ', & ! 25
                                                   'B 10         ', & ! 26
                                                   'V 11         ', & ! 27
                                                   'B 11         ', & ! 28
                                                   'V 12         ', & ! 29
                                                   'B 12         ', & ! 30
                                                   'V 13         ', & ! 31
                                                   'B 13         ', & ! 32
                                                   'V 14         ', & ! 33
                                                   'B 14         ', & ! 34
                                                   'V 15         ', & ! 35
                                                   'B 15         ', & ! 36
                                                   'V 16         ', & ! 37
                                                   'B 16         '/)  ! 38

  integer, private :: nit,nit_included,kpoint_iter,nint_used,nint_used_virt,min_it,ncalls,pass_cuts_point,ng,npg,k
  integer, dimension(ndimmax), private :: icell,ncell
  integer, dimension(nintegrals), private :: non_zero_point,ntotcalls
  integer, dimension(nintervals,ndimmax,maxchannels), private :: nhits
  integer, dimension(maxchannels), private :: nhits_in_grids
  integer, dimension(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels), private :: nvirt,nvirt_acc
  integer, dimension(13), private :: gen_counters
  logical, private :: double_events,reset,even_rn,firsttime
  ! Reset grids need a per-channel pilot even when a spreading fit is ready.
  logical, private :: initial_channel_scan_pending=.false.
  logical, dimension(maxchannels), private :: regridded
  double precision, dimension(0:nintervals,ndimmax,maxchannels), private :: xgrid,xacc
  double precision, dimension(nintervals,ndimmax,maxchannels), private :: ymax,xmmm
  double precision, dimension(nintegrals,0:maxchannels), private :: vtot,etot,chi2
  ! Keep the legacy harmonic errors used to choose iteration weights apart
  ! from the propagated uncertainty of the weighted estimator itself.
  double precision, dimension(nintegrals), private :: iteration_weight_unc
  double precision, dimension(nintegrals,3), private :: ans3,unc3
  double precision, dimension(nintegrals), private :: ans_l3,unc_l3,chi2_l3,f
  double precision, dimension(0:maxchannels), private :: ymax_virt,ans_chan
  double precision, dimension(maxchannels), private :: channel_moment,channel_squares,channel_probability
  integer(kind=8), dimension(maxchannels), private :: channel_moment_calls,channel_calls
  double precision, dimension(2), private :: HwU_values
  double precision, dimension(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels), private :: ave_virt,ave_virt_acc,ave_born_acc
  double precision, private :: upper_bound,vol_chan
  double precision, dimension(ndimmax), private :: rand
  double precision, dimension(0:nintervals,ndimmax) :: xgrid_new

! Common blocks used elsewhere in the code
  integer                                   npoints
  double precision            cross_section
  common /for_FixedOrder_lhe/ cross_section,npoints
  logical              fixed_order,nlo_ps
  common /c_fnlo_nlops/fixed_order,nlo_ps

! functions and subroutines:
  public :: mint,gen,read_grids_from_file,get_mint_wgt
  public :: born_spread_configure,born_spread_set_point
  public :: born_spread_observe_sample,born_spread_get_factor
  public :: sector_learning_begin_point,sector_learning_select_slot
  public :: sector_learning_add_contribution,sector_learning_reject_point
  public :: sector_learning_tile_displacement
  public :: born_spread_load_table,born_spread_write_table
  private :: initialise_mint,setup_basic_mint &
       &,update_accumulated_results,prepare_next_iteration &
       &,check_desired_accuracy,update_integration_grids &
       &,combine_final_three_iterations &
       &,print_results_accumulated_three_iterations &
       &,update_virtual_fraction,combine_iterations &
       &,print_results_accumulated,check_fractional_uncertainty &
       &,print_results_current_iteration &
       &,compute_fractional_uncertainty,combine_results_channels &
       &,check_for_special_channels_loop &
       &,combine_results_channels_special_loop,get_amount_of_points &
       &,add_point_to_grids,add_point_to_bounding_envelope &
       &,accumulate_the_point,compute_integrand,get_random_x &
       &,start_iteration,reset_accumulated_grids_for_updating &
       &,check_evenly_random_numbers,finalise_mint,write_results &
       &,write_channel_info,setup_imode_1 &
       &,reset_upper_bounding_envelope,setup_imode_m1,setup_imode_0 &
       &,reset_mint_grids,setup_common,write_grids_to_file &
       &,double_grid,regrid,smooth_xacc,nextlexi ,init_ave_virt&
       &,get_ave_virt,fill_ave_virt,regrid_ave_virt ,double_ave_virt&
       &,get_channel,update_channel_sampling,set_channel_probabilities,close_run_zero_res,ran3 &
       &,initialize_even_random_numbers,get_ran &
       &,increase_gen_counters_middle,increase_gen_counters_before &
       &,increase_gen_counters_end,check_upper_bound &
       &,get_random_cell_flat,get_weighted_cell,initialise_mint_gen &
       &,print_gen_counters

contains

  subroutine sl_require_matched111()
    ! Learned maps act on the unfolded radiation coordinates of one point.
    if(born_spread_active.or.any(ifold(1:ndim).ne.1)) &
         error stop 'Sector learning requires folding 1 1 1 and Born spreading off'
  end subroutine sl_require_matched111

  subroutine flat_grid_position(x,kdim,cell,width,u)
    ! Adaptive-grid cell of x in dimension kdim of the current channel, its
    ! width, and the flat (unit-measure) coordinate u of x.
    double precision,intent(in) :: x
    integer,intent(in) :: kdim
    integer,intent(out) :: cell
    double precision,intent(out) :: width,u
    integer :: upper,middle
    ! First cell whose upper edge reaches x (the grid is increasing).
    cell=1;upper=nint_used
    do while(cell.lt.upper)
       middle=(cell+upper)/2
       if(xgrid(middle,kdim,ichan).ge.x)then
          upper=middle
       else
          cell=middle+1
       endif
    enddo
    width=xgrid(cell,kdim,ichan)-xgrid(cell-1,kdim,ichan)
    u=(cell-1+(x-xgrid(cell-1,kdim,ichan))/width)/nint_used
  end subroutine flat_grid_position

  subroutine sector_learning_tile_displacement(x_original,geometry,permutation,delta)
    double precision,intent(in) :: x_original(ndimmax)
    integer,intent(in) :: geometry(3),permutation(sl_nodes)
    double precision,intent(out) :: delta(3)
    integer :: digit(3),target(3),axis,kdim,cell,node,remainder
    double precision :: width,u
    if(any(geometry.lt.1).or.product(geometry).ne.sl_nodes)error stop 'Invalid learned tile geometry'
    digit=0
    delta=0d0
    do axis=1,3
       if(geometry(axis).eq.1)cycle
       kdim=ndim-3+axis
       call flat_grid_position(x_original(kdim),kdim,cell,width,u)
       if(width.le.0d0)error stop 'Invalid adaptive grid for learned tile'
       digit(axis)=min(int(u*geometry(axis)),geometry(axis)-1)
    enddo
    node=1+(digit(1)*geometry(2)+digit(2))*geometry(3)+digit(3)
    remainder=permutation(node)-1
    if(remainder.lt.0.or.remainder.ge.sl_nodes)error stop 'Invalid production learned tile'
    do axis=3,1,-1
       target(axis)=mod(remainder,geometry(axis))
       remainder=remainder/geometry(axis)
    enddo
    delta=dble(target-digit)/dble(geometry)
  end subroutine sector_learning_tile_displacement

  subroutine sector_learning_begin_point()
    ! Ordinary MINT0 points split their ABS weight over the grid (sl_split_*);
    ! learning probes record their Born and sector parts (sl_point_*).
    sl_split_point=sector_learning_active.and.imode.eq.0.and.point_evaluation%integration
    sl_split_valid=.false.
    if(sl_split_point)then
       if(allocated(sl_split_b))sl_split_b=0d0
       if(allocated(sl_split_r))sl_split_r=0d0
       if(allocated(sl_split_h))sl_split_h=0d0
       if(allocated(sl_split_cell))sl_split_cell=0
       sl_split_h0=0d0
       sl_split_abs=0d0
    endif
    if(.not.point_evaluation%learning_probe)return
    if(allocated(sl_point_b))sl_point_b=0d0
    if(allocated(sl_point_r))sl_point_r=0d0
    sl_point_slot=0
    sl_point_flavours=0
    sl_point_rejected=.false.
  end subroutine sector_learning_begin_point

  subroutine sector_learning_reject_point()
    sl_split_valid=.false.
    if(point_evaluation%learning_probe)sl_point_rejected=.true.
  end subroutine sector_learning_reject_point

  subroutine add_abs_weight(acc)
    ! Add the point's ABS weight f(1) to the grid accumulator acc, in the
    ! sampled cells; with learned maps, in the xi/y dimensions the Born
    ! part there and each sector's part where that sector was evaluated.
    double precision,intent(inout) :: acc(0:,:,:)
    integer :: kdim,i,cell
    do kdim=1,ndim
       if(sl_split_valid.and.(kdim.eq.ndim-2.or.kdim.eq.ndim-1))then
          acc(icell(kdim),kdim,ichan)=acc(icell(kdim),kdim,ichan)+sl_split_share0
          do i=1,size(sl_entries(sl_current)%ids)
             if(sl_split_share(i).eq.0d0)cycle
             cell=sl_split_cell(kdim-ndim+3,i)
             if(cell.lt.1)cell=icell(kdim)
             acc(cell,kdim,ichan)=acc(cell,kdim,ichan)+sl_split_share(i)
          enddo
       else
          acc(icell(kdim),kdim,ichan)=acc(icell(kdim),kdim,ichan)+f(1)
       endif
    enddo
  end subroutine add_abs_weight

  subroutine sector_learning_select_slot(slot)
    integer,intent(in) :: slot
    integer :: i
    if(.not.sector_learning_active)return
    sl_current=0
    if(allocated(sl_entries))then
       do i=1,size(sl_entries)
          if(sl_entries(i)%slot.eq.slot)sl_current=i
       enddo
    endif
    if(sl_current.eq.0)error stop 'Unregistered learned sector Born slot'
    if(point_evaluation%learning_probe)then
       if(sl_point_slot.ne.0.and.sl_point_slot.ne.sl_current)error stop 'Learning orbit changes Born slot'
       sl_point_slot=sl_current
    endif
  end subroutine sector_learning_select_slot

  subroutine sector_learning_add_contribution(owner,fold,canonical_flavour,value)
    ! A probe's S contribution at one fold; owner is its outer sector, 0
    ! for the Born part (learning_owner).
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer,intent(in) :: owner,fold,canonical_flavour
    double precision,intent(in) :: value
    double precision,allocatable :: grown_b(:,:),grown_r(:,:,:)
    integer :: nf,ns,previous_f,previous_s,i
    if(.not.point_evaluation%learning_probe)return
    if(sl_current.eq.0.or.owner.lt.0.or.fold.lt.1.or.fold.gt.sl_nodes.or.canonical_flavour.lt.1.or. &
         .not.ieee_is_finite(value))error stop 'Invalid learned sector contribution'
    nf=canonical_flavour
    ns=size(sl_entries(sl_current)%ids)
    previous_f=0
    previous_s=0
    if(allocated(sl_point_b))then
       previous_f=size(sl_point_b,1)
       previous_s=size(sl_point_r,3)
    endif
    if(nf.gt.previous_f.or.ns.gt.previous_s)then
       allocate(grown_b(max(nf,previous_f),sl_nodes),source=0d0)
       allocate(grown_r(max(nf,previous_f),sl_nodes,max(ns,previous_s)),source=0d0)
       if(previous_f.gt.0)then
          grown_b(1:previous_f,:)=sl_point_b
          grown_r(1:previous_f,:,1:previous_s)=sl_point_r
       endif
       call move_alloc(grown_b,sl_point_b)
       call move_alloc(grown_r,sl_point_r)
    endif
    sl_point_flavours=max(sl_point_flavours,nf)
    if(owner.eq.0)then
       sl_point_b(nf,fold)=sl_point_b(nf,fold)+value
    else
       i=sl_sector_index(owner)
       if(i.eq.0)error stop 'Unknown learned sector contribution owner'
       sl_point_r(nf,fold,i)=sl_point_r(nf,fold,i)+value
    endif
  end subroutine sector_learning_add_contribution

  subroutine sl_unified_add_phase(sector,shift,x_original)
    integer,intent(in) :: sector
    double precision,intent(out) :: shift(2)
    double precision,intent(in) :: x_original(ndimmax)
    integer :: index
    double precision :: delta(3)
    if(sl_current.lt.1)error stop 'Missing unified Born slot'
    index=sl_sector_index(sector)
    if(index.eq.0)error stop 'Unknown unified outer sector'
    shift=dble(unified_slots(sl_current)%qcodes(:,index))/4d0
    if(point_evaluation%learning_probe)return
    if(imode.ne.0.and..not.sl_ready)error stop 'Missing frozen unified map in production'
    call sector_learning_tile_displacement(x_original,[4,4,1],sl_entries(sl_current)%permutations(:,index),delta)
    shift=modulo(shift+delta(1:2),1d0)
  end subroutine sl_unified_add_phase

  subroutine sl_epoch_replay_begin_iteration()
    type(epoch_replay_bank_entry),allocatable :: current(:)
    integer :: i,n,nf,nt
    call sl_unified_ensure()
    n=0
    if(allocated(sl_entries))n=size(sl_entries)
    allocate(current(n))
    do i=1,n
       nf=sl_entries(i)%nflavours;nt=sl_entries(i)%ntrain
       if(nt.eq.0.or.nf.eq.0)cycle
       current(i)%nf=nf;current(i)%nt=nt
       current(i)%b=sl_entries(i)%b(1:nf,:,1:nt)
       current(i)%r=sl_entries(i)%r(1:nf,:,:,1:nt)
    enddo
    ! This also replaces positive history by empty after an empty epoch.
    call move_alloc(current,epoch_replay_previous)
    ! Open the new epoch: snapshot the grid, reset the probe banks.
    if(online_epoch.eq.0)then
       online_budget=FKSSectorLearningBudget
       write(*,*)'ONLINE MINT0: mixed-map ABS summaries are TRAINING heuristics; MINT1 recomputes final normalization.'
       write(*,*)'ONLINE policy: (4,4,1) probes; budget:',online_budget,' probe cost:',sl_probe_cost, &
            ' ordinary points; stride16 warmup64.'
       if(sector_learning_archive)open(newunit=online_unit,file='sector_learning_online.bin', &
            access='stream',form='unformatted',status='replace')
    endif
    online_epoch=online_epoch+1
    online_nint_before=nint_used
    ! The grid accounting exists only for the archive audit, which checks
    ! that ordinary points alone feed the grid.
    if(sector_learning_archive)then
       online_grid_before=xgrid(0:nint_used,1:ndim,1:nchans)
       online_base_acc=xacc(0:nint_used,1:ndim,1:nchans)
       online_base_hits=nhits(1:nint_used,1:ndim,1:nchans)
       if(allocated(online_ordinary_acc))deallocate(online_ordinary_acc,online_ordinary_hits)
       allocate(online_ordinary_acc(0:nint_used,ndim,nchans),online_ordinary_hits(nint_used,ndim,nchans))
       online_ordinary_acc=0d0
       online_ordinary_hits=0
    endif
    if(allocated(sl_entries))then
       do i=1,size(sl_entries)
          if(allocated(sl_entries(i)%b))deallocate(sl_entries(i)%b,sl_entries(i)%r)
          sl_entries(i)%ntrain=0
          sl_entries(i)%nflavours=0
          online_slots(i)%before=sl_entries(i)%permutations
          online_slots(i)%oldgx=online_slots(i)%gx
          online_slots(i)%oldgy=online_slots(i)%gy
       enddo
    endif
    if(.not.allocated(unified_slots))return
    do i=1,size(unified_slots)
       unified_slots(i)%fit_status=0;unified_slots(i)%selected_orientation=0
       unified_slots(i)%primary_gx=0;unified_slots(i)%primary_gy=0;unified_slots(i)%primary_selected=0
       unified_slots(i)%alternative_gx=0;unified_slots(i)%alternative_gy=0;unified_slots(i)%alternative_selected=0
       unified_slots(i)%primary=unified_fit_attempt()
       unified_slots(i)%alternative=unified_fit_attempt()
    enddo
  end subroutine sl_epoch_replay_begin_iteration

  subroutine sl_epoch_replay_fit_epoch()
    integer :: i,nf,nt,ns,gx,gy,k
    integer,allocatable :: a(:,:),bmap(:,:)
    double precision :: before,aa,bb,dummy
    double precision,allocatable :: fit_b(:,:,:),fit_r(:,:,:,:)
    call sl_unified_ensure()
    if(sector_learning_archive)then
       if(any(xgrid(0:online_nint_before,1:ndim,1:nchans).ne.online_grid_before)) &
            error stop 'Grid changed inside online iteration'
       online_total_acc=xacc(0:nint_used,1:ndim,1:nchans)
       online_total_hits=nhits(1:nint_used,1:ndim,1:nchans)
       if(any(online_total_hits.ne.online_base_hits+online_ordinary_hits)) &
            error stop 'Unified grid hit accounting differs'
       if(maxval(abs(online_total_acc-online_base_acc-online_ordinary_acc)).gt. &
            1d-10*max(1d0,maxval(abs(online_total_acc))))error stop 'Unified grid accounting differs'
    endif
    if(.not.allocated(sl_entries))return
    do i=1,size(sl_entries)
       nf=sl_entries(i)%nflavours;nt=sl_entries(i)%ntrain;ns=size(sl_entries(i)%ids)
       online_slots(i)%total_samples=online_slots(i)%total_samples+nt
       online_slots(i)%loss_before=0d0;online_slots(i)%loss_after=0d0
       online_slots(i)%alternative_x=0d0;online_slots(i)%alternative_y=0d0
       if(nt.eq.0.or.nf.eq.0)cycle
       unified_slots(i)%fit_status=1
       gx=online_slots(i)%gx;gy=online_slots(i)%gy;k=gx*gy
       if(nt.ge.64.and.online_epoch.ge.3)then
          gx=4;gy=4
       elseif(nt.ge.32.and.online_epoch.ge.2.and.k.lt.8)then
          gx=4;gy=2
       elseif(nt.ge.16.and.k.lt.4)then
          gx=2;gy=2
       endif
       ! Each capacity needs four probes per tile. A slot without a fit
       ! builds its replay bank only for the archive's loss diagnostic.
       if(gx*gy.lt.4.or.nt.lt.4*gx*gy)then
          if(sector_learning_archive)then
             call sl_epoch_replay_build_bank(i,fit_b,fit_r)
             block
               double precision :: losses(size(fit_b,3))
               call sl_tile_losses_unchecked(fit_b,fit_r,sl_entries(i)%permutations,losses)
               online_slots(i)%loss_before=sum(losses);online_slots(i)%loss_after=sum(losses)
             end block
             deallocate(fit_b,fit_r)
          endif
          cycle
       endif
       call sl_epoch_replay_build_bank(i,fit_b,fit_r)
       allocate(a(sl_nodes,ns),bmap(sl_nodes,ns))
       unified_slots(i)%fit_status=2;unified_slots(i)%selected_orientation=1
       unified_slots(i)%primary_gx=gx;unified_slots(i)%primary_gy=gy
       call sector_learning_unified_fit(fit_b,fit_r, &
            sl_entries(i)%permutations,gx,gy,unified_slots(i)%qcodes,a,before,aa, &
            unified_slots(i)%primary_selected,unified_slots(i)%primary)
       online_slots(i)%loss_before=before
       if(gx*gy.eq.8.and.k.lt.8)then
          unified_slots(i)%alternative_gx=2;unified_slots(i)%alternative_gy=4
          call sector_learning_unified_fit(fit_b,fit_r, &
               sl_entries(i)%permutations,2,4,unified_slots(i)%qcodes,bmap,dummy,bb, &
               unified_slots(i)%alternative_selected,unified_slots(i)%alternative)
          online_slots(i)%alternative_x=aa;online_slots(i)%alternative_y=bb
          if(bb.lt.aa)then
             gx=2;gy=4;a=bmap;aa=bb;unified_slots(i)%selected_orientation=2
          endif
       endif
       online_slots(i)%gx=gx;online_slots(i)%gy=gy
       online_slots(i)%loss_after=aa;sl_entries(i)%permutations=a
       deallocate(a,bmap,fit_b,fit_r)
    enddo
  end subroutine sl_epoch_replay_fit_epoch

  subroutine sl_epoch_replay_write_epoch()
    integer :: i,nf,ns,nt,nlegs
    if(.not.allocated(sl_entries))return
    if(.not.sector_learning_archive)goto 10
    write(online_unit)73110620,sl_format_version,sl_method_id,online_epoch,online_nint_before,nint_used, &
         ndim,nchans,size(sl_entries),online_ordinary_calls,online_probes,online_record_rejections
    write(online_unit)online_grid_before,xgrid(0:nint_used,1:ndim,1:nchans)
    write(online_unit)online_base_acc,online_total_acc,online_ordinary_acc
    write(online_unit)online_base_hits,online_total_hits,online_ordinary_hits
    do i=1,size(sl_entries)
       nf=sl_entries(i)%nflavours;ns=size(sl_entries(i)%ids);nt=sl_entries(i)%ntrain
       nlegs=size(sl_entries(i)%pdgs,2)
       ! The final 0 is the (always empty) legacy tile-trace length.
       write(online_unit)sl_entries(i)%slot,nf,ns,nt,nlegs,online_slots(i)%oldgx,online_slots(i)%oldgy, &
            online_slots(i)%gx,online_slots(i)%gy,online_slots(i)%total_samples,0
       write(online_unit)sl_entries(i)%ids,sl_entries(i)%providers,sl_entries(i)%groups,sl_entries(i)%pdgs, &
            online_slots(i)%before,sl_entries(i)%permutations
       write(online_unit)online_slots(i)%loss_before,online_slots(i)%loss_after, &
            online_slots(i)%alternative_x,online_slots(i)%alternative_y
       call sl_unified_write_extension(online_unit,i)
       if(nt.gt.0)write(online_unit)sl_entries(i)%b(1:nf,:,1:nt),sl_entries(i)%r(1:nf,:,:,1:nt)
    enddo
    flush(online_unit)
10  continue
    write(*,*)'ONLINE epoch:',online_epoch,' ordinary/probes/rejects:',online_ordinary_calls, &
         online_probes,online_record_rejections
  end subroutine sl_epoch_replay_write_epoch

  subroutine sl_epoch_replay_finish()
    integer :: i
    call sl_unified_ensure()
    if(.not.allocated(sl_entries))error stop 'Online MINT0 observed no Born layouts'
    if(sector_learning_archive)then
       write(online_unit)-73110620
       close(online_unit);online_unit=0
    endif
    sl_grid_nint=nint_used;sl_grid_ndim=ndim;sl_grid_nchans=nchans
    sl_grid_snapshot=xgrid(0:nint_used,1:ndim,1:nchans)
    sl_ready=.true.
    do i=1,size(sl_entries)
       if(allocated(sl_entries(i)%b))deallocate(sl_entries(i)%b,sl_entries(i)%r)
    enddo
    write(*,*)'ONLINE frozen final map; no separate calibration; normalization deferred to MINT1.'
  end subroutine sl_epoch_replay_finish

  subroutine sl_epoch_replay_read_table(unit)
    use,intrinsic :: ieee_arithmetic,only:ieee_is_finite
    integer,intent(in) :: unit
    character(len=8192) :: line
    character(len=64) :: marker,tag
    integer :: ios,version,mode,count_entries,i,s,slot,ns,nlegs,nf,gx,gy,total,read_slot,read_index,node,c,d,n
    integer :: warm,stride,cost
    double precision :: budget
    integer :: gauge,gauge_version,phase_passes,tile_passes,selection_version,expected_gauge
    integer,allocatable :: expected_q(:,:)
    rewind(unit)
    do
       read(unit,'(a)',iostat=ios)line
       if(ios.ne.0)error stop 'Missing SCL10 header'
       line=adjustl(line)
       if(index(line,'SCL10 H').eq.1)exit
       if(index(line,'SCL').eq.1)error stop 'Unexpected learning metadata before SCL10'
    enddo
    ! The version comes first: another version may have another layout.
    read(line,*,iostat=ios)marker,tag,version
    if(ios.ne.0)error stop 'Malformed SCL10 header'
    ! Expected after a code update, so reported like a setting mismatch.
    if(version.ne.sl_format_version)then
       write(*,*)'ERROR: SCL10 table from another code version; rerun MINT0.'
       stop 1
    endif
    call sl_epoch_replay_validate_line(line,'H',18)
    read(line,*,iostat=ios)marker,tag,version,mode,count_entries,sl_grid_nint,sl_grid_ndim,sl_grid_nchans, &
         online_epoch,online_ordinary_calls,online_probes,online_record_rejections, &
         warm,stride,gauge_version,phase_passes,tile_passes,selection_version
    if(ios.ne.0.or.mode.ne.sl_method_id.or.count_entries.lt.1.or. &
         sl_grid_nint.ne.nintervals.or.sl_grid_ndim.ne.ndim.or.sl_grid_nchans.ne.nchans.or. &
         min(online_epoch,online_ordinary_calls,online_record_rejections).lt.0.or. &
         online_record_rejections.gt.online_probes.or.count_entries.gt.100000.or. &
         warm.ne.64.or.stride.ne.16) &
         error stop 'Malformed SCL10 header'
    if(gauge_version.ne.1.or.phase_passes.ne.3.or.tile_passes.ne.3.or.selection_version.ne.1) &
         error stop 'Unknown unified map convention'
    read(unit,'(a)',iostat=ios)line
    if(ios.ne.0)error stop 'Missing SCL10 budget'
    call sl_epoch_replay_validate_line(line,'B',4)
    read(line,*,iostat=ios)marker,tag,budget,cost
    if(ios.ne.0.or..not.ieee_is_finite(budget).or..not.(budget.gt.0d0.and.budget.le.1d0).or. &
         cost.ne.sl_probe_cost) &
         error stop 'Malformed SCL10 budget'
    ! The budget only steered the MINT0 probes; keep the training value.
    online_budget=budget
    if(allocated(sl_entries))deallocate(sl_entries)
    if(allocated(online_slots))deallocate(online_slots)
    if(allocated(unified_slots))deallocate(unified_slots)
    allocate(sl_entries(count_entries),online_slots(count_entries),unified_slots(count_entries))
    do i=1,count_entries
       read(unit,'(a)',iostat=ios)line
       if(ios.ne.0)error stop 'Missing SCL10 slot'
       call sl_epoch_replay_validate_line(line,'L',10)
       read(line,*,iostat=ios)marker,tag,slot,ns,nlegs,nf,gx,gy,total,gauge
       if(ios.ne.0.or.min(slot,ns,nlegs).lt.1.or. &
            min(nf,total).lt.0.or..not.((gx.eq.1.and.gy.eq.1).or.sl_valid_capacity(gx,gy))) &
            error stop 'Malformed SCL10 slot'
       if(gauge.ne.0.and.gauge.ne.1)error stop 'Malformed unified gauge kind'
       unified_slots(i)%gauge=gauge
       allocate(unified_slots(i)%qcodes(2,ns))
       if(i.gt.1)then
          if(any(sl_entries(1:i-1)%slot.eq.slot))error stop 'Duplicate SCL10 slot'
       endif
       sl_entries(i)%slot=slot;sl_entries(i)%nflavours=nf
       online_slots(i)%gx=gx;online_slots(i)%gy=gy;online_slots(i)%total_samples=total
       allocate(sl_entries(i)%ids(ns),sl_entries(i)%providers(ns),sl_entries(i)%groups(ns), &
            sl_entries(i)%pdgs(ns,nlegs),sl_entries(i)%permutations(sl_nodes,ns))
       do s=1,ns
          read(unit,'(a)',iostat=ios)line
          if(ios.ne.0)error stop 'Missing SCL10 sector'
          call sl_epoch_replay_validate_line(line,'R',9+nlegs)
          read(line,*,iostat=ios)marker,tag,read_slot,read_index,sl_entries(i)%ids(s), &
               sl_entries(i)%providers(s),sl_entries(i)%groups(s),sl_entries(i)%pdgs(s,:),unified_slots(i)%qcodes(:,s)
          if(ios.ne.0.or.read_slot.ne.slot.or.read_index.ne.s.or. &
               min(sl_entries(i)%ids(s),sl_entries(i)%providers(s),sl_entries(i)%groups(s)).lt.1) &
               error stop 'Malformed SCL10 sector'
          read(unit,'(a)',iostat=ios)line
          if(ios.ne.0)error stop 'Missing SCL10 permutation'
          call sl_epoch_replay_validate_line(line,'P',20)
          read(line,*,iostat=ios)marker,tag,read_slot,read_index,sl_entries(i)%permutations(:,s)
          if(ios.ne.0.or.read_slot.ne.slot.or.read_index.ne.s) &
               error stop 'Malformed SCL10 permutation'
          if(.not.sl_is_bijection(sl_entries(i)%permutations(:,s)))error stop 'SCL10 map not bijective'
          if(gx.eq.1.and.gy.eq.1)then
             if(any(sl_entries(i)%permutations(:,s).ne.[(node,node=1,sl_nodes)])) &
                  error stop 'Nonidentity SCL10 map with identity capacity'
          elseif(.not.sl_map_representable(sl_entries(i)%permutations(:,s),gx,gy))then
             error stop 'SCL10 capacity does not represent map'
          endif
       enddo
       if(any([(count(sl_entries(i)%ids.eq.sl_entries(i)%ids(s)).ne.1,s=1,ns)])) &
            error stop 'Duplicate SCL10 sector'
       call sl_unified_check_groups(sl_entries(i)%ids,sl_entries(i)%providers, &
            sl_entries(i)%pdgs,sl_entries(i)%groups)
       allocate(expected_q(2,ns))
       call sector_learning_unified_gauge(sl_entries(i)%ids,expected_gauge,expected_q)
       if(expected_gauge.ne.gauge.or.any(expected_q.ne.unified_slots(i)%qcodes)) &
            error stop 'Saved unified gauge differs from outer IDs'
       deallocate(expected_q)
       online_slots(i)%before=sl_entries(i)%permutations
    enddo
    if(allocated(sl_grid_snapshot))deallocate(sl_grid_snapshot)
    allocate(sl_grid_snapshot(0:sl_grid_nint,sl_grid_ndim,sl_grid_nchans))
    do c=1,sl_grid_nchans
       do d=1,sl_grid_ndim
          read(unit,'(a)',iostat=ios)line
          if(ios.ne.0)error stop 'Missing SCL10 grid'
          call sl_epoch_replay_validate_line(line,'G',sl_grid_nint+6)
          read(line,*,iostat=ios)marker,tag,read_slot,read_index,n,sl_grid_snapshot(:,d,c)
          ! Check n before it indexes the row: Fortran does not short-circuit.
          if(ios.ne.0.or.read_slot.ne.c.or.read_index.ne.d.or. &
               n.ne.sl_grid_nint)error stop 'Malformed SCL10 grid'
          if(.not.all(ieee_is_finite(sl_grid_snapshot(:,d,c))).or. &
               sl_grid_snapshot(0,d,c).ne.0d0.or.sl_grid_snapshot(n,d,c).ne.1d0.or. &
               any(sl_grid_snapshot(1:n,d,c).le.sl_grid_snapshot(0:n-1,d,c)))error stop 'Malformed SCL10 grid'
       enddo
    enddo
    do
       read(unit,'(a)',iostat=ios)line
       if(ios.lt.0)exit
       if(ios.ne.0.or.len_trim(line).ne.0)error stop 'Trailing unified learning metadata'
    enddo
    sl_ready=.true.;rewind(unit)
  end subroutine sl_epoch_replay_read_table

  subroutine sl_online_observe_ordinary()
    ! Called once an ordinary MINT0 point is accumulated.
    integer :: d
    online_ordinary_calls=online_ordinary_calls+1
    if(.not.sector_learning_archive)return
    ! The same per-cell split as the grid (add_point_to_grids).
    call add_abs_weight(online_ordinary_acc)
    do d=1,ndim
       online_ordinary_hits(icell(d),d,ichan)=online_ordinary_hits(icell(d),d,ichan)+1
    enddo
  end subroutine sl_online_observe_ordinary

  subroutine save_point_state(state)
    type(mint_point_state),intent(out) :: state
    state%ifold=ifold;state%ncell=ncell;state%icell=icell;state%ichan=ichan
    state%iconfig=iconfig;state%sl_current=sl_current;state%f=f;state%rand=rand
    state%vol_chan=vol_chan;state%hwu_values=HwU_values
    state%average_virtual=average_virtual;state%polyfit=polyfit;state%even_rn=even_rn
    state%pass_cuts_check=pass_cuts_check;state%new_point=new_point
    state%evaluation=point_evaluation
  end subroutine save_point_state

  subroutine restore_point_state(state)
    type(mint_point_state),intent(in) :: state
    ifold=state%ifold;ncell=state%ncell;icell=state%icell;ichan=state%ichan
    iconfig=state%iconfig;sl_current=state%sl_current;f=state%f;rand=state%rand
    vol_chan=state%vol_chan;HwU_values=state%hwu_values
    average_virtual=state%average_virtual;polyfit=state%polyfit;even_rn=state%even_rn
    pass_cuts_check=state%pass_cuts_check;new_point=state%new_point
    point_evaluation=state%evaluation
  end subroutine restore_point_state

  subroutine swap_probe_random_numbers()
    ! Exchange the ranmar state (ran2) with the learning probes' own stream.
    ! That stream is seeded once, from the ordinary stream at the first
    ! probe (a fixed number of ordinary points into MINT0), so the run's
    ! seed fixes both streams.
    double precision :: ranu(97),ranc,rancd,rancm,su(97),sc,scd,scm
    integer :: iranmr,jranmr,si,sj
    common/raset1/ranu,ranc,rancd,rancm
    common/raset2/iranmr,jranmr
    if(.not.probe_rng_ready)then
       probe_ranu=ranu;probe_ranc=ranc;probe_rancd=rancd;probe_rancm=rancm
       probe_iranmr=iranmr;probe_jranmr=jranmr
       call rmarin(int(ranu(1)*31328d0),int(ranu(2)*30081d0))
       su=ranu;ranu=probe_ranu;probe_ranu=su
       sc=ranc;ranc=probe_ranc;probe_ranc=sc
       scd=rancd;rancd=probe_rancd;probe_rancd=scd
       scm=rancm;rancm=probe_rancm;probe_rancm=scm
       si=iranmr;iranmr=probe_iranmr;probe_iranmr=si
       sj=jranmr;jranmr=probe_jranmr;probe_jranmr=sj
       probe_rng_ready=.true.
    endif
    su=ranu;ranu=probe_ranu;probe_ranu=su
    sc=ranc;ranc=probe_ranc;probe_ranc=sc
    scd=rancd;rancd=probe_rancd;probe_rancd=scd
    scm=rancm;rancm=probe_rancm;probe_rancm=scm
    si=iranmr;iranmr=probe_iranmr;probe_iranmr=si
    sj=jranmr;jranmr=probe_jranmr;probe_jranmr=sj
  end subroutine swap_probe_random_numbers

  subroutine sl_unified_maybe_probe(fun)
    double precision,external :: fun
    type(mint_point_state) :: saved
    integer :: folds(ndimmax)
    double precision :: x(ndimmax),vol
    if(online_ordinary_calls.lt.64.or.mod(online_ordinary_calls,16).ne.0)return
    ! Admission counts points only, so the probes, maps, grids and events
    ! are fixed by the seed: a probe costs sl_probe_cost ordinary points.
    if(sl_probe_cost*online_probes.ge.online_budget*online_ordinary_calls)return
    if(mod(nint_used,4).ne.0.or.mod(nint_used_virt,4).ne.0)return
    ! The probe leaves the per-point state as the last ordinary point set it.
    call save_point_state(saved)
    ifold=1; ifold(ndim-2:ndim)=[4,4,1]; even_rn=.false.
    point_evaluation=learning_probe_evaluation; new_point=.true.
    ! Probes draw from their own stream: the ordinary points' random numbers
    ! do not depend on how many probes the budget admits.
    call swap_probe_random_numbers()
    call get_random_x(x,vol,folds)
    call compute_integrand(fun,x,vol)
    call swap_probe_random_numbers()
    online_probes=online_probes+1
    if(sl_point_rejected)then
       online_record_rejections=online_record_rejections+1
    else
       call sl_append_orbit()
    endif
    call restore_point_state(saved)
  end subroutine sl_unified_maybe_probe

  subroutine sl_verify_grid()
    if(.not.sector_learning_active)return
    if(.not.sl_ready.or..not.allocated(sl_grid_snapshot))error stop 'Missing learned grid provenance'
    ! read_grids_from_file consumes the full fixed-size file grid before
    ! either MINT1 or GEN initializes nint_used in its fresh process.
    ! Verify the dimensions that were actually read, without changing
    ! the legacy startup state or accepting a partial training grid.
    if(sl_grid_nint.ne.nintervals.or.sl_grid_ndim.ne.ndim.or.sl_grid_nchans.ne.nchans) &
         error stop 'Learned adaptive grid dimensions differ from saved setup'
    if(any(sl_grid_snapshot.ne.xgrid(0:nintervals,1:ndim,1:nchans))) &
         error stop 'Learned phase table belongs to a different adaptive grid'
  end subroutine sl_verify_grid

  subroutine mint(fun)
    implicit none
    integer kpoint
    double precision :: vol
    double precision, dimension(ndimmax) :: x
    integer, dimension(ndimmax) :: kfold
    double precision, external :: fun
    logical :: enough_points,channel_loop_done,sl_train
    call initialise_mint
    ! Sector maps are trained only in MINT0; later steps use the frozen table.
    sl_train=sector_learning_active.and.imode.eq.0
    if(sector_learning_active)then
       call sl_require_matched111()
       if(imode.eq.0.and.sl_ready)error stop 'A trained sector table cannot be refitted in a MINT0 restart'
    endif
10  continue
    do while (nit.lt.itmax)
       call start_iteration
       if(sl_train)call sl_epoch_replay_begin_iteration()
2      kpoint_iter=kpoint_iter+1
       do kpoint=1,ncalls
          new_point=.true.
          call get_random_x(x,vol,kfold)
          call compute_integrand(fun,x,vol)
          call accumulate_the_point(x)
          if(sl_train)then
             call sl_online_observe_ordinary()
             call sl_unified_maybe_probe(fun)
          endif
       enddo
       call get_amount_of_points(enough_points)
       if (.not.enough_points) goto 2
       ! Only a grid reset installs the unit channel markers for this scan.
       ! A calibrated restart retains its grids; bad-iteration recovery does not.
       if (initial_channel_scan_pending) then
          call check_for_special_channels_loop(channel_loop_done)
          if (.not.channel_loop_done) goto 2
          call combine_results_channels_special_loop
       else
          call combine_results_channels
       endif
       if(sl_train)call sl_epoch_replay_fit_epoch()
       call update_accumulated_results
       if(sl_train)call sl_epoch_replay_write_epoch()
    enddo
    if (imode.eq.0 .and. born_spread_active .and. &
         .not.born_spread_ready) then
       call calibrate_born_spreading(fun)
       call born_spread_write_table
       born_spread_ready=.true.
       born_spread_phase=3
       if (double_events) born_spread_restart_ncalls= &
            80*ndim*(nchans/3+1)
       ncalls0=born_spread_restart_ncalls
       call setup_common
       call reset_MC_grid
       call reset_accumulated_grids_for_updating
       if (even_rn) call initialize_even_random_numbers
       write(*,*) 'Restarting MINT with the normalized born-spreading table'
       goto 10
    endif
    if(sl_train)call sl_epoch_replay_finish()
    call finalise_mint
  end subroutine mint

  subroutine calibrate_born_spreading(fun)
    implicit none
    double precision, external :: fun
    double precision :: x(ndimmax),vol
    integer :: kfold(ndimmax),ipoint
    logical :: old_even_rn
    born_spread_phase=1
    point_evaluation=born_spread_evaluation
    born_spread_training_count=0
    born_spread_validation_count=0
    born_spread_mean_delta=0d0
    born_spread_m2_delta=0d0
    born_spread_base_total=0d0
    born_spread_spread_total=0d0
    born_spread_point_base=0d0
    born_spread_point_spread=0d0
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    old_even_rn=even_rn
    even_rn=.false.
    write(*,*) 'Training born-spreading table with ', &
         born_spread_train_points,' independent phase-space points'
    do ipoint=1,born_spread_train_points
       new_point=.true.
       call get_random_x(x,vol,kfold)
       call compute_integrand(fun,x,vol)
    enddo
    call solve_born_spreading_table
    born_spread_phase=2
    write(*,*) 'Validating born-spreading table with ', &
         born_spread_validation_points,' independent phase-space points'
    do ipoint=1,born_spread_validation_points
       new_point=.true.
       call get_random_x(x,vol,kfold)
       call compute_integrand(fun,x,vol)
    enddo
    point_evaluation=integration_evaluation
    even_rn=old_even_rn
    call finish_born_spreading_validation
    call deallocate_born_spread_lines
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
    born_spread_nbreak=0
    born_spread_capacity=0
  end subroutine calibrate_born_spreading

  subroutine solve_born_spreading_table
    implicit none
    integer :: isector,first,last
    if (born_spread_nbreak.gt.1) &
         call born_spread_sort_breaks(1,born_spread_nbreak)
    first=1
    do isector=1,born_spread_nfks
       last=first-1
       do while (last.lt.born_spread_nbreak)
          if (born_spread_break_bin(last+1).gt.isector*born_spread_nbins) exit
          last=last+1
       enddo
       call solve_born_spreading_sector(isector,first,last)
       first=last+1
    enddo
  end subroutine solve_born_spreading_table

  subroutine solve_born_spreading_sector(isector,first_break,last_break)
    implicit none
    integer, intent(in) :: isector,first_break,last_break
    integer, allocatable :: seg_bin(:)
    double precision, allocatable :: seg_slope(:),seg_length(:)
    double precision, dimension(born_spread_nbins) :: caps,bin_area
    double precision :: deriv,prev,turn,jump_total,length,remaining
    double precision :: group_slope,cap_total,low,high,mid,trial,total_add
    double precision :: max_factor,weighted_norm,slope_scale
    integer :: ibin,k,j,nseg,iseg,first,last,pass,global_bin,capacity
    capacity=last_break-first_break+1+born_spread_nbins
    allocate(seg_bin(capacity),seg_slope(capacity),seg_length(capacity))
    nseg=0
    k=first_break
    do ibin=1,born_spread_nbins
       global_bin=(isector-1)*born_spread_nbins+ibin
       bin_area(ibin)=born_spread_bin_area()
       max_factor=1d0/bin_area(ibin)
       deriv=born_spread_deriv0(ibin,isector)
       slope_scale=abs(deriv)
       j=k
       do while (j.le.last_break)
          if (born_spread_break_bin(j).ne.global_bin) exit
          slope_scale=slope_scale+born_spread_jump(j)
          j=j+1
       enddo
       prev=0d0
       do while (k.le.last_break)
          if (born_spread_break_bin(k).ne.global_bin) exit
          turn=born_spread_break(k)
          jump_total=0d0
          j=k
          do while (j.le.last_break)
             if (born_spread_break_bin(j).ne.global_bin) exit
             if (.not.born_spread_same_break(turn, &
                  born_spread_break(j))) exit
             jump_total=jump_total+born_spread_jump(j)
             j=j+1
          enddo
          if (turn.gt.prev.and.prev.lt.max_factor) then
             length=min(turn-prev,max_factor-prev)
             if (length.gt.0d0) then
                nseg=nseg+1
                seg_bin(nseg)=ibin
                seg_slope(nseg)=deriv/bin_area(ibin)
                if (abs(deriv).le.1d-13*slope_scale) seg_slope(nseg)=0d0
                seg_length(nseg)=length
             endif
          endif
          deriv=deriv+jump_total
          prev=max(prev,turn)
          k=j
       enddo
       if (prev.lt.max_factor) then
          nseg=nseg+1
          seg_bin(nseg)=ibin
          seg_slope(nseg)=deriv/bin_area(ibin)
          ! Cancelling the initial slope against its jumps can leave a tiny
          ! residual. A truly flat tail must participate in the closest-to-
          ! one tie break, rather than attract an entire sector's mass.
          if (abs(deriv).le.1d-13*slope_scale) seg_slope(nseg)=0d0
          seg_length(nseg)=max_factor-prev
       endif
    enddo
    if (nseg.le.0) then
       born_spread_factor(:,:,isector)=1d0
       deallocate(seg_bin,seg_slope,seg_length)
       return
    endif
    if (nseg.gt.1) call born_spread_sort_segments( &
         seg_slope,seg_bin,seg_length,1,nseg)
    born_spread_factor(:,:,isector)=0d0
    remaining=1d0
    iseg=1
    do while (iseg.le.nseg.and.remaining.gt.1d-12)
       first=iseg
       group_slope=seg_slope(iseg)
       do while (iseg.le.nseg)
          if (.not.born_spread_same_slope(group_slope, &
               seg_slope(iseg))) exit
          iseg=iseg+1
       enddo
       last=iseg-1
       caps=0d0
       do j=first,last
          caps(seg_bin(j))=caps(seg_bin(j))+seg_length(j)
       enddo
       cap_total=sum(caps*bin_area)
       if (remaining.ge.cap_total-1d-12) then
          do ibin=1,born_spread_nbins
             if (caps(ibin).gt.0d0) then
                j=(ibin-1)/born_spread_nxi+1
                k=ibin-(j-1)*born_spread_nxi
                born_spread_factor(k,j,isector)=born_spread_factor(k,j,isector)+caps(ibin)
             endif
          enddo
          remaining=max(0d0,remaining-cap_total)
       else
          low=-maxval(1d0/bin_area)
          high=maxval(1d0/bin_area)
          do pass=1,100
             mid=(low+high)/2d0
             total_add=0d0
             do ibin=1,born_spread_nbins
                if (caps(ibin).le.0d0) cycle
                j=(ibin-1)/born_spread_nxi+1
                k=ibin-(j-1)*born_spread_nxi
                trial=max(born_spread_factor(k,j,isector), &
                     min(born_spread_factor(k,j,isector)+caps(ibin),1d0+mid))
                total_add=total_add+bin_area(ibin)* &
                     (trial-born_spread_factor(k,j,isector))
             enddo
             if (total_add.lt.remaining) then
                low=mid
             else
                high=mid
             endif
          enddo
          mid=(low+high)/2d0
          do ibin=1,born_spread_nbins
             if (caps(ibin).le.0d0) cycle
             j=(ibin-1)/born_spread_nxi+1
             k=ibin-(j-1)*born_spread_nxi
             trial=max(born_spread_factor(k,j,isector), &
                  min(born_spread_factor(k,j,isector)+caps(ibin),1d0+mid))
             born_spread_factor(k,j,isector)=trial
          enddo
          remaining=0d0
       endif
    enddo
    if (remaining.gt.1d-7) then
       write(*,*) 'ERROR: born-spreading optimizer could not satisfy', &
            ' its normalization constraint',remaining
       stop 1
    endif
    weighted_norm=0d0
    do ibin=1,born_spread_nbins
       j=(ibin-1)/born_spread_nxi+1
       k=ibin-(j-1)*born_spread_nxi
       weighted_norm=weighted_norm+bin_area(ibin)* &
            born_spread_factor(k,j,isector)
    enddo
    if (weighted_norm.le.0d0.or.abs(weighted_norm-1d0).gt.1d-7) then
       write(*,*) 'ERROR: born-spreading optimizer produced invalid ', &
            'normalization',weighted_norm
       stop 1
    endif
    born_spread_factor(:,:,isector)=born_spread_factor(:,:,isector)/weighted_norm
    deallocate(seg_bin,seg_slope,seg_length)
  end subroutine solve_born_spreading_sector

  pure double precision function born_spread_bin_area()
    implicit none
    ! The Born prefactor contains the radiation Jacobians. The table
    ! coordinates are uniform in that measure, not in the raw MINT x's.
    born_spread_bin_area=1d0/born_spread_nbins
  end function born_spread_bin_area

  subroutine born_spread_solver_self_test
    implicit none
    integer :: ibin,ix,iy
    double precision :: expected,bval(1),cval(1),area
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_phase=1
    do ibin=1,born_spread_nbins
       born_spread_current_bin=ibin
       area=born_spread_bin_area()
       ix=mod(ibin-1,born_spread_nxi)+1
       bval(1)=area
       cval(1)=-2d0*area
       if (ix.gt.born_spread_nxi/2) then
          bval(1)=-area
          cval(1)=-area
       endif
       call born_spread_observe_sample(bval,cval,1)
    enddo
    born_spread_phase=0
    call solve_born_spreading_table
    do ibin=1,born_spread_nbins
       iy=(ibin-1)/born_spread_nxi+1
       ix=ibin-(iy-1)*born_spread_nxi
       expected=0d0
       if (ix.le.born_spread_nxi/2) expected=2d0
       if (abs(born_spread_factor(ix,iy,1)-expected).gt.1d-10) then
          write(*,*) 'ERROR: born-spreading optimizer self-test failed'
          stop 1
       endif
    enddo
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_training_count=0
    call solve_born_spreading_table
    if (maxval(abs(born_spread_factor-1d0)).gt.1d-10) then
       write(*,*) 'ERROR: born-spreading tie-break self-test failed'
       stop 1
    endif
    born_spread_factor=1d0
    born_spread_deriv0=0d0
    born_spread_current_bin=1
    born_spread_nbreak=0
    born_spread_capacity=0
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
  end subroutine born_spread_solver_self_test

  subroutine finish_born_spreading_validation
    implicit none
    double precision :: validation_error,improvement,required
    validation_error=0d0
    if (born_spread_validation_count.gt.1) &
         validation_error=sqrt(max(0d0,born_spread_m2_delta/ &
         (born_spread_validation_count-1)/born_spread_validation_count))
    improvement=born_spread_base_total-born_spread_spread_total
    required=2d0*validation_error*born_spread_validation_count
    if (born_spread_validation_count.lt.2.or. &
         born_spread_base_total.le.0d0.or.improvement.le.required) then
       born_spread_factor=1d0
       write(*,*) 'Born spreading validation found no significant reduction; ', &
            'using the normalized unit table'
    else
       write(*,*) 'Born spreading validation reduced sampled negative mass by ', &
            improvement,' +/- ',validation_error*born_spread_validation_count
    endif
  end subroutine finish_born_spreading_validation

  recursive subroutine born_spread_sort_breaks(left,right)
    implicit none
    integer, intent(in) :: left,right
    integer :: i,j,pivot_bin,temp_bin
    double precision :: pivot_turn,temp_turn,temp_jump
    i=left
    j=right
    pivot_bin=born_spread_break_bin((left+right)/2)
    pivot_turn=born_spread_break((left+right)/2)
    do
       do while (born_spread_break_bin(i).lt.pivot_bin.or. &
            (born_spread_break_bin(i).eq.pivot_bin.and. &
            born_spread_break(i).lt.pivot_turn))
          i=i+1
       enddo
       do while (born_spread_break_bin(j).gt.pivot_bin.or. &
            (born_spread_break_bin(j).eq.pivot_bin.and. &
            born_spread_break(j).gt.pivot_turn))
          j=j-1
       enddo
       if (i.le.j) then
          temp_bin=born_spread_break_bin(i)
          born_spread_break_bin(i)=born_spread_break_bin(j)
          born_spread_break_bin(j)=temp_bin
          temp_turn=born_spread_break(i)
          born_spread_break(i)=born_spread_break(j)
          born_spread_break(j)=temp_turn
          temp_jump=born_spread_jump(i)
          born_spread_jump(i)=born_spread_jump(j)
          born_spread_jump(j)=temp_jump
          i=i+1
          j=j-1
       endif
       if (i.gt.j) exit
    enddo
    if (left.lt.j) call born_spread_sort_breaks(left,j)
    if (i.lt.right) call born_spread_sort_breaks(i,right)
  end subroutine born_spread_sort_breaks

  pure recursive subroutine born_spread_sort_segments(slopes,bins,lengths,left,right)
    implicit none
    integer, intent(in) :: left,right
    integer, intent(inout) :: bins(:)
    double precision, intent(inout) :: slopes(:),lengths(:)
    integer :: i,j,temp_bin
    double precision :: pivot,temp_slope,temp_length
    i=left
    j=right
    pivot=slopes((left+right)/2)
    do
       do while (slopes(i).lt.pivot)
          i=i+1
       enddo
       do while (slopes(j).gt.pivot)
          j=j-1
       enddo
       if (i.le.j) then
          temp_slope=slopes(i)
          slopes(i)=slopes(j)
          slopes(j)=temp_slope
          temp_length=lengths(i)
          lengths(i)=lengths(j)
          lengths(j)=temp_length
          temp_bin=bins(i)
          bins(i)=bins(j)
          bins(j)=temp_bin
          i=i+1
          j=j-1
       endif
       if (i.gt.j) exit
    enddo
    if (left.lt.j) call born_spread_sort_segments( &
         slopes,bins,lengths,left,j)
    if (i.lt.right) call born_spread_sort_segments( &
         slopes,bins,lengths,i,right)
  end subroutine born_spread_sort_segments

  pure logical function born_spread_same_break(a,b)
    implicit none
    double precision, intent(in) :: a,b
    born_spread_same_break=(a.eq.b)
  end function born_spread_same_break

  pure logical function born_spread_same_slope(a,b)
    implicit none
    double precision, intent(in) :: a,b
    double precision :: scale
    scale=max(abs(a),abs(b),1d-300)
    born_spread_same_slope=abs(a-b).le.1d-13*scale
  end function born_spread_same_slope

  subroutine born_spread_configure(enabled,nexternal,nincoming,nfks,ndim_in)
    implicit none
    logical, intent(in) :: enabled
    integer, intent(in) :: nexternal,nincoming,nfks,ndim_in
    born_spread_active=enabled
    born_spread_ready=.false.
    born_spread_phase=0
    if (allocated(born_spread_factor)) deallocate(born_spread_factor)
    if (allocated(born_spread_deriv0)) deallocate(born_spread_deriv0)
    if (nfks.lt.1) call born_spread_table_error('Invalid Born-spreading sector count')
    allocate(born_spread_factor(born_spread_nxi,born_spread_ny,nfks))
    allocate(born_spread_deriv0(born_spread_nbins,nfks))
    born_spread_factor=1d0
    born_spread_x=0d0
    born_spread_y=0d0
    born_spread_current_bin=1
    born_spread_current_sector=1
    born_spread_nexternal=nexternal
    born_spread_bin_fold=1
    born_spread_sector_fold=1
    born_spread_nincoming=nincoming
    born_spread_nfks=nfks
    born_spread_ndim=ndim_in
    born_spread_restart_ncalls=0
    born_spread_training_count=0
    born_spread_validation_count=0
    born_spread_mean_delta=0d0
    born_spread_m2_delta=0d0
    born_spread_base_total=0d0
    born_spread_spread_total=0d0
    born_spread_point_base=0d0
    born_spread_point_spread=0d0
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_capacity=0
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
    if (enabled) call born_spread_solver_self_test
  end subroutine born_spread_configure

  subroutine born_spread_set_point(x,y)
    implicit none
    double precision, intent(in) :: x,y
    integer :: ix,iy
    born_spread_x=max(0d0,min(x,1d0))
    born_spread_y=max(0d0,min(y,1d0))
    ix=min(int(born_spread_x*born_spread_nxi)+1,born_spread_nxi)
    iy=min(int(born_spread_y*born_spread_ny)+1,born_spread_ny)
    born_spread_current_bin=(iy-1)*born_spread_nxi+ix
  end subroutine born_spread_set_point

  double precision function born_spread_get_factor()
    implicit none
    integer :: ix,iy
    iy=(born_spread_current_bin-1)/born_spread_nxi+1
    ix=born_spread_current_bin-(iy-1)*born_spread_nxi
    born_spread_get_factor=born_spread_factor(ix,iy,born_spread_current_sector)
  end function born_spread_get_factor

  subroutine born_spread_observe_sample(bvals,cvals,nvals,point_complete)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer, intent(in) :: nvals
    double precision, intent(in) :: bvals(*),cvals(*)
    logical, intent(in), optional :: point_complete
    integer :: j,ibin,isector
    double precision :: b,c,turn,base_neg,spread_neg,delta,old_mean
    if (born_spread_phase.eq.1) then
       if (.not.present(point_complete)) then
          born_spread_training_count=born_spread_training_count+1
       elseif (point_complete) then
          born_spread_training_count=born_spread_training_count+1
       endif
       ibin=born_spread_current_bin
       isector=born_spread_current_sector
       do j=1,nvals
          b=bvals(j)
          c=cvals(j)
          if (.not.ieee_is_finite(b) .or. .not.ieee_is_finite(c)) cycle
          if (b.eq.0d0) cycle
          if (c.lt.0d0.or.(c.eq.0d0.and.b.lt.0d0)) &
               born_spread_deriv0(ibin,isector)=born_spread_deriv0(ibin,isector)-b
          turn=-c/b
          if (.not.ieee_is_finite(turn)) cycle
          if (turn.eq.0d0.and.c.lt.0d0.and.b.gt.0d0) &
               born_spread_deriv0(ibin,isector)=born_spread_deriv0(ibin,isector)+b
          if (turn.gt.0d0.and.turn.lt.1d0/born_spread_bin_area())call &
               born_spread_append((isector-1)*born_spread_nbins+ibin,turn,abs(b))
       enddo
    elseif (born_spread_phase.eq.2) then
       base_neg=0d0
       spread_neg=0d0
       do j=1,nvals
          b=bvals(j)
          c=cvals(j)
          if (.not.ieee_is_finite(b) .or. .not.ieee_is_finite(c)) cycle
          base_neg=base_neg+max(0d0,-(b+c))
          spread_neg=spread_neg+max(0d0,-(b*born_spread_get_factor()+c))
       enddo
       born_spread_point_base=born_spread_point_base+base_neg
       born_spread_point_spread=born_spread_point_spread+spread_neg
       if (present(point_complete)) then
          if (point_complete) then
             born_spread_validation_count=born_spread_validation_count+1
             delta=born_spread_point_spread-born_spread_point_base
             old_mean=born_spread_mean_delta
             born_spread_mean_delta=old_mean+ &
                  (delta-old_mean)/born_spread_validation_count
             born_spread_m2_delta=born_spread_m2_delta+ &
                  (delta-old_mean)*(delta-born_spread_mean_delta)
             born_spread_base_total=born_spread_base_total+ &
                  born_spread_point_base
             born_spread_spread_total=born_spread_spread_total+ &
                  born_spread_point_spread
             born_spread_point_base=0d0
             born_spread_point_spread=0d0
          endif
       else
! Preserve the standalone observer contract for single-fold callers.
          born_spread_validation_count=born_spread_validation_count+1
          delta=born_spread_point_spread-born_spread_point_base
          old_mean=born_spread_mean_delta
          born_spread_mean_delta=old_mean+ &
               (delta-old_mean)/born_spread_validation_count
          born_spread_m2_delta=born_spread_m2_delta+ &
               (delta-old_mean)*(delta-born_spread_mean_delta)
          born_spread_base_total=born_spread_base_total+ &
               born_spread_point_base
          born_spread_spread_total=born_spread_spread_total+ &
               born_spread_point_spread
          born_spread_point_base=0d0
          born_spread_point_spread=0d0
       endif
    endif
  end subroutine born_spread_observe_sample

  subroutine born_spread_append(ibin,turn,jump)
    implicit none
    integer, intent(in) :: ibin
    double precision, intent(in) :: turn,jump
    integer, allocatable :: itemp(:)
    double precision, allocatable :: dtemp(:)
    integer :: new_capacity
    if (born_spread_nbreak.eq.born_spread_capacity) then
       new_capacity=max(1024,2*born_spread_capacity)
       allocate(itemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            itemp(1:born_spread_nbreak)=born_spread_break_bin
       call move_alloc(itemp,born_spread_break_bin)
       allocate(dtemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            dtemp(1:born_spread_nbreak)=born_spread_break
       call move_alloc(dtemp,born_spread_break)
       allocate(dtemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            dtemp(1:born_spread_nbreak)=born_spread_jump
       call move_alloc(dtemp,born_spread_jump)
       born_spread_capacity=new_capacity
    endif
    born_spread_nbreak=born_spread_nbreak+1
    born_spread_break_bin(born_spread_nbreak)=ibin
    born_spread_break(born_spread_nbreak)=turn
    born_spread_jump(born_spread_nbreak)=jump
  end subroutine born_spread_append

  subroutine born_spread_load_table
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: ios,version,nxi,ny,nexternal,nincoming,nfks,ndim_in
    integer :: i,j,isector
    character(len=32) :: tag
    integer, parameter :: lun=71
    open(unit=lun,file='born_spreading.dat',status='old',iostat=ios)
    if (ios.ne.0) call born_spread_table_error( &
         'Cannot open born_spreading.dat; rerun integration step 0 or set born_spreading = False')
    read(lun,*,iostat=ios) tag,version
    if (ios.ne.0) &
         call born_spread_table_error('Unsupported born_spreading.dat format')
    if (trim(tag).ne.'BORN_SPREAD'.or.version.ne.3) &
         call born_spread_table_error('Unsupported born_spreading.dat format; rerun integration step 0')
    read(lun,*,iostat=ios) nxi,ny,nexternal,nincoming,nfks,ndim_in
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat signature')
    if (nxi.ne.born_spread_nxi.or.ny.ne.born_spread_ny.or. &
         nexternal.ne.born_spread_nexternal.or. &
         nincoming.ne.born_spread_nincoming.or. &
         nfks.ne.born_spread_nfks.or.ndim_in.ne.born_spread_ndim) &
         call born_spread_table_error( &
         'born_spreading.dat belongs to an incompatible process; rerun step 0')
    read(lun,*,iostat=ios) born_spread_training_count, &
         born_spread_validation_count
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat statistics')
    read(lun,*,iostat=ios) born_spread_base_total, &
         born_spread_spread_total,born_spread_mean_delta,born_spread_m2_delta
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat statistics')
    do isector=1,born_spread_nfks
       do j=1,born_spread_ny
          read(lun,*,iostat=ios) (born_spread_factor(i,j,isector),i=1,born_spread_nxi)
          if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat table')
       enddo
    enddo
    close(lun)
    if (any(.not.ieee_is_finite(born_spread_factor)).or. &
         any(born_spread_factor.lt.0d0)) &
         call born_spread_table_error('born_spreading.dat contains invalid factors')
    do isector=1,born_spread_nfks
       if (abs(born_spread_normalization(isector)-1d0).gt.1d-7) &
            call born_spread_table_error('born_spreading.dat violates a sector normalization constraint')
    enddo
    born_spread_ready=.true.
    born_spread_phase=3
    write(*,*) 'Loaded normalized born-spreading table from born_spreading.dat'
  end subroutine born_spread_load_table

  pure double precision function born_spread_normalization(isector)
    implicit none
    integer, intent(in) :: isector
    born_spread_normalization=sum(born_spread_bin_area()*born_spread_factor(:,:,isector))
  end function born_spread_normalization

  subroutine born_spread_write_table
    implicit none
    integer :: ios,i,j,isector
    integer, parameter :: lun=72
    open(unit=lun,file='born_spreading.dat',status='replace',iostat=ios)
    if (ios.ne.0) then
       write(*,*) 'ERROR: cannot write born_spreading.dat'
       stop 1
    endif
    write(lun,*) 'BORN_SPREAD',3
    write(lun,*) born_spread_nxi,born_spread_ny,born_spread_nexternal, &
         born_spread_nincoming,born_spread_nfks,born_spread_ndim
    write(lun,*) born_spread_training_count,born_spread_validation_count
    write(lun,'(4(1x,es24.16))') born_spread_base_total, &
         born_spread_spread_total,born_spread_mean_delta,born_spread_m2_delta
    do isector=1,born_spread_nfks
       do j=1,born_spread_ny
          write(lun,'(40(1x,es24.16))') &
               (born_spread_factor(i,j,isector),i=1,born_spread_nxi)
       enddo
    enddo
    close(lun)
  end subroutine born_spread_write_table

  subroutine born_spread_table_error(message)
    implicit none
    character(len=*), intent(in) :: message
    write(*,*) 'ERROR: ',trim(message)
    stop 1
  end subroutine born_spread_table_error

  subroutine initialise_mint
    implicit none
    initial_channel_scan_pending=.false.
    channel_moment=0d0
    channel_moment_calls=0
    if (imode.ne.0) call read_grids_from_file
    call setup_basic_mint
    if (imode.eq.0) then
       call setup_imode_0
    elseif (imode.eq.-1) then
       call setup_imode_m1
    elseif (imode.eq.1) then
       call setup_imode_1
    endif
    call set_channel_probabilities
    cross_section=ans_chan(0) * wgt_mult
    call setup_common
  end subroutine initialise_mint

  subroutine setup_basic_mint
    implicit none
    ! if ncalls0 is greater than 0, use the default running, i.e. do not
    ! double the events after each iteration as well as use a fixed number
    ! of intervals in the grids.
    if (ncalls0.gt.0) then
       double_events=.false.
       nint_used=nintervals
       nint_used_virt=nintervals_virt
    else
       ! if ncalls0.le.0, reset it and double the events per iteration
       ncalls0=80*ndim*(nchans/3+1)
       double_events=.true.
       if (imode.eq.1 .or. imode.eq.-1) then
          nint_used=nintervals
          nint_used_virt=nintervals_virt
       else
          nint_used=min_inter
          nint_used_virt=min_inter
       endif
    endif
    reset=.false.
    ncalls=0  ! # PS points (updated below)
    if (imode.eq.0.and.born_spread_active.and. &
         born_spread_restart_ncalls.eq.0) &
         born_spread_restart_ncalls=ncalls0
  end subroutine setup_basic_mint

  subroutine update_accumulated_results
    implicit none
    double precision, dimension(nintegrals) :: efrac
    logical :: iterations_done
    call compute_fractional_uncertainty(efrac)
    call print_results_current_iteration(efrac)
! Keep sampling information even from an iteration rejected for a large
! fluctuation: these observations identify channels needing more points.
    call update_channel_sampling
    call check_fractional_uncertainty(efrac)
    if (reset) return ! iteration was not accurate enough: do not include it
    call combine_iterations
    call combine_final_three_iterations
    if (fixed_order) then
       call accum(.true.)
       call HwU_accum_iter(.true.,ntotcalls(1),HwU_values)
    endif
    if (imode.eq.0) then
       call update_virtual_fraction
       call update_integration_grids
    endif
    call check_desired_accuracy(iterations_done)
    if (.not. iterations_done) then
       call prepare_next_iteration
    else
       nit=itmax
    endif
  end subroutine update_accumulated_results

  subroutine prepare_next_iteration
    implicit none
    integer :: kchan,kdim,k_ord_virt
    if (double_events) then
! Double the number of intervals in the grids if not yet reach the maximum
       if (2*nint_used.le.nintervals) then
          do kchan=1,nchans
             do kdim=1,ndim
                call double_grid(kdim,kchan)
             enddo
          enddo
          nint_used=2*nint_used
       endif
       if (2*nint_used_virt.le.nintervals_virt) then
          do k_ord_virt=0,n_ord_virt
             call double_ave_virt(k_ord_virt)
          enddo
          nint_used_virt=2*nint_used_virt
       endif
! double the number of points for the next iteration
       ncalls0=ncalls0*2
    endif
  end subroutine prepare_next_iteration
  
  subroutine check_desired_accuracy(iterations_done)
    implicit none
    logical :: iterations_done
    integer :: i
! Quit if the desired accuracy has been reached
    iterations_done=.false.
    if (nit_included.ge.min_it .and. accuracy.gt.0d0) then
       if (unc(1,0)/ans(1,0)*max(1d0,chi2(1,0)/dble(nit_included-1)).lt.accuracy) then
          write (*,*) 'Found desired accuracy'
          iterations_done=.true.
       elseif(unc_l3(1)/ans_l3(1)*max(1d0,chi2_l3(1)).lt.accuracy) then
          write (*,*) 'Found desired accuracy in last 3 iterations'
          iterations_done=.true.
          ! overwrite results with the results from the last three iterations
          do i=1,nintegrals
             ans(i,0)=ans_l3(i)
             unc(i,0)=unc_l3(i)
             chi2(i,0)=chi2_l3(i)*dble(nit_included-1)
          enddo
       endif
    endif
  end subroutine check_desired_accuracy
  
  subroutine update_integration_grids
    implicit none
    integer :: kchan,kdim,k_ord_virt
    do kchan=1,nchans
       do kdim=1,ndim
          call regrid(kdim,kchan)
       enddo
       ! overwrite xgrid with the new xgrid
       if (regridded(kchan)) xgrid(1:nint_used,1:ndim,kchan)=xgrid_new(1:nint_used,1:ndim)
    enddo
    if (use_poly_virtual) then
       call do_polyfit()
    else
       do k_ord_virt=0,n_ord_virt
          call regrid_ave_virt(k_ord_virt)
       enddo
    endif
    if (VirtTrickSurrogate.ge.1) then
       do kchan=1,nchans
          call vs_solve(kchan,.not.fixed_order)
       enddo
    endif
! Regrid the MC over integers (used for the MC over FKS dirs)
    call regrid_MC_integer
  end subroutine update_integration_grids

  
  subroutine combine_final_three_iterations
    implicit none
    integer :: i,j
    double precision :: weight_unc,old_weight,new_weight
    logical :: initialized
! Update the results of the last tree iterations
    do j=1,2
       ans3(1:nintegrals,j)=ans3(1:nintegrals,j+1)
       unc3(1:nintegrals,j)=unc3(1:nintegrals,j+1)
    enddo
    ans3(1:nintegrals,3)=vtot(1:nintegrals,0)
    unc3(1:nintegrals,3)=etot(1:nintegrals,0)
! Compute the results of the last three iterations
    if (nit_included.ge.4) then
       do i=1,nintegrals
          ans_l3(i)=0d0
          unc_l3(i)=0d0
          weight_unc=0d0
          initialized=.false.
          chi2_l3(i)=0d0
          do j=1,3 ! the three final iterations
             if (i.ne.1 .and. unc3(i,j).eq.0d0) then
                continue ! do not do anything
             elseif (.not.initialized) then
                ! Start with an actual observation instead of the signed
                ! ans3*1d99 sentinel, which fails for a zero first mean.
                ans_l3(i)=ans3(i,j)
                unc_l3(i)=unc3(i,j)
                weight_unc=unc3(i,j)
                initialized=.true.
             else
                old_weight=unc3(i,j)/(weight_unc+unc3(i,j))
                new_weight=weight_unc/(weight_unc+unc3(i,j))
                ans_l3(i)=(ans_l3(i)/weight_unc+ans3(i,j)/unc3(i,j))/(1d0/weight_unc+1d0/unc3(i,j))
                unc_l3(i)=sqrt((old_weight*unc_l3(i))**2+(new_weight*unc3(i,j))**2)
                weight_unc=1d0/sqrt(1d0/weight_unc**2+1d0/unc3(i,j)**2)
                chi2_l3(i)=chi2_l3(i)+(ans3(i,j)-ans_l3(i))**2/unc3(i,j)**2
             endif
          enddo
          chi2_l3(i)=chi2_l3(i)/2d0 ! three iterations, so 2 degrees of freedom
       enddo
       call print_results_accumulated_three_iterations
    endif
  end subroutine combine_final_three_iterations

  subroutine print_results_accumulated_three_iterations
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,2
       if (ans_l3(i).ne.0d0) then
          efrac(i)=abs(unc_l3(i)/ans_l3(i))
       else
          efrac(i)=0d0
       endif
       if (ans_l3(i).ne.0d0 .and. unc_l3(i).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')  &
               'accumulated results last 3 iterations '//title(i)//' =' , &
               ans_l3(i),' +/- ',unc_l3(i) ,' (',efrac(i)*100d0 ,'%)'
       endif
    enddo
    write(*,'(a,1x,e10.4)') 'accumulated result over the last 3 iterations Chi^2 per DoF =' &
         ,chi2_l3(1)
  end subroutine print_results_accumulated_three_iterations

  
  subroutine update_virtual_fraction
! Update the fraction of the events for which we include the virtual corrections
! in the calculation
    implicit none
    integer kchan,k_ord_virt
    double precision :: error_virt
    do kchan=1,nchans
       error_virt=0d0
       do k_ord_virt=1,n_ord_virt
          error_virt=error_virt+etot(2*k_ord_virt+5,kchan)**2
       enddo
       error_virt=sqrt(error_virt)
       virtual_fraction(kchan)=max(min(virtual_fraction(kchan) &
            *max(min(2d0*error_virt/etot(1,kchan),2d0),0.25d0),1d0) &
            ,Min_virt_fraction_mint)
    enddo
  end subroutine update_virtual_fraction

  
  subroutine combine_iterations
    implicit none
    integer i,kchan
    double precision :: old_weight,new_weight
    HwU_values(1)=etot(1,0)
    HwU_values(2)=iteration_weight_unc(1)
    if(nit.eq.1) then ! first iteration
       ans(1:nintegrals,0:nchans)=vtot(1:nintegrals,0:nchans)
       unc(1:nintegrals,0:nchans)=etot(1:nintegrals,0:nchans)
       iteration_weight_unc(1:nintegrals)=etot(1:nintegrals,0)
       ans_chan(0:nchans)=ans(1,0:nchans)
       write (*,'(a,1x,e10.4)') 'Chi^2 per d.o.f.',0d0
    else
       do kchan=nchans,0,-1 ! go backwards so that kchan=0 goes last
                            ! (this makes sure central value is correctly updated).
          do i=1,nintegrals
             if (i.ne.1 .and. (etot(i,0).eq.0d0 .or. iteration_weight_unc(i).eq.0d0)) then
                continue ! do not do anything
             else
                old_weight=etot(i,0)/(iteration_weight_unc(i)+etot(i,0))
                new_weight=iteration_weight_unc(i)/(iteration_weight_unc(i)+etot(i,0))
                ans(i,kchan)=(ans(i,kchan)/iteration_weight_unc(i)+vtot(i,kchan)/etot(i,0)) &
                     /(1d0/iteration_weight_unc(i)+1d0/etot(i,0))
                ! Every channel uses the total-integral weights. Propagate
                ! its variance with those same weights, including zero
                ! channel errors; the total retains its own covariance.
                ! As for HwU, this treats the estimated weights as fixed.
                unc(i,kchan)=sqrt((old_weight*unc(i,kchan))**2+(new_weight*etot(i,kchan))**2)
                if (kchan.eq.0) iteration_weight_unc(i)= &
                     1d0/sqrt(1d0/iteration_weight_unc(i)**2+1d0/etot(i,0)**2)
                if (etot(i,kchan).gt.0d0) chi2(i,kchan)= &
                     chi2(i,kchan)+(vtot(i,kchan)-ans(i,kchan))**2/etot(i,kchan)**2
             endif
          enddo
          ans_chan(kchan)=ans(1,kchan)
       enddo
       write (*,'(a,1x,e10.4)') 'Chi^2=',(vtot(1,0)-ans(1,0))**2/etot(1,0)**2
    endif
    nit_included=nit_included+1
    call print_results_accumulated
    cross_section=ans(1,0)
  end subroutine combine_iterations

  subroutine print_results_accumulated
    implicit none
    integer i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (ans(i,0).ne.0d0) then
          efrac(i)=abs(unc(i,0)/ans(i,0))
       else
          efrac(i)=0d0
       endif
       if (ans(i,0).ne.0d0 .and. unc(i,0).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')  &
               'accumulated results '//title(i)//' =',ans(i,0),' +/- ',unc(i,0) ,' (',efrac(i)*100d0,'%)'
       endif
    enddo
    if (nit_included.le.1) then
       write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF =',0d0
    else
       write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF =',chi2(1,0)/dble(nit_included-1)
    endif
  end subroutine print_results_accumulated

 
  subroutine check_fractional_uncertainty(efrac)
    implicit none
    double precision, dimension(nintegrals) :: efrac
    logical, save :: bad_iteration=.false.
    integer iappl
    common /for_applgrid/ iappl
! If there was a large fluctation in this iteration, be careful with
! including it in the accumulated results and plots.
    if (efrac(1).gt.0.3d0 .and. iappl.eq.0 .and. nit.gt.3) then
! Do not include the results in the plots
       if (fixed_order) call accum(.false.)
       if (fixed_order) call HwU_accum_iter(.false.,ntotcalls(1),HwU_values)
! Do not include the results in the updating of the grids.
       write (*,*) 'Large fluctuation ( >30 % ). Not including iteration in results.'
! empty the accumulated results in the MC over integers
       call empty_MC_integer
! empty the accumulated results for the MINT grids (Cannot really
! skip the increase of the upper bounding envelope. So, simply
! continue here. Note that no matter how large the integrand for the
! PS point, the upper bounding envelope is at most increased by a
! factor 2, so this should be fine).
       reset=.true.
! double the number of points for the next iteration
       if (double_events) ncalls0=ncalls0*2
       if (bad_iteration .and. imode.eq.0 .and. double_events) then
! 2nd bad iteration is a row. Reset grids
          write (*,*)'2nd bad iteration in a row. Resetting grids and starting from scratch...'
          if (double_events) then
             if (imode.eq.0) nint_used=min_inter ! reset number of intervals
             ncalls0=ncalls0/8   ! Start with larger number
          endif
          call reset_mint_grids
          call reset_MC_grid  ! reset the grid for the integers
          if (fixed_order) call initplot  ! Also reset all the plots
          call setup_common
          bad_iteration=.false.
       else
          bad_iteration=.true.
       endif
    else
       bad_iteration=.false.
    endif
  end subroutine check_fractional_uncertainty

  

  subroutine print_results_current_iteration(efrac)
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (vtot(i,0).ne.0d0 .and. etot(i,0).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)') &
               title(i)//' =',vtot(i,0),' +/- ',etot(i,0),' (',efrac(i)*100d0 ,'%)'
       endif
    enddo
  end subroutine print_results_current_iteration

  subroutine compute_fractional_uncertainty(efrac)
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (vtot(i,0).ne.0d0) then
          efrac(i)=abs(etot(i,0)/vtot(i,0))
       else
          efrac(i)=0d0
       endif
    enddo
  end subroutine compute_fractional_uncertainty
  
  subroutine combine_results_channels
    implicit none
    integer :: kchan
    vtot(1:nintegrals,0)=sum(vtot(1:nintegrals,1:nchans),dim=2)
    etot(1:nintegrals,0)=sum(etot(1:nintegrals,1:nchans),dim=2)
    do kchan=0,nchans
       vtot(1:nintegrals,kchan)=vtot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
       etot(1:nintegrals,kchan)=etot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
       etot(1:nintegrals,kchan)=sqrt(abs(etot(1:nintegrals,kchan)-vtot(1:nintegrals,kchan)**2)  &
                                /dble(ntotcalls(1:nintegrals)))
    enddo
  end subroutine combine_results_channels
  
  subroutine check_for_special_channels_loop(channel_loop_done)
    implicit none
    logical, intent(out) :: channel_loop_done
    integer :: kchan
    do kchan=nchans,1,-1
       if (ans_chan(kchan).eq.1d0) then
! results of the current channel
          vtot(1:nintegrals,kchan)=vtot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
          etot(1:nintegrals,kchan)=etot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
          etot(1:nintegrals,kchan)=sqrt(abs(etot(1:nintegrals,kchan)-vtot(1:nintegrals,kchan)**2)  &
                                   /dble(ntotcalls(1:nintegrals)))
          if (kchan.eq.nchans) then
! done all channels
             channel_loop_done=.true.
             return
          endif
! prepare for the next channel
          ans_chan(kchan)=0d0
          ans_chan(kchan+1)=1d0
          ntotcalls(1:nintegrals)=0
          non_zero_point(1:nintegrals)=0
          pass_cuts_point=0
          kpoint_iter=0
          channel_loop_done=.false.
          return
       endif
    enddo
    error stop 'Missing active channel in initial MINT channel loop'
  end subroutine check_for_special_channels_loop

  subroutine combine_results_channels_special_loop
    implicit none
! set the total result for the first iteration to the sum over all the channels
    vtot(1:nintegrals,0)=sum(vtot(1:nintegrals,1:nchans),dim=2)
    etot(1:nintegrals,0)=sum(etot(1:nintegrals,1:nchans)**2,dim=2)
    etot(1:nintegrals,0)=sqrt(etot(1:nintegrals,0))
    ncalls0=ncalls0*nchans
    initial_channel_scan_pending=.false.
  end subroutine combine_results_channels_special_loop
  

  subroutine get_amount_of_points(enough_points)
    ! fill the ntotcalls() array with the total number of calls used
    ! and check if this is enough for this iteration.
    implicit none
    logical :: enough_points
    integer :: i
    do i=1,nintegrals
! Number of phase-space points used
       ntotcalls(i)=ncalls*kpoint_iter
! Special for the computation of the 'computed virtual'
       if (i.eq.4 .and. non_zero_point(i).ne.0 ) &
            ntotcalls(i) = non_zero_point(i)
    enddo
    
    if (.not.double_events) then
! If not doubling the number of events for each iteration, nothing
! needs to be done here.
       enough_points=.true.
       return
    endif
    if (pass_cuts_point.lt.25) then
! Not enough points have passed to cuts to get a reliable estimate
       if (ntotcalls(1).gt.max_points) then
! tried many points already. Need to crash. 
          write (*,*) 'ERROR: NOT ENOUGH POINTS PASS THE CUTS. ' // &
               'RESULTS CANNOT BE TRUSTED. ' // &
               'LOOSEN THE GENERATION CUTS, OR ADAPT SET_TAU_MIN()' // &
               ' IN SETCUTS.F ACCORDINGLY.'
          stop 1
       else
          enough_points=.false.
          return
       endif
    endif
    if (non_zero_point(1).lt.int(0.99*ncalls)) then
! Not enough (non-zero) points have been generated
       if ( pass_cuts_point.gt.ncalls .and. &
            non_zero_point(1).lt.2) then
! Many points passed the cuts, but less than 2 non-zero integrand
! values: must be that the PDFs or the matrix elements (e.g. coupling
! constants) are numerically zero. End the run gracefully
          if (nit.gt.1 .or. imode.ne.0) then
             write (*,*) 'THE INTEGRAL APPEARS TO BE ZERO: END THE RUN GRACEFULLY.'
             write (*,*) 'TRIED',ntotcalls(1),'PS POINTS AND ONLY '  &
                  ,non_zero_point(1),' GAVE A NON-ZERO INTEGRAND.'
             call close_run_zero_res
             stop 0
          else
! This is for the special channels loop. Simply assume that the result
! for this channel is zero, and go to the next channel. If all
! channels give a zero result, end the run gracefully.
             vtot(1,ichan)=0d0
             if(ichan.eq.nchans .and. all(vtot(1,1:nchans).eq.0d0) ) then
                write (*,*) 'THE INTEGRAL APPEARS TO BE ZERO: END THE RUN GRACEFULLY.'
                write (*,*) 'TRIED',ntotcalls(1),'PS POINTS AND ONLY '  &
                     ,non_zero_point(1),' GAVE A NON-ZERO INTEGRAND.'
                call close_run_zero_res
                stop 0
             endif
             enough_points=.true.
             return
          endif
       else
          if (ntotcalls(1).lt.max_points) then
             enough_points=.false.
             return
          endif
       endif
    endif
    enough_points=.true.
  end subroutine get_amount_of_points
  
  

  subroutine add_point_to_grids(x)
    implicit none
    integer :: kdim,k_ord_virt,ithree,isix
    double precision, dimension(ndimmax) :: x
    double precision :: virtual,born,ratio
! accumulate the function in xacc(icell(kdim),kdim) to adjust the grid later
    call add_abs_weight(xacc)
! Set the Born contribution (to compute the average_virtual) to zero if
! the virtual was not computed for this phase-space point. Compensate by
! including the virtual_fraction.
    do k_ord_virt=0,n_ord_virt
       if (k_ord_virt.eq.0) then
          ithree=3
          isix=6
       else
          ithree=2*k_ord_virt+5
          isix=2*k_ord_virt+6
       endif
       if (f(ithree).ne.0d0.or.(vs_vcomp.and.f(isix).ne.0d0)) then
          born=f(isix)
          ! Undo the actual surrogate used for this point so the fallback
          ! continues learning the exact virtual alongside the fit.
          if (vs_ratio_set) then
             ratio=vs_ratio
          elseif (use_poly_virtual) then
             ratio=polyfit(k_ord_virt)
          else
             ratio=average_virtual(k_ord_virt,ichan)
          endif
          virtual=f(ithree)*virtual_fraction(ichan)+ratio*born
          if (use_poly_virtual) then
             call add_point_polyfit(ichan,k_ord_virt,x(1:ndim-3), &
                  virtual/born,born/wgt_mult)
          else
             call fill_ave_virt(x,k_ord_virt,virtual,born)
          endif
       else
          f(isix)=0d0
       endif
    enddo
  end subroutine add_point_to_grids

  subroutine add_point_to_bounding_envelope
    implicit none
    integer :: kdim,k_ord_virt,ithree,isix
    double precision :: prod
! update the upper bounding envelope total rate
    prod=1d0
    do kdim=1,ndim
       prod=prod*ymax(ncell(kdim),kdim,ichan)
    enddo
    prod=(f(1)/prod)
    if (prod.gt.1d0) then
! Weight for this PS point is larger than current upper bound. Increase
! the bound so that it is equal to the current max weight.  If the new
! point is more than twice as large as current upper bound, increase
! bound by factor 2 only to prevent a single unstable points to
! completely screw up the efficiency
       prod=min(2d0,prod)
       prod=prod**(1d0/dble(ndim))
       do kdim=1,ndim
          ymax(ncell(kdim),kdim,ichan)=ymax(ncell(kdim),kdim,ichan)*prod
       enddo
    endif
! Update the upper bounding envelope virtual. Do not include the
! enhancement due to the virtual_fraction. (And again limit by factor 2
! at most).
    if (f(5)*virtual_fraction(ichan).gt.ymax_virt(ichan)) &
         ymax_virt(ichan) = min(f(5)*virtual_fraction(ichan),ymax_virt(ichan)*2d0)
! for consistent printing in the log files (in particular when doing LO
! runs), set also f(6) to zero when imode.eq.1 and the virtuals are not
! included.
    do k_ord_virt=0,n_ord_virt
       if (k_ord_virt.eq.0) then
          ithree=3
          isix=6
       else
          ithree=2*k_ord_virt+5
          isix=2*k_ord_virt+6
       endif
       if (f(ithree).eq.0) f(isix)=0d0
    enddo
  end subroutine add_point_to_bounding_envelope
     
  subroutine accumulate_the_point(x)
    implicit none
    integer :: i
    double precision :: channel_weight
    double precision, dimension(ndimmax) :: x
    if (fixed_order.and.nchans.gt.1) then
! Remove the channel-selection compensation and split-job normalization.
! The second moment of this conditional weight determines the optimal
! random-channel probabilities. Count rejected/zero points as well.
       channel_weight=f(1)*vol_chan/wgt_mult
       channel_squares(ichan)=channel_squares(ichan)+channel_weight**2
       channel_calls(ichan)=channel_calls(ichan)+1
    endif
    if(imode.eq.0) then
       call add_point_to_grids(x)
    else
       call add_point_to_bounding_envelope
    endif
    do i=1,nintegrals
       if (f(i).ne.0d0) non_zero_point(i)=non_zero_point(i)+1
    enddo
    if (pass_cuts_check) pass_cuts_point=pass_cuts_point+1
! Add the PS point to the result of this iteration
    vtot(1:nintegrals,ichan)=vtot(1:nintegrals,ichan)+f(1:nintegrals)
    etot(1:nintegrals,ichan)=etot(1:nintegrals,ichan)+f(1:nintegrals)**2
! Accumulate the points in the HwU histograms    
    if (f(1).ne.0d0) call HwU_add_points
  end subroutine accumulate_the_point

  
  subroutine compute_integrand(fun,x,vol)
    implicit none
    integer :: ifirst,iret
    integer, dimension(ndimmax) :: kfold
    double precision :: dummy,vol
    double precision, dimension(nintegrals) :: f1
    double precision, dimension(ndimmax) :: x
    double precision, external :: fun
    ! Select once per complete point; never consume physics random numbers.
    call timing_begin_point()
    ! contribution to integral
    ifirst=0
    if(imode.eq.0.and..not.point_evaluation%all_folds) then
       dummy=fun(x,vol,ifirst,f1)
       if (.not. fixed_order) dummy=fun(x,vol,2,f1)
       f(1:nintegrals)=f1(1:nintegrals)
    else
       f(1:nintegrals)=0d0
       kfold(1:ndim)=1
1      continue
       ! this accumulated value will not be used
       dummy=fun(x,vol,ifirst,f1)
       ifirst=1
       call nextlexi(ifold,kfold,iret)
       if(iret.eq.0) then
          call get_random_x_next_fold(x,vol,kfold)
          goto 1
       endif
       !closing call: accumulated value with correct sign
       ifirst=2
       dummy=fun(x,vol,ifirst,f1)
       f(1:nintegrals)=f1(1:nintegrals)
    endif
    call timing_end_point()
  end subroutine compute_integrand

  subroutine get_mint_wgt(x,vol)
    ! Given a random point x, compute the corresponding wgt from the
    ! importance sampling accompanying that point.
    implicit none
    double precision, dimension(ndimmax),intent(in) :: x
    double precision,intent(out) :: vol
    integer :: kdim,icell
    double precision :: dx,u
    ! This assumes we are in channel 'ichan'
    vol=1d0/vol_chan * wgt_mult
    do kdim=1,ndim
       call flat_grid_position(x(kdim),kdim,icell,dx,u)
       vol=vol*dx*nint_used/ifold(kdim)
    enddo
  end subroutine get_mint_wgt

  subroutine resample_sector_radiation(x_in,x_out,weight_ratio,sector_id)
    ! Give an outer FKS sector its own radiation sample: translate the flat
    ! adaptive-grid xi/y coordinates by the sector's static quarter-period
    ! Sobol gauge composed with its learned tile map. The translation is a
    ! function of the latent MINT coordinates, with no additional RNG, and
    ! the exact binary offsets preserve the integration measure.
    implicit none
    double precision,intent(in) :: x_in(ndimmax)
    double precision,intent(out) :: x_out(ndimmax),weight_ratio
    integer,intent(in) :: sector_id
    integer :: kdim,old_cell,new_cell
    double precision :: old_dx,new_dx,shift(2),u,fraction
    integer :: index
    x_out=x_in
    weight_ratio=1d0
    ! The sector's gauge offset composed with its learned tile displacement.
    call sl_unified_add_phase(sector_id,shift,x_in)
    index=0
    if(sl_split_point)then
       call sl_split_ensure(0)
       index=sl_split_index(sector_id)
    endif
    do kdim=ndim-2,ndim-1
       call flat_grid_position(x_in(kdim),kdim,old_cell,old_dx,u)
       if(index.gt.0)sl_split_cell(kdim-ndim+3,index)=old_cell
       ! Preserve a zero-shift sector exactly, including endpoints.
       if(shift(kdim-ndim+3).eq.0d0)cycle
       u=modulo(u+shift(kdim-ndim+3),1d0)
       new_cell=min(int(u*nint_used)+1,nint_used)
       fraction=u*nint_used-(new_cell-1)
       new_dx=xgrid(new_cell,kdim,ichan)-xgrid(new_cell-1,kdim,ichan)
       x_out(kdim)=xgrid(new_cell-1,kdim,ichan)+fraction*new_dx
       weight_ratio=weight_ratio*new_dx/old_dx
       if(index.gt.0)sl_split_cell(kdim-ndim+3,index)=new_cell
    enddo
  end subroutine resample_sector_radiation
  
  subroutine get_random_x(x,vol,kfold)
    implicit none
    integer :: kdim,k_ord_virt,nintcurr
    integer, dimension(ndimmax) :: kfold
    double precision :: vol,dx
    double precision, dimension(ndimmax) :: x
    call get_channel
! find random x, and its random cell
    do kdim=1,ndim
! if(even_rn), we should compute the ncell and the rand from the ran3()
       if (even_rn) then
          rand(kdim)=ran3(even_rn)
          ncell(kdim)= min(int(rand(kdim)*nint_used)+1,nint_used)
          rand(kdim)=rand(kdim)*nint_used-(ncell(kdim)-1)
       else
          ncell(kdim)=min(int(nint_used/ifold(kdim)*ran3(even_rn))+1,nint_used/ifold(kdim))
          rand(kdim)=ran3(even_rn)
       endif
    enddo
    kfold(1:ndim)=1
    entry get_random_x_next_fold(x,vol,kfold)
    vol=1d0/vol_chan * wgt_mult
! convert 'flat x' ('rand') to 'vegas x' ('x') and include jacobian ('vol')
    do kdim=1,ndim
       nintcurr=nint_used/ifold(kdim)
       icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
       dx=xgrid(icell(kdim),kdim,ichan)-xgrid(icell(kdim)-1,kdim,ichan)
       vol=vol*dx*nintcurr
       x(kdim)=xgrid(icell(kdim)-1,kdim,ichan)+rand(kdim)*dx
       if(imode.eq.0.and.point_evaluation%integration) &
            nhits(icell(kdim),kdim,ichan)= &
            nhits(icell(kdim),kdim,ichan)+1
    enddo
    do k_ord_virt=0,n_ord_virt
       if (use_poly_virtual) then
          call get_polyfit(ichan,k_ord_virt,x(1:ndim-3),polyfit(k_ord_virt))
       else
          call get_ave_virt(x,k_ord_virt)
       endif
    enddo
  end subroutine get_random_x
  

  subroutine start_iteration
    implicit none
    call write_channel_info
    nit=nit+1
    write (*,*) '------- iteration',nit
    call check_evenly_random_numbers
    if (imode.eq.0) then
       call reset_accumulated_grids_for_updating
    else
       ! Start MINT1 with a fresh iteration-rejection flag.
       reset=.false.
    endif
    vtot(1:nintegrals,0:nchans)=0d0
    etot(1:nintegrals,0:nchans)=0d0
    kpoint_iter=0
    non_zero_point(1:nintegrals)=0
    pass_cuts_point=0
    channel_squares=0d0
    channel_calls=0
  end subroutine start_iteration

  subroutine reset_accumulated_grids_for_updating
    implicit none
    integer :: kchan
    do kchan=1,nchans
       ! only reset if grids were updated (or there is a forced reset)
       if (regridded(kchan).or.reset) then
          if (regridded(kchan) .and. .not. reset) then
             ! set nhits_in_grids equal to the number of points used for the last update
             nhits_in_grids(kchan)=sum(nhits(1:nint_used,1,kchan),dim=1)
          elseif (regridded(kchan) .and. reset) then
             nhits_in_grids(kchan)=0
          endif
          xacc(0:nint_used,1:ndim,kchan)=0d0
          nhits(1:nint_used,1:ndim,kchan)=0
       endif
    enddo
    reset=.false.
  end subroutine reset_accumulated_grids_for_updating

  subroutine check_evenly_random_numbers
    implicit none
    if (even_rn .and. ncalls.ne.ncalls0) then
       ! Uses more evenly distributed random numbers. This overwrites
       ! the number of calls
       call initialize_even_random_numbers
       write (*,*) 'Update # PS points (even_rn): ',ncalls0,' --> ',ncalls
    elseif (ncalls0.ne.ncalls) then
       ncalls=ncalls0
       write (*,*) 'Update # PS points: ',ncalls0,' --> ',ncalls
    endif
    npoints=ncalls
  end subroutine check_evenly_random_numbers

  subroutine finalise_mint
    implicit none
    integer :: kchan
    call write_channel_info
    if (nit_included.ge.2) then
       chi2(1,0:nchans)=chi2(1,0:nchans)/dble(nit_included-1)
    else
       chi2(1,0:nchans)=0d0
    endif
    write (*,*) '-------'
    ncalls0=ncalls*kpoint_iter ! return number of points used
    if (double_events) then
       itmax=2
    else
       itmax=nit_included
    endif
    cross_section=ans(2,0)
    do kchan=1,nchans
       if (regridded(kchan)) then
       ! set equal to number of points used for the last update
          nhits_in_grids(kchan)=sum(nhits(1:nint_used,1,kchan),dim=1) 
       endif
    enddo
    call write_grids_to_file
    call write_results
  end subroutine finalise_mint

  subroutine write_results
    implicit none
    integer :: kchan
    if (fixed_order) then
       write(*,*)'Final result [ABS]:',ans(1,0),' +/-',unc(1,0)
       write(*,*)'Final result:',ans(2,0),' +/-',unc(2,0)
       write(*,*)'chi**2 per D.o.F.:',chi2(1,0)
       open(unit=58,file='results.dat',status='unknown')
       do kchan=0,nchans
          write(58,*) ans(1,kchan),unc(2,kchan),0d0,0,0,0,0,0d0,0d0,ans(2,kchan)
       enddo
       close(58)
    else
       if (imode.eq.0) then
          open(unit=58,file='res_0',status='unknown')
          write(58,*)'Final result [ABS]:',ans(1,1),' +/-',unc(1,1)
          write(58,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          close(58)
          write(*,*)'Final result [ABS]:',ans(1,1),' +/-',unc(1,1)
          write(*,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          write(*,*)'chi**2 per D.o.F.:',chi2(1,1)
       elseif (imode.eq.1) then
! If integrating the virtuals alone, we include the virtuals in
! ans(1). Therefore, no need to have them in ans(5) and we have to set
! them to zero.
          if (only_virt) then
             ans(3,1)=0d0 ! virtual Xsec
             ans(5,1)=0d0 ! ABS virtual Xsec
          endif
          open(unit=58,file='res_1',status='unknown')
          write(58,*)'Final result [ABS]:',ans(1,1)+ans(5,1),' +/-',sqrt(unc(1,1)**2+unc(5,1)**2)
          write(58,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          close(58)
          write(*,*)'Final result [ABS]:',ans(1,1)+ans(5,1),' +/-',sqrt(unc(1,1)**2+unc(5,1)**2)
          write(*,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          write(*,*)'chi**2 per D.o.F.:',chi2(1,1)
          open(unit=58,file='results.dat',status='unknown')
          write(58,*)ans(1,1)+ans(5,1),unc(2,1),0d0,0,0,0,0,0d0,0d0,ans(2,1) 
          close(58)
       else
          continue
       endif
    endif
  end subroutine write_results
  
  subroutine write_channel_info
    implicit none
    integer :: kchan,np
    do kchan=1,nchans
       np=sum(nhits(1:nint_used,1,kchan))
       write (*,250) 'channel',kchan,':',iconfigs(kchan) &
            ,regridded(kchan),np,nhits_in_grids(kchan)   &
            ,ans_chan(kchan),ans(2,kchan),virtual_fraction(kchan)
    enddo
    if (fixed_order.and.nchans.gt.1) &
         write (*,*) 'Channel sampling probabilities:',channel_probability(1:nchans)
    call flush(6)
    return
250 format(a7,i5,1x,a1,1x,i5,1x,l,1x,i8,1x,i8,2x,e10.4,2x,e10.4,2x,e10.4)
  end subroutine write_channel_info
  

  subroutine setup_imode_1
    implicit none
    even_rn=.false.
    min_it=min_it1
    call reset_upper_bounding_envelope
    ans_chan(1:nchans)=ans(1,1:nchans)
    ans_chan(0)=sum(ans(1,1:nchans))
  end subroutine setup_imode_1

  subroutine reset_upper_bounding_envelope
    implicit none
    integer :: kdim,kint,nintcurr,nintcurr_virt
    do kdim=1,ndim
       nintcurr=nint_used/ifold(kdim)
       nintcurr_virt=nint_used_virt/ifold(kdim)
       if(nintcurr*ifold(kdim).ne.nint_used .or. &
            nintcurr_virt*ifold(kdim).ne.nint_used_virt) then
          write(*,*) 'mint: the values in the ifold array should be divisors of', &
               nint_used,'and',nint_used_virt
          stop 1
       endif
       do kint=1,nintcurr
          ymax(kint,kdim,1:nchans)=ans(1,1:nchans)**(1d0/ndim)
       enddo
    enddo
    ymax_virt(1:nchans)=ans(5,1:nchans)
  end subroutine reset_upper_bounding_envelope
  
  subroutine setup_imode_m1
    implicit none
    even_rn=.true.
    imode=0
    min_it=min_it0
    ans_chan(1:nchans)=ans(1,1:nchans)
    ans_chan(0)=sum(ans(1,1:nchans))
  end subroutine setup_imode_m1
  
  subroutine setup_imode_0
    implicit none
    even_rn=.true.
    min_it=min_it0
    call reset_mint_grids
  end subroutine setup_imode_0

  subroutine reset_mint_grids
    implicit none
    integer :: kint
    call vs_reset
    do kint=0,nint_used
       xgrid(kint,1:ndim,1:nchans)=dble(kint)/nint_used
    enddo
    nhits(1:nint_used,1:ndim,1:nchans)=0
    regridded(1:nchans)=.true.
    nhits_in_grids(1:nchans)=0
    if (use_poly_virtual) then
       call init_polyfit(ndim-3,nchans,n_ord_virt,1000)
    else
       call init_ave_virt
    endif
    virtual_fraction(1:nchans)=max(virt_fraction,min_virt_fraction)
    average_virtual(0:n_ave_virt,1:nchans)=0d0
    ans_chan(0:nchans)=0d0
    initial_channel_scan_pending=double_events
    if (double_events) then
       ! when double events, start with the very first channel only. For the
       ! first iteration, we compute each channel separately.
       ans_chan(0)=1d0
       ans_chan(1)=1d0
       ncalls0=ncalls0/nchans
    endif
  end subroutine reset_mint_grids
  
  subroutine setup_common
    implicit none
    nit=0
    nit_included=0
    ans(1:nintegrals,0:nchans)=0d0
    unc(1:nintegrals,0:nchans)=0d0
    iteration_weight_unc=0d0
    chi2(1:nintegrals,0:nchans)=0d0
    ans3(1:nintegrals,1:3)=0d0
    unc3(1:nintegrals,1:3)=0d0
    ans_l3(1:nintegrals)=0d0
    unc_l3(1:nintegrals)=0d0
    chi2_l3(1:nintegrals)=0d0
    vtot(1:nintegrals,0:nchans)=0d0
    etot(1:nintegrals,0:nchans)=0d0
    ntotcalls(1:nintegrals)=0
    non_zero_point(1:nintegrals)=0
    pass_cuts_point=0
    HwU_values(1:2)=0d0
  end subroutine setup_common
  
  
  subroutine write_grids_to_file
! Write the MINT integration grids to file
    implicit none
    integer :: i,j,k,kchan
    open (unit=12,file='mint_grids',status='unknown')
    do kchan=1,nchans
       do j=0,nintervals
          write (12,*) 'AVE',(xgrid(j,i,kchan),i=1,ndim)
       enddo
       if (imode.ge.1) then
          do j=1,nintervals
             write (12,*) 'MAX',(ymax(j,i,kchan),i=1,ndim)
          enddo
       endif
       if (.not.use_poly_virtual) then
          do j=1,nintervals_virt
             do k=0,n_ord_virt
                write (12,*) 'AVE',(ave_virt(j,i,k,kchan),i=1,ndim-3)
             enddo
          enddo
       endif
       if (imode.ge.1) then
          write (12,*) 'MAX',ymax_virt(kchan)
       endif
       write (12,*) 'SUM',(ans(i,kchan),i=1,nintegrals)
       write (12,*) 'QSM',(unc(i,kchan),i=1,nintegrals)
       if (fixed_order.and.nchans.gt.1) then
! Optional trailing fields are ignored by older readers. The moment is
! independent of wgt_mult, so split jobs can pool it using the call count.
          write (12,*) 'SPE',ncalls0,itmax,nhits_in_grids(kchan), &
               channel_moment(kchan),channel_moment_calls(kchan)
       else
          write (12,*) 'SPE',ncalls0,itmax,nhits_in_grids(kchan)
       endif
       write (12,*) 'AVE',virtual_fraction(kchan),average_virtual(0,kchan)
    enddo
    if (use_poly_virtual) call save_polyfit(12)
    if (VirtTrickSurrogate.ge.1) call vs_write_grids(12,nchans)
    ! Learned sector maps also act during MINT0 training: store the frozen
    ! table at every step. A zero-result grid (close_run_zero_res, possibly
    ! before training ends) has none: read_grids_from_file stops before
    ! it would be used.
    if(sector_learning_active.and.(any(ans(1,1:nchans).ne.0d0).or. &
         any(unc(1,1:nchans).ne.0d0)))call sl_epoch_replay_write_table(12)
    close (12)
  end subroutine write_grids_to_file
  
  subroutine read_grids_from_file
! Read the MINT integration grids from file
    implicit none
    integer :: i,j,k,kchan,idum,ios
    integer,dimension(maxchannels) :: points
    integer(kind=8) :: saved_calls
    double precision :: saved_moment
    character(len=3) :: dummy
    character(len=4096) :: marker
    logical :: saved_surrogate,saved_learning,zero_result
    character(len=512) :: channel_info
    open (unit=12, file='mint_grids',status='old')
    saved_surrogate=.false.
    saved_learning=.false.
    call vs_reset
    do
       read(12,'(a)',iostat=ios) marker
       if (ios.lt.0) exit
       if (ios.ne.0) then
          write(*,*) 'ERROR: cannot inspect mint_grids.'
          stop 1
       endif
       marker=adjustl(marker)
       if (marker(1:3).eq.'VSU') saved_surrogate=.true.
       ! Any learning metadata marks a learned grid; its table is checked strictly.
       if (marker(1:3).eq.'SCL') saved_learning=.true.
    enddo
    if(sector_learning_active)call sl_require_matched111()
    rewind(12)
    ans(1,0)=0d0
    unc(1,0)=0d0
    do kchan=1,nchans
       do j=0,nintervals
          read (12,*) dummy,(xgrid(j,i,kchan),i=1,ndim)
       enddo
       if (imode.ge.2) then
          do j=1,nintervals
             read (12,*) dummy,(ymax(j,i,kchan),i=1,ndim)
          enddo
       endif
       if (.not.use_poly_virtual) then
          do j=1,nintervals_virt
             do k=0,n_ord_virt
                read (12,*) dummy,(ave_virt(j,i,k,kchan),i=1,ndim-3)
             enddo
          enddo
       endif
       if (imode.ge.2) then
          read (12,*) dummy,ymax_virt(kchan)
       endif
       read (12,*) dummy,(ans(i,kchan),i=1,nintegrals)
       read (12,*) dummy,(unc(i,kchan),i=1,nintegrals)
       read (12,'(a)') channel_info
       read (channel_info,*) dummy,idum,idum,nhits_in_grids(kchan)
! Legacy grids have no second moment. Their ABS integral supplies the
! initial relative importance; exploration also covers zero estimates.
       channel_moment(kchan)=ans(1,kchan)**2
       channel_moment_calls(kchan)=0
       read (channel_info,*,iostat=ios) dummy,idum,idum,idum,saved_moment,saved_calls
       if (ios.eq.0) then
          channel_moment(kchan)=max(0d0,saved_moment)
          channel_moment_calls(kchan)=max(0_8,saved_calls)
       endif
       read (12,*) dummy,virtual_fraction(kchan),average_virtual(0,kchan)
       ans(1,0)=ans(1,0)+ans(1,kchan)
       unc(1,0)=unc(1,0)+unc(1,kchan)**2
    enddo
    unc(1,0)=sqrt(unc(1,0))
    ! polyfit stuff:
    if (use_poly_virtual) then
       do kchan=1,nchans
          read (12,*) dummy,points(kchan)
       enddo
       do kchan=1,nchans
          backspace(12)
       enddo
       call init_polyfit(ndim-3,nchans,n_ord_virt,maxval(points(1:nchans)))
       call restore_polyfit(12)
       call do_polyfit()
    endif
    ! Older grids have no VSU rows and keep the grid/polynomial fallback.
    ! Consume saved rows even when the fitted surrogate has been disabled.
    if (saved_surrogate) call vs_read_grids(12,nchans,VirtTrickSurrogate.ge.1)
    zero_result=imode.ne.0.and.ans(1,0).eq.0d0.and.unc(1,0).eq.0d0
    ! Zero-result grids carry no frozen sector table (write_grids_to_file).
    if(.not.zero_result)then
       if(saved_learning.neqv.sector_learning_active)then
          write(*,*) 'FKSSectorLearning differs from the saved MINT training.'
          write(*,*) 'Rerun integration and grid setup before restarting.'
          stop 1
       endif
       if(sector_learning_active)then
          call sl_epoch_replay_read_table(12)
          call sl_verify_grid
       endif
    endif
    close (12)
! check for zero cross-section: if restoring grids corresponding to
! sigma=0, just terminate the run
    if (zero_result) then
       call initplot()
       call close_run_zero_res
       stop 0
    endif
  end subroutine read_grids_from_file

  subroutine double_grid(kdim,kchan)
    implicit none
    integer :: kchan,kdim,i
    do i=nint_used,1,-1
       xgrid(i*2,kdim,kchan)=xgrid(i,kdim,kchan)
       xgrid(i*2-1,kdim,kchan)=(xgrid(i,kdim,kchan)+xgrid(i-1,kdim,kchan))/2d0
       if ((.not.regridded(kchan)) .and. (.not.reset)) then
          nhits(i*2,kdim,kchan)=nhits(i,kdim,kchan)/2
          nhits(i*2-1,kdim,kchan)=nhits(i,kdim,kchan)-nhits(i*2,kdim,kchan)
          if (nhits(i,kdim,kchan).ne.0) then
             xacc(i*2,kdim,kchan)=xacc(i,kdim,kchan)*nhits(i*2,kdim,kchan)/dble(nhits(i,kdim,kchan))
             xacc(i*2-1,kdim,kchan)=xacc(i,kdim,kchan)*nhits(i*2-1,kdim,kchan)/dble(nhits(i,kdim,kchan))
          else
             xacc(i*2,kdim,kchan)=0d0
             xacc(i*2-1,kdim,kchan)=0d0
          endif
       endif
    enddo
  end subroutine double_grid

  subroutine regrid(kdim,kchan)
    implicit none
    integer :: kdim,kchan,kint,jint
    double precision :: r,total
    double precision, parameter :: tiny=1d-8
! compute total number of points and update grids if large
    regridded(kchan)=.false.
    if (sum(nhits(1:nint_used,kdim,kchan),dim=1).lt.nint(0.9*nhits_in_grids(kchan))) return
    regridded(kchan)=.true.
! Use the same smoothing as in VEGAS uses for the grids, i.e. use the
! average of the central and the two neighbouring grid points: (Only do
! this if we are already at the maximum intervals, because the doubling
! of the grids also includes a smoothing).
    if (nint_used.eq.nintervals) then
       call smooth_xacc(kdim,kchan)
    endif
    do kint=1,nint_used
       if (nhits(kint,kdim,kchan).ne.0) then
          xacc(kint,kdim,kchan)=abs(xacc(kint,kdim,kchan))/nhits(kint,kdim,kchan)
       else
          xacc(kint,kdim,kchan)=0d0
       endif
    enddo
! Overwrite xacc so that it accumulates the cross section with each
! successive interval.  It already contains a factor equal to the
! interval size. Thus the integral of rho is performed by summing up
    total=sum(xacc(1:nint_used,kdim,kchan),dim=1)
    do kint=1,nint_used
       if(nhits(kint,kdim,kchan).ne.0) then
!     take logarithm to help convergence (taken from LO dsample.f)
          if (xacc(kint,kdim,kchan).ne.total) then
             xacc(kint,kdim,kchan)=((xacc(kint,kdim,kchan)/total-1d0)/log(xacc(kint,kdim,kchan)/total))**1.5
          else
             xacc(kint,kdim,kchan)=1d0
          endif
          xacc(kint,kdim,kchan)= xacc(kint-1,kdim,kchan) + abs(xacc(kint,kdim,kchan))
       else
          xacc(kint,kdim,kchan)=xacc(kint-1,kdim,kchan)
       endif
    enddo
! No valid points. Simply return
    if (xacc(nint_used,kdim,kchan).eq.0d0) return
! normalise xacc so that it goes from 0 to 1.
    xacc(1:nint_used,kdim,kchan)=xacc(1:nint_used,kdim,kchan)/xacc(nint_used,kdim,kchan)
! Check that we have a reasonable result and update the accumulated results if need be
    do kint=1,nint_used
       if (xacc(kint,kdim,kchan).lt.(xacc(kint-1,kdim,kchan)+tiny)) then
          xacc(kint,kdim,kchan)=xacc(kint-1,kdim,kchan)+tiny
       endif
    enddo
! it could happen that the change above yielded xacc() values greater than 1: one more update needed
    xacc(nint_used,kdim,kchan)=1d0
    do kint=1,nint_used
       if (xacc(nint_used-kint,kdim,kchan).gt.(xacc(nint_used-kint+1,kdim,kchan)-tiny)) then
          xacc(nint_used-kint,kdim,kchan)=1d0-dble(kint)*tiny
       else
          exit
       endif
    enddo
! adjust 'xgrid_new' (temporary grid) so that each element contains identical cross section    
    xgrid_new(0,kdim)=0d0
    do kint=1,nint_used
       r=dble(kint)/dble(nint_used)
       do jint=1,nint_used
          if(r.lt.xacc(jint,kdim,kchan)) then
             xgrid_new(kint,kdim)=xgrid(jint-1,kdim,kchan)+(r-xacc(jint-1,kdim,kchan))/ &
                  (xacc(jint,kdim,kchan)-xacc(jint-1,kdim,kchan))* &
                  (xgrid(jint,kdim,kchan)-xgrid(jint-1,kdim,kchan))
             goto 11
          endif
       enddo
       if(jint.ne.nint_used+1.and.kint.ne.nint_used) then
          write(*,*) 'ERROR',jint,nint_used
          stop 1
       endif
11     continue
    enddo
    xgrid_new(nint_used,kdim)=1d0
  end subroutine regrid

  subroutine smooth_xacc(kdim,kchan)
    implicit none
    integer :: kdim,kchan,kint,kk,itot,kkint
    double precision :: tot
    integer, parameter :: isize=1
    integer, dimension(1:nintervals) :: local_nhits
    double precision, dimension(1:nintervals) :: local_xacc
    do kint=1,nint_used
       tot=0d0
       itot=0
       do kk=-isize,isize
          kkint=kint+kk
          if (kkint.le. 0) kkint=1
          if (kkint.ge.nint_used+1) kkint=nint_used
          tot=tot+xacc(kkint,kdim,kchan)
          itot=itot+nhits(kkint,kdim,kchan)
       enddo
       local_xacc(kint)=tot/dble(2*isize+1)
       local_nhits(kint)=nint(itot/dble(2*isize+1))
    enddo
    xacc(1:nint_used,kdim,kchan)=local_xacc(1:nint_used)
    nhits(1:nint_used,kdim,kchan)=local_nhits(1:nint_used)
  end subroutine smooth_xacc
    
  subroutine nextlexi(iii,kkk,iret)
! kkk: array of integers 1 <= kkk(j) <= iii(j), j=1,ndim
! at each call iii is increased lexicographycally.
! for example, starting from ndim=3, kkk=(1,1,1), iii=(2,3,2)
! subsequent calls to nextlexi return
!         kkk(1)      kkk(2)      kkk(3)    iret
! 0 calls   1           1           1       0
! 1         1           1           2       0    
! 2         1           2           1       0
! 3         1           2           2       0
! 4         1           3           1       0
! 5         1           3           2       0
! 6         2           1           1       0
! 7         2           1           2       0
! 8         2           2           1       0
! 9         2           2           2       0
! 10        2           3           1       0
! 11        2           3           2       0
! 12        2           3           2       1
    implicit none
    integer :: k,iret
    integer,dimension(ndimmax) :: kkk,iii
    k=ndim
1   continue
    if(kkk(k).lt.iii(k)) then
       kkk(k)=kkk(k)+1
       iret=0
       return
    else
       kkk(k)=1
       k=k-1
       if(k.eq.0) then
          iret=1
          return
       endif
       goto 1
    endif
  end subroutine nextlexi

  subroutine init_ave_virt
    implicit none
    if (n_ord_virt.gt.n_ave_virt) then
       write (*,*) 'Too many grids to keep track off',n_ord_virt,n_ave_virt
       stop 1
    endif
    nvirt(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0
    ave_virt(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    nvirt_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0
    ave_virt_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    ave_born_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
  end subroutine init_ave_virt

  subroutine get_ave_virt(x,k_ord_virt)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision, dimension(ndimmax) :: x
    average_virtual(k_ord_virt,ichan)=0d0
    do kdim=1,ndim-3
       ncell=min(int(x(kdim)*nint_used_virt)+1,nint_used_virt)
       average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan) &
                                        +ave_virt(ncell,kdim,k_ord_virt,ichan)
    enddo
    average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan)/(ndim-3)
  end subroutine get_ave_virt

  subroutine fill_ave_virt(x,k_ord_virt,virtual,born)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision,dimension(ndimmax) :: x(ndimmax)
    double precision :: virtual,born
    do kdim=1,ndim-3
       ncell=min(int(x(kdim)*nint_used_virt)+1,nint_used_virt)
       nvirt_acc(ncell,kdim,k_ord_virt,ichan)=nvirt_acc(ncell,kdim,k_ord_virt,ichan)+1
       ave_virt_acc(ncell,kdim,k_ord_virt,ichan)=ave_virt_acc(ncell,kdim,k_ord_virt,ichan)+virtual
       ave_born_acc(ncell,kdim,k_ord_virt,ichan)=ave_born_acc(ncell,kdim,k_ord_virt,ichan)+born
    enddo
  end subroutine fill_ave_virt

  subroutine regrid_ave_virt(k_ord_virt)
    implicit none
    integer kchan,kdim,i,k_ord_virt
! need to solve for k_new = (virt+k_old*born)/born = virt/born + k_old
    do kchan=1,nchans
       do kdim=1,ndim-3
          do i=1,nint_used_virt
             if (ave_born_acc(i,kdim,k_ord_virt,kchan).eq.0d0) cycle
             if (ave_virt(i,kdim,k_ord_virt,kchan).eq.0d0) then ! i.e. first iteration
                ave_virt(i,kdim,k_ord_virt,kchan)= ave_virt_acc(i,kdim,k_ord_virt,kchan) &
                     /ave_born_acc(i,kdim,k_ord_virt,kchan)+ave_virt(i,kdim,k_ord_virt,kchan)
             else  ! give some importance to the iterations already done
                ave_virt(i,kdim,k_ord_virt,kchan)=(nvirt_acc(i,kdim,k_ord_virt,kchan)* &
                     ave_virt_acc(i,kdim,k_ord_virt,kchan)/ave_born_acc(i,kdim,k_ord_virt,kchan)+ &
                     nvirt(i,kdim,k_ord_virt,kchan)*ave_virt(i,kdim,k_ord_virt,kchan))/ &
                     dble(nvirt_acc(i,kdim,k_ord_virt,kchan)+nvirt(i,kdim,k_ord_virt,kchan))
             endif
          enddo
       enddo
    enddo
! reset the acc values
    nvirt(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)= &
                 nvirt(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)  &
                 + nvirt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)
    nvirt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0
    ave_born_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0d0
    ave_virt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0d0
  end subroutine regrid_ave_virt

  subroutine double_ave_virt(k_ord_virt)
    implicit none
    integer :: kdim,i,k_ord_virt,kchan
    do kchan=1,nchans
       do kdim=1,ndim-3
          do i=nint_used_virt,1,-1
             ave_virt(i*2,kdim,k_ord_virt,kchan)=ave_virt(i,kdim,k_ord_virt,kchan)
             if (nvirt(i,kdim,k_ord_virt,kchan).ne.0) then
                nvirt(i*2,kdim,k_ord_virt,kchan)=max(nvirt(i,kdim,k_ord_virt,kchan)/2,1)
             else
                nvirt(i*2,kdim,k_ord_virt,kchan)=0
             endif
             if (i.ne.1) then
                ave_virt(i*2-1,kdim,k_ord_virt,kchan)=(ave_virt(i,kdim,k_ord_virt,kchan)  &
                                                      +ave_virt(i-1,kdim,k_ord_virt,kchan))/2d0
                if (nvirt(i,kdim,k_ord_virt,kchan)+nvirt(i-1,kdim,k_ord_virt,kchan).ne.0) then
                   nvirt(i*2-1,kdim,k_ord_virt,kchan)= &
                        max((nvirt(i,kdim,k_ord_virt,kchan)+nvirt(i-1,kdim,k_ord_virt,kchan))/4,1)
                else
                   nvirt(i*2-1,kdim,k_ord_virt,kchan)=0
                endif
             else
                if (nvirt(1,kdim,k_ord_virt,kchan).ne.0) then
                   nvirt(1,kdim,k_ord_virt,kchan)=max(nvirt(1,kdim,k_ord_virt,kchan)/2,1)
                else
                   nvirt(1,kdim,k_ord_virt,kchan)=0
                endif
             endif
          enddo
       enddo
    enddo
  end subroutine double_ave_virt

  subroutine update_channel_sampling
    implicit none
    integer :: kchan
    if (.not.fixed_order.or.nchans.le.1) return
    do kchan=1,nchans
! An unvisited channel has no new estimate; do not replace its history
! with a spurious zero integral and zero error.
       if (channel_calls(kchan).eq.0) cycle
       channel_moment(kchan)=channel_squares(kchan)/dble(channel_calls(kchan))
       channel_moment_calls(kchan)=channel_calls(kchan)
    enddo
    call set_channel_probabilities
  end subroutine update_channel_sampling

  subroutine set_channel_probabilities
    implicit none
    double precision :: scores(maxchannels),largest
! For random channel choice, variance is minimized by p_i proportional
! to sqrt(<w_i**2>). The same moment can be estimated from the current
! iteration as p_i*(A_i**2+N*error_i**2), not the relative error alone.
    scores(1:nchans)=sqrt(max(0d0,channel_moment(1:nchans)))
    largest=maxval(scores(1:nchans))
    if (largest.gt.0d0) then
       scores(1:nchans)=scores(1:nchans)/largest
       channel_probability(1:nchans)=(1d0-channel_exploration)* &
            scores(1:nchans)/sum(scores(1:nchans))+channel_exploration/dble(nchans)
    else
       channel_probability(1:nchans)=1d0/dble(nchans)
    endif
! Only the existing total-error criterion determines convergence. Tiny
! channels do not need small relative errors; their exploration share
! is bounded by channel_exploration for the entire job.
  end subroutine set_channel_probabilities

  subroutine get_channel
! Picks one random 'ichan' among the 'nchans' integration channels and
! fills the channels common block in mint.inc.
    implicit none
    double precision :: trgt,total
    if (nchans.eq.1) then
       ichan=1
       iconfig=iconfigs(ichan)
       vol_chan=1d0
    elseif (nchans.gt.1) then
       if (fixed_order.and..not.initial_channel_scan_pending) then
! Preserve the forced per-channel pilot loop in the first automatic
! iteration. All later selections use the RMS weights plus exploration.
          trgt=ran3(.false.)
          ichan=1
          total=channel_probability(ichan)
          do while (total.lt.trgt.and.ichan.lt.nchans)
             ichan=ichan+1
             total=total+channel_probability(ichan)
          enddo
          iconfig=iconfigs(ichan)
          vol_chan=channel_probability(ichan)
          return
       endif
       if (ans_chan(0).le.0d0) then
!     pick one at random (flat)
          ichan=int(ran3(.false.)*nchans)+1
          iconfig=iconfigs(ichan)
          vol_chan=1d0/dble(nchans)
       else
!     pick one at random (weighted by cross section)
          total=sum(ans_chan(1:nchans))
          if (abs(total-ans_chan(0))/(total+ans_chan(0)).gt.1d-8) then
             write (*,*) 'ERROR: total should be equal to ans',total,ans_chan(0)
             stop 1
          endif
          trgt=ans_chan(0)*ran3(.false.)
          total=0d0
          ichan=0
          do while (total.lt.trgt)
             ichan=ichan+1
             total=total+ans_chan(ichan)
          enddo
          if (ichan.eq.0 .or. ichan.gt.nchans) then
             write (*,*) 'ERROR: ichan cannot be zero or larger than nchans',ichan,nchans
             stop 1
          endif
          iconfig=iconfigs(ichan)
          vol_chan=ans_chan(ichan)/ans_chan(0)
       endif
    endif
  end subroutine get_channel

  subroutine close_run_zero_res
    implicit none
    integer :: kchan
    xgrid(0:nintervals,1:ndim,1:nchans)=0d0
    ymax(1:nintervals,1:ndim,1:nchans)=0d0
    ave_virt(1:nintervals_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    ymax_virt(1:nchans)=0d0
    ans(1:nintegrals,1:nchans)=0d0
    unc(1:nintegrals,1:nchans)=0d0
    nhits_in_grids(1:nchans)=0
    virtual_fraction(1:nchans)=1d0
    average_virtual(0,1:nchans)=0d0
    call write_grids_to_file
    call write_results
    call regrid_MC_integer
    open (unit=12, file='res.dat',status='unknown')
    do kchan=0,nchans
       write (12,*)ans(1,kchan),unc(1,kchan),ans(2,kchan),unc(2,kchan) &
            ,itmax,ncalls0,0d0
    enddo
    close(12)
  end subroutine close_run_zero_res

  function ran3(even)
    implicit none
    double precision :: ran3
    logical :: even
    double precision, external :: ran2
    if (even) then
       ran3=get_ran()
    else
       ran3=ran2()
    endif
  end function ran3

  subroutine initialize_even_random_numbers
! Recompute the number of calls. Uses the algorithm from VEGAS
    implicit none
! Make sure that hypercubes are newly initialized
    firsttime=.true.
! Number of elements in which we can split one dimension
    ng=int((ncalls0/2.)**(1./ndim))
! Total number of hypercubes
    k=ng**ndim
! Number of PS points in each hypercube (at least 2)
    npg=max(ncalls0/k,2)
! Number of PS points for this iteration
    ncalls=npg*k
  end subroutine initialize_even_random_numbers

  function get_ran()
    implicit none
    double precision :: get_ran
    double precision, save :: dng
    double precision, external ::  ran2
    integer, dimension(ndimmax), save ::  iii,kkk
    integer, save :: current_dim
    integer :: i,iret
    if (firsttime) then
! initialise the hypercubes
       dng=1d0/dble(ng)
       current_dim=0
       do i=1,ndim
          iii(i)=ng
          kkk(i)=1
       enddo
       firsttime=.false.
    endif
    current_dim=mod(current_dim,ndim)+1
! This is the random number in the hypercube 'k' for current_dim
    get_ran=dng*(ran2()+dble(kkk(current_dim)-1))
! Got random numbers for all dimensions, update kkk() for the next call
    if (current_dim.eq.ndim) then
       call nextlexi(iii,kkk,iret)
       if (iret.eq.1) then
          call nextlexi(iii,kkk,iret)
       endif
    endif
  end function get_ran

  subroutine gen(fun,gen_mode,vn,x)
    implicit none
    integer :: vn,gen_mode
    logical :: found_point
    double precision, external :: fun
    double precision, dimension(ndimmax) :: x
    double precision :: vol
    if (gen_mode.eq.0) then
       call initialise_mint_gen
    elseif(gen_mode.eq.3) then
       call print_gen_counters
    elseif(gen_mode.eq.1) then
       call increase_gen_counters_before(vn)
10     continue
       new_point=.true.
       if (vn.eq.1) then
          call get_random_cell_flat(x,vol)
       else
          call get_weighted_cell(x,vol)
       endif
       call compute_integrand(fun,x,vol)
       call increase_gen_counters_middle(vn)
       call check_upper_bound(vn,found_point)
       if (.not.found_point) goto 10
       call increase_gen_counters_end(vn)
    else
       write (*,*) "Unknown gen_mode in gen (from mint_module)",gen_mode
       stop 1
    endif
  end subroutine gen

  subroutine increase_gen_counters_middle(vn)
    implicit none
    integer :: vn
    gen_counters(3)=gen_counters(3)+1
    if (vn.eq.1) then
       gen_counters(5)=gen_counters(5)+1
    else
       gen_counters(6)=gen_counters(6)+1
    endif
    if (f(1).eq.0d0) then
       gen_counters(4)=gen_counters(4)+1
    endif
  end subroutine increase_gen_counters_middle
  
  subroutine increase_gen_counters_before(vn)
    implicit none
    integer :: vn
    if (vn.eq.1) then
       gen_counters(1)=gen_counters(1)+1
    else
       gen_counters(2)=gen_counters(2)+1
    endif
  end subroutine increase_gen_counters_before

  subroutine increase_gen_counters_end(vn)
    implicit none
    integer :: vn
    if (vn.eq.2) then
       gen_counters(11)=gen_counters(11)+1
    elseif (vn.eq.1) then
       gen_counters(12)=gen_counters(12)+1
    elseif (vn.eq.3) then
       gen_counters(13)=gen_counters(13)+1
    endif
  end subroutine increase_gen_counters_end

  subroutine check_upper_bound(vn,found_point)
    implicit none
    logical :: found_point
    integer :: vn
    if (f(1).gt.upper_bound) then
       if (vn.eq.2) then
          gen_counters(7)=gen_counters(7)+1
       elseif (vn.eq.1) then
          gen_counters(8)=gen_counters(8)+1
       elseif(vn.eq.3) then
          gen_counters(9)=gen_counters(9)+1
       endif
    endif
    upper_bound=upper_bound*ran3(.false.)
    if (upper_bound.gt.f(1)) then
       gen_counters(10)=gen_counters(10)+1
       found_point=.false.
    else
       found_point=.true.
    endif
  end subroutine check_upper_bound
  
  subroutine get_random_cell_flat(x,vol)
    implicit none
    double precision :: vol
    double precision, dimension(ndimmax) :: x
    integer, dimension(ndimmax) :: kfold
    call get_random_x(x,vol,kfold)
    upper_bound=ymax_virt(ichan)
  end subroutine get_random_cell_flat

  subroutine get_weighted_cell(x,vol)
    implicit none
    integer :: kdim,nintcurr,kint
    integer, dimension(ndimmax) :: kfold
    double precision :: vol,r
    double precision, dimension(ndimmax) :: x
    call get_channel
    do kdim=1,ndim
       nintcurr=nintervals/ifold(kdim)
       r=ran3(.false.)
       do kint=1,nintcurr
          if(r.lt.xmmm(kint,kdim,ichan)) then
             ncell(kdim)=kint
             exit
          endif
       enddo
       rand(kdim)=ran3(.false.)
    enddo
    kfold(1:ndim)=1
    call get_random_x_next_fold(x,vol,kfold)
    upper_bound=1d0
    do kdim=1,ndim
       upper_bound=upper_bound*ymax(ncell(kdim),kdim,ichan)
    enddo
  end subroutine get_weighted_cell
  

  subroutine initialise_mint_gen
    implicit none
    integer :: kdim,kint
    integer :: nintcurr
    if (nchans.ne.1) then
       write (*,*) 'ERROR in mint_module: for event generation, can do only 1 channel at a time',nchans
       stop 1
    endif
    even_rn=.false.
    nint_used=nintervals
    nint_used_virt=nintervals_virt
    do kdim=1,ndim
       nintcurr=nintervals/ifold(kdim)
       xmmm(1,kdim,1:nchans)=ymax(1,kdim,1:nchans)
       do kint=2,nintcurr
          xmmm(kint,kdim,1:nchans)=xmmm(kint-1,kdim,1:nchans)+ymax(kint,kdim,1:nchans)
       enddo
       do kint=1,nintcurr
          xmmm(kint,kdim,1:nchans)=xmmm(kint,kdim,1:nchans)/xmmm(nintcurr,kdim,1:nchans)
       enddo
    enddo
    gen_counters(1:13)=0
  end subroutine initialise_mint_gen

  subroutine print_gen_counters
    implicit none
    double precision :: unwgt_eff, unwgt_eff_virt
    if (gen_counters( 3).ne.0) write (*,*) 'another call to the function:',gen_counters(3)
    if (gen_counters(11).ne.0) write (*,*) 'events generated, novi:',gen_counters(11)
    if (gen_counters(12).ne.0) write (*,*) 'events generated, virt:',gen_counters(12)
    if (gen_counters(13).ne.0) write (*,*) 'events generated, born:',gen_counters(13)
    if (gen_counters( 7).ne.0) write (*,*) 'upper bound failure, novi:',gen_counters(7)
    if (gen_counters( 8).ne.0) write (*,*) 'upper bound failure, virt:',gen_counters(8)
    if (gen_counters( 9).ne.0) write (*,*) 'upper bound failure, born:',gen_counters(9)
    if (gen_counters(10).ne.0) write (*,*) 'vetoed calls in inclusive cross section:',gen_counters(10)
    if (gen_counters( 4).ne.0) write (*,*) 'failed generation cuts:',gen_counters(4)
    if(gen_counters(6).ne.0) then
       unwgt_eff=dble(gen_counters(2))/dble(gen_counters(6))
    else
       unwgt_eff=-1d0
    endif
    if(gen_counters(5).ne.0) then
       unwgt_eff_virt=dble(gen_counters(1))/dble(gen_counters(5))
    else
       unwgt_eff_virt=-1d0
    endif
    write (*,*) 'Generation efficiencies:',unwgt_eff,unwgt_eff_virt
  end subroutine print_gen_counters

  
end module mint_module
  
! Dummy subroutine (normally used with vegas when resuming plots)
subroutine resume()
end subroutine resume
