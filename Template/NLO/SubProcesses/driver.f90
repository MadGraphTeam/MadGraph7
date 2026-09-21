module driver_vec
  implicit none
  ! state variables to see the status of driver_vec
  logical, public :: driver_is_allocated = .false.
  integer, public :: driver_vector_size = 0
  ! surrounding infrastructure variables
  integer, allocatable, public :: MCcnt_vec(:)
  double precision, allocatable, public :: vegas_wgt_vec(:)
  double precision, allocatable, public :: x_mint_vec(:,:)
  double precision, allocatable, public :: x_vegas_vec(:,:)
  double precision, allocatable, public :: x_save_vec(:,:,:)
  double precision, allocatable, public :: f_vec(:,:)
  logical, allocatable, public :: skip_iter_vec(:)
  logical, allocatable, public :: pass_cuts_check_vec(:)
  logical, allocatable, public :: passcuts_born_vec(:)
  logical, allocatable, public :: passcuts_nbody_vec(:,:)
  logical, allocatable, public :: passcuts_n1body_vec(:,:)
  double precision, allocatable, public :: virt_wgt_vec(:,:), born_wgt_vec(:,:)
 ! born helicities
  integer, allocatable, public :: nhel_vec(:,:,:)
  ! n-body kinematics Borns
  double precision, allocatable, public :: snb_amp2(:,:,:)
  double precision, allocatable, public :: snb_jamp2(:,:,:)
  complex*16, allocatable, public :: snb_ans_cnt(:,:,:,:)
  double precision, allocatable, public :: snb_amp_split(:,:,:)
  double complex, allocatable, public :: snb_amp_split_cnt(:,:,:,:,:)
  double complex, allocatable, public :: snb_saveamp(:,:,:,:)
  double precision, allocatable, public :: snb_split_soft(:,:,:)
  double precision, allocatable, public :: swgt_nb(:,:)
  ! n+1-body kinematics Borns
  double precision, allocatable, public :: sn1_amp2(:,:,:)
  double precision, allocatable, public :: sn1_jamp2(:,:,:)
  complex*16, allocatable, public :: sn1_ans_cnt(:,:,:,:)
  double precision, allocatable, public :: sn1_amp_split(:,:,:)
  double complex, allocatable, public :: sn1_amp_split_cnt(:,:,:,:,:)
  double complex, allocatable, public :: sn1_saveamp(:,:,:,:)
  double precision, allocatable, public :: sn1_split_soft(:,:,:)
  double precision, allocatable, public :: swgt_n1(:,:)
  !Store storage for born-like collinears
    complex*16, allocatable, public :: scb_ans_cnt(:,:,:,:)
    double precision, allocatable, public :: scb_amp_split(:,:,:)
    double complex, allocatable, public :: scb_amp_split_cnt(:,:,:,:,:)
    double complex, allocatable, public :: scb_saveamp(:,:,:,:)
  ! Store storage for born-like collinears
    complex*16, allocatable, public :: sc1_ans_cnt(:,:,:,:)
    double precision, allocatable, public :: sc1_amp_split(:,:,:)
    double complex, allocatable, public :: sc1_amp_split_cnt(:,:,:,:,:)
    double complex, allocatable, public :: sc1_saveamp(:,:,:,:)
  ! Store storage for rotated Borns
    double precision, allocatable, public :: srot_jamp2(:,:,:)
    complex*16, allocatable, public :: srot_ans_cnt(:,:,:,:)
    ! double precision, allocatable, public :: srot_amp_split(:,:,:)
    double complex, allocatable, public :: srot_amp_split_cnt(:,:,:,:,:)
    ! double complex, allocatable, public :: srot_saveamp(:,:,:,:)
! norad and ev amplitudes
    double precision, allocatable, public :: snorad_amp2(:,:,:)
    double precision, allocatable, public :: sev_amp2(:,:,:)
    ! Born contributions storage
    double precision, allocatable, public :: sborn_amp2(:,:)
    double precision, allocatable, public :: sborn_jamp2(:,:)
    complex*16, allocatable, public :: sborn_ans_cnt(:,:,:)
    double precision, allocatable, public :: sborn_amp_split(:,:)
    double complex, allocatable, public :: sborn_amp_split_cnt(:,:,:,:)
    double complex, allocatable, public :: sborn_saveamp(:,:,:)
    double precision, allocatable, public :: sborn_split_soft(:,:)
    double precision, allocatable, public :: swgt_born(:)
! Real amplitudes and weights
    double precision, allocatable, public :: sreal_amp_split(:,:,:)
    double precision, allocatable, public :: sfx_ev(:,:)

    ! Momenta

    double precision, allocatable, public :: spb(:,:,:,:), spb_rot(:,:,:,:)
    double precision, allocatable, public :: spb_coll(:,:,:,:), spb_ev(:,:,:,:)
    double precision, allocatable, public :: spb_norad(:,:,:,:), sp1_cnt(:,:,:,:,:)
    double precision, allocatable, public :: sp1(:,:,:,:)

    public :: allocate_storage, reset_storage, deallocate_storage

 contains
 subroutine allocate_storage(vector_size,ndimmax,max_fold,nintegrals,n_ave_virt)
   implicit none
   integer, intent(in) :: vector_size
   integer, intent(in) :: ndimmax,max_fold,nintegrals,n_ave_virt
    include 'nexternal.inc'
    include 'nFKSconfigs.inc'
   include 'genps.inc'
   include 'orders.inc'
   include 'born_nhel.inc'
    driver_vector_size = vector_size
    ! surrounding infrastructure variables
    allocate(MCcnt_vec(vector_size))
    allocate(vegas_wgt_vec(vector_size))
    allocate(x_mint_vec(ndimmax,vector_size))
    allocate(x_vegas_vec(99,vector_size))
    allocate(x_save_vec(ndimmax,max_fold,vector_size))
    allocate(f_vec(nintegrals,vector_size))
    allocate(pass_cuts_check_vec(vector_size))
    allocate(passcuts_born_vec(vector_size))
    allocate(passcuts_nbody_vec(FKS_configs,vector_size))
    allocate(passcuts_n1body_vec(FKS_configs,vector_size))
    ! Virtual and born weights for MINT
    allocate(virt_wgt_vec(0:n_ave_virt,vector_size))
    allocate(born_wgt_vec(0:n_ave_virt,vector_size))
    ! born helicities
    allocate(nhel_vec(nexternal-1,max_bhel,vector_size))
   ! n-body kinematics Borns
   allocate(snb_amp2(ngraphs,FKS_configs,vector_size))
   allocate(snb_jamp2(0:ncolor,FKS_configs,vector_size))
   allocate(snb_ans_cnt(2,nsplitorders,FKS_configs,vector_size))
   allocate(snb_amp_split(amp_split_size,FKS_configs,vector_size))
   allocate(snb_amp_split_cnt(amp_split_size,2,nsplitorders,FKS_configs,vector_size))
   allocate(snb_saveamp(ngraphs,max_bhel,FKS_configs,vector_size))
   allocate(snb_split_soft(amp_split_size,FKS_configs,vector_size))
   allocate(swgt_nb(FKS_configs,vector_size))
   ! n+1-body kinematics Borns
   allocate(sn1_amp2(ngraphs,FKS_configs,vector_size))
   allocate(sn1_jamp2(0:ncolor,FKS_configs,vector_size))
   allocate(sn1_ans_cnt(2,nsplitorders,FKS_configs,vector_size))
   allocate(sn1_amp_split(amp_split_size,FKS_configs,vector_size))
   allocate(sn1_amp_split_cnt(amp_split_size,2,nsplitorders,FKS_configs,vector_size))
   allocate(sn1_saveamp(ngraphs,max_bhel,FKS_configs,vector_size))
    allocate(sn1_split_soft(amp_split_size,FKS_configs,vector_size))
   allocate(swgt_n1(FKS_configs,vector_size))
   ! Store storage for born-like collinears
    allocate(scb_ans_cnt(2,nsplitorders,FKS_configs,vector_size))
    allocate(scb_amp_split(amp_split_size,FKS_configs,vector_size))
    allocate(scb_amp_split_cnt(amp_split_size,2,nsplitorders,FKS_configs,vector_size))
    allocate(scb_saveamp(ngraphs,max_bhel,FKS_configs,vector_size))
   ! Store storage for born-like collinears
    allocate(sc1_ans_cnt(2,nsplitorders,FKS_configs,vector_size))
    allocate(sc1_amp_split(amp_split_size,FKS_configs,vector_size))
    allocate(sc1_amp_split_cnt(amp_split_size,2,nsplitorders,FKS_configs,vector_size))
    allocate(sc1_saveamp(ngraphs,max_bhel,FKS_configs,vector_size))
   ! Store storage for rotated Borns
    allocate(srot_jamp2(0:ncolor,FKS_configs,vector_size))
    allocate(srot_ans_cnt(2,nsplitorders,FKS_configs,vector_size))
    ! allocate(srot_amp_split(amp_split_size,FKS_configs,vector_size))
    allocate(srot_amp_split_cnt(amp_split_size,2,nsplitorders,FKS_configs,vector_size))
    ! allocate(srot_saveamp(ngraphs,max_bhel,FKS_configs,vector_size))
   ! norad and ev amplitudes
    allocate(snorad_amp2(ngraphs,FKS_configs,vector_size))
    allocate(sev_amp2(ngraphs,FKS_configs,vector_size))
   ! Born contributions storage
    allocate(sborn_amp2(ngraphs,vector_size))
    allocate(sborn_jamp2(0:ncolor,vector_size))
    allocate(sborn_ans_cnt(2,nsplitorders,vector_size))
    allocate(sborn_amp_split(amp_split_size,vector_size))
    allocate(sborn_amp_split_cnt(amp_split_size,2,nsplitorders,vector_size))
    allocate(sborn_saveamp(ngraphs,max_bhel,vector_size))
    allocate(sborn_split_soft(amp_split_size,vector_size))
    allocate(swgt_born(vector_size))
    ! Reals
    allocate(sreal_amp_split(amp_split_size,FKS_configs,vector_size))
    allocate(sfx_ev(FKS_configs,vector_size))
    ! Momenta
    allocate(spb(0:3,nexternal-1,0:FKS_configs,vector_size))
    allocate(spb_rot(0:3,nexternal-1,FKS_configs,vector_size))
    allocate(spb_coll(0:3,nexternal-1,FKS_configs,vector_size))
    allocate(spb_ev(0:3,nexternal-1,FKS_configs,vector_size))
    allocate(spb_norad(0:3,nexternal-1,FKS_configs,vector_size))
    allocate(sp1_cnt(0:3,nexternal,-2:2,0:FKS_configs,vector_size))
    allocate(sp1(0:3,nexternal,FKS_configs,vector_size))
    driver_is_allocated = .true.
end subroutine allocate_storage

subroutine reset_storage()
  implicit none
   include 'nFKSconfigs.inc'
   include 'born_nhel.inc'
   ! Born and virtual storage reset
   virt_wgt_vec(:,:) = 0d0
    born_wgt_vec(:,:) = 0d0
   ! n-body kinematics Borns
   snb_amp2(:,:,:) = 0d0
   snb_jamp2(:,:,:) = 0d0
   snb_ans_cnt(:,:,:,:) = (0d0,0d0)
   snb_amp_split(:,:,:) = 0d0
   snb_amp_split_cnt(:,:,:,:,:) = (0d0,0d0)
   snb_saveamp(:,:,:,:) = (0d0,0d0)
    snb_split_soft(:,:,:) = 0d0
   swgt_nb(:,:) = 0d0
   ! n+1-body kinematics Borns
   sn1_amp2(:,:,:) = 0d0
   sn1_jamp2(:,:,:) = 0d0
   sn1_ans_cnt(:,:,:,:) = (0d0,0d0)
   sn1_amp_split(:,:,:) = 0d0
   sn1_amp_split_cnt(:,:,:,:,:) = (0d0,0d0)
   sn1_saveamp(:,:,:,:) = (0d0,0d0)
    sn1_split_soft(:,:,:) = 0d0
   swgt_n1(:,:) = 0d0
   ! Store storage for born-like collinears
    scb_ans_cnt(:,:,:,:) = (0d0,0d0)
    scb_amp_split(:,:,:) = 0d0
    scb_amp_split_cnt(:,:,:,:,:) = (0d0,0d0)
    scb_saveamp(:,:,:,:) = (0d0,0d0)
   ! Store storage for born-like collinears
    sc1_ans_cnt(:,:,:,:) = (0d0,0d0)
    sc1_amp_split(:,:,:) = 0d0
    sc1_amp_split_cnt(:,:,:,:,:) = (0d0,0d0)
    sc1_saveamp(:,:,:,:) = (0d0,0d0)
   ! Store storage for rotated Borns
    srot_jamp2(:,:,:) = 0d0
    srot_ans_cnt(:,:,:,:) = (0d0,0d0)
    ! srot_amp_split(:,:,:) = 0d0
    srot_amp_split_cnt(:,:,:,:,:) = (0d0,0d0)
    ! srot_saveamp(:,:,:,:) = (0d0,0d0)
  ! norad and ev amplitudes
    snorad_amp2(:,:,:) = 0d0
    sev_amp2(:,:,:) = 0d0
   ! Born contributions storage
    sborn_amp2(:,:) = 0d0
    sborn_jamp2(:,:) = 0d0
    sborn_ans_cnt(:,:,:) = (0d0,0d0)
    sborn_amp_split(:,:) = 0d0
    sborn_amp_split_cnt(:,:,:,:) = (0d0,0d0)
    sborn_saveamp(:,:,:) = (0d0,0d0)
    sborn_split_soft(:,:) = 0d0
    swgt_born(:) = 0d0
! Real amplitudes and weights
    sreal_amp_split(:,:,:) = 0d0
    sfx_ev(:,:) = 0d0
    ! Momenta
    spb(:,:,:,:) = 0d0
    spb_rot(:,:,:,:) = 0d0
    spb_coll(:,:,:,:) = 0d0
    spb_ev(:,:,:,:) = 0d0
    spb_norad(:,:,:,:) = 0d0
    sp1_cnt(:,:,:,:,:) = 0d0
    sp1(:,:,:,:) = 0d0
end subroutine reset_storage

subroutine deallocate_storage()
    implicit none
    driver_is_allocated = .false.
    driver_vector_size = 0
    ! surrounding infrastructure variables
    if (allocated(MCcnt_vec)) deallocate(MCcnt_vec)
    if (allocated(vegas_wgt_vec)) deallocate(vegas_wgt_vec)
    if (allocated(x_mint_vec)) deallocate(x_mint_vec)
    if (allocated(x_vegas_vec)) deallocate(x_vegas_vec)
    if (allocated(x_save_vec)) deallocate(x_save_vec)
    if (allocated(f_vec)) deallocate(f_vec)
    if (allocated(pass_cuts_check_vec)) deallocate(pass_cuts_check_vec)
    if (allocated(passcuts_born_vec)) deallocate(passcuts_born_vec)
    if (allocated(passcuts_nbody_vec)) deallocate(passcuts_nbody_vec)
    if (allocated(passcuts_n1body_vec)) deallocate(passcuts_n1body_vec)
    if (allocated(virt_wgt_vec)) deallocate(virt_wgt_vec)
    if (allocated(born_wgt_vec)) deallocate(born_wgt_vec)
  ! born helicities
    if (allocated(nhel_vec)) deallocate(nhel_vec)
    ! n-body kinematics Borns
    if (allocated(snb_amp2)) deallocate(snb_amp2)
    if (allocated(snb_jamp2)) deallocate(snb_jamp2)
    if (allocated(snb_ans_cnt)) deallocate(snb_ans_cnt)
    if (allocated(snb_amp_split)) deallocate(snb_amp_split)
    if (allocated(snb_amp_split_cnt)) deallocate(snb_amp_split_cnt)
    if (allocated(snb_saveamp)) deallocate(snb_saveamp)
    if (allocated(snb_split_soft)) deallocate(snb_split_soft)
    if (allocated(swgt_nb)) deallocate(swgt_nb)
    ! n+1-body kinematics Borns
    if (allocated(sn1_amp2)) deallocate(sn1_amp2)
    if (allocated(sn1_jamp2)) deallocate(sn1_jamp2)
    if (allocated(sn1_ans_cnt)) deallocate(sn1_ans_cnt)
    if (allocated(sn1_amp_split)) deallocate(sn1_amp_split)
    if (allocated(sn1_amp_split_cnt)) deallocate(sn1_amp_split_cnt)
    if (allocated(sn1_saveamp)) deallocate(sn1_saveamp)
    if (allocated(sn1_split_soft)) deallocate(sn1_split_soft)
    if (allocated(swgt_n1)) deallocate(swgt_n1)
    ! Store storage for born-like collinears
    if (allocated(scb_ans_cnt)) deallocate(scb_ans_cnt)
    if (allocated(scb_amp_split)) deallocate(scb_amp_split)
    if (allocated(scb_amp_split_cnt)) deallocate(scb_amp_split_cnt)
    if (allocated(scb_saveamp)) deallocate(scb_saveamp)
    ! Store storage for born-like collinears
    if (allocated(sc1_ans_cnt)) deallocate(sc1_ans_cnt)
    if (allocated(sc1_amp_split)) deallocate(sc1_amp_split)
    if (allocated(sc1_amp_split_cnt)) deallocate(sc1_amp_split_cnt)
    if (allocated(sc1_saveamp)) deallocate(sc1_saveamp)
    ! Store storage for rotated Borns
    if (allocated(srot_jamp2)) deallocate(srot_jamp2)
    if (allocated(srot_ans_cnt)) deallocate(srot_ans_cnt)
    ! if (allocated(srot_amp_split)) deallocate(srot_amp_split)
    if (allocated(srot_amp_split_cnt)) deallocate(srot_amp_split_cnt)
    ! if (allocated(srot_saveamp)) deallocate(srot_saveamp)
    ! norad and ev amplitudes
    if (allocated(snorad_amp2)) deallocate(snorad_amp2)
    if (allocated(sev_amp2)) deallocate(sev_amp2)
    ! Born contributions storage
    if (allocated(sborn_amp2)) deallocate(sborn_amp2)
    if (allocated(sborn_jamp2)) deallocate(sborn_jamp2)
    if (allocated(sborn_ans_cnt)) deallocate(sborn_ans_cnt)
    if (allocated(sborn_amp_split)) deallocate(sborn_amp_split)
    if (allocated(sborn_amp_split_cnt)) deallocate(sborn_amp_split_cnt)
    if (allocated(sborn_saveamp)) deallocate(sborn_saveamp)
    if (allocated(sborn_split_soft)) deallocate(sborn_split_soft)
    if (allocated(swgt_born)) deallocate(swgt_born)
! Real amplitudes and weights
    if (allocated(sreal_amp_split)) deallocate(sreal_amp_split)
    if (allocated(sfx_ev)) deallocate(sfx_ev)
    ! Momenta
    if (allocated(spb)) deallocate(spb)
    if (allocated(spb_rot)) deallocate(spb_rot)
    if (allocated(spb_coll)) deallocate(spb_coll)
    if (allocated(spb_ev)) deallocate(spb_ev)
    if (allocated(spb_norad)) deallocate(spb_norad)
    if (allocated(sp1_cnt)) deallocate(sp1_cnt)
    if (allocated(sp1)) deallocate(sp1)
end subroutine deallocate_storage

end module driver_vec