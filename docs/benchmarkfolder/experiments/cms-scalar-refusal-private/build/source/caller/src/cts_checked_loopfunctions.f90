! Extracted from the pinned real loop-function routines; checked scaloop=2 only.
module cts_checked_loopfunctions
  include 'cts_mprec.h'
  use denominators
  use scale
  use dimensions
  use maxnumden, only: maxden
  use loopfunctions, only: aloopfun,bloopfun,b1loopfun,b11loopfun,cloopfun,dloopfun
  use mp_loopfunctions, only: mp_aloopfun,mp_bloopfun,mp_b1loopfun,mp_b11loopfun,mp_cloopfun,mp_dloopfun
  use cms_scalar_observer
  use avh_olo_forCutTools_dp, only: dp_checked_a0=>cms_checked_a0_c,dp_checked_b11=>cms_checked_b11rc, &
       dp_checked_c0=>cms_checked_c0rc,dp_checked_d0=>cms_checked_d0rc
  use avh_olo_forCutTools_qp, only: qp_checked_a0=>cms_checked_a0_c,qp_checked_b11=>cms_checked_b11rc, &
       qp_checked_c0=>cms_checked_c0rc,qp_checked_d0=>cms_checked_d0rc
  implicit none
  private
  public :: getloop_checked,get_mp_loop_checked,cts_checked_reset,cms_report_accepted
  public :: checked_dp_valid,checked_qp_valid
  logical,save :: checked_dp_valid=.false.,checked_qp_valid=.false.
contains
  subroutine cts_checked_reset()
    checked_dp_valid=.false.
    checked_qp_valid=.false.
  end subroutine
  logical function cms_report_accepted(report)
    type(cms_scalar_report),intent(in) :: report
    cms_report_accepted=report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called
  end function
  subroutine getloop_checked(nprop,direction,report)
    integer,intent(in) :: nprop,direction
    type(cms_scalar_report),intent(out) :: report
    report=cms_scalar_report()
    report%precision=8
    report%direction=direction
    checked_dp_valid=.false.
    if(scaloop.ne.2.or.nprop.lt.1.or.nprop.gt.maxden.or.abs(direction).ne.1) then
      report%status=CMS_OVERLOAD_UNSUPPORTED
      return
    endif
    if(nprop.ge.1) then
      if(dmns_1.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(aloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(aloopfun,2).lt.dmns_1) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.2) then
      if(dmns_2.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(bloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(bloopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(b1loopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(b1loopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(b11loopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(b11loopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.3) then
      if(dmns_3.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(cloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(cloopfun,2).lt.dmns_3) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.4) then
      if(dmns_4.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(dloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(dloopfun,2).lt.dmns_4) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.1) then
      call getaloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.2) then
      call getbloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.3) then
      call getcloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.4) then
      call getdloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    checked_dp_valid=.true.
  end subroutine getloop_checked
subroutine getaloop_checked(direction,report)
!
!  A 1-loop scalar function
!
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12
   include 'cts_dpc.h' 
    :: cm12
   include 'cts_dpc.h' 
    :: a0,qlI1
   do ib= 1,dmns_1
     cm12= den(bbn1(1,ib))%m2
     call dp_checked_a0(value,cm12,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     aloopfun(:,ib)= value(:) 
   enddo
  end subroutine getaloop_checked
subroutine getbloop_checked(direction,report)
!
!  B,B1 and B11 1-loop scalar functions
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib
   include 'cts_dpr.h' 
    :: k12
   include 'cts_dpr.h'
    , dimension(0:3) :: k1
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpc.h'
    , dimension(0:2) :: valb11,valb00,valb1,valb0
   include 'cts_dpr.h'    
    :: m12,m22
   include 'cts_dpc.h' 
    :: cm12,cm22
   include 'cts_dpc.h' 
    :: b0,b1,b11,qlI2
   do ib= 1,dmns_2
    k1= den(bbn2(2,ib))%p-den(bbn2(1,ib))%p
    call contr(k1,k1,k12)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
     cm12= den(bbn2(1,ib))%m2
     cm22= den(bbn2(2,ib))%m2
     call dp_checked_b11(valb11,valb00,valb1,valb0,k12,cm12,cm22,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     bloopfun(:,ib)  =  valb0(:)
     b1loopfun(:,ib) =  valb1(:)
     b11loopfun(:,ib)=  valb11(:)
   enddo
  end subroutine getbloop_checked
subroutine getcloop_checked(direction,report)
!
!  C 1-loop scalar function
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib
   include 'cts_dpr.h' 
    :: k12,k22,k32
   include 'cts_dpr.h'
    , dimension(0:3) :: k1,k2,k3
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12,m22,m32
   include 'cts_dpc.h' 
    :: cm12,cm22,cm32
   include 'cts_dpc.h' 
    :: c0,qlI3
   do ib= 1,dmns_3
    k1= den(bbn3(2,ib))%p-den(bbn3(1,ib))%p
    k2= den(bbn3(3,ib))%p-den(bbn3(2,ib))%p
    k3= den(bbn3(3,ib))%p-den(bbn3(1,ib))%p
    call contr(k1,k1,k12)
    call contr(k2,k2,k22)
    call contr(k3,k3,k32)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8) k22= 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8) k32= 0.d0
     cm12= den(bbn3(1,ib))%m2
     cm22= den(bbn3(2,ib))%m2
     cm32= den(bbn3(3,ib))%m2
     call dp_checked_c0(value,k12,k22,k32,cm12,cm22,cm32,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     cloopfun(:,ib)  =        value(:) 
   enddo
  end subroutine getcloop_checked
subroutine getdloop_checked(direction,report)
!
!  D 1-loop scalar function
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib
   include 'cts_dpr.h' 
    :: k12,k22,k32,k42,k122,k232
   include 'cts_dpr.h'
    , dimension(0:3) :: k1,k2,k3,k4,p12,p23
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12,m22,m32,m42
   include 'cts_dpc.h' 
    :: cm12,cm22,cm32,cm42
   include 'cts_dpc.h' 
    :: d0,qlI4
   do ib= 1,dmns_4
    k1 = den(bbn4(2,ib))%p-den(bbn4(1,ib))%p
    k2 = den(bbn4(3,ib))%p-den(bbn4(2,ib))%p
    k3 = den(bbn4(4,ib))%p-den(bbn4(3,ib))%p
    k4 = den(bbn4(4,ib))%p-den(bbn4(1,ib))%p
    p12= den(bbn4(3,ib))%p-den(bbn4(1,ib))%p
    p23= den(bbn4(4,ib))%p-den(bbn4(2,ib))%p
    call contr(k1 ,k1 ,k12)
    call contr(k2 ,k2 ,k22)
    call contr(k3 ,k3 ,k32)
    call contr(k4 ,k4 ,k42)
    call contr(p12,p12,k122)
    call contr(p23,p23,k232)
!!    if (dabs(k12/roots/roots).lt.1.d-8)  k12 = 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8)  k22 = 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8)  k32 = 0.d0
!!    if (dabs(k42/roots/roots).lt.1.d-8)  k42 = 0.d0
!!    if (dabs(k122/roots/roots).lt.1.d-8) k122= 0.d0
!!    if (dabs(k232/roots/roots).lt.1.d-8) k232= 0.d0
     cm12= den(bbn4(1,ib))%m2
     cm22= den(bbn4(2,ib))%m2
     cm32= den(bbn4(3,ib))%m2
     cm42= den(bbn4(4,ib))%m2
     call dp_checked_d0(value,k12,k22,k32,k42,k122,k232,cm12,cm22,cm32,cm42,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     dloopfun(:,ib)  =        value(:) 
   enddo
  end subroutine getdloop_checked
  subroutine get_mp_loop_checked(nprop,direction,report)
    integer,intent(in) :: nprop,direction
    type(cms_scalar_report),intent(out) :: report
    report=cms_scalar_report()
    report%precision=16
    report%direction=direction
    checked_qp_valid=.false.
    if(scaloop.ne.2.or.nprop.lt.1.or.nprop.gt.maxden.or.abs(direction).ne.1) then
      report%status=CMS_OVERLOAD_UNSUPPORTED
      return
    endif
    if(nprop.ge.1) then
      if(dmns_1.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_aloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_aloopfun,2).lt.dmns_1) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.2) then
      if(dmns_2.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_bloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_bloopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_b1loopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_b1loopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_b11loopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_b11loopfun,2).lt.dmns_2) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.3) then
      if(dmns_3.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_cloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_cloopfun,2).lt.dmns_3) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.4) then
      if(dmns_4.lt.1) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(.not.allocated(mp_dloopfun)) then
        report%status=CMS_BAD_INPUT
        return
      endif
      if(size(mp_dloopfun,2).lt.dmns_4) then
        report%status=CMS_BAD_INPUT
        return
      endif
    endif
    if(nprop.ge.1) then
      call get_mp_aloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.2) then
      call get_mp_bloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.3) then
      call get_mp_cloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    if(nprop.ge.4) then
      call get_mp_dloop_checked(direction,report)
      if(.not.cms_report_accepted(report)) return
    endif
    checked_qp_valid=.true.
  end subroutine get_mp_loop_checked
subroutine get_mp_aloop_checked(direction,report)
!
!  A 1-loop scalar function
!
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib,ic
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12
   include 'cts_mpc.h' 
    :: cm12
   include 'cts_dpc.h' 
    :: qlI1
   include 'cts_dpr.h'    
    :: dm12
   do ib= 1,dmns_1
     cm12= mp_den(bbn1(1,ib))%m2
     call qp_checked_a0(value,cm12,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     mp_aloopfun(:,ib)= value(:) 
   enddo
  end subroutine get_mp_aloop_checked
subroutine get_mp_bloop_checked(direction,report)
!
!  B,B1 and B11 1-loop scalar functions
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib,ic
   include 'cts_mpr.h' 
    :: k12
   include 'cts_mpr.h'
    , dimension(0:3) :: k1
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpc.h'
    , dimension(0:2) :: valb11,valb00,valb1,valb0
   include 'cts_mpr.h'    
    :: m12,m22
   include 'cts_mpc.h' 
    :: cm12,cm22
   include 'cts_dpc.h' 
    :: qlI2
   include 'cts_dpc.h'
    , dimension(0:2) :: dvalb11,dvalb00,dvalb1,dvalb0
   include 'cts_dpr.h'    
    :: dm12,dm22
   include 'cts_dpr.h' 
    :: dk12
   do ib= 1,dmns_2
    do ic= 0,3
     k1(ic)= mp_den(bbn2(2,ib))%p(ic) &
            -mp_den(bbn2(1,ib))%p(ic)
    enddo
    call contr(k1,k1,k12)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
     cm12= mp_den(bbn2(1,ib))%m2
     cm22= mp_den(bbn2(2,ib))%m2
     call qp_checked_b11(valb11,valb00,valb1,valb0,k12,cm12,cm22,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     mp_bloopfun(:,ib)  =  valb0(:)
     mp_b1loopfun(:,ib) =  valb1(:)
     mp_b11loopfun(:,ib)=  valb11(:)
   enddo
  end subroutine get_mp_bloop_checked
subroutine get_mp_cloop_checked(direction,report)
!
!  C 1-loop scalar function
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib,ic
   include 'cts_mpr.h' 
    :: k12,k22,k32
   include 'cts_mpr.h'
    , dimension(0:3) :: k1,k2,k3
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12,m22,m32
   include 'cts_mpc.h' 
    :: cm12,cm22,cm32
   include 'cts_dpc.h' 
    :: qlI3
   include 'cts_dpr.h'    
    :: dm12,dm22,dm32
   include 'cts_dpr.h' 
    :: dk12,dk22,dk32
   do ib= 1,dmns_3
    do ic= 0,3
     k1(ic)= mp_den(bbn3(2,ib))%p(ic) &
            -mp_den(bbn3(1,ib))%p(ic)
     k2(ic)= mp_den(bbn3(3,ib))%p(ic) &
            -mp_den(bbn3(2,ib))%p(ic)
     k3(ic)= mp_den(bbn3(3,ib))%p(ic) &
            -mp_den(bbn3(1,ib))%p(ic)
    enddo
    call contr(k1,k1,k12)
    call contr(k2,k2,k22)
    call contr(k3,k3,k32)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8) k22= 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8) k32= 0.d0
     cm12= mp_den(bbn3(1,ib))%m2
     cm22= mp_den(bbn3(2,ib))%m2
     cm32= mp_den(bbn3(3,ib))%m2
     call qp_checked_c0(value,k12,k22,k32,cm12,cm22,cm32,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     mp_cloopfun(:,ib)  =        value(:) 
   enddo
  end subroutine get_mp_cloop_checked
subroutine get_mp_dloop_checked(direction,report)
!
!  D 1-loop scalar function
!
   use tensor_operations  
   integer,intent(in) :: direction
   type(cms_scalar_report),intent(out) :: report
   integer :: ib,ic
   include 'cts_mpr.h' 
    :: k12,k22,k32,k42,k122,k232
   include 'cts_mpr.h'
    , dimension(0:3) :: k1,k2,k3,k4,p12,p23
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12,m22,m32,m42
   include 'cts_mpc.h' 
    :: cm12,cm22,cm32,cm42
   include 'cts_dpc.h' 
    :: qlI4
   include 'cts_dpr.h'    
    :: dm12,dm22,dm32,dm42
   include 'cts_dpr.h' 
    :: dk12,dk22,dk32,dk42,dk122,dk232
   do ib= 1,dmns_4
    do ic= 0,3
     k1(ic) = mp_den(bbn4(2,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     k2(ic) = mp_den(bbn4(3,ib))%p(ic) &
             -mp_den(bbn4(2,ib))%p(ic)
     k3(ic) = mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(3,ib))%p(ic)
     k4(ic) = mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     p12(ic)= mp_den(bbn4(3,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     p23(ic)= mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(2,ib))%p(ic)
    enddo
    call contr(k1 ,k1 ,k12)
    call contr(k2 ,k2 ,k22)
    call contr(k3 ,k3 ,k32)
    call contr(k4 ,k4 ,k42)
    call contr(p12,p12,k122)
    call contr(p23,p23,k232)
!!    if (dabs(k12/roots/roots).lt.1.d-8)  k12 = 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8)  k22 = 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8)  k32 = 0.d0
!!    if (dabs(k42/roots/roots).lt.1.d-8)  k42 = 0.d0
!!    if (dabs(k122/roots/roots).lt.1.d-8) k122= 0.d0
!!    if (dabs(k232/roots/roots).lt.1.d-8) k232= 0.d0
     cm12= mp_den(bbn4(1,ib))%m2
     cm22= mp_den(bbn4(2,ib))%m2
     cm32= mp_den(bbn4(3,ib))%m2
     cm42= mp_den(bbn4(4,ib))%m2
     call qp_checked_d0(value,k12,k22,k32,k42,k122,k232,cm12,cm22,cm32,cm42,report)
     report%direction=direction
     report%subtopology=ib
     if (.not.cms_report_accepted(report)) then
      if (report%status.eq.CMS_OK) report%status=CMS_BAD_INPUT
      report%result_valid=.false.
      return
     endif
     mp_dloopfun(:,ib)  =        value(:) 
   enddo
  end subroutine get_mp_dloop_checked
end module cts_checked_loopfunctions
