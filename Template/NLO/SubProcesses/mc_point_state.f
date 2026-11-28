      module mc_point_state
c Point-owned data at the native full-real boundary. Kinematics use the
c FKS frame on entry; the chosen real provider applies the ME frame.
      implicit none
      private
      include 'nexternal.inc'
      include 'orders.inc'
      integer, parameter, public :: real_idle=0,real_waiting=1,
     $     real_ready=2
      type, public :: native_real_request
         integer :: status=real_idle
         double precision :: p(0:3,nexternal),xi,y
         double precision :: total,components(amp_split_size)
      end type
c Decisions whose lifetime spans the folds of a single MINT draw.
      type, public :: native_fold_sample
         integer :: group=0,born_flow=0,fold_number=0
         double precision :: group_probability=0d0,flow_probability=0d0
         integer,allocatable :: partner(:)
         logical :: rejected=.false.
      end type
      type, public :: native_flow_cursor
         integer :: stage=0,flow=0,flow_owner=0,called_owner=0
         double precision :: probability,flow_probability,flow_fraction,
     $        factors(9),mc_factors(8),pdf_factors(3),gfun(3),sumborn
         double precision,allocatable :: flow_weights(:),emsca_base(:,:),
     $        emsca_owner(:,:)
         logical,allocatable :: leading(:)
         logical :: dipoles(nexternal,nexternal)
         double precision :: scales(nexternal,nexternal)
         integer :: colours_s(2,nexternal-1),colours_h(2,nexternal)
      end type
      type, public :: native_real_emission
         type(native_real_request) request
         double precision :: sector_fraction,sudakov
      end type
      public :: request_native_real,complete_native_real,
     $     consume_native_real
      contains
      subroutine request_native_real(request,p,xi,y)
      type(native_real_request),intent(inout) :: request
      double precision,intent(in) :: p(0:3,nexternal),xi,y
      if(request%status.ne.real_idle) error stop
     $     'A native real request is already outstanding'
      request%p=p
      request%xi=xi
      request%y=y
      request%status=real_waiting
      end subroutine

      subroutine complete_native_real(request,total,components)
      type(native_real_request),intent(inout) :: request
      double precision,intent(in) :: total,components(amp_split_size)
      if(request%status.ne.real_waiting) error stop
     $     'Unexpected native real completion'
      request%total=total
      request%components=components
      request%status=real_ready
      end subroutine

      subroutine consume_native_real(request,wgt,components)
      type(native_real_request),intent(inout) :: request
      double precision,intent(out) :: wgt,components(amp_split_size)
      if(request%status.ne.real_ready) error stop
     $     'Native real result is unavailable or was already consumed'
c Preserve the scalar evaluation order and signed split-order components.
      wgt=request%total*request%xi**2*(1d0-request%y)
      components=request%components*request%xi**2*(1d0-request%y)
      request%status=real_idle
      end subroutine
      end module
