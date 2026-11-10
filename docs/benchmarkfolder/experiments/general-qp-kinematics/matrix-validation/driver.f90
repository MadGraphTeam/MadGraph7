program matrix_policy_check
  use iso_fortran_env, only: real128, int64
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  integer :: u, ios, count, c, which, g, ct, n, i, j, expectation, massmode, internalmode, inject, k, has_golden
  real(real128), allocatable :: p(:,:)
  complex(real128), allocatable :: masses(:), legacy(:,:), compat(:,:), computed(:,:), selected(:,:)
  real(real128) :: reference, bound, target, invariant, golden
  complex(real128) :: internal, expected
  logical :: h0(3), h1(3)
  integer(int64) :: words(4)
  character(len=4096) :: path
  call get_command_argument(1,path)
  open(newunit=u,file=trim(path),status='old',action='read',iostat=ios)
  if(ios/=0) error stop 1
  read(u,*) count
  call ieee_get_halting_mode(ieee_invalid,h0(1))
  call ieee_get_halting_mode(ieee_divide_by_zero,h0(2))
  call ieee_get_halting_mode(ieee_overflow,h0(3))
  do c=1,count
    read(u,*) which,g,ct,n,i,j,expectation,massmode,internalmode,inject,has_golden
    allocate(p(n,0:3),masses(n),legacy(n,n),compat(n,n),computed(n,n),selected(n,n))
    read(u,*) p
    read(u,*) reference,bound,golden
    call INIT_CONTEXT(massmode,internal)
    do k=1,n
      masses(k)=internal+cmplx(k,0,kind=real128)
    end do
    if(internalmode==0) masses=0
    if(inject==1) then
      ! Pair i,j stays finite, but an unrelated shift is invalid.
      if(i==1.or.j==1) error stop 2
      p(1,0)=ieee_value(0.0_real128,ieee_quiet_nan)
    end if
    call KC_MP_BUILD_KINEMATIC_MATRIX(n,p,masses,legacy)
    if(which==1) then
      call KC_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,0,n,p,masses,compat)
      call KC_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,2,n,p,masses,computed)
      call KC_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,1,n,p,masses,selected)
    else
      call OFF_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,0,n,p,masses,compat)
      call OFF_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,2,n,p,masses,computed)
      call OFF_MP_BUILD_KINEMATIC_MATRIX_KIN(g,ct,1,n,p,masses,selected)
    end if
    if(any(transfer(legacy,[0_int64],4*n*n)/=transfer(compat,[0_int64],4*n*n))) then
      write(*,*) 'FAIL_COMPAT',c
      error stop 3
    end if
    invariant=real(computed(i,j)+masses(i)+masses(j),kind=real128)
    ! Independent rational Minkowski reference allows finite operation and
    ! internal-mass subtraction/addition rounding, not a wrong shell target.
    if(.not.ieee_is_finite(invariant)) error stop 4
    if(abs(invariant-reference)>bound+64*epsilon(0.0_real128)*max(1.0_real128,abs(masses(i)),abs(masses(j)))) then
      write(*,*) 'FAIL_COMPUTED',c,invariant,reference,bound
      error stop 5
    end if
    if(has_golden==1) then
      if(golden==0.0_real128) error stop 9
      if(any(transfer(invariant,[0_int64],2)/=transfer(golden,[0_int64],2))) then
        write(*,*) 'FAIL_RETAINED_COMPUTED_WORDS',c
        error stop 10
      end if
    end if
    select case(expectation)
    case(0)
      expected=computed(i,j)
    case(1)
      expected=cmplx(0.0_real128,0.0_real128,kind=real128)-masses(i)-masses(j)
    case(2)
      expected=cmplx(4.0_real128,0.0_real128,kind=real128)-masses(i)-masses(j)
    case(3)
      target=4.0_real128+2.0_real128**(-109)
      expected=cmplx(target,0.0_real128,kind=real128)-masses(i)-masses(j)
    case default
      error stop 6
    end select
    if(any(transfer(selected(i,j),[0_int64],4)/=transfer(expected,[0_int64],4))) then
      write(*,*) 'FAIL_POLICY',c,g,ct,i,j,expectation
      write(*,*) 'ACTUAL_EXPECTED',selected(i,j),expected
      error stop 7
    end if
    call ieee_get_halting_mode(ieee_invalid,h1(1))
    call ieee_get_halting_mode(ieee_divide_by_zero,h1(2))
    call ieee_get_halting_mode(ieee_overflow,h1(3))
    if(any(h0.neqv.h1)) error stop 8
    words=transfer(selected(i,j),words)
    write(*,'(A,8(1X,I0),4(1X,Z16.16))') 'CASE_PASS',c,which,g,ct,n,i,j,expectation,words
    deallocate(p,masses,legacy,compat,computed,selected)
  end do
  close(u)
  write(*,'(A,1X,I0)') 'MATRIX_POLICY_PASS',count
end program
