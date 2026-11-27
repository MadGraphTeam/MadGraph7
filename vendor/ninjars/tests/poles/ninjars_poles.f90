! Focused independent comparison against COLLIER tensor components.
program ninjars_poles
  use mninjars
  implicit none
  integer :: jobs, job, n, rank, count, k, mode, status, unit_in, unit_out, argc
  integer :: modes(8) = [0, 1, 2, 3, 0, 2, 1, 3]
  real(KI_QNRS) :: re, im, scale
  complex(KI_QNRS), allocatable :: coeff(:), mass(:)
  real(KI_QNRS), allocatable :: momenta(:,:), smatrix(:,:)
  complex(KI_QNRS) :: result(0:2), rational
  complex(KI_NRS) :: dpresult(0:2), dprational
  character(len=4096) :: input_path, output_path, precision
  character(len=1024) :: error_message
  argc = command_argument_count()
  if (argc /= 3) stop 2
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  call get_command_argument(3, precision)
  open(newunit=unit_in, file=trim(input_path), status='old')
  open(newunit=unit_out, file=trim(output_path), status='replace')
  read(unit_in, *) jobs
  do job=1,jobs
    read(unit_in, *) n, rank
    count=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24
    allocate(coeff(count),mass(n),momenta(0:3,n),smatrix(n,n))
    do k=1,count
      read(unit_in, *) re, im
      coeff(k)=cmplx(re,im,kind=KI_QNRS)
    end do
    do k=1,n
      read(unit_in, *) re, im
      mass(k)=cmplx(re,im,kind=KI_QNRS)
    end do
    do k=1,n
      read(unit_in, *) momenta(:,k)
    end do
    do k=1,n
      read(unit_in, *) smatrix(:,k)
    end do
    read(unit_in, *) scale
    do k=1,size(modes)
      mode=modes(k)
      if (trim(precision)=='mp') then
        call ninjars_tensor_evaluate(coeff,n,rank,smatrix,momenta,mass,scale,result,rational,status,mode)
      else
        call ninjars_tensor_evaluate(cmplx(coeff,kind=KI_NRS),n,rank,real(smatrix,kind=KI_NRS), &
          real(momenta,kind=KI_NRS),cmplx(mass,kind=KI_NRS),real(scale,kind=KI_NRS),dpresult,dprational,status,mode)
        result=cmplx(dpresult,kind=KI_QNRS)
        rational=cmplx(dprational,kind=KI_QNRS)
      end if
      write(unit_out,'(3(I0,1X),8(ES44.35E3,1X))') job,mode,status,result,rational
      flush(unit_out)
      if(status /= NINJARS_SUCCESS) then
        call ninjars_error_message(error_message)
        print *, job,mode,status,trim(error_message)
      end if
    end do
    deallocate(coeff,mass,momenta,smatrix)
  end do
  call ninjars_finalize(status)
  if(status /= NINJARS_SUCCESS) stop 3
  close(unit_in)
  close(unit_out)
end program
