! Adapted from NinjaRS validation/rank-n2/harness/collier_tensor_driver.f90.
! Separate poles use independent DeltaUV/DeltaIR shifts, matching MadLoop.
program collier_tensor_driver
  use COLLIER
  implicit none
  integer :: njobs, nmax, rmax, ijob, N, R, k, iset, n0, n1, n2, n3, errflag, accflag
  integer :: errmin, accmax, mode, nargs, ios, unit_in, unit_out
  double precision :: mu2, re, im, reqacc, p(0:3), pi2o6
  double precision :: d_uv(4), d_ir1(4), d_ir2(4)
  character(len=4096) :: path_in, path_out, arg
  double complex, allocatable :: mass2(:), MomInv(:), MomVec(:,:)
  double complex, allocatable :: TN(:,:,:,:), TNuv(:,:,:,:), TNs(:,:,:,:,:), TNuv1(:,:,:,:)
  double precision, allocatable :: TNerr(:), TNerr1(:)

  pi2o6 = 4d0*atan(1d0)
  pi2o6 = pi2o6*pi2o6/6d0
  d_uv  = (/ 0d0, 1d0, 0d0, 0d0 /)
  d_ir1 = (/ 0d0, 0d0, 1d0, 0d0 /)
  d_ir2 = (/ pi2o6, pi2o6, pi2o6, 1d0 + pi2o6 /)

  nargs = command_argument_count()
  if (nargs < 2) then
    write(0,*) 'usage: collier_tensor_driver JOBS.in OUT.txt [MODE] [REQACC]'
    stop 2
  end if
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  mode = 1
  if (nargs >= 3) then
    call get_command_argument(3, arg)
    read(arg, *) mode
  end if
  reqacc = -1d0
  if (nargs >= 4) then
    call get_command_argument(4, arg)
    read(arg, *) reqacc
  end if

  open(newunit=unit_in, file=trim(path_in), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(0,*) 'cannot open ', trim(path_in)
    stop 2
  end if
  read(unit_in, *) njobs, nmax, rmax

  call Init_cll(max(nmax, 2), max(rmax, 1), '')
  call SetMode_cll(mode)
  if (reqacc > 0d0) call SetReqAcc_cll(reqacc)

  open(newunit=unit_out, file=trim(path_out), status='replace', action='write')

  do ijob = 1, njobs
    read(unit_in, *) N, R
    read(unit_in, *) mu2
    allocate(mass2(0:max(N-1,1)), MomVec(0:3, max(N-1,1)), MomInv(max(N*(N-1)/2,1)))
    mass2 = (0d0, 0d0)
    MomVec = (0d0, 0d0)
    MomInv = (0d0, 0d0)
    do k = 0, N-1
      read(unit_in, *) re, im
      mass2(k) = dcmplx(re, im)
    end do
    do k = 1, N-1
      read(unit_in, *) p
      MomVec(:, k) = dcmplx(p, 0d0)
    end do
    do k = 1, N*(N-1)/2
      read(unit_in, *) re
      MomInv(k) = dcmplx(re, 0d0)
    end do

    allocate(TN(0:R,0:R,0:R,0:R), TNuv(0:R,0:R,0:R,0:R), TNs(0:R,0:R,0:R,0:R,4), &
             TNuv1(0:R,0:R,0:R,0:R), TNerr(0:R), TNerr1(0:R))
    call SetMuUV2_cll(mu2)
    call SetMuIR2_cll(mu2)
    errmin = 0
    accmax = 0
    do iset = 1, 4
      call SetDeltaUV_cll(d_uv(iset))
      call SetDeltaIR_cll(d_ir1(iset), d_ir2(iset))
      call InitErrFlag_cll()
      call InitAccFlag_cll()
      TN = (0d0, 0d0)
      TNuv = (0d0, 0d0)
      if (N == 1) then
        call Aten_cll(TN, TNuv, mass2(0:0), R, TNerr)
      else
        call TNten_cll(TN, TNuv, MomVec(:, 1:N-1), MomInv(1:N*(N-1)/2), mass2(0:N-1), N, R, TNerr)
      end if
      call GetErrFlag_cll(errflag)
      call GetAccFlag_cll(accflag)
      errmin = min(errmin, errflag)
      accmax = min(accmax, accflag)
      TNs(:,:,:,:,iset) = TN
      if (iset == 1) then
        TNuv1 = TNuv
        TNerr1 = TNerr
      end if
    end do

    write(unit_out, '(A,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0)') 'JOB', ijob, N, R, errmin, accmax
    do k = 0, R
      write(unit_out, '(A,1X,I0,1X,ES25.16E3)') 'ERR', k, TNerr1(k)
    end do
    do n0 = 0, R
      do n1 = 0, R - n0
        do n2 = 0, R - n0 - n1
          do n3 = 0, R - n0 - n1 - n2
            write(unit_out, '(4(I0,1X),8(ES25.16E3,1X))') n0, n1, n2, n3, &
              TNs(n0,n1,n2,n3,1), TNs(n0,n1,n2,n3,2) - TNs(n0,n1,n2,n3,1), &
              TNs(n0,n1,n2,n3,3) - TNs(n0,n1,n2,n3,1), TNs(n0,n1,n2,n3,4) - TNs(n0,n1,n2,n3,1)
          end do
        end do
      end do
    end do
    deallocate(mass2, MomVec, MomInv, TN, TNuv, TNs, TNuv1, TNerr, TNerr1)
  end do
  close(unit_out)
  close(unit_in)
end program collier_tensor_driver
