! PRIVATE STANDALONE QUALIFICATION SOURCE. No automatic/native execution.
! All input/output occurs with the saved original environment restored.
program qualify_exact_geometry
  use, intrinsic :: iso_c_binding, only: c_int,c_int64_t
  use, intrinsic :: iso_fortran_env, only: input_unit,output_unit
  use ml_private_exact_geometry, only: geometry_ik,exact_geometry_difference
  implicit none
  interface
    function qualifier_pid() result(pid) bind(C,name='ml_geom_qual_pid')
      import c_int
      integer(c_int) :: pid
    end function
    subroutine qualifier_constants(values) bind(C,name='ml_geom_qual_constants')
      import c_int64_t
      integer(c_int64_t), intent(out) :: values(10)
    end subroutine
    function qualifier_observe(values) result(status) bind(C,name='ml_geom_qual_observe')
      import c_int,c_int64_t
      integer(c_int64_t), intent(out) :: values(7)
      integer(c_int) :: status
    end function
    function qualifier_begin(mode,requests,baseline) result(status) bind(C,name='ml_geom_qual_begin')
      import c_int,c_int64_t
      integer(c_int), value :: mode
      integer(c_int64_t), intent(out) :: requests(5),baseline(7)
      integer(c_int) :: status
    end function
    function qualifier_restore(restored) result(status) bind(C,name='ml_geom_qual_restore')
      import c_int,c_int64_t
      integer(c_int64_t), intent(out) :: restored(7)
      integer(c_int) :: status
    end function
    subroutine qualifier_raw_restore_failure(id,mode,status,restore_result) &
        bind(C,name='ml_geom_qual_raw_restore_failure')
      import c_int
      integer(c_int), value :: id,mode,status,restore_result
    end subroutine
  end interface
  integer, parameter :: storage_external=65,mode_count=20,max_cases=10000
  integer(geometry_ik) :: external_bits(2,0:3,storage_external),q_bits(2,0:3)
  integer :: coefficients(storage_external)
  integer(c_int64_t) :: constants(10),requests(5),baseline(7),before(7),after(7),restored(7)
  integer(c_int) :: status,observed_before,observed_after,restore_ok
  integer :: iu,ou,ios,ncases,case_number,id,width,reference_width,n,shape,leg,mu,mode,accepted_bit
  logical :: accepted
  character(len=4096) :: fixture_path,observation_path,line
  character(len=32) :: command

  if(command_argument_count()/=2) error stop 'Expected fixtures.hex observations.txt'
  call get_command_argument(1,fixture_path,status=ios)
  if(ios/=0) error stop 'Invalid fixture path'
  call get_command_argument(2,observation_path,status=ios)
  if(ios/=0) error stop 'Invalid observation path'
  write(output_unit,'(A,1X,I0)') 'READY',qualifier_pid()
  flush(output_unit)
  read(input_unit,'(A)',iostat=ios) command
  if(ios/=0.or.trim(command)/='GO') error stop 'GO handshake required'
  open(newunit=iu,file=trim(fixture_path),status='old',action='read',iostat=ios)
  if(ios/=0) error stop 'Cannot open fixtures'
  open(newunit=ou,file=trim(observation_path),status='new',action='write',iostat=ios)
  if(ios/=0) error stop 'Cannot create fresh observations'
  read(iu,*,iostat=ios) ncases
  if(ios/=0) error stop 'Missing fixture count'
  if(ncases<1.or.ncases>max_cases) error stop 'Fixture count out of bounds'
  call qualifier_constants(constants)
  write(ou,'(A)') 'SCHEMA exact-geometry-fenv-v1'
  write(ou,'(A,*(1X,I0))') 'CONSTANTS',constants
  write(ou,'(A,1X,I0)') 'NCASES',ncases
  write(ou,'(A,1X,I0)') 'MODES',mode_count
  write(ou,'(A)') 'SNAPSHOT_FIELDS round sticky halts mxcsr_control x87cw mxcsr_raw x87sw'
  write(ou,'(A)') 'CASE_FIELDS id mode status accepted restore_ok request_round request_sticky request_halts '// &
      'request_ftz request_daz baseline(7) before(7) after(7) restored(7)'
  do case_number=1,ncases
    external_bits=0_geometry_ik;q_bits=0_geometry_ik;coefficients=0
    read(iu,*,iostat=ios) id,width,reference_width,n,shape
    if(ios/=0) error stop 'Malformed fixture header'
    if(n<0.or.n>storage_external.or.shape<0.or.shape>5) error stop 'Fixture storage shape out of bounds'
    ! Consume exactly one coefficient line even when N=0.
    read(iu,'(A)',iostat=ios) line
    if(ios/=0) error stop 'Missing coefficient line'
    if(n>0) then
      read(line,*,iostat=ios) coefficients(:n)
      if(ios/=0) error stop 'Malformed coefficients'
    endif
    do leg=1,n
      do mu=0,3
        read(iu,'(Z32,1X,Z32)',iostat=ios) external_bits(:,mu,leg)
        if(ios/=0) error stop 'Malformed external word pair'
      enddo
    enddo
    do mu=0,3
      read(iu,'(Z32,1X,Z32)',iostat=ios) q_bits(:,mu)
      if(ios/=0) error stop 'Malformed Q word pair'
    enddo
    write(ou,'(A,5(1X,I0))') 'INPUT',id,width,reference_width,n,shape
    write(ou,'(A,1X,I0,*(1X,I0))') 'COEFF',id,coefficients(:n)
    do leg=1,n
      do mu=0,3
        write(ou,'(A,3(1X,I0),2(1X,Z32.32))') 'X',id,leg,mu,external_bits(:,mu,leg)
      enddo
    enddo
    do mu=0,3
      write(ou,'(A,2(1X,I0),2(1X,Z32.32))') 'Q',id,mu,q_bits(:,mu)
    enddo
    do mode=0,mode_count-1
      status=qualifier_begin(int(mode,c_int),requests,baseline)
      observed_before=qualifier_observe(before)
      accepted_bit=-1
      if(status==0.and.observed_before==0) then
        select case(shape)
        case(0)
          accepted=exact_geometry_difference(width,reference_width,external_bits(:,:,:n),coefficients(:n),q_bits)
        case(1)
          accepted=exact_geometry_difference(width,reference_width,external_bits(1:1,:,:n),coefficients(:n),q_bits)
        case(2)
          accepted=exact_geometry_difference(width,reference_width,external_bits(:,0:2,:n),coefficients(:n),q_bits)
        case(3)
          accepted=exact_geometry_difference(width,reference_width,external_bits(:,:,1:max(0,n-1)), &
              coefficients(:n),q_bits)
        case(4)
          accepted=exact_geometry_difference(width,reference_width,external_bits(:,:,:n),coefficients(:n),q_bits(1:1,:))
        case(5)
          accepted=exact_geometry_difference(width,reference_width,external_bits(:,:,:n),coefficients(:n),q_bits(:,0:2))
        end select
        accepted_bit=merge(1,0,accepted)
      endif
      observed_after=qualifier_observe(after)
      restore_ok=qualifier_restore(restored)
      ! The C failure reporter uses only integer formatting/write/_exit and
      ! never returns. A failed restoration cannot reach formatted Fortran I/O.
      if(restore_ok/=1) then
        call qualifier_raw_restore_failure(int(id,c_int),int(mode,c_int),status,restore_ok)
      endif
      ! No I/O, decimal conversion or formatting ran while the test environment
      ! was active; even SKIP/error records are emitted only after restoration.
      write(ou,'(A,*(1X,I0))') 'CASE',id,mode,status,accepted_bit,restore_ok,requests,baseline,before,after,restored
      if(status==2.or.observed_before/=0.or.observed_after/=0) then
        flush(ou)
        error stop 'Environment setup/observation/restoration failure'
      endif
    enddo
    flush(ou)
  enddo
  read(iu,'(A)',iostat=ios) line
  if(ios>=0) error stop 'Unexpected extra fixture record'
  close(iu)
  write(ou,'(A,1X,I0)') 'DONE',ncases
  close(ou)
  write(output_unit,'(A,1X,I0)') 'DONE',qualifier_pid()
  flush(output_unit)
  read(input_unit,'(A)',iostat=ios) command
  if(ios/=0.or.trim(command)/='EXIT') error stop 'EXIT handshake required'
end program
