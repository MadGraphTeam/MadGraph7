
program caller_consumers
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: wl(4,0:LOOPMAXCOEFS-1,4,2),coefs(4,0:VERTEXMAXCOEFS-1,4)
common/CALLER_WL/wl,coefs
complex(8) :: loopcoefs(0:LOOPMAXCOEFS-1,3,2),amp(2)
common/TEST_LCOEFS/loopcoefs
common/TEST_AMPS/amp
integer :: cf_d(2,2),cf_n(2,2),goodhel(2),heloffset,helpicked
common/TEST_CF/cf_d,cf_n
logical :: checkphase,heldoublechecked,goodamp(3,2)
common/TEST_INIT/checkphase,heldoublechecked
common/TEST_Filters/goodamp,goodhel,heloffset
common/TEST_HELCHOICE/helpicked
integer :: vertex_mode
common/VERTEX_MODE_CONTROL/vertex_mode
logical :: update_flags(size(ieee_all))
common/UPDATER_FLAGS/update_flags
logical :: reference_update(size(ieee_all)),reference_final(size(ieee_all)),actual_final(size(ieee_all))
complex(8) :: saved_coefs(0:LOOPMAXCOEFS-1,3,2),saved_wl(4,0:LOOPMAXCOEFS-1,4),sentinel
integer(int64) :: cbits(size(loopcoefs)*2),wbits(size(saved_wl)*2)
integer(int64) :: input_bits(4*LOOPMAXCOEFS*4*2),vertex_bits(size(coefs)*2)
integer :: i,j,k,trial
type(ieee_round_type) :: after_rounding
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
sentinel=cmplx(-123.25_8,456.125_8,8)
cf_d=reshape([3,-5,-7,11],[2,2])
cf_n=reshape([2,-3,5,7],[2,2])
amp=[cmplx(0.375_8,-0.625_8,8),cmplx(1.125_8,0.25_8,8)]
goodhel=[2,3]
goodamp=.true.
heloffset=0
heldoublechecked=.true.
helpicked=-1
do trial=1,2
  checkphase=trial==2
  do k=1,4
    do j=0,LOOPMAXCOEFS-1
      do i=1,4
        wl(i,j,k,1)=cmplx(real(3*i-2*j+k,8)/8,real(i+j-k,8)/16,8)
      enddo
    enddo
    do j=0,VERTEXMAXCOEFS-1
      do i=1,4
        coefs(i,j,k)=cmplx(real(2*i+j+k,8)/4,real(i-2*j+k,8)/8,8)
      enddo
    enddo
  enddo
  input_bits=transfer(wl(:,:,:,1),input_bits)
  vertex_bits=transfer(coefs,vertex_bits)
  do vertex_mode=1,2
    wl(:,:,:,2)=sentinel
    loopcoefs=sentinel
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_flag(ieee_inexact,vertex_mode==2)
    call CHECK_REFERENCE()
    call ieee_get_flag(ieee_all,reference_final)
    reference_update=update_flags
    saved_coefs=loopcoefs
    saved_wl=wl(:,:,:,2)
    wl(:,:,:,2)=sentinel
    loopcoefs=sentinel
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_flag(ieee_inexact,vertex_mode==2)
    call CHECK_CANDIDATE()
    call ieee_get_flag(ieee_all,actual_final)
    if(any(reference_update.neqv.update_flags)) stop 41
    if(any(reference_final.neqv.actual_final)) stop 42
    if(any(transfer(saved_coefs,cbits)/=transfer(loopcoefs,cbits))) stop 43
    if(any(transfer(wl(:,:,:,1),input_bits)/=input_bits)) stop 44
    if(any(transfer(coefs,vertex_bits)/=vertex_bits)) stop 45
    if(vertex_mode==1) then
      ! Vertex-generated INEXACT must admit after a clear caller entry.
      do i=1,4
        do j=1,4
          if(i/=j.and.any(transfer(wl(j,0:69,i,2),wbits,140)/=0_int64)) stop 46
        enddo
        if(any(transfer(saved_wl(i,:,i),wbits,LOOPMAXCOEFS*2)/= &
           transfer(wl(i,:,i,2),wbits,LOOPMAXCOEFS*2))) stop 47
      enddo
    else
      ! A vertex that clears an incoming flag must force complete fallback.
      if(any(transfer(saved_wl,wbits)/=transfer(wl(:,:,:,2),wbits))) stop 48
    endif
    if(any(wl(:,70:,:,2)/=cmplx(0.0_8,0.0_8,8))) stop 49
    call ieee_get_rounding_mode(after_rounding)
    if(after_rounding/=ieee_nearest) stop 50
  enddo
enddo
write(*,*) 'PASS CALLER CREATE MERGE'
end program

subroutine LAST_VERTEX()
use, intrinsic :: ieee_exceptions
implicit none
integer :: vertex_mode
common/VERTEX_MODE_CONTROL/vertex_mode
real(8),volatile :: x,y,z
if(vertex_mode==1) then
  x=0.1_8
  y=0.1_8
  z=x*y
else
  call ieee_set_flag(ieee_inexact,.false.)
endif
end subroutine

subroutine RECORD_UPDATE_FLAGS()
use, intrinsic :: ieee_exceptions
implicit none
logical :: update_flags(size(ieee_all))
common/UPDATER_FLAGS/update_flags
call ieee_get_flag(ieee_all,update_flags)
end subroutine

integer function TEST_ML5SOINDEX_FOR_BORN_AMP(i)
integer :: i
TEST_ML5SOINDEX_FOR_BORN_AMP=i
end function
integer function TEST_ML5SOINDEX_FOR_LOOP_AMP(i)
integer :: i
TEST_ML5SOINDEX_FOR_LOOP_AMP=i
end function
integer function TEST_ML5SQSOINDEX(i,j)
integer :: i,j
TEST_ML5SQSOINDEX=i+j-1
end function
