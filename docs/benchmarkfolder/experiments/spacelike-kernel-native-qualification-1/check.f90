program check_certificate
  use ml_private_spacelike_certificate
  use, intrinsic :: ieee_exceptions
  implicit none
  integer :: ios, id, width, cls, mask, have, env, seed, e(0:3), j
  integer(ik) :: words(5)
  logical :: answer, initial(size(ieee_all)), final(size(ieee_all))
  do
    read(*,*,iostat=ios) id,width,cls,mask,have,env,seed,e,words
    if(ios<0) exit
    if(ios/=0) stop 2
    do j=1,size(initial)
      initial(j)=btest(seed,j-1)
    enddo
    call ieee_set_flag(ieee_all,initial)
    answer=certified_spacelike(words(1:4),words(5),e,width,cls, &
       mask/=0,have/=0,env/=0)
    call ieee_get_flag(ieee_all,final)
    write(*,'(I0,1X,L1,1X,L1)') id,answer,all(initial.eqv.final)
  enddo
end program
