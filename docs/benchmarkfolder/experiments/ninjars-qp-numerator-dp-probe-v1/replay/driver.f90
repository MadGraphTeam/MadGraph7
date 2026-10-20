! Standalone stored-input diagnostic; no MadLoop controller or amplitude calls.
program stored_input_replay
  use mninjars
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_round_type, ieee_get_rounding_mode, ieee_nearest, operator(/=)
  implicit none
  integer :: inp,out,ios,key,point,direction,group,order,n,rank,nc,i,j,cell,status,final_status
  integer :: seen,calls
  integer(int64) :: w(4)
  character(len=4096) :: input_path,output_path
  character(len=16) :: tag
  type(ieee_round_type) :: rounding
  complex(KI_NRS), allocatable :: cd(:),md(:),cwork(:)
  real(KI_NRS), allocatable :: sd(:,:),pd(:,:)
  real(KI_NRS) :: ud
  complex(KI_QNRS), allocatable :: cq(:),mq(:),cx(:),mx(:)
  real(KI_QNRS), allocatable :: sq(:,:),pq(:,:),sx(:,:),px(:,:)
  real(KI_QNRS) :: uq,ux
  complex(KI_NRS) :: rd(0:2),r1d
  complex(KI_QNRS) :: rq(0:2),r1q
  integer(int64), allocatable :: before(:),after(:)

  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  if (len_trim(input_path)==0.or.len_trim(output_path)==0) error stop 'input and output required'
  if (storage_size(1.0_KI_NRS)/=64.or.digits(1.0_KI_NRS)/=53) error stop 'binary64 required'
  if (storage_size(1.0_KI_QNRS)/=128.or.digits(1.0_KI_QNRS)/=113) error stop 'binary128 required'
  w(1)=transfer(1.0_KI_NRS,w(1))
  w(2:3)=transfer(1.0_KI_QNRS,w(2:3))
  if(w(1)/=int(z'3FF0000000000000',int64).or.w(2)/=0_int64.or. &
     w(3)/=int(z'3FFF000000000000',int64)) error stop 'unsupported word order'
  w(1:2)=transfer(cmplx(1,2,kind=KI_NRS),w(1:2))
  if(any(w(1:2)/=[int(z'3FF0000000000000',int64),int(z'4000000000000000',int64)])) &
    error stop 'unsupported complex64 layout'
  w=transfer(cmplx(1,2,kind=KI_QNRS),w)
  if(any(w/=[0_int64,int(z'3FFF000000000000',int64),0_int64,int(z'4000000000000000',int64)])) &
    error stop 'unsupported complex128 layout'
  call ieee_get_rounding_mode(rounding)
  if(rounding/=ieee_nearest) error stop 'round-to-nearest required'
  open(newunit=inp,file=trim(input_path),status='old',action='read')
  open(newunit=out,file=trim(output_path),status='new',action='write')
  seen=0
  calls=0
  do
    read(inp,*,iostat=ios) tag,key,point,direction,group,order,n,rank
    if(ios<0) exit
    if(ios/=0.or.tag/='CASE') error stop 'bad case header'
    seen=seen+1
    if(key/=seen.or.seen>464) error stop 'case sequence'
    if((point/=44.and.point/=80).or.direction<1.or.direction>2.or.group<1.or.group>116.or. &
       order/=1.or.n<1.or.n>5.or.rank<0.or.rank>4) error stop 'unexpected domain'
    nc=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24
    allocate(cd(nc),md(n),cwork(nc),sd(n,n),pd(0:3,n))
    allocate(cq(nc),mq(n),cx(nc),mx(n),sq(n,n),pq(0:3,n),sx(n,n),px(0:3,n))
    read(inp,*) tag
    if(tag/='DP') error stop 'missing DP'
    do i=1,nc
      read(inp,'(Z16,1X,Z16)') w(1:2)
      cd(i)=transfer(w(1:2),cd(i))
    enddo
    do j=1,n
      do i=1,n
        read(inp,'(Z16)') w(1)
        sd(i,j)=transfer(w(1),sd(i,j))
      enddo
      do i=0,3
        read(inp,'(Z16)') w(1)
        pd(i,j)=transfer(w(1),pd(i,j))
      enddo
      read(inp,'(Z16,1X,Z16)') w(1:2)
      md(j)=transfer(w(1:2),md(j))
    enddo
    read(inp,'(Z16)') w(1)
    ud=transfer(w(1),ud)
    read(inp,*) tag
    if(tag/='QP') error stop 'missing QP'
    do i=1,nc
      read(inp,'(4(Z16,1X))') w
      cq(i)=transfer(w,cq(i))
    enddo
    do j=1,n
      do i=1,n
        read(inp,'(Z16,1X,Z16)') w(1:2)
        sq(i,j)=transfer(w(1:2),sq(i,j))
      enddo
      do i=0,3
        read(inp,'(Z16,1X,Z16)') w(1:2)
        pq(i,j)=transfer(w(1:2),pq(i,j))
      enddo
      read(inp,'(4(Z16,1X))') w
      mq(j)=transfer(w,mq(j))
    enddo
    read(inp,'(Z16,1X,Z16)') w(1:2)
    uq=transfer(w(1:2),uq)
    read(inp,*) tag
    if(tag/='END') error stop 'missing end'
    do cell=1,8
      write(out,*) 'BEGIN',key,point,direction,group,order,n,rank,cell
      if(cell==1.or.cell==2.or.cell==4.or.cell==5) then
        cwork=cd
        if(cell==4.or.cell==5) cwork=cmplx(real(cq,kind=KI_NRS),real(aimag(cq),kind=KI_NRS),kind=KI_NRS)
        call words_dp(before)
        call echo_dp()
        flush(out)
        if(cell==2.or.cell==5) then
          call ninjars_tensor_evaluate_compensated(cwork,n,rank,sd,pd,md,ud,rd,r1d,status)
        else
          call ninjars_tensor_evaluate(cwork,n,rank,sd,pd,md,ud,rd,r1d,status)
        endif
        call words_dp(after)
        if(any(before/=after)) error stop 'DP input mutation'
        write(out,*) 'RESULT',64,status
        do i=0,2
          write(out,'(A,1X,I1,1X,2(Z16.16,1X))') 'R',i,transfer(rd(i),w(1:2))
        enddo
        write(out,'(A,1X,2(Z16.16,1X))') 'R1',transfer(r1d,w(1:2))
      else
        sx=real(sd,kind=KI_QNRS)
        px=real(pd,kind=KI_QNRS)
        mx=cmplx(real(md,kind=KI_QNRS),real(aimag(md),kind=KI_QNRS),kind=KI_QNRS)
        ux=real(ud,kind=KI_QNRS)
        if(cell==3) then
          cx=cmplx(real(cd,kind=KI_QNRS),real(aimag(cd),kind=KI_QNRS),kind=KI_QNRS)
        else if(cell==6) then
          cwork=cmplx(real(cq,kind=KI_NRS),real(aimag(cq),kind=KI_NRS),kind=KI_NRS)
          cx=cmplx(real(cwork,kind=KI_QNRS),real(aimag(cwork),kind=KI_QNRS),kind=KI_QNRS)
        else
          cx=cq
        endif
        if(cell==8) then
          sx=sq
          px=pq
          mx=mq
          ux=uq
        endif
        call words_qp(before)
        call echo_qp()
        flush(out)
        call ninjars_tensor_evaluate(cx,n,rank,sx,px,mx,ux,rq,r1q,status)
        call words_qp(after)
        if(any(before/=after)) error stop 'QP input mutation'
        write(out,*) 'RESULT',128,status
        do i=0,2
          write(out,'(A,1X,I1,1X,4(Z16.16,1X))') 'R',i,transfer(rq(i),w)
        enddo
        write(out,'(A,1X,4(Z16.16,1X))') 'R1',transfer(r1q,w)
      endif
      deallocate(before,after)
      calls=calls+1
      write(out,*) 'END',key,cell
      flush(out)
    enddo
    deallocate(cd,md,cwork,sd,pd,cq,mq,cx,mx,sq,pq,sx,px)
  enddo
  if(seen/=464.or.calls/=3712) error stop 'incomplete replay'
  call ninjars_finalize(final_status)
  write(out,*) 'COMPLETE',seen,calls,final_status
  close(inp)
  close(out)
  if(final_status/=0) error stop 'finalize failed'
contains
  subroutine words_dp(words)
    integer(int64),allocatable,intent(out) :: words(:)
    words=[transfer(cwork,[0_int64]),transfer(sd,[0_int64]),transfer(pd,[0_int64]), &
           transfer(md,[0_int64]),transfer(ud,[0_int64])]
  end subroutine
  subroutine words_qp(words)
    integer(int64),allocatable,intent(out) :: words(:)
    words=[transfer(cx,[0_int64]),transfer(sx,[0_int64]),transfer(px,[0_int64]), &
           transfer(mx,[0_int64]),transfer(ux,[0_int64])]
  end subroutine
  subroutine echo_dp()
    integer :: a,b
    write(out,*) 'INPUT',64
    do a=1,nc
      write(out,'(A,1X,I0,1X,2(Z16.16,1X))') 'C',a-1,transfer(cwork(a),w(1:2))
    enddo
    do b=1,n
      do a=1,n
        write(out,'(A,2(1X,I0),1X,Z16.16)') 'S',a,b,transfer(sd(a,b),w(1))
      enddo
      do a=0,3
        write(out,'(A,2(1X,I0),1X,Z16.16)') 'P',a,b,transfer(pd(a,b),w(1))
      enddo
      write(out,'(A,1X,I0,1X,2(Z16.16,1X))') 'M',b,transfer(md(b),w(1:2))
    enddo
    write(out,'(A,1X,Z16.16)') 'U',transfer(ud,w(1))
  end subroutine
  subroutine echo_qp()
    integer :: a,b
    write(out,*) 'INPUT',128
    do a=1,nc
      write(out,'(A,1X,I0,1X,4(Z16.16,1X))') 'C',a-1,transfer(cx(a),w)
    enddo
    do b=1,n
      do a=1,n
        write(out,'(A,2(1X,I0),1X,2(Z16.16,1X))') 'S',a,b,transfer(sx(a,b),w(1:2))
      enddo
      do a=0,3
        write(out,'(A,2(1X,I0),1X,2(Z16.16,1X))') 'P',a,b,transfer(px(a,b),w(1:2))
      enddo
      write(out,'(A,1X,I0,1X,4(Z16.16,1X))') 'M',b,transfer(mx(b),w)
    enddo
    write(out,'(A,1X,2(Z16.16,1X))') 'U',transfer(ux,w(1:2))
  end subroutine
end program
