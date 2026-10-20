program reversal
  use mninjars
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: n,rank,nc,k,j,i,d,first,last,case_id,status,mode,phase
  complex(KI_NRS),allocatable :: c(:),cr(:),m(:),mr(:)
  real(KI_NRS),allocatable :: p(:,:),pr(:,:),s(:,:),sr(:,:)
  complex(KI_QNRS),allocatable :: qc(:),qcr(:),qm(:),qmr(:)
  real(KI_QNRS),allocatable :: qp(:,:),qpr(:,:),qs(:,:),qsr(:,:)
  complex(KI_NRS) :: r(0:2),r1,base(0:2,2),back(0:2,2),negative(0:2,2)
  complex(KI_QNRS) :: qr(0:2),qr1,qbase(0:2),qback(0:2),qnegative(0:2)
  real(KI_QNRS) :: norm,common_norm,bound,fraction,largest,negative_norm,negative_bound,negative_difference
  character(len=1024) :: message
  do case_id=1,2
    n=case_id+2
    rank=n
    nc=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24
    allocate(c(nc),cr(nc),m(n),mr(n),p(0:3,n),pr(0:3,n),s(n,n),sr(n,n))
    allocate(qc(nc),qcr(nc),qm(n),qmr(n),qp(0:3,n),qpr(0:3,n),qs(n,n),qsr(n,n))
    do k=0,nc-1
      c(k+1)=cmplx(real(mod(7*k,13)-6,KI_NRS)/16,real(mod(11*k,17)-8,KI_NRS)/32,KI_NRS)
    enddo
    p=0
    do j=2,n
      p(j-1,j)=j-1
    enddo
    do j=1,n
      m(j)=cmplx(real((j+1)**2,KI_NRS),-real((j+1)**2,KI_NRS)/128,KI_NRS)
      do i=1,n
        s(i,j)=(p(0,i)-p(0,j))**2-sum((p(1:3,i)-p(1:3,j))**2)
      enddo
    enddo
    cr=c
    do d=1,rank,2
      first=d*(d+1)*(d+2)*(d+3)/24+1
      last=(d+1)*(d+2)*(d+3)*(d+4)/24
      cr(first:last)=-cr(first:last)
    enddo
    if(all(c(2:5)==0).or.all(aimag(m)==0).or.all(p==0)) stop 90
    do j=1,n
      pr(:,j)=-p(:,n+1-j)
      mr(j)=m(n+1-j)
      do i=1,n
        sr(i,j)=s(n+1-i,n+1-j)
      enddo
    enddo
    qc=cmplx(real(c,KI_QNRS),real(aimag(c),KI_QNRS),KI_QNRS)*(1.0_KI_QNRS+2.0_KI_QNRS**(-100))
    qcr=cmplx(real(cr,KI_QNRS),real(aimag(cr),KI_QNRS),KI_QNRS)*(1.0_KI_QNRS+2.0_KI_QNRS**(-100))
    qp=real(p,KI_QNRS); qpr=real(pr,KI_QNRS)
    qs=real(s,KI_QNRS); qsr=real(sr,KI_QNRS)
    qm=cmplx(real(m,KI_QNRS),real(aimag(m),KI_QNRS),KI_QNRS)
    qmr=cmplx(real(mr,KI_QNRS),real(aimag(mr),KI_QNRS),KI_QNRS)
    do phase=1,5
      if(phase==4) then
        call ninjars_finalize(status)
        if(status/=NINJARS_SUCCESS) stop 20
      endif
      do mode=1,2
        r=cmplx(-999,-999,KI_NRS); r1=r(0)
        if(phase==2) then
          if(mode==1) call ninjars_tensor_evaluate(cr,n,rank,sr,pr,mr,49.0_KI_NRS,r,r1,status)
          if(mode==2) call ninjars_tensor_evaluate_compensated(cr,n,rank,sr,pr,mr,49.0_KI_NRS,r,r1,status)
        else if(phase==5) then
          ! Deliberately wrong: reverse propagators but leave odd coefficients unchanged.
          if(mode==1) call ninjars_tensor_evaluate(c,n,rank,sr,pr,mr,49.0_KI_NRS,r,r1,status)
          if(mode==2) call ninjars_tensor_evaluate_compensated(c,n,rank,sr,pr,mr,49.0_KI_NRS,r,r1,status)
        else
          if(mode==1) call ninjars_tensor_evaluate(c,n,rank,s,p,m,49.0_KI_NRS,r,r1,status)
          if(mode==2) call ninjars_tensor_evaluate_compensated(c,n,rank,s,p,m,49.0_KI_NRS,r,r1,status)
        endif
        write(*,'(A,4I5,8ES49.39E4)') 'ROW ',case_id,phase,mode,status,r,r1
        if(status/=NINJARS_SUCCESS) then
          call ninjars_error_message(message)
          print *, trim(message)
          stop 10
        endif
        if(phase==1) base(:,mode)=r
        if(phase==2) back(:,mode)=r
        if(phase==5) negative(:,mode)=r
        if(phase==3.or.phase==4) then
          if(any(transfer(r,[0_int64],6)/=transfer(base(:,mode),[0_int64],6))) stop 11
        endif
      enddo
      qr=cmplx(-999,-999,KI_QNRS); qr1=qr(0)
      if(phase==2) then
        call ninjars_tensor_evaluate(qcr,n,rank,qsr,qpr,qmr,49.0_KI_QNRS,qr,qr1,status)
      else if(phase==5) then
        call ninjars_tensor_evaluate(qc,n,rank,qsr,qpr,qmr,49.0_KI_QNRS,qr,qr1,status)
      else
        call ninjars_tensor_evaluate(qc,n,rank,qs,qp,qm,49.0_KI_QNRS,qr,qr1,status)
      endif
      write(*,'(A,4I5,8ES49.39E4)') 'ROW ',case_id,phase,3,status,qr,qr1
      if(status/=NINJARS_SUCCESS) then
        call ninjars_error_message(message)
        print *, trim(message)
        stop 12
      endif
      if(phase==1) qbase=qr
      if(phase==2) qback=qr
      if(phase==5) qnegative=qr
      if(phase==3.or.phase==4) then
        if(any(transfer(qr,[0_int64],12)/=transfer(qbase,[0_int64],12))) stop 13
      endif
    enddo
    norm=max(maxval(abs(qbase)),maxval(abs(qback)))
    if(norm<=0) stop 14
    common_norm=max(norm,real(maxval(abs(base)),KI_QNRS),real(maxval(abs(back)),KI_QNRS))
    largest=0
    do i=0,2
      fraction=abs(qbase(i)-qback(i))/(1.e-31_KI_QNRS*norm)
      largest=max(largest,fraction)
      if(fraction>1) stop 15
      do mode=1,2
        bound=1.e-7_KI_QNRS*max(abs(base(i,mode)),abs(back(i,mode)))+1.e-10_KI_QNRS*common_norm
        if(abs(base(i,mode)-back(i,mode))>bound) stop 16
        bound=1.e-7_KI_QNRS*max(abs(cmplx(base(i,mode),kind=KI_QNRS)),abs(qbase(i)))+1.e-10_KI_QNRS*common_norm
        if(abs(cmplx(base(i,mode),kind=KI_QNRS)-qbase(i))>bound) stop 17
        bound=1.e-7_KI_QNRS*max(abs(cmplx(back(i,mode),kind=KI_QNRS)),abs(qback(i)))+1.e-10_KI_QNRS*common_norm
        if(abs(cmplx(back(i,mode),kind=KI_QNRS)-qback(i))>bound) stop 18
      enddo
    enddo
    ! A resolved negative control is required independently of the observed answer:
    ! ten times the approved comparison bound evaluated on a common norm.
    negative_norm=max(common_norm,maxval(abs(qnegative)),real(maxval(abs(negative)),KI_QNRS))
    negative_bound=10.0_KI_QNRS*(1.e-7_KI_QNRS+1.e-10_KI_QNRS)*negative_norm
    negative_difference=maxval(abs(qnegative-qback))
    if(negative_difference<=negative_bound) stop 21
    do mode=1,2
      if(real(maxval(abs(negative(:,mode)-back(:,mode))),KI_QNRS)<=negative_bound) stop 22
      do i=0,2
        bound=1.e-7_KI_QNRS*max(abs(cmplx(negative(i,mode),kind=KI_QNRS)),abs(qnegative(i))) &
              +1.e-10_KI_QNRS*negative_norm
        if(abs(cmplx(negative(i,mode),kind=KI_QNRS)-qnegative(i))>bound) stop 23
      enddo
    enddo
    write(*,'(A,I5,3ES49.39E4)') 'NEGATIVE_CASE ',case_id,negative_difference,negative_bound, &
                                negative_difference/negative_bound
    write(*,'(A,I5,ES49.39E4)') 'PASS_CASE ',case_id,largest
    deallocate(c,cr,m,mr,p,pr,s,sr,qc,qcr,qm,qmr,qp,qpr,qs,qsr)
  enddo
  call ninjars_finalize(status)
  if(status/=NINJARS_SUCCESS) stop 19
  print *, 'PASS_REVERSAL_ABI3'
end program
