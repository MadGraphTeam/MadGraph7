      program isr_frame
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'run.inc'
      double precision pb(0:3,nexternal-1),pc(0:3,nexternal-1)
      common/pborn/pb
      common/pborn_coll/pc
      double precision p(0:3,nexternal),p2(0:3,nexternal)
      double precision xi_ev,y_ev,pi_ev(0:3),pi_cnt(0:3,-2:2)
      common/fksvariables/xi_ev,y_ev,pi_ev,pi_cnt
      double precision kp(0:3)
      common/cxij_kperp/kp
      double complex xij
      common/cxij_aor/xij
      double precision ylab,ycm,sqrtshat,shat
      common/parton_cms_stuff/ylab,ycm,sqrtshat,shat
      integer i_fks,j_fks,m_type,i_type,j_type
      common/fks_indices/i_fks,j_fks
      double precision ch_i,ch_j,ch_m,iden
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      common/c_iden_comp/iden
      logical use_evpr,split_type(nsplitorders)
      common/to_use_evpr/use_evpr
      common/c_split_type/split_type
      integer iextra,isborn,iscnt
      common/c_extra_cnt/iextra,isborn,iscnt
      double complex ans(2,nsplitorders),ans2(2,nsplitorders)
      double complex cnt(amp_split_size,2,nsplitorders),cnt2(amp_split_size,2,nsplitorders)
      double complex ca(2,nsplitorders),cc(amp_split_size,2,nsplitorders)
      double precision ret(amp_split_size),ret2(amp_split_size),cs(amp_split_size)
      double precision w,w2,xi,y,norm,angle
      integer i,j,k,ifr,ix,iy
      call setpara('param_card.dat')
      do j=1,nexternal
         read(*,*) (p(i,j),i=0,3)
      enddo
      do j=1,nexternal-1
         read(*,*) (pb(i,j),i=0,3)
      enddo
      do j=1,nexternal
         read(*,*) (p2(i,j),i=0,3)
      enddo
      do j=1,nexternal-1
         read(*,*) (pc(i,j),i=0,3)
      enddo
C Synthetic nonzero complex spin correlations isolate the phase contraction.
      ch_i=0d0
      ch_j=0d0
      ch_m=0d0
      m_type=8
      i_type=8
      j_type=8
      i_fks=nexternal
      iextra=0
      use_evpr=.false.
      iden=1d0
      split_type=.true.
      sqrtshat=2d0*pb(0,1)
      shat=sqrtshat**2
      cs=0d0
      ca=(3d0,-2d0)
      cc=(3d0,-2d0)
      angle=.37d0
      kp=(/0d0,cos(angle),sin(angle),0d0/)
      xij=-exp(dcmplx(0d0,2d0*angle))
      do ifr=1,2
      frame_id=6
      if (ifr.eq.2) frame_id=8
      do j_fks=1,2
      do ix=0,1
      xi=.2d0*ix
      do iy=0,1
      y=1d0-1d-7*iy
      pi_ev=(/10d0,10d0*sqrt(1d0-y*y)*cos(angle),
     $        10d0*sqrt(1d0-y*y)*sin(angle),10d0*y/)
      if(j_fks.eq.2) pi_ev(3)=-pi_ev(3)
      ans=(2d0,1d0)
      cnt=(2d0,1d0)
      ret=0d0
      ans2=ans
      cnt2=cnt
      ret2=ret
      if(ix.eq.1) then
         ans=ca
         cnt=cc
      endif
      call sborncol_isr(p,xi,y,w,ret,ans,cnt)
      call sborncol_isr_store(p,xi,y,w2,ret2,ans2,cnt2,cs,ca,cc)
      norm=max(abs(w),maxval(abs(ret)),1d-99)
      write(*,*) 'ISR_FRAME',ifr,j_fks,ix,iy,w,w2,
     $            maxval(abs(ret-ret2))/norm
      if(abs(w-w2)/norm.gt.1d-12) stop 1
      if(maxval(abs(ret-ret2))/norm.gt.1d-12) stop 2
      enddo
      enddo
      enddo
      enddo
      end
C No extra Born exists for this isolated contraction probe.
      subroutine extra_cnt(p,idx,ans)
      stop 3
      end
