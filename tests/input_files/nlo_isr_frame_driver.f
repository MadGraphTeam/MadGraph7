      module isr_probe_inputs
      implicit none
      include 'orders.inc'
      double complex input_ans(2,nsplitorders)
      double complex input_cnt(amp_split_size,2,nsplitorders)
      end module

      program isr_frame
      use isr_probe_inputs, only: input_ans,input_cnt
      use fks_phase_space_data, only: pb=>p_born,pc=>p_born_coll,
     $     pi_ev=>p_i_fks_ev,kp=>xij_kperp,xij=>xij_aor,
     $     sqrtshat,shat
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      include 'run.inc'
      double precision p(0:3,nexternal),p2(0:3,nexternal)
      integer i_fks,j_fks,m_type,i_type,j_type,j_pdg
      common/fks_indices/i_fks,j_fks
      double precision ch_i,ch_j,ch_m,iden,iden_fks(fks_configs)
      common/cparticle_types/ch_i,ch_j,ch_m,
     $     i_type,j_type,m_type,j_pdg
      common/c_iden_comp/iden,iden_fks
      logical use_evpr,split_type(nsplitorders),fixed_order,nlo_ps
      common/to_use_evpr/use_evpr
      common/c_split_type/split_type
      common/c_fnlo_nlops/fixed_order,nlo_ps
      integer iextra,isborn,iscnt
      common/c_extra_cnt/iextra,isborn,iscnt
      double precision ret(amp_split_size),ret2(amp_split_size)
      double precision w,w2,xi,y,norm,angle,z,t,ap(2),q(2)
      double precision born_used(0:3,nexternal-1),pi(0:3),pj(0:3)
      double complex phase,factor
      logical boosted
      integer i,j,k,ifr,ix,iy,iord
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
C Controlled nonzero complex Born tensors isolate the phase contraction;
C the full lane probe separately checks actual Born/real matrix elements.
      ch_i=0d0
      ch_j=0d0
      ch_m=0d0
      m_type=8
      i_type=8
      j_type=8
      j_pdg=21
      i_fks=nexternal
      iextra=0
      use_evpr=.false.
      fixed_order=.true.
      nlo_ps=.false.
      iden=1d0
      iden_fks=1d0
      split_type=.true.
      sqrtshat=2d0*pb(0,1)
      shat=sqrtshat**2
      input_ans(1,:)=(2d0,1d0)
      input_ans(2,:)=(3d0,-2d0)
      do i=1,amp_split_size
         input_cnt(i,1,:)=dcmplx(2*i,i)
         input_cnt(i,2,:)=dcmplx(3*i,-2*i)
      enddo
      angle=.37d0
      kp=(/0d0,cos(angle),sin(angle),0d0/)
      xij=-exp(dcmplx(0d0,2d0*angle))
      do ifr=1,2
      frame_id=6
      if (ifr.eq.2) frame_id=8
      do j_fks=1,2
      do ix=0,1
      xi=.2d0*ix
      z=1d0-xi
      t=z*shat/4d0
      call AP_reduced(m_type,i_type,ch_m,ch_i,t,z,ap)
      call Qterms_reduced_spacelike(m_type,i_type,ch_m,ch_i,t,z,q)
      born_used=pb
      if(xi.gt.0d0)born_used=pc
      do iy=0,1
      y=1d0-1d-7*iy
      pi_ev=(/10d0,10d0*sqrt(1d0-y*y)*cos(angle),
     $        10d0*sqrt(1d0-y*y)*sin(angle),10d0*y/)
      if(j_fks.eq.2) pi_ev(3)=-pi_ev(3)
      call sborncol_isr(p,xi,y,w)
      ret=amp_split
C Independently contract the unmodified input tensor with the selected
C splitting kernels and phase, rather than calling the removed store API.
      call azifact_me_frame(born_used,j_fks,pi_ev,p(0,j_fks),
     $     y,phase,boosted)
      if(.not.boosted)then
         phase=xij
         if(1d0-y.ge.1d-8)then
            pi=p(:,i_fks)
            if(xi.lt.1d-8)pi=pi_ev
            pj=p(:,j_fks)
            if(j_fks.eq.2)then
               pi(1)=-pi(1)
               pi(3)=-pi(3)
               pj(1)=-pj(1)
               pj(3)=-pj(3)
            endif
            call azifact_from_spinors(pi,pj,phase)
         endif
      endif
      factor=-dconjg(phase)
      w2=0d0
      ret2=0d0
      do iord=1,nsplitorders
         k=0
         if(iord.eq.qcd_pos)k=1
         if(iord.eq.qed_pos)k=2
         if(k.eq.0)cycle
         w2=w2+dble(input_ans(1,iord)*ap(k)
     $        +factor*input_ans(2,iord)*q(k))
         ret2=ret2+dble(input_cnt(:,1,iord)*ap(k)
     $        +factor*input_cnt(:,2,iord)*q(k))
      enddo
      norm=max(abs(w2),maxval(abs(ret2)),1d-99)
      write(*,*) 'ISR_FRAME',ifr,j_fks,ix,iy,w,w2,
     $            maxval(abs(ret-ret2))/norm
      if(abs(w-w2)/norm.gt.1d-12) stop 1
      if(maxval(abs(ret-ret2))/norm.gt.1d-12) stop 2
      enddo
      enddo
      enddo
      enddo
      end

      subroutine isr_probe_born(p,w)
      use isr_probe_inputs, only: input_ans,input_cnt
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),w
      double complex ans_cnt(2,nsplitorders)
      common/c_born_cnt/ans_cnt
      ans_cnt=input_ans
      amp_split_cnt=input_cnt
      w=1d0
      end

      subroutine isr_probe_no_extra(p,idx,ans)
C No extra Born participates in this isolated phase contraction.
      stop 3
      end
