      program probe
      use extra_weights
      use fks_phase_space_data, only: pb=>p_born,pc=>p_born_coll,
     $     ylab=>ybst_til_tolab,ycm=>ybst_til_tocm,
     $     rootshat=>sqrtshat,shat
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'orders.inc'
      include 'coupl.inc'
      include 'run.inc'
      include 'q_es.inc'
      include '../../Source/PDF/pdf.inc'
      double precision p(0:3,4),savedp(0:3,4)
      double precision delta,iden,iden_fks(1)
      common/cdelta_used/delta
      common/c_iden_comp/iden,iden_fks
      integer ifks,jfks,itype,jtype,mtype,extra,isborn,iscnt,jpdg
      common/fks_indices/ifks,jfks
      double precision chi,chj,chm
      common/cparticle_types/chi,chj,chm,itype,jtype,mtype,jpdg
      common/c_extra_cnt/extra,isborn,iscnt
      logical projected,split_type(2),calculated
      common/to_use_evpr/projected
      common/c_split_type/split_type
      common/ccalculatedBorn/calculated
      double complex ans(2,2),cnt(2,2,2),ca(2,2),cc(2,2,2)
      double precision out(14),ref(14),xi,w,ret(2),coll(2)
      integer ip,ix,kind,scheme,checks,hooks
      common/probe_hooks/hooks
      pb=2d0
      pc=3d0
      p=0d0
      p(0,1)=10d0
      p(3,1)=10d0
      p(0,2)=10d0
      p(3,2)=-10d0
      savedp=p
      rootshat=20d0
      shat=400d0
      ylab=0d0
      ycm=0d0
      delta=.2d0
      iden=1d0
      q2fact=(/7d0,11d0/)
      qes2=25d0
      xiAI=1d0
      RAI=1d0
      ifks=4
      jtype=3
      chj=1d0/3d0
      extra=0
      isborn=1
      iscnt=2
      hooks=0
      checks=0
      split_type=.true.
      doreweight=.false.
      do scheme=0,1
      pdfscheme=scheme
      do kind=1,4
      mtype=8
      itype=8
      chm=0d0
      chi=0d0
      if (kind.eq.2.or.kind.eq.4) then
         mtype=3
         chm=1d0/3d0
      endif
      if (kind.eq.3.or.kind.eq.4) then
         itype=3
         chi=1d0/3d0
      endif
      do ip=0,1
      do ix=0,1
      do jfks=1,3
         projected=ip.eq.1
         xi=.2d0*ix
         g=1.5d0
         gal=(.3d0,.4d0)
         calculated=.true.
         ans=(2d0,1d0)
         cnt=(2d0,1d0)
         ca=(3d0,-1d0)
         cc=(3d0,-1d0)
         amp_split_cnt=(-9d0,-8d0)
         out=-999d0
         call evaluate(p,xi,out,ans,cnt,ca,cc)
         if (any(ans.ne.(2d0,1d0))) stop 20
         if (any(cnt.ne.(2d0,1d0))) stop 21
         if (any(ca.ne.(3d0,-1d0))) stop 22
         if (any(cc.ne.(3d0,-1d0))) stop 23
         if (any(p.ne.savedp).or.any(pb.ne.2d0)) stop 24
         if (any(pc.ne.3d0)) stop 25
         if (g.ne.1.5d0.or.any(gal.ne.(.3d0,.4d0))) stop 26
         if (.not.calculated) stop 27
         if (any(amp_split_cnt.ne.(-9d0,-8d0))) stop 28
         if (hooks.ne.0) stop 29
         if (jfks.eq.3) then
            if (any(out.ne.0d0)) stop 30
         else if (kind.le.2.or.ix.ne.0) then
            if (out(2).eq.0d0) stop 31
         endif
c Independent projection choice selects the same explicit collinear arrays.
         if (ip.eq.0.and.ix.eq.1) then
            projected=.true.
            ans=ca
            cnt=cc
            call evaluate(p,xi,ref,ans,cnt,ca,cc)
            if (any(out.ne.ref)) stop 32
         endif
         checks=checks+1
      enddo
      enddo
      enddo
      enddo
      enddo
c Exercise the actual ISR consumer, including its no-projection overwrite.
c A quark mother has no spin-phase call; any unexpected such call stops.
      mtype=3
      itype=8
      chm=1d0/3d0
      chi=0d0
      do ip=0,1
      do ix=0,1
      do jfks=1,2
         projected=ip.eq.1
         xi=.2d0*ix
         ans=(2d0,1d0)
         cnt=(2d0,1d0)
         ca=(3d0,-1d0)
         cc=(3d0,-1d0)
         ret=0d0
         coll=3d0
         call sborncol_isr_store_buffered(p,xi,1d0,w,ret,ans,cnt,coll,ca,cc)
         if (.not.(w.gt.0d0).or.any(.not.(ret.gt.0d0))) stop 36
         if (ip.eq.0.and.ix.eq.1) then
            if (any(ans.ne.ca).or.any(cnt.ne.cc)) stop 37
         else
            if (any(ans.ne.(2d0,1d0))) stop 38
            if (any(cnt.ne.(2d0,1d0))) stop 39
         endif
         if (any(ca.ne.(3d0,-1d0))) stop 40
         if (any(cc.ne.(3d0,-1d0))) stop 41
         if (.not.calculated.or.g.ne.1.5d0) stop 42
      enddo
      enddo
      enddo
c Prove the extra-counterterm branch really permits state mutation.
      extra=1
      jfks=1
      call evaluate(p,.2d0,out,ans,cnt,ca,cc)
      if (hooks.ne.1.or.calculated) stop 33
      if (g.ne.7d0.or.any(amp_split_cnt.ne.(8d0,9d0))) stop 34
      if (checks.ne.96) stop 35
      print *, 'DEGENERATE_PASS',checks
      end
      subroutine evaluate(p,xi,out,ans,cnt,ca,cc)
      implicit none
      double precision p(0:3,4),xi,out(14)
      double complex ans(2,2),cnt(2,2,2),ca(2,2),cc(2,2,2)
      call sreal_deg_buffered(p,xi,1d0,out(1),out(2),out(3),out(5),out(7),
     $               out(9),out(11),out(13),ans,cnt,ca,cc)
      end
      double precision function dot(p,q)
      double precision p(0:3),q(0:3)
      dot=p(0)*q(0)-sum(p(1:3)*q(1:3))
      end
      subroutine extra_cnt_frame(p,index,ans)
      implicit none
      include 'coupl.inc'
      include 'orders.inc'
      double precision p(0:3,3)
      integer index,hooks
      common/probe_hooks/hooks
      double complex ans(2,2)
      logical calculated
      common/ccalculatedBorn/calculated
      hooks=hooks+1
      g=7d0
      calculated=.false.
      amp_split_cnt=(8d0,9d0)
      ans=(8d0,9d0)
      end
      subroutine azifact_me_frame
      stop 90
      end
      subroutine IXXXXX
      stop 91
      end
      subroutine OXXXXX
      stop 92
      end
