      subroutine sbornsoft_onia(pp,xi_i_fks,y_ij_fks,wgt)
c**************************************************************************
c     The soft current of a P-wave quarkonium (arXiv:2402.19221, sect. 3.2;
c     madgraph/fks/fks_onia.py), added by sbornsoft to its eikonal sum. For
c     a soft gluon k_i = xi sqrt(s)/2 khat it returns the limit of
c     xi^2 (1-y) |M_real|^2 beyond the colour-linked Borns:
c
c       g^2 (1-y) { sum_K w_mu(K) VEC^mu(K)
c                   - [g_mu nu/(P.khat)^2 + P^2 khat_mu khat_nu/(P.khat)^4]
c                     TEN^{mu nu} }
c
c     with P the momentum of the onium, the sum over the coloured Born legs
c     K (the onium included) and
c       w_mu(K) = K_mu/(K.khat P.khat) - (P.K) khat_mu/(K.khat (P.khat)^2),
c     VEC and TEN from BORN_ONIA_SOFT (born_onia.f, the cross vectors and
c     the self tensor of the paper). For a massless j_fks the collinear
c     factor (1-y)/(k_j.khat) is replaced as in eikonal_reduced. All terms
c     vanish in the collinear limit y -> 1. Fills amp_split at the
c     position of the Born orders + g^2, like the colour-linked Borns.
c**************************************************************************
      use fks_phase_space_helpers, only: dot
      use fks_phase_space_data, only: p_born,resonance_momentum,
     &     resonance_recoil,p_i_fks_cnt,sqrtshat
      implicit none
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "coupl.inc"
      include "orders.inc"
      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,wgt
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      double precision iden_comp,iden_comp_FKS(fks_configs)
      common /c_iden_comp/iden_comp,iden_comp_FKS
      double precision zero,pmass(nexternal),tiny
      parameter (zero=0d0)
      parameter (tiny=1d-8)
      integer naux_max
      parameter (naux_max=2)
      double precision vec(0:3,nexternal-1),baux(naux_max)
      double complex ten(0:3,0:3),zself
      integer ionium,ionium_r,kb,kr,mu,nu,ord(nsplitorders),ipos
      double precision phat(0:3),pk(0:3),w(0:3),ponium(0:3)
      double precision pkk,fact,contr,metric(0:3),phat_low(0:3)
      data metric /1d0,-1d0,-1d0,-1d0/
      integer orders_to_amp_split_pos
      external orders_to_amp_split_pos
      include "pmass.inc"

      wgt=0d0
      if (.not.need_color_links) return
      if (1d0-y_ij_fks.lt.tiny) return
c the reduced momentum of i_fks, as in eikonal_reduced (softcol = 0)
      if (p_i_fks_cnt(0,0).lt.0d0) then
         if (xi_i_fks.eq.0d0 .or. pp(0,i_fks).eq.0d0) then
            write (*,*) 'Error #1 in sbornsoft_onia',xi_i_fks,y_ij_fks
            stop 1
         endif
         phat(0:3)=pp(0:3,i_fks)/xi_i_fks
      else
         phat(0:3)=p_i_fks_cnt(0:3,0)
      endif
      call born_onia_soft(p_born,vec,ten,baux,ionium)
      ionium_r=ionium
      if (ionium.ge.i_fks) ionium_r=ionium+1
      ponium(0:3)=pp(0:3,ionium_r)
      pkk=dot(ponium,phat)
c the cross vectors
      contr=0d0
      do kb=1,nexternal-1
         if (all(vec(0:3,kb).eq.0d0)) cycle
         kr=kb
         if (kb.ge.i_fks) kr=kb+1
         pk(0:3)=pp(0:3,kr)
         if (kr.eq.j_fks .and. pmass(j_fks).eq.zero) then
c           (1-y)/(k_j.khat) for the collinear partner
            fact=2d0/(sqrtshat*pp(0,j_fks))
            if (resonance_recoil)
     &           fact=2d0/dot(resonance_momentum,pp(0:3,j_fks))
         else
            fact=(1d0-y_ij_fks)/dot(pk,phat)
         endif
         w(0:3)=fact*(pk(0:3)/pkk-dot(ponium,pk)*phat(0:3)/pkk**2)
         contr=contr+dot(w,vec(0:3,kb))
      enddo
c the self term
      phat_low(0:3)=metric(0:3)*phat(0:3)
      zself=(0d0,0d0)
      do mu=0,3
         zself=zself+metric(mu)*ten(mu,mu)/pkk**2
         do nu=0,3
            zself=zself+dot(ponium,ponium)*phat_low(mu)*phat_low(nu)
     &           *ten(mu,nu)/pkk**4
         enddo
      enddo
      contr=contr-(1d0-y_ij_fks)*dble(zself)
      wgt=contr*g**2*iden_comp
c the coupling orders: those of the Born, and g^2
      ord(1:nsplitorders)=born_orders(1:nsplitorders)
      ord(qcd_pos)=ord(qcd_pos)+2
      ipos=orders_to_amp_split_pos(ord)
      amp_split(ipos)=amp_split(ipos)+wgt
      return
      end


      subroutine bsv_onia(p,xicut_used,wgt)
c**************************************************************************
c     The integrated soft counterterm of a P-wave quarkonium beyond the
c     colour-linked Borns (arXiv:2402.19221, sect. 4.3), finite part in the
c     conventions of bornsoftvirtual: momenta of the soft counter-event,
c     shat, Q_ES and xicut_used; (alpha_s/2 pi) times
c
c       sum_K [E(K,P) K_mu/(P.K) - E_mu({1,2},{K,P})] VEC^mu(K)
c       - [g_mu nu E(P,P)/P^2 + E_mu nu({2,2},{P,P})] TEN^{mu nu}
c       + (1P1 only) 2/P^2 sum_S' C_eff M(r_S')
c       + 8/(m_Q m_Qbar) sum_S' c_S' M(r_S') log(Q_ES^2/mu_NRQCD^2)
c
c     with P the onium momentum, E the finite parts of the eikonal
c     integrals (eikonal_Ireg, fks_onia_eikonal.f), VEC, TEN and
c     M(r_S') = BAUX (2S+1)/(2J+1) from born_onia.f, and the LDME
c     coefficients c_S' = 1/(2 N_c) ([18]), B_F ([88]), C_F ([81]) times
c     (2J+1)/9 for 3P_J. The 1P1 term is the remnant of -(2 eps - 2) of
c     the polarisation sum in d dimensions (paper, sect. 4.3.3).
c**************************************************************************
      use fks_phase_space_data, only: shat,p_born
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "q_es.inc"
      include "run.inc"
      double precision p(0:3,nexternal),xicut_used,wgt
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer naux_max
      parameter (naux_max=2)
      double precision pi,nc,cf,bf
      parameter (pi=3.1415926535897932385d0,nc=3d0,cf=4d0/3d0,
     &     bf=5d0/12d0)
      double precision vec(0:3,nexternal-1),baux(naux_max)
      double complex ten(0:3,0:3)
      integer ionium,ionium_r,kb,kr,mu,nu,a,naux,spin,jtot
      character*2 label(naux_max)
      double precision ceff(naux_max),mq,mqb,ponium(0:3),pk(0:3)
      double precision p2v(0:3),p1v(0:3),finv(0:3),t1(0:3,0:3)
      double precision t2(0:3,0:3),eik,contr,metric(0:3),onia_dot
      double precision mr,cldme,mun2,ktot2
      external onia_dot
      data metric /1d0,-1d0,-1d0,-1d0/

      wgt=0d0
      call born_onia_soft(p_born,vec,ten,baux,ionium)
      call born_onia_info(naux,spin,jtot,label,ceff,mq,mqb)
      ionium_r=ionium
      if (ionium.ge.i_fks) ionium_r=ionium+1
      ponium(0:3)=p(0:3,ionium_r)
      ktot2=onia_dot(ponium,ponium)
      contr=0d0
c the cross vectors
      do kb=1,nexternal-1
         if (all(vec(0:3,kb).eq.0d0)) cycle
         kr=kb
         if (kb.ge.i_fks) kr=kb+1
         pk(0:3)=p(0:3,kr)
         if (kr.eq.ionium_r) then
            call onia_eik_r1(ponium,ponium,.true.,shat,qes2,xicut_used,
     &           p2v,p1v,finv)
            contr=contr-onia_dot(finv,vec(0,kb))
         else
            call eikonal_Ireg(p,kr,ionium_r,xicut_used,eik)
            call onia_eik_r1(pk,ponium,.false.,shat,qes2,xicut_used,
     &           p2v,p1v,finv)
            contr=contr+eik*onia_dot(pk,vec(0,kb))/onia_dot(ponium,pk)
     &           -onia_dot(finv,vec(0,kb))
         endif
      enddo
c the self term
      call eikonal_Ireg(p,ionium_r,ionium_r,xicut_used,eik)
      call onia_eik_r2_self(ponium,shat,qes2,xicut_used,t1,t2)
      do mu=0,3
         contr=contr-metric(mu)*dble(ten(mu,mu))*eik/ktot2
         do nu=0,3
            contr=contr-metric(mu)*metric(nu)*t2(mu,nu)
     &           *dble(ten(mu,nu))
         enddo
      enddo
c the 1P1 remnant and the LDME counterterms
      if (mu_nrqcd_ref_fixed.gt.0d0) then
         mun2=mu_nrqcd_ref_fixed**2
      else
         mun2=((mq+mqb)/2d0)**2
      endif
      do a=1,naux
         mr=baux(a)*dble(2*spin+1)/dble(2*jtot+1)
         if (spin.eq.0) contr=contr+2d0/ktot2*ceff(a)*mr
         if (label(a).eq.'18') then
            cldme=1d0/(2d0*nc)
         elseif (label(a).eq.'81') then
            cldme=cf
         else
            cldme=bf
         endif
         if (spin.eq.1) cldme=cldme*dble(2*jtot+1)/9d0
         contr=contr+8d0/(mq*mqb)*cldme*mr*dlog(qes2/mun2)
      enddo
      wgt=g**2/(8d0*pi**2)*contr
      return
      end
