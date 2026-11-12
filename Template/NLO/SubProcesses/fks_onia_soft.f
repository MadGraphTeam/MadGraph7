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
