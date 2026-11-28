c**************************************************************************
c     The soft currents of P-wave quarkonia at NLO (FKS): arXiv:2402.19221
c     for one P-wave quarkonium, and its extension to several P-wave pairs
c     (JHEP_2Pwave: the squared soft current and the one-loop pole operator
c     are sums over pairs of legs, so every pair of P-wave quarkonia adds
c     its dipole-dipole terms). The Born-level ingredients are those of
c     BORN_ONIA_SOFT (born_onia.f, export_fks.write_born_onia_file), per
c     P-wave quarkonium H:
c       VEC(:,K,H)  the cross vectors of Born leg K (monopole-dipole),
c       TEN(:,:,H)  the self tensor sum_S' C_eff M_J^{mu nu}(r'),
c       BAUX(S',H)  -g_{mu nu} M_J^{mu nu}(r'),
c     and per pair IP = (H1, H2), mu of H1, nu of H2:
c       DD(:,:,IP)   the dipole-dipole interference of the real emission
c                    (cross class, eq. Mdipdip),
c       VPAR(:,:,IP) the doubly-dipole-linked Born of the virtual
c                    (parallel class, eq. MV).
c**************************************************************************

      subroutine sbornsoft_onia(pp,xi_i_fks,y_ij_fks,wgt)
c**************************************************************************
c     The soft currents of the P-wave quarkonia, added by sbornsoft to its
c     eikonal sum. For a soft gluon k_i = xi sqrt(s)/2 khat it returns the
c     limit of xi^2 (1-y) |M_real|^2 beyond the colour-linked Borns:
c
c       g^2 (1-y) { sum_H [ sum_K w_mu(K,P_H) VEC^mu(K,H)
c                   - [g_mu nu/(P_H.khat)^2 + P_H^2 khat_mu khat_nu
c                      /(P_H.khat)^4] TEN^{mu nu}(H) ]
c                 + sum_IP S_mu nu(P_H1,P_H2;khat) DD^{mu nu}(IP) }
c
c     with P_H the momentum of onium H, the sum over the coloured Born legs
c     K (the onia included) and
c       w_mu(K,P) = K_mu/(K.khat P.khat) - (P.K) khat_mu/(K.khat (P.khat)^2),
c       S_mu nu(K1,K2;k) = d^2/dK1^mu dK2^nu [K1.K2/(K1.k K2.k)]
c     (eqs. softkP, softPP, softP1P2, Sdipdip). For a massless j_fks the
c     collinear factor (1-y)/(k_j.khat) is replaced as in eikonal_reduced.
c     All terms vanish in the collinear limit y -> 1. Fills amp_split at
c     the position of the Born orders + g^2, like the colour-linked Borns.
c**************************************************************************
      use fks_phase_space_helpers, only: dot
      use fks_phase_space_data, only: p_born,resonance_momentum,
     &     resonance_recoil,p_i_fks_cnt,sqrtshat
      implicit none
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "coupl.inc"
      include "orders.inc"
      include "born_onia.inc"
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
      double precision vec(0:3,nexternal-1,npq),baux(nauxmax,npq)
      double precision dd(0:3,0:3,npairx)
      double complex ten(0:3,0:3,npq),vpar(0:3,0:3,npairx),zself
      integer ionium(npq),naux(npq),spin(npq),jtot(npq),ipair(2,npairx)
      character*2 label(nauxmax,npq)
      double precision ceff(nauxmax,npq),mq(npq),mqb(npq)
      integer h,ip,ir,ir1,ir2,kb,kr,mu,nu,ord(nsplitorders),ipos
      double precision phat(0:3),pk(0:3),w(0:3),ponium(0:3)
      double precision pkk,fact,contr,metric(0:3),phat_low(0:3)
      double precision k1(0:3),k2(0:3),k1l(0:3),k2l(0:3),a1,a2,x12
      double precision smn
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
      phat_low(0:3)=metric(0:3)*phat(0:3)
      call born_onia_soft(p_born,vec,ten,baux,dd,vpar)
      call born_onia_info(ionium,naux,spin,jtot,label,ceff,mq,mqb,ipair)
      contr=0d0
      do h=1,npq
         ir=ionium(h)
         if (ir.ge.i_fks) ir=ir+1
         ponium(0:3)=pp(0:3,ir)
         pkk=dot(ponium,phat)
c the cross vectors
         do kb=1,nexternal-1
            if (all(vec(0:3,kb,h).eq.0d0)) cycle
            kr=kb
            if (kb.ge.i_fks) kr=kb+1
            pk(0:3)=pp(0:3,kr)
            if (kr.eq.j_fks .and. pmass(j_fks).eq.zero) then
c              (1-y)/(k_j.khat) for the collinear partner
               fact=2d0/(sqrtshat*pp(0,j_fks))
               if (resonance_recoil)
     &              fact=2d0/dot(resonance_momentum,pp(0:3,j_fks))
            else
               fact=(1d0-y_ij_fks)/dot(pk,phat)
            endif
            w(0:3)=fact*(pk(0:3)/pkk-dot(ponium,pk)*phat(0:3)/pkk**2)
            contr=contr+dot(w,vec(0:3,kb,h))
         enddo
c the self term
         zself=(0d0,0d0)
         do mu=0,3
            zself=zself+metric(mu)*ten(mu,mu,h)/pkk**2
            do nu=0,3
               zself=zself+dot(ponium,ponium)*phat_low(mu)*phat_low(nu)
     &              *ten(mu,nu,h)/pkk**4
            enddo
         enddo
         contr=contr-(1d0-y_ij_fks)*dble(zself)
      enddo
c the dipole-dipole terms of the pairs of P-wave quarkonia
      do ip=1,npair
         ir1=ionium(ipair(1,ip))
         if (ir1.ge.i_fks) ir1=ir1+1
         ir2=ionium(ipair(2,ip))
         if (ir2.ge.i_fks) ir2=ir2+1
         k1(0:3)=pp(0:3,ir1)
         k2(0:3)=pp(0:3,ir2)
         k1l(0:3)=metric(0:3)*k1(0:3)
         k2l(0:3)=metric(0:3)*k2(0:3)
         a1=dot(k1,phat)
         a2=dot(k2,phat)
         x12=dot(k1,k2)
         do mu=0,3
            do nu=0,3
               smn=-k2l(mu)*phat_low(nu)/(a1*a2**2)
     &              -k1l(nu)*phat_low(mu)/(a1**2*a2)
     &              +x12*phat_low(mu)*phat_low(nu)/(a1**2*a2**2)
               if (mu.eq.nu) smn=smn+metric(mu)/(a1*a2)
               contr=contr+(1d0-y_ij_fks)*smn*dd(mu,nu,ip)
            enddo
         enddo
      enddo
      wgt=contr*g**2*iden_comp
c the coupling orders: those of the Born, and g^2
c (the Born has a single combination of coupling orders, amp_split
c position 1; born_orders is the generation constraint)
      call amp_split_pos_to_orders(1,ord)
      ord(qcd_pos)=ord(qcd_pos)+2
      ipos=orders_to_amp_split_pos(ord)
      amp_split(ipos)=amp_split(ipos)+wgt
      return
      end


      subroutine bsv_onia(p,xicut_used,wgt)
c**************************************************************************
c     The integrated soft counterterms of the P-wave quarkonia beyond the
c     colour-linked Borns, and the NRQCD counterterms (arXiv:2402.19221,
c     sect. 4.3; JHEP_2Pwave, sects. 5 and 6.2), finite parts in the
c     conventions of bornsoftvirtual: momenta of the soft counter-event,
c     shat, Q_ES and xicut_used; (alpha_s/2 pi) times
c
c       sum_H { sum_K [E(K,P) K_mu/(P.K) - E_mu({1,2},{K,P})] VEC^mu(K,H)
c               - [g_mu nu E(P,P)/P^2 + E_mu nu({2,2},{P,P})] TEN^{mu nu}(H)
c               + 2/P^2 sum_S' C_eff(S') eta_SJ M(r_S') log(Q_ES^2/mu_N^2) }
c     + sum_IP { Dbar_mu nu(P_1,P_2) DD^{mu nu}
c               - 2/(M_1 M_2) [f1 g_mu nu + f2 v2_mu v1_nu] DD^{mu nu}
c                 log(Q_ES^2/mu_N^2)
c               - 4/(M_1 M_2) Re{[f1c g_mu nu + f2c v2_mu v1_nu] VPAR^{mu nu}}
c                 log(Q_ES^2/mu_N^2) }
c
c     with P = P_H the onium momentum, E the finite parts of the eikonal
c     integrals (eikonal_Ireg, fks_onia_eikonal.f), Dbar the dipole-dipole
c     tensor (eq. Dbar, onia_eik_dd), f1, f2 (eqs. f1f2, f1f2c) of
c     omega = v1.v2 at the Born momenta, M(r_S') = BAUX (2S+1)/(2J+1) and
c     eta_SJ = 1 (1P1) or (2J+1)/9 (3P_J). The third line is the LDME
c     counterterm (eq. LDMECT) with kappa c_S' = 2 C_eff(S') eta_SJ/M^2
c     (corrections to arXiv:2402.19221, item 1, 2026-11-24; JHEP_2Pwave,
c     footnote of sect. 5.1): 8/(m_Q m_Qbar) {1/(2N_c), C_F, B_F} eta_SJ
c     for equal masses; for m_Q /= m_Qbar, kappa = 2 M^2/(m_Q m_Qbar)^2 for
c     [18], [81] and kappa (14/5 - 36 m_Q m_Qbar/(5 M^2)) for [88], whose
c     C_eff(88) = 7/6 Sigma^2 - 3 Sigma (Sigma = M^2/(m_Q m_Qbar)). Its
c     pole cancels that of the self term for any masses, transition by
c     transition, as the virtual has no intra-pair pole. The last two lines are the MS-bar counterterms of the
c     eight-quark operators, cross class (eq. CTcross, which cancels the
c     pole of Dbar) and parallel class (eq. CTparallel, which cancels the
c     dipole-dipole pole of the virtual). mu_N = mu_nrqcd_ref_fixed, or by
c     default the mean constituent mass (m_Q+m_Qbar)/2 of the quarkonium
c     (M_1+M_2)/4 for a pair).
c     There is no extra 1P1 term. The self term contracts the
c     d-dimensional rank-2 eikonal integral (its angular average
c     <n_i n_j> = delta_ij/(d-1)) with the 4-dimensional TEN: the
c     't Hooft-Veltman computation, 3 states normalised as the LDME
c     <O>/(N_C (2J+1)), d-dimensional soft gluon (JHEP_2Pwave, footnote
c     of sect. 3.2). Check: the d-dimensional real of g g -> 1P1[1] + g
c     (Larin's gamma5) integrated in d dimensions with the MS-bar NRQCD
c     counterterm gives the delta(1-z) constant -5/6 of HELAC-Onia,
c     which this reproduces.
c**************************************************************************
      use fks_phase_space_data, only: shat,p_born
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "q_es.inc"
      include "run.inc"
      include "born_onia.inc"
      double precision p(0:3,nexternal),xicut_used,wgt
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      double precision vec(0:3,nexternal-1,npq),baux(nauxmax,npq)
      double precision dd(0:3,0:3,npairx)
      double complex ten(0:3,0:3,npq),vpar(0:3,0:3,npairx)
      integer ionium(npq),naux(npq),spin(npq),jtot(npq),ipair(2,npairx)
      character*2 label(nauxmax,npq)
      double precision ceff(nauxmax,npq),mq(npq),mqb(npq)
      integer h,h1,h2,ip,ir,ir1,ir2,kb,kr,mu,nu,a
      double precision ponium(0:3),pk(0:3)
      double precision p2v(0:3),p1v(0:3),finv(0:3),t1(0:3,0:3)
      double precision t2(0:3,0:3),eik,contr,metric(0:3),onia_dot
      double precision mr,eta,mun2,ktot2,mm1,mm2,omega,f1,f2,f1i,f2i
      double precision k1(0:3),k2(0:3),v1l(0:3),v2l(0:3),dpole(0:3,0:3)
      double precision dfin(0:3,0:3),x1d,x2d,lnmu
      double complex x1p,x2p
      external onia_dot
      data metric /1d0,-1d0,-1d0,-1d0/

      wgt=0d0
      call born_onia_soft(p_born,vec,ten,baux,dd,vpar)
      call born_onia_info(ionium,naux,spin,jtot,label,ceff,mq,mqb,ipair)
      contr=0d0
      do h=1,npq
         ir=ionium(h)
         if (ir.ge.i_fks) ir=ir+1
         ponium(0:3)=p(0:3,ir)
         ktot2=onia_dot(ponium,ponium)
c the cross vectors
         do kb=1,nexternal-1
            if (all(vec(0:3,kb,h).eq.0d0)) cycle
            kr=kb
            if (kb.ge.i_fks) kr=kb+1
            pk(0:3)=p(0:3,kr)
            if (kr.eq.ir) then
               call onia_eik_r1(ponium,ponium,.true.,shat,qes2,
     &              xicut_used,p2v,p1v,finv)
               contr=contr-onia_dot(finv,vec(0,kb,h))
            else
               call eikonal_Ireg(p,kr,ir,xicut_used,eik)
               call onia_eik_r1(pk,ponium,.false.,shat,qes2,xicut_used,
     &              p2v,p1v,finv)
               contr=contr+eik*onia_dot(pk,vec(0,kb,h))
     &              /onia_dot(ponium,pk)-onia_dot(finv,vec(0,kb,h))
            endif
         enddo
c the self term
         call eikonal_Ireg(p,ir,ir,xicut_used,eik)
         call onia_eik_r2_self(ponium,shat,qes2,xicut_used,t1,t2)
         do mu=0,3
            contr=contr-metric(mu)*dble(ten(mu,mu,h))*eik/ktot2
            do nu=0,3
               contr=contr-metric(mu)*metric(nu)*t2(mu,nu)
     &              *dble(ten(mu,nu,h))
            enddo
         enddo
c the LDME counterterms
         if (mu_nrqcd_ref_fixed.gt.0d0) then
            mun2=mu_nrqcd_ref_fixed**2
         else
            mun2=((mq(h)+mqb(h))/2d0)**2
         endif
         eta=1d0
         if (spin(h).eq.1) eta=dble(2*jtot(h)+1)/9d0
         do a=1,naux(h)
            mr=baux(a,h)*dble(2*spin(h)+1)/dble(2*jtot(h)+1)
            contr=contr+2d0*ceff(a,h)/(mq(h)+mqb(h))**2*eta*mr
     &           *dlog(qes2/mun2)
         enddo
      enddo
c the dipole-dipole terms of the pairs of P-wave quarkonia
      do ip=1,npair
         h1=ipair(1,ip)
         h2=ipair(2,ip)
         ir1=ionium(h1)
         if (ir1.ge.i_fks) ir1=ir1+1
         ir2=ionium(h2)
         if (ir2.ge.i_fks) ir2=ir2+1
c        the integrated soft term
         call eikonal_Ireg(p,ir1,ir2,xicut_used,eik)
         call onia_eik_dd(p(0,ir1),p(0,ir2),eik,shat,qes2,xicut_used,
     &        dpole,dfin)
         do mu=0,3
            do nu=0,3
               contr=contr+metric(mu)*metric(nu)*dfin(mu,nu)
     &              *dd(mu,nu,ip)
            enddo
         enddo
c        the eight-quark counterterms, at the Born momenta
         mm1=mq(h1)+mqb(h1)
         mm2=mq(h2)+mqb(h2)
         if (mu_nrqcd_ref_fixed.gt.0d0) then
            mun2=mu_nrqcd_ref_fixed**2
         else
            mun2=((mm1+mm2)/4d0)**2
         endif
         lnmu=dlog(qes2/mun2)
         k1(0:3)=p_born(0:3,ionium(h1))
         k2(0:3)=p_born(0:3,ionium(h2))
         omega=onia_dot(k1,k2)/(mm1*mm2)
         call onia_f12(omega,f1,f2,f1i,f2i)
         v1l(0:3)=metric(0:3)*k1(0:3)/mm1
         v2l(0:3)=metric(0:3)*k2(0:3)/mm2
         x1d=0d0
         x2d=0d0
         x1p=(0d0,0d0)
         x2p=(0d0,0d0)
         do mu=0,3
            x1d=x1d+metric(mu)*dd(mu,mu,ip)
            x1p=x1p+metric(mu)*vpar(mu,mu,ip)
            do nu=0,3
               x2d=x2d+v2l(mu)*v1l(nu)*dd(mu,nu,ip)
               x2p=x2p+v2l(mu)*v1l(nu)*vpar(mu,nu,ip)
            enddo
         enddo
         contr=contr-2d0/(mm1*mm2)*(f1*x1d+f2*x2d)*lnmu
         contr=contr-4d0/(mm1*mm2)*(f1*dble(x1p)-f1i*dimag(x1p)
     &        +f2*dble(x2p)-f2i*dimag(x2p))*lnmu
      enddo
      wgt=g**2/(8d0*pi**2)*contr
      return
      end


      subroutine getpoles_onia(single)
c**************************************************************************
c     The single pole of the virtual of a Born with P-wave quarkonia
c     beyond the colour-linked Borns of getpoles (arXiv:2402.19221,
c     appendix B; JHEP_2Pwave, eq. Vpoles), in the normalisation of
c     getpoles: the coefficient of 1/eps in front of
c     (alpha_s/2 pi) (4 pi)^eps/Gamma(1-eps) (mu^2/Q_ES^2)^eps, times
c     alpha_s/(2 pi),
c
c       sum_H { sum_k k_mu VEC^mu(k,H)/(P.k)                (k massless)
c       + sum_k k_mu VEC^mu(k,H)/(P.k) m_k^2 P^2/(2 v_k^3 (P.k)^2)
c             [2 v_k/(1-v_k^2) - log((1+v_k)/(1-v_k))]     (k massive)
c     + sum_IP 4/(M_1 M_2) Re{[f1c g_mu nu + f2c v2_mu v1_nu] VPAR^{mu nu}}
c
c     with P = P_H, v_k = sqrt(1 - m_k^2 P^2/(P.k)^2), the sums over the
c     Born legs but onium H (the other octet onia are massive legs), and
c     VEC as in bsv_onia. The virtual has no self-energy-like (intra-pair)
c     pole for any masses (JHEP_2Pwave sect. 4.1; corrections to
c     arXiv:2402.19221, item 1: the last lines of its (B.4)-(B.8) do not
c     belong in the virtual): the pole of the integrated self term,
c     -2/(3 P^2) g_mu nu TEN^{mu nu}, and that of the LDME counterterm,
c     2/P^2 sum_S' C_eff eta_SJ M(r_S'), cancel each other in bsv_onia and
c     are left out, as are the cross-class poles (Dbar and its eight-quark
c     counterterm). The iPi parts of the massive-leg kernel, left out as in
c     appendix B, vanish after the spin and colour sums in
c     parity-conserving processes. The last line is
c     the dipole-dipole pole of the virtual (eq. Vdipdip), cancelled by
c     the parallel eight-quark counterterm of bsv_onia. There is no double
c     pole. Born momenta p_born.
c**************************************************************************
      use fks_phase_space_data, only: p_born
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "born_onia.inc"
      double precision single
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      double precision vec(0:3,nexternal-1,npq),baux(nauxmax,npq)
      double precision dd(0:3,0:3,npairx)
      double complex ten(0:3,0:3,npq),vpar(0:3,0:3,npairx)
      integer ionium(npq),naux(npq),spin(npq),jtot(npq),ipair(2,npairx)
      character*2 label(nauxmax,npq)
      double precision ceff(nauxmax,npq),mq(npq),mqb(npq)
      integer h,h1,h2,ip,kb,mu,nu,a
      double precision ponium(0:3),pk(0:3)
      double precision contr,metric(0:3),onia_dot,ktot2,pkk,mk2,v
      double precision mm1,mm2,omega,f1,f2,f1i,f2i,v1l(0:3)
      double precision v2l(0:3)
      double complex x1p,x2p
      external onia_dot
      data metric /1d0,-1d0,-1d0,-1d0/

      single=0d0
      call born_onia_soft(p_born,vec,ten,baux,dd,vpar)
      call born_onia_info(ionium,naux,spin,jtot,label,ceff,mq,mqb,ipair)
      contr=0d0
      do h=1,npq
         ponium(0:3)=p_born(0:3,ionium(h))
         ktot2=onia_dot(ponium,ponium)
c the cross vectors
         do kb=1,nexternal-1
            if (kb.eq.ionium(h)) cycle
            if (all(vec(0:3,kb,h).eq.0d0)) cycle
            pk(0:3)=p_born(0:3,kb)
            pkk=onia_dot(ponium,pk)
            mk2=onia_dot(pk,pk)
            if (abs(mk2).lt.1d-8*pk(0)**2) then
               contr=contr+onia_dot(pk,vec(0,kb,h))/pkk
            else
               v=dsqrt(1d0-mk2*ktot2/pkk**2)
               contr=contr+onia_dot(pk,vec(0,kb,h))/pkk
     &              *mk2*ktot2/(2d0*v**3*pkk**2)
     &              *(2d0*v/(1d0-v**2)-dlog((1d0+v)/(1d0-v)))
            endif
         enddo
      enddo
c the dipole-dipole pole of the virtual (parallel class)
      do ip=1,npair
         h1=ipair(1,ip)
         h2=ipair(2,ip)
         mm1=mq(h1)+mqb(h1)
         mm2=mq(h2)+mqb(h2)
         omega=onia_dot(p_born(0,ionium(h1)),p_born(0,ionium(h2)))
     &        /(mm1*mm2)
         call onia_f12(omega,f1,f2,f1i,f2i)
         v1l(0:3)=metric(0:3)*p_born(0:3,ionium(h1))/mm1
         v2l(0:3)=metric(0:3)*p_born(0:3,ionium(h2))/mm2
         x1p=(0d0,0d0)
         x2p=(0d0,0d0)
         do mu=0,3
            x1p=x1p+metric(mu)*vpar(mu,mu,ip)
            do nu=0,3
               x2p=x2p+v2l(mu)*v1l(nu)*vpar(mu,nu,ip)
            enddo
         enddo
         contr=contr+4d0/(mm1*mm2)*(f1*dble(x1p)-f1i*dimag(x1p)
     &        +f2*dble(x2p)-f2i*dimag(x2p))
      enddo
      single=g**2/(8d0*pi**2)*contr
      return
      end
