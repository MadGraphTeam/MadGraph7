      subroutine BinothLHA(p,born_wgt,virt_wgt)
c**************************************************************************
c     Built-in one-loop virtuals of 2 -> 1 Borns with a bound state
c     (madgraph/fks/fks_onia.py, builtin_virtual), linked as BinothLHA.f
c     for [QCD] processes with onia instead of the MadLoop interface.
c
c     Petrelli et al. (hep-ph/9707223, eqs. 121-133) write the virtual as
c       Born_D (alpha_s/pi) f_eps(M^2) [A/eps^2 + B/eps + b_0/eps_UV + C]
c     with f_eps(M^2) = (4 pi mu^2/M^2)^eps Gamma(1+eps), Born_D the
c     D-dimensional Born, the Coulomb term dropped (absorbed into the LDME).
c     After MSbar renormalisation of the n powers of alpha_s of the Born
c     and at mu = Q_ES, in the normalisation of getpoles,
c       (alpha_s/2 pi) (4 pi)^eps/Gamma(1-eps) (mu^2/Q_ES^2)^eps
c       x [2A/eps^2 + 2(B + A L)/eps
c          + 2C + A pi^2/3 + (2B + n b_0) L + A L^2] Born,
c     L = log(Q_ES^2/M^2), M^2 = (p1+p2)^2. The Born is taken at
c     alpha_s(Q_ES), as MadLoop does here (mu_r = Q_ES). Cases (A, B, C):
c       1  g g -> 1S0, 3P0, 3P2 [1]: (-C_A, -b_0, B_n)
c       2  g g -> 1S0, 3P0, 3P2 [8]: (-C_A, -b_0 - C_A/2, B_n)
c       3  q qbar -> 3S1[8]: (-C_F, -3/2 C_F - C_A/2,
c                             B_n - 10/9 n_f T_F - 16/9 T_F)
c          (-16/9 T_F: the bound heavy quark in the s-channel gluon,
c          decoupling scheme)
c       4  q qbar -> leptonium: (-C_F, -3/2 C_F, C_F (pi^2 - 8)/2
c                                + C_F pi^2/6), n = 0
c       5  vanishing Born: no virtual
c       0  none: stop, unless the Born vanishes at this point
c     with b_0 = (11 C_A - 2 n_f)/6 (n_f = NF of coupl.inc, as in
c     setfksfactor) and B_n = onia_virt_b (onia_virtual.inc).
c     The poles are checked against getpoles (appendix B of
c     arXiv:2402.19221 for P waves) on the first point, and always when
c     force_polecheck is set (check_poles).
c**************************************************************************
      use FKSParams
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include 'orders.inc'
      include 'onia_virtual.inc'
      double precision p(0:3,nexternal-1)
      double precision virt_wgt,born_wgt
      double precision pi,ca,cf,tf
      parameter (pi=3.1415926535897932385d0,ca=3d0,cf=4d0/3d0,tf=0.5d0)
      double precision qes2
      common /coupl_es/ qes2
      double precision amp_split_finite_ML(amp_split_size)
      common /to_amp_split_finite/amp_split_finite_ML
      double precision amp_split_poles_FKS(amp_split_size,2)
      common /to_amp_split_poles_FKS/amp_split_poles_FKS
      logical force_polecheck, polecheck_passed
      common /to_polecheck/force_polecheck, polecheck_passed
      integer ret_code_common
      common /to_ret_code/ret_code_common
      integer ord(nsplitorders),ipos,orders_to_amp_split_pos,nbad
      external orders_to_amp_split_pos
      double precision g_in,born_q,ao2pi,b0,a,b,c,l,m2,nas
      double precision double,single,finite,madfks_double,madfks_single
      double precision diff,scale
      logical firsttime
      save firsttime,nbad
      data firsttime,nbad /.true.,0/

      virt_wgt=0d0
      amp_split_finite_ML(1:amp_split_size)=0d0
      ret_code_common=0
      polecheck_passed=.true.
      if (onia_virt_case.eq.5) return
      if (onia_virt_case.eq.0) then
         if (born_wgt.eq.0d0) return
         write (*,*) 'ERROR: no built-in virtual for this process with'
     &        //' bound states (built-in: 2 -> 1 Borns, see'
     &        //' madgraph/fks/fks_onia.py). Replace BinothLHA.f in'
     &        //' this directory by your own virtual (template:'
     &        //' ../BinothLHA_user.f), or run [real=QCD].'
         stop 1
      endif
c the virtual at mu_r = Q_ES
      g_in=g
      mu_r=dsqrt(qes2)
      call update_as_param()
      ao2pi=g**2/(8d0*pi**2)
      born_q=born_wgt
      if (g_in.ne.0d0) born_q=born_wgt*(g/g_in)**born_orders(qcd_pos)
      nas=dble(born_orders(qcd_pos))/2d0
      b0=(11d0*ca-2d0*nf)/6d0
      if (onia_virt_case.eq.1) then
         a=-ca
         b=-b0
         c=onia_virt_b
      elseif (onia_virt_case.eq.2) then
         a=-ca
         b=-b0-ca/2d0
         c=onia_virt_b
      elseif (onia_virt_case.eq.3) then
         a=-cf
         b=-1.5d0*cf-ca/2d0
         c=onia_virt_b-10d0/9d0*nf*tf-16d0/9d0*tf
      elseif (onia_virt_case.eq.4) then
         a=-cf
         b=-1.5d0*cf
         c=cf*(pi**2-8d0)/2d0+cf*pi**2/6d0
      else
         write (*,*) 'ERROR in BinothLHA_onia: unknown case',
     &        onia_virt_case
         stop 1
      endif
      m2=(p(0,1)+p(0,2))**2-(p(1,1)+p(1,2))**2-(p(2,1)+p(2,2))**2
     &     -(p(3,1)+p(3,2))**2
      l=dlog(qes2/m2)
      double=ao2pi*born_q*2d0*a
      single=ao2pi*born_q*2d0*(b+a*l)
      finite=ao2pi*born_q*(2d0*c+a*pi**2/3d0+(2d0*b+nas*b0)*l+a*l**2)
      virt_wgt=finite
      ord(1:nsplitorders)=born_orders(1:nsplitorders)
      ord(qcd_pos)=ord(qcd_pos)+2
      ipos=orders_to_amp_split_pos(ord)
      amp_split_finite_ML(ipos)=finite
c the pole check against MadFKS
      if (firsttime .or. force_polecheck) then
         call getpoles(p,qes2,madfks_double,madfks_single,.true.)
         diff=dabs(single-madfks_single)+dabs(double-madfks_double)
         scale=dabs(single)+dabs(double)+dabs(madfks_single)
     &        +dabs(madfks_double)
         polecheck_passed=diff.le.IRPoleCheckThreshold*scale/2d0
         if (polecheck_passed) then
            if (firsttime) write(*,*) "---- POLES CANCELLED ----"
         else
            write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     &           IRPoleCheckThreshold
         endif
         if (firsttime .or. force_polecheck
     &        .or. .not.polecheck_passed) then
            write(*,*) " COEFFICIENT DOUBLE POLE:"
            write(*,*) "       MadFKS: ", madfks_double,
     &           "          OLP: ", double
            write(*,*) " COEFFICIENT SINGLE POLE:"
            write(*,*) "       MadFKS: ",madfks_single,
     &           "          OLP: ",single
            write(*,*) " FINITE:"
            write(*,*) "          OLP: ",virt_wgt
            write(*,*) "          BORN: ",born_wgt
         endif
         if (.not.polecheck_passed .and. .not.force_polecheck) then
            nbad=nbad+1
            if (nbad.ge.5) then
               write(*,*) " TOO MANY FAILURES, QUITTING"
               stop 1
            endif
         else
            firsttime=.false.
         endif
      endif
      return
      end


      subroutine BinothLHAInit(filename)
      implicit none
      character*13 filename
      return
      end


c MadLoop entry points called by check_poles (no MadLoop for onia)
      subroutine get_nsqso_loop(nsqso)
      implicit none
      integer nsqso
      nsqso=1
      end

      subroutine get_answer_dimension(n)
      implicit none
      integer n
      n=1
      end

      subroutine force_stability_check(flag)
      implicit none
      logical flag
      end

      subroutine collier_compute_uv_poles(flag)
      implicit none
      logical flag
      end

      subroutine collier_compute_ir_poles(flag)
      implicit none
      logical flag
      end
