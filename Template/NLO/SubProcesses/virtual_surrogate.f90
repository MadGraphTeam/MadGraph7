module virt_surrogate
! Per-channel fits of the finite virtual, frozen within each iteration.
!
! VirtTrickSurrogate 1 and 2 fit one surrogate, the 'fit' candidate below
! (colour dipoles; for 2 also the logarithms of all Born pairs).
! VirtTrickSurrogate 3 fits, on the same points, up to three candidates
! and uses in each channel the one with the smallest validation residual,
! or the grid when none beats it:
!   1 fit : the features of VirtTrickSurrogate 2;
!   2 IR  : V_IR, B and the loop coupling times B, where V_IR is the finite
!           part at mu^2 = Q_ES^2 of the one-loop infrared operator
!           (Catani; Catani, Dittmaier and Trocsanyi), built from the
!           colour links, or from the charge links for EW corrections;
!   3 DP  : B and the four Denner-Pozzorini EW Sudakov terms of the Born
!           (LSC, SSC, C, parameter renormalisation), for a process
!           generated with --ewsudakov.
! The IR features cost a few logarithms per link. The DP features cost the
! EW Sudakov matrix elements of the Born; they are computed only where the
! virtual is computed and where DP is the surrogate of the channel.
  implicit none
  private
  integer, parameter :: vs_maxfeat=94, vs_maxchan=20, vs_minpts=50
  integer, parameter, public :: vs_ncand=3
  integer, parameter :: vs_maxsfeat=6
  integer, parameter :: vs_nfeat_c(2:vs_ncand)=(/3,5/)
  character(len=3), parameter :: vs_cname(0:vs_ncand)=(/'grd','fit','IR ','DP '/)
  public :: vs_begin_point,vs_clear_links,vs_add_link,vs_add_ir_link,vs_features,vs_select
  public :: vs_reset,vs_value,vs_accumulate,vs_solve,vs_write_grids,vs_read_grids
  public :: vs_single_born_order,vs_ir_value,vs_dump,vs_any_featok
  public :: vs_dp_wanted,vs_set_dp,vs_dp_count
  public :: vs_maxfeat

! Current-point data shared by the matrix-element and weighted-integrand stages.
  logical, public :: vs_vcomp=.false.,vs_featok=.false.,vs_ratio_set=.false.
  double precision, public :: vs_virt=0d0,vs_born=0d0,vs_fvirt=1d0,vs_ratio=0d0
! A separate virtual event omits the approximation weight. One nonzero
! residual order supplies its normalization after PDFs, bias and folds.
  integer, public :: vs_residual_order=0
  double precision, public :: vs_residual_reference=0d0
! Features of the 'fit' candidate (vs_featok: they are available)
  integer, public :: vs_nfeat=4
  double precision, public :: vs_feat(vs_maxfeat)=0d0
! Features of the IR and DP candidates at the current point
  logical :: vs_sok(2:vs_ncand)=.false.
  double precision :: vs_sfeat(vs_maxsfeat,2:vs_ncand)=0d0
! VirtTrickSurrogate of the current run (3: choose among the candidates);
! vs_dp_avail: the process has the EW Sudakov matrix elements
  integer :: vs_mode=2
  logical, public :: vs_dp_avail=.false.
! number and CPU time of the DP feature evaluations (reported in the log)
  integer(kind=8) :: vs_dp_ncall=0
  double precision :: vs_dp_cpu=0d0
  logical :: have_links=.false.
  double precision :: dipole_features(3)=0d0
! Finite part at mu^2 = Q_ES^2 of the one-loop infrared structure, and the
! sum of the off-diagonal links, in units of the links of the code
! (g^2 <T_k.T_l> B); divided by 8 pi^2 they are in units of the virtual.
  double precision :: vs_ir=0d0, vs_linksum=0d0
! Per-point record of the links and Born legs (for the optional dump)
  integer, parameter, public :: vs_maxpair=200, vs_maxleg=20
  integer, public :: vs_npair=0, vs_pk(vs_maxpair)=0, vs_pl(vs_maxpair)=0
  double precision, public :: vs_pb(vs_maxpair)=0d0
  double precision, public :: vs_leg_gc(vs_maxleg)=0d0, vs_leg_m(vs_maxleg)=0d0
  double precision, parameter :: vs_pi=3.1415926535897932385d0
  integer :: vs_nin=2, vs_nleg=0
  double precision :: vs_qes2=0d0, vs_pborn(0:3,vs_maxleg)=0d0

! Normal equations and frozen coefficients for each integration channel:
! the 'fit' candidate (large), then the IR and DP candidates (small).
  double precision :: vs_xtx(vs_maxfeat,vs_maxfeat,vs_maxchan)=0d0, &
       vs_xty(vs_maxfeat,vs_maxchan)=0d0
  double precision :: vs_theta(vs_maxfeat,vs_maxchan)=0d0
  double precision :: vs_sxtx(vs_maxsfeat,vs_maxsfeat,vs_maxchan,2:vs_ncand)=0d0, &
       vs_sxty(vs_maxsfeat,vs_maxchan,2:vs_ncand)=0d0, &
       vs_stheta(vs_maxsfeat,vs_maxchan,2:vs_ncand)=0d0
  double precision :: vs_yty(vs_maxchan,vs_ncand)=0d0
  logical :: vs_fitted(vs_maxchan,vs_ncand)=.false.
! Surrogate used in each channel: 0 = grid, else the candidate
  integer :: vs_choice(vs_maxchan)=0
  integer :: vs_npt(vs_maxchan,vs_ncand)=0, vs_it_npt(vs_maxchan,vs_ncand)=0
! The first fit uses its training residual. Later choices compare each
! frozen candidate with the grid on the points observed after its fit.
  double precision :: vs_it_s2grid(vs_maxchan,vs_ncand)=0d0
  integer :: vs_cum_npt(vs_maxchan,vs_ncand)=0
  double precision :: vs_cum_s2grid(vs_maxchan,vs_ncand)=0d0, &
       vs_cum_s2fit(vs_maxchan,vs_ncand)=0d0, &
       vs_cum_neggrid(vs_maxchan,vs_ncand)=0d0, vs_cum_negfit(vs_maxchan,vs_ncand)=0d0

contains

  subroutine vs_begin_point
    vs_vcomp=.false.
    vs_featok=.false.
    vs_sok=.false.
    vs_ratio_set=.false.
    vs_virt=0d0
    vs_born=0d0
    vs_fvirt=1d0
    vs_ratio=0d0
    vs_residual_order=0
    vs_residual_reference=0d0
    call vs_clear_links
  end subroutine vs_begin_point

  logical function vs_single_born_order(born_weights,born) result(allowed)
! The scalar fit predicts one approximation A to the summed virtual. Its
! ratio A/B must therefore multiply exactly one virtual Born normalizer B.
! Broadcasting it to several orders would subtract several copies of A,
! while selection still compared V-A. A different Born normalizer also
! changes the applied approximation. Retain the per-order grid/polynomial
! in either case until the fit can validate those actual residual orders.
    double precision, intent(in) :: born_weights(:),born
    allowed=count(born_weights.ne.0d0).eq.1
    if (allowed) allowed=sum(born_weights).eq.born
    if (.not.allowed) call vs_begin_point
  end function vs_single_born_order

  logical function vs_any_featok()
    vs_any_featok=vs_featok.or.any(vs_sok)
  end function vs_any_featok

  subroutine vs_clear_links
    have_links=.false.
    dipole_features=0d0
    vs_ir=0d0
    vs_linksum=0d0
    vs_npair=0
  end subroutine vs_clear_links

  subroutine vs_add_ir_link(p,qes2,unresolved,m,n,bkl,mm,mn,gcm,gcn,nincoming)
! Adds the link (m,n) (real-process labels, n<=m in the caller's loop, m=n
! only for a massive parton) to the finite part at mu^2=Q_ES^2 of the
! one-loop infrared operator of Catani, Dittmaier and Trocsanyi
! (hep-ph/0011222, eq. 9, CDR, renormalised amplitude), with, for massless
! partons, the collinear term gamma_k/eps multiplied by (Q_ES^2/|s_kl|)^eps
! as in Catani (hep-ph/9802439), which gives -gamma_k/C_k <T_k.T_l> L_kl.
! bkl = g^2 <T_m.T_n> B, as passed to vs_add_link; gcm, gcn = gamma_k/C_k
! for massless coloured (or charged) partons, 0 otherwise; mm, mn the
! masses. The heavy-flavour decoupling term of the gluon is not included.
    double precision, intent(in) :: p(0:,:),qes2,bkl,mm,mn,gcm,gcn
    integer, intent(in) :: unresolved,m,n,nincoming
    integer :: k,l
    double precision :: pkl,s,el,v,lk,ll,f
    logical :: timelike
    if (bkl.eq.0d0.or.qes2.le.0d0) return
    vs_nin=nincoming
    k=m
    l=n
    if (k.gt.unresolved) k=k-1
    if (l.gt.unresolved) l=l-1
    if (vs_npair.lt.vs_maxpair) then
       vs_npair=vs_npair+1
       vs_pk(vs_npair)=k
       vs_pl(vs_npair)=l
       vs_pb(vs_npair)=bkl
    endif
    if (k.le.vs_maxleg) then
       vs_leg_gc(k)=gcm
       vs_leg_m(k)=mm
    endif
    if (l.le.vs_maxleg) then
       vs_leg_gc(l)=gcn
       vs_leg_m(l)=mn
    endif
    if (k.eq.l) then
! Gamma_Q of a massive quark: -C_F (1/2 ln(m^2/Q^2) - 2); the self-link of
! the code is g^2 C_F B/2 (checked against colour conservation)
       if (mm.gt.0d0) vs_ir=vs_ir-2d0*bkl*(0.5d0*log(mm**2/qes2)-2d0)
       return
    endif
    vs_linksum=vs_linksum+bkl
    pkl=p(0,k)*p(0,l)-p(1,k)*p(1,l)-p(2,k)*p(2,l)-p(3,k)*p(3,l)
    s=abs(2d0*pkl)
    if (s.le.0d0) return
    el=log(s/qes2)
    timelike=(k.le.nincoming).eqv.(l.le.nincoming)
    if (mm.eq.0d0.and.mn.eq.0d0) then
       f=el**2-(gcm+gcn)*el
       if (timelike) f=f-vs_pi**2
    elseif (mm.eq.0d0.or.mn.eq.0d0) then
       lk=log(max(mm,mn)**2/qes2)
       f=el**2-0.5d0*lk**2-vs_pi**2/6d0-(gcm+gcn)*el
       if (timelike) f=f-vs_pi**2
    else
       v=sqrt(max(0d0,1d0-(mm*mn/pkl)**2))
       lk=log(mm**2/s)
       ll=log(mn**2/s)
       if (v.gt.1d-6) then
          f=-el/v*log((1d0-v)/(1d0+v))
          if (timelike) f=f-vs_pi**2/v
       else
          f=2d0*el
       endif
       f=f-0.5d0*(lk**2+ll**2)-vs_pi**2/3d0
    endif
    vs_ir=vs_ir+bkl*f
  end subroutine vs_add_ir_link

  double precision function vs_ir_value()
! the infrared finite part in units of the virtual: b_kl = alpha_s/(2 pi)
! <T_k.T_l> B = bkl/(8 pi^2)
    vs_ir_value=vs_ir/(8d0*vs_pi**2)
  end function vs_ir_value

  subroutine vs_add_link(p,qes2,unresolved,m,n,bkl)
! Link indices belong to the real process; p contains the native Born legs.
    double precision, intent(in) :: p(0:,:),qes2,bkl
    integer, intent(in) :: unresolved,m,n
    integer :: k,l
    double precision :: logarithm
    if (bkl.eq.0d0.or.qes2.le.0d0) return
    k=m
    l=n
    if (k.gt.unresolved) k=k-1
    if (l.gt.unresolved) l=l-1
    have_links=.true.
    if (k.eq.l) then
       dipole_features(3)=dipole_features(3)+bkl
    else
       logarithm=log_invariant(p(:,k),p(:,l),qes2)
       dipole_features(1)=dipole_features(1)+bkl*logarithm
       dipole_features(2)=dipole_features(2)+bkl*logarithm**2
    endif
  end subroutine vs_add_link

  subroutine vs_features(p,born,qes2,mode)
    double precision, intent(in) :: p(0:,:),born,qes2
    integer, intent(in) :: mode
    integer :: nlegs,k,l,m
    double precision :: logarithm
    logical, save :: warned=.false.
    nlegs=size(p,2)
    vs_mode=mode
    vs_qes2=qes2
    vs_nleg=min(nlegs,vs_maxleg)
    vs_pborn(0:3,1:vs_nleg)=p(0:3,1:vs_nleg)
    vs_nfeat=4
    if (mode.ge.2) then
       if (4+nlegs*(nlegs-1).le.vs_maxfeat) then
          vs_nfeat=4+nlegs*(nlegs-1)
       else
          if (.not.warned) write(*,*) 'Too many Born particles for '// &
               'VirtTrickSurrogate>=2: using the dipole features only'
          warned=.true.
       endif
    endif
    vs_feat(1)=born
    vs_feat(2:vs_nfeat)=0d0
    vs_featok=have_links.and.qes2.gt.0d0
    vs_sok=.false.
    if (.not.vs_featok) return
! IR candidate: A = t1 V_IR + t2 B + t3 (sum of the links), the last being
! -sum_k C_k (alpha/2pi) B, i.e. the loop coupling of the point times B
    if (mode.eq.3) then
       vs_sfeat(1:3,2)=(/vs_ir_value(),born,vs_linksum/(8d0*vs_pi**2)/)
       vs_sok(2)=.true.
    endif
    vs_feat(2:4)=dipole_features
    if (vs_nfeat.eq.4) return
    m=4
    do k=1,nlegs-1
       do l=k+1,nlegs
          logarithm=log_invariant(p(:,k),p(:,l),qes2)
          vs_feat(m+1)=born*logarithm
          vs_feat(m+2)=born*logarithm**2
          m=m+2
       enddo
    enddo
  end subroutine vs_features

  logical function vs_dp_wanted(kchan,vcomp)
! The DP features are needed at this point: to train and validate (where
! the virtual is computed, vcomp) or because DP is the channel's surrogate.
    integer, intent(in) :: kchan
    logical, intent(in) :: vcomp
    vs_dp_wanted=.false.
    if (vs_mode.ne.3.or..not.vs_dp_avail.or.vs_sok(3)) return
    if (kchan.lt.1.or.kchan.gt.vs_maxchan) return
    vs_dp_wanted=vcomp.or.vs_choice(kchan).eq.3
  end function vs_dp_wanted

  subroutine vs_dp_count(cpu)
    double precision, intent(in) :: cpu
    vs_dp_ncall=vs_dp_ncall+1
    vs_dp_cpu=vs_dp_cpu+cpu
  end subroutine vs_dp_count

  subroutine vs_set_dp(born,dp)
! dp = LSC, SSC, C and parameter-renormalisation terms, units of the virtual
    double precision, intent(in) :: born,dp(4)
    vs_sfeat(1,3)=born
    vs_sfeat(2:5,3)=dp
    vs_sok(3)=all(dp.eq.dp)
  end subroutine vs_set_dp

  double precision function log_invariant(p,q,qes2) result(logarithm)
    double precision, intent(in) :: p(0:3),q(0:3),qes2
    double precision :: invariant
    invariant=abs(2d0*(p(0)*q(0)-p(1)*q(1)-p(2)*q(2)-p(3)*q(3)))
    logarithm=0d0
    if (invariant.gt.0d0) logarithm=log(invariant/qes2)
  end function log_invariant

  logical function cand_ok(c)
    integer, intent(in) :: c
    if (c.eq.1) then
       cand_ok=vs_featok
    else
       cand_ok=vs_sok(c)
    endif
  end function cand_ok

  subroutine vs_select(kchan,born)
    integer, intent(in) :: kchan
    double precision, intent(in) :: born
    vs_ratio_set=.false.
    if (born.eq.0d0) return
    if (kchan.lt.1.or.kchan.gt.vs_maxchan) return
    if (vs_choice(kchan).eq.0) return
    if (.not.cand_ok(vs_choice(kchan))) return
    vs_ratio=vs_value(kchan)/born
    vs_ratio_set=.true.
  end subroutine vs_select

  subroutine vs_reset
! A fresh grid setup must not retain a fit from a previous integration.
    call vs_begin_point
    vs_nfeat=4
    vs_feat=0d0
    vs_sfeat=0d0
    vs_xtx=0d0
    vs_xty=0d0
    vs_sxtx=0d0
    vs_sxty=0d0
    vs_yty=0d0
    vs_theta=0d0
    vs_stheta=0d0
    vs_fitted=.false.
    vs_choice=0
    vs_npt=0
    vs_it_npt=0
    vs_it_s2grid=0d0
    vs_cum_npt=0
    vs_cum_s2grid=0d0
    vs_cum_s2fit=0d0
    vs_cum_neggrid=0d0
    vs_cum_negfit=0d0
  end subroutine vs_reset

  double precision function cand_value(c,kchan)
    integer, intent(in) :: c,kchan
    integer :: nf
    if (c.eq.1) then
       cand_value=dot_product(vs_theta(1:vs_nfeat,kchan),vs_feat(1:vs_nfeat))
    else
       nf=vs_nfeat_c(c)
       cand_value=dot_product(vs_stheta(1:nf,kchan,c),vs_sfeat(1:nf,c))
    endif
  end function cand_value

  double precision function vs_value(kchan)
! the surrogate of channel kchan at the current point (matrix-element
! level): the chosen candidate, or the 'fit' candidate if none is chosen
    integer, intent(in) :: kchan
    vs_value=cand_value(max(1,vs_choice(kchan)),kchan)
  end function vs_value

  subroutine vs_accumulate(kchan,virt,surr_grid,w)
! add the current point (virtual 'virt' and grid surrogate 'surr_grid', at
! matrix-element level, weight w) to the fits of channel kchan
    integer, intent(in) :: kchan
    double precision, intent(in) :: virt,surr_grid,w
    integer :: i,c,nf
    double precision :: afit
    if (kchan.lt.1 .or. kchan.gt.vs_maxchan) return
    do c=1,vs_ncand
       if (.not.cand_ok(c)) cycle
       if (c.eq.1) then
          nf=vs_nfeat
          do i=1,nf
             vs_xtx(1:nf,i,kchan)=vs_xtx(1:nf,i,kchan)+w*vs_feat(1:nf)*vs_feat(i)
          enddo
          vs_xty(1:nf,kchan)=vs_xty(1:nf,kchan)+w*vs_feat(1:nf)*virt
       else
          nf=vs_nfeat_c(c)
          do i=1,nf
             vs_sxtx(1:nf,i,kchan,c)=vs_sxtx(1:nf,i,kchan,c)+w*vs_sfeat(1:nf,c)*vs_sfeat(i,c)
          enddo
          vs_sxty(1:nf,kchan,c)=vs_sxty(1:nf,kchan,c)+w*vs_sfeat(1:nf,c)*virt
       endif
       vs_yty(kchan,c)=vs_yty(kchan,c)+w*virt**2
       vs_npt(kchan,c)=vs_npt(kchan,c)+1
       vs_it_npt(kchan,c)=vs_it_npt(kchan,c)+1
       vs_it_s2grid(kchan,c)=vs_it_s2grid(kchan,c)+w*(virt-surr_grid)**2
       if (vs_fitted(kchan,c)) then
          afit=cand_value(c,kchan)
          vs_cum_npt(kchan,c)=vs_cum_npt(kchan,c)+1
          vs_cum_s2fit(kchan,c)=vs_cum_s2fit(kchan,c)+w*(virt-afit)**2
          vs_cum_s2grid(kchan,c)=vs_cum_s2grid(kchan,c)+w*(virt-surr_grid)**2
          vs_cum_negfit(kchan,c)=vs_cum_negfit(kchan,c)+sqrt(w)*max(0d0,afit-virt)
          vs_cum_neggrid(kchan,c)=vs_cum_neggrid(kchan,c)+sqrt(w)*max(0d0,surr_grid-virt)
       endif
    enddo
  end subroutine vs_accumulate

  subroutine solve_normal(nf,xtx,xty,b,ok)
! weighted least squares from the normal equations, with a small ridge for
! (nearly) degenerate features; Gaussian elimination, partial pivoting
    integer, intent(in) :: nf
    double precision, intent(in) :: xtx(:,:),xty(:)
    double precision, intent(out) :: b(nf)
    logical, intent(out) :: ok
    integer :: i,j,k
    double precision :: a(nf,nf),ridge,piv,fac
    ok=.false.
    a(1:nf,1:nf)=xtx(1:nf,1:nf)
    b(1:nf)=xty(1:nf)
    ridge=0d0
    do i=1,nf
       ridge=ridge+a(i,i)
    enddo
    ridge=1d-9*ridge/nf
    do i=1,nf
       a(i,i)=a(i,i)+ridge
    enddo
    do k=1,nf
       j=k
       do i=k+1,nf
          if (abs(a(i,k)).gt.abs(a(j,k))) j=i
       enddo
       if (j.ne.k) then
          do i=1,nf
             piv=a(k,i)
             a(k,i)=a(j,i)
             a(j,i)=piv
          enddo
          piv=b(k)
          b(k)=b(j)
          b(j)=piv
       endif
       if (a(k,k).eq.0d0) return
       do i=k+1,nf
          fac=a(i,k)/a(k,k)
          a(i,k:nf)=a(i,k:nf)-fac*a(k,k:nf)
          b(i)=b(i)-fac*b(k)
       enddo
    enddo
    do k=nf,1,-1
       b(k)=(b(k)-dot_product(a(k,k+1:nf),b(k+1:nf)))/a(k,k)
    enddo
    ok=.true.
  end subroutine solve_normal

  subroutine vs_solve(kchan,signaware)
! end of an iteration: refit each candidate of channel kchan on all its
! points so far and choose the surrogate for the next iteration: the
! candidate whose residual, relative to the grid's on the same points, is
! the smallest and below 1, else the grid (signaware: compare the weights
! of the negative residuals, else the variances)
    integer, intent(in) :: kchan
    logical, intent(in) :: signaware
    integer :: c,nf,best
    double precision :: b(vs_maxfeat),s2fit,r,rbest
    double precision :: rc(vs_ncand)
    logical :: ok,solved,on
    if (kchan.lt.1 .or. kchan.gt.vs_maxchan) return
    solved=.false.
    best=0
    rbest=1d0
    rc=-1d0
    do c=1,vs_ncand
       if (c.eq.1) then
          nf=vs_nfeat
       else
          nf=vs_nfeat_c(c)
       endif
       if (vs_npt(kchan,c).lt.max(vs_minpts,3*nf)) cycle
       if (c.eq.1) then
          call solve_normal(nf,vs_xtx(:,:,kchan),vs_xty(:,kchan),b(1:nf),ok)
       else
          call solve_normal(nf,vs_sxtx(:,:,kchan,c),vs_sxty(:,kchan,c),b(1:nf),ok)
       endif
       if (.not.ok) cycle
       on=.false.
       r=-1d0
       if (vs_cum_npt(kchan,c).gt.0) then
          if (signaware) then
             on=vs_cum_negfit(kchan,c).lt.vs_cum_neggrid(kchan,c)
             if (on) r=vs_cum_negfit(kchan,c)/vs_cum_neggrid(kchan,c)
          else
             on=vs_cum_s2fit(kchan,c).lt.vs_cum_s2grid(kchan,c)
             if (on) r=vs_cum_s2fit(kchan,c)/vs_cum_s2grid(kchan,c)
          endif
       elseif (vs_it_npt(kchan,c).gt.0) then
          ! first fit: in-sample residual of the fit against the grid's
          if (c.eq.1) then
             s2fit=vs_yty(kchan,c)-2d0*dot_product(b(1:nf),vs_xty(1:nf,kchan)) &
                  +dot_product(b(1:nf),matmul(vs_xtx(1:nf,1:nf,kchan),b(1:nf)))
          else
             s2fit=vs_yty(kchan,c)-2d0*dot_product(b(1:nf),vs_sxty(1:nf,kchan,c)) &
                  +dot_product(b(1:nf),matmul(vs_sxtx(1:nf,1:nf,kchan,c),b(1:nf)))
          endif
          on=s2fit/vs_npt(kchan,c).lt.vs_it_s2grid(kchan,c)/vs_it_npt(kchan,c)
          if (on) r=(s2fit/vs_npt(kchan,c))/(vs_it_s2grid(kchan,c)/vs_it_npt(kchan,c))
       else
          ! no new point since the last fit: keep the previous decision
          on=vs_choice(kchan).eq.c
          if (on) r=0d0
       endif
       if (c.eq.1) then
          vs_theta(1:nf,kchan)=b(1:nf)
       else
          vs_stheta(1:nf,kchan,c)=b(1:nf)
       endif
       vs_fitted(kchan,c)=.true.
       solved=.true.
       rc(c)=r
       if (on.and.(best.eq.0.or.r.lt.rbest)) then
          best=c
          rbest=r
       endif
    enddo
    if (solved) then
       vs_choice(kchan)=best
       write (*,'(a,i3,a,i7,a,l2)') ' fitted surrogate of the virtual, channel',kchan, &
            ': points',vs_npt(kchan,1),', used',best.gt.0
       if (vs_mode.eq.3) write (*,'(a,i3,a,a,3(1x,a,1x,f8.4),a,i10,a,es10.3,a)') &
            ' virtual surrogate candidates, channel',kchan,': choice ',vs_cname(best), &
            (vs_cname(c),rc(c),c=1,vs_ncand),'; DP features',vs_dp_ncall,' calls,', &
            vs_dp_cpu,' s'
    endif
    vs_it_npt(kchan,:)=0
    vs_it_s2grid(kchan,:)=0d0
  end subroutine vs_solve

  subroutine vs_dump(kchan,w,agrid)
! Optional record of each grid-setup point that enters the fit, written to
! vt_dump.dat when the environment variable VT_DUMP is 1: channel, weight w,
! V, B, grid surrogate, Q_ES^2, V_IR, sum of the links/(8 pi^2), number of
! fit features, features, Born legs (momentum, mass, gamma/C), links
! (k, l, g^2 <T_k.T_l> B), and the DP features (flag, LSC, SSC, C, PR).
    integer, intent(in) :: kchan
    double precision, intent(in) :: w,agrid
    logical, save :: first=.true., on=.false.
    integer, save :: lun=0
    character(len=8) :: env
    integer :: ios,i,k
    double precision :: rec(17+vs_maxfeat+6*vs_maxleg+3*vs_maxpair)
    if (first) then
       first=.false.
       call get_environment_variable('VT_DUMP',env,status=ios)
       on=ios.eq.0.and.env(1:1).eq.'1'
       if (on) open(newunit=lun,file='vt_dump.dat',status='replace')
    endif
    if (.not.on) return
    rec(1:9)=(/dble(kchan),w,vs_virt,vs_born,agrid,vs_qes2,vs_ir_value(), &
         vs_linksum/(8d0*vs_pi**2),dble(vs_nfeat)/)
    k=9
    rec(k+1:k+vs_nfeat)=vs_feat(1:vs_nfeat)
    k=k+vs_nfeat
    rec(k+1:k+2)=(/dble(vs_nleg),dble(vs_nin)/)
    k=k+2
    do i=1,vs_nleg
       rec(k+1:k+4)=vs_pborn(0:3,i)
       rec(k+5:k+6)=(/vs_leg_m(i),vs_leg_gc(i)/)
       k=k+6
    enddo
    rec(k+1)=dble(vs_npair)
    k=k+1
    do i=1,vs_npair
       rec(k+1:k+3)=(/dble(vs_pk(i)),dble(vs_pl(i)),vs_pb(i)/)
       k=k+3
    enddo
    rec(k+1)=merge(1d0,0d0,vs_sok(3))
    rec(k+2:k+5)=vs_sfeat(2:5,3)
    k=k+5
    write(lun,'(*(es24.16e3,1x))') rec(1:k)
    flush(lun)
  end subroutine vs_dump

  subroutine vs_write_grids(lun,nchans)
! VSU records follow the grid/polynomial fallback in mint_grids:
! 'VSU', fit used, fit fitted, nfeat, its coefficients, then, for
! VirtTrickSurrogate 3, the choice and for the IR and DP candidates
! fitted, nfeat and coefficients (ignored by older readers)
    integer, intent(in) :: lun,nchans
    integer :: kchan,c,nf
    character(len=8192) :: line
    character(len=40) :: item
    do kchan=1,nchans
       if (vs_mode.ne.3) then
          write(lun,*) 'VSU',merge(1,0,vs_choice(kchan).eq.1),merge(1,0,vs_fitted(kchan,1)), &
               vs_nfeat,vs_theta(1:vs_nfeat,kchan)
          cycle
       endif
       write(line,*) 'VSU',merge(1,0,vs_choice(kchan).eq.1),merge(1,0,vs_fitted(kchan,1)), &
            vs_nfeat,vs_theta(1:vs_nfeat,kchan),vs_choice(kchan)
       do c=2,vs_ncand
          nf=vs_nfeat_c(c)
          write(item,*) merge(1,0,vs_fitted(kchan,c)),nf
          line=trim(line)//' '//trim(adjustl(item))
          write(line(len_trim(line)+1:),*) vs_stheta(1:nf,kchan,c)
       enddo
       write(lun,'(a)') trim(line)
    enddo
  end subroutine vs_write_grids

  subroutine vs_read_grids(lun,nchans,enabled)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: lun,nchans
    logical, intent(in) :: enabled
    integer :: kchan,ios,ion,ifit,nfeat,i,ichoice,c,nf,ifc(2:vs_ncand),nfc(2:vs_ncand)
    double precision :: th(vs_maxsfeat,2:vs_ncand),dummy(vs_maxfeat)
    character(len=3) :: tag
    character(len=8192) :: line
    do kchan=1,nchans
       read(lun,'(a)',iostat=ios) line
       if (ios.ne.0) call grid_error('Missing virtual surrogate in mint_grids.')
       read(line,*,iostat=ios) tag,ion,ifit,nfeat
       if (ios.ne.0) call grid_error('Malformed virtual surrogate in mint_grids.')
       if (tag.ne.'VSU'.or.nfeat.lt.1.or.nfeat.gt.vs_maxfeat.or. &
            (ion.ne.0.and.ion.ne.1).or.(ifit.ne.0.and.ifit.ne.1)) &
            call grid_error('Invalid virtual surrogate in mint_grids.')
       read(line,*,iostat=ios) tag,ion,ifit,nfeat,(vs_theta(i,kchan),i=1,nfeat)
       if (ios.ne.0) call grid_error('Incomplete virtual surrogate in mint_grids.')
       if (.not.all(ieee_is_finite(vs_theta(1:nfeat,kchan)))) &
            call grid_error('Nonfinite virtual surrogate in mint_grids.')
       vs_nfeat=nfeat
       ichoice=ion
       vs_fitted(kchan,1)=ifit.eq.1
       vs_fitted(kchan,2:vs_ncand)=.false.
! VirtTrickSurrogate 3: choice and the IR and DP candidates
       read(line,*,iostat=ios) tag,ion,ifit,nfeat,(dummy(i),i=1,nfeat),ichoice, &
            (ifc(c),nfc(c),(th(i,c),i=1,vs_nfeat_c(c)),c=2,vs_ncand)
       if (ios.eq.0) then
          if (ichoice.lt.0.or.ichoice.gt.vs_ncand.or.any(nfc.ne.vs_nfeat_c)) &
               call grid_error('Invalid virtual surrogate candidates in mint_grids.')
          do c=2,vs_ncand
             nf=vs_nfeat_c(c)
             if (.not.all(ieee_is_finite(th(1:nf,c)))) &
                  call grid_error('Nonfinite virtual surrogate in mint_grids.')
             vs_stheta(1:nf,kchan,c)=th(1:nf,c)
             vs_fitted(kchan,c)=ifc(c).eq.1
          enddo
       else
          ichoice=ion
       endif
       vs_choice(kchan)=0
       if (enabled) vs_choice(kchan)=ichoice
    enddo
  end subroutine vs_read_grids

  subroutine grid_error(message)
    character(len=*), intent(in) :: message
    write(*,*) 'ERROR: ',message
    stop 1
  end subroutine grid_error

end module virt_surrogate
