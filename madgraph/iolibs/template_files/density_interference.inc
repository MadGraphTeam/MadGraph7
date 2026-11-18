C Exact spectator sum of the linear interference, retaining all colours.
C The physical Born is kept. Longitudinal corrections are mapped from the
C Goldstone amplitudes with HELAS equivalence phases. With lrel=1 the
C second output is a spectator-resolved Goldstone-normalised projection
C for comparison, and zero_gold counts the states where it is undefined;
C with lrel=0 both stay zero. Only the first nvar variants are evaluated;
C the others stay zero.

      subroutine sdk_density_interference(p,nres,pos,dims,nvar,lrel,
     $     born,correction,relative,zero_gold,status)
      implicit none
      include 'nexternal.inc'
      %(constants)s
      integer nres,pos(10),dims(10),nvar,lrel,zero_gold,status
      double precision p(0:3,nexternal-1)
      double complex born(36,36),correction(36,36,5)
      double complex relative(36,36,5)
      integer NCOMB
      parameter (NCOMB=%(ncomb)d)
      integer nhel(nexternal-1,NCOMB),i
      %(helicity_lines)s
      integer ispec,h,hp,r,k,d,iv,s,ip,ib,base(36),source_base(NS)
      data source_base /%(source_bases)s/
      logical polarised(nexternal-1)
      data polarised /%(polarised)s/
      integer hels(nexternal-1,36),iflist(nexternal-1)
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      %(den_factor_lines)s
      double precision invariants(nexternal-1,nexternal-1,5),norm(36)
      double complex amps(NC,NO,NS,36),deriv(NC,NO,6,36)
      double complex ub(NC,NO,NS,36)
      double complex coeff(NS),parcoeff(6,5),phase(36)
      double complex delta(36,5),mixed(36,36,5),brow(36,36)
      double complex xs(36,NS),ys(NS),xd(36,6),yd(6)
      double complex sdk_density_inter,sdk_density_dot,term
      external sdk_density_inter,sdk_density_dot
      logical dep(6,NB),used(NS),ok
      integer sud_mod,saved_mod
      common/to_sud_mod/sud_mod
      logical s_to_rij,rij_ge_mw,saved_stor,saved_rij
      common/to_s_to_rij/s_to_rij
      common/rij_ge_mw/rij_ge_mw

      born=(0d0,0d0)
      correction=(0d0,0d0)
      relative=(0d0,0d0)
      zero_gold=0
C One to five variants, distinct resonance positions with two or three
C spin states each, and at most 36 resonance spin states in total.
      status=1
      if (nvar.lt.1.or.nvar.gt.5) return
      if (lrel.lt.0.or.lrel.gt.1) return
      if (nres.lt.1.or.nres.gt.10) return
      d=1
      do r=1,nres
        if (dims(r).lt.2.or.dims(r).gt.3) return
        if (pos(r).lt.1.or.pos(r).gt.nexternal-1) return
        if (any(pos(1:r-1).eq.pos(r))) return
        if (d.gt.36/dims(r)) return
        d=d*dims(r)
      enddo
C The helicity table restricts a polarised leg, so it cannot be retained.
      status=4
      if (any(polarised(pos(1:nres)))) return
      status=0
      saved_mod=sud_mod
      saved_stor=s_to_rij
      saved_rij=rij_ge_mw
      iflist=(/%(iflist)s/)
C Each variant builds its invariants with its own floor.
      do iv=1,nvar
        call sdk_density_variant(iv)
        call sdk_get_invariants(p,iflist,invariants(1,1,iv))
        call sdk_density_parcoeff(invariants(1,1,iv),parcoeff(1,iv))
      enddo
C The input parameters that each base amplitude depends on.
      call sdk_density_dependence(dep,ok)
      if (.not.ok) then
        status=3
        goto 900
      endif

C One representative of each spectator tuple. No good-helicity filtering:
C a zero Born component can have a nonzero mixed-helicity interference.
C Every amplitude read below is evaluated first for this tuple.
      do ispec=1,NCOMB
        if (any(nhel(pos(1:nres),ispec).ne.-1)) cycle
        do h=1,d
          hels(:,h)=nhel(:,ispec)
          k=h-1
          do r=1,nres
            ip=mod(k,dims(r))
            k=k/dims(r)
            if (dims(r).eq.2) then
              hels(pos(r),h)=2*ip-1
            else
              hels(pos(r),h)=ip-1
            endif
          enddo
          call sdk_density_base(hels(1,h),base(h),phase(h))
          call sdk_density_eval(1,p,hels(1,h),amps(1,1,1,h))
        enddo
C Every term has a Born amplitude of this tuple on its conjugated side.
        if (all(amps(:,:,1,1:d).eq.(0d0,0d0))) cycle
        used=.false.
        used(1)=.true.
        do h=1,d
          ib=base(h)
          if (ib.eq.0) cycle
          do s=2,NS
            if (source_base(s).ne.ib) cycle
            used(s)=.true.
            call sdk_density_eval(s,p,hels(1,h),amps(1,1,s,h))
          enddo
        enddo
        call sdk_density_derivatives(p,d,base,hels,dep,deriv)
C The conjugated Born side of the colour contractions, once per tuple.
        do hp=1,d
          do s=1,NS
            if (used(s))
     $        call sdk_density_left(1,s,amps(1,1,1,hp),ub(1,1,s,hp))
          enddo
        enddo
        do h=1,d
          do hp=1,d
            brow(h,hp)=sdk_density_dot(ub(1,1,1,hp),amps(1,1,1,h))
          enddo
          if (lrel.eq.0) cycle
          ib=base(h)
          if (ib.eq.0) then
            norm(h)=0d0
          else
            norm(h)=dble(sdk_density_inter(ib,ib,
     $                           amps(1,1,ib,h),amps(1,1,ib,h)))
          endif
          if (norm(h).le.0d0.and.dble(brow(h,h)).gt.0d0)
     $        zero_gold=zero_gold+1
        enddo
        mixed(1:d,1:d,:)=(0d0,0d0)
        delta(1:d,:)=(0d0,0d0)
        do h=1,d
          ib=base(h)
          if (ib.eq.0) cycle
C The colour contractions do not depend on the variant. coeff(s) can be
C nonzero only for the sources of this base, a derivative only for the
C parameters the base depends on.
          do s=1,NS
            if (source_base(s).ne.ib) cycle
            if (lrel.eq.1) ys(s)=sdk_density_inter(ib,s,amps(1,1,ib,h),
     $                                             amps(1,1,s,h))
            do hp=1,d
              xs(hp,s)=sdk_density_dot(ub(1,1,s,hp),amps(1,1,s,h))
            enddo
          enddo
          do ip=1,6
            if (.not.dep(ip,ib)) cycle
            if (lrel.eq.1) yd(ip)=sdk_density_inter(ib,ib,
     $                               amps(1,1,ib,h),deriv(1,1,ip,h))
            do hp=1,d
              xd(hp,ip)=sdk_density_dot(ub(1,1,ib,hp),deriv(1,1,ip,h))
            enddo
          enddo
          do iv=1,nvar
            call sdk_density_variant(iv)
            call sdk_density_coeff(ib,hels(1,h),invariants(1,1,iv),
     $                             coeff)
            term=(0d0,0d0)
            do s=1,NS
              if (coeff(s).eq.(0d0,0d0)) cycle
              if (lrel.eq.1) term=term+coeff(s)*ys(s)
              do hp=1,d
                mixed(h,hp,iv)=mixed(h,hp,iv)+phase(h)*coeff(s)*
     $            xs(hp,s)
              enddo
            enddo
            do ip=1,6
              if (.not.dep(ip,ib)) cycle
              if (lrel.eq.1) term=term+parcoeff(ip,iv)*yd(ip)
              do hp=1,d
                mixed(h,hp,iv)=mixed(h,hp,iv)+phase(h)*
     $            parcoeff(ip,iv)*xd(hp,ip)
              enddo
            enddo
            if (lrel.eq.1) then
              if (norm(h).gt.0d0) delta(h,iv)=term/norm(h)
            endif
          enddo
        enddo
        do iv=1,nvar
          do h=1,d
            do hp=1,d
              correction(h,hp,iv)=correction(h,hp,iv)+
     $             mixed(h,hp,iv)+dconjg(mixed(hp,h,iv))
            enddo
          enddo
        enddo
        if (lrel.eq.1) then
          do iv=1,nvar
            do h=1,d
              do hp=1,d
                relative(h,hp,iv)=relative(h,hp,iv)+brow(h,hp)*
     $               (delta(h,iv)+dconjg(delta(hp,iv)))
              enddo
            enddo
          enddo
        endif
        born(1:d,1:d)=born(1:d,1:d)+brow(1:d,1:d)
      enddo
      born(1:d,1:d)=born(1:d,1:d)/dble(iden_values(nfksprocess))
      correction(1:d,1:d,:)=correction(1:d,1:d,:)/
     $  dble(iden_values(nfksprocess))
      relative(1:d,1:d,:)=relative(1:d,1:d,:)/
     $  dble(iden_values(nfksprocess))
 900  continue
      sud_mod=saved_mod
      s_to_rij=saved_stor
      rij_ge_mw=saved_rij
      end

      subroutine sdk_density_variant(iv)
      implicit none
      integer iv,sud_mod
      logical s_to_rij,rij_ge_mw
      common/to_sud_mod/sud_mod
      common/to_s_to_rij/s_to_rij
      common/rij_ge_mw/rij_ge_mw
C Same order as ewsudakov_py results(2:6): QED, weak, s/r off,
C s/r off and invariant floor off, invariant floor off.
      sud_mod=1
      if (iv.eq.1) sud_mod=0
      s_to_rij=iv.ne.3.and.iv.ne.4
      rij_ge_mw=iv.lt.4
      end

      subroutine sdk_density_base(hels,ibase,phase)
      implicit none
      include 'nexternal.inc'
      integer hels(nexternal-1),ibase
      double complex phase
      %(base_selection)s
      end

      subroutine sdk_density_eval(isource,p,hels,jamp)
      implicit none
      include 'nexternal.inc'
      %(constants)s
      integer isource,hels(nexternal-1)
      double precision p(0:3,nexternal-1)
      double complex jamp(NC,NO)
      %(source_selection)s
      end

      subroutine sdk_density_coeff(ibase,hels,invariants,coeff)
      implicit none
      include 'nexternal.inc'
      include 'ewsudakov_haslo.inc'
      %(constants)s
      integer ibase,hels(nexternal-1),iflist(nexternal-1)
      integer pdglist(nexternal-1)
      double precision invariants(nexternal-1,nexternal-1)
      double complex coeff(NS)
      double complex get_lsc_diag,get_ssc_n_diag,get_xxc_diag
      double complex get_lsc_nondiag,get_xxc_nondiag
      double complex get_ssc_n_nondiag_1,get_ssc_n_nondiag_2,get_ssc_c
      external get_lsc_diag,get_ssc_n_diag,get_xxc_diag
      external get_lsc_nondiag,get_xxc_nondiag
      external get_ssc_n_nondiag_1,get_ssc_n_nondiag_2,get_ssc_c
      coeff=(0d0,0d0)
      if (.not.has_lo1) return
      iflist=(/%(iflist)s/)
      %(coefficient_calls)s
      end

      subroutine sdk_density_left(ibase,isource,a,u)
C u(j,n)=sum_i conj(metric(i,j)*a(i,n)), the conjugated side of the
C colour contraction of base ibase with source isource.
      implicit none
      %(constants)s
      integer ibase,isource,i,j,n
      double complex a(NC,NO),u(NC,NO),metric(NC,NC,NS,NB)
      %(metric_data)s
      do n=1,NO
        do j=1,NC
          u(j,n)=dconjg(metric(1,j,isource,ibase)*a(1,n))
          do i=2,NC
            u(j,n)=u(j,n)+dconjg(metric(i,j,isource,ibase)*a(i,n))
          enddo
        enddo
      enddo
      end

      double complex function sdk_density_dot(u,b)
C Contraction of a conjugated side u of sdk_density_left with b, over
C the coupling-order pairs of the target order.
      implicit none
      %(constants)s
      integer j,m,n,mask(NO,NO)
      double complex u(NC,NO),b(NC,NO)
      data mask /%(order_mask)s/
      sdk_density_dot=(0d0,0d0)
      do m=1,NO
        do n=1,NO
          if (mask(m,n).eq.0) cycle
          do j=1,NC
            sdk_density_dot=sdk_density_dot+u(j,n)*b(j,m)
          enddo
        enddo
      enddo
      end

      double complex function sdk_density_inter(ibase,isource,a,b)
      implicit none
      %(constants)s
      integer ibase,isource
      double complex a(NC,NO),b(NC,NO),u(NC,NO)
      double complex sdk_density_dot
      external sdk_density_dot
      call sdk_density_left(ibase,isource,a,u)
      sdk_density_inter=sdk_density_dot(u,b)
      end

      subroutine sdk_density_dependence(dep,ok)
C dep(ip,b): the amplitudes of base b depend on input parameter ip. A
C shift that leaves its couplings, masses and widths bitwise unchanged
C leaves its amplitudes unchanged, so that derivative is exactly zero.
      implicit none
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      %(constants)s
      integer NV
      parameter (NV=%(ninputs)d)
      logical dep(6,NB),ok
      integer ip,b,k
      double precision original(6),step(6)
      double complex central(NV,NB),shifted(NV)
      original=(/%(first_parameter)s,mdl_mz,mdl_mw,mdl_mt,mdl_mh,
     $           mdl_ntadpole/)
      step=abs(original)*1d-5
      ok=mdl_mt.eq.mdl_ymt.and.mdl_ntadpole.eq.1d0.and.
     $   all(step.ne.0d0)
      if (.not.ok) return
      do b=1,NB
        call sdk_density_inputs(b,central(1,b))
      enddo
      dep=.false.
      do ip=1,6
        do k=-1,1,2
          call sdk_density_set_parameter(ip,original(ip)+k*step(ip))
          call coup()
          do b=1,NB
            call sdk_density_inputs(b,shifted)
            if (any(shifted.ne.central(:,b))) dep(ip,b)=.true.
          enddo
        enddo
        call sdk_density_set_parameter(ip,original(ip))
      enddo
      call coup()
      end

      subroutine sdk_density_inputs(ibase,values)
C The couplings, masses and widths that the amplitudes of base ibase use.
      implicit none
      include 'coupl.inc'
      integer ibase
      double complex values(%(ninputs)d)
      values=(0d0,0d0)
      %(base_inputs)s
      end

      subroutine sdk_density_derivatives(p,d,base,hels,dep,deriv)
C Central differences of the base amplitudes in the input parameters
C they depend on. Each shift updates the couplings once for all
C resonance helicities.
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      %(constants)s
      integer d,base(36),hels(nexternal-1,36),ip,h
      double precision p(0:3,nexternal-1),original(6),step(6)
      double complex deriv(NC,NO,6,36),up(NC,NO,36),dn(NC,NO)
      logical dep(6,NB),need
      original=(/%(first_parameter)s,mdl_mz,mdl_mw,mdl_mt,mdl_mh,
     $           mdl_ntadpole/)
      step=abs(original)*1d-5
      do ip=1,6
        need=.false.
        do h=1,d
          if (base(h).ne.0) need=need.or.dep(ip,base(h))
        enddo
        if (.not.need) cycle
        call sdk_density_set_parameter(ip,original(ip)+step(ip))
        call coup()
        do h=1,d
          if (base(h).eq.0) cycle
          if (dep(ip,base(h)))
     $      call sdk_density_eval(base(h),p,hels(1,h),up(1,1,h))
        enddo
        call sdk_density_set_parameter(ip,original(ip)-step(ip))
        call coup()
        do h=1,d
          if (base(h).eq.0) cycle
          if (.not.dep(ip,base(h))) cycle
          call sdk_density_eval(base(h),p,hels(1,h),dn)
          deriv(:,:,ip,h)=(up(:,:,h)-dn)/(2d0*step(ip))
        enddo
        call sdk_density_set_parameter(ip,original(ip))
      enddo
      call coup()
      end

      subroutine sdk_density_set_parameter(ip,value)
      implicit none
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      integer ip
      double precision value
      if (ip.eq.1) %(first_parameter)s=value
      if (ip.eq.2) mdl_mz=value
      if (ip.eq.3) mdl_mw=value
      if (ip.eq.4) then
        mdl_mt=value
        mdl_ymt=value
      endif
      if (ip.eq.5) mdl_mh=value
      if (ip.eq.6) mdl_ntadpole=value
      end

      subroutine sdk_density_parcoeff(invariants,coeff)
C Reuse the scalar kernel's renormalisation constants. It acts linearly
C on derivatives of |M|^2 and divides by two; probe with unit derivatives
C and undo that division for derivatives of the complex amplitude.
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'ewsudakov_haslo.inc'
      double precision invariants(nexternal-1,nexternal-1)
      double complex coeff(6),der(amp_split_size,6)
      double complex der2(amp_split_size,6),ans(amp_split_size)
      common/to_amp_split_ewsud_der/der
      common/to_amp_split_ewsud_der2/der2
      common/to_amp_split_ewsud/ans
      double complex saved(amp_split_size,6),saved2(amp_split_size,6)
      double complex savedans(amp_split_size)
      integer ip
      coeff=(0d0,0d0)
      if (.not.has_lo1) return
      saved=der
      saved2=der2
      savedans=ans
      der2=(0d0,0d0)
      do ip=1,6
        der=(0d0,0d0)
        der(lo1_pos,ip)=(1d0,0d0)
        call %(par_ren)s(invariants)
        coeff(ip)=2d0*ans(lo1_pos)
      enddo
      der=saved
      der2=saved2
      ans=savedans
      end
