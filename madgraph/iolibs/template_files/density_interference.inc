C Exact spectator sum of the linear interference, retaining all colours.
C The physical Born is kept. Longitudinal corrections are mapped from the
C Goldstone amplitudes with HELAS equivalence phases. The second output is
C a spectator-resolved Goldstone-normalised projection for comparison.

      subroutine sdk_density_interference(p,nres,pos,dims,born,
     $     correction,relative,zero_gold,status)
      implicit none
      include 'nexternal.inc'
      %(constants)s
      integer nres,pos(10),dims(10),zero_gold,status
      double precision p(0:3,nexternal-1)
      double complex born(36,36),correction(36,36,5)
      double complex relative(36,36,5)
      integer NCOMB
      parameter (NCOMB=%(ncomb)d)
      integer nhel(nexternal-1,NCOMB),i
      %(helicity_lines)s
      integer ispec,h,hp,r,k,d,iv,s,ip,ib,base(36)
      integer hels(nexternal-1,36),iflist(nexternal-1)
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      %(den_factor_lines)s
      double precision invariants(nexternal-1,nexternal-1,5),norm(36)
      double complex amps(NC,NO,NS,36),deriv(NC,NO,6,36)
      double complex coeff(NS),parcoeff(6,5),phase(36)
      double complex delta(36,5),mixed(36,36,5),brow(36,36)
      double complex sdk_density_inter,term
      external sdk_density_inter
      logical ok,selected
      integer sud_mod,saved_mod
      common/to_sud_mod/sud_mod
      logical s_to_rij,rij_ge_mw,saved_stor,saved_rij
      common/to_s_to_rij/s_to_rij
      common/rij_ge_mw/rij_ge_mw

      born=(0d0,0d0)
      correction=(0d0,0d0)
      relative=(0d0,0d0)
      zero_gold=0
      status=0
C The underlying Sudakov charge/reshuffling functions do not implement
C massive vector beams. Do not return their undefined numerical results.
      if (.not.%(supported_initial)s) then
        status=4
        return
      endif
      d=1
      if (nres.lt.1.or.nres.gt.10) then
        status=1
        return
      endif
      do r=1,nres
        if (dims(r).lt.2.or.dims(r).gt.3.or.pos(r).lt.1.or.
     $      pos(r).gt.nexternal-1) then
          status=1
          return
        endif
        do k=1,r-1
          if (pos(k).eq.pos(r)) then
            status=1
            return
          endif
        enddo
        if (d.gt.36/dims(r)) then
          status=1
          return
        endif
        d=d*dims(r)
      enddo
      saved_mod=sud_mod
      saved_stor=s_to_rij
      saved_rij=rij_ge_mw
      iflist=(/%(iflist)s/)
C Each variant builds its invariants with its own floor.
      do iv=1,5
        call sdk_density_variant(iv)
        call sdk_get_invariants(p,iflist,invariants(1,1,iv))
        call sdk_density_parcoeff(invariants(1,1,iv),parcoeff(1,iv))
      enddo

C One representative of each spectator tuple. No good-helicity filtering:
C a zero Born component can have a nonzero mixed-helicity interference.
      do ispec=1,NCOMB
        selected=.true.
        do r=1,nres
          if (nhel(pos(r),ispec).ne.-1) selected=.false.
        enddo
        if (.not.selected) cycle
        amps=(0d0,0d0)
        deriv=(0d0,0d0)
        mixed=(0d0,0d0)
        delta=(0d0,0d0)
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
          call sdk_density_eval(1,p,hels(1,h),amps(1,1,1,h),ok)
          if (.not.ok) then
            status=2
            goto 900
          endif
          ib=base(h)
          if (ib.eq.0) cycle
          do s=2,NS
            call sdk_density_needed(ib,s,selected)
            if (.not.selected) cycle
            call sdk_density_eval(s,p,hels(1,h),amps(1,1,s,h),ok)
            if (.not.ok) then
              status=2
              goto 900
            endif
          enddo
          call sdk_density_derivatives(ib,p,hels(1,h),
     $         deriv(1,1,1,h),ok)
          if (.not.ok) then
            status=3
            goto 900
          endif
        enddo
        do h=1,d
          do hp=1,d
            brow(h,hp)=sdk_density_inter(1,1,amps(1,1,1,hp),
     $                                  amps(1,1,1,h))
          enddo
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
        do iv=1,5
          call sdk_density_variant(iv)
          do h=1,d
            ib=base(h)
            if (ib.eq.0) cycle
            call sdk_density_coeff(ib,hels(1,h),invariants(1,1,iv),
     $                             coeff)
            term=(0d0,0d0)
            do s=1,NS
              if (coeff(s).eq.(0d0,0d0)) cycle
              term=term+coeff(s)*sdk_density_inter(ib,s,
     $                    amps(1,1,ib,h),amps(1,1,s,h))
              do hp=1,d
                mixed(h,hp,iv)=mixed(h,hp,iv)+phase(h)*coeff(s)*
     $              sdk_density_inter(1,s,amps(1,1,1,hp),
     $                                    amps(1,1,s,h))
              enddo
            enddo
            do ip=1,6
              term=term+parcoeff(ip,iv)*sdk_density_inter(ib,ib,
     $                        amps(1,1,ib,h),deriv(1,1,ip,h))
              do hp=1,d
                mixed(h,hp,iv)=mixed(h,hp,iv)+phase(h)*
     $             parcoeff(ip,iv)*sdk_density_inter(1,ib,
     $                          amps(1,1,1,hp),deriv(1,1,ip,h))
              enddo
            enddo
            if (norm(h).gt.0d0) delta(h,iv)=term/norm(h)
          enddo
          do h=1,d
            do hp=1,d
              correction(h,hp,iv)=correction(h,hp,iv)+
     $             mixed(h,hp,iv)+dconjg(mixed(hp,h,iv))
              relative(h,hp,iv)=relative(h,hp,iv)+brow(h,hp)*
     $             (delta(h,iv)+dconjg(delta(hp,iv)))
            enddo
          enddo
        enddo
        born(1:d,1:d)=born(1:d,1:d)+brow(1:d,1:d)
      enddo
      born=born/dble(iden_values(nfksprocess))
      correction=correction/dble(iden_values(nfksprocess))
      relative=relative/dble(iden_values(nfksprocess))
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

      subroutine sdk_density_needed(ibase,isource,needed)
      implicit none
      %(constants)s
      integer ibase,isource,source_base(NS)
      logical needed
      data source_base /%(source_bases)s/
      needed=source_base(isource).eq.ibase
      end

      subroutine sdk_density_eval(isource,p,hels,jamp,ok)
      implicit none
      include 'nexternal.inc'
      %(constants)s
      integer isource,hels(nexternal-1)
      double precision p(0:3,nexternal-1)
      double complex jamp(NC,NO)
      logical ok
      jamp=(0d0,0d0)
      ok=.false.
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

      double complex function sdk_density_inter(ibase,isource,a,b)
      implicit none
      %(constants)s
      integer ibase,isource,i,j,m,n,mask(NO,NO)
      double complex a(NC,NO),b(NC,NO),metric(NC,NC,NS,NB)
      data mask /%(order_mask)s/
      %(metric_data)s
      sdk_density_inter=(0d0,0d0)
      do m=1,NO
        do n=1,NO
          if (mask(m,n).eq.0) cycle
          do i=1,NC
            do j=1,NC
              sdk_density_inter=sdk_density_inter+
     $         dconjg(metric(i,j,isource,ibase)*a(i,n))*b(j,m)
            enddo
          enddo
        enddo
      enddo
      end

      subroutine sdk_density_derivatives(ibase,p,hels,deriv,ok)
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      %(constants)s
      integer ibase,hels(nexternal-1),ip
      double precision p(0:3,nexternal-1),value,step,original
      double complex deriv(NC,NO,6),up(NC,NO),dn(NC,NO)
      logical ok,okup,okdn
      deriv=(0d0,0d0)
      ok=.false.
      if (mdl_mt.ne.mdl_ymt.or.mdl_ntadpole.ne.1d0) return
      do ip=1,6
        call sdk_density_parameter(ip,value,.false.)
        original=value
        step=abs(value)*1d-5
        if (step.eq.0d0) return
        value=value+step
        call sdk_density_parameter(ip,value,.true.)
        call coup()
        call sdk_density_eval(ibase,p,hels,up,okup)
        value=original-step
        call sdk_density_parameter(ip,value,.true.)
        call coup()
        call sdk_density_eval(ibase,p,hels,dn,okdn)
        value=original
        call sdk_density_parameter(ip,value,.true.)
        call coup()
        if (.not.okup.or..not.okdn) return
        deriv(:,:,ip)=(up-dn)/(2d0*step)
      enddo
      ok=.true.
      end

      subroutine sdk_density_parameter(ip,value,write_value)
      implicit none
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      integer ip
      double precision value,params(6)
      logical write_value
      if (write_value) then
        if (ip.eq.1) %(first_parameter)s=value
        if (ip.eq.2) mdl_mz=value
        if (ip.eq.3) mdl_mw=value
        if (ip.eq.4) then
          mdl_mt=value
          mdl_ymt=value
        endif
        if (ip.eq.5) mdl_mh=value
        if (ip.eq.6) mdl_ntadpole=value
      else
        params=(/%(first_parameter)s,mdl_mz,mdl_mw,mdl_mt,mdl_mh,
     $            mdl_ntadpole/)
        value=params(ip)
      endif
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

      subroutine sdk_density_dimensions(ncolor,norder,nsource,nbase)
      implicit none
      %(constants)s
      integer ncolor,norder,nsource,nbase
      ncolor=NC
      norder=NO
      nsource=NS
      nbase=NB
      end
