c The matrix elements of the Sudakov approximation of the EW corrections
c (--ewsudakov) for fks_provider.f, those that sudakov_wrapper of
c ewsudakov_wrapper.f takes per helicity combination of the Born, for
c EWSudakov of madspace. madgraph/mg7/nlo/provider.py links it in a P*
c directory of an output with --ewsudakov, with
c fks_provider_ewsudakov_calls.f, the calls to the amplitudes of the
c process, which it writes from fks_configurations.json, and
c fks_provider_ewsudakov_dummy.f in its place without --ewsudakov.


c Whether the output has the Sudakov approximation, the number of
c helicity combinations of the Born and of interferences, and has_lo1
c and has_lo2 of ewsudakov_haslo.inc.
      subroutine provider_sdk_sizes(sizes)
      implicit none
      include 'ewsudakov_haslo.inc'
      integer sizes(5)
      integer sdk_get_ncomb,provider_sdk_interference_count
      external sdk_get_ncomb,provider_sdk_interference_count
      sizes=[1,sdk_get_ncomb(),provider_sdk_interference_count(),0,0]
      if (has_lo1) sizes(4)=1
      if (has_lo2) sizes(5)=1
      end


c The base amplitude of each helicity combination (0 for the Born, k for
c the Goldstone amplitude k) and of each interference, and the parameters
c of the logarithms (mdl_mw, mdl_mz, mdl_mh, mdl_mt, the mass of the b
c quark, gal(1), aewm1, mdl_ee, mdl_gf).
      subroutine provider_sdk_info(ncomb,nint,bases,interference_bases,
     $     parameters)
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include '../../Source/MODEL/input.inc'
      integer ncomb,nint,bases(ncomb),interference_bases(nint)
      double precision parameters(9)
      integer h,t,hels(nexternal-1)
      integer provider_sdk_base,provider_sdk_interference_base
      double precision get_mass_from_id
      external provider_sdk_base,provider_sdk_interference_base
      external get_mass_from_id
      do h=1,ncomb
         call sdk_get_hels(h,hels)
         bases(h)=provider_sdk_base(hels)
      enddo
      do t=1,nint
         interference_bases(t)=provider_sdk_interference_base(t)
      enddo
      parameters=[mdl_mw,mdl_mz,mdl_mh,mdl_mt,get_mass_from_id(5),
     $     dble(gal(1)),aewm1,mdl_ee,dble(mdl_gf)]
      end


c The inputs of EWSudakov at the Born p (in its partonic center-of-mass
c frame), as dump_ewsudakov.f of the tests of madspace takes them, all
c with comp_idfac of the configuration as sudakov_wrapper: per helicity
c combination the base amplitude at the LO1 and at the LO2 and its
c derivatives with respect to the six parameters of numder_*.f (out(1:8,
c h)), then per helicity combination and interference the interference
c at the LO1 (real and imaginary parts), zero for the interferences of
c another base amplitude.
      subroutine provider_sdk_evaluate(p,ncomb,nint,out)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'ewsudakov_haslo.inc'
      integer ncomb,nint
      double precision p(0:3,nexternal-1)
      double precision out(8*ncomb+2*nint*ncomb)
      integer h,t,j,base,k,hels(nexternal-1),iflist(nexternal-1)
      double precision comp,invariants(nexternal-1,nexternal-1)
      integer provider_sdk_base,provider_sdk_interference_base
      double precision provider_sdk_idfac
      external provider_sdk_base,provider_sdk_interference_base
      external provider_sdk_idfac
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex amp_split_ewsud_lo2(amp_split_size)
      common /to_amp_split_ewsud_lo2/ amp_split_ewsud_lo2
      double complex amp_split_ewsud_der(amp_split_size,6)
      common /to_amp_split_ewsud_der/ amp_split_ewsud_der
      do j=1,nexternal-1
         iflist(j)=1
         if (j.le.nincoming) iflist(j)=-1
      enddo
      out=0d0
      do h=1,ncomb
         call sdk_get_hels(h,hels)
         base=provider_sdk_base(hels)
         comp=provider_sdk_idfac(base)
         k=8*(h-1)
         calculatedborn=.false.
         call provider_sdk_base_me(base,p,hels,h)
         if (has_lo1) out(k+1)=dble(amp_split_ewsud(lo1_pos))*comp
         if (has_lo2) out(k+2)=dble(amp_split_ewsud_lo2(lo2_pos))*comp
         call sdk_get_invariants(p,iflist,invariants)
         calculatedborn=.false.
         call provider_sdk_par_ren(base,p,hels,h,invariants)
         if (has_lo1) then
            do j=1,6
               out(k+2+j)=dble(amp_split_ewsud_der(lo1_pos,j))*comp
            enddo
         endif
         do t=1,nint
            if (provider_sdk_interference_base(t).ne.base) cycle
            calculatedborn=.false.
            call provider_sdk_interference(t,p,hels)
            if (has_lo1) then
               k=8*ncomb+2*(nint*(h-1)+t-1)
               out(k+1)=dble(amp_split_ewsud(lo1_pos))*comp
               out(k+2)=dimag(amp_split_ewsud(lo1_pos))*comp
            endif
         enddo
      enddo
      calculatedborn=.false.
      end


c The weights of compute_ewsudakov of fks_singular.f at the Born p for
c sud_mod 0 and 1 (s_to_rij and rij_ge_mw as the run sets them), without
c f_b / g^qcd_power, at the renormalization scale scale_in of the LO2
c part: per mode those of the LO1 at two powers of QED more and of the
c LO2 at two powers of QCD more (both at the NLO order of the LO1), and
c the sum of the others, those of the other split-order positions.
      subroutine provider_sdk_weights(p,scale_in,out)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'run.inc'
      include 'ewsudakov_haslo.inc'
      double precision p(0:3,nexternal-1),scale_in,out(2,0:1)
      integer iamp,mode
      double complex lo1,lo2
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      integer sud_mod
      common /to_sud_mod/ sud_mod
      logical sud_mc_hel
      common /to_mc_hel/ sud_mc_hel
      integer ewsud_helselect
      common/to_ewsud_helselect/ewsud_helselect
      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double complex amp_split_ewsud_par(amp_split_size)
      common /to_amp_ewsud_par/amp_split_ewsud_par
      double complex amp_split_ewsud_qcd(amp_split_size)
      common /to_amp_ewsud_qcd/amp_split_ewsud_qcd
      double complex amp_split_ewsud_parqcd(amp_split_size)
      common /to_amp_ewsud_parqcd/amp_split_ewsud_parqcd
      sud_mc_hel=.false.
      ewsud_helselect=0
      scale=scale_in
      out=0d0
      do mode=0,1
         sud_mod=mode
         calculatedborn=.false.
         call sudakov_wrapper(p)
         do iamp=1,amp_split_size
            lo1=amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)
     $           +amp_split_ewsud_xxc(iamp)+amp_split_ewsud_par(iamp)
            lo2=amp_split_ewsud_qcd(iamp)+amp_split_ewsud_parqcd(iamp)
            if (has_lo1) then
               if (iamp.eq.lo1_pos) then
                  out(1,mode)=out(1,mode)+2d0*dble(lo1)
               else
                  out(2,mode)=out(2,mode)+2d0*dble(lo1)
               endif
            endif
            if (has_lo2) then
               if (iamp.eq.lo2_pos) then
                  out(1,mode)=out(1,mode)+2d0*dble(lo2)
               else
                  out(2,mode)=out(2,mode)+2d0*dble(lo2)
               endif
            endif
         enddo
      enddo
      calculatedborn=.false.
      end
