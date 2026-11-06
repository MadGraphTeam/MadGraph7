      program dump_ewsudakov
c Reference values of the Sudakov approximation of the EW corrections of
c ewsudakov_wrapper.f, ewsudakov_functions.f and fks_singular.f
c (compute_ewsudakov), for make_ewsudakov_reference.py. Built in the P*
c directory of an 'output ewsudakovsa' with dump_ewsudakov_calls.f, the
c calls to the matrix elements of the process, which
c make_ewsudakov_reference.py writes from its fks_configurations.json.
c
c stdin: the number of points, G and the renormalization scale (scale of
c run.inc, in GET_QCD_LO2), then the amplitudes as written by
c make_ewsudakov_reference.py, then the Born momenta of each point.
c
c For every point it writes the inputs of the Sudakov approximation per
c helicity (the base amplitude of each helicity, its LO1, LO2 and
c parameter derivatives, the interferences), the coefficients of
c ewsudakov_functions.f per helicity and per interference, and the
c contributions of sudakov_wrapper per helicity and summed, for every
c mode m = 4 sud_mod + 2 s_to_rij + rij_ge_mw (sud_mod 0 and 1). The
c values of the matrix elements are multiplied by comp_idfac, as
c sudakov_wrapper takes them.
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'orders.inc'
      include 'ewsudakov_haslo.inc'
      include 'run.inc'
      include '../../Source/MODEL/input.inc'
      integer n
      parameter (n=nexternal-1)
      integer maxgold, maxint
      parameter (maxgold=64, maxint=512)
c the amplitudes
      integer ngold, ngleg(maxgold), gleg(n, maxgold)
      integer bpdg(n, 0:maxgold)
      integer nint, itype(maxint), ibase(maxint)
      integer ileg(2, maxint), iold(2, maxint), inew(2, maxint)
c the helicities
      integer ncomb, sdk_get_ncomb
      integer hels(n), base
      integer npoints, ipt, i, j, k, h, m, t, sm, s2r, rge
      integer iflist(n)
      double precision p(0:3, n), invariants(n, n)
      double precision born, comp, scale_in, g_in
      double precision diag, lo2, der(6)
      double complex intf, c1, c2, c3
      logical ok
c commons of ewsudakov_functions.f and of the wrapper
      integer sud_mod
      common /to_sud_mod/ sud_mod
      logical sud_mc_hel
      common /to_mc_hel/ sud_mc_hel
      logical s_to_rij
      common /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      common /rij_ge_mw/ rij_ge_mw
      integer ewsud_helselect
      common/to_ewsud_helselect/ewsud_helselect
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex amp_split_ewsud_lo2(amp_split_size)
      common /to_amp_split_ewsud_lo2/ amp_split_ewsud_lo2
      double complex amp_split_ewsud_der(amp_split_size, 6)
      common /to_amp_split_ewsud_der/ amp_split_ewsud_der
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
c functions
      double complex get_lsc_diag, get_lsc_nondiag
      double complex get_ssc_n_diag, get_ssc_n_nondiag_1
      double complex get_ssc_n_nondiag_2, get_ssc_c
      double complex get_xxc_diag, get_xxc_nondiag, get_qcd_lo2
      double precision dump_idfac, get_mass_from_id
      external get_lsc_diag, get_lsc_nondiag
      external get_ssc_n_diag, get_ssc_n_nondiag_1
      external get_ssc_n_nondiag_2, get_ssc_c
      external get_xxc_diag, get_xxc_nondiag, get_qcd_lo2
      external dump_idfac, get_mass_from_id

      nfksprocess = 1
      sud_mc_hel = .false.
      ewsud_helselect = 0
      call setpara('param_card.dat')

      read(*,*) npoints
      read(*,*) g_in, scale_in
      g = g_in
      call update_as_param()
      scale = scale_in
c the Goldstone amplitudes, then the PDG codes of the Born and of
c each of them
      read(*,*) ngold
      do k = 1, ngold
        read(*,*) ngleg(k), (gleg(i, k), i = 1, ngleg(k))
      enddo
      do k = 0, ngold
        read(*,*) (bpdg(i, k), i = 1, n)
      enddo
c the interferences: type (1 cew, 2 iz1, 3 iz2, 4 ipm2), base, then
c leg, old and new PDG code of the two replaced particles (zero for
c the second of cew and iz1)
      read(*,*) nint
      do t = 1, nint
        read(*,*) itype(t), ibase(t),
     $   (ileg(j, t), iold(j, t), inew(j, t), j = 1, 2)
      enddo

      do i = 1, n
        iflist(i) = 1
        if (i.le.nincoming) iflist(i) = -1
      enddo

      write(*,'(a,6es26.17)') 'PARAMS', mdl_mw, mdl_mz, mdl_mh,
     $ mdl_mt, dble(gal(1)), g
      write(*,'(a,4es26.17)') 'PARAMS2', aewm1, mdl_ee, scale,
     $ dble(mdl_gf)
      write(*,'(a,2l3,2i6)') 'LO', has_lo1, has_lo2, lo1_pos, lo2_pos
      write(*,'(a,es26.17)') 'MB', get_mass_from_id(5)

      ncomb = sdk_get_ncomb()
      do ipt = 1, npoints
        do i = 1, n
          read(*,*) (p(j, i), j = 0, 3)
        enddo
        write(*,'(a,i6)') 'POINT', ipt
        do i = 1, n
          write(*,'(a,i4,4es26.17)') 'P', i, (p(j, i), j = 0, 3)
        enddo
        calculatedborn = .false.
        call sborn(p, born)
        write(*,'(a,es26.17)') 'BORN', born

c the inputs per helicity
        do h = 1, ncomb
          call sdk_get_hels(h, hels)
          base = 0
          do k = ngold, 1, -1
            ok = .true.
            do i = 1, ngleg(k)
              ok = ok .and. hels(gleg(i, k)).eq.0
            enddo
            if (ok) base = k
          enddo
          comp = dump_idfac(base)
          calculatedborn = .false.
          call dump_base_me(base, p, hels, h)
          diag = 0d0
          lo2 = 0d0
          if (has_lo1) diag = dble(amp_split_ewsud(lo1_pos)) * comp
          if (has_lo2) lo2 = dble(amp_split_ewsud_lo2(lo2_pos)) * comp
          call sdk_get_invariants(p, iflist, invariants)
          calculatedborn = .false.
          call dump_par_ren(base, p, hels, h, invariants)
          do j = 1, 6
            der(j) = 0d0
            if (has_lo1) der(j) = dble(amp_split_ewsud_der(lo1_pos, j))
     $       * comp
          enddo
          write(*,'(a,2i4,8es26.17)') 'HEL', h, base, diag, lo2, der
          do t = 1, nint
            if (ibase(t).ne.base) cycle
            calculatedborn = .false.
            call dump_interference(t, p, hels)
            intf = (0d0, 0d0)
            if (has_lo1) intf = amp_split_ewsud(lo1_pos) * comp
            write(*,'(a,2i4,2es26.17)') 'INT', h, t, intf
          enddo
        enddo

c the coefficients per helicity and per interference
        do sm = 0, 1
        do s2r = 0, 1
        do rge = 0, 1
          m = 4 * sm + 2 * s2r + rge
          sud_mod = sm
          s_to_rij = s2r.eq.1
          rij_ge_mw = rge.eq.1
          call sdk_get_invariants(p, iflist, invariants)
          do h = 1, ncomb
            call sdk_get_hels(h, hels)
            base = 0
            do k = ngold, 1, -1
              ok = .true.
              do i = 1, ngleg(k)
                ok = ok .and. hels(gleg(i, k)).eq.0
              enddo
              if (ok) base = k
            enddo
            c1 = get_lsc_diag(bpdg(1, base), hels, iflist, invariants)
            c2 = get_ssc_n_diag(bpdg(1, base), hels, iflist, invariants)
            c3 = get_xxc_diag(bpdg(1, base), hels, iflist, invariants)
            write(*,'(a,2i4,6es26.17)') 'COEF', m, h, c1, c2, c3
            if (has_lo2) then
              c1 = get_qcd_lo2(bpdg(1, base), hels, iflist, invariants,
     $         lo2_pos)
              write(*,'(a,2i4,2es26.17)') 'QCOEF', m, h, c1
            endif
            do t = 1, nint
              if (ibase(t).ne.base) cycle
              if (itype(t).eq.1) then
                c1 = get_lsc_nondiag(bpdg(1, base), hels, iflist,
     $           invariants, ileg(1, t), iold(1, t), inew(1, t))
     $           + get_xxc_nondiag(bpdg(1, base), hels, iflist,
     $           invariants, ileg(1, t), iold(1, t), inew(1, t))
              else if (itype(t).eq.2) then
                c1 = get_ssc_n_nondiag_1(bpdg(1, base), hels, iflist,
     $           invariants, ileg(1, t), iold(1, t), inew(1, t))
              else if (itype(t).eq.3) then
                c1 = get_ssc_n_nondiag_2(bpdg(1, base), hels, iflist,
     $           invariants, ileg(1, t), iold(1, t), inew(1, t),
     $           ileg(2, t), iold(2, t), inew(2, t))
              else
                c1 = get_ssc_c(ileg(1, t), ileg(2, t), bpdg(1, base),
     $           inew(1, t), inew(2, t), hels, iflist, invariants)
              endif
              write(*,'(a,3i4,2es26.17)') 'ICOEF', m, h, t, c1
            enddo
          enddo

c the contributions of sudakov_wrapper per helicity, then summed
          do h = 0, ncomb
            ewsud_helselect = h
            calculatedborn = .false.
            call sudakov_wrapper(p)
            if (has_lo1) write(*,'(a,2i4,8es26.17)') 'WLO1', m, h,
     $       amp_split_ewsud_lsc(lo1_pos), amp_split_ewsud_ssc(lo1_pos),
     $       amp_split_ewsud_xxc(lo1_pos), amp_split_ewsud_par(lo1_pos)
            if (has_lo2) write(*,'(a,2i4,4es26.17)') 'WLO2', m, h,
     $       amp_split_ewsud_qcd(lo2_pos),
     $       amp_split_ewsud_parqcd(lo2_pos)
          enddo
          ewsud_helselect = 0
        enddo
        enddo
        enddo
      enddo
      end
