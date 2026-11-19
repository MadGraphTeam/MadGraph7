      subroutine sample_beamstrahlung(x1,omx1,q2f1,id1,
     &                                x2,omx2,q2f2,id2,
     &                                xbs1,xbs2)
c**************************************************************************
c     Split the two beam momentum fractions of the current event into
c     their beamstrahlung and ISR parts, x = xbs * xisr.
c
c     eepdf.f gives a dressed-lepton density as a sum over n_ee
c     components; component i weights the event with comp1(i)*comp2(i),
c     which is exactly what ee_comp_prod() sums over.  For each beam,
c     eepdf_tilde_type() distinguishes the components that carry a
c     beamstrahlung convolution on that beam from those that are a
c     plain delta(1-xbs) -- eepdf_bstrahl_pars() in bstrahl.f exposes
c     the same split.  So sampling the component and then the split of
c     each radiating beam reproduces the density exactly, by
c     construction: the marginal in x is untouched.
c
c     This is called once per written event rather than per
c     phase-space point, so recomputing the components here is cheap
c     and keeps the sampling out of the cross-section path entirely.
c
c     omx1/omx2 are 1-x, passed in rather than recomputed because the
c     (1-x)**(-k) factor inside compute_eepdf loses precision as x->1.
c**************************************************************************
      implicit none
      include 'eepdf.inc'
c
c     Arguments
c
      double precision x1, omx1, q2f1, x2, omx2, q2f2
      double precision xbs1, xbs2
      integer id1, id2
c
c     Local
c
      double precision comp1(n_ee), comp2(n_ee)
      double precision wgt(n_ee), wtot, wsum, target
      double precision u1, u2, xisr
      integer i, isel
c
c     External
c
      double precision compute_eepdf
      logical eepdf_has_bstrahl
      real ran1
      external ran1
c
c     Global
c
      integer iseed
      data iseed /0/
      save iseed
c-----
c  Begin Code
c-----
      xbs1 = 1d0
      xbs2 = 1d0
      if (.not.eepdf_has_bstrahl()) return

      wtot = 0d0
      do i = 1, n_ee
        comp1(i) = compute_eepdf(x1, omx1, dsqrt(q2f1), i, id1, id1)
        comp2(i) = compute_eepdf(x2, omx2, dsqrt(q2f2), i, id2, id2)
c       components may individually be negative, so sample on the
c       absolute contribution -- the same convention PD(0) uses when
c       it accumulates DABS(PD(IPROC)) in auto_dsig.f
        wgt(i) = dabs(comp1(i)*comp2(i))
        wtot = wtot + wgt(i)
      enddo
      if (wtot.le.0d0) return

      target = dble(ran1(iseed))*wtot
      isel = n_ee
      wsum = 0d0
      do i = 1, n_ee
        wsum = wsum + wgt(i)
        if (wsum.ge.target) then
          isel = i
          exit
        endif
      enddo

      u1 = dble(ran1(iseed))
      u2 = dble(ran1(iseed))
      call eepdf_sample_bstrahl(x1,q2f1,isel,id1,id1,u1,u2,xbs1,xisr)

      u1 = dble(ran1(iseed))
      u2 = dble(ran1(iseed))
      call eepdf_sample_bstrahl(x2,q2f2,isel,id2,id2,u1,u2,xbs2,xisr)

      return
      end


      integer function bstrahl_beam_id(lppin)
c**************************************************************************
c     PDG id of the beam lepton behind lpp = +-3 (electron) or +-4
c     (muon), matching the ih_local convention of pdg2pdf.f.  Returns 0
c     for any other beam type.
c**************************************************************************
      implicit none
      integer lppin

      if (iabs(lppin).eq.3) then
        bstrahl_beam_id = isign(11,lppin)
      else if (iabs(lppin).eq.4) then
        bstrahl_beam_id = isign(13,lppin)
      else
        bstrahl_beam_id = 0
      endif

      return
      end
