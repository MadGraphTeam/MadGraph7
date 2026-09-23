c     ------------------------------------------------------------------
c     Beamstrahlung / ISR splitting for the 'isrbetll' lepton density.
c
c     This density is pure ISR: there is no beamstrahlung to split off,
c     so the whole of x is ISR off a beam lepton at its nominal energy.
c     See sample_bstrahl.f for how this is consumed.
c     ------------------------------------------------------------------

      logical function eepdf_has_bstrahl()
      implicit none
      eepdf_has_bstrahl = .false.
      return
      end


      subroutine eepdf_sample_bstrahl(x,Q2,n,partonid,beamid,u1,u2,
     &                                xbs,xisr)
      implicit none
      double precision x, Q2, u1, u2, xbs, xisr
      integer n, partonid, beamid
      xbs = 1d0
      xisr = x
      return
      end
