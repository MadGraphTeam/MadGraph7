c     ------------------------------------------------------------------
c     Beamstrahlung / ISR splitting: no dressed-lepton density.
c
c     Default placeholder, overwritten by copy_lep_densities() when a
c     leptonic density is selected.  With no such density there is no
c     beamstrahlung to split off.  See sample_bstrahl.f.
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
