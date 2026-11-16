      ! Per-event beamstrahlung momentum fraction of each beam.
      !
      ! A dressed-lepton density convolutes beamstrahlung with ISR, so
      ! the momentum fraction x that ends up in the LHE initial-state
      ! momenta is the product of the two: x = xbs * xisr.  A shower
      ! reading those momenta has no way to tell them apart and would
      ! attribute all of 1-x to ISR off a full-energy beam, which gives
      ! the hard system a recoil that beamstrahlung does not produce.
      !
      ! xbs_beam is therefore sampled at event-writing time and written
      ! to the event file, so the shower can be told the degraded beam
      ! energy and backward-evolve xisr only.  write_bstrahl is false
      ! for densities that carry no beamstrahlung at all.
      logical write_bstrahl
      double precision xbs_beam(2)
      common /to_bstrahl/ xbs_beam, write_bstrahl
