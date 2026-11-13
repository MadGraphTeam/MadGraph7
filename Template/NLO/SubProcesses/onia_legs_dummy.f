      logical function fks_onia_leg(i)
c**************************************************************************
c     No bound state in this process: no leg is an onium (see onia_legs.f,
c     written by the exporter for processes with bound states).
c**************************************************************************
      implicit none
      integer i
      fks_onia_leg=.false.
      return
      end
