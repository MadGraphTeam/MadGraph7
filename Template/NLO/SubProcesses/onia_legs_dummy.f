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


      logical function fks_onia_nosector(i)
c**************************************************************************
c     No bound state in this process: no leg is left out of the FKS
c     partition (see onia_legs.f).
c**************************************************************************
      implicit none
      integer i
      fks_onia_nosector=.false.
      return
      end


      subroutine onia_set_ldme_common()
c**************************************************************************
c     No bound state in this process: no LDME for MadLoop (see onia_legs.f).
c**************************************************************************
      implicit none
      return
      end


      logical function onia_born_vanishes()
c**************************************************************************
c     No bound state in this process: the Born does not vanish identically
c     (see onia_legs.f).
c**************************************************************************
      implicit none
      onia_born_vanishes=.false.
      return
      end
