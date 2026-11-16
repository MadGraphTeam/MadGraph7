c The fixed-order analysis of the references of
c make_fks_cross_section_reference.py for an output with --ewsudakov:
c the total rate, that of the Born and those of the Sudakov approximation
c of the EW corrections for sud_mod 0 and 1 (ibody 10 and 11 of outfun,
c the types 20 and 21 of compute_ewsudakov), each a histogram of one bin.
      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      call HwU_inithist(nwgt,weights_info)
      call HwU_book(1,'total rate           ',1,0.5d0,1.5d0)
      call HwU_book(2,'total rate Born      ',1,0.5d0,1.5d0)
      call HwU_book(3,'EW Sudakov sud_mod 0 ',1,0.5d0,1.5d0)
      call HwU_book(4,'EW Sudakov sud_mod 1 ',1,0.5d0,1.5d0)
      return
      end


      subroutine analysis_end(dummy)
      implicit none
      double precision dummy
      call HwU_write_file
      return
      end


      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal),iPDG(nexternal),ibody
      double precision p(0:4,nexternal),wgts(*)
      call HwU_fill(1,1d0,wgts)
      if (ibody.eq.3) call HwU_fill(2,1d0,wgts)
      if (ibody.eq.10) call HwU_fill(3,1d0,wgts)
      if (ibody.eq.11) call HwU_fill(4,1d0,wgts)
      return
      end
