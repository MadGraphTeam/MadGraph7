      subroutine sbornsoft_onia(pp,xi_i_fks,y_ij_fks,wgt)
c**************************************************************************
c     No P-wave quarkonium in this Born: no soft current beyond the
c     colour-linked Borns (see fks_onia_soft.f).
c**************************************************************************
      implicit none
      include "nexternal.inc"
      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,wgt
      wgt=0d0
      return
      end


      subroutine bsv_onia(p,xicut_used,wgt)
c**************************************************************************
c     No P-wave quarkonium: no integrated soft current beyond the
c     colour-linked Borns (see fks_onia_soft.f).
c**************************************************************************
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),xicut_used,wgt
      wgt=0d0
      return
      end
