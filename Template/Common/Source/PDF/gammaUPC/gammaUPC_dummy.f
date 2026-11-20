       double precision function photonpdfsquare(x1,x2)
       implicit none
       double precision x1,x2
       write(*,*) "WRONG gamma UPC linked"
       photonpdfsquare = 1.0
       stop 1
       return
       end


       subroutine Get_nucleus_RA(nb_p,nb_n,RAI)
       implicit none
       integer nb_p,nb_n
       double precision RAI
       write(*,*) "WRONG gamma UPC linked"
       RAI = 1d0
       stop 2
       return
       end


       double precision function gammaupc_single_flux(x, ebeam, ichff)
       implicit none
       double precision x, ebeam
       integer ichff
       write(*,*) "WRONG gamma UPC linked (pdlabel edff or chff needed)"
       gammaupc_single_flux = 0d0
       stop 3
       return
       end


       double precision function gammaupc_single_flux_pp(x, ebeam,
     &      ichff, sqrts)
       implicit none
       double precision x, ebeam, sqrts
       integer ichff
       write(*,*) "WRONG gamma UPC linked (pdlabel edff or chff needed)"
       gammaupc_single_flux_pp = 0d0
       stop 3
       return
       end


       double precision function gammaupc_single_flux_ion(xn, ie,
     &      ichff, igap, itag1, itag2, iconj)
       implicit none
       double precision xn
       integer ie, ichff, igap, itag1, itag2, iconj
       write(*,*) "WRONG gamma UPC linked (pdlabel edff or chff needed)"
       gammaupc_single_flux_ion = 0d0
       stop 3
       return
       end
