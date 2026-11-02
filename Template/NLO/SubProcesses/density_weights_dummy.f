c The spin-density-matrix weights of an output made without --density:
c there is none (density_weights.f of the P directory replaces this file
c in an output made with --density=i,j; see makefile_fks_dir).
      subroutine density_n_elements(n)
c     number of spin-density-matrix elements carried by each weight: none
      implicit none
      integer n
      n=0
      end

      subroutine density_element(k, i, j, kind)
      implicit none
      integer k, i, j, kind
      write (*,*) 'ERROR: no density matrix in this output'
      stop 1
      end

      subroutine density_set_element(k)
      implicit none
      integer k
      write (*,*) 'ERROR: no density matrix in this output'
      stop 1
      end

      subroutine density_unset()
      end

      subroutine density_element_label(k, label)
      implicit none
      integer k
      character*(*) label
      label='none'
      end

      subroutine density_frame_rephase(p, npart)
      implicit none
      integer npart
      double precision p(0:3,npart)
      end
