c The parton densities of a built-in set of MadFKS at nodes in x and Q, for
c madgraph/mg7/nlo/grids.py: fdist of pdf.f after pdfwrap, as pdg2pdf takes
c them for a proton. It is linked with the libraries of an aMC@NLO output
c (libpdf.a) and runs in its lib directory, where pdfwrap finds the tables
c (Pdfdata). Its argument is the file it writes. stdin: pdlabel, the number
c of nodes in x and in Q, then the nodes. The file: per node in x and per
c node in Q, x f of the flavours -5 to 5 (0 the gluon) and of the photon.
      program pdf_table
      implicit none
      include 'pdf.inc'
      integer nx,nq,ix,iq,i
      double precision, allocatable :: x(:),q(:)
      double precision fx(-7:7)
      character*512 path
      call get_command_argument(1,path)
      read(*,'(a)') pdlabel
      call pdfwrap
      read(*,*) nx,nq
      allocate(x(nx),q(nq))
      read(*,*) (x(ix),ix=1,nx)
      read(*,*) (q(iq),iq=1,nq)
      open(unit=91,file=path,status='replace')
      do ix=1,nx
         do iq=1,nq
            call fdist(1,x(ix),q(iq),fx)
            write(91,'(12es26.17)') (x(ix)*fx(i),i=-5,5),x(ix)*fx(7)
         enddo
      enddo
      close(91)
      end
