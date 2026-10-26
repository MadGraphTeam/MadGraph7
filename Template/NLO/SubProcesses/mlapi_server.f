      module mlapi_pipes
c Raw reads and writes on the named pipes through POSIX read/write.
c Fortran stream I/O treats a partial read from a pipe as the end of the
c file, so every transfer loops until all bytes have gone through.
      use iso_c_binding
      implicit none
      interface
         function c_open(path,flags) bind(C,name='open')
         import :: c_int,c_char
         integer(c_int) :: c_open
         character(kind=c_char) :: path(*)
         integer(c_int), value :: flags
         end function c_open
         function c_read(fd,buf,n) bind(C,name='read')
         import :: c_int,c_ptr,c_size_t,c_long
         integer(c_long) :: c_read
         integer(c_int), value :: fd
         type(c_ptr), value :: buf
         integer(c_size_t), value :: n
         end function c_read
         function c_write(fd,buf,n) bind(C,name='write')
         import :: c_int,c_ptr,c_size_t,c_long
         integer(c_long) :: c_write
         integer(c_int), value :: fd
         type(c_ptr), value :: buf
         integer(c_size_t), value :: n
         end function c_write
      end interface
      contains

      integer function open_pipe(path,write_mode)
      character(len=*) :: path
      logical write_mode
      integer(c_int) flags
      flags=0
      if (write_mode) flags=1
      open_pipe=c_open(trim(path)//c_null_char,flags)
      if (open_pipe.lt.0) then
         write (*,*) 'mlapi_server: cannot open ',trim(path)
         stop 1
      endif
      end function open_pipe

      subroutine read_bytes(fd,buf,nbytes)
      integer fd
      type(c_ptr) buf
      integer(c_size_t) nbytes,done
      integer(c_long) got
      character(kind=c_char), pointer :: bytes(:)
      call c_f_pointer(buf,bytes,[nbytes])
      done=0
      do while (done.lt.nbytes)
         got=c_read(int(fd,c_int),c_loc(bytes(done+1)),nbytes-done)
         if (got.le.0) then
            write (*,*) 'mlapi_server: the input pipe was closed'
            stop 1
         endif
         done=done+got
      enddo
      end subroutine read_bytes

      subroutine write_bytes(fd,buf,nbytes)
      integer fd
      type(c_ptr) buf
      integer(c_size_t) nbytes,done
      integer(c_long) put
      character(kind=c_char), pointer :: bytes(:)
      call c_f_pointer(buf,bytes,[nbytes])
      done=0
      do while (done.lt.nbytes)
         put=c_write(int(fd,c_int),c_loc(bytes(done+1)),nbytes-done)
         if (put.le.0) then
            write (*,*) 'mlapi_server: the output pipe was closed'
            stop 1
         endif
         done=done+put
      enddo
      end subroutine write_bytes
      end module mlapi_pipes


      program mlapi_server
c Serves the ML API of mlapi.f over two named pipes, so that a caller in
c another language can evaluate MC@NLO weights at its own phase-space
c points.  Run from the P* directory as
c
c   ./madevent_mlapi <pipe to read from> <pipe to write to>
c
c Everything on the pipes is raw binary in native byte order: 4-byte
c integers and 8-byte reals, arrays in Fortran order.  Standard output
c keeps MadGraph's log messages.
c
c At start-up the server writes
c   info(8)                                (see mlapi_init)
c   members(0:fks_configs) for every group  (see mlapi_group)
c   idiag for every channel index           (see mlapi_config)
c Then it loops over commands:
c   command 0  stop
c   command 1  read n, x(ndim,n), ichan(n), igroup(n); evaluate (see
c              mlapi_eval) and write totals(4,n), nev(n),
c              ev_int(3,maxev,n), ev_dbl(3,maxev,n),
c              ev_mom(0:3,nexternal,maxev,n), parts(16,n)
      use mlapi_pipes
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      integer nparts,maxev
      parameter (nparts=16,maxev=fks_configs+1)
      integer(c_int), target :: info(8),members(0:fks_configs),idiag
     $     ,command,n
      integer i,ndim,ngroups,nconfigs,nmax,fdin,fdout
      integer(c_size_t) i4,r8
      parameter (i4=4,r8=8)
      character*512 pipe_in,pipe_out
      double precision, allocatable, target :: x(:,:),totals(:,:)
     $     ,ev_dbl(:,:,:),ev_mom(:,:,:,:),parts(:,:)
      integer(c_int), allocatable, target :: ichan(:),igroup(:),nev(:)
     $     ,ev_int(:,:,:)

      if (command_argument_count().ne.2) then
         write (*,*) 'usage: madevent_mlapi <pipe in> <pipe out>'
         stop 1
      endif
      call get_command_argument(1,pipe_in)
      call get_command_argument(2,pipe_out)
c the caller opens the same two pipes in the same order
      fdin=open_pipe(pipe_in,.false.)
      fdout=open_pipe(pipe_out,.true.)

      call mlapi_init(info)
      ndim=info(1)
      ngroups=info(3)
      nconfigs=info(4)
      call write_bytes(fdout,c_loc(info),8*i4)
      do i=1,ngroups
         call mlapi_group(i,members)
         call write_bytes(fdout,c_loc(members),(fks_configs+1)*i4)
      enddo
      do i=1,nconfigs
         call mlapi_config(i,idiag)
         call write_bytes(fdout,c_loc(idiag),i4)
      enddo

      nmax=0
      do
         call read_bytes(fdin,c_loc(command),i4)
         if (command.eq.0) exit
         if (command.ne.1) then
            write (*,*) 'mlapi_server: unknown command',command
            stop 1
         endif
         call read_bytes(fdin,c_loc(n),i4)
         if (n.gt.nmax) then
            if (allocated(x)) deallocate(x,ichan,igroup,totals,nev
     $           ,ev_int,ev_dbl,ev_mom,parts)
            nmax=n
            allocate(x(ndim,n),ichan(n),igroup(n),totals(4,n),nev(n)
     $           ,ev_int(3,maxev,n),ev_dbl(3,maxev,n)
     $           ,ev_mom(0:3,nexternal,maxev,n),parts(nparts,n))
         endif
c the first n columns of every array are contiguous
         call read_bytes(fdin,c_loc(x),ndim*n*r8)
         call read_bytes(fdin,c_loc(ichan),n*i4)
         call read_bytes(fdin,c_loc(igroup),n*i4)
         call mlapi_eval(n,x,ichan,igroup,totals,nev,ev_int,ev_dbl
     $        ,ev_mom,parts)
         call write_bytes(fdout,c_loc(totals),4*n*r8)
         call write_bytes(fdout,c_loc(nev),n*i4)
         call write_bytes(fdout,c_loc(ev_int),3*maxev*n*i4)
         call write_bytes(fdout,c_loc(ev_dbl),3*maxev*n*r8)
         call write_bytes(fdout,c_loc(ev_mom),4*nexternal*maxev*n*r8)
         call write_bytes(fdout,c_loc(parts),nparts*n*r8)
      enddo
      end
