c The matrix-element blocks of the FKS integrand from the Fortran of an
c aMC@NLO output, for madgraph/mg7/nlo/provider.py, the Fortran provider
c of the FKS integrand of madspace. It is linked in a P* directory as
c check_poles (the objects of madevent_mintFO without its driver) and
c serves requests over two pipes, whose file descriptors are its two
c arguments. Every block is taken with the strong coupling g = 1, so
c that each split-order position carries no power of g. The answers
c are those of the calls that sigint of driver_mintFO.f and
c fks_integrand_dump.f make:
c - REAL: smatrix_real_frame at the event of one FKS configuration,
c   amp_split per position.
c - BORN: sborn_frame at the Born with the azimuthal part of the
c   configuration, amp_split per position (the Born orders), the
c   counterterm parts amp_split_cnt(:,1:2,:) and amp2 of the Born
c   diagrams of the channels (mapconfig(:,0)).
c - LINKS: sborn_sf_frame for the given pairs of legs of the real
c   process of one configuration, amp_split_soft per position.
c - VIRTUAL: sloopmatrix_thres of MadLoop at the Born with mu_R given,
c   the finite part, the poles and the accuracy per position, for the
c   orders of the NLO, with the return code of MadLoop, and with tagged
c   photons and QED corrections the change between alpha(0) and G_mu of
c   their couplings (get_virtual_a0Gmu_conv), as BinothLHA adds it.
c - CHANNELS: the Born configurations (born_conf.inc, born_props.inc),
c   per configuration its diagram, for each propagator its two daughters,
c   sprop and tprid, then the masses and widths of the propagators, as
c   write_born_forest of fks_kinematics_dump.f writes them.
c - COUPLINGS: the number of tagged photons (get_n_tagged_photons), the QED
c   coupling gal(1)^2 / (4 pi) of the counterterms, and for the given
c   powers of e the factor of a contribution with tagged photons
c   (get_rescale_alpha_factor), by which add_wgt multiplies its weight.
c - SCALE: the reference scale of the user (user_dynamical_scale of
c   dummy_fct.f) at kinematics of the real process, for
c   dynamical_scale_choice 0 and 10.
c - EXTRA: extra_cnt_frame at the Born of a configuration with an extra
c   counterterm (born_cnt_<n>.f, the Born of the other mother of a quark
c   pair), its counterterm parts amp_split_cnt(:,1:2,:), as sborncol_fsr
c   and sborncol_isr of fks_singular.f take them.
c - EWSUDAKOV_INFO: whether the output has the Sudakov approximation of
c   the EW corrections (--ewsudakov), its sizes, the base amplitude of
c   each helicity combination of the Born and of each interference, and
c   its parameters (fks_provider_ewsudakov.f).
c - EWSUDAKOV: the matrix elements of the Sudakov approximation per
c   helicity combination at the Born of one configuration, those that
c   sudakov_wrapper of ewsudakov_wrapper.f takes, for EWSudakov of
c   madspace.
c - EWSUDAKOV_WEIGHTS: the weights of compute_ewsudakov of fks_singular.f
c   at the Born of one configuration and a renormalization scale per
c   point, for the tests.
c A request is three integers (command, number of points, argument)
c then its data, an answer is a status (0 for success) then its data,
c all in the native byte order. Command 0 stops the server, 1 gives
c the sizes and the orders of the split-order positions.
      module fks_provider_io
      use iso_c_binding
      implicit none
      integer(c_int) fd_in,fd_out
      interface
         function c_read(fd,buf,n) bind(C,name='read')
         import c_int,c_ptr,c_size_t,c_long
         integer(c_int),value :: fd
         type(c_ptr),value :: buf
         integer(c_size_t),value :: n
         integer(c_long) c_read
         end function c_read
         function c_write(fd,buf,n) bind(C,name='write')
         import c_int,c_ptr,c_size_t,c_long
         integer(c_int),value :: fd
         type(c_ptr),value :: buf
         integer(c_size_t),value :: n
         integer(c_long) c_write
         end function c_write
      end interface
      contains
      subroutine transfer_bytes(buf,nbytes,reading)
      type(c_ptr) buf
      integer(c_size_t) nbytes,done
      integer(c_long) got
      logical reading
      integer(c_int8_t),pointer :: bytes(:)
      if (nbytes.eq.0) return
      call c_f_pointer(buf,bytes,[nbytes])
      done=0
      do while (done.lt.nbytes)
         if (reading) then
            got=c_read(fd_in,c_loc(bytes(done+1)),nbytes-done)
         else
            got=c_write(fd_out,c_loc(bytes(done+1)),nbytes-done)
         endif
         if (got.le.0) then
            write (*,*) 'fks_provider: the pipe closed'
            stop 1
         endif
         done=done+got
      enddo
      end subroutine transfer_bytes
      subroutine read_ints(n,values)
      integer n
      integer(c_int32_t),target :: values(n)
      call transfer_bytes(c_loc(values),int(4*n,c_size_t),.true.)
      end subroutine read_ints
      subroutine read_reals(n,values)
      integer n
      real(c_double),target :: values(n)
      call transfer_bytes(c_loc(values),int(8*n,c_size_t),.true.)
      end subroutine read_reals
      subroutine write_ints(n,values)
      integer n
      integer(c_int32_t),target :: values(n)
      call transfer_bytes(c_loc(values),int(4*n,c_size_t),.false.)
      end subroutine write_ints
      subroutine write_reals(n,values)
      integer n
      real(c_double),target :: values(n)
      call transfer_bytes(c_loc(values),int(8*n,c_size_t),.false.)
      end subroutine write_reals
      subroutine write_status(status)
      integer status
      integer(c_int32_t) values(1)
      values(1)=status
      call write_ints(1,values)
      end subroutine write_status
      end module fks_provider_io


      program fks_provider
      use fks_provider_io
      implicit none
      integer(c_int32_t) header(3)
      character*32 arg
      call get_command_argument(1,arg)
      read (arg,*) fd_in
      call get_command_argument(2,arg)
      read (arg,*) fd_out
      call provider_init
      do
         call read_ints(3,header)
         select case (header(1))
         case (0)
            exit
         case (1)
            call provider_info
         case (2)
            call provider_real(header(2),header(3))
         case (3)
            call provider_born(header(2),header(3))
         case (4)
            call provider_links(header(2),header(3))
         case (5)
            call provider_virtual(header(2))
         case (6)
            call provider_channels
         case (7)
            call provider_extra(header(2),header(3))
         case (8)
            call provider_couplings(header(2))
         case (9)
            call provider_scale(header(2))
         case (10)
            call provider_ewsudakov_info
         case (11)
            call provider_ewsudakov(header(2),header(3))
         case (12)
            call provider_ewsudakov_weights(header(2),header(3))
         case default
            call write_status(1)
         end select
      enddo
      end


c The setup of init_test_limits of test_soft_col_limits.f at fixed
c order, with g = 1.
      subroutine provider_init
      use FKSParams, only: FKSParamReader,paramFileName,
     $     print_fks_isr_mapping,configure_fxfx_sudakov
      use mint_module
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      include 'genps.inc'
      include 'coupl.inc'
      logical Hevents
      common/SHevents/Hevents
      nlo_ps=.false.
      fixed_order=.true.
      call setrun
      call FKSParamReader(paramFileName,.true.,.false.)
      call configure_fxfx_sudakov(ickkw,nlo_ps,shower_mc,.true.)
      call print_fks_isr_mapping(nlo_ps,shower_mc)
      call setpara('param_card.dat')
      call setcuts
      call fill_configurations_common
      call check_amp_split
      call init_process_module_global(shower_mc,'all ',nexternal
     $     ,nincoming,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow
     $     ,ickkw)
      call init_process_module_nbody_wrapper()
      call init_scale_module(nexternal,shower_scale_factor,fks_configs
     $     ,1)
      Hevents=.true.
      g=1d0
      call update_as_param()
      end


c The configuration of the FKS integrand, as sigint sets it for the
c n+1-body contributions.
      subroutine provider_configuration(nfks)
      implicit none
      integer nfks,dum
      call update_fks_dir(nfks)
      call init_process_module_n1body_wrapper(dum)
      end


      subroutine provider_info
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      integer(c_int32_t) values(9+(amp_split_size+2)*nsplitorders)
      integer orders(nsplitorders),iamp,k,i
      values(1:9)=[nexternal,nincoming,amp_split_size
     $     ,amp_split_size_born,nsplitorders,qcd_pos,qed_pos
     $     ,fks_integrated,mapconfig(0,0)]
      i=9
      do iamp=1,amp_split_size
         call amp_split_pos_to_orders(iamp,orders)
         do k=1,nsplitorders
            values(i+k)=orders(k)
         enddo
         i=i+nsplitorders
      enddo
      values(i+1:i+nsplitorders)=born_orders
      values(i+nsplitorders+1:i+2*nsplitorders)=nlo_orders
      call write_status(0)
      call write_ints(size(values),values)
      end


c The real per split-order position at count events of configuration
c nfks, the momenta in the order of its real process.
      subroutine provider_real(count,nfks)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer count,nfks,k
      double precision, allocatable :: p(:,:,:),out(:,:)
      double precision wme
      allocate(p(0:3,nexternal,count),out(amp_split_size,count))
      call read_reals(4*nexternal*count,p)
      call provider_configuration(nfks)
      do k=1,count
         call smatrix_real_frame(p(0,1,k),wme)
         out(:,k)=amp_split(1:amp_split_size)
      enddo
      call write_status(0)
      call write_reals(amp_split_size*count,out)
      deallocate(p,out)
      end


c The Born at count Born kinematics, with the azimuthal part of
c configuration nfks: per point amp_split, amp_split_cnt (real and
c imaginary parts in the order of memory) and amp2 of the Born
c diagrams of the channels.
      subroutine provider_born(count,nfks)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      double precision amp2(ngraphs),jamp2(0:ncolor)
      common/to_amps/amp2,jamp2
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer count,nfks,k,i,size_cnt,width
      double precision, allocatable :: p(:,:,:),out(:,:)
      double precision wme
      size_cnt=4*amp_split_size*nsplitorders
      width=amp_split_size+size_cnt+mapconfig(0,0)
      allocate(p(0:3,nexternal-1,count),out(width,count))
      call read_reals(4*(nexternal-1)*count,p)
      call provider_configuration(nfks)
      do k=1,count
         calculatedBorn=.false.
         call sborn_frame(p(0,1,k),wme)
         out(1:amp_split_size,k)=amp_split(1:amp_split_size)
         out(amp_split_size+1:amp_split_size+size_cnt,k)=
     $        transfer(amp_split_cnt,0d0,size_cnt)
         do i=1,mapconfig(0,0)
            out(amp_split_size+size_cnt+i,k)=amp2(mapconfig(i,0))
         enddo
      enddo
      call write_status(0)
      call write_reals(width*count,out)
      deallocate(p,out)
      end


c The reference scale of the user at count kinematics of the real process
c (nexternal momenta, in the partonic center-of-mass frame as sigint takes
c them).
      subroutine provider_scale(count)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      integer count,k
      double precision, allocatable :: p(:,:,:),out(:)
      double precision user_dynamical_scale
      external user_dynamical_scale
      allocate(p(0:3,nexternal,count),out(count))
      call read_reals(4*nexternal*count,p)
      do k=1,count
         out(k)=user_dynamical_scale(p(0,1,k))
      enddo
      call write_status(0)
      call write_reals(count,out)
      deallocate(p,out)
      end


c The number of tagged photons, the QED coupling and the factors of the
c contributions with tagged photons for count powers of e.
      subroutine provider_couplings(count)
      use fks_provider_io
      implicit none
      include 'coupl.inc'
      integer count,k
      integer(c_int32_t) ntag(1)
      integer(c_int32_t), allocatable :: powers(:)
      double precision, allocatable :: out(:)
      integer get_n_tagged_photons
      double precision get_rescale_alpha_factor
      external get_n_tagged_photons,get_rescale_alpha_factor
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      allocate(powers(count),out(count+1))
      call read_ints(count,powers)
      call provider_configuration(1)
      ntag(1)=get_n_tagged_photons()
      out(1)=abs(gal(1))**2/(4d0*pi)
      do k=1,count
         out(k+1)=get_rescale_alpha_factor(ntag(1),powers(k))
      enddo
      call write_status(0)
      call write_ints(1,ntag)
      call write_reals(count+1,out)
      deallocate(powers,out)
      end


c Whether the output has the Sudakov approximation of the EW corrections:
c five integers (one with it, the number of helicity combinations of the
c Born and of interferences, has_lo1 and has_lo2), with it then the base
c amplitude of each helicity combination and of each interference and
c nine parameters.
      subroutine provider_ewsudakov_info
      use fks_provider_io
      implicit none
      integer(c_int32_t) sizes(5)
      integer(c_int32_t), allocatable :: bases(:)
      double precision parameters(9)
      integer ncomb,nint
      call provider_sdk_sizes(sizes)
      call write_status(0)
      call write_ints(5,sizes)
      if (sizes(1).eq.0) return
      ncomb=sizes(2)
      nint=sizes(3)
      allocate(bases(ncomb+nint))
      call provider_sdk_info(ncomb,nint,bases(1),bases(ncomb+1),
     $     parameters)
      call write_ints(ncomb+nint,bases)
      call write_reals(9,parameters)
      deallocate(bases)
      end


c The matrix elements of the Sudakov approximation at count Born
c kinematics of configuration nfks (in the partonic center-of-mass
c frame), per point those of provider_sdk_evaluate, status 2 for an
c output without it.
      subroutine provider_ewsudakov(count,nfks)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      integer count,nfks,k,width
      integer(c_int32_t) sizes(5)
      double precision, allocatable :: p(:,:,:),out(:,:)
      call provider_sdk_sizes(sizes)
      width=8*sizes(2)+2*sizes(3)*sizes(2)
      allocate(p(0:3,nexternal-1,count),out(max(width,1),count))
      call read_reals(4*(nexternal-1)*count,p)
      if (sizes(1).eq.0) then
         call write_status(2)
         deallocate(p,out)
         return
      endif
      call provider_configuration(nfks)
      do k=1,count
         call provider_sdk_evaluate(p(0,1,k),sizes(2),sizes(3),out(1,k))
      enddo
      call write_status(0)
      call write_reals(width*count,out)
      deallocate(p,out)
      end


c The weights of compute_ewsudakov at count Born kinematics of
c configuration nfks (in the partonic center-of-mass frame) and a
c renormalization scale per point, per point those of
c provider_sdk_weights (per sud_mod the weight at the NLO order of the
c LO1 and the sum of the others), status 2 for an output without it.
      subroutine provider_ewsudakov_weights(count,nfks)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      integer count,nfks,k
      integer(c_int32_t) sizes(5)
      double precision, allocatable :: p(:,:,:),scales(:),out(:,:)
      allocate(p(0:3,nexternal-1,count),scales(count),out(4,count))
      call read_reals(count,scales)
      call read_reals(4*(nexternal-1)*count,p)
      call provider_sdk_sizes(sizes)
      if (sizes(1).eq.0) then
         call write_status(2)
         deallocate(p,scales,out)
         return
      endif
      call provider_configuration(nfks)
      do k=1,count
         call provider_sdk_weights(p(0,1,k),scales(k),out(1,k))
      enddo
      call write_status(0)
      call write_reals(4*count,out)
      deallocate(p,scales,out)
      end


c The counterterm parts amp_split_cnt of the Born of the extra counterterm
c of configuration nfks at count Born kinematics (real and imaginary parts
c in the order of memory), status 2 for a configuration without one.
      subroutine provider_extra(count,nfks)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer iextra_cnt,isplitorder_born,isplitorder_cnt
      common /c_extra_cnt/iextra_cnt,isplitorder_born,isplitorder_cnt
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer count,nfks,k,size_cnt
      double precision, allocatable :: p(:,:,:),out(:,:)
      double complex cnts(2,nsplitorders)
      size_cnt=4*amp_split_size*nsplitorders
      allocate(p(0:3,nexternal-1,count),out(size_cnt,count))
      call read_reals(4*(nexternal-1)*count,p)
      call provider_configuration(nfks)
      if (iextra_cnt.le.0) then
         call write_status(2)
         deallocate(p,out)
         return
      endif
      do k=1,count
         calculatedBorn=.false.
         call extra_cnt_frame(p(0,1,k),iextra_cnt,cnts)
         out(:,k)=transfer(amp_split_cnt,0d0,size_cnt)
      enddo
      call write_status(0)
      call write_reals(size_cnt*count,out)
      deallocate(p,out)
      end


c The colour- or charge-linked Borns of sborn_sf at count Born
c kinematics for npairs pairs of legs of the real process of
c configuration nfks (that sbornsoft takes, fks_j_from_i of i_fks),
c amp_split_soft per pair.
      subroutine provider_links(count,npairs)
      use fks_provider_io
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer count,npairs,k,l
      integer(c_int32_t) nfks(1)
      integer(c_int32_t), allocatable :: pairs(:,:)
      double precision, allocatable :: p(:,:,:),out(:,:,:)
      double precision wme
      allocate(pairs(2,npairs),p(0:3,nexternal-1,count)
     $     ,out(amp_split_size,npairs,count))
      call read_ints(1,nfks)
      call read_ints(2*npairs,pairs)
      call read_reals(4*(nexternal-1)*count,p)
      call provider_configuration(nfks(1))
      do k=1,count
         calculatedBorn=.false.
         call sborn_frame(p(0,1,k),wme)
         do l=1,npairs
            call sborn_sf_frame(p(0,1,k),pairs(1,l),pairs(2,l),wme)
            out(:,l,k)=amp_split_soft(1:amp_split_size)
         enddo
      enddo
      call write_status(0)
      call write_reals(amp_split_size*npairs*count,out)
      deallocate(pairs,p,out)
      end


c The virtual of MadLoop at count Born kinematics, each with its
c renormalization scale (the scale of the finite part), summed over
c helicities as BinothLHA calls it without helicity sampling: per point
c the return code of MadLoop, then per split-order position the
c finite part, the single and double poles and the accuracy MadLoop
c estimates, zero outside the orders of the NLO. Without the pole
c check of BinothLHA and the conversion of tagged photons.
      subroutine provider_virtual(count)
      use fks_provider_io
      use FKSParams, only: PrecisionVirtualAtRunTime
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'orders.inc'
      double precision qes2
      common /coupl_es/ qes2
      logical updateloop
      common /to_updateloop/updateloop
      integer count,k,i,j,ret_code,amp_orders(nsplitorders),pos
      integer nsqso,MLResArrayDim,ntagph,qed_pow_b
      integer getordpowfromindex_ml5,orders_to_amp_split_pos
      integer get_n_tagged_photons
      double precision get_virtual_a0Gmu_conv,born_wgt
      external get_n_tagged_photons,get_virtual_a0Gmu_conv
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      double precision, allocatable :: p(:,:,:),out(:,:,:),scales(:)
      double precision pf(0:3,nexternal-1)
      integer ids(nexternal-1)
      integer(c_int32_t), allocatable :: codes(:)
      double precision, allocatable, save :: virt_wgts(:,:)
      double precision, allocatable, save :: accuracies(:)
      logical, allocatable, save :: keep_order(:)
      logical firsttime
      data firsttime/.true./
      save firsttime,nsqso,MLResArrayDim,ntagph
      if (firsttime) then
         firsttime=.false.
c the tagged photons of the Born, those of every configuration
         call provider_configuration(1)
         ntagph=get_n_tagged_photons()
         call set_forbid_hel_doublecheck(.true.)
         call get_nsqso_loop(nsqso)
         call get_answer_dimension(MLResArrayDim)
         allocate(accuracies(0:nsqso),virt_wgts(0:3,0:MLResArrayDim)
     $        ,keep_order(nsqso))
         call force_stability_check(.true.)
         call collier_compute_uv_poles(.true.)
         call collier_compute_ir_poles(.true.)
         do i=1,nsqso
            keep_order(i)=.true.
            do j=1,nsplitorders
               if (getordpowfromindex_ml5(j,i).gt.nlo_orders(j))
     $              keep_order(i)=.false.
            enddo
         enddo
      endif
      allocate(scales(count),p(0:3,nexternal-1,count)
     $     ,out(4,amp_split_size,count),codes(count))
      call read_reals(count,scales)
      call read_reals(4*(nexternal-1)*count,p)
      call get_frame_mask_born(ids)
      out=0d0
      do k=1,count
c the couplings of the loops as well, as BinothLHA updates them
         mu_r=scales(k)
         qes2=scales(k)**2
         updateloop=.true.
         call update_as_param()
         updateloop=.false.
         call boost_to_me_frame(p(0,1,k),nexternal-1,ids,pf)
         call sloopmatrix_thres(pf,virt_wgts,PrecisionVirtualAtRunTime
     $        ,accuracies,ret_code)
         codes(k)=ret_code
         do i=1,nsqso
            if (.not.keep_order(i)) cycle
            do j=1,nsplitorders
               amp_orders(j)=getordpowfromindex_ml5(j,i)
            enddo
            pos=orders_to_amp_split_pos(amp_orders)
            out(1:3,pos,k)=virt_wgts(1:3,i)
            out(4,pos,k)=accuracies(i)
         enddo
c the change of scheme of the couplings of tagged photons, per Born order
c at the order of its QED correction
         if (ntagph.ne.0 .and. qed_pos.gt.0) then
            calculatedBorn=.false.
            call sborn_frame(p(0,1,k),born_wgt)
            do i=1,amp_split_size_born
               call amp_split_pos_to_orders(i,amp_orders)
               born_wgt=amp_split(i)
               qed_pow_b=amp_orders(qed_pos)
               amp_orders(qed_pos)=amp_orders(qed_pos)+2
               if (amp_orders(qed_pos).gt.nlo_orders(qed_pos)) cycle
               pos=orders_to_amp_split_pos(amp_orders)
               out(1,pos,k)=out(1,pos,k)+
     $              get_virtual_a0Gmu_conv(qed_pow_b,ntagph,0,born_wgt)
               out(2,pos,k)=out(2,pos,k)+
     $              get_virtual_a0Gmu_conv(qed_pow_b,ntagph,1,born_wgt)
            enddo
         endif
      enddo
      call write_status(0)
      call write_ints(count,codes)
      call write_reals(4*amp_split_size*count,out)
      deallocate(scales,p,out,codes)
      end


c The Born configurations: their number and that of the propagators,
c then per configuration its diagram (mapconfig) and per propagator
c iforest(1:2), sprop and tprid, then per configuration and propagator
c the mass and the width.
      subroutine provider_channels
      use fks_provider_io
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_conf.inc'
      include 'coupl.inc'
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      integer i,k,branches,l
      integer(c_int32_t), allocatable :: ints(:)
      double precision, allocatable :: reals(:)
      pmass=0d0
      pwidth=0d0
      include 'born_props.inc'
      branches=nexternal-(2+nincoming)
      allocate(ints(2+mapconfig(0)*(1+4*branches))
     $     ,reals(2*branches*mapconfig(0)))
      ints(1)=mapconfig(0)
      ints(2)=branches
      l=2
      do i=1,mapconfig(0)
         ints(l+1)=mapconfig(i)
         l=l+1
         do k=-1,-branches,-1
            ints(l+1:l+4)=[iforest(1,k,i),iforest(2,k,i),sprop(k,i)
     $           ,tprid(k,i)]
            l=l+4
         enddo
      enddo
      l=0
      do i=1,mapconfig(0)
         do k=-1,-branches,-1
            reals(l+1:l+2)=[pmass(k,i),pwidth(k,i)]
            l=l+2
         enddo
      enddo
      call write_status(0)
      call write_ints(size(ints),ints)
      call write_reals(size(reals),reals)
      deallocate(ints,reals)
      end


c The wrappers of init_test_limits (test_soft_col_limits.f) for the
c process module, which this program links without that file.
      subroutine init_process_module_nbody_wrapper()
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      integer colour(1:nexternal-1),i,get_color
      external get_color
      double precision mass(1:nexternal-1),get_mass_from_id
      external get_mass_from_id
      logical valid_dipole(1:nexternal-1,1:nexternal-1,1:max_bcol)
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      do i=1,nexternal-1
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      call init_process_module_nbody(nexternal-1,mass,colour
     $     ,max_bcol,valid_dipole)
      end


      subroutine init_process_module_n1body_wrapper(dum)
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer colour(1:nexternal),i,get_color,dum
      double precision mass(1:nexternal),get_mass_from_id
      external get_color
      external get_mass_from_id
      logical valid_dipole(1:nexternal,1:nexternal)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     dummy(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,dummy,niprocs
      do i=1,nexternal
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      call init_process_module_n1body(nexternal,mass,colour
     $     ,maxflow,valid_dipole)
      end
