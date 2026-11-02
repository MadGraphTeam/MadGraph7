CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                          C
C                  HwU: Histograms with Uncertainties                      C
C                 By Rikkert Frederix, 12-2014--05-2017                    C
C                                                                          C
C     Book, fill and write out histograms. Particularly suited for NLO     C
C     computations with correlations between points (ie. event and         C
C     counter-event) and multiple weights for given points (e.g. scale     C
C     and PDF uncertainties through reweighting).                          C
C                                                                          C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      module HwU_wgts_info_len
         use iso_c_binding
         integer, parameter :: wgts_info_len=80 
         contains 
             integer function get_wgts_info_len() bind(c,name="get_wgts_info_len")
                 get_wgts_info_len = wgts_info_len
                 return
             end function get_wgts_info_len

      end module HwU_wgts_info_len


c The module contains effectively the common block with allocatable
c variables (something not possible in old fortran version)
      module HwU_variables
         use HwU_wgts_info_len
         implicit none
         integer :: max_plots,max_points,max_bins,nwgts,np
         integer :: error_estimation=3
         logical, allocatable :: booked(:)
         integer, allocatable :: nbin(:),histi(:,:),p_bin(:),p_label(:)
         character(len=50), allocatable :: title(:)
         character(len=wgts_info_len), allocatable :: wgts_info(:)
         double precision, allocatable :: histy(:,:,:),histy_acc(:,:,:)
     $        ,histy2(:,:),histy_err(:,:),histxl(:,:),histxm(:,:)
     $        ,step(:),p_wgts(:,:)
c The spin-density-matrix weights (labels starting with 'dens '): their
c positions idens(1:ndens) among the weights and, per bin, the sums of
c the squares of w_K (histd2(2K-1,:,:)) and of w_K + w_1 (histd2(2K,:,:))
c for the current iteration, and the accumulated uncertainties of
c these two (histd_err), written as the extra columns '<label> @dy' and
c '<label> @dyc'. With both, the covariance of w_K with the central
c value, and so the uncertainty of w_K/w_1, can be computed.
         integer :: ndens=0
         integer, allocatable :: idens(:)
         double precision, allocatable :: histd2(:,:,:),histd_err(:,:,:)
         save
      end module HwU_variables

      
c To be called once at the start of each run. Initialises the packages
c and sets the number of weights that need to be included for each point.
      subroutine HwU_inithist(nweights,wgt_info)
      use HwU_variables
      implicit none
      integer i,nweights
      character*(*) wgt_info(*)

c     PineAPPL commons
      include "reweight_pineappl.inc"
      include "pineappl_common.inc"
      logical pineappl
      common /for_pineappl/ pineappl

C     Initialize the number of bins of the aMCfast grids
      if(pineappl) appl_obs_nbins = 0

      call HwU_deallocate_all
      max_plots=0
      max_points=0
      max_bins=0
      np=0
c     Number of weights associated to each point. Note that the first
c     weight should always be the 'central value' and it should not be
c     zero if any of the other weights are non-zero.
      nwgts=nweights
      allocate(wgts_info(nwgts))
      do i=1,nwgts
         wgts_info(i)=wgt_info(i)
      enddo
      ndens=0
      do i=1,nwgts
         if (wgts_info(i)(1:5).eq.'dens ') ndens=ndens+1
      enddo
      allocate(idens(ndens))
      ndens=0
      do i=1,nwgts
         if (wgts_info(i)(1:5).eq.'dens ') then
            ndens=ndens+1
            idens(ndens)=i
         endif
      enddo
      return
      end

      subroutine set_error_estimation(input)
c Error estimation
c     input 0: Simply sum all PS points; error is estimated by assuming
c        Binomial statistics: use for unweighted events. (Assumes that
c        all events have equal weight (up to a sign)).
c     input 1: Simply sum all PS points; error is estimated by the
c        variance between the separate iterations. This assumes that
c        each iteration has the same number of PS points (This is the
c        topdrawer default in MadGraph7).
c     input 2: Sum PS points for a given iteration and error estimate by
c       square root of the sum of the squares. Perform a weighted average
c       iteration-by-iteration
c     input 3: Same as input 2, but weighted average is same as from MINT
      use HwU_variables
      implicit none
      integer input
      if (input.ge.0 .and. input.le.3) then
         error_estimation=input
      else
         write (*,*) 'unknown error estimation',input
         stop 1
      endif
      return
      end
            
c Book the histograms at the start of the run. Give a 'label' (an
c integer) that identifies the plot when filling it and a title
c ('title_l') for each plot. Also the number of bins ('nbin_l') and the
c plot range (from 'xmin' to 'xmax') should be given.
      subroutine HwU_book(label,title_l,nbin_l,xmin,xmax)
      use HwU_variables
      implicit none
      integer label,nbin_l,i,j
      character*(*) title_l
      double precision xmin,xmax
c     PineAPPL commons
      include "reweight_pineappl.inc"
      include "pineappl_common.inc"
      logical pineappl
      common /for_pineappl/ pineappl
      double precision del

c     Initialize the grids only if the switch "pineappl" is set to True
c     and if the title  does not contain the word "Born". 
      if(pineappl.and.index(title_l,"Born").eq.0)then
c     Observable parameters
c     Compute number of bins and edges only if they have not been given by the user.
         if(appl_obs_nbins.eq.0)then
            appl_obs_nbins = nbin_l
            ! bin width
            del = (xmax - xmin) / nbin_l
c     compute bin edges
            do i=0,appl_obs_nbins
               appl_obs_bins(i) = xmin + i * del
            enddo
         endif
         appl_obs_min = appl_obs_bins(0)
         appl_obs_max = appl_obs_bins(appl_obs_nbins)
         if(abs(appl_obs_max-xmax).gt.0.00000001d0)then
            write(*,*) 'PineAPPL Histogram: ', 
     1                 'Change of the upper limit:',xmax,'-->',
     2                  appl_obs_max
         endif
c     Initialize PineAPPL routines
         call APPL_init
c     Keep track of the position of this histogram
         nh_obs = nh_obs + 1
         ih_obs(nh_obs) = label
c     Reset number of bins to zero
         appl_obs_nbins = 0
      endif

c     Allocate space for new histograms if needed      
      call HwU_allocate_histo(label,nbin_l)
c     Setup the histogram
      booked(label)=.true.
      title(label)=title_l
      nbin(label)=nbin_l
c     Compute the bin width
      step(label)=(xmax-xmin)/dble(nbin(label))
      do i=1,nbin(label)
c     Compute the lower and upper bin edges
         histxl(label,i)=xmin+step(label)*dble(i-1)
         histxm(label,i)=xmin+step(label)*dble(i)
c     Set all the bins to zero.
         do j=1,nwgts
            histy(j,label,i)=0d0
            histy_acc(j,label,i)=0d0
         enddo
         histi(label,i)=0
         histy2(label,i)=0d0
         histy_err(label,i)=0d0
         histd2(:,label,i)=0d0
         histd_err(:,label,i)=0d0
      enddo
      return
      end
      
c Fill the histograms identified with 'label' with a point at x with
c weights giving by 'wgts'. The dimension of the 'wgts' array should
c always be as large as the number of weights ('nweights') specified in
c the 'HwU_inithist' subroutine. That means that each point should have
c the same number of weights.
      subroutine HwU_fill(label,x,wgts)
      use HwU_variables
      implicit none
      integer label,i,j,bin
      double precision x, wgts(*)

c     PineAPPL commons
      include "reweight_pineappl.inc"
      include "pineappl_common.inc"
      logical pineappl
      common /for_pineappl/ pineappl
      if(pineappl)then
         do j=1,nh_obs
            if(label.eq.ih_obs(j))then
               appl_obs_num   = j
               appl_obs_histo = x
c     Fill the reference PineAPPL histograms
c     Fill the PineAPPL files
               call APPL_fill
            endif
         enddo
      endif

c     If central weight is zero do not add this point.
      if (wgts(1).eq.0d0) return
c     Check if point is within plotting range
      if (x.lt.histxl(label,1) .or.
     $     x.gt.histxm(label,nbin(label))) return
c     Compute the bin to fill
      bin=int((x-histxl(label,1))/step(label)) +1
c     To prevent numerical inaccuracy, double check that bin is
c     reasonable
      if (bin.lt.1 .or. bin.gt.nbin(label)) return
c     Check if we already had another (correlated) point in this bin. If
c     so, add the current weight to that point.
      do i=1,np
         if ( p_label(i).eq.label .and. p_bin(i).eq.bin) then
            do j=1,nwgts
               p_wgts(j,i)=p_wgts(j,i)+wgts(j)
            enddo
            return
         endif
      enddo
c     If a new bin, add it to the list of points
      np=np+1
      call HwU_allocate_p
      p_label(np)=label
      p_bin(np)=bin
      do j=1,nwgts
         p_wgts(j,np)=wgts(j)
      enddo
      return
      end
      
c Call after all correlated contributions for a give phase-space
c point. I.e., every time you get a new set of random numbers from
c MINT/VEGAS. It adds the current list of points to the histograms. Add
c the squares to compute the statistical uncertainty on the bin. Do the
c second only for the weight corresponding to the 'central value'. In
c this way, correlations between events and counter-events can be
c correctly taken into account.
      subroutine HwU_add_points
      use HwU_variables
      implicit none
      integer i,j,k
      do i=1,np
         do j=1,nwgts
            histy(j,p_label(i),p_bin(i))=
     $           histy(j,p_label(i),p_bin(i))+p_wgts(j,i)
         enddo
         histi(p_label(i),p_bin(i))=histi(p_label(i),p_bin(i))+1
         histy2(p_label(i),p_bin(i))=
     $        histy2(p_label(i),p_bin(i))+p_wgts(1,i)**2
         do k=1,ndens
            histd2(2*k-1,p_label(i),p_bin(i))=histd2(2*k-1,p_label(i)
     $           ,p_bin(i))+p_wgts(idens(k),i)**2
            histd2(2*k,p_label(i),p_bin(i))=histd2(2*k,p_label(i)
     $           ,p_bin(i))+(p_wgts(idens(k),i)+p_wgts(1,i))**2
         enddo
      enddo
      np=0
      return
      end


c *****For plotting during fixed order f(N)LO runs*****
c Call after every iteration. This adds the histograms of the current
c iteration ('histy') to the accumulated results ('histy_acc'), with the
c uncertainty estimate given in 'histy_err' and empties the arrays for
c the current iteration so that they can be filled with the next
c iteration.
      subroutine HwU_accum_iter(inclde,nPSpoints,values)
      use HwU_variables
      implicit none
      logical inclde
      integer nPSpoints,label,i,j
      double precision nPSinv,etot,niter,y_squared,values(2)
      data niter /0d0/
      nPSinv = 1d0/dble(nPSpoints)
      if (inclde) niter = niter+1d0
      do label=1,max_plots
         if (.not. booked(label)) cycle
         if (inclde) then
            call accumulate_results(label,nPSinv,niter,values)
         endif
c     Reset the histo of the current iteration to zero so that they are
c     ready for the next iteration.
         do i=1,nbin(label)
            do j=1,nwgts
               histy(j,label,i)=0d0
            enddo
            histy2(label,i)=0d0
            histd2(:,label,i)=0d0
            histi(label,i)=0
         enddo
      enddo
      return
      end

c *****For plotting unweighted events****
c Overwrites the accumulated results ('histy_acc') with the current
c results ('histy') and provides an uncertainty estimate in
c 'histy_err'. This means that during the plotting of events, at
c intermediate stages this function can be called (together with
c HwU_output) to write intermediate plots to disk.
      subroutine finalize_histograms(nPSpoints)
      use HwU_variables
      implicit none
      integer label,nPSpoints,i,j
      double precision nPSinv,niter,dummy(2)
      nPSinv=1d0/dble(nPSpoints)
      niter=1d0
      do label=1,max_plots
         if (.not. booked(label)) cycle
         do i=1,nbin(label)
c     Set all the bins to zero.
            do j=1,nwgts
               histy_acc(j,label,i)=0d0
            enddo
            histy_err(label,i)=0d0
            histd_err(:,label,i)=0d0
         enddo
         call accumulate_results(label,nPSinv,niter,dummy)
      enddo
      return
      end

c This adds the histograms of the current iteration ('histy') to the
c accumulated results ('histy_acc'), with the uncertainty estimate given
c in 'histy_err'. When error_estimation==2 it adds using a weight
c corresponding to the statistical uncertainties on the 'central value'
c weights ('etot' (from 'histy2') and 'histy_err', respectively), if
c error_estimation==1 it simply averages over the iterations, if
c error_estimation==0 it uses Poisson statistics to compute the
c uncertainty. Note that this means that during the filling of the
c histograms the central value should not be zero if any of the other
c weights are non-zero.
      subroutine accumulate_results(label,nPSinv,niter,values)
      use HwU_variables
      implicit none
      integer label,i,j,k
      double precision nPSinv,etot,niter,y_squared
     $     ,values(2),a1,a2
      double precision,allocatable :: vtot(:),dtot(:),detot(:),dacc(:)
      if (.not. allocated(vtot)) allocate(vtot(nwgts))
c the density weights: their value (dtot: w_K and w_K + w_1, as in
c histd2) and the uncertainty of the current iteration (detot), and
c their accumulated values (dacc)
      if (.not. allocated(dtot)) allocate(dtot(2*ndens),detot(2*ndens)
     $     ,dacc(2*ndens))
      if (error_estimation.eq.2) then
c     Use the weighted average bin-by-bin. This is not really justified
c     for fNLO computations, because for bins with low statistics, the
c     variance computation cannot be trusted.
         do i=1,nbin(label)
c     Skip bin if no entries
            if (histi(label,i).eq.0) cycle
c     Divide weights by the number of PS points. This means that this is
c     now normalised to the total cross section in that bin
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
            enddo
c     Error estimation of the current bin
            etot=sqrt(abs(histy2(label,i)*nPSinv-vtot(1)**2)*nPSinv)
c     Include "Bessel's correction" to have a corrected (even though
c     still biased) estimator of the standard deviation.
            if (histi(label,i).gt.1) then
               etot=etot* sqrt(dble(histi(label,i))
     &                    /(dble(histi(label,i))-1.5d0))
            else
               etot=abs(vtot(1))*10d0 ! multiply by 10 to make it large
            endif
            call HwU_dens_iter_err(label,i,vtot,nPSinv,dtot,detot)
c     If the error estimation of the accumulated results is still zero
c     (i.e. no points were added yet, e.g. because it is the first
c     iteration) simply copy the results of this iteration over the
c     accumulated results.
            if (histy_err(label,i).eq.0d0) then
               do j=1,nwgts
                  histy_acc(j,label,i)=vtot(j)
               enddo
               histy_err(label,i)=etot
               histd_err(1:2*ndens,label,i)=detot(1:2*ndens)
            else
c     the density weights are combined with the same coefficients
               a1=(1d0/etot)/(1d0/histy_err(label,i)+1d0/etot)
               a2=(1d0/histy_err(label,i))/(1d0/histy_err(label,i)+1d0
     $              /etot)
               do k=1,2*ndens
                  histd_err(k,label,i)=sqrt((a2*histd_err(k,label,i))**2
     $                 +(a1*detot(k))**2)
               enddo
c     Add the results of the current iteration to the accumulated results
               do j=1,nwgts
                  histy_acc(j,label,i)=(histy_acc(j,label,i)
     &                 /histy_err(label,i)+vtot(j)/etot)/(1d0
     &                 /histy_err(label,i) + 1d0/etot)
               enddo
               histy_err(label,i)=
     $              1d0/sqrt(1d0/histy_err(label,i)**2+1d0/etot**2)
            endif
         enddo
      elseif (error_estimation.eq.3) then
         do i=1,nbin(label)
c     Skip bin if no entries
            if (histi(label,i).eq.0) cycle
c     Divide weights by the number of PS points. This means that this is
c     now normalised to the total cross section in that bin
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
            enddo
c     Error estimation of the current bin
            etot=sqrt(abs(histy2(label,i)*nPSinv-vtot(1)**2)*nPSinv)
c     Include "Bessel's correction" to have a corrected (even though
c     still biased) estimator of the standard deviation.
            if (histi(label,i).gt.1) then
               etot=etot* sqrt(dble(histi(label,i))
     &                    /(dble(histi(label,i))-1.5d0))
            else
               etot=abs(vtot(1))*10d0 ! multiply by 10 to make it large
            endif
            call HwU_dens_iter_err(label,i,vtot,nPSinv,dtot,detot)
c     If the error estimation of the accumulated results is still zero
c     (i.e. no points were added yet, e.g. because it is the first
c     iteration) simply copy the results of this iteration over the
c     accumulated results.
            if (histy_err(label,i).eq.0d0) then
               do j=1,nwgts
                  histy_acc(j,label,i)=vtot(j)
               enddo
               histy_err(label,i)=etot
               histd_err(1:2*ndens,label,i)=detot(1:2*ndens)
            else
c     Add the results of the current iteration to the accumulated results
               do j=1,nwgts
                  histy_acc(j,label,i)=(histy_acc(j,label,i)
     &                 /values(2)+vtot(j)/values(1))/(1d0
     &                 /values(2) + 1d0/values(1))
               enddo
               a1=((1d0/values(1))/((1d0/values(1))+1d0/values(2)))**2
               a2=((1d0/values(2))/((1d0/values(1))+1d0/values(2)))**2
               histy_err(label,i)=sqrt(a2*histy_err(label,i)**2 +
     &              a1*etot**2)
c     the density weights are combined with the same coefficients
               do k=1,2*ndens
                  histd_err(k,label,i)=sqrt(a2*histd_err(k,label,i)**2
     $                 +a1*detot(k)**2)
               enddo
            endif
         enddo 
      elseif(error_estimation.eq.1) then
c     simply sum the weights in the bins
         do i=1,nbin(label)
            if (histi(label,i).eq.0 .and.
     &           histy_acc(1,label,i).eq.0d0) cycle
            if (niter.ne.1d0) y_squared=((niter-1)*histy_err(label,i))
     &           **2+(niter-1)*histy_acc(1,label,i)**2
c     the density weights: the same, with their accumulated values
c     before this iteration
            do k=1,ndens
               dacc(2*k-1)=histy_acc(idens(k),label,i)
               dacc(2*k)=histy_acc(idens(k),label,i)+histy_acc(1,label,i)
               dtot(2*k-1)=histy(idens(k),label,i)*nPSinv
               dtot(2*k)=(histy(idens(k),label,i)+histy(1,label,i))
     $              *nPSinv
            enddo
            do k=1,2*ndens
               if (niter.eq.1d0) then
                  histd_err(k,label,i)=0d0
               else
                  a1=((niter-1)*histd_err(k,label,i))**2+(niter-1)
     $                 *dacc(k)**2
                  a2=(dacc(k)*(niter-1d0)+dtot(k))/niter
                  histd_err(k,label,i)=sqrt(abs((a1+dtot(k)**2)/niter
     $                 -a2**2)/niter)
               endif
            enddo
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
               histy_acc(j,label,i)=(histy_acc(j,label,i)*(niter-1d0)
     &              +vtot(j))/niter
            enddo
c     base the error on the variance in the results per iteration. For a
c     small number of iterations, this underestimates the actual
c     uncertainty.
            if (niter.eq.1d0) then
               histy_err(label,i)=0d0
            else
               histy_err(label,i)= sqrt(((y_squared+vtot(1)**2)/niter
     &              -histy_acc(1,label,i)**2)/niter)
            endif
         enddo
      elseif(error_estimation.eq.0) then
c     simply sum the weights in the bins
         do i=1,nbin(label)
            if (histi(label,i).eq.0 .and.
     &           histy_acc(1,label,i).eq.0d0) cycle
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
               histy_acc(j,label,i)=(histy_acc(j,label,i)*(niter-1d0)
     &              +vtot(j))/niter
            enddo
c     Error estimation of the current bin using Poisson statistics,
c     making sure we normalise with the number of iterations.
            if (histi(label,i).ne.0) then
               etot=sqrt(histy2(label,i))*nPSinv
               histy_err(label,i)= sqrt(((niter-1d0)*histy_err(label,i))
     &              **2+etot**2)/niter
               do k=1,2*ndens
                  histd_err(k,label,i)=sqrt(((niter-1d0)*histd_err(k
     $                 ,label,i))**2+histd2(k,label,i)*nPSinv**2)/niter
               enddo
            else
               histy_err(label,i)=(niter-1d0)/niter*histy_err(label,i)
               histd_err(:,label,i)=(niter-1d0)/niter*histd_err(:,label
     $              ,i)
            endif
         enddo
      endif
      end


      subroutine HwU_dens_iter_err(label,i,vtot,nPSinv,dtot,detot)
c The density weights of bin i of histogram label in the current
c iteration: their values (dtot: w_K, w_K + w_1) and uncertainties
c (detot), estimated as that of the central value in accumulate_results
      use HwU_variables
      implicit none
      integer label,i,k
      double precision vtot(nwgts),nPSinv,dtot(2*ndens),detot(2*ndens)
      do k=1,ndens
         dtot(2*k-1)=vtot(idens(k))
         dtot(2*k)=vtot(idens(k))+vtot(1)
      enddo
      do k=1,2*ndens
         detot(k)=sqrt(abs(histd2(k,label,i)*nPSinv-dtot(k)**2)*nPSinv)
         if (histi(label,i).gt.1) then
            detot(k)=detot(k)*sqrt(dble(histi(label,i))
     &           /(dble(histi(label,i))-1.5d0))
         else
            detot(k)=abs(dtot(k))*10d0
         endif
      enddo
      end
      
c Write the histograms to disk at the end of the run, multiplying the
c output by 'xnorm'
      subroutine HwU_output(unit,xnorm)
      use HwU_variables
      implicit none
      integer unit,i,j,k,label
      integer max_length
      character(len=:), allocatable :: buffer
      character*4 str_nbin
      double precision xnorm
      if (.not. allocated(buffer))
     &     allocate(character(len=(nwgts+3)*17) :: buffer)
c     column info: x_min, x_max, y (central value), dy, {extra
c     weights}.
      write (unit,'(a)', advance='no') '##& xmin'
      write (unit,'(a)', advance='no') ' & xmax'
      write (unit,'(a)', advance='no') ' & '//trim(adjustl(wgts_info(1)))
      write (unit,'(a)', advance='no') ' & dy'
      do j=2,nwgts
         write (unit,'(a)', advance='no') ' & '//trim(adjustl(wgts_info(j)))
      enddo
      do k=1,ndens
         write (unit,'(a)', advance='no') ' & '//
     $        trim(adjustl(wgts_info(idens(k))))//' @dy'
         write (unit,'(a)', advance='no') ' & '//
     $        trim(adjustl(wgts_info(idens(k))))//' @dyc'
      enddo
      write (unit,'(a)') ''
      write (unit,'(a)') ''
      do label=1,max_plots
         if (.not. booked(label)) cycle
c     title
c        For some weird reason, it is no possible to include directly
c        the integer in two line below with the format '(12a,i4.4,3a,a,2a)'
c        this is why I have to do it in two steps instead.
         write (str_nbin,'(i4.4)') nbin(label)
         write (unit, '(12a,4a,3a,a,2a)') '<histogram> ',str_nbin,' " ',trim(title(label)), ' "'
c         write (*, '(12a,4a,3a,a,2a)') '<histogram> ',str_nbin,' " ',trim(title(label)), ' "'
c     data
         do i=1,nbin(label)
           write (unit,'(2x,e14.7)', advance='no') histxl(label,i)
           write (unit,'(2x,e14.7)', advance='no') histxm(label,i)
           write (unit,'(2x,e14.7)', advance='no') histy_acc(1,label,i)*xnorm
           write (unit,'(2x,e14.7)', advance='no') histy_err(label,i)*xnorm
           do j=2,nwgts
              write (unit, '(2x,e14.7)', advance='no') histy_acc(j,label,i)*xnorm
           enddo
           do k=1,2*ndens
              write (unit, '(2x,e14.7)', advance='no') histd_err(k,label,i)*abs(xnorm)
           enddo
	   write(unit, *) ''
         enddo
c     2 empty lines after each plot
         write (unit,'(12a)') '<\histogram>'
         write (unit,'(a)') ''
         write (unit,'(a)') ''         
      enddo
      deallocate(buffer)
      return
      end

c Clean all the allocatable variables:
      subroutine HwU_deallocate_all
      use HwU_variables
      implicit none
      if (allocated(wgts_info)) deallocate(wgts_info)
      if (allocated(booked)) deallocate(booked)
      if (allocated(title)) deallocate(title)
      if (allocated(nbin)) deallocate(nbin)
      if (allocated(step)) deallocate(step)
      if (allocated(histxl)) deallocate(histxl)
      if (allocated(histxm)) deallocate(histxm)
      if (allocated(histy)) deallocate(histy)
      if (allocated(histy_acc)) deallocate(histy_acc)
      if (allocated(histi)) deallocate(histi)
      if (allocated(histy2)) deallocate(histy2)
      if (allocated(histy_err)) deallocate(histy_err)
      if (allocated(p_bin)) deallocate(p_bin)
      if (allocated(p_label)) deallocate(p_label)
      if (allocated(p_wgts)) deallocate(p_wgts)
      if (allocated(idens)) deallocate(idens)
      if (allocated(histd2)) deallocate(histd2)
      if (allocated(histd_err)) deallocate(histd_err)
      return
      end


      subroutine HwU_allocate_p
      use HwU_variables
      implicit none
      integer,allocatable :: itemp1(:)
      double precision, allocatable :: temp2(:,:)
      if (.not. allocated(p_bin)) then
         allocate(p_bin(max_plots))
         allocate(p_label(max_plots))
         allocate(p_wgts(nwgts,max_plots))
         max_points=max_plots
      else
         if (np.gt.max_points) then
c p_bin
            allocate(itemp1(np+max_plots))
            itemp1(1:max_points)=p_bin
            call move_alloc(itemp1,p_bin)
      
c p_label
            allocate(itemp1(np+max_plots))
            itemp1(1:max_points)=p_label
            call move_alloc(itemp1,p_label)
c p_wgts
            allocate(temp2(nwgts,np+max_plots))
            temp2(1:nwgts,1:max_points)=p_wgts
            call move_alloc(temp2,p_wgts)
            max_points=np+max_plots
         endif
      endif
      return
      end

      subroutine HwU_allocate_histo(label,nbin_l)
      use HwU_variables
      implicit none
      logical,allocatable :: ltemp(:)
      integer,allocatable :: itemp1(:),itemp2(:,:)
      character(len=50),allocatable :: ctemp(:)
      double precision, allocatable :: temp1(:),temp2(:,:),temp3(:,:,:)
      integer label,i,nbin_l,label_max,nbin_max
      logical debug
      parameter (debug=.true.)
c Check if variables are already allocated. If not, simply allocate a
c single histogram
      if (.not. allocated(booked)) then
         allocate(booked(1))
         booked(1)=.false.
         allocate(title(1))
         allocate(nbin(1))
         allocate(step(1))
         allocate(histxl(1,nbin_l))
         allocate(histxm(1,nbin_l))
         allocate(histy(nwgts,1,nbin_l))
         allocate(histy_acc(nwgts,1,nbin_l))
         allocate(histi(1,nbin_l))
         allocate(histy2(1,nbin_l))
         allocate(histy_err(1,nbin_l))
         allocate(histd2(2*ndens,1,nbin_l))
         allocate(histd_err(2*ndens,1,nbin_l))
         max_plots=1
         max_bins=nbin_l
      endif
c If current label is greater than the plots already allocated, increase
c the size of the allocated arrays. This is kind of slow, but shouldn't
c really matter since it's only done at the start of a run.
      if (label.gt.max_plots .or. nbin_l.gt.max_bins) then
         label_max=max(label,max_plots)
         nbin_max=max(nbin_l,max_bins)
c booked
         allocate(ltemp(label_max))
         ltemp(1:max_plots)=booked
         call move_alloc(ltemp,booked)
         do i=max_plots+1,label_max
            booked(i)=.false. ! histos have not yet been setup
         enddo
c title
         allocate(ctemp(label_max))
         ctemp(1:max_plots)=title
         call move_alloc(ctemp,title)
c nbin
         allocate(itemp1(label_max))
         itemp1(1:max_plots)=nbin
         call move_alloc(itemp1,nbin)
c step
         allocate(temp1(label_max))
         temp1(1:max_plots)=step
         call move_alloc(temp1,step)
c histxl
         allocate(temp2(label_max,nbin_max))
         temp2(1:max_plots,1:max_bins)=histxl
         call move_alloc(temp2,histxl)
c histxm
         allocate(temp2(label_max,nbin_max))
         temp2(1:max_plots,1:max_bins)=histxm
         call move_alloc(temp2,histxm)
c histy
         allocate(temp3(nwgts,label_max,nbin_max))
         temp3(1:nwgts,1:max_plots,1:max_bins)=histy
         call move_alloc(temp3,histy)
c histy_acc
         allocate(temp3(nwgts,label_max,nbin_max))
         temp3(1:nwgts,1:max_plots,1:max_bins)=histy_acc
         call move_alloc(temp3,histy_acc)
c histi
         allocate(itemp2(label_max,nbin_max))
         itemp2(1:max_plots,1:max_bins)=histi
         call move_alloc(itemp2,histi)
c histy2
         allocate(temp2(label_max,nbin_max))
         temp2(1:max_plots,1:max_bins)=histy2
         call move_alloc(temp2,histy2)
c histy_err
         allocate(temp2(label_max,nbin_max))
         temp2(1:max_plots,1:max_bins)=histy_err
         call move_alloc(temp2,histy_err)
c histd2 and histd_err
         allocate(temp3(2*ndens,label_max,nbin_max))
         temp3(1:2*ndens,1:max_plots,1:max_bins)=histd2
         call move_alloc(temp3,histd2)
         allocate(temp3(2*ndens,label_max,nbin_max))
         temp3(1:2*ndens,1:max_plots,1:max_bins)=histd_err
         call move_alloc(temp3,histd_err)
c Update maximums
         max_plots=label_max
         max_bins=nbin_max
      elseif (booked(label)) then
         write (*,*) 'ERROR in HwU.f: histogram already booked',label
         stop
      endif
      return
      end
      
      
c dummy subroutine
      subroutine accum(idummy)
      integer idummy
      end
c dummy subroutine
      subroutine addfil(string)
      character*(*) string
      end



