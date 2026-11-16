c fks_provider_ewsudakov.f for an output without --ewsudakov: no Sudakov
c approximation of the EW corrections.
      subroutine provider_sdk_sizes(sizes)
      implicit none
      integer sizes(5)
      sizes=0
      end


      subroutine provider_sdk_info(ncomb,nint,bases,interference_bases,
     $     parameters)
      implicit none
      integer ncomb,nint,bases(*),interference_bases(*)
      double precision parameters(9)
      parameters=0d0
      end


      subroutine provider_sdk_evaluate(p,ncomb,nint,out)
      implicit none
      integer ncomb,nint
      double precision p(*),out(*)
      write (*,*) 'fks_provider: the output has no --ewsudakov'
      stop 1
      end


      subroutine provider_sdk_weights(p,scale_in,out)
      implicit none
      double precision p(*),scale_in,out(*)
      write (*,*) 'fks_provider: the output has no --ewsudakov'
      stop 1
      end
