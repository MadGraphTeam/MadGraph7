c Writes the fixed-order integrand of MadFKS at the points of a run, for
c make_fks_integrand_reference.py. make_fks_integrand_reference.py appends
c this file to driver_mintFO.f of the output and calls it from sigint,
c right after include_PDF_and_alphas, so it sees the weight_lines module
c as MadFKS sums it: per contribution the type (add_wgt), the FKS
c configuration, the power of g_strong, the coefficients wgt(1:3) of
c 1, log(mu_R^2/Q^2) and log(mu_F^2/Q^2), the momentum fractions, the
c scales, g_strong, the boost to the lab, the momenta and PDG codes, the
c luminosity per subprocess and the final central weight wgts(1), and
c the phase-space point of the last FKS configuration (the Born with its
c Jacobians, the event and the counter-events with theirs). The
c environment variable MADFKS_INTEGRAND_DUMP sets the number of points
c it writes (none without it), to fks_integrand_dump.txt in the channel
c directory.
      subroutine fks_integrand_dump(xx,x,vegas_wgt,nFKS_picked
     $     ,iran_picked,vol,nFKS_born,sum,real_type,phase_space)
      use weight_lines
      use mint_module, only: ichan,iconfig,ndimmax
      use fks_phase_space, only: fks_phase_space_point
      implicit none
      include 'nexternal.inc'
      type(fks_phase_space_point) phase_space
      double precision xx(ndimmax),x(99),vegas_wgt,vol
      integer nFKS_picked,iran_picked,nFKS_born,real_type
      logical sum
      integer i,j,k,ndump,ncalls,nwritten,lun
      integer         nndim
      common/tosigint/nndim
      character*20 env
      logical firsttime
      data firsttime/.true./
      data ncalls,nwritten/0,0/
      save firsttime,ndump,ncalls,nwritten,lun
      if (firsttime) then
         firsttime=.false.
         ndump=0
         call getenv('MADFKS_INTEGRAND_DUMP',env)
         if (env.ne.' ') read(env,*) ndump
         lun=97
         if (ndump.gt.0) open(unit=lun,file='fks_integrand_dump.txt'
     $        ,status='unknown')
      endif
      ncalls=ncalls+1
      if (nwritten.ge.ndump) return
      nwritten=nwritten+1
      write(lun,'(a,9(1x,i8))') 'POINT',ncalls,ichan,iconfig
     $     ,nFKS_picked,iran_picked,nFKS_born,real_type,nndim
     $     ,merge(1,0,sum)
      write(lun,'(a,2(1x,es24.16e3))') 'WEIGHTS',vegas_wgt,vol
      write(lun,'(a,99(1x,es24.16e3))') 'XX',(xx(i),i=1,nndim)
      write(lun,'(a,99(1x,es24.16e3))') 'X',(x(i),i=1,nndim)
      write(lun,'(a,1x,i2,9(1x,es24.16e3))') 'BORN'
     $     ,merge(1,0,phase_space%born%valid),phase_space%born%xjac
     $     ,phase_space%born%xpswgt,phase_space%born%shat
     $     ,phase_space%born%tau,phase_space%born%ycm
     $     ,phase_space%born%xbjrk(1),phase_space%born%xbjrk(2)
     $     ,phase_space%born%stot,phase_space%weight
      write(lun,'(a,99(1x,es24.16e3))') 'BORNP',((phase_space%born%p(j
     $     ,k),j=0,3),k=1,nexternal-1)
      write(lun,'(a,1x,i2,7(1x,es24.16e3))') 'EVENT'
     $     ,merge(1,0,phase_space%event%valid)
     $     ,phase_space%event%jacobian,phase_space%event%xi
     $     ,phase_space%event%y,phase_space%event%xi_hat
     $     ,phase_space%event%xi_max,phase_space%event%xbjrk(1)
     $     ,phase_space%event%xbjrk(2)
      do k=0,2
         write(lun,'(a,1x,i2,1x,i2,7(1x,es24.16e3))') 'CNT',k
     $        ,merge(1,0,phase_space%counterevent(k)%valid)
     $        ,phase_space%counterevent(k)%jacobian
     $        ,phase_space%counterevent(k)%xi
     $        ,phase_space%counterevent(k)%y
     $        ,phase_space%counterevent(k)%xi_hat
     $        ,phase_space%counterevent(k)%xi_max
     $        ,phase_space%counterevent(k)%xbjrk(1)
     $        ,phase_space%counterevent(k)%xbjrk(2)
      enddo
      write(lun,'(a,3(1x,es24.16e3))') 'RAD'
     $     ,(phase_space%radiation_coordinates(k),k=1,3)
      write(lun,'(a,1x,i8)') 'NCONTR',icontr
      do i=1,icontr
         write(lun,'(a,9(1x,i12))') 'CINT',i,itype(i),nFKS(i)
     $        ,QCDpower(i),plot_id(i),merge(1,0,H_event(i)),niproc(i)
     $        ,orderstag(i),amppos(i)
         write(lun,'(a,14(1x,es24.16e3))') 'CWGT',(wgt(k,i),k=1,3)
     $        ,(bjx(k,i),k=1,2),(scales2(k,i),k=1,3),g_strong(i)
     $        ,y_bst(i),cpower(i),wgts(1,i),wgt_ME_tree(1,i)
     $        ,wgt_ME_tree(2,i)
         write(lun,'(a,99(1x,es24.16e3))') 'CMOM',((momenta(j,k,i)
     $        ,j=0,3),k=1,nexternal)
         write(lun,'(a,99(1x,i8))') 'CPDG',(pdg(k,i),k=1,nexternal)
         write(lun,'(a,99(1x,es24.16e3))') 'CLUM',(parton_iproc(k,i)
     $        ,k=1,niproc(i))
      enddo
      write(lun,'(a)') 'END'
      call flush(lun)
      return
      end
