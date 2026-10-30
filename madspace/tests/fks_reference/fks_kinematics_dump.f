      program fks_kinematics_dump
c Reference values of the FKS phase space of genps_fks.f for the
c madspace tests. Built in the P* directory of an
c 'output standalone_fortran --fks --limits' process (see
c make_fks_reference.py) and linked like test_soft_col_limits.
c
c Reads from stdin: the number of points per FKS configuration and the
c Born configuration (iconfig). For every FKS configuration and point
c it calls generate_momenta at uniform random x, which runs the full
c production chain (Born map, FKS radiation, counter-events). It then
c calls generate_FKS_kinematics again at the same x with xjac0 =
c xpswgt0 = 1, so that the second set of Jacobians holds the radiation
c part alone, times compute_flux. Everything is printed with keywords
c that make_fks_reference.py parses.
      use mint_module
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'born_conf.inc'
      integer i,j,k,ipt,npts,iconf_in,fks_loop
      double precision x(99),p(0:3,nexternal),wgt,jac_full
      double precision jac_full_cnt(-2:2)
      double precision stot,shat_born,sqrtshat_born,tau_born
      double precision ycm_born,ycmhat,xbjrk_born(2),jac_rad
      double precision m(-max_branch:max_particles),m_born(nexternal-1)
      double precision pb(0:3),xjac0,xpswgt0
      logical pass
      integer nndim
      common/tosigint/nndim
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision emass(nexternal)
      common/to_mass/emass
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     &     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision p_ev(0:3,nexternal)
      common/pev/p_ev
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev
     &     ,p_i_fks_cnt
      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt
      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev
      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt
      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      integer isolsign
      common /c_isolsign/isolsign
      double precision xij_kperp(0:3)
      common/cxij_kperp/xij_kperp
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision ran2
      external ran2
      character*24 fmt
      parameter (fmt='(a,1x,*(es25.17e3,1x))')

      read (*,*) npts
      read (*,*) iconf_in

      call setrun
      call setpara('param_card.dat')
      call fill_configurations_common
      softtest=.false.
      colltest=.false.

      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)).ge.1) ndim=ndim+1
      if (abs(lpp(2)).ge.1) ndim=ndim+1
      nndim=ndim

      write (*,'(a,1x,*(i0,1x))') 'HEADER',nexternal,nincoming
     &     ,fks_configs,ndim,lpp(1),lpp(2),iconf_in
      write (*,fmt) 'EBEAM',ebeam(1),ebeam(2)

      do fks_loop=1,fks_configs
         nFKSprocess=fks_loop
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         iconfig=iconf_in
         ichan=1
         iconfigs(1)=iconfig
         call setfksfactor(.false.)
         call setcuts
         write (*,'(a,1x,*(i0,1x))') 'CONF',nFKSprocess,i_fks,j_fks
     &        ,(pdg_type(i),i=1,nexternal)
         write (*,fmt) 'MASS',(emass(i),i=1,nexternal)
         do ipt=1,npts
            do j=1,ndim
               x(j)=ran2()
            enddo
            new_point=.true.
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            jac_full=wgt
            do k=-2,2
               jac_full_cnt(k)=jac_cnt(k)
            enddo
            write (*,'(a,1x,i0,1x,i0,1x,i0)') 'POINT',nFKSprocess,ipt
     &           ,isolsign
            write (*,fmt) 'X',(x(j),j=1,ndim)
            write (*,fmt) 'JAC',jac_full,(jac_full_cnt(k),k=-2,2)
            write (*,fmt) 'PBORN',((p_born_l(j,i),j=0,3),i=1
     &           ,nexternal-1)
            write (*,fmt) 'PEV',((p_ev(j,i),j=0,3),i=1,nexternal)
            do k=-2,2
               write (*,'(a,1x,i0,1x,*(es25.17e3,1x))') 'PCNT',k
     &              ,((p1_cnt(j,i,k),j=0,3),i=1,nexternal)
            enddo
            write (*,fmt) 'XI',xi_i_fks_ev,xi_i_hat_ev,y_ij_fks_ev
     &           ,xiimax_ev,(xi_i_fks_cnt(k),k=-2,2)
     &           ,(xi_i_hat_cnt(k),k=-2,2),(xiimax_cnt(k),k=-2,2)
            write (*,fmt) 'XBJ',xbjrk_ev(1),xbjrk_ev(2)
     &           ,((xbjrk_cnt(j,k),j=1,2),k=-2,2)
            write (*,fmt) 'SHAT',shat_ev,(shat_cnt(k),k=-2,2),tau_ev
     &           ,ycm_ev,(tau_cnt(k),k=-2,2),(ycm_cnt(k),k=-2,2)
            write (*,fmt) 'KPERP',(xij_kperp(j),j=0,3)
c Radiation part alone: rerun generate_FKS_kinematics on the Born of the
c point with unit Born weights.
            if (jac_full.gt.0d0 .and. p_born_l(0,1).gt.0d0) then
               do j=0,3
                  pb(j)=p_born_l(j,1)+p_born_l(j,2)
               enddo
               shat_born=pb(0)**2-pb(1)**2-pb(2)**2-pb(3)**2
               sqrtshat_born=sqrt(shat_born)
               stot=4d0*ebeam(1)*ebeam(2)
c A final-state emitter leaves ycm unchanged. For an initial-state
c emitter fill_FKS_commons stores ycm_born in ycm_cnt(0), also when the
c soft counter-event is not generated, while xbjrk_cnt(:,0) is then
c stale. (ycm_cnt(0) is stale for the second fold of a massive j_fks.)
               tau_born=shat_born/stot
               if (lpp(1).eq.0 .and. lpp(2).eq.0) then
                  ycm_born=0d0
               elseif (j_fks.gt.nincoming) then
                  ycm_born=ycm_ev
               else
                  ycm_born=ycm_cnt(0)
               endif
               xbjrk_born(1)=sqrt(tau_born)*exp(ycm_born)
               xbjrk_born(2)=sqrt(tau_born)*exp(-ycm_born)
               if (tau_born.lt.1d0) then
                  ycmhat=ycm_born/(-0.5d0*log(tau_born))
               else
                  ycmhat=0d0
               endif
               do i=1,nexternal-1
                  if (i.lt.i_fks) then
                     m_born(i)=emass(i)
                  else
                     m_born(i)=emass(i+1)
                  endif
               enddo
               xjac0=1d0
               xpswgt0=1d0
               call generate_FKS_kinematics(x,ndim,xjac0,xpswgt0,stot
     &              ,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat
     &              ,xbjrk_born,.false.,m,m_born,jac_rad,p,pass)
               write (*,fmt) 'BORNX',shat_born,tau_born,ycm_born
     &              ,xbjrk_born(1),xbjrk_born(2)
               write (*,fmt) 'RADJAC',jac_rad,(jac_cnt(k),k=-2,2)
               write (*,fmt) 'RADPEV',((p(j,i),j=0,3),i=1,nexternal)
            endif
         enddo
      enddo
      end
