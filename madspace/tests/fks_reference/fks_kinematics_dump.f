      program fks_kinematics_dump
c Reference values of the FKS phase space of genps_fks.f for the
c madspace tests. Built in the P* directory of an
c 'output standalone_fortran --fks --limits' process (see
c make_fks_reference.py) and linked like test_soft_col_limits.
c
c Reads from stdin: the number of points per FKS configuration, the
c Born configuration (iconfig) and the number of extra points per FKS
c configuration with a massive final-state emitter. For every FKS
c configuration and point it calls generate_momenta at uniform random x,
c which runs the full production chain (Born map, FKS radiation,
c counter-events). The extra points follow in a second pass, so that the
c first one does not depend on their number. They have y_ij < 0 and xi
c close to its upper end, where the second solution for the momentum of
c the emitter (isolsign = -1) is used for part of the range. It then
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
      include 'coupl.inc'
      include 'fks_powers.inc'
      integer i,j,k,ipt,npts,iconf_in,fks_loop,ipass,nmassive,nloop
      double precision x(99),p(0:3,nexternal),wgt,jac_full
      double precision jac_full_cnt(-2:2)
      double precision stot,shat_born,sqrtshat_born,tau_born
      double precision ycm_born,ycmhat,xbjrk_born(2),jac_rad
      double precision m(-max_branch:max_particles),m_born(nexternal-1)
      double precision pb(0:3),xjac0,xpswgt0,tau_lower_bound_save
      integer irerun
      character*3 pre
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
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
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge
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
      double precision ran2,fks_Sij,s_ev,s_cnt(0:2)
      external ran2,fks_Sij
      integer neik,mm,nn,eik_m(nexternal*nexternal)
     &     ,eik_n(nexternal*nexternal)
      double precision eik(nexternal*nexternal)
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      double precision e_i,e_j,zz,tt,ap(2),qq(2),coll(6,2)
      double precision f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision fkssymmetryfactor,fkssymmetryfactorBorn
     &     ,fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn
     &     ,fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat
     &     ,shat
      character*24 fmt
      parameter (fmt='(a,1x,*(es25.17e3,1x))')

      read (*,*) npts
      read (*,*) iconf_in
      read (*,*) nmassive

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

      do ipass=1,2
      do fks_loop=1,fks_configs
         nFKSprocess=fks_loop
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         iconfig=iconf_in
         ichan=1
         iconfigs(1)=iconfig
         call setfksfactor(.false.)
         call setcuts
         if (ipass.eq.1) then
            nloop=npts
         elseif (j_fks.gt.nincoming .and. emass(j_fks).gt.0d0) then
            nloop=nmassive
         else
            cycle
         endif
         write (*,'(a,1x,*(i0,1x))') 'CONF',nFKSprocess,i_fks,j_fks
     &        ,(pdg_type(i),i=1,nexternal)
         write (*,fmt) 'MASS',(emass(i),i=1,nexternal)
c the FKS information of the configuration (fks_inc_chooser): colour
c types, charges and FKS pairs of the real process
         write (*,'(a,1x,*(i0,1x))') 'PTYPE',(particle_type(i),i=1
     &        ,nexternal)
         write (*,fmt) 'PCHARGE',(particle_charge(i),i=1,nexternal)
         do i=1,nexternal
            write (*,'(a,1x,*(i0,1x))') 'JFROMI',i,(fks_j_from_i(i,j),j=1
     &           ,fks_j_from_i(i,0))
         enddo
         do ipt=1,nloop
            do j=1,ndim
               x(j)=ran2()
            enddo
            if (ipass.eq.2) then
               x(ndim-2)=1d0-0.12d0*x(ndim-2)
               x(ndim-1)=0.71d0+0.29d0*x(ndim-1)
            endif
            new_point=.true.
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            jac_full=wgt
            do k=-2,2
               jac_full_cnt(k)=jac_cnt(k)
            enddo
            write (*,'(a,1x,i0,1x,i0,1x,i0)') 'POINT',nFKSprocess
     &           ,ipt+(ipass-1)*npts,isolsign
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
            write (*,fmt) 'PIHAT',(p_i_fks_ev(j),j=0,3)
     &           ,((p_i_fks_cnt(j,k),j=0,3),k=-2,2)
c The sector function S_ij of the configuration (fks_Sij) at the event and
c at its counter-events, with the arguments of compute_soft_counter_term
c and its siblings, -1 where genps_fks.f did not generate the momenta (and
c for the collinear ones of a massive emitter)
            s_ev=-1d0
            do k=0,2
               s_cnt(k)=-1d0
            enddo
            if (jac_full.gt.0d0) then
               call set_cms_stuff(-100)
               s_ev=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
               if (jac_full_cnt(0).gt.0d0) then
                  call set_cms_stuff(0)
                  s_cnt(0)=fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,0d0
     &                 ,y_ij_fks_ev)
               endif
               if (jac_full_cnt(1).gt.0d0.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(1)
                  s_cnt(1)=fks_Sij(p1_cnt(0,1,1),i_fks,j_fks
     &                 ,xi_i_fks_cnt(1),1d0)
               endif
               if (jac_full_cnt(2).gt.0d0.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(2)
                  s_cnt(2)=fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,0d0,1d0)
               endif
            endif
            write (*,fmt) 'SIJ',s_ev,(s_cnt(k),k=0,2)
c The reduced eikonal factors at the soft counter-event, for the pairs
c (m, n) of sbornsoft in its order, as sbornsoft calls eikonal_reduced
c (none where the soft counter-event is not generated)
            neik=0
            if (jac_full.gt.0d0.and.jac_full_cnt(0).gt.0d0) then
               call set_cms_stuff(0)
               do k=1,fks_j_from_i(i_fks,0)
                  do j=1,k
                     mm=fks_j_from_i(i_fks,k)
                     nn=fks_j_from_i(i_fks,j)
                     if ((mm.ne.nn .or. emass(mm).ne.0d0) .and.
     &                    nn.ne.i_fks .and. mm.ne.i_fks) then
                        neik=neik+1
                        eik_m(neik)=mm
                        eik_n(neik)=nn
                        call eikonal_reduced(p1_cnt(0,1,0),mm,nn,i_fks
     &                       ,j_fks,0d0,y_ij_fks_ev,eik(neik))
                     endif
                  enddo
               enddo
            endif
            write (*,'(a,1x,*(i0,1x))') 'EIKPAIRS',neik,(eik_m(k)
     &           ,eik_n(k),k=1,neik)
            write (*,fmt) 'EIK',(eik(k),k=1,neik)
c The collinear factors of sborncol_fsr and sborncol_isr at the collinear
c (k = 1) and soft-collinear (k = 2) counter-events: z, t and AP_reduced and
c Qterms_reduced_timelike/spacelike without the couplings (QCD over g^2,
c QED over gal(1)^2); z = -1 where the counter-event is not generated or j
c is massive
            do k=1,2
               coll(1,k)=-1d0
               do j=2,6
                  coll(j,k)=0d0
               enddo
               if (jac_full.gt.0d0.and.jac_full_cnt(k).gt.0d0.and.
     &              emass(j_fks).eq.0d0) then
                  call set_cms_stuff(k)
                  if (j_fks.gt.nincoming) then
                     e_i=p1_cnt(0,i_fks,k)
                     e_j=p1_cnt(0,j_fks,k)
                     zz=1d0-e_i/(e_i+e_j)
                     tt=zz*shat/4d0
                     call AP_reduced(j_type,i_type,ch_j,ch_i,tt,zz,ap)
                     call Qterms_reduced_timelike(j_type,i_type,ch_j
     &                    ,ch_i,tt,zz,qq)
                  else
                     if (k.eq.1) then
                        zz=1d0-xi_i_fks_cnt(1)
                     else
                        zz=1d0
                     endif
                     tt=zz*shat/4d0
                     call AP_reduced(m_type,i_type,ch_m,ch_i,tt,zz,ap)
                     call Qterms_reduced_spacelike(m_type,i_type,ch_m
     &                    ,ch_i,tt,zz,qq)
                  endif
                  coll(1,k)=zz
                  coll(2,k)=tt
                  coll(3,k)=ap(1)/g**2
                  coll(4,k)=ap(2)/dble(gal(1))**2
                  coll(5,k)=qq(1)/g**2
                  coll(6,k)=qq(2)/dble(gal(1))**2
               endif
            enddo
            write (*,fmt) 'COLL',((coll(j,k),j=1,6),k=1,2)
c The prefactors of the real emission and its counterterms of
c compute_prefactors_n1body, with unit vegas weight and the Jacobian of the
c event, its symmetry factors and the parameters of the subtraction
c (nocntevents as 1 or 0), where the event is generated
            if (jac_full.gt.0d0) then
               call set_cms_stuff(-100)
               call compute_prefactors_n1body(1d0,jac_full)
               if (nocntevents) then
                  s_ev=1d0
               else
                  s_ev=0d0
               endif
               write (*,fmt) 'PREF',f_r,f_s,f_c,f_dc,f_sc,(f_dsc(k),k=1
     &              ,4),fkssymmetryfactor,fkssymmetryfactorDeg,xinorm_ev
     &              ,(xinorm_cnt(k),k=-2,2),xicut_used,xiScut_used
     &              ,delta_used,deltaS,deltaO,deltaI,s_ev
            endif
c Radiation part alone: rerun generate_FKS_kinematics on the Born of the
c point with unit Born weights, once with the lower bound on the partonic
c energy of the real emission that set_tau_min derives from the cuts
c (tau_lower_bound, which restricts xi and y for an initial-state emitter;
c the CUT entries) and once without it (the FKS map itself; the RAD
c entries).
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
               write (*,fmt) 'BORNX',shat_born,tau_born,ycm_born
     &              ,xbjrk_born(1),xbjrk_born(2)
               write (*,fmt) 'TAULB',tau_lower_bound
               tau_lower_bound_save=tau_lower_bound
               do irerun=1,2
                  if (irerun.eq.1) then
                     pre='CUT'
                     tau_lower_bound=tau_lower_bound_save
                  else
                     pre='RAD'
                     tau_lower_bound=0d0
                  endif
                  xjac0=1d0
                  xpswgt0=1d0
                  call generate_FKS_kinematics(x,ndim,xjac0,xpswgt0,stot
     &                 ,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat
     &                 ,xbjrk_born,.false.,m,m_born,jac_rad,p,pass)
                  write (*,fmt) pre//'JAC',jac_rad,(jac_cnt(k),k=-2,2)
                  write (*,fmt) pre//'PEV',((p(j,i),j=0,3),i=1
     &                 ,nexternal)
                  do k=-2,2
                     write (*,'(a,1x,i0,1x,*(es25.17e3,1x))') pre/
     &                    /'PCNT',k,((p1_cnt(j,i,k),j=0,3),i=1
     &                    ,nexternal)
                  enddo
                  write (*,fmt) pre//'XI',xi_i_fks_ev,xi_i_hat_ev
     &                 ,y_ij_fks_ev,xiimax_ev,(xi_i_fks_cnt(k),k=-2,2)
     &                 ,(xi_i_hat_cnt(k),k=-2,2),(xiimax_cnt(k),k=-2,2)
                  write (*,fmt) pre//'XBJ',xbjrk_ev(1),xbjrk_ev(2)
     &                 ,((xbjrk_cnt(j,k),j=1,2),k=-2,2)
                  write (*,fmt) pre//'SHAT',shat_ev,(shat_cnt(k),k=-2,2)
                  write (*,fmt) pre//'PIHAT',(p_i_fks_ev(j),j=0,3)
     &                 ,((p_i_fks_cnt(j,k),j=0,3),k=-2,2)
               enddo
               tau_lower_bound=tau_lower_bound_save
            endif
         enddo
      enddo
      enddo
      end
