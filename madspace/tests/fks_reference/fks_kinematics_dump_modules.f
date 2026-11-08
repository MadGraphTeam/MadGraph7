      program fks_kinematics_dump
c Reference values of the FKS phase space of genps_fks.f for the
c madspace tests. Built in the P* directory of an
c 'output standalone_fortran --fks --limits' process (see
c make_fks_reference.py) and linked like test_soft_col_limits.
c
c Reads from stdin: the number of points per FKS configuration, the
c Born configuration (iconfig), the number of extra points per FKS
c configuration with a massive final-state emitter, the number of
c points per FKS configuration close to the limits and the lpp of both
c beams (0 keeps those of the run_card, 3 for lepton PDFs, whose
c run_card the Source of an NLO output does not take). With lepton PDFs
c an incoming emitter radiates without the projection of the event
c (generate_noevpr_kinematics), and the dump also prints the luminosity
c and the Born of each counter-event (see NOEVPR and BORNREC below). For every FKS
c configuration and point it calls generate_momenta at uniform random x,
c which runs the full production chain (Born map, FKS radiation,
c counter-events). The extra points follow in a second pass, so that the
c first one does not depend on their number. They have y_ij < 0 and xi
c close to its upper end, where the second solution for the momentum of
c the emitter (isolsign = -1) is used for part of the range. The points
c close to the limits follow in a third pass, with xi_hat within 1e-8 of
c its lower end stiny and every second one also with y close to the upper
c end of its range, where sreal and fks_Sij switch to the soft and
c collinear limits (below tiny = 1e-6 in xi or 1 - y). At every
c point it prints the pieces of the subtraction of fks_singular.f: the
c sector functions, the factors of the soft and collinear limits, the
c prefactors, sreal at the event and the counter-events, the Born
c with its colour (or charge) links, the collinear remnants of sreal_deg
c and the n-body part of bornsoftvirtual without the virtual, also per
c position of the amp_split arrays (the split orders). It then
c calls generate_FKS_kinematics again at the same x with xjac0 =
c xpswgt0 = 1, so that the second set of Jacobians holds the radiation
c part alone, times compute_flux. Everything is printed with keywords
c that make_fks_reference.py parses.
c
c This copy is ported to the module API of Rikkert's MCcntRefactor_Sfun
c branch (fks_phase_space, fks_phase_space_data). It reads a sixth input,
c FKSISRMapping (1 symmetric as MadFKS main, 2 asymmetric), sets
c VirtTrickSurrogate = 0 and runs at fixed order.
      use mint_module
      use FKSParams, only: FKSParamReader,paramFileName,FKSISRMapping
     &     ,VirtTrickSurrogate
      use fks_phase_space, only: fks_born_point,fks_phase_space_point
     &     ,initialize_fks_phase_space,generate_prepared_fks_point
     &     ,generate_fks_radiation
      use fks_phase_space_data, only: tau_Born_lower_bound
     &     ,tau_lower_bound,p1_cnt,jac_cnt,p_born,p_born_l,p_ev
     &     ,xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt,xi_i_fks_cnt
     &     ,xi_i_hat_ev,xi_i_hat_cnt,xbjrk_ev,xbjrk_cnt,sqrtshat_ev
     &     ,shat_ev,sqrtshat_cnt,shat_cnt,tau_ev,ycm_ev,tau_cnt,ycm_cnt
     &     ,xiimax_ev,xiimax_cnt,isolsign,xij_kperp,xinorm_ev,xinorm_cnt
     &     ,nocntevents,ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
     &     ,p_born_coll
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
      integer nlimit
      double precision x(99),p(0:3,nexternal),wgt,jac_full
      double precision jac_full_cnt(-2:2)
      double precision stot,shat_born,sqrtshat_born,tau_born
      double precision ycm_born,xbjrk_born(2),jac_rad
      integer irerun,isr_in
      character*3 pre
      type(fks_phase_space_point) point,point2
      type(fks_born_point) born,born2
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
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision ran2,fks_Sij,s_ev,s_cnt(0:2)
      external ran2,fks_Sij
      integer neik,mm,nn,eik_m(nexternal*nexternal)
     &     ,eik_n(nexternal*nexternal)
      double precision eik(nexternal*nexternal)
      integer i_type,j_type,m_type,j_pdg
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/ch_i,ch_j,ch_m,i_type,j_type,m_type,j_pdg
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
      include 'orders.inc'
      include 'q_es.inc'
      double precision amp_split_wgtdegrem_xi(amp_split_size),
     &     amp_split_wgtdegrem_lxi(amp_split_size),
     &     amp_split_wgtdegrem_muF(amp_split_size)
      common /to_amp_split_deg/amp_split_wgtdegrem_xi,
     &     amp_split_wgtdegrem_lxi,amp_split_wgtdegrem_muF
      double precision amp_split_wgtpsch_p(amp_split_size),
     &     amp_split_wgtpsch_l(amp_split_size),
     &     amp_split_wgtpsch_d(amp_split_size)
      common /to_amp_split_dis/amp_split_wgtpsch_p,
     &     amp_split_wgtpsch_l,amp_split_wgtpsch_d
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      double precision deg(6,2),deg_xi,deg_lxi
      double precision psch_o(amp_split_size,3,2,7)
      integer isch,ischeme_save
      integer lpp_in
      logical use_evpr
      common /to_use_evpr/use_evpr
      double precision omx_ee(2)
      common /to_ee_omx1/omx_ee
      double precision srec_k(4),wrec_k(4),prec_k(0:3,nexternal-1,4)
      double precision amp_split_wgtnstmp(amp_split_size),
     &     amp_split_wgtwnstmpmuf(amp_split_size),
     &     amp_split_wgtwnstmpmur(amp_split_size)
      common /to_amp_split_bsv/amp_split_wgtnstmp,
     &     amp_split_wgtwnstmpmuf,amp_split_wgtwnstmpmur
      integer fold,ifold_counter
      common /cfl/fold,ifold_counter
      double precision bsv(5),bsv_wgt,virt_wgt
c the same per split order, for the amp_split arrays of fks_singular.f
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft
      double precision sreal_o(amp_split_size,4),born_o(amp_split_size,2)
      double precision link_o(amp_split_size,nexternal*nexternal)
      double precision deg_o(amp_split_size,6,2),bsv_o(amp_split_size,4)
      integer ord_o(nsplitorders),iamp
      include '../../Source/PDF/pdf.inc'
      complex*16 ans_cnt(2,nsplitorders)
      common /c_born_cnt/ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt,isplitorder_born,isplitorder_cnt
      common /c_extra_cnt/iextra_cnt,isplitorder_born,isplitorder_cnt
      double precision extra_o(amp_split_size,2)
      double precision iden_comp,iden_comp_FKS(fks_configs)
      common /c_iden_comp/iden_comp,iden_comp_FKS
      logical need_color_links,need_charge_links
      common /c_need_links/need_color_links,need_charge_links
      double precision fx(4),born_wgt,blink(nexternal*nexternal)
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      character*24 fmt
      parameter (fmt='(a,1x,*(es25.17e3,1x))')

      read (*,*) npts
      read (*,*) iconf_in
      read (*,*) nmassive
      read (*,*) nlimit
      read (*,*) lpp_in
      read (*,*) isr_in

      fixed_order=.true.
      nlo_ps=.false.
      call setrun
      call FKSParamReader(paramFileName,.false.,.false.)
      FKSISRMapping=isr_in
      VirtTrickSurrogate=0
      if (lpp_in.ne.0) then
         lpp(1)=lpp_in
         lpp(2)=lpp_in
      endif
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
c the scheme of the PDFs (0 MSbar, 1 DIS), the number of light flavors
c and the number of massless leptons
      write (*,'(a,1x,i0,2(1x,es25.17e3))') 'SCHEME',pdfscheme,nf,nl
c the split orders: amp_split_size, amp_split_size_born, the positions
c of QCD and QED (-1 if absent), and the powers of each split order of
c every position of the amp_split arrays; the QED coupling e = gal(1)
      write (*,'(a,1x,*(i0,1x))') 'ORDERS',amp_split_size
     &     ,amp_split_size_born,qcd_pos,qed_pos,nsplitorders
      do iamp=1,amp_split_size
         call amp_split_pos_to_orders(iamp,ord_o)
         write (*,'(a,1x,*(i0,1x))') 'AMPSPLIT',iamp,(ord_o(i),i=1
     &        ,nsplitorders)
      enddo
      write (*,fmt) 'COUPL',dble(gal(1))

      do ipass=1,3
      write (*,'(a,1x,i0)') 'PASS',ipass
      do fks_loop=1,fks_integrated
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
         elseif (ipass.eq.3) then
            nloop=nlimit
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
            elseif (ipass.eq.3) then
               x(ndim-2)=1d-4*x(ndim-2)
               if (mod(ipt,2).eq.0) x(ndim-1)=1d-4*x(ndim-1)
            endif
            new_point=.true.
            wgt=1d0
            point=fks_phase_space_point()
            call initialize_fks_phase_space(iconfig)
            call generate_prepared_fks_point(ndim,x,.false.,wgt,point)
            p=point%p
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
c Without the projection of the event (lepton PDFs and an incoming
c emitter): 1, tau_Born_lower_bound, the momentum fractions and 1 - x of
c generate_ee_tau_y, and the Born of the collinear counter-event
c (p_born_coll). Then for the event, the soft, collinear and
c soft-collinear counter-event (k = -100, 0, 1, 2): the energy squared
c of its Born (s_rec = (1 - xi) shat), the weight of generate_momenta_born
c at it with unit Jacobians (Kajantie's normalization) and its momenta in
c their rest frame.
            if (jac_full.gt.0d0.and..not.use_evpr) then
               write (*,fmt) 'NOEVPR',1d0,tau_Born_lower_bound
     &              ,xbjrk_ev(1),xbjrk_ev(2),omx_ee(1),omx_ee(2)
               write (*,fmt) 'PBORNCOLL',((p_born_coll(j,i),j=0,3),i=1
     &              ,nexternal-1)
               srec_k(1)=shat_ev*(1d0-xi_i_fks_ev)
               srec_k(2)=shat_ev
               srec_k(3)=shat_ev*(1d0-xi_i_fks_cnt(1))
               srec_k(4)=shat_ev
               call born_at_srec(x,srec_k,wrec_k,prec_k)
               do k=1,4
                  write (*,'(a,1x,i0,1x,*(es25.17e3,1x))') 'BORNREC',k
     &                 ,srec_k(k),wrec_k(k),((prec_k(j,i,k),j=0,3),i=1
     &                 ,nexternal-1)
               enddo
            endif
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
     &              ,delta_used,deltaS,deltaO,deltaI,s_ev,f_pdfsch_d
     &              ,f_pdfsch_p,f_pdfsch_l
            endif
c The reduced matrix elements of sreal at the event and at the soft,
c collinear and soft-collinear counter-events, with the arguments of
c compute_real_emission and compute_*_counter_term (-1 where the momenta
c are not generated, and for the collinear ones of a massive emitter).
c As in driver_mintFO.f, the Born is computed anew at every point.
            calculatedBorn=.false.
            do k=1,4
               fx(k)=-1d0
               do iamp=1,amp_split_size
                  sreal_o(iamp,k)=0d0
               enddo
            enddo
            if (jac_full.gt.0d0) then
               call set_cms_stuff(-100)
               if (.not.use_evpr) calculatedBorn=.false.
               call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx(1))
               sreal_o(:,1)=amp_split(:)
               if (jac_full_cnt(0).gt.0d0) then
                  call set_cms_stuff(0)
                  if (.not.use_evpr) calculatedBorn=.false.
                  call sreal(p1_cnt(0,1,0),0d0,y_ij_fks_ev,fx(2))
                  sreal_o(:,2)=amp_split(:)
               endif
               if (jac_full_cnt(1).gt.0d0.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(1)
                  if (.not.use_evpr) calculatedBorn=.false.
                  call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),1d0,fx(3))
                  sreal_o(:,3)=amp_split(:)
               endif
               if (jac_full_cnt(2).gt.0d0.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(2)
                  if (.not.use_evpr) calculatedBorn=.false.
                  call sreal(p1_cnt(0,1,2),0d0,1d0,fx(4))
                  sreal_o(:,4)=amp_split(:)
               endif
            endif
            write (*,fmt) 'SREAL',(fx(k),k=1,4)
c the same per position of amp_split (zero where not generated)
            write (*,fmt) 'SREALO',((sreal_o(iamp,k),iamp=1
     &           ,amp_split_size),k=1,4)
c The Born at p_born as sbornsoft and sborncol_fsr call it (the QCD part
c of the counterterms, ans_cnt(1,qcd_pos), and the sum over the split
c orders), the colour-linked Borns of sborn_sf for the pairs of
c sbornsoft in its order (with g^2, where i has colour links), g and
c iden_comp, where the event is generated
            if (jac_full.gt.0d0) then
               if (.not.use_evpr) calculatedBorn=.false.
               call sborn_frame(p_born,born_wgt)
               write (*,fmt) 'BORN',g,iden_comp,born_wgt
     &              ,dble(ans_cnt(1,qcd_pos))
c the Born per position of amp_split for a QCD and for a QED splitting
c (amp_split_cnt, at the NLO position the counterterm contributes to)
               do iamp=1,amp_split_size
                  born_o(iamp,1)=0d0
                  born_o(iamp,2)=0d0
                  if (qcd_pos.gt.0) born_o(iamp,1)=
     &                 dble(amp_split_cnt(iamp,1,qcd_pos))
                  if (qed_pos.gt.0) born_o(iamp,2)=
     &                 dble(amp_split_cnt(iamp,1,qed_pos))
               enddo
               write (*,fmt) 'BORNO',((born_o(iamp,k),iamp=1
     &              ,amp_split_size),k=1,2)
c the same for the Born of the extra counterterm (born_cnt_<n>.f, the
c other mother of a quark pair), zero without one
               do iamp=1,amp_split_size
                  extra_o(iamp,1)=0d0
                  extra_o(iamp,2)=0d0
               enddo
               if (iextra_cnt.gt.0) then
                  if (.not.use_evpr) calculatedBorn=.false.
                  call extra_cnt_frame(p_born,iextra_cnt,ans_extra_cnt)
                  do iamp=1,amp_split_size
                     if (qcd_pos.gt.0) extra_o(iamp,1)=
     &                    dble(amp_split_cnt(iamp,1,qcd_pos))
                     if (qed_pos.gt.0) extra_o(iamp,2)=
     &                    dble(amp_split_cnt(iamp,1,qed_pos))
                  enddo
               endif
               write (*,fmt) 'BORNXO',((extra_o(iamp,k),iamp=1
     &              ,amp_split_size),k=1,2)
               neik=0
               if (need_color_links.or.need_charge_links) then
                  do k=1,fks_j_from_i(i_fks,0)
                     do j=1,k
                        mm=fks_j_from_i(i_fks,k)
                        nn=fks_j_from_i(i_fks,j)
                        if ((mm.ne.nn .or. emass(mm).ne.0d0) .and.
     &                       nn.ne.i_fks .and. mm.ne.i_fks) then
                           neik=neik+1
                           if (.not.use_evpr) calculatedBorn=.false.
                           call sborn_frame(p_born,born_wgt)
                           call sborn_sf_frame(p_born,mm,nn,blink(neik))
                           link_o(:,neik)=amp_split_soft(:)
                        endif
                     enddo
                  enddo
               endif
               write (*,fmt) 'BLINK',(blink(k),k=1,neik)
c the colour- (or charge-) linked Borns per position of amp_split
               write (*,fmt) 'BLINKO',((link_o(iamp,k),iamp=1
     &              ,amp_split_size),k=1,neik)
            endif
c The collinear remnants of sreal_deg at the collinear (k = 1) and
c soft-collinear (k = 2) counter-events of a massless incoming emitter,
c summed over the split orders: wgtdegrem_xi, wgtdegrem_lxi,
c wgtdegrem_muF and the parts p, l and d of the PDF scheme (zero where
c not generated). The factorization and Ellis-Sexton scales are set here
c (sreal_deg reads them), the strong coupling of the Fortran is kept.
            q2fact(1)=200d0**2
            q2fact(2)=200d0**2
            QES2=91.188d0**2
            do k=1,2
               do j=1,6
                  deg(j,k)=0d0
                  do iamp=1,amp_split_size
                     deg_o(iamp,j,k)=0d0
                  enddo
               enddo
               if (jac_full.gt.0d0.and.jac_full_cnt(k).gt.0d0.and.
     &              j_fks.le.nincoming.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(k)
                  if (k.eq.1) then
                     if (.not.use_evpr) calculatedBorn=.false.
                     call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(1)
     &                    ,deg_xi,deg_lxi)
                  else
                     if (.not.use_evpr) calculatedBorn=.false.
                     call sreal_deg(p1_cnt(0,1,2),0d0,deg_xi
     &                    ,deg_lxi)
                  endif
                  do i=1,amp_split_size
                     deg(1,k)=deg(1,k)+amp_split_wgtdegrem_xi(i)
                     deg(2,k)=deg(2,k)+amp_split_wgtdegrem_lxi(i)
                     deg(3,k)=deg(3,k)+amp_split_wgtdegrem_muF(i)
                     deg(4,k)=deg(4,k)+amp_split_wgtpsch_p(i)
                     deg(5,k)=deg(5,k)+amp_split_wgtpsch_l(i)
                     deg(6,k)=deg(6,k)+amp_split_wgtpsch_d(i)
                  enddo
                  deg_o(:,1,k)=amp_split_wgtdegrem_xi(:)
                  deg_o(:,2,k)=amp_split_wgtdegrem_lxi(:)
                  deg_o(:,3,k)=amp_split_wgtdegrem_muF(:)
                  deg_o(:,4,k)=amp_split_wgtpsch_p(:)
                  deg_o(:,5,k)=amp_split_wgtpsch_l(:)
                  deg_o(:,6,k)=amp_split_wgtpsch_d(:)
               endif
            enddo
            write (*,fmt) 'DEG',((deg(j,k),j=1,6),k=1,2),q2fact(1)
     &           ,q2fact(2),QES2
            write (*,fmt) 'DEGO',(((deg_o(iamp,j,k),iamp=1
     &           ,amp_split_size),j=1,6),k=1,2)
c The parts p, l and d of the PDF scheme of sreal_deg per position of
c amp_split for each scheme 1 to 7 of pdfscheme (DIS, the leptonic eta,
c beta, mixed, nobeta and delta, UPC), at the collinear (k = 1) and
c soft-collinear (k = 2) counter-events, at the scales above. The nuclei
c of UPC are set here: xiAI = 1 and RAI = 31 and 35 GeV^-1.
            ischeme_save=pdfscheme
            xiAI(1)=1d0
            xiAI(2)=1d0
            RAI(1)=31d0
            RAI(2)=35d0
            psch_o(:,:,:,:)=0d0
            do k=1,2
               if (jac_full.gt.0d0.and.jac_full_cnt(k).gt.0d0.and.
     &              j_fks.le.nincoming.and.emass(j_fks).eq.0d0) then
                  call set_cms_stuff(k)
                  do isch=1,7
                     pdfscheme=isch
                     if (k.eq.1) then
                        if (.not.use_evpr) calculatedBorn=.false.
                        call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(1)
     &                       ,deg_xi,deg_lxi)
                     else
                        if (.not.use_evpr) calculatedBorn=.false.
                        call sreal_deg(p1_cnt(0,1,2),0d0,deg_xi
     &                       ,deg_lxi)
                     endif
                     psch_o(:,1,k,isch)=amp_split_wgtpsch_p(:)
                     psch_o(:,2,k,isch)=amp_split_wgtpsch_l(:)
                     psch_o(:,3,k,isch)=amp_split_wgtpsch_d(:)
                  enddo
               endif
            enddo
            pdfscheme=ischeme_save
            write (*,fmt) 'PSCHO',((((psch_o(iamp,j,k,isch),iamp=1
     &           ,amp_split_size),j=1,3),k=1,2),isch=1,7)
c The n-body part of bornsoftvirtual at the soft counter-event, where i
c has colour links (as compute_nbody_noborn calls it for those
c configurations), without the virtual (fold = 1 keeps its saved value
c and leaves ran2 alone): bsv_wgt, born_wgt and the sums over the split
c orders of wgtnstmp, wgtwnstmpmur and wgtwnstmpmuf, with the scales
c set here (two factorization scales, the renormalization and the
c Ellis-Sexton scale)
            q2fact(1)=150d0**2
            q2fact(2)=250d0**2
            scale=120d0
            QES2=91.188d0**2
            do j=1,5
               bsv(j)=0d0
            enddo
            if (jac_full.gt.0d0.and.jac_full_cnt(0).gt.0d0.and.
     &           (need_color_links.or.need_charge_links)) then
               call set_cms_stuff(0)
               fold=1
               if (.not.use_evpr) calculatedBorn=.false.
               call bornsoftvirtual(p1_cnt(0,1,0),bsv_wgt,virt_wgt
     &              ,born_wgt)
               fold=0
               bsv(1)=bsv_wgt
               bsv(2)=born_wgt
               do i=1,amp_split_size
                  bsv(3)=bsv(3)+amp_split_wgtnstmp(i)
                  bsv(4)=bsv(4)+amp_split_wgtwnstmpmur(i)
                  bsv(5)=bsv(5)+amp_split_wgtwnstmpmuf(i)
               enddo
               write (*,fmt) 'BSV',(bsv(j),j=1,5),q2fact(1),q2fact(2)
     &              ,scale**2,QES2
c per position of amp_split: amp_split after bornsoftvirtual (the Born at
c its positions plus the n-body part), wgtnstmp, wgtwnstmpmur and
c wgtwnstmpmuf
               bsv_o(:,1)=amp_split(:)
               bsv_o(:,2)=amp_split_wgtnstmp(:)
               bsv_o(:,3)=amp_split_wgtwnstmpmur(:)
               bsv_o(:,4)=amp_split_wgtwnstmpmuf(:)
               write (*,fmt) 'BSVO',((bsv_o(iamp,k),iamp=1
     &              ,amp_split_size),k=1,4)
            endif
c Radiation part alone: rerun generate_FKS_kinematics on the Born of the
c point with unit Born weights, once with the lower bound on the partonic
c energy of the real emission that set_tau_min derives from the cuts
c (tau_lower_bound, which restricts xi and y for an initial-state emitter;
c the CUT entries) and once without it (the FKS map itself; the RAD
c entries).
            if (jac_full.gt.0d0 .and. p_born_l(0,1).gt.0d0 .and.
     &           use_evpr) then
c The Born point of the staged generation, with unit Born measures:
c generate_fks_radiation then returns the radiation part times the flux.
               born=point%born
               shat_born=born%shat
               tau_born=born%tau
               ycm_born=born%ycm
               xbjrk_born(1)=born%xbjrk(1)
               xbjrk_born(2)=born%xbjrk(2)
               write (*,fmt) 'BORNX',shat_born,tau_born,ycm_born
     &              ,xbjrk_born(1),xbjrk_born(2)
               write (*,fmt) 'TAULB',born%bounds(3)
               do irerun=1,2
                  born2=born
                  born2%xjac=1d0
                  born2%xpswgt=1d0
                  if (irerun.eq.1) then
                     pre='CUT'
                  else
                     pre='RAD'
                     born2%bounds(3)=0d0
                  endif
                  jac_rad=1d0
                  point2=fks_phase_space_point()
                  call generate_fks_radiation(born2,x(ndim-2),.false.
     &                 ,jac_rad,point2,pass)
                  p=point2%p
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
            endif
         enddo
      enddo
      enddo
      end


      subroutine born_at_srec(x,srec,wrec,prec)
c The Born of generate_noevpr_kinematics at each energy squared srec(k):
c generate_momenta_born with the random numbers x of the point and unit
c Jacobians, its weight and its momenta in their rest frame. The module
c data it writes (p_born, p_born_l, p_born_ev) are restored.
      use fks_phase_space_data, only: p_born,p_born_l,p_born_ev
      use fks_born_sampling, only: generate_momenta_born
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision x(99),srec(4),wrec(4),prec(0:3,nexternal-1,4)
      double precision qmass_common(-nexternal:0),qwidth_common(
     &     -nexternal:0)
      common /c_qmass_qwidth/qmass_common,qwidth_common
      double precision pmass(nexternal)
      common /to_mass/pmass
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision save_b(0:3,nexternal-1),save_l(0:3,nexternal-1)
     &     ,save_e(0:3,nexternal-1)
      double precision m(-max_branch:max_particles)
      double precision s(-max_branch:max_particles),m_born(nexternal-1)
      double precision totmass,xpswgt,xjac
      integer i,k
      save_b(:,:)=p_born(:,:)
      save_l(:,:)=p_born_l(:,:)
      save_e(:,:)=p_born_ev(:,:)
      do k=1,4
         m(:)=0d0
         s(:)=0d0
         do i=1,nexternal-1
            if (i.lt.i_fks) then
               m(i)=pmass(i)
            else
               m(i)=pmass(i+1)
            endif
         enddo
         totmass=0d0
         do i=nincoming+1,nexternal-1
            totmass=totmass+m(i)
         enddo
         xpswgt=1d0
         xjac=1d0
         call generate_momenta_born(x,srec(k),sqrt(srec(k)),totmass,m,s
     &        ,qmass_common,qwidth_common,m_born,xpswgt,xjac)
         wrec(k)=xpswgt*xjac
         prec(:,:,k)=p_born_l(:,:)
      enddo
      p_born(:,:)=save_b(:,:)
      p_born_l(:,:)=save_l(:,:)
      p_born_ev(:,:)=save_e(:,:)
      return
      end
