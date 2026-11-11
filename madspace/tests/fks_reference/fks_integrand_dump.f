c Writes the fixed-order integrand of MadFKS at the points of a run, for
c make_fks_integrand_reference.py, which appends this file to
c driver_mintFO.f of the output and calls it from sigint right after
c include_PDF_and_alphas. It sees the weight_lines module as MadFKS sums
c it and writes
c - per contribution the type (add_wgt), the FKS configuration, the
c   power of g_strong, the coefficients wgt(1:3) of 1, log(mu_R^2/Q^2)
c   and log(mu_F^2/Q^2), the momentum fractions, the scales, g_strong,
c   the boost to the lab, the momenta and PDG codes, the luminosity per
c   subprocess and the final central weight wgts(1).
c - the phase-space point of the last FKS configuration: the Born with
c   its Jacobians and the bounds on tau (tau_Born_lower_bound,
c   tau_lower_bound_resonance, tau_lower_bound), the event and the
c   counter-events with theirs, the resonance of the resonance-local
c   recoil of the event with its mass squared.
c - the prefactors f_* of compute_prefactors_nbody and
c   compute_prefactors_n1body as sigint leaves them (with the
c   multichannel factor where include_multichannel_enhance ran, where
c   the cuts passed), the FKS symmetry factors and the normalizations of
c   xi.
c - where the Born is valid, the matrix-element blocks of the point,
c   each with the coupling g it took: the real per split-order position
c   at the event of the picked configuration (smatrix_real), the Born
c   and its azimuthal part per position (amp_split_cnt(:,1:2,qcd_pos) of
c   sborn) at that configuration with xij_kperp and xij_aor of the event
c   (the phase of the azimuthal part that sborncol_isr and sborncol_fsr
c   take at the collinear counter-events), the colour-linked Borns of
c   sborn_sf per position at the configuration of the n-body part for
c   the pairs that sbornsoft takes (fks_j_from_i of i_fks, a pair of the
c   same leg only if it is massive), the multichannel factor at both
c   configurations with amp2 of the Born diagrams it takes, and, where
c   bornsoftvirtual evaluated the virtual (a contribution of type 14),
c   the finite part of MadLoop it got (amp_split_finite_ML, at the Born
c   and the Ellis-Sexton scale, with the coupling of the n-body part),
c   the virtual fraction of the channel and that scale.
c The environment variable MADFKS_INTEGRAND_DUMP sets the number of
c points it writes (none without it), to fks_integrand_dump.txt in the
c channel directory.
      subroutine fks_integrand_dump(xx,x,vegas_wgt,nFKS_picked
     $     ,iran_picked,vol,nFKS_born,sum,real_type,phase_space)
      use weight_lines
      use mint_module, only: ichan,iconfig,ndimmax,virtual_fraction
      use fks_phase_space, only: fks_phase_space_point
      use fks_phase_space_data, only: p_born,xij_kperp,xij_aor
     $     ,xinorm_ev,xinorm_cnt,xiimax_ev,xiimax_cnt,resonance_recoil
     $     ,resonance_mass2,resonance_momentum
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'orders.inc'
      type(fks_phase_space_point) phase_space
      double precision f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      double precision fkssymmetryfactor,fkssymmetryfactorBorn
     $     ,fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn
     $     ,fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      double precision qes2
      common /coupl_es/ qes2
      double precision amp_split_finite_ML(amp_split_size)
      common /to_amp_split_finite/amp_split_finite_ML
      double precision born_wgt
      integer i_fks,j_fks,m,n,iamp,orders(nsplitorders)
      common/fks_indices/i_fks,j_fks
      double precision amp_split_soft(amp_split_size),wme
      common /to_amp_split_soft/amp_split_soft
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision emass(nexternal)
      common/to_mass/emass
      logical need_color_links,need_charge_links
      common /c_need_links/need_color_links,need_charge_links
      integer kk,jj
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
      if (nwritten.eq.0) then
         do iamp=1,amp_split_size
            call amp_split_pos_to_orders(iamp,orders)
            write(lun,'(a,99(1x,i4))') 'ORDERS',iamp,qcd_pos,qed_pos
     $           ,(orders(k),k=1,nsplitorders)
         enddo
      endif
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
      write(lun,'(a,3(1x,es24.16e3))') 'BOUNDS'
     $     ,(phase_space%born%bounds(k),k=1,3)
      write(lun,'(a,1x,i2,5(1x,es24.16e3))') 'RESONANCE'
     $     ,merge(1,0,resonance_recoil),resonance_mass2
     $     ,(resonance_momentum(k),k=0,3)
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
      write(lun,'(a,99(1x,es24.16e3))') 'FACTORS',f_b,f_nb,f_r,f_s
     $     ,f_c,f_dc,f_sc,(f_dsc(k),k=1,4),f_pdfsch_d,f_pdfsch_p
     $     ,f_pdfsch_l
      write(lun,'(a,99(1x,es24.16e3))') 'SYMMETRY',fkssymmetryfactor
     $     ,fkssymmetryfactorBorn,fkssymmetryfactorDeg
      write(lun,'(a,99(1x,es24.16e3))') 'XINORM',xinorm_ev
     $     ,(xinorm_cnt(k),k=0,2),xiimax_ev,(xiimax_cnt(k),k=0,2)
      write(lun,'(a,1x,i8)') 'NCONTR',icontr
      do i=1,icontr
         write(lun,'(a,8(1x,i12))') 'CINT',i,itype(i),nFKS(i)
     $        ,QCDpower(i),merge(1,0,H_event(i)),niproc(i),orderstag(i)
     $        ,amppos(i)
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
      if (phase_space%born%valid .and. real_type.eq.0) then
         write(lun,'(a,1x,es24.16e3)') 'G',g
         call update_fks_dir(nFKS_picked)
         write(lun,'(a,99(1x,i4))') 'FKS',nFKS_picked,i_fks,j_fks
         if (phase_space%event%valid) then
            call smatrix_real_frame(phase_space%p,wme)
            write(lun,'(a,99(1x,es24.16e3))') 'REAL',g,wme
     $           ,(amp_split(iamp),iamp=1,amp_split_size)
         endif
         calculatedBorn=.false.
         call sborn_frame(p_born,wme)
         write(lun,'(a,99(1x,es24.16e3))') 'KPERP',(xij_kperp(k),k=0
     $        ,3),dble(xij_aor),dimag(xij_aor)
         write(lun,'(a,99(1x,es24.16e3))') 'BORN_BLOCK',g,wme
     $        ,(dble(amp_split_cnt(iamp,1,qcd_pos)),iamp=1
     $        ,amp_split_size),(dble(amp_split_cnt(iamp,2,qcd_pos))
     $        ,iamp=1,amp_split_size),(dimag(amp_split_cnt(iamp,2
     $        ,qcd_pos)),iamp=1,amp_split_size)
         call fks_integrand_dump_channel(lun,'CHANNEL')
         call update_fks_dir(nFKS_born)
         write(lun,'(a,99(1x,i4))') 'FKSB',nFKS_born,i_fks,j_fks
         calculatedBorn=.false.
         call sborn_frame(p_born,born_wgt)
         call fks_integrand_dump_channel(lun,'CHANNELB')
         if (any(itype(1:icontr).eq.14)) then
            write(lun,'(a,99(1x,es24.16e3))') 'VIRTUAL',qes2
     $           ,virtual_fraction(ichan),(amp_split_finite_ML(iamp)
     $           ,iamp=1,amp_split_size)
         endif
         if (.not.(need_color_links.or.need_charge_links)) goto 50
         do kk=1,fks_j_from_i(i_fks,0)
            do jj=1,kk
               m=fks_j_from_i(i_fks,kk)
               n=fks_j_from_i(i_fks,jj)
               if (m.eq.i_fks .or. n.eq.i_fks) cycle
               if (m.eq.n .and. emass(m).eq.0d0) cycle
               calculatedBorn=.false.
               call sborn_frame(p_born,wme)
               call sborn_sf_frame(p_born,m,n,wme)
               write(lun,'(a,2(1x,i4),99(1x,es24.16e3))') 'LINK',m,n
     $              ,g,wme,(amp_split_soft(iamp),iamp=1,amp_split_size)
            enddo
         enddo
 50      continue
      endif
      write(lun,'(a)') 'END'
      call flush(lun)
      return
      end


c The multichannel factor of include_multichannel_enhance from amp2 of
c the last call of sborn: the channel (this_config), its Born diagram,
c diagramsymmetryfactor and the factor, then amp2 of the Born diagrams
c of mapconfig(:,0) in their order.
      subroutine fks_integrand_dump_channel(lun,label)
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer lun,i
      character*(*) label
      double precision xtot,enhance
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      integer this_config
      common/to_mconfigs/this_config
      double precision amp2(ngraphs),jamp2(0:ncolor)
      common/to_amps/amp2,jamp2
      double precision diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor
      xtot=0d0
      do i=1,mapconfig(0,0)
         xtot=xtot+amp2(mapconfig(i,0))
      enddo
      enhance=0d0
      if (xtot.ne.0d0) enhance=amp2(mapconfig(this_config,0))/xtot
     $     *diagramsymmetryfactor
      write(lun,'(a,2(1x,i4),2(1x,es24.16e3))') label,this_config
     $     ,mapconfig(this_config,0),diagramsymmetryfactor,enhance
      write(lun,'(a,99(1x,es24.16e3))') 'AMP2',(amp2(mapconfig(i,0)),i
     $     =1,mapconfig(0,0))
      return
      end
