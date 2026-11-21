      program frame_pipeline
      use driver_vec
      use couplings
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'coupl.inc'
      integer nfksprocess,i,j,k,l,ifr,icoup,step,m,n
      integer proc_map(0:fks_configs,0:fks_configs)
      common/c_nfksprocess/nfksprocess
      double precision p(0:3,nexternal,2)
      double precision pb(0:3,nexternal-1,2),ref,ret(amp_split_size)
      double precision amp2(ngraphs),jamp2(0:ncolor),rwgt
      double complex cnt(amp_split_size,2,nsplitorders)
      double complex ans(2,nsplitorders),sav(ngraphs,max_bhel)
      double complex linkcnt(amp_split_size,2,nsplitorders)
      double precision linkref,linkval,softref(amp_split_size)
      double precision softval(amp_split_size),linkscale
      logical need_color_links,need_charge_links
      common/c_need_links/need_color_links,need_charge_links
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      call setpara('param_card.dat')
      call allocate_storage(3,4,2,6,amp_split_size)
      step=4*fks_configs+1
      call allocate_couplings(3*step)
      read(*,*) ifr
      frame_id=ifr
      do l=1,2
         do j=1,nexternal
            read(*,*) (p(i,j,l),i=0,3)
         enddo
         do j=1,nexternal-1
            read(*,*) (pb(i,j,l),i=0,3)
         enddo
      enddo
         call reset_storage()
         proc_map=0
         proc_map(0,0)=1
         proc_map(1,0)=1
         proc_map(1,1)=1
         lane_group=1
         spb=-1d0
         do l=1,2
            g=1d0+l/10d0
            call update_as_param()
            do icoup=(l-1)*step+1,l*step
               g_vec(icoup)=g
               call update_as_param_vec(icoup)
            enddo
            nfksprocess=1
            call smatrix_real_frame(p(:,:,l),ref)
            calculatedborn=.false.
            call sborn_amp_frame(pb(:,:,l),amp2,jamp2,ret,cnt,
     $                           ref,ans,sav)
            spb(:,:,0,l)=pb(:,:,l)
            spb(:,:,1,l)=pb(:,:,l)
            spb_ev(:,:,1,l)=pb(:,:,l)
            spb_norad(:,:,1,l)=pb(:,:,l)
            spb_coll(:,:,1,l)=pb(:,:,3-l)
            spb_rot(:,:,1,l)=pb(:,:,3-l)
            sp1(:,:,1,l)=p(:,:,l)
            passcuts_nbody_vec(1,l)=.true.
            passcuts_n1body_vec(1,l)=.true.
C The caller's common FKS selector must not override the explicit row.
            nfksprocess=fks_configs
            call amplitudes_ivec(proc_map,rwgt,3,1,l)
            write(*,*) 'FRAME_CHECK',ifr,l,swgt_born(l),ref
            write(*,*) 'FRAME_ARRAYS',ifr,l,
     $           maxval(abs(sborn_amp_split(:,l)-ret)),
     $           maxval(abs(sborn_amp_split_cnt(:,:,:,l)-cnt)),
     $           maxval(abs(sborn_saveamp(:,:,l)-sav))
            nfksprocess=1
            need_color_links=.true.
            need_charge_links=.false.
            do m=1,nexternal-1
               if(m.eq.3) cycle
               do n=m+1,nexternal-1
                  if(n.eq.3) cycle
C Checked scalar links use the framed Born cache; stored links use lane data.
                  calculatedborn=.false.
                  call sborn_amp_frame(pb(:,:,l),amp2,jamp2,ret,cnt,
     $                                 ref,ans,sav)
                  call sborn_sf_amp_frame(pb(:,:,l),m,n,linkref,
     $                                 ans,cnt,sav,softref)
                  linkcnt=sborn_amp_split_cnt(:,:,:,l)
                  call sborn_sf_store(pb(:,:,l),m,n,linkval,
     $                 sborn_ans_cnt(:,:,l),linkcnt,
     $                 sborn_saveamp(:,:,l),softval)
                  linkscale=max(abs(linkref),maxval(abs(softref)),1d-99)
                  write(*,*) 'FRAME_LINK',ifr,l,m,n,linkval,linkref,
     $                        maxval(abs(softval-softref))/linkscale
               enddo
            enddo
            calculatedborn=.false.
            call sborn_amp_frame(pb(:,:,3-l),amp2,jamp2,ret,cnt,
     $                           ref,ans,sav)
            write(*,*) 'FRAME_COL',ifr,l,
     $           maxval(abs(scb_amp_split_cnt(:,:,:
     $           ,1,l)-cnt)),maxval(abs(scb_saveamp(:,:,1,l)-sav))
         enddo
         nfksprocess=fks_configs
         call amplitudes_real_vec(proc_map,3)
         do l=1,2
            g=1d0+l/10d0
            call update_as_param()
            nfksprocess=1
            call smatrix_real_frame(p(:,:,l),ref)
            write(*,*) 'FRAME_CHECK',ifr,l,sfx_ev(1,l),ref
            if (any(sp1(:,:,1,l).ne.p(:,:,l))) stop 10
            if (any(spb(:,:,0,l).ne.pb(:,:,l))) stop 11
         enddo
         if (any(sfx_ev(:,3).ne.0d0)) stop 12
      end
