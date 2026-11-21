      module mc_caller_configuration
      use mc_born_types, only: BornModelConfiguration
      implicit none
      type(BornModelConfiguration),save::configuration
      integer(8),save::saved_revision=-1
      end module
      subroutine mc_born_local(p,request,result)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'timing_variables.inc'
      double precision p(0:3,nexternal-1)
      type(BornRequest) request
      type(BornResult) result
      type(BornModelState),save::state
      integer status,nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision charges(nexternal-1)
      common/c_charges_born/charges
      if(timing_sampled)call timing_enter(timing_born)
      call mc_capture_model_state(state)
      request%sector=nfksprocess
      request%charges=charges
      request%helicity_frame=born_active_helicity_frame
      call born_evaluate(2,2,p,state,request,result,status)
      if(status.ne.0)then
      write(*,*)'Born support evaluation failed',status
      stop 1
      endif
      if(timing_sampled)call timing_leave(timing_born)
      end
      subroutine mc_capture_model_state(state)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'coupl.inc'
      type(BornModelState) state
      integer nr,nc
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      call mc_prepare_model_configuration(state%configuration_epoch)
      call born_model_dimensions(nr,nc)
      call born_resize_model_state(state,nr,nc)
      state%real_values(1)=g
      state%complex_values(1)=gc_10
      state%complex_values(2)=gc_11
      state%complex_values(3)=gc_12
      end
      subroutine mc_prepare_model_configuration(epoch)
      use mc_born_support
      use mc_born_types
      use mc_caller_configuration
      implicit none
      include 'coupl.inc'
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      integer(8) epoch,current_revision,mc_born_model_revision
      external mc_born_model_revision
      type(BornModelConfiguration),save::candidate
      integer status
      logical same_configuration
      current_revision=mc_born_model_revision()
      if(saved_revision.ne.current_revision.or.configuration%epoch.eq.0)then
      if(.not.allocated(candidate%real_values))allocate(candidate%real_values(2))
      if(.not.allocated(candidate%complex_values))allocate(candidate%complex_values(2))
      candidate%complex_values(1:2)=gal
      candidate%real_values(1)=mdl_mt
      candidate%real_values(2)=mdl_wt
      same_configuration=.false.
      if(configuration%epoch.gt.0)then
      same_configuration=all(candidate%real_values.eq.configuration%real_values).and.all(candidate%complex_values.eq.configuration%complex_values)
      endif
      if(.not.same_configuration)then
      configuration%real_values=candidate%real_values
      configuration%complex_values=candidate%complex_values
      call born_configure_model(configuration,status)
      if(status.ne.0)stop "Invalid native Born model configuration"
      endif
      saved_revision=current_revision
      endif
      if(configuration%epoch.ne.born_active_configuration_epoch)then
      call born_activate_model(configuration,status)
      if(status.ne.0)stop "Cannot reactivate native Born model configuration"
      endif
      epoch=configuration%epoch
      end
      integer(8) function mc_live_model_bytes()
      implicit none
      mc_live_model_bytes=600_8
      end
      subroutine mc_park_full_model(state)
      use mc_born_support
      use mc_born_types
      use mc_caller_configuration
      implicit none
      include 'coupl.inc'
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      type(BornLiveModel),intent(inout)::state
      integer(8) epoch
      call mc_prepare_model_configuration(epoch)
      if(state%configuration%epoch.ne.epoch)state%configuration=configuration
      state%helicity_frame=born_active_helicity_frame
      if(.not.allocated(state%real_values))allocate(state%real_values(49))
      if(.not.allocated(state%complex_values))allocate(state%complex_values(10))
      state%real_values(1)=g
      state%real_values(2)=all_g
      state%complex_values(1:2)=gal
      state%real_values(3)=mu_r
      state%real_values(4)=all_mu_r
      state%real_values(5)=mdl_mb
      state%real_values(6)=mdl_mh
      state%real_values(7)=mdl_mt
      state%real_values(8)=mdl_mta
      state%real_values(9)=mdl_mw
      state%real_values(10)=mdl_mz
      state%real_values(11)=mdl_wh
      state%real_values(12)=mdl_wt
      state%real_values(13)=mdl_ww
      state%real_values(14)=mdl_wz
      state%complex_values(3)=gc_10
      state%complex_values(4)=gc_11
      state%complex_values(5)=gc_12
      state%real_values(15)=mdl_sqrt__as
      state%real_values(16)=mdl_g__exp__2
      state%real_values(17)=mdl_conjg__ckm3x3
      state%real_values(18)=mdl_conjg__ckm1x1
      state%real_values(19)=mdl_ckm3x3
      state%real_values(20)=mdl_mz__exp__2
      state%real_values(21)=mdl_mz__exp__4
      state%real_values(22)=mdl_sqrt__2
      state%real_values(23)=mdl_mh__exp__2
      state%real_values(24)=mdl_aew
      state%real_values(25)=mdl_sqrt__aew
      state%real_values(26)=mdl_ee
      state%real_values(27)=mdl_mw__exp__2
      state%real_values(28)=mdl_sw2
      state%real_values(29)=mdl_cw
      state%real_values(30)=mdl_sqrt__sw2
      state%real_values(31)=mdl_sw
      state%real_values(32)=mdl_g1
      state%real_values(33)=mdl_gw
      state%real_values(34)=mdl_vev
      state%real_values(35)=mdl_vev__exp__2
      state%real_values(36)=mdl_lam
      state%real_values(37)=mdl_yb
      state%real_values(38)=mdl_yt
      state%real_values(39)=mdl_ytau
      state%real_values(40)=mdl_muh
      state%real_values(41)=mdl_ee__exp__2
      state%real_values(42)=mdl_sw__exp__2
      state%real_values(43)=mdl_cw__exp__2
      state%real_values(44)=aewm1
      state%real_values(45)=mdl_gf
      state%real_values(46)=as
      state%real_values(47)=mdl_ymb
      state%real_values(48)=mdl_ymt
      state%real_values(49)=mdl_ymtau
      state%complex_values(6)=mdl_complexi
      state%complex_values(7)=mdl_i1x33
      state%complex_values(8)=mdl_i2x33
      state%complex_values(9)=mdl_i3x33
      state%complex_values(10)=mdl_i4x33
      end
      subroutine mc_activate_full_model(state)
      use mc_born_support
      use mc_born_types
      use mc_caller_configuration
      implicit none
      include 'coupl.inc'
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      type(BornLiveModel),intent(in)::state
      integer status
      integer(8),external::mc_born_model_revision
      if(state%configuration%epoch.le.0)error stop "Missing native caller model"
      g=state%real_values(1)
      all_g=state%real_values(2)
      gal=state%complex_values(1:2)
      mu_r=state%real_values(3)
      all_mu_r=state%real_values(4)
      mdl_mb=state%real_values(5)
      mdl_mh=state%real_values(6)
      mdl_mt=state%real_values(7)
      mdl_mta=state%real_values(8)
      mdl_mw=state%real_values(9)
      mdl_mz=state%real_values(10)
      mdl_wh=state%real_values(11)
      mdl_wt=state%real_values(12)
      mdl_ww=state%real_values(13)
      mdl_wz=state%real_values(14)
      gc_10=state%complex_values(3)
      gc_11=state%complex_values(4)
      gc_12=state%complex_values(5)
      mdl_sqrt__as=state%real_values(15)
      mdl_g__exp__2=state%real_values(16)
      mdl_conjg__ckm3x3=state%real_values(17)
      mdl_conjg__ckm1x1=state%real_values(18)
      mdl_ckm3x3=state%real_values(19)
      mdl_mz__exp__2=state%real_values(20)
      mdl_mz__exp__4=state%real_values(21)
      mdl_sqrt__2=state%real_values(22)
      mdl_mh__exp__2=state%real_values(23)
      mdl_aew=state%real_values(24)
      mdl_sqrt__aew=state%real_values(25)
      mdl_ee=state%real_values(26)
      mdl_mw__exp__2=state%real_values(27)
      mdl_sw2=state%real_values(28)
      mdl_cw=state%real_values(29)
      mdl_sqrt__sw2=state%real_values(30)
      mdl_sw=state%real_values(31)
      mdl_g1=state%real_values(32)
      mdl_gw=state%real_values(33)
      mdl_vev=state%real_values(34)
      mdl_vev__exp__2=state%real_values(35)
      mdl_lam=state%real_values(36)
      mdl_yb=state%real_values(37)
      mdl_yt=state%real_values(38)
      mdl_ytau=state%real_values(39)
      mdl_muh=state%real_values(40)
      mdl_ee__exp__2=state%real_values(41)
      mdl_sw__exp__2=state%real_values(42)
      mdl_cw__exp__2=state%real_values(43)
      aewm1=state%real_values(44)
      mdl_gf=state%real_values(45)
      as=state%real_values(46)
      mdl_ymb=state%real_values(47)
      mdl_ymt=state%real_values(48)
      mdl_ymtau=state%real_values(49)
      mdl_complexi=state%complex_values(6)
      mdl_i1x33=state%complex_values(7)
      mdl_i2x33=state%complex_values(8)
      mdl_i3x33=state%complex_values(9)
      mdl_i4x33=state%complex_values(10)
      if(configuration%epoch.ne.state%configuration%epoch)configuration=state%configuration
      saved_revision=mc_born_model_revision()
      call born_activate_model(configuration,status)
      if(status.ne.BORN_OK)error stop "Invalid restored native model configuration"
      born_active_helicity_frame=state%helicity_frame
      end
