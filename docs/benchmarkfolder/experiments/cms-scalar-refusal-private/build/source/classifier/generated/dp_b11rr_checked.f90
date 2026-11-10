  subroutine cms_checked_b11rr(b11,b00,b1,b0,pp,m1,m2,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  real(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2),cms_result_2(0:2),cms_result_3(0:2),cms_result_4(0:2)
  call cms_preflight_b11rr(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rr(cms_result_1,cms_result_2,cms_result_3,cms_result_4,pp,m1,m2)
  if (.not.all(ieee_is_finite(real((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  b11=cms_result_1
  b00=cms_result_2
  b1=cms_result_3
  b0=cms_result_4
  report%result_valid=.true.
  end subroutine cms_checked_b11rr

