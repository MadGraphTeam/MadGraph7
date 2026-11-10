  subroutine cms_checked_d0rr(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  real(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_d0rr(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rr(cms_result_1,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_d0rr

