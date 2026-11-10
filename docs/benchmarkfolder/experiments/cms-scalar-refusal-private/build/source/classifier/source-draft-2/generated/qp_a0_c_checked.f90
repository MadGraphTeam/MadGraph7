  subroutine cms_checked_a0_c(rslt,mm,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  complex(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_a0_c(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_c(cms_result_1,mm)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_a0_c

