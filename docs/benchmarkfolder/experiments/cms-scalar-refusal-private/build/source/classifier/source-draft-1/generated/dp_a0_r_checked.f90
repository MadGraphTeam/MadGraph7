  subroutine cms_checked_a0_r(rslt,mm,report)
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  call cms_preflight_a0_r(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_r(rslt,mm)
  report%result_valid=.true.
  end subroutine cms_checked_a0_r

