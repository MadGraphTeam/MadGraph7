  subroutine cms_checked_c0rr(rslt,p1,p2,p3,m1,m2,m3,report)
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3
  real(kindr2), intent(in) :: m1,m2,m3
  type(cms_scalar_report), intent(out) :: report
  call cms_preflight_c0rr(report,p1,p2,p3,m1,m2,m3)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call c0rr(rslt,p1,p2,p3,m1,m2,m3)
  report%result_valid=.true.
  end subroutine cms_checked_c0rr

