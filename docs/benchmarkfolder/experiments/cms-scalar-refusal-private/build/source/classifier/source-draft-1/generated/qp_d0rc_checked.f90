  subroutine cms_checked_d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  complex(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  call cms_preflight_d0rc(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  report%result_valid=.true.
  end subroutine cms_checked_d0rc

