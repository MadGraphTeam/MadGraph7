  subroutine cms_checked_b11rc(b11,b00,b1,b0,pp,m1,m2,report)
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  complex(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  call cms_preflight_b11rc(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rc(b11,b00,b1,b0,pp,m1,m2)
  report%result_valid=.true.
  end subroutine cms_checked_b11rc

