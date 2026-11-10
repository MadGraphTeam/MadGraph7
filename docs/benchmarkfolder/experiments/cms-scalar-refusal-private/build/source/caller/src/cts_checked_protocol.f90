module cts_checked_protocol
  use cms_scalar_observer
  implicit none
  private
  public :: cts_checked_report
  type cts_checked_report
    integer :: status=CMS_OK,precision=0,direction=0,scalar_refusals=0
    logical :: result_valid=.false.,internally_rescued=.false.
    type(cms_scalar_report) :: first_refusal,last_scalar
  end type
end module cts_checked_protocol
