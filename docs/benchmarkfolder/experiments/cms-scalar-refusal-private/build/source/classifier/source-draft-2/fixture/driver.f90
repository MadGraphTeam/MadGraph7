program classifier_mock
  use cms_mock_dp,only:check_dp=>fixture_check
  use cms_mock_qp,only:check_qp=>fixture_check
  implicit none
  if (precision(0._16).lt.33) error stop 2
  call check_dp()
  call check_qp()
  print *, 'COMPLETE_CLASSIFIER_MOCK_ONLY'
end program
