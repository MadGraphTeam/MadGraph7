! Test-only recording seam. No reducer, tensor, amplitude or formatted I/O.
module gp_recording_state
  use, intrinsic :: iso_c_binding, only: c_int64_t
  implicit none
  integer :: provenance_calls = 0, legacy_calls = 0
  integer :: recorded_we(3), recorded_rank, recorded_so, recorded_loop
  integer :: recorded_chord(0:3,3), recorded_class(3,3)
  logical :: recorded_have(3)
  integer(c_int64_t) :: recorded_environment, recorded_mass_words(2,3)
contains
  subroutine clear_recording()
    provenance_calls = 0
    legacy_calls = 0
    recorded_we = -999
    recorded_rank = -999
    recorded_so = -999
    recorded_loop = -999
    recorded_chord = -999
    recorded_class = -999
    recorded_have = .false.
    recorded_environment = 0_c_int64_t
    recorded_mass_words = 0_c_int64_t
  end subroutine
end module

subroutine ml5_0_loop_3_provenance_dispatch(w1,w2,w3,m1,m2,m3,rank, &
    squaredsoindex,loopnum,gp_chord,gp_class,gp_have,gp_env)
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double_complex
  use gp_recording_state
  implicit none
  integer :: w1,w2,w3,rank,squaredsoindex,loopnum
  complex(c_double_complex) :: m1,m2,m3
  integer :: gp_chord(0:3,3),gp_class(3,3)
  logical :: gp_have(3)
  integer(c_int64_t) :: gp_env
  provenance_calls = provenance_calls + 1
  recorded_chord = gp_chord
  recorded_class = gp_class
  recorded_have = gp_have
  recorded_environment = gp_env
  ! Keep the original emitted seam's legacy call and argument order.
  call ml5_0_loop_3(w1,w2,w3,m1,m2,m3,rank,squaredsoindex,loopnum)
end subroutine

subroutine ml5_0_loop_3(w1,w2,w3,m1,m2,m3,rank,squaredsoindex,loopnum)
  use, intrinsic :: iso_c_binding, only: c_double_complex
  use gp_recording_state
  implicit none
  integer :: w1,w2,w3,rank,squaredsoindex,loopnum
  complex(c_double_complex) :: m1,m2,m3
  legacy_calls = legacy_calls + 1
  recorded_we = [w1,w2,w3]
  recorded_rank = rank
  recorded_so = squaredsoindex
  recorded_loop = loopnum
  recorded_mass_words(:,1) = transfer(m1,recorded_mass_words(:,1))
  recorded_mass_words(:,2) = transfer(m2,recorded_mass_words(:,2))
  recorded_mass_words(:,3) = transfer(m3,recorded_mass_words(:,3))
end subroutine
