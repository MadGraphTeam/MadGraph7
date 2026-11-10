"""Materialize a private mtirust copy; do not touch installed/production code."""
import difflib
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def materialize():
    source = (ROOT / 'vendor/tirust/mtirust.f90').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == '3cb9cf98e82965e546365b293fdfd622cbae2ec09a7ae3d364de3ccf9b9bd56f'
    public = '  public :: tirust_context_generation, tirust_qp_cache_statistics\n'
    assert source.count(public) == 1
    result = source.replace(public, public + '  public :: tirust_qp_precision_available, tirust_qp_precision_snapshot\n')
    anchor = '    function set_cache_budget(handle,bytes) bind(C,name="ml_tirust_set_cache_budget") result(status)\n'
    assert result.count(anchor) == 1
    result = result.replace(anchor, '''    function qp_precision_words() bind(C,name="ml_tirust_private_qp_precision_words") result(count)
      import
      integer(c_int) :: count
    end function
    function qp_precision_snapshot(handle,words,count) &
        bind(C,name="ml_tirust_private_qp_precision_snapshot") result(status)
      import
      type(c_ptr), value :: handle
      integer(c_int64_t), intent(out) :: words(*)
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
''' + anchor)
    anchor = '  subroutine tirust_set_cache_budget(bytes,status)\n'
    assert result.count(anchor) == 1
    result = result.replace(anchor, '''  logical function tirust_qp_precision_available() result(available)
    available = qp_precision_words() == 11
  end function

  subroutine tirust_qp_precision_snapshot(words,generation,status,error_before,query_error)
    ! PRIVATE optional getter. Never creates a context or clears an event.
    ! Read after the caller consumes numerical status; guard() clears native
    ! LAST_ERROR, so save the existing error before querying and keep it local.
    ! First nine words are cumulative unsigned counters; last two are metadata.
    integer(c_int64_t), intent(out) :: words(:)
    integer, intent(out) :: generation,status
    character(len=*), intent(out) :: error_before,query_error
    integer(c_int64_t) :: temporary(11)
    character(len=512) :: previous_error
    call tirust_error_message(previous_error)
    error_before = previous_error
    query_error = ''
    words = 0_c_int64_t
    generation = tirust_context_generation()
    status = TIRUST_ERROR
    if (size(words) /= 11) then
      query_error = 'Adaptive QP telemetry requires exactly eleven words.'
    else if (.not.tirust_qp_precision_available()) then
      status = TIRUST_UNSUPPORTED
      query_error = 'This bridge has no adaptive QP precision getter.'
    else if (.not.c_associated(context)) then
      query_error = 'No live TIRust context; telemetry does not initialize it.'
    else
      status = qp_precision_snapshot(context,temporary,11_c_size_t)
      if (status == TIRUST_SUCCESS) then
        words = temporary
      else
        ! Read the getter error separately, without replacing the numerical error.
        local_error = ''
        call tirust_error_message(query_error)
        if (len_trim(query_error) == 0) query_error = 'Invalid adaptive QP telemetry response.'
      endif
    endif
    local_error = previous_error
  end subroutine

''' + anchor)
    (HERE / 'mtirust.f90').write_text(result)
    (HERE / 'mtirust.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True),
        fromfile='a/vendor/tirust/mtirust.f90', tofile='b/vendor/tirust/mtirust.f90')))

if __name__ == '__main__':
    assert Path.cwd() == ROOT
    assert not (HERE / 'mtirust.f90').exists()
    materialize()
