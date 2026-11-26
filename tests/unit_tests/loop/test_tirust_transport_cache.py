"""Exercise the actual QP decimal transport with an independent caching C ABI.

No Rust library or reducer is linked.  The mock deliberately caches records
before Fortran validates them, including records outside binary128's range.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from madgraph import MG5DIR


MODULE = Path(MG5DIR) / "vendor/tirust/mtirust.f90"


@unittest.skipUnless(shutil.which("gfortran") and shutil.which("cc"),
                     "gfortran and cc are required")
class QPTransportCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tirust-transport-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        (cls.work / "provider.c").write_text(PROVIDER)
        (cls.work / "driver.f90").write_text(DRIVER)
        cls.command(["cc", "-std=c99", "-O0", "-c", "provider.c"])
        cls.build(MODULE, "actual")
        cls.build(MODULE, "actual-o3", "-O3")
        # Remove only the production cache invalidation.  This is the previous
        # behavior, not a second implementation of decimal transport.
        source = MODULE.read_text()
        needle = "        ios = clear_cache(context)\n"
        if source.count(needle) != 1:
            raise AssertionError("expected one public transport invalidation")
        mutant = cls.work / "without_invalidation.f90"
        mutant.write_text(source.replace(needle, "", 1))
        cls.build(mutant, "without-invalidation")
        # Compile only the generated clear procedure from the live template.
        # Two prefixes exercise the shared MTIRUST state without a Rust reducer.
        template = (Path(MG5DIR) / "madgraph/iolibs/template_files/loop_optimized/"
                    "TIRUST_interface.inc").read_text()
        procedure = re.search(
            r"(?ms)^      SUBROUTINE %\(proc_prefix\)sCLEAR_TIRUST_CACHE\(\)\n.*?^      END\n",
            template)
        if procedure is None:
            raise AssertionError("generated TIRust clear procedure is missing")
        (cls.work / "generated_clear.f").write_text("\n".join(
            procedure.group() % {"proc_prefix": prefix} for prefix in ("ctx_a_", "ctx_b_")))
        (cls.work / "clear_driver.f90").write_text(CLEAR_DRIVER)
        for executable, optimization in (("generated-clear", "-O0"), ("generated-clear-o3", "-O3")):
            cls.command(["gfortran", optimization, "-fPIC", "-fcheck=all", "-frounding-math",
                         "-ffixed-line-length-132", str(MODULE), "generated_clear.f",
                         "clear_driver.f90", "provider.o", "-o", executable])

    @classmethod
    def command(cls, command):
        result = subprocess.run(command, cwd=cls.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout)
        return result.stdout

    @classmethod
    def build(cls, source, executable, optimization="-O0"):
        cls.command(["gfortran", optimization, "-fPIC", "-fcheck=all", "-frounding-math",
                     str(source), "driver.f90", "provider.o", "-o", executable])

    def invoke(self, executable, mode):
        return subprocess.run([str(self.work / executable), mode], cwd=self.work,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=30)

    def test_generated_event_clear_uses_only_the_shared_existing_context(self):
        for executable in ("generated-clear", "generated-clear-o3"):
            with self.subTest(executable=executable):
                result = self.invoke(executable, "lifecycle")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("PASS generated-clear lifecycle", result.stdout)

    def test_failed_public_transport_invalidates_cached_records(self):
        for executable in ("actual", "actual-o3"):
            with self.subTest(executable=executable):
                result = self.invoke(executable, "transport")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("PASS transport cases=28", result.stdout)

    def test_clear_failure_preserves_public_failure_and_local_message(self):
        for executable in ("actual", "actual-o3"):
            with self.subTest(executable=executable):
                result = self.invoke(executable, "clear-failure")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("PASS clear-failure cases=7", result.stdout)

    def test_old_behavior_is_rejected_by_the_same_cache_assertion(self):
        result = self.invoke("without-invalidation", "transport")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("FAIL rejected entries must be absent", result.stdout)

    def test_optional_route_statistics_preserves_context_and_diagnostics(self):
        for executable in ('actual', 'actual-o3'):
            with self.subTest(executable=executable):
                result = self.invoke(executable, 'routes')
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn('PASS routes cases=4', result.stdout)

    def test_finite_underflow_transport_is_unchanged(self):
        observed = []
        for executable in ("actual", "actual-o3", "without-invalidation"):
            result = self.invoke(executable, "underflow")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("PASS underflow cases=2", result.stdout)
            observed.append(result.stdout)
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[0], observed[2])


PROVIDER = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { WIDTH = 64, RECORDS = 8, CAPACITY = 8 };
struct entry {
    char key[WIDTH], output[WIDTH * RECORDS];
    int poles, finite;
};
static struct entry entries[CAPACITY];
static int entry_count, error_kind, clear_fails, clear_calls, peak_entries;
static int64_t hits, misses;
static int handle, context_creates, context_destroys, plan_registrations;

void mock_reset(int bad, int fail_clear) {
    entry_count = clear_calls = peak_entries = 0;
    hits = misses = 0;
    error_kind = bad;
    clear_fails = fail_clear;
}
void mock_counters(int *clears, int *peak) {
    *clears = clear_calls;
    *peak = peak_entries;
}
void mock_context_counters(int *creates, int *destroys, int *registrations) {
    *creates = context_creates;
    *destroys = context_destroys;
    *registrations = plan_registrations;
}
int ml_tirust_context_create(int version, void **out) {
    if (version != 1) return 2;
    ++context_creates;
    *out = &handle;
    return 0;
}
int ml_tirust_context_destroy(void *ctx) { ++context_destroys; return 0; }
int ml_tirust_clear_cache(void *ctx) {
    ++clear_calls;
    if (clear_fails) return 3;  /* Must not become the public result. */
    entry_count = 0;
    return 0;
}
int ml_tirust_register_plan(void *ctx, int n, int rank) { ++plan_registrations; return 0; }
int ml_tirust_set_cache_budget(void *ctx, int64_t bytes) { return 0; }
int ml_tirust_supports(int n, int rank, int finite) { return 0; }
int ml_tirust_qp_cache_statistics(void *ctx, int64_t *out, size_t count) {
    if (count != 6) return 2;
    memset(out, 0, count * sizeof(*out));
    out[0] = hits;
    out[1] = misses;
    out[2] = entry_count;
    out[3] = entry_count * WIDTH * RECORDS;
    return 0;
}
int ml_tirust_dp_route_statistics(void *ctx, uint64_t *out, size_t count) {
    if (count != 20) return 2;
    for (size_t i = 0; i < count; ++i) out[i] = (uint64_t)(i + 1);
    return error_kind == 7 ? 2 : 0;
}
int ml_tirust_cache_statistics(void *ctx, int64_t *out, size_t count) {
    memset(out, 0, count * sizeof(*out));
    return 0;
}
size_t ml_tirust_last_error(char *out, size_t capacity) {
    const char *text = "mock native error must not replace transport error";
    if (capacity) {
        size_t n = strlen(text);
        if (n >= capacity) n = capacity - 1;
        memcpy(out, text, n);
        out[n] = 0;
    }
    return strlen(text) + 1;
}
int ml_tirust_evaluate_dp(void *ctx, int n, int rank, const double *p,
    size_t plen, const double *s, size_t slen, const void *m, size_t mlen,
    double mu2, void *out, size_t olen, int poles, int finite, int check, int cache) {
    return 3;
}
static void record(char *out, const char *text) {
    memset(out, ' ', WIDTH);
    memcpy(out, text, strlen(text));
}
int ml_tirust_evaluate_mp(void *ctx, int n, int rank, const char *p,
    size_t plen, const char *s, size_t slen, const char *m, size_t mlen,
    const char *mu2, size_t scale_len, char *out, size_t olen,
    int poles, int finite, int check, int cache) {
    static const char *bad[] = {"unused", "not-a-number-record", "NaN",
                               "Infinity", "1.0E+5000", "1.0E-5000", "1.0E-4950"};
    struct entry *e;
    int i, is_bad = 0;
    if (ctx != &handle || n != 1 || rank != 0 || plen != 4 * WIDTH ||
        slen != WIDTH || mlen != 2 * WIDTH || scale_len != WIDTH ||
        olen != RECORDS * WIDTH || poles < 0 || poles > 3 ||
        (finite != 0 && finite != 1) || check != 0 || cache != 1) return 2;
    for (i = 0; i < entry_count; ++i) {
        e = &entries[i];
        if (memcmp(e->key, mu2, WIDTH) == 0 && e->poles == poles && e->finite == finite) {
            ++hits;
            memcpy(out, e->output, olen);
            return 0;
        }
    }
    if (entry_count == CAPACITY) return 2;
    ++misses;
    e = &entries[entry_count++];
    if (entry_count > peak_entries) peak_entries = entry_count;
    memcpy(e->key, mu2, WIDTH);
    e->poles = poles;
    e->finite = finite;
    /* The caller uses exact scales 1 and 2; inspect the first nonspace byte. */
    for (i = 0; i < WIDTH; ++i) if (mu2[i] != ' ') { is_bad = mu2[i] == '2'; break; }
    for (i = 0; i < RECORDS; ++i)
        record(e->output + i * WIDTH, i % 2 ? "-0.0" : "1.25");
    if (!finite) {
        record(e->output + 6 * WIDTH, "0.0");
        record(e->output + 7 * WIDTH, "0.0");
    }
    /* Put the bad record in a requested slot, after a valid real component.
       The finite-only setting selects its finite slot; partial UV selects UV. */
    if (is_bad) {
        int bad_record = poles == 3 ? 7 : poles == 1 ? 1 : 5;
        record(e->output + bad_record * WIDTH, bad[error_kind]);
    }
    memcpy(out, e->output, olen);
    return 0;
}
"""


DRIVER = r"""program transport_cache
  use mtirust
  use iso_c_binding
  use ieee_arithmetic
  use ieee_exceptions
  implicit none
  interface
    subroutine mock_reset(bad, fail_clear) bind(C)
      import c_int
      integer(c_int), value :: bad, fail_clear
    end subroutine
    subroutine mock_counters(clears, peak) bind(C)
      import c_int
      integer(c_int), intent(out) :: clears, peak
    end subroutine
  end interface
  real(KI_QTIR) :: p(0:3,1), s(1,1), scale
  complex(KI_QTIR) :: m(1), coefs(1,4)
  integer(c_int64_t) :: stats(6)
  integer(c_int) :: clears, peak
  integer :: bad, mode, finite, status, count
  logical :: old_halts(size(ieee_all)), halts(size(ieee_all)), sticky
  logical :: flags(5)
  integer(c_int64_t) :: bits(2)
  type(ieee_status_type) :: old_status
  character(32) :: action
  character(512) :: message

  call get_command_argument(1, action)
  call ieee_get_status(old_status)
  call ieee_get_halting_mode(ieee_all, old_halts)
  call ieee_set_halting_mode(ieee_all, .false.)
  p = 0
  s = 0
  m = cmplx(1, 0, kind=KI_QTIR)
  count = 0
  if (trim(action) == 'routes') then
    call check_routes()
    stop
  endif
  if (trim(action) == 'underflow') then
    do bad = 5, 6
      call mock_reset(int(bad,c_int), 0_c_int)
      call ieee_set_flag(ieee_all, .false.)
      call ieee_set_flag(ieee_divide_by_zero, .true.)
      scale = 2
      call tirust_tensor_evaluate(1, 0, p, s, m, scale, coefs, status, &
        pole_mode=0, compute_finite=.true., check_pass=.false., cache_enabled=.true.)
      call ieee_get_flag(ieee_invalid, flags(1))
      call ieee_get_flag(ieee_divide_by_zero, flags(2))
      call ieee_get_flag(ieee_overflow, flags(3))
      call ieee_get_flag(ieee_underflow, flags(4))
      call ieee_get_flag(ieee_inexact, flags(5))
      call ieee_get_halting_mode(ieee_all, halts)
      call require(.not.any(halts), 'underflow halting modes unchanged')
      call require(flags(2), 'underflow entry sticky flag preserved')
      call require(status == TIRUST_SUCCESS, 'finite underflow remains accepted')
      call require(ieee_is_finite(aimag(coefs(1,3))), 'underflow result finite')
      if (bad == 5) call require(aimag(coefs(1,3)) == 0, 'below-range value rounds to zero')
      if (bad == 6) call require(aimag(coefs(1,3)) > 0, 'representable subnormal remains nonzero')
      bits = transfer(aimag(coefs(1,3)), bits)
      call mock_counters(clears, peak)
      call require(clears == 0, 'accepted underflow does not invalidate')
      call check_stats(0, 1, 1)
      write(*,'(A,I0,A,5L2,A,2Z17.16)') 'UNDERFLOW ', bad, ' flags=', flags, ' bits=', bits
      count = count + 1
    enddo
  else
  do finite = 0, 1
    do mode = 0, 3
      if (finite == 0 .and. mode == 3) cycle
      if (trim(action) == 'clear-failure') then
        call mock_reset(1_c_int, 1_c_int)
        call evaluate(2, .false.)
        call mock_counters(clears, peak)
        call require(clears == 1, 'clear failure was attempted')
        call check_stats(0, 1, 1)
        count = count + 1
      else
        do bad = 1, 4
          call mock_reset(int(bad,c_int), 0_c_int)
          call evaluate(1, .true.)
          call check_stats(0, 1, 1)
          call evaluate(1, .true.)
          call check_stats(1, 1, 1)
          call mock_counters(clears, peak)
          call require(clears == 0, 'successful transport must not clear')
          call evaluate(2, .false.)
          call check_stats(1, 2, 0)
          call mock_counters(clears, peak)
          call require(clears == 1 .and. peak == 2, 'clear includes prior valid entry')
          call evaluate(2, .false.)
          call check_stats(1, 3, 0)
          call mock_counters(clears, peak)
          call require(clears == 2, 'identical failed key reevaluated and cleared')
          call evaluate(1, .true.)
          call check_stats(1, 4, 1)
          call mock_counters(clears, peak)
          call require(clears == 2, 'later success must not clear')
          count = count + 1
        enddo
      endif
    enddo
  enddo
  endif
  call tirust_finalize(status)
  call require(status == TIRUST_SUCCESS, 'finalize')
  call ieee_set_status(old_status)
  call ieee_set_halting_mode(ieee_all, old_halts)
  write(*,'(A,A,A,I0)') 'PASS ', trim(action), ' cases=', count
contains
  subroutine check_routes()
    integer(c_int64_t) :: routes(20)
    integer :: i,epoch
    character(512) :: before_message,after_message
    call tirust_set_cache_budget(-1_c_int64_t,status)
    call tirust_error_message(before_message)
    call tirust_dp_route_statistics(routes,status)
    call require(status == TIRUST_SUCCESS.and.all(routes == 0), 'unused route counters')
    call require(tirust_context_generation() == -1, 'route query must not create')
    call tirust_error_message(after_message)
    call require(before_message == after_message, 'unused local diagnostic preserved')
    call tirust_register_plan(1,0,status)
    epoch=tirust_context_generation()
    call tirust_error_message(before_message)
    call tirust_dp_route_statistics(routes,status)
    call require(status == TIRUST_SUCCESS, 'native diagnostic route status')
    call tirust_error_message(after_message)
    call require(before_message == after_message, 'C diagnostic preserved by module query')
    call tirust_set_cache_budget(-1_c_int64_t,status)
    call tirust_error_message(before_message)
    call tirust_dp_route_statistics(routes,status)
    call require(status == TIRUST_SUCCESS, 'active route status')
    call require(all(routes == [(int(i,c_int64_t),i=1,20)]), 'both ten-slot route blocks')
    call require(tirust_context_generation() == epoch, 'active generation preserved')
    call tirust_error_message(after_message)
    call require(before_message == after_message, 'active local diagnostic preserved')
    call mock_reset(7_c_int,0_c_int)
    call tirust_dp_route_statistics(routes,status)
    call require(status == TIRUST_ERROR.and.all(routes == 0), 'failed route counts unusable')
    call tirust_error_message(after_message)
    call require(before_message == after_message, 'failed query local diagnostic preserved')
    call tirust_finalize(status)
    call tirust_dp_route_statistics(routes,status)
    call require(status == TIRUST_SUCCESS.and.all(routes == 0), 'finalized route counters')
    call require(tirust_context_generation() == -1, 'finalized query must not create')
    write(*,'(A)') 'PASS routes cases=4'
  end subroutine
  subroutine require(condition, description)
    logical, intent(in) :: condition
    character(*), intent(in) :: description
    if (.not.condition) then
      write(*,'(A,A)') 'FAIL ', description
      stop 1
    endif
  end subroutine
  subroutine check_stats(expected_hits, expected_misses, expected_entries)
    integer, intent(in) :: expected_hits, expected_misses, expected_entries
    call tirust_qp_cache_statistics(stats, status)
    call require(status == TIRUST_SUCCESS, 'statistics status')
    call require(stats(3) == expected_entries, 'rejected entries must be absent')
    call require(stats(1) == expected_hits, 'cache hit count')
    call require(stats(2) == expected_misses, 'provider evaluation count')
  end subroutine
  subroutine evaluate(key, success)
    integer, intent(in) :: key
    logical, intent(in) :: success
    integer :: j
    call ieee_set_flag(ieee_all, .false.)
    call ieee_set_flag(ieee_divide_by_zero, .true.)
    scale = real(key, kind=KI_QTIR)
    coefs = cmplx(123, 456, kind=KI_QTIR)
    call tirust_tensor_evaluate(1, 0, p, s, m, scale, coefs, status, &
      pole_mode=mode, compute_finite=finite == 1, check_pass=.false., cache_enabled=.true.)
    call ieee_get_flag(ieee_divide_by_zero, sticky)
    call require(sticky, 'entry sticky flag preserved')
    call ieee_get_halting_mode(ieee_all, halts)
    call require(.not.any(halts), 'halting modes unchanged')
    if (success) then
      call require(status == TIRUST_SUCCESS, 'successful status')
      do j = 1, 4
        if (j == 4 .and. finite == 0) then
          call require(coefs(1,j) == cmplx(0,0,kind=KI_QTIR), 'partial finite placeholder')
        else
          call require(coefs(1,j) == cmplx(1.25_KI_QTIR,0,kind=KI_QTIR), 'decoded coefficient')
          call require(ieee_is_negative(aimag(coefs(1,j))), 'negative zero preserved')
        endif
      enddo
    else
      call require(status == TIRUST_ERROR, 'public transport must remain ERROR')
      call require(all(coefs == cmplx(0,0,kind=KI_QTIR)), 'failure outputs remain zero')
      call tirust_error_message(message)
      call require(trim(message) == &
        'TIRust returned an invalid or nonfinite binary128 output record.', 'local transport error preserved')
    endif
  end subroutine
end program
"""


CLEAR_DRIVER = r"""program generated_clear_lifecycle
  use mtirust
  use iso_c_binding
  use ieee_arithmetic
  use ieee_exceptions
  implicit none
  interface
    subroutine mock_reset(bad, fail_clear) bind(C)
      import c_int
      integer(c_int), value :: bad, fail_clear
    end subroutine
    subroutine mock_counters(clears, peak) bind(C)
      import c_int
      integer(c_int), intent(out) :: clears, peak
    end subroutine
    subroutine mock_context_counters(creates, destroys, registrations) bind(C)
      import c_int
      integer(c_int), intent(out) :: creates, destroys, registrations
    end subroutine
  end interface
  integer :: status, first_generation
  integer(c_int) :: creates, destroys, registrations, clears, peak
  integer(c_int64_t) :: stats(6)
  real(KI_QTIR) :: p(0:3,1), s(1,1)
  complex(KI_QTIR) :: m(1), coefs(1,4)
  logical :: before(5), after(5), old_halts(size(ieee_all))
  type(ieee_status_type) :: old_status
  type(ieee_round_type) :: round_before, round_after
  character(512) :: message_before, message_after

  call ieee_get_status(old_status)
  call ieee_get_halting_mode(ieee_all, old_halts)
  call ieee_set_halting_mode(ieee_all, .false.)
  call mock_reset(1_c_int, 0_c_int)
  call require(tirust_context_generation() == -1, 'initially absent')
  call tirust_set_cache_budget(-1_c_int64_t, status)
  call require(status == TIRUST_ERROR, 'local diagnostic fixture')
  call tirust_error_message(message_before)
  call ieee_set_flag(ieee_divide_by_zero, .true.)
  call ieee_get_flag(ieee_all, before)
  call ieee_get_rounding_mode(round_before)
  call ctx_a_clear_tirust_cache()
  call ctx_b_clear_tirust_cache()
  call ieee_get_flag(ieee_all, after)
  call ieee_get_rounding_mode(round_after)
  call require(all(before.eqv.after), 'inactive IEEE flags preserved')
  call require(round_before == round_after, 'inactive rounding preserved')
  call tirust_error_message(message_after)
  call require(message_before == message_after, 'inactive local diagnostic preserved')
  call require(tirust_context_generation() == -1, 'generated clear must not create')
  call counts(0, 0, 0, 0)

  call tirust_clear_cache(status)
  call require(status == TIRUST_SUCCESS, 'public clear remains creating')
  first_generation = tirust_context_generation()
  call require(first_generation > 0, 'public clear created generation')
  call counts(1, 0, 0, 1)
  p = 0
  s = 0
  m = cmplx(1, 0, kind=KI_QTIR)
  call seed()
  ! B has never registered a plan, but shares A's initialized module context.
  call ctx_b_clear_tirust_cache()
  call check_entries(0)
  call require(tirust_context_generation() == first_generation, 'clear preserves generation')
  call counts(1, 0, 0, 2)
  call tirust_register_plan(1, 0, status)
  call require(status == TIRUST_SUCCESS, 'first registration')
  call tirust_finalize(status)
  call require(status == TIRUST_SUCCESS, 'first finalization')
  call ctx_a_clear_tirust_cache()
  call ctx_b_clear_tirust_cache()
  call require(tirust_context_generation() == -1, 'finalized context stays absent')
  call counts(1, 1, 1, 2)
  call tirust_register_plan(1, 0, status)
  call require(status == TIRUST_SUCCESS, 'recreated registration')
  call require(tirust_context_generation() > first_generation, 'recreated epoch advances')
  call seed()
  call ctx_a_clear_tirust_cache()
  call check_entries(0)
  call counts(2, 1, 2, 3)
  call tirust_finalize(status)
  call require(status == TIRUST_SUCCESS, 'finalization')
  call counts(2, 2, 2, 3)
  call ieee_set_status(old_status)
  call ieee_set_halting_mode(ieee_all, old_halts)
  write(*,'(A)') 'PASS generated-clear lifecycle'
contains
  subroutine require(condition, description)
    logical, intent(in) :: condition
    character(*), intent(in) :: description
    if (.not.condition) then
      write(*,'(A,A)') 'FAIL ', description
      stop 1
    endif
  end subroutine
  subroutine counts(expected_creates, expected_destroys, expected_registrations, expected_clears)
    integer, intent(in) :: expected_creates, expected_destroys, expected_registrations, expected_clears
    call mock_context_counters(creates, destroys, registrations)
    call mock_counters(clears, peak)
    call require(creates == expected_creates, 'native creation count')
    call require(destroys == expected_destroys, 'native destruction count')
    call require(registrations == expected_registrations, 'native registration count')
    call require(clears == expected_clears, 'native clear count')
  end subroutine
  subroutine check_entries(expected)
    integer, intent(in) :: expected
    call tirust_qp_cache_statistics(stats, status)
    call require(status == TIRUST_SUCCESS, 'cache statistics status')
    call require(stats(3) == expected, 'shared cached entry count')
  end subroutine
  subroutine seed()
    call tirust_tensor_evaluate(1, 0, p, s, m, 1.0_KI_QTIR, coefs, status)
    call require(status == TIRUST_SUCCESS, 'seed valid cached tensor')
    call check_entries(1)
  end subroutine
end program
"""
