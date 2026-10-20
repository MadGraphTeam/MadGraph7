#define _GNU_SOURCE
#include <fenv.h>
#include <stdint.h>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

/* Observation only: a rejected attempt must not change caller flags or traps.
   Other platforms use the original upstream callback in this experiment. */
int ninjars_t3_standard_environment(void) {
#if defined(__linux__) && defined(__x86_64__)
    unsigned short cw;
    __asm__ volatile ("fnstcw %0" : "=m"(cw));
    return fegetround() == FE_TONEAREST && fegetexcept() == 0
        && (_mm_getcsr() & ~UINT32_C(0x3f)) == UINT32_C(0x1f80)
        && (cw & 0x1f3f) == 0x033f;
#else
    return 0;
#endif
}

/* Test helpers keep platform FE_* constants out of Rust. */
int ninjars_t3_flags(void) { return fetestexcept(FE_ALL_EXCEPT); }
int ninjars_t3_clear_flags(void) { return feclearexcept(FE_ALL_EXCEPT); }
int ninjars_t3_bad_flags(void) {
    return fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
}
int ninjars_t3_round_upward(void) { return fesetround(FE_UPWARD); }
int ninjars_t3_round_nearest(void) { return fesetround(FE_TONEAREST); }
int ninjars_t3_raise_inexact(void) { return feraiseexcept(FE_INEXACT); }
