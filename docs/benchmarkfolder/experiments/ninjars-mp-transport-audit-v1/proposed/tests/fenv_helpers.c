#include <fenv.h>
/* Test helpers keep platform FE_* constants out of Rust. */
int ninjars_t3_flags(void) { return fetestexcept(FE_ALL_EXCEPT); }
int ninjars_t3_clear_flags(void) { return feclearexcept(FE_ALL_EXCEPT); }
int ninjars_t3_bad_flags(void) {
    return fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
}
int ninjars_t3_round_upward(void) { return fesetround(FE_UPWARD); }
int ninjars_t3_round_nearest(void) { return fesetround(FE_TONEAREST); }
int ninjars_t3_raise_inexact(void) { return feraiseexcept(FE_INEXACT); }
