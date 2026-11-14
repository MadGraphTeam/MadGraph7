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

