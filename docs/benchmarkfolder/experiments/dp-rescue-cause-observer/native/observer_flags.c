/* Private, GNU/Linux diagnostic. Read state; never clear/restore exceptions. */
#define _GNU_SOURCE
#include <fenv.h>
#include <stdint.h>

/* MPFR's public mpfr_flags_t is unsigned int; no MPFR numerical operation. */
extern unsigned int mpfr_flags_save(void);

int ml_tirust_private_observer_fp_read(uint64_t out[4])
{
    const int flags = fetestexcept(FE_ALL_EXCEPT);
    const int rounding = fegetround();
    const int traps = fegetexcept();
    const unsigned int mpfr = mpfr_flags_save();
    out[0] = (uint64_t)(unsigned int)flags;
    out[1] = (uint64_t)(unsigned int)rounding;
    out[2] = (uint64_t)(unsigned int)traps;
    out[3] = (uint64_t)mpfr;
    return flags >= 0 && rounding >= 0 && traps >= 0 && (mpfr & ~63u) == 0;
}
