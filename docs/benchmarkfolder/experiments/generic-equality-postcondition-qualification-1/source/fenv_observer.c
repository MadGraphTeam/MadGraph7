#define _GNU_SOURCE 1
/* PRIVATE QUALIFICATION DRIVER ONLY. Never linked into the mathematical helper.
 * The helper cannot call this file. All environment setters are confined here.
 * SCHEMA snapshot fields: round,sticky,halts,mxcsr_control,x87cw,mxcsr_raw,x87sw.
 */
#include <fenv.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

#if defined(__x86_64__)
#define ML_GEOM_X86 1
#include <xmmintrin.h>
#else
#define ML_GEOM_X86 0
#endif

static fenv_t saved_environment;
static int saved_active;
static int64_t saved_snapshot[7];
#if ML_GEOM_X86
static uint32_t saved_mxcsr;
static uint16_t saved_x87cw;

static uint16_t get_cw(void) {
    uint16_t word;
    __asm__ volatile("fnstcw %0" : "=m" (word));
    return word;
}

static uint16_t get_sw(void) {
    uint16_t word;
    __asm__ volatile("fnstsw %0" : "=m" (word));
    return word;
}

static void set_cw(uint16_t word) {
    __asm__ volatile("fldcw %0" : : "m" (word));
}

/* Mask before clearing: no floating arithmetic is executed with old pending,
 * unmasked exceptions. Bit1 (denormal) is masked too, beyond FE_ALL_EXCEPT.
 */
static void mask_and_clear_hardware(void) {
    set_cw((uint16_t)(get_cw() | UINT16_C(0x003f)));
    _mm_setcsr((_mm_getcsr() | UINT32_C(0x00001f80)) & ~UINT32_C(0x003f));
    __asm__ volatile("fnclex");
}

static void clear_hardware_flags(void) {
    _mm_setcsr(_mm_getcsr() & ~UINT32_C(0x003f));
    __asm__ volatile("fnclex");
}

static uint32_t mxcsr_writable(void) {
    struct { unsigned char bytes[512]; } __attribute__((aligned(16))) area;
    uint32_t mask;
    memset(&area, 0, sizeof area);
    __asm__ volatile("fxsave %0" : "=m" (area));
    memcpy(&mask, area.bytes + 28, sizeof mask);
    /* Intel's architectural fallback excludes DAZ, whose support is optional. */
    return mask ? mask : UINT32_C(0x0000ffbf);
}
#endif

int ml_geom_qual_pid(void) { return (int)getpid(); }

/* Failure path: restoration was not observed, so formatted stdio or a Fortran
 * runtime error reporter must not run. Integer-only formatting and one raw
 * write retain the identity if stderr is available, then exit unconditionally.
 */
static char *append_text(char *out, const char *text) {
    while (*text) *out++ = *text++;
    return out;
}

static char *append_integer(char *out, int value) {
    char digits[32];
    unsigned count = 0;
    int64_t signed_value = value;
    uint64_t magnitude;
    if (signed_value < 0) {
        *out++ = '-';
        magnitude = (uint64_t)(-(signed_value + 1)) + UINT64_C(1);
    } else magnitude = (uint64_t)signed_value;
    do {
        digits[count++] = (char)('0' + magnitude % UINT64_C(10));
        magnitude /= UINT64_C(10);
    } while (magnitude);
    while (count) *out++ = digits[--count];
    return out;
}

_Noreturn void ml_geom_qual_raw_restore_failure(int id, int mode, int status, int restore_result) {
    char message[256], *end = message;
    end = append_text(end, "RESTORE_FAILURE id=");
    end = append_integer(end, id);
    end = append_text(end, " mode=");
    end = append_integer(end, mode);
    end = append_text(end, " status=");
    end = append_integer(end, status);
    end = append_text(end, " restore=");
    end = append_integer(end, restore_result);
    *end++ = '\n';
    (void)write(STDERR_FILENO, message, (size_t)(end - message));
    _exit(86);
}

void ml_geom_qual_constants(int64_t out[10]) {
    out[0] = FE_TONEAREST; out[1] = FE_DOWNWARD;
    out[2] = FE_UPWARD; out[3] = FE_TOWARDZERO;
    out[4] = FE_INVALID; out[5] = FE_DIVBYZERO; out[6] = FE_OVERFLOW;
    out[7] = FE_UNDERFLOW; out[8] = FE_INEXACT; out[9] = FE_ALL_EXCEPT;
}

int ml_geom_qual_observe(int64_t out[7]) {
    out[0] = fegetround();
    out[1] = fetestexcept(FE_ALL_EXCEPT);
#if defined(__GLIBC__)
    out[2] = fegetexcept();
#else
    out[2] = -1;
#endif
#if ML_GEOM_X86
    {
        uint32_t csr = _mm_getcsr();
        out[3] = csr & ~UINT32_C(0x003f);
        out[4] = get_cw();
        out[5] = csr;
        out[6] = get_sw();
    }
#else
    out[3] = -1; out[4] = -1; out[5] = -1; out[6] = -1;
#endif
    return out[0] == -1 ? -1 : 0;
}

/* Requests: actual platform round code, sticky mask, enabled halt mask, FTZ,
 * DAZ. -1 means preserve/unspecified. Return0=applied,1=explicit SKIP,2=error.
 * Every successful save MUST be paired with restore, including SKIP/error.
 */
int ml_geom_qual_begin(int mode, int64_t requests[5], int64_t baseline[7]) {
    int j, round_mode = FE_TONEAREST, sticky = 0, halts = 0;
    int ftz = -1, daz = -1;
    const int flags[5] = {FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW,
                          FE_UNDERFLOW, FE_INEXACT};
    fexcept_t raised;
    for (j = 0; j < 5; ++j) requests[j] = -1;
    if (ml_geom_qual_observe(baseline) != 0 || saved_active) return 2;
    if (fegetenv(&saved_environment) != 0) return 2;
    saved_active = 1;
    memcpy(saved_snapshot, baseline, sizeof saved_snapshot);
#if ML_GEOM_X86
    saved_mxcsr = _mm_getcsr(); saved_x87cw = get_cw();
#endif
    if (mode < 0 || mode >= 20) return 2;
    if (mode == 0) return 0; /* Untouched original-state control. */
    if (mode >= 1 && mode <= 4) {
        const int rounds[4] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
        round_mode = rounds[mode - 1];
    } else if (mode >= 5 && mode <= 9) sticky = flags[mode - 5];
    else if (mode == 10) sticky = FE_ALL_EXCEPT;
    else if (mode >= 11 && mode <= 15) halts = flags[mode - 11];
    else if (mode == 16) { ftz = 1; daz = 0; }
    else if (mode == 17) { ftz = 0; daz = 1; }
    else if (mode == 18) { ftz = 1; daz = 1; }
    else if (mode == 19) halts = FE_ALL_EXCEPT;
    requests[0] = round_mode; requests[1] = sticky; requests[2] = halts;
    requests[3] = ftz; requests[4] = daz;
#if !defined(__GLIBC__) || !ML_GEOM_X86
    return 1; /* This first qualification targets GNU libc on x86. */
#else
    if (ftz >= 0) {
        uint32_t wanted = (ftz ? UINT32_C(0x8000) : 0) |
                          (daz ? UINT32_C(0x0040) : 0);
        if ((wanted & ~mxcsr_writable()) != 0) return 1;
    }
    mask_and_clear_hardware();
    if (fedisableexcept(FE_ALL_EXCEPT) == -1) return 2;
    if (fesetround(round_mode) != 0) return 2;
    clear_hardware_flags();
    if (sticky) {
        /* Raising overflow may also raise inexact. Capture, clear ALL, then
         * install ONLY the requested bits using the standard opaque flag API.
         */
        if (feraiseexcept(sticky) != 0) return 2;
        if (fegetexceptflag(&raised, sticky) != 0) return 2;
        clear_hardware_flags();
        if (fesetexceptflag(&raised, sticky) != 0) return 2;
    }
    if (ftz >= 0) {
        uint32_t csr = _mm_getcsr() & ~UINT32_C(0x8040);
        if (ftz) csr |= UINT32_C(0x8000);
        if (daz) csr |= UINT32_C(0x0040);
        _mm_setcsr(csr);
    }
    if (halts) {
        /* Halting cohorts have zero sticky flags; never unmask pending ones. */
        if (sticky || fetestexcept(FE_ALL_EXCEPT) != 0) return 2;
        if (feenableexcept(halts) == -1) return 2;
    }
    if (fegetround() != round_mode || fetestexcept(FE_ALL_EXCEPT) != sticky ||
        fegetexcept() != halts) return 2;
    if (ftz >= 0 && (((_mm_getcsr() >> 15) & 1u) != (unsigned)ftz ||
                     ((_mm_getcsr() >> 6) & 1u) != (unsigned)daz)) return 2;
    return 0;
#endif
}

int ml_geom_qual_restore(int64_t restored[7]) {
    int result = 0;
    if (!saved_active) {
        ml_geom_qual_observe(restored);
        return -1;
    }
#if ML_GEOM_X86
    mask_and_clear_hardware();
#elif defined(__GLIBC__)
    fedisableexcept(FE_ALL_EXCEPT);
    feclearexcept(FE_ALL_EXCEPT);
#endif
    if (fesetenv(&saved_environment) != 0) result = -1;
#if ML_GEOM_X86
    _mm_setcsr(saved_mxcsr);
    set_cw(saved_x87cw);
#endif
    if (ml_geom_qual_observe(restored) != 0) result = -1;
    if (result == 0) result = memcmp(restored, saved_snapshot, sizeof saved_snapshot) == 0 ? 1 : 0;
    saved_active = 0;
    return result;
}
