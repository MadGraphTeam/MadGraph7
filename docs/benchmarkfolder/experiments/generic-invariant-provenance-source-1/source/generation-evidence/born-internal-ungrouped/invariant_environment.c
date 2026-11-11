/* SOURCE ONLY. Read controls, never load/change modes or exception flags. */
#include <stdint.h>

int64_t ML5_0_kinematic_environment_token(void) {
#if defined(__x86_64__) && defined(__GNUC__)
    uint32_t mxcsr;
    uint16_t x87cw;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    __asm__ __volatile__("fnstcw %0" : "=m"(x87cw));
    /* IEEE status bits are excluded, all relevant control bits retained. */
    return (int64_t)(UINT64_C(0x4b45000000000000) |
                    ((uint64_t)(mxcsr & UINT32_C(0xffc0)) << 16) |
                    (uint64_t)(x87cw & UINT16_C(0x1f3f)));
#else
    return INT64_C(0); /* Unknown architecture/compiler: ineligible. */
#endif
}
