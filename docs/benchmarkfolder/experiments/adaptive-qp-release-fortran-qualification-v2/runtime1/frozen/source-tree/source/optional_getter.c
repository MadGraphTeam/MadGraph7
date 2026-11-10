/* Private GNU/Linux optional-symbol adapter. Released ABI1 remains linkable.
 * No dlopen/dlsym, allocation, floating arithmetic, context creation or reset.
 * This shim is additive and is not part of an installed bridge ABI promise. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

extern int32_t ml_tirust_qp_precision_statistics(void *, uint64_t *, size_t)
    __attribute__((weak));

int32_t ml_tirust_private_qp_precision_words(void) {
    return ml_tirust_qp_precision_statistics ? 11 : 0;
}

int32_t ml_tirust_private_qp_precision_snapshot(void *context, uint64_t *out, size_t count) {
    uint64_t words[11];
    int32_t status;
    if (count != 11 || !out || (uintptr_t)out % _Alignof(uint64_t)) return 2;
    if (!ml_tirust_qp_precision_statistics) return 3;
    if (!context) return 2;
    memset(words, 0xff, sizeof(words));
    status = ml_tirust_qp_precision_statistics(context, words, 11);
    if (status != 0) return status;
    /* These metadata values have a restricted domain, unlike u64 counters.
     * Sentinels detect a truncated 0..10-word successful reply. The original
     * ABI has no returned length, so it cannot prove arbitrary middle writes. */
    if (words[9] != 0 && words[9] != 256 && words[9] != 512) return 2;
    if (words[10] && ((words[10] >> 63) || (words[10] & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))) return 2;
    memcpy(out, words, sizeof(words));
    return 0;
}
