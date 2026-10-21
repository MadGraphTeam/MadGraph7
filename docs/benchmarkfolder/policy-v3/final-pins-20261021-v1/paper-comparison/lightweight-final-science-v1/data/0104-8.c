/* Private diagnostic interposition only; all permitted C ABI calls are forwarded. */
#include "ninjars.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

/* create, native DP, compensated DP, MP, destroy, non-NULL destroy. */
static uint64_t counts[6];

static void report(const char *event, int has_status, int32_t status) {
    const char *name = getenv("NINJARS_GUARD_LOG");
    FILE *file = name ? fopen(name, "a") : NULL;
    if (!file) _Exit(90);
    if (fprintf(file, "%s", event) < 0) _Exit(91);
    if (has_status && fprintf(file, " status=%" PRId32, status) < 0) _Exit(91);
    if (fprintf(file,
        " create=%" PRIu64 " dp=%" PRIu64 " compensated=%" PRIu64
        " mp=%" PRIu64 " destroy=%" PRIu64 " nonnull=%" PRIu64 "\n",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]) < 0)
        _Exit(91);
    if (fclose(file)) _Exit(91);
}

static int32_t forbidden(int index, const char *event) {
    counts[index]++;
    report(event, 0, 0);
    _Exit(92);
}

extern int32_t __real_ninjars_context_create(uint32_t version,
                                           ninjars_context **output);
int32_t __wrap_ninjars_context_create(uint32_t version, ninjars_context **output) {
    counts[0]++;
    int32_t status = __real_ninjars_context_create(version, output);
    report("CREATE_RETURN", 1, status);
    return status;
}

#define DP_ARGUMENTS ninjars_context *context, int32_t n, int32_t rank, size_t count, \
    const ninjars_complex *coefficients, const double *s_matrix, size_t s_length, \
    const double *momenta, size_t p_length, const ninjars_complex *masses, size_t m_length, \
    double scale_squared, ninjars_complex *laurent, size_t laurent_length, \
    ninjars_complex *rational, size_t rational_length
#define DP_VALUES context, n, rank, count, coefficients, s_matrix, s_length, \
    momenta, p_length, masses, m_length, scale_squared, laurent, laurent_length, \
    rational, rational_length

extern int32_t __real_ninjars_evaluate_dp(DP_ARGUMENTS);
int32_t __wrap_ninjars_evaluate_dp(DP_ARGUMENTS) {
    counts[1]++;
    int32_t status = __real_ninjars_evaluate_dp(DP_VALUES);
    report("NATIVE_DP_RETURN", 1, status);
    return status;
}

int32_t __wrap_ninjars_evaluate_dp_compensated(DP_ARGUMENTS) {
    return forbidden(2, "FORBIDDEN_COMPENSATED_DP_ENTRY");
}

int32_t __wrap_ninjars_evaluate_mp(ninjars_context *context, int32_t n,
    int32_t rank, size_t count,
    const unsigned char *coefficients, size_t coefficient_bytes,
    const unsigned char *s_matrix, size_t s_bytes,
    const unsigned char *momenta, size_t p_bytes,
    const unsigned char *masses, size_t m_bytes,
    const unsigned char *scale_squared, size_t scale_bytes,
    unsigned char *output, size_t output_bytes) {
    return forbidden(3, "FORBIDDEN_MP_ENTRY");
}

extern int32_t __real_ninjars_context_destroy(ninjars_context *context);
int32_t __wrap_ninjars_context_destroy(ninjars_context *context) {
    counts[4]++;
    if (context) counts[5]++;
    int32_t status = __real_ninjars_context_destroy(context);
    report("DESTROY_RETURN", 1, status);
    return status;
}

__attribute__((constructor)) static void started(void) {
    report("GUARD_BEGIN", 0, 0);
}
__attribute__((destructor)) static void finished(void) {
    report("GUARD_FINAL", 0, 0);
}
