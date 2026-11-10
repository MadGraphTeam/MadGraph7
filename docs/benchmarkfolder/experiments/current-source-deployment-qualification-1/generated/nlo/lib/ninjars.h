#ifndef MADLOOP_NINJARS_H
#define MADLOOP_NINJARS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { double re, im; } ninjars_complex;
typedef struct ninjars_context ninjars_context;
#define NINJARS_ABI_VERSION 4
enum { NINJARS_POLES_BOTH = 0, NINJARS_POLES_UV = 1, NINJARS_POLES_IR = 2, NINJARS_POLES_NONE = 3 };
enum { NINJARS_SUCCESS = 0, NINJARS_UNSTABLE_KINEMATICS = 1, NINJARS_ERROR = 2 };
uint32_t ninjars_abi_version(void);
int32_t ninjars_context_create(uint32_t version, ninjars_context **output);
/* Destroy only after outstanding calls complete; NULL is a no-op. */
int32_t ninjars_context_destroy(ninjars_context *context);
/*
 * One numerator, rank <= n+2. count=(rank+1)...(rank+4)/24.
 * Coefficients use Ninja/MadLoop tensor order. momenta[4*n] stores (E,x,y,z)
 * per propagator; s_matrix[n*n] is symmetric and masses[n] are squared.
 * scale_squared is positive mu_R^2. Results: finite, single, double pole.
 * Inputs and outputs must be valid and mutually disjoint for the entire call.
 * Each DP length counts elements (real or complex as declared), not bytes.
 * All lengths must exactly match the dimensions; mismatch is rejected before
 * reading or writing any numeric buffer. count is the coefficients length.
 * On failure output buffers remain unchanged. No exception crosses this ABI.
 */
int32_t ninjars_evaluate_dp(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const ninjars_complex *coefficients, const double *s_matrix, size_t s_length,
    const double *momenta, size_t p_length, const ninjars_complex *masses, size_t m_length,
    double scale_squared, ninjars_complex *laurent, size_t laurent_length,
    ninjars_complex *rational, size_t rational_length);
/* Explicit pole selection. Finite terms are unchanged; the selector is per call.
 * UV/IR DP uses the generic binary64 engine on the shared Symbolica worker
 * so scaleless-bubble coefficients remain available. Its rational/CC split
 * can differ from the fast engine; the complete finite term is equivalent.
 * Invalid modes fail. */
int32_t ninjars_evaluate_dp_with_poles(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const ninjars_complex *coefficients, const double *s_matrix, size_t s_length,
    const double *momenta, size_t p_length, const ninjars_complex *masses, size_t m_length,
    double scale_squared, ninjars_complex *laurent, size_t laurent_length,
    ninjars_complex *rational, size_t rational_length, int32_t pole_mode);

/* Explicit compensated DP retry. Same buffer and transactional contract as
 * ninjars_evaluate_dp; owns separate lazy state. The caller controls stability
 * checks and fallback. Ordinary DP calls never enable this path implicitly. */
int32_t ninjars_evaluate_dp_compensated(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const ninjars_complex *coefficients, const double *s_matrix, size_t s_length,
    const double *momenta, size_t p_length, const ninjars_complex *masses, size_t m_length,
    double scale_squared, ninjars_complex *laurent, size_t laurent_length,
    ninjars_complex *rational, size_t rational_length);
/* Selected UV/IR retries use the same generic binary64 engine as the ordinary
 * selected call, without the fast engine's compensated tensor callback.
 * Combined/no-pole retries retain that callback. Invalid modes fail. */
int32_t ninjars_evaluate_dp_compensated_with_poles(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const ninjars_complex *coefficients, const double *s_matrix, size_t s_length,
    const double *momenta, size_t p_length, const ninjars_complex *masses, size_t m_length,
    double scale_squared, ninjars_complex *laurent, size_t laurent_length,
    ninjars_complex *rational, size_t rational_length, int32_t pole_mode);
/*
 * MP records are 64-byte space-padded ASCII, 40 significant decimal digits,
 * with E exponent. Each number originated in binary128 (113 mantissa bits).
 * Counts in records: coefficients=2*count, matrix=n*n, momenta=4*n,
 * masses=2*n, scale=1. Complex records are real followed by imaginary.
 * output has 8 records: three Laurent complex values then the rational term.
 * Every MP buffer length is in bytes, not records; count is still the number
 * of complex tensor coefficients. All byte lengths must match exactly.
 * The complete reduction uses 256-bit arithmetic. All contexts share one
 * worker, with independent retained numerical state; last destroy joins it.
 * The last complex value is the generic engine's basis-dependent rational
 * diagnostic. Its split from the cut-constructible part can differ from DP
 * and original Ninja; the complete finite Laurent value includes both.
 * Symbolica licensing applies to MP and selected UV/IR DP execution.
 * Its restricted mode permits
 * one active Symbolica thread per user across all processes; concurrent MP
 * processes need a suitable license. Permit/license failures in upstream
 * Symbolica can terminate the process rather than return an ABI status.
 */
int32_t ninjars_evaluate_mp(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const unsigned char *coefficients, size_t coefficient_bytes,
    const unsigned char *s_matrix, size_t s_bytes,
    const unsigned char *momenta, size_t p_bytes,
    const unsigned char *masses, size_t m_bytes,
    const unsigned char *scale_squared, size_t scale_bytes,
    unsigned char *output, size_t output_bytes);
/* Explicit pole selection in the generic 256-bit engine. Finite terms and
 * the binary128 record contract are unchanged. Invalid modes fail. */
int32_t ninjars_evaluate_mp_with_poles(ninjars_context *context, int32_t n, int32_t rank, size_t count,
    const unsigned char *coefficients, size_t coefficient_bytes,
    const unsigned char *s_matrix, size_t s_bytes,
    const unsigned char *momenta, size_t p_bytes,
    const unsigned char *masses, size_t m_bytes,
    const unsigned char *scale_squared, size_t scale_bytes,
    unsigned char *output, size_t output_bytes, int32_t pole_mode);
/* Required length includes NUL; NULL/zero queries it. Error is calling-thread local. */
size_t ninjars_last_error(unsigned char *output, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
