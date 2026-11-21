#ifndef MADLOOP_TIRUST_H
#define MADLOOP_TIRUST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TIRUST_ABI_VERSION 1
typedef struct { double re, im; } ml_tirust_complex;
typedef struct ml_tirust_context ml_tirust_context;
enum { TIRUST_SUCCESS = 0, TIRUST_UNSTABLE = 1, TIRUST_ERROR = 2, TIRUST_UNSUPPORTED = 3 };
enum { TIRUST_POLES_BOTH = 0, TIRUST_POLES_UV = 1, TIRUST_POLES_IR = 2, TIRUST_POLES_NONE = 3 };

uint32_t ml_tirust_abi_version(void);
int32_t ml_tirust_context_create(uint32_t version, ml_tirust_context **output);
/* NULL destruction succeeds. A context must not have outstanding calls. */
int32_t ml_tirust_context_destroy(ml_tirust_context *context);
/* Clear retained integral values at each new phase-space point. Plans survive. */
int32_t ml_tirust_clear_cache(ml_tirust_context *context);
int32_t ml_tirust_register_plan(ml_tirust_context *context, int32_t n, int32_t rank);
/* Pure capability query; the qualified v1 range is n=1..6 and rank=0..6. */
int32_t ml_tirust_supports(int32_t n, int32_t rank, int32_t compute_finite);
/* bytes >= 0; zero disables insertion. Default total numerical cache budget:
 * 256 MiB. Half is reserved for QP whole-tensor results and a quarter for the
 * primary full-DP subintegral store. The last quarter is split between the
 * pole-only DP store (half of it, when poles are supported), the check-pass
 * subintegral store and the repeated DP whole-tensor cache.
 * Changing the budget clears numerical entries; setting the same value is a no-op. */
int32_t ml_tirust_set_cache_budget(ml_tirust_context *context, int64_t bytes);
/* count must equal nine. Statistics are unsigned counters, in order:
 * subsets_reused, subsets_computed, masters_reused, masters_computed,
 * moments_reused, moments_computed, entries, bytes, not_stored, summed over
 * the primary and check-pass stores. The last five also include the QP and
 * repeated-DP whole-tensor hits, misses, entries, bytes and refused insertions. */
int32_t ml_tirust_cache_statistics(ml_tirust_context *context, uint64_t *counts, size_t count);
/* count must equal six. QP whole-tensor counters, in order:
 * hits, misses, entries, bytes, not_stored, bypassed.
 * Hits/misses count tensor requests; a miss may subsequently fail reduction.
 * Bytes account for retained payloads and estimated entry metadata, excluding
 * plans/workspaces. These are not process resident-memory measurements.
 * Summed over the primary and check-pass caches, which never share an entry
 * and share one byte budget. cache_enabled=0 bypasses both without changing
 * hits or misses. Full/poles-only requests and pole selectors have distinct
 * keys. */
int32_t ml_tirust_qp_cache_statistics(ml_tirust_context *context, uint64_t *counts, size_t count);

/* Optional cumulative full-DP diagnostics, since context creation. count=20.
 * Primary then check blocks: Direct, MethodChoice, Tier106, Tier256, Tier512,
 * Refused, unreported/reduction-error, attempted106, attempted256, attempted512.
 * Accepted routes require successful finite DP output; attempts include failed
 * tiers. Public argument-validation failures are outside these counters.
 * Successful queries preserve the existing thread-local last error.
 * Cache clears preserve counters. A repeated-DP whole-tensor hit is not an
 * execution and changes no counter. Tier106 is DD; Tier256/512 evaluate
 * binary64 inputs with a DP numerator. None is genuine binary128-input QP. */
int32_t ml_tirust_dp_route_statistics(ml_tirust_context *context, uint64_t *counts, size_t count);

/*
 * Denominators: (q+p_i)^2 - m_i^2, metric (+,-,-,-). shifts[4*n] stores
 * time,x,y,z per propagator. invariants[n*n] is the symmetric column-major
 * matrix (p_i-p_j)^2; exact on-shell supplied values must be retained.
 * masses[n] contains complex squared masses with nonpositive imaginary part;
 * scale_squared is positive mu_R^2. All DP lengths count declared elements.
 *
 * count=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24 Cartesian tensor moments cover
 * every rank from zero through rank in IREGI order. Convention::MadLoop has
 * already applied the odd-rank sign; no multinomial factor is included.
 * output[4*count] is coefficient-fast: output[slot*count+coefficient], with
 * slots [UV1,IR1,IR2,finite]. Pole selection is applied after reduction.
 * When compute_finite=0, the finite slot is an UNEVALUATED zero placeholder:
 * callers must ignore it, never interpret it as a physical vanishing result.
 * MadLoop's user-facing partial-result diagnostic marks that slot NaN.
 * compute_finite, check_pass and cache_enabled are booleans encoded as 0/1.
 * A check pass never reads the primary pass's retained values: check passes
 * share subintegrals only with each other (bit-identical to unshared
 * evaluation). An exact repeat of a primary full-DP request within a point
 * (identical input bits, dimensions and pole selection) returns the stored
 * whole tensor.
 *
 * All lengths must match exactly. Buffers must remain valid and mutually
 * disjoint throughout the call. Failed calls leave outputs unchanged. No
 * Rust panic may unwind through this ABI. The Fortran adapter zeros outputs
 * on failure. Separate callers must use separate contexts.
 */
int32_t ml_tirust_evaluate_dp(ml_tirust_context *context, int32_t n, int32_t rank,
    const double *shifts, size_t shifts_length,
    const double *invariants, size_t invariants_length,
    const ml_tirust_complex *masses, size_t masses_length, double scale_squared,
    ml_tirust_complex *output, size_t output_length,
    int32_t pole_mode, int32_t compute_finite, int32_t check_pass, int32_t cache_enabled);

/*
 * Binary128 inputs use 64-byte space-padded ES64.39E4 ASCII records (40
 * significant digits). All MP lengths count bytes, not records. Record
 * counts: shifts=4*n, invariants=n*n, masses=2*n, scale_squared=1,
 * output=8*count. Each complex value has consecutive real/imaginary records;
 * the output uses the same coefficient-fast, four-slot layout as DP.
 * Inputs are parsed at 113 bits then widened to the working precision,
 * preserving signed zero. The complete calculation uses at least 256 bits.
 * The output is rounded by the Fortran adapter to genuine binary128.
 * GNU Fortran on Linux is the qualified target. Its native double-complex
 * layout matches the adjacent re/im fields of ml_tirust_complex.
 */
int32_t ml_tirust_evaluate_mp(ml_tirust_context *context, int32_t n, int32_t rank,
    const unsigned char *shifts, size_t shifts_bytes,
    const unsigned char *invariants, size_t invariants_bytes,
    const unsigned char *masses, size_t masses_bytes,
    const unsigned char *scale_squared, size_t scale_bytes,
    unsigned char *output, size_t output_bytes,
    int32_t pole_mode, int32_t compute_finite, int32_t check_pass, int32_t cache_enabled);

/* Required capacity includes NUL. NULL/zero queries it. Error is thread-local. */
size_t ml_tirust_last_error(unsigned char *output, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
