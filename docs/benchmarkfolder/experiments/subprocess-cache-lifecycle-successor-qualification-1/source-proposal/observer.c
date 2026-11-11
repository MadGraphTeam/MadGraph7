/*
 * Source proposal: serial, forwarding observer for the real TIRust ABI v1.
 * Compile with LC_OBSERVED and GNU --wrap for the six exports below. Without
 * LC_OBSERVED the driver API is present, but there are no __real_* references.
 * No wrapper allocates, writes files, performs floating arithmetic, modifies
 * input/output buffers, repairs the floating environment, or retries a call.
 * Every wrapper forwards exactly once, including after observer failure.
 *
 * Driver API (Fortran ISO_C_BINDING, exact names via bind(C)):
 *   void    lc_open(void);                       writes fixed "trace.txt"
 *   void    lc_begin(int32_t action, int32_t process);  both arguments VALUE
 *   void    lc_flush(void);                      only outside amplitude calls
 *   void    lc_close(void);
 *   int32_t lc_health(void);                     sticky failure bit mask
 *   int64_t lc_generation(void);                 observer context token
 *
 * A token increases on each successful nonnull creation, even if malloc
 * reuses an address. -1 means no observed active context. In a plain build
 * it ALWAYS means observation is absent, not that MTIRUST has no context.
 * The driver must separately record TIRUST_CONTEXT_GENERATION in both builds.
 * This observer intentionally accepts only one live context and serial calls.
 *
 * Text schema (ASCII whitespace, all numeric fields decimal integers):
 *   H 1 observed capacity sizeof_record
 *   R seq action process kind token_before token_after status n rank poles
 *     finite check cache version stats_pre_aggregate stats_pre_qp
 *     stats_post_aggregate stats_post_qp
 *     flags_before_observer flags_before_native flags_after_native
 *     flags_after_observer rounding_before_observer rounding_before_native
 *     rounding_after_native rounding_after_observer
 *     pre_A0 ... pre_A8 pre_Q0 ... pre_Q5 post_A0 ... post_A8 post_Q0 ... post_Q5
 *   F flush_serial records_in_flush total_record_attempts health active_token
 * One R line has 56 integer fields after R. Sequence numbers start at one
 * and do not reset on flush. action/process are supplied by the latest begin.
 * Kinds: 1 BEGIN, 2 CREATE, 3 DESTROY, 4 CLEAR, 5 REGISTER, 6 DP, 7 MP.
 * Not-applicable signed 32-bit fields are INT32_MIN; version is zero except
 * CREATE. Nonqueried counters are zero and their statistics statuses remain
 * INT32_MIN, so zero-filled data must never be treated as a valid snapshot.
 * Aggregate A and Q counter orders are exactly those in tirust.h. Counter
 * snapshots occur ONLY around CLEAR, only for the known nonnull active
 * context, and after CLEAR only when its native return is SUCCESS.
 *
 * For CREATE token_before is the previously active token, token_after the
 * newly observed token (or unchanged on failure). DESTROY records the target
 * token before and the current active token after. Other wrapper records use
 * the target token in both positions. An untracked nonnull target uses -2
 * and fails health. No pointer values are written to the trace.
 *
 * Health: 1 overflow, 2 statistics failure, 4 observer floating-environment
 * interference/query failure, 8 file I/O failure, 16 protocol/lifecycle
 * misuse, 32 unexpected context. Native reduction failures remain trace
 * statuses; they are not themselves observer failures. An F footer is a
 * checkpoint; the caller must also require lc_health()==0 after lc_close.
 *
 * The four floating-environment samples bracket the observer-only pre and
 * post sections separately. Changes made by the real numerical call are
 * recorded, never blamed on or repaired by the observer. The observation
 * code compares FE_ALL_EXCEPT and fegetround; GNU/Linux is the target here.
 * Statistics clear Rust's last-error string. Before CLEAR this is harmless:
 * the real clear itself unconditionally resets that string. After a failed
 * clear, and after every evaluation, no statistics API is called. Therefore
 * evaluation/clear failure messages remain available to the Fortran caller.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#include "tirust.h"

#define LC_CAPACITY 32768u
#define LC_NA INT32_MIN
enum {
    LC_OVERFLOW = 1, LC_STATS = 2, LC_FENV = 4, LC_IO = 8,
    LC_PROTOCOL = 16, LC_CONTEXT = 32
};
enum {
    LC_BEGIN = 1, LC_CREATE = 2, LC_DESTROY = 3, LC_CLEAR = 4,
    LC_REGISTER = 5, LC_DP = 6, LC_MP = 7
};

typedef struct {
    uint64_t sequence;
    int32_t action, process, kind;
    int64_t token_before, token_after;
    int32_t status, n, rank, poles, finite, check, cache;
    uint32_t version;
    int32_t statistics_status[4];
    int32_t flags[4], rounding[4];
    uint64_t before[15], after[15];
} lc_record;

typedef struct { int flags, rounding; } lc_environment;

static lc_record journal[LC_CAPACITY];
typedef char lc_journal_under_16_mib[
    sizeof(journal) < 16u * 1024u * 1024u ? 1 : -1];
static size_t used;
static uint64_t attempts, flush_serial;
static int32_t health, action_tag = -1, process_tag = -1;
static FILE *trace;
static int opened_once, closed;
#ifdef LC_OBSERVED
static ml_tirust_context *active_context;
static int64_t token_serial, active_token = -1;
#endif

int32_t lc_health(void) { return health; }

int64_t lc_generation(void)
{
#ifdef LC_OBSERVED
    return active_context != NULL ? active_token : -1;
#else
    return -1;
#endif
}

static lc_environment environment(void)
{
    lc_environment value;
    value.flags = fetestexcept(FE_ALL_EXCEPT);
    value.rounding = fegetround();
    if (value.flags < 0 || value.rounding < 0) health |= LC_FENV;
    return value;
}

static void unchanged(lc_environment before, lc_environment after)
{
    if (before.flags != after.flags || before.rounding != after.rounding)
        health |= LC_FENV;
}

static lc_record *reserve(int32_t kind)
{
    lc_record *record;
    size_t i;
    if (!opened_once || closed) health |= LC_PROTOCOL;
    if (attempts == UINT64_MAX) {
        health |= LC_OVERFLOW;
        return NULL;
    }
    ++attempts;
    if (used == LC_CAPACITY) {
        health |= LC_OVERFLOW;
        return NULL;
    }
    record = &journal[used++];
    memset(record, 0, sizeof(*record));
    record->sequence = attempts;
    record->action = action_tag;
    record->process = process_tag;
    record->kind = kind;
    record->token_before = record->token_after = lc_generation();
    record->status = record->n = record->rank = record->poles = LC_NA;
    record->finite = record->check = record->cache = LC_NA;
    for (i = 0; i < 4; ++i) {
        record->statistics_status[i] = LC_NA;
        record->flags[i] = record->rounding[i] = LC_NA;
    }
    return record;
}

static void sample(lc_record *record, size_t index, lc_environment value)
{
    if (record != NULL) {
        record->flags[index] = (int32_t)value.flags;
        record->rounding[index] = (int32_t)value.rounding;
    }
}

/* Runtime-map capture occurs outside every amplitude/ABI wrapper. */
static void lc_capture_maps(const char *name)
{
    FILE *input = fopen("/proc/self/maps", "rb");
    FILE *output = fopen(name, "wb");
    unsigned char buffer[4096];
    size_t size;
    if (input == NULL || output == NULL) {
        health |= LC_IO;
    } else {
        while ((size = fread(buffer, 1, sizeof(buffer), input)) != 0)
            if (fwrite(buffer, 1, size, output) != size) { health |= LC_IO; break; }
        if (ferror(input) || ferror(output)) health |= LC_IO;
    }
    if (input != NULL && fclose(input) != 0) health |= LC_IO;
    if (output != NULL && fclose(output) != 0) health |= LC_IO;
}

void lc_open(void)
{
    lc_environment before = environment();
    if (opened_once) {
        health |= LC_PROTOCOL;
        unchanged(before, environment());
        return;
    }
    opened_once = 1;
    trace = fopen("trace.txt", "w");
    if (trace == NULL) {
        health |= LC_IO;
    } else {
#ifdef LC_OBSERVED
        const int observed = 1;
#else
        const int observed = 0;
#endif
        if (fprintf(trace, "H 1 %d %u %llu\n", observed, LC_CAPACITY,
                    (unsigned long long)sizeof(lc_record)) < 0)
            health |= LC_IO;
    }
    lc_capture_maps("runtime-maps-open.txt");
    unchanged(before, environment());
}

void lc_begin(int32_t action, int32_t process)
{
    lc_environment before = environment(), after;
    lc_record *record;
    action_tag = action;
    process_tag = process;
    record = reserve(LC_BEGIN);
    if (record != NULL) record->status = TIRUST_SUCCESS;
    after = environment();
    unchanged(before, after);
    sample(record, 0, before);
    sample(record, 1, after);
    sample(record, 2, after);
    sample(record, 3, after);
}

void lc_flush(void)
{
    lc_environment before = environment();
    size_t i, j;
    if (!opened_once || closed || trace == NULL) {
        health |= LC_IO | LC_PROTOCOL;
        unchanged(before, environment());
        return;
    }
    if (flush_serial == UINT64_MAX) health |= LC_OVERFLOW;
    else ++flush_serial;
    for (i = 0; i < used; ++i) {
        const lc_record *r = &journal[i];
        if (fprintf(trace,
                    "R %llu %d %d %d %lld %lld %d %d %d %d %d %d %d %u",
                    (unsigned long long)r->sequence, (int)r->action,
                    (int)r->process, (int)r->kind,
                    (long long)r->token_before, (long long)r->token_after,
                    (int)r->status, (int)r->n, (int)r->rank, (int)r->poles,
                    (int)r->finite, (int)r->check, (int)r->cache,
                    (unsigned)r->version) < 0) health |= LC_IO;
        for (j = 0; j < 4; ++j)
            if (fprintf(trace, " %d", (int)r->statistics_status[j]) < 0)
                health |= LC_IO;
        for (j = 0; j < 4; ++j)
            if (fprintf(trace, " %d", (int)r->flags[j]) < 0) health |= LC_IO;
        for (j = 0; j < 4; ++j)
            if (fprintf(trace, " %d", (int)r->rounding[j]) < 0) health |= LC_IO;
        for (j = 0; j < 15; ++j)
            if (fprintf(trace, " %llu", (unsigned long long)r->before[j]) < 0)
                health |= LC_IO;
        for (j = 0; j < 15; ++j)
            if (fprintf(trace, " %llu", (unsigned long long)r->after[j]) < 0)
                health |= LC_IO;
        if (fputc('\n', trace) == EOF) health |= LC_IO;
    }
    unchanged(before, environment());
    if (fprintf(trace, "F %llu %llu %llu %d %lld\n",
                (unsigned long long)flush_serial, (unsigned long long)used,
                (unsigned long long)attempts, (int)health,
                (long long)lc_generation()) < 0) health |= LC_IO;
    if (fflush(trace) != 0 || ferror(trace)) health |= LC_IO;
    used = 0;
    unchanged(before, environment());
}

void lc_close(void)
{
    lc_environment before = environment();
    if (!opened_once || closed) {
        health |= LC_PROTOCOL;
        unchanged(before, environment());
        return;
    }
    lc_flush();
    if (trace != NULL && fclose(trace) != 0) health |= LC_IO;
    trace = NULL;
    closed = 1;
    lc_capture_maps("runtime-maps-close.txt");
    unchanged(before, environment());
}

#ifdef LC_OBSERVED

extern int32_t __real_ml_tirust_context_create(uint32_t, ml_tirust_context **);
extern int32_t __real_ml_tirust_context_destroy(ml_tirust_context *);
extern int32_t __real_ml_tirust_clear_cache(ml_tirust_context *);
extern int32_t __real_ml_tirust_register_plan(ml_tirust_context *, int32_t, int32_t);
extern int32_t __real_ml_tirust_evaluate_dp(
    ml_tirust_context *, int32_t, int32_t, const double *, size_t,
    const double *, size_t, const ml_tirust_complex *, size_t, double,
    ml_tirust_complex *, size_t, int32_t, int32_t, int32_t, int32_t);
extern int32_t __real_ml_tirust_evaluate_mp(
    ml_tirust_context *, int32_t, int32_t, const unsigned char *, size_t,
    const unsigned char *, size_t, const unsigned char *, size_t,
    const unsigned char *, size_t, unsigned char *, size_t,
    int32_t, int32_t, int32_t, int32_t);

static int64_t target_token(ml_tirust_context *context)
{
    if (context == NULL) return -1;
    if (context == active_context) return active_token;
    health |= LC_CONTEXT;
    return -2;
}

static lc_record *wrapper_start(int32_t kind, ml_tirust_context *context,
                                lc_environment before)
{
    lc_record *record = reserve(kind);
    int64_t token = target_token(context);
    if (record != NULL) record->token_before = record->token_after = token;
    sample(record, 0, before);
    return record;
}

static void before_native(lc_record *record, lc_environment before)
{
    lc_environment after = environment();
    unchanged(before, after);
    sample(record, 1, after);
}

static lc_environment after_native(lc_record *record, int32_t status)
{
    lc_environment value = environment();
    if (record != NULL) record->status = status;
    sample(record, 2, value);
    return value;
}

static void wrapper_finish(lc_record *record, lc_environment before)
{
    lc_environment after = environment();
    unchanged(before, after);
    sample(record, 3, after);
}

static void statistics(ml_tirust_context *context, lc_record *record, int after)
{
    uint64_t discarded[15];
    uint64_t *counts = record == NULL ? discarded :
                       (after ? record->after : record->before);
    int32_t aggregate, qp;
    if (context == NULL || context != active_context) {
        health |= LC_CONTEXT;
        return;
    }
    aggregate = ml_tirust_cache_statistics(context, counts, 9);
    qp = ml_tirust_qp_cache_statistics(context, counts + 9, 6);
    if (record != NULL) {
        record->statistics_status[after ? 2 : 0] = aggregate;
        record->statistics_status[after ? 3 : 1] = qp;
    }
    if (aggregate != TIRUST_SUCCESS || qp != TIRUST_SUCCESS) health |= LC_STATS;
}

int32_t __wrap_ml_tirust_context_create(uint32_t version, ml_tirust_context **output)
{
    lc_environment before = environment(), after;
    lc_record *record = reserve(LC_CREATE);
    int32_t status;
    sample(record, 0, before);
    if (record != NULL) record->version = version;
    before_native(record, before);
    status = __real_ml_tirust_context_create(version, output);
    after = after_native(record, status);
    if (status == TIRUST_SUCCESS) {
        if (output == NULL || *output == NULL) {
            health |= LC_CONTEXT;
        } else {
            if (active_context != NULL) health |= LC_CONTEXT;
            if (token_serial == INT64_MAX) health |= LC_OVERFLOW;
            else ++token_serial;
            active_context = *output;
            active_token = token_serial;
        }
    }
    if (record != NULL) record->token_after = lc_generation();
    wrapper_finish(record, after);
    return status;
}

int32_t __wrap_ml_tirust_context_destroy(ml_tirust_context *context)
{
    lc_environment before = environment(), after;
    lc_record *record = wrapper_start(LC_DESTROY, context, before);
    const int was_active = context != NULL && context == active_context;
    int32_t status;
    before_native(record, before);
    status = __real_ml_tirust_context_destroy(context);
    after = after_native(record, status);
    if (status == TIRUST_SUCCESS && was_active) {
        active_context = NULL;
        active_token = -1;
    }
    if (record != NULL) record->token_after = lc_generation();
    wrapper_finish(record, after);
    return status;
}

int32_t __wrap_ml_tirust_clear_cache(ml_tirust_context *context)
{
    lc_environment before = environment(), after;
    lc_record *record = wrapper_start(LC_CLEAR, context, before);
    int32_t status;
    if (context != NULL && context == active_context) statistics(context, record, 0);
    before_native(record, before);
    status = __real_ml_tirust_clear_cache(context);
    after = after_native(record, status);
    if (status == TIRUST_SUCCESS && context != NULL && context == active_context)
        statistics(context, record, 1);
    wrapper_finish(record, after);
    return status;
}

int32_t __wrap_ml_tirust_register_plan(ml_tirust_context *context, int32_t n, int32_t rank)
{
    lc_environment before = environment(), after;
    lc_record *record = wrapper_start(LC_REGISTER, context, before);
    int32_t status;
    if (record != NULL) { record->n = n; record->rank = rank; }
    before_native(record, before);
    status = __real_ml_tirust_register_plan(context, n, rank);
    after = after_native(record, status);
    wrapper_finish(record, after);
    return status;
}

static void evaluation_fields(lc_record *record, int32_t n, int32_t rank,
                              int32_t poles, int32_t finite, int32_t check,
                              int32_t cache)
{
    if (record == NULL) return;
    record->n = n;
    record->rank = rank;
    record->poles = poles;
    record->finite = finite;
    record->check = check;
    record->cache = cache;
}

int32_t __wrap_ml_tirust_evaluate_dp(
    ml_tirust_context *context, int32_t n, int32_t rank,
    const double *shifts, size_t shifts_length,
    const double *invariants, size_t invariants_length,
    const ml_tirust_complex *masses, size_t masses_length, double scale_squared,
    ml_tirust_complex *output, size_t output_length,
    int32_t pole_mode, int32_t compute_finite, int32_t check_pass, int32_t cache_enabled)
{
    lc_environment before = environment(), after;
    lc_record *record = wrapper_start(LC_DP, context, before);
    int32_t status;
    evaluation_fields(record, n, rank, pole_mode, compute_finite, check_pass, cache_enabled);
    before_native(record, before);
    status = __real_ml_tirust_evaluate_dp(context, n, rank, shifts, shifts_length,
        invariants, invariants_length, masses, masses_length, scale_squared,
        output, output_length, pole_mode, compute_finite, check_pass, cache_enabled);
    after = after_native(record, status);
    wrapper_finish(record, after);
    return status;
}

int32_t __wrap_ml_tirust_evaluate_mp(
    ml_tirust_context *context, int32_t n, int32_t rank,
    const unsigned char *shifts, size_t shifts_bytes,
    const unsigned char *invariants, size_t invariants_bytes,
    const unsigned char *masses, size_t masses_bytes,
    const unsigned char *scale_squared, size_t scale_bytes,
    unsigned char *output, size_t output_bytes,
    int32_t pole_mode, int32_t compute_finite, int32_t check_pass, int32_t cache_enabled)
{
    lc_environment before = environment(), after;
    lc_record *record = wrapper_start(LC_MP, context, before);
    int32_t status;
    evaluation_fields(record, n, rank, pole_mode, compute_finite, check_pass, cache_enabled);
    before_native(record, before);
    status = __real_ml_tirust_evaluate_mp(context, n, rank, shifts, shifts_bytes,
        invariants, invariants_bytes, masses, masses_bytes, scale_squared, scale_bytes,
        output, output_bytes, pole_mode, compute_finite, check_pass, cache_enabled);
    after = after_native(record, status);
    wrapper_finish(record, after);
    return status;
}
#endif
