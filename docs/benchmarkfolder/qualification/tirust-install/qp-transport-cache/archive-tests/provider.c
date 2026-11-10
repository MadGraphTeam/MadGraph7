
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { WIDTH = 64, RECORDS = 8, CAPACITY = 8 };
struct entry {
    char key[WIDTH], output[WIDTH * RECORDS];
    int poles, finite;
};
static struct entry entries[CAPACITY];
static int entry_count, error_kind, clear_fails, clear_calls, peak_entries;
static int64_t hits, misses;
static int handle;

void mock_reset(int bad, int fail_clear) {
    entry_count = clear_calls = peak_entries = 0;
    hits = misses = 0;
    error_kind = bad;
    clear_fails = fail_clear;
}
void mock_counters(int *clears, int *peak) {
    *clears = clear_calls;
    *peak = peak_entries;
}
int ml_tirust_context_create(int version, void **out) {
    if (version != 1) return 2;
    *out = &handle;
    return 0;
}
int ml_tirust_context_destroy(void *ctx) { return 0; }
int ml_tirust_clear_cache(void *ctx) {
    ++clear_calls;
    if (clear_fails) return 3;  /* Must not become the public result. */
    entry_count = 0;
    return 0;
}
int ml_tirust_register_plan(void *ctx, int n, int rank) { return 0; }
int ml_tirust_set_cache_budget(void *ctx, int64_t bytes) { return 0; }
int ml_tirust_supports(int n, int rank, int finite) { return 0; }
int ml_tirust_qp_cache_statistics(void *ctx, int64_t *out, size_t count) {
    if (count != 6) return 2;
    memset(out, 0, count * sizeof(*out));
    out[0] = hits;
    out[1] = misses;
    out[2] = entry_count;
    out[3] = entry_count * WIDTH * RECORDS;
    return 0;
}
int ml_tirust_cache_statistics(void *ctx, int64_t *out, size_t count) {
    memset(out, 0, count * sizeof(*out));
    return 0;
}
size_t ml_tirust_last_error(char *out, size_t capacity) {
    const char *text = "mock native error must not replace transport error";
    if (capacity) {
        size_t n = strlen(text);
        if (n >= capacity) n = capacity - 1;
        memcpy(out, text, n);
        out[n] = 0;
    }
    return strlen(text) + 1;
}
int ml_tirust_evaluate_dp(void *ctx, int n, int rank, const double *p,
    size_t plen, const double *s, size_t slen, const void *m, size_t mlen,
    double mu2, void *out, size_t olen, int poles, int finite, int check, int cache) {
    return 3;
}
static void record(char *out, const char *text) {
    memset(out, ' ', WIDTH);
    memcpy(out, text, strlen(text));
}
int ml_tirust_evaluate_mp(void *ctx, int n, int rank, const char *p,
    size_t plen, const char *s, size_t slen, const char *m, size_t mlen,
    const char *mu2, size_t scale_len, char *out, size_t olen,
    int poles, int finite, int check, int cache) {
    static const char *bad[] = {"unused", "not-a-number-record", "NaN",
                               "Infinity", "1.0E+5000", "1.0E-5000", "1.0E-4950"};
    struct entry *e;
    int i, is_bad = 0;
    if (ctx != &handle || n != 1 || rank != 0 || plen != 4 * WIDTH ||
        slen != WIDTH || mlen != 2 * WIDTH || scale_len != WIDTH ||
        olen != RECORDS * WIDTH || poles < 0 || poles > 3 ||
        (finite != 0 && finite != 1) || check != 0 || cache != 1) return 2;
    for (i = 0; i < entry_count; ++i) {
        e = &entries[i];
        if (memcmp(e->key, mu2, WIDTH) == 0 && e->poles == poles && e->finite == finite) {
            ++hits;
            memcpy(out, e->output, olen);
            return 0;
        }
    }
    if (entry_count == CAPACITY) return 2;
    ++misses;
    e = &entries[entry_count++];
    if (entry_count > peak_entries) peak_entries = entry_count;
    memcpy(e->key, mu2, WIDTH);
    e->poles = poles;
    e->finite = finite;
    /* The caller uses exact scales 1 and 2; inspect the first nonspace byte. */
    for (i = 0; i < WIDTH; ++i) if (mu2[i] != ' ') { is_bad = mu2[i] == '2'; break; }
    for (i = 0; i < RECORDS; ++i)
        record(e->output + i * WIDTH, i % 2 ? "-0.0" : "1.25");
    if (!finite) {
        record(e->output + 6 * WIDTH, "0.0");
        record(e->output + 7 * WIDTH, "0.0");
    }
    /* Put the bad record in a requested slot, after a valid real component.
       The finite-only setting selects its finite slot; partial UV selects UV. */
    if (is_bad) {
        int bad_record = poles == 3 ? 7 : poles == 1 ? 1 : 5;
        record(e->output + bad_record * WIDTH, bad[error_kind]);
    }
    memcpy(out, e->output, olen);
    return 0;
}
