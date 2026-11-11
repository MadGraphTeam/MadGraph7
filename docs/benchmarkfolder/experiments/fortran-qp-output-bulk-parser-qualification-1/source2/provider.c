/* Recording ABI only. No reducer or floating-point arithmetic is linked here.
 * The supplied 128-hex-character lines encode 64 raw transport bytes each.
 * Cache insertion deliberately precedes public Fortran output validation.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

enum { WIDTH = 64, MAX_COUNT = 210, MAX_RECORDS = 8 * MAX_COUNT, CAPACITY = 2 };
struct entry {
    int key, n, rank, poles, finite;
    size_t length;
    char output[WIDTH * MAX_RECORDS];
};
static struct entry entries[CAPACITY];
static char supplied[WIDTH * MAX_RECORDS];
static size_t supplied_records;
static int handle, loaded, clear_fails, entry_count;
static int64_t hits, misses, refused, bypasses, budget = INT64_MAX;
/* clears, peak entries, creates, destroys, registrations, MP calls, DP calls,
 * rejected ABI requests. mock_reset resets counters after context admission. */
static int64_t counters[11];

static int marker(const char *text, size_t length) {
    return write(1, text, length) == (ssize_t)length ? 0 : 2;
}
void ml_bulk_observe(int mode) {
    if (mode == 1) { ++counters[8]; marker("BULK_ENTER\n", 11); }
    else if (mode == 0) { ++counters[9]; marker("SCALAR_ENTER\n", 13); }
    else if (mode == 2) { ++counters[10]; marker("FORCED_ENTER\n", 13); }
}

static int reject(void) { ++counters[7]; return 2; }
static int nibble(unsigned char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}
static int count_for_rank(int rank) {
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) / 24;
}
static int supported(int n, int rank, int finite) {
    return n >= 1 && n <= 6 && rank >= 0 && rank <= 6 &&
           (finite == 0 || finite == 1);
}
static int64_t cached_bytes(void) {
    int i;
    int64_t result = 0;
    for (i = 0; i < entry_count; ++i) result += (int64_t)entries[i].length;
    return result;
}
static void record(char *output, const char *value) {
    memset(output, ' ', WIDTH);
    memcpy(output, value, strlen(value));
}

int mock_load(int records, const char *hex, size_t hex_length) {
    size_t i;
    loaded = 0;
    if (records < 8 || records > MAX_RECORDS || records % 8 != 0 ||
        hex_length != (size_t)records * WIDTH * 2 || hex == NULL) return 2;
    for (i = 0; i < (size_t)records * WIDTH; ++i) {
        int high = nibble((unsigned char)hex[2 * i]);
        int low = nibble((unsigned char)hex[2 * i + 1]);
        if (high < 0 || low < 0) return 2;
        supplied[i] = (char)(16 * high + low);
    }
    supplied_records = (size_t)records;
    loaded = 1;
    return 0;
}
void mock_reset(int fail_clear) {
    entry_count = 0;
    clear_fails = fail_clear;
    hits = misses = refused = bypasses = 0;
    budget = INT64_MAX;
    memset(counters, 0, sizeof(counters));
}
int mock_counters(int64_t *output, size_t count) {
    if (count != 11 || output == NULL) return 2;
    memcpy(output, counters, sizeof(counters));
    return 0;
}
uint32_t ml_tirust_abi_version(void) { return 1; }
int ml_tirust_context_create(int version, void **output) {
    if (version != 1 || output == NULL) return reject();
    ++counters[2];
    *output = &handle;
    return 0;
}
int ml_tirust_context_destroy(void *context) {
    if (context != &handle && context != NULL) return reject();
    ++counters[3];
    entry_count = 0;
    return 0;
}
int ml_tirust_clear_cache(void *context) {
    if (context != &handle) return reject();
    ++counters[0];
    if (clear_fails) return 3;
    entry_count = 0;
    return 0;
}
int ml_tirust_register_plan(void *context, int n, int rank) {
    if (context != &handle || !supported(n, rank, 1)) return reject();
    ++counters[4];
    return 0;
}
int ml_tirust_set_cache_budget(void *context, int64_t bytes) {
    if (context != &handle || bytes < 0) return reject();
    budget = bytes;
    return 0;
}
int ml_tirust_supports(int n, int rank, int finite) {
    return supported(n, rank, finite) ? 0 : 3;
}
int ml_tirust_qp_cache_statistics(void *context, int64_t *output, size_t count) {
    if (context != &handle || output == NULL || count != 6) return reject();
    output[0] = hits;
    output[1] = misses;
    output[2] = entry_count;
    output[3] = cached_bytes();
    output[4] = refused;
    output[5] = bypasses;
    return 0;
}
int ml_tirust_cache_statistics(void *context, int64_t *output, size_t count) {
    if (context != &handle || output == NULL || count != 9) return reject();
    memset(output, 0, count * sizeof(*output));
    output[4] = hits;
    output[5] = misses;
    output[6] = entry_count;
    output[7] = cached_bytes();
    output[8] = refused;
    return 0;
}
size_t ml_tirust_last_error(char *output, size_t capacity) {
    const char *message = "mock native recording provider";
    size_t length = strlen(message);
    if (capacity && output != NULL) {
        size_t copied = length < capacity - 1 ? length : capacity - 1;
        memcpy(output, message, copied);
        output[copied] = 0;
    }
    return length + 1;
}
int ml_tirust_evaluate_dp(void *context, int n, int rank, const double *p,
    size_t plen, const double *s, size_t slen, const void *m, size_t mlen,
    double mu2, void *output, size_t olen, int poles, int finite, int check, int cache) {
    (void)context; (void)n; (void)rank; (void)p; (void)plen; (void)s; (void)slen;
    (void)m; (void)mlen; (void)mu2; (void)output; (void)olen; (void)poles;
    (void)finite; (void)check; (void)cache;
    ++counters[6];
    return 3;
}
int ml_tirust_evaluate_mp(void *context, int n, int rank, const char *p,
    size_t plen, const char *s, size_t slen, const char *m, size_t mlen,
    const char *mu2, size_t scale_length, char *output, size_t olen,
    int poles, int finite, int check, int cache) {
    int i, key = 0, eligible;
    size_t k, expected;
    struct entry *entry;
    ++counters[5];
    if (marker("ABI_ENTER\n", 10)) return reject();
    if (context != &handle || !loaded || !supported(n, rank, finite) ||
        poles < 0 || poles > 3 || (!finite && poles == 3) ||
        (check != 0 && check != 1) || (cache != 0 && cache != 1)) return reject();
    expected = (size_t)8 * count_for_rank(rank) * WIDTH;
    if (p == NULL || s == NULL || m == NULL || mu2 == NULL || output == NULL ||
        plen != (size_t)4 * n * WIDTH || slen != (size_t)n * n * WIDTH ||
        mlen != (size_t)2 * n * WIDTH || scale_length != WIDTH ||
        olen != expected || supplied_records * WIDTH != expected) return reject();
    /* Inputs use exact positive integer scales 1 and 2. No C decimal parser
     * or floating-point conversion can perturb the caller's IEEE state. */
    for (k = 0; k < WIDTH; ++k) {
        if (mu2[k] != ' ') { key = mu2[k] == '1' ? 1 : mu2[k] == '2' ? 2 : 0; break; }
    }
    if (key == 0) return reject();
    eligible = cache && !check;
    if (eligible) {
        for (i = 0; i < entry_count; ++i) {
            entry = &entries[i];
            if (entry->key == key && entry->n == n && entry->rank == rank &&
                entry->poles == poles && entry->finite == finite && entry->length == olen) {
                ++hits;
                memcpy(output, entry->output, olen);
                return 0;
            }
        }
        ++misses;
    } else ++bypasses;
    if (key == 2) memcpy(output, supplied, olen);
    else {
        for (k = 0; k < supplied_records; ++k)
            record(output + k * WIDTH, k % 2 ? "-0.0" : "1.25");
    }
    if (eligible) {
        if (entry_count >= CAPACITY || (int64_t)olen > budget - cached_bytes()) ++refused;
        else {
            entry = &entries[entry_count++];
            entry->key = key; entry->n = n; entry->rank = rank;
            entry->poles = poles; entry->finite = finite; entry->length = olen;
            memcpy(entry->output, output, olen);
            if (entry_count > counters[1]) counters[1] = entry_count;
        }
    }
    return 0;
}
