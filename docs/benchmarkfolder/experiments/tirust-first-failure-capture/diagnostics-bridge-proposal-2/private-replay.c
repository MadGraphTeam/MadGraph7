/* Source-only diagnostic replay. Build with -I <the pinned repository root>.
 * No floating-point arithmetic or classification is performed in this file.
 * Inputs and successful outputs cross the C ABI as their captured raw bits.
 * No result from a failed bridge call is inspected. No automatic retry occurs.
 */
#define _POSIX_C_SOURCE 200809L
#include "vendor/tirust/tirust.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(double) == 8, "Captured DP words require binary64");
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "Captured DP words require IEEE binary64");
_Static_assert(sizeof(ml_tirust_complex) == 16, "Unexpected complex ABI");
_Static_assert(offsetof(ml_tirust_complex, re) == 0 &&
               offsetof(ml_tirust_complex, im) == 8, "Unexpected complex layout");

enum { MAX_REQUESTS = 6, MAX_EVALUATIONS = 12, MAX_PATH_BYTES = 4096,
       MAX_INPUT_BYTES = 32768, ERROR_CAPACITY = 4096, MAX_OUTPUT = 504 };

/* Private bridge metadata only. This accessor never clears LAST_ERROR. */
extern size_t ml_tirust_private_dp_diagnostics_v1(unsigned char *, size_t);
enum { PRIVATE_DIAGNOSTIC_CAPACITY = 65536 };

struct request {
    char path[MAX_PATH_BYTES + 1];
    int32_t case_id, phase, point, method, group, n, rank;
    int32_t plen, slen, mlen, olen, poles, finite, check, cache;
    double mu2, p[24], s[36];
    ml_tirust_complex masses[6];
};

struct bridge_error {
    size_t required, copied;
    int truncated;
    unsigned char bytes[ERROR_CAPACITY];
};

/* This is the first action after each guarded bridge call. In particular,
 * context destruction, another evaluation and plan registration must never
 * overwrite the thread-local diagnostic before this copy has completed.
 */
static void copy_error(struct bridge_error *error)
{
    error->required = ml_tirust_last_error(error->bytes, sizeof error->bytes);
    error->truncated = error->required > sizeof error->bytes;
    error->copied = error->required ? error->required - 1 : 0;
    if (error->copied > sizeof error->bytes - 1)
        error->copied = sizeof error->bytes - 1;
}

static void json_bytes(FILE *out, const unsigned char *bytes, size_t length)
{
    size_t i;
    fputc('"', out);
    for (i = 0; i < length; ++i) {
        unsigned int byte = bytes[i];
        if (byte == '"' || byte == '\\') {
            fputc('\\', out); fputc((int)byte, out);
        } else if (byte < 0x20 || byte >= 0x7f) {
            fprintf(out, "\\u%04x", byte);
        } else {
            fputc((int)byte, out);
        }
    }
    fputc('"', out);
}

static int record(FILE *out, const struct request *r, const char *stage,
                  int repetition, int32_t status, const struct bridge_error *error,
                  const ml_tirust_complex *output, int context_present,
                  unsigned int evaluation_count)
{
    int32_t i;
    fprintf(out, "{\"record_type\":\"%s\",\"input_key\":{"
            "\"case_id\":%" PRId32 ",\"phase\":%" PRId32
            ",\"point\":%" PRId32 ",\"method\":%" PRId32
            ",\"group\":%" PRId32 "},\"input_path\":",
            stage, r->case_id, r->phase, r->point, r->method, r->group);
    json_bytes(out, (const unsigned char *)r->path, strlen(r->path));
    fprintf(out, ",\"repetition\":%d,\"status\":%" PRId32
            ",\"error_required\":%zu,\"error_truncated\":%s,"
            "\"error_encoding\":\"latin1-bytes\",\"error\":",
            repetition, status, error->required,
            error->truncated ? "true" : "false");
    json_bytes(out, error->bytes, error->copied);
    fprintf(out, ",\"context_present\":%s,\"evaluation_count\":%u",
            context_present < 0 ? "null" : context_present ? "true" : "false",
            evaluation_count);
    if (strcmp(stage, "destroy") == 0)
        fprintf(out, ",\"destroy_succeeded\":%s",
                status == TIRUST_SUCCESS ? "true" : "false");
    if (strcmp(stage, "evaluate") == 0) {
        unsigned char diagnostic[PRIVATE_DIAGNOSTIC_CAPACITY];
        size_t needed, copied;
        needed = ml_tirust_private_dp_diagnostics_v1(diagnostic, sizeof diagnostic);
        if (needed == 0 || needed > sizeof diagnostic) return 0;
        copied = needed - 1;
        if (diagnostic[copied] != 0) return 0;
        fprintf(out, ",\"private_diagnostic_required\":%zu,"
                "\"private_diagnostic_copy_truncated\":false,"
                "\"private_diagnostic_encoding\":\"latin1-bytes\","
                "\"private_diagnostic\":", needed);
        json_bytes(out, diagnostic, copied);
    }
    /* Even a poisoned or unchanged output buffer is not read on failure. */
    if (strcmp(stage, "evaluate") == 0 && status == TIRUST_SUCCESS) {
        fprintf(out, ",\"output_bits\":[");
        for (i = 0; i < r->olen; ++i) {
            uint64_t words[2];
            memcpy(words, output + i, sizeof words);
            fprintf(out, "%s\"%016" PRIx64 "\",\"%016" PRIx64 "\"",
                    i ? "," : "", words[0], words[1]);
        }
        fputc(']', out);
    }
    fputs("}\n", out);
    return !ferror(out) && fflush(out) == 0;
}

/* Printable ASCII records with LF termination only. No NUL, CR, tabs,
 * overlong line or silently accepted trailing bytes are permitted.
 * Return 1 for a record, 0 for clean EOF, -1 for malformed input/I/O.
 */
static int line(FILE *in, char *buffer, size_t capacity)
{
    size_t used = 0;
    int c;
    while ((c = fgetc(in)) != EOF) {
        if (c == '\n') {
            buffer[used] = '\0';
            return 1;
        }
        if (c < 0x20 || c > 0x7e || used + 1 >= capacity)
            return -1;
        buffer[used++] = (char)c;
    }
    return ferror(in) || used ? -1 : 0;
}

static FILE *regular_input(const char *path, off_t maximum)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat st;
    FILE *in;
    if (fd < 0) return NULL;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size < 0 || st.st_size > maximum) {
        close(fd); return NULL;
    }
    in = fdopen(fd, "r");
    if (!in) close(fd);
    return in;
}

static int decimal_header(const char *text, int32_t values[15])
{
    size_t field;
    const unsigned char *p = (const unsigned char *)text;
    for (field = 0; field < 15; ++field) {
        uint32_t value = 0;
        if (*p < '0' || *p > '9') return 0;
        do {
            uint32_t digit = (uint32_t)(*p++ - '0');
            if (value > ((uint32_t)INT32_MAX - digit) / 10) return 0;
            value = value * 10 + digit;
        } while (*p >= '0' && *p <= '9');
        values[field] = (int32_t)value;
        if (field == 14) return *p == '\0';
        if (*p++ != ' ') return 0;
    }
    return 0;
}

static int hex_word(FILE *in, double *output)
{
    char text[64];
    uint64_t word = 0;
    size_t i;
    if (line(in, text, sizeof text) != 1 || strlen(text) != 16) return 0;
    for (i = 0; i < 16; ++i) {
        unsigned int digit;
        unsigned char c = (unsigned char)text[i];
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return 0;
        word = (word << 4) | digit;
    }
    memcpy(output, &word, sizeof word);
    return 1;
}

static int parse_request(struct request *r)
{
    FILE *in = regular_input(r->path, MAX_INPUT_BYTES);
    char text[256];
    int32_t h[15], i;
    int ok = 0;
    if (!in) return 0;
    if (line(in, text, sizeof text) != 1 || strcmp(text, "MLTCAP_INPUT_V1") != 0)
        goto done;
    if (line(in, text, sizeof text) != 1 || !decimal_header(text, h)) goto done;
    r->case_id = h[0]; r->phase = h[1]; r->point = h[2];
    r->method = h[3]; r->group = h[4]; r->n = h[5]; r->rank = h[6];
    r->plen = h[7]; r->slen = h[8]; r->mlen = h[9]; r->olen = h[10];
    r->poles = h[11]; r->finite = h[12]; r->check = h[13]; r->cache = h[14];
    if (r->case_id != 1 || r->phase != 1 || r->method < 1 || r->method > 3 ||
        r->poles != TIRUST_POLES_BOTH || r->finite != 1 ||
        r->cache != 0 || r->check != (r->method == 2)) goto done;
    if (r->case_id == 1) {
        if (r->point != 25 || (r->group != 59 && r->group != 61) ||
            r->n != 5 || r->rank != 4 || r->olen != 4 * 70) goto done;
    } else if (r->case_id == 2) {
        if (r->point != 2 || (r->group != 408 && r->group != 422) ||
            r->n != 6 || r->rank != 5 || r->olen != 4 * 126) goto done;
    } else goto done;
    if (r->plen != 4 * r->n || r->slen != r->n * r->n ||
        r->mlen != r->n || r->olen > MAX_OUTPUT) goto done;
    if (!hex_word(in, &r->mu2)) goto done;
    for (i = 0; i < r->plen; ++i) if (!hex_word(in, r->p + i)) goto done;
    for (i = 0; i < r->slen; ++i) if (!hex_word(in, r->s + i)) goto done;
    for (i = 0; i < r->mlen; ++i) {
        if (!hex_word(in, &r->masses[i].re) || !hex_word(in, &r->masses[i].im))
            goto done;
    }
    if (line(in, text, sizeof text) != 0) goto done;
    ok = 1;
done:
    if (fclose(in) != 0) ok = 0;
    return ok;
}

static int same_key(const struct request *a, const struct request *b)
{
    return a->case_id == b->case_id && a->phase == b->phase &&
        a->point == b->point && a->method == b->method && a->group == b->group;
}

static int manifest(const char *path, struct request requests[MAX_REQUESTS],
                    size_t *count)
{
    FILE *in = regular_input(path, 65536);
    char text[MAX_PATH_BYTES + 1];
    int state;
    size_t i;
    *count = 0;
    if (!in) return 0;
    while ((state = line(in, text, sizeof text)) == 1) {
        size_t length = strlen(text);
        if (strspn(text, " ") == length) continue;
        if (text[0] != '/' || *count == MAX_REQUESTS) goto bad;
        memcpy(requests[*count].path, text, length + 1);
        if (!parse_request(requests + *count)) goto bad;
        for (i = 0; i < *count; ++i)
            if (same_key(requests + i, requests + *count) ||
                strcmp(requests[i].path, text) == 0) goto bad;
        ++*count;
    }
    {
        int closed = fclose(in);
        if (state < 0 || closed != 0) return 0;
    }
    if (*count != 6) return 0;
    return 1;
bad:
    fclose(in);
    return 0;
}

static int replay(FILE *out, const struct request *r, unsigned int *evaluations)
{
    ml_tirust_context *context = NULL;
    ml_tirust_complex output[MAX_OUTPUT];
    struct bridge_error error;
    int32_t status;
    int repetition, io_ok = 1, setup_ok = 1;

    status = ml_tirust_context_create(TIRUST_ABI_VERSION, &context);
    copy_error(&error);
    io_ok = record(out, r, "create", 1, status, &error, NULL,
                   context != NULL, *evaluations);
    if (status != TIRUST_SUCCESS || context == NULL) return 0;
    if (!io_ok) goto destroy;
    status = ml_tirust_register_plan(context, r->n, r->rank);
    copy_error(&error);
    io_ok = record(out, r, "register", 1, status, &error, NULL, 1, *evaluations);
    if (status != TIRUST_SUCCESS) setup_ok = 0;
    if (!io_ok || !setup_ok) goto destroy;
    for (repetition = 1; repetition <= 2; ++repetition) {
        if (*evaluations >= MAX_EVALUATIONS) { io_ok = 0; break; }
        ++*evaluations;
        status = ml_tirust_evaluate_dp(context, r->n, r->rank,
            r->p, (size_t)r->plen, r->s, (size_t)r->slen,
            r->masses, (size_t)r->mlen, r->mu2,
            output, (size_t)r->olen, r->poles, r->finite, r->check, r->cache);
        copy_error(&error);
        io_ok = record(out, r, "evaluate", repetition, status, &error,
                       output, 1, *evaluations);
        if (!io_ok) break;
    }
destroy:
    status = ml_tirust_context_destroy(context);
    copy_error(&error);
    if (status != TIRUST_SUCCESS) setup_ok = 0;
    if (!record(out, r, "destroy", 2, status, &error, NULL,
                status == TIRUST_SUCCESS ? 0 : -1, *evaluations))
        io_ok = 0;
    return io_ok && setup_ok;
}

int main(int argc, char **argv)
{
    struct request requests[MAX_REQUESTS];
    size_t count, i;
    unsigned int evaluations = 0;
    int fd, ok = 1;
    FILE *out;
    if (argc != 2) {
        fputs("usage: capture-replay REQUEST_MANIFEST\n", stderr); return 2;
    }
    /* Validate the entire allowlisted batch before any bridge invocation.
     * Exactly the six captured Zgg requests are required; missing ZZgg
     * inputs remain unavailable and are never reconstructed here.
     */
    if (!manifest(argv[1], requests, &count)) {
        fputs("invalid replay manifest or captured input\n", stderr); return 2;
    }
    fd = open("tirust-replay.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { perror("tirust-replay.jsonl"); return 2; }
    out = fdopen(fd, "w");
    if (!out) { close(fd); return 2; }
    for (i = 0; i < count; ++i) {
        if (!replay(out, requests + i, &evaluations)) ok = 0;
        /* Do not start more numerical work once its journal cannot be kept. */
        if (ferror(out)) break;
    }
    if (fclose(out) != 0) ok = 0;
    return ok ? 0 : 1;
}
