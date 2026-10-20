/* Untimed diagnostic linker wrapper; no change to either numerical library. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

extern int32_t __real_ninjars_evaluate_mp(
    void *, int32_t, int32_t, size_t,
    const unsigned char *, size_t, const unsigned char *, size_t,
    const unsigned char *, size_t, const unsigned char *, size_t,
    const unsigned char *, size_t, unsigned char *, size_t);

static void records(FILE *stream, const char *name,
                    const unsigned char *values, size_t bytes) {
    if (bytes % 64) abort();
    fprintf(stream, ",\"%s\":[", name);
    for (size_t offset = 0; offset < bytes; offset += 64) {
        size_t end = 64, begin = 0;
        while (end && values[offset + end - 1] == ' ') --end;
        while (begin < end && values[offset + begin] == ' ') ++begin;
        if (offset) fputc(',', stream);
        fputc('"', stream);
        for (size_t i = begin; i < end; ++i) {
            unsigned char c = values[offset + i];
            if (!((c >= '0' && c <= '9') || c == '+' || c == '-' ||
                  c == '.' || c == 'E' || c == 'e')) abort();
            fputc(c, stream);
        }
        fputc('"', stream);
    }
    fputc(']', stream);
}

int32_t __wrap_ninjars_evaluate_mp(
    void *context, int32_t n, int32_t rank, size_t count,
    const unsigned char *coefficients, size_t coefficient_bytes,
    const unsigned char *s_matrix, size_t s_bytes,
    const unsigned char *momenta, size_t p_bytes,
    const unsigned char *masses, size_t m_bytes,
    const unsigned char *scale, size_t scale_bytes,
    unsigned char *output, size_t output_bytes) {
    static FILE *stream;
    static int initialized;
    if (!initialized) {
        const char *path = getenv("NINJARS_TENSOR_CAPTURE");
        if (!path || !(stream = fopen(path, "wx"))) abort();
        initialized = 1;
    }
    int32_t status = __real_ninjars_evaluate_mp(
        context, n, rank, count, coefficients, coefficient_bytes,
        s_matrix, s_bytes, momenta, p_bytes, masses, m_bytes,
        scale, scale_bytes, output, output_bytes);
    fprintf(stream, "{\"n\":%d,\"rank\":%d,\"count\":%zu,\"status\":%d",
            n, rank, count, status);
    records(stream, "coefficients", coefficients, coefficient_bytes);
    records(stream, "s_matrix", s_matrix, s_bytes);
    records(stream, "momenta", momenta, p_bytes);
    records(stream, "masses", masses, m_bytes);
    records(stream, "scale", scale, scale_bytes);
    if (status == 0) records(stream, "output", output, output_bytes);
    if (fputs("}\n", stream) < 0 || fflush(stream) || ferror(stream)) abort();
    return status;
}
