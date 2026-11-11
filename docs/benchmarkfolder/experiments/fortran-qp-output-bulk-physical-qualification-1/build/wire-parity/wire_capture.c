/* Observer-only native wire recorder. Never link into scientific/timing lanes.
 * Integer-only metadata/byte output; every native argument/status/result is
 * forwarded unchanged. I/O errors are recorded separately, never substituted.
 */
#define _GNU_SOURCE
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

static FILE *wire;
static int enabled, failure;
static int64_t calls, records;

extern int __real_ml_tirust_evaluate_mp(void *, int, int, const char *, size_t,
    const char *, size_t, const char *, size_t, const char *, size_t,
    char *, size_t, int, int, int, int);

void qp_wire_begin_(void) {
    if (enabled || wire != NULL || calls != 0 || records != 0) { failure = 1; return; }
    wire = fopen("real-wire.hex", "wx");
    if (wire == NULL) { failure = 2; return; }
    enabled = 1;
    if (fputs("# real-tirust-qp-wire-v1 width64\n", wire) == EOF) failure = 3;
    if (fprintf(wire,"ENVCONST %d %d\n",FE_TONEAREST,
        FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INEXACT)<0) failure = 3;
}

int __wrap_ml_tirust_evaluate_mp(void *context, int n, int rank,
    const char *p, size_t plen, const char *s, size_t slen,
    const char *m, size_t mlen, const char *scale, size_t scale_len,
    char *output, size_t output_len, int poles, int finite, int check, int cache) {
    static const char digits[] = "0123456789abcdef";
    int status = __real_ml_tirust_evaluate_mp(context,n,rank,p,plen,s,slen,
        m,mlen,scale,scale_len,output,output_len,poles,finite,check,cache);
    size_t k, j;
    int rounding, halting, sticky;
    if (!enabled || wire == NULL) return status;
    rounding = fegetround();
    halting = fegetexcept();
    sticky = fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INEXACT);
    ++calls;
    if (fprintf(wire,"CALL %lld %d %d %d %d %d %d %d %zu %d %d %d\n",
        (long long)calls,n,rank,poles,finite,check,cache,status,output_len,
        rounding,halting,sticky) < 0) failure = 4;
    if (status == 0) {
        if (output == NULL || output_len % 64 != 0) failure = 5;
        else {
            for (k=0; k<output_len/64; ++k) {
                if (fprintf(wire,"RECORD %zu ",k+1) < 0) failure = 6;
                for (j=0; j<64; ++j) {
                    unsigned int byte = (unsigned char)output[64*k+j];
                    if (fputc(digits[byte>>4],wire)==EOF || fputc(digits[byte&15],wire)==EOF) failure = 7;
                }
                if (fputc('\n',wire)==EOF) failure = 8;
                ++records;
            }
        }
    }
    if (fprintf(wire,"END %lld\n",(long long)calls)<0 || fflush(wire)!=0) failure = 9;
    return status;
}

void qp_wire_end_(void) {
    enabled = 0;
    if (wire != NULL) {
        if (fclose(wire)!=0) failure = 10;
        wire = NULL;
    }
}

void qp_wire_health_(int *status, int64_t *native_calls, int64_t *native_records) {
    if (status != NULL) *status = failure;
    if (native_calls != NULL) *native_calls = calls;
    if (native_records != NULL) *native_records = records;
}
