#include "ninjars.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double number(void) {
    double value;
    if (scanf("%lf", &value) != 1) exit(11);
    return value;
}
static void *allocate(size_t bytes) {
    void *memory = malloc(bytes);
    if (!memory) exit(12);
    return memory;
}
static unsigned char *records(const double *values, size_t count) {
    unsigned char *output = allocate(64 * count);
    memset(output, ' ', 64 * count);
    for (size_t i = 0; i < count; ++i) {
        char text[64];
        int length = snprintf(text, sizeof(text), "%.39e", values[i]);
        if (length < 0 || length >= 64) exit(13);
        memcpy(output + 64*i, text, length);
    }
    return output;
}
static void check(int status) {
    if (status) {
        unsigned char error[4096];
        ninjars_last_error(error, sizeof(error));
        fprintf(stderr, "bridge status %d: %s\n", status, error);
        exit(14);
    }
}
int main(void) {
    ninjars_context *context = NULL;
    check(ninjars_context_create(NINJARS_ABI_VERSION, &context));
    int cases;
    if (scanf("%d", &cases) != 1 || cases < 1) return 10;
    for (int c = 0; c < cases; ++c) {
        int loop, direction, n, rank;
        size_t count;
        if (scanf("%d%d%d%d%zu", &loop, &direction, &n, &rank, &count) != 5)
            return 10;
        double *coefficients = allocate(sizeof(double)*2*count);
        double *s = allocate(sizeof(double)*n*n);
        double *p = allocate(sizeof(double)*4*n);
        double *m = allocate(sizeof(double)*2*n);
        for (size_t i = 0; i < 2*count; ++i) coefficients[i] = number();
        for (int i = 0; i < n*n; ++i) s[i] = number();
        for (int i = 0; i < 4*n; ++i) p[i] = number();
        for (int i = 0; i < 2*n; ++i) m[i] = number();
        double scale = number();
        ninjars_complex result[3], rational;
        int status = ninjars_evaluate_dp(context, n, rank, count,
            (ninjars_complex *)coefficients, s, n*n, p, 4*n,
            (ninjars_complex *)m, n, scale, result, 3, &rational, 1);
        check(status);
        printf("DP %d %d %d", loop, direction, status);
        for (int i = 0; i < 3; ++i)
            printf(" %.17e %.17e", result[i].re, result[i].im);
        printf(" %.17e %.17e\n", rational.re, rational.im);
        unsigned char *ct = records(coefficients, 2*count);
        unsigned char *st = records(s, n*n);
        unsigned char *pt = records(p, 4*n);
        unsigned char *mt = records(m, 2*n);
        unsigned char *scale_record = records(&scale, 1);
        unsigned char output[512];
        memset(output, '!', sizeof(output));
        status = ninjars_evaluate_mp(context, n, rank, count,
            ct, 128*count, st, 64*n*n, pt, 256*n, mt, 128*n,
            scale_record, 64, output, sizeof(output));
        check(status);
        printf("MP %d %d %d", loop, direction, status);
        for (int i = 0; i < 8; ++i) printf(" %.64s", output + 64*i);
        printf("\n");
        fflush(stdout);
        free(coefficients); free(s); free(p); free(m);
        free(ct); free(st); free(pt); free(mt); free(scale_record);
    }
    check(ninjars_context_destroy(context));
    return 0;
}
