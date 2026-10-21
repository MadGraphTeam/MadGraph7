/* Public-ABI replay of exact stored tensor inputs on an installed archive.
 * The Python adapter supplies bounded records; this program does no reduction
 * mathematics and leaves all numerical policy to the unmodified bridge.
 */
#define _GNU_SOURCE
#include "ninjars.h"
#include <link.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RECORD 64
#define MAX_N 16
#define MAX_COEFFICIENTS 100000

static void fail(const char *message) {
    fprintf(stderr, "precision driver: %s\n", message);
    exit(2);
}

static void *allocate(size_t count, size_t width) {
    void *result = calloc(count, width);
    if (!result) fail("allocation failed");
    return result;
}

static void doubles(double *values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (scanf("%lf", &values[i]) != 1) fail("incomplete double input");
}

static void complexes(ninjars_complex *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        doubles(&values[i].re, 1);
        doubles(&values[i].im, 1);
    }
}

static unsigned char *records(size_t count) {
    unsigned char *result = allocate(count, RECORD);
    for (size_t i = 0; i < count; ++i) {
        char text[RECORD];
        if (scanf("%63s", text) != 1) fail("incomplete ASCII input");
        size_t width = strlen(text);
        memset(result+i*RECORD, ' ', RECORD);
        memcpy(result+i*RECORD, text, width);
    }
    return result;
}

static void ascii_result(const unsigned char *value) {
    size_t width = RECORD;
    while (width && value[width-1] == ' ') --width;
    putchar('"');
    for (size_t i = 0; i < width; ++i) {
        unsigned char c = value[i];
        if (c < 32 || c >= 127 || c == '"' || c == '\\')
            fail("non-numeric ASCII output");
        putchar(c);
    }
    putchar('"');
}

static int loaded_object(struct dl_phdr_info *info, size_t size, void *unused) {
    (void)size;
    (void)unused;
    if (info->dlpi_name[0]) fprintf(stderr,"NINJARS_LOADED %s\n",info->dlpi_name);
    return 0;
}

int main(void) {
    ninjars_context *context = NULL;
    if (ninjars_abi_version() != NINJARS_ABI_VERSION ||
        ninjars_context_create(NINJARS_ABI_VERSION, &context))
        fail("cannot create matching ABI context");
    int mode, n, rank;
    size_t count, index = 0;
    while (scanf("%d %d %d %zu", &mode, &n, &rank, &count) == 4) {
        if (mode < 0 || mode > 2 || n < 1 || n > MAX_N || rank < 0 ||
            rank > n+2 || count == 0 || count > MAX_COEFFICIENTS)
            fail("invalid request dimensions");
        int status;
        if (mode < 2) {
            ninjars_complex *c = allocate(count, sizeof(*c));
            double *s = allocate(n*n, sizeof(*s));
            double *p = allocate(4*n, sizeof(*p));
            ninjars_complex *m = allocate(n, sizeof(*m));
            double scale;
            ninjars_complex result[3] = {{NAN,NAN},{NAN,NAN},{NAN,NAN}};
            ninjars_complex rational = {NAN,NAN};
            complexes(c, count);
            doubles(s, n*n);
            doubles(p, 4*n);
            complexes(m, n);
            doubles(&scale, 1);
            status = (mode == 0 ? ninjars_evaluate_dp : ninjars_evaluate_dp_compensated)(
                context,n,rank,count,c,s,n*n,p,4*n,m,n,scale,result,3,&rational,1);
            printf("NINJARS_INSTALLED {\"record\":%zu,\"mode\":%d,\"status\":%d,\"output\":[",index,mode,status);
            for (size_t i = 0; i < 8; ++i) {
                ninjars_complex z = i < 6 ? result[i/2] : rational;
                double x = i % 2 ? z.im : z.re;
                if (i) putchar(',');
                if (isfinite(x)) printf("%.17e",x); else printf("null");
            }
            putchar(']');
            free(c); free(s); free(p); free(m);
        } else {
            unsigned char *c = records(2*count), *s = records(n*n), *p = records(4*n);
            unsigned char *m = records(2*n), *scale = records(1);
            unsigned char result[8*RECORD];
            memset(result,'!',sizeof(result));
            status = ninjars_evaluate_mp(context,n,rank,count,c,2*count*RECORD,
                s,n*n*RECORD,p,4*n*RECORD,m,2*n*RECORD,scale,RECORD,result,sizeof(result));
            printf("NINJARS_INSTALLED {\"record\":%zu,\"mode\":%d,\"status\":%d,\"output\":[",index,mode,status);
            for (size_t i = 0; i < 8; ++i) {
                if (i) putchar(',');
                ascii_result(result+i*RECORD);
            }
            putchar(']');
            free(c); free(s); free(p); free(m); free(scale);
        }
        puts("}");
        fflush(stdout);
        ++index;
        if (status) {
            unsigned char message[2048];
            ninjars_last_error(message,sizeof(message));
            fprintf(stderr,"request %zu returned status %d: %s\n",index-1,status,message);
        }
    }
    if (!feof(stdin)) fail("malformed request header");
    dl_iterate_phdr(loaded_object,NULL);
    if (ninjars_context_destroy(context)) fail("context destruction failed");
    return 0;
}
