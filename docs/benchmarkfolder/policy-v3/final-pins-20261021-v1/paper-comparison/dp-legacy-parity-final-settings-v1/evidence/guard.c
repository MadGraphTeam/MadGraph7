/* Diagnostic link guard only: no replacement numerical implementation. */
#include "ninjars.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static unsigned long counts[6]; /* create,dp,compensated,mp,destroy,nonnull */
static void report(const char *event) {
    const char *name=getenv("NINJARS_GUARD_LOG");
    FILE *file=name ? fopen(name,"a") : NULL;
    if (!file) _Exit(90);
    fprintf(file,"%s create=%lu dp=%lu compensated=%lu mp=%lu destroy=%lu nonnull=%lu\n",
            event,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5]);
    if (fclose(file)) _Exit(91);
}
static int32_t forbidden(int index) {
    counts[index]++;
    report("FORBIDDEN_NUMERICAL_ENTRY");
    _Exit(92);
}
int32_t __wrap_ninjars_context_create(uint32_t version,ninjars_context **output) {
    return forbidden(0);
}
#define DP_ARGUMENTS ninjars_context *context,int32_t n,int32_t rank,size_t count, \
    const ninjars_complex *coefficients,const double *s_matrix,size_t s_length, \
    const double *momenta,size_t p_length,const ninjars_complex *masses,size_t m_length, \
    double scale_squared,ninjars_complex *laurent,size_t laurent_length, \
    ninjars_complex *rational,size_t rational_length
int32_t __wrap_ninjars_evaluate_dp(DP_ARGUMENTS) {return forbidden(1);}
int32_t __wrap_ninjars_evaluate_dp_compensated(DP_ARGUMENTS) {return forbidden(2);}
int32_t __wrap_ninjars_evaluate_mp(ninjars_context *context,int32_t n,int32_t rank,size_t count,
    const unsigned char *coefficients,size_t coefficient_bytes,
    const unsigned char *s_matrix,size_t s_bytes,const unsigned char *momenta,size_t p_bytes,
    const unsigned char *masses,size_t m_bytes,const unsigned char *scale_squared,size_t scale_bytes,
    unsigned char *output,size_t output_bytes) {return forbidden(3);}
extern int32_t __real_ninjars_context_destroy(ninjars_context *context);
int32_t __wrap_ninjars_context_destroy(ninjars_context *context) {
    counts[4]++;
    if (context) {counts[5]++;report("FORBIDDEN_NONNULL_TEARDOWN");_Exit(93);}
    int32_t status=__real_ninjars_context_destroy(context);
    if (status!=NINJARS_SUCCESS) {report("FAILED_NULL_TEARDOWN");_Exit(94);}
    report("NULL_TEARDOWN_FORWARDED");
    return status;
}
__attribute__((destructor)) static void finished(void) {report("GUARD_FINAL");}
