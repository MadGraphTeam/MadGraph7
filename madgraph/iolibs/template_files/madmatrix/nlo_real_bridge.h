/* Exception-safe C ABI between generated FKS Fortran and MadMatrix UMAMI. */
#ifndef MG7_NLO_REAL_BRIDGE_H
#define MG7_NLO_REAL_BRIDGE_H 1

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mg7_nlo_real_initialize(void **context, const char *param_card,
                            const char *library_dir, const char *backend);
int mg7_nlo_real_evaluate(void *context, int real_me_id, size_t event_count,
                          const double *momenta, const double *g_strong,
                          const int32_t *flavour, double *squared_orders);
int mg7_nlo_real_finalize(void **context);
const char *mg7_nlo_real_last_error(void *context);

#ifdef __cplusplus
}
#endif
#endif
