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
int mg7_nlo_real_available(const void *context, int real_me_id);
int mg7_nlo_real_evaluate(void *context, int real_me_id, size_t event_count,
                          const double *momenta, const double *g_strong,
                          const int32_t *flavour, double *squared_orders);
/* Physical components for the caller's G and e, reconstructed from unit
 * coupling blocks and the local order powers published by the provider. */
int mg7_nlo_real_evaluate_ew(void *context, int real_me_id, size_t event_count,
                             const double *momenta, const double *g_strong,
                             double g_ew, const int32_t *flavour, double *squared_orders);
/* Optional validation callbacks observe the actual packed inputs at dispatch
 * and the physical components before they are mapped back into FKS slots. */
typedef void (*mg7_trace_input_fn)(size_t, int, int, const double *, const double *, double, const int32_t *);
typedef void (*mg7_trace_output_fn)(size_t, int, const double *);
void mg7_nlo_real_set_trace(void *context, mg7_trace_input_fn input, mg7_trace_output_fn output);
int mg7_nlo_real_finalize(void **context);
const char *mg7_nlo_real_last_error(void *context);

#ifdef __cplusplus
}
#endif
#endif
