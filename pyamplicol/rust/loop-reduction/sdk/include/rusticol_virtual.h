/* SPDX-License-Identifier: 0BSD */
#ifndef RUSTICOL_VIRTUAL_H
#define RUSTICOL_VIRTUAL_H
/* Linux glibc floor of the shared library (README, "glibc floor (Linux)").
 * librusticol_loop_sdk.so of a release native build needs the glibc symbol
 * versions it links, at most GLIBC_2.38 and no glibc ABI marker
 * (`readelf -V` lists a build's own requirements), so the floor is glibc
 * RUSTICOL_VIRTUAL_GLIBC_FLOOR_MAJOR.RUSTICOL_VIRTUAL_GLIBC_FLOOR_MINOR (2.38).
 * Linking a client executable with a toolchain on an older glibc fails at
 * link time ("librusticol_loop_sdk.so: undefined reference to
 * `__isoc23_strtol@GLIBC_2.38'"); a client built on glibc 2.38 or newer and
 * run on an older one is stopped by the dynamic loader before main
 * ("libc.so.6: version `GLIBC_2.38' not found (required by
 * .../librusticol_loop_sdk.so)"). Define RUSTICOL_VIRTUAL_SHARED when
 * compiling a client of the release build's shared library to stop at compile
 * time instead, on a toolchain whose glibc is older (every glibc release
 * defines the numbered versions of the releases before it). The macro states
 * the release build's floor: omit it for an SDK built from source on an older
 * host. */
#define RUSTICOL_VIRTUAL_GLIBC_FLOOR_MAJOR 2
#define RUSTICOL_VIRTUAL_GLIBC_FLOOR_MINOR 38
#if defined(RUSTICOL_VIRTUAL_SHARED) && defined(__linux__)
#include <features.h>
#if !defined(__GLIBC__)
#error "librusticol_loop_sdk.so needs glibc >= 2.38 but this toolchain's C library is not glibc (the SDK's release libraries are built for glibc)"
#elif __GLIBC__ < RUSTICOL_VIRTUAL_GLIBC_FLOOR_MAJOR || \
    (__GLIBC__ == RUSTICOL_VIRTUAL_GLIBC_FLOOR_MAJOR && \
     __GLIBC_MINOR__ < RUSTICOL_VIRTUAL_GLIBC_FLOOR_MINOR)
#error "librusticol_loop_sdk.so needs glibc >= 2.38 but this toolchain's glibc is older; build and run the client with glibc 2.38 or newer (RUSTICOL_VIRTUAL_SHARED states the release build's floor: omit it for an SDK built from source on this host)"
#endif
#endif
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Additive source-only ABI version 1; independent of the tree scalar handles.
 * Handles are exclusive, non-concurrently callable. Results own their buffers
 * and remain usable after the originating runtime is cleared or destroyed.
 * All buffer capacities count elements (JSON counts bytes including NUL).
 * Nonempty pointers must be aligned and valid for the documented counts.
 * Output buffers must not alias handles, requests, or one another.
 */
typedef struct VirtualHandle rusticol_virtual_handle;
typedef struct VirtualResult rusticol_virtual_result;
enum rusticol_virtual_function_status {
  RUSTICOL_VIRTUAL_OK = 0, RUSTICOL_VIRTUAL_INVALID_ARGUMENT = 1,
  RUSTICOL_VIRTUAL_BUFFER_TOO_SMALL = 2, RUSTICOL_VIRTUAL_RUNTIME_ERROR = 3,
  RUSTICOL_VIRTUAL_PANIC = 4
};
enum rusticol_virtual_point_status {
  RUSTICOL_VIRTUAL_SUCCESS = 0, /* initial success, accuracy not established */
  RUSTICOL_VIRTUAL_VERIFIED = 1, /* explicit requested-bound agreement */
  RUSTICOL_VIRTUAL_RESCUED = 2,
  RUSTICOL_VIRTUAL_FAILED = 3,
  RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED = 4
};
typedef struct {
  size_t external_count, born_orders, virtual_orders, helicities;
  size_t born_color_components, virtual_color_components, epsilon_count;
  uint32_t resolved_available;
} rusticol_virtual_shape_info;
typedef struct {
  const double *momenta; /* [point,external_leg,(E,px,py,pz)] */
  size_t momentum_count, point_count;
  const uint64_t *point_ids; /* NULL chooses 0..point_count; otherwise unique */
  const double *mu_r2; /* explicit positive real scale squared, never loop mu² */
  size_t scale_count; /* one or point_count; zero allowed for an empty batch */
  uint32_t resolved; /* 0 inclusive, 1 helicity-resolved */
  const char *r2_strategy; /* NULL inherits open; auto/model, direct unavailable */
  const char *verification_json; /* NULL: no hidden verification or rescue */
} rusticol_virtual_batch_request;
typedef struct {
  size_t requested_points, successful_points, born_values, virtual_values;
  uint32_t resolved;
  rusticol_virtual_shape_info axes;
} rusticol_virtual_result_shape_info;

uint32_t rusticol_virtual_abi_version(void);
/* Borrowed thread-local string; valid until the next failing call on this thread. */
const char *rusticol_virtual_last_error_message(void);
/* Optional parameters JSON maps names to exact [real-string,imag-string] pairs.
 * Example: {"aS":["0.118","0"]}. NULL retains source defaults.
 * process may be NULL only when ordinary artifact selection is unambiguous.
 */
int rusticol_virtual_open(const char *path, const char *process,
  const char *parameters_json, const char *r2_strategy,
  rusticol_virtual_handle **output);
/* Optional retained verification policy uses the batch-request JSON schema.
 * It is configured before exact source realization. Per-call policy overrides
 * this default; NULL per-call policy inherits it, including after clear.
 */
int rusticol_virtual_open_with_policy(const char *path, const char *process,
  const char *parameters_json, const char *r2_strategy,
  const char *verification_json, rusticol_virtual_handle **output);
int rusticol_virtual_free(rusticol_virtual_handle *handle); /* NULL is accepted */
int rusticol_virtual_clear(rusticol_virtual_handle *handle);
/* Plan v4: the whole mapping is validated first, as the Python API's
 * set_model_parameters; an empty name, a value that is not a finite decimal
 * (ASCII text, as Python's Decimal reads it), an unknown name, the
 * renormalisation scale's parameter (set per point by mu_r2) or a
 * parameter fixed at generation at another value returns
 * RUSTICOL_VIRTUAL_INVALID_ARGUMENT, a nonzero imaginary part
 * RUSTICOL_VIRTUAL_RUNTIME_ERROR (the Python CompatibilityError), and changes
 * nothing (the same applies to parameters_json at open). */
int rusticol_virtual_update_parameters(rusticol_virtual_handle *handle,
  const char *changes_json, uint64_t *source_revision);
/* Pole selection of later evaluations (REQUIREMENTS_ADDENDUM I): "all" (NULL,
 * the default), "none" (plan v4: the finite part alone, its pole-only work
 * skipped; the single and double pole entries are quiet NaNs), "uv" (plan v4:
 * the poles are the bare loop's UV poles) or "ir" (plan v4: the poles are the
 * renormalised virtual's IR poles). The split selections run the stored
 * companion of the certified-scaleless families; the finite part is the same
 * in every selection. Plan v1-v3 artifacts accept "all" only.
 */
int rusticol_virtual_set_poles(rusticol_virtual_handle *handle, const char *poles);
int rusticol_virtual_shape(const rusticol_virtual_handle *handle,
  rusticol_virtual_shape_info *output);
/* JSON size query: output=NULL, capacity=0; needed includes terminal NUL.
 * Metadata contains schemes, squared orders and contributing Born/loop pairs.
 */
int rusticol_virtual_metadata_json(const rusticol_virtual_handle *handle,
  char *output, size_t capacity, size_t *needed);
int rusticol_virtual_capabilities_json(const rusticol_virtual_handle *handle,
  char *output, size_t capacity, size_t *needed);
/* Returns OK for a completed batch even if some points failed. Inspect statuses.
 * Numerical rows exist ONLY for successful_point_ids in their original order.
 * Every finite binary64 input is preserved exactly in higher-carrier retries.
 * Optional verification JSON:
 * {"higher":[{"arithmetic":"double-double","precision":null},
 *            {"arithmetic":"arbitrary","precision":60}],
 *  "relative_tolerance":"1e-8","absolute_tolerance":"1e-20",
 *  "verify_successful":true}
 * Actual precision must strictly increase; numerical agreement is not an
 * independent physics certificate. Global source binding errors precede rescue.
 */
int rusticol_virtual_evaluate(rusticol_virtual_handle *handle,
  const rusticol_virtual_batch_request *request, rusticol_virtual_result **output);
int rusticol_virtual_result_free(rusticol_virtual_result *result);
int rusticol_virtual_result_shape(const rusticol_virtual_result *result,
  rusticol_virtual_result_shape_info *output);
/* Inclusive layout: [successful_point,order,(epsilon)]. Resolved layout:
 * [successful_point,order,helicity,colour_component,(epsilon)].
 * Epsilon order is finite,single_pole,double_pole = (0,-1,-2).
 * Resolved LC colour entries are signed Born-anchored interference components.
 * Inclusive projected identity correlations and NLC/full use one contracted
 * contribution. Query shape and metadata for all axes.
 * Resolved helicities retain the inclusive incoming spin average; their sum
 * reconstructs inclusive results. Physical couplings are included.
 * Short buffers return BUFFER_TOO_SMALL without writing any element.
 */
int rusticol_virtual_result_copy_born(const rusticol_virtual_result *, double *, size_t);
int rusticol_virtual_result_copy_virtual(const rusticol_virtual_result *, double *, size_t);
int rusticol_virtual_result_copy_requested_ids(const rusticol_virtual_result *, uint64_t *, size_t);
int rusticol_virtual_result_copy_successful_ids(const rusticol_virtual_result *, uint64_t *, size_t);
int rusticol_virtual_result_copy_statuses(const rusticol_virtual_result *, uint32_t *, size_t);
int rusticol_virtual_result_diagnostics_json(const rusticol_virtual_result *, char *, size_t, size_t *);
/* Additive grouped complex interface. Ordinary v1 structures and functions are
 * unchanged. Every saved history is evaluated once in the recorded ID order;
 * request.resolved must be zero. Histories are never summed implicitly.
 * Shape counts below count complex pairs, not individual real components.
 */
typedef struct CorrelatedResult rusticol_virtual_correlated_result;
typedef struct { double real, imag; } rusticol_virtual_complex64;
/* Literal sources: values[row][selected_leg][component], four complex Lorentz
 * components. One row broadcasts, or supply point_count rows. Labels are
 * one-based physical legs and form one saved joint class. No normalization or
 * conjugation is applied. All storage is borrowed only during the call. */
typedef struct {
  const size_t *legs;
  size_t leg_count;
  const rusticol_virtual_complex64 *values;
  size_t value_count;
} rusticol_virtual_spin_sources;
int rusticol_virtual_spin_correlation_classes_json(const rusticol_virtual_handle *, char *, size_t, size_t *);

typedef struct {
  size_t histories, born_orders, virtual_orders, epsilon_count;
} rusticol_virtual_correlated_shape_info;
typedef struct {
  size_t requested_points, successful_points, born_values, virtual_values;
  rusticol_virtual_correlated_shape_info axes;
} rusticol_virtual_correlated_result_shape_info;
int rusticol_virtual_correlated_shape(const rusticol_virtual_handle *, rusticol_virtual_correlated_shape_info *);
/* Ordered history IDs only; no second colour-operation label convention. */
int rusticol_virtual_correlation_histories_json(const rusticol_virtual_handle *, char *, size_t, size_t *);
int rusticol_virtual_evaluate_correlated(rusticol_virtual_handle *, const rusticol_virtual_batch_request *, rusticol_virtual_correlated_result **);
int rusticol_virtual_evaluate_correlated_with_sources(rusticol_virtual_handle *, const rusticol_virtual_batch_request *, const rusticol_virtual_spin_sources *, rusticol_virtual_correlated_result **);
int rusticol_virtual_correlated_result_free(rusticol_virtual_correlated_result *);
int rusticol_virtual_correlated_result_shape(const rusticol_virtual_correlated_result *, rusticol_virtual_correlated_result_shape_info *);
/* Born [successful_point,history,born_order]; loop
 * [successful_point,history,virtual_order,epsilon] with epsilon=(0,-1,-2).
 * Both real and imaginary components are physical signed interference outputs.
 * Failure/status/ownership/short-buffer rules are identical to ordinary calls.
 */
int rusticol_virtual_correlated_result_copy_born(const rusticol_virtual_correlated_result *, rusticol_virtual_complex64 *, size_t);
int rusticol_virtual_correlated_result_copy_virtual(const rusticol_virtual_correlated_result *, rusticol_virtual_complex64 *, size_t);
int rusticol_virtual_correlated_result_copy_requested_ids(const rusticol_virtual_correlated_result *, uint64_t *, size_t);
int rusticol_virtual_correlated_result_copy_successful_ids(const rusticol_virtual_correlated_result *, uint64_t *, size_t);
int rusticol_virtual_correlated_result_copy_statuses(const rusticol_virtual_correlated_result *, uint32_t *, size_t);
int rusticol_virtual_correlated_result_diagnostics_json(const rusticol_virtual_correlated_result *, char *, size_t, size_t *);
int rusticol_virtual_correlated_result_histories_json(const rusticol_virtual_correlated_result *, char *, size_t, size_t *);
int rusticol_virtual_correlated_result_spin_legs_json(const rusticol_virtual_correlated_result *, char *, size_t, size_t *);
#ifdef __cplusplus
}
#endif
#endif
