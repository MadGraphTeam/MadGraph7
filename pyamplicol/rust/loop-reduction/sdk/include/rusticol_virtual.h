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

/* Source-only ABI version 2 (it dropped version 1's verification policy);
 * independent of the tree scalar handles.
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
  RUSTICOL_VIRTUAL_VERIFIED = 1, /* reserved: plan v4 never returns it */
  RUSTICOL_VIRTUAL_RESCUED = 2, /* reserved: the rescue is the Python API's */
  RUSTICOL_VIRTUAL_FAILED = 3,
  RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED = 4
};
/* Loop-induced processes (metadata_json "contribution_kind": "loop-induced",
 * "interference": "conj(loop)*loop"; |A_loop|^2, no Born) are opened directly,
 * with the same shapes: born_orders = 0 and born_color_components = 0 here,
 * born_values = 0 in every result (copy_born copies nothing); the virtual
 * orders are the loop x loop sectors; each value is |A_loop|^2 per sector and
 * epsilon power, full colour (one contracted colour component). Its poles are
 * computed as a finiteness cross-check: they vanish up to rounding.
 * Its resolved and correlated outputs (resolved_available = 1) are quadratic
 * forms on the point's one amplitude vector (the Python API's, the same Rust
 * code in binary64): request.resolved gives |A(h)|^2/avg per helicity row with
 * one colour-contracted component (the rows sum to the ordinary value); the
 * correlated history "born" is the colour metric itself, i.e. |A|^2 (the
 * identity metric's history, no Born), each declared history X its stored
 * matrix M_X on the amplitude basis (real for a dipole T_i.T_j); a spin source
 * of a declared vector leg enters through its projection on the leg's
 * polarisation vectors (the transverse part). Born counts stay zero there too.
 * The amplitude-basis anchors and the helicity selection are served by
 * rusticol_virtual_evaluate_selected (below). */
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
 * in every selection. Plan v1-v3 artifacts accept "all" only. A loop-induced
 * process serves "all" and "none" only (its UV and IR parts are not additive
 * in |A|^2): "uv" and "ir" return RUSTICOL_VIRTUAL_RUNTIME_ERROR with the
 * Python API's message and change nothing (the capabilities' "pole_selections").
 */
int rusticol_virtual_set_poles(rusticol_virtual_handle *handle, const char *poles);
/* Arithmetic of later evaluations (the Python API's arithmetic=/precision=):
 * "double" (NULL, the default; digits 0), "double-double" (digits 0) or
 * "arbitrary" with digits > 0 decimal digits. The extended carriers run the
 * native executor in their arithmetic and assemble its outputs in Python's
 * Decimal arithmetic at the working digits (max(digits + 20, 40); 52 for
 * double-double), as the Python API: their exact values are decimal literals
 * (rusticol_virtual_result_decimal_json); the double copies are the nearest
 * binary64 values. Ordinary, resolved and correlated evaluations take it.
 */
int rusticol_virtual_set_arithmetic(rusticol_virtual_handle *handle,
  const char *arithmetic, uint32_t digits);
/* Reduction backend of later evaluations (the Python API's reduction=;
 * additive, ABI version 2 kept): "auto" (NULL, the default), "ninjars" or
 * "tirust". "auto" takes the process type's default where it serves the call
 * (one policy, shared with the Python API: TIRust for a loop-induced process,
 * NinjaRS for a Born-interfered one, TIRust on a TIR-only artifact; the reason
 * recorded), else NinjaRS. On TIRust, "auto" has a PER-POINT NinjaRS FALLBACK
 * (where NinjaRS serves the call: not on a TIR-only artifact): a point TIRust
 * refuses or fails (also in the enabled binary64 monitoring's own
 * evaluations, rusticol_virtual_set_stability_checks) is evaluated again on
 * NinjaRS from its original exact inputs, at the call's own arithmetic, after
 * the TIRust call; a point NinjaRS fails too fails with both reasons. In
 * double-double and arbitrary precision every point the fallback serves is
 * guarded (m-lit lit-fallback-guard): its NinjaRS value runs the
 * reduction-direction check (reversed re-test frame, cached masters) in the
 * call's arithmetic, and a point whose relative estimate of the outputs the
 * call returns exceeds the guard's limit (the reduction target, never below
 * 1e-5) is RUSTICOL_VIRTUAL_FAILED with one reason naming the fallback, the
 * arithmetic, the estimate and the limit; a point within it keeps the
 * fallback's value. In binary64 the fallback's value is served unguarded. An
 * explicit "ninjars" or "tirust" has no fallback (its refusals and failures
 * stay failed points). An explicit backend this library is built without
 * (TIRust needs the cargo feature tir; there "auto" takes NinjaRS) is refused
 * here, one that cannot serve a later call's carrier (the library's
 * capability query) by that call; both RUSTICOL_VIRTUAL_RUNTIME_ERROR (the
 * Python API's ReductionUnavailable), never a silent switch. Every result's
 * diagnostics record "reduction": {"requested","effective","reason",
 * "fallback" (null: none),"fallback_points"} and "fallbacks": {point id:
 * {"effective" (null: failed on both, or the guard rejected it),"reason",
 * "guard_estimate" (the guard's estimate; null where the guard did not run:
 * binary64, a point NinjaRS failed too)}} per point the fallback evaluated
 * again (every other point is the effective backend's); the
 * handle's capabilities list what this library serves ("reduction":
 * "native").
 */
int rusticol_virtual_set_reduction(rusticol_virtual_handle *handle,
  const char *reduction);
/* Target accuracy of later evaluations (the Python API's reduction_target=;
 * additive, ABI version 2 kept): the relative accuracy of each point's result
 * (every squared-order sector against its own size), 1e-5 by default.
 * TIRust's method-choice acceptance follows it
 * (calibrated against consistent truths); NinjaRS has no setting it drives.
 * In double-double and arbitrary precision it is also the threshold of the
 * "auto" per-point fallback's guard, never below 1e-5
 * (rusticol_virtual_set_reduction).
 * A value that is not a finite positive number is
 * RUSTICOL_VIRTUAL_INVALID_ARGUMENT and nothing changes. */
int rusticol_virtual_set_reduction_target(rusticol_virtual_handle *handle,
  double target);
/* Unchecked setting of later evaluations (the Python API's
 * reduction_unchecked=; additive, ABI version 2 kept): 1 reduces without the
 * reduction backends' own acceptance and refusal checks (TIRust: its gates and
 * method choice as method selection without the acceptance pass, plain PV at
 * gate 0 as the last resort; NinjaRS: zero stability thresholds), so only rows
 * without a finite value fail; "auto" keeps its backend choice and its
 * per-point fallback. 0, the default, restores the checks. The stability
 * checks stay separate settings. Each result records it in its "reduction"
 * diagnostics. Any other value is RUSTICOL_VIRTUAL_INVALID_ARGUMENT and
 * nothing changes. */
int rusticol_virtual_set_reduction_unchecked(rusticol_virtual_handle *handle,
  int unchecked);
/* Stability checks of later evaluations (the Python API's stability_checks=,
 * VirtualStabilityChecks; additive, ABI version 2 kept): rotation_count
 * default rotations (the Python API's count; any number), or, with
 * rotation_count 0, the rotations of the JSON array of rotation
 * specifications `rotations` ("<x|y|z>:<cos>:<sin>", "<ax>,<ay>,<az>:<radians>",
 * "euler:<alpha>,<beta>,<gamma>"; NULL: none); reduction_direction, rescaled
 * (requires reduction_direction) and cross_backend (with another check) are
 * 0 or 1. All 0 and NULL, the default, runs no check: no estimate and no
 * extra cost. Checks the Python API refuses are RUSTICOL_VIRTUAL_INVALID_ARGUMENT
 * with its message, and nothing changes. A binary64 ordinary evaluation (of a
 * Born-interfered process with "all" or "none" poles, of a loop-induced
 * process) then also evaluates every
 * point at each exactly rotated point, re-reduces every lane in the reduction
 * check's frame and evaluates every point on the other reduction backend,
 * and returns each successful point's estimate in its diagnostics:
 * "stability": {"estimates": [one record per successful point, in order]},
 * each the Python API's VirtualStabilityEstimate (relative, finite,
 * single_pole, double_pole, checks [[name, relative]], scale, absolute and
 * magnitude per sector and epsilon power, not_run [[name, reason]]), bitwise
 * its values (non-finite values as "inf"); a loop-induced call with poles
 * also "finiteness" per successful point (the Python API's lane diagnostic);
 * the other evaluations carry
 * {"estimates": null, "reason": ...}, and null without checks. With an
 * explicit backend, a point whose rotated evaluation fails fails
 * ("rotation-<k>: <reason>"), as in the Python API; under "auto" on TIRust
 * such a point (and one TIRust refuses) is evaluated again, with the same
 * checks, on NinjaRS (the per-point fallback, rusticol_virtual_set_reduction),
 * its cross-backend check then in not_run with the fallback's reason. A point
 * the other backend fails or refuses keeps the other checks' estimate with
 * the cross-backend check in not_run. A cross-backend check
 * without the other backend in this library is refused by the call
 * (RUSTICOL_VIRTUAL_RUNTIME_ERROR, the Python API's ReductionUnavailable).
 * Point statuses keep their meaning: comparing an estimate with a target,
 * and any precision rescue, is the caller's (the rescue is the Python API's). */
int rusticol_virtual_set_stability_checks(rusticol_virtual_handle *handle,
  uint64_t rotation_count, const char *rotations, uint32_t reduction_direction,
  uint32_t rescaled, uint32_t cross_backend);
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
 * Every finite binary64 input is read as its exact value. Stability estimates
 * follow rusticol_virtual_set_stability_checks; the precision rescue is the
 * Python API's.
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
/* Extended-precision results: {"arithmetic","digits","born","virtual"} with
 * decimal literals in the copy layouts (absent poles null); "null" for a
 * binary64 result. */
int rusticol_virtual_result_decimal_json(const rusticol_virtual_result *, char *, size_t, size_t *);
/* Selections (additive, ABI version 2 kept; the Python API's helicities= and
 * color_resolution=, the tree C ABI's selector conventions). Global selectors
 * are string IDs, per-point selectors zero-based indices, exactly one per
 * point; global and per-point selectors are mutually exclusive on the same
 * axis; a zero count omits a selector (an omitted axis keeps every row).
 * Every selector is validated before any evaluation; *output is written only
 * on success (NULL otherwise).
 *
 * Loop-induced processes (full colour only, RULED Q8):
 * - helicity IDs are the capabilities' "helicity_ids" (h<i>) or the tree
 *   engine's IDs (h:<helicities in leg order>, e.g. "h:-1,-1,+1,+1");
 *   helicity_by_point[p] is the zero-based helicity row of point p;
 * - request.resolved = 0: per point the sum over the selected rows of
 *   |A(h)|^2/avg with the total's average (RULED Q39: the rows of a complete
 *   selection sum to the ordinary value; no parity or Bose row mapping,
 *   RULED Q37), in the ordinary layout; with helicity_by_point, each point's
 *   value is that of the global selection of its own row on that point;
 * - request.resolved = 1: the selected rows (all without helicity IDs) in
 *   the resolved layout [successful_point,order,helicity,colour,(epsilon)];
 *   color_resolution NULL or "contracted": one colour-contracted component
 *   ("full"); "amplitude-basis": one component per amplitude-basis element t
 *   ("amplitude-<t>"), Re(conj(a_t) (G a)_t)/avg, an explicitly
 *   BASIS-DEPENDENT diagnostic split (the basis may be linearly dependent at
 *   N_c = 3; PLAN 2.a.10) whose components sum to the row's value. Per-point
 *   selectors are refused for resolved calls (RUSTICOL_VIRTUAL_INVALID_ARGUMENT);
 * - only the selected rows are dressed and reduced (the executor's row
 *   liveness, the Python API's path): a single-helicity call costs less than
 *   the full evaluation; binary64 values are bitwise the Python API's;
 *   double-double / arbitrary precision as the other calls (decimal_json);
 * - refused before any evaluation: colour flows (color_ids,
 *   color_flow_by_point) and an unknown color_resolution with the Python
 *   API's CompatibilityError texts (RUSTICOL_VIRTUAL_RUNTIME_ERROR); an
 *   unknown helicity ID (its EvaluationError text), an empty or repeated ID
 *   (its ValueError texts), an out-of-range index, a wrong per-point length,
 *   both selectors of one axis, per-point selectors of a resolved call and a
 *   color_resolution of an ordinary call (RUSTICOL_VIRTUAL_INVALID_ARGUMENT);
 * - the call takes the handle's poles, arithmetic, reduction backend
 *   (rusticol_virtual_set_reduction: the same per-call policy, its
 *   requested / effective / reason record in the diagnostics, failed points
 *   as FAILED statuses with the Python API's reasons) and model parameters;
 *   with stability checks set, its diagnostics carry
 *   "stability": {"estimates": null, "reason": ...} (the Python API
 *   estimates selections), a cross-backend check whose other backend this
 *   library lacks is refused by the call (as rusticol_virtual_evaluate's);
 *   diagnostics "selection": {"helicity_ids": per point the selected h<i>}
 *   for an ordinary selected call.
 * Every process: a NULL selection, or one without selectors whose
 * color_resolution is NULL or "contracted", is rusticol_virtual_evaluate
 * (values and diagnostics, its monitoring included).
 * Born-interfered processes: a color_resolution is
 * refused with the Python API's text; selectors at NLC and full colour with
 * its text; a leading-colour selection is served by the Python API only (both
 * RUSTICOL_VIRTUAL_RUNTIME_ERROR). */
typedef struct {
  const char *const *helicity_ids; /* global helicity IDs */
  size_t helicity_count;
  const char *const *color_ids; /* global colour-flow IDs */
  size_t color_count;
  const uint32_t *helicity_by_point; /* zero-based, point_count entries */
  size_t helicity_by_point_count;
  const uint32_t *color_flow_by_point; /* zero-based, point_count entries */
  size_t color_flow_by_point_count;
  const char *color_resolution; /* resolved: NULL/"contracted" or "amplitude-basis" */
} rusticol_virtual_selection;
int rusticol_virtual_evaluate_selected(rusticol_virtual_handle *handle,
  const rusticol_virtual_batch_request *request,
  const rusticol_virtual_selection *selection, rusticol_virtual_result **output);
/* A result's axes as JSON: {"helicity_ids": [...], "color_ids": [...]} (a
 * resolved result's helicity rows and colour components in its layout order,
 * also counted by rusticol_virtual_result_shape's axes.helicities and
 * axes.virtual_color_components; empty lists for an ordinary result). */
int rusticol_virtual_result_axes_json(const rusticol_virtual_result *, char *, size_t, size_t *);
/* Additive grouped complex interface. Ordinary v1 structures and functions are
 * unchanged. Every saved history is evaluated once in the recorded ID order
 * ("born", the inclusive Born metric, first; then the declared histories);
 * request.resolved must be zero. Histories are never summed implicitly.
 * Shape counts below count complex pairs, not individual real components.
 * Plan v4: the stored observable columns of the process (the Python API's
 * evaluate_correlated_many, one request per history with the same spin
 * vectors), in the handle's pole selection and arithmetic.
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
/* [real, imaginary] decimal literals of an extended-precision correlated
 * result in the copy layouts (absent poles null); "null" in binary64. */
int rusticol_virtual_correlated_result_decimal_json(const rusticol_virtual_correlated_result *, char *, size_t, size_t *);
#ifdef __cplusplus
}
#endif
#endif
