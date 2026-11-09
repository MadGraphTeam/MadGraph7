/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_VIRTUAL_V1_H
#define PYAMPLICOL_UMAMI_VIRTUAL_V1_H
#include "umami.h"
#include "rusticol_virtual.h"
#ifdef __cplusplus
extern "C" {
#endif

/* pyAmpliCol extension, NOT standard UMAMI virtual interoperability.
 * It adds no umami_* symbols and changes no upstream/tree enums or ABI.
 * Use rusticol_virtual_open/free/clear/update_parameters and its metadata,
 * capabilities and shape queries. A handle identifies one saved process.
 *
 * Layout version 2 (with the virtual C ABI version 2): the options struct
 * lost version 1's verification_json member. The file, symbol and key names
 * keep their historical _v1 / V1 spelling (renaming them would break every
 * client for no layout change); pyamplicol_umami_virtual_v1_version() and
 * this macro report the layout a client was compiled against.
 */
#define PYAMPLICOL_UMAMI_VIRTUAL_VERSION 2
/* These provider-specific keys are accepted only by the versioned function.
 * Standard UMAMI_IN_MOMENTA is also accepted, and UMAMI_IN_HELICITY_INDEX
 * (int[stride], as the tree provider's: per event the zero-based helicity row,
 * an index into the SDK capabilities' "helicity_ids"; ordinary evaluations,
 * options.resolved = 0) selects that row alone: the event's value is the
 * loop-induced |A(h)|^2/avg (RULED Q39; rusticol_virtual_evaluate_selected's
 * helicity_by_point; only the selected rows are dressed). COLOR_RESOLUTION
 * (a NUL-terminated char string, capacity its bytes; options.resolved = 1)
 * selects the colour axis of a loop-induced resolved output: "contracted"
 * (one colour-contracted component, the default) or "amplitude-basis" (one
 * component per amplitude-basis element t, Re(conj(a_t) (G a)_t)/avg, an
 * explicitly basis-dependent split summing to the row's value; C below is
 * then the basis size, the capabilities' "color_resolutions" list the IDs).
 * A Born-interfered process refuses both, as the C ABI does (its
 * leading-colour selection is the Python API's). Other standard inputs,
 * including random/probability selectors (UMAMI_IN_RANDOM_HELICITY),
 * UMAMI_IN_RANDOM_COLOR, channels and flavours, are unsupported; a
 * loop-induced process has no colour flows (full colour only, RULED Q8).
 */
enum pyamplicol_umami_virtual_v1_input_key {
    PYAMPLICOL_UMAMI_V1_IN_MU_R2 = 0x50410001, /* double[stride] */
    PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR = 0x50410002, /* double[1], broadcasts */
    PYAMPLICOL_UMAMI_V1_IN_POINT_ID = 0x50410003, /* uint64_t[stride], optional */
    PYAMPLICOL_UMAMI_V1_IN_SPIN_LEGS = 0x50410004, /* size_t[L], capacity defines L */
    PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS = 0x50410005, /* complex pair[stride*L*4] */
    PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS_SCALAR = 0x50410006, /* complex pair[L*4], broadcasts */
    PYAMPLICOL_UMAMI_V1_IN_COLOR_RESOLUTION = 0x50410007 /* char[], NUL-terminated */
};
enum pyamplicol_umami_virtual_v1_output_key {
    PYAMPLICOL_UMAMI_V1_OUT_BORN = 0x50410001, /* double, all Born squared orders */
    PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL = 0x50410002, /* double, all virtual orders/poles */
    PYAMPLICOL_UMAMI_V1_OUT_STATUS = 0x50410003, /* uint32_t[stride], REQUIRED */
    PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_BORN = 0x50410004, /* complex pair, history x Born order */
    PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL = 0x50410005 /* complex pair, history x virtual order x poles */
};
/* A loop-induced process (|A_loop|^2; rusticol_virtual_metadata_json names
 * "contribution_kind": "loop-induced" and "interference": "conj(loop)*loop")
 * is served by the same function: it has no Born, so the shape's born_orders
 * is 0 and PYAMPLICOL_UMAMI_V1_OUT_BORN (and OUT_CORRELATED_BORN) are empty
 * (zero width; a requested Born output is left untouched); OUT_VIRTUAL holds
 * |A_loop|^2 per loop x loop sector and epsilon power. */
typedef struct {
    uint32_t resolved; /* 0 inclusive; 1 all saved helicities/colour components */
    const char *r2_strategy; /* NULL inherits the SDK handle */
} pyamplicol_umami_virtual_v1_options;

uint32_t pyamplicol_umami_virtual_v1_version(void);
/* Borrowed error text, valid until the next failing extension/SDK call on this
 * thread. Function failures use UmamiStatus; numerical lane outcomes use the
 * unchanged rusticol_virtual_point_status values, never a positive weight.
 */
const char *pyamplicol_umami_virtual_v1_last_error_message(void);
UmamiStatus pyamplicol_umami_virtual_v1_supported_inputs(const int **keys, size_t *count);
UmamiStatus pyamplicol_umami_virtual_v1_supported_outputs(const int **keys, size_t *count);

/* UMAMI column-major layout, with event as the contiguous dimension:
 * momenta[event + stride*(leg + external_count*component)].
 * Inclusive Born:   out[event + stride*order].
 * Inclusive loop:   out[event + stride*(order + virtual_orders*epsilon)].
 * Resolved Born:    out[event + stride*(order + born_orders*(helicity + H*colour))].
 * Resolved loop:    out[event + stride*(order + virtual_orders*(helicity + H*(colour + C*epsilon)))].
 * epsilon=0,1,2 denotes finite,single_pole,double_pole. The Born and virtual
 * order catalogues are independent; obtain them from SDK metadata_json.
 *
 * Evaluate count events beginning at offset, with offset+count <= stride.
 * Capacities count typed elements for the complete stride-based array layout;
 * they accompany each pointer in inputs/outputs. Pointers must be aligned and
 * live for their capacities. Output buffers and diagnostic_result must not alias
 * inputs, options, each other or the runtime. Duplicate keys are errors.
 * Require momenta and exactly one scale input for a nonempty batch. Point IDs
 * default to the original event indices offset..offset+count. Status output is
 * mandatory for nonempty calls, making untouched failed numerical rows explicit.
 * On function failure no output array changes. On success, statuses are written
 * for every requested event; numerical outputs only for successful events.
 * All failed numerical slots and all slots outside the slice remain untouched.
 * Zero count accepts missing inputs/outputs, while supplied keys are validated.
 *
 * Exactly one native bulk evaluation is performed; packing/scattering are only
 * layout transport. Optional diagnostic_result returns the ordinary owned SDK
 * result (compact rows and full attempt diagnostics), freed with
 * rusticol_virtual_result_free. It is set to NULL on error. Pass NULL to discard.
 * No UMAMI_OUT_MATRIX_ELEMENT fallback, sampled helicity, probability, mapping
 * or correlated-output semantics are supplied by this extension.
 * With UMAMI_IN_HELICITY_INDEX or COLOR_RESOLUTION the bulk evaluation is
 * rusticol_virtual_evaluate_selected (still one native call per batch of
 * equal rows; validated before evaluation); a resolved call's output
 * capacities are checked against its own axes (H, C) before any write.
 */
UmamiStatus pyamplicol_umami_virtual_v1_evaluate(
    rusticol_virtual_handle *handle, size_t count, size_t stride, size_t offset,
    size_t input_count, const int *input_keys, const void *const *inputs,
    const size_t *input_capacities,
    size_t output_count, const int *output_keys, void *const *outputs,
    const size_t *output_capacities,
    const pyamplicol_umami_virtual_v1_options *options,
    rusticol_virtual_result **diagnostic_result);
/* Additive complex grouped operation, with the same input and slice rules.
 * Only *_CORRELATED_BORN/VIRTUAL and STATUS output keys are accepted.
 * Each complex element is rusticol_virtual_complex64 (real,imag), never a
 * probability. Query ordered history IDs with the standalone SDK.
 * Born: out[event + stride*(order + born_orders*history)].
 * Loop: out[event + stride*(order + virtual_orders*(history + J*epsilon))].
 * All saved histories are evaluated in one native call. options.resolved=0.
 * Empty/failed/outside-slice slots obey the same rules as ordinary evaluation.
 * The optional owned diagnostic result uses the correlated SDK free function.
 */
/* Correlated input adds optional SPIN_LEGS and exactly one SPIN_VECTORS key.
 * The leg list capacity is its exact length L, with one-based physical labels.
 * Per-point vectors use value[event + stride*(selected_leg + L*component)];
 * scalar/broadcast vectors use value[selected_leg + L*component]. Components
 * are literal complex pairs; no conjugation or normalization is performed.
 * Input vectors and labels are borrowed only for the call. Ordinary evaluation
 * rejects these keys. Class support comes from SDK spin_correlation_classes_json.
 */
UmamiStatus pyamplicol_umami_virtual_v1_supported_correlated_inputs(const int **keys, size_t *count);
UmamiStatus pyamplicol_umami_virtual_v1_supported_correlated_outputs(const int **keys, size_t *count);
UmamiStatus pyamplicol_umami_virtual_v1_evaluate_correlated(
    rusticol_virtual_handle *handle, size_t count, size_t stride, size_t offset,
    size_t input_count, const int *input_keys, const void *const *inputs,
    const size_t *input_capacities,
    size_t output_count, const int *output_keys, void *const *outputs,
    const size_t *output_capacities,
    const pyamplicol_umami_virtual_v1_options *options,
    rusticol_virtual_correlated_result **diagnostic_result);
#ifdef __cplusplus
}
#endif
#endif
