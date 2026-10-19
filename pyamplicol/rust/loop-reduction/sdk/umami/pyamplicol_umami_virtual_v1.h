/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_VIRTUAL_V1_H
#define PYAMPLICOL_UMAMI_VIRTUAL_V1_H
#include "umami.h"
#include "rusticol_virtual.h"
#ifdef __cplusplus
extern "C" {
#endif

/* pyAmpliCol extension version 1, NOT standard UMAMI virtual interoperability.
 * It adds no umami_* symbols and changes no upstream/tree enums or ABI.
 * Use rusticol_virtual_open/free/clear/update_parameters and its metadata,
 * capabilities and shape queries. A handle identifies one saved process.
 */
#define PYAMPLICOL_UMAMI_VIRTUAL_VERSION 1
/* These provider-specific keys are accepted only by the versioned function.
 * Standard UMAMI_IN_MOMENTA is also accepted. Other standard inputs, including
 * random/probability selectors, channels and flavours, are unsupported.
 */
enum pyamplicol_umami_virtual_v1_input_key {
    PYAMPLICOL_UMAMI_V1_IN_MU_R2 = 0x50410001, /* double[stride] */
    PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR = 0x50410002, /* double[1], broadcasts */
    PYAMPLICOL_UMAMI_V1_IN_POINT_ID = 0x50410003 /* uint64_t[stride], optional */
};
enum pyamplicol_umami_virtual_v1_output_key {
    PYAMPLICOL_UMAMI_V1_OUT_BORN = 0x50410001, /* double, all Born squared orders */
    PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL = 0x50410002, /* double, all virtual orders/poles */
    PYAMPLICOL_UMAMI_V1_OUT_STATUS = 0x50410003, /* uint32_t[stride], REQUIRED */
    PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_BORN = 0x50410004, /* complex pair, history x Born order */
    PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL = 0x50410005 /* complex pair, history x virtual order x poles */
};
typedef struct {
    uint32_t resolved; /* 0 inclusive; 1 all saved helicities/colour components */
    const char *r2_strategy; /* NULL inherits the SDK handle */
    const char *verification_json; /* NULL: no hidden verification/rescue */
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
