/* SPDX-License-Identifier: 0BSD */

#ifndef RUSTICOL_H
#define RUSTICOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RUSTICOL_ABI_VERSION 1u

enum rusticol_status {
    RUSTICOL_STATUS_OK = 0,
    RUSTICOL_STATUS_INVALID_ARGUMENT = 1,
    RUSTICOL_STATUS_BUFFER_TOO_SMALL = 2,
    RUSTICOL_STATUS_RUNTIME_ERROR = 3,
    RUSTICOL_STATUS_PANIC = 4
};

typedef struct RusticolRuntimeHandle RusticolRuntimeHandle;

/*
 * A handle is mutable and must not be called concurrently. Independent handles
 * may be used concurrently from separate threads.
 *
 * For string and variable-length metadata getters, callers may query the
 * required capacity with a null buffer and zero capacity. String capacities
 * include the trailing NUL. Query and short-buffer calls do not consume
 * warning state; rusticol_runtime_take_warnings_json consumes warnings only
 * after a successful copy.
 */

uint32_t rusticol_abi_version(void);
int rusticol_supported_runtime_capabilities_json(
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_last_error_message(char *buffer, size_t capacity, size_t *required);

/* process_key may be a stable process/alias ID or a concrete expression. */
int rusticol_runtime_load(
    const char *process_dir,
    const char *process_key,
    const char *model_parameters_path,
    RusticolRuntimeHandle **output
);
int rusticol_runtime_free(RusticolRuntimeHandle *handle);

int rusticol_runtime_metadata_json(
    const RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_physics_json(
    const RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_process(
    const RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_process_key(
    const RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_color_accuracy(
    const RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);

int rusticol_runtime_external_count(
    const RusticolRuntimeHandle *handle,
    size_t *output
);
int rusticol_runtime_external_pdg(
    const RusticolRuntimeHandle *handle,
    size_t index,
    int32_t *output
);

int rusticol_runtime_helicity_count(
    const RusticolRuntimeHandle *handle,
    size_t *output
);
int rusticol_runtime_helicity_id(
    const RusticolRuntimeHandle *handle,
    size_t index,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_helicity_vector(
    const RusticolRuntimeHandle *handle,
    size_t index,
    int32_t *output,
    size_t capacity,
    size_t *required
);

int rusticol_runtime_color_count(
    const RusticolRuntimeHandle *handle,
    size_t *output
);
int rusticol_runtime_color_id(
    const RusticolRuntimeHandle *handle,
    size_t index,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_color_kind(
    const RusticolRuntimeHandle *handle,
    size_t index,
    char *buffer,
    size_t capacity,
    size_t *required
);
int rusticol_runtime_color_word(
    const RusticolRuntimeHandle *handle,
    size_t index,
    size_t *output,
    size_t capacity,
    size_t *required
);

int rusticol_runtime_model_parameter_count(
    const RusticolRuntimeHandle *handle,
    size_t *output
);
int rusticol_runtime_model_parameter_name(
    const RusticolRuntimeHandle *handle,
    size_t index,
    char *buffer,
    size_t capacity,
    size_t *required
);

int rusticol_runtime_resolved_shape(
    const RusticolRuntimeHandle *handle,
    const char *const *helicity_ids,
    size_t helicity_count,
    const char *const *color_ids,
    size_t color_count,
    size_t *output_helicity_count,
    size_t *output_color_count
);

/* Momenta use [point][external particle][E, px, py, pz]. */
int rusticol_runtime_evaluate_f64(
    RusticolRuntimeHandle *handle,
    const double *momenta,
    size_t momentum_count,
    size_t point_count,
    double *output,
    size_t output_capacity
);

/* Resolved output uses [point][helicity][color]. */
int rusticol_runtime_evaluate_resolved_f64(
    RusticolRuntimeHandle *handle,
    const double *momenta,
    size_t momentum_count,
    size_t point_count,
    const char *const *helicity_ids,
    size_t helicity_count,
    const char *const *color_ids,
    size_t color_count,
    double *output,
    size_t output_capacity,
    size_t *output_helicity_count,
    size_t *output_color_count
);

int rusticol_runtime_set_model_parameters(
    RusticolRuntimeHandle *handle,
    const char *const *names,
    const double *real,
    const double *imaginary,
    size_t count
);
int rusticol_runtime_set_model_parameter(
    RusticolRuntimeHandle *handle,
    const char *name,
    double real,
    double imaginary
);
int rusticol_runtime_set_model_parameters_json(
    RusticolRuntimeHandle *handle,
    const char *path
);

int rusticol_runtime_mute_warnings(
    RusticolRuntimeHandle *handle,
    int muted
);
int rusticol_runtime_take_warnings_json(
    RusticolRuntimeHandle *handle,
    char *buffer,
    size_t capacity,
    size_t *required
);

#ifdef __cplusplus
}
#endif

#endif
