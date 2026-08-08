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

enum rusticol_warm_up_event_kind {
    RUSTICOL_WARM_UP_EVENT_START = 0,
    RUSTICOL_WARM_UP_EVENT_UPDATE = 1,
    RUSTICOL_WARM_UP_EVENT_END = 2
};

enum rusticol_warm_up_stage {
    RUSTICOL_WARM_UP_STAGE_PROCESS_PREPARATION = 0,
    RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY = 1,
    RUSTICOL_WARM_UP_STAGE_FAMILY_FINALIZATION = 2,
    RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION = 3
};

/*
 * Fixed-layout progress snapshot. RSS fields are meaningful only when their
 * matching *_available field is nonzero. The event pointer is borrowed for
 * the callback invocation only.
 */
typedef struct RusticolWarmUpProgressEvent {
    uint32_t schema_version;
    uint32_t kind;
    uint32_t stage;
    uint32_t reserved;
    uint64_t completed;
    uint64_t total;
    double elapsed_seconds;
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    uint64_t workers;
    uint32_t current_rss_available;
    uint32_t peak_rss_available;
} RusticolWarmUpProgressEvent;

typedef struct RusticolWarmUpResult {
    uint32_t schema_version;
    uint32_t reserved;
    double elapsed_seconds;
    uint64_t query_count;
    uint64_t warmed_query_count;
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    uint32_t current_rss_available;
    uint32_t peak_rss_available;
    uint32_t already_warm;
    uint32_t first_evaluation_completed;
} RusticolWarmUpResult;

/* Return nonzero to continue, or zero to request cancellation. */
typedef int (*RusticolWarmUpProgressCallback)(
    const RusticolWarmUpProgressEvent *event,
    void *user_data
);

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
int rusticol_runtime_execution_mode(
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
int rusticol_runtime_representative_process_key(
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
/* Representative-index to public/requested-index external-leg permutation. */
int rusticol_runtime_external_permutation(
    const RusticolRuntimeHandle *handle,
    size_t *output,
    size_t capacity,
    size_t *required
);
/* Load one public-order [external][4] JSON point into a flat f64 buffer. */
int rusticol_runtime_load_kinematics_json(
    const RusticolRuntimeHandle *handle,
    const char *path,
    double *output,
    size_t capacity,
    size_t *required
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

/*
 * Construct and retain one selected OTF query family, then evaluate exactly
 * one binary64 point. Momenta use [external particle][E, px, py, pz]. Global
 * selector IDs are optional; a null pointer with zero count sums that axis.
 * Progress callbacks run on this coordinating caller thread. The terminal
 * FIRST_EVALUATION/END event is post-commit and cannot cancel the result.
 */
int rusticol_runtime_warm_up_f64(
    RusticolRuntimeHandle *handle,
    const double *momenta,
    size_t momentum_count,
    const char *const *helicity_ids,
    size_t helicity_count,
    const char *const *color_ids,
    size_t color_count,
    RusticolWarmUpProgressCallback progress_callback,
    void *progress_user_data,
    RusticolWarmUpResult *output
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

/*
 * Evaluate one total per point with optional selectors. Global selectors are
 * physical string IDs. Per-point selectors are zero-based physical indices and
 * must have length point_count. Global and per-point selectors are mutually
 * exclusive on the same axis. A null pointer with zero length omits that axis.
 */
int rusticol_runtime_evaluate_selected_f64(
    RusticolRuntimeHandle *handle,
    const double *momenta,
    size_t momentum_count,
    size_t point_count,
    const char *const *helicity_ids,
    size_t helicity_count,
    const char *const *color_ids,
    size_t color_count,
    const uint32_t *helicity_by_point,
    size_t helicity_by_point_count,
    const uint32_t *color_flow_by_point,
    size_t color_flow_by_point_count,
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
