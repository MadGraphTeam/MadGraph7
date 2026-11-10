/* SPDX-License-Identifier: 0BSD */
/* UMAMI provider of the ratio of a target-accuracy matrix element (e.g. full colour)
 * to the leading-colour one summed over all colour flows, at a fixed helicity.
 * With the parameter flavor_switch, events can also be switched to the flavours
 * they were not generated for, see the flavour switch below. */
#include "umami.h"
#include "reweight_provider.h"
#include <rusticol.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef UMAMI_REWEIGHT_HEADER
#define UMAMI_REWEIGHT_HEADER "reweight_p0.h"
#endif
#include UMAMI_REWEIGHT_HEADER

/* Events are evaluated in batches of at most this many points per native call. */
#define UMAMI_CHUNK 512

typedef struct {
    char *name;
    double real, imaginary;
} Override;

/* Helicity vectors and, for reference processes, the switch flow of each colour. */
typedef struct {
    size_t helicity_count, color_count;
    int *helicities, *flow_slots;
} RuntimeTables;

typedef struct {
    char *base;
    RusticolRuntimeHandle **reference, **target;
    RuntimeTables *reference_tables, *target_tables;
    double *momenta;
    /* Parameters set so far, applied to the processes loaded later. */
    Override *overrides;
    size_t override_count;
    bool flavor_switch;
    /* scratch of the flavour switch */
    double *values, *batch, *resolved;
    size_t values_capacity, batch_capacity, resolved_capacity;
    /* Per call: the unswitched events, their flavour, enhancement and the values of
     * both sides, and their order sorted by the process of one side. */
    size_t *order, *sorted, order_capacity, sorted_capacity;
    ReweightFlavour const **flavours;
    size_t flavours_capacity;
    double *enhancements, *side_values[2];
    size_t enhancements_capacity, side_capacity[2];
    size_t *starts, *cursor;
    /* Per chunk of UMAMI_CHUNK events. */
    double *chunk_values;
    size_t *events, *positions;
    ReweightFlavour const **entries;
    uint32_t *helicities;
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
    [UMAMI_IN_RANDOM_FLAVOR] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_RANDOM_COLOR] = true,
    [UMAMI_IN_PARTON_DENSITIES] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT_RATIO] = true,
    [UMAMI_OUT_FLAVOR_INDEX] = true,
    [UMAMI_OUT_COLOR_INDEX] = true,
    [UMAMI_OUT_HELICITY_INDEX] = true,
};

static UmamiStatus native_error(int status) {
    if (status == RUSTICOL_STATUS_OK) return UMAMI_SUCCESS;
    char message[1024];
    size_t required = 0;
    if (rusticol_last_error_message(message, sizeof(message), &required) ==
        RUSTICOL_STATUS_OK) fprintf(stderr, "umami: %s\n", message);
    return UMAMI_ERROR;
}

static UmamiStatus load(char const *base, char const *relative, char const *identity,
                        ReweightProcess const *process, RusticolRuntimeHandle **result) {
    size_t length = strlen(base) + strlen(relative) + 2;
    char *path = malloc(length);
    if (path == NULL) return UMAMI_ERROR;
    snprintf(path, length, "%s/%s", base, relative);
    RusticolRuntimeHandle *runtime = NULL;
    int status = rusticol_runtime_load(path, process->key, NULL, &runtime);
    free(path);
    if (status != RUSTICOL_STATUS_OK) return native_error(status);
    size_t count = 0, required = 0;
    status = rusticol_runtime_external_count(runtime, &count);
    bool compatible = status == RUSTICOL_STATUS_OK && count == reweight_provider.particle_count;
    for (size_t j = 0; compatible && j < count; ++j) {
        int32_t pdg = 0;
        status = rusticol_runtime_external_pdg(runtime, j, &pdg);
        compatible = status == RUSTICOL_STATUS_OK && pdg == process->pdgs[j];
    }
    if (compatible) {
        status = rusticol_runtime_helicity_count(runtime, &count);
        compatible = status == RUSTICOL_STATUS_OK && count == process->helicity_count;
    }
    if (compatible && identity != NULL) {
        char *loaded = malloc(strlen(identity) + 1);
        compatible = loaded != NULL && rusticol_runtime_artifact_id(
            runtime, loaded, strlen(identity) + 1, &required) == RUSTICOL_STATUS_OK &&
            strcmp(loaded, identity) == 0;
        free(loaded);
    }
    if (!compatible) {
        (void)rusticol_runtime_free(runtime);
        fprintf(stderr, "umami: artifact %s is incompatible with reweight provider %s\n",
                relative, reweight_provider.id);
        return UMAMI_ERROR;
    }
    *result = runtime;
    return UMAMI_SUCCESS;
}

/* Loads a reference (side 0) or target process on first use, with the parameters set. */
static UmamiStatus ensure_loaded(Instance *instance, int side, size_t index) {
    RusticolRuntimeHandle **slot =
        side == 0 ? &instance->reference[index] : &instance->target[index];
    if (*slot != NULL) return UMAMI_SUCCESS;
    RusticolRuntimeHandle *runtime = NULL;
    UmamiStatus status = side == 0
        ? load(instance->base, reweight_provider.reference_artifact,
               reweight_provider.reference_artifact_id,
               &reweight_provider.reference_processes[index], &runtime)
        : load(instance->base, reweight_provider.target_artifact,
               reweight_provider.target_artifact_id,
               &reweight_provider.target_processes[index], &runtime);
    if (status != UMAMI_SUCCESS) return status;
    for (size_t i = 0; i < instance->override_count; ++i) {
        Override const *override = &instance->overrides[i];
        double re = 0.0, im = 0.0;
        if (rusticol_runtime_get_model_parameter(runtime, override->name, &re, &im) !=
            RUSTICOL_STATUS_OK) continue;
        status = native_error(rusticol_runtime_set_model_parameter(
            runtime, override->name, override->real, override->imaginary));
        if (status != UMAMI_SUCCESS) { (void)rusticol_runtime_free(runtime); return status; }
    }
    *slot = runtime;
    return UMAMI_SUCCESS;
}

static UmamiStatus record_override(Instance *instance, char const *name,
                                   double real, double imaginary) {
    for (size_t i = 0; i < instance->override_count; ++i)
        if (strcmp(instance->overrides[i].name, name) == 0) {
            instance->overrides[i].real = real;
            instance->overrides[i].imaginary = imaginary;
            return UMAMI_SUCCESS;
        }
    Override *overrides = realloc(instance->overrides,
                                  (instance->override_count + 1) * sizeof(*overrides));
    if (overrides == NULL) return UMAMI_ERROR;
    instance->overrides = overrides;
    char *copy = malloc(strlen(name) + 1);
    if (copy == NULL) return UMAMI_ERROR;
    strcpy(copy, name);
    overrides[instance->override_count++] = (Override){copy, real, imaginary};
    return UMAMI_SUCCESS;
}

static bool reserve(double **buffer, size_t *capacity, size_t size) {
    if (size <= *capacity) return true;
    if (size > SIZE_MAX / sizeof(double)) return false;
    double *grown = realloc(*buffer, size * sizeof(double));
    if (grown == NULL) return false;
    *buffer = grown;
    *capacity = size;
    return true;
}

static bool reserve_indices(size_t **buffer, size_t *capacity, size_t size) {
    if (size <= *capacity) return true;
    if (size > SIZE_MAX / sizeof(size_t)) return false;
    size_t *grown = realloc(*buffer, size * sizeof(size_t));
    if (grown == NULL) return false;
    *buffer = grown;
    *capacity = size;
    return true;
}

UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)reweight_provider.particle_count; break;
    case UMAMI_META_HELICITY_COUNT:
        *(int *)result = (int)reweight_provider.exported_helicity_count; break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_supported_inputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_required_inputs(bool const **required, int *count) {
    if (required == NULL || count == NULL) return UMAMI_ERROR;
    *required = required_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_supported_outputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_outputs; *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    if (handle == NULL) return UMAMI_ERROR;
    *handle = NULL;
    struct stat info;
    if (param_card_path == NULL || param_card_path[0] != '/' ||
        stat(param_card_path, &info) != 0 || !S_ISDIR(info.st_mode) ||
        reweight_provider.particle_count > INT_MAX ||
        reweight_provider.particle_count > SIZE_MAX / (4 * sizeof(double) * UMAMI_CHUNK))
        return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    instance->reference = calloc(reweight_provider.reference_process_count + 1,
                                 sizeof(*instance->reference));
    instance->target = calloc(reweight_provider.target_process_count + 1,
                              sizeof(*instance->target));
    instance->reference_tables = calloc(reweight_provider.reference_process_count + 1,
                                        sizeof(*instance->reference_tables));
    instance->target_tables = calloc(reweight_provider.target_process_count + 1,
                                     sizeof(*instance->target_tables));
    instance->momenta = calloc(4 * reweight_provider.particle_count, sizeof(double));
    instance->base = malloc(strlen(param_card_path) + 1);
    size_t chunk = UMAMI_CHUNK, processes = reweight_provider.reference_process_count;
    if (reweight_provider.target_process_count > processes)
        processes = reweight_provider.target_process_count;
    instance->starts = calloc(processes + 1, sizeof(size_t));
    instance->cursor = calloc(processes + 1, sizeof(size_t));
    instance->chunk_values = calloc(chunk, sizeof(double));
    instance->events = calloc(chunk, sizeof(size_t));
    instance->positions = calloc(chunk, sizeof(size_t));
    instance->entries = calloc(chunk, sizeof(*instance->entries));
    instance->helicities = calloc(chunk, sizeof(uint32_t));
    if (instance->reference == NULL || instance->target == NULL || instance->momenta == NULL ||
        instance->reference_tables == NULL || instance->target_tables == NULL ||
        instance->base == NULL || reweight_provider.reference_process_count == 0 ||
        reweight_provider.target_process_count == 0 || instance->starts == NULL ||
        instance->cursor == NULL || instance->chunk_values == NULL || instance->events == NULL ||
        instance->positions == NULL || instance->entries == NULL || instance->helicities == NULL ||
        !reserve(&instance->batch, &instance->batch_capacity,
                 4 * reweight_provider.particle_count * chunk)) {
        umami_free(instance); return UMAMI_ERROR;
    }
    strcpy(instance->base, param_card_path);
    /* Validate both artifacts now; other processes are loaded lazily on first use. */
    for (int side = 0; side < 2; ++side) {
        UmamiStatus status = ensure_loaded(instance, side, 0);
        if (status != UMAMI_SUCCESS) { umami_free(instance); return status; }
    }
    *handle = instance;
    return UMAMI_SUCCESS;
}

/* Runs body for every loaded runtime, reference ones first. */
#define FOR_EACH_RUNTIME(instance, runtime, body) \
    for (size_t side = 0; side < 2; ++side) { \
        size_t count = side == 0 ? reweight_provider.reference_process_count \
                                 : reweight_provider.target_process_count; \
        for (size_t i = 0; i < count; ++i) { \
            RusticolRuntimeHandle *runtime = \
                side == 0 ? (instance)->reference[i] : (instance)->target[i]; \
            if (runtime == NULL) continue; \
            body \
        } \
    }

UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                              double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    Instance *instance = handle;
    if (strcmp(name, "flavor_switch") == 0 && reweight_provider.flavor_switch != NULL) {
        *real = instance->flavor_switch;
        if (imaginary != NULL) *imaginary = 0.0;
        return UMAMI_SUCCESS;
    }
    double re = 0.0, im = 0.0;
    FOR_EACH_RUNTIME(instance, runtime, {
        if (rusticol_runtime_get_model_parameter(runtime, name, &re, &im) ==
            RUSTICOL_STATUS_OK) {
            *real = re;
            if (imaginary != NULL) *imaginary = im;
            return UMAMI_SUCCESS;
        }
    })
    return UMAMI_ERROR;
}

UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                              double real, double imaginary) {
    if (handle == NULL || name == NULL || !isfinite(real)) return UMAMI_ERROR;
    Instance *instance = handle;
    if (strcmp(name, "flavor_switch") == 0) {
        if (reweight_provider.flavor_switch == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
        instance->flavor_switch = real != 0.0;
        return UMAMI_SUCCESS;
    }
    bool found = false;
    double re = 0.0, im = 0.0;
    FOR_EACH_RUNTIME(instance, runtime, {
        if (rusticol_runtime_get_model_parameter(runtime, name, &re, &im) !=
            RUSTICOL_STATUS_OK) continue;
        found = true;
        if (native_error(rusticol_runtime_set_model_parameter(runtime, name, real, imaginary)) !=
            UMAMI_SUCCESS) return UMAMI_ERROR;
    })
    if (!found) return UMAMI_ERROR;
    return record_override(instance, name, real, imaginary);
}

/* Value of one process at a fixed runtime helicity, with an optional alpha_s. */
static UmamiStatus evaluate(Instance *instance, RusticolRuntimeHandle *runtime,
                            ReweightProcess const *process, uint32_t helicity,
                            double const *alpha_s, double *result) {
    size_t n = reweight_provider.particle_count;
    double previous_real = 0.0, previous_imaginary = 0.0;
    bool changed = false;
    UmamiStatus status;
    if (alpha_s != NULL) {
        if (process->alpha_s_name == NULL) return UMAMI_ERROR_UNSUPPORTED_INPUT;
        status = native_error(rusticol_runtime_get_model_parameter(
            runtime, process->alpha_s_name, &previous_real, &previous_imaginary));
        if (status != UMAMI_SUCCESS) return status;
        changed = previous_real != *alpha_s || previous_imaginary != 0.0;
        if (changed) {
            status = native_error(rusticol_runtime_set_model_parameter(
                runtime, process->alpha_s_name, *alpha_s, 0.0));
            if (status != UMAMI_SUCCESS) return status;
        }
    }
    status = native_error(rusticol_runtime_evaluate_selected_f64(
        runtime, instance->momenta, 4*n, 1, NULL, 0, NULL, 0,
        &helicity, 1, NULL, 0, result, 1));
    if (changed) {
        UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
            runtime, process->alpha_s_name, previous_real, previous_imaginary));
        if (restored != UMAMI_SUCCESS) return restored;
    }
    return status;
}

/* ---- flavour switch ------------------------------------------------------ */

static UmamiStatus helicity_table(RusticolRuntimeHandle *runtime, RuntimeTables *tables) {
    size_t n = reweight_provider.particle_count, required = 0;
    UmamiStatus status =
        native_error(rusticol_runtime_helicity_count(runtime, &tables->helicity_count));
    if (status != UMAMI_SUCCESS) return status;
    tables->helicities = malloc((tables->helicity_count * n + 1) * sizeof(int));
    if (tables->helicities == NULL) return UMAMI_ERROR;
    for (size_t h = 0; h < tables->helicity_count; ++h) {
        status = native_error(rusticol_runtime_helicity_vector(
            runtime, h, (int32_t *)&tables->helicities[h * n], n, &required));
        if (status != UMAMI_SUCCESS) return status;
    }
    return UMAMI_SUCCESS;
}

/* The tables of a reference process, with its runtime loaded. */
static UmamiStatus reference_tables(Instance *instance, size_t index, RuntimeTables **result) {
    RuntimeTables *tables = &instance->reference_tables[index];
    *result = tables;
    if (tables->helicities != NULL) return UMAMI_SUCCESS;
    UmamiStatus status = ensure_loaded(instance, 0, index);
    if (status != UMAMI_SUCCESS) return status;
    RusticolRuntimeHandle *runtime = instance->reference[index];
    status = native_error(rusticol_runtime_color_count(runtime, &tables->color_count));
    if (status != UMAMI_SUCCESS) return status;
    size_t helicities = 0, colors = 0;
    status = native_error(rusticol_runtime_resolved_shape(
        runtime, NULL, 0, NULL, 0, &helicities, &colors));
    if (status != UMAMI_SUCCESS) return status;
    tables->flow_slots = malloc((tables->color_count + 1) * sizeof(int));
    if (tables->flow_slots == NULL) return UMAMI_ERROR;
    ReweightSwitchFlows const *flows = &reweight_provider.flavor_switch->flows[index];
    char name[512];
    for (size_t c = 0; c < tables->color_count; ++c) {
        size_t required = 0;
        status = native_error(
            rusticol_runtime_color_id(runtime, c, name, sizeof(name), &required));
        if (status != UMAMI_SUCCESS) return status;
        tables->flow_slots[c] = -1;
        for (size_t f = 0; f < flows->flow_count; ++f)
            if (strcmp(flows->flow_ids[f], name) == 0) tables->flow_slots[c] = (int)f;
    }
    status = helicity_table(runtime, tables);
    if (status != UMAMI_SUCCESS) return status;
    if (helicities != tables->helicity_count || colors != tables->color_count) {
        fprintf(stderr, "umami: unexpected resolved shape of a reference process\n");
        return UMAMI_ERROR;
    }
    return UMAMI_SUCCESS;
}

static UmamiStatus target_tables(Instance *instance, size_t index, RuntimeTables **result) {
    RuntimeTables *tables = &instance->target_tables[index];
    *result = tables;
    if (tables->helicities != NULL) return UMAMI_SUCCESS;
    UmamiStatus status = ensure_loaded(instance, 1, index);
    if (status != UMAMI_SUCCESS) return status;
    return helicity_table(instance->target[index], tables);
}

/* Leading colour of every switch evaluation for the events of the batch. */
static UmamiStatus switch_totals(Instance *instance, size_t count, size_t stride,
                                 size_t offset, double const *momenta) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    size_t n = reweight_provider.particle_count;
    if (!reserve(&instance->values, &instance->values_capacity, sw->evaluation_count * count) ||
        !reserve(&instance->batch, &instance->batch_capacity, 4 * n * count))
        return UMAMI_ERROR;
    for (size_t e = 0; e < sw->evaluation_count; ++e) {
        ReweightSwitchEvaluation const *evaluation = &sw->evaluations[e];
        UmamiStatus status = ensure_loaded(instance, 0, evaluation->process);
        if (status != UMAMI_SUCCESS) return status;
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < 4; ++k)
                    instance->batch[4 * (n * i + j) + k] =
                        momenta[offset + i + stride * (n * k + evaluation->legs[j])];
        status = native_error(rusticol_runtime_evaluate_selected_f64(
            instance->reference[evaluation->process], instance->batch, 4 * n * count, count,
            NULL, 0, NULL, 0, NULL, 0, NULL, 0, &instance->values[e * count], count));
        if (status != UMAMI_SUCCESS) return status;
    }
    return UMAMI_SUCCESS;
}

/* Share of an ordering in the summed leading colour of the event. */
static double switch_share(Instance *instance, ReweightSwitchOrdering const *ordering,
                           double const *densities, size_t event, size_t index,
                           size_t count, size_t stride) {
    double pdf0 = densities[event + stride * (0 + 2 * ordering->pdf[0])];
    double pdf1 = densities[event + stride * (1 + 2 * ordering->pdf[1])];
    return ordering->weight * pdf0 * pdf1 *
        instance->values[ordering->evaluation * count + index];
}

/* The switched event: its helicity, colour and ratio at the target. */
static UmamiStatus switch_event(Instance *instance, ReweightSwitchTarget const *target,
                                double const *momenta, size_t event, size_t stride,
                                double random_helicity, double random_color,
                                int *helicity_out, int *color_out, double *ratio) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    ReweightSwitchEvaluation const *evaluation = &sw->evaluations[target->evaluation];
    size_t n = reweight_provider.particle_count;
    RuntimeTables *lc = NULL, *fc = NULL;
    UmamiStatus status = reference_tables(instance, evaluation->process, &lc);
    if (status == UMAMI_SUCCESS) status = target_tables(instance, target->target_process, &fc);
    if (status != UMAMI_SUCCESS) return status;
    size_t size = lc->helicity_count * lc->color_count;
    if (!reserve(&instance->resolved, &instance->resolved_capacity, size)) return UMAMI_ERROR;
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < 4; ++k)
            instance->momenta[4 * j + k] = momenta[event + stride * (n * k + evaluation->legs[j])];
    size_t helicities = 0, colors = 0;
    status = native_error(rusticol_runtime_evaluate_resolved_f64(
        instance->reference[evaluation->process], instance->momenta, 4 * n, 1, NULL, 0,
        NULL, 0, instance->resolved, size, &helicities, &colors));
    if (status != UMAMI_SUCCESS) return status;
    double total = 0.0;
    for (size_t i = 0; i < size; ++i) total += instance->resolved[i];
    if (!(total > 0.0)) return UMAMI_ERROR;
    /* the helicity in proportion to the flows it sums, then a flow at it */
    size_t h = 0, c = 0;
    double row = 0.0, threshold = random_helicity * total;
    for (h = 0; h < lc->helicity_count; ++h) {
        row = 0.0;
        for (c = 0; c < lc->color_count; ++c) row += instance->resolved[h * lc->color_count + c];
        if (threshold < row || h + 1 == lc->helicity_count) break;
        threshold -= row;
    }
    threshold = random_color * row;
    for (c = 0; c < lc->color_count; ++c) {
        double value = instance->resolved[h * lc->color_count + c];
        if (threshold < value || c + 1 == lc->color_count) break;
        threshold -= value;
    }
    int const *vector = &lc->helicities[h * n];
    int exported = -1, fc_helicity = -1;
    for (size_t e = 0; exported < 0 && e < reweight_provider.exported_helicity_count; ++e) {
        bool equal = true;
        for (size_t j = 0; equal && j < n; ++j)
            equal = sw->helicities[e * n + evaluation->legs[j]] == vector[j];
        if (equal) exported = (int)e;
    }
    for (size_t f = 0; fc_helicity < 0 && f < fc->helicity_count; ++f)
        if (memcmp(&fc->helicities[f * n], vector, n * sizeof(int)) == 0) fc_helicity = (int)f;
    int slot = lc->flow_slots[c];
    if (exported < 0 || slot < 0) {
        fprintf(stderr, "umami: switch target without exported helicity or flow\n");
        return UMAMI_ERROR;
    }
    double value = 0.0;
    if (fc_helicity >= 0) {
        status = evaluate(instance, instance->target[target->target_process],
                          &reweight_provider.target_processes[target->target_process],
                          (uint32_t)fc_helicity, NULL, &value);
        if (status != UMAMI_SUCCESS) return status;
    }
    *ratio = row > 0.0 ? value / row : 0.0;
    *helicity_out = exported;
    *color_out = target->colors[slot];
    return UMAMI_SUCCESS;
}

/* ---- batched evaluation ------------------------------------------------ */

/* Values of one process at the points events[0..count), each through the permutation
 * of its flavour, at the runtime helicities[point], natively batched over runs of equal
 * alpha_s (or alpha_s NULL). */
static UmamiStatus evaluate_points(Instance *instance, RusticolRuntimeHandle *runtime,
                                   ReweightProcess const *process,
                                   ReweightFlavour const *const *entries, double const *momenta,
                                   size_t stride, size_t const *events, size_t count,
                                   uint32_t const *helicities, double const *alpha_s,
                                   double *values) {
    size_t n = reweight_provider.particle_count;
    if (alpha_s != NULL && process->alpha_s_power >= 0) {
        /* ME(alpha_s) = (alpha_s / alpha_s_0)^power ME(alpha_s_0), proven at generation */
        double reference = 0.0, imaginary = 0.0;
        UmamiStatus status = native_error(rusticol_runtime_get_model_parameter(
            runtime, process->alpha_s_name, &reference, &imaginary));
        if (status != UMAMI_SUCCESS) return status;
        if (reference > 0.0 && imaginary == 0.0) {
            status = evaluate_points(instance, runtime, process, entries, momenta, stride,
                                     events, count, helicities, NULL, values);
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < count; ++i)
                values[i] *= pow(alpha_s[events[i]] / reference, process->alpha_s_power);
            return UMAMI_SUCCESS;
        }
    }
    for (size_t start = 0, end; start < count; start = end) {
        end = count;
        if (alpha_s != NULL)
            for (end = start + 1; end < count && alpha_s[events[end]] == alpha_s[events[start]];)
                ++end;
        size_t size = end - start;
        for (size_t i = 0; i < size; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < 4; ++k)
                    instance->batch[4 * (n * i + j) + k] = momenta[
                        events[start + i] + stride * (n * k + entries[start + i]->permutation[j])];
        double previous_real = 0.0, previous_imaginary = 0.0;
        bool changed = false;
        UmamiStatus status;
        if (alpha_s != NULL) {
            double value = alpha_s[events[start]];
            status = native_error(rusticol_runtime_get_model_parameter(
                runtime, process->alpha_s_name, &previous_real, &previous_imaginary));
            if (status != UMAMI_SUCCESS) return status;
            changed = previous_real != value || previous_imaginary != 0.0;
            if (changed) {
                status = native_error(rusticol_runtime_set_model_parameter(
                    runtime, process->alpha_s_name, value, 0.0));
                if (status != UMAMI_SUCCESS) return status;
            }
        }
        status = native_error(rusticol_runtime_evaluate_selected_f64(
            runtime, instance->batch, 4 * n * size, size, NULL, 0, NULL, 0,
            helicities + start, size, NULL, 0, values + start, size));
        if (changed) {
            UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
                runtime, process->alpha_s_name, previous_real, previous_imaginary));
            if (restored != UMAMI_SUCCESS) return restored;
        }
        if (status != UMAMI_SUCCESS) return status;
    }
    return UMAMI_SUCCESS;
}

/* One side (reference 0 or target 1) of the pending events, grouped by its process;
 * absent helicities give zero. */
static UmamiStatus evaluate_side(Instance *instance, int side, void const *const *in,
                                 size_t stride, size_t pending) {
    int const *labels = in[UMAMI_IN_HELICITY_INDEX];
    size_t processes = side == 0 ? reweight_provider.reference_process_count
                                 : reweight_provider.target_process_count;
    double *result = instance->side_values[side];
    size_t *starts = instance->starts;
    for (size_t p = 0; p <= processes; ++p) starts[p] = 0;
    for (size_t i = 0; i < pending; ++i) {
        ReweightFlavour const *entry = instance->flavours[i];
        ++starts[(side == 0 ? entry->reference_process : entry->target_process) + 1];
        result[i] = 0.0;
    }
    for (size_t p = 0; p < processes; ++p) starts[p + 1] += starts[p];
    for (size_t p = 0; p < processes; ++p) instance->cursor[p] = starts[p];
    for (size_t i = 0; i < pending; ++i) {
        ReweightFlavour const *entry = instance->flavours[i];
        instance->sorted[instance->cursor[
            side == 0 ? entry->reference_process : entry->target_process]++] = i;
    }
    for (size_t index = 0; index < processes; ++index)
        for (size_t start = starts[index]; start < starts[index + 1]; start += UMAMI_CHUNK) {
            size_t size = starts[index + 1] - start, kept = 0;
            if (size > UMAMI_CHUNK) size = UMAMI_CHUNK;
            for (size_t i = 0; i < size; ++i) {
                size_t position = instance->sorted[start + i];
                ReweightFlavour const *entry = instance->flavours[position];
                size_t event = instance->order[position];
                int const *map = side == 0 ? entry->reference_helicity : entry->target_helicity;
                int helicity = map[labels[event]];
                if (helicity < 0) continue;
                instance->events[kept] = event;
                instance->positions[kept] = position;
                instance->entries[kept] = entry;
                instance->helicities[kept++] = (uint32_t)helicity;
            }
            if (kept == 0) continue;
            UmamiStatus status = ensure_loaded(instance, side, index);
            if (status != UMAMI_SUCCESS) return status;
            status = evaluate_points(
                instance, side == 0 ? instance->reference[index] : instance->target[index],
                side == 0 ? &reweight_provider.reference_processes[index]
                          : &reweight_provider.target_processes[index],
                instance->entries, in[UMAMI_IN_MOMENTA], stride, instance->events, kept,
                instance->helicities, in[UMAMI_IN_ALPHA_S], instance->chunk_values);
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < kept; ++i)
                result[instance->positions[i]] = instance->chunk_values[i];
        }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride-offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL)) ||
        (stride && reweight_provider.particle_count > SIZE_MAX / stride / 4)) return UMAMI_ERROR;
    void const *in[UMAMI_INPUT_KEY_COUNT] = {0};
    void *out[UMAMI_OUTPUT_KEY_COUNT] = {0};
    for (size_t i = 0; i < input_count; ++i) {
        int key = input_keys[i];
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !supported_inputs[key])
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (inputs[i] == NULL || in[key] != NULL) return UMAMI_ERROR;
        in[key] = inputs[i];
    }
    for (size_t i = 0; i < output_count; ++i) {
        int key = output_keys[i];
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || !supported_outputs[key])
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        if (outputs[i] == NULL || out[key] != NULL) return UMAMI_ERROR;
        out[key] = outputs[i];
    }
    if (count == 0) return UMAMI_SUCCESS;
    if (in[UMAMI_IN_MOMENTA] == NULL || in[UMAMI_IN_HELICITY_INDEX] == NULL)
        return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    double const *momenta = in[UMAMI_IN_MOMENTA];
    size_t n = reweight_provider.particle_count;
    bool switching = instance->flavor_switch;
    double const *densities = in[UMAMI_IN_PARTON_DENSITIES];
    if (switching) {
        if (densities == NULL || in[UMAMI_IN_RANDOM_FLAVOR] == NULL ||
            in[UMAMI_IN_RANDOM_HELICITY] == NULL || in[UMAMI_IN_RANDOM_COLOR] == NULL)
            return UMAMI_ERROR_MISSING_INPUT;
        UmamiStatus status = switch_totals(instance, count, stride, offset, momenta);
        if (status != UMAMI_SUCCESS) return status;
    }
    if (count > instance->flavours_capacity) {
        if (count > SIZE_MAX / sizeof(*instance->flavours)) return UMAMI_ERROR;
        ReweightFlavour const **flavours =
            realloc(instance->flavours, count * sizeof(*instance->flavours));
        if (flavours == NULL) return UMAMI_ERROR;
        instance->flavours = flavours;
        instance->flavours_capacity = count;
    }
    if (!reserve_indices(&instance->order, &instance->order_capacity, count) ||
        !reserve_indices(&instance->sorted, &instance->sorted_capacity, count) ||
        !reserve(&instance->enhancements, &instance->enhancements_capacity, count) ||
        !reserve(&instance->side_values[0], &instance->side_capacity[0], count) ||
        !reserve(&instance->side_values[1], &instance->side_capacity[1], count))
        return UMAMI_ERROR;
    size_t pending = 0;
    for (size_t event = offset; event < offset+count; ++event) {
        int switched_flavor = -1, switched_color = -1, switched_helicity = -1;
        double enhancement = 1.0;
        bool switched = false;
        if (switching) {
            ReweightSwitch const *sw = reweight_provider.flavor_switch;
            size_t index = event - offset;
            double generated = 0.0, dropped = 0.0;
            for (size_t o = 0; o < sw->ordering_count; ++o) {
                double share = switch_share(instance, &sw->orderings[o], densities, event,
                                            index, count, stride);
                if (sw->orderings[o].target < 0) generated += share;
                else dropped += share;
            }
            if (!(generated > 0.0) || !isfinite(dropped)) return UMAMI_ERROR;
            enhancement = 1.0 + dropped / generated;
            double u = ((double const *)in[UMAMI_IN_RANDOM_FLAVOR])[event] * enhancement;
            if (u >= 1.0 && dropped > 0.0) {
                double threshold = (u - 1.0) * generated;
                int target = -1;
                for (size_t o = 0; o < sw->ordering_count; ++o) {
                    if (sw->orderings[o].target < 0) continue;
                    target = sw->orderings[o].target;
                    threshold -= switch_share(instance, &sw->orderings[o], densities, event,
                                              index, count, stride);
                    if (threshold < 0.0) break;
                }
                double ratio = 0.0;
                UmamiStatus status = switch_event(
                    instance, &sw->targets[target], momenta, event, stride,
                    ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event],
                    ((double const *)in[UMAMI_IN_RANDOM_COLOR])[event],
                    &switched_helicity, &switched_color, &ratio);
                if (status != UMAMI_SUCCESS) return status;
                switched = true;
                switched_flavor = target;
                ratio *= enhancement;
                if (!isfinite(ratio)) return UMAMI_ERROR;
                if (out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])
                    ((double *)out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])[event] = ratio;
            }
        }
        if (out[UMAMI_OUT_FLAVOR_INDEX]) ((int *)out[UMAMI_OUT_FLAVOR_INDEX])[event] = switched_flavor;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = switched_color;
        if (out[UMAMI_OUT_HELICITY_INDEX])
            ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = switched_helicity;
        if (switched) continue;
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
            ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
            ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        int helicity = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
        if (channel >= reweight_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= reweight_provider.channels[channel].flavour_count ||
            helicity < 0 || (size_t)helicity >= reweight_provider.exported_helicity_count)
            return UMAMI_ERROR;
        ReweightFlavour const *entry = &reweight_provider.channels[channel].flavours[flavour];
        for (size_t k = 0; k < 4 * n; ++k)
            if (!isfinite(momenta[event + stride * k])) return UMAMI_ERROR;
        if (in[UMAMI_IN_ALPHA_S] &&
            ((entry->reference_helicity[helicity] >= 0 &&
              reweight_provider.reference_processes[entry->reference_process].alpha_s_name ==
                  NULL) ||
             (entry->target_helicity[helicity] >= 0 &&
              reweight_provider.target_processes[entry->target_process].alpha_s_name == NULL)))
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        instance->flavours[pending] = entry;
        instance->order[pending] = event;
        instance->enhancements[pending++] = enhancement;
    }
    /* The unswitched events, in batches by the process of each side. */
    for (int side = 0; side < 2; ++side) {
        UmamiStatus status = evaluate_side(instance, side, in, stride, pending);
        if (status != UMAMI_SUCCESS) return status;
    }
    for (size_t i = 0; i < pending; ++i) {
        double reference = instance->side_values[0][i];
        double ratio = reference != 0.0 ?
            instance->enhancements[i] * instance->side_values[1][i] / reference : 0.0;
        if (!isfinite(ratio)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])
            ((double *)out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])[instance->order[i]] = ratio;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    if (instance->reference != NULL)
        for (size_t i = 0; i < reweight_provider.reference_process_count; ++i)
            if (instance->reference[i] != NULL) (void)rusticol_runtime_free(instance->reference[i]);
    if (instance->target != NULL)
        for (size_t i = 0; i < reweight_provider.target_process_count; ++i)
            if (instance->target[i] != NULL) (void)rusticol_runtime_free(instance->target[i]);
    for (size_t i = 0; i < instance->override_count; ++i) free(instance->overrides[i].name);
    for (size_t i = 0; instance->reference_tables != NULL &&
                       i < reweight_provider.reference_process_count; ++i) {
        free(instance->reference_tables[i].helicities);
        free(instance->reference_tables[i].flow_slots);
    }
    for (size_t i = 0; instance->target_tables != NULL &&
                       i < reweight_provider.target_process_count; ++i)
        free(instance->target_tables[i].helicities);
    free(instance->reference_tables); free(instance->target_tables);
    free(instance->values); free(instance->batch); free(instance->resolved);
    free(instance->order); free(instance->sorted); free(instance->flavours);
    free(instance->enhancements); free(instance->side_values[0]); free(instance->side_values[1]);
    free(instance->starts); free(instance->cursor); free(instance->chunk_values);
    free(instance->events); free(instance->positions); free(instance->entries);
    free(instance->helicities);
    free(instance->overrides); free(instance->base);
    free(instance->reference); free(instance->target); free(instance->momenta); free(instance);
    return UMAMI_SUCCESS;
}
