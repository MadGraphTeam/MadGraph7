/* SPDX-License-Identifier: 0BSD */
/* UMAMI provider of the ratio of a target-accuracy matrix element (e.g. full colour)
 * to the leading-colour one summed over all colour flows, at a fixed helicity.
 * With the parameter flavor_switch, events can also be switched to the flavours
 * they were not generated for, see the flavour switch below. */
/* POSIX (mmap, pthread) under -std=c11 -pedantic */
#define _POSIX_C_SOURCE 200809L
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
/* the constants of the provider; its tables are loaded from its data file */
#include UMAMI_REWEIGHT_HEADER
#include "reweight_data.h"

/* Events are evaluated in batches of at most this many points per native call. */
#define UMAMI_CHUNK 512

typedef struct {
    char *name;
    double real, imaginary;
} Override;

/* A switched event: its target, then the helicity and colour drawn at it. */
typedef struct {
    size_t event;
    int target, exported_helicity, fc_helicity, color;
    double enhancement, row, value;
} SwitchedEvent;

/* Helicity vectors and, for reference processes, the switch flow of each colour
 * and the colour of each switch flow (-1: none). */
typedef struct {
    size_t helicity_count, color_count;
    int *helicities, *flow_slots, *slot_colors;
} RuntimeTables;

typedef struct {
    char *base;
    RusticolRuntimeHandle **reference, **target;
    RuntimeTables *reference_tables, *target_tables;
    /* Parameters set so far, applied to the processes loaded later. */
    Override *overrides;
    size_t override_count;
    bool flavor_switch;
    /* scratch of the flavour switch */
    double *values, *batch, *resolved;
    size_t values_capacity, batch_capacity, resolved_capacity;
    /* Per call: the switched events, sorted by their evaluation or target process. */
    SwitchedEvent *switched;
    size_t switched_capacity;
    size_t *switch_sorted, *switch_starts, switch_sorted_capacity, switch_starts_capacity;
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
    /* Flavour-union requests: per chunk one assignment per point, and those of the
     * switch calls; helicity values in event slots. */
    size_t *offsets, *call_offsets, call_offsets_capacity;
    uint32_t *assignments, *call_ids;
    size_t call_ids_capacity;
    /* per switch call with relabelled requests: their runtime assignment ids */
    uint32_t **relabelled;
    double *call_values;
    size_t call_values_capacity;
    int *slot_helicities;
    /* per switch evaluation: its runtime helicities in physical order (lazy) */
    uint32_t **helicity_orders;
    /* holds a reference to the provider's data */
    bool data_acquired;
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

/* Whether flavour-union assignment `assignment` of a runtime has the process id `key`. */
static bool assignment_matches(RusticolRuntimeHandle *runtime, int assignment,
                               char const *key) {
    char name[1024];
    size_t required = 0;
    return key != NULL &&
        rusticol_runtime_assignment_process_id(runtime, (size_t)assignment, name,
            sizeof(name), &required) == RUSTICOL_STATUS_OK &&
        strcmp(name, key) == 0;
}

/* Whether the core's momentum (helicity = false) or helicity signs of an assignment
 * are `expected` (NULL: all +1, uncrossed). */
static bool signs_match(RusticolRuntimeHandle *runtime, uint32_t assignment,
                        int32_t const *expected, bool helicity) {
    int32_t signs[64];
    size_t required = 0;
    int status = helicity
        ? rusticol_runtime_assignment_helicity_signs(runtime, assignment, signs, 64, &required)
        : rusticol_runtime_assignment_momentum_signs(runtime, assignment, signs, 64, &required);
    if (status != RUSTICOL_STATUS_OK || required != reweight_provider.particle_count)
        return false;
    for (size_t j = 0; j < required; ++j)
        if (signs[j] != (expected == NULL ? 1 : expected[j])) return false;
    return true;
}

/* Whether the flavour-union assignments a reference (side 0) or target process is
 * used with are those of its runtime, with the same crossings. */
static bool assignments_compatible(RusticolRuntimeHandle *runtime, size_t process,
                                   int side) {
    for (size_t c = 0; c < reweight_provider.channel_count; ++c)
        for (size_t f = 0; f < reweight_provider.channels[c].flavour_count; ++f) {
            ReweightFlavour const *entry = &reweight_provider.channels[c].flavours[f];
            size_t index = side == 0 ? entry->reference_process : entry->target_process;
            int assignment =
                side == 0 ? entry->reference_assignment : entry->target_assignment;
            if (index != process || assignment < 0) continue;
            if (!assignment_matches(runtime, assignment, side == 0
                    ? entry->assignment_key : entry->target_assignment_key) ||
                !signs_match(runtime, (uint32_t)assignment, side == 0
                    ? entry->reference_signs : entry->target_signs, false)) return false;
        }
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    for (size_t e = 0; sw != NULL && side == 0 && e < sw->evaluation_count; ++e) {
        ReweightSwitchEvaluation const *evaluation = &sw->evaluations[e];
        if (evaluation->process != process || evaluation->assignment < 0) continue;
        uint32_t assignment = (uint32_t)evaluation->assignment;
        if (!assignment_matches(runtime, evaluation->assignment,
                evaluation->assignment_key) ||
            !signs_match(runtime, assignment, evaluation->signs, false) ||
            !signs_match(runtime, assignment, evaluation->helicity_signs, true))
            return false;
    }
    for (size_t t = 0; sw != NULL && side == 1 && t < sw->target_count; ++t) {
        ReweightSwitchTarget const *target = &sw->targets[t];
        if (target->target_process != process || target->assignment < 0) continue;
        if (!assignment_matches(runtime, target->assignment, rw_text(target->assignment_key)) ||
            !signs_match(runtime, (uint32_t)target->assignment, NULL, false))
            return false;
    }
    return true;
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
    if (!assignments_compatible(runtime, index, side)) {
        (void)rusticol_runtime_free(runtime);
        fprintf(stderr, "umami: flavour-union assignments of reweight provider %s do not "
                "match the %s artifact\n", reweight_provider.id,
                side == 0 ? "reference" : "target");
        return UMAMI_ERROR;
    }
    ReweightSwitch const *sw = side == 0 ? reweight_provider.flavor_switch : NULL;
    for (size_t g = 0; sw != NULL && g < sw->call_count; ++g) {
        ReweightSwitchCall const *call = &sw->calls[g];
        if (call->entry_permutations == NULL ||
            sw->evaluations[call->evaluations[0]].process != index) continue;
        size_t n = reweight_provider.particle_count;
        free(instance->relabelled[g]);
        instance->relabelled[g] = malloc(call->entry_count * sizeof(uint32_t));
        status = instance->relabelled[g] == NULL ? UMAMI_ERROR : UMAMI_SUCCESS;
        for (size_t e = 0; status == UMAMI_SUCCESS && e < call->entry_count; ++e) {
            size_t permutation[REWEIGHT_PARTICLE_COUNT];
            for (size_t j = 0; j < n; ++j) permutation[j] = call->entry_permutations[e * n + j];
            status = native_error(rusticol_runtime_relabel_assignment(
                runtime, call->assignments[e], permutation, n, &instance->relabelled[g][e]));
        }
        /* a relabelled request reads the call's gather: its crossing must be the call's */
        int32_t const *signs = sw->evaluations[call->evaluations[0]].signs;
        for (size_t e = 0; status == UMAMI_SUCCESS && e < call->entry_count; ++e)
            if (!signs_match(runtime, instance->relabelled[g][e], signs, false)) {
                fprintf(stderr, "umami: a relabelled switch request changes its crossing\n");
                status = UMAMI_ERROR;
            }
        if (status != UMAMI_SUCCESS) { (void)rusticol_runtime_free(runtime); return status; }
    }
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
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)REWEIGHT_PARTICLE_COUNT; break;
    case UMAMI_META_HELICITY_COUNT:
        *(int *)result = (int)REWEIGHT_EXPORTED_HELICITY_COUNT; break;
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
        REWEIGHT_PARTICLE_COUNT > INT_MAX ||
        REWEIGHT_PARTICLE_COUNT > SIZE_MAX / (4 * sizeof(double) * UMAMI_CHUNK))
        return UMAMI_ERROR;
    /* the tables, before anything reads their counts */
    UmamiStatus loaded = reweight_data_acquire(param_card_path);
    if (loaded != UMAMI_SUCCESS) return loaded;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) { reweight_data_release(); return UMAMI_ERROR; }
    /* every later error path releases it through umami_free */
    instance->data_acquired = true;
    instance->reference = calloc(reweight_provider.reference_process_count + 1,
                                 sizeof(*instance->reference));
    instance->target = calloc(reweight_provider.target_process_count + 1,
                              sizeof(*instance->target));
    instance->reference_tables = calloc(reweight_provider.reference_process_count + 1,
                                        sizeof(*instance->reference_tables));
    instance->target_tables = calloc(reweight_provider.target_process_count + 1,
                                     sizeof(*instance->target_tables));
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
    instance->offsets = calloc(chunk + 1, sizeof(size_t));
    instance->assignments = calloc(chunk, sizeof(uint32_t));
    instance->slot_helicities = calloc(reweight_provider.particle_count + 1, sizeof(int));
    size_t switch_calls = reweight_provider.flavor_switch == NULL
        ? 0 : reweight_provider.flavor_switch->call_count;
    instance->relabelled = calloc(switch_calls + 1, sizeof(uint32_t *));
    instance->helicity_orders = calloc(
        (reweight_provider.flavor_switch == NULL
             ? 0 : reweight_provider.flavor_switch->evaluation_count) + 1,
        sizeof(uint32_t *));
    if (instance->relabelled == NULL || instance->helicity_orders == NULL || instance->offsets == NULL || instance->assignments == NULL ||
        instance->slot_helicities == NULL || instance->reference == NULL || instance->target == NULL ||
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
        if (status != UMAMI_SUCCESS) {
            /* helicities != NULL marks the tables complete */
            free(tables->helicities);
            tables->helicities = NULL;
            return status;
        }
    }
    return UMAMI_SUCCESS;
}

/* Drops partly built reference tables, so that the next call rebuilds them. */
static UmamiStatus reset_reference_tables(RuntimeTables *tables, UmamiStatus status) {
    free(tables->flow_slots);
    tables->flow_slots = NULL;
    free(tables->slot_colors);
    tables->slot_colors = NULL;
    free(tables->helicities);
    tables->helicities = NULL;
    return status;
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
    free(tables->flow_slots);
    tables->flow_slots = malloc((tables->color_count + 1) * sizeof(int));
    if (tables->flow_slots == NULL) return UMAMI_ERROR;
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    ReweightSwitchFlows const *flows = &sw->flows[index];
    uint32_t const *ids = &sw->flow_ids[flows->first];
    char name[512];
    for (size_t c = 0; c < tables->color_count; ++c) {
        size_t required = 0;
        status = native_error(
            rusticol_runtime_color_id(runtime, c, name, sizeof(name), &required));
        if (status != UMAMI_SUCCESS) return reset_reference_tables(tables, status);
        tables->flow_slots[c] = -1;
        for (size_t f = 0; f < flows->count; ++f)
            if (strcmp(rw_strings + ids[f], name) == 0) tables->flow_slots[c] = (int)f;
    }
    free(tables->slot_colors);
    tables->slot_colors = malloc((flows->count + 1) * sizeof(int));
    if (tables->slot_colors == NULL) return reset_reference_tables(tables, UMAMI_ERROR);
    for (size_t f = 0; f < flows->count; ++f) tables->slot_colors[f] = -1;
    for (size_t c = 0; c < tables->color_count; ++c)
        if (tables->flow_slots[c] >= 0) tables->slot_colors[tables->flow_slots[c]] = (int)c;
    status = helicity_table(runtime, tables);
    if (status != UMAMI_SUCCESS) return reset_reference_tables(tables, status);
    if (helicities != tables->helicity_count || colors != tables->color_count) {
        fprintf(stderr, "umami: unexpected resolved shape of a reference process\n");
        return reset_reference_tables(tables, UMAMI_ERROR);
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
    for (size_t g = 0; g < sw->call_count; ++g) {
        ReweightSwitchCall const *call = &sw->calls[g];
        ReweightSwitchEvaluation const *evaluation = &sw->evaluations[call->evaluations[0]];
        UmamiStatus status = ensure_loaded(instance, 0, evaluation->process);
        if (status != UMAMI_SUCCESS) return status;
        /* the call's evaluations share their crossing (same template sides per slot) */
        int32_t const *signs = evaluation->signs;
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < 4; ++k)
                    instance->batch[4 * (n * i + j) + k] = (signs == NULL ? 1.0 : signs[j]) *
                        momenta[offset + i + stride * (n * k + evaluation->legs[j])];
        RusticolRuntimeHandle *runtime = instance->reference[evaluation->process];
        if (call->assignments == NULL) {
            status = native_error(rusticol_runtime_evaluate_selected_f64(
                runtime, instance->batch, 4 * n * count, count, NULL, 0, NULL, 0, NULL, 0,
                NULL, 0, &instance->values[call->evaluations[0] * count], count));
            if (status != UMAMI_SUCCESS) return status;
            continue;
        }
        /* every request of the call at every event, in one native call */
        size_t width = call->entry_count;
        uint32_t const *requested =
            call->entry_permutations == NULL ? call->assignments : instance->relabelled[g];
        if (width > SIZE_MAX / count / sizeof(double) ||
            !reserve_indices(&instance->call_offsets, &instance->call_offsets_capacity,
                             count + 1) ||
            !reserve(&instance->call_values, &instance->call_values_capacity, count * width))
            return UMAMI_ERROR;
        if (count * width > instance->call_ids_capacity) {
            uint32_t *grown = realloc(instance->call_ids, count * width * sizeof(uint32_t));
            if (grown == NULL) return UMAMI_ERROR;
            instance->call_ids = grown;
            instance->call_ids_capacity = count * width;
        }
        for (size_t i = 0; i <= count; ++i) instance->call_offsets[i] = i * width;
        for (size_t i = 0; i < count; ++i)
            memcpy(&instance->call_ids[i * width], requested, width * sizeof(uint32_t));
        status = native_error(rusticol_runtime_evaluate_assignments_f64(
            runtime, instance->batch, 4 * n * count, count, instance->call_offsets, count + 1,
            instance->call_ids, count * width, NULL, 0, NULL, 0, 0, instance->call_values,
            count * width));
        if (status != UMAMI_SUCCESS) return status;
        for (size_t j = 0; j < call->evaluation_count; ++j)
            for (size_t i = 0; i < count; ++i)
                instance->values[call->evaluations[j] * count + i] =
                    instance->call_values[i * width + call->evaluation_entries[j]];
    }
    return UMAMI_SUCCESS;
}

/* Share of an ordering in the summed leading colour of the event. */
static double switch_share(Instance *instance, ReweightSwitchOrdering const *ordering,
                           double const *densities, size_t event, size_t index,
                           size_t count, size_t stride) {
    double pdf0 = densities[event + stride * (0 + 2 * ordering->pdf[0])];
    double pdf1 = densities[event + stride * (1 + 2 * ordering->pdf[1])];
    return reweight_provider.flavor_switch->constants[ordering->weight] * pdf0 * pdf1 *
        instance->values[ordering->evaluation * count + index];
}

/* Sorts the switched events by their evaluation (0) or target process (1) into
 * switch_sorted, with the range of key k at switch_starts[k..k+1). */
static bool sort_switched(Instance *instance, size_t switched, int key_kind) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    size_t keys = key_kind == 0 ? sw->evaluation_count : reweight_provider.target_process_count;
    if (!reserve_indices(&instance->switch_starts, &instance->switch_starts_capacity, keys + 2) ||
        !reserve_indices(&instance->switch_sorted, &instance->switch_sorted_capacity, switched))
        return false;
    size_t *starts = instance->switch_starts;
    for (size_t k = 0; k < keys + 2; ++k) starts[k] = 0;
    for (size_t i = 0; i < switched; ++i) {
        ReweightSwitchTarget const *target = &sw->targets[instance->switched[i].target];
        ++starts[(key_kind == 0 ? target->evaluation : target->target_process) + 2];
    }
    for (size_t k = 0; k < keys; ++k) starts[k + 2] += starts[k + 1];
    for (size_t i = 0; i < switched; ++i) {
        ReweightSwitchTarget const *target = &sw->targets[instance->switched[i].target];
        instance->switch_sorted[
            starts[(key_kind == 0 ? target->evaluation : target->target_process) + 1]++] = i;
    }
    return true;
}

/* The legs of a switch target in event slots. */
static uint32_t const *switch_target_legs(ReweightSwitchTarget const *target) {
    return target->legs == REWEIGHT_NULL
        ? reweight_provider.flavor_switch->evaluations[target->evaluation].legs
        : (uint32_t const *)(rw_pool + target->legs);
}

/* The momenta of the points sorted[0..count) in the legs of their evaluations, or of
 * their targets. */
static void gather_switched(Instance *instance, double const *momenta, size_t stride,
                            size_t const *sorted, size_t count, bool target_legs) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    size_t n = reweight_provider.particle_count;
    for (size_t i = 0; i < count; ++i) {
        SwitchedEvent const *s = &instance->switched[sorted[i]];
        ReweightSwitchTarget const *target = &sw->targets[s->target];
        ReweightSwitchEvaluation const *evaluation = &sw->evaluations[target->evaluation];
        uint32_t const *legs = target_legs ? switch_target_legs(target) : evaluation->legs;
        /* the targets are uncrossed */
        int32_t const *signs = target_legs ? NULL : evaluation->signs;
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k)
                instance->batch[4 * (n * i + j) + k] = (signs == NULL ? 1.0 : signs[j]) *
                    momenta[s->event + stride * (n * k + legs[j])];
    }
}

typedef struct {
    int const *vector;
    size_t n;
    uint32_t index;
} HelicityKey;

static int compare_helicity_keys(void const *a, void const *b) {
    HelicityKey const *x = a, *y = b;
    for (size_t j = 0; j < x->n; ++j)
        if (x->vector[j] != y->vector[j]) return x->vector[j] < y->vector[j] ? -1 : 1;
    return 0;
}

/* The runtime helicities of a switch evaluation ordered by their values in event
 * slots: the same physical order for every runtime that evaluates the flavour. */
static uint32_t const *helicity_order(Instance *instance, size_t e, RuntimeTables const *lc) {
    if (instance->helicity_orders[e] != NULL) return instance->helicity_orders[e];
    size_t n = reweight_provider.particle_count, count = lc->helicity_count;
    ReweightSwitchEvaluation const *evaluation = &reweight_provider.flavor_switch->evaluations[e];
    uint32_t const *legs = evaluation->legs;
    int32_t const *rho = evaluation->helicity_signs;
    int *vectors = malloc((count * n + 1) * sizeof(int));
    HelicityKey *keys = malloc((count + 1) * sizeof(*keys));
    uint32_t *order = malloc((count + 1) * sizeof(uint32_t));
    if (vectors == NULL || keys == NULL || order == NULL) {
        free(vectors); free(keys); free(order);
        return NULL;
    }
    for (size_t h = 0; h < count; ++h) {
        for (size_t j = 0; j < n; ++j)
            vectors[h * n + legs[j]] = (rho == NULL ? 1 : rho[j]) * lc->helicities[h * n + j];
        keys[h] = (HelicityKey){&vectors[h * n], n, (uint32_t)h};
    }
    qsort(keys, count, sizeof(*keys), compare_helicity_keys);
    for (size_t h = 0; h < count; ++h) order[h] = keys[h].index;
    free(vectors); free(keys);
    instance->helicity_orders[e] = order;
    return order;
}

/* The leading colour of a target's flows in physical order, then of the colours
 * without a switch flow (unmatched) in runtime order. */
static double switch_row(RuntimeTables const *lc, ReweightSwitchFlows const *flows,
                         ReweightSwitchTarget const *target, double const *values,
                         double *unmatched) {
    double row = 0.0, rest = 0.0;
    uint32_t const *order = (uint32_t const *)(rw_pool + target->flows);
    for (size_t k = 0; k < flows->count; ++k) {
        int c = lc->slot_colors[order[k]];
        if (c >= 0) row += values[c];
    }
    for (size_t c = 0; c < lc->color_count; ++c)
        if (lc->flow_slots[c] < 0) rest += values[c];
    if (unmatched != NULL) *unmatched = rest;
    return row + rest;
}

/* The helicity in proportion to the flows it sums, then a flow at it, from the
 * leading colour resolved at the event; both drawn in physical order. */
static UmamiStatus draw_switched(Instance *instance, SwitchedEvent *s, RuntimeTables const *lc,
                                 double const *resolved, double random_helicity,
                                 double random_color) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    ReweightSwitchTarget const *target = &sw->targets[s->target];
    ReweightSwitchEvaluation const *evaluation = &sw->evaluations[target->evaluation];
    ReweightSwitchFlows const *flows = &sw->flows[evaluation->process];
    size_t n = reweight_provider.particle_count, colors = lc->color_count;
    RuntimeTables *fc = NULL;
    UmamiStatus status = target_tables(instance, target->target_process, &fc);
    if (status != UMAMI_SUCCESS) return status;
    uint32_t const *order = helicity_order(instance, target->evaluation, lc);
    if (order == NULL) return UMAMI_ERROR;
    double total = 0.0;
    for (size_t k = 0; k < lc->helicity_count; ++k)
        total += switch_row(lc, flows, target, &resolved[order[k] * colors], NULL);
    if (!(total > 0.0)) return UMAMI_ERROR;
    size_t h = 0;
    double row = 0.0, unmatched = 0.0, threshold = random_helicity * total;
    for (size_t k = 0; k < lc->helicity_count; ++k) {
        h = order[k];
        row = switch_row(lc, flows, target, &resolved[h * colors], &unmatched);
        if (threshold < row || k + 1 == lc->helicity_count) break;
        threshold -= row;
    }
    double const *values = &resolved[h * colors];
    threshold = random_color * row;
    int chosen = -1;
    bool drawn = false;
    uint32_t const *flow_order = (uint32_t const *)(rw_pool + target->flows);
    for (size_t k = 0; !drawn && k < flows->count; ++k) {
        int c = lc->slot_colors[flow_order[k]];
        if (c < 0) continue;
        chosen = (int)k;
        drawn = threshold < values[c];
        threshold -= values[c];
    }
    /* past the flows: a colour without a switch flow (or rounding at the end) */
    if (!drawn && unmatched > 0.0) chosen = -1;
    /* the runtime helicity in event slots, rho applied at crossed slots */
    int const *vector = &lc->helicities[h * n];
    int32_t const *rho = evaluation->helicity_signs;
    for (size_t j = 0; j < n; ++j)
        instance->slot_helicities[evaluation->legs[j]] = (rho == NULL ? 1 : rho[j]) * vector[j];
    int exported = -1, fc_helicity = -1;
    for (size_t e = 0; exported < 0 && e < reweight_provider.exported_helicity_count; ++e) {
        bool equal = true;
        for (size_t j = 0; equal && j < n; ++j)
            equal = sw->helicities[e * n + j] == instance->slot_helicities[j];
        if (equal) exported = (int)e;
    }
    /* the target helicity of the same event-slot values, in the target's legs */
    uint32_t const *target_legs = switch_target_legs(target);
    for (size_t f = 0; fc_helicity < 0 && f < fc->helicity_count; ++f) {
        bool equal = true;
        for (size_t j = 0; equal && j < n; ++j)
            equal = fc->helicities[f * n + j] == instance->slot_helicities[target_legs[j]];
        if (equal) fc_helicity = (int)f;
    }
    if (exported < 0 || chosen < 0) {
        fprintf(stderr, "umami: switch target without exported helicity or flow\n");
        return UMAMI_ERROR;
    }
    s->row = row;
    s->exported_helicity = exported;
    s->fc_helicity = fc_helicity;
    s->color = rw_pool[target->colors + (uint32_t)chosen];
    return UMAMI_SUCCESS;
}

/* The switched events: their helicity and colour from the leading colour resolved
 * in batches per evaluation, then the full colour at the target in batches per
 * target process (zero at a helicity the target does not have). */
static UmamiStatus evaluate_switched(Instance *instance, double const *momenta, size_t stride,
                                     double const *random_helicity, double const *random_color,
                                     size_t switched) {
    ReweightSwitch const *sw = reweight_provider.flavor_switch;
    size_t n = reweight_provider.particle_count;
    if (switched == 0) return UMAMI_SUCCESS;
    if (!sort_switched(instance, switched, 0)) return UMAMI_ERROR;
    for (size_t e = 0; e < sw->evaluation_count; ++e) {
        size_t process = sw->evaluations[e].process;
        RuntimeTables *lc = NULL;
        size_t begin = instance->switch_starts[e], end = instance->switch_starts[e + 1];
        if (begin == end) continue;
        UmamiStatus status = reference_tables(instance, process, &lc);
        if (status != UMAMI_SUCCESS) return status;
        size_t size = lc->helicity_count * lc->color_count;
        for (size_t start = begin; start < end; start += UMAMI_CHUNK) {
            size_t points = end - start < UMAMI_CHUNK ? end - start : UMAMI_CHUNK;
            if (size > SIZE_MAX / points ||
                !reserve(&instance->resolved, &instance->resolved_capacity, points * size))
                return UMAMI_ERROR;
            size_t const *sorted = &instance->switch_sorted[start];
            gather_switched(instance, momenta, stride, sorted, points, false);
            size_t helicities = 0, colors = 0;
            int assignment = sw->evaluations[e].assignment;
            if (assignment >= 0) {
                /* [helicity][flow] of the flavour-union assignment */
                for (size_t i = 0; i <= points; ++i) instance->offsets[i] = i;
                for (size_t i = 0; i < points; ++i)
                    instance->assignments[i] = (uint32_t)assignment;
                status = native_error(rusticol_runtime_evaluate_assignments_f64(
                    instance->reference[process], instance->batch, 4 * n * points, points,
                    instance->offsets, points + 1, instance->assignments, points, NULL, 0,
                    NULL, 0, 2, instance->resolved, points * size));
            } else {
                status = native_error(rusticol_runtime_evaluate_resolved_f64(
                    instance->reference[process], instance->batch, 4 * n * points, points,
                    NULL, 0, NULL, 0, instance->resolved, points * size, &helicities,
                    &colors));
            }
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < points; ++i) {
                SwitchedEvent *s = &instance->switched[sorted[i]];
                status = draw_switched(instance, s, lc, &instance->resolved[i * size],
                                       random_helicity[s->event], random_color[s->event]);
                if (status != UMAMI_SUCCESS) return status;
            }
        }
    }
    if (!sort_switched(instance, switched, 1)) return UMAMI_ERROR;
    for (size_t t = 0; t < reweight_provider.target_process_count; ++t) {
        size_t begin = instance->switch_starts[t], end = instance->switch_starts[t + 1];
        for (size_t start = begin; start < end; start += UMAMI_CHUNK) {
            size_t points = end - start < UMAMI_CHUNK ? end - start : UMAMI_CHUNK, kept = 0;
            for (size_t i = 0; i < points; ++i) {
                size_t position = instance->switch_sorted[start + i];
                SwitchedEvent *s = &instance->switched[position];
                s->value = 0.0;
                if (s->fc_helicity < 0) continue;
                instance->positions[kept] = position;
                instance->helicities[kept++] = (uint32_t)s->fc_helicity;
            }
            if (kept == 0) continue;
            UmamiStatus status = ensure_loaded(instance, 1, t);
            if (status != UMAMI_SUCCESS) return status;
            gather_switched(instance, momenta, stride, instance->positions, kept, true);
            if (sw->targets[instance->switched[instance->positions[0]].target].assignment >= 0) {
                /* the targets of a template process are all its assignments */
                for (size_t i = 0; i <= kept; ++i) instance->offsets[i] = i;
                for (size_t i = 0; i < kept; ++i)
                    instance->assignments[i] = (uint32_t)
                        sw->targets[instance->switched[instance->positions[i]].target].assignment;
                status = native_error(rusticol_runtime_evaluate_assignments_f64(
                    instance->target[t], instance->batch, 4 * n * kept, kept, instance->offsets,
                    kept + 1, instance->assignments, kept, instance->helicities, kept, NULL, 0,
                    0, instance->chunk_values, kept));
            } else {
                status = native_error(rusticol_runtime_evaluate_selected_f64(
                    instance->target[t], instance->batch, 4 * n * kept, kept, NULL, 0, NULL, 0,
                    instance->helicities, kept, NULL, 0, instance->chunk_values, kept));
            }
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < kept; ++i) {
                SwitchedEvent *s = &instance->switched[instance->positions[i]];
                s->value = sw->constants[sw->targets[s->target].factor] *
                    instance->chunk_values[i];
            }
        }
    }
    return UMAMI_SUCCESS;
}

/* ---- batched evaluation ------------------------------------------------ */

/* Values of one process of a side at the points events[0..count), each through the
 * permutation of its flavour (and its flavour-union assignment), at the runtime
 * helicities[point], natively batched over runs of equal alpha_s (or alpha_s NULL). */
static UmamiStatus evaluate_points(Instance *instance, int side, RusticolRuntimeHandle *runtime,
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
            status = evaluate_points(instance, side, runtime, process, entries, momenta, stride,
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
        for (size_t i = 0; i < size; ++i) {
            ReweightFlavour const *entry = entries[start + i];
            /* crossed union members read sign-flipped momenta (q = sigma p) */
            int32_t const *signs = side == 0 ? entry->reference_signs : entry->target_signs;
            uint32_t const *permutation =
                side == 0 ? entry->permutation : entry->target_permutation;
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < 4; ++k)
                    instance->batch[4 * (n * i + j) + k] = (signs == NULL ? 1.0 : signs[j]) *
                        momenta[events[start + i] + stride * (n * k + permutation[j])];
        }
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
        int first = side == 0 ? entries[start]->reference_assignment
                              : entries[start]->target_assignment;
        if (first >= 0) {
            /* a template's entries are all assignments */
            for (size_t i = 0; i <= size; ++i) instance->offsets[i] = i;
            for (size_t i = 0; i < size; ++i)
                instance->assignments[i] = (uint32_t)(side == 0
                    ? entries[start + i]->reference_assignment
                    : entries[start + i]->target_assignment);
            status = native_error(rusticol_runtime_evaluate_assignments_f64(
                runtime, instance->batch, 4 * n * size, size, instance->offsets, size + 1,
                instance->assignments, size, helicities + start, size, NULL, 0, 0,
                values + start, size));
        } else {
            status = native_error(rusticol_runtime_evaluate_selected_f64(
                runtime, instance->batch, 4 * n * size, size, NULL, 0, NULL, 0,
                helicities + start, size, NULL, 0, values + start, size));
        }
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
                int32_t const *map =
                    side == 0 ? entry->reference_helicity : entry->target_helicity;
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
                instance, side, side == 0 ? instance->reference[index] : instance->target[index],
                side == 0 ? &reweight_provider.reference_processes[index]
                          : &reweight_provider.target_processes[index],
                instance->entries, in[UMAMI_IN_MOMENTA], stride, instance->events, kept,
                instance->helicities, in[UMAMI_IN_ALPHA_S], instance->chunk_values);
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < kept; ++i)
                result[instance->positions[i]] = side == 0 ? instance->chunk_values[i]
                    : instance->entries[i]->target_factor * instance->chunk_values[i];
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
    if (switching && count > instance->switched_capacity) {
        if (count > SIZE_MAX / sizeof(*instance->switched)) return UMAMI_ERROR;
        SwitchedEvent *grown = realloc(instance->switched, count * sizeof(*grown));
        if (grown == NULL) return UMAMI_ERROR;
        instance->switched = grown;
        instance->switched_capacity = count;
    }
    if (!reserve_indices(&instance->order, &instance->order_capacity, count) ||
        !reserve_indices(&instance->sorted, &instance->sorted_capacity, count) ||
        !reserve(&instance->enhancements, &instance->enhancements_capacity, count) ||
        !reserve(&instance->side_values[0], &instance->side_capacity[0], count) ||
        !reserve(&instance->side_values[1], &instance->side_capacity[1], count))
        return UMAMI_ERROR;
    size_t pending = 0, switched_count = 0;
    for (size_t event = offset; event < offset+count; ++event) {
        double enhancement = 1.0;
        bool switched = false;
        if (switching) {
            ReweightSwitch const *sw = reweight_provider.flavor_switch;
            size_t index = event - offset;
            /* the event's non-partons keep their slots in every flavour summed */
            size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
                ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
            int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
                ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
            if (channel >= reweight_provider.channel_count || flavour < 0 ||
                (size_t)flavour >= reweight_provider.channels[channel].flavour_count)
                return UMAMI_ERROR;
            int arrangement =
                reweight_provider.channels[channel].flavours[flavour].switch_arrangement;
            double generated = 0.0, dropped = 0.0;
            for (size_t o = 0; o < sw->ordering_count; ++o) {
                if (sw->orderings[o].arrangement != arrangement) continue;
                double share = switch_share(instance, &sw->orderings[o], densities, event,
                                            index, count, stride);
                if (sw->orderings[o].target < 0) generated += share;
                else dropped += share;
            }
            if (!isfinite(generated) || !isfinite(dropped)) return UMAMI_ERROR;
            /* no leading colour of its own (zero weight): nothing to switch from */
            if (generated > 0.0) enhancement = 1.0 + dropped / generated;
            else dropped = 0.0;
            double u = ((double const *)in[UMAMI_IN_RANDOM_FLAVOR])[event] * enhancement;
            if (u >= 1.0 && dropped > 0.0) {
                double threshold = (u - 1.0) * generated;
                int target = -1;
                for (size_t o = 0; o < sw->ordering_count; ++o) {
                    if (sw->orderings[o].target < 0 ||
                        sw->orderings[o].arrangement != arrangement) continue;
                    target = sw->orderings[o].target;
                    threshold -= switch_share(instance, &sw->orderings[o], densities, event,
                                              index, count, stride);
                    if (threshold < 0.0) break;
                }
                for (size_t k = 0; k < 4 * n; ++k)
                    if (!isfinite(momenta[event + stride * k])) return UMAMI_ERROR;
                instance->switched[switched_count++] = (SwitchedEvent){
                    .event = event, .target = target, .enhancement = enhancement};
                switched = true;
            }
        }
        if (out[UMAMI_OUT_FLAVOR_INDEX]) ((int *)out[UMAMI_OUT_FLAVOR_INDEX])[event] = -1;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = -1;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = -1;
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
    UmamiStatus switch_status = evaluate_switched(
        instance, momenta, stride, in[UMAMI_IN_RANDOM_HELICITY], in[UMAMI_IN_RANDOM_COLOR],
        switched_count);
    if (switch_status != UMAMI_SUCCESS) return switch_status;
    for (size_t i = 0; i < switched_count; ++i) {
        SwitchedEvent const *s = &instance->switched[i];
        double ratio = s->enhancement * (s->row > 0.0 ? s->value / s->row : 0.0);
        if (!isfinite(ratio)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])
            ((double *)out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])[s->event] = ratio;
        if (out[UMAMI_OUT_FLAVOR_INDEX]) ((int *)out[UMAMI_OUT_FLAVOR_INDEX])[s->event] = s->target;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[s->event] = s->color;
        if (out[UMAMI_OUT_HELICITY_INDEX])
            ((int *)out[UMAMI_OUT_HELICITY_INDEX])[s->event] = s->exported_helicity;
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
        free(instance->reference_tables[i].slot_colors);
    }
    for (size_t e = 0; instance->helicity_orders != NULL &&
                       reweight_provider.flavor_switch != NULL &&
                       e < reweight_provider.flavor_switch->evaluation_count; ++e)
        free(instance->helicity_orders[e]);
    free(instance->helicity_orders);
    for (size_t i = 0; instance->target_tables != NULL &&
                       i < reweight_provider.target_process_count; ++i)
        free(instance->target_tables[i].helicities);
    free(instance->reference_tables); free(instance->target_tables);
    free(instance->values); free(instance->batch); free(instance->resolved);
    free(instance->order); free(instance->sorted); free(instance->flavours);
    free(instance->enhancements); free(instance->side_values[0]); free(instance->side_values[1]);
    free(instance->starts); free(instance->cursor); free(instance->chunk_values);
    free(instance->events); free(instance->positions); free(instance->entries);
    free(instance->helicities); free(instance->offsets); free(instance->assignments);
    free(instance->call_offsets); free(instance->call_ids); free(instance->call_values);
    free(instance->slot_helicities);
    if (instance->relabelled != NULL && reweight_provider.flavor_switch != NULL)
        for (size_t g = 0; g < reweight_provider.flavor_switch->call_count; ++g)
            free(instance->relabelled[g]);
    free(instance->relabelled);
    free(instance->switched); free(instance->switch_sorted); free(instance->switch_starts);
    free(instance->overrides); free(instance->base);
    free(instance->reference); free(instance->target);
    /* last: the loops above read the provider's counts */
    bool acquired = instance->data_acquired;
    free(instance);
    if (acquired) reweight_data_release();
    return UMAMI_SUCCESS;
}
