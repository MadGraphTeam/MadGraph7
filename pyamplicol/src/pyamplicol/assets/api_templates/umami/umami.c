/* SPDX-License-Identifier: 0BSD */
#include "umami.h"
#include "umami_provider.h"
#include <rusticol.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef UMAMI_PROVIDER_HEADER
#define UMAMI_PROVIDER_HEADER "provider_p0.h"
#endif
#include UMAMI_PROVIDER_HEADER

/* Events are evaluated in batches of at most this many points per native call. */
#define UMAMI_CHUNK 512

typedef struct {
    char *name;
    double real, imaginary;
} Override;

typedef struct {
    char *artifact;
    RusticolRuntimeHandle **runtimes;
    /* Parameters set so far, applied to the processes loaded later. */
    Override *overrides;
    size_t override_count;
    /* Flat (channel, flavour) selections: offset of each channel, and their count. */
    size_t *selection_offsets, selection_count;
    /* Flat contributions: offset of each selection, and the group of each one.
     * Contributions of one group share their runtime process and colour flow. */
    size_t *contribution_offsets, *groups, group_count;
    /* Per call: the (event, contribution) pairs, their group, their order sorted
     * by group, and the total of every event. */
    size_t *pair_events, *pair_positions, *keys, *sorted, pair_capacity;
    UmamiContribution const **pair_entries;
    double *totals;
    size_t totals_capacity;
    size_t *starts, *cursor;
    /* Per chunk of UMAMI_CHUNK pairs. */
    double *batch, *values;
    size_t *events, *positions;
    UmamiContribution const **entries;
    uint32_t *helicities;
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT] = true,
    [UMAMI_OUT_COLOR_INDEX] = UMAMI_HAS_COLOR_FLOW,
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

static UmamiParameter const *parameter(size_t process, char const *name) {
    UmamiProcess const *entry = &umami_provider.processes[process];
    for (size_t i = 0; i < entry->parameter_count; ++i)
        if (strcmp(entry->parameters[i].name, name) == 0) return &entry->parameters[i];
    return NULL;
}

static UmamiStatus load_process(Instance *instance, size_t process) {
    if (instance->runtimes[process] != NULL) return UMAMI_SUCCESS;
    UmamiProcess const *entry = &umami_provider.processes[process];
    RusticolRuntimeHandle *runtime = NULL;
    int status = rusticol_runtime_load(instance->artifact, entry->key, NULL, &runtime);
    if (status != RUSTICOL_STATUS_OK) return native_error(status);
    size_t count = 0, required = 0;
    char accuracy[16];
    status = rusticol_runtime_external_count(runtime, &count);
    bool compatible = status == RUSTICOL_STATUS_OK && count == umami_provider.particle_count;
    char const *expected_identity = UMAMI_ARTIFACT_ID;
    if (compatible && expected_identity != NULL) {
        size_t capacity = strlen(expected_identity) + 1;
        char *identity = malloc(capacity);
        if (identity == NULL) compatible = false;
        else {
            status = rusticol_runtime_artifact_id(runtime, identity, capacity, &required);
            compatible = status == RUSTICOL_STATUS_OK && strcmp(identity, expected_identity) == 0;
            free(identity);
        }
    }
    for (size_t j = 0; compatible && j < count; ++j) {
        int32_t pdg = 0;
        status = rusticol_runtime_external_pdg(runtime, j, &pdg);
        compatible = status == RUSTICOL_STATUS_OK && pdg == entry->pdgs[j];
    }
    if (compatible) {
        status = rusticol_runtime_color_accuracy(runtime, accuracy, sizeof(accuracy), &required);
        compatible = status == RUSTICOL_STATUS_OK && strcmp(accuracy, entry->color_accuracy) == 0;
    }
    if (compatible) {
        status = rusticol_runtime_helicity_count(runtime, &count);
        compatible = status == RUSTICOL_STATUS_OK && count == entry->helicity_count;
    }
    for (size_t j = 0; compatible && j < entry->helicity_count; ++j) {
        size_t capacity = strlen(entry->helicity_ids[j]) + 1;
        char *identifier = malloc(capacity);
        if (identifier == NULL) { compatible = false; break; }
        status = rusticol_runtime_helicity_id(runtime, j, identifier, capacity, &required);
        compatible = status == RUSTICOL_STATUS_OK && strcmp(identifier, entry->helicity_ids[j]) == 0;
        free(identifier);
    }
    if (!compatible) {
        (void)rusticol_runtime_free(runtime);
        fprintf(stderr, "umami: artifact is incompatible with provider %s, process %s\n",
                umami_provider.id, entry->key);
        return UMAMI_ERROR;
    }
    for (size_t i = 0; i < instance->override_count; ++i) {
        Override const *override = &instance->overrides[i];
        UmamiParameter const *known = parameter(process, override->name);
        if (known == NULL) continue;
        status = rusticol_runtime_set_model_parameter(runtime, override->name,
            override->real, known->is_complex ? override->imaginary : 0.0);
        if (status != RUSTICOL_STATUS_OK) {
            (void)rusticol_runtime_free(runtime);
            return native_error(status);
        }
    }
    instance->runtimes[process] = runtime;
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

UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)umami_provider.particle_count; break;
    case UMAMI_META_HELICITY_COUNT: *(int *)result = (int)umami_provider.helicity_count; break;
    case UMAMI_META_COLOR_COUNT:
        if (!UMAMI_HAS_COLOR_FLOW) return UMAMI_ERROR_UNSUPPORTED_META;
        *(int *)result = (int)umami_provider.color_count; break;
    case UMAMI_META_MASSES:
        memcpy(result, umami_provider.masses, umami_provider.particle_count * sizeof(double));
        break;
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

static int same_runtime(UmamiContribution const *a, UmamiContribution const *b) {
    if (a->process_index != b->process_index)
        return a->process_index < b->process_index ? -1 : 1;
    if (a->color_id == NULL || b->color_id == NULL)
        return (a->color_id != NULL) - (b->color_id != NULL);
    return strcmp(a->color_id, b->color_id);
}

typedef struct {
    UmamiContribution const *entry;
    size_t index;
} Ranked;

static int compare_ranked(void const *a, void const *b) {
    Ranked const *left = a, *right = b;
    int order = same_runtime(left->entry, right->entry);
    return order != 0 ? order : (left->index > right->index) - (left->index < right->index);
}

/* Groups the contributions that share their runtime process and colour flow. */
static UmamiStatus assign_groups(Instance *instance) {
    size_t total = 0, s = 0;
    for (size_t c = 0; c < umami_provider.channel_count; ++c)
        for (size_t f = 0; f < umami_provider.channels[c].flavour_count; ++f) {
            instance->contribution_offsets[s++] = total;
            total += umami_provider.channels[c].flavours[f].contribution_count;
        }
    instance->contribution_offsets[s] = total;
    instance->groups = calloc(total + 1, sizeof(size_t));
    Ranked *ranked = calloc(total + 1, sizeof(Ranked));
    if (instance->groups == NULL || ranked == NULL) { free(ranked); return UMAMI_ERROR; }
    size_t i = 0;
    for (size_t c = 0; c < umami_provider.channel_count; ++c)
        for (size_t f = 0; f < umami_provider.channels[c].flavour_count; ++f) {
            UmamiFlavour const *flavour = &umami_provider.channels[c].flavours[f];
            for (size_t j = 0; j < flavour->contribution_count; ++j, ++i)
                ranked[i] = (Ranked){&flavour->contributions[j], i};
        }
    qsort(ranked, total, sizeof(Ranked), compare_ranked);
    for (i = 0; i < total; ++i) {
        if (i > 0 && same_runtime(ranked[i - 1].entry, ranked[i].entry) != 0)
            ++instance->group_count;
        instance->groups[ranked[i].index] = instance->group_count;
    }
    if (total > 0) ++instance->group_count;
    free(ranked);
    return UMAMI_SUCCESS;
}

UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    if (handle == NULL) return UMAMI_ERROR;
    *handle = NULL;
    struct stat info;
    if (param_card_path == NULL || param_card_path[0] != '/' ||
        stat(param_card_path, &info) != 0 || !S_ISDIR(info.st_mode) ||
        umami_provider.process_count == 0 || umami_provider.particle_count > INT_MAX ||
        umami_provider.particle_count > SIZE_MAX / (4 * sizeof(double) * UMAMI_CHUNK))
        return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    size_t length = strlen(param_card_path) + 1;
    instance->artifact = malloc(length);
    instance->runtimes = calloc(umami_provider.process_count, sizeof(*instance->runtimes));
    instance->selection_offsets = calloc(umami_provider.channel_count + 1, sizeof(size_t));
    for (size_t c = 0; instance->selection_offsets != NULL && c < umami_provider.channel_count; ++c) {
        instance->selection_offsets[c] = instance->selection_count;
        instance->selection_count += umami_provider.channels[c].flavour_count;
    }
    size_t chunk = UMAMI_CHUNK, width = umami_provider.helicity_count + 1;
    instance->contribution_offsets = calloc(instance->selection_count + 1, sizeof(size_t));
    if (instance->artifact == NULL || instance->runtimes == NULL ||
        instance->selection_offsets == NULL || instance->contribution_offsets == NULL ||
        assign_groups(instance) != UMAMI_SUCCESS) {
        umami_free(instance); return UMAMI_ERROR;
    }
    instance->starts = calloc(instance->group_count + 1, sizeof(size_t));
    instance->cursor = calloc(instance->group_count + 1, sizeof(size_t));
    instance->batch = calloc(4 * umami_provider.particle_count * chunk, sizeof(double));
    instance->values = calloc(width * chunk, sizeof(double));
    instance->events = calloc(chunk, sizeof(size_t));
    instance->positions = calloc(chunk, sizeof(size_t));
    instance->entries = calloc(chunk, sizeof(*instance->entries));
    instance->helicities = calloc(chunk, sizeof(uint32_t));
    if (instance->starts == NULL || instance->cursor == NULL || instance->batch == NULL ||
        instance->values == NULL || instance->events == NULL || instance->positions == NULL ||
        instance->entries == NULL || instance->helicities == NULL) {
        umami_free(instance); return UMAMI_ERROR;
    }
    memcpy(instance->artifact, param_card_path, length);
    /* Validate the artifact now; other processes are loaded lazily on first use. */
    UmamiStatus status = load_process(instance, 0);
    if (status != UMAMI_SUCCESS) { umami_free(instance); return status; }
    *handle = instance;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                              double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    Instance *instance = handle;
    for (size_t i = 0; i < umami_provider.process_count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        if (entry->is_complex && imaginary == NULL) return UMAMI_ERROR;
        UmamiStatus status = load_process(instance, i);
        if (status != UMAMI_SUCCESS) return status;
        double re, im;
        status = native_error(rusticol_runtime_get_model_parameter(instance->runtimes[i], name, &re, &im));
        if (status != UMAMI_SUCCESS) return status;
        *real = re;
        if (imaginary != NULL) *imaginary = im;
        return UMAMI_SUCCESS;
    }
    return UMAMI_ERROR;
}

UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                              double real, double imaginary) {
    if (handle == NULL || name == NULL || !isfinite(real)) return UMAMI_ERROR;
    Instance *instance = handle;
    size_t count = umami_provider.process_count;
    double *previous = calloc(count * 2, sizeof(double));
    bool *changed = calloc(count, sizeof(bool));
    if (previous == NULL || changed == NULL) { free(previous); free(changed); return UMAMI_ERROR; }
    UmamiStatus status = UMAMI_ERROR;
    bool found = false;
    for (size_t i = 0; i < count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        /* Load the first process using it, so that the value is validated now. */
        if (!found && load_process(instance, i) != UMAMI_SUCCESS) goto rollback;
        found = true;
        if (!entry->mutable || (entry->is_complex && !isfinite(imaginary))) goto rollback;
        if (instance->runtimes[i] == NULL) continue;
        status = native_error(rusticol_runtime_get_model_parameter(
            instance->runtimes[i], name, &previous[2*i], &previous[2*i+1]));
        if (status != UMAMI_SUCCESS) goto rollback;
        status = native_error(rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, real, entry->is_complex ? imaginary : 0.0));
        if (status != UMAMI_SUCCESS) goto rollback;
        changed[i] = true;
    }
    if (found && record_override(instance, name, real, imaginary) != UMAMI_SUCCESS)
        goto rollback;
    free(previous); free(changed);
    return found ? UMAMI_SUCCESS : UMAMI_ERROR;
rollback:
    for (size_t i = 0; i < count; ++i)
        if (changed[i]) (void)rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, previous[2*i], previous[2*i+1]);
    free(previous); free(changed);
    return UMAMI_ERROR;
}

static bool grow_indices(size_t **buffer, size_t count) {
    size_t *grown = realloc(*buffer, count * sizeof(size_t));
    if (grown != NULL) *buffer = grown;
    return grown != NULL;
}

/* Grows the per-call arrays to at least pairs (event, contribution) pairs and events. */
static bool reserve(Instance *instance, size_t pairs, size_t events) {
    if (pairs > instance->pair_capacity) {
        size_t capacity = pairs > 2 * instance->pair_capacity ? pairs : 2 * instance->pair_capacity;
        if (capacity > SIZE_MAX / sizeof(size_t)) return false;
        if (!grow_indices(&instance->pair_events, capacity) ||
            !grow_indices(&instance->pair_positions, capacity) ||
            !grow_indices(&instance->keys, capacity) || !grow_indices(&instance->sorted, capacity))
            return false;
        UmamiContribution const **entries =
            realloc(instance->pair_entries, capacity * sizeof(*entries));
        if (entries == NULL) return false;
        instance->pair_entries = entries;
        instance->pair_capacity = capacity;
    }
    if (events > instance->totals_capacity) {
        if (events > SIZE_MAX / sizeof(double)) return false;
        double *totals = realloc(instance->totals, events * sizeof(double));
        if (totals == NULL) return false;
        instance->totals = totals;
        instance->totals_capacity = events;
    }
    return true;
}

/* One native evaluation of the points events[0..count) of one group, each through its
 * own contribution, which share one alpha_s (or NULL). With resolved, values gets
 * [point][runtime helicity], otherwise the totals, at helicities[point] if not NULL. */
static UmamiStatus evaluate_points(Instance *instance, UmamiContribution const *const *entries,
                                   double const *momenta, size_t stride,
                                   size_t const *events, size_t count,
                                   uint32_t const *helicities, double const *alpha_s,
                                   bool resolved, double *values) {
    UmamiContribution const *first = entries[0];
    size_t process = first->process_index;
    UmamiProcess const *description = &umami_provider.processes[process];
    RusticolRuntimeHandle *runtime = instance->runtimes[process];
    size_t n = umami_provider.particle_count;
    double *batch = instance->batch;
    for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k)
                batch[4*(n*i + j) + k] =
                    momenta[events[i] + stride * (n*k + entries[i]->permutation[j])];
    double previous_real = 0.0, previous_imaginary = 0.0;
    bool changed = false;
    UmamiStatus status;
    if (alpha_s != NULL) {
        status = native_error(rusticol_runtime_get_model_parameter(
            runtime, description->alpha_s_name, &previous_real, &previous_imaginary));
        if (status != UMAMI_SUCCESS) return status;
        changed = previous_real != *alpha_s || previous_imaginary != 0.0;
        if (changed) {
            status = native_error(rusticol_runtime_set_model_parameter(
                runtime, description->alpha_s_name, *alpha_s, 0.0));
            if (status != UMAMI_SUCCESS) return status;
        }
    }
    char const *const *color = first->color_id == NULL ? NULL : &first->color_id;
    size_t color_count = first->color_id == NULL ? 0 : 1;
    int native_status;
    if (resolved) {
        size_t helicity_count = 0, resolved_colors = 0;
        native_status = rusticol_runtime_evaluate_resolved_f64(
            runtime, batch, 4*n*count, count, NULL, 0, color, color_count,
            values, count * description->helicity_count, &helicity_count, &resolved_colors);
        if (native_status == RUSTICOL_STATUS_OK &&
            (helicity_count != description->helicity_count || resolved_colors != 1)) {
            fprintf(stderr, "umami: unexpected resolved shape of process %s\n", description->key);
            native_status = RUSTICOL_STATUS_RUNTIME_ERROR;
        }
    } else {
        native_status = rusticol_runtime_evaluate_selected_f64(
            runtime, batch, 4*n*count, count, NULL, 0, color, color_count,
            helicities, helicities == NULL ? 0 : count, NULL, 0, values, count);
    }
    status = native_error(native_status);
    if (changed) {
        UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
            runtime, description->alpha_s_name, previous_real, previous_imaginary));
        if (restored != UMAMI_SUCCESS) return restored;
    }
    return status;
}

/* evaluate_points over runs of events with equal alpha_s; values as there. */
static UmamiStatus evaluate_runs(Instance *instance, UmamiContribution const *const *entries,
                                 double const *momenta, size_t stride,
                                 size_t const *events, size_t count,
                                 uint32_t const *helicities, double const *alpha_s,
                                 bool resolved, double *values) {
    UmamiProcess const *description = &umami_provider.processes[entries[0]->process_index];
    size_t width = resolved ? description->helicity_count : 1;
    if (alpha_s != NULL && description->alpha_s_power >= 0) {
        /* ME(alpha_s) = (alpha_s / alpha_s_0)^power ME(alpha_s_0), proven at generation */
        double reference = 0.0, imaginary = 0.0;
        UmamiStatus status = native_error(rusticol_runtime_get_model_parameter(
            instance->runtimes[entries[0]->process_index], description->alpha_s_name,
            &reference, &imaginary));
        if (status != UMAMI_SUCCESS) return status;
        if (reference > 0.0 && imaginary == 0.0) {
            status = evaluate_points(instance, entries, momenta, stride, events, count,
                                     helicities, NULL, resolved, values);
            if (status != UMAMI_SUCCESS) return status;
            for (size_t i = 0; i < count; ++i) {
                double scale = pow(alpha_s[events[i]] / reference, description->alpha_s_power);
                for (size_t h = 0; h < width; ++h) values[i * width + h] *= scale;
            }
            return UMAMI_SUCCESS;
        }
    }
    for (size_t start = 0, end; start < count; start = end) {
        end = count;
        if (alpha_s != NULL)
            for (end = start + 1; end < count && alpha_s[events[end]] == alpha_s[events[start]];)
                ++end;
        UmamiStatus status = evaluate_points(
            instance, entries + start, momenta, stride, events + start, end - start,
            helicities == NULL ? NULL : helicities + start,
            alpha_s == NULL ? NULL : &alpha_s[events[start]], resolved, values + start * width);
        if (status != UMAMI_SUCCESS) return status;
    }
    return UMAMI_SUCCESS;
}

/* Adds the pairs[0..count) of one group, at most UMAMI_CHUNK of them, to the totals. */
static UmamiStatus evaluate_group(Instance *instance, void const *const *in, void *const *out,
                                  size_t const *pairs, size_t count, size_t stride) {
    double const *momenta = in[UMAMI_IN_MOMENTA];
    double const *alpha_s = in[UMAMI_IN_ALPHA_S];
    int const *fixed = in[UMAMI_IN_HELICITY_INDEX];
    bool sample = fixed == NULL && out[UMAMI_OUT_HELICITY_INDEX] != NULL;
    size_t kept = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t pair = pairs[i];
        UmamiContribution const *entry = instance->pair_entries[pair];
        if (fixed != NULL) {
            /* a fixed exported helicity, absent ones do not contribute */
            int runtime = entry->helicity_runtime[fixed[instance->pair_events[pair]]];
            if (runtime < 0) continue;
            instance->helicities[kept] = (uint32_t)runtime;
        }
        instance->events[kept] = instance->pair_events[pair];
        instance->positions[kept] = instance->pair_positions[pair];
        instance->entries[kept++] = entry;
    }
    if (kept == 0) return UMAMI_SUCCESS;
    UmamiStatus status = load_process(instance, instance->entries[0]->process_index);
    if (status != UMAMI_SUCCESS) return status;
    double *values = instance->values, *totals = instance->totals;
    status = evaluate_runs(instance, instance->entries, momenta, stride, instance->events, kept,
                           fixed == NULL ? NULL : instance->helicities, alpha_s, sample, values);
    if (status != UMAMI_SUCCESS) return status;
    if (!sample) {
        for (size_t i = 0; i < kept; ++i)
            totals[instance->positions[i]] += instance->entries[i]->factor * values[i];
        return UMAMI_SUCCESS;
    }
    /* sample the helicity in proportion to its contribution */
    double const *random = in[UMAMI_IN_RANDOM_HELICITY];
    size_t width = umami_provider.processes[instance->entries[0]->process_index].helicity_count;
    int *labels = out[UMAMI_OUT_HELICITY_INDEX];
    for (size_t i = 0; i < kept; ++i) {
        double const *row = &values[i * width];
        double total = 0.0;
        for (size_t h = 0; h < width; ++h) total += row[h];
        size_t event = instance->events[i];
        double target = (random ? random[event] : 0.5) * total, cumulative = 0.0;
        size_t chosen = 0;
        for (size_t h = 0; h < width; ++h) {
            if (row[h] <= 0.0) continue;
            chosen = h;
            cumulative += row[h];
            if (cumulative > target) break;
        }
        labels[event] = (int)instance->entries[i]->helicity_export[chosen];
        totals[instance->positions[i]] += instance->entries[i]->factor * total;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride-offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL)) ||
        (stride && umami_provider.particle_count > SIZE_MAX / stride / 4)) return UMAMI_ERROR;
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
    if (in[UMAMI_IN_MOMENTA] == NULL) return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    size_t n = umami_provider.particle_count;
    /* Validate every event and list its (event, contribution) pairs by group. */
    size_t *starts = instance->starts, pairs = 0;
    for (size_t g = 0; g <= instance->group_count; ++g) starts[g] = 0;
    double const *momenta = in[UMAMI_IN_MOMENTA];
    for (size_t i = 0; i < count; ++i) {
        size_t event = offset + i;
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
            ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
            ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        if (channel >= umami_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= umami_provider.channels[channel].flavour_count) return UMAMI_ERROR;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        if (selection->contribution_count == 0) return UMAMI_ERROR;
        if (in[UMAMI_IN_HELICITY_INDEX]) {
            int label = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
            if (label < 0 || (size_t)label >= umami_provider.exported_helicity_count)
                return UMAMI_ERROR;
        } else if (out[UMAMI_OUT_HELICITY_INDEX] && selection->contribution_count != 1) {
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        }
        if (in[UMAMI_IN_RANDOM_HELICITY]) {
            double random = ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event];
            if (!isfinite(random) || random < 0.0 || random >= 1.0) return UMAMI_ERROR;
        }
        if (in[UMAMI_IN_ALPHA_S]) {
            double alpha_s = ((double const *)in[UMAMI_IN_ALPHA_S])[event];
            for (size_t j = 0; j < selection->contribution_count; ++j)
                if (umami_provider.processes[selection->contributions[j].process_index]
                        .alpha_s_name == NULL) return UMAMI_ERROR_UNSUPPORTED_INPUT;
            if (!isfinite(alpha_s) || alpha_s < 0.0) return UMAMI_ERROR;
        }
        for (size_t k = 0; k < 4 * n; ++k)
            if (!isfinite(momenta[event + stride * k])) return UMAMI_ERROR;
        if (!reserve(instance, pairs + selection->contribution_count, count)) return UMAMI_ERROR;
        size_t first = instance->contribution_offsets[
            instance->selection_offsets[channel] + (size_t)flavour];
        for (size_t j = 0; j < selection->contribution_count; ++j, ++pairs) {
            size_t group = instance->groups[first + j];
            instance->pair_events[pairs] = event;
            instance->pair_positions[pairs] = i;
            instance->pair_entries[pairs] = &selection->contributions[j];
            instance->keys[pairs] = group;
            ++starts[group + 1];
        }
        instance->totals[i] = 0.0;
    }
    for (size_t g = 0; g < instance->group_count; ++g) starts[g + 1] += starts[g];
    for (size_t g = 0; g < instance->group_count; ++g) instance->cursor[g] = starts[g];
    for (size_t p = 0; p < pairs; ++p) instance->sorted[instance->cursor[instance->keys[p]]++] = p;
    for (size_t g = 0; g < instance->group_count; ++g)
        for (size_t start = starts[g]; start < starts[g + 1]; start += UMAMI_CHUNK) {
            size_t size = starts[g + 1] - start;
            if (size > UMAMI_CHUNK) size = UMAMI_CHUNK;
            UmamiStatus status =
                evaluate_group(instance, in, out, &instance->sorted[start], size, stride);
            if (status != UMAMI_SUCCESS) return status;
        }
    int const *fixed = in[UMAMI_IN_HELICITY_INDEX];
    for (size_t i = 0; i < count; ++i) {
        size_t event = offset + i;
        if (!isfinite(instance->totals[i])) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT])
            ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = instance->totals[i];
        if (fixed != NULL && out[UMAMI_OUT_HELICITY_INDEX])
            ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = fixed[event];
        if (out[UMAMI_OUT_COLOR_INDEX]) {
            size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
                ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
            int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
                ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
            ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = (int)umami_provider.channels[channel]
                .flavours[flavour].contributions[0].color_index;
        }
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    if (instance->runtimes != NULL)
        for (size_t i = 0; i < umami_provider.process_count; ++i)
            if (instance->runtimes[i] != NULL) (void)rusticol_runtime_free(instance->runtimes[i]);
    for (size_t i = 0; i < instance->override_count; ++i) free(instance->overrides[i].name);
    free(instance->overrides);
    free(instance->runtimes); free(instance->selection_offsets);
    free(instance->contribution_offsets); free(instance->groups);
    free(instance->pair_events); free(instance->pair_positions); free(instance->keys);
    free(instance->sorted); free(instance->pair_entries); free(instance->totals);
    free(instance->starts); free(instance->cursor); free(instance->batch); free(instance->values);
    free(instance->events); free(instance->positions); free(instance->entries);
    free(instance->helicities);
    free(instance->artifact); free(instance);
    return UMAMI_SUCCESS;
}
