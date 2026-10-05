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

typedef struct {
    char *artifact;
    RusticolRuntimeHandle **runtimes;
    double *momenta;
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
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
    instance->runtimes[process] = runtime;
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

UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    if (handle == NULL) return UMAMI_ERROR;
    *handle = NULL;
    struct stat info;
    if (param_card_path == NULL || param_card_path[0] != '/' ||
        stat(param_card_path, &info) != 0 || !S_ISDIR(info.st_mode) ||
        umami_provider.process_count == 0 || umami_provider.particle_count > INT_MAX ||
        umami_provider.particle_count > SIZE_MAX / (4 * sizeof(double))) return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    size_t length = strlen(param_card_path) + 1;
    instance->artifact = malloc(length);
    instance->runtimes = calloc(umami_provider.process_count, sizeof(*instance->runtimes));
    instance->momenta = calloc(4 * umami_provider.particle_count, sizeof(double));
    if (instance->artifact == NULL || instance->runtimes == NULL || instance->momenta == NULL) {
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
        found = true;
        if (!entry->mutable || (entry->is_complex && !isfinite(imaginary))) goto rollback;
        status = load_process(instance, i);
        if (status != UMAMI_SUCCESS) goto rollback;
        status = native_error(rusticol_runtime_get_model_parameter(
            instance->runtimes[i], name, &previous[2*i], &previous[2*i+1]));
        if (status != UMAMI_SUCCESS) goto rollback;
        status = native_error(rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, real, entry->is_complex ? imaginary : 0.0));
        if (status != UMAMI_SUCCESS) goto rollback;
        changed[i] = true;
    }
    free(previous); free(changed);
    return found ? UMAMI_SUCCESS : UMAMI_ERROR;
rollback:
    for (size_t i = 0; i < count; ++i)
        if (changed[i]) (void)rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, previous[2*i], previous[2*i+1]);
    free(previous); free(changed);
    return UMAMI_ERROR;
}

static UmamiStatus evaluate_contribution(Instance *instance, UmamiContribution const *entry,
                                        double const *momenta, size_t event, size_t stride,
                                        double const *alpha_s, double *result) {
    size_t process = entry->process_index;
    UmamiProcess const *description = &umami_provider.processes[process];
    UmamiStatus status = load_process(instance, process);
    if (status != UMAMI_SUCCESS) return status;
    RusticolRuntimeHandle *runtime = instance->runtimes[process];
    size_t n = umami_provider.particle_count;
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < 4; ++k) {
            double value = momenta[event + stride * (n*k + entry->permutation[j])];
            if (!isfinite(value)) return UMAMI_ERROR;
            instance->momenta[4*j+k] = value;
        }
    double previous_real = 0.0, previous_imaginary = 0.0;
    bool changed = false;
    if (alpha_s != NULL) {
        if (description->alpha_s_name == NULL) return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (!isfinite(*alpha_s) || *alpha_s < 0.0) return UMAMI_ERROR;
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
    int native_status = rusticol_runtime_evaluate_selected_f64(
        runtime, instance->momenta, 4*n, 1,
        NULL, 0,
        entry->color_id == NULL ? NULL : &entry->color_id,
        entry->color_id == NULL ? 0 : 1,
        NULL, 0, NULL, 0, result, 1);
    status = native_error(native_status);
    if (changed) {
        UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
            runtime, description->alpha_s_name, previous_real, previous_imaginary));
        if (restored != UMAMI_SUCCESS) return restored;
    }
    return status;
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
    for (size_t event = offset; event < offset+count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
            ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
            ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        if (channel >= umami_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= umami_provider.channels[channel].flavour_count) return UMAMI_ERROR;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        if (selection->contribution_count == 0) return UMAMI_ERROR;
        double random = in[UMAMI_IN_RANDOM_HELICITY] ?
            ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        if (!isfinite(random) || random < 0.0 || random >= 1.0) return UMAMI_ERROR;
        size_t helicities = umami_provider.processes[selection->contributions[0].process_index].helicity_count;
        int label = (int)(random * (double)helicities);
        double total = 0.0;
        for (size_t j = 0; j < selection->contribution_count; ++j) {
            double value = 0.0;
            UmamiStatus status = evaluate_contribution(instance, &selection->contributions[j],
                in[UMAMI_IN_MOMENTA], event, stride,
                in[UMAMI_IN_ALPHA_S] ? &((double const *)in[UMAMI_IN_ALPHA_S])[event] : NULL, &value);
            if (status != UMAMI_SUCCESS) return status;
            total += selection->contributions[j].factor * value;
        }
        if (!isfinite(total)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = total;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] =
            (int)selection->contributions[0].color_index;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    if (instance->runtimes != NULL)
        for (size_t i = 0; i < umami_provider.process_count; ++i)
            if (instance->runtimes[i] != NULL) (void)rusticol_runtime_free(instance->runtimes[i]);
    free(instance->runtimes); free(instance->momenta); free(instance->artifact); free(instance);
    return UMAMI_SUCCESS;
}
