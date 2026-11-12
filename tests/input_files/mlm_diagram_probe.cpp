// Test-only UMAMI proxy: observe the real MLM -> MadMatrix handoff without
// changing the momenta, random numbers, matrix elements or event weights.
#include "madspace/umami.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <set>

namespace {
constexpr int subprocess = @SUBPROCESS@;
constexpr int ndiagrams = @NDIAGRAMS@;
constexpr int ncolors = @NCOLORS@;
constexpr bool allowed[ndiagrams][ncolors] = @ALLOWED@;

void* open_library() {
    auto* library = dlopen(@LIBRARY@, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "MLM probe: cannot load target library: %s\n", dlerror());
        std::abort();
    }
    return library;
}
struct Functions {
    void* library = open_library();
#define LOAD(member, name) decltype(&name) member = reinterpret_cast<decltype(&name)>(dlsym(library, #name))
    LOAD(meta, umami_get_meta);
    LOAD(inputs, umami_supported_inputs);
    LOAD(required, umami_required_inputs);
    LOAD(outputs, umami_supported_outputs);
    LOAD(initialize, umami_initialize);
    LOAD(set, umami_set_parameter);
    LOAD(get, umami_get_parameter);
    LOAD(matrix, umami_matrix_element);
    LOAD(free, umami_free);
#undef LOAD
};
Functions& api() { static Functions functions; return functions; }

struct State {
    UmamiHandle real;
    size_t calls = 0, forced = 0, sampled = 0, errors = 0, colored = 0;
    std::set<int> diagrams;
};
}

UmamiStatus umami_get_meta(UmamiMetaKey key, void* out) { return api().meta(key, out); }
UmamiStatus umami_supported_inputs(const bool** out, int* count) { return api().inputs(out, count); }
UmamiStatus umami_required_inputs(const bool** out, int* count) { return api().required(out, count); }
UmamiStatus umami_supported_outputs(const bool** out, int* count) { return api().outputs(out, count); }
UmamiStatus umami_initialize(UmamiHandle* handle, const char* card) {
    auto* state = new State;
    auto status = api().initialize(&state->real, card);
    if (status != UMAMI_SUCCESS) { delete state; return status; }
    *handle = state;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_set_parameter(UmamiHandle handle, const char* name, double re, double im) {
    return api().set(static_cast<State*>(handle)->real, name, re, im);
}
UmamiStatus umami_get_parameter(UmamiHandle handle, const char* name, double* re, double* im) {
    return api().get(static_cast<State*>(handle)->real, name, re, im);
}
UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t ni, const UmamiInputKey* ik, const void* const* in,
    size_t no, const UmamiOutputKey* ok, void* const* out) {
    auto& state = *static_cast<State*>(handle);
    ++state.calls;
    const int* diagrams = nullptr;
    int *chosen = nullptr, *colors = nullptr;
    double* matrix = nullptr;
    for (size_t k = 0; k < ni; ++k)
        if (ik[k] == UMAMI_IN_DIAGRAM_INDEX) diagrams = static_cast<const int*>(in[k]);
    for (size_t k = 0; k < no; ++k) {
        if (ok[k] == UMAMI_OUT_DIAGRAM_INDEX) chosen = static_cast<int*>(out[k]);
        if (ok[k] == UMAMI_OUT_COLOR_INDEX) colors = static_cast<int*>(out[k]);
        if (ok[k] == UMAMI_OUT_MATRIX_ELEMENT) matrix = static_cast<double*>(out[k]);
    }
    if (!diagrams || !chosen || !colors || !matrix) {
        ++state.errors;
        std::fprintf(stderr, "MLM probe: subprocess %d missing diagram handoff/output\n", subprocess);
        return UMAMI_ERROR;
    }
    auto status = api().matrix(state.real, count, stride, offset, ni, ik, in, no, ok, out);
    if (status != UMAMI_SUCCESS) return status;
    for (size_t i = offset; i < offset + count; ++i) {
        int diagram = diagrams[i];
        if (diagram == -1) { ++state.sampled; continue; }
        ++state.forced;
        state.diagrams.insert(diagram);
        bool mismatch = diagram < 0 || diagram >= ndiagrams || chosen[i] != diagram;
        if (std::isfinite(matrix[i]) && matrix[i] != 0.) {
            ++state.colored;
            mismatch = mismatch || colors[i] < 0 || colors[i] >= ncolors ||
                !allowed[diagram][colors[i]];
        }
        if (mismatch) {
            ++state.errors;
            std::fprintf(stderr, "MLM probe: subprocess %d expected diagram %d, got %d/color %d\n",
                         subprocess, diagram, chosen[i], colors[i]);
            return UMAMI_ERROR;
        }
    }
    return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle handle) {
    auto* state = static_cast<State*>(handle);
    auto status = api().free(state->real);
    if (const char* path = std::getenv("MG7_MLM_PROBE_LOG")) {
        if (FILE* log = std::fopen(path, "a")) {
            std::fprintf(log, "{\"subprocess\":%d,\"calls\":%zu,\"forced\":%zu,\"sampled\":%zu,"
                "\"errors\":%zu,\"colored\":%zu,\"diagrams\":%zu}\n", subprocess,
                state->calls, state->forced, state->sampled, state->errors, state->colored,
                state->diagrams.size());
            std::fclose(log);
        }
    }
    delete state;
    return status;
}
