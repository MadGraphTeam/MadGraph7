/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#define PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *key;
    const int *pdgs;
    const char *alpha_s_name;
    /* ME(alpha_s) = (alpha_s / alpha_s_0)^alpha_s_power ME(alpha_s_0), -1 if unproven. */
    int alpha_s_power;
    size_t helicity_count;
} ReweightProcess;

typedef struct {
    size_t reference_process, target_process;
    /* runtime[j] = input[permutation[j]], including both incoming legs. */
    const size_t *permutation;
    /* Exported helicity index -> runtime index of each process, -1 if absent. */
    const int *reference_helicity, *target_helicity;
    /* Flavour-union assignment of the reference runtime (-1: none) and its process id. */
    int reference_assignment;
    const char *assignment_key;
    /* The target's runtime[j] = input[target_permutation[j]], times target_factor. */
    const size_t *target_permutation;
    double target_factor;
    /* Flavour-union assignment of the target runtime (-1: none). */
    int target_assignment;
    /* Arrangement of the switch's non-partons in the entry (0 if it has one). */
    int switch_arrangement;
} ReweightFlavour;

typedef struct {
    size_t flavour_count;
    const ReweightFlavour *flavours;
} ReweightChannel;

/* Flavour switch: every flavour of the provider in event slot order. */
typedef struct {
    size_t process;
    /* runtime[j] = input[legs[j]] */
    const size_t *legs;
    /* flavour-union assignment of the runtime (-1: none) and its process id */
    int assignment;
    const char *assignment_key;
} ReweightSwitchEvaluation;

/* Evaluations sharing one native call: one evaluation, or flavour-union
 * assignments of one runtime at the legs of its first evaluation. */
typedef struct {
    size_t evaluation_count;
    const size_t *evaluations;
    /* the requested assignment ids, NULL for a plain evaluation */
    const uint32_t *assignments;
    size_t entry_count;
    /* NULL, or per request the template slot relabelling of its assignment
     * (particle_count values each, rusticol_runtime_relabel_assignment) */
    const size_t *entry_permutations;
    /* request of each evaluation, NULL for a plain evaluation */
    const size_t *evaluation_entries;
} ReweightSwitchCall;

typedef struct {
    size_t evaluation;
    /* parton density columns of the incoming legs, PDG id + 6 (gluon 6) */
    unsigned char pdf[2];
    /* x1 f x2 f weight * LC is the ordering's share of the summed flavour */
    double weight;
    /* switch target index, -1 for a flavour events are generated for */
    int target;
    /* summed for the events with this arrangement of the non-partons only */
    int arrangement;
} ReweightSwitchOrdering;

typedef struct {
    size_t evaluation, target_process;
    /* colour index of each flow of the reference process, in physical order (the
     * order of the colour draw), and the flow (ReweightSwitchFlows) of each */
    const int *colors;
    const int *flows;
    /* target runtime[j] = input[legs[j]], times factor */
    const size_t *legs;
    double factor;
    /* flavour-union assignment of the target runtime (-1: none) */
    int assignment;
} ReweightSwitchTarget;

typedef struct {
    size_t flow_count;
    const char *const *flow_ids;
} ReweightSwitchFlows;

typedef struct {
    size_t evaluation_count;
    const ReweightSwitchEvaluation *evaluations;
    size_t call_count;
    const ReweightSwitchCall *calls;
    size_t ordering_count;
    const ReweightSwitchOrdering *orderings;
    size_t target_count;
    const ReweightSwitchTarget *targets;
    /* per reference process */
    const ReweightSwitchFlows *flows;
    /* the exported helicities, particle_count values each */
    const int *helicities;
} ReweightSwitch;

typedef struct {
    const char *id;
    size_t particle_count, exported_helicity_count;
    /* Artifact directories relative to the directory passed to umami_initialize. */
    const char *reference_artifact, *target_artifact;
    const char *reference_artifact_id, *target_artifact_id;
    size_t reference_process_count;
    const ReweightProcess *reference_processes;
    size_t target_process_count;
    const ReweightProcess *target_processes;
    size_t channel_count;
    const ReweightChannel *channels;
    bool supports_alpha_s;
    const ReweightSwitch *flavor_switch;
} ReweightProvider;
#endif
