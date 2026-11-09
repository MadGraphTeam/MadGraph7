/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#define PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *key;
    const int *pdgs;
    const char *alpha_s_name;
    size_t helicity_count;
} ReweightProcess;

typedef struct {
    size_t reference_process, target_process;
    /* runtime[j] = input[permutation[j]], including both incoming legs. */
    const size_t *permutation;
    /* Exported helicity index -> runtime index of each process, -1 if absent. */
    const int *reference_helicity, *target_helicity;
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
} ReweightSwitchEvaluation;

typedef struct {
    size_t evaluation;
    /* parton density columns of the incoming legs, PDG id + 6 (gluon 6) */
    unsigned char pdf[2];
    /* x1 f x2 f weight * LC is the ordering's share of the summed flavour */
    double weight;
    /* switch target index, -1 for a flavour events are generated for */
    int target;
} ReweightSwitchOrdering;

typedef struct {
    size_t evaluation, target_process;
    /* colour index of each flow of the reference process, see ReweightSwitchFlows */
    const int *colors;
} ReweightSwitchTarget;

typedef struct {
    size_t flow_count;
    const char *const *flow_ids;
} ReweightSwitchFlows;

typedef struct {
    size_t evaluation_count;
    const ReweightSwitchEvaluation *evaluations;
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
