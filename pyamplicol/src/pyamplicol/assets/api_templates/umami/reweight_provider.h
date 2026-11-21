/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#define PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The tables of a reweight provider, filled from its data file by reweight_data.h:
 * arrays point into the mapped file. */
typedef struct {
    const char *key;
    const int32_t *pdgs;
    const char *alpha_s_name;
    /* ME(alpha_s) = (alpha_s / alpha_s_0)^alpha_s_power ME(alpha_s_0), -1 if unproven. */
    int alpha_s_power;
    size_t helicity_count;
} ReweightProcess;

typedef struct {
    size_t reference_process, target_process;
    /* runtime[j] = input[permutation[j]], including both incoming legs. */
    const uint32_t *permutation;
    /* Crossed union member: runtime[j] = reference_signs[j] * input[permutation[j]]
     * (NULL: all +1); target_signs likewise for target_permutation. */
    const int32_t *reference_signs;
    /* Exported helicity index -> runtime index of each process, -1 if absent. */
    const int32_t *reference_helicity, *target_helicity;
    /* Flavour-union assignment of the reference runtime (-1: none) and its process id. */
    int reference_assignment;
    const char *assignment_key;
    /* The target's runtime[j] = input[target_permutation[j]], times target_factor. */
    const uint32_t *target_permutation;
    const int32_t *target_signs;
    double target_factor;
    /* Flavour-union assignment of the target runtime (-1: none) and its process id. */
    int target_assignment;
    const char *target_assignment_key;
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
    const uint32_t *legs;
    /* flavour-union assignment of the runtime (-1: none) and its process id */
    int assignment;
    const char *assignment_key;
    /* crossed assignment: runtime[j] = signs[j] * input[legs[j]], and the helicity in
     * event slot legs[j] is helicity_signs[j] * runtime helicity j (NULL: all +1) */
    const int32_t *signs, *helicity_signs;
} ReweightSwitchEvaluation;

/* Evaluations sharing one native call: one evaluation, or flavour-union
 * assignments of one runtime at the legs of its first evaluation. */
typedef struct {
    size_t evaluation_count;
    const uint32_t *evaluations;
    /* the requested assignment ids, NULL for a plain evaluation */
    const uint32_t *assignments;
    size_t entry_count;
    /* NULL, or per request the template slot relabelling of its assignment
     * (particle_count values each, rusticol_runtime_relabel_assignment) */
    const uint32_t *entry_permutations;
    /* request of each evaluation, NULL for a plain evaluation */
    const uint32_t *evaluation_entries;
} ReweightSwitchCall;

/* The orderings and targets are the records of the data file, used in place:
 * arrays are refs (offsets into the int pool, REWEIGHT_NULL: none), weights and
 * factors indices into the constants. */
typedef struct {
    /* the summed LC read: the flavour's, or its charge conjugate's when equal */
    uint32_t evaluation;
    /* switch target index, -1 for a flavour events are generated for */
    int32_t target;
    /* x1 f x2 f constants[weight] * LC is the ordering's share of the summed flavour */
    uint32_t weight;
    /* summed for the events with this arrangement of the non-partons only */
    uint16_t arrangement;
    /* parton density columns of the incoming legs, PDG id + 6 (gluon 6) */
    uint8_t pdf[2];
} ReweightSwitchOrdering;

typedef struct {
    uint32_t evaluation, target_process;
    /* colour index of each flow of the reference process, in physical order (the
     * order of the colour draw), and the flow (ReweightSwitchFlows) of each */
    uint32_t colors, flows;
    /* target runtime[j] = input[legs[j]] (REWEIGHT_NULL: the evaluation's legs),
     * times constants[factor] */
    uint32_t legs;
    /* flavour-union assignment of the target runtime (-1: none) and its process id */
    int32_t assignment;
    uint32_t factor, assignment_key;
} ReweightSwitchTarget;

/* The flow ids of a reference process: string offsets ids[first..first+count). */
typedef struct {
    uint32_t first, count;
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
    /* per reference process, and the string offsets of the flow ids */
    const ReweightSwitchFlows *flows;
    const uint32_t *flow_ids;
    /* the exported helicities, particle_count values each */
    const int32_t *helicities;
    /* the weights of the orderings and factors of the targets */
    const double *constants;
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
