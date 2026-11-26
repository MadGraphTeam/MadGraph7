/* SPDX-License-Identifier: 0BSD */
/* Loads the tables of a reweight provider from its data file REWEIGHT_DATA_FILE
 * (written by pyamplicol.artifacts.umami_reweight_data), shared and refcounted by
 * all instances of the library. Include once, after reweight_provider.h and the
 * generated header. */
#ifndef PYAMPLICOL_UMAMI_REWEIGHT_DATA_H
#define PYAMPLICOL_UMAMI_REWEIGHT_DATA_H
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define REWEIGHT_NULL UINT32_MAX
#define REWEIGHT_FORMAT_VERSION 1u
#define REWEIGHT_HEADER_SIZE 128u
#define REWEIGHT_TAG(a, b, c, d) \
    ((uint32_t)(a) | (uint32_t)(b) << 8 | (uint32_t)(c) << 16 | (uint32_t)(d) << 24)

static ReweightProvider reweight_provider;
static int32_t const *rw_pool;
static char const *rw_strings;

typedef struct {
    uint32_t tag, elem_size;
    uint64_t offset, count, reserved;
} RwdSection;

typedef struct {
    uint32_t particle_count, exported_helicity_count, reference_process_count,
        target_process_count, channel_count, flags, arrangement_count, reserved;
    uint32_t id, reference_artifact, target_artifact, reference_artifact_id,
        target_artifact_id, producer;
} RwdProvider;

typedef struct {
    uint32_t key, pdgs, alpha_s_name;
    int32_t alpha_s_power;
    uint32_t helicity_count;
} RwdProcess;

typedef struct {
    uint32_t first, count;
} RwdChannel;

typedef struct {
    uint32_t reference_process, target_process, permutation, reference_signs,
        reference_helicity, target_helicity, target_permutation, target_signs;
    int32_t reference_assignment;
    uint32_t assignment_key;
    int32_t target_assignment;
    uint32_t target_assignment_key, target_factor;
    int32_t switch_arrangement;
} RwdFlavour;

typedef struct {
    uint32_t process, legs;
    int32_t assignment;
    uint32_t assignment_key, signs, helicity_signs;
} RwdEvaluation;

typedef struct {
    uint32_t evaluation_count, evaluations, assignments, entry_count, entry_permutations,
        evaluation_entries;
} RwdCall;

_Static_assert(sizeof(RwdSection) == 32, "section entry layout");
_Static_assert(sizeof(RwdProvider) == 56, "PROV layout");
_Static_assert(sizeof(RwdProcess) == 20, "PROC layout");
_Static_assert(sizeof(RwdFlavour) == 56, "FLAV layout");
_Static_assert(sizeof(RwdEvaluation) == 24, "SEVL layout");
_Static_assert(sizeof(RwdCall) == 24, "SCAL layout");
_Static_assert(sizeof(ReweightSwitchOrdering) == 16 &&
               offsetof(ReweightSwitchOrdering, weight) == 8 &&
               offsetof(ReweightSwitchOrdering, arrangement) == 12 &&
               offsetof(ReweightSwitchOrdering, pdf) == 14, "SORD layout");
_Static_assert(sizeof(ReweightSwitchTarget) == 32 &&
               offsetof(ReweightSwitchTarget, legs) == 16 &&
               offsetof(ReweightSwitchTarget, assignment_key) == 28, "STGT layout");
_Static_assert(sizeof(ReweightSwitchFlows) == 8, "SFLW layout");
_Static_assert(REWEIGHT_PARTICLE_COUNT <= 64, "at most 64 particles");

enum {
    RW_PROV, RW_STRS, RW_I32P, RW_F64T, RW_PROC, RW_CHAN, RW_FLAV, RW_SEVL, RW_SCAL,
    RW_SORD, RW_STGT, RW_SFLW, RW_SFID, RW_SHEL, RW_SECTIONS
};
static uint32_t const rw_tags[RW_SECTIONS] = {
    REWEIGHT_TAG('P', 'R', 'O', 'V'), REWEIGHT_TAG('S', 'T', 'R', 'S'),
    REWEIGHT_TAG('I', '3', '2', 'P'), REWEIGHT_TAG('F', '6', '4', 'T'),
    REWEIGHT_TAG('P', 'R', 'O', 'C'), REWEIGHT_TAG('C', 'H', 'A', 'N'),
    REWEIGHT_TAG('F', 'L', 'A', 'V'), REWEIGHT_TAG('S', 'E', 'V', 'L'),
    REWEIGHT_TAG('S', 'C', 'A', 'L'), REWEIGHT_TAG('S', 'O', 'R', 'D'),
    REWEIGHT_TAG('S', 'T', 'G', 'T'), REWEIGHT_TAG('S', 'F', 'L', 'W'),
    REWEIGHT_TAG('S', 'F', 'I', 'D'), REWEIGHT_TAG('S', 'H', 'E', 'L'),
};
static uint32_t const rw_sizes[RW_SECTIONS] = {56, 1, 4, 8, 20, 8, 56, 24, 24, 16, 32, 8, 4, 4};

static pthread_mutex_t rw_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t rw_users;
static unsigned char *rw_data;
static size_t rw_size;
static bool rw_mapped;
static ReweightProcess *rw_processes;
static ReweightFlavour *rw_flavours;
static ReweightChannel *rw_channels;
static ReweightSwitchEvaluation *rw_evaluations;
static ReweightSwitchCall *rw_calls;
static ReweightSwitch rw_switch;

static uint32_t rw_crc32(unsigned char const *data, size_t size) {
    static uint32_t table[256];
    static bool ready;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1u ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        ready = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

static uint32_t rw_u32_at(size_t offset) {
    uint32_t value;
    memcpy(&value, rw_data + offset, sizeof(value));
    return value;
}

/* The checked data: section pointers and counts. */
typedef struct {
    char const *path;
    void const *at[RW_SECTIONS];
    uint64_t count[RW_SECTIONS];
} RwdView;

static bool rw_fail(RwdView const *view, char const *what) {
    fprintf(stderr, "umami: reweight data %s: %s\n", view->path, what);
    return false;
}

static bool rw_string(RwdView const *view, uint32_t offset, bool nullable) {
    return offset == REWEIGHT_NULL ? nullable : offset < view->count[RW_STRS];
}

/* A ref of `length` ints within the pool, each in [low, high); NULL if nullable. */
static bool rw_array(RwdView const *view, uint32_t ref, uint64_t length, bool nullable,
                     int64_t low, int64_t high) {
    if (ref == REWEIGHT_NULL) return nullable;
    if ((uint64_t)ref + length > view->count[RW_I32P]) return false;
    int32_t const *values = (int32_t const *)view->at[RW_I32P] + ref;
    if (low <= INT32_MIN && high > INT32_MAX) return true;
    for (uint64_t k = 0; k < length; ++k)
        if (values[k] < low || values[k] >= high) return false;
    return true;
}

static bool rw_signs(RwdView const *view, uint32_t ref) {
    if (!rw_array(view, ref, REWEIGHT_PARTICLE_COUNT, true, -1, 2)) return false;
    for (uint32_t j = 0; ref != REWEIGHT_NULL && j < REWEIGHT_PARTICLE_COUNT; ++j)
        if (((int32_t const *)view->at[RW_I32P])[ref + j] == 0) return false;
    return true;
}

/* Every index the provider code dereferences, so that it can run unchecked. */
static bool rw_validate(RwdView *view) {
    uint64_t const n = REWEIGHT_PARTICLE_COUNT;
    char const *strings = view->at[RW_STRS];
    if (view->count[RW_STRS] == 0 || strings[view->count[RW_STRS] - 1] != '\0')
        return rw_fail(view, "the string pool is not terminated");
    if (view->count[RW_PROV] != 1) return rw_fail(view, "needs one provider record");
    RwdProvider const *p = view->at[RW_PROV];
    bool has_switch = (p->flags & 2u) != 0;
    if (p->particle_count != REWEIGHT_PARTICLE_COUNT ||
        p->exported_helicity_count != REWEIGHT_EXPORTED_HELICITY_COUNT ||
        (p->flags & 1u) != (UMAMI_HAS_ALPHA_S ? 1u : 0u))
        return rw_fail(view, "does not match the constants of the library");
    if (p->reference_process_count < 1 || p->target_process_count < 1 ||
        !rw_string(view, p->id, false) || !rw_string(view, p->reference_artifact, false) ||
        !rw_string(view, p->target_artifact, false) ||
        !rw_string(view, p->reference_artifact_id, true) ||
        !rw_string(view, p->target_artifact_id, true) || !rw_string(view, p->producer, true))
        return rw_fail(view, "invalid provider record");
    uint64_t references = p->reference_process_count, targets = p->target_process_count;
    if (view->count[RW_PROC] != references + targets)
        return rw_fail(view, "process count mismatch");
    RwdProcess const *processes = view->at[RW_PROC];
    for (uint64_t i = 0; i < references + targets; ++i)
        if (!rw_string(view, processes[i].key, false) ||
            !rw_string(view, processes[i].alpha_s_name, true) ||
            !rw_array(view, processes[i].pdgs, n, false, INT32_MIN, INT64_C(1) << 32))
            return rw_fail(view, "invalid process record");
    if (view->count[RW_CHAN] != p->channel_count) return rw_fail(view, "channel count mismatch");
    RwdChannel const *channels = view->at[RW_CHAN];
    for (uint64_t c = 0; c < p->channel_count; ++c)
        if ((uint64_t)channels[c].first + channels[c].count > view->count[RW_FLAV])
            return rw_fail(view, "channel out of range");
    RwdFlavour const *flavours = view->at[RW_FLAV];
    uint64_t constants = view->count[RW_F64T];
    for (uint64_t f = 0; f < view->count[RW_FLAV]; ++f) {
        RwdFlavour const *e = &flavours[f];
        if (e->reference_process >= references || e->target_process >= targets ||
            !rw_array(view, e->permutation, n, false, 0, (int64_t)n) ||
            !rw_array(view, e->target_permutation, n, true, 0, (int64_t)n) ||
            !rw_signs(view, e->reference_signs) || !rw_signs(view, e->target_signs) ||
            !rw_array(view, e->reference_helicity, REWEIGHT_EXPORTED_HELICITY_COUNT, false,
                      -1, processes[e->reference_process].helicity_count) ||
            !rw_array(view, e->target_helicity, REWEIGHT_EXPORTED_HELICITY_COUNT, false,
                      -1, processes[references + e->target_process].helicity_count) ||
            !rw_string(view, e->assignment_key, e->reference_assignment < 0) ||
            !rw_string(view, e->target_assignment_key, e->target_assignment < 0) ||
            e->target_factor >= constants)
            return rw_fail(view, "invalid flavour record");
    }
    if (!has_switch) return true;
    uint64_t evaluations = view->count[RW_SEVL];
    RwdEvaluation const *sevl = view->at[RW_SEVL];
    for (uint64_t e = 0; e < evaluations; ++e)
        if (sevl[e].process >= references ||
            !rw_array(view, sevl[e].legs, n, false, 0, (int64_t)n) ||
            !rw_string(view, sevl[e].assignment_key, sevl[e].assignment < 0) ||
            !rw_signs(view, sevl[e].signs) || !rw_signs(view, sevl[e].helicity_signs))
            return rw_fail(view, "invalid switch evaluation");
    RwdCall const *calls = view->at[RW_SCAL];
    /* switch_totals writes only the evaluations of the calls: the orderings read no others */
    bool *written = calloc(evaluations + 1, sizeof(bool));
    if (written == NULL) return rw_fail(view, "out of memory");
    for (uint64_t g = 0; g < view->count[RW_SCAL]; ++g) {
        RwdCall const *call = &calls[g];
        bool plain = call->assignments == REWEIGHT_NULL;
        if (call->evaluation_count < 1 ||
            !rw_array(view, call->evaluations, call->evaluation_count, false, 0,
                      (int64_t)evaluations) ||
            (plain && (call->evaluation_count != 1 || call->entry_count != 1 ||
                       call->entry_permutations != REWEIGHT_NULL ||
                       call->evaluation_entries != REWEIGHT_NULL)) ||
            (!plain && (call->entry_count < 1 ||
                        !rw_array(view, call->assignments, call->entry_count, false,
                                  INT32_MIN, INT64_C(1) << 32) ||
                        !rw_array(view, call->evaluation_entries, call->evaluation_count,
                                  false, 0, call->entry_count) ||
                        !rw_array(view, call->entry_permutations,
                                  (uint64_t)call->entry_count * n, true, 0, (int64_t)n)))) {
            free(written);
            return rw_fail(view, "invalid switch call");
        }
        uint32_t const *members = (uint32_t const *)view->at[RW_I32P] + call->evaluations;
        for (uint32_t j = 0; j < call->evaluation_count; ++j) written[members[j]] = true;
    }
    uint64_t switch_targets = view->count[RW_STGT];
    ReweightSwitchOrdering const *orderings = view->at[RW_SORD];
    for (uint64_t o = 0; o < view->count[RW_SORD]; ++o)
        if (orderings[o].evaluation >= evaluations || !written[orderings[o].evaluation] ||
            orderings[o].target < -1 ||
            (orderings[o].target >= 0 && (uint64_t)orderings[o].target >= switch_targets) ||
            orderings[o].pdf[0] >= 13 || orderings[o].pdf[1] >= 13 ||
            orderings[o].weight >= constants) {
            free(written);
            return rw_fail(view, "invalid switch ordering");
        }
    free(written);
    if (view->count[RW_SFLW] != references) return rw_fail(view, "flow table count mismatch");
    ReweightSwitchFlows const *flows = view->at[RW_SFLW];
    uint32_t const *ids = view->at[RW_SFID];
    for (uint64_t i = 0; i < references; ++i) {
        if ((uint64_t)flows[i].first + flows[i].count > view->count[RW_SFID])
            return rw_fail(view, "flow ids out of range");
        for (uint64_t k = 0; k < flows[i].count; ++k)
            if (!rw_string(view, ids[flows[i].first + k], false))
                return rw_fail(view, "invalid flow id");
    }
    ReweightSwitchTarget const *stgt = view->at[RW_STGT];
    /* the targets share few flow-order arrays: each (ref, width) is checked once */
    uint16_t *checked = switch_targets > 0
        ? calloc(view->count[RW_I32P] + 1, sizeof(uint16_t)) : NULL;
    if (switch_targets > 0 && checked == NULL) return rw_fail(view, "out of memory");
    bool valid = true;
    for (uint64_t t = 0; valid && t < switch_targets; ++t) {
        ReweightSwitchTarget const *target = &stgt[t];
        if (target->evaluation >= evaluations || target->target_process >= targets) {
            valid = false;
            break;
        }
        uint64_t width = flows[sevl[target->evaluation].process].count;
        bool order = target->flows != REWEIGHT_NULL && target->flows < view->count[RW_I32P] &&
                     width < UINT16_MAX && checked[target->flows] == width + 1;
        if (!order) {
            order = rw_array(view, target->flows, width, false, 0, (int64_t)width);
            if (order && width < UINT16_MAX && target->flows < view->count[RW_I32P])
                checked[target->flows] = (uint16_t)(width + 1);
        }
        valid = order && rw_array(view, target->colors, width, false, INT32_MIN, INT64_MAX) &&
                rw_array(view, target->legs, n, true, 0, (int64_t)n) &&
                rw_string(view, target->assignment_key, target->assignment < 0) &&
                target->factor < constants;
    }
    free(checked);
    if (!valid) return rw_fail(view, "invalid switch target");
    if (view->count[RW_SHEL] != REWEIGHT_EXPORTED_HELICITY_COUNT * n)
        return rw_fail(view, "switch helicity count mismatch");
    return true;
}

/* Header, section table and checksums; fills the section pointers. */
static bool rw_check(RwdView *view) {
    static char const magic[8] = {'P', 'Y', 'A', 'C', 'R', 'W', 'D', '\0'};
    if (rw_size < REWEIGHT_HEADER_SIZE) return rw_fail(view, "file is truncated");
    if (memcmp(rw_data, magic, 8) != 0) return rw_fail(view, "not a reweight data file");
    uint32_t version = rw_u32_at(8);
    if (version != REWEIGHT_FORMAT_VERSION) {
        fprintf(stderr, "umami: reweight data %s has format version %u, this library reads "
                "%u: regenerate the output\n", view->path, (unsigned)version,
                (unsigned)REWEIGHT_FORMAT_VERSION);
        return false;
    }
    if (rw_u32_at(16) != 0x01020304u) return rw_fail(view, "wrong byte order");
    if (rw_u32_at(12) != REWEIGHT_HEADER_SIZE) return rw_fail(view, "wrong header size");
    uint64_t file_size;
    memcpy(&file_size, rw_data + 24, sizeof(file_size));
    if (file_size != rw_size) return rw_fail(view, "file size mismatch (truncated?)");
    unsigned char header[REWEIGHT_HEADER_SIZE];
    memcpy(header, rw_data, sizeof(header));
    memset(header + 36, 0, 4);
    if (rw_crc32(header, sizeof(header)) != rw_u32_at(36))
        return rw_fail(view, "header checksum mismatch");
    uint32_t sections = rw_u32_at(20);
    if ((uint64_t)sections * sizeof(RwdSection) > rw_size - REWEIGHT_HEADER_SIZE)
        return rw_fail(view, "section table out of bounds");
    if (rw_crc32(rw_data + REWEIGHT_HEADER_SIZE, rw_size - REWEIGHT_HEADER_SIZE) !=
        rw_u32_at(32))
        return rw_fail(view, "payload checksum mismatch (corrupted)");
    char hex[65];
    for (int k = 0; k < 32; ++k) snprintf(hex + 2 * k, 3, "%02x", rw_data[40 + k]);
    if (strcmp(hex, REWEIGHT_DATA_SHA256) != 0)
        return rw_fail(view, "content digest differs from the library's: rebuild the "
                       "library (make) or regenerate the output");
    bool seen[RW_SECTIONS] = {false};
    for (uint32_t s = 0; s < sections; ++s) {
        RwdSection entry;
        memcpy(&entry, rw_data + REWEIGHT_HEADER_SIZE + s * sizeof(entry), sizeof(entry));
        int k = 0;
        while (k < RW_SECTIONS && rw_tags[k] != entry.tag) ++k;
        if (k == RW_SECTIONS || seen[k]) return rw_fail(view, "unknown or repeated section");
        if (entry.elem_size != rw_sizes[k] || entry.offset % 8 != 0 ||
            entry.offset > rw_size || entry.count > (rw_size - entry.offset) / rw_sizes[k])
            return rw_fail(view, "malformed section");
        seen[k] = true;
        view->at[k] = rw_data + entry.offset;
        view->count[k] = entry.count;
    }
    for (int k = 0; k < RW_SECTIONS; ++k)
        if (!seen[k]) return rw_fail(view, "a section is missing");
    return true;
}

static char const *rw_text(uint32_t offset) {
    return offset == REWEIGHT_NULL ? NULL : rw_strings + offset;
}

static int32_t const *rw_ints(uint32_t ref) {
    return ref == REWEIGHT_NULL ? NULL : rw_pool + ref;
}

static void rw_unload(void) {
    free(rw_processes); free(rw_flavours); free(rw_channels);
    free(rw_evaluations); free(rw_calls);
    rw_processes = NULL; rw_flavours = NULL; rw_channels = NULL;
    rw_evaluations = NULL; rw_calls = NULL;
    if (rw_mapped) munmap(rw_data, rw_size);
    else free(rw_data);
    rw_data = NULL; rw_size = 0; rw_mapped = false;
    rw_pool = NULL; rw_strings = NULL;
    memset(&reweight_provider, 0, sizeof(reweight_provider));
    memset(&rw_switch, 0, sizeof(rw_switch));
}

/* The pointer tables of the checked data; in-place sections stay in the file. */
static bool rw_materialise(RwdView const *view) {
    RwdProvider const *p = view->at[RW_PROV];
    rw_pool = view->at[RW_I32P];
    rw_strings = view->at[RW_STRS];
    double const *constants = view->at[RW_F64T];
    size_t processes = p->reference_process_count + (size_t)p->target_process_count;
    rw_processes = calloc(processes, sizeof(*rw_processes));
    rw_flavours = calloc(view->count[RW_FLAV] + 1, sizeof(*rw_flavours));
    rw_channels = calloc(view->count[RW_CHAN] + 1, sizeof(*rw_channels));
    rw_evaluations = calloc(view->count[RW_SEVL] + 1, sizeof(*rw_evaluations));
    rw_calls = calloc(view->count[RW_SCAL] + 1, sizeof(*rw_calls));
    if (rw_processes == NULL || rw_flavours == NULL || rw_channels == NULL ||
        rw_evaluations == NULL || rw_calls == NULL)
        return false;
    RwdProcess const *proc = view->at[RW_PROC];
    for (size_t i = 0; i < processes; ++i)
        rw_processes[i] = (ReweightProcess){
            rw_text(proc[i].key), rw_ints(proc[i].pdgs), rw_text(proc[i].alpha_s_name),
            proc[i].alpha_s_power, proc[i].helicity_count};
    RwdFlavour const *flav = view->at[RW_FLAV];
    for (size_t f = 0; f < view->count[RW_FLAV]; ++f) {
        RwdFlavour const *e = &flav[f];
        uint32_t const *permutation = (uint32_t const *)rw_ints(e->permutation);
        uint32_t const *target_permutation = e->target_permutation == REWEIGHT_NULL
            ? permutation : (uint32_t const *)rw_ints(e->target_permutation);
        rw_flavours[f] = (ReweightFlavour){
            e->reference_process, e->target_process, permutation,
            rw_ints(e->reference_signs), rw_ints(e->reference_helicity),
            rw_ints(e->target_helicity), e->reference_assignment,
            rw_text(e->assignment_key), target_permutation, rw_ints(e->target_signs),
            constants[e->target_factor], e->target_assignment,
            rw_text(e->target_assignment_key), e->switch_arrangement};
    }
    RwdChannel const *chan = view->at[RW_CHAN];
    for (size_t c = 0; c < view->count[RW_CHAN]; ++c)
        rw_channels[c] = (ReweightChannel){chan[c].count, rw_flavours + chan[c].first};
    reweight_provider = (ReweightProvider){
        rw_text(p->id), p->particle_count, p->exported_helicity_count,
        rw_text(p->reference_artifact), rw_text(p->target_artifact),
        rw_text(p->reference_artifact_id), rw_text(p->target_artifact_id),
        p->reference_process_count, rw_processes,
        p->target_process_count, rw_processes + p->reference_process_count,
        p->channel_count, rw_channels, (p->flags & 1u) != 0, NULL};
    if ((p->flags & 2u) == 0) return true;
    RwdEvaluation const *sevl = view->at[RW_SEVL];
    for (size_t e = 0; e < view->count[RW_SEVL]; ++e)
        rw_evaluations[e] = (ReweightSwitchEvaluation){
            sevl[e].process, (uint32_t const *)rw_ints(sevl[e].legs), sevl[e].assignment,
            rw_text(sevl[e].assignment_key), rw_ints(sevl[e].signs),
            rw_ints(sevl[e].helicity_signs)};
    RwdCall const *scal = view->at[RW_SCAL];
    for (size_t g = 0; g < view->count[RW_SCAL]; ++g)
        rw_calls[g] = (ReweightSwitchCall){
            scal[g].evaluation_count, (uint32_t const *)rw_ints(scal[g].evaluations),
            (uint32_t const *)rw_ints(scal[g].assignments), scal[g].entry_count,
            (uint32_t const *)rw_ints(scal[g].entry_permutations),
            (uint32_t const *)rw_ints(scal[g].evaluation_entries)};
    rw_switch = (ReweightSwitch){
        view->count[RW_SEVL], rw_evaluations, view->count[RW_SCAL], rw_calls,
        view->count[RW_SORD], view->at[RW_SORD], view->count[RW_STGT], view->at[RW_STGT],
        view->at[RW_SFLW], view->at[RW_SFID], view->at[RW_SHEL], constants};
    reweight_provider.flavor_switch = &rw_switch;
    return true;
}

static bool rw_read(char const *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat info;
    if (fstat(fd, &info) != 0 || info.st_size < 0 || (uint64_t)info.st_size > SIZE_MAX) {
        close(fd);
        return false;
    }
    rw_size = (size_t)info.st_size;
    void *map = rw_size == 0 ? MAP_FAILED
                             : mmap(NULL, rw_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map != MAP_FAILED) {
        rw_data = map;
        rw_mapped = true;
    } else {
        rw_data = malloc(rw_size + 1);
        size_t done = 0;
        while (rw_data != NULL && done < rw_size) {
            ssize_t got = read(fd, rw_data + done, rw_size - done);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;
            done += (size_t)got;
        }
        if (rw_data == NULL || done != rw_size) {
            free(rw_data);
            rw_data = NULL;
            close(fd);
            return false;
        }
    }
    close(fd);
    return true;
}

/* Loads the tables from <base>/REWEIGHT_DATA_FILE on the first call. */
static UmamiStatus reweight_data_acquire(char const *base) {
    pthread_mutex_lock(&rw_lock);
    if (rw_users > 0) {
        ++rw_users;
        pthread_mutex_unlock(&rw_lock);
        return UMAMI_SUCCESS;
    }
    size_t length = strlen(base) + sizeof(REWEIGHT_DATA_FILE) + 1;
    char *path = malloc(length);
    UmamiStatus status = UMAMI_ERROR;
    if (path != NULL) {
        snprintf(path, length, "%s/%s", base, REWEIGHT_DATA_FILE);
        RwdView view = {.path = path};
        if (!rw_read(path)) {
            fprintf(stderr, "umami: cannot read the reweight data %s\n", path);
        } else if (rw_check(&view) && rw_validate(&view)) {
            if (rw_materialise(&view)) {
                status = UMAMI_SUCCESS;
                rw_users = 1;
            }
        }
        if (status != UMAMI_SUCCESS) rw_unload();
        free(path);
    }
    pthread_mutex_unlock(&rw_lock);
    return status;
}

static void reweight_data_release(void) {
    pthread_mutex_lock(&rw_lock);
    if (rw_users > 0 && --rw_users == 0) rw_unload();
    pthread_mutex_unlock(&rw_lock);
}
#endif
