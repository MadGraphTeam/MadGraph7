// Copyright (C) 2026 CERN.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by A. Valassi (Aug 2026) for project MadVFold (arXiv:2609.23500).
//
// Phase 3b/3c/4: dlopen-based dispatcher for CUDACPP Born and Real libraries.
//
// Why dlopen: libmg5amc_cudacpp_born.so (Born) and libmg5amc_cudacpp_real*.so
// (Real) export the same Fortran symbols (fbridgecreate_, fbridgedelete_,
// fbridgesequence_nomultichannel_) and therefore cannot both be linked into
// the same executable.
//
// Phase 4 addition: fbridge_born_sequence_ wraps fbridgesequence_ (multichannel
// variant) with a channelIds array — used for the SDE (Single Diagram Enhancement)
// batch call. Only Born needs this; Real uses nomultichannel only.
//
// Real bridge generalisation (found 2026-09-07, F-vs-C investigation on
// eemumubbg_vfalt): NFKSPROCESS can dispatch to several physically distinct
// real-emission subprocesses (e.g. "add a gluon" vs "split the Born gluon
// into a light q qbar pair", one physical process per flavour group) -- a
// single Real bridge can only ever cover ONE of them. There is therefore no
// longer a single libmg5amc_cudacpp_real.so: instead there are however many
// libmg5amc_cudacpp_real<N>.so libraries the process needs (N=1,2,...),
// numbered in whatever order copyCudacppLibs.sh lists ccRealProcs. A
// process with only one real subprocess (the common case) has just
// libmg5amc_cudacpp_real1.so, so this generalisation is a no-op for it.
// Real<N> is loaded lazily, indexed by N, the first time Fortran asks for
// it via the new WHICH argument on fbridge_real_create_ etc. below.
// fbridge_real_library_exists_ lets the Fortran side (vfold_real_batch.f90)
// probe which N actually have a library on disk BEFORE routing any entry
// to the bridge, so a process that hasn't (yet) built every real<N>
// library still runs correctly, falling back to scalar SMATRIX_REAL for
// the ones missing.
//
// Per-group common library (found 2026-09-07, real2/uux F-vs-C investigation;
// naming scheme reworked 2026-09-09 to fix a cuda incompatibility found on
// eebb_vfalt): libmg5amc_cudacpp_common*.so (Parameters_sm/SLHAReader,
// holding the process's coupling-constant table) is NOT interchangeable
// across Born/Real<N> -- each is its own independent MG5 export and can lay
// out its couplings (GC_N numbering) differently. Loading e.g. Real2's
// matrix-element code together with Born's common.so silently reads
// couplings from the wrong slots (confirmed: this was the root cause of
// real2/uux giving wildly wrong MEs).
//
// copyCudacppLibs.sh copies each of Born/Real1..N's OWN common.so into its
// OWN subdirectory, AND renames it (patchelf --set-soname) to a name unique
// to that group -- libmg5amc_cudacpp_common_born.so,
// libmg5amc_cudacpp_common_real1.so, etc. -- and repoints (patchelf
// --replace-needed) that group's own matrix-element .so's DT_NEEDED entry
// at the renamed name. This is why plain dlopen() below is enough on its
// own: since each group's common.so now has a distinct name, the dynamic
// linker's normal process-global name-based dependency resolution can never
// confuse one group's common.so for another's -- there is nothing left to
// disambiguate by directory or by namespace.
//
// (An earlier version of this file used dlmopen(LM_ID_NEWLM, ...) to put
// each group in its own linker namespace instead, precisely to stop
// same-named common.so files colliding under plain dlopen(). That worked
// for the CPU/SIMD backends, but broke the cuda backend found 2026-09-08 on
// eebb_vfalt: Born/Real<N> link no separate libcudart.so/libcuda.so (the
// CUDA runtime is statically linked into each .so), and dlmopen's separate
// linker namespace gives each such copy its own independent runtime state;
// two independent copies both trying to claim the same physical device
// crashed immediately on the second one's first CUDA call (cudaSetDevice ->
// cudaErrorOperatingSystem, code 304). Renaming instead of namespacing
// avoids this entirely: it uses only the plain dlopen() mechanism already
// proven CUDA-compatible for every single-Real-library process, uniformly
// for single- and multi-Real-library processes alike -- no runtime
// detection of "how many Real libraries does this process have" is needed
// any more.)
//
// Library layout for FPTYPE=d (copied by copyCudacppLibs.sh), one
// subdirectory per group, each with its OWN uniquely-named common.so:
//   <BASE>/build.<BACKEND>_d_inl0_hrd0/born/libmg5amc_cudacpp_common_born.so
//   <BASE>/build.<BACKEND>_d_inl0_hrd0/born/libmg5amc_cudacpp_born.so
//   <BASE>/build.<BACKEND>_d_inl0_hrd0/real<N>/libmg5amc_cudacpp_common_real<N>.so
//   <BASE>/build.<BACKEND>_d_inl0_hrd0/real<N>/libmg5amc_cudacpp_real<N>.so
//
// Born and every Real<N> have RPATH=$ORIGIN, resolving to their own
// subdirectory's (uniquely-named) common.so.
//
// BASE must be defined at runtime via ENV_VFOLD_CUDACPP_BASE; the compile-time
// path CUDACPP_BASE_PATH used in earlier code is no longer defined.
//
// SIMD backend is selected via ENV_VFOLD_CUDACPP_BACKEND (default: 512y).

#include <dlfcn.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

//#ifndef CUDACPP_BASE_PATH
//#define CUDACPP_BASE_PATH "<path>/DEV/eebb_vfold/lib/cudacpp"
//#endif

namespace {

// Generous upper bound on the number of distinct Real subprocesses any one
// process might need (observed so far: 1 for the simpler vfalt processes,
// 3 for eemumubbg_vfalt's "add gluon" + two flavour-group splits). Bump if
// a future process genuinely needs more; nothing below scales with this
// beyond a few small fixed-size arrays. Index 0 is unused (WHICH is 1-based,
// matching libmg5amc_cudacpp_real<N>.so's own N).
constexpr int kMaxRealLibs = 9;

// Fortran-callable C signatures. INTEGER*8 -> long long*, INTEGER*4 -> int*,
// DOUBLE PRECISION -> double*, LOGICAL (default 4 bytes in gfortran) -> int*.
typedef void (*fp_create_t)(long long* pbridge, int* nevt, int* npar, int* np4);
// Delete a bridge: fbridgedelete_ also resets the GPU (used for the Born, deleted
// last), fbridgedelete_nohwreset_ does not (used for the Reals, deleted first)
typedef void (*fp_delete_t)(long long* pbridge);
typedef void (*fp_sequence_nomc_t)(
    long long* pbridge,
    double* momenta,
    double* gs, double* rndhel, double* rndcol,
    const int* igraph, // 381 port: new in cudacpp for MLM (nullptr = no MLM)
    double* mes,
    int* selhel, int* selcol,
    int* helonly);
typedef void (*fp_sequence_mc_t)(
    long long* pbridge,
    double* momenta,
    double* gs, double* rndhel, double* rndcol,
    const unsigned int* channelIds,
    const int* igraph, // 381 port: new in cudacpp for MLM (nullptr = no MLM)
    double* mes,
    int* selhel, int* selcol,
    int* helonly);

void* g_born_handle   = nullptr;
fp_create_t        g_born_create        = nullptr;
fp_delete_t        g_born_delete        = nullptr;
fp_sequence_nomc_t g_born_sequence_nomc = nullptr;
fp_sequence_mc_t   g_born_sequence_mc   = nullptr;
// AV (Road B step 9i+9j): OPTIONAL accessor for the ten colour-linked Borns
// computed by the last Born sequence call (fbridgegetbsf_, exported by the
// eebbbb Born library, by its C++ backends since step 9i+9j and by its cuda
// backend since step 10a; nullptr if absent, e.g. other processes).
typedef void (*fp_getbsf_t)(const int* nevt, double* bsfs, int* status);
fp_getbsf_t        g_born_getbsf        = nullptr;
// AV road B (step 9i+9j): OPTIONAL accessor for the whole per-event block
// (10 colour-linked Borns, 16 per-diagram |amp|^2, 2 per-colour |jamp|^2)
typedef void (*fp_getextras_t)(const int* nevt, double* extras, int* status);
fp_getextras_t     g_born_getextras     = nullptr;
bool g_born_initialized = false;

// One slot per possible Real<N>, indexed 1..kMaxRealLibs (index 0 unused).
void*              g_real_handle[kMaxRealLibs + 1]        = {};
fp_create_t        g_real_create[kMaxRealLibs + 1]        = {};
fp_delete_t        g_real_delete[kMaxRealLibs + 1]        = {};
fp_sequence_nomc_t g_real_sequence_nomc[kMaxRealLibs + 1] = {};
bool               g_real_initialized[kMaxRealLibs + 1]   = {};

[[noreturn]] void fatal(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::fputs("[cudacpp_dispatcher] FATAL: ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    std::exit(1);
}

const char* getenv_or(const char* var, const char* fallback) {
    const char* s = std::getenv(var);
    return (s && *s) ? s : fallback;
}

std::string get_lib_dir() {
    const char* backend = getenv_or("ENV_VFOLD_CUDACPP_BACKEND", "512y");

    //const char* base    = getenv_or("ENV_VFOLD_CUDACPP_BASE", CUDACPP_BASE_PATH);
    const char* base    = getenv_or("ENV_VFOLD_CUDACPP_BASE", "UNKNOWN");

    std::fprintf(stderr, "[cudacpp_dispatcher] backend=%s base=%s\n", backend, base);
    return std::string(base) + "/build." + backend + "_d_inl0_hrd0";
}

std::string real_group_dir(int which) {
    return get_lib_dir() + "/real" + std::to_string(which);
}

std::string real_lib_path(int which) {
    return real_group_dir(which) + "/libmg5amc_cudacpp_real" + std::to_string(which) + ".so";
}

void init_born() {
    const std::string path = get_lib_dir() + "/born/libmg5amc_cudacpp_born.so";
    std::fprintf(stderr, "[cudacpp_dispatcher] born:   %s\n", path.c_str());
    // Plain dlopen(): born's own DT_NEEDED now names its own uniquely-named
    // common.so (see the top-of-file comment), so ordinary process-global
    // name-based dependency resolution can never confuse it with another
    // group's common.so.
    g_born_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_born_handle) fatal("dlopen born failed: %s", dlerror());

    g_born_create =
        (fp_create_t) dlsym(g_born_handle, "fbridgecreate_");
    if (!g_born_create) fatal("dlsym fbridgecreate_ (born) failed: %s", dlerror());

    g_born_delete =
        (fp_delete_t) dlsym(g_born_handle, "fbridgedelete_");
    if (!g_born_delete) fatal("dlsym fbridgedelete_ (born) failed: %s", dlerror());

    g_born_sequence_nomc =
        (fp_sequence_nomc_t) dlsym(g_born_handle, "fbridgesequence_nomultichannel_");
    if (!g_born_sequence_nomc)
        fatal("dlsym fbridgesequence_nomultichannel_ (born) failed: %s", dlerror());

    g_born_sequence_mc =
        (fp_sequence_mc_t) dlsym(g_born_handle, "fbridgesequence_");
    if (!g_born_sequence_mc)
        fatal("dlsym fbridgesequence_ (born) failed: %s", dlerror());

    // Optional: no fatal() if absent (see g_born_getbsf above)
    g_born_getbsf =
        (fp_getbsf_t) dlsym(g_born_handle, "fbridgegetbsf_");
    if (!g_born_getbsf) dlerror(); // clear the error state
    std::fprintf(stderr, "[cudacpp_dispatcher] born:   fbridgegetbsf_ %s\n",
                 g_born_getbsf ? "available" : "NOT available");
    g_born_getextras =
        (fp_getextras_t) dlsym(g_born_handle, "fbridgegetbornextras_");
    if (!g_born_getextras) dlerror(); // clear the error state
    std::fprintf(stderr, "[cudacpp_dispatcher] born:   fbridgegetbornextras_ %s\n",
                 g_born_getextras ? "available" : "NOT available");

    g_born_initialized = true;
}

void init_real(int which) {
    const std::string path = real_lib_path(which);
    std::fprintf(stderr, "[cudacpp_dispatcher] real%d:  %s\n", which, path.c_str());
    // Plain dlopen(): see init_born() -- real<which>'s own DT_NEEDED names
    // its own uniquely-named common.so, so it can never resolve to another
    // group's.
    g_real_handle[which] = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_real_handle[which]) fatal("dlopen real%d failed: %s", which, dlerror());

    g_real_create[which] =
        (fp_create_t) dlsym(g_real_handle[which], "fbridgecreate_");
    if (!g_real_create[which]) fatal("dlsym fbridgecreate_ (real%d) failed: %s", which, dlerror());

    g_real_delete[which] =
        (fp_delete_t) dlsym(g_real_handle[which], "fbridgedelete_nohwreset_");
    if (!g_real_delete[which]) fatal("dlsym fbridgedelete_nohwreset_ (real%d) failed: %s", which, dlerror());

    g_real_sequence_nomc[which] =
        (fp_sequence_nomc_t) dlsym(g_real_handle[which], "fbridgesequence_nomultichannel_");
    if (!g_real_sequence_nomc[which])
        fatal("dlsym fbridgesequence_nomultichannel_ (real%d) failed: %s", which, dlerror());

    g_real_initialized[which] = true;
}

void check_which(int which, const char* caller) {
    if (which < 1 || which > kMaxRealLibs)
        fatal("%s: WHICH=%d out of range [1,%d]", caller, which, kMaxRealLibs);
}

inline void ensure_born_initialized() {
    if (!g_born_initialized) init_born();
}
inline void ensure_real_initialized(int which) {
    check_which(which, "ensure_real_initialized");
    if (!g_real_initialized[which]) init_real(which);
}

} // namespace

extern "C" {

// CALL FBRIDGE_BORN_CREATE(PBRIDGE, NEVT, NPAR, NP4)
void fbridge_born_create_(long long* pbridge, int* nevt, int* npar, int* np4) {
    ensure_born_initialized();
    g_born_create(pbridge, nevt, npar, np4);
}

// CALL FBRIDGE_BORN_DELETE(PBRIDGE)
void fbridge_born_delete_(long long* pbridge) {
    ensure_born_initialized();
    g_born_delete(pbridge); // fbridgedelete_: also resets the GPU (only for the Born)
}

// CALL FBRIDGE_BORN_SEQUENCE_NOMULTICHANNEL(PBRIDGE, MOMENTA, GS, RNDHEL, RNDCOL,
//                                          MES, SELHEL, SELCOL, HELONLY)
void fbridge_born_sequence_nomultichannel_(
    long long* pbridge,
    double* momenta,
    double* gs, double* rndhel, double* rndcol,
    double* mes,
    int* selhel, int* selcol,
    int* helonly)
{
    ensure_born_initialized();
    g_born_sequence_nomc(pbridge, momenta, gs, rndhel, rndcol,
                         nullptr, mes, selhel, selcol, helonly); // 381 port: igraph=nullptr (no MLM)
}

// CALL FBRIDGE_BORN_SEQUENCE(PBRIDGE, MOMENTA, GS, RNDHEL, RNDCOL,
//                            CHANNELIDS, MES, SELHEL, SELCOL, HELONLY)
// Multichannel variant: MES(i) = ME_total(i) * amp2[channelIds(i)] / sum(amp2)
// Used for SDE (Single Diagram Enhancement) batch in Phase 4.
// CHANNELIDS is INTEGER(4) array of length NEVT (Fortran 1-based channel indices).
void fbridge_born_sequence_(
    long long* pbridge,
    double* momenta,
    double* gs, double* rndhel, double* rndcol,
    const unsigned int* channelIds,
    double* mes,
    int* selhel, int* selcol,
    int* helonly)
{
    ensure_born_initialized();
    g_born_sequence_mc(pbridge, momenta, gs, rndhel, rndcol,
                       channelIds, nullptr, mes, selhel, selcol, helonly); // 381 port: igraph=nullptr (no MLM)
}

// CALL FBRIDGE_BORN_GET_BSF(NEVT, BSFS, STATUS)
// AV (Road B step 9i+9j): the ten colour-linked Borns, BSFS(10,NEVT), of the
// events of the LAST Born sequence call. STATUS=0 on success, 1 if NEVT does
// not match that call, 2 if this backend's library does not provide them.
void fbridge_born_get_bsf_(int* nevt, double* bsfs, int* status) {
    ensure_born_initialized();
    if (!g_born_getbsf) { *status = 2; return; }
    g_born_getbsf(nevt, bsfs, status);
}

// CALL FBRIDGE_BORN_GET_EXTRAS(NEVT, EXTRAS, STATUS)
// AV road B (step 9i+9j): EXTRAS(28,NEVT) = the 10 colour-linked Borns, then
// AMP2(1:16), then JAMP2(1:2), for the events of the LAST Born sequence
// call. STATUS=0 on success, 1 if NEVT does not match that call, 2 if this
// backend's library does not provide them.
void fbridge_born_get_extras_(int* nevt, double* extras, int* status) {
    ensure_born_initialized();
    if (!g_born_getextras) { *status = 2; return; }
    g_born_getextras(nevt, extras, status);
}

// CALL FBRIDGE_REAL_CREATE(WHICH, PBRIDGE, NEVT, NPAR, NP4)
// WHICH selects libmg5amc_cudacpp_real<WHICH>.so (1-based, see this file's
// header comment) -- loaded lazily on first use for that WHICH.
void fbridge_real_create_(int* which, long long* pbridge, int* nevt, int* npar, int* np4) {
    ensure_real_initialized(*which);
    g_real_create[*which](pbridge, nevt, npar, np4);
}

// CALL FBRIDGE_REAL_DELETE(WHICH, PBRIDGE)
void fbridge_real_delete_(int* which, long long* pbridge) {
    ensure_real_initialized(*which);
    g_real_delete[*which](pbridge); // fbridgedelete_nohwreset_: no GPU reset
}

// CALL FBRIDGE_REAL_SEQUENCE_NOMULTICHANNEL(WHICH, PBRIDGE, MOMENTA, GS, RNDHEL,
//                                          RNDCOL, MES, SELHEL, SELCOL, HELONLY)
void fbridge_real_sequence_nomultichannel_(
    int* which,
    long long* pbridge,
    double* momenta,
    double* gs, double* rndhel, double* rndcol,
    double* mes,
    int* selhel, int* selcol,
    int* helonly)
{
    ensure_real_initialized(*which);
    g_real_sequence_nomc[*which](pbridge, momenta, gs, rndhel, rndcol,
                                 nullptr, mes, selhel, selcol, helonly); // 381 port: igraph=nullptr (no MLM)
}

// CALL FBRIDGE_REAL_LIBRARY_EXISTS(WHICH, EXISTS)
// Pure existence probe: sets EXISTS to 1 if libmg5amc_cudacpp_real<WHICH>.so
// is present on disk at the configured lib dir, 0 otherwise. Does NOT
// dlopen or initialize anything -- lets the Fortran side (vfold_real_batch.f90)
// decide, per NFKSPROCESS, whether to route through the bridge or fall back
// to scalar SMATRIX_REAL, without paying for (or risking a fatal error
// from) a real dlopen attempt against a library that was never built for
// this process. WHICH outside [1,kMaxRealLibs] is treated as "does not
// exist" rather than a fatal error, so callers can probe upward without
// having to know kMaxRealLibs themselves.
void fbridge_real_library_exists_(int* which, int* exists) {
    if (*which < 1 || *which > kMaxRealLibs) { *exists = 0; return; }
    const std::string path = real_lib_path(*which);
    *exists = (access(path.c_str(), F_OK) == 0) ? 1 : 0;
}

} // extern "C"
