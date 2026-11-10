# Closed QCD n3 genuine-QP diagnostic — source proposal 3

This proposal reuses the exact successful source2 build2 executable and all
fifteen original donor runtime libraries. It changes only target library
resolution and bookkeeping. No compilation, linking, loader trace or amplitude
is performed during proposal preparation. Pure tests use synthetic traces,
text providers, temporary directories and fake supervisors.

Closed run2 traced successfully, then rejected three compiler support providers
with different bytes before creating any worker. Its complete failed attempt
is retained in failed-run2-retention-1. Static ELF audit records all hashes,
SONAMEs, GNU build-ID absence where applicable, versions and link dependencies.
All three substituted files differ; substituted quadmath instruction bytes also
differ. ABI consistency does not establish numerical equivalence, and none of
those replacement files is admitted here.

All 5,040 permutations of the seven existing library directories fail to
recover the mixed donor providers: quadmath requires chqq before xc0, while
libstdc++ and libgcc_s require xc0 before chqq. The independent additive
run2-donor-closure-resolution-proposal-1 packet preserves this proof.

The future fresh one-shot run3 creates exactly fifteen direct absolute SONAME
symlinks in run3/closed-runtime. Each alias comes from pinned ELF DT_SONAME and
points to its original canonical provider with the original SHA256. The manifest
stays outside that directory. Missing/extra entries, hwcaps subdirectories,
regular files, indirect aliases, retargeting and changed provider bytes reject.
The executable and all original providers lack DT_RPATH; every needed basename
has an alias. No library is omitted or replaced.

The target-only LD_LIBRARY_PATH names only that private directory. All outer
host utilities and Python retain the clean source2 environment. Unchanged
exec_target.py verifies executable bytes and the original glibc2.40 PT_INTERP
before target execve. No LD_PRELOAD or loader substitution is introduced.
Historical host-ldd loader exclusion is unchanged; fifteen target libraries
remain mandatory. Source2/build2/run2 are read-only, and source3 has no build
stage or relink. Exact existing executable SHA256 is
ed6a51fff0312b06a45cd97a409aafa7d2447eef5ae7800579b44c0c904b8474.
Link-map/compiler-support observations first pinned after the original link
remain labelled post-link; no retrospective pre-link claim is made.

All source/object/runtime/executable dependencies and aliases are checked before
the first trace and each worker. Each worker has a separate strict pre-entry
loader trace, followed by another integrity check. Existing actual /proc maps
validation rejects an unexpected ELF and retains original hashes/CPU12 checks.
Those maps are observational during execution, not a pre-entry barrier or
continuous race-proof monitoring. A passing trace is not an amplitude result.

The scientific specification is unchanged: qp-specification1 index
741813a73a19891091f5597d8032033e2e1723866c45b10f7f8ebacbe6c1c9a9.
Driver, observers, analyzer, process supervisor, exec wrapper, shape checks,
preparation provenance and every fixture remain byte-identical to source2.
Existing compiler flags/generated numerical objects are unchanged. prepare.py
is retained provenance and must not be rerun over this sealed proposal.

Sixteen serial workers cover backends1/6/8/9, modes4/5 and plain/observed arms.
Each uses one point1 bootstrap, at most eight initialization calls, then physical
points1,93,39 in order: at most192 public calls including48 diagnostic calls and
96 returned order columns. Initialization retains ordinary checked policy;
only afterward are forced stability checks disabled and unchecked modes4/5
selected. MLStabThres1e-6, ImprovePSPoint2, all QP numerator flags, full finite and
pole evaluation, cards and journal limits200000records/128MiB remain unchanged.
Only backend ID changes in each card. There is no adaptive bridge or fake failure.

Observers copy opaque bytes without changing arithmetic or IEEE state. Exact
plain/observed parity and common physical/Born/model/repair/QP-numerator equality
are required before coefficient comparison. Native bases are recorded separately:
CutTools keeps its DP denominator ABI with QP callbacks, Ninja uses binary128,
and bridges retain decimal113 transport. Missing nonzero groups, native failure,
W03 warnings, nonfinite outputs, incomplete traces or observation drift fail.
Status120/accuracy-1 stay unchecked. Protocol success is not numerical truth,
tolerance acceptance, campaign acceptance or a performance claim.

The nonblocking common lock precedes run3 creation; busy75 is a scheduling hold.
CPU12 remains nonexclusive, one thread and4GiB. The shared3000s stage includes
integrity checks and all17 possible traces (30s caps) plus16 workers (180s caps).
The first failed worker stops execution; no retry, capacity expansion or point
replacement is allowed. All failures and warnings remain retained.

Pure tests currently pass58 cases; an independent run also passes55 with process
and exec calls forbidden outside injected fakes. mechanical.patch exposes the
source2 delta; mechanical-change.json lists preserved scientific payloads and
prior evidence. Actual run requires separate root review and admission.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal3/run.py verify
# Only after root admission:
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal3/run.py run
```

The future controller depends on named closed workspace donors. A portable
scientific replay capsule can only be made after an actual closed attempt.
