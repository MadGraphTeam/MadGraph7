# Production direction-workspace optimization

The failed qualified n1 benchmark motivated the isolated attribution in
`../bridge/` and the seven-triad whole-amplitude experiment in
`../whole-amplitude/split-workspace-1/`. The latter measured a split/original
ratio of 0.978537 with 95% interval [0.963998, 0.993296] and bit-identical saved
amplitudes. It remains diagnostic evidence: the split/baseline ratio was
1.113072 [1.099917, 1.126384], above the requested limit, and it was not a quiet
acceptance run. A new campaign must identify the optimized source and library.

## Implementation and scope

The production bridge keeps one immutable full-DP reduction plan per `(N,rank)`
and two embedded workspaces keyed by the independent-check flag. Each workspace
retains its own geometry variants and scratch buffers, avoiding repeated
selection when propagator order reverses. The check workspace remains detached
from the numerical tensor store. Any failed full-DP call clears the store and
removes both directional workspaces while retaining the immutable plan.
Numerator contraction and stability decisions remain in the existing Fortran
adapter. No profiler code or extra public ABI symbols enter production. The exact source,
test and documentation diff is retained in `production-change.patch`;
`production-change.json` identifies its Git base and before/after source hashes.

This changes structural reuse, not the numeric cache budget or precision
policy. The existing full-DP embedder may select another DP reduction method;
its default partial rescue and precision-tier options remain disabled. The
QP and accelerated pole-only paths are unchanged.

## Native unit tests

`native-test-build-provenance.json` records the exact Cargo invocation, compiler
and manifest/lock/source identities; `native-test-build.jsonl` and
`native-test-build.log` retain the build output. `run_native_tests.py` runs the
identified executable under the shared numerical lock. All **11 tests passed**:
the original nine plus tests for persistent separate workspaces sharing one
plan, check-cache bypass, and both-direction invalidation/recovery after an
upstream overflow refusal. The original tests also exercise QP transport,
complex masses, finite skipping, selection and cache lifecycle. Unit tests
inspect workspace/plan identity directly; this is stronger than inferring
object lifetime from external cache counters.

## Release reconstruction and qualification

`build_release.py` builds only the production bridge against the exact existing
release dependency artifacts. It verifies their SHA256 hashes and uses Rust
optimization level 3, thin LTO and one codegen unit. This is an incremental
release reconstruction, not a fresh network installer qualification. The
separate clean-installer campaign captured the earlier bridge source and must
not be cited as evidence for this optimization.

The new prefix is `.tirust-build/tirust-split-install`. The original
`HEPTools/tirust` prefix is retained intact for the frozen c91 campaign. The
Fortran archive/module, dependency notices and locked numerical dependencies
are unchanged. The new prefix records an explicit rebuild provenance object
and updated bridge source/artifact hashes. Its metadata retains the original
dependency build provenance because those exact artifacts were reused.

The release builder checks that exported symbols exactly match the original
production ABI and contain no profiler entry points, then compiles and runs the
actual Fortran `smoke --require-poles`. The first invocation built the DSO but
its smoke compile selected an ignored stale `vendor/tirust/mtirust.mod` beside
the source, before the installed module search path. That module lacked the
context-generation accessor. `build_release_attempt1.py`, the preparation
record and `release-smoke-build-attempt1.log` preserve this failure.
`finish_release.py` compiled an exact copy of the smoke source from the isolated
build directory and the installed module, without rebuilding or changing the
DSO. That smoke passed. The original stale artifact is identified by hash and
was not removed. The corrected `build_release.py` uses this isolated source
copy on future builds. Installer staging already excludes stale modules and
rebuilds its module; its qualification separately compares the smoke-visible
and installed module hashes. `qualify_release.py` verifies availability
through MadGraph's installer checks, records runtime dependencies, and runs
`validate_release.py` against the old and new release libraries in sequential
processes under the numerical lock. The release validator is the original
90-call isolated validator with unchanged fixtures/assertions, the temporary
profile-control call removed, and explicit release metadata identities added.
Both versions are retained for source comparison. Numerical calls cover full
and partial DP tensors, cache bypass/budgets, changed inputs/restoration, rank-6
scaleless moments, complex masses and expected error recovery. QP is covered by
the native unit and Fortran smoke tests, not by the 90-call DP comparison.

Run from the MadGraph root with the pinned environment and fresh output paths:

```sh
source tools/tirust_env.sh
export PYTHONDONTWRITEBYTECODE=1
python docs/benchmarkfolder/experiments/n1-profile/production/build_release.py
python docs/benchmarkfolder/experiments/n1-profile/production/qualify_release.py
```

The release builder refuses to overwrite an existing prefix, and the validator
refuses to overwrite its evidence. Replays must use new destinations. Native
executables are kept under `.tirust-build`; this evidence folder contains only
source, commands, hashes and text results. The final release result and exact
DSO identity are recorded in `release-qualification.json` once qualification
completes; absence of that record means the release is not yet qualified.

## Qualified result

The new release DSO is
`92da56e247d793e51b9e6b85154996fd056e754b3224ba15ecc0a8ff2004f3d6`,
built from bridge source
`455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c`.
All 11 native unit tests and the installed Fortran smoke passed. The release
comparison made 90 calls per library: 80 successful evaluations, eight expected
invalid-input errors, and two expected overflow refusals. All 110 event records
are identical, including error messages, output bytes, sentinels and cache
counters. All 13 optional fixture calls were served successfully. The native
unit tests separately confirm destruction/recreation of both workspace objects.

MadGraph recognizes both prefixes as available; all 12 production exports match,
without profiler symbols. DT_NEEDED and RUNPATH are unchanged: the Linux loader,
GNU compiler runtime, libm and libc are system requirements; GMP and MPFR remain
linked into the numerical library. No system C runtime is bundled. The original
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`
DSO and every file in its installed library directory remain unchanged.

`release-qualification.json` and the portable `evidence-index.json` bind the
retained result files. This qualifies the interface and lifecycle optimization;
a new full physics/runtime campaign and an actual current-source CLI installer
publication remain separate gates.
