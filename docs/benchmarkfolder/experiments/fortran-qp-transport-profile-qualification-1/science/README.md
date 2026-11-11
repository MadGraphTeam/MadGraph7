# Private QP transport profile — source proposal

This is an **uncompiled, unexecuted instrumentation proposal**, not an
optimization or performance result. It observes the closed current-source
`qcd_n3_uux_zgg` export from `entry-latch-cache-guard-1`, with production source
`0fcd2cd873d61d134837e1d9e87afb13a47824fd` and unchanged native DSO
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
The source-only transform, exact generated preview and diff are reviewable here.
There is no compiler/native controller in this capsule and no runtime admission.

The first question is whether naturally repeated, eligible QP tensors incur
significant Fortran kinematic/decimal transport costs. No cache is added. The
existing native whole-tensor lookup runs after Fortran encoding; its hits still
decode all output records. The generic MadLoop tensor cache is DP-only and is
absent in ordinary Born/single-order exports. Repeated helicities must not be
assumed: steady Born reductions already use helicity-summed numerators.

## Fixed scope

The existing worker's model setup, bootstrap and complete 100-point steady
learning sequence remain byte-identical. Its private derivative accepts one
`PROFILE 1 MODE` command after READY, then evaluates saved points **1 and 16**
once each with the existing automatic stability policy. It rejects `RUN`.
Point 1 has accepted status 219 (DP); point 16 has status 323 (genuine QP after
automatic rescue), with maximum retained ordinary-reference relative residual
`2.775086784358554e-12`. This is the ordinary real-mass QCD case, not a fixed-QP
or unchecked CMS diagnostic. No fresh point, changed numerator, fixed-QP mode,
threshold, card, scale prescription, filter or shortened learning stream is used.

`input-pins.json` binds both baseline/current source and build receipts, the
full point stream, model identity, physical/numerical cards, installed Fortran
source/archive, DSO, compiler settings and the closed original observations.
`expected.json` retains both orders' exact binary64 result words and original
point text. The baseline 9228 export is an identity/reference binding only;
this proposal does not run a new baseline performance comparison.

Three separate fresh workers are proposed, in fixed order:

1. Mode 0: hooks disabled, coefficients/status control against retained results.
2. Mode 1: stage clocks and native QP cache snapshots; no shadow-byte copying.
3. Mode 2: untimed exact-byte/key trace and cache snapshots; no stage clocks.

Each worker independently performs the unchanged full learning sequence first.
Original campaign files, artifacts, cards and filters remain read-only. Private
clones have local cards/resources/filters. The modes never concatenate state or
reuse tensors across workers/events. The extra selected calls are diagnostic;
they do not update any ordinary qualification or precise timing gate.

## Instrumentation and interpretation

The small `qp_profile.f90` module reuses the prior profile's integer
`SYSTEM_CLOCK` convention. Seven buckets cover initialization/budget, kinematics,
allocation/input encoding, inclusive FFI, output decoding/mapping, contraction,
and clearing. Existing numerical statements, validation and returns are kept.
Incomplete intervals on failed calls are omitted; status/validity is retained.
DP bucket four includes its complete Fortran module call; QP bucket four covers
the native FFI call. They are labeled separately and must not be equated.
Only clocks with matching count/precision/direction scopes may be compared.
No empty-clock correction, confidence interval or improvement estimate is made.

The native six-counter API is queried before/after QP FFI and immediately before
real cache clearing, including failed QP transport cleanup. Queries are disabled
throughout learning and mode 0; pre-clear queries never create a missing context.
Counters are stored as raw snapshots, so resets cannot appear as negative hits.
Independent-check directions remain uncached. New timing code does not alter the
native store, budget, plan, precision, method choice or failure handling.

Mode 2 records exact memory bytes using `TRANSFER` and hexadecimal output:
pre-switch PL/M2L/MU_R/OSThres, native shifts/invariants/masses/scale, and actual
decimal input records. Metadata distinguishes call/event epoch, context
generation, direction, topology/rank, precision, finite/pole mode and cache
eligibility/budget. No floating conversion or comparison is used to capture keys.
All output formatting occurs after each selected point. The 64 MiB trace cap
fails explicitly on overflow rather than silently losing events.

`analyze_shadow.py` counts only primary CTMODE=4, full, cache-enabled, validated
QP requests. Keys include exact inputs and all observed policy/context fields.
An equal pre-kinematic key mapping to different native records fails analysis.
Failures clear the hypothetical opportunity history; check directions never
populate it; event epochs prevent cross-event counts. These are **potential
exact repeats without a budget simulation**, not actual Fortran cache hits or a
speedup. Any future cache must share the existing worker budget with native
stores, preserve all clearing boundaries, and recontract the current numerator.

## Proposed bounded stages — not admitted

Preparation copies the already qualified export and unchanged model/numerical
objects to a new ignored directory; it never writes through source/lib symlinks.
Before any build, verify all source/object/library/flag receipts using the
existing benchmark provenance helpers and retain exact copy hashes. Stop on
any discrepancy. Keep the original prefix, source and executable immutable.

One serial Fortran-only build, proposed cap **180 seconds including cleanup**,
would compile the profile module, the private `mtirust.f90`, two generated
interfaces and private worker, then link existing numerical objects. Use GNU
15.2 and the retained recipes: generated units keep their recorded flags;
the installed module's observed recipe is `gfortran -O3 -fPIC` (installation
log pinned). Place private module objects before the original archive in the
link and verify all symbol resolution/runtime dependencies. Only these five
objects may change. No Rust, dependency, model, HELAS or polynomial rebuild.

Each of the three proposed native workers has **300 seconds total**, including
full learning and 60 seconds reserved cleanup; total proposed admission budget
is **1,080 seconds** including the 180-second build and three native stages.
Lock admission consumes the relevant native stage budget. Use the already
reviewed StageDeadline/OwnedChild/SignalRequests lifecycle, observing actual
descendants and stopping on uncertain cleanup; no plain external timeout.
Nonexclusive CPU 12 is a resource choice, not reservation. Every native worker
holds `.tirust-build/numerical-tests.lock`; explicit coordinator admission is
required because other diagnostics may use it. No retry or alternative point
selection is authorized. Deadline/refusal/nonfinite/parity/status changes or
trace overflow retain an incomplete attempt and stop remaining stages.

After each worker, require the full warmup parser's 100 steady points; compare
selected coefficients/status/accuracy against `expected.json` and mode 0; retain
all differences. Floating-environment parity is not established by these source
tests and remains an execution-qualification requirement before interpreting
small timings. Hash private/current inputs before/after and retain actual
loader identities and terminal/cleanup outcomes. Exact build/worker command
supervision must be reviewed before executing the proposal.

If inclusive FFI dominates, a later separately reviewed extension can reuse the
existing private bridge profile and upstream `DirectWorkspace::set_profiling` /
`last_profile` to split MPFR parsing/reduction/output formatting. That is not part
of this Fortran-only proposal. Final precise timings remain Hua-Sheng's local TODO.

Pure source checks, without compiler or reducer:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/fortran-qp-transport-profile-1 \
  -p test_source.py
```

The source emitter is append-only. `prepare.py` created `source-preview/` once;
tests re-render in memory against the exact original hashes instead of replacing
that preview. Source-preparation anchor/syntax failures were corrected before the
first seal; no build or physical attempt resulted from them.
