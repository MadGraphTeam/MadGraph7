# Isolated bridge attribution

These scripts create a diagnostic copy of the bridge for the failed qualified
`qcd_n1_uux_z` timing gate at integration revision `c91cb85f3a`. They never edit
the production bridge, installed prefix, original benchmark processes, or
baseline. The matching Fortran profiler uses the actual generated process and
the same phase-space points and cards.

`prepare.py` applies timer instrumentation to a copy of `vendor/tirust/src/lib.rs`
under `.tirust-build/n1-profile/bridge`, with `profile.rs` as a temporary module.
`build.py` directly invokes the pinned Rust compiler against the exact release
dependency artifacts, verifies their SHA256 hashes, and retains the release
optimization settings (`opt-level=3`, thin LTO, one codegen unit). No Cargo or
upstream source rebuild is involved. Build commands, source and library hashes
are recorded alongside the isolated DSO.

The temporary ABI adds `ml_tirust_profile_control(level)` and
`ml_tirust_profile_json(buffer,capacity)`. Control returns zero on success and
resets the calling thread's counters. JSON returns the required byte count
including its terminating NUL; a null buffer queries the size. Levels are:

- `0`: timers and upstream profiling disabled, for instrumentation-cost checks.
- `1`: bridge timers and lane-route counts.
- `2`: bridge timers plus the existing upstream `set_profiling`/`last_profile`.

Changes to the upstream profiling flag are applied once to existing workspaces
when the requested level changes; new workspaces inherit the requested level.
The embedded reduction algorithm, precision policy and cache attachments are
unchanged. Primary and independent-check calculations keep their original
cache behavior. The profiler additionally counts direct, method-choice,
precision-tier and refused lanes.

Counters are grouped by `(N, rank, check_pass, cache_enabled)`. This experiment
fixes finite computation and pole selection; snapshots must not combine other
settings. Timers report input-vector construction, lock acquisition, workspace
setup, tensor-store attachment, reduction, output packing and final copying.
Separate counters cover cache clearing, cache-budget calls and plan registration.
Upstream durations report lane input/geometry, scalar masters, recurrence and
its total reduction time, as detailed below.

Times are **inclusive and overlapping**: the FFI total contains the state call,
which contains store attachment/reduction/packing; upstream stages lie within
reduction. Do not sum all columns. Timer and bookkeeping costs are not
subtracted. Compare profiling levels and the original installed-library control
before interpreting small components. First snapshot-row creation and a
requested profiling-level change also incur diagnostic overhead. Inner-stage
timers complete only on successful returns, so decomposition requires
`failed_calls == 0`. The production error checks and cleanup remain unchanged.

Build only after the benchmark controller releases the timing window:

```sh
source tools/tirust_env.sh
export PYTHONDONTWRITEBYTECODE=1
python docs/benchmarkfolder/experiments/n1-profile/bridge/prepare.py
python docs/benchmarkfolder/experiments/n1-profile/bridge/build.py
```

Numerical runs must hold `.tirust-build/numerical-tests.lock` exclusively. These
instrumented durations are attribution evidence, never an acceptance timing.
Any proposed optimization needs separate numerical qualification and an
uninstrumented paired runtime gate.

## Retained attribution

The matching Fortran runner's `../fortran/attribution-1/results.json` contains
18 runs: three repetitions of the pristine library and each diagnostic level,
with 200 traversals of the same 100 points per batch. All saved numerical outputs
from the final traversal are bit-identical; statuses cover all timed points.
Every enabled bridge snapshot contains 20,000 primary and 20,000
independent-check calls, all `N=3`, rank 2, direct DP lanes. There are no method
choice, precision tier, refused or failed lanes. The profiling library hash is
`aef3b0fb1404bdcd12e31ceee9eea57ee431a48d1eea922cd879436e7c9cc732`.

Reproduce the independent aggregation without running native code:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/n1-profile/bridge/analyze.py \
  docs/benchmarkfolder/experiments/n1-profile/fortran/attribution-1/results.json \
  --output docs/benchmarkfolder/experiments/n1-profile/bridge/analysis.json
```

The JSON records all observations and their ranges, not just medians. Whole
amplitude medians in microseconds are 70.773 for pristine, 70.672 with both
profilers disabled, 72.262 for Fortran-only profiling, 72.365 for bridge level 1,
75.192 for bridge level 2, and 74.698 with both profilers enabled. Some individual
runs are substantially slower (the pristine range is 70.616–116.919 us). Three
repetitions under shared-host variation support coarse attribution; they do not
provide a new runtime acceptance result or a causal estimate from differences
between profiling levels.

Bridge-only level 1 gives these median costs per amplitude, which contains two
tensor calls. Each timer includes its instrumentation overhead.

| Measured region | us per amplitude |
| --- | ---: |
| Complete DP FFI call, inclusive | 8.886 |
| Reduction call, within FFI | 6.382 |
| Input validation and batch packing | 0.487 |
| Workspace lookup/setup | 0.315 |
| Tensor-store attachment | 0.163 |
| Output allocation and packing | 0.189 |
| Final output copy | 0.082 |
| Mutex acquisition | 0.074 |
| Cache clearing, separate entry points | 0.323 |
| Cache-budget setter, separate entry points | 0.099 |

Level 2 adds the upstream profiler. In the embedded direct engine its public
`arguments` field means **lane input and geometry**, including Gram matrix
solves, variant selection and conditioning. It does not measure Symbolica
expression evaluation here. This mapping is explicit in
`crates/iregi-rust/src/dispatch/direct_hybrid.rs`, `add_direct_profile`:
`arguments += geometry`, `scalar_masters += masters`, and
`reduction += recurrence + gather + output`. The observed medians are 2.856 us
for geometry, 1.148 us for scalar masters, and 1.961 us for recurrence, gathers
and output, within an upstream total of 7.083 us. These more deeply instrumented
times must not be substituted into the less instrumented level-1 total.

The measurements rule out store attachment, final copying or the budget setter
as a sufficient explanation of the failed runtime gate: each is a small part of
the call. Removing their entire measured cost would still not establish the
required speedup. Input packing and Fortran preparation may provide incremental
gains, but the largest measured backend region is the reduction itself,
especially low-dimensional geometry. Any optimization hypothesis needs an
isolated numerical check and uninstrumented comparison before production
changes or a claim that the 5% gate is met.

## Direction-workspace probe

`context_probe.py` tests one concrete explanation for the geometry cost with the
existing profiling DSO. A fixed massless triangle has lightlike bubble masks
`{3,6}` in one propagator ordering and `{5,6}` after reversing the two nonzero
shifts. The direct executor retains its last active lightlike-plan variant.
Alternating orderings in a shared workspace can therefore reject the previous
variant and repeat geometry while selecting the appropriate cached plan. The
relevant upstream code is `direct/exec.rs`, `select_variant` and the geometry
retry loop in `reduce_batch`.

The probe compares the same alternating calls in one context with calls in two
independent contexts, which retain separate workspaces. Every call clears its
context's numerical cache first; independent-check calls remain detached from
the tensor store. This changes no library source and shares no computed tensor
between directions. The exact input vectors, library/source hashes and all
profiles are retained in `context-probe.json`.

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/n1-profile/bridge/context_probe.py \
  --library .tirust-build/n1-profile/bridge/libmadloop_tirust.so \
  --output docs/benchmarkfolder/experiments/n1-profile/bridge/context-probe.json
```

The script owns the numerical lock. It runs three repetitions at profiling
levels 0 and 2, with 10,000 primary/check pairs per configuration. All 240,000
measured tensor evaluations match the per-order reference byte for byte; the
warmup calls are checked too. No call fails or uses a method-choice/precision
tier. The level-2 median geometry time per pair drops from 6.191 us with one
context to 2.447 us with two; inclusive FFI time drops from 19.468 to 14.368 us.
All three paired repetitions improve. Level-0 Python-loop medians are 32.989
and 29.422 us per pair respectively, but include ctypes dispatch, assertions
and cache-clear calls, so they are only an instrumentation-disabled control.

These absolute timings differ from the full-process attribution: the probe
uses fixed synthetic kinematics and a Python caller. It establishes an
optimization hypothesis, not a MadLoop speedup or acceptance result.
`split_experiment.py --build` then creates an isolated library implementing
only separate full-DP workspace keys `(N,rank,check_pass)` within one context.
It preserves all cache attachments and removes both directional workspaces on
a failed full-DP reduction. The exact change is `split-workspaces.patch`; the
preparation and build records bind its inputs and output. Production and the
original profiling library remain unchanged.

The split DSO built successfully with SHA256
`1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7`.
`validate_split.py` ran both DSOs in separate sequential processes under the
numerical lock. Each executed 90 calls: 80 successful evaluations, eight
expected invalid requests, and two deliberate overflow refusals. All per-call
status codes, output bytes and cache-counter snapshots are identical between
libraries. Both overflow directions preserve the caller's output sentinel,
empty the numerical store and recover correctly in both directions. The
overflow is explicitly rejected by the upstream direct engine as a nonfinite
coefficient; this does not simulate an upstream success containing NaNs.

The checks cover alternating and crossed ordering/check flags, changed
kinematics/masses/scale followed by restoration, cache on/off, unchanged and
zero budgets, all pole selectors, finite-skipping mode, analytic scaleless
rank-6 bubble moments, and a complex-mass rank-4 box. Every optional mutation
and complex-box fixture was served successfully in this run. QP code is
unchanged and is outside this experiment's test scope. External tests establish
empty-store and recovery behavior; removal of both workspace objects is
established by the retained source diff. Results and exact library/source hashes
are in `split-validation.json` and `split-validation-workers/*.json`.

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/n1-profile/bridge/validate_split.py \
  --output docs/benchmarkfolder/experiments/n1-profile/bridge/split-validation-replay.json
```

The validator requires a fresh output path.

## Whole-amplitude follow-up and production change

The uninstrumented, seven-triad diagnostic in
`../whole-amplitude/split-workspace-1/results.json` compares the original
NinjaRS baseline, original TIRust library, and isolated split-workspace library
on CPU 9. Every saved original/split amplitude coefficient is bit-identical.
The split/original ratio is 0.978537 with a 95% paired-bootstrap interval
[0.963998, 0.993296], supporting an approximately 2.15% improvement. The
split/baseline ratio remains 1.113072 [1.099917, 1.126384], so this experiment
does not establish the required 5% limit. Concurrent shared-host work makes
these diagnostic timings ineligible as a new acceptance result.

Three additional split profile snapshots each contain 20,000 primary and
20,000 check calls, all direct DP with no failures or fallback. Their median
geometry cost is 1.330395 us per amplitude (range 1.321776–1.357345); complete
FFI cost is 10.703622 us, including 6.944414 us of upstream work. These values
must not be causally subtracted from earlier CPU-1 profiles.

The measured improvement justified the production change recorded separately
in `../production/`. Production uses separate primary/check full-DP workspaces
and additionally shares their immutable reduction plan. It contains no timer
instrumentation and invalidates both workspaces after a failed full-DP call.
The original installed library and the failed c91 benchmark remain unchanged.
The historical preparation scripts in this folder target the original bridge
source at c91cb85f3a; running them against the newer production source is not a
replay of the recorded experiment. Preserve the recorded source revision and
artifact hashes when reproducing this attribution.
