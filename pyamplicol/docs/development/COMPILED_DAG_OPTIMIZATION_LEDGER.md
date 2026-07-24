<!-- SPDX-License-Identifier: 0BSD -->

# Compiled-DAG Optimization Ledger

## Scope and acceptance

This ledger covers only the existing fused compiled-DAG execution lane. It
does not introduce recurrence/eager execution, a plane-aware evaluator ABI, or
color-contraction redesign.

Performance claims use warmed native wall markers after caller-side momentum
packing. Every retained comparison uses byte-identical artifacts, momenta, and
selectors; at least five independent samples; and reports median and median
absolute deviation (MAD). Numerical comparisons use `rtol=1e-12` and
`atol=1e-15`.

## Authoritative inputs

- Feature base: `58dbba1234643508e6fc33cfa6d990a8f520dad8`
  (`origin/main`, fetched 2026-07-23).
- Design memo checkout: `f52081aec80944d36d35d883c9abcc30a2f840e0`.
- Design memo SHA-256:
  `4d74022551b2103f4d844a9e4d26ca3599a4c115274bdaf041f9139f5aa137a7`.
- Canonical `Cargo.lock` SHA-256:
  `58fec7351b18f93bb38b41bea5ecdad3b0b2f1bd4d5d3a302e7f471c58ccc9c0`.
- Candidate `Cargo.lock` SHA-256:
  `d67fc57d0def30dd4642a57c651a34529e0a7e4102d13822daa78fde6a438735`.
- Contributor lock SHA-256:
  `c1a56124b1d0a38051a0e202bdea72ce55e07e1f94158922c7e1b240531e2824`.
- Candidate native-input fingerprint:
  `2cbd458e0fd6700ea7c63dac7768ffae31f4bba5e3f9a23f34d863fcbeb4372d`.
- Baseline candidate fingerprint: `86379ddd8713`.
- Baseline wheel SHA-256:
  `72b11fb9d402532ebc81a210351cb428b4cc61bac23d02e0be488a0e2fa2c598`.

The strict release dependency lane is intentionally fail-closed at this base:
`dependencies/release-lock.toml` marks the required Symbolica/SymJIT
combination unverified, and candidate source distributions are forbidden.
Candidate wheel/source/deployment gates remain authoritative executable
evidence; the expected release/sdist failure is retained rather than weakened.

## Baseline host and build

- Host: Apple arm64, macOS 15.0 (Darwin 24.0.0).
- Python: 3.12.6.
- Rust: 1.89.0.
- C/C++: Apple clang 16.0.0.
- Build: candidate overlay, release profile, thin LTO, one codegen unit,
  optimization level 3.
- Initial candidate runtime build watchdog peak: 1.739 GiB.
- Primary artifact generation watchdog peak: 5.084 GiB.

The ordinary candidate wheel gate at the base failed before runtime compilation
because the packaged eager-model metadata did not match the active candidate
dependency lock. The focused candidate runtime was therefore built through the
same repository overlay without staging packaged eager assets. This pre-existing
base defect is kept separate from compiled-DAG results.

## Primary retained artifact

- Workload: compiled JIT O3 `u u~ > Z g g g g g g`, LC
  `all-flow-union`, all flows, fixed runtime helicity
  `h:-1,+1,-1,+1,-1,+1,-1,+1,-1`.
- Artifact ID:
  `8bf4bda1942ee60de3380fca9e0444373852ba4423f0fbb1869d346ee7f77d81`.
- Artifact manifest SHA-256:
  `885b1b84392c99e07de2674bfe82460c31006809749993aa0f6a8782306a3fa7`.
- `evaluators.pacbin` SHA-256:
  `b2c5ee4312002e683828ad12a61d860fe02af87eb7f5d924edee94d07efb47a5`.
- Validation momenta SHA-256:
  `b9b6b525bd227d3c4d5021d6bf09a24cd61aa34ebd4f76139d9c12e2b59ea7b9`.
- Payload size: 436,182,463 bytes; 19,752 evaluator members.

## Non-union retained artifact

- Workload: compiled JIT O3 `u u~ > Z g g g g g g`, LC
  `topology-replay`, runtime color flow
  `flow:2,4,5,6,7,8,9,1`, and no helicity selector (all retained
  helicities are summed).
- Exact-base source: `58dbba1234643508e6fc33cfa6d990a8f520dad8`;
  the disposable generator's native module is byte-identical to the frozen
  baseline module (SHA-256
  `92031a0d5555d3d5b5b0215fb89e4e13578eb64d3d313617281b80c62ce28e7f`).
- Artifact ID:
  `bf35165c70100605dfc41e8572fb40a1c726f8b90565b4c058b7462d679bc431`.
- Artifact manifest SHA-256:
  `6be814c96e1afe64f11a5bd89711a807c4abc69c467dd5bd2a6963d9b7b2dcbe`.
- `evaluators.pacbin` SHA-256:
  `a3d87242ea6e3e04a1c4c0f443860e3aa1b3130b5eb8aad6dd58b803702de3c3`.
- Execution manifest SHA-256:
  `5152df831dda5a496a3bf3240e0f96ebee8bbac23ff96e6d5be4e5a64a3f1a05`.
- Validation momenta SHA-256:
  `b9b6b525bd227d3c4d5021d6bf09a24cd61aa34ebd4f76139d9c12e2b59ea7b9`.
- Artifact tree SHA-256:
  `a17ea2c7d0daf63d1273cb6fccb9d984818ad0dfccf4e7c38de0bf48ce8d7844`;
  20,800,596 bytes in ten files.
- Physics coverage: 720 physical flows and 768 physical helicities, represented
  by one computed flow and two computed helicities.
- Exact-base generation watchdog peak: 0.263 GiB.

## Exact pre-change baseline

Five independent blocks were measured per batch. The native unprofiled timer
starts after Python/NumPy momentum packing. A paired instrumented pass used the
same batch, selector, and repetitions.

| Batch | Repetitions | Median ms/point | MAD ms/point |
|---:|---:|---:|---:|
| 1 | 341 | 3.5131 | 0.1396 |
| 64 | 28 | 0.5974 | 0.0122 |
| 128 | 14 | 0.5936 | 0.0042 |
| 1024 | 2 | 0.5694 | 0.0086 |

At batch 64, the instrumented median was 0.5810 ms/point. The existing
exclusive fields reported 0.0449 ms stage-union packing, 0.3468 ms evaluator
calls, 0.0493 ms output assignment, 0.0014 ms amplitude packing, 0.0076 ms
amplitude evaluation, and 0.0161 ms reduction. Together with source and
momentum setup they explain only about 82% of wall; the hidden leaf gather and
materialization/copy work remain in the residual.

The current-host baseline already outperforms the memo's historical
1.704 ms/point figure by roughly 65%. Optimization acceptance therefore uses
the exact local baseline above, phase ownership, and statistically robust
relative changes rather than treating the memo number as current provenance.

## Milestones

1. Add exclusive accounting and deterministic movement/call/allocation
   counters explaining at least 98% of profiled wall.
2. Split genuinely unprofiled compiled execution from diagnostic execution.
   Measure both LC workloads independently: non-union single-flow/helicity-sum
   and all-flow-union all-flows/fixed-helicity.
3. Add borrowed flat momentum views and direct caller-output execution while
   keeping allocating APIs as wrappers.
4. Add totals-only selector execution without resolved-component
   materialization.
5. Precompose global-state-to-leaf mappings and gather leaf inputs once.
6. Retain authoritative output slabs only if interleaved A/B measurements show
   a robust gain without correctness or RSS regressions.

## Milestone 1: exact accounting

Status: implementation and focused rebuilt-runtime validation complete. Full
CI, packaging, and release validation are intentionally deferred to the final
optimization integration.

The benchmark headline now always comes from the warmed native unprofiled
`_benchmark_f64_wall_time` pass. A paired `profile_repeated` pass uses the
byte-identical batch and repetition count only for attribution. Profiled wall
time is never substituted for the performance headline.

The regression driver treats a performance result as authoritative only when
both interpreters use one read-only shared artifact or when all
performance-relevant payloads rehash identically. It fingerprints the installed
distribution, native module, and build information so reinstalling into the
same environment invalidates an artifact cache. Each timing subprocess also
performs a bounded warmed evaluation; the driver compares the native results at
`rtol=1e-12`, `atol=1e-15` and makes correctness part of the top-level gate.

Compiled execution now reports exclusive top-level clocks for native input
packing and crossing, runtime orchestration, reusable-state preparation and
clearing, sources, momenta, model parameters, stage-union input packing, stage
evaluator calls, stage output assignment, amplitude input packing/evaluation,
reduction, totals materialization, runtime output copies, and selector
planning/gather/scatter. Evaluator-call attribution separately exposes
leaf-input gathering, backend calls, chunk-output gathering, and amplitude
output remapping without adding those nested fields to top-level coverage.
Resolved reduction/materialization is explicitly labelled as inclusive within
the reduction timer. Both Rust and Python reject a profile whose exclusive
top-level phases exceed wall time. All timing state that survives generated
evaluator calls is integer-backed `Duration`; this avoids the native evaluator
ABI's use of the full floating-point register file.

Movement counters distinguish actual copied/gathered components from borrowed
inputs. Allocation counters are deliberately narrow: native input-container
and final-output vector allocations are explicit, while
`observed_scratch_reallocation_count` is only a lower-bound count of
capacity-changing reallocations in instrumented hot reusable buffers. It is
not presented as a process-wide allocator count or a peak-memory measurement.

The public `pyamplicol profile` human report exposes the same data in colored
top-level timing, per-stage timing, internal stage attribution, and normalized
native-work-counter tables. Its JSON report retains the typed aggregate and
per-stage fields for machine processing. Internal attribution is explicitly
non-additive: full-stage evaluator envelopes own leaf gathering, while composed
selected-chunk input-pack envelopes own it.

The final rebuilt retained batch-64 accounting measurement used the exact
baseline artifact, the fixed helicity above, seven paired samples, and 24
repetitions per sample. Instrumented wall was `0.466443 ms/point` median with
`0.000850 ms/point` MAD. The independently summed exclusive phases explain
`99.9364%, 99.9353%, 99.9278%, 99.9363%, 99.9248%, 99.9379%, 99.9390%` of
their respective wall samples. Median per-sample coverage is `99.9363%`, and
every sample exceeds the required 98%.

Median diagnostic phase values from that run are:

- Stage-union input copy: `0.037533 ms/point`.
- Previously hidden stage leaf-input copy: `0.069524 ms/point`.
- Evaluator backend calls: `0.303062 ms/point`.
- Stage output assignment: `0.034182 ms/point`.
- Amplitude input/leaf/backend/output-gather attribution:
  `0.000780 / 0.000979 / 0.003275 / 0.001413 ms/point`.
- Reduction/resolved materialization: `0.014192 ms/point`.
- Totals materialization: `0.000416 ms/point`.

This run is accounting evidence, not an optimization claim. Native unprofiled
wall time from the version-independent paired regression harness remains the
arbiter for every optimization milestone.

A final rebuilt CLI smoke of the topology-replay selected-flow/helicity-sum
lane at batch 64 reported `101.352 us/point` paired-profile wall and
`1.128 us/point` unexplained core time: `98.887%` exclusive coverage. It also
rendered the top-level, seven-stage, nested-stage, and counter tables and
reported zero warmed scratch reallocations. This independently confirms the
98% accounting threshold in the second required LC workload; it does not
replace the native unprofiled headline.

The final batch-1 shared-artifact smoke exercises both required LC modes. The
all-flow-union fixed-helicity lane measured `3.220667 ms/point` baseline
(MAD `0.322734`) and `2.997758 ms/point` current, a `-6.92%` diagnostic
change. The topology-replay selected-flow helicity-sum lane measured
`0.271086 ms/point` baseline (MAD `0.006622`) and `0.272144 ms/point`
current, a `+0.39%` diagnostic change. Both gates are authoritative, both pass
the two-percent/no-three-MAD regression rule, and both return bitwise-identical
warmed f64 values across the frozen and current native modules. These timings
still describe the accounting milestone, not a claimed hot-path optimization.

On the second warmed batch-64 profile, observed instrumented scratch
reallocations were zero. Per evaluation the runtime copied 3,037,824
stage-union components and then 4,170,752 leaf-input components, confirming
and quantifying the duplicate gather. It made 50 backend calls, assigned
1,468,544 stage output components, gathered 92,160 amplitude evaluator output
components, materialized 46,080 resolved components and 64 totals, allocated
65 nested native-input containers, and allocated one native output vector.

Evidence:

- `/private/tmp/pyamplicol-compiled-dag-optimization-accounting-z6g-b64-v3.json`
  SHA-256
  `9de86ed2289f692c504e964f2dcad9f3e14b040d1efc22a190d75033f8fbfd4d`.
- Version-independent shared-artifact regression smoke:
  `/private/tmp/pyamplicol-compiled-dag-regression-accounting-final-union/result.json`,
  SHA-256
  `36ce5bbd8c8c90fbfee9f270419e39b8db678ed2f1a0f32f38e870a974e59e7c`.
- Version-independent non-union shared-artifact regression smoke:
  `/private/tmp/pyamplicol-compiled-dag-regression-accounting-final-nonunion/result.json`,
  SHA-256
  `a77d41d7f5b95e2f98f60674e1379def1f9aaf617d6f8c506ee4b7384fb853de`.
  Both results preserve their artifact tree digests, and all twenty lane
  samples use batch SHA-256
  `7c61a6add4c4a75c7fd5239e321e4e31dfc318096d637287cfe44e08d3e64336`.
- Final focused accounting candidate rebuild watchdog peak: `1.287 GiB`.
- Rust/Python check/clippy watchdog peaks: `1.303 / 1.096 GiB`.

## Milestone 2: clock-free ordinary compiled execution

Status: implemented and measured; final robust median/MAD validation remains
part of the cumulative optimization gate.

Ordinary compiled f64 totals, resolved values, global selectors, per-point
selectors, topology replay, and helicity recurrence now use result-only engine
entry points. Those entry points call direct stage, amplitude, and evaluator
loops with no `Instant` reads, timing vectors, profile structures, counter
bookkeeping, or profile-fold allocations. Explicit `profile` entry points
retain the instrumented implementations and the same attribution schema. The
outer `_benchmark_f64_wall_time` marker remains the sole clock in the ordinary
benchmark envelope.

An independent diff audit found one LC-sector selector restoration omission in
the first draft; the result-only path now uses the same set/call/restore guard
as the established profiled path. No other routing, summation-order,
structural-zero, or profile-path divergence was found. The Rust library suite
passes 247 tests with the one documented stale exact-base test filtered.

An early five-block, same-helper A/B smoke used the retained artifacts,
byte-identical deterministic batches and selectors, and a 0.05-second target
per block:

| LC workload | Batch | Accounting `e9b28fe` ms/point | Clock-free ms/point | Change |
|---|---:|---:|---:|---:|
| all-flow-union, fixed helicity | 128 | 0.552008 | 0.537119 | -2.70% |
| all-flow-union, fixed helicity | 1024 | 0.567789 | 0.529601 | -6.73% |
| topology-replay, selected flow, helicity sum | 128 | 0.103348 | 0.085786 | -16.99% |
| topology-replay, selected flow, helicity sum | 1024 | 0.094683 | 0.086619 | -8.52% |

The warmed values are bitwise identical to the accounting build for every
listed batch: union `0x1.98ab33a27f939p-64`, non-union
`0x1.03c2893f24a72p-68`. Batch-1 provisional wall is `2.730917 ms/point`
for union and `0.274444 ms/point` for non-union; the latter is effectively flat
within the earlier batch-1 noise and is not presented as a gain.

Evidence:

- Accounting-build samples:
  `/private/tmp/pyamplicol-compiled-dag-e9b28fe-baseline`.
- Clock-free samples: `/private/tmp/pyamplicol-compiled-dag-clock-free`.
- Current result SHA-256 values for union batch 128/1024:
  `9fa3438cd377c794cdfd4175bdfb7290ce987696a08d7f475afeb05bfc81cd74` /
  `96a5fae2d70bdbbc48883e03e4f9cb9955121d7aa6fc2302f8592b62d36c9208`.
- Current result SHA-256 values for non-union batch 128/1024:
  `b2ca185761a85e78110a26defdad40657b2a3ee938166893fd6ceb7090f6c740` /
  `0673e2779c182487278036aeef8f2d98df424998a3f5d9070de5bef86b1d8c6d`.
- Focused candidate rebuild watchdog peak: `1.290 GiB`; Rust library test
  watchdog peak: `1.053 GiB`.

## Milestone 3: borrowed flat input and direct output

Status: implemented and measured at commits `b707da9` and `e237d5c`; final
robust median/MAD validation remains part of the cumulative optimization gate.

The ordinary f64 lane now borrows contiguous row-major momentum input with
layout `[point][external][E,px,py,pz]`. A prevalidated view applies an alias
permutation on access, while the dominant identity case has a separate
branch-free accessor that performs neither a lookup nor multiplication by
one. Contiguous selector partitions remain borrowed; only genuinely gathered
random partitions and real topology permutations materialize momentum rows.
Noncontiguous NumPy arrays and generic Python sequences retain the validating
owned fallback.

Rust exposes `evaluate_f64_into` and output-last
`evaluate_f64_into_with_selectors`; the established allocating entry points
are wrappers. Python allocates its public result once and the C ABI writes
directly into the validated caller prefix. The native benchmark allocates and
reuses its output before starting the wall clock. Existing resolved APIs,
native signatures, artifact schemas and evaluator ABIs are unchanged.

The first borrowed-view draft retained a per-leg optional crossing branch and
four multiplications by one in the identity case. A current-host paired check
showed that draft about 12--13% slower than exact main at batch 128 in both
required workloads. Commit `e237d5c` specializes identity access; the same
check then measured `0.474549 ms/point` for all-flow-union fixed helicity
(`-15.81%` versus paired exact main) and `0.077799 ms/point` for non-union
selected-flow/helicity-sum (`-19.38%`). The regressing draft is not used as
performance evidence.

A five-block, same-helper smoke against the clock-free milestone measured:

| LC workload | Batch | Clock-free ms/point | Borrowed/into ms/point | Change |
|---|---:|---:|---:|---:|
| all-flow-union, fixed helicity | 1 | 2.730917 | 1.164927 | -57.34% |
| all-flow-union, fixed helicity | 128 | 0.537119 | 0.474549 | -11.65% |
| all-flow-union, fixed helicity | 1024 | 0.529601 | 0.495361 | -6.47% |
| topology-replay, selected flow, helicity sum | 1 | 0.274444 | 0.259004 | -5.63% |
| topology-replay, selected flow, helicity sum | 128 | 0.085786 | 0.077799 | -9.31% |
| topology-replay, selected flow, helicity sum | 1024 | 0.086619 | 0.084251 | -2.73% |

Every warmed value is bitwise identical to the preceding clock-free build.
Relative to the accounting build, the cumulative batch-128/1024 changes are
`-14.03% / -12.76%` for union and `-24.72% / -11.02%` for non-union. The
batch-1 union number is deliberately provisional because its five-block
standard deviation was `0.143603 ms/point`; the final long regression gate,
not this smoke, decides acceptance.

Explicit profile APIs still use the owned diagnostic input-preparation lane.
Their native input pack/cross clocks and container counters therefore describe
the paired profile pass, not ordinary borrowed evaluation; the independently
timed unprofiled headline remains authoritative.

Evidence:

- Measurements: `/private/tmp/pyamplicol-compiled-dag-flat-identity`.
- Union batch 1/128/1024 SHA-256:
  `2a4e2e01ef1d73243d5a2956bed4566771619730457def857a04feeb64bd8699` /
  `a8599471496984b94d9a3b8fb0e1afbe960004b66dcb0d2455ea94dcce564dc6` /
  `2bf145392a7fbfcb166ecd1256ef28e37334a9f0ff4863db280ecad0dd70b562`.
- Non-union batch 1/128/1024 SHA-256:
  `8609c38f88744173d04ebfcab0532fd72813df0dd3e90d17feccc5cd89f7fb9a` /
  `325e609745220b890925a64f78179b0af80a47a9f66035497f058b6cc3452179` /
  `dfba535db6dbb67629f9485c266b081c498578e4996de3effefc8cc744339d66`.
- Native build-input SHA-256:
  `32a045bb291a396838f10ec42eb3aee94384d85d2aaa452fd98a28c2da7ee6de`;
  module SHA-256:
  `bd87c30f1c613dc84433652e06a946d43dfafc359439271e27a2f0901171117e`.
  The measurement wheel was built from the exact `e237d5c` file content before
  that one-file corrective commit was created, so its informational build
  revision still records parent `b707da9`; final evidence will be rebuilt from
  the tested commit.
- Focused candidate rebuild watchdog peak: `1.288 GiB`. Formatting and affected
  crate checks pass; broader correctness and release gates remain deferred to
  final integration as requested.

## Milestone 4: totals-only compiled selectors

Status: implemented and measured at commit `bf3a704`; final robust median/MAD
validation remains part of the cumulative optimization gate.

Ordinary compiled total evaluation no longer requires a batch-wide resolved
tensor for the two target selector shapes. All-flow-union fixed-helicity
execution accumulates the existing coherent-group contributions into one
persistent manifest-color row, then performs the exact `+0.0` left fold in
manifest color order. Topology-replay selected-flow/helicity-sum execution
uses the same resolved group order in a persistent source row, applies one
mapping's routes in their established order to a persistent target row, then
folds the target helicity/color cells in manifest order.

The direct topology route is deliberately guarded to one source group, one
mapping, one replay entry, one diagonal-LC materialized selector lane.
Multi-mapping, recursive, contracted-color and multi-helicity shapes retain
the established resolved fallback. Resolved and profile APIs are unchanged.

Global and homogeneous selectors write directly into caller output.
Contiguous per-point partitions write their caller slice; gathered partitions
reuse the existing partition-total scratch and then scatter. The eager lane
retains resolved-plus-total fallback. Evaluation and reduction were separated
so resolved evaluation no longer computes and discards an ordinary total
first.

Retained-artifact checks compare ordinary totals with an explicit resolved
`+0.0` left fold. They are bitwise identical for the union primary nonzero
helicity (manifest index 234), a structural-zero helicity (including exact
positive-zero bits), and non-union identity and noncomputed/permuted flows
(indices 0 and 1). Homogeneous, pooled and alternating per-point selectors
agree with global-selector oracles at the required `rtol=1e-12`,
`atol=1e-15`. Different SIMD partition widths can differ by a few ULP, so
cross-width per-point results are not incorrectly claimed bitwise identical.

A five-block, same-helper smoke against milestone 3 measured:

| LC workload | Batch | Borrowed/into ms/point | Totals-only ms/point | Change |
|---|---:|---:|---:|---:|
| all-flow-union, fixed helicity | 1 | 1.164927 | 1.114137 | -4.36% |
| all-flow-union, fixed helicity | 128 | 0.474549 | 0.462946 | -2.45% |
| all-flow-union, fixed helicity | 1024 | 0.495361 | 0.483370 | -2.42% |
| topology-replay, selected flow, helicity sum | 1 | 0.259004 | 0.226892 | -12.40% |
| topology-replay, selected flow, helicity sum | 128 | 0.077799 | 0.075475 | -2.99% |
| topology-replay, selected flow, helicity sum | 1024 | 0.084251 | 0.079319 | -5.85% |

Every warmed result remains bitwise identical to milestone 3. Relative to the
accounting build, cumulative batch-128/1024 changes are
`-16.13% / -14.87%` for union and `-26.97% / -16.23%` for non-union.

Evidence:

- Measurements: `/private/tmp/pyamplicol-compiled-dag-totals-only`.
- Union batch 1/128/1024 SHA-256:
  `8950ab1651d8d3b43bfe9421a63c72e14c721303e06eaed3825fb0dae04569f8` /
  `775e3b00c74b3c045443c05cf0678ff56af779ad5da3885141f10d9ceb4a5e68` /
  `001efe54f585681ee895643855c1febf9b9527fc9f9ba5b5f6802436fba846de`.
- Non-union batch 1/128/1024 SHA-256:
  `e5b7e6f15c406af5251cadcb250a4fd54aca0861042e1805777fb92dfd82e30d` /
  `fe223ff8b7e06174504771552bdbad2a5ccc563fa502f8d80acb8ee57c5d2070` /
  `90bf8f0109a4c098f0fd575c1da145c91136772c510176b6c9d537867ecadf69`.
- Native build-input SHA-256:
  `8cb4af6ad4d2b06f76b80d4a6b600827131384a13bd932119f2bd2c40e9e2eff`;
  module SHA-256:
  `a928482a8e513306c18f6f6ec3a13412cee8ce5ca2c20844e70b3a1f2565728c`.
  As in milestone 3, the measured source content exactly matches `bf3a704`,
  while the informational wheel revision records its parent because the wheel
  was built immediately before the source commit. Final evidence will rebuild
  the immutable tested head.
- Focused candidate rebuild watchdog peak: `1.252 GiB`; largest measurement
  peak: `2.655 GiB`. `cargo fmt --all -- --check` and
  `cargo check -p rusticol-core --tests` pass. An independent read-only diff
  audit reported no actionable findings.

## Milestone 5: precomposed leaf mappings

Status: measured and rejected; no implementation code retained.

A runtime-only experiment composed each stage's parent-local and leaf-local
maps into ordered global-state-to-leaf layouts. It preserved duplicates and
ordering, borrowed exact identity layouts, kept the legacy snapshot path for
overlapping stage input/output, and left generic/exact execution and artifact
serialization unchanged. The expected copy reduction was achieved. At union
batch 128, stage parent copies fell from `6,075,648` to `19,456` and amplitude
parent copies from `369,152` to zero; leaf copies remained exactly once.
Warmed scratch reallocations remained zero and all numerical results were
bitwise identical to milestone 4.

Wall time nevertheless regressed consistently at useful batch sizes:

| LC workload | Batch | Milestone 4 ms/point | Precomposed ms/point | Change |
|---|---:|---:|---:|---:|
| all-flow-union, fixed helicity | 1 | 1.114137 | 1.979075 | +77.63% |
| all-flow-union, fixed helicity | 128 | 0.462946 | 0.487780 | +5.36% |
| all-flow-union, fixed helicity | 1024 | 0.483370 | 0.521280 | +7.84% |
| topology-replay, selected flow, helicity sum | 1 | 0.226892 | 0.230991 | +1.81% |
| topology-replay, selected flow, helicity sum | 128 | 0.075475 | 0.076819 | +1.78% |
| topology-replay, selected flow, helicity sum | 1024 | 0.079319 | 0.082341 | +3.81% |

The direct gathers reduce copied elements but lose the locality supplied by
the compact parent-stage slab; several later stage gathers and backend calls
become slower. Native wall time is the retention gate, so the experiment was
discarded in full rather than committing a counter-only improvement.

Evidence:

- Measurements: `/private/tmp/pyamplicol-compiled-dag-direct-leaf`.
- Union batch 1/128/1024 SHA-256:
  `b71d06a503a0809b8c42d1967bb5141f48fd963604cbf67f4dccc94a96438637` /
  `8edbe0e1678fc2229a1f6b4d9435c735173ccb76a3018f0b79f26af28fe3b013` /
  `9c276c5cedc8a99e13409c50535b2ff48d34a0ffa3bea2f6a0246579890686ea`.
- Non-union batch 1/128/1024 SHA-256:
  `0361cf982d1e9574f9bf281e9c322ef4039c72993037ed73cbe9a3b664ed9aca` /
  `9b30f4599d659156a6fcccbfb2100e95f225d51bd71274013cb762cd90109e7c` /
  `12462f9b291379701746c82fc25e9fa8704882e3fcaab9c9b5e3d992dfe30d78`.
- Native build-input SHA-256:
  `c8a7769c2e13d5d0c2a72f4e769acde6ab64b661f4a381900cdde652fa06a2ab`;
  module SHA-256:
  `f2a328a1290a257238fbc655cb42b44fcbf19305b4424f36bca5e5622d4070e8`.
- Focused candidate rebuild watchdog peak: `1.243 GiB`; largest measurement
  peak: `2.263 GiB`. Before the wall gate, formatting, affected-crate
  `cargo check`, and two independent read-only audits found no correctness
  issue. The measured implementation existed only as uncommitted work atop
  `b232ba1` and was restored after rejection.

## Milestone 6: persistent authoritative stage-output slabs

Status: deferred; no implementation code created.

The post-milestone-5 runtime topology removes the premise that this remains a
low-risk experiment. Each fused evaluator leaf currently writes a contiguous
row-major scratch matrix and then copies into one compact global-state matrix.
Keeping those outputs authoritative requires one retained matrix per leaf:
multiple leaf matrices cannot write directly into one stage matrix without
the explicitly out-of-scope strided/plane-aware backend ABI. Every downstream
stage and amplitude input would then gather from segmented
`(base state | producer leaf slab | structural zero)` sources.

Static mapping of the retained artifacts and milestone-4 profiles gives a
clear upper bound and locality risk:

| LC workload | Assigned components/point | Assignment ms/point (b128 / b1024) | Share of wall | Global-state source runs | Leaf-slab source runs |
|---|---:|---:|---:|---:|---:|
| all-flow-union, fixed helicity | 22,946 over 47 leaf slabs | 0.03788 / 0.04874 | 8.3% / 10.0% | 6,979 | 11,612 (+66%) |
| topology-replay, selected flow, helicity sum | 196 over 19 leaf slabs | 0.00341 / 0.00436 | 4.5% / 5.4% | 63 | 141 (+124%) |

The union amplitude input is particularly unfavorable: two contiguous
global-state runs become 1,441 producer/zero runs. Keeping the global arena
for safe fallback would add about 44.8 MiB of slabs at batch 128 and
358.5 MiB at batch 1024. Removing it instead would expand the change into
source/momentum/model storage, structural-zero ownership, selector liveness,
and exact/resolved compatibility.

Milestone 5 already supplied the relevant measured A/B lesson: reducing
copies while weakening compact-parent locality regressed union by 5--8% and
non-union by 2--4%. Authoritative slabs have a smaller theoretical ceiling
and substantially worse source fragmentation. A prototype therefore does not
meet this project's low-risk threshold. Reconsider it only with a separate
segmented/strided backend ABI project, or if future measurements put stage
assignment materially above 10% of wall.

## Milestone 7: SymJIT 2.21.1 and repeated colour-block reduction

Status: validated, pushed, and fast-forwarded to `main` at
`64f60d01f8aa2d147449219515a8f950da03e826`.

The candidate dependency is pinned to exact upstream SymJIT commit
`48197f32536c894b51ef25b2cf05ddd05c22675f` (release `2.21.1`, archive
SHA-256 `876930348cc06761ca780570fb282d009f143f4a469e321e3b5039c5ee217424`).
The NLC/full-colour totals runtime now recognizes disconnected,
bit-identical colour matrices repeated over helicity components. It gathers
coherent amplitudes once into local-colour-major/component-minor order and
contracts the canonical matrix across the contiguous component dimension.
Real weights use a four-accumulator Hermitian product kernel; complex and
non-repeated plans retain generic fallbacks. The implementation is ordinary
portable Rust and contains no Apple- or Arm-specific intrinsics.

Interleaved exact-v2.21.1 batch-128 measurements against frozen parent
`f41cb7e` were:

| Process and accuracy | Baseline ms/point | `64f60d0` ms/point | Wall change | Reduction change |
|---|---:|---:|---:|---:|
| `d d~ > t t~ + 3g`, NLC | 0.558152 | 0.539107 | -3.41% | -56.18% |
| `d d~ > t t~ + 3g`, full | 0.594886 | 0.561501 | -5.61% | -56.26% |
| `g g > t t~ + 3g`, NLC | 2.242961 | 2.019784 | -9.95% | -60.66% |
| `g g > t t~ + 3g`, full | 3.007659 | 2.215970 | -26.32% | -60.60% |

Maximum relative numerical disagreement was `3.18e-14`, below the required
`rtol=1e-12`, `atol=1e-15`. Focused and full Python/Rust checks passed except
for the pre-existing stale assertion
`compiled_color_topology_lane_requires_physics_reduction`, whose expected
diagnostic is already documented by the exact-main ledger.

## Milestone 8: compact repeated colour-contraction ABI

Status: validated, pushed, and fast-forwarded to `main` at
`54eb9ddff092f4181ba0d7018ccbfdcd5301234c`.

Generated NLC/full-colour plans now encode one canonical upper-triangular
matrix plus a local-colour-major/component-minor coherent-group map when at
least two helicity components have identical sector identities, binary64
helicity weights, and coefficients. Expanded entries remain the fallback for
non-repeated plans. Rust constructs the hot repeated block directly instead
of parsing and retaining every helicity copy. Exact Python and every f64,
resolved, materialized-helicity, generic-precision, eager-v2, and eager-v3
consumer iterate the same logical entries. Malformed mixed storage, bad
shapes/maps/indices, missing storage, and invalid compact weight widths fail
closed.

High-colour artifact results:

| Artifact | Total bytes, expanded | Total bytes, compact | Execution JSON change | Warm load change |
|---|---:|---:|---:|---:|
| `d d~ > t t~ + 3g`, NLC | 100,226,563 | 97,701,552 | -3.07% | -0.30% |
| `d d~ > t t~ + 3g`, full | 104,443,953 | 97,964,422 | -7.48% | -6.17% |
| `g g > t t~ + 3g`, NLC | 166,088,606 | 146,980,892 | -17.67% | -9.57% |
| `g g > t t~ + 3g`, full | 242,269,360 | 148,729,429 | -50.75% | -36.60% |

The largest sum lane replaces `929,280` expanded full-colour entries with
`7,260` template entries and a `15,360`-group map. The complete execution
manifest falls from `184,313,350` to `90,771,938` bytes. Seven-load alternating
medians are `3.258793 s` expanded versus `2.066087 s` compact. Fresh generation
remains on par: `22.93--24.10 s` for the two `d d~` artifacts and
`61.98--62.39 s` for the two `g g` artifacts, with watchdog peaks
`0.738--1.126 GiB` (below the exact-baseline peaks up to `1.784 GiB`).
All four generated artifacts load and evaluate successfully; focused evidence
currently comprises 53 Python tests, three compact Rust malformed/parity tests,
the 259-test Rust suite up to its known stale diagnostic assertion, Cargo
all-target checks, Ruff, rustfmt, and artifact smokes.

The final compact-artifact runtime gate preserved the milestone-7 kernel gain:
`g g > t t~ + 3g` full colour improved by `21.72%` at batch 128 and
`20.68%` at batch 1024 relative to the expanded exact-parent runtime; NLC
improved by `7.37%` and `12.17%`. The corresponding batch-128 changes for
`d d~ > t t~ + 3g` were `16.43%` full and `4.42%` NLC. Maximum absolute
disagreement was `1.55e-23` and maximum relative disagreement was below
`3e-14`.

## Milestone 9: direct portable SymJIT AoSoA execution

Status: validated and integrated into `main` at
`44242aec31cc8c26da0455cc7baee89f156bc2f4`.

The compiled-DAG runtime now packs fused stage leaves directly into SymJIT's
native complex SIMD block layout and invokes the already-loaded SIMD machine
code with transposition disabled. Results are scattered directly from block
AoSoA into the authoritative state or amplitude output. This removes
SymJIT's internal row-major transpose buffers without changing an artifact,
generation, public, or native ABI. Both complete-stage and selector-pruned
execution use the path, including their profiled variants.

The implementation discovers the generated lane count dynamically and has
portable specialized packers for widths 2, 4, and 8 plus a generic fallback.
It contains no architecture intrinsics or Apple-specific assumptions.
Incomplete blocks repeat the final valid input row and scatter only valid
lanes. Applications requesting threaded SymJIT matrix execution retain the
existing threaded path. Non-SymJIT, scalar batch, exact/generic-precision, and
SIMD-rejection cases retain the established fallback. A static
input-plus-output footprint gate keeps tiny amplitude leaves such as LC
`4 -> 1` kernels on that fallback; substantial colour kernels such as
`484 -> 120` use direct AoSoA. This is a fixed shape rule, not runtime
autotuning.

Seven-sample native unprofiled medians against the exact `54eb9dd` runtime,
using shared artifacts and byte-identical batches, were:

| Workload | Batch | `54eb9dd` ms/point | Direct AoSoA ms/point | Change |
|---|---:|---:|---:|---:|
| Z+6g all-flow union, fixed helicity | 128 | 0.497402 | 0.422662 | -15.03% |
| Z+6g all-flow union, fixed helicity | 1024 | 0.514180 | 0.453264 | -11.85% |
| Z+6g selected flow, helicity sum | 128 | 0.084113 | 0.081425 | -3.20% |
| Z+6g selected flow, helicity sum | 1024 | 0.084985 | 0.082595 | -2.81% |
| `g g > t t~ + 3g`, full colour | 128 | 2.338664 | 1.943635 | -16.89% |
| `g g > t t~ + 3g`, full colour | 1024 | 2.507651 | 2.138550 | -14.72% |
| `g g > t t~ + 3g`, NLC | 128 | 2.027607 | 1.641926 | -19.02% |
| `g g > t t~ + 3g`, NLC | 1024 | 2.182931 | 1.825885 | -16.36% |

Every pair has the same batch digest and bitwise-identical warmed f64 output.
Odd-tail batch 129 checks passed for union, selected-flow, full-colour,
selected-helicity full-colour, and the two-amplitude-chunk
`d d~ > t t~ + 3g` runtime. The selected-helicity colour lane improved
`7.07%`; the two-chunk `d d~` total improved `19.95%`. Two interleaved
batch-1 repetitions found no scalar regression. The largest measured peak RSS
was `3.762 GiB` at batch 1024.

The direct amplitude path reduces the `g g` amplitude envelope by about
`27 us/point`: leaf packing fell from `23.91` to `18.96 us/point` and backend
execution from `43.00` to `19.23 us/point`. Stage packing and backend gains
remain the dominant wall reduction. Padded movement counters report physical
rather than logical lane copies, mapping-scratch reallocations are visible,
and warmed retained-artifact profiles report zero reallocations.

Validation evidence:

- Native measurements:
  `/private/tmp/pyamplicol-dag-aosoa-final-paired`,
  `/private/tmp/pyamplicol-dag-aosoa-amplitude-tail-gates`, and
  `/private/tmp/pyamplicol-dag-aosoa-b1-gates-repeat`.
- Baseline native-input/module SHA-256:
  `689e09ea0c89f2d4593af0bb8226dc9042ed1b9907aac3a29385993cd5584971` /
  `912d5f6d0498a69488ede4da0f8d7732d8db344be7d91671019efca4640318bc`.
- Candidate native-input/module SHA-256:
  `1baa36e12a89a81c38557b40af78264e43abe4cde0996bc6297f4dc4cf071fd4` /
  `3ebb7fcb690e975e34da1ee9a6bd296f94dd93a3c17b46853c40d589e8aeb8dd`.
- Candidate wheel SHA-256:
  `fc6eca384f6dc4fca3e5f85d0edf83ee0753106d4aa0f5050dafc68254ca4a54`;
  build peak RSS `1.017 GiB`.
- `cargo check -p rusticol-core --tests`, four direct AoSoA unit tests, and
  the serial full Rust suite pass: 263 library plus 26 integration tests.
  The stale compact-colour fixture assertion was repaired by stripping
  nested auxiliary lanes from the deliberately invalid test lane.
- The installed-wheel LC integration file passed 15 tests overall, including
  a clean 14-test compiled-focused subset. Two built-in eager cases remain
  blocked by the documented stale packaged prepared-model candidate
  fingerprint; policy and metadata checks were not weakened.

## Milestone 10: static output-chunk coarsening

Status: measured and rejected; no default or runtime policy change retained.

A generation-only experiment raised the existing fixed SymJIT output chunk
from 512 to 1024. For `g g > t t~ + 3g` NLC this cut packed members from 456
to 238 and payload bytes from `146,975,129` to `145,620,586`. A controlled
same-process, alternating batch-128 A/B gave `1.752226 -> 1.704814 ms/point`
(`-3.05%`) with identical values. The JIT phase, however, rose from
`46.082 s` to `57.067 s` (`+23.84%`).

More importantly, the same policy regressed the latency-sensitive
topology-replay Z+6g selected-flow/helicity-sum lane:

| Batch | Chunk 512 ms/point | Chunk 1024 ms/point | Paired change |
|---:|---:|---:|---:|
| 128 | 0.082496 | 0.087987 | +7.16% |
| 1024 | 0.081863 | 0.085225 | +4.28% |

Generation there remained essentially unchanged (`4.021 -> 4.076 s` JIT),
so the regression is generated-code locality rather than a resource trade.
The fixed larger-chunk policy and a stage-wide single-applet direction are
therefore rejected. Evidence is under
`/private/tmp/pyamplicol-dag-chunk1024-44242ae`.

## Milestone 11: wider repeated-colour Hermitian accumulation

Status: validated for a compact runtime milestone.

The real-weight repeated-colour kernel now carries eight independent
Hermitian-product dependency chains for wide repeated-helicity blocks, with
the prior four-chain shape retained for the remainder. This changes neither
artifact storage nor generation, uses ordinary portable scalar Rust, and
gives LLVM more independent work on both Arm and x86.

Five alternating outer samples against a frozen exact-`44242ae` wheel at
batch 128 gave:

| `g g > t t~ + 3g` accuracy | `44242ae` ms/point | Wider kernel ms/point | Median paired change |
|---|---:|---:|---:|
| full | 1.938389 | 1.902407 | -1.20% |
| NLC | 1.638698 | 1.625563 | -0.67% |

The maximum relative numerical difference was `2.00e-16` for full colour and
zero for NLC. Focused real/complex repeated-block and compact-manifest Rust
tests pass together with rustfmt, affected-crate check, and `git diff --check`.
The final focused build peaked at `1.276 GiB`; its wheel/native-module SHA-256
values are
`43f594e062f154c4a9921dd0a2eac7e8c336f07dd996e5a4edf5e44575297bbc` /
`40a6f1a9b3ce90b796b5f2b7918726491f312610cc46d1bfcef3318c30f65561`.
A final seven-sample installed-binary canary retained the same
`2.00e-16` maximum relative difference.

## Milestone 12: persistent AoSoA state and composed full-stage gathers

Status: measured and rejected; neither prototype is retained.

The first prototype kept the full compiled state in a dense referenced-slot
AoSoA arena across stages. It was eligible only when every stage and amplitude
leaf used the same non-threaded SymJIT SIMD width; scalar, threaded, mixed,
profiled, and exact execution retained the row-major fallback. The plan
precomposed every global-to-leaf map, handled selectors and odd tails, and
preserved bitwise totals on the retained artifacts. It nevertheless enlarged
and fragmented the active working set while still gathering each leaf into
contiguous evaluator input. Native batch-128 measurements rejected it:

| Workload | `0dde005` ms/point | Persistent AoSoA ms/point | Change |
|---|---:|---:|---:|
| Z+6g selected flow, helicity sum | 0.082898 | 0.091534 | +10.42% |
| Z+6g all-flow union, fixed helicity | 0.688121 | 1.026229 | +49.14% |
| `g g > t t~ + 3g`, NLC canary | 1.779332 | 2.446874 | +37.52% |

The union window was noisy, but the direction and the two independent
regressions were decisive. Batch-128 union peak RSS reached `1.608 GiB`, and
the runtime could retain both the AoSoA and row-major arenas after crossing
between profiled and unprofiled paths. The build peaked at `1.302 GiB`.

The smaller follow-up reused the selector lane's already-composed
global-state-to-leaf mappings for ordinary full chunked stages, avoiding the
parent-stage slab without changing evaluator layout. This was allocation-free
and exact, but random global gathers lost the parent slab's locality. Union
improved only `0.421021 -> 0.418381 ms/point` (`-0.63%`, noisy), while
`g g > t t~ + 3g` NLC regressed
`1.662814 -> 1.790170 ms/point` (`+7.66%`). It too was removed.

These A/B results reject copy-count reduction by itself. A future state-layout
change must tile the live working set or change the evaluator ABI so that it
also improves locality; it should not add a workload-adaptive policy merely
to hide these regressions.

## Milestone 13: exact Klein-four Walsh full-colour contraction

Status: validated for a compact full-colour milestone.

One-quark-line full-colour permutation orbits now carry an optional,
generation-time `klein-four-walsh` plan when their complete scalar
permutation matrix is exactly invariant under a free
`C2 x C2` subgroup action. Recognition is structural and works for eligible
`S_m` orbits; it does not contain a process ID, an `S5` constant, an
architecture threshold, or runtime tuning. The generator retains the compact
logical repeated entries for exact, resolved, and unsupported fallbacks.
NLC deliberately remains on its sparse expanded reducer.

Rust validates the coset partition, finite real weights, unique matrix
entries, and exact subgroup invariance at load. It then derives four real
symmetric blocks once. The f64 totals-only singleton path fuses the existing
coherent-output gather with normalized four-point Walsh transforms and
contracts contiguous helicity components. For
`g g > t t~ + 3g`, the plan maps 120 flows into 30 cosets and reduces the hot
upper-triangular work from 7,260 to 1,812 Hermitian dots.

A controlled A/B used one generated evaluator payload and one candidate
runtime. The control artifact differed only by removing the optional
factorized plan and updating its canonical payload and artifact digests:

| Batch | Expanded median +/- MAD ms/point | Walsh median +/- MAD ms/point | Change |
|---:|---:|---:|---:|
| 128 | 2.028861 +/- 0.038119 | 1.715008 +/- 0.018649 | -15.47% |
| 1024 | 1.955683 +/- 0.001316 | 1.700167 +/- 0.016882 | -13.07% |

The maximum observed total disagreement was `4.14e-25` absolute and
`8.00e-16` relative. A fresh NLC fallback canary was bitwise exact and moved
`1.564733 -> 1.552362 ms/point`; the selected-flow and union LC fallbacks
were also bitwise exact with no regression. The generated execution JSON grew
by only 488 bytes; total artifact bytes were effectively unchanged
(`148,729,429 -> 148,724,640`). Comparable generation phase sums were
`57.207 -> 55.261 s`, and the candidate generation watchdog peak was
`1.159 GiB`.

The standalone proof, including a conservative no-vectorization build, is
under `/private/tmp/pyamplicol-s5-walsh-proof`. Focused evidence comprises
13 Python contraction tests, two malformed/parity Walsh Rust tests, four
repeated-colour Rust tests, Ruff, rustfmt, affected-crate checks, a guarded
candidate build (`1.286 GiB` peak), and native artifact evaluations.

## Milestone 14: selected-flow backend gap audit

Status: accounting complete; no code change retained.

The selected Z+6g closure is already equivalent to legacy AmpliCol after its
terminal-zero filter: pyAmpliCol has 8,338 interactions, 1,425 currents, and
384 roots, versus 8,324, 1,422, and 382. The 14-interaction difference is
`0.17%` and cannot explain a roughly 23 microsecond backend gap.

The actual 13 SymJIT applets instead contain `2,869,640` bytes of AArch64 SIMD
machine code, about 50.8 times the legacy library's `56,532` bytes of hot
text. Small applets sustain 15--17 GInstr/s, while the 300--730 KiB late-stage
applets fall to 5--6.5 GInstr/s. SymJIT already executes fewer arithmetic
instructions per point through two-lane NEON, so the remaining gap is
instruction-fetch/front-end locality rather than absent SIMD or missing DAG
recycling.

The credible generic direction is compact, table-driven repeated-Lorentz
microkernels or generalized MIR outlining while retaining vector-across-point
execution. A kernelized lowering has a measured-throughput ceiling of roughly
26--37 microseconds in the selected backend and a credible 15--25 microsecond
gain. Cross-chunk shared prologues could recover another 3--8 microseconds:
late stages duplicate 1,847 of 6,962 unique chunk inputs. Naively enlarging
applets remains rejected by milestone 10.
