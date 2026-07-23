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
