# Independent stability-policy and reporting audit

The benchmark freezes `MLStabThres=1e-6` for both the starting revision and
candidate, every explicit reducer (1, 6, 7, 8, 9), and both materialized rescue
chains (`8|6|7|1` and `9|8|6|7|1`). The production default remains `1e-3`.
This is a common validation policy selected before acceptance measurements,
not a candidate-only advantage. The complete `cases.json` hash binds each
acceptance run; earlier pilots using the original threshold remain separate
and cannot supply acceptance timings. The policy record is
`../../inputs/stability-policy.json`.

## Evidence motivating the threshold

I independently inspected the retained native adapter/estimator evidence and
adjudication in `../tirust-bridge/stability-audit/`. The synthetic massive
rank-five box has a same-input DP/QP finite discrepancy of
`2.665154525962104e-4` under the symmetric complex comparison. The actual
MadLoop estimator returns approximately `3.79833e-5` and `2.17083e-4` for
separate real and imaginary projections. Both estimates are below the
historical `1e-3` threshold, even though the accepted midpoint finite answers
have symmetric errors `1.5251293305061354e-4` and `7.028800383773444e-4`.
Both answers fail the independent `3e-6` regression gate. Two rotations do
not resolve these correlated errors. Explicit QP agrees with the retained
reference to `4.1672708817260766e-16` over the complete lane.

The installed release replay reproduces these same numerical values. I
verified that `release-adjudication.json` binds the retained release input and
the installed `libmadloop_tirust.so` by their SHA-256 hashes. This confirms that
the observed issue is present in the installed release build as well as the
initial debug audit; it remains numerical evidence, not a timing result.

The common `1e-6` threshold therefore requests further evaluation for both
demonstrated DP projections. This evidence is stronger than selecting a
threshold from the bridge's success flag: it executes the production Fortran
adapter, check-pass cache bypass, and native accuracy function. However, the
original `adjudication.json` executes the estimator, not automatic dispatch.
An additional installed-release helper now executes the extracted production
rescue control block with the same native integral callback. Its narrower
automatic-dispatch result is confirmed below.

The original evidence was recorded before the benchmark policy amendment.
Its statements that the tighter threshold was diagnostic and the benchmark
policy unchanged describe that earlier observation. They are preserved with
their original source hashes rather than rewritten as later acceptance data.

## Executed automatic-dispatch check

I independently checked all eight rows of `host-dispatch-release.json`, all
twelve recorded build-source/library hashes, the runner/input hashes, and the
exact extracted production control block. The evidence file SHA-256 is
`e77c9380f6af78f0da125e931f7ddd538dfb06912af2fffc1351600b51a41c01`.
Generated availability and QP-index initialization are preserved.

For both projections, `1e-3` returns code 219 after DP modes `[1,2]`; with two
rotations the sequence is `[1,2,1,1]`. All four accepted finite results fail
the independent `3e-6` symmetric comparison. At `1e-6`, code 323 follows
`[1,2,4,5]`, or `[1,2,1,1,4,5,4,4]` with rotations. All callbacks succeed,
and each returned answer equals its same-input native QP reference in the
reported precision. Relative whole-projection errors against the fixture are
`2.169450454841158e-16` (real) and `1.1945816650004379e-15` (imaginary).

Thus this synthetic callback demonstrably enters genuine QP through the real
host rescue branch at the common benchmark threshold. It does not execute
the complete physical `SLOOPMATRIX` initialization, filtering, numerator
construction, and event pipeline. Its internal mode trace is separate from
the ordinary amplitude worker's final-result histogram.

## Limits and independent gates

- This is one synthetic tensor-contraction failure, with separate real and
  imaginary projections and no physical Born/R2/counterterm contributions.
  It is not a complete physical matrix-element qualification.
- A smaller estimator threshold does not bound the true error or guarantee
  detection of every correlated error. The independent `3e-6` regression
  gate and ordinary-point `1e-8` target remain unchanged and separate.
- Physical comparisons, precision-rescue observations, and measured runtime
  under the newly frozen policy are still required. Passing a rescue test
  does not remove the retained default-policy failure.
- Native debug-library correctness evidence cannot establish release-library
  performance. An estimated zero accuracy after equal rounded QP answers
  must not be described as zero mathematical error.

## Precision and timing language

The persistent worker records the final accepted backend and precision for
every timed amplitude call, and retains failed evaluations. These histograms
do **not** enumerate internal reducer attempts, all tensor evaluations, or
every intermediate DP failure. A QP-accepted call establishes a QP outcome;
its complete internal attempt sequence requires separate instrumentation.

Timing uses one allowed CPU and must begin after this campaign's compilation
finishes. The machine is shared: unrelated host activity cannot be excluded.
Host-load records and alternating pairs help characterize those conditions;
their confidence intervals describe the sampled shared-host measurements,
not exclusive-machine throughput. Cold timing includes first-amplitude
initialization plus warming the saved point stream, and excludes process
startup and parameter-card loading.

No physical timing was run as part of this independent audit.
