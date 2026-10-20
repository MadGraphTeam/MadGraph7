# Representative collection history

These runs exercise fresh-process versus persistent-worker parity, using every
retained point and at least two persistent commands. Their timings are
exploratory. They do not replace final cross-backend correctness or performance
measurements.

The first candidate generation is the clean `fd72f8914` tree. Its completed
rows remain valid historical evidence, but the later compiler-compatibility
fix changes the generator identity and requires a refreshed candidate proof.
The original and corrected-original roles can be reused when their exact
runtime bindings remain unchanged. New original-only DP runs avoid mixing
superseded candidate targets into the final role-aware proof.

The strict forced-QP control uses all ten retained `gg → tt~` points at
`MLStabThres=1e-20`. Ninja returned status `420` in each independently attempted
source role:

- `representative-legacy-qp-20261020`: original Ninja; the combined attempt
  stopped before candidate collection.
- `representative-candidate-legacy-qp-20261020`: corrected candidate Ninja.
- `../reference-runs/representative-reference-ninja-forced-qp-20261020`:
  corrected-original Ninja.

Those failures remain in their original run-status and rejected-output records.
They are ineligible for representative proof. CutTools and NinjaRS completed
their separate ten-point forced-QP runs. Ninja representative rescue coverage
instead uses the complete 100-point core QCD3 stream, with the unchanged core
`1e-9` threshold, including every DP and QP outcome. This is a separate
representative process, not a filtered or relaxed version of the failed
forced-QP control.

The original core exports also have two genuine link configurations: QCD1–3
and EW1–3 lack IREGI; QCD4, EW4 and the controls include it. Both full runtime
signatures are retained. The additional `ct-qcd1-qp` runs use every one of the
100 original QCD1 points with a separately labelled, stricter representative
`1e-20` setting. They cover CutTools QP without IREGI. The
`ninja-near-singular` runs use all ten pre-existing near-singular `gg → tt~`
points at the unchanged `1e-20` control setting. They cover Ninja QP with IREGI.
Their explicit point-root alias and provenance are retained under
`../representative-point-streams/near-singular`; no core point or threshold
is changed, and none of the failed ordinary forced-control points is removed.

`representative-candidate-ninja-qcd3-20261020` was interrupted after its fresh
run while preparing the persistent calibration, at the root agent's request:
the generator was being superseded. Frozen worker cleanup ran; the failure and
explicit cancellation record remain. This attempt is not proof evidence.

Runtime refreshes live outside these run directories in
`../runtime-bindings-20261020`. They preserve the original source/run binding
and add exact retained library-interface and link metadata. A final proof must
explicitly bind only its eligible complete runs and chosen candidate identity.
