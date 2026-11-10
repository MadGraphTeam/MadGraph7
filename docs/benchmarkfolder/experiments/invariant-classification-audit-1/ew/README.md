# Closed EW group5 clamp arithmetic

This pure audit adds scale and branch diagnostics for the saved ordinary point1
and threshold point2, group5. It reads five hash-bound historical files and
copies no native traces or libraries. It changes no input, threshold or guard
and evaluates no integral or amplitude.

The reconstructed normal binary128 inputs use the retained 36-significant-digit
round-trip records. Rational arithmetic simulates nearest-even113-bit rounding
after each displayed real operation. All twelve offdiagonal invariants match
the captured supplied values exactly. The same branches fire at both points:

| Pair | Branch | Corresponding captured external momentum |
|---|---|---|
|1,2|Snap to loop mass1|p3 (top), with a retained origin defect|
|2,3|Snap to loop mass3|p4 (antitop), bit exact|
|3,4|Snap to zero|p5 (photon), bit exact|
|1,3;1,4;2,4|No snap|p3+p4; p3+p4+p5; p4+p5|

The external correspondences are numerical checks against the saved repaired
`MP_P` values, not reconstructed symbolic graph routing. Pair1 correspondences
have nonzero origin defects: maximum `2.07e-32 GeV` at threshold. Those defects
remain in the summary rather than being declared exact affine identities.

At threshold, pair3,4 has rounded113-bit `DIFFSQ=0`; the exact Minkowski square
of the fixed binary inputs is about `-2.96474e-47 GeV²`. Its original-energy
normalizer is `1.20131572013e5 GeV²`; its difference-vector Euclidean norm squared
is `2.40263083118e-9 GeV²`. The latter relative residual is `1.23396e-38`.
This is the captured physical photon shell, **not** the genuine forward
spacelike channel erased in the separate CMS fixtures.

The two massive snaps expose a different discrepancy. The loop mass squared
is formed in DP then promoted, whereas the repaired external top shell uses
the square of the promoted model mass in QP. Their difference is
`-8.8448359747417284e-13 GeV²`. The source snaps the corresponding supplied
invariant to the loop mass. The exact repaired top/antitop shell residuals
relative to the QP model square are around `1e-30 GeV²`; the DP input residuals
are around `1e-12 GeV²` at threshold. Thus the retained tensor receives mutually
different shift geometry and snapped invariants. This does not establish a
wrong integral or a qualified mass-transport correction; the current native
input contract and independent pinch/finite-oracle gaps remain.

The summary also retains original-energy, Euclidean difference-vector,
max-component and pair-centered-energy scales for every pair. A common shift
of both chord energies leaves `DIFFSQ` unchanged but changes the original
energy denominator. Difference-vector scales remove that particular origin
dependence, but are descriptive diagnostics, not proposed replacements or
certified on-shell tests. Neither these two fixtures nor their source-order
arithmetic simulation establishes a universal threshold policy, compiler
transformation/rounding behavior, hardware flags, true error or performance.

Recompute without a compiler or numerical library:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/invariant-classification-audit-1/ew/recompute.py --check
```

`--check` checks the retained summary; `--write` is the explicit artifact-authoring
mode and is not needed for verification.
