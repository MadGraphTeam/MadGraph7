# EW threshold adjudication: current milestone

This status supersedes the chronological state described in the original
[README](README.md), which is intentionally preserved byte for byte. The
original README predates the completed tensor replay and private
higher-precision experiment below. All results here use the original saved
physical point; this milestone contains no production change.

1. **Historical amplitude trace.** [Trace summary](summary.json) and
   [raw tensor records](ew-tensor-trace.dat) establish that the ordinary
   control serves all 200 groups, while the threshold point serves 163 and
   refuses 37 with ill-conditioned-Gram errors. Production automatic status
   `420` correctly rejects the point. Positive fixed-QP status `120` bypasses
   that rejection and exposes an incomplete sum; its finite and pole values
   are invalid amplitude references.
2. **Geometry and CutTools isolation.** The [geometry audit](geometry/README.md)
   records the actual QP momentum repair, mass-shell conventions and scale.
   The [private CutTools QP-denominator experiment](ct-qp-transport/README.md)
   preserves the original scalar cutoff and does not resolve the threshold
   finite or simple pole. Its positive-mode results remain unchecked.
3. **Completed matched-input tensor replay.** The [256-bit replay results](tensor-replay/RESULTS.md)
   compare the refused box, an ordinary box and a successful massless box
   against NinjaRS unit numerators. The difficult box is refused by historical
   TIRust; all 35 NinjaRS units are served. Ordinary components agree closely.
   The [historical reversal](tensor-replay/reversal/RESULTS.md) also leaves
   TIRust refused. NinjaRS poles are unchanged by reversal; its maximum
   finite complex-relative change is `8.19e-9`.
4. **Independent analytic poles.** The [source-reviewed pole check](tensor-replay/analytic-poles/README.md)
   specializes the MadFKS formula to the exact model and untagged process.
   NinjaRS satisfies its Born-independent single/double-pole ratio to
   `1.68e-11`. Ninja's automatic result, accepted with status `328`, disagrees
   by `111.13` relative. That accepted inaccurate single pole remains an
   accuracy failure. The check does not certify the finite coefficient.
5. **Exact input and Gram audits.** The [input consistency audit](tensor-replay/input-consistency/README.md)
   and [exact Gram audit](tensor-replay/gram-consistency/README.md) distinguish
   supplied on-shell-aware invariants from the geometry of fixed Cartesian
   shifts. For the refused box, their determinants differ by `1.018469%`,
   while both normalized determinants remain approximately `5.2e-30`.
   These source-defined conditioning estimates are not error bounds.
6. **Completed private 512-bit experiment.** The [precision/order results](tensor-replay/higher-precision/analysis/README.md)
   use a separately built bridge with 512 working bits and guard `2^-392`,
   retaining 113-bit transport and the historical lightlike policy. The box
   now returns success, but 31 finite and 31 simple-pole tensor components
   disagree strongly with NinjaRS even though the scalar component agrees
   exactly. All 105 TIRust forward/reverse component-slot values are identical
   at recorded precision. Success and order stability do not qualify this
   private result. No full amplitude was evaluated with the private bridge.

The current production guard and its correct refusal remain unchanged. The
original threshold amplitude, the private relaxed-guard tensor result, and
the finite coefficient remain unqualified. Further source-level geometry
adjudication is maintained separately, outside this frozen milestone; it
does not retroactively change these observations.

The outer [artifact index](artifact-index.json) covers the complete milestone,
including failed setup/admission attempts and ignored raw logs. Verify its
portable file inventory from the repository root with:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/freeze.py
```

This verification uses only the committed files and Python's standard library;
it does not compile, load a reducer or claim a new numerical measurement.
