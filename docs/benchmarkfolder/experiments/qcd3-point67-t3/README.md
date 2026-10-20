# Compensated binary64 T3 callback experiment

This is an isolated numerical experiment, not an installed change or an accepted
performance result. The production bridge, precision, stability thresholds,
cut geometry, scalar masters and MP rescue are unchanged.

The frozen qcd3 point 67 diagnosis identified numerator T3 expansion rounding,
rather than scalar-master rounding or final integral summation, as the largest
source of its extra NinjaRS direction discrepancy. See the adjacent
[`qcd3-point67-masters`](../qcd3-point67-masters/README.md) and
[`qcd3-point67-compensated`](../qcd3-point67-compensated/README.md) experiments.
All six original tensor requests, master values and Float256 callback targets
are retained there; no sibling reference clone is required for replay.

## Kernel and guards

`t3.rs` first compares seven arithmetic variants on the frozen requests.
Mode 6 applies compensated transposed Horner evaluation: FMA supplies each
binary64 product residual, Neumaier summation retains local addition residuals,
and the first-order residual continues through the next polynomial level.
Residual products are themselves rounded. This is **not a general guarantee of
correctly rounded polynomial evaluation**.

`guarded.rs` implements that recurrence with scratch owned by the retained
tensor. Its supported domain is an ordinary `N(q)` tensor of rank 2..8 with
T3 `mindeg=2`. Every non-T3 callback, every other T3 layout, ranks outside this
range, and the rank-n+2 callbacks delegate to the pinned native tensor.
The public bridge has exactly `ncoeff(rank)` coefficients: explicit-mu input
blocks are rejected by the existing ABI dimension checks, not interpreted as
ordinary tensor coefficients.

Before any compensated arithmetic, integer-only inspection of input binary
representations bounds the exact common binary lattice and intermediate
magnitudes. Subnormal/nonfinite inputs and ranges outside conservative
`[-900,900]` exponent bounds go directly to the native callback.
`guarded_fenv.c` requires GNU/Linux x86-64, round-to-nearest, masked exceptions,
FTZ/DAZ off and ordinary hardware controls; other environments also fall back.
The observation does not alter flags or trap masks. No correction is attempted
and abandoned, so fallback cannot inherit flags from speculative arithmetic.
Successful compensated arithmetic can set the ordinary inexact flag.

The indexing and traversal follow pinned NinjaRS
`0d0d7e7a8033431b60a8580731084742e0655cda`,
`crates/ninjars/src/fast/{tensor.rs,tensor_layout.rs,tensor_jets.rs}`.
The comparison modes 3..5 adapt the original GPL Ninja
`tensor_ninja.cc` tensor construction/contraction order. The original-order
binary64 Rust result matches the actual original Ninja callback on all 480
captured scalar components. Actual original quadNinja, with the same binary64
inputs widened exactly, matches all 480 Float256-rounded targets too.

## Numerical evidence

`results.json` and `summary.json` retain every variant and all three complex
Laurent coefficients for loops 3/52/71 in both directions. Mode 6 matches all 480
captured Float256-rounded callback components; native fast tensor matches 63.
The whole reduction still uses binary64 cut geometry and reduction algebra:
improved callback accuracy does not remove all common reduction error.

Four added bridge controls passed in 0.30 s:

- The guarded scratch implementation reproduces all 480 frozen targets.
- Ranks 0..8, four dense/sparse/integer/adversarial patterns and T3 layouts 0..5
  provide 2448 scalar oracle comparisons at 256 bits. All 224 supported corrected
  components match their rounded oracle;116 improve over native and none worsen
  in this sample. Other layouts must remain bitwise native, and may retain large
  cancellation error. This sample establishes no universal rounding guarantee.
- Every non-T3 route remains native; extra explicit-mu ABI coefficients fail
  dimension validation.
- Nonfinite/extreme/subnormal inputs and directed rounding preserve native
  fallback values and floating-point flags. An additional test-only extension
  exercises finite inputs causing intermediate underflow/overflow; its separate
  result is recorded in `guarded-extremes.log`.

All 13 tests, including the nine existing production bridge tests, passed in
4.50s (`guarded-all-tests.log`). The inherited tests include actual complex
masters, 256-to-384-bit convergence, changed coefficients/masses/scale and fresh
versus retained contexts, actual binary128 transport, and transactional invalid
buffer handling. `guarded-tests-initial.json` binds those results to their
binary/source hashes and distinguishes the subsequent test-only extension.

The actual generated normal DP decision for point 67 is retained in
`guarded-normal-dp-decision.dat`:

| Quantity | Original bridge | Guarded experiment |
|---|---:|---:|
| DP stability estimate |1.74797848921705276e-9|2.16494704065548449e-10|
| Required threshold |1e-9|1e-9|
| DP orientations |2|2|
| Floating-point exception detected |false|false|

This driver stops immediately after the normal DP decision, before any possible
MP call. It does not produce a final stable/rescued return code. All 232 original
requests and paired original Ninja outputs are exactly unchanged; all 232
reductions succeed. The three complex Laurent components for every reduction
are retained. The final artifact was relinked and reproduces the first guarded
run exactly; the initial run is preserved under `guarded-initial-run`.

The direction-averaged finite coefficient is
`2.1582557399957516e-8`. Its finite-only relative difference from historical
CutTools QP is about 1.58e-7, which illustrates the remaining common bias.
The prescribed acceptance rule includes `1e-10*S`, with the Born value setting
`S=2.59179917621121855e-5`. All components pass that unchanged mixed rule against
the retained CutTools/Ninja/Collier coefficients; the largest tolerance fraction
is 0.819445. `guarded-process-comparison.json` retains the exact comparisons.
This is only one frozen point; current-source 100-point correctness and runtime
overhead remain separate requirements.

## Reproduction

Run from the repository root after sourcing `.ninjars-build/dev-env.sh`.
Builds use the pinned Cargo dependencies and a repository-local target.

```sh
python docs/benchmarkfolder/experiments/qcd3-point67-t3/prototype.py build
python docs/benchmarkfolder/experiments/qcd3-point67-t3/prototype.py run
python docs/benchmarkfolder/experiments/qcd3-point67-t3/original.py
python docs/benchmarkfolder/experiments/qcd3-point67-t3/original.py --quad
python docs/benchmarkfolder/experiments/qcd3-point67-t3/guarded.py prepare
python docs/benchmarkfolder/experiments/qcd3-point67-t3/guarded.py build
```

The original Ninja comparison commands use the retained baseline installation
under `.benchmark-work/original/HEPTools/ninja`. Callback and scalar replay do
not use that installation. The generated DP control copies the previously
prepared frozen point67 diagnostic output; it refuses to overwrite an existing
copy:

```sh
python docs/benchmarkfolder/experiments/qcd3-point67-t3/guarded.py process --cpu 4
```

Run bridge tests serially, sharing the task's Symbolica lock directory and
avoiding another process that holds a licensed runtime permit:

```sh
export CARGO_TARGET_DIR="$PWD/.benchmark-work/experiments/qcd3-point67-t3/guarded-target"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
cargo test --release --locked \
  --manifest-path .benchmark-work/experiments/qcd3-point67-t3/guarded-bridge/Cargo.toml \
  -- --test-threads=1 --nocapture
```

The process-tested, frozen archive SHA256 is
`d0f42d52da8b87f6debdf84c765a76c9833d33efee7e70ea56ef0e5f0b1d2973`.
Its build/source manifests are `guarded-build.json` and `guarded-source.json`.
No performance claim is made here. The root agent owns separate current-source
qcd1/EW2/qcd3 overhead and correctness measurements.
