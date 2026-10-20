# Explicit compensated DP retry experiment

This isolated prototype tests whether compensated T3 expansion can recover a
finite DP stability failure without imposing its cost on ordinary reductions.
It is not installed, is not production code, and is not benchmark acceptance
evidence. Its generated processes come from the retained `fd72` corrected-model
exports, not a newly generated current-source acceptance series.

The production `DpState` and `ninjars_evaluate_dp` implementation are unchanged.
An experimental `ninjars_evaluate_dp_compensated` entry point owns separate,
lazy context state and the previously tested guarded T3 numerator. It leaves
other callbacks, unsupported layouts, unsafe exponent domains, and unsupported
floating-point environments on the original callback. The original guarded
kernel and its numerical controls are retained in
[`../qcd3-point67-t3`](../qcd3-point67-t3/README.md). The original unconditional
archive is frozen; this experiment has a separate archive and module.

The cloned generated Fortran first performs the ordinary DP evaluation. Only
a finite stability failure from backend 8, without a hard reduction error or
FPE, enables one compensated DP retry. That retry performs the same directions
and configured rotations with the same threshold. Failed retries proceed to
the existing backend/QP chain. A public wrapper clears retry mode on entry and
on every ordinary return; mode also clears before advancing a backend or
entering genuine QP. For rotated checks, the retry rebuilds the numerator in
the original frame. With zero rotations, it reuses the same-momentum tensor
coefficients. `retry-stats.dat` records the native and retry accuracies and
counts separately.

The additive ABI symbol is explicitly experimental. This prototype retains
ABI version 2 for isolated linking only. Promotion would require a capability
or ABI change, installer/export changes, independent review, and fresh
numerical and performance validation.

## Checks completed

The three Rust tests pass without instantiating Symbolica:

- All six retained native reduction requests reproduce every Laurent bit.
  Ordinary calls leave compensated and MP state unallocated. Explicit
  compensated calls reproduce the frozen corrected results, followed by
  unchanged ordinary results on the same context.
- Changed coefficients, masses, and scales match fresh contexts; malformed
  inputs and output lengths preserve output buffers.
- Active rank-2/3/4 compensated tensors also match fresh contexts after changing
  nonzero masses, scales, and coefficients; repeated native/compensated
  interleaving and return to old parameters do not reuse stale state.

The mocked generated state-machine suite passes 24 scenarios: rotations 0/2,
backend chains `8` and `8|6`, compensated success, compensated failure, native
success, hard-status/FPE bypass, and successful points after each failure.
It verifies once-only retry, all expected direction/rotation phases, backend
transition, and mode cleanup. These are control-flow tests: the fake QP wrapper
does **not** establish genuine MP numerical rescue. The initial harness
assertion counted the initial coefficient construction as a label-200 rebuild;
the implementation initially enters label 208. That failed assertion and the
corrected test interpretation are retained.

The actual QCD3 100-point run passes every physical coefficient against the
native and unconditional-compensation references at the unchanged mixed
tolerance. The worst tolerance fractions are 0.717465 and 0.00004457,
respectively. Native failures at points 44, 67, and 80 each trigger one retry.
Point 67 improves from `1.74798e-9` to `2.16495e-10` at the unchanged `1e-9`
threshold and returns DP status 218. Points 44 and 80 remain unstable after the
retry and correctly enter genuine 256-bit rescue, returning 322. The full
result is 98 DP points and 2 QP points, with no leaked retry mode. See
[`qcd-3/runs/full-100/analysis.json`](qcd-3/runs/full-100/analysis.json).

The rotation-2 sequence `67,1,44,1,80,1,67` also passes all physical comparisons.
It returns four genuine-QP statuses 322 and three DP statuses 218. With these
additional rotations, point 67 remains unstable after compensation and is
correctly rescued. Every failing point retries once; each following ordinary
point starts with cleared state. The threshold remains `1e-9`.

The actual loop-induced ggH and mixed-order controls each pass all ten points
and complete split-order rows, returning 218 with zero retries. All retained
corrected-cohort backend comparisons pass (worst tolerance fractions 0.00904
and 0.00000908). Literal-original results are classified as diagnostics; they
also agree in these controls.

Seven alternating quiet pairs on CPU207 measure the added ordinary-path cost:
QCD1's retry/native ratio is 1.005424, with 95% paired-bootstrap interval
[1.000627, 1.008376]; EW2's ratio is 0.987828, interval
[0.972794, 0.998872]. All 100 points remain native DP with zero retries and
exact coefficients, accuracy, and status across variants and repetitions.
These ratios compare the retry machinery to the ordinary NinjaRS bridge.
They are **not** the final original-Ninja performance gate.
See [`overhead/quiet-seven-pairs/result.json`](overhead/quiet-seven-pairs/result.json).

## Reproduction

Run from the repository root after loading the pinned development environment:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/t3-dp-retry/reproduce.py prepare
python docs/benchmarkfolder/experiments/t3-dp-retry/reproduce.py build
python docs/benchmarkfolder/experiments/t3-dp-retry/process.py prepare --case qcd-3
python docs/benchmarkfolder/experiments/t3-dp-retry/process.py run --case qcd-3 --cpu 5 --label full-100
python docs/benchmarkfolder/experiments/t3-dp-retry/analyze.py --case qcd-3 --label full-100
```

Preparation refuses to overwrite existing experiments. The commands use the
shared task `SYMBOLICA_LOCK_DIR`; real MP runs must be sequential with other
task MP processes. Process runs verify their compiled sources, executable,
archive, and resolved parameter card before evaluation, remove stale output,
and retain cards, points, commands, output hashes, and return status.

`bridge.patch`, per-process patches, source snapshots, build metadata, and test
logs retain the exact modifications. `mock_flow.py` prepares and executes the
mocked state-machine suite after the `qcd-1` process is prepared.
