# Exact propagator-order reversal

This separately authorized diagnostic replays only threshold point 2/group 5
with permutation `[4, 3, 2, 1]`. It permutes the exact decimal shift and mass
records and both axes of the invariant matrix. It changes no component value,
scale, tensor rank, unit numerator, precision or stability guard. In particular,
it does not translate the loop-momentum origin to make the new first shift zero.
The denominator product and its polynomial numerator are therefore unchanged.

One TIRust call and 35 NinjaRS unit calls use the same historical 256-bit bridges
as the forward replay. The exact executable and object are copied into the new
ignored `.tirust-build/ew-threshold-tensor-replay-reversal` directory; no new
compiler invocation is needed. The previous source package, raw logs, input
stream and native artifacts remain unchanged. The new source manifest binds
the complete frozen forward package and all new protocol sources.

`derivation.patch` records the limited changes to the prior launcher/analyzer:
new directory depths and private paths, plus removal of the build action.
The Fortran source remains byte-identical. Input extraction verifies the
original trace's identity and status, then applies the explicit permutation.
Exact echoes, zero cache use, native statuses, complete coefficient coverage,
zero process exit and actual historical DSO mapping are checked as before.

The native launcher retains the shared numerical lock, assigned CPU and
180-second timeout. It runs only after the coordinated priority jobs release
their slot. Native failures remain excluded from tensor comparisons.
`compare.py` compares each successful NinjaRS unit with the corresponding
forward value and retains both complex values and absolute/relative differences.
The relative metric is `abs(a-b)/max(abs(a),abs(b))`, with zero–zero set to zero.
Tensor coefficients close to zero require their absolute scales; this is not
an amplitude reconstruction or an accuracy/performance acceptance test.

Preparation and, after explicit slot release, execution from the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/prepare.py prepare
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/run.py run --cpu 9
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/analyze.py
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/compare.py
```

`compilation.json` explicitly records executable reuse and binds the original
compiler transcript; its inherited commands describe the earlier build.
All preparation/run steps refuse to overwrite existing evidence. Subsequent
results belong in `RESULTS.md`, keeping this protocol manifest unchanged.
