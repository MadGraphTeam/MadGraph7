# Retained common-input rank-six diagnostic

The completed private diagnostic contains 30 synthetic tensor rows. All effective
inputs match the frozen dyadic fixtures exactly at the TIRust and COLLIER
boundaries. This is an append-only result of the earlier source proposal;
`complete: true` records protocol completion, not numerical qualification.

The six-point QP finite comparison fails against COLLIER for all three pole
selectors, with a maximum symmetric residual of 1. Six six-point DP requests
are refused by the Gram stability guard. Eighteen host IEEE observations and
all refusals and mismatches remain in the original results. No input, tolerance,
or production source was changed while packaging this evidence.

Nine accepted full/partial pole comparisons have residual zero. Nine independent
pole comparisons pass the frozen criterion, which permits poles negligible
relative to the finite coefficient; a symmetric residual of 1 can therefore
coexist with a passed negligible-pole comparison. This is not a claim that all
independent pole coefficients are exactly equal. The two-point rank-six UV
witness contains all 210 nonzero coefficients. Six-point finite nonzero coverage
also spans 210 coefficients, but the finite accuracy failures remain open.

`data/` retains every original run file. Large text logs use deterministic gzip;
`copies.json` records both stored and decompressed hashes and sizes. JSON is
retained byte-for-byte. `frozen/repo/` includes the unchanged analyzer, its source
dependencies, fixture, source index, and prior build text evidence. Absolute paths
in these historical records are provenance only. No executable or library is
included or loaded by replay.

From the repository root, verify hashes and reparse with the frozen analyzer:

```sh
python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/runtime1/replay.py
```

The directory may also be copied elsewhere and its `replay.py` run directly;
only Python's standard library is needed. Replay uses temporary plain-text files
and compares the complete parsed result to the original `data/results.json`.
It does not compile, load numerical libraries, or launch a native worker.

`summary.json` is a compact navigation aid; the raw output and frozen parser are
authoritative. This package does not qualify physical amplitudes, the complex-mass
stress campaign, performance, or the cause of the finite discrepancy. Precise
timing remains a separate local TODO for Hua-Sheng.
