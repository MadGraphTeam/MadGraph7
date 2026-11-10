# Closed EW n3 numerical comparison

This offline analysis reparses the six completed streams for
`e+ e- > t t~ a [virt=QED]`: baseline CutTools and candidate backends
1, 6, 7, 8 and 9. Each stream contains 100 saved physical points and
200 answer-order rows. Its 82 retained input files are bound through the
immutable `numerical-checkpoints/completed9-1/index.json`; active QCD n4,
EW n4 and mutable supervisor status are outside this analysis.

All five explicit-backend comparisons with baseline CutTools pass the agreed
`3e-6` criterion and the `1e-8` ordinary-point target. All ten pairs among
the five candidate backends also pass both thresholds. There are no target
misses in Born, finite, single-pole or double-pole coefficients. The largest
pairwise symmetric residual is `1.81178562220549e-9`, in a finite coefficient
between NinjaRS and TIRust. The largest single-pole and double-pole symmetric
residuals are `1.2931831829498408e-10` and `1.052519950515903e-11`.

CutTools, Ninja, COLLIER and NinjaRS each finish all 100 points in DP.
TIRust finishes 82 in DP and 18 in QP. `analysis.json` retains every point's
accepted status, precision and stability estimate, together with component
maxima and the full original comparison summaries. These counts describe
final accepted results; FPE flags and internal retries were not observed.

Agreement is not an independent truth certificate. No analytic pole or
higher-precision adjudication is added, and no original comparison or gate
is changed. This analysis performs no new reducer evaluations and provides
no timing qualification; warm-up durations are not reduction timings.

From the repository root, replay the exact retained parser and comparator:

```sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/ew-n3-target-triage-1/analyze.py
```

The replay verifies the local package inventory, checkpoint and every
consumed retained input, then reproduces `analysis.json`.
No ignored build tree, installed library or network connection is needed.
