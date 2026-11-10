# Closed QCD n4 numerical checkpoint

This append-only checkpoint retains the completed `u u~ > z z g g [virt=QCD]`
case from campaign `split-dp-terminal-1`. It covers baseline CutTools and
candidate backends 1, 6, 7, 8 and 9, each with all 100 saved physical points
and 200 answer-order rows from its final steady warm-up traversal.

All six lanes pass the existing `3e-6` regression criterion. CutTools matches
the baseline at the `1e-8` ordinary target; Ninja, COLLIER, NinjaRS and TIRust
retain target misses. The exact residuals remain in each original result.
Final accepted DP/QP counts are 88/12 for baseline and candidate CutTools,
97/3 for Ninja, 100/0 for COLLIER, 96/4 for NinjaRS and 14/86 for TIRust.
These are final accepted results, not counts of internal reduction attempts.
FPE flags and internal retry counters were not observed.

The immutable inventory includes the complete closed raw case directory,
its closed console log, copied completed-case status and static input,
source, compiler, executable, runtime, model, card and point bindings.
The active EW n4 directory and mutable supervisor status are excluded.
The case began at 01:30:21 UTC and closed at 05:02:17 UTC on 10 November 2026;
this elapsed administrative interval is not a warmed runtime measurement.

No native evaluation, compilation, timing or performance acceptance is
performed by this checkpoint. All observation times remain null for timing
purposes. This one-case checkpoint does not certify the full numerical
matrix, any performance gate or completion of the overall goal.

From the repository root, verify all retained hashes and reproduce the
six raw numerical observations and comparisons:

```sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-checkpoints/qcd-n4-1/verify.py
```

Replay requires the retained protocol source hashes to match the current
pure parser/comparator helper files. It does not execute a reducer or read
an ignored generated build tree.
