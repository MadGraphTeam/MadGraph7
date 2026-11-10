# Closed numerical checks for the entry/cache successor

The freshly exported QCD `u u~ > z [virt=QCD]` and EW
`u d~ > w+ [virt=QED]` processes pass both the `3e-6` regression criterion and
the `1e-8` target. Each uses 100 saved points, baseline CutTools and candidate
CutTools, Ninja, COLLIER, NinjaRS and TIRust: 12 lanes and 1,200 final point calls.
All accepted final results use the selected backend's DP mode. Internal retry
counters and software/hardware exception flags were not recorded. These are
untimed numerical checks. No precise performance gate is qualified.

Independent replay reproduces all 9,600 coefficient comparisons. The largest
symmetric differences against baseline CutTools are `2.69e-13` for QCD and
`1.95e-12` for EW. Candidate CutTools matches baseline coefficients exactly.
The two candidate exports are fresh full builds; baseline numerical objects
are preserved with worker-only recompilation/relinking. Frozen cards, points,
compiler policies, source identities and before/after inputs pass their checks.
Runtime evidence records resolved `ldd` libraries and their hashes; worker
process maps were not sampled. All five supervised stages exit zero and their
cleanup receipts have no surviving owned processes.

[The closed inventory](../n1-qualification-1/closure-index.json) binds the exact
218 payloads, source/protocol receipts and raw results. Its source anchor is
`0fcd2cd873`; the observed documentary revision `85c6211b14` has the same exporter
inventory. The unchanged native TIRust DSO remains `eb3d67fd…`.

Reproduce every payload hash and numerical comparison using only the retained
Python protocols and raw data, with no compiler, reducer or original installed
library:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/n1-closed-replay-1/verify.py
```

`--root PATH` accepts a relocated tree containing the indexed files at their
repository-relative paths. Original absolute paths inside receipts are never
opened. `verification.json` records an identical replay after copying only that
closed inventory into a temporary tree. This verifier replays numerical data;
it does not re-execute physical amplitudes or establish independent true error.

These two processes qualify **12 of 66 lanes** for this successor. Nine other
processes, the separate unresolved stress/CMS cases and precise performance
acceptance remain open. All 66 fresh local lanes still need qualification for
Hua-Sheng's local build; follow the
[local numerical/timing handoff](../timing-handoff-1/LOCAL_TIMING_TODO.md).
The existing 31-page canonical PDF remains historical for its preceding source.
