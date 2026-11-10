# Closed EW n4 numerical checkpoint

This checkpoint retains `e+ e- > mu+ mu- a a [virt=QED]`: baseline
CutTools and candidate backends1,6,7,8,9, each with100 saved physical points
and200 final steady answer rows. All six lanes pass3e-6. CutTools meets
the1e-8 target; the other four reducers retain target misses. No finite
truth or timing acceptance is inferred from reducer agreement.

Accepted DP/QP counts are93/7 for both CutTools lanes,100/0 for Ninja,
COLLIER and NinjaRS, and38/62 for TIRust. These are final accepted results,
not internal-attempt counts. FPE flags were not observed by this campaign.

All raw case files, the closed console/status, and static input/source/build
bindings are indexed. The mutable supervisor status is excluded. This closes
the last ordinary case, but this individual checkpoint covers only its own
six lanes. Stress, precision, performance and overall qualification remain
separate. Reparse retained raw streams from the repository root with:

```sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-checkpoints/ew-n4-1/verify.py
```

The verifier reads text and hashes only. It launches no numerical library.
