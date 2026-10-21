These controls compare current generated Ninja 6 and NinjaRS 8 amplitudes for
`g g > h h h [sqrvirt=QCD]` and `g g > h h h h [sqrvirt=QCD]`. Each process uses
exactly two common fixed physical points at 1 TeV, with the same zero-width
parameter card, normal rescue, full Laurent coefficients, and split orders.
This is a small correctness control, not a timing or random-stability study.

The four fresh native invocations completed with exit zero. Both tools returned
double-precision success for both points in both processes, with zero Born
components. The numerical comparisons pass the common relative-plus-point-scale
tolerance. The independent review is in `independent-review-v1/receipt.json`.

The original Python controller then exited with an analysis-only import error:
its comparison call named the v3 policy module, while the existing pure
`compare_rows` function belongs to v2. The frozen controller, specification, raw
outputs and original parent exit status are preserved. The separate
`finalize_v1.py` corrects only that local reader reference, recomputes the complete
retained validation, and writes `run-v1/result-finalized-v1.json`. It makes no
native calls and changes no original file. The finalizer's real comparator API
is covered by the ten-test successor proof in `software-proof-v2`.

From the repository root in the documented Python environment, revalidate with:

```sh
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/higher-higgs-controls/finalize_v1.py validate
```

The retained reader uses the public source, card, point, raw and runtime metadata
proofs. Fresh numerical regeneration additionally requires the installed
libraries and compiler described by the sibling `higher-higgs-generation`
authority. Its actual exports were made at revision `235901b9` with the recorded
dirty source patch and source inventory; subsequent committed CLI sources must
not be substituted for that generation identity. Native invocations must not be
repeated as part of retained-data validation.
