# Closed QCD 2→3 comparison plots

These two figures use the already closed current-source
`u u~ > z g g [virt=QCD]` result. They add no native evaluations or timings.
The portable analysis capsule is pinned at index
`79f57a8d83ea74249af83d12908de299372714cd7587d59d4ae31cfa9f63e457`;
its verifier is replayed before either input series is constructed.

- [Accepted composite stability estimate](figures/accepted-internal-estimate.pdf)
  shows the recorded MadLoop estimate **after rescue**, including finite and
  pole diagnostics. It is not the paper's raw-DP finite-only diagnostic.
  COLLIER may use a different estimator. Thirteen TIRust accepted QP estimates
  equal zero; these are diagnostic equalities, not certified digits. The
  selected capsule lacks the current effective numerical card bytes, so the
  actual host-policy vertical marker is omitted. The 10⁻³ line is explicitly
  a paper reference for a different diagnostic.
- [Finite coefficient disagreement](figures/finite-disagreement-current-cuttools.pdf)
  uses `abs(a-b)/abs(a+b)` relative to **current CutTools backend 1**. CutTools
  supplies a comparison reference, not independent truth. Its self-comparison
  has 100 exact zeros. All five current backends remain in the denominator.

Each figure contains 100 physical points. The total and order-1 answer rows
are verified exact duplicates and collapsed once. This is the original
ordinary campaign sample; no claim is made that it passes the appendix C cuts.
The empirical tail resolution is 0.01. Confidence about rarer population
tails is not inferred. Actual accepted TIRust outcomes are 87 DP and 13 QP;
the other four current lanes contain 100 accepted DP outcomes each. These
counts do not recover unrecorded internal attempts. The raw-DP pilot is a
separate future execution with new cut-qualified common points.

The two panels use the same target domain. Ratios are relative to the Ninja
tail and are masked when Ninja has no exceedance; valid zero ratios remain
visible. Exact-zero tail fractions cannot appear on logarithmic y axes.
Raw metrics and all zero counts are retained in JSON sidecars.

Reproduce with explicit closed dependencies and a new output directory:

```
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/retrospective-qcd-n3-3/adapter.py --execute --destination /tmp/madloop-stability-reproduction-1
```

Verify the delivered normalized data, counts, summaries and rendered file
hashes without rendering or evaluating amplitudes:

```
python3 -B docs/benchmarkfolder/stability_appendix_c/retrospective-qcd-n3-3/adapter.py --verify-only
```

`--capsule` and `--plotter` can name explicitly relocated copies of the pinned
portable analysis and plotter-source-4 packages. The adapter never follows
the original historical paths recorded inside that analysis. It executes
only the verified helper bytes; it cannot select a latest campaign attempt.

The earlier actual render namespaces `retrospective-qcd-n3-1` and `-2` remain
unchanged as layout history. Their scientific series are identical. This
version fixes clipped titles, annotation overlap, common x-axis limits and
the visible CutTools reference label. Source4's metric and statistical
function ASTs are identical to admitted source2.
