# Closed invariant classification: forward CMS and EW threshold

This audit adds exact input arithmetic and scale diagnostics to existing closed
fixtures. It evaluates no integral or amplitude, changes no production file,
and proposes no lower `OSThres` or replacement guard. Two small recomputation
helpers read the original hash-bound evidence in this checkout; no large trace
packet is copied.

MadLoop's massless test compares `abs(DIFFSQ)/N` with `OSThres`, where
`DIFFSQ=(qi-qj)^2` and `N=((abs(qi[0])+abs(qj[0]))/2)^2`. The default threshold
is the binary64 value read from `1e-8`, also used by the QP helper. The report
retains that exact threshold, nearest-even53/113-bit operation models, exact
rational input geometry, and each input's shell residuals separately.

## Forward CMS points20/21

For the saved repaired `p1-p3` transfer:

| Point | Rounded113-bit transfer (GeV²) | Original-energy N (GeV²) | abs(DIFFSQ)/N |
|---|---:|---:|---:|
|20|−2.49999999999979135e−7|250000|9.99999999999916541e−13|
|21|−2.50000000000000047e−11|250000|1.00000000000000019e−16|

Both genuine nonzero spacelike transfers therefore satisfy the primary `1e-8`
zeroing test. Their time differences are exactly zero. The Euclidean squared
norm of each difference vector equals the absolute exact Minkowski square;
there is no local cancellation between timelike and spacelike squares.
Relative to the difference-vector scale, the transfer is approximately unity.
Thus the small original-energy ratio here measures a small transfer relative
to the individual energies, rather than a nearly on-shell difference vector.

The outgoing-leg shell is a different quantity. At point20 its exact raw DP
input residue is `−2.64385e−12 GeV²` even though the simulated DP operation
returns zero. At point21 the exact residue is `−2.5e−11 GeV²` and the simulated
DP operation returns `−2.91038e−11 GeV²`. After the saved QP repair, sequential
113-bit shell evaluations are `+2^-95` and `−2^-95 GeV²` (about `2.52435e−29`).
Those differ from the separately retained exact dyadic input products. The
physical transfer is well separated from the repaired shell residue in these
fixtures; that observation alone is not a universal error bound or classifier.

All96 saved probe helper rows agree with the arithmetic model. The separate
point20 actual-matrix trace contains40 exact rounded physical-transfer delta
matches across104 matrices, in two ordered input/output classes, both with zero
internal masses. That trace used alternate `OSThres=1e-24`; its retained S
preserves the transfer. It supplies no primary-policy native result, no point21
matrix trace and no loop-group labels. The primary-policy zeroing conclusion
is source-bound arithmetic, not a new native evaluation. All35-significant-digit
coordinate/trace reconstructions have unique normal-binary candidates; zeros
use normalized scientific-format semantics rather than an arbitrary decimal
zero interval. Shorter raw DIFFSQ prints are interval-checked, not assumed exact.

## EW threshold group5 is a separate case

For ordinary point1 and threshold point2, the independently recomputed helper
matches all12 captured pair invariants exactly. Both points take the same
branches: mass snaps for pairs1,2 and2,3, plus zeroing of pair3,4. The latter
is bit-exactly the captured external photon `p5`; threshold rounded113-bit
DIFFSQ is already zero. No forward spacelike channel is erased in this example.

Here the massive snaps expose the separate mismatch between a DP-formed loop
mass squared and the QP square of the promoted model mass:
`−8.8448359747417284e−13 GeV²`. The repaired top-shell residuals relative to the
latter are around `1e−30 GeV²`. Supplied snapped invariants and fixed shift
geometry therefore differ. This corroborates their recorded inconsistency;
it neither identifies a faulty recurrence nor qualifies changing the shared
upstream mass contract. The external-channel correspondences retain their
nonzero origin defects, and no exact symbolic routing is invented.

The original-energy scale also depends on a common translation of chord
energies, while their difference and its norm do not. Difference-vector norm
scales remove that particular origin dependence but remain diagnostic scales,
not qualified on-shell predicates. A genuine small almost-lightlike channel can
still require physical provenance. These fixtures do not justify changing
thresholds, treating every small invariant as physical, or modifying scalar
fallback behavior. Compiler transformations, actual rounding modes, hardware
flags, independent finite/single-pole truth, full stress/CMS acceptance and
performance remain outside this audit.

Recompute from the repository after sourcing `tools/tirust_env.sh`:

```sh
python3 -B docs/benchmarkfolder/experiments/invariant-classification-audit-1/cms/audit.py --check
python3 -B docs/benchmarkfolder/experiments/invariant-classification-audit-1/ew/recompute.py --check
```

`index.json` binds the six new payloads and the18 existing input paths. The
private `--write` modes authored these summaries; verification uses `--check`
and writes nothing. Original fixtures and their failure classifications remain
unchanged.
