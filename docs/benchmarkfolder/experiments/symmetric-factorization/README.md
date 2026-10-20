# Factored momentum-dependent accumulation experiment

The [research note](research.md) describes a source-proved way to share
contracted rows between momentum coefficients. This is a changed-rounding
prototype proposal, not a production optimization or measured speedup.

`proof-seal.json` binds the symbolic analysis, exact-parser checks, initial
analysis, and retained input/source snapshots. It covers only the source-proof
stage; subsequent compiled experiments and independent numerical-oracle work
have separate evidence and are not certified by this seal.

The proof verifies 1,024 identities across 16 fermion/scalar maps and explicitly
excludes two vector maps. Its static selected-product reductions are 38.59%
for QCD3 and 41.12% for QCD4 before runtime zero skips and guard failures.
The constant vertex term remains unchanged. No timing follows from these
counts.

From the repository root, without a generated output tree:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization/verify-proof.py
```

This rehashes retained sources, replays the exact polynomial identities,
recomputes the static counts, and checks decimal parsing. It runs no compiler,
amplitude, or scalar-integral provider.
