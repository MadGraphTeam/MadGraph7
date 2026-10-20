# Completed generated controls for guarded QP tensor sharing

The four complete ten-point controls ggH, ggHH, mixed QCD/QED and complex
masses pass all 21 pairwise comparisons among the four corrected candidate
backends and three helper-corrected original backends at both repetitions.
Every physical point/order row and status is checked under the unchanged
`1e-7` relative plus `1e-10 S` tolerance.

The sealed inventory covers these two newly completed runs:

- `control-runs/controls-candidate-abi3-tensor-factor-v1-hh-complex-20261020`
  (eight targets);
- `control-runs/controls-candidate-abi3-tensor-factor-v1-mixed-legacy-20261020`
  (three targets).

The unchanged, previously committed
`reference-runs/controls-reference-hh-complex-20261020` supplies six
helper-corrected original targets and is revalidated through its original binding.

It includes their raw two-command series, fresh parity where requested,
drivers, cards, point streams, source/build identities, diagnostics, two
runtime bindings and retained runtime assets. The comparison and its frozen
plan/source are also bound. The existing final ggH and mixed candidate
representatives and unchanged helper-corrected original ggH/mixed results are
reused through their existing validated bindings and representative proof.
The verifier follows that complete retained dependency chain.

These are correctness controls and diagnostic timings, not seven-repetition
performance acceptance. The ordinary forced-Ninja failures remain failures
in their historical namespaces. Rotations, mass/scale rebinding, fresh-context
and near-singular variants are excluded from this package, as is the still
incomplete eight-case corrected-original core collection.

From the repository root:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/controls-retention-abi3-tensor-factor-v1/verify.py
```

Verification hashes the immutable inventory, revalidates source/runtime/proof
bindings, reconstructs every complete stream and recomputes the comparison.
It performs no compilation or numerical evaluation and does not require the
disposable generated-output trees.
