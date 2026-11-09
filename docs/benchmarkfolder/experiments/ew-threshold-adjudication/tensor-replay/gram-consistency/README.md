# Fixed-input Gram geometry

This source/input audit reconstructs the exact rational Gram determinants of
the three saved rank-three box requests. It makes no reducer call and changes
no input. It compares the authoritative supplied invariants with the exact
geometry of the same fixed binary128 shift components; the latter is not a
new stream sent to either reducer.

For threshold group 5, the supplied-invariant Gram determinant is
`1.2226858389020917935e-13`. The normalized Hadamard determinant is
`5.2240781136634144476e-30`, reproducing the captured native refusal condition
to better than `1e-55` relative. Reconstructing the Gram matrix directly from
the fixed shifts gives determinant `1.2102331616577179281e-13`, a
`-1.018469%` change, and normalized determinant `5.1708724932341610240e-30`.
Both determinants are positive and nonzero. The tiny supplied/shift mismatch
therefore affects this near-degenerate geometry, but does not create or remove
its severe conditioning.

The ordinary group-5 determinant changes only `-6.7393e-17` relative. The
successful threshold massless group-24 control changes `1.4981e-28` relative.
All matrices, signed determinants, exact determinant fractions, recursion
scales, source-defined condition estimates and rank-three powers are retained
in `summary.json`.

The comparison implements `direct/exec.rs`'s minimum of the Hadamard-relative
determinant and the scale-relative cube root of the determinant, followed by
the coefficient-rank power. `direct/numeric.rs` supplies the Hadamard
definition. Fractions decode the binary128 inputs exactly; only roots and
logarithms use 90-digit decimal arithmetic. These source-defined estimates
are heuristics, not certified condition numbers or numerical error bounds.
They do not establish tensor or amplitude accuracy. Replacing supplied
invariants with shift-derived invariants would change API inputs and requires
a separate, explicitly labeled diagnostic.

Reproduce from the repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/gram-consistency/audit.py
```
