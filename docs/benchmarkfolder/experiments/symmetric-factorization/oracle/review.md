# Independent review of the initial FFV factorization

Reviewed generator: `factor.py` SHA256
`fec33d37a7f668355dd9073462117aa330cc81329ed7c4a069000c4b0962e570`.
The initial compiler campaigns retain that generator and their exact generated
Fortran sources. The sealed source proof is unchanged.

No algebra or indexing blocker was found for the admitted FFV1 maps:

- Runtime comparisons check the actual stored derivative rows against signed
  component permutations of `B(:,1,:)`; independently rounded source expressions
  are not assumed to obey a relation that their stored values violate.
- The constant `B0` products retain their original order for each output. The
  derivative terms deliberately change grouping, so numerical forward error,
  physical accuracy and unchanged stability decisions require validation.
- `U` sums use the original K traversal. Its scatter uses the original polynomial
  coefficient-position table. The independent oracle instead reconstructs the
  monomials and direct stored-input contraction.
- Terminal `U` rows are selected for the final diagonal cut component. Zero
  off-diagonal outputs and the full-output false-entry fallback are tested
  separately.
- The admission guard requires bounded finite operands, nearest rounding,
  supported/disabled halting modes and a true entry-inexact flag. The independently
  owned safety suite checks rejection and original fallback behavior.
- The numerical equality guard intentionally ignores zero-component signs.
  These are not silently treated as bit equality: signed-zero fixtures are
  accepted, output zero-sign differences are counted, and fallback bits remain
  subject to exact equality.

The exact stored-input campaigns currently recorded here cover both dominant
FFV1 maps, full/terminal helpers, NLO ranks 2–4, and standalone ranks 2/4. All
200 fixtures and 250,880 complex outputs pass. There are 2,582 output coefficients
whose exact magnitude is below `2^-80` times the absolute real-product scale.
The maximum factored forward error is 0.719030 times
`epsilon * absolute-product-scale`, with `epsilon=2^-112`; the maximum fraction
of the declared conservative bound is 0.048841. Inputs remain bit-unchanged,
cumulative IEEE flags match, and false-entry fallback outputs are bit-identical.

This establishes the checked stored-input numerical behavior. It does not
establish universal correct rounding, a physical amplitude tolerance, unchanged
rescue frequencies, a portable non-GNU binary128 contract, or a process speedup.
Those require their separately retained generated-process and environment
controls. Compiled FFS factor coverage is not inferred from FFV coverage; it is
a separate proposed extension.

One additional safety coverage request remains at this review point: the
48,000-case fallback suite always supplies a false entry-inexact flag, so those
cases short-circuit before the subsequent shape, rounding, range and stored-row
guards. They establish exact fallback identity, rather than independently
exercising those later guards. A separately retained true-entry campaign should
start with compatible stored rows, vary each admission condition, and check the
fallback counter, original output and flags. Enabled-trap controls already
exercise their early guard. This is a coverage gap, not an observed arithmetic
or guard defect.

That request was subsequently addressed by `true-entry-guards-v1`: all 152
controls pass across ranks 2/4, full/terminal and standalone/NLO flags. The driver
starts with compatible phase-related rows and actual sticky inexact set, then
applies one of 19 nonnearest, range, subnormal, nonfinite, row-relation or shape
violations. It requires one fallback, exact whole input/output bits and equal
flags. Independent source review confirms these cases reach beyond the
false-entry guard. This closes the identified coverage gap in the tested GNU
environment; it does not imply support for untested floating-point environments.

`test_oracle.py` also passes five focused checks of the oracle itself: IEEE
binary128 boundary decoding, exact complex products/scales, hand-tabulated
monomial positions, a known sparse contraction with a deliberately corrupted
output rejection, and source/provenance checks for all 60 rank/map/full-terminal
driver templates.

The read-only verifier rehashes all raw streams, numerical results and generated
sources, recomputes campaign totals, and optionally repeats the exact rational
contractions:

```sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization/oracle/verify.py \
  --recompute \
  --guards docs/benchmarkfolder/experiments/symmetric-factorization/true-entry-guards-v1
```

It creates no files, imports no compiler or numerical runtime, and suppresses
Python bytecode caches. Compiled executables are checked if present; their
absence is allowed because reproducible textual source/dump evidence need not
retain build products.
