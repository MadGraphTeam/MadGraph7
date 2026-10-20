# Exact stored-input oracle for derivative-row factorization

This directory independently checks the isolated Fortran factorization prototype.
It does not modify production code, evaluate amplitudes, or use Symbolica.
`driver.py` generates a free-form Fortran driver; the experiment owner compiles
that driver with the original and factored helpers. `oracle.py` checks its raw
hexadecimal dump using exact Python `Fraction` arithmetic.

The reference is the complete mathematical contraction of the **stored**
binary128 inputs:

```
OUT(j,n,i) = sum[A(k,l,i) * B(j,m,k) : monomial(l)+monomial(m)=monomial(n)]
```

The oracle independently constructs monomial indices and sums real and imaginary
products. It does not reuse the prototype's intermediate `U`, operation order, or
row-factorization formula to compute the reference. Terminal accepted calls check
the diagonal `i=j` and require zero off-diagonal outputs. The false-entry
terminal control invokes the existing full-output fallback and checks that full
output instead.

`driver.py` binds the sealed source proof and its exact generated FFV/FFS source
hashes. Fixtures 1–6 and 8–10 construct derivative rows from stored `B(:,1,:)`
by exact component sign changes and swaps. Fixture 7 preserves the independently
rounded generated ALOHA expressions, so a relation failure must fall back. These
are controlled stored-input algebra fixtures; they do not replace physical
full-process validation of the actual generated vertices.

The implemented runtime guard compares finite numerical component values.
It ignores zero-component signs, including the zero component of a nonzero
complex coefficient. The sealed research note described a prospective stricter
bit guard; this implemented contract is deliberately different. Fixtures 9 and
10 include `(nonzero,+0)`, `(nonzero,-0)`, `(+0,nonzero)`, and `(-0,nonzero)`.
Accepted factored outputs may change rounding and zero signs. Fallback output
must remain bit-identical to the original output.

The ten fixtures are:

1. Dense inputs with perturbations below binary64 precision.
2. Sparse inputs including signed zero.
3. Severe contraction cancellation for FFV maps; alternating shifted coefficients
   for one-entry scalar rows.
4. Inputs scaled by `2^900`.
5. Inputs scaled by `2^-900`.
6. Coefficient-dependent `2^900` and `2^-900` scales.
7. Original independently rounded ALOHA rows, with runtime fallback permitted.
8. False entry-inexact flag, requiring the original fallback.
9. Real derivative-row entries with both signs of imaginary zero.
10. Imaginary derivative-row entries with both signs of real zero.

The dump records every active A/B component and every relevant output as two
16-digit hexadecimal words per real binary128 component. An explicit
storage/digits/exponent/word-order descriptor is checked. Inactive input regions
contain sentinels; whole-array integer `TRANSFER` comparisons verify that neither
helper mutates any active or inactive input. The driver records cumulative IEEE
flags immediately after each call, both path counters, and zero inactive output
tails. The validator requires all ten complete cases in order, complete arrays,
matching flags, unchanged inputs, correct path selection, and the final marker.

For each output, let `N` be the number of nonzero complex input products and let
`S` be the exact sum of absolute values of their four real products. The declared
forward bound for **both** implementations is

```
max(abs(real_error), abs(imag_error)) <= gamma_(2N+8)(epsilon) * S
epsilon = 2^-112
gamma_k(epsilon) = k*epsilon/(1-k*epsilon)
```

This intentionally conservative componentwise bound covers complex-product
rounding and the original or grouped summations: at most 20 complex terms per
output, at most four terms per `U`, and at most four constant terms plus four
`U` contributions in the final sum. `epsilon` here is twice the unit roundoff.
The fixture exponent range keeps every nonzero arithmetic result away from
binary128 overflow and gradual-underflow boundaries; the separately owned
safety suite checks rejected environments and exceptional inputs. This is a
forward-error validation for the specified operation graphs and fixtures, not
a universal correctly-rounded polynomial claim. Severe cancellation is
reported independently; no relative-to-near-zero-output gate is used.

Results retain exact rational references, observed values, scales, and errors at
each implementation's worst normalized output, as well as all raw input/output
bits in the separate dump. A result JSON binds the dump, source proof, and
validator hashes and refuses to overwrite an existing result.

Example from the repository root after the experiment owner produces a dump:

```sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization/oracle/oracle.py \
  path/to/dump.txt.gz --vertex MP_FFV1L2_1 --result path/to/oracle-result.json
```

The driver factory is `driver(rank, vertex, terminal=False)` and uses the agreed
`TEST_UPDATE_WL_<rank>_1_FACTOR` / `TEST_TERMINAL_WL_<rank>_1_FACTOR` signatures
with explicit `WL_ENTRY_INEXACT`. Instrumented helpers expose
`COMMON/FACTOR_TRACE/FAST,FALLBACK`. Compiler/safety/timing orchestration belongs
to the sibling prototype runner; this directory contains no performance claim.
