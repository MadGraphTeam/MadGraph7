# Additive offline analysis for the parameter-integral check

This parser compares explicitly closed quadrature records with both retained
native results. It imports no reducer and performs no quadrature. The original
prototype, inputs and historical disagreement remain unchanged.

The input join is fixed to fixture108, TIRust QP action3007 and uncached COLLIER
action3013. Every component index is checked against the frozen monomial list.
The finite Laurent slot is0; its binary128 words0/1 and2/3 encode the real and
imaginary parts. All finite binary128 numbers, including subnormals, decode
exactly. The four lower-rank controls precede the three disputed components in
the output.

Every completed order retains the estimate, both native values, absolute
differences, relative differences and the historical symmetric residual
`abs(x-y)/abs(x+y)`. Comparison flags describe the estimates at the existing
`3e-6` criterion and `1e-8` target. Same-order binary64/80-digit Decimal
comparisons isolate observed arithmetic differences from changes in quadrature
order.

The exact analytic Gaussian error bound is checked against the immutable
derivation. The additional residual enclosure uses triangle inequalities,
exact rational arithmetic and outward-rounded decimal displays. It includes
the analytic integration error only; it explicitly excludes finite-precision
node, weight and arithmetic errors. The full result therefore preserves its
finite-precision qualification limit. A practical inference from convergence,
the independent arithmetic check, the bounded analytic tail and the actual
backend flags must remain scoped to these sampled components. It does not
certify every rank-six component or a physical amplitude.

The CLI refuses to read a result until the root confirms its process terminated
and supplies `--admit-closed-result`. It rejects running sequences, changed
fixture identities, component remapping, altered analytic bounds, duplicate
orders and nonfinite values. Output files must be new, so prior results remain
immutable.

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/rank6-feynman-parameter-analysis-proposal-1 \
  -p 'test_*.py' -v
python3 -B docs/benchmarkfolder/experiments/rank6-feynman-parameter-analysis-proposal-1/compare.py \
  --result /absolute/path/to/closed-float-result.json \
  --paired-result /absolute/path/to/closed-decimal8-result.json \
  --output /absolute/path/to/new-comparison.json --admit-closed-result
```

No COLLIER cache code is changed. Its recorded accuracy flag−1, error flag0 and
IEEE observations remain visible; neither backend is assumed true.
