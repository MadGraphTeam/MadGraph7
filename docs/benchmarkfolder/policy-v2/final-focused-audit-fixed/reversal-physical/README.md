# Physical-kinematics installed-ABI3 reversal control

This separate attempt passed with all thirty required output rows and status
zero in native DP, compensated DP and genuine MP. Original/reversed complete
Laurent outputs satisfy the fixed bounds; retained/fresh outputs are bitwise
identical within each mode. Both negative controls are resolved. Raw buffers,
all outputs and archive/module/source identities are retained in `run.log`
and `result.json`.

The original all-spatial synthetic attempt remains unchanged under
`../reversal`. Its first native
triangle returned `UNSTABLE`; `../reversal-initial-diagnosis.json` binds that
failure and the upstream real-basis negative-square-root explanation. No
invariant comparison completed in that run and no scalar-master failure is
inferred.

This attempt uses the retained `sqrt_s_416.7` ggHH rank-three triangle and
rank-four box fixtures in `vendor/ninjars/tests/data/cases`. The original
nonzero odd coefficients, momenta, supplied invariants and real squared masses
are preserved. Distinct exactly representable negative imaginary mass
components `-(index+1)/4` test the associated propagator/mass permutation.
Inputs and source fixture hashes are retained in `inputs.json` and the actual
readable numeric buffers are in `inputs.dat`.

The denominator convention, q→−q transformation, unchanged thresholds,
original/reversed/repeated/fresh sequence, genuine MP `1+2^-100` perturbation,
and wrong-unflipped-numerator negative control are exactly those described in
`../reversal/README.md`. The original and transformed first shifts need not be
zero; no affine translation is introduced. Thirty complete output rows are
required. The fixed negative-control bound remains
`10 × (1e-7 + 1e-10) × S`, set before any execution.

The negative-control differences exceed this bound by factors of `7.83e5`
and `9.01e5`. The two MP invariant differences are fractions `2.39e-48`
and `2.66e-45` of the `1e-31` bound. These small residuals arise in tiny
pole components; they do not imply that binary128 finite parts have
77-digit accuracy. Every finite, single-pole and double-pole component is
included in the invariant comparison. The separately recorded rational term
is an engine-basis diagnostic.

From the repository root, after root allocates the shared Symbolica slot:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python -B docs/benchmarkfolder/policy-v2/final-focused-audit-fixed/reversal-physical/control.py run --cpu 5
```

The runner refuses to overwrite prior results or build products. Every failure
is retained. Source preparation alone uses the same script's `prepare` action.
