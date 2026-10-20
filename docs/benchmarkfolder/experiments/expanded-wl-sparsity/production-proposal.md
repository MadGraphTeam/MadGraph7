# Conditional production patch proposal

This records the pre-implementation boundaries. The quiet seven-pair QCD3
improvement and EW2 DP regression check subsequently passed, and the
production change has been implemented and independently reviewed. Fresh
final-candidate benchmark generation remains a separate step. The
model-helper precision corrections remain a separate change.

The production scope is:

1. `madgraph/various/q_polynomial.py`: emit the tested version-4 sparse
   Fortran only when `coef_format == 'complex*32'`. Keep both DP updaters and
   the alternate compact loop ordering byte-for-byte unchanged. Apply
   compact pruning to the normal vertex-coefficient-first form and expanded
   pruning to the expanded form. Preserve all active/inactive output clearing
   and every retained accumulation order.
2. Factor the unchanged dense body into a private generated `_DENSE` routine
   with the same argument list. Non-nearest rounding, any nonfinite active
   operand, or an unsupported REAL16 representation calls this body before
   any pruning. The exceptional fallback must remain separately compiled;
   merely disabling skip flags changed NaN signs in the rejected first
   prototype.
3. Emit the version-4 elemental finite classifier, which uses integer
   `TRANSFER`/exponent inspection after verifying IEEE binary128 storage,
   radix, precision and exponent range. Its word order is derived from the
   representation of `1.0_16`. Do not replace it with `IEEE_IS_FINITE` or
   `IEEE_CLASS`: both raised INVALID for signaling NaNs in empty-output
   calls that did no original arithmetic.
4. Emit the exact-zero masks and guards from `prototype.py` without changing
   their arithmetic. Expanded masks cover B blocks; compact masks cover A
   coefficients after the existing B-zero check. No magnitude threshold,
   rational reassociation, changed accumulator initialization or model
   parameter specialization is allowed.

The implementation should build these statements directly in the existing
generator methods, sharing the finite-classifier/fallback emission through
one private method. It should not import an experiment script, depend on
formatted-source regex matching, or add state to numerical COMMON blocks.
The current generated-source transform is the reference for the emitted
Fortran, not the intended production Python architecture.

The meaningful regression belongs in
`tests/unit_tests/various/test_q_polynomial_compiled.py`, with an independent
pre-change dense oracle retained in the test. Promote the existing complete
output/input-bit and IEEE-flag checks, including:

- expanded `(1,1)` and compact `(2,1)` plus higher-rank representatives;
- dimensions including zero, active and inactive regions;
- dense, sparse, cancellation, signed-zero, subnormal and overflow inputs;
- quiet/signaling NaN and infinity, including zero multipliers and empty
  output dimensions;
- nearest, upward, downward and toward-zero rounding;
- both `-O3` and `-O3 -frounding-math`.

The existing 210,000 full-rank comparisons and 24,000 signaling comparisons
remain retained experiment evidence; unit tests can use representative ranks
to keep routine test cost bounded. Test DP output strings against the
unchanged pre-patch generator and compile the existing DP differential suite.
Do not derive the dense oracle by removing guards from the new generator.

After a production patch exists, compare its formatted Fortran against the
measured prototype, regenerate fresh standalone outputs, repeat complete
coefficient/status checks, update only affected QP IO snapshots, and refresh
hash-bound benchmark generation/compilation and protocol parity provenance.
A successful isolated experiment alone does not authorize reusing old
accepted executable measurements for the changed generator.
