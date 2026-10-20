# Minimal production proposal, held for numerical and process validation

`production-tensor.patch` changes only the existing bridge adaptation in
`vendor/ninjars/src/tensor.rs`. It expands the pinned complex multiplication
into its identical four scalar products and two additions, preserving operand
and accumulation order while avoiding redundant outer complex clones. Existing
finite zero/one scratch resets reuse their buffers, and vector growth lazily
clones only new slots. This does not modify MadLoop's Fortran, scalar providers,
thresholds, transport, precision, topology, recurrence, or rescue policy.

The exact proposed tensor body is the measured allocation source with only the
experiment's trailing `cfg(test)` module declarations removed. Retain upstream
attribution and the explanatory scalar-expression comment. Do not generalize
`set_from` to arbitrary inputs: prior NaN-source sign counterexamples remain
documented, while these reset sources are always the constructor's finite
zero/one values.

Durable regression recommendation:

- Add the frozen dense implementation as a test-only reference and the four
  tests in `reused/tensor_allocation_tests.rs`, with the same two `cfg(test)`
  declarations used by the validated experiment. They independently compare
  stored rational values, tracked precision, sign and finite state over ranks,
  all expansion kinds, mixed precisions, nonfinite operands, empty/reordered
  powers, lane rebinding, and actual zero/one reset sources. The reference source
  is the previous production implementation, not a mirror of the new code.
- Keep the existing two transport-contract tests and the inherited complex-mass
  256/384-bit convergence test unchanged. These cover the corrected parser's
  negative zeros, branch lips, subnormals and values distinguishable beyond DP.
- Retain `reused/tensor_capture_tests.rs` as explicitly ignored integration
  tests: all21 golden lanes and the full1,392-record retained request stream must
  compare every dense/optimized callback exactly. The existing production
  MpState replay separately checks all11,136 final ASCII fields with reusable
  contexts. Run these serially under the exclusive MP slot before promotion.
- Keep the pure-kernel timing seal as research evidence; it is not a unit test
  and does not establish MadLoop performance. Require the separately declared
  complete-process comparison before deciding to port this patch.

Root owns any production application, tests, installation, commits and pushes.
This directory only contains the reviewable proposal and isolated validation.
