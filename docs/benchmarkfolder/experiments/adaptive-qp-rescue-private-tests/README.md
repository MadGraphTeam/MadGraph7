# Private adaptive-QP unit qualification

One authorized serial Rust test process passed **20/20 tests** on CPU 11. The
reviewed controller used the shared numerical lock, an inclusive 180-second
deadline and no retry. The owned child was reaped normally in 0.32414 seconds;
this duration is administrative evidence, not a performance measurement.

The exact test executable has SHA256
`d22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7`.
Its source is private revision 3, index
`f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213`,
from the immutable compile package `adaptive-qp-rescue-private-build`
(`1a4af7f8…`). The exact controller is preserved by
`adaptive-qp-rescue-private-test-controller` (`615bc8f5…`), which records its
earlier conditional authorization; this result records the later admitted run.
No compilation, installation or production modification occurred in this run.

The eleven previous bridge tests pass unchanged. Nine added tests cover narrow
conditioning-refusal eligibility; unchanged ordinary 256-bit output; at most
one 512-bit retry with 113-bit inputs and signed zero preserved; no retry for
unrelated errors; independent full/pole/check/tier workspaces; exact cache keys,
budget and check bypass; and failure invalidation. Injected outcomes test dispatch
and failure paths explicitly and are not numerical reference data.

Three added tests also execute real scalar/tensor kernels. In particular, the
captured massless N4R3 group 10 fixture refuses at 256 bits and succeeds after the
single 512-bit retry in both full and poles-only modes. Independent checks
reevaluate both tiers; full and partial residues satisfy the declared
`2^-100 * (|a|+|b|+1)` comparison. The finite slot in the poles-only result is
an unevaluated internal zero placeholder. The fixture is not physics truth.

The panic message in `attempt1/tests.log` is intentional: the poison-control
test catches that panic and verifies that it cannot become a precision retry.
All twenty named tests completed, with no failures, ignored tests or filtering.

This directly linked Rust binary was tested; the separately built adaptive
cdylib has not been loaded by this test. Actual Fortran transport, dynamic-runtime
mapping, IEEE/trap behavior and physical amplitudes mixing 256/512-bit results
remain separate qualification gates. No actual-runtime mapping observer was
used by this test controller. No production or benchmark acceptance is implied.

`python -B verify.py` checks the portable inventory, exact dependency bindings,
the twenty named log results and recorded before/after hashes without executing
native code. Add `--live` to rehash the original protected inputs and binary.
Only text and hashes are included; no compiled artifact is packaged.
