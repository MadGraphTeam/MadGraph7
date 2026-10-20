# Signed-zero loss in MP input widening

The source-bound scalar and GNU REAL16 checks reproduce the old defect and pass
with direct precision widening. The isolated complete bridge passes all 18
ordinary tests and reproduces all 11,136 output fields from 1,392 stored requests
exactly. Production code and the completed capture package were unchanged during
all preparation, compilation, and execution in this package.

`parse_record_at_precision` first recovers the binary128 decimal record at 113
bits, but then converts that Float to an exact rational before creating a
256-bit Float. A rational has one zero, so this erases the sign of every negative
zero. The existing transport regression checks one positive nonzero beyond-DP
value by rational equality; it cannot detect this defect. Nonfinite rejection
is already explicit and should remain unchanged.

The proposed minimal fix is `input.set_prec(precision); Ok(input)` on the parsed
finite Float. At the pinned Symbolica revision, this delegates to Rug's
`set_prec`, then MPFR's `prec_round`. Increasing precision preserves every finite
represented dyadic, including signed zero. MPFR's own retained source test
explicitly checks that `prec_round` preserves both zero signs. All actual MP
working precisions are at least 113 bits (256 production, 384 convergence tests).
This also avoids an unnecessary rational conversion, but no runtime benefit is
claimed or measured here.

Signed zero is observable downstream: pinned `Complex::arg` uses `atan2`, and
its square root explicitly preserves signed-zero branch lips. Thus `log(-1-i0)`
and `sqrt(-1-i0)` differ from their `+i0` versions. Complex squared masses accept
non-positive imaginary parts, so `-0` is a permitted input. Momentums, S entries
and numerator coefficients also pass through the affected parser. Their zero
signs can propagate into divisions, roots and complex arguments.

This establishes an interface-contract defect, not a demonstrated wrong
physical amplitude in the existing benchmarks. In particular, oneloopmaster
explicitly selects the physical lower logarithm lip on real axes, and the master
provider deliberately snaps sufficiently small masses/invariants to positive
zero. Those existing conventions can mask or supersede an input zero sign for
some master calls. They do not justify erasing signs in the generic transport
before geometry or numerator evaluation. Source excerpts and exact dependency
hashes are retained in `source-evidence.json`.

The sealed QCD3 capture contains two negative-zero coefficient components among
its 464 QP records; its S, P, M and U components contain no negative zeros. The
integer-only inventory is `captured-zero-inventory.json`. Therefore the current
capture controls reproduce the old canonicalizing transport. A future corrected
library is a different runtime identity and must not be substituted into those
controls while claiming original byte-identical runtime attribution.

Output transport is different: production formats `Float` with `.39e`, which
delegates to Rug's exponential formatter. Rug checks `is_zero()` and writes
`-0` when `is_sign_negative()` is true. The production record copier only pads
the resulting bytes. Fortran's module unpacks the text with a REAL16 read, with
no intervening rational conversion. The proposed tests include output-format
and 113-bit reparse sign checks. Actual GNU Fortran read/TRANSFER checks now pass
all 14 cases with the proposed parser, while the old parser fails negative zero.

`minimal.patch` applies only to the input widening. `proposed_tests.rs` supplies
the separate Rust contract test module and uses exact rational dyadics as independent
expected values. The 14 finite fixtures include both signs of zero, minimum and
maximum subnormal binary128, minimum normal, maximum finite, a beyond-binary64
bit and the final binary128 significand bit. Every fixture is tested at 256 and
384 working bits, with exact rational value, precision, sign, output record
width, and formatted-output reparse assertions. NaN, infinities and malformed
records remain rejected. Separate complex log/sqrt tests demonstrate the branch
contract directly without creating a reducer context.

The validation record is:

1. `scalar-result.json` retains both exact parser bodies in a standalone program:
   the old path fails two sign, two formatted-sign and two complex branch checks;
   the proposed path passes all checks. Both preserve the independently specified
   nonzero dyadics and reject nonfinite/malformed records. No engine was created.
2. The same result binds actual GNU REAL16 text-read and integer `TRANSFER` checks:
   the old negative-zero output fails, while all 14 proposed outputs and Fortran
   pack/read round trips pass, including normal/subnormal extremes.
3. `bridge-build.json` records a successful isolated full-crate compilation with
   unchanged production optimization flags and the source-bound parser/tests.
   `production-ready.patch` reproduces the entire compiled source difference,
   including a separate ignored captured-request replay module.
4. `bridge-result.json` records 18 passing ordinary tests (16 existing plus two
   transport tests), followed by the complete 1,392-request captured replay.
   Every status and all eight ASCII output fields per request match the retained
   expected result exactly: zero changed records or fields. No historical
   artifact or expected runtime hash was updated. The corrected input transport
   remains a distinct runtime identity even though these sampled outputs agree.
5. Treat installation and generated-process checks as a separately versioned
   runtime refresh if the fix is promoted; allocation optimization remains a
   distinct source change and evidence package.

`production-ready.patch` is the exact complete source delta of the tested crate:
the finite input-widening fix, two ordinary contract tests in a separate module,
and one ignored captured-request replay test. The latter requires explicit input
and output environment paths and does not run in the ordinary suite. The scalar
red/green failures are expected counterexamples, not discarded failed inputs.

`verify.py` rehashes the sealed files and independently matches every replay row
to the frozen captured request without building or invoking any engine. The
initial proposal documents remain as historical source-only preparation records;
the subsequent build and result JSON files state what was actually executed.
