# Independent source review

Reviewer: `/root/audit_fortran`, 2026-10-20, read-only review before numerical tests.

The reviewer found no arithmetic or state blocker. `borrowed_product` reproduces
the pinned `Complex::mul` scalar operand order and expressions, removing only the
outer complex clone. `Complex::set_from` delegates both components and
`Float::set_from` uses the underlying `clone_from`, consistent with a complete
stored-state copy. `resize_with` avoids the unused fill clone at unchanged length.

Recurrence, addition, contraction order and literal-zero operations remain
intact. Mixed-precision, sign, nonfinite product and assignment controls, the
dense recurrence/rebinding comparison, and actual-request replay address the
relevant risks. Numerical claims remain conditional on the tests. The reviewer
did not build or execute a numerical process.

Subsequent test finding: the complete-copy interpretation above is too broad for
an arbitrary NaN source. Rug 1.30's `clone_from` can retain the destination's NaN
sign. The prototype's actual `set_from` sources are finite zero and one, so the
new test verifies those resets from arbitrary prior states. The original failed
test and counterexample remain retained. The reviewer was informed before the
corrected test was run; the arithmetic patch has not changed.

The reviewer acknowledged the correction: the source review supports the pinned
borrowed-multiplication expressions and the actual finite zero/one reset domain,
and does not establish general NaN-source copy semantics. The retained
counterexample and separate strict exceptional-product checks bound the claim.
