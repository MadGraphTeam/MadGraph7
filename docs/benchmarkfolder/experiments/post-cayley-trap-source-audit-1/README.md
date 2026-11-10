# Source audit of the two remaining division sites

**Source-only proposal. No source patch has been applied, compiled, loaded or
physically replayed.** The closed localization package identifies two different
DIVSD sites in private DSO
`69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b`.
Neither is inside the guarded Cayley function. No actual operand registers,
MXCSR, signal subtype or core dump were retained. Association with the following
source expressions is an inference from the closed symbol/disassembly/source
evidence; the specific exceptional operands and trap causes remain unknown.

## Gram normalization: a signed-zero-only proposal is defensible

The checked-worker site at ELF `0x11fcecb` is in `gram_indicator::<f64>`.
Retained `gram.rs:724–725` computes
`w = largest / sqrt(cnorm2(delta).to_f64())`, then returns
`(largest > 0.0 && w.is_finite()).then_some(w)`.
The resulting `Option` is consumed by `alternative_estimate_of` at lines
963–965 to select a reduction estimate; it is not a disposable diagnostic.

`gram-zero-only.patch` proposes computing the same denominator, then checking
its integer bits for **either signed zero**, returning `None` only in that
case. For all nonzero bit patterns, leave the old division and predicate
unchanged. All determinant, numerator, conversion and square-root arithmetic
preceding the division remains in place.

The old masked `Option` value is provably `None` for a zero denominator:

| Numerator | Masked division by either signed zero | Old returned value |
|---|---|---|
| Nonzero finite value or infinity | Infinity, with sign determined by operands | `None` |
| Either signed zero | NaN | `None` |
| Any NaN | NaN | `None` |

This covers the returned variant across rounding modes. It does not promise
identical discarded NaN payloads or uninitialized `Option` representation bytes.
The integer test adds no floating comparison of a signaling NaN. Do not broaden
the guard to reject nonfinite denominators: a positive finite `largest` divided
by positive infinity currently produces `Some(+0)`. Nonfinite and nonzero
values retain their old path under this proposal.

This **intentionally changes exception flags and trap execution** on the
selected branch; it is not full IEEE equivalence. Earlier arithmetic can still
trap. A tiny nonzero denominator can still overflow the division, so this guard
is not established as a remedy for the observed failure. Rust/LLVM code
generation must also be inspected: a source-level control guard alone is not
proof that an exceptional operation will not be speculated in a new binary.

Before any promotion, root must approve a new isolated build and validate
signed zeros, zero/nonzero/infinite/NaN numerators, all nonzero/nonfinite
denominator paths, rounding modes, inherited flags and positive trap controls.
Then replay the same masked/trapped physical fixtures, separately checking
amplitudes, host stability, actual precision path and trap completion. No
unchanged-path or physical-success claim follows from this source proof.

## Dilogarithm: the integer estimate changes the recurrence

The forced-QP site at ELF `0x2523cc3` is inside the MPFR `series` symbol, but
the instruction is binary64. The loaded numerator is 39.47841760435743,
consistent with inlined `series_terms` at `oneloop-dilog.rs:73–75`:

```text
ratio = 4*pi*pi / modulus
count = ceil((bits+6)*ln(2) / ln(ratio)) as usize + 1
```

`series:125–131` forms `v=u*u`, converts its components to binary64, computes
their hypotenuse, and selects zero terms if that modulus compares equal to
zero. Otherwise it uses the estimated count, capped by the coefficient table.
Lines 141–148 use `coefficients[count-1]` and the corresponding Horner steps
to compute the term added to the numerical answer. Thus this is a convergence
estimator **with a direct effect on the numerical recurrence**.

Under nearest rounding, a positive overflowing ratio becomes infinity. The
remaining expression then gives one term. A very large finite ratio instead
produces a positive quotient and at least two terms before the table cap.
A clamp or log-domain rewrite therefore cannot claim the old masked result.
Directed downward/toward-zero overflow may produce the largest finite value
instead of infinity, so an unconditional tiny-modulus-to-one guard is unsound.

There is a conditional proof worth recording, **not an implemented patch**:
if nearest rounding is explicitly verified, positive modulus words at most
`0x0030000000000000` represent values at most `2^-1020`. Since the numerator
exceeds 32, the exact ratio exceeds `2^1025`, guaranteeing a nearest-rounded
overflow to positive infinity. Substituting `ratio=INFINITY` only on that
integer-classified interval and retaining the original remaining expression
would preserve its one-term estimate. This conservative interval does not cover
every possible overflow input. It must not be used without the rounding
contract, and it would suppress the skipped overflow/inexact exceptions.

Keep the caller's signed-zero bypass: both zero signs currently select zero
terms there, whereas calling `series_terms` directly with zero follows a
different exceptional path. Preserve negative, nonfinite and all other input
bit patterns; do not introduce new NaN comparisons. DAZ/FTZ behavior, the actual
divisor, rounding mode and exception subtype are unobserved in the closed run.
No dilogarithm fix is proposed for implementation until that evidence or an
explicit, checked rounding contract exists. Independent scalar and amplitude
accuracy qualification would still be necessary.

`proposal.json` binds the exact retained sources, instructions and localization
receipt, including pinned oneloopmaster
`bd703d09a78f7849a6b86a07096e6f222fc01193`. Its verifier reads text only,
reconstructs the proposed Gram change in memory and checks its hash. It neither
applies a patch nor exercises arithmetic in a numerical library:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/post-cayley-trap-source-audit-1/verify.py
```
