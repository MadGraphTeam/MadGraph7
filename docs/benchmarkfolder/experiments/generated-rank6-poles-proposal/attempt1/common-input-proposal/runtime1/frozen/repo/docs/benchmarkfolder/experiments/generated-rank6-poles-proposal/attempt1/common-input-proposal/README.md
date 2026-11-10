# Separate common-input mapping fixture — source-only proposal

This is an independent synthetic tensor fixture. It does not replace, zero,
repair, or relabel any captured physical point. It uses the previously compiled
generated rank-six adapters and observer, with a new private driver and two
explicit closed polygons from `fixtures.json`. No physical numerator or
amplitude/capture routine is called. The card initializes model/global state
only; all diagnostic loop shifts, squared masses and scale are specified here.

Fixture108 has six propagators/rank6; fixture109 has two propagators/rank6.
Every input is an integer or dyadic rational. The reference shift is exactly
zero, external differences sum exactly to zero, and no sum/product can incur
binary64 rounding at this scale. Complex squared-mass components are exact
multiples of1/8. The origin plus five nonzero six-point shifts have nonzero
exact Gram determinants for every subset of2 through5 points; the six-point
Gram rank is necessarily at most4 in four dimensions. This is a source-level
nondegeneracy check, not a native stability prediction. All mapping components
receive generic nonzero momenta and unequal complex masses. The six-point finite
tensor and the two-point UV-pole tensor provide complementary witnesses for
all210 coefficients throughrank6. Missing or zero witnesses remain unqualified.

The30-row schedule contains both fixtures, DP/QP, full/poles-only, all three
UV/IR selectors, and COLLIER. Exact effective shifts, invariant matrix, masses
and scale must match the frozen mathematical inputs at the actual ABI boundary
before comparisons count. The adapters still execute their own ordinary
kinematic constructors; no production invariant is overridden. Because these
inputs are exactly representable and are away from the mass-shell thresholds,
their cyclic and matrix constructions should agree bit-for-bit. A discrepancy
is a failed protocol assertion, not permission to change an input.

Comparisons retain existing3e-6 independent-backend acceptance and1e-8 target,
with the same finite-relative small-pole rule; full/partial DP and QP bounds
remain1e-10 and1e-24. Every refusal, bad IEEE flag and missing coefficient
witness remains explicit. Fresh-process IEEE state is observed before and after
model initialization and at each adapter/native boundary. No flag is cleared.
This distinguishes newly raised native exceptions from inherited state in the
original physical diagnostic without changing that original evidence.

Proposed next step: source review, then one private driver compile and one link
with all original numerical objects, using the already reviewed ownership
helper and120-second/4GiB compile-only bound. A later separately admitted
30-row worker would retain the original900-second/4GiB bound, common lock,
runtime hashes and actual mapping check. CPU affinity is nonexclusive. No
compiler, loader or native worker has been run for this proposal. Final precise
timing remains a TODO for Hua-Sheng's local machine.

`controller.py` reuses the frozen CMS controller and reviewed ownership helper.
Only its work/source paths, the source-verification callback, scope text,
and omission of the unused physical point file change; compilation still creates
one new driver and one executable. `verify_fixture.py` independently reconstructs
the exact driver transformation and checks all56 nonzero Gram minors using
rational arithmetic. No new process supervisor is introduced.
