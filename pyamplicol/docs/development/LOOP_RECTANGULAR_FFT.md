# Rectangular Born–loop colour contraction

The first implementation boundary extends the existing native symmetric-group
transform with a complex adjoint and a normalized inverse. It does **not** yet
connect FFT colour contraction to the loop scheduler or advertise public loop
FFT support.

The transform in `recurrence/symmetric_group_fft.rs` uses real orthogonal Young
representations and the convention

```
F_lambda = sum_g f(g) rho_lambda(g).
```

Its adjoint is the transpose of those real numerical instructions, acting on
complex values without conjugating them. The normalized inverse applies the
irrep weight `dimension(lambda) / order(G)` before this adjoint. Implementation
reverses the existing subgroup stages: sum becomes distribution, the branch
embedding becomes additive extraction, Young generators act in reverse order,
and the original permutation gather becomes a scatter. The same representation
metadata, generator operations and reusable native workspaces are used.

Scalar and lane-innermost batch methods share this implementation. As with the
existing forward API, an active transform has positive lane count; an enclosing
loop scheduler handles an empty logical batch without invoking a transform.
The arbitrary native degree-ten bound has been removed. Admission checks the
actual factorial and `u32` reorder-index storage instead. The existing tree
artifact/UI degree-ten guards are a separate boundary and are not a policy for
new loop plans.

## The required rectangular operator

Born and loop channels must have independent regular-orbit maps under the same
actual external-label action. For every **ordered** channel pair, define

```
G[(a,g),(b,h)] = k_ab(g^-1 h),
K_ab = Fourier[g -> k_ab(g^-1)].
```

The desired dual weights are `w = B^dagger G`. They follow from

```
R_b = sum_a Bhat_a K_ab^dagger,
r_b = inverse(R_b),
w_b = conjugate(r_b).
```

Neither a square matrix, a Hermitian kernel, a real result nor an upper-triangle
symmetry factor is implied. Exact direct entries remain for every pair outside
two certified regular orbits: orbit-to-residual, residual-to-orbit, and
residual-to-residual. Each contributes once.

Independent controls construct a complex nonsymmetric 13-by-20 matrix with two
Born orbits, three loop orbits, and nonzero residual terms on both sides. Native
transform/inverse dual weights agree with its full direct matrix action for
three simultaneous lanes. This is a mathematical interface test, not a physical
loop FFT result. Other controls compare arbitrary inverse/adjoint inputs against
explicit dense representation matrices, test both round-trip directions,
exercise trivial groups and changing batch tails, and retain existing forward
transform checks.

## Loop integration still required

Raw Born amplitudes and loop coefficients already meet in `loop_execution.rs`
before the family projector. The current producer embeds
`conjugate(B) * G * numerator_coefficient` separately into each family's Symbolica
instructions. The next scheduling change must compute a shared bank of dual
Born weights once per point, Born order, helicity/source and operator, then feed
those weights to family coefficient assembly. Denominator families need not
form complete colour orbits; colour maps and family/routing maps stay separate.

Cold preparation must certify actual loop tensor relabellings using dependency
canonicalization and exact overlap operations. Existing tree DDM, colour-sector
or helicity certificates cannot establish this. Nontrivial stabilizers,
incomplete orbits and marked-leg operators require their own certificates or
exact direct residuals. Model UV/R2 and field contributions remain complete even
when their first implementation uses direct contraction.

Formal Nc projection retains the existing common structural degree and exact
weights. Binary64 FFT binding happens only after that projection. DoubleFloat
and arbitrary precision use genuine exact-input direct evaluation until their
own transform implementation exists, and report the actual algorithm. Saved
plans, precision rescue, runtime policy and physical comparisons are required
before exposing public `auto`, `direct` or forced FFT behaviour.

The dependency audit found the required index renaming, canonicalization,
conjugation and formal colour operations in FeynKit/Community. Their source,
documentation and examples/tests were checked. Fourier work reuses pyAmpliCol's
existing implementation; no second group representation algebra is introduced.
