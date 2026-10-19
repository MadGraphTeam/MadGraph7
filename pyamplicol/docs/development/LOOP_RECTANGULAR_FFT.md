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

Raw Born amplitudes and loop coefficients meet in `loop_execution.rs` before
the family projector. The shared exact dual-Born schedule described below now
computes a bank once per point, Born order, helicity/source and operator, then
feeds its selected columns to family coefficient assembly. The original
`conjugate(B) * G * numerator_coefficient` projector remains an independent
baseline. Denominator families need not form complete colour orbits; colour
maps and family/routing maps stay separate.

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

## Cold tensor certificate

`loops/color_orbits.py` now prepares a separate, internal certificate from actual
Born and loop dependency tensors, absolute coupling orders, a common adjoint-port
action and exact rectangular metric banks. Bank names retain the owning
observable's sector/operator/source signature; physical admission remains with
that owner. This module neither sums amplitudes nor evaluates an FFT.

The two sides have independent alias partitions. Equal canonical tensors only
share an alias when their absolute orders agree, and every original current
column is retained. Relabellings and canonicalization use the dependency.
Every typed external port is checked before and after canonicalization,
including fixed fundamental/dual ports. A dependency failure or changed interface
leaves the tensor unproved and direct; the open-duality defect cannot provide a
false orbit certificate.

A regular orbit contains each lexicographic group element once. Incomplete and
stabilized actions remain residual. If the group order exceeds the number of
actual classes, a regular orbit is mathematically impossible, so preparation
does not enumerate that group. This permits large action domains with no
arbitrary degree or column cap.

Each metric bank separately proves equivariance for every ordered Born/loop
orbit pair. A failing pair is kept wholly direct. Every alias pair is represented
exactly once by a relative kernel or a direct entry, including both residual
cross directions. Alias metric entries must agree exactly; formal Nc comparisons
never substitute a numerical colour value. The certificate does not alter the
existing structural LC/NLC projection.

Focused controls reconstruct a complex rectangular physical overlap with
independent aliases and residuals, separate absolute-order banks, a symbolic
`Nc-3` counterexample, incomplete/stabilized/trivial/empty actions and a
12-adjoint-port domain. An actual S3 tensor action with a complex unequal loop
superposition produces noncentral kernels through the existing formal full/LC/NLC
projector. An independent explicit SU3 component sum fixes a complex
overlap, and a corrupted typed canonicalization is rejected. These are cold
mathematical controls, not public loop FFT or scheduling acceptance.

## Shared exact dual-Born schedule

`loops/dual_weights.py` prepares a single exact map from Born current columns to
complex dual weights for each native point/helicity or source row. Columns retain
the direction, virtual squared-order sector and original loop colour term. Every
allowed term remains present even when its ordinary metric happens to vanish.
Different directions, source classes and resolved Born-component masks own
separate banks; no Hermitian shortcut combines the two directed terms.

Families select explicit columns from this bank and combine them with their
existing routed numerator coefficients. Colour domains therefore remain
independent of denominator families. Neither amplitudes nor physical terms are
erased by aliasing. UV, model R2 and external-field projectors remain unchanged.

Prepared views retain only dependency evaluator bytes and column identities.
Double, DoubleFloat and arbitrary precision execute the same instructions in
their selected carrier. Weights are recomputed for each call and compacted with
their original point/source rows; no event-value cache survives parameter updates
or rescue. The optional `exact-dual-born-v1` descriptor lives in the existing
PACBIN payload. Native validation checks state shape, key uniqueness and domain,
term graph ownership, and every complete ordered family selection.

The unchanged original bilinear projector and its descriptor-free native
schedule remain the independent baseline. Focused tests compare mixed orders,
complex directed and masked component banks, zero columns, independent family
maps, changing batches and all three carriers. A retained source photon process
also compares every complete component and both integrated directed terms for
ordinary, singleton and joint sources. The saved/native checks cover transport,
weight-stage lane failures and separate higher-precision subset rescue.
An actual declared-model gluon LC owner additionally passes its six signed Born
components and 16 helicities through weighted and original saved native schedules
in all three carriers, with empty/singleton/tail rows and inclusive reconstruction.

This establishes the shared scheduling seam. Actual process certificate
attachment, rectangular Fourier execution and truthful public policy selection
remain subsequent work; the presence of a dual-weight descriptor does not imply
that an FFT was performed.
