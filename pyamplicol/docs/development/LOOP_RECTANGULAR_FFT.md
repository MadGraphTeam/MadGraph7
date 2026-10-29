# Rectangular Born–loop colour contraction

> Historical: this describes the per-diagram legacy one-loop engine, removed on
> 2026-10-29 (MQ-LEGACY-REMOVE; v2 is the only one-loop engine, see
> `loop_v2/DESIGN.md` and §13.3 there). The modules, tools and tests named
> here may no longer exist.

Certified ordinary and resolved NLC/full-colour loop artifacts now connect the
existing symmetric-group transform to the shared dual-Born weight schedule.
Binary64 uses the certified rectangular transform plus all direct residuals;
DoubleFloat and arbitrary precision execute the original exact direct map.
UV, model R2, external fields and Born-square contractions remain complete and
direct. LC and correlated/source FFT composition remain separate feature gates.
This is feature acceptance, with no new speed or benchmark claim.

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

## Shared runtime boundary

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
plans, precision rescue and public policy use the same operation and report the
actual selected bank, as described below.

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

The exact schedule remains the high-precision implementation and numerical
oracle for the rectangular operation. A dual-weight descriptor alone does not
imply FFT execution: its optional certified rectangle and the actual arithmetic
select the operation.

## Formal metric ownership and actual process certificates

The existing common-degree colour projection now retains its exact Born and
Born–loop matrices before substituting Nc=3 or applying external averaging.
One immutable `FormalColorMetrics` record owns the actual tensor columns,
absolute orders, allowed pairings, Born/virtual sector degrees, accuracy and
metric revision. The virtual degree is still selected jointly with UV, model R2
and external-field terms. Prepared Python views retain this same record; it is
not another artifact manifest or a reconstruction from numerical colour values.

Only `project_physical_color` attaches the record it actually computes. Ordinary
`set_color_metrics` always clears formal ownership and cannot accept a supplied
record, even if that record would have identical values at Nc=3. Certification
checks tensor identity, order domains, normalization, degree and revision before
reuse. An arbitrary added factor proportional to Nc−3 cannot acquire a proof
through the numerical replacement boundary.

`prepare_certified_dual_born_weights` is an internal cold factory for ordinary
and signed resolved banks. It takes explicit physical adjoint labels, masks the
retained symbolic rectangle by actual orders and components, and invokes the
existing dependency-backed tensor certificate. Independent Born/loop aliases,
all original columns and both residual cross directions survive. Family routing
does not enter the certificate. Binding Nc=3 and averaging happens afterward;
the saved exact evaluator remains identical to the direct weight schedule.
Directed inserted operators need their own formal bank and are not inferred
from this ordinary rectangle.

Full/LC/NLC controls verify exact reconstruction, mixed orders, complex entries,
aliases, component masks and owner invalidation. A retained physical gluon LC
producer supplies an actual order-two action on six Born and 258 loop columns:
two regular Born orbits, 45 regular loop orbits, 90 ordinary kernels and 1,062
residual alias pairs. Its six component banks reconstruct exactly as well.
The producer predates retention, so this single probe rebuilds symbolic SUN
matrices from its actual tensors using its already-proved common degree P=5;
the Nc=3 comparison is only a binding consistency check. No production path
attempts to recover missing formal data. This is a cold feature certificate,
not a native Fourier execution or speed result.


## Public selection, saved execution and diagnostics

The existing `[color] contraction` setting accepts `direct`, `auto` and
`symmetric-group-fft`. For ordinary/resolved full or NLC requests, cold generation
searches actual compatible adjoint-port actions and requires at least one
nonzero regular kernel after the declared projection and normalization. Born
and loop orbits remain independent. A wholly residual or zero-kernel rectangle
cannot satisfy a forced request. An explicitly requested unsupported loop basis
is rejected; tree DDM and trace certificates are not reused.

Full-colour `auto` may continue directly when no formal continuation is declared,
when the declaration explicitly lacks a selected kernel/field capability, or when
no eligible action is proved. Missing selected coverage is a typed capability
outcome. A fallback recompiles from the untouched prepared model with identical
physical inputs; it cannot reuse partly graded tensors or counterterms. Malformed
declarations, mismatched owners and algebra errors remain hard failures. LC/NLC
still require their existing complete formal continuation. LC `auto` remains
direct and retains the originally requested `auto` in diagnostics. Forced LC FFT
remains unsupported.

Declared correlations retain their existing literal direct path in this slice;
forced correlated FFT is rejected. No ordinary certificate is copied onto an
unproved inserted-operator or spin-source bank. A future composition must prove
its actual bank and both interference directions separately.

The existing `exact-dual-born-v1` record optionally carries a typed
`rectangular-regular-orbits-v1` descriptor and a Symbolica exact constant-state
reference. It stores actual absolute-order aliases, independent regular orbits,
ordered direction/sector banks, kernels and all residual pairs. Core validation
checks exact one-time coverage and shapes before binding. Normalization applies
Nc=3 and the existing initial average exactly once after formal projection.
Repeated resolved exports keep their local helicity projector separate from the
three ordinary model evaluator roles even after prepared-cache invalidation.

Prepared Python and saved native consumers bind the same `DualBornWeights`
operation. Binary64 evaluates constant kernels in its own arithmetic and uses
the existing native transform/inverse workspaces. DD/AP never round these
constants through binary64: they bind the original exact evaluator. Empty rows,
changing batch sizes, opaque original IDs and parameter updates preserve the
normal execution semantics.

`Runtime.inspect()["color_contraction"]` reports the operation for each available
ordinary, resolved, directed and source bank in the handle's actual arithmetic.
Python lane diagnostics and standalone SDK JSON name the initial choice
`primary_color_contraction`; each verification attempt separately exposes its
`color_contraction`. Thus a binary64 FFT primary followed by exact AP rescue is
explicitly distinguishable from a rescued result computed through FFT. A forced
eligible artifact permits this documented exact high-precision fallback.

The actual retained four-gluon controls have six Born and 258 loop columns.
The selected NLC S2 action has two/45 regular Born/loop orbits, 90 kernels and
1,062 residual alias pairs. The independently prepared full-colour action has
two/54 regular orbits, 108 kernels and 360 residual pairs. Their exact symbolic
rectangles and direct weight states agree before execution; full-colour source
preparation uses the authorized corrected Community package and does not imply
that every publicly published package already contains that dependency fix.

The executable controls compare every complete Born, bare, UV, R2, external-field,
order and Laurent component, both ordinary and resolved layouts, all three
carriers, exact saved weights, changing row counts and original IDs. Independent
noncentral S3 controls exercise complex aliasing, multiple direction/component
banks and mixed-order scatter. Native malformed-coverage checks, repeated-export
admission, graph-free Python binding, public artifact loading and standalone SDK
output cover the transport boundaries. Subset scale-underflow rescue checks the
primary FFT metadata and each exact AP attempt. Existing performance table
measurements retain their original snapshot identities; this schedule is not
retimed by these correctness controls.
