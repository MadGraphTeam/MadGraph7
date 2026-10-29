# Local direct-R2 feasibility boundary

> Historical: this describes the per-diagram legacy one-loop engine, removed on
> 2026-10-29 (MQ-LEGACY-REMOVE; v2 is the only one-loop engine, see
> `loop_v2/DESIGN.md` and §13.3 there). The modules, tools and tests named
> here may no longer exist.

`tools/loop_reference/local_r2_oracle.py` is an executable local-kernel oracle.
It does not enable the public direct-R2 strategy, certify a complete physical
process, or add loop-induced observables. The bounded physical gluon control
below separately exercises complete model and direct strategies. Public direct
dispatch remains disabled. A direct selection must cover every contributing
graph; these experiments do not justify mixing direct and model completion
graph by graph.

## Reused local state

`CoefficientDAG` now optionally accepts extra polynomial variables and explicit
nonnegative rank weights. The ordinary four-momentum interface is unchanged.
The oracle uses `(q0, q1, q2, q3, mu, epsilon)` with weights
`(1, 1, 1, 1, 1, 0)`. Each sparse coefficient retains its tensor component and
all six powers. `mu` is a slash parameter, not the reducer's `mu2`: only after
spin closure are even powers mapped to `mu2**(power/2)`.

Every transition contracts one dependency tensor with a placeholder coefficient
current. Symbolica extracts coefficients and saves scalar evaluator instructions.
The existing Rust `CoefficientPlan` executes those instructions over native
binary64, DoubleFloat or arbitrary-precision buffers. No current is expanded
into a whole-diagram expression in that numerical path. Exact expansion occurs
only in the small independent comparison oracle. Weighted rank is structural;
relations between predecessor coefficients can lower the actual polynomial rank.

## Closed fermion-chain experiment

The tested chain is the stripped massive scalar–fermion–two-vector triangle in
both ordered orientations, with physical vector ports and one common evanescent
loop momentum. Its local propagator is represented by the dependency's explicit
four-dimensional matrices as

`slash(q4 + shift) + m + i*mu*gamma5`.

Here `gamma5` is an auxiliary matrix realization, **not** a physical axial
coupling or a D-dimensional gamma-five prescription. Dependency contractions
verify its square `-mu**2`, anticommutation with each physical gamma, the massive
propagator product, `trace(1)=4`, and unchanged evanescent part under a physical
routing shift. No matrix entries or Clifford simplifier are implemented here.

Odd-mu coefficients are retained through every open segment. At closure the
D-dimensional trace selects even evanescent parity. The component realization
otherwise produces nonzero spurious odd traces. A negative control that drops
odd currents early loses the required even rational numerator. This prescription
is checked against an independent Idenso D-dimensional trace for all sixteen
vector tensor components, both orientations, arbitrary symbolic fermion mass,
and two routings, including a nontrivial rational four-dimensional shift.
The independent trace uses `Pi_D.Pj_D = Pi_4.Pj_4 - mu**2` for all routed loop
momenta, with physical external ports. No rotational integration identity is
used pointwise.

The physical transverse projection, with the normalization of
`tools/loop_reference/diphoton_r2.json`, yields `y*mu2` per orientation. Native
coefficient buffers from both orientations are combined before NinjaRS reduction.
The massive subthreshold triangle yields finite `y`, with both poles zero.
Changing/empty batches and original lane identities are checked in all three
arithmetic modes. Inputs beyond binary64 resolution remain visible in both
DoubleFloat and arbitrary precision. This local experiment agrees with the
existing separately generated full-diagram diphoton oracle; it does not import
arbitrary UFO diagrams into direct kernels.

A second local transition uses Idenso to reduce the open word
`gamma_a gamma_b gamma^a` before materializing its remaining physical port.
The exact factor `(2-D)` survives; closure against a physical temporal gamma
produces `-8+8*epsilon` at `D=4-2*epsilon`. Its native reduction against the
massless spacelike bubble checks that the epsilon times single pole contributes
`+8` to the finite coefficient. This certifies that particular contracted kernel
and the epsilon transport/convolution, not a universal vector-loop rule set.

## Dependency audit and remaining gates

The inspected Rust snapshot is the existing pinned FeynKit
`474478a7a80bc9f9f21bead0e1cde6acbcffbe93`. Executable Python probes use the
installed Symbolica Community distribution 3.0.2, whose kernel reports 3.0.1.
These are the current development dependency selections; the existing source
publication/package-coherence gate is unchanged.

The three checks were source, public API/documentation, and regression/examples:

- Source: `crates/idenso/src/dirac/simplify.rs`,
  `DiracRuleDimension::{gamma_compatible,merge_known_gamma_dimension}`, requires
  matching known Lorentz dimensions for ordinary dimensional identities and
  explicit dimension four for four-dimensional identities. This does not encode
  orthogonal physical and evanescent Clifford subspaces.
- Public API: `crates/spynso3/src/expression.rs`, `dirac_gamma`, `gamma5`,
  `with_lorentz_dimension`, `simplify_algebra`, and `undo_chain`; the corresponding
  installed `symbolica.community.tensor` stubs document four-dimensional explicit
  matrices. `with_lorentz_dimension` promotes four-dimensional ports; it is not a
  split-space constructor. Its scalar coefficients are deliberately unchanged.
- Tests/examples: `crates/idenso/src/dirac/test/fixed_frame_regressions.rs`
  preserves symbolic `(D-4)` spectator factors. Community's
  `examples/hep/gamma_simplification.py` explicitly keeps mixed dimensions and
  D-dimensional gamma-five words unreduced. Its Higgs diphoton example/test
  establishes a full-trace evanescent oracle, not local split-space transitions.

No dedicated general `4 + (D-4)` Clifford transition/closure API was found in
those surfaces or the relevant tensor/idenso source searches. Existing
`feynkit-tensor::TensorReducer` supplies integrated tensor projection; it is not
an evanescent local spin-state representation and is not substituted for NinjaRS.
The restricted prototype therefore reuses existing supported matrices, traces,
polynomial operations and evaluators; it introduces no replacement algebra.

Before production direct completion, the model compiler must certify source
fermion templates, signs, gauge, dimensional spin conventions and complete graph
coverage. Open fermion chains, physical chiral/projector vertices, arbitrary
D-vector/ghost contractions, multiple independent evanescent structures and
anomaly-sensitive traces remain outside this certificate. There is no universal
FDH-to-HV correction inferred from these results. General split-space support,
if required for those domains, is an explicit dependency/API design gate rather
than a private algebra implementation. No dependency was modified by this work.

## Graph-derived closed vectorlike cycles

`ClosedVectorlikeFermionKernels` now plugs into the ordinary dependency-owned
open-loop traversal. Before materialization it verifies each cycle propagator
against its source `normalization*(slash(P)+m)` and each cycle vertex against a
scalar identity or one physical vector gamma. Directed opposite fermion flows,
unit loop signatures and the common sign of the native affine routing are
checked explicitly. The extra slash carries both the source momentum signature
and the routing sign. Physical momentum variables remain separate from auxiliary
polynomial variables when constructing exact denominators and S matrices.

Exact checks use actual massive-quark two-, three- and four-gluon cycles from the
QCD model's four-gluon graph set, including their off-shell tree attachments,
both oriented words, alternative bubble cuts and canonical family routing.
An independent oracle contracts the unmodified source graph in D dimensions
before selecting physical external ports. All six even closed polynomials agree;
the box contains nonzero mu-four terms. Local odd factors are checked before
closure, since the complete trace can already be even and hide a wrong common
routing sign. The construction accepts longer certified chains within the
reducer's domain; this graph-derived test set currently covers lengths two to
four.

The bubble and box are also reduced through the native local coefficient DAG
and NinjaRS in double, DoubleFloat and 80-digit arbitrary arithmetic. An
independently compiled full-D-trace polynomial supplies the coefficient oracle;
both feed the same reducer, so this checks the graph-to-reducer boundary rather
than supplying an independent master-integral oracle. Real and complex fermion
masses, canonical family ordering, exact S matrices, original lane IDs and
batch sizes 1, 5, 0, 33, 1 are covered. All pole coefficients vanish for these
positive-mu rational numerators and their nonzero finite parts agree. These
checks use NinjaRS 5f35b07 and oneloopmaster 264a6ce; the separate analytic scalar
anchors and physical Born-interference controls retain their own validation
responsibilities.

This provider remains an internal graph-numerator prototype. It does not certify
physical axial vertices, open fermion chains, internal D-vector/ghost states or
whole-process direct completion, and does not mix direct and effective-rule
completion within a process. Tree attachments retain the supported physical
four-dimensional source rules. Source-template checks do not by themselves prove
all possible model conventions or gauge sectors.

## Massless Yang--Mills and ghost state experiment

`DimensionalYangMillsKernels` supplies a second bounded graph provider. It
continues the source covariant metric propagators and momentum/metric cubic
and quartic vertices using dependency tensors. The accepted graph has one pure
massless vector or ghost cycle and physical massless-vector tree attachments.
Longitudinal propagator additions, higher-derivative vertices, massive particles,
mixed spin cycles and unexpected source ports are rejected. These are algebraic
source-template checks, not a claim of gauge invariance for an arbitrary model.

The local state has six explicit coordinates: four physical directions, the
common extra loop-momentum direction, and one transverse spectator. Every source
momentum has fifth component `a_i * routing_sign * mu` and zero sixth component;
physical external states and tree momenta have both extra components zero.
The source rule itself is promoted with `with_lorentz_dimension(6)`, so all
physical/extra transitions and their metric signs come from the dependency.
Odd-mu transitions survive every open current. A negative control that projects
each local rule to even mu loses nonzero closed even terms.

The native rank-one spectator projector is normalized by its own Minkowski
norm. Exactly one chosen cycle propagator receives `(D-6)*P_s`, at
`D=4-2*epsilon`. Its six-state metric trace is therefore D, and the spectator
weight is `D-5 = -1-2*epsilon`. Dependency component contractions show that
physical/aligned states do not mix with the spectator, that distinct spectator
states do not mix, and that their diagonal transitions agree. These checks
include arbitrary symbolic physical attachments and each local colour term of
an actual quartic QCD source vertex; they compare six and seven component
libraries. The accepted metric/momentum templates have no further transverse
structure. Changing the weighted edge and the cut independently preserves the
closed numerator. Independently generated unweighted five- and six-dimensional
recurrences also satisfy `N(D) = N5 + (D-5)*(N6-N5)` on the tested graphs.

Actual vector bubble/triangle and ghost bubble/triangle graphs with off-shell
tree attachments agree exactly with an independent raw-source D-dimensional
contraction before physical materialization. The vector results retain nonzero
mu-dependent and explicit epsilon terms. The ghost cycles retain the native
source statistics/global factors and agree with their four-dimensional
numerators: a physical attachment has no extra component for its ghost momentum
vertex to contract. This does not omit their ordinary four-dimensional bare
contribution or its denominator-induced rational terms.

For this domain, the epsilon-free, mu-free closure equals the original
four-dimensional source numerator. `rational_sector_columns` selects the
remaining closed even-mu and epsilon columns. The vector bubble and triangle
then pass through the native coefficient scheduler, exact S-matrix binding,
NinjaRS numerator sectors and Laurent convolution in double, DoubleFloat and
80-digit arbitrary arithmetic. An independently compiled full-D polynomial
provides the coefficient oracle; both use the same reduction library. Native
batch sizes 1, 5, 0, 33, 1 preserve row identity, and coupling differences below
binary64 resolution remain visible in the reduced result. All three Laurent
coefficients are compared. These checks do not introduce a second reducer or a
process-level model-R2 comparison.

The dependency inventory for this construction was checked at three surfaces:

- Source: `crates/spynso3/src/structure.rs` exposes `Representation::mink` with
  signature `(+,-,...,-)`; `expression.rs::with_lorentz_dimension` promotes
  source index spaces before contraction. `crates/spenso/src/network/library/`
  supplies dimension-dependent metrics and native tensor materialization.
- Public Python API/documentation: `Representation.mink`,
  `TensorExpression.with_lorentz_dimension`, `TensorLibrary` signatures,
  `Tensor.dense`, `Tensor.dot` and component access support the entire local
  construction. The projector's norm and all contractions use these APIs.
- Existing regressions/examples: `fixed_basis_products_reject_unproven_or_incompatible_components`
  and the adjacent larger-space metric checks in
  `crates/spenso/src/network/library/symbolic.rs`, plus
  `crates/feynkit-py/tests/installed_metric_factory.py` and
  `installed_dimensional_spin_sums.py`, establish arbitrary-dimension metric
  behavior and symbolic-dimensional traces. No dedicated physical/evanescent
  local gauge-cycle constructor was found in these source/API/example searches;
  the prototype composes the existing metric, projector and routing operations.

These first integrated graph checks cover cubic vector/ghost bubbles and
triangles; the complete retained gluon-graph coverage below extends them.
Massive gauge and Goldstone sectors, physical chiral vertices, open fermion
chains and mixed spin cycles remain outside this provider. No universal
scheme-conversion formula is inferred, and no dependency source was modified.

## Complete retained gluon-graph kernel coverage

`direct_kernels.dimensional_kernels(graph)` selects the closed vectorlike
fermion or massless Yang--Mills/ghost provider. Binding still checks every local
source factor. Unsupported or mixed cycles fail explicitly; this dispatch does
not establish eligibility for a complete process.

The actual `qcd_qed` source selection for `g g -> g g`, Born order QCD=2 and
loop order QCD=4 with QED=0, retains 186 loops and 258 Lorentz/colour terms:

| Cycle | Bubbles | Triangles | Boxes |
| --- | ---: | ---: | ---: |
| Closed quarks, all six source flavours | 18 | 72 | 36 |
| Ghosts | 3 | 12 | 6 |
| Vectors, no quartic vertex | 3 | 6 | 3 |
| Vectors, one quartic vertex | 18 | 6 | 0 |
| Vectors, two quartic vertices | 3 | 0 | 0 |

Every retained graph builds with the selected dimensional provider. The largest
structural weighted numerator rank is four. No tadpole remains in this source
selection; this observation does not certify tadpoles in other processes.
At an exact physical momentum control with general external vector components,
all auxiliary-free polynomial coefficients agree exactly with ordinary
four-dimensional recurrences, retaining each separate native colour tensor.
The comparison covers all 186 graphs, including both fermion orientations and
massless and massive source quarks. It is a full coefficient comparison at that
kinematic control, not a proof over all possible phase space.

Independent full-D source contractions additionally cover a one-quartic bubble,
a two-quartic bubble, a quartic triangle, a vector box and a ghost box. Each uses
two nonzero exact external-colour projections. This small oracle expands and
contracts the original unsplit source graph with dependency operations; it
does not substitute recurrence instructions. Both the dimensional trace and
the local recurrence retain the entire loop-momentum polynomial. The two-quartic
bubble is also invariant under independent choices of cut and weighted edge,
so its dimensional multiplicity enters exactly once.

`closed_reduction_powers(recurrence, node=...)` is the shared reducer boundary.
It maps scalar closed output labels to `(q0,q1,q2,q3,mu2,epsilon)` powers and
allows scalar descendants such as a tree-denominator multiplication. For a
closed vectorlike fermion, `None` explicitly denotes the proven odd auxiliary
gamma5-trace projection; such nonzero auxiliary terms are expected and must not
be rejected as if they were physical evanescent traces. No open current may be
projected. Gauge/ghost closures must instead have no odd-mu labels and reject
them. Fermion epsilon powers and more than one gauge dimensional-trace factor
also reject. The auxiliary-free columns give N4; the remaining accepted columns
give the direct dimensional remainder before reduction and Laurent convolution.

This kernel coverage does not itself enable public direct generation. Complete
Born interference, model-R2 agreement in the same HV convention, exclusive
process-wide completion, precision and performance remain responsibilities of
the physical assembler and its independent controls.

## Complete physical gluon diagnostic

`gluon_scattering_model_virtual_program(..., r2_strategy="direct")` is an
internal full-SU(3) control in the same HV convention as its source-model path.
Every selected loop uses its dimensional provider; an unsupported graph fails
construction. One closed coefficient DAG per graph retains the common
`(q[4],mu,epsilon)` frame. Born and exact denominator kinematics use only its
four physical momentum variables.

The native family projector applies complex Born weights and sums compatible
helicities and graph terms before reduction. Its canonical columns are
`(q[4],mu2,epsilon)`. The auxiliary-free columns form N4; positive auxiliary
columns form the dimensional remainder. Epsilon has zero tensor-rank weight
and selects one of the reducer's three epsilon sectors. The diagnostic reduces
N4 and the remainder separately from the shared native coefficient buffer,
never by subtracting integrated amplitudes. It exposes the resulting bare and
R2 components separately. A future complete-only schedule may combine their
sectors in a single reduction without changing the numerator construction.

Direct selection excludes every model R2 insertion. Source local UV and
external-field half-Z factors are retained unchanged, including per-point
renormalization scales and the six-internal-flavour/five-renormalized-flavour
convention. The selection is fixed for the complete call. Model selection
continues to use the ordinary four-dimensional program and source R2 trees.

The physical regression checks all three reference points against independently
instrumented MadLoop bare, R2, local UV, external-field and complete values.
It also compares the direct remainder with independently evaluated exact source
R2 trees in double, DoubleFloat and 50-digit arbitrary arithmetic. The complete
remainder's poles are checked after summing all families; individual family
poles are not discarded. Native empty batches and failure guards remain in
place. Reference finite-amplitude comparisons themselves remain binary64
controls; the same-carrier R2 comparison is a separate precision check.

The binary64 diagnostic also accepts `reduction_engine="fast-tensor"` using
NinjaRS b5b93be's dense μ² numerator. Owner coefficient/layout APIs feed each
existing HV epsilon sector without a scheme transformation. The complete
gluon component and independent source-R2 controls also pass with this option.
The original `fast` callback remains available; a sparse rank-five box timing
did not show a speedup, so no automatic selection or performance claim follows.
DoubleFloat/arbitrary arithmetic retain the generic reducer. Nonfinite dense
coefficient-head errors remain batch errors and exclude this engine from the
optional partial/rescue view.

The bounded direct gluon factory also accepts formal full/LC/NLC colour. Its
UV scalar grades and rectangular overlaps are projected together, with one
virtual degree P=5 across N4, the remainder, local UV and external fields; Born
uses P=4. This does not infer a continuation of the excluded model R2 quartic
rules. Other mixed-grade multi-term CTs require a separate incidence proof.

These controls do not enable public direct eligibility, general chiral or
massive-gauge continuation, or a performance selection policy. Ordinary
diagnostic evaluation still raises for a reduction failure in either channel;
the optional bounded partial/rescue view is described below.


The internal full-colour QCD four-gluon model wrapper supports optional sparse
verification/rescue with direct R2 fixed across every attempt. N4 and Delta
reduction failures are collected before a common whole-point gather, preserving
original IDs and previously integrated families. Both final sums are gathered
together. Retried coefficient DAGs preserve physical/evanescent sectors and
explicit epsilon powers; model R2 is never substituted during a direct retry.
The native result records its executed strategy. Cross-precision agreement is
empirical evidence under the configured bounds, including requested-carrier
rounding, not independent certification of the reducer. This scope does not
include source-owned direct kernels, CMS, formal colour or correlated views.
