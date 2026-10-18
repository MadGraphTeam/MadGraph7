# Local direct-R2 feasibility boundary

`tools/loop_reference/local_r2_oracle.py` is an executable local-kernel oracle.
It does not enable the public direct-R2 strategy, certify a complete physical
process, or add loop-induced observables. Model-rule completion remains the
available physical path. A future direct selection must cover every contributing
graph in the requested process; this experiment does not justify mixing direct
and model-rule completion graph by graph.

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
