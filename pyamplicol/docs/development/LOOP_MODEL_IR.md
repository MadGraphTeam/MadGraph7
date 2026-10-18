# Exact loop-model preparation

`pyamplicol.loops.model.build_loop_model(raw_sidecar)` prepares exact scalar
definitions and individual tree/UV/R2 colour × Lorentz × loop-group entries.
It consumes the existing raw UFO sidecar, including its exact-source contract;
it does not import a second source manifest. Scalar definitions are indexed by
`(collection, raw_object_id)`. Vertex display names need not be unique.

`parse_exact_ufo` uses standard Python AST/tokenization to validate arithmetic
syntax and retain numeric lexemes. `Fraction` converts decimals into exact
rationals before the existing UFO parser runs. The parser and Symbolica own
complex arithmetic, model-function calls and expression transformations. No
Python expression evaluation or binary64 conversion occurs on this path.
Unknown model functions retain their raw definitions and remain symbolic;
arbitrary Python helper bodies are not interpreted.

`ScalarDefinition.value` is a `LaurentExpression`. CT dependencies, including
ordinary parameter or coupling aliases containing CT/Laurent definitions, are
substituted before requesting a coefficient. The full expression remains
available. `series(maximum_power)` returns Symbolica's native series and its
remainder order; `coefficient(power)` extracts a native exact coefficient.
For example, an O(epsilon) term multiplying a pole is retained in the finite
part. Ordinary epsilon-independent parameter definitions remain separate,
so input values and formal colour constants are not prematurely substituted.

`CovariantKernelCatalogue(ir, structural_model).lower(entry)` produces separate
native `TensorExpression` objects for colour and Lorentz structure. The
structural input is a native HepKit model or its public normalized JSON mapping.
Only species and propagator metadata are reused. Cached numerical scalar values
from the normalized model are never used for the entry's coefficient. Each
component is localized by a public HepKit single-vertex diagram with unit scalar
coupling and the other tensor domain set to one. Component results are cached.
This reuses native UFO index, momentum, gamma and colour lowering directly.

The local momenta are all incoming native `q_edge` symbols. `leg_half_edges`
identifies the UFO leg order. Each tensor retains its native axis order, which
can differ: `Metric(1,4)*Metric(2,3)` has axes `(0,3,1,2)`. Consumers must use the
native slots/index labels and native axis permutation/rebinding operations.
The coefficient stays separate from both tensors.

The initial `loop_sm` controls cover every pure-QCD component, all 57 active
four-gluon R2 basis entries, and two-point UV mass insertions. A synthetic
one-point UV rule checks native scalar lowering without extending process
support. A mixed QCD/QED quark-photon R2 control checks the exact model formula.
Run these controls with `PYAMPLICOL_NLO_UFO_REFERENCE` pointing to `loop_sm` and
a licensed Symbolica runtime. No reference files are changed.

This preparation does not certify a scheme, UV/R2 completeness, arbitrary
precision physics, or formal-Nc continuation. Loop-group alternatives and
particle counterterm metadata are retained; process-dependent matching,
external-wavefunction insertions, restrictions and scalar-kernel compilation
remain separate work. The active four-gluon `loop_sm` R2 coefficients contain
fixed SU(3) constants. Preserving them does not reconstruct their Nc dependence.
`FormalColorIdentities` states explicit SU(Nc) CA/CF/TR/TF/Ncol substitutions
with fixed Nf, but never applies them automatically or certifies LC/NLC.
Legacy inconsistent non-QCD Lorentz spin signatures fail when requested.

## Local colour alternatives

`loops.color_terms.split_local_color_terms` expands one native local factor and
retains each independent colour tensor. The existing single-colour separation
proof classifies the local monomials. `model_basis_terms` instead consumes the
unsummed UFO entries directly and extracts an explicitly selected epsilon
coefficient. It does not select eligible loop groups; its caller must supply
those entries after process matching.

`ColorTermDAG` wraps the coefficient DAG. An open `ColorCurrent` carries
`(coefficient_node, native_colour_tensor)` alternatives. Only local basis choices
branch, while previous Lorentz currents remain node references. At a named
site, identical local tensors and dependency nodes reuse a coefficient
instruction. Alternatives sharing that instruction add their colour factors
using native tensor arithmetic, preserving multiplicity. Distinct coefficient
instructions remain separate, with visible branch counts. There is no global
Lorentz expansion, trace, colour approximation or inferred symmetry factor.

This primitive covers the physical four-gluon rule's three independent local
colour tensors and the model's selected four-gluon R2 basis. Its tests include
a two-site, nine-branch coefficient calculation against an independent native
local contraction, exact scalar/node reuse, and R2 tensor-DAG construction.
Graph insertion/rebinding, physical external states, process-level colour
contraction and branch-growth optimization are separate integration steps.
Local catalogue frames must be rebound before two distinct sites are combined;
reusing their original half-edge labels would incorrectly identify indices.

## Native counterterm process matching

`loops.counterterms.prepare_counterterm_model` appends homogeneous CT rules to
the ordinary native model with a private insertion order. Exact Laurent scalar
expressions remain unevaluated; aliases use the native coupling symbols, including
their namespaces. The returned model must generate both the ordinary catalogue
and CT candidates. `prepared.process(..., counterterms=False)` restricts generation
to ordinary rules; use `build_loop_process(..., ordinary_model=prepared.ordinary_model)`
so effective CT interactions cannot change perturbative species membership.

`match_counterterms` generates zero-loop trees with exactly one tagged CT. It
compares their attached currents to actual admitted loops, matching the sum of
orders on the cycle vertices rather than the total amplitude order. R2 and
UVmass alternatives match the unique positive loop species; empty alternatives
are wildcards. UVloop alternatives impose forbidden-species restrictions only.
UVtree uses tagged tree generation without a loop witness. Alternative indices
are retained: repeated identical alternatives deliberately repeat the insertion.
The legacy UFO type `UV` is classified with MadLoop 3.7's import rule: an identical
two-particle pair is UVmass, otherwise UVloop. The reference is
`models/import_ufo.py:1571–1655` and
`madgraph/loop/loop_diagram_generation.py:1210–1525` in the supplied MG5 source.

HepKit finds components and excises each attached current. Excision's artificial
boundary can be incoming or outgoing depending on the original internal edge
orientation, so its raw diagram ID is insufficient. Public `Symbolica.Graph`
canonicalizes a metadata projection retaining native interaction IDs, interaction
port slots, particle identities and directed flow, plus all physical external
labels and states. Only the artificial boundary state is normalized. The native
model fingerprint is checked before using these current keys. Tests distinguish
identical external support with different internal rules and external label
assignments. No canonical-labeling, traversal or graph-generation algorithm is
implemented in pyAmpliCol.

The counterterm inherits its freshly generated tree weight, never the sign or
symmetry denominator of a loop witness. Multiple crossed loops can witness the
same insertion without multiplying its weight. Bivalent CT rules require a
finite native generation bound: with one CT of valence at least two and ordinary
vertices of valence at least three, `max_vertices = n_external - 1` is exact.
Ordinary one/two-point rules and CT tadpoles fail explicitly at this boundary.

Eight native controls cover Drell–Yan, crossed fermion boxes, repeated/wildcard
alternatives, UVtree versus UVloop, joint correction selection and current
identity. The full compatible `loop_sm` subset selects one `R2_UUA` and no UV
insertion for photon-only QCD Drell–Yan. Source CT tensors are still uncalibrated:
the bundled native model uses a different FFV spinor-slot convention from the
standard loop UFO. Matching graphs does not validate that convention, supply
external-leg wavefunction CTs, certify a renormalization scheme, or certify
formal-Nc continuation. Missing species of a restricted ordinary model remain
explicitly recorded as unavailable basis entries.

`OrdinaryBasisRepresentative` provides a narrow, explicit convention bridge.
Its source ordinary basis key, native rule and colour/Lorentz slot identify the
representative; its required calibration label records the caller's prior
physical control. Preparation checks identical ordered signed PDGs, spins and
colour representations, exact ordinary scalar and coupling order, and exact
colour structure. A CT can reuse that representative only when its source
Lorentz and colour expressions equal the source ordinary basis exactly. The
counterterm's Laurent coefficient is retained. Native model symbols are used
for substitutions, and approximate numeric atoms in representatives are rejected.

This does not infer any Lorentz convention from names or transpose a gamma
matrix. The first explicit mapping is the source `u~ u A` vector current to
the bundled native `V_132` basis `(0,0)`. Each CT rule records its proven source
identity and whether all entries have a representative. Unmatched structures
remain uncalibrated; chiral, Sigma and new two/four-point structures acquire no
certificate from this mapping. A calibration label alone is not a physical
validation result, and the whole-model convention flag remains false.

## Native fundamental colour traces

The current HepKit UFO dispatcher does not lower `Tr`: passing it through would
leave a rank-zero scalar with unresolved colour indices. `model_color` prepares
that head through public `tensor.trace`, `TensorExpression.color_t`, and typed
HepKit UFO indices before native graph localization. The original sidecar colour
expression stays unchanged. Both local catalogue lowering and augmented CT model
construction use this boundary. Unknown UFO colour function heads fail explicitly.

The initial trace envelope is SU(3) with distinct positive adjoint external-leg
indices; negative dummy indices and other representations are unsupported.
Native algebra and component tests establish actual adjoint ports,
`Tr(Ta Tb) = delta_ab/2`, cyclic invariance, and the noncommuting three-generator
identity `Tr(Ta Tb Tc) - Tr(Tb Ta Tc) = i f_abc/2`. No private trace or colour
algebra is implemented, and this does not establish general-Nc model validity.

`tools/loop_reference/ct_tensor_oracle.py` independently compares the three
`loop_sm` gluon two-point R2 Lorentz structures with generated ALOHA routines:
`p^2 g`, `p_mu p_nu`, and `g`. It checks every polarization basis pair and a
complex polarization pair at real and complex off-shell momenta, removing the
explicit ALOHA `-i*COUP` convention. Native arithmetic is exact; ALOHA is binary64.
The report records source hashes and 102 comparisons. This local check does not
validate CT scalar values, a whole process, chiral tensors or arbitrary-precision
physics. It supplies a separate primitive control rather than extrapolating the
ordinary FFV convention bridge to new structures.
