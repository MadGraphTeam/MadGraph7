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

An independently controlled new primitive may be supplied explicitly through
`CountertermPrimitiveValidation` and `prepare_counterterm_model`'s
`primitive_validations` argument. The immutable record binds the exact source
basis key, colour/Lorentz expressions, ordered PDGs/spins/colour representations
and named conformance control. Preparation rejects stale expressions, wrong ports,
missing entries, duplicate evidence and empty provenance. It retains the original
tensor and exact scalar; it cannot authorize another basis entry by name alone.
`CountertermRule.has_validated_tensor_basis` requires every entry to have either
an ordinary correspondence or this independent primitive evidence. The record
does not run a comparison, validate scalar renormalization or set whole-model
validation flags. Supplying it is an explicit assertion that the named local
control has already been performed for precisely these structures and ports.

## External-field factors

`loops.wavefunction` reads `particle.counterterm` independently of effective
vertex insertion. One key contains a power for each coupling order in the UFO's
original registry order, followed by the declared loop-group index. Exactly one
unit power selects a one-loop field correction with two additional fundamental
coupling powers. Higher-loop entries are skipped. Laurent arithmetic and
CT-dependent alias substitution reuse the existing native scalar lowering.
Particle-field values currently require exact expression strings or integers;
unobserved float/complex attributes fail rather than acquire false exactness.

`match_external_field_counterterms` retains each already-generated Born graph
and records its external-leg index, selected model factor and loop-alternative
index. It uses the positive-PDG particle definition for the antiparticle too.
Each allowed alternative contributes separately, including repeated alternatives;
an empty alternative is a wildcard. The caller supplies the complete resolved
forbidden-species set and, when needed, an explicit allowed amplitude-order
envelope. This stream does not infer those filters from bare-loop order
differences or require a loop witness. It creates no new graph or Lorentz tensor.

MadLoop's `import_ufo.py:1355–1383` and
`loop_diagram_generation.py:1290–1332` define this convention. In `loop_sm`,
`tWcft_UV` already includes the field factor delta-Z/2: the two external top legs
contribute twice that model coefficient to the Born amplitude. No extra half,
loop integration factor or loop symmetry factor is applied. Physical contraction
then uses the usual `2 Re(Born* CT)`. The separate field IR leaves scheme
validation false; vertex CT matching's external-field-completion flag is not
silently changed by constructing it.

The executable `tools/loop_reference/wavefunction_oracle.py` derives these two
top factors from the source model and evaluates all three Laurent coefficients
in a native batch against isolated MadLoop external-field output. It uses the
reference Born values and binary64 comparison, so it validates field scalar
normalization and multiplicity, not a full amplitude or arbitrary-precision
physics. Full native physical assembly is a separate integration check.

## Physical fixed-Nf colour controls

The internal DY, distinct-flavour four-quark and massive top-pair factories
accept `color_accuracy="full"`, `"lc"` or `"nlc"`. Their default full-colour
path remains the explicit SU(3) component oracle; `formal_color=True` selects
the independent formal full-colour calculation. The direct-R2 gluon factory
also supports these projections with six Born colour terms and rectangular
Born/CT matrices. One admitted squared-order sector is still required. These
controls do not certify arbitrary models or the four-gluon effective model
R2 rules' general-Nc continuation.

`physical_color` contracts semantic native colour tensors with `SUNColorSpace`.
Source `Ncol`, `CA`, `CF` and `TF` are bound explicitly to their exact SU(Nc)
expressions. Numeric constants are never guessed to be Casimirs. Each CT alias
coefficient is first split into exact Nc grades. The shared native CT executor
evaluates the colour-free aliases for each grade, while projection acts on
Nc^grade times the complete Born-term by CT-term overlap matrix. A single
projected weight is never absorbed into an alias shared by multiple terms.
Each field grade uses its corresponding columns of the full Born Gram matrix.
All bare, local UV, R2 and external-field rows share one structural maximum degree across
the finite and both pole outputs, before numerical model values or helicities
are evaluated. Born squared has its own maximum degree. Projection retains
degree P for LC and every degree at least P-2 for NLC, with Nf fixed. Nc=3 and the
physical initial-state average are applied only after projection. Result
metadata records both degrees.

The four-quark Born degree is 2 and its virtual degree is 3: the leading Born
colour factor is proportional to Nc^2-1, while the virtual CF factor adds one
power. Fixed-Nf quark-loop and heavy-quark decoupling terms can have degree 2,
so NLC retains this odd suppression as well as degree 1. For DY the respective
degrees are 1 and 2. Exact scalar algebra combines terms within each row;
kinematically distinct currents remain distinct without a proven identity.
The gluon Born degree is 4 and its virtual degree is 5. Closed-quark rows have
degrees 4 and 2; gauge/ghost rows have degrees 5 and 3. Source gauge UV factors
carry explicit CA (grade 1), while quark UV and external fields carry TF
(grade 0). Consequently the heavy-quark degree-4 terms vanish at LC and survive
NLC, which keeps every degree at least 3. The full, LC and NLC double poles
are checked against the four-adjoint universal factor.

Multiple grades remain supported for single-colour-term currents, including
the existing CF controls. A multi-term CT currently requires one uniform
nonzero grade across its aliases. Other grade-to-term patterns are rejected until an
exact incidence proof identifies inactive terms before choosing the common P.

Native `if` branches with Nc-independent conditions are collected separately
with the dependency's Laurent coefficient operation and recombined with the
same runtime condition for every coefficient. This retains the model's zero-
and nonzero-mass branches without fixing a numerical mass during generation.
Nc-dependent conditions and opaque functional Nc dependence fail explicitly.
The saved native evaluators run these branches in double, DoubleFloat and
arbitrary precision. Formal projection owns the bare program's weight
evaluators; another model wrapper cannot mutate or reuse a projected bare
program. Construct a fresh bare program for each projection.

## Four-gluon primitive and multicolour assembly boundary

The independent `four_gluon_oracle.py` control covers the active `V_R2RGA` rule:
four Lorentz bases (1,040 ALOHA comparisons), twelve colour bases plus two
complex trace probes (196 exact overlaps against MadGraph), and all 57 source
basis keys. Exact native SU(3) component contractions are used for the colour
oracle: some products of compact symmetric traces remain unevaluated by the
current symbolic colour simplifier. Complex probes retain the relative `d*f`
orientation, while the full Lorentz component basis includes nontransverse
off-shell currents. The named conformance scope is
`local-four-gluon-r2-aloha-and-mg-color-control` and can be attached to those
exact source tensors/ports with `CountertermPrimitiveValidation`.

Only the active source rule is tested. This neither reactivates the commented
alternate four-gluon rule nor supplies its missing general-Nc continuation.
Scalar R2 coefficients, gauge cancellation and complete physical interference
still require process-level comparisons.

Physical assembly must retain `ColorCurrent`'s `(node, colour)` alternatives
through both Born and loop closure. Native coefficient execution can expose
the unique terminal nodes as multiple outputs, with a rectangular Born-term
by loop-term colour matrix. Shared physical external vectors multiply every
local Lorentz alternative in the same native half-edge frame. Graph signs and
symmetry factors enter each closed term once; branching introduces no extra
graph multiplicity. Contracting or rebinding local tensors by ordinal axis
position instead of their native port labels would mix distinct tensor bases.


`loops.graph_basis` now supplies those preparation interfaces. The
`build_color_tree_recurrence` result includes every tree propagator denominator;
`build_color_loop_recurrence` keeps attached tree denominators explicit for
later Born weighting. Both expose `terms`, `final_nodes`, `color_expressions`,
the common coefficient `dag`, exact momentum symbols and shared independent
external spin components. Feed `final_nodes` to `NativeCoefficientDAG` to execute
all alternatives in one native batch. The original single-colour
`build_open_loop_recurrence` delegates to this same traversal and rejects a
multi-term result explicitly.

Optional local kernel providers bind to one graph/routing, validate each raw
Lorentz term, declare auxiliary polynomial variables/weights, and materialize
individual local tensors. They cannot replace colour factors or graph traversal.
The graph's overall sign and symmetry factor multiply the closed current after
provider materialization; dimensional-algebra certification belongs to the
provider and is not implied by accepting auxiliary variables.

Exact small-graph tests compare the complete unsplit native numerator with the
term recurrence for all four gluon Born graphs and representative one-/two-quartic
bubbles and one-quartic triangles. They cover every cut, family routing, bridge
denominators and the bubble symmetry factor. Separate multiple-output native
batch checks cover double, DoubleFloat and 70-digit arithmetic with retained row
identities and empty batches. These preparation/kernel checks do not use
physical vector states; the process controls below supply that separate check.

`NativeBareProcessProgram` now consumes the colour-term recurrence directly.
One native executor per graph returns its ordered terminal coefficients; the
Born and loop colour matrices are indexed by terms, while exact kinematics and
denominator families remain indexed by graphs. Born weighting and coherent
helicity sums still happen on complex coefficients before one reduction per
family and point. Tree recurrences already contain their bridge denominators;
loop attachments receive those factors once before execution.

The internal `gluon_scattering_qcd_program` uses actual native massless vector
states, six quark flavours with a global top mass, and the explicit overall
normalization 1/512 (initial spin/colour averaging and the identical final-state
factor 1/2). It retains four Born graphs with six colour terms and 186 loop
graphs with 258 terms in 52 families. This factory is the full-SU(3) bare
control; the model/direct wrappers supply UV/R2 completion and formal projection.
General vector states with nonzero mass are rejected until their helicity and
input consistency contract is implemented.


## Pure-gluon source basis controls

`prepare_gluon_counterterm_basis(source, native_model)` provides a bounded
pure-QCD gluon catalogue for `prepare_counterterm_model`: its `entries`,
`ordinary_representatives` and `primitive_validations` stay separate from graph
matching. Every ordinary representative requires exact ordered ports, source
colour, normalized Lorentz tensor and scalar coupling identity. For `loop_sm`
and the bundled QCD/QED model these are `V_36`'s cubic slot and `V_37`'s three
diagonal quartic slots. Unknown or modified pure-gluon structures fail; model
scalar renormalization and the active R2 quartic's formal-Nc continuation remain
uncertified. The primitive whitelist contains only the already checked two-point
and active four-point R2 tensors, and never reactivates the commented model rule.

`gluon_vertex_oracle.py` independently checks all Lorentz components at real and
complex off-shell momenta for the ordinary/UV/R2 cubic rule and ordinary/UV quartic
rules. Its 1,672 ALOHA comparisons and 34 exact MadGraph colour overlaps cover
26 source basis records. `four_gluon_oracle.py` retains its independent active
R2 quartic checks. No reference routine or source model is patched.

For the initial six-flavour `gg -> gg` graph set, native matching gives four Born
graphs, 186 bare loops and 119 local CT insertions: 42 R2 cubic, 42 UV cubic,
21 R2 propagator, seven active R2 quartic and seven UV quartic insertions. The
seven alternatives are the six quarks and the source gauge group; no additional
ghost multiplier is introduced. External fields remain a separate stream:
four Born graphs times four gluon legs times the three declared heavy-quark
groups gives 48 source records. With massless bottom/charm, only the top group
is active. MadLoop groups the four identical leg factors into `4*UVWFCT_G_2`
for each of its six Born colour/Lorentz amplitudes and two Laurent sectors.
The generic field-factor oracle verifies the same factor against the isolated
reference field contribution at three points, using the independent Born values.

`NativeModelVirtualProgram` now preserves those alternatives through physical
assembly. A local CT graph has one native multi-output coefficient program;
its rectangular Born-term by CT-term matrix is contracted coherently. A field
record multiplies every coefficient term of its referenced Born graph, rather
than interpreting a graph index as a single amplitude column. These factors
share the already computed native external states and Born values. Source CT
rules retain their own loop factors and field half-Z convention.

`gluon_scattering_model_virtual_program` combines the bare gluon program with
119 matched source insertions (126 active Laurent currents) and 48 field
records (32 active coefficients when bottom/charm are massless). It returns
separate bare, local UV, external-field UV, total UV, R2 and total virtual values.
The independently instrumented MadLoop reference checks all of these components;
exact heavy-quark decoupling and field formulas check original decimal global
parameters and point-specific scales in each arithmetic carrier. Full finite
physics comparisons remain binary64 reference checks. The control supports
full SU(3) with model R2. The direct-R2 branch supports the bounded formal
projection described above; the active R2 quartic model's SU(Nc) continuation
remains explicitly unsupported.

## Quark/gluon counterterm tensor catalogue

`prepare_qcd_counterterm_basis(source, native, quark_pdgs=(2,))` combines the
controlled gauge catalogue with explicitly selected massless up/down vector
currents. Exact source and native tensor, coupling and ordered species checks
protect the existing vector-current calibration. Unknown QCD quark structures
fail; a general chiral or massive-quark tensor catalogue is not implied.

The new source primitive is `R2_QQ_1=P(-1,1)*Gamma(-1,2,1)` with fundamental
colour identity. Its local oracle compares all ordered spinor components at
real and complex off-shell momenta against generated ALOHA: 34 checks. Another
48 exact component identities at generic symbolic momentum establish its relation to the source ordinary
FFV contracted with the first-leg momentum and the corresponding calibrated
native FFV contraction. The latter gives `P(-1,1)*Gamma(-1,1,2)` in the bundled
model convention. This is a bounded derivation, not an inferred universal
spinor transpose. A separate `CountertermTensorRepresentative` preserves exact
source evidence while selecting that explicit native expression at model
preparation. Scalar coefficients and process insertion phase/routing remain
separate controls; the helper does not set a whole-process validation flag.

For `u ubar -> gg`, matching gives three Born graphs, 45 loop graphs and 63
local counterterm insertions: 35 quark/gluon UV vertices, five quark/gluon R2
vertices, seven each cubic-gluon UV, cubic-gluon R2 and gluon two-point R2, and
two quark two-point R2 insertions. There is no source QCD qqgg contact or new
external ghost tensor. The independent field stream contains 18 source records
(three Born graphs, two gluon legs and the three c/b/t groups). With massless
charm/bottom only six top-group records remain active. MadLoop's different
69 local HELAS amplitudes count includes separate tensor/Laurent structures;
it is not a counterterm graph count.

The internal `quark_gluon_model_virtual_program` uses the same native process
and CT assembly with mixed Dirac/vector states and the explicit normalization
1/72. Its isolated `V_R2UU` subset agrees with independently filtered MadLoop
at all three points, including the nonzero finite contribution and zero poles.
The physical assembler applies no additional transpose, insertion phase or
normalization adjustment to the contraction-derived representative.

Complete component tests cover bare, local UV, external-field UV, total UV,
R2 and the full virtual result in double, DoubleFloat and arbitrary precision.
Independent analytic Born, double-pole and heavy-quark UV/field formulas check
global coupling/mass rebinding and separate point scales beyond binary64 input
precision. The local and field UV finite terms cancel in this process; testing
them separately prevents that cancellation from hiding a scale-binding error.
Reference finite-amplitude agreement remains a binary64 check. This control
still selects one Born coupling order and one virtual squared-order sector;
projected multiple-colour-term CT assembly is unsupported.
