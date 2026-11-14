# Exact loop-model preparation

> Partly historical: the model preparation (`loops.model`, `loops.compiler`
> `prepare_loop_request_model`) is shared with engine v2; the kernels,
> recurrence, physical programs and oracles described for the per-diagram
> legacy one-loop engine were removed on 2026-10-29 (MQ-LEGACY-REMOVE; see
> `loop_v2/DESIGN.md` §13.3).

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
Formal-Nc identities of colour parameters (CA, CF, TR, ...) are never assumed:
a model supplies them in its `LoopColorDeclaration.parameter_identities`.
Legacy inconsistent non-QCD Lorentz spin signatures fail when requested.

## Exact restrictions and external requests

`prepare_loop_restrictions(raw, simplify=True)` applies the captured restriction
card to a derived raw declaration view and retains the original unchanged. The
installed UFO `ParamCard` parser owns block/LHA addresses. Source-line annotations
in comments preserve the original numeric lexemes; its binary64 values are never
used. Every source record and address must survive exactly, so duplicate records
(including values that round to the same double), malformed headers, scans,
continuations, nonfinite literals, scale-dependent blocks and decay tables are
rejected. As in MadGraph's restriction reader, an entry whose LHA address names
no external parameter of the model (for example `DECAY 15` in models without a
tau width) is ignored and recorded under `ignored_entries`; names and comments
never bind an entry. Signed decimal and Fortran D-exponent literals are exact.
Complex parameters require both real and `IM` blocks; an `IM` entry only
addresses a complex parameter, and partial component updates are explicitly
unsupported and cannot erase an omitted component. All restriction cards shipped
with loop_sm, loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu load, except Gmu's
`restrict_parallel_test.dat`, which assigns `SMINPUTS 1` twice and which
MadGraph 3.7.0 rejects as well.

Nonzero entries replace adjustable defaults. Simplification fixes exactly zero
external parameters, including omitted exact-zero defaults, under the existing
loader convention. It does not identify equal nonzero parameters. Structural
lowering carries these fixed values into scalar binding and rejects contradictory
overrides. Without simplification, even a zero-valued adjustable mass retains
its structural massive classification. The existing raw restriction metadata
holds this result; no additional identity or manifest is introduced.

`prepare_loop_external_request(model, incoming, outgoing)` resolves particle
names or PDGs through the native model. It derives directed scalar/Dirac/vector
state kinds, structural helicities, component offsets, initial spin and colour
averages, and identical-final factorials. The complete divisor follows the
existing physical engine: initial spin × initial colour × identical-final
factor, with couplings retained in the local kernels. Ghosts, Goldstones,
nonpropagating states, Majorana fermions and unsupported spins/representations
are rejected. This is request preparation, not a process capability certificate.

`source_bare_program` and `source_model_program` now use these model-derived
defaults when state kinds or normalization are omitted. Existing explicit
internal overrides remain supported. The derived defaults are retained as
plain request metadata for the existing artifact boundary. Complete neutral
source controls exercise the defaults against independent Born/bare/UV/R2
references, per-point scales and retained model snapshots in all three carriers.
Public loop generation remains guarded until portable plan integration and
selected-model capability checks are complete.

`prepare_loop_request_scope` may exclude a source basis before native tensor
lowering only when an explicitly bounded coupling power proves it cannot occur.
Every source coupling first requires declared nonnegative integer order data.
Ordinary rules use the conservative coordinatewise union of Born/loop upper
bounds; CT rules use loop upper bounds. Missing or unbounded coordinates do not
exclude anything, and lower bounds cannot prune local factors. Original basis
keys, including CT loop-group indices, and each exceeded-bound witness remain
available in the scope. The full raw source and scalar program remain intact.

Scoped structural lowering retains the same raw object, and a source physical
call must use the bounds that prepared it. Species perturbative-order facts
come from the full ordinary source catalogue after exact-zero pruning and
before request pruning. The graph builder validates these facts against native
species/order declarations and uses its existing joint-correction eligibility
formula; removed vertices cannot erase eligibility metadata. Scoped local CT coverage requires
exact equality with every admitted source CT key and no unavailable entries.
This covers local bases only: particle field counterterms, graph-derived
correction eligibility, tensor conformance, selected Born/loop pairings and the
physical scheme remain separate required checks. Unscoped callers retain the
complete-source catalogue guard. The real `loop_sm` pure-QCD graph control uses
this proof to exclude an unrelated QED scalar basis with an inconsistent source
spin signature; the unscoped model still rejects it.

`loops.compiler.prepare_loop_request_model` connects these boundaries without
process-name dispatch. It consumes either an existing `CompiledModel` raw UFO
sidecar or an explicit `RawLoopModel`; normalized binary64 parameter defaults
are not used. The returned preparation retains its compiled owner, original and
restricted declarations, scope, native structural model, one paired scalar
program and the selected complete local CT catalogue. Its `compile` method
resolves physical external identities and forwards exact request bounds to the
existing native complete engine.

The source catalogue reuses the exact-key local neutral-lepton, fundamental
FFV and gluon tensor certificates. Unknown bases stay present and reject if
matched by the requested process. It does not replace source coefficients,
infer an absent UV scheme, or certify formal-Nc continuation of fixed-SU(3)
four-gluon R2 rules. The source-QCD four-quark control agrees separately in bare,
UV, R2 and total Laurent coefficients with independent MadLoop components in
double, double-double and AP60. Exact analytic Born/UV controls test source
parameter updates beyond binary64 and per-point scale binding. The same facade
also exercises the complete neutral Gmu source. Source particle-field factors
use the same paired owner and scale binding, as described below. Public
generation and portable source-plan admission are unchanged.

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
path uses `PhysicalColorSpace` to contract the original native SU(3) tensors;
`formal_color=True` selects the separate formal full-colour calculation. The
direct-R2 gluon factory also supports these projections with six Born colour
terms and rectangular Born/CT matrices. One admitted squared-order sector is
still required. These
controls do not certify arbitrary models or the four-gluon effective model
R2 rules' general-Nc continuation.

The physical path uses the dependency's adjoint, dummy-index scoping, trace,
Fierz, projector and dimension-invariant operations. It never materializes
the external colour component grid. Born/loop and Born/CT matrices retain
exact complex entries; averaging is applied once after contraction. Physical
representation dimensions and literal scalar coefficients stay unchanged,
so fixed-SU(3) effective rules need no general-Nc continuation certificate.
Dense component contraction is retained only in small independent test oracles.
One cold process owns a cache of exact native tensor-expression pairs;
model values and event buffers never enter it. Open-tensor canonicalization
is not required for cache correctness. An unresolved closed network uses the
same dependency trace/chain expansion and canonicalization sequence as the
formal helper, and unknown residual tensors are rejected.

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
Each graph retains a native executor with ordered terminal coefficients; the
Born and loop colour matrices are indexed by terms, while exact kinematics and
denominator families remain indexed by graphs. Born weighting and coherent
helicity sums still happen on complex coefficients before one reduction per
family and point. Tree recurrences already contain their bridge denominators;
loop attachments receive those factors once before execution.

The process owns a native coefficient-kernel cache shared by Born, loop and
local CT executors. Equal saved instruction programs reuse cold compiled kernels
with independent mutable scratch. Born and loop schedules may also share their
outputs within one call, but only when ordered instructions, input/output slots,
polynomial labels, auxiliary domain, arithmetic and exact symbolic input
bindings agree. The bindings come from the same immutable model-definition
snapshot compiled into the model evaluator. No sampled parameter values enter
the proof. Every graph keeps its original position, colour, signs and family
mapping. `cache_coefficient_kernels=False` and
`share_coefficient_schedules=False` retain the uncached, unshared diagnostic path;
both switches default to true. Cache statistics and frozen representative maps
are available on the internal program. Numerical outputs are never reused
between evaluation calls.

The internal `gluon_scattering_qcd_program` uses actual native massless vector
states, six quark flavours with a global top mass, and the explicit overall
normalization 1/512 (initial spin/colour averaging and the identical final-state
factor 1/2). It retains four Born graphs with six colour terms and 186 loop
graphs with 258 terms in 52 families. This factory is the full-SU(3) bare
control; the model/direct wrappers supply UV/R2 completion and formal projection.
The shared physical assembler also accepts scalar and massive-vector state
layouts. `ExternalStateLayout` records each leg's component offset and physical
helicities: scalars have one component and helicity zero, Dirac and massless
vector legs have four components and two helicities, and massive vectors include
the longitudinal helicity. Exact fixed-zero model masses select the massless
catalogue; changing an adjustable mass numerically does not change the compiled
state space. The derived `initial_spin_average` is exposed separately from the
caller-owned total normalization, which remains unchanged.

The dependency prepares width-one and width-four states in at most two native
bulk calls, followed by native column packing into the original leg order.
Scalar unit wavefunctions have no recurrence tensor ports. Bound model columns,
compact vector-source views and partial-lane compaction use these actual widths.
A failed state removes its entire original physical point before helicity sums.
The dependency's undefined longitudinal rest-vector state remains an explicit
failure; no local polarization formula or momentum repair is introduced.
Supplied momenta must still satisfy the declared external model masses, and a
general scale-aware shell-consistency check remains open.

All three arithmetic carriers pass scalar contact-model Born and bubble-pole
controls, and an isolated scalar--massive-vector contact model agrees with the
independent physical-completeness Born sum. The latter is a layout control, not
an SM process prediction or a general source-model/renormalization certificate.
Arbitrary external-vector sources retain their separately certified massless
gluon domain.


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

## Ordered bare interference

`NativeBareProcessProgram.evaluate_ordered_native` and
`evaluate_ordered_decimal` retain every structural squared-order sector from
the generated catalogue. Born and loop currents are still evaluated once per
exact shared schedule. Each Born terminal has its graph's absolute amplitude
order; a family projector includes a Born/loop term only when its exact pair
occurs in the corresponding `LoopOrderPlan` sector. Joint correction eligibility
remains a graph-construction decision and is never reconstructed from an order
difference. Execution also retains the plan's exact contribution IDs, so a
singleton correction selection cannot inherit same-order mixed graphs from a
larger joint catalogue.

The native projector produces complex coefficients indexed by sector and
monomial. Reduction batches pack point and active sector into lanes, with the
reversible ID `point * number_of_virtual_sectors + sector`. One exact family
kinematic evaluation is broadcast to those lanes. Real parts are taken only
after integration. Status columns retain explicit `(sector, family)` metadata;
failures identify the original point, sector and family.

`OrderedBareProcessResult` has decimal complex-pair arrays with shapes
`born[point, born_squared_order, 2]` and
`virtual[point, virtual_squared_order, epsilon, 2]`. Epsilon powers are
`(0, -1, -2)`. Canonical order metadata includes all contributing amplitude
pairs and selected correction names. `born_order` and `virtual_order` reject
absent selectors. `sum_orders()` is explicit and uses sums evaluated in the
native arithmetic carrier before decimal transport. Result arrays are read-only.
Structurally generated sectors remain present even when an exact colour
identity makes their value zero.

The mixed QCD/QED distinct-flavour four-quark control has Born sectors 40, 22,
04 and virtual sectors 60, 42, 24, 06, where digits denote fundamental QCD and
QED coupling powers. Born 22 vanishes by colour. Virtual 42 contains both
B20/L22 and B02/L40, whose separate nonzero contributions are tested before
summing. Independent Born and universal double-pole formulas, coupling rescaling,
explicit order sums and empty/rebound batches cover all three arithmetic modes.
An independent MadLoop export with no counterterm amplitudes checks every Born
and virtual sector at three points, including all Laurent coefficients. Separate
Born-source filters independently check both contributions to virtual 42.
These reference comparisons are binary64 controls; the precision checks use
the exact Born, infrared and coupling-scaling identities separately.

The bare boundary remains four-dimensional interference. Full-colour model-rule
completion uses the ordered extension below; multi-order formal-colour and
direct-dimensional completion still reject explicitly. Single-sector adapters
remove the order axes only after checking that both Born and virtual catalogues
have exactly one entry.
For the existing single-sector dimensional diagnostic, ordered `virtual` still
contains N4 alone and `rational` contains the separate direct remainder; the
convention explicitly labels both channels without claiming UV completion.

## Ordered model-rule completion

`NativeModelVirtualProgram.evaluate_ordered_decimal` assembles local UV, model
R2 and external-field insertions on the bare program's exact squared-order
catalogue. A local CT keeps its absolute amplitude order and native graph
identity. Its source correction must be selected, and a matching loop witness
must occur among the bare program's admitted graph IDs. The source-defined
`UVtree` class retains its explicit correction-order eligibility without a loop
witness; other unmatched insertions cannot acquire that exemption. Matching
loop multiplicity is never applied to the already generated CT tree again.

For each sector, a local insertion interferes only with Born terminals whose
`(Born order, CT amplitude order)` occurs in that sector's explicit pair list.
External-field terms keep the exact right-hand Born graph, external leg and
source loop-group alternative. Their amplitude order includes the source field
correction; left-hand Born terms use the same pair mask. The source field
coefficient already contains the required half-Z factor. All complex
contractions precede the real part, and all scalars, currents, sector sums and
component sums execute in the selected native arithmetic carrier. A scalar or
current is evaluated once per batch and shared across sectors.

`OrderedModelVirtualResult` retains `born[point, Born order, 2]` and six
virtual-valued arrays on `[point, virtual order, epsilon, 2]`: the complete
`virtual`, `bare`, total `uv`, `r2`, `uv_local` and `uv_external_fields` components
(the Born array has its own order axis). Epsilon powers remain `(0,-1,-2)`.
Selectors and explicit `sum_orders()` follow the bare result conventions;
all component sums are computed natively before decimal serialization. Arrays
are read-only. `channels`/`statuses` describe bare reductions; nonfinite CT or
final assembly values raise before exposing the result, rather than claiming
successful per-lane rescue.

The first complete multi-order control is QCD correction to mixed gluon/photon
Born `u ubar -> d dbar`, using exact `loop_sm` model rules and calibrated native
quark/gluon and quark/photon bases. It retains 16 loop graphs, Born sectors
40/22/04 and virtual sectors 60/42/24. An independent MadLoop export retaining
all compatible UV/R2 interactions and particle counterterms checks full, bare,
local UV, R2 and field components at three points. Separately filtered Born
sources check both origins of sector 42: B20/L22 is zero by colour orthogonality
for this QCD-only selection, while B02/L40 is nonzero. These are distinct from
the joint QCD/QED bare catalogue above.

All three arithmetic modes additionally check the exact photon-Born sector
against the two-current massless form factor under separate point scales and
couplings with digits beyond binary64. A mixed heavy-quark Born control checks
source external-field factors against their exact on-shell formula and isolates
the photon graph's field insertions, proving that they cannot renormalize the
QCD Born ket. This is component evidence, not an independent complete mixed-top
reference. The complete mixed massless reference remains a binary64 comparison.
Joint QCD/QED model completion, multi-order formal colour/direct R2, general
model completeness preflight and public result integration remain open.

### Exact source scalar frontend

`loops.scalar_frontend.lower_scalar_program` is an explicit preparation entrypoint
for the bounded UFO scalar language. It preserves exact numeric tokens, source
conditionals and Python `and`/`or` operand-return semantics. Source helper bodies
are certified at construction, and actual `function_library` namespace bindings
select their record identities. The Gmu model's final CMS `reglog` binding therefore
replaces its earlier principal-log definition without deleting either raw record.
Unknown scope, arbitrary Python syntax and ambiguous argument strings fail.

The resulting `ScalarProgram` retains a diagnostic beside every value. Saved
native evaluators export interleaved `(value, diagnostic)` outputs; callers must
check **both** for finiteness. The diagnostic follows only active branches,
including inside separately serialized source helper bodies. This is necessary
because a NaN used directly as a conditional predicate can select a finite value.
The fixed native `pyamplicol_scalar_v1` truth, equality and real-order predicates
use the selected carrier's comparison operations. They reject nonfinite operands
and nonreal ordered comparisons. They are registered before native decoding and
cannot be rebound through the Python consumer API; their versioned definitions
are fixed for the lifetime of an evaluator cache. Source helper bodies, unlike
callback implementations, are included in saved evaluator bytes. Typed component
access uses `Complex.re`/`Complex.im` directly: arithmetic reconstruction through
`z +/- conj(z)` can overflow finite inputs or erase subnormal components.

CT dependencies are substituted before dependency-owned Laurent extraction.
Epsilon-dependent opaque function arguments are explicitly unsupported until
native helper expansion is implemented; they never become unspecified formal
derivatives. Ordinary parameter definitions remain separate for exact binding by
a future EW assembler. The existing QCD lowering/physical path is unchanged, and
successful source preparation does not enable EW process generation or certify
EW counterterm completeness, complex-mass conventions or numerical physics.

### Paired scalar ownership in model and counterterm preparation

`build_loop_model(raw, scalar_program=program)` accepts the exact source scalar
program as an additive alternative to legacy QCD lowering. The program must
belong to the identical immutable raw source object and contain its complete
declaration set. The IR and every kernel retain that owner and the original
`SourceScalarDefinition` objects. `KernelBasisEntry.scalar_expression(power)`
returns the value and active diagnostic together; source helper definitions are
kept in the owner's library rather than expanded or discarded.

Local CT insertions expose `coefficient_expressions(power)` and external-field
insertions expose `coefficient_expression(power)`. Field coefficients use the
same branch-preserving frontend, including CT-dependent source aliases. Native
CT and field evaluators serialize the owning helper library, evaluate paired
outputs in one native batch, and check every value and diagnostic before selecting
the physical value columns. A coefficient with an identically zero value is not
discarded when its diagnostic can be active. Legacy value-only preparation
accessors explicitly reject diagnostic-bearing or helper-dependent definitions.

An empty outer UFO `loop_particles` list denotes an unrestricted loop group,
matching MadGraph's import convention. Preparation normalizes this to a wildcard
for each declared coupling group while retaining the raw empty source list.

Regression controls cover complete QCD top-pair component parity in all three
arithmetic modes, source helper serialization, inactive invalid branches, and
active invalid CT/field conditions whose physical values remain finite or zero.
The full Gmu source can now prepare this unsummed IR. This does not certify its
physical tensor conventions or UV/R2 completeness. Paired source programs with
formal colour projection remain explicitly unsupported until scalar grading and
diagnostics are handled together. Native per-point scale binding is described
below; complete source-process integration has separate physics gates.

## Directed native colour weights

`loops.color_connections` adapts the existing typed `ColorConnection` histories
to native tensor expressions. It inserts dependency generators and structure
constants with physical `Tr(Ta Tb)=delta(ab)/2`, all-outgoing quark/antiquark
signs and adjoint orientation. It supports emissions from emitted legs and
gluon-to-quark-pair splitting without a kinematic splitting factor. Initial
colour ports use integer leg labels; dummy scoping, index contraction, adjoints
and SU(N) invariants belong to the tensor dependency.

`directed_color_bilinears` returns independent rectangular Born-loop and
loop-Born matrices. The latter is the adjoint of the Born-loop matrix for
**swapped histories**; directed interference need not equal twice the real part
of its first cross term. Existing exact tree contractions independently check
these weights, including a complex non-Hermitian control. `SUNColorSpace` now
retries the dependency's recognized quartic invariant substitution after
binding the explicit adjoint dimension, following ordinary projector expansion.
No private Gram formula is introduced. These are cold generation helpers;
the numerical directed contracts and their supported domains are described
below. Spin, mapping and FFT composition have separate implementation gates.

## Exact global scalar bindings

`prepare_scalar_bindings` resolves selected parameter/coupling Laurent outputs
with the source-scoped scalar frontend. External parameters remain independent
symbols; internal values and diagnostics are substituted together through the
source syntax so an inactive conditional dependency stays inactive. A supplied
`ScalarProgram` must belong to the same immutable raw source instance.

`NativeScalarBindings` computes exact source defaults and selected derived
values in double, DoubleFloat or arbitrary precision. Overrides accept decimal
strings and complex decimal pairs. They never enter a binary64 parameter map.
Real-type input validation rejects even imaginary strings below binary64 range.
Defaults are used only for parameters without overrides. Updates validate every
selected value and active diagnostic before atomically replacing state; failed
updates preserve the previous arguments, values, overrides and revision. The
immutable `snapshot()` exposes those four fields together, and retained native
buffers remain valid across later updates. Concurrent writers are serialized;
readers take one consistent snapshot. Returned values are native one-row batches
suitable for native event broadcasting.

`NativeScaleBindings(global_state, plan=None, scale_parameter="MU_R")` is a
compiled per-point view of that same global state. An alternate output plan can
select CT/Laurent coefficients, but must use the identical `ScalarProgram`,
external names/types/input order and exact fixed-parameter map. The selected
scale must be a real, adjustable external parameter. `evaluate_native(mu_r2)`
accepts one native scale column, validates positive real finite values, computes
the square root with the dependency evaluator and replaces only that input in
the broadcast global row. Both selected values and active source diagnostics
are evaluated natively. No intermediate decimal or Python complex transport is
used. Native carrier/precision checks apply even to empty batches; original
lane IDs survive broadcasts, tail batches and failure reporting. Each call
reads the owner snapshot once, and subsequent calls observe successful updates.
The optional `snapshot=` argument retains one revision across ordinary and UV
physical stages; snapshots from another state are rejected even with the same
plan. This is an in-memory ownership check, not an artifact identity.

`reconstruct_snapshot(source_owner, snapshot)` on another carrier binding of the
identical plan rebuilds exact defaults and captured overrides without publishing
an update. The source owner is checked before numerical work. Its native rounded
arguments and values are never used as higher-precision inputs.
`NativeScaleBindings.evaluate_partial_native` returns sparse native values and
original-ID numerical failures; it filters scale and active-diagnostic failures
before their values enter later evaluators. Metadata/carrier errors still raise,
including for empty batches. The ordinary method retains its raising behavior.

`ScalarBindingPlan.prepared` retains a `PreparedScalarBindings` record: exact
saved paired value/default/type-guard programs, ordered external names/types,
selected Laurent output labels and exact fixed values. A scale view's
`PreparedScaleBindings` retains its selected paired program and scale-check
program with a reference to that same owner record. These frozen records contain
no raw source, helper library, symbolic producer or numerical buffer. Binding
them in a new carrier reconstructs fresh native evaluators; defaults and captured
decimal overrides are evaluated directly in that carrier. Scale bindings require
the identical prepared owner, while snapshots still belong to one mutable native
state. Tests forbid source lowering/evaluator saving during all-three-carrier
rebinding and exercise anonymous field outputs, active diagnostics, fixed/type
guards, beyond-binary64 updates, sparse original IDs and empty batches.

These are scalar-state consumers, not a production EW runtime. Their bounded
physical UV integration is described below. No coupling running or scheme choices
are inferred here. Exact cold output expressions can also
prove structural zero mass aliases without evaluating adjustable defaults.

## Structural covariant source model

`build_covariant_structure` prepares a public native `ModelJSON` model from the
lossless raw species and ordinary vertices, independently of the restricted
tree importer. It preserves exact source charges, directed ghosts, ordered
species and every colour/Lorentz coupling slot. Numeric parameter and coupling
caches remain empty. Adjustable external masses keep independent symbols even
when their source default is zero. The shared scalar binding preparation resolves
mass aliases exactly before native mass classification; the installed Community
classifier alone recognizes literal zero but does not resolve a `ZERO` alias.
An unresolved active diagnostic cannot certify a structural zero.

Massive and massless vectors use the dependency's public custom-propagator
interface with the covariant numerator `-i Metric(1,2)`. The ordinary loader's
massive-vector numerator contains the longitudinal Proca term even with its
`Feynman` argument, so that argument is insufficient for this catalogue.
Scalars, Dirac fermions and directed ghosts use its existing propagator builder.
Optional explicit covariant cut multiplets pass through native model validation;
the Gmu control includes the neutral and both charged vector/Goldstone/ghost
quartets. This does not identify ghosts merely by taking opposite PDG signs.

The source observer retains `Particle.charge` arithmetic before Python rounding,
including `2/3`, and certifies antiparticle negation from the actual observed
`anti()` declaration. Raw runtime charge metadata remains available. Structural
preparation independently checks conjugate species, mass, width, colour, electric
charge, ghost/lepton numbers and Goldstone/propagating flags.

The explicit `gmu_particle_anti` compatibility profile recognizes the demonstrated
old positional `Particle.anti()` call against its actual constructor signature
and complete charged-Goldstone metadata. It restores only the derived G−
Goldstone flag, retaining raw `False`, derived `True` and the reason in adapter
diagnostics. No source file or antiparticle counterterm is changed. Unknown
metadata mismatches fail. The full Gmu structural control contains 43 particles,
163 ordinary vertices, 132 used coupling definitions and 22 local Lorentz spin
signatures; it requires 37 independent external bindings.

This is a bounded structural and kernel-preparation interface. Unspecified gauge,
custom propagators, unapplied restriction cards, Majorana or four-fermion rules,
nonpropagating auxiliary species and uncertified chiral hypercharge conventions
are rejected. The object retains the exact scalar program but does not bridge
the existing counterterm `LoopModelIR`, certify complex-mass parameter
transformations, support unstable external particles, or enable physical EW
generation. Those require separate source tensor and complete-process checks.

## Explicit fixed source parameters

`prepare_scalar_bindings` and `build_covariant_structure` accept
`fixed_parameters` as decimal strings or exact decimal complex pairs. These
are constants selected during generation, separate from adjustable runtime
parameters. The source model and scalar program remain unchanged. Fixed values
are inserted before dependency resolution and native mass classification;
`1e-1000` remains structurally nonzero, while an explicit zero can classify a
massless particle. Runtime updates to a fixed name fail atomically.

The covariant structure exposes `scalar_bindings(outputs)` to carry the same
fixed map into numerical evaluation. Ordinary source coupling slots disappear
only when both the resolved value and its active diagnostic are exactly zero;
a vertex disappears only when all its slots do. This removes zero Yukawa
vertices without using numerical defaults or discarding invalid active branches.
All remaining couplings retain their source definitions. Fixed kinematic masses
and independent Yukawa inputs must be specified separately.

This is an explicit specialization interface. It does not parse restriction
cards, infer ties between equal parameters, or certify EW counterterm completion.
Unapplied source restriction cards continue to reject. Model preparation must
pass the same fixed bindings to its subsequent physical consumers.

## Directed bare numerical diagnostics

`NativeDirectedBareProgram(bare, histories)` evaluates a list of ordered
`(bra, ket)` colour connections on an unmodified full-colour bare program.
`directed_color_legs` exposes the actual native amplitude ports with zero-based
physical momentum-leg labels. Their signed representations include the native
crossing convention and must not be inferred from physical particle names.
The dependency constructs the ordered rectangular colour weights. Each matrix
is evaluated at exact Nc=3 with the original physical averaging factor.

Physical states, global model values and each exact representative Born/loop
coefficient schedule execute once per batch, independently of the number of
requested histories. One family kinematic preparation is broadcast over all
active directions and squared-order sectors. Native reduction lanes have the
stable identity `point * (directions * sectors) + direction * sectors + sector`,
where directions `2*h` and `2*h+1` denote history `h` and its swap. Status columns
carry `(direction, sector, family)` metadata and retain those original lane IDs.

Both complex Born-weighted numerators use the same Feynman prescription. After
reduction and multiplication by `i/(16*pi**2)`, assembly adds the forward value
to the conjugate of the swapped value. Conjugating an unintegrated numerator
does not provide the second term. The result retains complex directed Born
contractions and both separate cross terms; it does not apply `2 Re` to a
non-Hermitian kernel. Decimal arrays have axes `point, history, squared order`
and, for virtual values, `(finite, single pole, double pole)`, followed by the
real/imaginary pair. Native buffers carry all intermediate arithmetic.

Actual `u ubar -> d dbar` controls in double, DoubleFloat and arbitrary precision
check identity histories, physical Casimirs, colour-charge conservation, swapped
conjugation and a non-Hermitian history with a nonzero absorptive finite part.
Instrumentation checks that histories do not repeat current or kinematic work.
A mixed-order control retains all three Born and four virtual squared-order
sectors, with exact complex coupling factors used to test a genuinely complex
Born contraction. This scalar-binding diagnostic is not an SM parameter claim.
Empty, changing and tail batches retain point/direction/sector identities.

This interface contains N4 only: no UV, model/direct R2, external-field, spin,
mapping or public correlated-completeness claim follows. Dimensional programs,
model-bound programs and changed colour metrics reject explicitly. Attachment
locks ordinary metric mutation; later model binding is detected before directed
evaluation. Dependency reduction success retains its existing accuracy semantics
and does not establish an independent rescue or accuracy certificate.

## Complete directed model-rule QCD controls

`NativeDirectedModelProgram(model, histories)` extends the same directed
contractions to an existing validated `NativeModelVirtualProgram`. Its current
scope is unmodified full SU(3) colour with model R2 and QCD corrections. Formal
colour projections and direct dimensional R2 reject; no EW, spin, flavour
mapping, FFT or public process-dispatch completeness is implied.

The ordinary model executor exposes checked unprojected CT amplitudes and field
factors on the already prepared physical batch. The directed executor reuses
those currents and their retained native colour tensors. It contracts the
forward and swapped-history Born/CT matrices independently, with the exact
Born/loop order pairings. The swapped term is conjugated after its complete
complex contraction. Source effective rules already contain the loop measure,
so no extra `i/(16*pi**2)` multiplies them. Field insertions retain the source
`delta-Z/2` factor and the right-hand terms of their specific Born graph.

The result retains complex Born, bare, local UV, external-field UV, total UV,
R2 and completed virtual arrays, plus the total first and second cross terms.
All values stay in native double, DoubleFloat or arbitrary-precision buffers
until final decimal transport. Histories share one ordinary state/current batch
and one execution of each CT current. Reduction statuses retain their original
point/direction/order/family identities; invalid CT scalars, active diagnostics,
currents or contractions raise before exposing a result. Successful reduction
still does not independently establish numerical accuracy or provide rescue.

The light-quark and top-pair QCD controls pass ordinary identity-history
agreement for every component, Casimir scaling, colour-charge conservation,
swapped-history conjugation and nonzero absorptive complex contributions in all
three arithmetic modes. The massive control includes both physical external
field factors. A separate mixed-Born QCD algebra diagnostic deliberately gives
effective couplings a complex phase: it retains a nonzero imaginary R2 result
and verifies that the two isolated Born origins recombine in the same squared
order for every component and both cross terms. That phase is a contraction
test, not a source-model renormalization convention.

## Literal gluon source preparation

`NativeSpinSourcePlan` reuses the tree `CorrelatorConfig` joint spin classes
and exact vector-input parser. Labels are public one-based legs. The current
bridge admits massless PDG21 vector slots and reads their existing tree
`SourceIR` crossing metadata once during preparation. Supplied temporal, spatial
and imaginary components remain literal: they are not projected, normalized,
negated or conjugated for an outgoing `epsilon_bar` slot. Only the ordinary
source crossing phase is applied, once, by a native dependency evaluator.

Each declared class has an immutable compact spectator-helicity layout; selected
legs contribute one source state, and unselected legs retain their ordinary
helicity sum. The shared bare process and its averaging factor remain unchanged.
Native vectors broadcast from one row or carry one row per physical point;
carrier, precision, finite components and point IDs are checked. The decimal
entrypoint uses the tree parser before native ingress. Empty batches and exact
complex inputs beyond binary64 are covered in all three arithmetic modes.

This source-preparation boundary alone does not certify a physical spin-
correlated loop result. The bounded complete-QCD execution below supplies its
separate physical checks.

## Complete QCD source-vector execution

`spin_sources.NativeSpinSourceProgram` composes declared joint gluon source
classes with `NativeDirectedModelProgram`. Source labels remain one-based;
directed colour histories retain their existing zero-based native physical
ports. Each class has an immutable spectator-helicity layout. The bare
program's ordinary helicities and initial spin/colour average are unchanged.
An omitted or empty source map selects the ordinary complete observable.

Literal native source columns replace the chosen external states before any
Born, loop, local UV/R2 or external-field current executes. Those currents are
shared by the forward and swapped colour contractions. Effective rules retain
their source normalization; the two reduced loop terms retain their existing
complex prescription and are combined only after integration. No source
component is projected, normalized or conjugated at ingress. Native batches
carry the actual helicity count and source class, so a precomputed current batch
from a different class rejects even when its row count is identical. Results
retain complex pairs and their point/history/order/pole axes, and add the
selected `spin_legs` tuple.

The complete `u u~ -> g g` control covers one or both outgoing gluon sources in
double, DoubleFloat and arbitrary precision. Summing two orthonormal transverse
linear sources reproduces the ordinary helicity sum, including all effective
components and the original averaging. Complex rescaling, joint-source
rescaling, input bits beyond binary64, empty batches and stable reduction lane
IDs are checked. A source shift `v -> v + kappa*p` preserves the complete
complex observable with a nonzero Born amplitude, testing the loop Ward
identity without a trivially vanishing Born projection. Identity, adjoint
Casimir and single-emission colour-pair contractions compose with these checks;
individual integrated cross terms retain nonzero absorptive parts.

Some two-emission histories for this process currently reach an unresolved
Community projector/chain materialization error during exact colour
construction. They fail before evaluation; the bounded source tests do not
certify those histories. The separate non-Hermitian complete `u u~ -> d d~`
and top-pair controls retain their previous coverage. Incoming-gluon numerical
source checks, other vector species, source-model electroweak completion,
flavour mappings, formal-colour/FFT source projections and public dispatch
remain separate work. Reduction success retains the existing accuracy
semantics and does not establish a new precision-rescue certificate.
## Source-owned bare electroweak control

`physical_source.source_bare_program` connects a `StructuralCovariantModel` to
the existing native process/family evaluator. Selected ordinary couplings and
mass aliases are evaluated by the source `ScalarProgram`, including active
diagnostics, directly into typed native buffers. One immutable global parameter
snapshot is retained on the physical batch. A `NativeScaleBindings` view applies
each point's `mu_r2` to all selected ordinary source outputs, including masses if
they depend on the source scale. Other parameters remain global and named
updates are atomic; fixed structural parameters cannot be changed at runtime.

Standard UFO Dirac vertex endpoints require an explicit physical frame. The
adapter acts only on the two external bispinor `Slot` keys of each localized
vertex, before external states attach. Dependency `rename_indices` performs a
simultaneous swap without interning; the original free-axis set is checked.
Lorentz indices, scalar momentum arguments, contracted dummies, propagators,
source records and the native structural model remain unchanged. The recurrence
records `vertex_spinor_frame="native-physical"`; existing builtin programs keep
their original frame and instructions. Saved adapted kernels also determine
schedule-sharing identity.

The current physical source gate admits the independently tested FFV1/2/4/6
bases and FFV3/5 as their exact chiral linear combinations. Independent ALOHA
checks cover 904 component/current choices, both endpoint orders, massive and
massless physical spinors, and complex off-shell currents. Typed-map tests also
reuse the same label in Lorentz and scalar momentum positions to ensure those
positions do not move. This convention adapter does not define general gamma5
algebra, Majorana or four-fermion rules. Active epsilon structures and source
dimensional providers remain gated pending their separate certificates.

`certified_ffs_source` separately identifies the five exact four-dimensional
Dirac/scalar bases `ProjM(2,1)`, `ProjP(2,1)`, their sum and difference, and
`Identity(2,1)`, with ordered spins `(2,2,1)`. The reproducible
`tools/loop_reference/dirac_scalar_oracle.py` compares these actual Gmu source
bases and the two chiral Higgs CT aliases with unmodified standalone ALOHA
as raw source matrices, without establishing the physical directed-current frame.
Its 812 comparisons cover the full bispinor component basis, both complex
off-shell endpoint orders, local open-current components, massive/massless
physical u/v states and complex scalar currents; maximum scaled error was
`4.45e-16`. The report retains all 357 matching source basis keys, while scalar
couplings and colour factors remain outside the local Lorentz certificate.

These diagonal matrices are symmetric under endpoint exchange: the previous
endpoint-only mapping passed all those raw `_0` controls while failing directed
charged-scalar currents. The physical FFS boundary additionally exchanges the
native `projm` and `projp` heads, exactly as the established tree tensor compiler
does for UFO `ProjM` and `ProjP`. The dependency performs simultaneous typed
replacements; scalar coefficients, momenta, FFV factors and propagators are
unchanged. The pseudoscalar difference changes sign and the scalar sum/identity
remain unchanged.

`tools/loop_reference/dirac_current_oracle.py` compares actual source graph
factors with unmodified ALOHA FFS/Dirac-propagator/chiral-FFV chains, both charge
flows, arbitrary complex components and momenta, and separate massive scalar
currents: 580 comparisons passed with maximum error `1.12e-16`. Unlike a
helicity-summed or diagonal vertex check, the chiral chains
fail without the projector mapping. The source and crossed HELAS current
orders are recorded separately; off-shell output labels do not identify the
next vertex's F1/F2 input position. Complete process and dimensional-gamma5
admission remain separate gates.

The real-mass full-Gmu `e- e+ -> m- m+` control retains all electroweak species:
two Born graphs, 98 loop graphs and 32 reduction families with the official
massless-lepton/zero-Yukawa restrictions. It matches the independent MadLoop N4
bare component at three phase-space points in double, DoubleFloat and arbitrary
precision. The high-precision tests additionally check the universal double
pole and exact global Gf rebinding with input bits beyond binary64; they do not
claim an independent 80-digit reference for the full finite part. Empty batches,
point-dependent ordinary scale bindings and retained snapshots are covered.
Arbitrary source tensor coverage, source dimensional numerators and complex-mass
certification remain separate work. Model CT composition requires the explicit
source frame and paired snapshot binding described below; it cannot mix a raw
source vertex with a calibrated builtin representative.

`physical_source.source_model_program` now assembles source UV and model R2 over
the same augmented native model as its Born and bare-loop catalogues. The
prepared model records the original `ufo` vertex frame. Every matched fermion
CT must have a tested FFV/FFS expression as well as exact local source evidence;
epsilon structures still reject. CT recurrences apply the same typed endpoint
adapter as ordinary vertices. The neutral-lepton conformance helper certifies
by structure, never by particle identity (genericity audit m5): an FFV entry of
any colour-singlet antifermion/fermion pair (the antiparticle first) with any
self-conjugate vector whose exact Lorentz tensor `certified_ffv_source` accepts,
and a two-point entry of two self-conjugate vectors whose tensor is one of the
tested neutral two-vector bases. Other source rules remain present and
unvalidated, so a process that needs one fails the existing tensor-basis guard.

The separate `massive_neutral_lepton_counterterm_validations` catalogue adds,
by the same structural rule, the FFS entries of any colour-singlet
antifermion/fermion pair with any self-conjugate colour-singlet scalar whose
tensor `certified_ffs_source` accepts, and the self-conjugate scalar-vector
mixing entries with a tested momentum tensor, retaining every source key and
all other unvalidated entries. On the Gmu models this certifies the tau's
A/Z/H/G0 and the neutrinos' A/Z entries alongside the electron's and muon's.
The real-mass massive-muon control sets `MM=ymm=0.105658`, retains the same full
Gmu electroweak species, and uses three exactly on-shell terminating-decimal
points. It has two Born graphs, 145 loop graphs and 64 matched local CT
insertions, with no source particle-field insertions. Independent MadLoop
checks retain N4, UV, R2 and total separately, and isolate the 47 loop records
containing scalar-Dirac vertices. The isolated interference is roughly
`1e-10`--`1e-9`, so a full-amplitude comparison alone is insufficient to test it.
Double, DoubleFloat and arbitrary-precision execution agree with the independent
binary64 components; the finite reference does not certify 80-digit physics.
The saved DoubleFloat/AP80 outputs agree to at least 31.0 decimal digits for
Born, 28.1 for N4, 30.8 for UV/R2 and 27.9 for the complete real Laurent
components. Scaling the complete difference by `|N4|+|UV|+|R2|` gives 28.1
digits; the much smaller isolated FFS terms agree to 26.3 digits. Structural
zero coefficients remain exact. Tiny imaginary accumulation residuals are
reported separately (at most `1.51e-36` in DoubleFloat and `8.24e-84` in AP80),
so they do not become misleading relative-error estimates near zero.
The massive control's double pole receives contributions only from the two
massless electron legs. Its exact pole identity and global Gf rebinding probe
the selected carrier without a fitted finite-part formula. Source dimensional
numerators, arbitrary charged/scalar external processes and CMS completeness
remain outside this admission. The separate selective-rescue source boundary
still requires structurally massless charged leptons; this massive control does
not relax that gate.

The source factory requires the full selected CT collection and all its species;
the preparation step records this from its existing source keys. This prevents
passing a UV/R2-pruned catalogue to the source-model interface. It does not infer
general renormalization or tensor completeness, and deliberately restricted
legacy QCD control factories keep their existing scope.

Native CT aliases select exact `(source coupling, Laurent power)` outputs from a
compiled `ScalarBindingPlan`. One `NativeScaleBindings` view evaluates those
outputs and their active diagnostics per point with the **captured bare-call
snapshot**, including the source MU_R substitution. No Python event loop, scalar
decimal roundtrip, helper-body inlining or second global parameter state is
introduced. Local CT currents then share the physical states and Born weights
and enter the existing sector-aware UV/R2 contraction.

Source particle-field insertions retain their original exact Laurent strings.
`prepare_scalar_bindings(..., anonymous_definitions=...)` resolves these terminal
expressions through the same source declaration graph and helper library.
Anonymous definitions cannot shadow source declarations, reference one another,
or accept rounded numerical literals. Values and active diagnostics preserve
their conditional branches. Their syntax is retained immutably in the plan;
the original raw model and its scalar program are unchanged.

Local CT aliases and field factors share one scale view. Its paired outputs are
checked before any value-only column selection or helicity broadcast. The
existing exact Born graph identity, field loop group and amplitude-order pair
masks are unchanged, and no extra half-Z or loop measure is introduced. Source
gluon scattering exercises all 48 field insertion records and compares bare,
local UV, fields, total UV, R2 and full results separately against MadLoop. Its
double, double-double and AP60 controls also test exact heavy-quark field and
decoupling identities after global parameter/per-point scale changes, retained
snapshots and empty batches. Source-field selective rescue and portable loading
remain explicitly outside the supported internal boundary.

For full real-mass Gmu `e- e+ -> m- m+`, 59 matched CT insertion graphs produce
75 retained Laurent currents. Independent MadLoop controls compare N4 bare, UV,
R2 and total separately at three points, both at MU_R=1000 and in one batch with
MU_R=(500,1500,2000). The generated reference recomputes its source EW scale
couplings for each point. All three arithmetic modes are checked, together with
global Gf bits beyond binary64, source snapshot retention across an update,
empty batches and rejection of mixed frames or missing source collections.

`model_cms.complex_mass_scheme` (M-SM MS-CMS, a port of MG5_aMC's
`Model.change_mass_to_complex_scheme`; DESIGN §6.8) replaces the former
bounded G_mu-only declaration view. It retains the original raw object and one
derived immutable raw model (`complex_mass_model` applies the same decisions to
the compiled tree payload), generic in the model: per nonzero width at the card
an external mass gets `CMASS_M = sqrt(M**2 - i*M*width)`, real mass and width
inputs are kept, dependent internal parameters become complex and read the pole
masses, CMSParam is enabled, Yukawa inputs become mass ties, and the
restriction card's parameters of value 1 become constants (MG5_aMC restricts
after its transform). Existing exact source name tokens are rewritten without
parsing numeric lexemes through binary64.

Pole-mass references cover the complete covariant multiplet, including ghosts
whose original width metadata is ZERO. The actual optimized MadLoop ee-to-mumu
CMS export groups the charged-ghost bubbles with W/Goldstone bubbles and supplies
the complex W mass to the reducer; its ghost numerator mass argument is unused.
This is evidence for that bounded optimized control, not a claim about every
MadLoop export. Source Gfbar/GfRedefinitionChoice, recms/crecms, branch logarithms,
all CT bases and all helper bindings remain source-defined. No new
renormalization formula or denominator-only width insertion is introduced.

Cold-view tests cover native numerator/denominator aliases and selected scalar
continuation in all three arithmetic modes, including exact pole-mass-square
identities beyond binary64 and zero-width updates without regeneration. The
declaration view itself carries no physical CMS certificate; complete bare,
UV/R2 and total component comparisons and the zero-width physical continuation
are separate gates.

The bounded full-process CMS comparison exposed a concrete reference semantic
boundary. The active Gmu UFO `reglog` subtracts `2*pi*i` when its argument has
negative real and positive imaginary parts. MadGraph 3.7's exporter reserves
that function name and emits principal `LOG` instead of the source helper body.
Born, N4 bare, R2 and every pole agree with the unmodified optimized CMS export;
only UV finite terms differ. An isolated scalar comparison locates the shift in
`eCoup_DGmua0_UV_EW`: four times the real difference predicts the entire UV/Born
offset. This is a source/export semantic difference, not a fitted amplitude
correction.

The declaration helper now accepts an explicit
`reglog_convention="ufo-standard"` compatibility choice. Its default remains
`"source"`. The selected convention is recorded in the existing declaration
metadata; diagnostics retain the original bound helper and its replacement,
and the original raw model remains unchanged. Only the recognized Gmu helper
bodies are accepted. `reglogp`, `reglogm`, other helpers and all CT formulas stay
unchanged. The generic UFO importer continues to preserve source semantics.

This choice follows the [UFO 2.0 definition, section 5.2](https://arxiv.org/pdf/2304.09883):
`reglog` denotes the principal logarithm; `reglogp` and `reglogm` explicitly
select shifted sheets. The actual MadGraph 3.7 exporter reserves these names,
skips their supplied helper bodies, and emits that convention. The legacy Gmu
source instead overwrites `reglog` with `reglogm` semantics. In the finite
`eCoup_DGmua0_UV_EW` expression, four of 56 `reglog` call sites already have
upper-left arguments at zero widths:
`-0.5+1.83351678i`, `-0.5+0.72646643i`,
`-0.06046135+0.99817054i` and `-0.64287724+0.93405746i`.
They are away from the cut at the real-mass anchor. Applying the legacy blanket
shift changes the complex charge CT by `-0.102935557513438i`; this does not vanish
as the widths approach zero. Its purely imaginary zero-width value was hidden
by real Born interference. The principal result
`0.03251413826093476+0.01510486664807422i` agrees with an independently compiled
Recola real-point `Delta r/2` scalar. The focused native test checks this complex
anchor, a small-width limit, the finite-width MadLoop scalar, all four local
helper shifts, and original-source preservation in double, DoubleFloat and
arbitrary precision. The binary64 reference accuracy is not a high-precision
certificate. The argument tracing diagnostic applies the declared principal
boundary `arg(z) in (-pi, pi]`, including MadGraph's explicit signed-zero
normalization, rather than Python's negative-zero `-pi` boundary.

The explicit `ufo-standard` view also passes the complete physical pipeline
against the original, unmodified MadLoop export in double and DoubleFloat.
The persistent component control checks Born, N4 bare, UV, R2 and the total at
four finite-width continuum/resonance points, three continuum points with all
four widths scaled by `0.001`, and three zero-width points. Each carrier reuses
one program and updates the original width inputs; graphs and scalar formulas
are not regenerated. The small-width reference calls the unchanged full and
independently filtered CT binaries. The source-semantic fixture remains a
separate control. The finite and small positive-width complete checks also pass
in the arbitrary-precision carrier at 80 decimal digits. These binary64 provider
comparisons validate the declared convention in the bounded domain; they do not
establish 80-digit reference accuracy or close the AP80 zero-width master gate
below, whose strict expected-failure test is retained separately.

The source-semantic physical control uses a copied reference helper object with
only REGLOG/MP_REGLOG changed to the exact bound UFO body. All original model
rules, scalar definitions and exporter files remain untouched. The compact
fixture labels this distinction and compares bare/UV/R2/total separately at a
continuum point and three points around the Z pole. Zero-width continuation
uses the original unmodified CMS reference and passes in double and DoubleFloat.
At 80 decimal digits, two real-mass box families exposed a separate dependency
branch failure through Symbolica e679e8c1: masses produced by `sqrt(M²)` then
squaring and numerically equal reparsed masses gave opposite scalar-box
imaginary parts. A native Rust probe confirmed exact mass-value equality with
positive zero signs, but component precision changes from `(267,266)` to
`(266,266)` bits. The constant-numerator probe in
`tools/loop_reference/reproduce_cms_zero_box.py` reproduces this without UFO
data or process generation. Symbolica 9cdc2949 (a `Float` product with an exact
±1 keeps the other factor's precision) makes both inputs take the same branch,
also with the earlier NinjaRS and oneloopmaster pins, and the AP80 zero-width
physical test passes against the zero-width reference. No mass rounding, tiny
term pruning or integral correction is applied. Finite-width tests pass in all
three carriers and also check exact Gf updates beyond binary64.
The independent Recola comparison requires the same Laurent normalization as
the QCD harness: `finite_Cepsilon = finite_Recola + pi^2/6 * double_pole`.
The original CMS diagnostic omitted this conversion. Applying it exactly once
to preserved raw provider coefficients makes N4, R2 and every pole agree with
the unmodified MadLoop export. A finite UV difference remains. Independently
dumped finite values of `delta Z_AZ`, `delta Z_ZZ`, `delta M_Z^2`, `delta s_w`
and `Delta r/2`, replayed through unchanged generated MadLoop couplings, explain
that entire residual at all four continuum/resonance points (largest absolute
UV difference after replay `4.8e-14`). This replay is a convention diagnostic,
not a production correction.

The installed Recola model evaluates relevant self-energies at real momentum
squares and includes the pole-to-real expansion terms. Its width dependence
differs from the source's direct complex-pole expressions. Such higher-order
differences from a real-momentum CMS expansion are discussed in
[the Recola2/REPT1L paper, section 5](https://arxiv.org/pdf/1705.06053).
The observed finite-width difference must remain an explicit model convention;
it is separate from both the legacy `reglog` incompatibility and the AP80 master
failure. These bounded controls do not enable a physical CMS capability flag.

Reproducible workspace evidence is retained in
`reference/runs/electroweak-mumu-cms-laurent-comparison.json`,
`electroweak-mumu-cms-uv-scalar-replay.json`,
`electroweak-mumu-cms-uv-width-comparison.json`, and
`electroweak-mumu-cms-reglog-arguments.json`. The corresponding scripts preserve
the original provider outputs and source model; no branch is fitted to an
interference result and no integral implementation is modified.

The separate `neutral_scalar_vector_counterterm_validations` helper binds exact
source keys for the neutral `G0-A/Z` two-point structures `P(2,1)` and `P(2,2)`.
The local oracle in `tools/loop_reference/scalar_vector_oracle.py` compares each
vector component and complex currents against unmodified loop-mode ALOHA, with
real, complex and independent leg momenta. All 60 controls pass within
`2.82e-16` scaled error. Scalar legs have no tensor port; their current value
multiplies the local vector contraction. The helper is separate from process
admission and does not certify scalar coefficients, mixing insertion phases or
the UV scheme. Its exact ordered-port/source-expression guard rejects charged
partners, reordered slots and untested linear combinations.

### Retained instructions and arithmetic views

`PreparedCoefficientDAG` separates exact saved dependency instructions and
integer slot maps from producer binding/domain metadata. Rebinding it creates a
fresh native coefficient arena and carrier-specific evaluators without reading
the producer graph or repeating expression optimization and evaluator saving.
Existing schedule identities and exact model-binding sharing proofs are retained.

`PreparedBareProcess` and `PreparedModelProcess` extend this boundary to the
complete full-colour controls with legacy exact model formulas or paired source
scalar bindings, including the certified internal direct-R2 gluon control in
the legacy path. They retain family routing/kinematic instructions, scalar coefficient
and contraction programs, CT/field current instructions, exact weights and order
maps. `rebind` creates independent native evaluators, reducers and kernel caches;
no event buffers or numerical model values are copied from another arithmetic.
The optional stability wrapper uses this path for those controls. The reducer
still constructs its own precision-specific polynomial workspace from the
retained expressions through its existing API.

Changing a bare colour metric replaces its saved projection instructions and
invalidates its cached prepared record. Previously retained records preserve
their original weights. An attached complete model still prevents metric changes
that would desynchronize its counterterms. Tests forbid recurrence generation,
colour overlap and instruction saving while rebinding complete DY/four-quark
views, compare all output components in all three carriers, and exercise exact
input changes beyond binary64, independent buffers and empty batches.

Direct records preserve the same dimensional coefficient DAGs, closed
epsilon/μ² labels, separate N4 and rational family reducers, and UV/field
completion. Their canonical `direct` strategy is retained in every result;
model-R2 insertions and currents cannot enter a direct record. The optional
stability wrapper reuses this plan while keeping the captured strategy fixed.
Focused gg tests forbid cold work during rebinding, compare independent source
R2 trees and MadLoop components, and check original point IDs, changed exact
couplings/scales, empty batches and the lifetime of previous output buffers.
This does not extend the certified dimensional-kernel or physical process domain.

Source records retain one `PreparedScalarBindings` owner and its ordinary and CT
`PreparedScaleBindings` views. Local CT coefficients and anonymous particle-field
columns use their original selected-output maps; they are not converted to
independent legacy formulas. The source bare/model facade's `rebind` captures
one immutable exact override map, reconstructs defaults and overrides in the
new carrier, and connects both scale views to that fresh owner. Later updates
are independent of the original state. Passing `parameters=` directly to a
prepared source record selects a new global state; omitting it selects exact
source defaults. Legacy records reject that source-only argument.

Tests generate each source QCD control once, then prohibit scalar lowering,
recurrence preparation, colour contractions and evaluator saving while binding
double, DoubleFloat and arbitrary precision. Independent MadLoop N4/UV/R2/full
components and the gluon field split check the selected columns. Exact Born/UV
and field relations, changed scales, retained snapshots, empty batches and
original decimal input bits check native state flow. Reference comparisons
remain limited by the independent binary64 provider values.

These are internal in-memory records. Existing producer graph references remain
available for diagnostics; they are neither serialized nor used to regenerate
physics during rebinding. Optional source plan references likewise support
existing inspection/capability checks; ordinary binding and evaluation work
without those references. The saved source descriptor and public runtime policy
are described in [the portable execution notes](LOOP_PORTABLE_EXECUTION.md).
Source direct-R2 and broader producer partial/rescue domains remain separate
work; the retained-instruction boundary alone does not widen those capability
gates.

### Live-producer source-binding rescue

Source-owned prepared records accept the existing `VirtualStabilityPolicy`
before realizing parameters in the requested arithmetic. Without a policy,
source binding and updates still reject numerical failures eagerly. With a
policy, a typed nonfinite realization may retain exact defaults and overrides
when a configured higher arithmetic can realize them. This admission does not
certify the amplitude: complete-result agreement and conversion checks remain
necessary. Malformed literals, nonreal real parameters, ownership mismatches and
unsupported settings remain hard errors.

An admitted failed snapshot holds immutable failure stage/detail and no numeric
buffers. Reading `arguments` or `values` raises a fresh typed exception rather
than retaining a traceback in shared state. Internal `NativeScalarBindings.update`
returns the newly admitted snapshot, and validates before publishing its revision;
a rejected update leaves the old snapshot intact. Source `rebind` inherits the
existing policy unless explicitly replaced. Changing the primary arithmetic must
still leave the configured higher sequence strictly increasing.

The live stability helper captures one exact source snapshot per call and
reconstructs it through the same retained prepared owner for every higher
attempt. A concurrent update belongs to the next call, including when higher
views are already cached. No failed numerical values, producer graph lowering or
instruction saving participate in that reconstruction. Global source failures
return actual zero-row component buffers and a failure for every original point
ID; original momentum and positive-scale validation still runs. The complete
model path includes source-derived UV, R2 and external-field columns, with later
point failures removing only their original lanes.

The admitted producer partial domain is structural: full colour, real masses,
the native physical frame, and coherent ordinary and
counterterm source-scale owners. It no longer requires a particular external-PDG
list, massless external states or the absence of source-derived field contributions. CMS,
formal-colour projected and broader correlated producer rescue
remain outside this admission.

The retained QCD photon control contains 23 loop graphs and six external-field
insertions. Complete Born, bare, UV, R2, field and total results in double,
DoubleFloat and arbitrary precision agree with the unchanged original-projector
control. An overflowing exponential added only to an existing real-input
diagnostic exercises double/DD failure followed by AP60/AP80 agreement, while
physical instructions and scalar values stay unchanged. Additional controls
cover original IDs, empty inputs, higher-policy replacement, captured-snapshot
updates, cached higher views and an injected late field-column failure. The
original producer owners, prepared caches, Python module dictionaries and native
module are preserved. These are correctness checks, not generation or runtime
measurements, and do not extend CMS or direct-R2 certification.

The retained stable `u d~ -> W+ g` owner adds 11 loop graphs and six field
insertions. Copied prepared records preserve the original modules, owner,
snapshots, caches and native backend while exercising all complete components in
double, DoubleFloat and AP80. Guarded source failures, captured-snapshot updates,
AP60/AP80 complete-result rescue, empty/invalid inputs and late field failures
pass. A real FeynKit longitudinal-rest-state failure removes its whole original
point in every carrier; the surviving point agrees with the unchanged AP80
ordinary control. No wavefunction formula or new execution path was introduced.

Producer physical IDs are opaque labels, including the native maximum-u64
value. Helicity/state packing and reduction use local row IDs, with transient
maps back to the caller's point and squared-order sector. Whole-point compaction
preserves those maps across prior state/current failures and later families;
source coefficients, completed components and stability replay retain original
labels. Strict external-state errors use the dependency's structured statuses
and preserve `ValueError`, without parsing messages or assigning an unproved
failing point. Invalid metadata and literals remain hard errors.

The retained W control passes all three arithmetic modes with maximum-u64 and
irregular IDs, exact snapshot updates, complete-result rescue, empty/tail batches
and permutations. Targeted current, local-CT, field and reduction failures test
original labels after compaction, including a nonfinite nominally successful
reduction. Literal singleton/joint sources preserve labels and both directed
terms across the identity and ordered histories. These source checks cover
complete evaluation, not broader correlated producer rescue. No artifact format,
native build, new capacity bound or independent high-precision physics claim is
added.

### Explicit loop_sm scalar declaration compatibility

The supplied MadGraph 3.7 `loop_sm` vertex `V_1001` lists four `G0` fields but
references `SSS1`, whose scalar spin signature has three entries. Its tensor is
the exact constant one, as is the existing four-slot `SSSS1` declaration.
`prepare_loop_sm_scalar_declarations` explicitly selects that existing reference
in a derived immutable source view. The original raw object is retained, the
choice is recorded in its ordinary source metadata, and no coefficient, particle,
restriction or other vertex changes. The helper requires this precise source
shape and rejects changed tensors, species or arity. Generic structural spin
signature validation remains strict. This enables request preparation with
neutral-current QED orders; it does not certify a six-leg process or new CT bases.

### Neutral-current QCD local counterterms

`neutral_gauge_oracle.py` compares the source `R2_GGZ`, `R2_GGGVa` and
`R2_GGVV` Lorentz tensors to freshly generated, unchanged ALOHA routines with
off-shell real/complex momenta, every basis current and arbitrary complex
currents. The native path uses the installed UFO phase and the existing exact
four-dimensional epsilon component library. Independent MadGraph colour
contractions compare the actual adjoint/singlet ports, including non-real
overlaps with ordered generator traces. The 1,158 Lorentz and 25 colour controls
bind only the exact `Tr × epsilon-momentum`, `f × epsilon`, and `d × metric-sum`
source products in `model_neutral_qcd.py`. They provide neither a dimensional
gamma-five prescription nor an anomaly, UV-scheme or full-process certificate.

The source quark two-point slash retains its original momentum leg. The typed
endpoint adapter gives exactly the explicit native representative previously
checked by `fermion_two_point_oracle.py`: contraction of the calibrated vector
current with that momentum. The endpoint-symmetric bispinor identity
`Identity(1,2)` (R2 and UV quark mass insertions) is admitted separately: it has
no momentum or projector, is invariant under the endpoint exchange, and its
lowered source kernels are rechecked against generated ALOHA `R2_QQ_2_0`
(`fermion_mass_two_point_oracle.py`); `g g > t t~` and `u u~ > t t~` compare
with MadLoop including the isolated mass insertions. Counterterms of
colour-singlet scalar couplings to an ordered fundamental pair (e.g. QCD
corrections of quark Yukawa/Goldstone vertices) use the same exact FFS bases and
physical frame as the tree vertices; `e+ e- > t t~ h` matches MadLoop for
every component. Family denominators reach the reducer in loop-cycle order
(`kinematics.py`), the conventional D0 argument order: oneloopmaster 09b2c74
misevaluated a soft-divergent box with massive emitters, such as the
(m, m, 0, m) boxes of these processes, when its propagators were labelled
non-adjacently (finite without the soft pole in binary64, NaN in DoubleFloat).
The pinned 8ec256fc relabels such boxes inside its numerical entry points; the
order is kept. Other two-point chiral, reversed-endpoint or momentum-leg tensors
remain outside this admission. Scalar source coefficients,
loop groups, matching and all process-level cancellations are independent gates.

### Flavour-changing Dirac/vector counterterms

The source FFV admission uses its exact calibrated Lorentz tensor, ordered
anti-fundamental/fundamental representations and exact `Identity` or `T` colour
tensor. It does not require equal fermion species or infer endpoint direction
from PDG signs. Actual particle identities, loop-particle alternatives, coupling
orders, scalar expressions and Laurent coefficients remain source-owned. No
mass, Yukawa or CKM parameter is equated by this local admission.

`charged_ffv_oracle.py` generates unchanged ALOHA `FFS4_1` and `FFS4_2` currents
with unequal masses, independent complex Yukawas and off-shell complex momenta,
then contracts them with `FFV2_0`. Both charge orientations of the source
`bbar–t–W-` and `tbar–b–W+` rule match the reference. Raw UFO endpoint slots are
checked before the existing typed endpoint translation; exchanging the current
bindings follows that declared frame change. No new gamma or colour algebra is
introduced. Negative local tests retain rejection of changed colour tensors,
reversed ports and uncalibrated Lorentz structures.

The independent complete-process control is `u ubar -> d dbar` with unrestricted
Wolfenstein CKM, mixed QCD/EW Born sectors and **QCD corrections only**. The full
SM source retains photon, Z and W Born exchange and the charged R2 insertions.
All three Born squared orders and all three virtual squared orders, including
both contributing Born–loop pairs at QCD4/QED2, agree with native MadLoop Laurent
outputs at three exact massless points. The fixture records averaging, physical
couplings, scales, restrictions and all changed model parameters. This is not a
validation of jointly selected QCD+QED corrections.

The same saved precision-independent plan is checked through the shared native
executor in double, DoubleFloat and arbitrary precision, including resolved
helicity reconstruction, original lane identifiers and parameter increments
below binary64 resolution. The provider comparison remains a binary64 check;
precision-preserving transport and internal inclusive/resolved agreement do not
turn its reference values into an arbitrary-precision oracle.

### Exact semantic colour views of source scalars

`prepare_source_color_scalars` prepares a cold symbolic view of an existing
source scalar program. Callers supply explicit ordinary-parameter identities,
for example `CA -> Nc` and `CF -> (Nc**2-1)/(2*Nc)`. The preparation checks their
exact agreement with the original source at `Nc=3`; agreement only at the default
external parameter values is insufficient. No numeric constant is interpreted
as a colour invariant. Additional symbols must belong to the source binding
layout or to the original exact expression.

The identity is substituted before dependent source declarations are expanded.
Its original active diagnostic remains paired with its value, including mutable
external-input conditions and inactive branches. Replacing an adjustable
external parameter is rejected unless it was explicitly fixed at generation;
ordinary native updates then reject changes to that fixed parameter. This avoids
silently accepting parameter updates that the colour view would ignore.

`SourceColorScalarPlan` retains the formal paired plan and an ordinary
`ScalarBindingPlan` containing Nc-independent coefficient outputs. `grades` and
`columns` map each colour grade back to the original scalar output. The existing
output epsilon powers are unchanged. Symbolica and the existing colour
projection helpers perform Laurent collection, including Nc-independent runtime
conditionals. Nc-dependent diagnostics, unsupported predicates and non-Laurent
functional dependence fail explicitly. Every coefficient retains its source
diagnostic, including exactly zero coefficients; an all-zero catalogue has a
zero grade solely to preserve diagnostic evaluation, not to set a physical
leading colour degree.

The coefficient plan shares its `ScalarProgram`, external layout, exact defaults
and fixed-parameter map with the ordinary source owner. `NativeScaleBindings`
evaluates it using the same atomic global snapshot and native per-point scale
rows. Saved prepared views retain this behavior in double, DoubleFloat and
arbitrary precision, without retaining a source producer or a second numerical
parameter engine.

This boundary does **not** certify a model's general-Nc continuation or enable
public LC/NLC generation. Physical integration still requires exact scalar-alias
incidence on local tensor terms, a common structural degree per squared-order
sector across bare/UV/R2 and all pole coefficients, and a formal continuation of
every selected effective rule. In particular, the fixed-Nc four-gluon R2 source
cannot be generalized by substituting its named CA/CF parameters alone.

`scalar_alias_incidence` supplies the next preparation boundary for counterterm
colour grading. It asks Symbolica to collect each existing local coefficient
kernel in its counterterm aliases and input placeholders, then propagates only
alias support and homogeneous degree through the DAG. Final scalar outputs must
have degree one in the selected insertion aliases; zero outputs are accepted.
Nonpolynomial alias/input dependence and alias-independent or nonlinear
contamination fail with the offending kernel/output identity.

Incidence support alone must not set the leading colour degree: different
nonzero alias coefficients may cancel within a tensor term. A separate exact
grade pass substitutes their values into each local kernel before collecting it,
propagates proven-zero components, and uses canonical references for equal local
expressions with equal dependencies. Thus `(g1+g2)*J` vanishes for a grade with
coefficients `(1,-1)` and cannot set a spurious leading degree. No nonzero
whole-current expression is expanded into a downstream kernel. Distinct remaining
currents define structural support independently of numerical helicity or
parameter cancellations.

`NativeModelVirtualProgram` uses these grade-aware masks in place of its blanket
rejection of mixed-grade, multi-term insertions. The source-model continuation
gate remains in place; incidence and local zero propagation are not by themselves
a general-Nc model certificate.

### Projection by squared coupling sector

The cold physical projection now uses the existing exact order-pair catalogue.
A rectangular Born–loop matrix cell belongs to the sector of its left Born order
and right loop order; a Born matrix cell similarly belongs to its Born squared
order. Counterterm and external-field rows supply the left order and their
actual insertion graph order explicitly. Missing pairings are zeroed and cannot
set a sector's leading degree. Each virtual sector combines its bare, UV, R2,
field and epsilon rows before selecting a common structural degree. Born sector
degrees are independent, and structurally zero sectors retain `None` on their
original catalogue axis.

The right insertion order is established before projection: source preparation
groups every actual CT entry by vertex, loop-particle-group and exact coupling
order, and `_rule_orders` checks every native coupling in each resulting rule.
Graph snapshots sum these homogeneous vertex orders; matching removes only the
auxiliary CT insertion tag. A focused source control checks one vertex with
multiple coupling orders and loop groups, preventing a representative coupling
from silently determining the whole rule's order.

Projection reassembles the same rectangular matrices and recompiles the existing
weight evaluators. It does not duplicate coherent currents or create another
numerical engine. `PhysicalColorProjection.born_degrees` and `virtual_degrees`
retain the catalogue-aligned degree vectors; the older singular degree fields
are set only for a single corresponding sector.

Retained model-rule views now preserve these exact projected weights, their
catalogue-aligned degree vectors and their declared accuracy while rebinding
arithmetic. The PACBIN plan has an optional `color_projection` descriptor with
the same accuracy and degree vectors; its numerical evaluator states already
contain the fixed-Nf projection evaluated at Nc=3. Core loading checks the
accuracy and order-axis lengths. An absent descriptor describes existing
unprojected full-colour plans. Portable binding restores the saved accuracy,
instead of labelling every loaded plan as full colour. Projected dimensional
numerator/direct-R2 transport remains explicitly unsupported in this slice.

Complete model-rule Drell–Yan controls cover full/LC/NLC PACBIN reload and
rebinding in double, DoubleFloat and arbitrary precision, including exact
parameter changes and native helicity-row reconstruction of every inclusive
component and Laurent coefficient. Mixed-order bare controls separately retain
all Born and virtual degree axes through rebinding. Publication checks compare
declared accuracy with the saved descriptor: the current full-only public
boundary cannot hide an LC/NLC projection. Source-model continuation and public
resolved LC's signed Born-component contract remain integration requirements
before public source LC/NLC can be enabled.

### Explicit four-gluon QCD rational continuation

`qcd_r2_color.prepare_qcd_four_gluon_r2` supplies a whole-rule HV continuation
from [2007.03713, Eqs. 5.30–5.38](https://arxiv.org/abs/2007.03713v2), using
`Tr(Ta Tb)=delta_ab/2`. Its scalar coefficients carry formal Nc, while native
fundamental/adjoint tensors retain the physical source port frame for the
existing `SUNColorSpace` lift. The library performs tensor construction and
contraction; pyAmpliCol declares the published local rule.

Preparation requires an explicit source group, gauge-coupling expression,
contribution (`gauge` or `fundamental-dirac`) and HV convention. It compares the
complete original source tensor at Nc=3, including scalar coefficients. Both
source and replacement lie in the checked three-metric-product span; every
SU(3) colour component of three independent Lorentz components is compared
exactly. The active 57-entry UFO gives 24,576 component agreements. A separate
formal-Nc test uses [0903.0356, Fig. 2](https://arxiv.org/abs/0903.0356v2), a
different published decomposition, rather than reusing the implementation.

The immutable record is keyed by original collection, vertex, loop group and
order. It retains the original entry objects, including source diagnostics and
all loop alternatives. One Dirac fundamental species per alternative and the
complete Yang–Mills gauge/ghost contribution are this local rule's certified
domain; the single-species requirement is not a general loop-group capacity
limit. No Nf aggregate or per-entry substitution is inferred, and source/model
names play no role in conformance. A partial group, incompatible species,
changed scalar normalization or unsupported scheme fails explicitly.

This local preparation alone does not alter process counterterm generation.
The explicit source binding described below applies the replacement once per
eligible original group, retains source diagnostics and species multiplicities,
and enforces HV parameter ties before the common sector colour projection.
Local Nc=3 agreement alone is not broader model, UV or scheme certification.

### Source-owned continuation and coefficient grades

`SourceColorContinuation` is explicit prepared-model data: the actual source
model, formal Nc, exact parameter identities, declared source kernel and
particle-field keys, and the existing complete scheme declaration. It does not
infer colour semantics from a model name or numerical constants. Missing
continuation for a selected kernel/field names that exact key and fails the
projection request. Whole rules whose numerical SU(3) coefficients do not define
their continuation still need independently checked replacements.

The generic `source_model_program` accepts this internal continuation. Before
projecting, it checks every selected ordinary scalar output, including derived
masses and couplings, for exact Nc independence across the full external
parameter space. Nontrivial ordinary scalar colour dependence is currently an
explicit unsupported feature; it cannot pass by matching default numbers.
Original conditional diagnostics remain attached to the exact source formulas.

Model UV/R2 and external-field coefficients then use one shared source owner
with a graded `NativeScaleBindings` view. Each current's alias columns select
its exact degree and epsilon coefficient. The existing local incidence and
zero-propagation proof controls its colour weights; zero field coefficients
cannot raise a sector's leading degree. The original diagnostic accompanies
every grade. Saved plans retain these ordinary exact scalar/evaluator states
and source column maps, so numerical evaluation uses the same batched shared
core and precision rebinding path as full colour.

Source Drell–Yan controls check the complete model-rule result in all three
carriers and through saved plans. The independent fundamental colour factor
predicts LC/full = 9/8 for every virtual component and NLC = full. The tests
also vary exact inputs and per-point scales and reject a missing selected rule.
These controls supply explicit model data and exclude the fixed-SU3 four-gluon
R2 rule; they are not a universal loop_sm continuation certificate or public
LC/NLC activation. Public resolved LC must still supply its signed
Born-component observable contract.

An unrestricted-CKM source `u u~ > d d~` control additionally preserves all three
Born and virtual squared orders and matches the independent MadLoop finite and
pole values after formal-full projection. Its correction selection remains QCD,
not joint QCD+QED. A source `u u~ > g g` control covers nonzero external-gluon
field renormalization: the heavy-flavour contribution is absent at LC and
retained at NLC, and its graded source columns survive PACBIN loading. Vanishing
Laurent components are compared at Born scale to account for cancellation
roundoff, without discarding any returned coefficient.

### Binding whole-rule colour input to a source process

The source coefficient consumer keeps the original CT coupling outputs in its
paired scalar plan when a checked whole rule is substituted. Every original
active diagnostic is evaluated even if none of the new coefficients uses the
old value. Replacement aliases select separate exact local coefficients. Their
referenced ordinary source parameters are resolved by the existing scalar
frontend and remain diagnostic-bearing outputs of the same source owner;
neither source defaults nor intermediate values pass through binary64.

The complete formal plan is graded once. Counterterm recurrence incidence uses
the active replacement aliases, while matching, loop alternatives, coupling
orders and signs continue to use the original rule. This retains per-species
eligibility and avoids assigning guessed Nc factors to the original numerical
four-gluon coefficient table. Saved plans carry the resulting ordinary
coefficient evaluator bytes and column maps, with no separate numerical path.

`SourceColorContinuation` now checks the identity of every actually applied
whole-rule input against the declaration. Scheme parameter ties use the existing
shared `fixed_parameters` mechanism: a required value must hold symbolically over
the entire runtime input domain. An external scheme parameter must therefore be
fixed during preparation, and an attempted update is rejected atomically. An
internal expression must reduce to the required value with the same fixed map.
Scheme declarations alone cannot establish these ties.

The explicit test-support profile
`tools/loop_reference/qcd_source_color.prepare_loop_sm_four_gluon_color` supplies
the source names `G` and `lhv`, always requires HV with `lhv=1`, and runs whole
tensor/scalar/species conformance on each selected four-adjoint-vector R2 group.
It preserves all unrelated caller declarations and original CT evidence. Model
display names do not grant admission, and this helper gives no certificate to
unrelated rules. It returns the prepared model with the checked native rules and
the matching source colour declaration; those objects must be used together.

These remain internal prepared-model interfaces. Public LC still requires its
signed Born-component observable contract, and a generic model needs declared,
validated continuation for every selected ordinary, UV, rational and field
structure before projection is available.

### Retained directed projectors on a source owner

`PreparedDirectedBare` and `PreparedDirectedModel` retain immutable dependency
instruction bytes for the directed Born contraction, each denominator family,
local UV/R2 and external-field terms, and final complex assembly. They refer to
existing retained bare/model plans. Binding a different arithmetic carrier uses
those bytes and the original source scalar/current owner; it does not regenerate
colour algebra, diagrams or counterterm currents. A different plan is rejected.
`NativeDirectedModelProgram` accepts a `NativeSourceModelProgram` through its
existing `.virtual` owner.

The two Born–loop terms remain separate through complex reduction. Only then is
the swapped second term conjugated. Local rules retain their individual order
pairings; external fields retain the corresponding Born-graph mask and delta-Z/2
factor. Ordinary real-result projection is not applied to directed histories.
The complete Python control wrapper uses dense internal helicity/reduction rows
and reports original physical point IDs separately from the direction/order/
family catalogue, including IDs at the native integer boundary. This is not a
partial-batch scheduler or precision-rescue implementation.
Early state/current exceptions and some finite-value guards still expose the
legacy dense row context; typed original-ID failures at every stage belong to
the subsequent native correlated scheduler.

The actual source `u u~ > d d~ [QCD]` control covers identity, a colour Casimir,
complex swapped histories, all UV/R2 components, two exact on-shell points and
scales, all three arithmetic carriers, and below-binary64 parameter changes.
Independent analytic Born/UV expressions supplement transport checks against the
ordinary source projection; sharing currents is not independent physics
validation. Source-directed artifact serialization, native/public correlation
scheduling, spin-source transport/compaction, correlated LC/NLC and flavour
mappings remain subsequent work. Existing internal spin controls remain separate.
