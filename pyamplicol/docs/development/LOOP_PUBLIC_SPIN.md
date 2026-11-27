# Public virtual spin sources

> Historical: this describes the per-diagram legacy one-loop engine, removed on
> 2026-10-29 (MQ-LEGACY-REMOVE; v2 is the only one-loop engine, see
> `loop_v2/DESIGN.md` and §13.3 there). The modules, tools and tests named
> here may no longer exist.

The existing correlated API accepts declared singleton or joint vector source
classes for QCD, with complete model-rule R2 and UV. Validated cases include
photons/gluons and the stable real-mass W± controls described below. Full-colour fixed-Nc
kernels support the admitted ordered colour histories. Formally projected LC/NLC
kernels support identity histories as described below. Generation
uses `CorrelatorConfig(spin_correlations=((3,), (4,), (3, 4)))`; the labels are the
same one-based physical external legs used by tree correlations. The saved plan
contains each requested class and the ordinary physical-helicity projection.
The generator validates every class through the normal transactional source versus
saved-runtime check before publication.

```python
runtime.set_spin_correlation_vectors({3: [0, 1, 0, 0]})
projected = runtime.evaluate_correlated(points, mu_r2=scales_squared)
grouped = runtime.evaluate_correlated_many(
    points,
    {
        "projected": CorrelatedRequest("born"),
        "physical": CorrelatedRequest("born", spin_vectors={}),
    },
    mu_r2=scales_squared,
)
```

Each vector is literal: no normalization, transverse projection or implicit
conjugation is added. A four-vector broadcasts; a list of vectors has one entry
per event. Decimal real components, strings and exact real/imaginary pairs remain
exact until the selected carrier receives them. The keyset must equal a declared
joint class. `None` inherits a snapshot of the runtime setter, while an explicit
empty map selects physical helicities. Per-call requests never change the setter.
A failed setter leaves previous defaults intact; `clear()` releases numerical
workspaces and preserves them. Empty event batches accept empty per-event vector
lists without losing their joint class identity.

The actual compiled model supplies each vector's `SourceIR`. Preparation checks
it against the physical port, actual particle/antiparticle identity and structural
two-state massless or three-state massive vector layout. Tree-adapter binary64 mass defaults do not decide loop mass
classification or enter its exact scalar owner. Raw internal producers supply
explicit contracts. The immutable source class retains its bare-plan owner,
contracts and compiled phases; the writer consumes these rather than rebuilding
a built-in gluon convention. Saved loading reuses the existing typed SourceIR
parser and checks the physical identity, orientation and states. Earlier internal
spin payloads require regeneration; no silent schema migration is applied.

The source phase and spectator-helicity projection are compiled once. Every call
replaces the selected external columns before the shared Born, loop, UV and R2
currents. The original incoming average is retained and replaced legs are counted
once. Identical exact source assignments share one native batch across all colour
histories. Different assignments each execute a native event batch. Python only
validates/formats inputs and selects result columns; it does not evaluate events
or contract amplitudes. Precision rescue retains original source strings and
recomputes failed points through the existing shared coordinator.

Results remain correlated `VirtualEvaluation` objects with separate Born orders,
all virtual squared orders and finite/single/double-pole coefficients. Binary64
returns complex values; DoubleFloat and arbitrary precision preserve decimal
real/imaginary components. `inspect()["spin_correlations"]` lists the actual saved
classes. Ordinary `evaluate` is independent of the spin setter.

Resolved correlated output, nonidentity LC/NLC colour histories,
neutral massive/chiral source cases and general EW corrections remain
unvalidated follow-ups. The independently
proved full-colour identity-order d/s flavour reuse is described in
`LOOP_FLAVOR_TRANSPORT.md`; formal-colour flavour reuse is a separate requirement.
No additional execution engine or fixed capacity limit is introduced.

Validation includes independent Recola singleton-gluon Born/all-pole projections
at two points/scales, and a joint linear-polarization sum reconstructing the
ordinary result. Recola is a binary64 reference; agreement does not establish
70 independent digits or non-Hermitian multi-emission imaginary histories. The
separate internal source controls exercise complex histories, exact changes below
binary64 resolution and AP60/AP80 subset replay.

The photon extension adds fresh public `u u~ -> a g` and crossed `u a -> u g`
artifacts with photon, gluon and joint classes. Complete ordinary and singleton
Born/Laurent coefficients agree with Recola in double, DoubleFloat and AP70.
Joint polarization sums, literal complex source changes below binary64
resolution, clear/empty/tail batches, and AP60/AP80 subset rescue also pass.
Malformed saved source identities/states are rejected. A safe-Rust client loads
the fresh photon artifact and checks source scaling, rejection and result
ownership against the actual newly built SDK.

Recola's spin API projects through physical helicities. Its independent photon
controls therefore use transverse complex vectors; they do not certify arbitrary
nontransverse source extension. That literal contract is checked through the
source-versus-saved recurrence comparison and exact homogeneity/replay, with no
implicit projection or conjugation. The reference uses `aEWM1=132.50698`, while
the supplied source default is `132.507`; the runtime binds the matched exact
value. Reference data, driver and conventions are in
`tools/loop_reference/PHOTON_SOURCES.md`. These photon controls do not imply
additional massive-vector or EW-correction certification.

## Stable charged massive-vector controls

The actual model-owned W± SourceIR now passes through the same source preparation
and saved-runtime path. No vector wavefunction formula was added: dependency
wavefunctions supply all three states, including the longitudinal state. Saved
loading also reuses the existing canonical and reciprocal antiparticle identity
checks across source classes. A three-state contract cannot be relabelled as a
massless state space; contradictory identities are rejected.

Complete full-colour QCD controls cover `u d~ -> W+ g` and crossed `d W+ -> u g`,
plus charge-conjugate `d u~ -> W- g`, with zero W width and matched exact input
parameters. Ordinary Born and all three
loop coefficients agree with MadLoop. Independent Recola coherent projections
use its own massive wavefunctions and three-state amplitude columns, rather than
its two-state public spin-projection helper. The outgoing/incoming maximum
relative differences are approximately `2.9e-13` and `3.2e-9`; these are binary64
physics comparisons. The W− coherent result agrees to approximately `4.3e-13`,
and its ordinary result agrees with MadLoop to `4.8e-15`. Its actual model
antiparticle orientation and reciprocal PDG labels are retained, without deriving
phases from the PDG sign. Fixed-helicity MadLoop/Recola controls also agree, with the
incoming spin average translated explicitly.

Native double, DoubleFloat and AP70 controls reconstruct every complete component
for all three processes from the three W states and the six joint W/gluon states, retaining both directed
interference terms and ordered histories. They exercise exact source changes
below binary64 resolution, arbitrary literal complex vectors, parameter snapshots,
empty/tail batches and original point IDs. Public saved artifacts pass coherent
reference checks, AP60/AP80 subset rescue and `clear`. An unchanged standalone
Rust source API loads the fresh artifact and checks the longitudinal result and
owned source labels.

The ordered-history checks are internal reconstruction and homogeneity controls;
the independent provider comparisons select the identity colour history.

This certifies the exercised stable W± QCD cases. It does not certify
Z axial closed-loop sums, electroweak corrections or complex external masses.
The narrower declared-colour composition below is a separate control. Recola only supplies an independent oracle for
the physical three-state subspace. Arbitrary nontransverse source behavior is
checked by literal recurrence transport and exact homogeneity. These tests are
not generation/runtime measurements. Driver and convention details are in
`tools/loop_reference/MASSIVE_VECTOR_SOURCES.md`.

## Declared-colour stable W composition

The explicit source-colour example includes the actual sixteen additional
ordinary/UV/R2 keys selected by `u d~ -> W+ g` QCD. Two separately inventoried
charged tree/R2 entries cover crossed `d W+ -> u g` and outgoing
`d u~ -> W- g`, retaining the source's conjugated CKM coupling. They preserve
the source's fundamental `Identity`/`T` tensors, named Nc-dependent invariants, loop-particle
groups, mass logarithms and existing `lhv=1` tie. Removing either the selected
charged tree entry or its R2 entry fails the existing declaration gate. No
process-specific compiler exception or scalar/tensor algebra was added.

Actual request owners in the corrected complete Community environment establish
the same common leading degrees, Born 2 and virtual 3, across the bare loop and all
UV/R2/field rows. The exact 322-row control includes 60 rows at the odd suppressed
power: LC removes them and NLC retains them. Full projection reconstructs every
unchanged complete component in double, DoubleFloat and AP70, using a
component-local Laurent norm for cancellations rather than division by Born.
The complete full-colour result retains the prior independent MadLoop control;
LC/NLC are checked by formal coefficient projection, not a finite-Nc provider.

Saved LC/NLC identity-source artifacts reconstruct the complete answer from the
three W states and the six joint W/gluon states in all three arithmetic modes.
They retain exact source and parameter changes, original opaque IDs, empty/tail
batches, both interference terms and literal complex-vector homogeneity. Normal
public artifact publication preserves ordinary LC signed Born anchors and NLC
contracted output; source-correlated LC/NLC output is contracted. All-carrier
resolved totals and saved public AP60/AP80 subset rescue use the existing APIs.
These controls add no generation or runtime timing measurement.

The crossed W+ and outgoing W− controls use their actual incoming/antiparticle
SourceIR and dependency wavefunctions. Each independently preserves full-colour
coherent Recola agreement while LC/NLC use the exact formal projection oracle.
All-carrier three-state/joint-source reconstruction and public subset rescue also
pass for these two channels, without inferring a phase from the particle's sign.

This covers the exercised stable W± QCD channels with identity histories.
Nonidentity formal-colour histories, EW corrections and complex masses remain
open. The corrected typed open-port Community package is required for this exercised preparation; existing
ordinary package dependency adoption is tracked separately. Unrelated original
full-colour owners and their numerical snapshots remain untouched.

## Projected identity histories

LC/NLC source classes reuse the complete model's already projected Born/loop
matrices and local/field counterterm weights. The common structural leading
colour degree remains the one established across each complete squared-order
sector, including UV and model R2. Source replacement and spectator helicity sums
do not recompute that degree. Both integrated interference directions remain
present, even for the identity observable.

Generation admits empty bra/ket histories, including named identity aliases,
and any actually supported declared joint source classes. It rejects requested
nonidentity histories and automatic colour-insertion catalogs before preparation.
Formal full-colour preparations also use this identity path until the graded
ordered-operator banks are available; fixed-Nc full-colour histories retain their
existing implementation. No guard is relaxed around an unprojected metric.

The saved descriptor convention is `sun-graded-identity-all-outgoing-v1`.
The native validator requires an actual colour projection and empty histories in
every ordinary/source descriptor. Correlated projected artifacts are inclusive:
they expose one contracted colour component, no resolved axes and no signed Born
anchors. Ordinary LC artifacts still require their canonical Born-component
partition and resolved payload. Python and standalone shape/capability queries
follow that actual partition; virtual contributions are never probabilities.

Actual retained four-gluon LC and independently generated public NLC controls
exercise singleton and joint polarization completeness against each ordinary
complete component, both interference directions, literal complex scaling, zero
sources and exact source changes in double, DoubleFloat and AP70. Original point
IDs, irregular source batches, empty classes and scalar rebinding are retained.
Public LC additionally checks AP60/AP80 subset rescue and `clear`; a packaged
Rust client loads the saved projected artifact and checks the contracted shape,
source ingress and owned output labels. NLC publication uses the ordinary public
generator's source-versus-saved checks, with its actual owner retained separately
from LC. These are correctness controls, not new timing measurements or an
independent 70-digit external-provider reference.
