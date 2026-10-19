# Public virtual spin sources

The existing correlated API accepts declared singleton or joint photon/gluon
source classes for full-colour QCD, with complete model-rule R2 and UV. Generation
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
it against the physical port, identity, representation and structurally massless
two-state domain. Tree-adapter binary64 mass defaults do not decide loop mass
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

Resolved correlated output, LC/NLC source composition, public flavour mapping of
spin classes, massive vectors and general EW source kernels remain unvalidated
follow-ups. No additional execution engine or fixed capacity limit is introduced.

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
`tools/loop_reference/PHOTON_SOURCES.md`. No new performance measurement or massive
vector/EW-source certification is implied.
