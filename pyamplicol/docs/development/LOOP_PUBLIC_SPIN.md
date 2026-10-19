# Public virtual spin sources

The existing correlated API accepts declared singleton or joint massless-gluon
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
