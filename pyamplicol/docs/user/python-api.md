---
title: "Python API"
nav_order: 4
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Python API

The public Python interface offers typed services for model compilation,
process planning/generation, runtime evaluation, Born correlations, selectors,
and profiling.
Importing `pyamplicol` is lightweight: Symbolica and model tooling are loaded
only when the requested operation needs them.

```python
import pyamplicol

print(pyamplicol.__version__)
```

## Public service map

| Service | Purpose |
| --- | --- |
| `ModelSource` | Resolve and compile built-in, JSON, UFO, compiled, or prepared models. |
| `Generator` | Plan or generate schema-v3 process artifacts. |
| `Runtime` | Load one concrete process and evaluate totals/resolved components. |
| `CorrelatorConfig`, `ColorCorrelator` | Declare allowed spin replacements and named colour operators at generation. |
| `CorrelatedValue` | A correlated result with `Decimal` real and imaginary parts. |
| `BenchmarkRunner` | Profile an artifact or loaded runtime. |
| `group_processes` | Group the subprocesses of an artifact with equal matrix elements. |
| `generate`, `load`, `benchmark` | One-shot convenience functions. |

The shipped scripts under
[`examples/python/`](https://github.com/MadGraphTeam/pyamplicol/tree/main/examples/python)
are complete runnable programs.

## Plan and generate with the built-in model

```python
from pyamplicol import GenerationConfig, Generator, ModelSource

generator = Generator(GenerationConfig(workers=2))
model = ModelSource.built_in_sm()

plan = generator.plan("d d~ > z g", model=model)
for process in plan.concrete_processes:
    print(process.name, process.expression)

result = generator.generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=model,
    mode="replace",
)
print(result.output)
```

`plan()` resolves the model/process/color/evaluator contract without writing a
process artifact. `generate()` returns an immutable `GenerationResult` with the
absolute output, stored process set, schema version, and file inventory.

The packaged scalar HEFT model uses the same public selector:

```python
from pyamplicol import Generator, ModelSource

heft = ModelSource.built_in_sm_heft()
Generator().plan("g g > H g g", model=heft)
```

Use an explicit `HIG = 1` coupling-order limit in a resolved run configuration
when generating the process; see [Models and Processes](models-and-processes.md#built-in-scalar-heft-model)
and the packaged `builtin_sm_heft.toml` card.

## Generate with adjoint FFT contraction

The Python API uses the same `RunConfig` and `ColorConfig` fields as TOML and
the CLI. Defaults `contraction="auto"` and `fft_basis="adjoint"` try adaptive
exact FFT for NLC/full recurrence and on-the-fly execution, with direct fallback
for unsupported FFT plans. This certified pure Yang–Mills example uses DDM:

```python
from pyamplicol import ColorConfig, Generator, ModelSource, RunConfig
from pyamplicol.config import EvaluatorConfig

config = RunConfig(
    action="generate",
    color=ColorConfig(accuracy="full"),
    evaluator=EvaluatorConfig(execution_mode="recurrence"),
)
result = Generator(config).generate(
    "g g > g g g",
    "artifacts/ggg_adjoint_fft",
    model=ModelSource.built_in_sm(),
)
```

Select `accuracy="nlc"` for NLC contraction, or `execution_mode="on-the-fly"`
for a compact OTF artifact. The effective configuration can retain `"auto"`
until per-process selection. LC, compiled/eager execution, and `correlators=`
use direct contraction and the trace basis under the automatic policy.

Use `ColorConfig(accuracy="full", contraction="direct")` to opt out of FFT;
direct contraction ignores `fft_basis`. To force FFT and retain errors for
unsupported plans, set `contraction="symmetric-group-fft"`, optionally with
`fft_basis="trace"` for the original representation. The string `"adjoint"`
is equivalent to `ColorFFTBasis.ADJOINT` (imported from `pyamplicol`). Explicit
FFT does not support compiled/eager modes or LC accuracy.

For certified trees with `n` external gluons, DDM fixes two anchors and reduces
the trace set from `(n-1)!` orderings to `(n-2)!`, representing the same amplitude.
Certified single-insertion HEFT also uses DDM; specify explicit coupling orders
with `HIG = 1`, as in `examples/builtin_sm_heft.toml`. Quark processes keep exact
fundamental chains or their products, not a newly assumed JO primitive basis;
uncertified adjoint interactions retain trace tensors. The saved
`fft_basis_selection` reports `requested`, `actual_basis`, `name`, `reason`,
`permutation_degree`, and `tensor_count`, making that choice explicit.
Forced adjoint FFT cannot be combined with `correlators=`. Runtime totals,
resolved helicities and native SDK evaluation calls are unchanged. No universal
speedup is implied.
In the benchmarked pure-gluon family the adjoint basis was faster than trace
for six or more external gluons (3.9x per sample at ten gluons); benchmark
other multiplicities and selector workloads.

## Generate a named process set

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("u u~ > Z g", name="uubar_Zg"),
        ProcessRequest.parse("u u~ > Z g g", name="uubar_Zgg"),
    )
)

result = Generator().generate(
    processes,
    "artifacts/z_ladder",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
```

Process names must be unique. For custom aliases and crossing metadata, use
`ProcessAlias`; runtime expression matching already handles unique
side-preserving permutations automatically.
The correlated reference path currently requires identity ordering; these
permutation facilities apply to ordinary evaluation.

## Compile and reuse an external model

```python
from pyamplicol import Generator, ModelSource

source = ModelSource.from_path(
    "models/json/sm/sm.json",
    restriction="default",
)
model = source.compile()

print(model.info.name)
print(model.capabilities.supported_color_accuracies)
print(model.supported)

Generator().plan("d d~ > z g", model=model)
```

For a trusted UFO directory:

```python
source = ModelSource.from_path(
    "/path/to/MyUFO",
    restriction="restrict_default.dat",
)
model = source.compile()
```

UFO modules execute Python while loading. See
[Models and Processes](models-and-processes.md) for trust, serialization, and
prepared bundles.

## Use a fully resolved run card

For complex configuration, resolve the same schema used by the CLI and pass the
result to `Generator`:

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config

card = {
    "schema_version": 1,
    "action": "generate",
    "model": {"source": "models/json/sm/sm.json", "restriction": "default"},
    "process": {
        "entries": [{"expression": "p p > Z j j"}],
        "multiparticles": {"p": ["d", "d~", "g"], "j": ["d", "d~", "g"]},
        "flavor_scheme": 2,
        "max_quark_lines": 2,
    },
    "color": {"accuracy": "lc"},
    "generation": {"output": "artifacts/pp_zjj", "mode": "replace"},
    "evaluator": {"execution_mode": "compiled", "backend": "jit"},
}

resolution = resolve_config(card)
model = ModelSource.from_config(resolution.effective.model).compile()
processes = ProcessSet(
    tuple(
        ProcessRequest.parse(entry.expression, name=entry.name)
        for entry in resolution.effective.process.entries
    )
)
result = Generator(resolution).generate(
    processes,
    resolution.effective.generation.output,
    model=model,
    mode=resolution.effective.generation.mode,
)
```

The complete maintained implementation is
[`examples/python/typed_generation.py`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/python/typed_generation.py).

## Load a runtime

```python
from pyamplicol import Runtime

runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
    model_parameters={"aS": 0.117, "MZ": 91.1876},
)

print(runtime.artifact_id)
print(runtime.execution_mode)
print(runtime.physics.process_id)
print(runtime.physics.process)
print(runtime.representative_process_key)
print(runtime.external_permutation)
```

`process` may be a stable process ID, explicit alias ID, exact stored expression,
or unique permutation-equivalent expression within each side. The loaded
runtime exposes metadata in the requested public order.
For correlated evaluation of a tree artifact, load the stored process in its
generated ordering; nonidentity aliases/permutations are not supported by that
path yet. A one-loop (virtual) artifact serves its aliases, side-permuted
expressions and exact process-group members in its evaluation calls,
correlated and resolved ones included: colour histories and spin legs are read
on the selected process's legs and relabelled onto the stored process's, and a
declared one whose relabelled image is not stored is not served. Helicity and
colour-flow selection of such a mapped process is refused; select on the
stored process (see
[Process grouping](process-grouping.md#one-loop-virtual-artifacts)).

## Evaluate totals and resolved components

```python
import json
import math
from pathlib import Path

points = json.loads(Path("data/pp_zjj_momenta.json").read_text())

totals = runtime.evaluate(points)
resolved = runtime.evaluate_resolved(points)

print(resolved.helicity_ids)
print(resolved.color_ids)
for optimized, explicit in zip(totals, resolved.total(), strict=True):
    assert math.isclose(optimized.real, explicit.real, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(optimized.imag, explicit.imag, rel_tol=1e-12, abs_tol=1e-15)
```

Input shape is:

```text
(point, external particle, [E, px, py, pz])
```

At LC, resolved shape is `(point, helicity, color_flow)`. At NLC/full,
the color dimension has length one because color is contracted.

## One-loop virtuals and loop-induced processes

A one-loop artifact evaluates to a `VirtualEvaluation` (per point, squared
order sector and Laurent coefficient `[finite, 1/eps, 1/eps^2]`):
`contribution = "virtual"` for the Born-virtual interference, `"loop-induced"`
for |A_loop|^2 of a process without a Born amplitude (`g g > h`; see
[Configuration](configuration.md#processes-and-multiparticles)). A
loop-induced result is the same type with its Born fields zeroed
(`born_orders == ()`, every per-point Born entry empty), sectors that pair
loop orders, and the pole selections `"all"` and `"none"` only; its computed
poles vanish, a necessary-only finiteness cross-check.

```python
from pyamplicol import Runtime, VirtualStabilityChecks

runtime = Runtime.load("artifacts/gg_h", model_parameters={"MT": "173"})
value = runtime.evaluate(
    points,
    mu_r2="15625",
    stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
)
assert value.contribution == "loop-induced" and value.born_orders == ()
lane = value.diagnostics[0]
print(lane.reduction)    # VirtualReduction(requested='auto', effective=..., reason=...)
print(lane.finiteness)   # ||a_{-k}||_G against the row's term floor (~1e-15)
estimate = value.stability[0]
print(estimate.absolute, estimate.magnitude, estimate.scale)   # E, S per sector; max E/S
```

The reduction backend is chosen once per call (`reduction="auto"`, `"ninjars"`
or `"tirust"` on `Runtime.load` and on every evaluation; default: the
artifact's `[loop] reduction`): `"auto"` takes TIRust for a loop-induced and
NinjaRS for a Born-interfered process (TIRust for both on a TIR-only
artifact) and falls back to NinjaRS, with the reason recorded, where TIRust
cannot serve the call (a native extension built without TIRust). On TIRust,
`"auto"` has a **per-point NinjaRS fallback**: a point TIRust refuses (a
kinematic row its conditioning gates and then its checked method choice
refuse, or a method-choice value beyond the target, below) or fails (also in
the call's own stability checks, e.g. a rotated evaluation) is evaluated again
on NinjaRS from its original exact inputs, in the call's own arithmetic,
after the TIRust call. That point's lane records its own `VirtualReduction`:
`effective="ninjars"` and the reason, which quotes TIRust's failure; a point
NinjaRS fails too fails with both reasons. A fallback point's cross-backend
check (below) is not run, with the fallback's reason in `not_run`. An
explicit `"ninjars"` or `"tirust"` has no fallback: `reduction="tirust"`
fails such a point closed with TIRust's reason, and `reduction="ninjars"`
forces NinjaRS for every point (for instance for a higher accuracy). A
TIR-only artifact has no fallback (NinjaRS is refused typed there).
`"tirust"` is served in every arithmetic (binary64, double-double and
arbitrary precision, also on every level of a rescue stack). An explicit
backend that cannot serve the call is refused before any point is evaluated
(`pyamplicol.ReductionUnavailable`, a `CompatibilityError` with key
`reduction-unavailable`). The effective backend is always one that the
call's executor runs: the summed `evaluate` and the resolved, correlated and
selected outputs each check it, so a result never names a backend that did
not reduce it. Every lane diagnostic records the requested and effective
backend, the reason, the call's fallback and target
(`VirtualReduction(requested, effective, reason, fallback, target)`;
[Configuration](configuration.md#reduction-backend)).

TIRust serves at a **target accuracy** (`reduction_target=` on `Runtime.load`
and on every evaluation; `[evaluation] reduction_target`; CLI
`--reduction-target`; default 1e-5): the relative accuracy of each point's
result, every squared-order sector against its own size (the rescue's
`"relative"` metric). It sets the acceptance of TIRust's checked method
choice (the rows its conditioning gates refuse, binary64): a method-choice
value is kept when its two frames agree within the target divided by the
calibrated factor 580, the largest ratio of a true error to that frame
deviation measured against consistent high-precision truths (a synthetic lane
set and the loop-induced benchmark rows' points). A tighter target refuses
more such values (served by the fallback under `"auto"`), a looser one admits
more; values TIRust serves at any target are the same values (the acceptance
only admits). The target governs only the values the method choice admits:
values of rows that pass the conditioning gates, and values already admitted
at the shipped TIRust acceptance, carry binary64's own reduction error
(measured up to about 1e-7 relative on the calibration sets), so a target
tighter than about 1e-7 can be missed by construction. The target is a
calibrated acceptance, not a certificate: request stability checks or a
rescue stack for an estimate. NinjaRS has no
setting the target drives, and in double-double and arbitrary precision,
where TIRust's method choice is off, the target acts on nothing. A value that
is not a finite positive number is refused (`ValueError`).

`VirtualStabilityChecks(..., cross_backend=True)` adds the cross-backend
check: every point is evaluated a second time on the other backend (TIRust
for a call on NinjaRS, NinjaRS for one on TIRust) and the deviation of its
reduction lanes enters the estimate as the check `"cross-backend"`. It
samples what neither backend's own frames see (the two reductions round
differently everywhere), costs one more evaluation of every point, and
combines with at least one other check: a point the other backend fails or
refuses keeps the estimate of the other checks and names the check, with
the reason, in its estimate's `not_run` (a refusal of the other backend says
nothing about the value). It is a runtime option of the call, never stored in
an artifact; a carrier the other backend does not serve (in this release
TIRust serves every carrier, so only a native extension built without
TIRust) refuses it before any point (`ReductionUnavailable`). The values are those
of the call without it, bit for bit. It runs on `evaluate` and on the
resolved, correlated and selected outputs of both process types (not on the
Python reference executor, which refuses it).

With stability checks every estimate carries, besides the relative
estimates, the absolute estimate E per sector and Laurent coefficient
(`absolute`), the scale S it is measured against (`magnitude`, the size of
the contributions before they cancel) and their largest ratio (`scale`). A
value with `|value| <= E` is zero within the estimate: zeros of loop-induced
processes (Landau-Yang, Furry) are never declared by a rule, they are
evaluated and reported this way. For a loop-induced result the pole
estimates `single_pole` and `double_pole` are E / S as well (the poles
vanish, so their own size is no scale).

Each lane's `finiteness` is the amplitude-level finiteness ratio of a
loop-induced point: the largest `||a_{-k}||_G / S_k(h)` over its loop
orders, helicity rows and poles, of the order of the carrier's unit
roundoff for a finite amplitude (the judgement the computed poles stand on).
It exists exactly when the call computes the poles and requests stability
checks (any enabled `VirtualStabilityChecks`, or a rescue level with
enabled checks; a bare `VirtualStabilityChecks()` enables none), and is
`None` otherwise, also for `poles="none"`. The reason: its scale S_k(h) is
the row's term floor, the amplitude's size before the reduction's own
cancellations, which needs every family's tensor moments at the point (one
unit-numerator reduction per family monomial and point with NinjaRS:
1.3 times a plain evaluation on `g g > z z`, 2 times on `g g > h`). The
stability checks form it anyway, as the floor of E; a call without checks
pays nothing for it. Without the ratio the computed poles still vanish
against the finite part (the necessary-only cross-check). An artifact
generated with `generation.loop_induced_probes = true` (off by default) also
passed the generation-time finiteness probe, in binary64 or, where binary64
failed or was refused, in double-double (a failed probe refuses the process
unless `generation.allow_failed_probes`, which is recorded); without the
option its record reads `{"status": "off"}`: not run. An exact
colour zero (every colour coefficient of the loop amplitude vanishes at
N_c = 3 by colour algebra, as for `h > g a`, Tr T^a = 0) evaluates to an
exact 0 and its lanes say so.

### Observables of loop-induced processes

The correlated, resolved and selected calls serve loop-induced processes with
the same result types (`contribution = "loop-induced"`), as quadratic forms on
the point's loop amplitude vector (nothing is reduced again; full colour
only). Colour histories and spin classes are declared at generation
(`correlators=CorrelatorConfig(...)`), their colour matrices stored with the
artifact:

```python
from pyamplicol.correlators import CorrelatedRequest

many = runtime.evaluate_correlated_many(
    points,
    {
        "total": CorrelatedRequest("born"),          # |A|^2
        "T1T4": CorrelatedRequest("T1T4"),           # <A|T_1.T_4|A>
        "spin": CorrelatedRequest("born", {4: v}),   # leg 4's wavefunction -> v
    },
    mu_r2=scales,
)
rows = runtime.evaluate_resolved(points, mu_r2=scales)           # per helicity
anchors = runtime.evaluate_resolved(
    points, mu_r2=scales, color_resolution="amplitude-basis"
)                                                                # per basis element
one = runtime.evaluate(points, mu_r2=scales, helicities=["h0"])  # |A(h0)|^2 / avg
```

A spin vector v is expanded on the leg's polarisation vectors, which span
exactly the vectors transverse to its momentum p (v.p = 0; for a massless leg
the part along p vanishes by the Ward identity): for a transverse v the
value is the literal-source one of a Born-interfered virtual. A part of v
with v.p != 0 (for example the rounding of a k_perp built in binary64) is not
read: the value is that of the transverse vector the expansion keeps, and
the stability estimate carries the dropped part's size (the check
`spin-transverse`), so a vector far from transverse is served with an
estimate of the value's own size rather than refused. One `evaluate_correlated_many` call is one evaluation
of the points whatever its spin vectors: every request is formed on the same
amplitudes. A single-helicity selection divides by the total's average, so
the selections of every row sum to the total, and its dressing is pruned to
the selected rows (a family keeping more than half of its links runs whole).
The amplitude-basis anchors are basis-dependent parts whose sum is the
helicity's value. Colour-flow selection does not exist for loop-induced
processes.

## Split totals by squared coupling order

Recurrence artifacts record the exact coupling orders of every amplitude
contribution. `evaluate_orders` returns, in one pass, the unchanged total and
one partial per squared coupling order, including the interference between
different amplitude orders:

```python
split = runtime.evaluate_orders(points)
print(split.amplitude_orders)  # e.g. ((("QCD", 2),), (("QED", 2),))
print(split.orders)            # (("QCD", 2), ("QED", 2)), (("QCD", 4),), ...
assert split.total == runtime.evaluate(points)
qcd = split.for_order((("QCD", 4),))
```

`values` has shape `(point, squared order)`; each `sectors` entry lists its
contributing ordered amplitude-order pairs, following the Born order axis of
`VirtualEvaluation`. Orders count fundamental couplings, list every pairwise
sum (also structurally vanishing interference) and omit zero powers.
`sum_orders()` adds the partials explicitly; it equals `total` up to rounding.
The global `helicities` and `color_flows` selectors and `precision` behave as
in `evaluate`; per-point selectors (`helicity_by_point`,
`color_flow_by_point`) are not accepted. Compiled, eager and on-the-fly
artifacts raise `CompatibilityError`.

## Split totals by power of N_c

Tree artifacts generated with `color.accuracy = "strict-lc"` or
`"strict-nlc"` (the formal 1/N_c expansion), or with full colour and
`color.graded = true`, report their colour sum split by N_c degree.
`evaluate_graded` returns, per point, the N_c^degree term of the expansion
evaluated at N_c = 3, summed over the selected helicities:

```python
config = RunConfig(color=ColorConfig(accuracy="full", graded=True))
runtime = Runtime.load(Generator(config).generate("u u~ > g g", "artifacts/uux_gg").output)
graded = runtime.evaluate_graded(points)
print(graded.degrees)             # (3, 2, 1, 0, -1): from the leading degree P down
print(graded.reference_degree)    # 3 = n_g + n_qqbar
leading = graded.for_degree(3)    # the strict LC value at every point
beyond_lc = [total - lc for total, lc in zip(graded.total, leading, strict=True)]
```

`values` has shape `(point, degree)` and `total` sums the degrees of each
point; for graded full colour it equals `evaluate`, for `strict-lc` and
`strict-nlc` it is their truncated colour sum. A degree reached by the colour
weights but cancelled by the amplitude relations (for example the U(1)
decoupling of gluon traces) is reported with a value zero up to rounding. Per
helicity, the same split is the colour axis of `evaluate_resolved`: one
contracted component per degree, ID `color:nc-degree:<d>`, with `nc_degree`
on the `ContractedColorComponent` of `runtime.physics`. `precision` behaves
as in `evaluate`. Artifacts without N_c degrees (inherited `lc`/`nlc`,
ungraded full colour) raise `CompatibilityError`. For correlated artifacts,
`evaluate_correlated_graded(points, color_correlation=..., precision=...)`
returns a `GradedCorrelatedEvaluation` of `CorrelatedValue` entries with the
same layout; each connection's degrees decrease from its reference degree
`n_g + n_qqbar + r - s`.

## Retain completed OTF preparation

For an on-the-fly runtime, `runtime.save(path)` writes the completed cache
left by evaluation or explicit warm-up. In a later program, load the original
process output with `Runtime.load(...)`, then call `runtime.load_cache(path)`
before evaluating with the same selectors. Both methods return `None`; the
restored calculation accepts new momenta and batch sizes while retaining the
receiving runtime's model parameters.

See the [complete four-gluon example](runtime-and-selectors.md#saving-an-otf-warm-cache)
for generation, saving and restoring, and a check that no queries are rebuilt.
The cache supplements the original process output; it does not replace it or
checkpoint an unfinished cold construction.

## Evaluate Born correlations

Correlations require an explicit `Generator.generate(..., correlators=...)`
declaration. They support **LC, NLC, full colour and the strict (formal 1/N_c)
`strict-lc`/`strict-nlc`**, through a separate Python
direct, generic compiled exact executor. `color.accuracy` selects the correlated
approximation. Supplying `correlators` retains complete underlying amplitudes
and selects direct contraction and compiled generation. Ordinary generation
and evaluation remain unchanged.

```python
from pyamplicol import (
    ColorCorrelator, CorrelatorConfig, Generator, ModelSource, Runtime,
)
from pyamplicol.config import ColorConfig, RunConfig

declarations = CorrelatorConfig(
    color_correlations=(ColorCorrelator.dipole("T13", 1, 3),),
    spin_correlations=((3,), (3, 4)),
)
generated = Generator(RunConfig(color=ColorConfig(accuracy="full"))).generate(
    "g g > g g", "artifacts/gg_correlated",
    model=ModelSource.built_in_sm(), correlators=declarations,
)
correlated_runtime = Runtime.load(generated.output)
point = (
    (500, 0, 0, 500), (500, 0, 0, -500),
    (500, 300, 0, 400), (500, -300, 0, -400),
)
colour = correlated_runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
correlated_runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0)})
spin_colour = correlated_runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
print(spin_colour.real, spin_colour.imag)  # Decimal values, not a forced real part.
correlated_runtime.set_spin_correlation_vectors(None)
```

Instead of explicit colour requests, use
`CorrelatorConfig.all_color(through_order=k, spin_correlations=...)` before
generation to prepare every compatible directed pair of tree-soft connections
through any positive order `k`. The nonminimal catalogue includes emissions from
emitted partons and splitting of auxiliary gluons, and grows rapidly with order.
It adds no kinematic kernels, flavour sums or multiplicity weights. Inspect the
operators actually stored for a selected process with:

```python
for entry in correlated_runtime.available_color_correlations():
    print(entry.id, entry.order, entry.bra, entry.ket)
```

This includes `"born"` and reads the catalogue without initializing amplitude
evaluation. See the [automatic catalogue guide](../correlators.md)
for generation and grouped evaluation.

`evaluate_correlated(momenta, *, color_correlation="born", helicities=None,
precision=16, arithmetic="arbitrary")` returns one `CorrelatedValue` per point. The reserved ID
`"born"` uses the Born metric at the requested colour accuracy; other IDs are the declared
operators. Optional batch-global helicity selectors accept stable IDs or typed
`HelicityConfiguration` objects. There is no colour-flow or per-point selector
argument on this method. Normalization, initial-state averages, and
identical-particle factors are retained, and runtime model-parameter updates
also apply to correlations.

For multiple combinations, `evaluate_correlated_many(points, requests)` takes
a mapping from your result labels to `CorrelatedRequest(color_correlation=...,
spin_vectors=...)`. Its result is a dictionary of tuples:
`result[request_label][point_index].real` or `.imag` selects one contraction
at one phase-space point. Label and point order are preserved. Identical spin
assignments share their amplitudes across colour operators, and unchanged
stages can be reused across different assignments. See the
[two-axis example](../correlators.md#several-requests-and-several-phase-space-points)
for request-centric and point-centric access. Ordinary `evaluate()` on the
underlying correlated output remains full-colour; use the correlated `"born"`
request when comparing its selected LC/NLC approximation. Strict correlated
artifacts are graded throughout: ordinary `evaluate()` there gives the strict
truncation of the Born, equal to the `"born"` request.

The spin-vector mapping uses one-based public leg labels. Each value is a
literal real or complex `(v0, vx, vy, vz)` vector, broadcast across the batch,
or one such vector per point. The nonempty key set must exactly match a
declared spin class: this example permits `{3}` and `{3, 4}`, but not `{4}`.
Vectors are neither normalized nor projected. They are copied into
instance-local state; a failed setter preserves the previous state. `None` or
`{}` resets to ordinary helicity sources; `clear()` releases warmed evaluators
without resetting vectors. The setter affects `evaluate_correlated` and
inherited requests in `evaluate_correlated_many`, not `evaluate` or
`evaluate_resolved`.

`Decimal` vector components retain their digits; a `(Decimal(real),
Decimal(imaginary))` pair represents one complex component without first
rounding it through Python's binary64 `complex`. For example:

```python
from decimal import Decimal

zero = Decimal(0)
correlated_runtime.set_spin_correlation_vectors({3: (
    zero, zero,
    (Decimal("1.0000000000000000000000001"), Decimal("2e-25")),
    zero,
)})
arb = correlated_runtime.evaluate_correlated((point,), precision=80)
dd = correlated_runtime.evaluate_correlated(
    (point,), arithmetic="double-double", precision=31,
)
correlated_runtime.set_spin_correlation_vectors(None)
```

The default is arbitrary precision. Explicit `arithmetic="double-double"`
uses genuine DoubleFloat arithmetic throughout sources, evaluation stages,
normalization and reduction; at most 31 result digits are supported. The same
keyword and Decimal vector syntax apply to `evaluate_correlated_many` and its
explicit request vectors. See the [grouped precision example](../correlators.md#decimal-inputs-and-double-double-arithmetic).

Artifacts without declarations, append generation, partial source coverage,
FFT, eager/recurrence/OTF correlated execution, replay reductions, and
nonidentity process permutations are not supported. Ordered `EmitGluon` and
`SplitGluon` operations have no fixed order cap; generic NkLO colour-connection
order is distinct from LC/NLC colour approximations and does not supply a
complete higher-order subtraction calculation. See [Born Correlations](../correlators.md)
for the bra-adjoint/ket convention, joint spin contractions, JSON declarations,
Ward checks, and current limitations.

The same generated colour catalogue and spin classes are available to the
[C/C++/Fortran/Rust SDKs](native-apis.md#correlated-born-evaluations), whose
correlated evaluation is binary64 and independent of Python/Symbolica.
The Python methods described here retain their precision-controlled executor.

## Select helicities and color flows

Batch-global selectors accept stable IDs or the typed objects exposed by
`runtime.physics`:

```python
selected = runtime.evaluate(
    points,
    helicities=[runtime.physics.helicities[0]],
    color_flows=[runtime.physics.color_flows[0]],
)
```

For one selector per point:

```python
mixed = runtime.evaluate(
    points,
    helicity_by_point=[
        runtime.physics.helicity_ids[index % 2]
        for index in range(len(points))
    ],
    color_flow_by_point=[
        runtime.physics.color_flow_ids[(index // 2) % 2]
        for index in range(len(points))
    ],
)
```

Batch-global and per-point selectors are mutually exclusive on the same axis.
Per-point color selection is LC-only. Rusticol groups mixed selectors while
returning values in original point order.

See [Runtime and Selectors](runtime-and-selectors.md) for the complete selector
contract.

## Update model parameters

```python
runtime.set_model_parameters({"aS": 0.1165, "MZ": 91.1876})
runtime.set_model_parameter("MT", 172.5)
```

Updates are atomic and refresh dependent derived parameters. Unknown,
immutable, non-finite, or invalid values reject the entire update.

`Runtime.load(..., model_parameters=...)` applies a complete mapping before the
runtime is returned. Complex inputs may be represented as Python complex values.

## Group equivalent subprocesses

```python
from pyamplicol.api import group_processes

summary = group_processes("artifacts/pp_jj")
for relation in summary["new_relations"]:
    print(relation["process"], "=", relation["factor"], "x", relation["representative"])
```

`group_processes` evaluates each group representative stored in
`process-groups.json` in f64 at three on-shell points, each with its own
deterministic variation of up to 10% of the external and normalization model
parameters other than masses and widths. Two subprocesses with the same
external particle types are grouped when, through a leg permutation (and
charge conjugation, unless `consider_charge_conjugation=False`), every
resolved helicity and colour
component of one is a common rational factor times the other's; numerator and
denominator are at most 99 and `only_unit_multiplier=True` accepts only 1.
Subprocesses generated with a selected helicity or colour subset are never
grouped. The map is updated in place and rerunning adds nothing; the artifact
ID and UMAMI files are unchanged. `analysis="none"` only writes the trivial map
of an artifact that has none. This is the `pyamplicol group-processes`
command; see [Process Grouping](process-grouping.md) for the details.

## Precision

```python
f64 = runtime.evaluate(points, precision=16)
high_precision = runtime.evaluate(points, precision=80)
```

For ordinary `evaluate` and `evaluate_resolved`, precision 16 uses the native
Rusticol runtime without Symbolica computations. Package startup imports
Symbolica to attempt registration of the pyAmpliCol library key; synchronous
registration failures retain the personal-license and restricted-mode paths.
Other positive precision requests load retained exact evaluator state when
the artifact supports it.
Decimal input preserves the supplied decimal digits;
binary64 input cannot gain information merely by requesting more arithmetic
precision.

`evaluate_correlated` always uses retained Symbolica evaluator states through
the separate Python exact executor, including at its default precision 16.
Its default `arithmetic="arbitrary"` is distinct from the explicit
`arithmetic="double-double", precision=31` mode; neither is selected by an
ordinary native SDK precision flag.
Its real and imaginary results remain `Decimal` values; `complex(value)` is an
explicit conversion to ordinary double precision.

Generated C/C++/Fortran/Rust standalone APIs support f64 (`precision=16`) only.

## Stability rescue (one-loop virtuals)

A one-loop virtual runtime can re-evaluate its numerically unstable points in
higher precision by itself. Pass a `VirtualRescuePolicy` to `evaluate` (per
call) or to `Runtime.load` (the default of every `evaluate` call that names
no arithmetic, precision or stability checks of its own). The rescue is off
unless a policy is passed, and it exists only in this Python API: the SDKs,
the CLI and the run card have no rescue (`[evaluation.stability]` stays
refused).

```python
from pyamplicol import Runtime, VirtualRescueLevel, VirtualRescuePolicy

runtime = Runtime.load("artifacts/gg_ggg", model_parameters={"aS": "0.118"})
result = runtime.evaluate(points, mu_r2="1000000", stability=VirtualRescuePolicy())
print(result.rescue_counts)   # {0: 126, 1: 2, 'not met': 0, 'failed': 0} (128 points)
record = result.rescue[17]    # level reached, its carrier, target, met, attempts

custom = VirtualRescuePolicy((
    VirtualRescueLevel("double", rotations=2, target=1e-7),
    VirtualRescueLevel("double-double", rotations=("euler:0.1,0.2,0.3",)),
    VirtualRescueLevel("arbitrary", 40, reduction_check="reversed", metric="finite"),
))
```

A policy is an ordered stack of levels in non-decreasing precision bits,
modelled on gammaloop's stability levels (`runtime.stability.levels`): each
level names how stability is assessed, its working precision and its
reduction backend. Two levels at the same precision must differ in their
backend (`reduction`), e.g. TIRust then NinjaRS in binary64; a repeated
(precision, backend) pair is refused (`ValueError`). Each `VirtualRescueLevel`
has:

- a carrier: `"double"`, `"double-double"` or `"arbitrary"` with `precision`
  decimal digits;
- its stability checks: `rotations` (a count of default rotations, exact
  `Rotation` records or rotation text, including gammaloop's Euler rotation
  `"euler:<alpha>,<beta>,<gamma>"` = Rz(gamma) Ry(beta) Rx(alpha) in radians,
  rounded once to an exact rotation) and `reduction_check` (`"none"`,
  `"reversed"` or `"rescaled"`), as in `VirtualStabilityChecks`;
- a `target` on the stability estimate's `metric`: `"relative"` (each squared
  order against its largest Laurent coefficient), `"finite"` (the finite
  part against itself) or `"scale"` (E <= target x S, the estimate against
  the scale of what cancelled: an opt-in acceptance rule for values that may
  vanish, such as a loop-induced zero, which no relative target can reach);
- its reduction backend `reduction`: `"auto"` (the default: the call's
  `reduction=` request when it names a backend, else the runtime's or the
  artifact's, else the per-type default with its fallback), `"ninjars"` or
  `"tirust"`. An `"auto"` level follows the call's `"auto"` rule (on TIRust:
  the per-point NinjaRS fallback at the level's arithmetic); a level naming a
  backend has no fallback, its next level is the fallback. TIRust's acceptance
  at a level follows the level's `target` (the same calibration as
  `reduction_target`). Naming one backend per level emulates MadLoop's
  `MLReductionLib` order, e.g. `VirtualRescueLevel("double",
  reduction="tirust")` followed by `VirtualRescueLevel("double",
  reduction="ninjars")`. Each level's backend is resolved once, for its
  carrier, before the first point: an explicit one (the level's, or the
  call's inherited by an `"auto"` level) that cannot serve the level's
  carrier is refused (`ReductionUnavailable`; for instance a `"tirust"` level
  in a native extension built without TIRust), never switched.
  Each `VirtualRescueAttempt` records the level's `reduction`
  (`VirtualReduction`: requested, effective, reason), so
  `result.rescue[i].attempts` shows the backend of every level a point
  reached (`result.rescue[i].reduction` is the returning level's).

Only the last level may run without a check; it then keeps every point it
evaluates, with no estimate. How a batch runs:

1. Level 0 evaluates the whole batch.
2. A point that fails (also when one of its rotated evaluations fails) or whose
   estimate misses the target is evaluated again at the next level, from its
   original exact inputs; no lower-precision value is reused. Escalated points
   are evaluated one per native call, so each carrier builds one executor and
   compiles its kernels once.
3. Accepted points are never evaluated again. At the last level a point that
   evaluates keeps its value with the target marked as not met; a point that
   failed at every level is a failed lane (raised unless
   `on_failure="return"`).

Each lane of the result is the evaluation of the level that returned it: its
values, pole parts, `stability` estimate and diagnostic carrier
(`arithmetic`, `precision_bits`). Its status is `"success"` at level 0,
`"rescued"` above and `"failed"` if every level failed;
`accuracy_established` is `True` exactly when the estimate met the target.
That is the estimate's verdict, not a certificate: the calibrated miss rates
of the stability checks apply (about 0.1 % of binary64 points above their
estimate with one rotation and the rescaled check). `result.rescue[i]` is a
`VirtualRescueRecord` (`level`, `arithmetic`, `precision_bits`, `target`,
`met`, and one `VirtualRescueAttempt` per level tried with its estimate or
failure); `result.rescue_counts` counts the lanes accepted per level, plus
`"not met"` and `"failed"`. Without a policy `rescue` is `None`.

The default `VirtualRescuePolicy()` runs binary64, then double-double, then
57-digit arbitrary precision, each with target 1e-5 on the relative estimate,
one rotation `"euler:0.1,0.2,0.3"` and the rescaled reduction check, every
level `"auto"`. For a loop-induced call requesting `"auto"` (where `"auto"`
takes TIRust with the NinjaRS fallback) it is the **loop-induced default
stack**: TIRust in binary64 at the target, then NinjaRS in binary64, then the
double-double and 57-digit levels (`"auto"`: TIRust with the per-point NinjaRS
fallback in their arithmetic). A point TIRust refuses at level 0 is recorded
at the NinjaRS level. A call naming a backend (`reduction="ninjars"` or
`"tirust"`) runs the default ladder on that one backend at every level (no
level on the other backend, no fallback): `evaluate(...,
reduction="ninjars", stability=VirtualRescuePolicy())` never evaluates a
point on TIRust. A stack you build keeps each level's named backend (a
level's backend wins over the call's) and its `"auto"` levels take the
call's request.

Field by field, a level is gammaloop's `StabilityLevelSetting` with a backend
(gammaloop `crates/gammalooprs/src/settings/runtime.rs`, `StabilitySettings`
and `StabilityLevelSetting`; the escalation loop of
`integrands/process/mod.rs`): `arithmetic` / `precision` is its `precision`
(`Double`, `Quad` = double-double, `Arb`); `rotations` is gammaloop's
`StabilitySettings.rotation_axis` (shared by every level there, set per level
here), to which pyAmpliCol adds `reduction_check`; `target` with `metric` is
its `required_precision_for_re` / `_im` (one target on pyAmpliCol's estimate;
`metric="relative"` normalises by the result as gammaloop's `check_on_norm`
does); `reduction` is the reduction backend, which gammaloop has no
counterpart for. As in gammaloop, the levels run in order, a point stops at
the first level that accepts it and the last level keeps its result.
gammaloop's `escalate_if_exact_zero`, `loop_momenta_norm_escalation_factor`
and `escalate_for_large_weight_threshold` have no counterpart (a loop-induced
zero is accepted by the `"scale"` metric).

The default ladder differs from gammaloop's defaults (f64, f128, 1000-bit arb,
target 1e-5, the same rotation) in three ways:

- the rescaled reduction check is added, because a rotation keeps every
  master integral's invariants and cannot see its own rounding;
- the last level has 57 digits (pyAmpliCol's calibration carrier) instead of
  1000 bits; set `VirtualRescueLevel("arbitrary", 300)` for gammaloop's;
- the target applies to pyAmpliCol's error estimate, which sums the deviations
  of every reduction lane with a safety factor of two, not to the spread of the
  rotated values.

For a Born-interfered process every pole selection (`poles=`), colour
accuracy and leading-colour `helicities` / `color_flows` selection is
supported. A loop-induced process is rescued with what it serves: the pole
selections `"all"` and `"none"`, full colour and its helicity selections
(`helicities=`); it has no colour flows to select (`CompatibilityError`). The reduction backend (`reduction=`) is
resolved for every level's carrier before the first point is evaluated: an
explicitly requested backend that cannot serve one of them is refused, an
`"auto"` one records its choice per level in the lane diagnostics and the
attempts. `stability=` together with
`arithmetic`, a non-default `precision` or `stability_checks` is refused with a
`CompatibilityError` (the levels own them), as is a policy on
`evaluate_resolved` and `evaluate_correlated(_many)`, and a load-time policy
with a load-time arithmetic or precision. Checks cost about one extra
evaluation per rotation plus a fifth to a third for the rescaled check (one
rotation and the rescaled check: 2.35 times plain binary64 on g g > g g g); never time a
benchmark with a policy. Double-double and arbitrary-precision levels read the
artifact's saved Symbolica state under the pyAmpliCol package key, as an
explicit `arithmetic="double-double"` call does.

## Profile from Python

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    target_runtime=1.0,
    batch_size=128,
    precision=16,
    warmup_runs=2,
    minimum_samples=5,
    color_flow_ids=("1",),
)

result = BenchmarkRunner(config).run(runtime, points=points)
print(result.wall_time_per_point)
print(result.evaluator_time_per_point)
print(result.uncertainty.standard_error)
print(result.uncertainty.relative_standard_error)
```

`BenchmarkResult` also contains calibrated repetitions, sample count,
environment metadata, native timing breakdown, stage attribution, and work
counters when the runtime exposes them.

The runnable reference is
[`examples/python/benchmark.py`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/python/benchmark.py).

## Convenience functions

```python
from pyamplicol import benchmark, generate, load

generated = generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
runtime = load(generated.output)
profile = benchmark(runtime, points=points)
```

The class-based services are preferable when reusing configuration, progress
sinks, compiled models, or loaded runtimes.
The one-shot `generate(...)` function also accepts the same
`correlators=CorrelatorConfig(...)` keyword as `Generator.generate(...)`.

## Errors

All public failures derive from `PyAmpliColError`:

```python
from pyamplicol import EvaluationError, GenerationError, PyAmpliColError

try:
    runtime = Runtime.load("artifacts/pp_zjj", process="missing")
except EvaluationError as error:
    print(f"could not load runtime: {error}")
```

Useful subclasses include configuration, model, generation, artifact,
compatibility, dependency, and evaluation errors. Catch the narrowest type you
can handle; catch `PyAmpliColError` at an application boundary.
`NativePlatformError`, a `DependencyError`, means that this platform cannot
load a native library at all (for example a glibc that lacks a version the
library needs; see
[Native libraries and glibc](installation.md#native-libraries-and-glibc)).

## API design notes

- Public result and metadata objects are immutable dataclasses or typed facades.
- Artifact paths are normalized to absolute paths at service boundaries.
- `Generator.generate(..., mode="error")` refuses an existing destination.
- `Runtime` is bound to one selected process; load another instance for another
  process in the same multiprocess artifact.
- Process artifacts are trusted executable inputs. Normal loading validates the
  schema, path confinement, references, target compatibility, and runtime ABI.
- Use explicit payload validation only for an intentional whole-artifact
  corruption audit; it is not required before every load.

## Further reading

- [Configuration](configuration.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Born Correlations](../correlators.md)
- [Correlation Conventions](../correlator-conventions.md)
- [Artifacts and Portability](artifacts-and-portability.md)
- [Native APIs](native-apis.md)
