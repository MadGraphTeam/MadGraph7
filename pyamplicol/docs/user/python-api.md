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
| `pyamplicol.api.check` | Pole and Ward checks of a one-loop request, regenerated in memory (`pyamplicol check`). |
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
the color dimension has length one because color is contracted. A one-loop
virtual's `evaluate_resolved(helicities=...)` returns the selected rows only
(the same layout over the kept rows).

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
from pyamplicol import Runtime
from pyamplicol.api import VirtualStabilityChecks

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

`VirtualStabilityChecks(1, True, rescaled=True)` are the **recommended
checks** (one rotation, the reduction-direction check and the rescaled
check; importable as `pyamplicol.loops.v2.stability.RECOMMENDED_CHECKS`).
Stability checks are off by default (no estimate is returned then); the
reduction backends' own acceptance and refusal checks stay on unless
`reduction_unchecked=True`.

The reduction backend is chosen once per call (`reduction="auto"`, `"ninjars"`
or `"tirust"` on `Runtime.load` and on every evaluation; default: the
artifact's `[loop] reduction`): `"auto"` takes TIRust for a loop-induced and
NinjaRS for a Born-interfered process (TIRust for both on a TIR-only
artifact) and falls back to NinjaRS, with the reason recorded, where TIRust
cannot serve the call (a native extension built without TIRust). On TIRust,
`"auto"` has a **per-point NinjaRS fallback**: a point TIRust refuses (in
binary64 a value of its checked method choice or of its verified last resort
that its acceptance check finds beyond the target, below; in double-double and
arbitrary precision a kinematic row its conditioning gates refuse) or fails (also in
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

In double-double and arbitrary precision every point the fallback serves
is **guarded**: its NinjaRS value runs a rotation re-test (the evaluated
point, i.e. the given inputs improved in the call's arithmetic, rotated
exactly by the recommended checks' rotation and evaluated again:
`VirtualStabilityChecks(1)`'s check) in the call's own arithmetic (never a
higher one), and the point fails typed
when that check's relative estimate (each squared-order sector against its
largest Laurent coefficient) exceeds the guard's limit: the call's target
(`reduction_target`, below; a rescue level's own target on that level),
never below 1e-5, i.e. max(target, 1e-5). The point's one reason names the
fallback, TIRust's failure, the arithmetic, the estimate and the limit,
e.g. `auto: TIRust refused or failed the point (...); the per-point NinjaRS
fallback's value failed its rotation check: estimate 1.4e2 >
1e-5 = max(target 1e-5, 1e-5) (double-double)`; `on_failure="raise"`
raises `EvaluationError` with it, `"return"` returns the point as a failed
lane (and a rescue stack escalates it), and the default `"best-effort"`
returns it with its unchecked value as a `"best-effort"` lane. The estimate is recorded in the
point's `VirtualReduction.guard_estimate` (a rejected point's too); a point
within the limit keeps the fallback's value unchanged, and the result's
`stability` stays what the call's own checks give (none without checks).
In binary64 the fallback's value is served unguarded, with no estimate
recorded: there the protection is the conditioning bound in NinjaRS's own
fast engine (below), and stability checks or the rescue
stack flag what it serves.
When the call's checks run that rotation first (the recommended checks) it
is reused; beside other checks (another rotation first, e.g. every default
rescue level's, or the reduction checks alone) the served points are
evaluated once more with it. The rotation re-test replaced the cached
reduction-direction check, which did not see the double-double
near-forward losses of NinjaRS (below); no false rejection was found on the
regular double-double points measured. The guard decides on the outputs the call returns: the summed value,
or on the resolved, correlated and selected calls each output (a helicity
row, a colour correlator, a helicity selection), from the fallback's
evaluation of the call's own rows only (only the selected helicities are
evaluated), each output's estimate against its own magnitude and an output
that vanishes (its magnitude within the estimate's rounding floor, 16 units
of the arithmetic's roundoff times the output's term scale: a numerical
zero) against the largest output the call returns for the point; the
largest ratio is the point's `guard_estimate`. Points TIRust serves run
nothing more.
The estimate is a sampled measure, not a proven bound.

**Near-planar NinjaRS hazard.** At near-planar 2 -> 3 points (the event
plane contains the beam: the pentagon Gram determinant nearly vanishes) of
pentagon families whose rank equals their propagator count (all-vector quark
loops), NinjaRS's reduction can produce finite, badly wrong values in every
arithmetic. Its box mu-expansion's conditioning bound refuses such lanes
typed (`Unstable`; the generic engine's `CutFailed { SingularParameter }`),
so under `"auto"` the fallback fails them typed (in double-double, e.g., the
measured g g > z z z points at tilts eps = 0, 1e-14 and 1e-13 out of the
plane); the guard stays as defence in depth on what the fallback still
serves. Without that bound (measured 2026-11-08 on the development host;
truth: TIRust at 78 digits), g g > z z z tilted by eps out of its plane was
served wrong in binary64 from eps ~5e-7 (2e-2 off at 1e-7) down to at least
1e-12, and in double-double at eps 0 (up to 1.3e7 off), 3.3e-6 off at 1e-14
and within 2.1e-10 from 1e-13 up. Measured affected: loop-induced
g g > z z z, z z g, z g g, a a g, a g g and W+ W- g; measured unaffected:
the rank-4 pentagon rows g g > h h h and g g > h z g, Born-interfered
g g > g g g and five Born-interfered NLO electroweak rows. The recommended
checks flag such values (estimates 22 to 892 in double-double, 3.0 to 236 in
binary64 on the measured wrong points) and the rescue stack escalates them.
A value the bound does not flag is no more verified than before: an explicit
`reduction="ninjars"` has no fallback guard, so request stability checks
where such kinematics matter.

**Inexact inputs near forward.** Near forward, backward or threshold
kinematics NinjaRS amplifies a point's on-shell and momentum-conservation
defect by about the inverse box Gram determinant, and neither the
reduction-direction check nor a rotation re-test sees it (a rotation keeps
the defect). Measured (2026-11-09, development host) at near-forward
g g > h h points given as binary64 texts (gluon legs off shell by up to
5.2e-13 GeV^2, Higgs legs by up to 2.0e-10 GeV^2): the +- helicity rows of
`evaluate_resolved` (1e-13 to 1e-21 of the ++ rows) moved 1.1e-4 to 9.3e11
from those of the exact point nearby on NinjaRS, while the summed |A|^2
stayed within 1.2e-9. The kinematics improvement (below; on by default in
double-double and arbitrary precision, in binary64 with
`kinematics_improvement_binary64=True`) removes that defect. On exactly
on-shell, momentum-conserving points double-double's own rounding remains:
there `"auto"` served the +- rows of one point 1.8e-5 and 2.0e-5 off with
the guard's estimate 9.3e-6, inside its limit 1e-5. This is the guard's
known limitation: its estimate is an estimate, and strongly suppressed
helicity rows near exactly forward kinematics can be served beyond their own
1e-5 (the summed observables are not affected at that level).

`"tirust"` is served in every arithmetic (binary64, double-double and
arbitrary precision, also on every level of a rescue stack). An explicit
backend that cannot serve the call is refused before any point is evaluated
(`pyamplicol.ReductionUnavailable`, a `CompatibilityError` with key
`reduction-unavailable`). The effective backend is always one that the
call's executor runs: the summed `evaluate` and the resolved, correlated and
selected outputs each check it, so a result never names a backend that did
not reduce it. Every lane diagnostic records the requested and effective
backend, the reason, the call's fallback, target and unchecked setting, and
on a point the fallback served in double-double or arbitrary precision its
guard's estimate
(`VirtualReduction(requested, effective, reason, fallback, target,
unchecked, guard_estimate)`;
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
setting the target drives; in double-double and arbitrary precision it is
also the per-point fallback's guard's threshold (above), never below 1e-5.
There, where TIRust's method choice is off, it acts on TIRust's values
through nothing else. A value that is not a finite positive
number is refused (`ValueError`).

**Unchecked reductions** (`reduction_unchecked=True` on `Runtime.load` and on
every evaluation; `[evaluation] reduction_unchecked = true`; CLI
`--reduction-unchecked`; the SDKs' `rusticol_virtual_set_reduction_unchecked`;
default **off**). The setting drops the reduction backends' own acceptance and
refusal checks, so that a backend serves every point it can give a finite
value for, the way the one-loop providers' most optimistic settings run
(OpenLoops and Recola with COLLIER's method choice and no stability system):

* TIRust keeps its a priori conditioning gates and its method choice
  (small-Gram expansions or the alternative reduction) as *method selection*
  and returns the method-choice values as computed: the acceptance check of
  those values (their re-evaluation at rescaled inputs and the comparison)
  is not computed at all, so the target accuracy drives nothing on TIRust
  under the setting. The rows the method choice refuses (an expansion that
  does not converge, an alternative estimate above its limit, a cancelling
  determinant, vanishing adjugates, a triangle below the requested rank, its
  own gate) are served by their **last resort**, plain Passarino-Veltman at
  gate 0. Every other row's value is bitwise the checked one.
* NinjaRS runs its binary64 engine with zero stability thresholds (Ninja's
  `stability_threshold` and `relative_tolerance` 0): a lane fails only where
  a tested quantity is exactly zero or NaN, or where its box mu-expansion's
  conditioning bound refuses it. The bound is not a stability threshold and
  stays active in every carrier: it flags a lane whose predicted loss of
  digits exceeds the carrier's precision minus 14 bits (near-planar 2 -> 3
  points with a rank-n numerator, e.g. g g > z z z with the event plane
  containing the beam); the binary64 engine refuses a flagged lane
  (`Unstable`) when the first-order rounding estimate of its boxes'
  mu-expansion terms exceeds 2^-17 of the lane's value, the generic engine
  (double-double, arbitrary precision) refuses every flagged lane
  (`CutFailed { SingularParameter }`), with or without the setting.
* The **residual failures**, which no scheme gives a finite value for, stay
  failed points with their reasons: an exactly singular pivot (TIRust, after
  its affine-relation variants), a non-finite output (TIRust's "nonfinite
  direct-engine coefficient", a non-finite numerator, NinjaRS's `NonFinite`),
  a master failure (TIRust's `MasterFailure`, NinjaRS's `MasterFailed`),
  TIRust's `InvalidInput` and a NinjaRS lane whose tested quantity is exactly
  zero or NaN or that its conditioning bound refuses (its `Unstable` status,
  e.g. a cut whose two momenta span a Euclidean plane, a near-planar box
  mu-expansion). A family outside TIRust's capability stays a call-level
  `ReductionUnavailable`. A served value is not thereby accurate: see below.
* `"auto"` keeps its rule, each backend in its unchecked form: TIRust for a
  loop-induced process, NinjaRS for a Born-interfered one, TIRust without a
  fallback on a TIR-only artifact, NinjaRS where the native extension has no
  TIRust. Its per-point NinjaRS fallback (unchecked, in the call's
  arithmetic) acts only on the residual failures of the unchecked TIRust,
  recorded per point, and in double-double and arbitrary precision its
  served points are guarded as above (the guard is pyAmpliCol's check, not a
  backend's own: it stays on); where nothing fails, an `"auto"` call is a
  pure TIRust call. Explicit `"tirust"` and `"ninjars"` have no fallback.
* Arithmetics: in double-double and arbitrary precision TIRust's method
  choice is off, so there the conditioning gates select and the last resort
  serves what they refuse. The setting changes nothing in NinjaRS's generic
  engine (double-double, arbitrary precision, and the binary64 lanes the
  fast engine routes to it): it has no stability threshold to set to zero,
  and its conditioning bound refuses near-planar box mu-expansions there
  too, at the carrier's precision. That is no accuracy statement: its
  values are unverified there too, with or without the setting (the
  near-planar hazard above); stability checks, a rescue stack or, in
  double-double and arbitrary precision, the guard of the per-point
  fallback (which stays on under the setting) check them.
* pyAmpliCol's own stability checks (`stability_checks=`, the SDKs'
  monitoring) and the rescue stack (`stability=`) stay their own settings:
  combined with the setting they check the unchecked values, and the rescue
  levels (each reducing unchecked) or the per-point fallback act on the
  points those checks fail.

The setting is recorded in every lane diagnostic
(`VirtualReduction.unchecked`) and in the SDKs' `"reduction"` diagnostics.
It is a **timing and benchmark mode with no accuracy certificate**, not a
production default: without it, the default `"auto"` serves loop-induced
processes on TIRust at the checked target accuracy above with the per-point
NinjaRS fallback. A value that is not a bool is refused (`ValueError`).

Measured accuracy of the setting (2026-11-08/09, development host; record
`docs/performance_reports/loop/data/li-unchecked-accuracy.json`):
binary64 finite part through this API (`Runtime.load(artifact,
reduction_unchecked=True)`, `evaluate(points, mu_r2=..., poles="none",
on_failure="return", reduction="tirust" | "ninjars")`) on each row's first
stream points against TIRust at 256 bits with supplied invariants; the
largest relative error over the points, then the points above 1e-5 and the
points that fail. TIRust's errors above 1e-5 all come from rows its last
resort serves (plain PV at gate 0 below the conditioning gate); NinjaRS's
zero thresholds return finite values off by up to 1e64 where its checks
refuse the point (g g > h h h: a box whose three momenta are linearly
dependent, which only its noise tests catch), and keep its few points off by
up to 3e-3 that the checked mode serves as well (a triangle cut with a small
Gram determinant); its conditioning bound refuses the near-planar g g > z z z
points in both modes (NinjaRS columns measured with that bound).

| process | points | TIRust unchecked | NinjaRS unchecked |
|---|---|---|---|
| g g > h | 128 | 1.2e-13 | 1.2e-12 |
| h > g g | 128 | 1.0e-13 | 1.1e-12 |
| h > a a | 128 | 2.4e-14 | 4.2e-13 |
| h > z a | 128 | 1.2e-14 | 1.6e-12 |
| a a > h | 128 | 1.4e-14 | 1.6e-13 |
| g g > a a | 128 | 6.6e-13 | 2.4e-13 |
| g g > z z | 128 | 5.5e-12 | 8.8e-11 |
| g g > w+ w- | 128 | 1.2e-12 | 5.2e-12 |
| g g > z a | 128 | 6.8e-13 | 6.4e-12 |
| g g > h h | 128 | 3.4e-9 | 8.8e-11 |
| g g > h z | 128 | 1.4e-12 | 1.4e-12 |
| g g > h g | 128 | 2.4e-14 | 7.3e-13 |
| u u~ > h g | 128 | 1.1e-14 | 1.5e-13 |
| u g > h u | 128 | 1.2e-14 | 1.1e-13 |
| g g > z g | 128 | 9.4e-13 | 9.7e-12 |
| g g > a g | 128 | 2.3e-13 | 1.7e-12 |
| e+ e- > h a | 128 | 3.8e-14 | 1.8e-13 |
| a a > a a | 128 | 1.8e-14 | 4.9e-14 |
| a a > z z | 128 | 1.5e-13 | 7.4e-13 |
| a a > h h | 128 | 8.8e-11 | 9.1e-11 |
| e+ e- > h h | 128 | 1.3e-10 | 1.3e-10 |
| h > g g g | 128 | 6.6e-13 | 1.5e-12 |
| z > g g g | 128 | 5.5e-11 | 3.7e-10 |
| e+ e- > g g g | 128 | 3.5e-7 | 3.2e-10 |
| g g > h h h | 128 | 7.6e-7 | 4.5e64: 2 above 1e-5 (the checked mode refuses them) |
| g g > h g g | 128 | 9.2e-10 | 8.0e-10 |
| g g > z z g | 128 | 1.2e-2: 18 above 1e-5 (all on the last resort) | 5.0e-8 |
| g g > a a g | 128 | 7.0e-6 | 1.0e-3: 1 above 1e-5 (also when checked) |
| g g > w+ w- g | 128 | 1.4e-2: 24 above 1e-5 (all on the last resort) | 6.2e-8 |
| g g > z z z | 128 | 2.2e-8 | 1.8e-9; 20 fail (`Unstable`) |
| g g > h z g | 128 | 8.2e-9 | 3.0e-8; 2 fail (`MasterFailed`, `Unstable`) |
| g g > a g g | 128 | 3.3e-6 | 2.0e-4: 1 above 1e-5 (also when checked) |
| g g > z g g | 128 | 3.1e-5: 3 above 1e-5 (all on the last resort) | 6.2e-6 |
| u u~ > h g g | 128 | 3.7e-9 | 3.5e-8 |
| u u~ > d d~ h | 128 | 4.1e-10 | 4.3e-9 |
| g g > h h g g | 32 | 7.1e-7 | 7.0e-7 |
| g g > h g g g | 32 | 5.9e-8 | 4.2e-5: 2 above 1e-5 (also when checked) |
| g g > e+ e- mu+ mu- | 32 | 5.4e-12 | 6.5e-10 |
| g g > a a g g | 128 | TIRust vs NinjaRS 7.0e-02 (no 256-bit truth: TIRust at 256 bits takes 405 s per point (measured 1-point probe; 32 points 3.6 h > the 3600 s row budget)); at its 2 largest-difference points vs the 256-bit truth: 1.3e-04 | at those points: 7.6e-02 |
| g g > z z z z | 112 | TIRust vs NinjaRS 4.9e-06 (no 256-bit truth: TIRust at 256 bits takes 764 s per point (measured 1-point probe; 32 points 6.8 h > the 3600 s row budget)) | — |
| g g > e+ e- mu+ mu- g | 16 | 8.3e-9 | 7.0e-9 |

Other processes and points are not measured: check a sample against the
default (checked) call before relying on the setting.

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

### Best-effort values of refused points

Since 1.0.3 a per-point refusal never interrupts a run by default. A point
that no configured alternative accepts (the reduction backend's own
acceptance checks, `"auto"`'s per-point NinjaRS fallback, the last level of
a rescue policy) is evaluated again, from its original exact inputs and on
that point only, without the backends' own acceptance and refusal checks:
exactly what `reduction_unchecked=True` gives for that point and backend
(TIRust's method choice and last resort as computed, NinjaRS with zero
stability thresholds). Where that yields a value, the lane is returned with
it under status `"best-effort"`: never `valid`, `has_value` true, its
refusal kept, and an uncertainty. A point without any value stays a
`"failed"` lane (value `None`).

```python
result = runtime.evaluate(points, mu_r2=scales, reduction="tirust")
result.statuses        # ("success", "best-effort", ...): one per point
result.has_warnings    # True when a point has no accepted value
for warning in result.warnings:  # ResultWarning: one per kind
    print(warning.kind, warning.count, warning.points, warning.message)
lane = result.diagnostics[1]
lane.best_effort.refusal      # e.g. "TIRust Unstable: ill-conditioned Gram ..."
lane.best_effort.uncertainty  # relative uncertainty, or None
lane.best_effort.source       # "stability-and-recommended-checks" | "stability-checks"
                              # | "recommended-checks" | "none"
```

`VirtualBestEffort` (the lane's `reduction.best_effort`, also
`lane.best_effort`) holds `refusal` (one message per distinct reason class,
with the count of the others), `uncertainty` (the relative estimate of the
value, the `"relative"` metric: each squared-order sector against its own
magnitude), its `source`, `backend_deviation` and `gram_conditioning`. The uncertainty always
includes the recommended checks (a rotation and the rescaled
reduction-direction check, `VirtualStabilityChecks(1, True, rescaled=True)`),
run on the refused points only, whatever checks the call asked for:
`"stability-and-recommended-checks"`: the call's own checks together with
the recommended ones (at least the larger of the two estimates; the lane's
public `stability` estimate stays that of the call's own checks);
`"stability-checks"`: the call's checks already include the recommended
ones; `"recommended-checks"`: the call asked for no checks (the public
result then carries no estimate); `"none"`: no estimate could be formed,
`uncertainty` is None, said so in the lane's detail, never invented. Why:
one rotation alone (the profile cards' check set) under-states the error of
an unchecked TIRust value of a Born-interfered process. Measured with the
shipped profile cards' check set (one default rotation; explicit TIRust,
binary64, flat points seed 1 at sqrt(s) = 1000, truth = double-double), the
one-rotation estimate covered the true error of 220 of 722 u u~ > t t~ g g
best-effort points and 39 of 65 u u~ > d d~ e+ e- ones; with the
recommended checks included, 722 of 722 and 65 of 65 (the C ABI alike: 221
-> 722 of 722 and 42 -> 64 of 65). A user who runs weak checks (one
rotation) thus still gets an uncertainty that covers the measured errors;
the lane's public `stability` estimate is that of the checks asked for and
can be optimistic for a best-effort value (compare it with
`best_effort.uncertainty`). `backend_deviation` and `gram_conditioning` are
TIRust's own lane-level measures of the refused lanes, returned by the native
reduction as data next to the reason texts (never read from them; the C ABI's
record carries the same two numbers): the largest method-choice deviation
between a lane's two frames relative to its term scale, and the smallest
conditioning estimate of a lane its stability threshold refused (Gram,
Cayley or coordinate-system measure; zero for an exactly singular pivot);
None where no refused lane states one (e.g. NinjaRS's refusals). Neither is
the value's accuracy: that is `uncertainty`. The lane's `reduction` records the
backend whose unchecked reduction gave the value (`unchecked=True`).

`result.warnings` is data, computed from the lanes, never emitted: a tuple
of `ResultWarning(source, kind, message, count, where, points)` (one entry
for the best-effort points, one for the failed ones; `"numerical"` source).
Nothing is printed, logged or warned per point or per call, so Monte-Carlo
logs cannot be swamped; `result.emit_warnings()` is the opt-in that turns a
result's warnings into ONE `pyamplicol.NumericalAccuracyWarning` (a
`UserWarning` subclass, filterable with the standard `warnings` module).
`Generator.generate` and `pyamplicol.check` return the known, benign
dependency log events they trigger in their result's `warnings` (source
`"symbolica"` or `"ufo-model-loader"`) instead of printing them, and so the
Python warnings a UFO model's own modules raise when they are compiled (source
`"ufo-model"`, for example Python 3.12's `SyntaxWarning` for an invalid escape
such as `'\lambda'` in a TeX name; one entry per compile).

`on_failure` selects the policy: `"best-effort"` (the default; `None`
means it), `"return"` (such points stay failed lanes, no extra evaluation)
and `"raise"` (the default before 1.0.3: `EvaluationError` at the first such
point). Accepted points are evaluated exactly as before, bitwise and at the
same cost; only refused points pay for their best-effort evaluation (one
unchecked evaluation with the call's checks and the recommended checks;
two when the call runs the cached reduction-direction check, which does not
combine with the rescaled one in one evaluation). A runtime already loaded with `reduction_unchecked=True` has no
checks to override: its failed points stay failed. Under a rescue policy
only the last level serves best-effort values (`rescue_counts` counts them
under `"best-effort"`).

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
the selected rows (every family that drops a link runs its pruned copy).
The amplitude-basis anchors are basis-dependent parts whose sum is the
helicity's value. Colour-flow selection does not exist for loop-induced
processes.

### Helicity selection of Born-interfered virtuals

A Born-interfered virtual selects its Born helicity rows at every colour
accuracy, with the tree engine's helicity IDs (`h:<values in leg
order>`) or the runtime's `h<index>`:

```python
one = runtime.evaluate(points, mu_r2=scales, helicities=["h3"])
half = runtime.evaluate(points, mu_r2=scales, helicities=ids[: len(ids) // 2])
rows = runtime.evaluate_resolved(points, mu_r2=scales, helicities=["h3", "h7"])
```

At next-to-leading and full colour the colour is contracted: a selection
keeps some helicity rows with every colour component of their Born, the
selected value is the sum of those rows' `evaluate_resolved` entries (one
helicity is its entry, bitwise up to the sign of a zero), and a complete set
of selections sums to the unselected value within rounding. Colour flows
exist at leading colour only (`color_flows=` there); at NLC and full colour
they are refused with a `CompatibilityError`. A selected call evaluates only
the selected helicities and nothing more: it forms the kept rows' Born and
counterterm trees, dresses the loop links those rows reach (every family
that drops a link runs its pruned copy) and reduces only the lanes of the
families they reach; `evaluate_resolved(helicities=...)` forms only the kept
rows' resolved columns (at leading colour the kept rows' and kept flows'
basis elements). A stored artifact serves the selection without
regeneration (artifacts generated before the feature included). The
correlated evaluations take no helicity or colour-flow selection (the v2
rule of both process types). The native SDKs serve the same helicity
selections ([Native APIs](native-apis.md)).

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

### Virtual and loop-induced results by squared order

On a virtual runtime (Born-interfered or loop-induced) `evaluate_orders` is
an order-split view of ONE `evaluate` call with the same arguments
(`mu_r2`, `poles`, `stability_checks`, `reduction`, ...): a
`VirtualOrderEvaluation` whose `evaluation` is that `VirtualEvaluation`.
Nothing is evaluated twice and no order is summed silently:

```python
view = runtime.evaluate_orders(points, mu_r2=scales)
view.orders            # the squared-order sectors (each with its (Born, loop) pairs)
view.born_orders       # the Born squared orders (empty for a loop-induced result)
view.for_order(view.orders[0])        # per point: LaurentCoefficients of one sector
view.born_for_order(view.born_orders[0])  # per point: the Born of one order
view.sum_orders()      # explicit per-point sum over the sectors, per Laurent coefficient
view.sum_born_orders() # explicit per-point sum over the Born orders
```

The sum of the partials is an identity. The test suite checks it against
`evaluate_resolved(...).total()` summed over the sectors, a different
contraction of the same amplitudes, to rounding: within gamma_n S, the
binary64 summation bound of the n resolved entries against the
pre-cancellation scale S. The walkthrough (`examples/python/loop_walkthrough.py`)
checks it against another public path at the same settings that does not
read the partials: for a loop-induced example (`g g > z g g`,
`g g > e+ e- m+ m-`) the sum of one call per helicity (`helicities=[h]`,
each forming only that helicity's rows, a different contraction); for a
Born-interfered one the correlated route's `"born"` history summed over the
sectors (within 28
units of roundoff of S for `u u~ > t t~ g g`; bitwise for
`u u~ > d d~ e+ e-`, where it reads the same contraction as the plain
evaluation, so there it checks the split's bookkeeping only: that example's
resolved evaluation takes about 9 GiB at one point). The walkthrough prints
the deviation in units of roundoff of S and requires it within ten times
the evaluation's own estimate.

## Check a one-loop request

```python
from pyamplicol.api import check

report = check("loop_gg_zgg_generate.toml", points=3)  # or an artifact directory
for process in report.processes:
    print(process.process, process.status)
    for row in process.rows:
        print(row.check, row.subject, row.carrier, row.ratio, row.verdict)
assert report.exit_code == 0
```

`check` regenerates the request in memory and runs the pole check (Catani's
operator; loop-induced: finiteness), the Ward check and the
Goldstone-equivalence probe where it applies; see
[the command-line guide](command-line-interface.md#check) for the criteria,
the carriers and the report's fields.

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
Above binary64 the result is the value at the *improved* point of the given
inputs (the kinematics improvement below: the point is first moved onto exact
energy-momentum conservation and the exact external shells in the carrier);
with `kinematics_improvement="off"` Decimal input is evaluated with the
supplied decimal digits as given. Binary64 input cannot gain information
merely by requesting more arithmetic precision: the improvement makes it
consistent, not more accurate.

**Kinematics improvement.** A one-loop or tree evaluation in
double-double or arbitrary precision evaluates the value at the *improved*
point of the given inputs: before anything is evaluated in that carrier,
each point is moved onto exact energy-momentum conservation and the exact
external shells in the carrier's arithmetic, by default with MadLoop's
default (`kinematics_improvement="z-shift"`, MadLoop's `ImprovePSPoint = 2`
with its own fallback to `ImprovePSPoint = 1`, `"rescaling"`), or with
`"projection"` (frame-independent) or not at all (`"off"`); binary64
evaluations only with `kinematics_improvement_binary64=True` (default
False: binary64 values are unchanged). Both are `Runtime.load` defaults and
per-call arguments of every `evaluate*` method; the variants, the chain, the
bounds and the record are described in
[Configuration](configuration.md#kinematics-improvement-of-the-input-points).
Each lane diagnostic records it (`VirtualLaneDiagnostic.kinematics`, a
`VirtualKinematics`; `None` for a binary64 evaluation without the switch), a
rescue attempt per level (`VirtualRescueAttempt.kinematics`: each level
improves the original inputs in its own carrier). Tree results are values
only: a tree runtime's `take_warnings()` returns the record of the most
recent improved call (a summary line, e.g. `"kinematics-improvement:
arbitrary at precision 32 (z-shift): z-shift 4"`, and every typed warning
with its point index; the lines of the C ABI's
`rusticol_runtime_take_warnings_json`), replaced by the next improved call
and not kept while the warnings are muted; the typed warnings are also
issued as Python warnings. The tree binary64 switch runs the native tree
runtime's binary64 code (the C ABI's: the same values bit for bit). Its typed warnings
(`"kinematics-not-improved"`: evaluated as given; `"kinematics-inexact"`: the
produced point misses exactness) are records, not failures. The default is
exact in every frame: where the beams lie along the z axis it is MadLoop's
z-shift bit for bit; in other frames MadLoop's chain produces the
rescaling's point, which keeps a conservation residual of the input's own
order, and the chain finishes it with the projection. Do not switch the
improvement off above binary64 (`kinematics_improvement="off"`): a
near-forward binary64 point can then be served far off without a flag. Near forward, backward and thresholds that
matters: NinjaRS amplifies an input's shell or conservation defect by about
the inverse box Gram determinant, and no rotation check sees it (a rotation
keeps the defect), which the improvement removes.

**Near-forward accuracy.** At a nearly degenerate 2 -> 2 (sub-)box
(forward, backward or threshold kinematics: a small transverse momentum or
a pair near its threshold) a binary64 call WITHOUT stability checks can be
far off with no estimate (NinjaRS's binary64 engine and oneloop's binary64
box integral; g g > z z at sqrt(s) = 500 GeV and theta = 1e-4 is off by a
factor 1e18), as in 1.0.2. Request the recommended checks
(`stability_checks=VirtualStabilityChecks(1, True, rescaled=True)`) there,
which flag such values, and the rescue (`stability=VirtualRescuePolicy()`),
which evaluates them in double-double or arbitrary precision on the
promoted, improved point (within 1e-5 on the points measured). An estimate
is an estimate, not a bound: at W+ W- production at 1 TeV with pT = 4.9 GeV
the binary64 estimate 9.2e-6 accepts a value 4.8e-5 off. Resolved outputs
far below the point's largest (suppressed helicity rows of a boosted
near-forward point, 1e-13 .. 1e-21 of the largest) need about
log10(largest / output) + the conditioning + the target's digits of working
precision: TIRust at 20-32 digits can serve them badly off, which the
recommended checks flag; use 60 digits or more there. The per-point NinjaRS
fallback's guard can miss them too (an estimate, not a bound): at g g > z z,
theta = 1e-4 in a rotated frame, double-double rows 5e-11 of the largest are
served 1e-2 off at a guard estimate of 1e-5.

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
   failed at every level is returned with its best-effort value
   (`"best-effort"` lane, the default policy), else as a failed lane
   (raised only under `on_failure="raise"`).

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
accuracy and selection (`helicities=` at every accuracy, `color_flows=` at
leading colour) is supported; every level and every rotation evaluates the
selected rows only. A loop-induced process is rescued with what it serves: the pole
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

A one-loop artifact or runtime returns a `benchmarking.VirtualProfileResult`
instead: the per-point time of `minimum_samples` calls of `batch_size`
points each, a new batch every call (a path without `points`: the distinct flat
points `benchmarking.profile_points` draws with `evaluation.phase_space`'s
seed and energy; given `points` are taken in order, cycled when there are
fewer than the calls need) (`seconds_per_point`, the median call; `low`,
`high`) and `stages`, the time
per point in each stage group (trees, dressing, scatter, reduction, rest)
read from the native executor's stage report of the same calls, with the
backend record (`reduction`) and the setting (`unchecked`). A `RunConfig`'s
`evaluation.reduction`, `reduction_unchecked` and `mu_r2` select the backend,
the unchecked timing mode and the scale when a path is loaded:

```python
from pyamplicol.config import resolve_config

config = resolve_config(
    {"action": "benchmark", "evaluation": {"reduction_unchecked": True}}
).effective
profile = BenchmarkRunner(config).run("artifacts/loop_gg_zgg", points=points)
print(profile.seconds_per_point, dict(profile.stages))
```

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

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

runtime-and-selectors
native-apis
artifacts-and-portability
umami-api
../correlators
../correlator-conventions
```
