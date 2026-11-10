# SPDX-License-Identifier: 0BSD
"""The reduction backend of a v2 one-loop evaluation (m-lit PLAN §2.b.4,
RULED Q17 / Q18).

Two backends reduce the loop numerators: NinjaRS (integrand reduction, the
Born-interfered default) and TIRust (tensor-integral reduction, the
loop-induced default: a loop-induced process reduces per loop amplitude, so
tensor integrals computed once per family and point serve every amplitude,
USER 2026-11-05 ~18:12). Both serve both process types.

A call requests ``"auto"``, ``"ninjars"`` or ``"tirust"`` (``evaluate(...,
reduction=)``, else ``Runtime.load(..., reduction=)``, else the artifact's
generation default ``[loop] reduction``, else ``"auto"``). ``resolve``
chooses the effective backend once per call and carrier, never per point:

* ``"auto"`` takes the per-type default (``DEFAULTS``; TIRust on a TIR-only
  artifact) when it serves the carrier, else NinjaRS, with the reason
  recorded; on TIRust where NinjaRS serves the call too it has NinjaRS as
  its PER-POINT FALLBACK (``VirtualReduction.fallback``, m-lit F3 (c)): a
  point TIRust refuses or fails (also in the call's stability checks' own
  evaluations) is evaluated again on NinjaRS from its original exact
  inputs, at the call's carrier, after the TIRust call (``with_fallback``),
  and the backend that served it and the reason are recorded per point
  (``VirtualLaneDiagnostic.reduction``); a point NinjaRS fails too fails
  with both reasons. In the extended carriers (double-double, arbitrary
  precision; ``fallback_guarded``) every point the fallback serves is
  GUARDED (m-lit lit-fallback-guard, DECISIONS 2026-11-09 05:56 (1) and
  11:53): NinjaRS before 71eddb15 (pin 5b6a8f57) returned finite, badly wrong values as
  success at near-planar 2 -> 3 points of rank-n pentagon families, and in
  the extended carriers it still loses digits ~1/Gram at a nearly
  degenerate 2 -> 2 (sub-)box (its cut stage), so the fallback's value runs a
  ROTATION RE-TEST (``guard_rotation``: the recommended checks' rotation
  of the evaluated point, exact, at the call's own carrier; DECISIONS
  2026-11-10 18:54 (c) / 19:20 (c): it sees the double-double near-forward
  misses of the cached reduction-direction check, with no false rejection
  on the regular double-double sets), and the point fails typed (one
  reason, ``text("rejected")``, naming the fallback, TIRust's failure, the
  carrier, the estimate and the limit) when the check's relative estimate
  exceeds the guard's limit, the call's target never below 1e-5
  (``fallback_accepted``: max(target, 1e-5), USER 2026-11-09 ~11:53),
  decided on the outputs the call returns (the summed value, or each
  resolved, correlated or selected output against its own magnitude, a
  vanishing one against the largest returned output: ``fallback_estimate``,
  DECISIONS 2026-11-09 14:30 Q7); the
  estimate is in the point's record (``VirtualReduction.guard_estimate``),
  an accepted point keeps the fallback's value bitwise and its public
  estimate is the call's checks' own. The call's own checks are reused
  when their first rotation is the guard's; otherwise the guard's runs in
  addition, on the served points only (``with_fallback``). A binary64 fallback point is
  served unguarded (its protection is NinjaRS's own conditioning bound,
  m-lit lit-ninja-planar). Points TIRust serves run nothing more;
* an explicit backend that cannot serve the call is refused typed
  (``ReductionUnavailable``): never a silent switch, and it has no
  fallback.

TIRust serves at a TARGET ACCURACY (``evaluate(..., reduction_target=)``,
else ``Runtime.load(..., reduction_target=)``, else ``DEFAULT_TARGET`` of
the native policy, 1e-5): the relative accuracy of each point's result
(every squared-order sector against its own size, the rescue's
``"relative"`` metric), from which TIRust's method-choice acceptance is
calibrated in the native extension (``v2_tir::acceptance_of``); a rescue
level's own target drives its level. NinjaRS has no setting it drives.

The UNCHECKED setting (``evaluate(..., reduction_unchecked=)``, else
``Runtime.load(..., reduction_unchecked=)``, else off, the native policy's
``DEFAULT_UNCHECKED``; m-lit lit-unchecked) drops the backends' own
acceptance and refusal checks for a call, every reduction of the call
(rescue levels, the per-point fallback, the call's own stability checks'
re-evaluations) included: TIRust keeps its conditioning gates and method
choice as method selection, never computes its acceptance pass (the target
then drives nothing), and serves what they refuse by plain PV at gate 0 (its
last resort); NinjaRS runs with zero stability thresholds. ``resolve``'s
backend choice is unchanged; ``"auto"``'s per-point fallback then acts only
on the points the unchecked TIRust still fails. ``running`` carries it to
the executors (``executor_options``, ``prepare``).

What a backend serves is the native extension's capability, never a
constant of this module: ``native.reduction_backends()`` (a JSON object
naming each built-in backend and the carriers it serves: TIRust at every
carrier in an extension built with it, the default cargo feature ``tir``,
m-lit lit-tir-mp; which families it serves is TIRust's own capability
query, refused typed when the plans are built, ``prepare``) when the
extension has the query, else NinjaRS at every carrier (an extension built
before the TIRust backend, m-lit lit-tir, which adds the query). An artifact
may admit fewer backends (``admitted``: TIR-only artifacts, whose
declarations record ``reductions = ["tirust"]`` for a family beyond NinjaRS's
rank <= N + 2, m-lit lit-eft; ``"auto"`` takes TIRust there at every carrier
and rescue level, NinjaRS is refused typed, and their rank > N + 2 results
carry no correctness claim). A call also runs on one route of the artifact runtime
(``Route``: ``SUMMED``, the summed virtual through the native executor, or
``RESOLVED``, the colour-, spin- and helicity-resolved outputs and the
leading-colour selections through the stored observable columns), and the
effective backend is one the route's executor runs: the guard of every
route, so that a result never names a backend its values were not reduced
by. Both routes run both backends, and ``"auto"`` takes either on both
(its ``auto`` set; m-lit F3: TIRust for loop-induced processes, with the
per-point NinjaRS fallback).

The policy has ONE definition, in Rust (``rusticol_loop_reduction::
reduction_policy``, m-lit DECISIONS 2026-11-07 21:50 R7), which the native
SDKs share: the per-type defaults (``DEFAULTS``), the artifact runtime's
routes with their ``auto`` sets (``SUMMED``, ``RESOLVED``), the default
target, ``resolve``'s choice, its per-point fallback and every reason text
are read from the native extension (``reduction_policy``,
``reduction_resolve``, ``reduction_auto_default``, ``reduction_text``),
never restated here.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from pyamplicol.api.errors import CompatibilityError, ReductionUnavailable
from pyamplicol.config.models import ReductionBackend, reduction_request

if TYPE_CHECKING:
    from pyamplicol.api.virtual_results import VirtualReduction

# The backends: the public choices (``config.ReductionBackend``, the one
# definition; ``config.models.reduction_request`` validates a request)
# without ``"auto"``.
BACKENDS = tuple(
    item.value for item in ReductionBackend if item is not ReductionBackend.AUTO
)
CARRIERS = ("double", "double-double", "arbitrary")
NAMES = {"ninjars": "NinjaRS", "tirust": "TIRust"}


@functools.lru_cache(maxsize=1)
def policy() -> Mapping[str, Any]:
    """The native reduction policy (``reduction_policy()``: the backends,
    carriers, per-type defaults and the artifact runtime's routes), read
    once."""

    from pyamplicol.loops.native import load_loop_native

    return json.loads(load_loop_native().reduction_policy())


class _Defaults(Mapping[str, str]):
    """The per-type defaults of ``"auto"`` (RULED Q17 / Q40), read from the
    native policy on first use."""

    def _table(self) -> Mapping[str, str]:
        return policy()["defaults"]

    def __getitem__(self, key: str) -> str:
        return self._table()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())


# The per-type defaults of ``"auto"`` (the native policy's).
DEFAULTS: Mapping[str, str] = _Defaults()


def default_target() -> float:
    """The default target accuracy of a call (the native policy's one
    definition, ``reduction_policy::DEFAULT_TARGET``)."""

    return float(policy()["target"]["default"])


def default_unchecked() -> bool:
    """The default of the unchecked setting (the native policy's one
    definition, ``reduction_policy::DEFAULT_UNCHECKED``: off)."""

    return bool(policy()["unchecked"]["default"])


def auto_default(contribution: str, admitted: Collection[str] | None = None) -> str:
    """The backend ``"auto"`` takes by default for ``contribution`` on an
    artifact admitting ``admitted`` (None: every backend): the native
    policy's one helper (``reduction_policy::auto_default``)."""

    from pyamplicol.loops.native import load_loop_native

    found = load_loop_native().reduction_auto_default(
        contribution, None if admitted is None else set(admitted)
    )
    if found is None:
        raise ValueError(f"unknown contribution {contribution!r}")
    return str(found)


def text(
    kind: str,
    a: str = "",
    b: str = "",
    detail: str = "",
    *,
    carrier: str = "",
    estimate: float = math.nan,
    target: float = math.nan,
) -> str:
    """A reason text of the native policy (``reduction_text``): ``"served"``
    (a point the fallback ``b`` served after ``a`` refused or failed it,
    ``detail``), ``"failed"`` (a reason ``detail`` of the fallback ``a``),
    ``"rejected"`` (a point the fallback ``b`` served after ``a`` refused or
    failed it, ``detail``, whose value the fallback's guard rejected at the
    call's ``carrier``: its reduction-direction ``estimate`` above the
    guard's limit at the call's ``target``; m-lit lit-fallback-guard) or
    ``"cross"`` (the cross-backend check's refusal ``detail``)."""

    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    if kind != "rejected":
        return str(native.reduction_text(kind, a, b, detail))
    return str(
        native.reduction_text(
            kind,
            a,
            b,
            detail,
            carrier=carrier,
            estimate=float(estimate),
            target=float(target),
        )
    )


def fallback_accepted(estimate: float, target: float) -> bool:
    """Whether the per-point fallback's guard accepts a served point whose
    reduction-direction estimate is ``estimate`` at the call's ``target``:
    the native policy's one rule (``reduction_policy::fallback_accepted``: a
    number at most the guard's limit, max(target, 1e-5), ``FALLBACK_FLOOR``,
    USER 2026-11-09 ~11:53; a NaN estimate is rejected)."""

    from pyamplicol.loops.native import load_loop_native

    return bool(load_loop_native().reduction_fallback_accepted(estimate, target))


def fallback_guarded(carrier: str) -> bool:
    """Whether the per-point fallback's guard runs in the carrier
    ``carrier``: the native policy's rule (``reduction_policy::
    fallback_guarded``: the extended carriers only, double-double and
    arbitrary precision; USER 2026-11-09 ~11:53 'Double-double+ only')."""

    from pyamplicol.loops.native import load_loop_native

    return bool(load_loop_native().reduction_fallback_guarded(carrier))


class Route(NamedTuple):
    """A route of the artifact runtime: what it evaluates (for messages), the
    reduction backends its executor runs, and those ``"auto"`` may take
    there (``auto``; None: every backend it runs)."""

    serves: str
    backends: frozenset[str]
    auto: frozenset[str] | None = None

    def automatic(self, backend: str) -> bool:
        """Whether ``"auto"`` may take ``backend`` on this route."""

        return backend in (self.backends if self.auto is None else self.auto)


def route(key: str) -> Route:
    """The artifact runtime's route ``key`` ("summed": the summed virtual;
    "resolved": the colour-, spin- and helicity-resolved outputs and
    selections) from the native policy: what it serves, the backends its
    executor runs (the native executor's ``reduction`` option, m-lit lit-tir)
    and those ``"auto"`` may take there (both, m-lit F3: which one a call
    takes is ``resolve``'s, TIRust for a loop-induced process with the
    per-point NinjaRS fallback, NinjaRS for a Born-interfered one)."""

    found = policy()["routes"][key]
    return Route(
        str(found["serves"]),
        frozenset(found["backends"]),
        None if found["auto"] is None else frozenset(found["auto"]),
    )


def __getattr__(name: str) -> Route:
    # ``SUMMED`` and ``RESOLVED``: the native policy's routes, read on first
    # use (PEP 562; importable by name, not listed in ``__all__``).
    if name in ("SUMMED", "RESOLVED"):
        found = route(name.lower())
        globals()[name] = found
        return found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The effective backend of the native executor calls of the current context
# (``running``): the artifact runtime sets it per carrier once ``resolve`` has
# chosen it; the executors' call sites read it (``executor_options``).
_EFFECTIVE: ContextVar[str] = ContextVar("pyamplicol_v2_reduction", default="ninjars")
# The target accuracy of the current context (None: the default target).
_TARGET: ContextVar[float | None] = ContextVar("pyamplicol_v2_target", default=None)
# The unchecked setting of the current context (off by default).
_UNCHECKED: ContextVar[bool] = ContextVar("pyamplicol_v2_unchecked", default=False)
_KEEP: Any = object()


@contextmanager
def running(
    backend: str | None, target: float | None = _KEEP, unchecked: bool = _KEEP
) -> Iterator[None]:
    """Run the native executor calls of this context on the effective
    ``backend`` (``resolve``'s ``VirtualReduction.effective``; None:
    NinjaRS), chosen once per call and carrier, never per point, at the
    target accuracy ``target`` (``VirtualReduction.target``; None: the
    default; omitted: the enclosing context's, e.g. for a cross-backend
    check's evaluations) and with the unchecked setting ``unchecked``
    (``VirtualReduction.unchecked``; omitted: the enclosing context's)."""

    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown reduction backend {backend!r}")
    token = _EFFECTIVE.set(backend or "ninjars")
    kept = target is _KEEP
    held = None if kept else _TARGET.set(target)
    mode = None if unchecked is _KEEP else _UNCHECKED.set(bool(unchecked))
    try:
        yield
    finally:
        if mode is not None:
            _UNCHECKED.reset(mode)
        if held is not None:
            _TARGET.reset(held)
        _EFFECTIVE.reset(token)


def current() -> str:
    """The effective backend of the current context (``running``)."""

    return _EFFECTIVE.get()


def current_target() -> float | None:
    """The target accuracy of the current context (``running``; None: the
    default)."""

    return _TARGET.get()


def current_unchecked() -> bool:
    """The unchecked setting of the current context (``running``)."""

    return _UNCHECKED.get()


def cross(backend: str) -> str:
    """The backend of the cross-backend check of a call reduced by
    ``backend`` (``stability.CROSS``): the other one of ``BACKENDS``."""

    others = [item for item in BACKENDS if item != backend]
    if backend not in BACKENDS or len(others) != 1:
        raise ValueError(f"no other reduction backend than {backend!r} is defined")
    return others[0]


def executor_options() -> dict[str, object]:
    """The native executor's ``reduction`` and ``unchecked`` options in the
    current context (``running``): empty for NinjaRS without the unchecked
    setting, the executor's defaults, so that such a call passes exactly
    what it passed before the options existed."""

    backend = _EFFECTIVE.get()
    options: dict[str, object] = {} if backend == "ninjars" else {"reduction": backend}
    if _UNCHECKED.get():
        options["unchecked"] = True
    return options


def prepare(executor: Any) -> None:
    """Build the TIRust plans of a native executor (``V2Executor``) from its
    plan's (propagators, rank) histogram when the current context runs
    TIRust (``running``), before the call's first point (m-lit PLAN §2.b.3:
    no first-call stall); kept for the executor's lifetime, so a later call
    only compares its settings: TIRust's defaults at the context's target
    accuracy (``current_target``; another target sets the acceptance on the
    built plans). The validation settings of ``V2Executor.prepare_reduction``
    serve runs on the native executor itself. TIRust's typed refusal of a
    family (its plan construction, or the rank admission) is
    ``ReductionUnavailable``. Nothing for NinjaRS."""

    if _EFFECTIVE.get() != "tirust":
        return
    unchecked = {"unchecked": True} if _UNCHECKED.get() else {}
    try:
        executor.prepare_reduction("tirust", target=_TARGET.get(), **unchecked)
    except ValueError as error:
        raise ReductionUnavailable(
            f"reduction 'tirust' cannot serve this artifact: {error}"
        ) from error


def native_backends(native: Any) -> dict[str, frozenset[str]]:
    """The carriers each reduction backend of the native extension serves
    (``native.reduction_backends()``; without the query: NinjaRS at every
    carrier)."""

    query = getattr(native, "reduction_backends", None)
    if query is None:
        return {"ninjars": frozenset(CARRIERS)}
    found = query()
    record = json.loads(found) if isinstance(found, (str, bytes)) else found
    if not isinstance(record, Mapping):
        raise CompatibilityError("the native reduction-backend query is malformed")
    out: dict[str, frozenset[str]] = {}
    for name, carriers in record.items():
        if name not in BACKENDS or not set(carriers) <= set(CARRIERS):
            raise CompatibilityError(
                f"the native extension reports an unknown reduction backend or "
                f"carrier: {name!r} {sorted(carriers)!r}"
            )
        out[name] = frozenset(carriers)
    return out


def resolve(
    requested: object,
    *,
    contribution: str,
    carrier: str,
    native: Any,
    route: Route,
    admitted: Collection[str] | None = None,
) -> VirtualReduction:
    """The ``VirtualReduction`` of a call on ``route``: the requested
    backend, the effective one at ``carrier`` and the reason (module
    docstring). An explicit request that cannot be served raises
    ``ReductionUnavailable``; ``"auto"`` without any backend for the carrier
    as well."""

    from pyamplicol.api.virtual_results import VirtualReduction
    from pyamplicol.loops.native import load_loop_native

    request = reduction_request(requested)
    served = native_backends(native)
    status, chosen, effective, reason, fallback = load_loop_native().reduction_resolve(
        request,
        contribution,
        carrier,
        {name: set(carriers) for name, carriers in served.items()},
        None if admitted is None else set(admitted),
        route.serves,
        set(route.backends),
        None if route.auto is None else set(route.auto),
    )
    if status == "unavailable":
        raise ReductionUnavailable(reason)
    if status != "ok":
        raise ValueError(reason)
    return VirtualReduction(chosen, effective, reason, fallback=fallback)


_R = TypeVar("_R")


def _outputs(result: Any) -> list[Any]:
    """The point results of one point of a batch: the point's ``PointResult``
    (the summed route), or every output's (a mapping of them: the resolved
    and correlated routes, ``runtime.virtual.V2ObservablesRuntime``)."""

    return list(result.values()) if isinstance(result, Mapping) else [result]


def _without_checks(result: Any) -> Any:
    """``result`` as an evaluation without stability checks returns it: no
    estimate, and no loop-induced finiteness ratio (formed only with the
    term scale the checks request, ``induced.point_finiteness``)."""

    if isinstance(result, Mapping):
        return {key: _without_checks(item) for key, item in result.items()}
    return replace(result, stability=None, finiteness=None)


def fallback_estimate(
    outputs: list[list[tuple[float, float, float]]], roundoff: float
) -> float:
    """The per-point fallback guard's estimate of one point from the
    outputs the call returns: the native policy's one rule
    (``reduction_policy::fallback_estimate``; m-lit lit-fallback-guard,
    DECISIONS 2026-11-09 14:30 RULING Q7). ``outputs[o][s]`` is (magnitude,
    error, scale) of sector s of output o (``_guard_sectors``), ``roundoff``
    the carrier's unit roundoff: each output against its own magnitude (the
    ``"relative"`` metric), an output that vanishes (its magnitude within the
    estimate's rounding floor, ``FLOOR_UNITS`` u S) against the largest
    returned output of the point; NaN when any input is not a number."""

    from pyamplicol.loops.native import load_loop_native

    return float(load_loop_native().reduction_fallback_estimate(outputs, roundoff))


def _guard_sectors(estimate: Any) -> list[tuple[float, float, float]]:
    """Per sector of one output's estimate (a ``StabilityEstimate`` of the
    guard's rotation re-test alone) its magnitude (the largest |Laurent
    coefficient|, ``stability.relative``'s scale), its largest absolute
    estimate E and its largest term scale S (NaN where a value is not a
    number: a NaN budget is no estimate, and ``max`` would pass over it)."""

    def largest(values: Any) -> float:
        found = [abs(float(x)) for x in values]
        return math.nan if any(math.isnan(x) for x in found) else max(found)

    return [
        (
            largest(laurent.coefficients),
            largest(estimate.absolute[sector].coefficients),
            largest(estimate.scale[sector].coefficients),
        )
        for sector, laurent in estimate.reference.items()
    ]


def _guard_estimate(result: Any) -> float:
    """The guard's estimate of one point (``fallback_estimate``) from the
    rotation re-test of every output the call returns (the guard's rotation,
    ``guard_rotation``, the first rotation of the checks that ran:
    ``stability.ROTATION`` 0 alone, ``StabilityEstimate.subset`` of the
    point's checks when they ran others too): the summed value (one output)
    or each resolved, correlated or selected output (DECISIONS 2026-11-09
    14:30 Q7: the guard decides on the returned outputs, never on the
    point's sum over rows the call does not return). NaN when an output has
    no such estimate."""

    from .stability import ROTATION

    check = ROTATION.format(0)
    outputs = []
    roundoff = math.nan
    for output in _outputs(result):
        estimate = getattr(output, "stability", None)
        if estimate is None or check not in estimate.budgets:
            return math.nan
        if set(estimate.budgets) != {check}:
            estimate = estimate.subset([check])
        roundoff = estimate.roundoff
        outputs.append(_guard_sectors(estimate))
    return fallback_estimate(outputs, roundoff)


def guard_rotation() -> Any:
    """The fallback guard's rotation re-test (``reduction_policy::
    FALLBACK_CHECK``; DECISIONS 2026-11-10 18:54 (c) / 19:20 (c)): the
    recommended checks' rotation, ``stability.default_rotation`` at the
    native policy's ``fallback_rotation`` (``FALLBACK_ROTATION``, 0; the
    native SDKs' ``v2_stability::default_rotation``), applied exactly to
    the point the fallback evaluated (the caller's exact inputs, improved
    in the call's carrier)."""

    from .stability import default_rotation

    return default_rotation(int(policy()["fallback_rotation"]))


def _guard_check(checks: Any) -> Any:
    """The fallback guard's checks: its rotation (``guard_rotation``) alone,
    of the caller's checks' type (``VirtualStabilityChecks`` or
    ``stability.StabilityChecks``; None: the latter)."""

    rotation = guard_rotation()
    if checks is None:
        from .stability import StabilityChecks

        return StabilityChecks(rotations=(rotation,))
    return replace(
        checks,
        rotations=(rotation,),
        reduction_direction=False,
        rescaled=False,
        cross_backend=False,
    )


def _carries_guard(checks: Any) -> bool:
    """Whether ``checks`` run the guard's rotation first (its estimate,
    ``stability.ROTATION`` 0, is then the guard's: reused)."""

    rotations = tuple(getattr(checks, "rotations", ()) or ())
    return bool(rotations) and rotations[0] == guard_rotation()


def with_fallback(
    run: Callable[[list[int] | None, Any], tuple[list[_R], dict[int, list[str]]]],
    choice: VirtualReduction,
    count: int,
    checks: Any,
    carrier: str,
) -> tuple[list[_R | None], dict[int, list[str]], dict[int, VirtualReduction]]:
    """``run`` of a batch on the call's effective backend at its target
    (``running``; ``run(None, checks)``: every point with the call's
    stability ``checks``; ``run(indices, checks)``: the points at those
    batch indices, in that order, with the given checks, a
    ``VirtualStabilityChecks`` or ``stability.StabilityChecks`` of the
    caller's type or None; each returns its per-point results and
    failures), then, for a choice with a per-point fallback
    (``VirtualReduction.fallback``, m-lit F3 (c)), the points it refused or
    failed again on the fallback, from their original exact inputs, at the
    same ``carrier`` (the call's: ``CARRIERS``; required, so that no caller
    skips the guard by default), after the primary call, with the call's
    checks less the cross-backend check (a fallback point's is not run,
    m-lit R6).

    In the extended carriers (``fallback_guarded``: double-double and
    arbitrary precision, USER 2026-11-09 ~11:53; a binary64 fallback point
    is served unguarded, with no estimate) every point the fallback SERVED
    is GUARDED (m-lit lit-fallback-guard, DECISIONS 2026-11-09 05:56 (1) and
    11:53; its check since DECISIONS 2026-11-10 19:20 (c)): the rotation
    re-test (``guard_rotation``: the point the fallback evaluated, rotated
    exactly by the recommended checks' rotation and evaluated again, the
    outputs compared, ``stability.ROTATION`` 0) runs at the call's carrier
    (never escalated, USER 2026-11-08 ~20:55) and its relative estimate
    (``_guard_estimate``) is recorded in the point's record
    (``VirtualReduction.guard_estimate``). When the call's checks run that
    rotation first they are reused (no second rotated evaluation:
    ``RECOMMENDED_CHECKS``); without checks the fallback's evaluation runs
    it and the public result stays what an evaluation without checks
    returns; beside other checks (another rotation first, e.g. every
    default rescue level's ``euler:0.1,0.2,0.3``, or the reduction checks
    alone) the served points are evaluated once more with the guard's
    rotation alone. The guard decides on the outputs the call RETURNS
    (DECISIONS 2026-11-09 14:30, RULING Q7: a point's summed value, or each
    output of a resolved, correlated or selected call, whose evaluation
    dresses the selected rows only, USER 2026-11-08 'only the selected
    helicities are evaluated and nothing more'): each output's cached
    estimate relative to its own magnitude, an output that vanishes
    relative to the largest returned output (``fallback_estimate``). A
    point the guard's check fails is a point the fallback failed. A point
    whose estimate exceeds
    the guard's limit (the call's target, ``choice.target`` else the
    default target, never below 1e-5: ``fallback_accepted``) fails typed
    with one reason (``text("rejected")``); an accepted point keeps the
    fallback's value and the call's own estimate. The public estimate is
    always the call's checks' own.

    Returns every point's result, the failures (a point the fallback fails
    too keeps both backends' reasons; a guard-rejected point the one
    rejection reason) and, per point the fallback evaluated again, its
    record (the backend that served it, the reason and the guard's
    estimate; a failed point's: the fallback's failure). A primary backend
    that refuses the call's plans (TIRust's capability query,
    ``ReductionUnavailable``) leaves all ``count`` points to the fallback. A
    choice without a fallback runs ``run`` once (nothing else:
    Born-interfered, explicit and TIR-only calls)."""

    from pyamplicol.api.virtual_results import VirtualReduction

    fallback = choice.fallback
    try:
        with running(choice.effective, choice.target, choice.unchecked):
            results, failures = run(None, checks)
    except ReductionUnavailable as error:
        # The primary backend refuses the call's plans (TIRust's capability
        # query of a family): with a fallback, every point is the
        # fallback's, with that reason.
        if fallback is None:
            raise
        results = [None] * count
        failures = {index: [str(error)] for index in range(count)}
    records: dict[int, VirtualReduction] = {}
    if fallback is None or not failures:
        return results, failures, records
    indices = sorted(failures)
    # The fallback's checks (a fallback point's cross-backend check is not
    # run, m-lit R6) and how the guard's rotation re-test runs beside them:
    # reused, carried by the fallback's evaluation, or once more.
    plain = (
        replace(checks, cross_backend=False)
        if checks is not None and checks.cross_backend
        else checks
    )
    enabled = plain is not None and bool(plain.rotations or plain.reduction_direction)
    reused = enabled and _carries_guard(plain)
    target = choice.target if choice.target is not None else default_target()
    checking = fallback_guarded(carrier)
    with running(fallback, choice.target, choice.unchecked):
        # Without the call's checks the fallback's evaluation carries the
        # guard's check.
        merged = checking and not enabled
        again, still = run(indices, _guard_check(plain) if merged else plain)
        still = dict(still)
        served = [local for local in range(len(indices)) if local not in still]
        guarded: dict[int, float] = {}
        if checking and (reused or merged):
            guarded = {local: _guard_estimate(again[local]) for local in served}
        elif checking and served:
            chosen = [indices[local] for local in served]
            found, missed = run(chosen, _guard_check(plain))
            for k, local in enumerate(served):
                if k in missed:
                    # The guard's check fails the point (as the fallback's
                    # evaluation with the check does without the call's
                    # checks): the fallback failed it.
                    still[local] = list(missed[k])
                else:
                    guarded[local] = _guard_estimate(found[k])
    results = list(results)
    failures = dict(failures)
    for local, index in enumerate(indices):
        first = "; ".join(failures[index])
        if local in still:
            second = [text("failed", fallback, "", reason) for reason in still[local]]
            failures[index] = [*failures[index], *second]
            records[index] = VirtualReduction(
                choice.requested,
                fallback,
                "; ".join(second),
                target=choice.target,
                unchecked=choice.unchecked,
            )
            continue
        estimate = guarded[local] if checking else None
        if estimate is not None and not fallback_accepted(estimate, target):
            reason = text(
                "rejected",
                choice.effective,
                fallback,
                first,
                carrier=carrier,
                estimate=estimate,
                target=target,
            )
            failures[index] = [reason]
            records[index] = VirtualReduction(
                choice.requested,
                fallback,
                reason,
                target=choice.target,
                unchecked=choice.unchecked,
                guard_estimate=estimate,
            )
            continue
        results[index] = _without_checks(again[local]) if merged else again[local]
        del failures[index]
        records[index] = VirtualReduction(
            choice.requested,
            fallback,
            text("served", choice.effective, fallback, first),
            target=choice.target,
            unchecked=choice.unchecked,
            guard_estimate=estimate,
        )
    return results, failures, records


__all__ = [
    "BACKENDS",
    "CARRIERS",
    "DEFAULTS",
    "ReductionUnavailable",
    "Route",
    "auto_default",
    "cross",
    "current",
    "current_target",
    "current_unchecked",
    "default_target",
    "default_unchecked",
    "executor_options",
    "fallback_accepted",
    "fallback_estimate",
    "fallback_guarded",
    "native_backends",
    "policy",
    "prepare",
    "resolve",
    "route",
    "running",
    "text",
    "with_fallback",
]
