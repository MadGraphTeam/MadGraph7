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

* ``"auto"`` takes the per-type default (``DEFAULTS``) when it serves the
  carrier, else NinjaRS, with the reason recorded;
* an explicit backend that cannot serve the call is refused typed
  (``ReductionUnavailable``): never a silent switch.

What a backend serves is the native extension's capability, never a
constant of this module: ``native.reduction_backends()`` (a JSON object
naming each built-in backend and the carriers it serves: TIRust in binary64
in an extension built with it, the default cargo feature ``tir``) when the
extension has the query, else NinjaRS at every carrier (an extension built
before the TIRust backend, m-lit lit-tir, which adds the query). An artifact
may admit fewer backends (``admitted``: TIR-only artifacts of rank > N + 2,
m-lit lit-eft). A call also runs on one route of the artifact runtime
(``Route``: ``SUMMED``, the summed virtual through the native executor, or
``RESOLVED``, the colour-, spin- and helicity-resolved outputs and the
leading-colour selections through the stored observable columns), and the
effective backend is one the route's executor runs: the guard of every
route, so that a result never names a backend its values were not reduced
by. Both routes run both backends; ``"auto"`` takes on a route only the
backends of its ``auto`` set: NinjaRS, until the loop-induced agreement of
the two backends is established (m-lit lit-cache), so ``"auto"`` keeps
NinjaRS for loop-induced processes too, with the reason recorded, and
TIRust serves explicit requests.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple, cast

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
# The per-type defaults of ``"auto"`` (RULED Q17 / Q40).
DEFAULTS: Mapping[str, str] = {"virtual": "ninjars", "loop-induced": "tirust"}
CARRIERS = ("double", "double-double", "arbitrary")
# The refusal of TIRust at an extended carrier on the Python reference path
# (the native one is ``runtime::TIRUST_BINARY64_ONLY``).
TIRUST_BINARY64_ONLY = "the TIRust reduction backend serves binary64 in this build"
NAMES = {"ninjars": "NinjaRS", "tirust": "TIRust"}
KINDS = {"virtual": "Born-interfered", "loop-induced": "loop-induced"}


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


# The routes and the backends their executors run (the native executor's
# ``reduction`` option, m-lit lit-tir). ``"auto"`` stays on NinjaRS until the
# loop-induced agreement of the two backends is checked (m-lit lit-cache,
# which adds TIRust to ``auto``).
SUMMED = Route(
    "the summed virtual", frozenset({"ninjars", "tirust"}), frozenset({"ninjars"})
)
RESOLVED = Route(
    "the colour-, spin- and helicity-resolved outputs and selections",
    frozenset({"ninjars", "tirust"}),
    frozenset({"ninjars"}),
)


# The effective backend of the native executor calls of the current context
# (``running``): the artifact runtime sets it per carrier once ``resolve`` has
# chosen it; the executors' call sites read it (``executor_options``).
_EFFECTIVE: ContextVar[str] = ContextVar("pyamplicol_v2_reduction", default="ninjars")


@contextmanager
def running(backend: str | None) -> Iterator[None]:
    """Run the native executor calls of this context on the effective
    ``backend`` (``resolve``'s ``VirtualReduction.effective``; None:
    NinjaRS), chosen once per call and carrier, never per point."""

    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown reduction backend {backend!r}")
    token = _EFFECTIVE.set(backend or "ninjars")
    try:
        yield
    finally:
        _EFFECTIVE.reset(token)


def executor_options() -> dict[str, str]:
    """The native executor's ``reduction`` option in the current context
    (``running``): empty for NinjaRS, the executor's default, so that a
    NinjaRS call passes exactly what it passed before the option existed."""

    backend = _EFFECTIVE.get()
    return {} if backend == "ninjars" else {"reduction": backend}


def prepare(executor: Any) -> None:
    """Build the TIRust plans of a native executor (``V2Executor``) from its
    plan's (propagators, rank) histogram when the current context runs
    TIRust (``running``), before the call's first point (m-lit PLAN §2.b.3:
    no first-call stall); kept for the executor's lifetime, so a later call
    only compares its settings, which are always TIRust's defaults here (the
    validation settings of ``V2Executor.prepare_reduction`` serve runs on the
    native executor itself). TIRust's typed refusal of a family (its plan
    construction, or the rank admission) is ``ReductionUnavailable``. Nothing
    for NinjaRS."""

    if _EFFECTIVE.get() != "tirust":
        return
    try:
        executor.prepare_reduction("tirust")
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


def _unavailable(
    backend: str,
    carrier: str,
    served: Mapping[str, frozenset[str]],
    admitted: Collection[str],
    route: Route,
) -> str | None:
    """Why ``backend`` cannot serve ``carrier`` on ``route`` (None: it can)."""

    name = NAMES[backend]
    if backend not in admitted:
        return (
            f"the artifact admits {[NAMES[item] for item in sorted(admitted)]} "
            f"only, not {name}"
        )
    if backend not in served:
        return f"{name} is not built into this native extension" + (
            " (the 'tir' feature)" if backend == "tirust" else ""
        )
    if carrier not in served[backend]:
        return f"{name} does not serve the {carrier} carrier in this native extension"
    if backend not in route.backends:
        return f"{name} is not wired to the executor of {route.serves} in this build"
    return None


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

    request = reduction_request(requested)
    if contribution not in DEFAULTS:
        raise ValueError(f"unknown contribution {contribution!r}")
    if carrier not in CARRIERS:
        raise ValueError(f"unknown carrier {carrier!r}")
    served = native_backends(native)
    allowed = frozenset(BACKENDS if admitted is None else admitted)
    if request != "auto":
        why = _unavailable(request, carrier, served, allowed, route)
        if why is not None:
            raise ReductionUnavailable(
                f"reduction {request!r} was requested, but {why}; request "
                "'auto' or the other backend"
            )
        return VirtualReduction(request, request, "requested")
    default = DEFAULTS[contribution]
    if allowed == {"tirust"}:
        # A TIR-only artifact: no NinjaRS fallback exists (m-lit Q18, Q24).
        default = "tirust"
    kind = KINDS[contribution]
    why = _unavailable(default, carrier, served, allowed, route)
    if (
        why is None
        and not route.automatic(default)
        and allowed != {"tirust"}
        and _unavailable("ninjars", carrier, served, allowed, route) is None
    ):
        # Served on explicit request, not yet by "auto" (``Route.auto``).
        return VirtualReduction(
            "auto",
            "ninjars",
            f"auto: NinjaRS; {NAMES[default]}, the {kind} default, serves "
            f"{route.serves} on explicit request until its agreement with "
            "NinjaRS on loop-induced processes is established",
        )
    if why is None:
        reason = (
            f"auto: {NAMES[default]}, the {kind} default"
            if default == DEFAULTS[contribution]
            else f"auto: {NAMES[default]}, the only backend the artifact admits"
        )
        return VirtualReduction("auto", cast(Any, default), reason)
    fallback = "ninjars"
    other = _unavailable(fallback, carrier, served, allowed, route)
    if fallback == default or other is not None:
        raise ReductionUnavailable(
            f"no reduction backend serves this {kind} call at the {carrier} "
            f"carrier: {why}" + ("" if other is None else f"; {other}")
        )
    return VirtualReduction(
        "auto",
        "ninjars",
        f"auto: {NAMES[default]}, the {kind} default, is unavailable ({why}); "
        "NinjaRS serves the call",
    )


__all__ = [
    "BACKENDS",
    "CARRIERS",
    "DEFAULTS",
    "RESOLVED",
    "SUMMED",
    "TIRUST_BINARY64_ONLY",
    "ReductionUnavailable",
    "Route",
    "executor_options",
    "native_backends",
    "prepare",
    "resolve",
    "running",
]
