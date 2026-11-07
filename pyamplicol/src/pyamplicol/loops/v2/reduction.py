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
naming each built-in backend and the carriers it serves) when the extension
has the query, else NinjaRS at every carrier (an extension built before the
TIRust backend, m-lit lit-tir, which adds the query). An artifact may admit
fewer backends (``admitted``: TIR-only artifacts of rank > N + 2, m-lit
lit-eft). A call also runs on one route of the artifact runtime (``Route``:
``SUMMED``, the summed virtual through the native executor, or
``RESOLVED``, the colour-, spin- and helicity-resolved outputs and the
leading-colour selections through the stored observable columns), and the
effective backend is one the route's executor runs: the guard of every
route, so that a result never names a backend its values were not reduced
by.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from pyamplicol.api.errors import CompatibilityError
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
NAMES = {"ninjars": "NinjaRS", "tirust": "TIRust"}
KINDS = {"virtual": "Born-interfered", "loop-induced": "loop-induced"}


class Route(NamedTuple):
    """A route of the artifact runtime: what it evaluates (for messages) and
    the reduction backends its executor runs."""

    serves: str
    backends: frozenset[str]


# The routes and the backends their executors run. Every native reduction is
# NinjaRS's until the TIRust backend is wired to an executor (m-lit lit-tir:
# the summed route; the resolved outputs after it, lit-obs).
SUMMED = Route("the summed virtual", frozenset({"ninjars"}))
RESOLVED = Route(
    "the colour-, spin- and helicity-resolved outputs and selections",
    frozenset({"ninjars"}),
)


class ReductionUnavailable(CompatibilityError):
    """An explicitly requested reduction backend cannot serve the call (the
    native extension lacks it, it does not serve the carrier, or the
    artifact does not admit it). ``key`` is ``"reduction-unavailable"``."""

    key = "reduction-unavailable"


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
    "ReductionUnavailable",
    "Route",
    "native_backends",
    "resolve",
]
