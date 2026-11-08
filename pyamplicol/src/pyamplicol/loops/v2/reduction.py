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
naming each built-in backend and the carriers it serves: TIRust at every
carrier in an extension built with it, the default cargo feature ``tir``,
m-lit lit-tir-mp; which families it serves is TIRust's own capability
query, refused typed when the plans are built, ``prepare``) when the
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

The policy has ONE definition, in Rust (``rusticol_loop_reduction::
reduction_policy``, m-lit DECISIONS 2026-11-07 21:50 R7), which the native
SDKs share: the per-type defaults (``DEFAULTS``), the artifact runtime's
routes with their ``auto`` sets (``SUMMED``, ``RESOLVED``) and ``resolve``'s
choice and reason texts are read from the native extension
(``reduction_policy``, ``reduction_resolve``), never restated here.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple

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
    and those ``"auto"`` takes there (NinjaRS until the loop-induced
    agreement of the two backends is checked; m-lit lit-cache adds TIRust in
    the native policy)."""

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


def current() -> str:
    """The effective backend of the current context (``running``)."""

    return _EFFECTIVE.get()


def cross(backend: str) -> str:
    """The backend of the cross-backend check of a call reduced by
    ``backend`` (``stability.CROSS``): the other one of ``BACKENDS``."""

    others = [item for item in BACKENDS if item != backend]
    if backend not in BACKENDS or len(others) != 1:
        raise ValueError(f"no other reduction backend than {backend!r} is defined")
    return others[0]


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
    status, chosen, effective, reason = load_loop_native().reduction_resolve(
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
    return VirtualReduction(chosen, effective, reason)


__all__ = [
    "BACKENDS",
    "CARRIERS",
    "DEFAULTS",
    "ReductionUnavailable",
    "Route",
    "cross",
    "current",
    "executor_options",
    "native_backends",
    "policy",
    "prepare",
    "resolve",
    "route",
    "running",
]
