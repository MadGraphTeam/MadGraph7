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
lit-eft).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any, Literal, cast

from pyamplicol.api.errors import CompatibilityError

# The public choices (``config.ReductionBackend``) and the backends.
REQUESTS = ("auto", "ninjars", "tirust")
BACKENDS = ("ninjars", "tirust")
# The per-type defaults of ``"auto"`` (RULED Q17 / Q40).
DEFAULTS: Mapping[str, str] = {"virtual": "ninjars", "loop-induced": "tirust"}
CARRIERS = ("double", "double-double", "arbitrary")
NAMES = {"ninjars": "NinjaRS", "tirust": "TIRust"}
KINDS = {"virtual": "Born-interfered", "loop-induced": "loop-induced"}


class ReductionUnavailable(CompatibilityError):
    """An explicitly requested reduction backend cannot serve the call (the
    native extension lacks it, it does not serve the carrier, or the
    artifact does not admit it). ``key`` is ``"reduction-unavailable"``."""

    key = "reduction-unavailable"


def requested_backend(value: object) -> Literal["auto", "ninjars", "tirust"]:
    """A validated request (``"auto"``, ``"ninjars"`` or ``"tirust"``; the
    ``config.ReductionBackend`` enum or its value)."""

    text = getattr(value, "value", value)
    if not isinstance(text, str) or text not in REQUESTS:
        raise ValueError(f"reduction must be one of {list(REQUESTS)}; got {value!r}")
    return cast(Literal["auto", "ninjars", "tirust"], text)


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
) -> str | None:
    """Why ``backend`` cannot serve ``carrier`` (None: it can)."""

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
    return None


def resolve(
    requested: object,
    *,
    contribution: str,
    carrier: str,
    native: Any,
    admitted: Collection[str] | None = None,
) -> Any:
    """The ``VirtualReduction`` of a call: the requested backend, the
    effective one at ``carrier`` and the reason (module docstring). An
    explicit request that cannot be served raises ``ReductionUnavailable``;
    ``"auto"`` without any backend for the carrier as well."""

    from pyamplicol.api.virtual_results import VirtualReduction

    request = requested_backend(requested)
    if contribution not in DEFAULTS:
        raise ValueError(f"unknown contribution {contribution!r}")
    if carrier not in CARRIERS:
        raise ValueError(f"unknown carrier {carrier!r}")
    served = native_backends(native)
    allowed = frozenset(BACKENDS if admitted is None else admitted)
    if request != "auto":
        why = _unavailable(request, carrier, served, allowed)
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
    why = _unavailable(default, carrier, served, allowed)
    if why is None:
        reason = (
            f"auto: {NAMES[default]}, the {kind} default"
            if default == DEFAULTS[contribution]
            else f"auto: {NAMES[default]}, the only backend the artifact admits"
        )
        return VirtualReduction("auto", cast(Any, default), reason)
    fallback = "ninjars"
    other = _unavailable(fallback, carrier, served, allowed)
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
    "REQUESTS",
    "ReductionUnavailable",
    "native_backends",
    "requested_backend",
    "resolve",
]
