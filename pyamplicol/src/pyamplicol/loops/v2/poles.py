# SPDX-License-Identifier: 0BSD
"""Pole selection of the v2 virtual: ``poles = "all" | "none" | "uv" | "ir"``
(design §12.4; user request and decision 2026-10-23, REQUIREMENTS_ADDENDUM I).

Every pole coefficient of the Born-virtual interference is split by origin
with two regulators, ``P/eps = P_UV/eps_UV + P_IR/eps_IR`` and
``D/eps^2 = D/eps_IR^2`` (at one loop the double pole is always infrared).
The selected virtual (``PointResult.components["virtual"]``, the public
``values``) is, per mode:

* ``"all"`` (default): the renormalised virtual's poles ``P_UV + P_IR`` and
  ``D`` (eps_UV = eps_IR, the existing result);
* ``"none"``: the finite-only mode (no pole is computed; absent, not zero);
* ``"uv"``: the UV poles of the BARE (unrenormalised) loop amplitude, the
  reduction's UV poles (``P_UV`` of ``bare``; the double pole is 0). The
  counterterms' UV poles are the selected ``uv`` (UV_local plus the
  external-field factors' UV part) and ``r2`` components, and both origins of
  every component are in ``PointResult.pole_origins``: bare + counterterm UV
  poles = the renormalised virtual's UV poles, zero where renormalisation is
  complete;
* ``"ir"``: the IR poles of the RENORMALISED virtual, which Catani's operator
  predicts (``catani_poles``).

The finite part is identical in all four modes: the selection never touches
a finite term. Origins per component (``assemble.COMPONENTS``):

* ``bare``: every reduction lane carries, besides its combined Laurent
  integrals, the UV residue of its integral (``V2Reducer`` ``poles="split"``,
  the exact large-q expansion of the lane's numerator; it equals the sum of
  the reduction coefficients times oneloopmaster's UV master residues,
  scaleless bubbles included). IR = combined - UV; the double pole is IR.
* ``r2`` and ``uv_local``: local counterterms of the model, ultraviolet by
  origin (renormalisation of couplings and masses; R2 is finite).
* ``uv_external_fields``: the model's on-shell external-field factors are
  written with one eps (UFO ``particle.counterterm``; a massless delta-Z,
  Z_UV (1/eps_UV - 1/eps_IR), vanishes and is omitted, a massive one mixes
  1/eps_UV and 2/eps_IR). A UFO model does not state the origin, so their
  ultraviolet part is fixed by the ultraviolet finiteness of the renormalised
  virtual: ``fields_UV = -(bare_UV + uv_local + r2)`` per sector and pole
  coefficient, and ``fields_IR = fields - fields_UV``. The bare-plus-
  counterterm UV cancellation is therefore exact by construction; what is
  tested is ``fields_UV`` against the textbook Feynman-gauge field
  renormalisation of every external leg of loop_sm, the residue against the
  generic engine's UV-selected masters and the ``"ir"`` poles against
  Catani's operator (``tests/unit/test_loop_v2_poles.py``,
  ``rust/loop-reduction/tests/runtime_v2_reduce.rs``).

``catani_poles`` is the IR-pole prediction of the renormalised virtual
(Catani, Dittmaier, Trocsanyi, Phys. Lett. B500 (2001) 149): per ordered pair
of coloured legs (j, k), (1/C_j) [C_j (mu^2/s_jk)^eps V^(S)_jk + Gamma_j]
<T_j.T_k>, with V^(S) = 1/eps^2 (both massless), 1/(2 eps^2) + ln(m^2/s)/
(2 eps) (one massive, mass m), ln(rho)/(v eps) (both massive) and Gamma_j =
gamma_j/eps (massless) or C_j/eps (massive); n_f light flavours in gamma_g.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from pyamplicol.config import PoleSelection

# The origins of a split pole (``PointResult.pole_origins``, the public
# ``VirtualPoleParts``).
ORIGINS = ("uv", "ir")
# Components whose poles are ultraviolet by origin (local counterterms).
LOCAL_UV = ("r2", "uv_local")


def pole_mode(poles: PoleSelection | str) -> PoleSelection:
    """The pole selection of an evaluation option: a ``PoleSelection`` or its
    value (one spelling; a bool is refused)."""

    if isinstance(poles, str):
        try:
            return PoleSelection(poles)
        except ValueError:
            pass
    choices = ", ".join(repr(item.value) for item in PoleSelection)
    raise ValueError(f"poles must be one of {choices}, not {poles!r}")


def reducer_poles(mode: PoleSelection) -> str:
    """The ``poles`` option of the native reducer for a selection: the
    combined Laurent integrals (``"all"``), the finite part (``"none"``) or
    ``"split"`` (combined Laurent plus each lane's UV residue)."""

    if mode == PoleSelection.ALL:
        return "all"
    if mode == PoleSelection.NONE:
        return "none"
    return "split"


def has_poles(mode: PoleSelection) -> bool:
    return mode != PoleSelection.NONE


def splits(mode: PoleSelection) -> bool:
    """Whether ``mode`` splits the poles by origin (``"uv"`` and ``"ir"``)."""

    return mode in (PoleSelection.UV, PoleSelection.IR)


def select_lane(
    values: Sequence[Any], mode: PoleSelection, zero: Any
) -> tuple[Any, ...]:
    """A lane's ``[finite, 1/eps, 1/eps^2]`` in ``mode`` from the reducer's
    split columns ``[finite, 1/eps, 1/eps^2, 1/eps_UV]``."""

    finite, single, double, ultraviolet = values
    if mode == "uv":
        return (finite, ultraviolet, zero)
    if mode == "ir":
        return (finite, single - ultraviolet, double)
    raise ValueError(f"lane selection needs mode uv or ir, not {mode!r}")


def split_sector(
    components: Mapping[str, Any],
    bare_uv: Any,
    field_component: str,
    laurent: Callable[..., Any],
) -> dict[str, dict[str, Any]]:
    """UV and IR parts of every component of one sector (or output column).

    ``components`` maps ``bare``, ``r2``, ``uv_local`` and the field
    component to their combined Laurent values (``finite, single_pole,
    double_pole``); ``bare_uv`` is the bare loop's UV Laurent (finite part
    unused, double pole zero). Returns ``{origin: {component: Laurent}}``
    for those four components and the sums ``uv`` (UV_local plus fields) and
    ``virtual`` (bare + uv + r2, the renormalised virtual): each part keeps
    the combined finite part, and the two parts' poles add up to the
    combined poles (to rounding).
    """

    def poles(value) -> tuple[Any, Any]:
        return value.single_pole, value.double_pole

    ultraviolet: dict[str, tuple[Any, Any]] = {"bare": poles(bare_uv)}
    for name in LOCAL_UV:
        ultraviolet[name] = poles(components[name])
    # The renormalised virtual is UV finite: the field factors carry the rest.
    ultraviolet[field_component] = tuple(
        -(ultraviolet["bare"][k] + ultraviolet["uv_local"][k] + ultraviolet["r2"][k])
        for k in (0, 1)
    )
    out: dict[str, dict[str, Any]] = {origin: {} for origin in ORIGINS}
    for name, (single, double) in ultraviolet.items():
        value = components[name]
        out["uv"][name] = laurent(value.finite, single, double)
        out["ir"][name] = laurent(
            value.finite, value.single_pole - single, value.double_pole - double
        )
    for part in out.values():
        part["uv"] = part["uv_local"] + part[field_component]
        part["virtual"] = part["bare"] + part["uv"] + part["r2"]
    return out


def select_sector(
    parts: Mapping[str, Mapping[str, Any]],
    mode: PoleSelection,
    laurent: Callable[..., Any],
) -> dict[str, Any]:
    """The components an evaluation in ``mode`` (``"uv"`` or ``"ir"``)
    returns for one sector, from ``split_sector``'s parts.

    ``"ir"``: every component's IR part; the virtual's are the IR poles of
    the renormalised virtual (Catani's operator). ``"uv"``: every
    component's UV part, except the virtual, whose poles are the bare
    loop's UV poles (the unrenormalised amplitude) with the renormalised
    virtual's finite part, as in every mode. The counterterms' UV poles are
    then the ``uv`` and ``r2`` components, and virtual + uv + r2 poles are
    the renormalised virtual's UV poles (``parts["uv"]["virtual"]``).
    """

    selected = dict(parts[mode])
    if mode == "uv":
        virtual, bare = selected["virtual"], selected["bare"]
        selected["virtual"] = laurent(
            virtual.finite, bare.single_pole, bare.double_pole
        )
    return selected


# --------------------------------------------------------------- Catani


def catani_poles(
    coloured: Sequence[int],
    representation: Mapping[int, int],
    momenta: Mapping[int, Sequence[Fraction]],
    masses: Mapping[int, Fraction],
    correlated: Callable[[int, int], Any],
    *,
    casimir: Callable[[int], Fraction],
    gamma: Callable[[int], Fraction],
    mu2: Any,
    log: Callable[[Any], Any],
    real: Callable[[Any], Any],
    sqrt: Callable[[Any], Any],
) -> tuple[Any, Any] | None:
    """(single, double) pole of the renormalised virtual over (alpha_s/2pi)
    for one sector, massless and massive coloured legs; None where the
    operator makes no prediction.

    ``correlated(j, k)`` is the sector's colour-correlated Born <T_j.T_k>,
    ``momenta`` the all-incoming exact momenta, ``masses`` the squared
    masses of the coloured legs by the model's runtime values (0: massless,
    whatever the p^2 of rounded momenta), ``gamma(j)`` gamma_j of a massless
    leg (n_f light flavours). ``log``, ``real`` and ``sqrt`` act in the
    carrier. No prediction (None) at a vanishing invariant 2 p_j.p_k or a
    pair of massive legs at or below threshold ((p_j.p_k)^2 <= m_j^2 m_k^2):
    the caller reports the prediction absent. The massless-only arithmetic is
    that of the previous implementation.
    """

    single = double = real(0)
    for j in coloured:
        c_j = casimir(representation[j])
        massive_j = masses[j] != 0
        g_j = c_j if massive_j else gamma(j)
        for k in coloured:
            if k == j:
                continue
            value = correlated(j, k)
            dot = _minkowski_dot(momenta[j], momenta[k])
            invariant = abs(2 * dot)
            if invariant == 0:
                return None
            massive_k = masses[k] != 0
            if not massive_j and not massive_k:
                double += value
                single += (real(g_j) / real(c_j) + log(mu2 / real(invariant))) * value
            elif massive_j != massive_k:
                mass2 = masses[j] if massive_j else masses[k]
                double += value / 2
                single += (
                    real(g_j) / real(c_j)
                    + (log(real(mass2) / real(invariant)) + log(mu2 / real(invariant)))
                    / 2
                ) * value
            else:
                # v = sqrt(1 - m_j^2 m_k^2 / (p_j.p_k)^2), rho^2 = (1-v)/(1+v)
                # = ratio / (1+v)^2 (no cancellation in 1 - v at small masses).
                ratio = (masses[j] * masses[k]) / (dot * dot)
                if ratio >= 1:
                    return None
                v = sqrt(real(1 - ratio))
                log_rho = log(real(ratio) / ((1 + v) * (1 + v))) / 2
                single += (real(g_j) / real(c_j) + log_rho / v) * value
    return single, double


def _minkowski_dot(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


__all__ = [
    "LOCAL_UV",
    "ORIGINS",
    "catani_poles",
    "has_poles",
    "pole_mode",
    "reducer_poles",
    "select_lane",
    "select_sector",
    "split_sector",
    "splits",
]
