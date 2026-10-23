# SPDX-License-Identifier: 0BSD
"""Pole selection of the v2 virtual: ``poles = "all" | "none" | "uv" | "ir"``
(design §12.4, user request 2026-10-23).

Every pole coefficient of the Born-virtual interference is split by origin
with two regulators, ``P/eps = P_UV/eps_UV + P_IR/eps_IR`` and
``D/eps^2 = D/eps_IR^2`` (at one loop the double pole is always infrared):

* ``"all"`` (default) returns ``P_UV + P_IR`` and ``D`` (eps_UV = eps_IR, the
  existing result);
* ``"none"`` is the finite-only mode (no pole is computed; absent, not zero);
* ``"uv"`` returns the ultraviolet parts, ``"ir"`` the infrared parts.

The finite part is identical in all four modes: the selection never touches a
finite term. Origins per component (``assemble.COMPONENTS``):

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
  1/eps_UV and 2/eps_IR). Their ultraviolet part follows from the ultraviolet
  finiteness of the renormalised virtual: ``fields_UV = -(bare_UV + uv_local
  + r2)`` per sector and pole coefficient, and ``fields_IR = fields -
  fields_UV``. For loop_sm this is the textbook Feynman-gauge field
  renormalisation of every external leg (``tests/unit/test_loop_v2_poles.py``
  checks it against those constants, independently of the reduction).

Consequently the renormalised virtual has no UV pole (``"uv"``: zero up to
rounding; the components report the bare-loop UV poles and the counterterm
UV poles separately), and its IR poles equal those of ``"all"``, which
Catani's operator predicts (``catani_poles``; massless and massive emitters).

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
from typing import Any, Literal

PoleMode = Literal["all", "none", "uv", "ir"]
MODES: tuple[PoleMode, ...] = ("all", "none", "uv", "ir")
ORIGINS = ("uv", "ir")
# Components whose poles are ultraviolet by origin (local counterterms).
LOCAL_UV = ("r2", "uv_local")


def pole_mode(poles: bool | str) -> PoleMode:
    """The pole mode of an evaluation option: ``True`` is ``"all"``,
    ``False`` is ``"none"`` (finite only)."""

    if poles is True:
        return "all"
    if poles is False:
        return "none"
    if isinstance(poles, str) and poles in MODES:
        return poles  # type: ignore[return-value]
    raise ValueError(f"poles must be one of {MODES} (or a bool), not {poles!r}")


def reducer_poles(mode: PoleMode) -> bool | str:
    """The ``poles`` option of the native reducer for a mode: the combined
    Laurent integrals (True), the finite part (False) or ``"split"``
    (combined Laurent plus each lane's UV residue)."""

    if mode == "all":
        return True
    if mode == "none":
        return False
    return "split"


def has_poles(mode: PoleMode) -> bool:
    return mode != "none"


def select_lane(values: Sequence[Any], mode: PoleMode, zero: Any) -> tuple[Any, ...]:
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
    """UV and IR parts of the local components of one sector.

    ``components`` maps ``bare``, ``r2``, ``uv_local`` and the field
    component to their combined Laurent values (``finite, single_pole,
    double_pole``); ``bare_uv`` is the bare loop's UV Laurent (finite part
    unused, double pole zero). Returns ``{origin: {component: Laurent}}``:
    each part keeps the combined finite part, and the two parts' poles add up
    to the combined poles (to rounding).
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
    out: dict[str, dict[str, Any]] = {"uv": {}, "ir": {}}
    for name, (single, double) in ultraviolet.items():
        value = components[name]
        out["uv"][name] = laurent(value.finite, single, double)
        out["ir"][name] = laurent(
            value.finite, value.single_pole - single, value.double_pole - double
        )
    return out


def slot_switches(mode: PoleMode) -> Callable[[str, int], bool]:
    """Which counterterm slots (component, eps power) an evaluation needs.

    ``"none"`` drops every pole slot (finite-only). ``"all"``, ``"uv"`` and
    ``"ir"`` need every slot: in ``"ir"`` the UV_local pole slots fix the IR
    part of the external-field factors (``split_sector``)."""

    def on(component: str, power: int) -> bool:
        del component
        return power == 0 or mode != "none"

    return on


# --------------------------------------------------------------- Catani


def catani_poles(
    coloured: Sequence[int],
    representation: Mapping[int, int],
    momenta: Mapping[int, Sequence[Fraction]],
    correlated: Callable[[int, int], Any],
    *,
    casimir: Callable[[int], Fraction],
    gamma: Callable[[int], Fraction],
    mu2: Any,
    log: Callable[[Any], Any],
    real: Callable[[Any], Any],
    sqrt: Callable[[Any], Any],
) -> tuple[Any, Any]:
    """(single, double) pole of the renormalised virtual over (alpha_s/2pi)
    for one sector, massless and massive coloured legs.

    ``correlated(j, k)`` is the sector's colour-correlated Born <T_j.T_k>,
    ``momenta`` the all-incoming exact momenta, ``gamma(j)`` gamma_j of a
    massless leg (n_f light flavours). ``log``, ``real`` and ``sqrt`` act in
    the carrier. The massless-only arithmetic is that of the previous
    implementation (``assemble._PointEvaluation._ir_poles``).
    """

    single = double = real(0)
    masses = {j: _minkowski(momenta[j]) for j in coloured}
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
                # v = sqrt(1 - m_j^2 m_k^2 / (p_j.p_k)^2), rho = sqrt((1-v)/(1+v)).
                ratio = (masses[j] * masses[k]) / (dot * dot)
                v = sqrt(real(1 - ratio))
                log_rho = log((1 - v) / (1 + v)) / 2
                single += (real(g_j) / real(c_j) + log_rho / v) * value
    return single, double


def _minkowski(vector: Sequence[Fraction]) -> Fraction:
    e, x, y, z = vector
    return e * e - x * x - y * y - z * z


def _minkowski_dot(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


__all__ = [
    "LOCAL_UV",
    "MODES",
    "ORIGINS",
    "PoleMode",
    "catani_poles",
    "has_poles",
    "pole_mode",
    "reducer_poles",
    "select_lane",
    "slot_switches",
    "split_sector",
]
