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

``qed_catani_poles`` is the same operator for QED corrections (WP3.2 EWI-6;
Catani-Seymour hep-ph/9605323 with the abelian substitution, massive legs as
in hep-ph/0201036, QED with masses Dittmaier hep-ph/9904440): T_j.T_k ->
Q_j Q_k with the legs' all-incoming charges (Q eta of the outgoing-particle
convention; sum_j Q_j = 0), C_j -> Q_j^2, gamma_f = 3/2 Q_f^2 (massless
fermions), gamma_s = 2 Q_s^2 (massless scalars), massive legs soft only. The
correlator is Q_j Q_k times the colour-summed Born interference of the
sector (``_ir_poles``), the prefactor alpha / 2 pi with alpha the coupling of
the loops (``qed_coupling``: the model's photon coupling, alpha_Gmu in a
G_mu model), not alpha(0). An on-shell external photon adds -gamma_gamma/eps
per photon only where its coupling is not alpha(0)
(``EXTERNAL_PHOTON_CONVENTIONS``): with alpha(0), delta Z_e^alpha(0) + 1/2
delta Z_AA has no light-fermion pole; with the loops' coupling the photon's
gamma_gamma = -2/3 sum_f N_c^f Q_f^2 (massless charged fermions) stays. The
convention is the user's pending decision [U9] (``EXTERNAL_PHOTON_CONVENTION``
is None): until it is taken there is no prediction with an external photon,
and the generation refuses electroweak requests with one (``ew_policy``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
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


# ------------------------------------------------------------------ QED Catani

# UFO spins (2s + 1) of the charged massless emitters with a QED gamma.
_FERMION_SPIN, _SCALAR_SPIN = 2, 1
# The coupling of an on-shell external photon: "alpha0" (alpha(0) with its
# charge-renormalisation term, the recommended default of [U9], MadLoop's and
# OpenLoops' convention) or "loop-coupling" (the loops' alpha, e.g. alpha_Gmu
# throughout). ``EXTERNAL_PHOTON_CONVENTION`` is the production choice: None
# until the user decides [U9] (no prediction with an external photon).
EXTERNAL_PHOTON_CONVENTIONS = ("alpha0", "loop-coupling")
EXTERNAL_PHOTON_CONVENTION: str | None = None


def exact_charge(charge: float | Fraction | str) -> Fraction:
    """A UFO electric charge as an exact rational: UFO models write charges
    as Python floats (2/3 -> 0.6666666666666666), so the float is mapped to
    the nearest fraction with denominator <= 1000 and that fraction must
    reproduce it to 1e-12."""

    if isinstance(charge, Fraction):
        return charge
    value = Fraction(charge) if isinstance(charge, str) else Fraction(float(charge))
    exact = value.limit_denominator(1000)
    if abs(float(exact) - float(value)) > 1e-12 * max(1.0, abs(float(value))):
        raise ValueError(f"electric charge {charge!r} is not a small rational")
    return exact


def qed_gamma(spin: int, charge: Fraction) -> Fraction | None:
    """gamma_j of a massless charged leg (3/2 Q^2 for a fermion, 2 Q^2 for a
    scalar); None where QED Catani makes no prediction (a massless charged
    vector)."""

    if spin == _FERMION_SPIN:
        return Fraction(3, 2) * charge * charge
    if spin == _SCALAR_SPIN:
        return 2 * charge * charge
    return None


def photon_gamma(light_charges: Iterable[tuple[Fraction, int]]) -> Fraction:
    """gamma_gamma = -2/3 sum_f N_c^f Q_f^2 over the massless charged fermion
    flavours, given as (charge, colour dimension) once per flavour."""

    return -Fraction(2, 3) * sum(
        (colours * charge * charge for charge, colours in light_charges),
        Fraction(0),
    )


def qed_catani_poles(
    charges: Mapping[int, Fraction],
    momenta: Mapping[int, Sequence[Fraction]],
    masses: Mapping[int, Fraction],
    born: Any,
    *,
    gamma: Callable[[int], Fraction | None],
    mu2: Any,
    log: Callable[[Any], Any],
    real: Callable[[Any], Any],
    sqrt: Callable[[Any], Any],
    photons: int = 0,
    gamma_photon: Fraction | None = None,
    convention: str | None = EXTERNAL_PHOTON_CONVENTION,
) -> tuple[Any, Any] | None:
    """(single, double) pole of the renormalised virtual over (alpha/2pi) for
    one sector's QED correction; None where the operator makes no
    prediction.

    ``charges`` are the all-incoming charges of the external legs (an
    outgoing leg carries its antiparticle's), every leg with a non-zero one;
    ``born`` the colour-summed Born interference the QED correction of the
    sector dresses; ``masses`` the legs' squared masses (``catani_poles``);
    ``gamma(j)`` gamma_j of a massless charged leg (``qed_gamma``).
    ``photons`` on-shell external photons add -gamma_gamma/eps each under
    the ``"loop-coupling"`` convention and nothing under ``"alpha0"``
    (module docstring); with ``convention`` None (the user's decision [U9]
    pending) there is no prediction with photons.
    """

    if photons:
        if convention not in EXTERNAL_PHOTON_CONVENTIONS:
            return None
        if convention == "loop-coupling" and gamma_photon is None:
            raise ValueError("the loop-coupling convention needs gamma_gamma")
    if sum(charges.values(), Fraction(0)) != 0:
        raise ValueError(
            f"the legs' all-incoming charges do not add to zero: {charges}"
        )
    charged = sorted(label for label, charge in charges.items() if charge != 0)
    gammas = {}
    for label in charged:
        if masses[label] == 0:
            value = gamma(label)
            if value is None:
                return None
            gammas[label] = value
    prediction = catani_poles(
        charged,
        {label: charges[label] for label in charged},
        momenta,
        masses,
        lambda j, k: real(charges[j] * charges[k]) * born,
        casimir=lambda charge: charge * charge,
        gamma=lambda j: gammas[j],
        mu2=mu2,
        log=log,
        real=real,
        sqrt=sqrt,
    )
    if prediction is None:
        return None
    single, double = prediction
    if photons and convention == "loop-coupling":
        single = single - real(photons * gamma_photon) * born
    return single, double


# The UFO Lorentz structure of a plain vector coupling, Gamma(mu, i, j).
_GAMMA = re.compile(
    r"^UFO::\{\}::Gamma\(UFO::\{\}::idx\(1,(\d)\),UFO::\{\}::idx\(1,(\d)\),"
    r"UFO::\{\}::idx\(1,(\d)\)\)$"
)


def qed_coupling(
    compiled: Any, is_photon: Callable[[Any], bool]
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """alpha = e^2 / 4 pi of the model's QED vertices, as a Symbolica
    expression in its external parameters: its canonical text and the
    (symbol, parameter name) pairs it holds (``reference_executor``'s
    ``("expr", ...)`` factor); None if the model has no such vertex, its
    photon couplings are not charge-universal or the expression holds a
    symbol that is no external parameter.

    e is read from every photon vertex with a charged fermion pair whose only
    term is the plain vector coupling Gamma(mu, i, j) (UFO's FFV1): the
    coupling is +-i e Q_f, so alpha = -(coupling / Q_f)^2 / (4 pi), which all
    of them must give. No parameter name is assumed: in a G_mu model this is
    alpha_Gmu, in an alpha(M_Z) model alpha(M_Z), the coupling of the loops.
    """

    from symbolica import E, Expression

    from pyamplicol._internal.physics.symbols import symbols as registry

    particles = {record.name: record for record in compiled.ir.particles}
    terms: dict[str, list[Any]] = {}
    for term in compiled.ir.vertex_terms:
        terms.setdefault(term.vertex, []).append(term)
    found: set[str] = set()
    expressions = []
    for vertex in sorted(terms):
        cells = terms[vertex]
        if len(cells) != 1 or len(cells[0].particles) != 3:
            continue
        (term,) = cells
        records = [particles.get(name) for name in term.particles]
        if any(record is None for record in records):
            continue
        photons = [i for i, record in enumerate(records) if is_photon(record)]
        fermions = [i for i, record in enumerate(records) if int(record.spin) == 2]
        if len(photons) != 1 or len(fermions) != 2:
            continue
        match = _GAMMA.match(term.lorentz_source)
        if match is None or int(match.group(1)) != photons[0] + 1:
            continue
        charge = exact_charge(records[fermions[0]].charge)
        if charge == 0 or exact_charge(records[fermions[1]].charge) != -charge:
            continue
        coupling = E(term.coupling_expression)
        alpha = (
            -(coupling**2) / (4 * Expression.PI * E(str(charge * charge)))
        ).expand()
        text = alpha.to_canonical_string()
        if text not in found:
            found.add(text)
            expressions.append(alpha)
    if len(found) != 1:
        return None
    (alpha,) = expressions
    model = registry.model(compiled.name)
    names = {
        model.symbol(record.name).to_canonical_string(): record.name
        for record in compiled.ir.parameters
        if record.nature == "external"
    }
    pairs = []
    for symbol in alpha.get_all_symbols(False):
        if symbol.get_name() == "symbolica::𝜋":
            continue
        text = symbol.to_canonical_string()
        if text not in names:
            return None
        pairs.append((text, names[text]))
    return alpha.to_canonical_string(), tuple(sorted(pairs))


__all__ = [
    "EXTERNAL_PHOTON_CONVENTION",
    "EXTERNAL_PHOTON_CONVENTIONS",
    "LOCAL_UV",
    "ORIGINS",
    "catani_poles",
    "exact_charge",
    "has_poles",
    "photon_gamma",
    "pole_mode",
    "qed_catani_poles",
    "qed_coupling",
    "qed_gamma",
    "reducer_poles",
    "select_lane",
    "select_sector",
    "split_sector",
    "splits",
]
