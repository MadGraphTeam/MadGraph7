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

**Models with pole regulators (M-SM MS-POLES).** MadLoop's electroweak UFOs
(loop_qcd_qed_sm and its G_mu variant) do state the origin: their CT
parameters are linear in two external parameters, ``epsUV`` and ``epsIR``
(UFO/SLHA block TECHNICAL, codes 1 and 2, default 1), which multiply every
1/eps_UV and 1/eps_IR of their pole coefficients (``(epsUV - epsIR)`` for a
massless delta-Z, ``(epsUV + 2 epsIR)`` for a massive one); the finite parts
do not read them. Where both are adjustable external parameters of the
request (``pole_regulators``), the counterterm components' UV parts are the
counterterm program at the point's parameters with the IR regulator at 0
(``regulated_parameters``; MadLoop's evaluation at ``epsIR = 0``) and their
IR parts the rest, ``CT(epsUV, epsIR) - CT(epsUV, 0)``, for ``r2``,
``uv_local`` and the field factors alike (``split_sector``'s
``counterterms_uv``): no remainder, so bare + counterterm UV poles = 0 is a
test (of the model's counterterms and of the bare residues), and the
``"ir"`` virtual is ``bare_IR + CT_IR``. An evaluation at ``(0, epsIR)``
would count a pole term that reads neither regulator twice (the UFOs'
``dMB_*`` massive-b corrections of the electroweak CT parameters are such
terms, UV in MadLoop's ``epsIR = 0`` reading), so the IR part is the
difference. ``"all"`` and ``"none"`` never form it: their values are
bitwise unchanged.

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
correlator is Q_j Q_k times the colour-summed Born interferences the sector's
QED correction dresses (``dressed_born``, cross-order ones included), the
prefactor alpha / 2 pi with alpha the coupling of the loops (``qed_coupling``:
the model's photon coupling, alpha_Gmu in a G_mu model), not alpha(0). A
joint request's prediction is the sum of its corrections' operators
(``sector_sum``). An on-shell external photon adds -gamma_gamma/eps per photon
only where its coupling is not alpha(0) (``EXTERNAL_PHOTON_CONVENTIONS``):
with alpha(0), delta Z_e^alpha(0) + 1/2 delta Z_AA has no light-fermion pole;
with the loops' coupling the photon's gamma_gamma = -2/3 sum_f N_c^f Q_f^2
(massless charged fermions) stays. A tagged photon (MG5_aMC's ``!a!``,
``SliceRequest.external_photons``: by default the final-state ones) couples
with alpha(0), the user's decision [U9] of 2026-10-25
(``EXTERNAL_PHOTON_CONVENTION``, WP3.2 EWI-6b); an untagged one (a PDF
photon) keeps the loops' coupling (``UNTAGGED_PHOTON_CONVENTION``, MS-PHOTON),
so the engine's operator passes the number of untagged photons.

The fixtures' provider poles of the electroweak rungs (MadLoop, Recola and
OpenLoops; g g > t t~, u u~ > e- e+ g, g g > t t~ e- e+ and
g g > t t~ e- e+ g at O(alpha)) and of the MadLoop G_mu controls
(e- e+ > mu- mu+, e- e+ > u u~) are the operator's references
(``tests/unit/test_loop_v2_qed_catani.py``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pyamplicol.config import PoleSelection

from .families import Orders, add_orders

# The origins of a split pole (``PointResult.pole_origins``, the public
# ``VirtualPoleParts``).
ORIGINS = ("uv", "ir")
# Components whose poles are ultraviolet by origin (local counterterms).
LOCAL_UV = ("r2", "uv_local")
# UFO/SLHA codes of a model's pole regulators (module docstring, "Models
# with pole regulators"): the parameters that multiply the 1/eps_UV and
# 1/eps_IR of its counterterm parameters (MadLoop's electroweak UFOs:
# epsUV and epsIR).
REGULATOR_CODES = {"uv": ("TECHNICAL", (1,)), "ir": ("TECHNICAL", (2,))}


@dataclass(frozen=True)
class PoleRegulators:
    """The names of a model's UV and IR pole regulators (``pole_regulators``)."""

    uv: str
    ir: str

    def record(self) -> dict[str, str]:
        return {"uv": self.uv, "ir": self.ir}

    @classmethod
    def from_record(cls, record: Mapping[str, str] | None) -> PoleRegulators | None:
        return None if record is None else cls(str(record["uv"]), str(record["ir"]))


def _by_code(externals: Iterable[tuple[str, str, Any]]) -> PoleRegulators | None:
    """The regulators among (name, LHA block, LHA code) of real external
    parameters (``REGULATOR_CODES``), None unless both exist once each."""

    externals = list(externals)
    found: dict[str, str] = {}
    for origin, (block, code) in REGULATOR_CODES.items():
        names = [
            name
            for name, lhablock, lhacode in externals
            if str(lhablock or "").upper() == block
            and tuple(int(c) for c in (lhacode or ())) == code
        ]
        if len(names) != 1:
            return None
        found[origin] = str(names[0])
    return PoleRegulators(**found)


def _value_texts(value: Mapping[str, Any]) -> list[tuple[int, str]]:
    """(epsilon power, source text) of a raw model value: a Laurent value's
    terms, else the value's expression at power 0."""

    if value.get("kind") == "laurent":
        return [(int(power), str(text)) for power, text in value["terms"]]
    text = value.get("expression")
    return [] if text is None else [(0, str(text))]


def model_pole_regulators(raw: Any) -> PoleRegulators | None:
    """The pole regulators a raw loop model declares: its real external
    parameters at ``REGULATOR_CODES`` (by code, not by name) that only the
    pole coefficients of its Laurent values (the counterterm parameters'
    1/eps terms) read; None when it has none, or when a tree parameter, a
    coupling or a finite part reads one (a model using those codes for
    something else: setting it to 0 would change more than the
    counterterms' IR poles)."""

    if raw is None:
        return None
    externals = []
    for record in raw.collections["parameters"]:
        fields = record["fields"]
        if fields.get("nature") == "external" and (
            str(fields.get("type", "real")).lower() == "real"
        ):
            externals.append(
                (record["name"], fields.get("lhablock"), fields.get("lhacode"))
            )
    regulators = _by_code(externals)
    if regulators is None:
        return None
    names = "|".join(re.escape(name) for name in (regulators.uv, regulators.ir))
    reads = re.compile(rf"(?<![A-Za-z0-9_.])(?:{names})(?![A-Za-z0-9_])")
    for collection in ("parameters", "counterterm_parameters", "couplings"):
        for record in raw.collections[collection]:
            if collection == "parameters" and record["name"] in (
                regulators.uv,
                regulators.ir,
            ):
                continue
            value = raw.exact_value(collection, record["id"])
            for power, text in _value_texts(value):
                if power >= 0 and reads.search(text):
                    return None
    return regulators


def pole_regulators(
    parameters: Iterable[Any], fixed: Container[str] = (), raw: Any = None
) -> PoleRegulators | None:
    """The pole regulators of a model (``REGULATOR_CODES``): its real
    external parameters at UFO/SLHA codes (TECHNICAL, 1) and (TECHNICAL, 2),
    by code, not by name; None unless both exist once each and neither is
    fixed (``fixed``: the request's and the preparation's fixed parameters,
    which the counterterm program substitutes). ``parameters`` are the
    compiled model's parameter records (``compiled.ir.parameters``); with
    its raw loop model ``raw`` the regulators must also be the ones it
    declares (``model_pole_regulators``: read only by pole coefficients)."""

    found = _by_code(
        (record.name, record.lhablock, record.lhacode)
        for record in parameters
        if record.nature == "external" and str(record.parameter_type).lower() == "real"
    )
    if found is None or found.uv in fixed or found.ir in fixed:
        return None
    if raw is not None and model_pole_regulators(raw) != found:
        return None
    return found


def regulated_parameters(
    values: Mapping[str, str], regulators: PoleRegulators
) -> dict[str, str]:
    """A point's complete parameter values with the IR regulator at 0: the
    counterterm program there is the counterterms' UV part (module
    docstring)."""

    out = dict(values)
    out[regulators.ir] = "0"
    return out


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
    counterterms_uv: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """UV and IR parts of every component of one sector (or output column).

    ``components`` maps ``bare``, ``r2``, ``uv_local`` and the field
    component to their combined Laurent values (``finite, single_pole,
    double_pole``); ``bare_uv`` is the bare loop's UV Laurent (finite part
    unused, double pole zero). ``counterterms_uv`` maps ``r2``, ``uv_local``
    and the field component to their UV Laurent values (finite parts
    unused): the counterterm program with the IR regulator at 0 for a model
    with pole regulators (module docstring); None (a model without them):
    the local counterterms' poles are UV and the field factors carry what
    makes the renormalised virtual UV finite. Returns ``{origin: {component:
    Laurent}}`` for those four components and the sums ``uv`` (UV_local plus
    fields) and ``virtual`` (bare + uv + r2, the renormalised virtual): each
    part keeps the combined finite part, and the two parts' poles add up to
    the combined poles (to rounding).
    """

    def poles(value) -> tuple[Any, Any]:
        return value.single_pole, value.double_pole

    ultraviolet: dict[str, tuple[Any, Any]] = {"bare": poles(bare_uv)}
    if counterterms_uv is not None:
        for name in (*LOCAL_UV, field_component):
            ultraviolet[name] = poles(counterterms_uv[name])
    else:
        for name in LOCAL_UV:
            ultraviolet[name] = poles(components[name])
        # The renormalised virtual is UV finite: the field factors carry the
        # rest.
        ultraviolet[field_component] = tuple(
            -(
                ultraviolet["bare"][k]
                + ultraviolet["uv_local"][k]
                + ultraviolet["r2"][k]
            )
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


# ------------------------------------------------- strong gamma_j and beta_0

# UFO spins (2s + 1).
_FERMION_SPIN, _SCALAR_SPIN = 2, 1
# SU(N_c), N_c = 3: Casimir C(R) and Dynkin index T(R) of the representations
# the strong operator covers (UFO colour 3, -3 and 8).
_N_C = Fraction(3)
_CASIMIRS = {3: (_N_C * _N_C - 1) / (2 * _N_C), 8: _N_C}
_DYNKIN = {3: Fraction(1, 2), 8: _N_C}
# The weight of one species in beta_0 (per unit T(R)): a Dirac fermion
# (particle and antiparticle) 2/3, a Majorana fermion 1/3, a complex scalar
# 1/6 and a real scalar 1/12 (the gluon and its ghosts give 11/6 C_A).
_BETA_WEIGHTS = {
    (_FERMION_SPIN, False): Fraction(2, 3),
    (_FERMION_SPIN, True): Fraction(1, 3),
    (_SCALAR_SPIN, False): Fraction(1, 6),
    (_SCALAR_SPIN, True): Fraction(1, 12),
}


def colour_casimir(colour: int) -> Fraction | None:
    """C(R) of a UFO colour representation; None outside 3, -3 and 8."""

    return _CASIMIRS.get(abs(int(colour)))


def strong_beta0(species: Iterable[tuple[int, int, bool]]) -> Fraction | None:
    """beta_0 = 11/6 C_A - sum_s w_s T(R_s) of a strong correction, the
    gluon's gamma_g (Catani: 11/6 C_A - 2/3 T_R n_f in QCD), over the
    massless coloured species ``(spin, colour, self-conjugate)`` other than
    the gluon and its ghosts that run in its loops (one entry per particle-
    antiparticle pair); None for a species the weights do not cover (a
    coloured vector, a representation without T(R))."""

    beta = Fraction(11, 6) * _N_C
    for spin, colour, real in species:
        weight = _BETA_WEIGHTS.get((int(spin), bool(real)))
        index = _DYNKIN.get(abs(int(colour)))
        if weight is None or index is None:
            return None
        beta -= weight * index
    return beta


def strong_gamma(
    spin: int, colour: int, beta0: Fraction | None, *, gauge: bool
) -> Fraction | None:
    """gamma_j of a massless coloured leg by its spin and representation: the
    gluon (``gauge``: the massless adjoint gauge vector) beta_0, a fermion
    3/2 C(R), a scalar 2 C(R); None for anything else (another coloured
    vector, a representation without C(R)) or a missing beta_0."""

    if gauge:
        return beta0
    casimir = colour_casimir(colour)
    if casimir is None:
        return None
    if int(spin) == _FERMION_SPIN:
        return Fraction(3, 2) * casimir
    if int(spin) == _SCALAR_SPIN:
        return 2 * casimir
    return None


# ----------------------------------------------- sectors the operators cover

# Why a sector has no prediction (``sector_predictions``; the reason of every
# unpredicted sector, ``LoopSlice.unpredicted_sectors``):
# * a soft exchange of a coupling the request does not correct is IR
#   divergent (its tree of order L - 2c exists, or may: a QCD correction of
#   u u~ > d d~ with gluon and photon Borns misses the photon exchanges);
UNCORRECTED_EXCHANGE = "uncorrected-exchange"
# * the operator sums a dressed Born interference whose loop order the request
#   does not compute;
UNCOMPUTED_BORN_PAIR = "uncomputed-born-pair"
# * the requested correction's loops leave out soft exchanges between legs
#   that a line of a species no requested correction circulates separates in
#   a Born tree (``separated_exchange``): u d > u d h at QCD = 0, the gluon
#   between the two quark lines of a Z- and a W-exchange tree.
SEPARATED_EXCHANGE = "separated-exchange"
UNPREDICTED_REASONS = (UNCORRECTED_EXCHANGE, UNCOMPUTED_BORN_PAIR, SEPARATED_EXCHANGE)

# The components of a Born tree (``separated_exchange``): the charged legs
# that lines of loop species join, one frozenset per component.
Partition = frozenset[frozenset[int]]


def separated_exchange(partitions: Iterable[Partition], *, abelian: bool) -> bool:
    """Whether a correction's computed loops miss soft exchanges its Catani
    operator assumes, for Born trees with the given ``partitions``.

    A loop of a soft exchange between charged legs j and k runs through the
    tree's lines between them, so the request computes it only where lines
    of species that a requested correction circulates join j and k
    (``LoopEligibility``'s species condition): within one component of the
    tree. The operator sums every pair. A non-abelian charge is conserved
    within each component and the colour tensor of a component is invariant,
    so <B|T_j.T_k|D> vanishes for j and k in two components of a tree D
    whenever every tree of the Born B has D's partition: the operator is
    then the computed loops' (u b > d t, the W joining two quark lines).
    With several partitions the cross terms do not vanish (u d > u d h at
    QCD = 0: Z and W exchange; u b > d t g: the gluon on either line). An
    abelian correlator Q_j Q_k <B|D> does not vanish across components, so
    an abelian operator needs one component per tree."""

    found = set(partitions)
    if abelian:
        return any(len(partition) > 1 for partition in found)
    return len(found) > 1


def sector_predictions(
    sectors: Sequence[tuple[Orders, Iterable[tuple[Orders, Orders]]]],
    born_orders: Iterable[Orders],
    corrections: Iterable[str],
    tree_exists: Callable[[Orders], bool | None],
    separated: Callable[[str, frozenset[Orders]], bool] | None = None,
) -> dict[Orders, str | None]:
    """Per sector of the slice, None where the requested corrections'
    operators predict its IR poles, else the reason (``UNPREDICTED_REASONS``)
    (``sectors``: (sector, its (Born order, loop order) pairs) of the
    slice; ``born_orders`` the Born amplitude orders of its tree program).

    The poles of a complete one-loop amplitude of order L are
    sum_c I_c A0_{L - 2c} over every coupling order c with L_c >= 2 (Catani:
    a soft or collinear exchange of a c-boson leaves a tree of order L - 2c).
    A request computes the loops of its corrections only (MadLoop's loop
    selection: a QCD correction drops the gluon-photon boxes), so a sector
    is predicted only if, for each of its loop orders L and each such c,

    * c is a requested correction and the tree of order L - 2c is one of the
      Born amplitudes of the slice or provably absent (its operator then
      holds every Born interference the loop's c-poles multiply), or
    * c is not requested and the tree of order L - 2c provably does not
      exist (the loop misses no IR-divergent c-exchange);

    if every Born interference a requested correction dresses in the
    sector (A0_a A0_b with a + b + 2c = sector) pairs with a requested loop
    order a + 2c (the operator sums over all of them); and if no requested
    correction's loops leave out an exchange its operator sums over the
    Born amplitudes it dresses there (``separated(c, orders)``,
    ``separated_exchange``). ``tree_exists(a)`` answers for an order a that
    is not a Born amplitude order: False when the process provably has no
    tree of order a, True when it has one, None when unknown (no
    prediction). A mixed QCD x EW sector of a QCD-only request (u u~ > d d~
    with gluon and photon Borns) is therefore not predicted: its order-g^2
    e^2 loops miss the photon exchanges that the QCD Born's QED poles
    need."""

    born = frozenset(born_orders)
    corrections = frozenset(corrections)
    loops = frozenset(loop for _sector, pairs in sectors for _born, loop in pairs)
    out: dict[Orders, str | None] = {}
    for sector, pairs in sectors:
        reason = None
        for _born, loop in pairs:
            for name, power in loop:
                if power < 2:
                    continue
                lower = add_orders(loop, ((name, -2),))
                exists = True if lower in born else tree_exists(lower)
                if name in corrections:
                    complete = exists is False or lower in born
                else:
                    complete = exists is False
                if not complete:
                    reason = UNCORRECTED_EXCHANGE
                    break
            if reason is not None:
                break
        if reason is None:
            dressed = {
                name: frozenset(
                    order
                    for a in born
                    for b in born
                    if add_orders(a, b, ((name, 2),)) == sector
                    for order in (a, b)
                )
                for name in sorted(corrections)
            }
            if not all(
                add_orders(a, ((name, 2),)) in loops
                for name, orders in dressed.items()
                for a in orders
            ):
                reason = UNCOMPUTED_BORN_PAIR
            elif separated is not None and any(
                orders and separated(name, orders) for name, orders in dressed.items()
            ):
                reason = SEPARATED_EXCHANGE
        out[sector] = reason
    return out


def predicted_sectors(
    sectors: Sequence[tuple[Orders, Iterable[tuple[Orders, Orders]]]],
    born_orders: Iterable[Orders],
    corrections: Iterable[str],
    tree_exists: Callable[[Orders], bool | None],
    separated: Callable[[str, frozenset[Orders]], bool] | None = None,
) -> frozenset[Orders]:
    """The sectors whose IR poles the requested corrections' operators
    predict (``sector_predictions`` without a reason)."""

    return frozenset(
        sector
        for sector, reason in sector_predictions(
            sectors, born_orders, corrections, tree_exists, separated
        ).items()
        if reason is None
    )


# ------------------------------------------------------------------ QED Catani

# The coupling of an on-shell external photon: "alpha0" (alpha(0) with its
# charge-renormalisation term, MadLoop's and OpenLoops' convention) or
# "loop-coupling" (the loops' alpha, e.g. alpha_Gmu throughout).
# ``EXTERNAL_PHOTON_CONVENTION`` is the one of a tagged photon, "alpha0" by
# the user's decision [U9] of 2026-10-25 (slice-status DECISIONS ~04:55);
# since MS-PHOTON the tags are per leg (``SliceRequest.external_photons``,
# MG5_aMC's ``!a!``) and an untagged photon takes
# ``UNTAGGED_PHOTON_CONVENTION``.
EXTERNAL_PHOTON_CONVENTIONS = ("alpha0", "loop-coupling")
EXTERNAL_PHOTON_CONVENTION = "alpha0"
UNTAGGED_PHOTON_CONVENTION = "loop-coupling"


def dressed_born(
    born: Mapping[Orders, Any], sector: Orders, correction: str, zero: Any
) -> Any:
    """The colour-summed Born interferences that the one-loop correction
    ``correction`` (a coupling-order name) of ``sector`` dresses: those of
    every squared Born order, cross-order interferences included, whose sum
    with the correction's two powers is ``sector`` (``zero`` if none)."""

    raised = add_orders(((correction, 2),))
    return sum(
        (value for order, value in born.items() if add_orders(order, raised) == sector),
        zero,
    )


def sector_sum(
    predictions: Sequence[Mapping[Orders, tuple[Any, Any]]],
) -> dict[Orders, tuple[Any, Any]]:
    """The (single, double) poles per sector of several corrections'
    operators (each over the same sectors), summed; one prediction is
    returned as it is."""

    first, *rest = predictions
    if not rest:
        return dict(first)
    if any(set(other) != set(first) for other in rest):
        raise ValueError("the corrections' predictions cover different sectors")
    return {
        sector: (
            sum((other[sector][0] for other in rest), single),
            sum((other[sector][1] for other in rest), double),
        )
        for sector, (single, double) in first.items()
    }


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
    convention: str = EXTERNAL_PHOTON_CONVENTION,
) -> tuple[Any, Any] | None:
    """(single, double) pole of the renormalised virtual over (alpha/2pi) for
    one sector's QED correction; None where the operator makes no
    prediction.

    ``charges`` are the all-incoming charges of the external legs (an
    outgoing leg carries its antiparticle's), every leg with a non-zero one;
    ``born`` the colour-summed Born interference the QED correction of the
    sector dresses; ``masses`` the legs' squared masses (``catani_poles``);
    ``gamma(j)`` gamma_j of a massless charged leg (``qed_gamma``).
    ``photons`` on-shell external photons add -gamma_gamma/eps each
    (``gamma_photon``, ``photon_gamma``) under the ``"loop-coupling"``
    convention and nothing under ``"alpha0"``, the production one (module
    docstring).
    """

    if convention not in EXTERNAL_PHOTON_CONVENTIONS:
        raise ValueError(f"unknown external-photon convention {convention!r}")
    if photons and convention == "loop-coupling" and gamma_photon is None:
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
        fermions = [
            i for i, record in enumerate(records) if int(record.spin) == _FERMION_SPIN
        ]
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


def coupling_value(
    coupling: tuple[str, tuple[tuple[str, str], ...]],
    parameters: Mapping[str, Fraction],
    carrier: Any,
) -> Any:
    """The real positive value of a ``qed_coupling`` expression at the
    exact external ``parameters`` in ``carrier`` (``reference_executor.
    Carrier``, evaluated as a kernel's irrational factor at the carrier's
    digits; an AP carrier inside its Decimal context), or None if it is not
    real and positive."""

    from .reference_executor import KernelConstants

    value = KernelConstants(parameters, carrier).constant(
        ((1, 0), (), ((("expr", coupling), Fraction(1)),), ())
    )
    if value.imag != 0 or not value.real > 0:
        return None
    return value.real


__all__ = [
    "EXTERNAL_PHOTON_CONVENTION",
    "EXTERNAL_PHOTON_CONVENTIONS",
    "LOCAL_UV",
    "ORIGINS",
    "SEPARATED_EXCHANGE",
    "UNCOMPUTED_BORN_PAIR",
    "UNCORRECTED_EXCHANGE",
    "UNPREDICTED_REASONS",
    "UNTAGGED_PHOTON_CONVENTION",
    "Partition",
    "catani_poles",
    "colour_casimir",
    "coupling_value",
    "dressed_born",
    "exact_charge",
    "has_poles",
    "photon_gamma",
    "pole_mode",
    "predicted_sectors",
    "qed_catani_poles",
    "qed_coupling",
    "qed_gamma",
    "reducer_poles",
    "sector_predictions",
    "sector_sum",
    "select_lane",
    "select_sector",
    "separated_exchange",
    "split_sector",
    "splits",
    "strong_beta0",
    "strong_gamma",
]
