# SPDX-License-Identifier: 0BSD
"""Goldstone-equivalence probes of loop-induced requests (M-LIT LIT-EW, m-lit
PLAN §2.a.7 "massive external vectors", §4 L4 / LI-W1).

The amplitude-level Ward probe (``induced.probe``) replaces a massless gauge
boson's polarisation by its momentum; a massive vector V instead obeys the
Slavnov-Taylor identity of its gauge fixing,

    k_mu A^mu(... V(k) ...) = eta M_V A(... G(k) ...),

with G its Goldstone boson and eta a phase fixed by the model's field
conventions (-i for an outgoing Z or W- in MG5_aMC's loop models,
``tests/unit/test_loop_v2_ew_gauge.py::test_slavnov_taylor_identities``).
It needs the Goldstone vertices, so it is probed only in 't Hooft-Feynman
gauge (the gauge of every electroweak request, ``ew_policy.
feynman_gauge_model``); a QCD-only request in unitary gauge is checked by
the both-gauges equality instead. At one loop the identity holds for the
loop-induced amplitude with the tree-level phase: its O(alpha) corrections
multiply a tree that does not exist.

The probe evaluates, at one point and per (loop order, helicity row) on the
amplitude basis, a_V with epsilon_j -> p_j / E_j (``ward.gauge_ward``) on the
vector's slice and a_G on the Goldstone's (the same request with leg j's
particle replaced by its Goldstone partner, admitted as an external leg in
``processes.model.goldstone_legs`` only), and the residue

    r(l, h) = || a_V(l, h) - eta (M_V / E_j) a_G(l, h_j -> 0) ||_G

against E_V + (M_V / E_j) E_G, the two amplitudes' absolute estimates
(``induced.amplitude_estimates``: re-test and rotation budgets plus the
floor on the physical amplitude's term scale); it passes within ten times
that (the user's x10 rule, m-lit Q12). eta is a field convention (a
Goldstone's phase can be redefined), so the probe reads it from the
amplitudes: the unit phase of sum <b, a_V>_G with b = (M_V / E_j) a_G over
every row, then requires every row to hold with that one phase and unit
modulus (a wrong Goldstone vertex, a missing R2 vertex or a wrong
counterterm breaks a row or the modulus; ``phase`` is recorded). Where the
Goldstone side vanishes (h > z a: k.A(V) = 0 by the photon's transversality)
the probe checks that both sides vanish within their estimates.

Widths: the probe slices are the generation-time probe's (``ew_policy.
probe_widths``: zero widths outside the complex-mass scheme, which keeps its
complex masses).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from . import induced
from .induced import ESTIMATE_FACTOR

# UFO spins are 2s + 1.
SCALAR_SPIN = 1
VECTOR_SPIN = 3


def goldstone_partners(compiled: Any) -> dict[int, int]:
    """Signed PDG of every massive physical vector -> signed PDG of its
    Goldstone boson: the scalar flagged ``goldstoneboson`` of the vector's
    charge, colour and mass parameter (in 't Hooft-Feynman gauge a Goldstone
    has its vector's mass: MG5_aMC's G0 with MZ, G+ with MW). A structural
    reading, never names; a vector without exactly one such scalar has no
    partner (no probe)."""

    from .ew_policy import is_zero

    parameters = {record.name: record for record in compiled.ir.parameters}
    goldstones = [
        record
        for record in compiled.ir.particles
        if record.goldstoneboson
        and int(record.spin) == SCALAR_SPIN
        and record.auxiliary_kind is None
    ]
    out = {}
    for record in compiled.ir.particles:
        if (
            int(record.spin) != VECTOR_SPIN
            or int(record.ghost_number)
            or record.goldstoneboson
            or record.auxiliary_kind is not None
            or is_zero(str(record.mass), parameters, {})
        ):
            continue
        found = [
            goldstone
            for goldstone in goldstones
            if str(goldstone.mass) == str(record.mass)
            and Fraction(str(goldstone.charge)) == Fraction(str(record.charge))
            and int(goldstone.color) == int(record.color)
        ]
        if len(found) == 1:
            out[int(record.pdg_code)] = int(found[0].pdg_code)
    return out


def massive_vector_legs(slice_: Any) -> dict[int, int]:
    """The first leg (1-based label) of every external massive-vector species
    of a slice that has a Goldstone partner (``goldstone_partners``) -> the
    partner's signed PDG (outgoing convention of the leg's particle: the
    partner of an incoming leg's particle). One leg per species: Bose
    symmetry relates the others."""

    partners = goldstone_partners(slice_.compiled)
    out: dict[int, int] = {}
    seen = set()
    for leg in sorted(slice_.blocks.process.legs, key=lambda leg: leg.label):
        pdg = int(leg.pdg)
        if pdg in partners and pdg not in seen:
            seen.add(pdg)
            out[int(leg.label)] = partners[pdg]
    return out


def goldstone_process(slice_: Any, leg: int, goldstone: int) -> str:
    """The slice's process with leg ``leg``'s particle replaced by the
    Goldstone of PDG ``goldstone`` (same side), as process text."""

    names = {
        int(record.pdg_code): str(record.name)
        for record in slice_.compiled.ir.particles
    }
    legs = sorted(slice_.blocks.process.legs, key=lambda leg: leg.label)
    tokens = [
        names[goldstone if int(entry.label) == leg else int(entry.pdg)]
        for entry in legs
    ]
    initial = sum(1 for entry in legs if entry.is_initial)
    return f"{' '.join(tokens[:initial])} > {' '.join(tokens[initial:])}"


def goldstone_slice(
    compiled: Any, request: Any, native: Any, slice_: Any, leg: int, goldstone: int
) -> Any:
    """The probe slice of the Goldstone side: ``request`` (a loop-induced
    probe request of ``slice_``, ``induced.probe_request``) with leg
    ``leg``'s vector replaced by its Goldstone ``goldstone``, generated on
    the full tree kernels like the vector side's Ward slice
    (``ward.ward_slice``) with the Goldstone admitted as an external leg
    (``processes.model.goldstone_legs``). Same loop orders: a Goldstone's
    vertices carry its vector's coupling orders in the identity."""

    from pyamplicol.processes.model import goldstone_legs

    from .ward import ward_slice

    process = goldstone_process(slice_, leg, goldstone)
    with goldstone_legs():
        return ward_slice(compiled, replace(request, process=process), native)


@dataclass(frozen=True)
class GoldstoneResult:
    """A Goldstone-equivalence probe of one leg (``probe``): per (loop
    order, row) of the vector side the ratio of the residue to its estimate,
    the fitted phase eta (re, im) and the size of the Goldstone side against
    its estimate (``goldstone_ratio``: how far it is from vanishing; the
    identity is vacuous where both sides vanish)."""

    leg: int
    ratios: Mapping[tuple[Any, Any], float]
    phase: tuple[float, float] | None
    goldstone_ratio: float
    scales: Mapping = field(default_factory=lambda: MappingProxyType({}))

    @property
    def worst(self) -> float:
        return max(self.ratios.values(), default=0.0)

    @property
    def passed(self) -> bool:
        return self.worst <= ESTIMATE_FACTOR


def _mass_over_energy(point: Sequence[Sequence[Any]], leg: int, digits: int) -> Decimal:
    """sqrt(p_j^2) / p_j^0 of leg ``leg`` at the (on-shell) probe point,
    exact momenta to ``digits`` decimal digits."""

    from .assemble import _fraction

    p = [_fraction(x) for x in point[leg - 1]]
    square = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]
    with localcontext() as context:
        context.prec = digits
        mass = (Decimal(square.numerator) / Decimal(square.denominator)).sqrt()
        return mass / (Decimal(p[0].numerator) / Decimal(p[0].denominator))


def _inner(a: Sequence[Any], b: Sequence[Any], gram: Any, numbers: Any) -> Any:
    """<a, b>_G = sum_{t,u} conj(a_t) G_tu b_u."""

    total = numbers.zero
    for t, x in enumerate(a):
        inner = numbers.zero
        for u, y in enumerate(b):
            inner = inner + gram[t][u] * y
        total = total + x.conjugate() * inner
    return total


def _unit(value: Any, numbers: Any) -> Any:
    """value / |value| (1 for zero) in the carrier."""

    if numbers.double:
        return value / abs(value) if value else 1 + 0j
    from .reference_executor import Complex

    with numbers.scope():
        modulus = (value.real * value.real + value.imag * value.imag).sqrt()
        if not modulus:
            return Complex(Decimal(1), Decimal(0))
        return Complex(value.real / modulus, value.imag / modulus)


def _row_map(row: Any, leg: int) -> Any:
    """The Goldstone side's helicity row of a vector-side row: leg ``leg``
    (1-based) at helicity 0 (a scalar's)."""

    return tuple((i, 0 if i == leg - 1 else h) for i, h in row)


def probe(
    vector: Any,
    goldstone: Any,
    point: Sequence[Sequence[Any]],
    scale: Any,
    leg: int,
    *,
    parameters: Mapping[str, str] | None = None,
    precision: Any = None,
    checks: Any = None,
    physical: Mapping | None = None,
) -> GoldstoneResult:
    """The Goldstone-equivalence probe of leg ``leg`` (module docstring) at
    one point on the native executors of the vector's slice (``vector``, on
    the full tree kernels) and of the Goldstone's (``goldstone``,
    ``goldstone_slice``). ``physical``: the vector slice's physical row
    floors at the point (``induced.physical_floors``; formed when None).
    ``checks``: the estimate's checks (default ``RECOMMENDED_CHECKS``)."""

    from .assemble import Precision, _fraction
    from .stability import RECOMMENDED_CHECKS
    from .ward import gauge_ward, rotated_ward

    precision = precision or Precision.double_double()
    checks = RECOMMENDED_CHECKS if checks is None else checks
    frame = checks.frame
    if frame is None:
        raise ValueError("a probe needs the reduction re-test")
    ward = gauge_ward(leg, (1, 0), point)
    exact = tuple(tuple(_fraction(x) for x in p) for p in point)
    moved = [rotation.apply(exact) for rotation in checks.rotations]

    def run(executor, wards):
        carrier, (evaluation,) = induced._probe_runs(
            executor, [(point, wards[0])], scale, parameters, precision, frame
        )
        turned = []
        if moved:
            _carrier, rotated = induced._probe_runs(
                executor,
                [
                    (
                        shifted,
                        None if wards[0] is None else rotated_ward(rotation, wards[0]),
                    )
                    for rotation, shifted in zip(checks.rotations, moved, strict=True)
                ],
                scale,
                parameters,
                precision,
                None,
            )
            turned = [induced.rotation_errors(evaluation, r) for r in rotated]
        return carrier, evaluation, turned

    carrier, side_v, turned_v = run(vector, [ward])
    _carrier, side_g, turned_g = run(goldstone, [None])
    if physical is None:
        physical = induced.physical_floors(vector, point, scale, parameters)
    floors_v = induced.ward_floors(physical, leg)
    floors_g = induced.physical_floors(goldstone, point, scale, parameters)
    roundoff = carrier.unit_roundoff
    estimates_v = induced.amplitude_estimates(side_v, roundoff, floors_v, turned_v)
    estimates_g = induced.amplitude_estimates(side_g, roundoff, floors_g, turned_g)
    numbers = side_v.numbers
    width = induced.width_of(side_v)
    ratio_digits = 2 * precision.working_digits
    factor = _mass_over_energy(point, leg, ratio_digits)
    gram = side_v.gram
    size = len(side_v.slice.amplitude)
    # The Goldstone side on the vector side's amplitude basis (same colour
    # content; a tensor the Goldstone side lacks is zero there).
    index = {tensor: t for t, tensor in enumerate(side_v.slice.amplitude.tensors)}
    mapping = [index[tensor] for tensor in side_g.slice.amplitude.tensors]
    zero_vector = [[numbers.zero] * width for _ in range(size)]

    def goldstone_vector(key):
        order, row = key
        found = side_g.amplitude_vectors.get((order, _row_map(row, leg)))
        if found is None:
            return zero_vector
        out = [[numbers.zero] * width for _ in range(size)]
        with numbers.scope():
            scaled = numbers.real(factor)
            for t, values in zip(mapping, found, strict=True):
                out[t] = [value * scaled for value in values]
        return out

    keys = list(side_v.amplitude_vectors)
    sides = {key: goldstone_vector(key) for key in keys}
    with numbers.scope():
        overlap = numbers.zero
        for key in keys:
            a = side_v.amplitude_vectors[key]
            b = sides[key]
            for k in range(width):
                overlap = overlap + _inner(
                    [b[t][k] for t in range(size)],
                    [a[t][k] for t in range(size)],
                    gram,
                    numbers,
                )
        eta = _unit(overlap, numbers)
    # How far the Goldstone side is from vanishing (its norm over its
    # estimate, scaled like b).
    goldstone_ratio = 0.0
    for entries in estimates_g.values():
        for residue, estimate in entries:
            if residue:
                goldstone_ratio = max(
                    goldstone_ratio, residue / estimate if estimate else float("inf")
                )
    ratios = {}
    scaled_factor = float(factor)
    for key in keys:
        a = side_v.amplitude_vectors[key]
        b = sides[key]
        entries_v = estimates_v.get(key, ())
        order, row = key
        entries_g = estimates_g.get((order, _row_map(row, leg)), ())
        ratio = 0.0
        for k in range(width):
            with numbers.scope():
                residue = float(
                    induced.g_norm(
                        [a[t][k] - eta * b[t][k] for t in range(size)], gram, numbers
                    )
                )
            if not residue:
                continue
            estimate = (
                entries_v[k][1] if k < len(entries_v) else 0.0
            ) + scaled_factor * (entries_g[k][1] if k < len(entries_g) else 0.0)
            ratio = max(ratio, residue / estimate if estimate else float("inf"))
        ratios[key] = ratio
    phase = complex(eta)
    return GoldstoneResult(
        leg,
        ratios,
        (phase.real, phase.imag) if goldstone_ratio > ESTIMATE_FACTOR else None,
        goldstone_ratio,
        MappingProxyType(physical),
    )


__all__ = [
    "GoldstoneResult",
    "goldstone_partners",
    "goldstone_process",
    "goldstone_slice",
    "massive_vector_legs",
    "probe",
]
