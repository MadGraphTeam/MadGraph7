# SPDX-License-Identifier: 0BSD
"""Ward identities at any multiplicity on the native per-point input path
(M-QCD MQ-WARD; REQUIREMENTS_ADDENDUM H; design §8.4, oracle O10).

The check replaces one external gluon's polarisation vector by its momentum
on the *loop side* of the Born-virtual interference (the asymmetric Ward
mode of ``observables``): the blocks the loop dresses, the counterterm
currents and the external-field insertions' Born amplitude read
epsilon_j -> c p_j; the Born bra keeps physical polarisations. For a
gauge-invariant amplitude the renormalised interference (bare + UV + R2)
vanishes per Laurent coefficient and squared-order sector as a complex
number; an inclusive output projects it to its real part, so c = 1 and
c = i together check all of it (``WARD_FACTORS``). The Born interference of
the physical bra with the substituted Born amplitude
(``PointResult.loop_born``) vanishes with the tree Ward identity. The parts
need not vanish on their own: the R2 part cancels a bare violation of its
own size (c = i on u u~ > g g g, O10), and with massive quarks the local UV
part holds the mass counterterm, whose insertion is not gauge invariant
without the bare self-energies (g g > t t~ g g, leg 1, epsilon_1 -> i p_1:
40 times the physical scale, the virtual 1.4e-24 of it in double-double).

* **Native inputs.** ``plan_export.NativeExecutor(..., inputs="native")
  .evaluate(..., ward=...)`` hands each point's loop-side vector to the
  native trees (``rusticol-core`` ``loop_v2::trees::TreeProgram::
  inputs_loop_side``): the tree program runs twice per point, physically for
  the Born bra and with the vector in the leg's source slots (every
  helicity source of the leg, times its crossing phase) for the block rows,
  the counterterm program and the field insertions' Born ket; nothing is
  formed in Python per point, so the check runs at any multiplicity the
  native lane evaluates (g g > g g g g g included).
* **Python oracle.** The same call on ``inputs="python"`` forms the inputs in
  the Python reference lane (``assemble._PointEvaluation(loop_sources=...)``,
  ``TreeRows`` in Decimal), which the native one must reproduce; the native
  executor dresses and reduces both.
* **Kernels.** The transverse Yang-Mills kernels drop terms proportional to
  an input current's p.J, exact only for conserved currents, so a Ward slice
  is generated on the model's full tree kernels (``ward_slice``;
  ``blocks.KERNEL_VARIANT_BY_BLOCK_CLASS["ward"]``); the executor refuses
  the Ward mode on a slice with other kernels (``require_full_kernels``),
  and on a stored process, whose record does not carry its kernel variant.

A Ward value is the leg j (c = 1), a (leg, c) pair with c a complex number
(a number, a decimal string or a (real, imaginary) pair) or a (leg, vector)
pair with a literal complex four-vector, the non-gauge control in which
nothing vanishes (``observables._ward``). ``gauge_ward`` gives the check's
vectors, c p_j / p_j^0: of the size of a polarisation vector, so that the
substituted amplitudes are measured against the physical ones on equal
footing (c p_j alone scales them by the leg's energy).

``ward_check`` runs the check of one point and carrier in one batch (the
physical entry and every leg and factor) and judges each row.

**The criterion** (``ward_row``), per point, leg and factor, against the
physical evaluation at the same point; the scale of a Laurent coefficient is
the largest physical component magnitude over the sectors
(``ward_scales``):

* Born: |sum of ``loop_born``| <= ``BORN_BOUNDS`` x |physical Born|
  (binary64 1e-12, double-double 1e-24: the O1 rule's binary64 tolerance and
  its double-double agreement);
* virtual, every sector and Laurent coefficient: in double-double
  |V_k| <= 1e-25 x scale_k (O10's double-double bound); in binary64, where
  the reduction's rounding grows with the multiplicity, the user's O1 rule
  (m-qcd PLAN §1.4 rules 2 and 3) with the exact value 0 and the scale in
  place of the coefficient: E_k, the absolute stability estimate of the
  Ward evaluation itself from the reduction-direction check
  (``REDUCTION``), is within O1's binary64 scope, E_k <= 1e-9 x scale_k;
  |V_k| <= max(1e-12 x scale_k, E_k) (rule 2, ``BINARY64_FLOOR``); and
  |V_k| <= 10 E_k (rule 3, the 10x rule). A gauge violation does not
  depend on the reduction frame, so the estimate does not cover it. A
  rotation check is no rounding estimate here: the substituted leg's bra
  sums over its helicities against a helicity-independent ket, so each
  reduction lane changes by the helicities' little-group phases under a
  rotation and only the vanishing sum is invariant (g g > t t~ g g, leg 1:
  per-lane rotation budget 1.6 times the physical scale, the virtual
  2e-12 of it before and after the rotation). An evaluation without the
  reduction check has no binary64 criterion (``passed`` None).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .assemble import COMPONENTS, LoopSlice
from .stability import REDUCTION, StabilityChecks

# epsilon_j -> c p_j for c = 1 and c = i (as (real, imaginary) parts): the
# inclusive output is the real part of the interference, so the two factors
# check all of it.
WARD_FACTORS = ((1, 0), (0, 1))

# The components a row reports (the virtual is the criterion).
PARTS = ("virtual", "bare", "r2", "uv_local", "uv_external_fields")

# Module docstring, "The criterion".
BORN_BOUNDS = {"double": 1e-12, "double-double": 1e-24}
VIRTUAL_BOUNDS = {"double-double": 1e-25}
ESTIMATE_FACTOR = 10
BINARY64_SCOPE = 1e-9
BINARY64_FLOOR = 1e-12
# The stability check the binary64 criterion reads.
WARD_CHECKS = StabilityChecks(reduction_direction=True)


def ward_slice(compiled: Any, request: Any, native: Any, **options: Any) -> LoopSlice:
    """``LoopSlice.generate`` of ``request`` on the full tree kernels of the
    compiled model (``catalogue.compile_tree_kernel_variant``), which the
    Ward mode needs; ``options`` are ``generate``'s."""

    from .catalogue import compile_tree_kernel_variant

    full = compile_tree_kernel_variant(compiled, "full")
    return LoopSlice.generate(full, request, native, **options)


def require_full_kernels(slice_: Any) -> None:
    """Refuse the Ward mode on a slice (or a stored process's view) that was
    not generated on the full tree kernels."""

    blocks = getattr(slice_, "blocks", None)
    if getattr(blocks, "kernel_variant", None) != "full":
        raise ValueError(
            "the Ward mode needs the full tree kernels (the transverse "
            "Yang-Mills kernel assumes conserved currents): generate the slice "
            "with ward.ward_slice; a stored process does not serve it"
        )


def point_wards(ward: Any, count: int) -> list[Any]:
    """One Ward value (or None) per point: ``ward`` for every point, or a
    list with one entry per point."""

    if isinstance(ward, list):
        if len(ward) != count:
            raise ValueError("one Ward value per point")
        return list(ward)
    return [ward] * count


def ward_leg(ward: Any) -> int:
    """The leg label of a Ward value."""

    from .observables import _ward

    return _ward(ward)[0]


def loop_vector(
    ward: Any, momentum: Sequence[Any]
) -> tuple[tuple[Fraction, Fraction], ...]:
    """The exact (real, imaginary) components of the loop-side vector of a
    Ward value at a point: c p_j for the leg's physical momentum ``momentum``
    (as given: incoming legs incoming, outgoing legs outgoing), or the
    literal vector."""

    from .assemble import _fraction
    from .observables import _ward

    _leg, value = _ward(ward)
    if len(value) == 4:
        return tuple((Fraction(re), Fraction(im)) for re, im in value)
    re, im = value
    return tuple((re * _fraction(x), im * _fraction(x)) for x in momentum)


def gauge_ward(leg: int, factor: Any, point: Sequence[Sequence[Any]]) -> Any:
    """The Ward value epsilon_j -> c p_j / p_j^0 of leg ``leg`` at ``point``
    (``factor`` c as a (real, imaginary) pair): exact, light-like for a
    massless leg, with components of the size of a polarisation vector's."""

    from .assemble import _fraction

    energy = _fraction(point[leg - 1][0])
    re, im = factor
    return (leg, (Fraction(re) / energy, Fraction(im) / energy))


def rotated_ward(rotation: Any, ward: Any) -> Any:
    """A Ward value for the rotated point (``observables._rotated_ward``: a
    literal vector rotates with the momenta, c p_j by itself)."""

    from .observables import _rotated_ward

    return _rotated_ward(rotation, ward)


def gauge_legs(slice_: Any) -> tuple[int, ...]:
    """The labels of the slice's external massless vector legs (gluons and
    photons: the legs whose polarisation the Ward check may replace)."""

    particles = {int(entry["pdg_code"]): entry for entry in slice_.data["particles"]}
    out = []
    for leg in sorted(slice_.blocks.process.legs, key=lambda x: x.label):
        entry = particles[int(leg.pdg)]
        if int(entry["spin"]) == 3 and entry.get("mass") in (None, "ZERO", "0"):
            out.append(int(leg.label))
    return tuple(out)


def ward_scales(physical: Any) -> list[float]:
    """Per Laurent coefficient (finite, single, double pole) the physical
    scale of a Ward result: the largest component magnitude over the
    sectors of the physical evaluation ``physical`` (a ``PointResult``)."""

    width = len(next(iter(physical.components["virtual"].values())).coefficients)
    return [
        max(
            abs(complex(laurent.coefficients[k]))
            for name in COMPONENTS
            for laurent in physical.components[name].values()
        )
        for k in range(width)
    ]


def lane_scales(result: Any, average: int) -> list[float]:
    """Per Laurent coefficient the absolute sum of a result's family lanes'
    bare contributions, 2 |Omega| / (16 pi^2 average): the conditioning of
    the Ward cancellation (``test_loop_v2_observables._lane_scale``)."""

    import math

    from .conventions import LOOP_FACTOR_DENOMINATOR

    factor = 2 / (LOOP_FACTOR_DENOMINATOR * math.pi**2 * average)
    lanes = list(result.families.values())
    width = len(lanes[0].coefficients) if lanes else 1
    return [
        factor * sum(abs(complex(lane.coefficients[k])) for lane in lanes)
        for k in range(width)
    ]


def largest(result: Any, part: str) -> list[float]:
    """Per Laurent coefficient the largest magnitude of a component over the
    sectors of a result."""

    laurents = list(result.components[part].values())
    width = len(laurents[0].coefficients)
    return [
        max(abs(complex(laurent.coefficients[k])) for laurent in laurents)
        for k in range(width)
    ]


def born_ratio(result: Any, physical: Any) -> float:
    """|sum of a Ward result's loop-side Born interference| over |the
    physical Born| (``PointResult.loop_born``, ``born``)."""

    if result.loop_born is None:
        raise ValueError("the result has no loop-side Born (no Ward value)")
    ward = sum(complex(value) for value in result.loop_born.values())
    born = sum(complex(value) for value in physical.born.values())
    return abs(ward) / abs(born)


def _ratio(value: float, scale: float) -> float:
    """``value / scale``; a vanishing scale (a coefficient no physical
    component has, e.g. the double pole of the "uv" selection) admits only a
    vanishing value (0, else infinity)."""

    if scale:
        return value / scale
    return 0.0 if value == 0 else float("inf")


def ward_row(physical: Any, result: Any, carrier: str) -> dict[str, Any]:
    """The criterion of one Ward evaluation (module docstring): ``physical``
    and ``result`` are the ``PointResult`` of the same point without and with
    the Ward value, ``carrier`` "double" or "double-double". Returns the
    ratios to the physical scale (per Laurent coefficient, the largest over
    the sectors), the binary64 estimate's (E_k / scale_k, None without
    checks) with |V_k| / E_k (``estimate_over``, rule 3) and, in binary64,
    |V_k| / max(1e-12 scale_k, E_k) (``o1_over``, rule 2), the Born ratio
    and ``passed`` (None: binary64 without checks)."""

    if carrier not in BORN_BOUNDS:
        raise ValueError(f"no Ward criterion for the carrier {carrier!r}")
    scale = ward_scales(physical)
    width = len(scale)
    parts = {
        name: [_ratio(v, s) for v, s in zip(largest(result, name), scale, strict=True)]
        for name in PARTS
    }
    born = born_ratio(result, physical)
    row: dict[str, Any] = {
        "born": born,
        "born_passed": born <= BORN_BOUNDS[carrier],
        "parts": parts,
        "estimate": None,
        "estimate_over": None,
    }
    virtual = result.components["virtual"]
    if carrier in VIRTUAL_BOUNDS:
        bound = VIRTUAL_BOUNDS[carrier]
        virtual_passed = all(
            abs(complex(laurent.coefficients[k])) <= bound * scale[k]
            for laurent in virtual.values()
            for k in range(width)
        )
    else:
        virtual_passed = None
    estimate = result.stability
    if estimate is not None and set(estimate.budgets) != {REDUCTION}:
        raise ValueError(
            "the Ward criterion reads the reduction-direction check alone (a "
            "rotation check is no rounding estimate of a Ward evaluation)"
        )
    if estimate is not None:
        # Per coefficient, the largest E_k / scale_k and |V_k| / E_k over the
        # sectors.
        absolute = {
            sector: [abs(float(x)) for x in estimate.absolute[sector].coefficients]
            for sector in virtual
        }
        row["estimate"] = [
            _ratio(max(absolute[sector][k] for sector in virtual), scale[k])
            for k in range(width)
        ]
        row["estimate_over"] = [
            max(
                _ratio(abs(complex(laurent.coefficients[k])), absolute[sector][k])
                for sector, laurent in virtual.items()
            )
            for k in range(width)
        ]
        bracketed = all(x <= ESTIMATE_FACTOR for x in row["estimate_over"])
        if carrier == "double":
            in_scope = all(e <= BINARY64_SCOPE for e in row["estimate"])
            # O1 rule 2 per sector and coefficient: |V_k| / max(1e-12
            # scale_k, E_k), the largest over the sectors (<= 1 passes).
            row["o1_over"] = [
                max(
                    _ratio(
                        abs(complex(laurent.coefficients[k])),
                        max(BINARY64_FLOOR * scale[k], absolute[sector][k]),
                    )
                    for sector, laurent in virtual.items()
                )
                for k in range(width)
            ]
            row["in_scope"] = in_scope
            virtual_passed = (
                bracketed and in_scope and all(x <= 1 for x in row["o1_over"])
            )
        else:
            virtual_passed = bool(virtual_passed) and bracketed
    row["virtual_passed"] = virtual_passed
    row["passed"] = (
        None if virtual_passed is None else bool(virtual_passed and row["born_passed"])
    )
    return row


def ward_check(
    executor: Any,
    point: Sequence[Sequence[Any]],
    scale: Any,
    carrier: str,
    *,
    legs: Sequence[int] | None = None,
    factors: Sequence[Any] = WARD_FACTORS,
    parameters: Any = None,
    checks: bool | None = None,
) -> dict[str, Any]:
    """The Ward check of one point in one batch of ``executor`` (a
    ``plan_export.NativeExecutor`` of a ``ward_slice``, either input path):
    the physical evaluation and, per leg (default: ``gauge_legs``) and
    factor, the evaluation with ``gauge_ward``, judged by ``ward_row``.
    ``carrier`` is "double" or "double-double"; ``checks`` (default: in
    binary64 only) runs ``WARD_CHECKS``, which the binary64 criterion
    reads. Returns ``physical`` (the ``PointResult``), ``results`` (one per
    row), ``rows`` (``ward_row`` plus ``leg`` and ``factor``) and
    ``passed`` (every row; None if a row has no criterion)."""

    from .assemble import Precision

    precisions = {"double": Precision.double, "double-double": Precision.double_double}
    if carrier not in precisions:
        raise ValueError(f"no Ward criterion for the carrier {carrier!r}")
    legs = tuple(legs if legs is not None else gauge_legs(executor.slice))
    checked = carrier == "double" if checks is None else checks
    labels = [(leg, factor) for leg in legs for factor in factors]
    results = executor.evaluate(
        [point] * (len(labels) + 1),
        [scale] * (len(labels) + 1),
        parameters=parameters,
        precision=precisions[carrier](),
        ward=[None, *(gauge_ward(leg, factor, point) for leg, factor in labels)],
        stability=WARD_CHECKS if checked else None,
    )
    physical, results = results[0], results[1:]
    rows = []
    for (leg, factor), result in zip(labels, results, strict=True):
        rows.append(
            {**ward_row(physical, result, carrier), "leg": leg, "factor": factor}
        )
    verdicts = [row["passed"] for row in rows]
    return {
        "physical": physical,
        "results": results,
        "rows": rows,
        "passed": None if None in verdicts else all(verdicts),
    }


__all__ = [
    "BINARY64_FLOOR",
    "BINARY64_SCOPE",
    "BORN_BOUNDS",
    "ESTIMATE_FACTOR",
    "PARTS",
    "VIRTUAL_BOUNDS",
    "WARD_CHECKS",
    "WARD_FACTORS",
    "born_ratio",
    "gauge_legs",
    "gauge_ward",
    "lane_scales",
    "largest",
    "loop_vector",
    "point_wards",
    "require_full_kernels",
    "rotated_ward",
    "ward_check",
    "ward_leg",
    "ward_row",
    "ward_scales",
    "ward_slice",
]
