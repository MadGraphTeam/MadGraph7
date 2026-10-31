# SPDX-License-Identifier: 0BSD
"""Formal 1/N_c expansion of tree colour sums (strict colour, graded output).

pyAmpliCol's tree ``lc``/``nlc`` are the *inherited* colour-matrix
approximation (``docs/correlator-conventions.md``, "Colour accuracy"): the
colour matrix is truncated with symbolic N_c, while the amplitude
coefficients stay at N_c = 3, including the numeric 1/3 of every
U(1)-subtraction (Fierz) piece. ``strict-lc``/``strict-nlc`` instead expand
the whole squared amplitude in N_c:

* every amplitude group carries its **N_c grade** g = -k, k the number of
  U(1)-subtraction auxiliaries of its trees (``generation.dag_grades``). Its
  value at N_c = 3 is 3^g times a kinematic coefficient that does not depend
  on N_c;
* a colour-matrix entry is an exact Laurent polynomial
  ``C[l, r] = sum_p c_p N_c^p`` (``color_contraction_laurent``);
* the term ``A_l C[l, r] conj(A_r)`` of power p has **N_c degree**
  ``d = p + g_l + g_r``. Its value at N_c = 3 is ``c_p 3^p A_l conj(A_r)``
  with the amplitudes as evaluated (``graded_weights``).

With P the reference degree of the colour sum (``reference_degree``: the
leading power of the inherited conventions, n_g + n_qqbar plus the gluon
emissions of a connection), ``strict-lc`` keeps the degree P,
``strict-nlc`` the degrees P, P-1 and P-2, and the graded full colour every
degree; N_c = 3 is inserted after the truncation. Graded output reports the
contribution of each kept degree separately; the degrees sum to the total.
The loop engine's projection follows the same rule (``loops.v2.colour``,
"Projection"), with its degree read from the exact seed cells.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .contraction_factors import (
    _coloured_word,
    _open_line_nc_power_terms,
    _rotate_to_reference_start,
    _two_line_gi_ui,
    _two_line_reference_start,
    exact_color_contraction_factor,
)
from .contraction_trace import _simplify_trace_terms_nc_power
from .contraction_types import NC
from .plan_types import GenericColorPlan, LCColorSector

StrictColorAccuracy = Literal["strict-lc", "strict-nlc"]
NcExpansionAccuracy = Literal["strict-lc", "strict-nlc", "full"]
STRICT_COLOR_ACCURACIES: tuple[str, ...] = ("strict-lc", "strict-nlc")
# Degrees below the reference degree P that each expansion keeps.
_DEPTH: dict[str, int | None] = {"strict-lc": 0, "strict-nlc": 2, "full": None}

Laurent = dict[int, Fraction]


@dataclass(frozen=True)
class NcExpansion:
    """The formal-N_c treatment of one tree artifact.

    ``accuracy`` is ``strict-lc`` (degree P), ``strict-nlc`` (P, P-1, P-2) or
    ``full`` (every degree: the graded full colour, equal to full colour at
    N_c = 3). An artifact with an expansion always reports its value per
    N_c degree.
    """

    accuracy: NcExpansionAccuracy

    def __post_init__(self) -> None:
        if self.accuracy not in _DEPTH:
            raise ValueError(f"unknown N_c expansion {self.accuracy!r}")

    @property
    def depth(self) -> int | None:
        return _DEPTH[self.accuracy]

    def kept(self, degree: int, reference: int) -> bool:
        """Whether the N_c degree ``degree`` survives at reference degree P."""

        depth = self.depth
        if depth is None:
            return True
        return reference - depth <= degree <= reference

    def to_json_dict(self) -> dict[str, object]:
        return {"accuracy": self.accuracy, "depth": self.depth}


def nc_expansion_for(accuracy: str, *, graded: bool = False) -> NcExpansion | None:
    """The expansion selected by ``color.accuracy`` and ``color.graded``."""

    if accuracy in STRICT_COLOR_ACCURACIES:
        return NcExpansion(accuracy)  # type: ignore[arg-type]
    if graded:
        if accuracy != "full":
            raise ValueError(
                "graded colour output needs a formal-N_c expansion: use "
                "color.accuracy='strict-lc', 'strict-nlc' or 'full'; the "
                f"inherited {accuracy!r} approximation has no N_c degrees"
            )
        return NcExpansion("full")
    return None


def reference_degree(
    color_plan: GenericColorPlan, *, emissions: int = 0, splittings: int = 0
) -> int:
    """P = n_g + n_qqbar + r - s of ``docs/correlator-conventions.md``.

    ``emissions`` is the number r of operations of a colour connection and
    ``splittings`` the number s of them that split a gluon into a quark pair
    (zero for the ordinary colour sum).
    """

    process = color_plan.process
    return (
        len(process.adjoint_labels)
        + process.color_endpoints.pair_count
        + emissions
        - splittings
    )


def color_contraction_laurent(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
) -> Laurent:
    """Exact Laurent coefficients ``{p: c_p}`` of one full-colour metric entry.

    The entry is the reference-normalised full-colour factor of
    ``exact_color_contraction_factor`` with N_c kept symbolic: the same trace
    reductions (``contraction_trace``), with the Fierz -1/N_c and every closed
    index loop counted as powers of N_c. The sum at N_c = 3 is checked
    against the full-colour factor exactly.
    """

    if color_plan.basis != "trace":
        raise ValueError(
            "the formal N_c expansion needs the trace/open-line colour basis; "
            "the adjoint DDM basis has no N_c-exact metric here"
        )
    terms = _metric_laurent(color_plan, left, right)
    full = exact_color_contraction_factor(color_plan, left, right, accuracy="full")
    value = sum(
        (coefficient * Fraction(NC) ** power for power, coefficient in terms.items()),
        Fraction(0),
    )
    if value != full:
        raise ValueError(
            f"formal N_c colour factor of sectors {left.id}, {right.id} is "
            f"{value} at N_c = 3, but the full-colour factor is {full}"
        )
    return terms


def _metric_laurent(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
) -> Laurent:
    open_line_count = color_plan.process.color_endpoints.pair_count
    iper = _coloured_word(left)
    jper = _coloured_word(right)
    n_ord = len(iper)
    if len(jper) != n_ord:
        return {}
    if open_line_count == 0:
        if n_ord == 0:
            return {0: Fraction(1)} if not iper and not jper else {}
        return _traces((iper, tuple(reversed(jper))))
    if open_line_count == 1:
        return _traces((tuple((*iper[1:-1], *reversed(jper[1:-1]))),))
    if open_line_count == 2:
        return _two_open_line_laurent(color_plan, left, right, n_ord)
    if left.kind == "open-lines" and right.kind == "open-lines":
        if len(left.open_color_lines) != len(right.open_color_lines):
            return {}
        return _nonzero(
            _open_line_nc_power_terms(left.open_color_lines, right.open_color_lines)
        )
    return {}


def _two_open_line_laurent(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
) -> Laurent:
    # Mirrors contraction_factors._two_open_line_full_factor_exact.
    reference_start = _two_line_reference_start(color_plan)
    iper = _rotate_to_reference_start(_coloured_word(left), reference_start)
    jper = _rotate_to_reference_start(_coloured_word(right), reference_start)
    reference = (
        _rotate_to_reference_start(
            _coloured_word(color_plan.sectors[0]), reference_start
        )
        if color_plan.sectors
        else iper
    )
    gi, ui = _two_line_gi_ui(color_plan, iper, reference)
    gj, uj = _two_line_gi_ui(color_plan, jper, reference)
    if ui == uj:
        return _traces(
            (
                tuple((*iper[1 : 1 + gi], *reversed(jper[1 : 1 + gj]))),
                tuple((*iper[gi + 3 : n_ord - 1], *reversed(jper[gj + 3 : n_ord - 1]))),
            )
        )
    if (ui, uj) in {(1, 2), (2, 1)}:
        return _traces(
            (
                tuple(
                    (
                        *iper[1 : 1 + gi],
                        *reversed(jper[gj + 3 : n_ord - 1]),
                        *iper[gi + 3 : n_ord - 1],
                        *reversed(jper[1 : 1 + gj]),
                    )
                ),
            ),
            coefficient=Fraction(-1),
        )
    return {}


def _traces(
    traces: tuple[tuple[int, ...], ...], *, coefficient: Fraction = Fraction(1)
) -> Laurent:
    return _nonzero(_simplify_trace_terms_nc_power(((coefficient, 0, traces),)))


def _nonzero(terms: Mapping[int, Fraction]) -> Laurent:
    return {int(power): Fraction(value) for power, value in terms.items() if value}


def graded_weights(
    terms: Mapping[int, Fraction],
    left_grade: int,
    right_grade: int,
) -> Laurent:
    """``{d: c_p 3^p}`` with d = p + g_l + g_r: the weight of
    ``A_l conj(A_r)`` (amplitudes as evaluated at N_c = 3) at each N_c degree.
    """

    out: Laurent = {}
    for power, coefficient in terms.items():
        if not coefficient:
            continue
        degree = int(power) + int(left_grade) + int(right_grade)
        out[degree] = out.get(degree, Fraction(0)) + coefficient * Fraction(NC) ** int(
            power
        )
    return {degree: value for degree, value in out.items() if value}


def kept_degrees(
    degrees: Iterable[int], expansion: NcExpansion, reference: int
) -> tuple[int, ...]:
    """The degrees ``expansion`` keeps among ``degrees``, decreasing."""

    return tuple(
        sorted(
            {int(d) for d in degrees if expansion.kept(int(d), reference)},
            reverse=True,
        )
    )


def nc_degree_color_id(degree: int) -> str:
    """Public colour-component ID of the contribution of N_c degree ``degree``."""

    return f"color:nc-degree:{int(degree)}"


__all__ = [
    "STRICT_COLOR_ACCURACIES",
    "NcExpansion",
    "NcExpansionAccuracy",
    "StrictColorAccuracy",
    "color_contraction_laurent",
    "graded_weights",
    "kept_degrees",
    "nc_degree_color_id",
    "nc_expansion_for",
    "reference_degree",
]
