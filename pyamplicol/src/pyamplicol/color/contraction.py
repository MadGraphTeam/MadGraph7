# SPDX-License-Identifier: 0BSD
"""Color-contraction planning and public factor selection."""

from __future__ import annotations

from collections.abc import Sequence

from .contraction_factors import (
    _coloured_word,
    _common_helicity_weight,
    _is_open_line_pair,
    _multi_quark_line_color_factors,
    _one_quark_line_color_factors,
    _pure_gluon_color_factor,
    _pure_gluon_color_factors,
    _two_quark_line_color_factors,
)
from .contraction_types import (
    ColorContractionEntry,
    ColorContractionPlan,
    ColorGroupDescriptor,
)
from .plan import GenericColorPlan, LCColorSector


def build_color_contraction_plan(
    color_plan: GenericColorPlan,
    groups: Sequence[ColorGroupDescriptor],
) -> ColorContractionPlan | None:
    """Build the final colour contraction over coherent amplitude groups.

    LC artifacts keep the historical Rusticol diagonal reduction path.  NLC and
    full-colour artifacts attach an explicit sparse colour matrix whose entries
    include AmpliCol's colour factors, so no leading-colour scalar factor should
    be applied again at runtime.
    """

    accuracy = color_plan.color_accuracy
    if accuracy == "lc":
        return None
    if accuracy not in {"nlc", "full"}:
        return ColorContractionPlan(
            color_accuracy=accuracy,
            supported=False,
            reason=f"unknown colour accuracy {accuracy!r}",
            group_count=len(groups),
            entries=(),
        )
    descriptors = tuple(groups)
    entries: list[ColorContractionEntry] = []
    sector_by_id = {sector.id: sector for sector in color_plan.sectors}
    factor_cache: dict[tuple[int, int], float] = {}
    descriptors_by_helicity: dict[tuple[object, ...], list[ColorGroupDescriptor]] = {}
    for descriptor in descriptors:
        descriptors_by_helicity.setdefault(descriptor.helicity_key, []).append(
            descriptor
        )
    for helicity_descriptors in descriptors_by_helicity.values():
        for left_offset, left in enumerate(helicity_descriptors):
            left_sector = sector_by_id.get(left.sector_id)
            if left_sector is None:
                return ColorContractionPlan(
                    color_accuracy=accuracy,
                    supported=False,
                    reason=f"missing colour sector {left.sector_id}",
                    group_count=len(groups),
                    entries=(),
                )
            for right in helicity_descriptors[left_offset:]:
                right_sector = sector_by_id.get(right.sector_id)
                if right_sector is None:
                    return ColorContractionPlan(
                        color_accuracy=accuracy,
                        supported=False,
                        reason=f"missing colour sector {right.sector_id}",
                        group_count=len(groups),
                        entries=(),
                    )
                sector_pair = (left.sector_id, right.sector_id)
                weight = factor_cache.get(sector_pair)
                if weight is None:
                    weight = color_contraction_factor(
                        color_plan,
                        left_sector,
                        right_sector,
                        accuracy=accuracy,
                        full_col_acc=20,
                    )
                    factor_cache[sector_pair] = weight
                if abs(weight) <= 0.0:
                    continue
                symmetry = 1.0 if left.group_id == right.group_id else 2.0
                helicity_weight = _common_helicity_weight(left, right)
                entries.append(
                    ColorContractionEntry(
                        left_group_id=left.group_id,
                        right_group_id=right.group_id,
                        weight_re=helicity_weight * weight,
                        symmetry_factor=symmetry,
                    )
                )
    return ColorContractionPlan(
        color_accuracy=accuracy,
        supported=True,
        reason=None,
        group_count=len(groups),
        entries=tuple(entries),
    )


def color_contraction_factors(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    full_col_acc: int = 20,
) -> tuple[float, float, float]:
    """Return reference-normalized (LC, NLC, full) colour factors."""

    n_quark_pairs = color_plan.process.quark_lines.quark_pair_count
    n_ord = len(_coloured_word(left))
    if len(_coloured_word(right)) != n_ord:
        return (0.0, 0.0, 0.0)
    if n_quark_pairs == 0:
        return _pure_gluon_color_factors(left, right, n_ord, full_col_acc)
    if n_quark_pairs == 1:
        return _one_quark_line_color_factors(left, right, n_ord)
    if n_quark_pairs == 2:
        return _two_quark_line_color_factors(color_plan, left, right, n_ord)
    if n_quark_pairs >= 3 and _is_open_line_pair(left, right):
        return _multi_quark_line_color_factors(color_plan, left, right)
    return (0.0, 0.0, 0.0)


def color_contraction_factor(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    accuracy: str,
    full_col_acc: int = 20,
) -> float:
    """Return only the requested reference-normalized colour factor."""

    n_quark_pairs = color_plan.process.quark_lines.quark_pair_count
    n_ord = len(_coloured_word(left))
    if len(_coloured_word(right)) != n_ord:
        return 0.0
    if n_quark_pairs == 0:
        return _pure_gluon_color_factor(
            left,
            right,
            n_ord,
            accuracy=accuracy,
            full_col_acc=full_col_acc,
        )
    values = color_contraction_factors(
        color_plan,
        left,
        right,
        full_col_acc=full_col_acc,
    )
    if accuracy == "lc":
        return values[0]
    if accuracy == "nlc":
        return values[1]
    if accuracy == "full":
        return values[2]
    return 0.0


__all__ = [
    "ColorContractionEntry",
    "ColorContractionPlan",
    "ColorGroupDescriptor",
    "build_color_contraction_plan",
    "color_contraction_factor",
    "color_contraction_factors",
]
