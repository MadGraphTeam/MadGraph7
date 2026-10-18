# SPDX-License-Identifier: 0BSD
"""Cold-path fixed-Nf color projection for bounded physical controls.

The Born observable has its own leading degree. Bare, UV, R2 and external-field
virtual rows share one leading degree across all Laurent sectors, before model
inputs or helicities are evaluated. Distinct currents remain independent rows
unless the caller has an exact identity permitting their combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyamplicol.config.models import ColorAccuracy

from .color_bilinears import SUNColorSpace
from .color_projection import prepare_color_projection


@dataclass(frozen=True, slots=True)
class PhysicalColorProjection:
    accuracy: str
    born_degree: int | None
    virtual_degree: int | None
    additional_rows: tuple[Any, ...]


def expand_color_conditionals(expression, nc):
    """Collect native Laurent coefficients through Nc-independent ``if``.

    Symbolica owns bottom-up branch mapping and polynomial collection. The
    conditional is linear in its selected branch, so each coefficient retains
    the *same runtime condition*. No mass or kinematic condition is evaluated.
    Nc-dependent conditions and opaque functional Nc dependence are rejected.
    """
    from symbolica import E, S

    condition, yes, no = S(
        "loop_color_if::condition_", "loop_color_if::yes_", "loop_color_if::no_"
    )
    conditional = S("symbolica::if")

    def coefficients(branch):
        result = {}
        for power, value in branch.together().expand().coefficient_list(nc):
            if nc in value.get_all_symbols():
                raise NotImplementedError(
                    "conditional branch has unsupported functional "
                    "or non-Laurent Nc dependence"
                )
            polynomial = power.to_rational_polynomial(vars=[nc])
            degree = polynomial.numerator().degree(
                nc
            ) - polynomial.denominator().degree(nc)
            if power != nc**degree:
                raise NotImplementedError(
                    "conditional branch requires integer Laurent Nc powers"
                )
            result[power] = value
        return result

    def transform(match):
        test = match[condition]
        if nc in test.get_all_symbols():
            raise NotImplementedError(
                "Nc-dependent conditions require a separate projection prescription"
            )
        left, right = coefficients(match[yes]), coefficients(match[no])
        return sum(
            (
                power
                * conditional(test, left.get(power, E("0")), right.get(power, E("0")))
                for power in left.keys() | right.keys()
            ),
            E("0"),
        )

    # Native replacement callbacks log Python exceptions rather than propagating
    # them. Record rejected branches and raise outside the dependency callback.
    failures = []

    def rewrite(match):
        try:
            return transform(match)
        except Exception as error:
            failures.append(error)
            return conditional(match[condition], match[yes], match[no])

    result = expression.replace(
        conditional(condition, yes, no), rewrite, nested=True, rhs_cache_size=0
    )
    if failures:
        raise NotImplementedError(str(failures[0])) from failures[0]
    return result


def project_physical_color(bare, nc, additional_rows, accuracy):
    """Project exact unaveraged rows, then bind Nc=3 and physical averages.

    Additional rows include *both* their model scalar color invariants and
    tensor overlaps. They are returned without an average for native CT alias
    binding. This seam does not infer a model's SU(Nc) continuation.
    """
    from symbolica import E

    accuracy = ColorAccuracy(accuracy)
    space = SUNColorSpace(nc)
    born = space.rectangular(bare.born_color_tensors, bare.born_color_tensors)
    loop = space.rectangular(bare.born_color_tensors, bare.loop_color_tensors)
    born_plan = prepare_color_projection(nc, [[value] for row in born for value in row])
    loop_rows = [[value] for row in loop for value in row]
    rows = [*loop_rows, *additional_rows]
    virtual_plan = prepare_color_projection(
        nc, [[expand_color_conditionals(sum(row, E("0")), nc)] for row in rows]
    )
    projected_born = born_plan.project(accuracy)
    projected_virtual = virtual_plan.project(accuracy)
    width = len(bare.born_colors)
    loops = len(bare.loop_color_tensors)

    def averaged(values):
        return tuple(value.replace(nc, 3) / bare.initial_average for value in values)

    bare.set_color_metrics(
        [averaged(projected_born[i * width : (i + 1) * width]) for i in range(width)],
        [
            averaged(projected_virtual[i * loops : (i + 1) * loops])
            for i in range(width)
        ],
    )
    bare.color_accuracy = accuracy.value
    bare.born_color_degree = born_plan.leading_degree
    bare.virtual_color_degree = virtual_plan.leading_degree
    return PhysicalColorProjection(
        accuracy.value,
        born_plan.leading_degree,
        virtual_plan.leading_degree,
        tuple(value.replace(nc, 3) for value in projected_virtual[len(loop_rows) :]),
    )
