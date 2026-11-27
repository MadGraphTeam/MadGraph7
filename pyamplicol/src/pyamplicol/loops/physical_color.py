# SPDX-License-Identifier: 0BSD
"""Cold-path fixed-Nf color projection for bounded physical controls.

The Born observable has its own leading degree. Bare, UV, R2 and external-field
virtual rows share one leading degree across all Laurent sectors, before model
inputs or helicities are evaluated. Distinct currents remain independent rows
unless the caller has an exact identity permitting their combination.
"""

from __future__ import annotations

from .._internal.physics.symbols import symbols
from .color_projection import prepare_color_projection


def expand_color_conditionals(expression, nc):
    """Collect native Laurent coefficients through Nc-independent ``if``.

    Symbolica owns bottom-up branch mapping and polynomial collection. The
    conditional is linear in its selected branch, so each coefficient retains
    the *same runtime condition*. No mass or kinematic condition is evaluated.
    Nc-dependent conditions and opaque functional Nc dependence are rejected.
    """
    from symbolica import E

    condition, yes, no = symbols.qualified(
        "loop_color_if::condition_", "loop_color_if::yes_", "loop_color_if::no_"
    )
    conditional = symbols.qualified("symbolica::if")

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


def split_color_grades(expressions, nc):
    """Separate exact scalar aliases into Nc powers without merging currents.

    Each returned ``(degree, coefficients)`` retains the original alias order,
    including zero entries. The coefficients contain no Nc. A caller evaluates
    a separate current for each grade and projects ``Nc**degree * overlap``;
    projecting the two factors independently would lose the common structural
    degree. This scalar split does not prove which Lorentz terms an alias feeds.
    """
    from symbolica import E

    expressions = tuple(expressions)
    normalized = tuple(
        expand_color_conditionals(expression, nc).together().expand()
        for expression in expressions
    )
    rows = prepare_color_projection(nc, [[value] for value in normalized]).rows
    grades = sorted({degree for row in rows for degree, _ in row})
    coefficients = tuple(dict(row) for row in rows)
    result = tuple(
        (degree, tuple(row.get(degree, E("0")) for row in coefficients))
        for degree in grades
    )
    for column, expected in enumerate(normalized):
        reconstructed = sum(
            (nc**degree * values[column] for degree, values in result), E("0")
        )
        if (reconstructed - expected).together().expand() != 0:
            raise ValueError("scalar color-grade reconstruction is incomplete")
    return result
