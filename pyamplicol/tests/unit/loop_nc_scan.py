# SPDX-License-Identifier: 0BSD
"""Full colour at another N_c: the N_c scan of ``test_loop_v2_fermion_lines``.

``python -m tests.unit.loop_nc_scan <compiled model> <name> <N> <precision>``
prints, as one JSON line, the full-colour result of the projection request
``name`` of ``test_loop_v2_fermion_lines`` at its first point with N_c = N
(a rational), in ``precision`` (``double`` or ``arbitrary:<digits>``, the
Python reference path).

Full colour never splits a block by N_c grade (``generation.dag_grades`` acts
on leading- and next-to-leading-colour requests only), so this value does not
use the grade bookkeeping that LC and NLC rest on. N_c = N replaces N_c = 3
everywhere the engine substitutes it: every module constant ``NC`` (seed
cells, the Born amplitude's grades, the counterterm and field cells and
couplings of the colour declaration, literal colour overlaps, the Born
metric), and in the trees the 1/3 of every U(1)-subtraction auxiliary becomes
1/N (the colour weight of each interaction producing an auxiliary times 3/N
before the colour-connection fold, as the grade scan of the test file doubles
it; Python DAG lanes). Powers of two keep 1/N exact in binary64. The external
colour average is the request's (N_c = 3) at every N. One N per process: the
patched constants stay out of the test session, and the caller gives each
process its own ``PYAMPLICOL_CACHE_DIR``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from fractions import Fraction


class _Rational(Fraction):
    """A non-integer N_c written in parentheses: the engine spells the
    U(1)-subtraction weight ``f"1/{NC}"`` in exact expressions."""

    def __str__(self) -> str:
        return f"({self.numerator}/{self.denominator})"

    def __format__(self, spec: str) -> str:
        return str(self) if not spec else super().__format__(spec)


def use_nc(n: Fraction) -> dict[str, int]:
    """Set N_c = ``n`` in this process (module constants and tree weights).

    Returns the counter of rescaled tree interactions (``"interactions"``).
    """

    import importlib

    from pyamplicol.color import contraction_types
    from pyamplicol.generation import dag_compiler
    from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND

    # The engine modules that read N_c, imported before the constant changes
    # so that each binds its own copy; modules imported later bind the new
    # value from ``contraction_types``.
    for name in ("pyamplicol.loops.v2.assemble", "pyamplicol.loops.v2.native_trees"):
        importlib.import_module(name)
    value = int(n) if n.denominator == 1 else _Rational(n)
    patched = [
        module
        for name, module in sorted(sys.modules.items())
        if name.startswith("pyamplicol.")
        and getattr(module, "NC", None) is contraction_types.NC
    ]
    assert contraction_types in patched
    for module in patched:
        module.NC = value
    from pyamplicol.loops.v2 import colour

    colour._nc_power.cache_clear()

    counter = {"interactions": 0}
    fold = dag_compiler.apply_colour_connection_signs

    def rescaled(dag, model):
        auxiliaries = {
            current.id
            for current in dag.currents
            if current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        }
        interactions = []
        for interaction in dag.interactions:
            if interaction.result_id in auxiliaries:
                counter["interactions"] += 1
                # 3 x (1/3 in binary64) rounds to 1 exactly.
                interaction = replace(
                    interaction,
                    color_weight=tuple(
                        (x * 3.0) / float(n) for x in interaction.color_weight
                    ),
                )
            interactions.append(interaction)
        return fold(replace(dag, interactions=tuple(interactions)), model)

    dag_compiler.apply_colour_connection_signs = rescaled
    return counter


def result_json(result) -> dict:
    """A ``PointResult``'s Born and components as decimal strings."""

    return {
        "born": {repr(order): str(value) for order, value in result.born.items()},
        "components": {
            component: {
                repr(sector): [str(c) for c in laurent.coefficients]
                for sector, laurent in sectors.items()
            }
            for component, sectors in result.components.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    model, name, n_text, precision_text = argv or sys.argv[1:]
    n = Fraction(n_text)
    counter = use_nc(n) if n != 3 else {"interactions": 0}

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.models.loading import load_compiled_model

    from .test_loop_v2_fermion_lines import _projection_request

    if precision_text == "double":
        precision = Precision.double()
    else:
        kind, digits = precision_text.split(":")
        assert kind == "arbitrary", precision_text
        precision = Precision.arbitrary(int(digits))
    compiled = load_compiled_model(model)
    request, point, scale, parameters = _projection_request(
        compiled, name, "full", "double"
    )
    with dag_input.native_flags(dag=False, program=False):
        full = LoopSlice.generate(compiled, request, load_loop_native())
    assert not full.blocks.graded
    if n != 3:
        assert counter["interactions"], "no U(1)-subtraction weight was rescaled"
    result = full.evaluate(point, scale, parameters=parameters, precision=precision)
    print(json.dumps({"rescaled": counter["interactions"], **result_json(result)}))


if __name__ == "__main__":
    main()
