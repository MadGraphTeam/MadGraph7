# SPDX-License-Identifier: 0BSD
"""Full colour at another N_c: the N_c scan of ``test_loop_v2_fermion_lines``.

``python -m tests.unit.loop_nc_scan <compiled model> <name> <carrier> <N>
<precision>`` prints, as one JSON line, the full-colour result of the
projection request ``name`` of ``test_loop_v2_fermion_lines`` at its
``carrier``'s point with N_c = N (a rational), in ``precision`` (``double``
or ``arbitrary:<digits>``, the Python reference path).

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

``python -m tests.unit.loop_nc_scan request <compiled model> <request JSON>
<N> <precision>`` does the same for any request (MQ-NC's oracles): the JSON
names ``process``, ``born`` and ``loop`` orders, optional
``fixed_parameters`` and ``particle_veto``, the ``point``, ``mu_r2`` and
``parameters``; the output adds the Catani prediction of each sector
(``ir_poles``). With ``"ward": true`` it runs the gauge-Ward check of the
request at N_c = N instead (``ward_main``; ``precision`` is the Ward
carrier, "double" or "double-double"); ``"identities"`` replaces some of the
declaration's identities (a control). With ``use_nc(n, catani=True)`` (this
entry) Catani's operator also takes N_c = N: C_A = N, C_F = (N^2-1)/(2N),
beta_0's C_A (``loops.v2.poles``) and the colour-correlated Borns at N (the
Python correlators; the native ones are contracted at N_c = 3). No colour
declaration enters it, so the renormalised virtual's poles at N against it
test the counterterms' UV N_c dependence.
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


def use_nc(n: Fraction, *, catani: bool = False) -> dict[str, int]:
    """Set N_c = ``n`` in this process (module constants and tree weights;
    with ``catani`` the Casimirs and beta_0 of Catani's operator too).

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
    if catani:
        from pyamplicol.loops.v2 import native_cells, poles

        # The native colour correlators (WP3.1 C4 "dipoles" stage) are
        # contracted at N_c = 3; the Python ones read the patched constant.
        native_cells.NATIVE_STAGES = native_cells.NATIVE_STAGES - {"dipoles"}
        nc = Fraction(n)
        poles._N_C = nc
        poles._CASIMIRS = {3: (nc * nc - 1) / (2 * nc), 8: nc}
        poles._DYNKIN = {3: Fraction(1, 2), 8: nc}

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


def _precision(text: str):
    from pyamplicol.loops.v2.assemble import Precision

    if text == "double":
        return Precision.double()
    kind, digits = text.split(":")
    assert kind == "arbitrary", text
    return Precision.arbitrary(int(digits))


def request_main(argv: list[str]) -> None:
    """The ``request`` entry (module docstring)."""

    model, request_text, n_text, precision_text = argv
    n = Fraction(n_text)
    counter = use_nc(n, catani=True) if n != 3 else {"interactions": 0}

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    spec = json.loads(request_text)
    compiled = load_compiled_model(model)
    # Full colour with the model's colour declaration (the shipped one of
    # loop_sm): its graded counterterm and field couplings take N_c = N.
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
    assert declaration is not None, "the model has no colour declaration"
    if spec.get("identities"):
        # A control: identities replaced (e.g. C_F by the number 4/3, which
        # passes the proof at N_c = 3 and is wrong at any other N).
        declaration = replace(
            declaration,
            parameter_identities={
                **declaration.parameter_identities,
                **spec["identities"],
            },
        )
    request = SliceRequest(
        spec["process"],
        spec["born"],
        spec["loop"],
        fixed_parameters=spec.get("fixed_parameters", {}),
        particle_veto=tuple(spec.get("particle_veto", ())),
        colour_declaration=declaration,
        colour_contraction="direct",
    )
    if spec.get("ward"):
        ward_main(compiled, request, spec, precision_text, counter)
        return
    with dag_input.native_flags(dag=False, program=False):
        full = LoopSlice.generate(compiled, request, load_loop_native())
    assert not full.blocks.graded
    result = full.evaluate(
        spec["point"],
        spec["mu_r2"],
        parameters=spec.get("parameters", {}),
        precision=_precision(precision_text),
    )
    poles = {
        repr(sector): [str(single), str(double)]
        for sector, (single, double) in (result.ir_poles or {}).items()
    }
    print(
        json.dumps(
            {
                "rescaled": counter["interactions"],
                "ir_poles": poles,
                **result_json(result),
            }
        )
    )


def ward_main(compiled, request, spec, carrier: str, counter) -> None:
    """The ``request`` entry with ``"ward": true``: the gauge-Ward check
    (``loops.v2.ward.ward_check``: every gluon leg, c = 1 and i) of the
    full-colour request at N_c = N in ``carrier`` ("double" or
    "double-double"), Python inputs (the patched constants and the Python
    tree lanes), the native executor dressing them. Prints each row's
    verdict and ratios."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import dag_input, ward
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    with dag_input.native_flags(dag=False, program=False):
        slice_ = ward.ward_slice(compiled, request, load_loop_native())
    executor = NativeExecutor(slice_, inputs="python", forward=False)
    check = ward.ward_check(
        executor,
        spec["point"],
        spec["mu_r2"],
        carrier,
        parameters=spec.get("parameters", {}),
    )
    rows = [
        {
            "leg": row["leg"],
            "factor": list(row["factor"]),
            "passed": row["passed"],
            "born": row["born"],
            "virtual": row["parts"]["virtual"],
            "estimate": row["estimate"],
        }
        for row in check["rows"]
    ]
    print(
        json.dumps(
            {
                "rescaled": counter["interactions"],
                "passed": check["passed"],
                "rows": rows,
                **result_json(check["physical"]),
            }
        )
    )


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if argv[0] == "request":
        request_main(argv[1:])
        return
    model, name, carrier, n_text, precision_text = argv
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
        compiled, name, "full", carrier
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
