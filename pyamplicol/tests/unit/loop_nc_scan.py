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
(``ir_poles``). With ``"points"`` (a list of ``[point, mu_r2]``) instead of
``point`` and ``mu_r2`` one generation serves every point, and the output's
``results`` hold one such record per point. At N != 3 the trees must pass
through the patched colour fold (``use_nc``), and trees holding a
U(1)-subtraction auxiliary (an internal gluon on quark lines) must have its
weight rescaled, or the entry fails. With ``"ward": true`` it runs the
gauge-Ward check of the request at N_c = N instead (``ward_main``;
``precision`` is the Ward carrier, "double" or "double-double");
``"identities"`` replaces some of the declaration's identities (a control).
With ``use_nc(n, catani=True)`` (this entry) Catani's operator also takes
N_c = N: C_A = N, C_F = (N^2-1)/(2N), beta_0's C_A (``loops.v2.poles``) and
the colour-correlated Borns at N (the Python correlators; the native ones
are contracted at N_c = 3). No colour declaration enters it, so the
renormalised virtual's poles at N against it test the counterterms' UV N_c
dependence. At N_c = 1, where C_F vanishes, and with ``"catani": false`` the
operator is not patched and ``ir_poles`` is null.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def exact_point(masses) -> list[list[str]]:
    """The first exact point (``tools/loop_reference/exact_points``) of the
    external ``masses``, as decimal strings."""

    from tools.loop_reference import exact_points

    point = exact_points.generate(list(masses), count=1)[0]
    return [[exact_points.decimal_string(x) for x in p] for p in point]


def request_spec(
    process, born, point, *, fixed=None, veto=(), mu_r2="1000000", **extra
) -> dict:
    """The request JSON of the ``request`` entry: QCD corrections of the
    Born orders ``born`` at ``point`` (``None`` with ``points`` in
    ``extra``), alpha_s = 0.118."""

    spec = {
        "process": process,
        "born": born,
        "loop": {**born, "QCD": born["QCD"] + 2},
        "fixed_parameters": dict(fixed or {}),
        "particle_veto": list(veto),
        "point": point,
        "mu_r2": mu_r2,
        "parameters": {"aS": "0.118"},
        **extra,
    }
    if point is None:
        del spec["point"], spec["mu_r2"]
    return spec


def scan(model, spec, values, precision, cache) -> dict[str, dict]:
    """The ``request`` entry of ``spec`` at each N of ``values`` (one process
    each, concurrently, each with its own ``PYAMPLICOL_CACHE_DIR`` under
    ``cache``): N -> its JSON."""

    def run(n):
        environment = dict(
            os.environ,
            PYAMPLICOL_CACHE_DIR=str(cache / f"nc-{n.replace('/', '_')}"),
        )
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.unit.loop_nc_scan",
                "request",
                str(model),
                json.dumps(spec),
                n,
                precision,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert process.returncode == 0, (n, process.stderr[-4000:])
        return json.loads(process.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(len(values)) as pool:
        return dict(zip(values, pool.map(run, values), strict=True))


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

    Returns the counters of the patched fold's calls (``"dags"``), the
    U(1)-subtraction auxiliary currents it saw (``"auxiliaries"``) and the
    tree interactions it rescaled (``"interactions"``).
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

    counter = {"dags": 0, "auxiliaries": 0, "interactions": 0}
    fold = dag_compiler.apply_colour_connection_signs

    def rescaled(dag, model):
        auxiliaries = {
            current.id
            for current in dag.currents
            if current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        }
        counter["dags"] += 1
        counter["auxiliaries"] += len(auxiliaries)
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


def slice_request(
    spec: dict, declaration, accuracy: str = "full", contraction: str = "direct"
):
    """The ``SliceRequest`` of a request JSON (``request`` entry) at
    ``accuracy`` with the colour declaration ``declaration`` and the seed
    contraction policy ``contraction`` (direct by default: the scan's Python
    lanes). Optional ``corrections`` (correction orders)."""

    from pyamplicol.loops.v2.assemble import SliceRequest

    corrections = spec.get("corrections")
    return SliceRequest(
        spec["process"],
        spec["born"],
        spec["loop"],
        corrections=None if corrections is None else tuple(corrections),
        fixed_parameters=spec.get("fixed_parameters", {}),
        particle_veto=tuple(spec.get("particle_veto", ())),
        accuracy=accuracy,
        colour_declaration=declaration,
        colour_contraction=contraction,
    )


def request_main(argv: list[str]) -> None:
    """The ``request`` entry (module docstring)."""

    model, request_text, n_text, precision_text = argv
    n = Fraction(n_text)
    spec = json.loads(request_text)
    # Catani's operator divides by C_F, which vanishes at N_c = 1: there
    # (and with ``"catani": false``) the output has no prediction.
    catani = n * n != 1 and spec.get("catani", True)
    counter = use_nc(n, catani=catani) if n != 3 else {"interactions": 0}

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

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
    request = slice_request(spec, declaration)
    if spec.get("ward"):
        ward_main(compiled, request, spec, precision_text, counter)
        return
    with dag_input.native_flags(dag=False, program=False):
        full = LoopSlice.generate(compiled, request, load_loop_native())
    assert not full.blocks.graded
    if n != 3:
        # The tree patch applied: the trees went through the patched fold,
        # and every U(1)-subtraction auxiliary they hold (an internal gluon
        # on quark lines) has its weight 1/N.
        assert counter["dags"], "the trees bypassed the patched colour fold"
        if counter["auxiliaries"]:
            assert counter["interactions"], "no U(1)-subtraction weight rescaled"

    precision = _precision(precision_text)
    parameters = spec.get("parameters", {})
    pairs = spec["points"] if "points" in spec else [(spec["point"], spec["mu_r2"])]
    if spec.get("dressing", "python") == "native":
        results = _native_dressing(full, pairs, parameters, precision)
    else:
        results = []
        for point, mu_r2 in pairs:
            result = full.evaluate(
                point, mu_r2, parameters=parameters, precision=precision
            )
            results.append((result, result.ir_poles))
    records = [
        {
            "ir_poles": {
                repr(sector): [str(single), str(double)]
                for sector, (single, double) in (poles or {}).items()
            }
            if catani or n == 3
            else None,
            **result_json(result),
        }
        for result, poles in results
    ]
    out = {"results": records} if "points" in spec else records[0]
    print(json.dumps({"rescaled": counter["interactions"], **out}))


def _native_dressing(full, pairs, parameters, precision) -> list:
    """(result, Catani poles) per (point, mu_r2) of ``pairs``: the Python
    point inputs of ``full`` (trees, Born, counterterm and field scalars at
    the patched N_c) dressed and reduced by the native executor in
    ``precision`` (``"dressing": "native"``); Catani's operator from the same
    Python point inputs (the native executor forms none)."""

    from pyamplicol.loops.v2.assemble import _Numbers
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    executor = NativeExecutor(full, inputs="python", forward=False)
    points = [point for point, _scale in pairs]
    scales = [scale for _point, scale in pairs]
    results = executor.evaluate(
        points, scales, parameters=parameters, precision=precision
    )
    _carrier, evaluations = executor._cached_points(
        points, scales, parameters, precision
    )
    poles = []
    for evaluation in evaluations:
        with _Numbers(precision).scope():
            poles.append(evaluation._ir_poles())
    return list(zip(results, poles, strict=True))


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
