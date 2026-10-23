# SPDX-License-Identifier: 0BSD
"""Native per-point inputs of the v2 executor (WP3.1 I3/I4/I5).

``NativeExecutor(inputs="native")`` exports the slice's tree program (the
interaction program), its Born root closure and its counterterm closure with
the external-field insertions (``loops/v2/native_trees.py``) and forms every
point's block rows, Born components, counterterm and field components in
``rusticol-core`` (``loop_v2::trees``, ``loop_v2::interference``), without
Python per point. Checks against the slice's Python inputs
(``inputs="python"``, the same exported plan):

* block rows and Born components per point in binary64;
* every component of the virtual per sector (bare, counterterms, external
  fields, complete) and |A0|^2 per squared Born order, in binary64 and in
  double-double, on processes with fixed parameters, massive external tops,
  and leading and next-to-leading colour (the Born metric per Born
  component; whole-rule colour continuations of the four-gluon R2);
* the counterterms of the tree recursion's program (production, a second
  native interaction program whose pole-slot currents finite-only batches
  skip) and of the site closure (the oracle route);
* per-point renormalisation scales in one batch, and massive external
  spinors in double-double with a mass that is no binary64 number (the
  executor takes exact masses and rounds them to its carrier).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
DOUBLE_DOUBLE = Precision.double_double()


def _points(source: str) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]


# process, Born orders, accuracy, points
PROCESSES = {
    # QED = 0: without the bound the Born has a QCD 1 QED 2 order (photon, Z
    # and W exchange) that the root closure does not build, and generation
    # refuses it (``test_loop_v2_generate``); before that refusal its
    # interference sector (QCD 6, QED 2) was silently zero here.
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "full",
        "required/uux_ddxg.json",
    ),
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, "full", "legacy/gg_gg.json"),
    "gg_gg_lc": ("g g > g g", {"QCD": 2, "QED": 0}, "lc", "legacy/gg_gg.json"),
    "uux_ddxg_nlc": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "nlc",
        "required/uux_ddxg.json",
    ),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, "full", "required/gg_ttx.json"),
    # The site closure of every counterterm rule (the oracle route) instead of
    # the tree recursion's counterterm program (production, WP2.2).
    "uux_ddxg_sites": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "full",
        "required/uux_ddxg.json",
        {"counterterms": "sites"},
    ),
}


@pytest.fixture(scope="module")
def executors(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built: dict[str, tuple[NativeExecutor, NativeExecutor]] = {}

    def build(name: str) -> tuple[NativeExecutor, NativeExecutor]:
        if name not in built:
            process, born, accuracy, _source, *options = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                accuracy=accuracy,
                fixed_parameters=FIXED,
                colour_declaration=None if accuracy == "full" else declaration,
                **(options[0] if options else {}),
            )
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            native = NativeExecutor(slice_, forward=False, inputs="native")
            python = NativeExecutor(slice_, forward=False, exported=native.exported)
            built[name] = native, python
        return built[name]

    return build


def _deviation(values, reference) -> float:
    """Largest deviation of two complex vectors on the reference's scale."""

    pairs = [(complex(*a), complex(*b)) for a, b in zip(values, reference, strict=True)]
    scale = max((abs(b) for _a, b in pairs), default=0.0) or 1.0
    return max((abs(a - b) for a, b in pairs), default=0.0) / scale


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg", "gg_ttx"])
def test_native_rows_and_born_components_equal_the_python_inputs(executors, name):
    """Block rows and the plan's Born components of every point."""

    native, python = executors(name)
    points, scales = _points(PROCESSES[name][3])
    rows, born, _amplitudes, _squared, _counterterms = native.native_inputs(
        points, scales, PARAMETERS
    )
    _precision, evaluations = python._points(points, scales, PARAMETERS, None, True)
    reference = python._inputs(evaluations, "double")
    for k in range(len(points)):
        assert len(rows[k]) == len(reference["block_rows"][k])
        assert _deviation(rows[k], reference["block_rows"][k]) <= 1e-14
        assert _deviation(born[k], reference["born"][k]) <= 1e-13


def _real(value) -> Decimal:
    """A real binary64 or Decimal component value, exactly."""

    if isinstance(value, complex):
        assert value.imag == 0
        value = value.real
    return Decimal(value)


def _assert_same(native, python, tolerance) -> None:
    """Every component per sector, on the scale of the sector's complete
    virtual, and |A0|^2 per squared Born order."""

    for a, b in zip(native, python, strict=True):
        assert a.components.keys() == b.components.keys()
        for name, sectors in b.components.items():
            for sector, value in sectors.items():
                mine = a.components[name][sector].coefficients
                scale = max(
                    abs(_real(x)) for x in b.components["virtual"][sector].coefficients
                )
                for x, y in zip(mine, value.coefficients, strict=True):
                    deviation = abs(_real(x) - _real(y))
                    assert deviation <= Decimal(tolerance) * scale, (name, sector)
        assert a.born.keys() == b.born.keys()
        for order, value in b.born.items():
            assert abs(float(a.born[order]) - float(value)) <= tolerance * abs(
                float(value)
            )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    ["uux_ddxg", "gg_gg", "gg_ttx", "gg_gg_lc", "uux_ddxg_nlc", "uux_ddxg_sites"],
)
def test_native_virtual_and_born_equal_the_python_inputs(executors, name):
    """Every component of the virtual per sector and |A0|^2 (at the
    request's colour accuracy) in binary64, with the counterterms of the
    tree recursion's program (a second native interaction program) or of
    the site closure."""

    native, python = executors(name)
    program = native.exported.trees.program
    assert (program["counterterm_trees"] is None) == name.endswith("_sites")
    points, scales = _points(PROCESSES[name][3])
    mine = native.evaluate(points, scales, parameters=PARAMETERS)
    theirs = python.evaluate(points, scales, parameters=PARAMETERS)
    _assert_same(mine, theirs, 1e-12)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg", "gg_ttx"])
def test_native_double_double_equals_the_python_inputs(executors, name):
    """Double-double: native rows in DoubleFloat against Decimal inputs."""

    native, python = executors(name)
    points, scales = _points(PROCESSES[name][3])
    points, scales = points[:1], scales[:1]
    mine = native.evaluate(
        points, scales, parameters=PARAMETERS, precision=DOUBLE_DOUBLE
    )
    theirs = python.evaluate(
        points, scales, parameters=PARAMETERS, precision=DOUBLE_DOUBLE
    )
    _assert_same(mine, theirs, 1e-26)


@pytest.mark.loop_physics
def test_native_finite_only_keeps_every_finite_part(executors):
    """Finite-only batches skip the pole-only counterterm couplings and the
    pole outputs: every component's finite part is bitwise that of a batch
    with poles."""

    native, _python = executors("uux_ddxg")
    # The counterterm program flags the currents of its pole slots, which
    # finite-only batches skip.
    steps = native.exported.trees.program["counterterm_trees"]["interactions"]["steps"]
    assert any(step.get("pole_interactions") for step in steps)
    points, scales = _points(PROCESSES["uux_ddxg"][3])
    points, scales = points[:1], scales[:1]
    (full,) = native.evaluate(points, scales, parameters=PARAMETERS)
    (finite,) = native.evaluate(points, scales, parameters=PARAMETERS, poles=False)
    assert finite.components.keys() == full.components.keys()
    for name, sectors in full.components.items():
        for sector, value in sectors.items():
            assert finite.components[name][sector].coefficients == (
                value.coefficients[0],
            ), (name, sector)


@pytest.mark.loop_physics
def test_native_lane_binds_the_external_parameters(executors):
    """A native batch carries momenta, scales and the external parameters
    only; the executor binds the kernel constants, the loop and external
    masses, the Born couplings and the tree program's derived parameters
    itself (the parameter program of ``native_trees``), and the scale
    parameter from each point's mu_R^2. Away from the
    defaults (alpha_s and the top mass, which enters the loop propagators,
    the massive spinors and the couplings' dependence) the native results
    equal the Python inputs, and differ from those at the defaults."""

    native, python = executors("gg_ttx")
    points, scales = _points(PROCESSES["gg_ttx"][3])
    points, scales = points[:1], scales[:1]
    parameters = {"aS": "0.13", "MT": "171.5"}
    evaluations = native._native_points(points, scales, parameters, Precision.double())
    batch = native._inputs(evaluations, "double")
    assert set(batch) == {"momenta", "scales", "parameters"}
    names = native.exported.trees.parameter_names
    assert batch["parameters"][names.index("aS")] == "0.13"
    mine = native.evaluate(points, scales, parameters=parameters)
    theirs = python.evaluate(points, scales, parameters=parameters)
    _assert_same(mine, theirs, 1e-12)
    (default,) = native.evaluate(points, scales, parameters=PARAMETERS)
    assert default.born != mine[0].born


@pytest.mark.loop_physics
def test_native_scales_may_change_per_point(executors):
    """One batch at two scales equals the two single-scale batches."""

    native, _python = executors("uux_ddxg")
    points, _scales = _points(PROCESSES["uux_ddxg"][3])
    point = points[0]
    both = native.evaluate([point, point], ["490000", "1960000"], parameters=PARAMETERS)
    for result, scale in zip(both, ["490000", "1960000"], strict=True):
        (alone,) = native.evaluate([point], [scale], parameters=PARAMETERS)
        for sector, value in alone.components["bare"].items():
            assert result.components["bare"][sector].coefficients == value.coefficients
    assert both[0].components != both[1].components


@pytest.mark.loop_physics
def test_native_massive_sources_carry_the_carrier_precision(executors):
    """MT = 173.1 is no binary64 number: in double-double the native massive
    spinors take the mass in the carrier and equal the Decimal inputs."""

    from fractions import Fraction

    from tools.loop_reference.exact_points import generate

    native, python = executors("gg_ttx")
    mass = Fraction(1731, 10)
    (point,) = generate([0, 0, mass, mass], count=1)
    points = [[[str(x) for x in p] for p in point]]
    parameters = {**PARAMETERS, "MT": "173.1"}
    mine = native.evaluate(
        points, ["1000000"], parameters=parameters, precision=DOUBLE_DOUBLE
    )
    theirs = python.evaluate(
        points, ["1000000"], parameters=parameters, precision=DOUBLE_DOUBLE
    )
    _assert_same(mine, theirs, 1e-26)


# A fresh interpreter creates ``argv[2]`` unrelated Symbolica symbols, then
# exports u u~ > g g g's native tree program and forms one point's native
# inputs; it prints digests of the program, the saved kernel evaluators and
# the binary64 rows and Born.
_HISTORY_PROBE = """
import hashlib, json, sys
from symbolica import S

model, unrelated, fixture = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for k in range(unrelated):
    S(f"history::unrelated{k}")
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.models.loading import load_compiled_model

request = SliceRequest(
    "u u~ > g g g", {"QCD": 3}, {"QCD": 5}, fixed_parameters={"MB": "0"}
)
slice_ = LoopSlice.generate(load_compiled_model(model), request, load_loop_native())
executor = NativeExecutor(slice_, forward=False, inputs="native")
trees = executor.exported.trees
points = json.load(open(fixture))["points"][:1]
rows, born, _amplitudes, squared, counterterms = executor.native_inputs(
    points, ["10000"], {"aS": "0.118", "MT": "173"}
)
digest = lambda data: hashlib.sha256(data).hexdigest()
print(json.dumps({
    "program": digest(trees.json().encode()),
    "kernels": [digest(saved) for saved in trees.kernels],
    "inputs": digest(repr((rows, born, squared, counterterms)).encode()),
}))
"""


@pytest.mark.loop_physics
def test_native_trees_do_not_depend_on_the_process_history(loop_sm_compiled_model):
    """Three fresh interpreters, two of which first create 250 and 300
    unrelated Symbolica symbols, export u u~ > g g g's native tree program:
    the program, every saved kernel evaluator and the binary64 rows and Born
    are identical. Before the interaction kernels were compiled on canonical
    symbols (``assemble._canonical_stage``), 7 of the 15 kernels differed
    after 300 symbols and so did the rows' last bits (measured)."""

    import subprocess
    import sys

    fixture = FIXTURES / "legacy" / "uux_ggg.json"
    runs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _HISTORY_PROBE,
                str(loop_sm_compiled_model),
                str(unrelated),
                str(fixture),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for unrelated in (0, 250, 300)
    ]
    outputs = []
    for run in runs:
        out, err = run.communicate(timeout=900)
        assert run.returncode == 0, err[-4000:]
        outputs.append(json.loads(out.strip().splitlines()[-1]))
    for output in outputs[1:]:
        assert output == outputs[0]


_ROUTE_ROWS_PROBE = """
import hashlib, json, sys
from decimal import Decimal

model, route, process, qcd, fixture = sys.argv[1:6]
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.v2 import assemble
from pyamplicol.loops.v2.assemble import (
    LoopSlice, Precision, SliceRequest, _fraction, _Numbers,
)
from pyamplicol.models.loading import load_compiled_model

# Both routes build every block current (the site route always does), so
# their DAGs and current ids are the same.
assemble.LOOP_BLOCK_REACHABILITY = False
qcd = int(qcd)
request = SliceRequest(
    process, {"QCD": qcd, "QED": 0}, {"QCD": qcd + 2, "QED": 0},
    fixed_parameters={"MB": "0"}, counterterms=route,
)
slice_ = LoopSlice.generate(load_compiled_model(model), request, load_loop_native())
# The block rows are built on first use: after the CT-lowered model of the
# recursion route, before any CT object of the site route.
assert "tree-rows" not in slice_._cache
point = json.load(open(fixture))["points"][0]
precision = Precision.double()
numbers = _Numbers(precision)
with numbers.scope():
    parameters = slice_._point_parameters(
        {"aS": "0.118", "MT": "173"}, _fraction("10000"), precision
    )
    rows = slice_.tree_rows.evaluate(
        [tuple(Decimal(str(x)) for x in p) for p in point], parameters, numbers
    )
print(hashlib.sha256(repr(sorted(rows.items())).encode()).hexdigest())
"""


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "qcd", "fixture"),
    [
        ("g g > g g g", 3, "legacy/uux_ggg.json"),
        ("u u~ > g g g", 3, "legacy/uux_ggg.json"),
    ],
)
def test_lazy_block_rows_do_not_depend_on_the_counterterm_route(
    loop_sm_compiled_model, process, qcd, fixture
):
    """The slice builds its Python block rows (``LoopSlice.tree_rows``) on
    first use, after the recursion route's CT-lowered model has created its
    symbols. Their stages are compiled on canonical symbols
    (``assemble._stage_evaluator``), so the binary64 rows of a point are
    bitwise those of the site route (each route in a fresh interpreter):
    the rounding independence the eager rows protected (RUST-MIGRATION.md
    §4 item 1(e))."""

    import subprocess
    import sys

    runs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _ROUTE_ROWS_PROBE,
                str(loop_sm_compiled_model),
                route,
                process,
                str(qcd),
                str(FIXTURES / fixture),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for route in ("recursion", "sites")
    ]
    digests = []
    for run in runs:
        out, err = run.communicate(timeout=900)
        assert run.returncode == 0, err[-4000:]
        digests.append(out.strip().splitlines()[-1])
    assert digests[0] == digests[1]


@pytest.mark.loop_physics
def test_counterterm_roots_outside_the_request_orders_are_not_closed(
    loop_sm_compiled_model, loop_native
):
    """u u~ > d d~ a at QCD 2, QED 1: the counterterm program holds W
    currents, and a clone root on the u line with a W sits at QED 3, outside
    the request. Its vertex's terms are not built (the closure refuses a W
    vertex not closed under conjugation; generation failed before), and the
    native counterterms equal the Python inputs."""

    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import exact_points

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "u u~ > d d~ a",
        {"QCD": 2, "QED": 1},
        {"QCD": 4, "QED": 1},
        fixed_parameters=FIXED,
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    assert slice_.counterterm_program.closure.terms
    native = NativeExecutor(slice_, forward=False, inputs="native")
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    points = [
        [[exact_points.decimal_string(x) for x in p] for p in point]
        for point in exact_points.generate([0] * 5, count=1)
    ]
    scales = ["1000000"]
    _assert_same(
        native.evaluate(points, scales, parameters=PARAMETERS),
        python.evaluate(points, scales, parameters=PARAMETERS),
        1e-12,
    )
