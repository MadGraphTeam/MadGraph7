# SPDX-License-Identifier: 0BSD
"""Electroweak native point inputs (WP3.2 EWI-N, package ew-native).

The SymJIT plane check of the native export (``native_trees.
_plane_applications``, rusticol-core ``loop_v2::trees::
compile_plane_application_checked``) compares every saved evaluator's
binary64 plane application with its exact evaluator

* at two complex rows off the real axis, where no input-dependent argument
  sits on a branch cut: they certify the translation whatever the inputs'
  declaration order, names and values;
* at the real inputs the program runs at, which certify its real-axis
  evaluation there: the parameter program's row is the request's parameter
  values (``native_trees._request_points``). A program that reads a branch-cut
  or discontinuous function of its inputs and has no real point is refused.

Evidence for the choice (slice-status DECISIONS 2026-10-26 ~21:10 item 6,
user-confirmed 2026-10-27 ~19:20): SymJIT gives a negated real a -0
imaginary part where Symbolica's evaluator gives +0, so at the former real
probes, assigned by input index, a square root of 1 - MW^2/MZ^2 sat on its
cut whenever the probe made MW > MZ and refused a correct translation,
depending on the declaration order (loop_sm with MW a card input; the G_mu
EW counterterms c_UVepemA1EW and UV_GTTLEW). At the physical point the
argument is positive, and an argument exactly on a cut at real masses takes
MadLoop's +0 side in every evaluator (``ct_recursion._RealArguments``); the
real-axis row checks both. A real point on a cut where the two evaluators
take opposite sides is a binary64 against DoubleFloat inconsistency there,
and stays the listed refusal ``plane-mismatch``.

Nightly: g g > t t~ EW and e- e+ > mu- mu+ (G_mu UFO) and u u~ > d d~ with
QED corrections (loop_qcd_qed_sm) export native point inputs, which equal
the Python inputs, and electroweak artifacts generated through the public
API (u u~ > d d~ and u u~ > a g, whose on-shell photon couples with alpha(0))
store them (``point_inputs`` "native-v2") and relocate bitwise. The photons'
native values: ``test_loop_v2_external_photons.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
# The physical W and Z masses (the G_mu card) and the former real probes of
# the first two inputs (0.5 + frac(0.618... (k + 1))).
PHYSICAL = (80.419, 91.188)
FORMER_PROBES = (1.1180339887498949, 0.7360679774997898)
TOLERANCE = 1e-12


def _state(texts, names=("x", "y")) -> bytes:
    from symbolica import E, Expression, S

    return bytes(
        Expression.evaluator_multiple(
            [E(text) for text in texts], [S(name) for name in names], jit_compile=False
        ).save()
    )


def _refusal(native, state, points=None):
    with pytest.raises(native.PlaneMismatchError) as error:
        native.compile_v2_plane(state, 2, False, points)
    return error.value.args


def _plane_values(application: bytes, inputs: int, outputs: int, rows):
    from pyamplicol import _rusticol

    rows = np.ascontiguousarray(rows, dtype=np.complex128)
    flat = _rusticol._evaluate_symjit_plane_application_v2(
        bytes(application),
        inputs,
        outputs,
        rows.view(np.float64).ravel().tolist(),
        len(rows),
    )
    return np.asarray(flat, dtype=np.float64).view(np.complex128).reshape(len(rows), -1)


# ------------------------------------------------------------- plane check


W_COUPLING = ("(1 - x^2/y^2)^(-1/2)", "sqrt(1 - x^2/y^2)", "log(1 - x^2/y^2)")


@pytest.mark.loop_fast
def test_plane_check_certifies_the_translation_in_any_input_order(loop_native):
    """The W coupling's (1 - MW^2/MZ^2)^(-1/2), its square root and a
    logarithm, with the request's point: accepted with the inputs in either
    declaration order (the former real probes refused the order (MW, MZ),
    whose first probe exceeds the second, and accepted the other), and the
    plane application equals the exact evaluator at the point."""

    from symbolica import Evaluator

    swapped = [
        t.replace("x", "#").replace("y", "x").replace("#", "y") for t in W_COUPLING
    ]
    for texts, point in ((W_COUPLING, PHYSICAL), (swapped, PHYSICAL[::-1])):
        state = _state(texts)
        plane, inputs, outputs = loop_native.compile_v2_plane(
            state, 2, False, [list(point)]
        )
        assert (inputs, outputs) == (2, 3)
        rows = [[complex(x) for x in point]]
        expected = np.asarray(Evaluator.load(state).evaluate_complex(rows))
        got = _plane_values(plane, inputs, outputs, rows)
        assert np.allclose(got, expected, rtol=1e-14, atol=0), (got, expected)
        assert np.all(got.imag == 0) and np.all(np.isfinite(got))
    # The former probes as the real point: a correct translation refused on
    # the cut (the imaginary part's sign: SymJIT's -0 against Symbolica's +0).
    message, _output, row, imaginary, real_axis = _refusal(
        loop_native, _state(W_COUPLING), [list(FORMER_PROBES)]
    )
    assert (row, imaginary, real_axis) == (0, True, True), message
    assert "imaginary part, real-axis probe row 0" in message, message


@pytest.mark.loop_fast
def test_plane_check_refuses_a_real_point_on_a_cut_where_the_sides_differ(
    loop_native,
):
    """sqrt(-x) at the real input x = 2: the plane application gives
    -i sqrt(2) and the exact evaluator +i sqrt(2) (a negated real's zero
    imaginary part), so binary64 and DoubleFloat would disagree there: the
    listed refusal, naming the output, part and real-axis row. Off the cut
    (x = -2) the same program is accepted."""

    from pyamplicol.loops.v2 import native_trees

    state = _state(["x + y", "sqrt(-x)"])
    message, output, row, imaginary, real_axis = _refusal(
        loop_native, state, [[-2.0, 1.0], [2.0, 1.0]]
    )
    assert (output, row, imaginary, real_axis) == (1, 1, True, True), message
    assert (
        "(imaginary part, real-axis probe row 1: -1.4142135623730951e0 against "
        "1.4142135623730951e0)" in message
    ), message
    assert loop_native.compile_v2_plane(state, 2, False, [[-2.0, 1.0]])
    with pytest.raises(native_trees.NativeExportUnsupported) as refused:
        native_trees._plane_applications((state,), None, {0: [[2.0, 1.0]]})
    assert refused.value.feature == "plane-mismatch"
    assert str(refused.value).startswith(
        "plane-mismatch: output 1 of saved evaluator 0: loop v2 trees: the SymJIT "
        "plane application disagrees with its exact evaluator at output 1 "
        "(imaginary part, real-axis probe row 0"
    ), str(refused.value)


@pytest.mark.loop_fast
def test_plane_check_needs_a_real_point_for_branch_cuts_of_its_inputs(
    loop_native,
):
    """Without a real point (a tree kernel: complex currents and couplings),
    a program reading a branch-cut or discontinuous function of its inputs
    is refused, naming the first such output; a branch cut of constants
    (sqrt(2)), polynomials and the continuous functions (exp, sin, cos, abs,
    conj) need none."""

    from pyamplicol.loops.v2 import native_trees

    for texts, output in (
        (["x*y", "sqrt(1 + x^2)"], 1),
        (["log(x)", "x"], 0),
        (["x", "y", "x^(1/3)"], 2),
    ):
        message, found, row, imaginary, real_axis = _refusal(loop_native, _state(texts))
        assert (found, row, imaginary) == (output, None, None), message
        assert real_axis, message
        assert f"output {output} reads a branch-cut or discontinuous function" in (
            message
        ), message
        # With a real point it is accepted.
        assert loop_native.compile_v2_plane(_state(texts), 2, False, [[0.75, 1.5]])
    for texts in (
        ["3*sqrt(2)*x + y^2", "x/y - 1"],
        ["exp(x) + sin(y)", "cos(x*y)", "abs(x - y)", "conj(x)*y"],
    ):
        assert loop_native.compile_v2_plane(_state(texts), 2, False)
    with pytest.raises(native_trees.NativeExportUnsupported) as refused:
        native_trees._plane_applications((_state(["x*y", "sqrt(1 + x^2)"]),))
    assert "output 1 of saved evaluator 0: loop v2 trees: output 1 reads" in str(
        refused.value
    )


@pytest.mark.loop_fast
def test_the_parameter_programs_real_point_is_the_requests_values():
    """The parameter program's real point: each external parameter's value
    fixed at generation, else its default (the batch's values without a
    caller's, ``batch_parameters``), in binary64."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.native_trees import _request_points

    slice_ = SimpleNamespace(fixed_parameters={"WZ": "0", "MB": "0"})
    record = {
        "names": ["MZ", "WZ", "MB", "MU_R"],
        "defaults": ["22797/250", "2.4952", "4.7", "91.188"],
    }
    assert _request_points(slice_, record) == [(91.188, 0.0, 0.0, 91.188)]


@pytest.mark.loop_fast
def test_counterterm_couplings_take_madloops_side_of_real_cuts(loop_native):
    """The native lane's counterterm and field couplings project every
    branch-cut argument that is manifestly real and not manifestly
    non-negative on the real axis (``ct_recursion._RealArguments`` with
    ``signed``; ``native_trees._parameter_program``): the plane application
    and the exact evaluator then take MadLoop's +0 side at a real point on the
    cut (the G_mu UFO's (MZ^2 (MZ^2 - 4 MW^2))^(1/2) at the card's masses),
    where without it they disagree; manifestly non-negative arguments
    (log(mu^2/m^2)) and arguments that are not manifestly real keep their
    form, and every value is unchanged."""

    from types import SimpleNamespace

    from symbolica import E, Expression, S

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.scalar_frontend import _producer_predicate
    from pyamplicol.loops.v2.ct_recursion import _RealArguments

    predicates = {
        name: symbols.qualified(
            f"pyamplicol_scalar_v1::{name}", eval={"complex": _producer_predicate(name)}
        )
        for name in ("equal", "real_part", "imaginary_part")
    }
    library = SimpleNamespace(
        predicates=predicates,
        choose=symbols.qualified("symbolica::if"),
        zero=E("0"),
        helpers={},
    )
    mw, mz, mu = S("ewn_mw", "ewn_mz", "ewn_mu")
    side = _RealArguments({mw, mz, mu}, library, signed=True)
    cut = (mz**4 * (1 - 4 * mw**2 / mz**2)) ** E("1/2")
    projected = side(cut)
    assert projected == predicates["real_part"](mz**4 * (1 - 4 * mw**2 / mz**2)) ** E(
        "1/2"
    )
    for kept in (
        E("log(ewn_mu^2/ewn_mz^2)"),
        E("sqrt(ewn_mw)"),
        E("sqrt(ewn_z - ewn_mw)"),
    ):
        kept = kept.replace(E("ewn_mu"), mu)
        assert side(kept) == kept
    point = [[80.419, 91.188]]

    def state(expression) -> bytes:
        return bytes(
            Expression.evaluator_multiple(
                [expression], [mw, mz], jit_compile=False
            ).save()
        )

    with pytest.raises(loop_native.PlaneMismatchError, match="real-axis probe row 0"):
        loop_native.compile_v2_plane(state(cut), 2, False, point)
    plane, inputs, outputs = loop_native.compile_v2_plane(
        state(projected), 2, False, point
    )
    rows = [[complex(x) for x in point[0]]]
    exact = Expression.evaluator_multiple([cut], [mw, mz], jit_compile=False)
    expected = np.asarray(exact.evaluate_complex(rows))
    got = _plane_values(plane, inputs, outputs, rows)
    assert expected[0, 0].imag > 0 and expected[0, 0].real == 0
    assert np.allclose(got, expected, rtol=1e-14, atol=0), (got, expected)


# ------------------------------------------------- electroweak processes


ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
G_MU_FIXED = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0", **ZERO_WIDTHS}


@pytest.fixture(scope="module")
def ew_slices(loop_gmu_ufo, loop_qcd_qed_sm_ufo, loop_native, loop_reference):
    """``slice(name)``: g g > t t~ EW (QCD and QED corrections, the request
    of ``required/gg_ttx_ew.json``) and e- e+ > mu- mu+ (full EW) on the G_mu
    UFO, u u~ > d d~ with QED corrections on loop_qcd_qed_sm, each generated
    once ('t Hooft-Feynman gauge; about 3, 6 and 1 min), with a point and
    mu_R^2."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source
    from tools import loop_v2_generate

    massless = json.loads((ROOT / "tools/loop_reference/uudd_qcd.json").read_text())
    massless = [[str(x) for x in leg] for leg in massless["points"][0]]
    built: dict[str, tuple] = {}

    def build(name: str):
        if name in built:
            return built[name]
        if name == "gg_ttx_ew":
            reference = loop_reference("required/gg_ttx_ew")
            compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
            request, _ignored = loop_v2_generate._fixture_request(
                reference, compiled, argparse.Namespace()
            )
            point = reference["points"][0]
            scale = str(int(reference["mu_r"][0]) ** 2)
        elif name == "eemm":
            compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
            request = SliceRequest(
                "e- e+ > m- m+",
                {"QED": 2, "QCD": 0},
                {"QED": 4, "QCD": 0},
                corrections=("QED",),
                fixed_parameters=G_MU_FIXED,
                compatibility="gmu_particle_anti",
                scale_parameter="MU_R",
            )
            point, scale = massless, "1000000"
        else:
            compiled = compile_model_source(
                loop_qcd_qed_sm_ufo, require_supported=False
            )
            request = SliceRequest(
                "u u~ > d d~",
                {"QCD": 2, "QED": 0},
                {"QCD": 2, "QED": 2},
                corrections=("QED",),
                fixed_parameters=ZERO_WIDTHS,
                compatibility="gmu_particle_anti",
                scale_parameter="MU_R",
            )
            point, scale = massless, "1000000"
        built[name] = LoopSlice.generate(compiled, request, loop_native), point, scale
        return built[name]

    return build


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_ttx_ew", "eemm", "uudd_qed"])
def test_electroweak_native_inputs_equal_the_python_inputs(ew_slices, name):
    """The native point inputs (tree program in 't Hooft-Feynman gauge with
    its massive vectors, Goldstones and ghosts' trees, Born and counterterm
    root closures, the Weyl lines of the colour-singlet lepton lines, the
    parameter program with the EW counterterm coefficients) equal the Python
    inputs on the same exported plan: Born and every component per sector
    within 1e-12 of the sector's virtual, in binary64 (the inputs-equality
    harness of ``test_loop_v2_ew_lines.test_native_inputs_equal_the_python_
    inputs``; the lanes are not bitwise equal, DESIGN §12.1 item 4)."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    slice_, point, scale = ew_slices(name)
    native = NativeExecutor(slice_, forward=False, inputs="native")
    trees = native.exported.trees
    assert len(trees.planes) == len(trees.kernels) + 1
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    (mine,) = native.evaluate([point], [scale])
    (theirs,) = python.evaluate([point], [scale])
    assert mine.born.keys() == theirs.born.keys() and theirs.born
    for order, value in theirs.born.items():
        assert abs(float(mine.born[order]) - float(value)) <= TOLERANCE * abs(
            float(value)
        ), order
    assert mine.components.keys() == theirs.components.keys()
    worst = 0.0
    for component, sectors in theirs.components.items():
        for sector, laurent in sectors.items():
            virtual = theirs.components["virtual"][sector].coefficients
            size = max(abs(float(x)) for x in virtual)
            for x, y in zip(
                mine.components[component][sector].coefficients,
                laurent.coefficients,
                strict=True,
            ):
                deviation = abs(float(x) - float(y)) / size
                worst = max(worst, deviation)
                assert deviation <= TOLERANCE, (component, sector, x, y)
    print(name, f"native - python: {worst:.1e} of the virtual")


_EW_CARD = """
action = "generate"
[model]
source = {model}{restriction}
[process]
entries = [{{expression = "{expression}", name = "{name}"}}]
loop_coupling_orders = ["QED"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.model]
loop_coupling_orders = ["QED"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO on-shell electroweak"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_qcd_qed_sm lhv=1 supplied QED rules"
[loop.preparation]
fixed_parameters = {{{fixed}}}
compatibility = "gmu_particle_anti"
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _fixed_text(fixed) -> str:
    return ", ".join(f'{name} = "{value}"' for name, value in fixed.items())


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("name", "expression", "model", "restriction", "fixed"),
    [
        ("uudd", "u u~ > d d~", "loop_qcd_qed_sm_ufo", None, ZERO_WIDTHS),
        ("uuxag", "u u~ > a g", "loop_gmu_ufo", "a0", G_MU_FIXED),
    ],
)
def test_electroweak_artifact_stores_native_inputs_and_relocates_bitwise(
    request, loop_native, tmp_path, name, expression, model, restriction, fixed
):
    """QED corrections through the public API: u u~ > d d~ on
    loop_qcd_qed_sm (Z, W, H, Goldstones and ghosts in the loops) and
    u u~ > a g on the G_mu UFO under restriction a0 (its on-shell photon
    with alpha(0), 1/alpha(0) a runtime parameter). The stored process
    declares native point inputs ("native-v2", never a regeneration at load),
    and the artifact-portability check passes: no host strings, storage-v4
    plane applications only, and bitwise-identical binary64 results at a
    relocated copy evaluated without a licence."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.generation.loop_artifact import LOOP_V2_NATIVE_POINT_INPUTS
    from tools import loop_artifact_portability as portability

    source = request.getfixturevalue(model)
    output = tmp_path / "artifact"
    card = tmp_path / "card.toml"
    card.write_text(
        _EW_CARD.format(
            model=json.dumps(str(source)),
            restriction=""
            if restriction is None
            else f'\nrestriction = "{restriction}"',
            expression=expression,
            name=name,
            fixed=_fixed_text(fixed),
            output=json.dumps(str(output)),
        )
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(expression, name=name),)), output
    )
    declared = [
        json.loads(path.read_text())["point_inputs"]
        for path in output.rglob("execution.json")
    ]
    assert declared == [LOOP_V2_NATIVE_POINT_INPUTS], declared
    verdict = portability.check(
        output, name, tmp_path / "check", count=2, licence=False
    )
    scan = verdict["scan"]
    assert scan["clean"], scan["findings"]
    assert set(scan["member_kinds"]) == {"1", "2", "11"}, scan["member_kinds"]
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]
