# SPDX-License-Identifier: 0BSD
"""The certified symmetric-group FFT of the seed contraction in the native
executor (design §7.3; M-QCD package MQ-FFT, slice-status DECISIONS
2026-10-28 ~15:35/~15:40).

``colour.seed_fourier`` detects the orbits of S_m on the external adjoint
labels and certifies every Born/loop orbit pair structurally on the exact
seed cells; ``plan_export.fourier_record`` maps them into the plan, and the
native loader (``loop_v2::fourier::SeedFourier``) reads the kernels from the
plan's own exact weights and certifies every pair again on them before the
covered weights leave the direct contraction. So the FFT and the direct
contraction are equal before rounding, and the checks here are:

* FFT = direct to rounding, binary64 and double-double, on the Born, every
  component and the virtual, for pure-gluon processes (S_3 ... S_6) and
  quark-gluon processes with orbits, at full colour and NLC; the native FFT
  also equals the Python stage's FFT (``SeedFourier.seeds``, the oracle);
* processes without a certified pair fall back to direct, reported, and the
  forced FFT refuses them at generation; the forced FFT runs in every
  floating carrier (the exact carrier contracts directly);
* the public route (``color.contraction = "symmetric-group-fft"`` with
  ``loop.engine = "v2"``): the artifact stores the record, the runtime and
  the SDKs' Rust runtime run and report the FFT, and the values equal the
  direct artifact's to rounding.
"""

from __future__ import annotations

import json

import pytest

from pyamplicol.loops.v2.assemble import COMPONENTS, LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.colour import SYMMETRIC_GROUP_FFT
from pyamplicol.loops.v2.plan_export import NativeExecutor
from tools.loop_reference import legacy_capture

PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
# FFT against direct: the seeds differ by the FFT's rounding only (a few
# units of the carrier's roundoff times the group's depth), so every
# coefficient agrees to a small multiple of the roundoff of the component's
# largest coefficient.
TOLERANCE = {"double": 1e-13, "double-double": 1e-28}
CARRIERS = {"double": Precision.double(), "double-double": Precision.double_double()}


def _points(masses, count):
    """Exact on-shell points as terminating decimal strings."""

    from tools.loop_reference.exact_points import decimal_string, generate

    return [
        [[decimal_string(x) for x in p] for p in point]
        for point in generate(masses, count=count)
    ]


def _scales(points):
    """One mu_R^2 per point (GeV-scale points, ``exact_points``)."""

    return ["1000000"] * len(points)


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


def _slice(compiled, native, process, born, loop, **options) -> LoopSlice:
    return LoopSlice.generate(
        compiled,
        SliceRequest(process, born, loop, fixed_parameters=FIXED, **options),
        native,
    )


def _pair(slice_: LoopSlice) -> tuple[NativeExecutor, NativeExecutor]:
    """The FFT and the direct native executor of one exported plan."""

    fft = NativeExecutor(slice_, forward=False, inputs="native")
    direct = NativeExecutor(
        slice_,
        forward=False,
        inputs="native",
        exported=fft.exported,
        plan_json=fft._plan_json,
        contraction="direct",
    )
    return fft, direct


def _coefficients(result):
    """Born per order and every component's Laurent coefficients per sector."""

    out = {("born", order): (value,) for order, value in result.born.items()}
    for component in COMPONENTS:
        for sector, laurent in result.components[component].items():
            out[component, sector] = tuple(laurent.coefficients)
    return out


def _assert_equal_to_rounding(mine, theirs, tolerance, label):
    a, b = _coefficients(mine), _coefficients(theirs)
    assert a.keys() == b.keys(), label
    for key, values in b.items():
        scale = max(abs(complex(x)) for x in values)
        for x, y in zip(a[key], values, strict=True):
            error = abs(complex(x) - complex(y))
            assert error <= tolerance * scale, (label, key, error / (scale or 1))


def _assert_fft_equals_direct(slice_, points, scales, *, dd_points=1):
    fft, direct = _pair(slice_)
    stats = fft.native("double").fourier
    assert stats is not None and stats["pairs"] >= 1 and stats["covered"] >= 1
    for carrier, precision in CARRIERS.items():
        count = len(points) if carrier == "double" else dd_points
        mine = fft.evaluate(
            points[:count], scales[:count], parameters=PARAMETERS, precision=precision
        )
        theirs = direct.evaluate(
            points[:count], scales[:count], parameters=PARAMETERS, precision=precision
        )
        assert fft.last_report["counts"].get("symmetric-group-fft seed jobs")
        assert "symmetric-group-fft seed jobs" not in direct.last_report["counts"]
        for k, (a, b) in enumerate(zip(mine, theirs, strict=True)):
            assert a.colour_contraction == SYMMETRIC_GROUP_FFT
            assert b.colour_contraction == "direct"
            _assert_equal_to_rounding(
                a, b, TOLERANCE[carrier], (slice_.request.process, carrier, k)
            )
    return stats


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "born", "loop", "masses", "group"),
    [
        ("g g > g g", {"QCD": 2}, {"QCD": 4}, [0] * 4, 6),
        ("u u~ > g g g", {"QCD": 3}, {"QCD": 5}, [0] * 5, 6),
    ],
)
def test_native_seed_fft_equals_direct(
    compiled, loop_native, process, born, loop, masses, group
):
    """FFT = direct to rounding at full colour, binary64 and double-double,
    on a pure-gluon process and a quark-gluon one (19 loop orbits of
    adjoint rings, closed traces and quark rings), with the native inputs."""

    slice_ = _slice(compiled, loop_native, process, born, loop)
    assert slice_.fourier is not None
    points = _points(masses, 2)
    stats = _assert_fft_equals_direct(slice_, points, _scales(points))
    assert stats["group_order"] == group


@pytest.mark.loop_physics
def test_native_seed_fft_at_nlc_equals_direct(compiled, loop_native):
    """NLC: the kernels are the cells projected per sector degree P and Born
    grade, as the direct weights are; FFT = direct to rounding."""

    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    slice_ = _slice(
        compiled,
        loop_native,
        "g g > g g",
        {"QCD": 2},
        {"QCD": 4},
        accuracy="nlc",
        colour_declaration=declaration,
    )
    points = _points([0] * 4, 2)
    _assert_fft_equals_direct(slice_, points, _scales(points))


@pytest.mark.loop_physics
def test_native_seed_fft_equals_the_python_stage(compiled, loop_native):
    """The Python stage is the oracle: the native FFT's virtual equals the
    reference executor's, whose seeds the Python ``SeedFourier`` contracts
    with the S_m FFT (binary64), and both report the FFT."""

    slice_ = _slice(compiled, loop_native, "g g > g g", {"QCD": 2}, {"QCD": 4})
    native = NativeExecutor(slice_)
    points = _points([0] * 4, 2)
    scales = _scales(points)
    mine = native.evaluate(points, scales, parameters=PARAMETERS)
    for point, scale, result in zip(points, scales, mine, strict=True):
        python = slice_.evaluate(point, scale, parameters=PARAMETERS)
        assert python.colour_contraction == result.colour_contraction
        assert result.colour_contraction == SYMMETRIC_GROUP_FFT
        _assert_equal_to_rounding(result, python, 1e-12, "python stage")


@pytest.mark.loop_physics
def test_processes_without_certified_pairs_fall_back_to_direct(compiled, loop_native):
    """One external gluon (S_1) has no orbit pair: "auto" contracts directly
    and reports it, no record is exported, and the forced FFT is refused at
    generation instead of silently contracting directly."""

    request = ("u u~ > d d~ g", {"QCD": 3}, {"QCD": 5})
    slice_ = _slice(compiled, loop_native, *request)
    assert slice_.fourier is None
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    assert executor.exported.fourier is None
    assert executor.native("double").fourier is None
    points = _points([0] * 5, 1)
    (result,) = executor.evaluate(points, _scales(points), parameters=PARAMETERS)
    assert result.colour_contraction == "direct"
    with pytest.raises(ValueError, match="no certified symmetric-group orbit pair"):
        _slice(compiled, loop_native, *request, colour_contraction="fft")


@pytest.mark.loop_physics
def test_forced_fft_runs_in_every_floating_carrier(compiled, loop_native):
    """``colour_contraction="fft"`` requires the FFT: the native executor
    runs it in binary64, double-double and MPFR (the Young factors formed
    in the carrier) and equals "auto"; "direct" exports no record. The
    exact carrier (numerators only) contracts directly."""

    required = _slice(
        compiled,
        loop_native,
        "g g > g g",
        {"QCD": 2},
        {"QCD": 4},
        colour_contraction="fft",
    )
    executor = NativeExecutor(required, forward=False, inputs="native")
    assert executor.contraction == "fft"
    points = _points([0] * 4, 1)
    scales = _scales(points)
    auto = NativeExecutor(
        required,
        forward=False,
        inputs="native",
        exported=executor.exported,
        plan_json=executor._plan_json,
        contraction="auto",
    )
    for precision, tolerance in (
        (Precision.double(), 0.0),
        (Precision.double_double(), 0.0),
        (Precision.arbitrary(40), 0.0),
    ):
        (mine,) = executor.evaluate(
            points, scales, parameters=PARAMETERS, precision=precision
        )
        (theirs,) = auto.evaluate(
            points, scales, parameters=PARAMETERS, precision=precision
        )
        assert mine.colour_contraction == SYMMETRIC_GROUP_FFT
        # The same plan, record and carrier: bitwise.
        _assert_equal_to_rounding(mine, theirs, tolerance, precision)
    direct = _slice(
        compiled,
        loop_native,
        "g g > g g",
        {"QCD": 2},
        {"QCD": 4},
        colour_contraction="direct",
    )
    assert direct.fourier is None
    assert NativeExecutor(direct, forward=False).exported.fourier is None


@pytest.mark.loop_physics
def test_a_record_must_certify_on_the_plan(compiled, loop_native):
    """The native loader certifies every pair on the plan's exact weights: a
    record whose orbits do not match the plan's weights is refused."""

    slice_ = _slice(compiled, loop_native, "g g > g g", {"QCD": 2}, {"QCD": 4})
    executor = NativeExecutor(slice_, forward=False)
    record = json.loads(executor.exported.fourier)
    job = record["jobs"][0]
    # Swap two Born elements of the first orbit: no longer the group function.
    born = job["born"][0]
    born[0], born[1] = born[1], born[0]
    with pytest.raises(Exception, match="not the group function"):
        loop_native.V2Executor(executor._plan_json, fourier=json.dumps(record))


# ------------------------------------------------------------------ public route

SCHEME = """
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
"""


def _card(model, output, contraction):
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression="g g > g g", name="gg"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "{contraction}"
[loop]
r2_strategy = "model"
engine = "v2"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 1
"""


@pytest.mark.loop_physics
def test_public_route_serves_the_seed_fft(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """``color.contraction = "symmetric-group-fft"`` with ``loop.engine =
    "v2"`` generates (the dry run lists no blocker), stores the record, and
    the loaded runtime and the SDKs' Rust runtime run and report the FFT; the
    values equal the direct artifact's to rounding."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet, Runtime
    from pyamplicol.config import resolve_config

    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    points = _points([0] * 4, 2)
    momenta = [[x for p in point for x in p] for point in points]
    scales = _scales(points)
    loaded = {}
    for contraction in ("symmetric-group-fft", "direct"):
        output = tmp_path / contraction
        card = tmp_path / f"{contraction}.toml"
        card.write_text(_card(loop_sm_compiled_model, output, contraction))
        generator = Generator(resolve_config(card))
        assert not generator.plan(processes).unsupported_features
        generator.generate(processes, output)
        runtime = Runtime.load(output, process="gg", model_parameters=PARAMETERS)
        assert runtime.inspect()["color_contraction"] == (
            "fft" if contraction != "direct" else "direct"
        )
        evaluation = runtime.evaluate(points, mu_r2=scales)
        operations = {
            lane.primary_color_contraction.algorithm for lane in evaluation.diagnostics
        }
        assert operations == {
            SYMMETRIC_GROUP_FFT if contraction != "direct" else "direct"
        }
        sdk = loop_native.evaluate_v2_artifact(
            str(output), momenta, scales, process="gg", parameters=PARAMETERS
        )
        # The SDKs' Rust runtime runs the same executor: bitwise.
        for index, (values, failures) in enumerate(sdk):
            assert not failures, failures
            born, virtual = values
            assert born == list(evaluation.born_values[index])
            assert virtual == [list(laurent) for laurent in evaluation.values[index]]
        loaded[contraction] = evaluation
    fft, direct = loaded["symmetric-group-fft"], loaded["direct"]
    for a_point, b_point in zip(fft.values, direct.values, strict=True):
        for a, b in zip(a_point, b_point, strict=True):
            scale = max(abs(float(x)) for x in b)
            for x, y in zip(a, b, strict=True):
                assert abs(float(x) - float(y)) <= TOLERANCE["double"] * scale


# ---------------------------------------------------------------- nightly tier


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("process", "masses", "group"),
    [
        ("g g > g g g", [0] * 5, 24),
        ("g g > g g g g", [0] * 6, 120),
        ("u u~ > g g g g", [0] * 6, 24),
        ("g g > t t~ g g", [0, 0, 173, 173, 0, 0], 24),
        ("g g > u u~ d d~", [0] * 6, 2),
    ],
)
def test_native_seed_fft_equals_direct_at_higher_multiplicity(
    compiled, loop_native, process, masses, group
):
    """FFT = direct to rounding (binary64, double-double) on S_4 and S_5
    pure-gluon processes and on quark-gluon processes with orbits (one and
    two quark lines, massive quarks): the same certification and contraction
    as at 2 -> 2, no process key."""

    n = len(masses) - 2
    slice_ = _slice(compiled, loop_native, process, {"QCD": n}, {"QCD": n + 2})
    points = _points(masses, 2)
    stats = _assert_fft_equals_direct(slice_, points, _scales(points))
    assert stats["group_order"] == group


@pytest.mark.loop_nightly
def test_three_quark_lines_fall_back_to_direct(compiled, loop_native):
    """Three quark lines and no gluon: no adjoint label, so no orbit; the
    contraction is direct and reported (a three-line process with orbits
    needs two external gluons, a 2 -> 6 process)."""

    slice_ = _slice(compiled, loop_native, "u u~ > d d~ s s~", {"QCD": 4}, {"QCD": 6})
    assert slice_.fourier is None
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    assert executor.exported.fourier is None
    points = _points([0] * 6, 1)
    (result,) = executor.evaluate(points, _scales(points), parameters=PARAMETERS)
    assert result.colour_contraction == "direct"
