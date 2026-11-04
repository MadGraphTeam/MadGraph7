# SPDX-License-Identifier: 0BSD
"""Catani's pole prediction on the native-input route (M-FINAL MF-8G-3 (i),
F11): the executor contracts the stored dipole orbits with the Born vectors
of every helicity row (``NativeExecutor(catani=True)``,
``V2Executor.evaluate(..., dipoles=)``; no dense operator), and
``_NativePoint`` forms the prediction from them, on a live slice with
``_PointEvaluation``'s operators and on a stored process with its Catani
record (``dipoles.catani_record``: massless QCD).

The oracle is the Python-input route's prediction (``_PointEvaluation.
_qcd_poles`` on the Python Born vectors and ``LoopSlice.dipoles``), one point
per process (HANDOFF §0.3): binary64 within 1e-12 relative, double-double
within 1e-24. A massive coloured leg is served on a live slice; a stored
process records no Catani record for it (no prediction). The executor's
correlations obey colour conservation (sum_{j != i} <T_i.T_j> = -C_i |A0|^2).
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.native_process import NativeProcess
from pyamplicol.loops.v2.plan_export import NativeExecutor

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0", "WT": "0"}


def _legacy_point(name: str):
    data = json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())
    return data["points"][0], data["scales_squared"][0]


def _required_point(name: str):
    from decimal import Decimal

    data = json.loads((FIXTURES / "required" / f"{name}.json").read_text())
    mu = Decimal(str(data["mu_r"][0]))
    return data["points"][0], format(mu * mu, "f")


# name: process, Born orders, point
PROCESSES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, lambda: _legacy_point("gg_gg")),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        lambda: _required_point("uux_ddxg"),
    ),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, lambda: _required_point("gg_ttx")),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            process, born, _point = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            built[name] = LoopSlice.generate(
                compiled,
                SliceRequest(process, born, loop, fixed_parameters=FIXED),
                loop_native,
            )
        return built[name]

    return build


def _evaluate(executor, name, precision):
    point, scale = PROCESSES[name][2]()
    (result,) = executor.evaluate(
        [point], [scale], parameters=PARAMETERS, precision=precision
    )
    return result


def _assert_equal_predictions(native, python, tolerance):
    assert native.ir_poles is not None and python.ir_poles is not None
    assert set(native.ir_poles) == set(python.ir_poles)
    for sector, (single, double) in python.ir_poles.items():
        got = native.ir_poles[sector]
        for a, b in zip(got, (single, double), strict=True):
            a, b = complex(a), complex(b)
            assert abs(a - b) <= tolerance * max(abs(b), 1e-300), (sector, a, b)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ("gg_gg", "uux_ddxg", "gg_ttx"))
def test_native_catani_equals_the_python_route(slices, name):
    slice_ = slices(name)
    python = _evaluate(NativeExecutor(slice_), name, Precision.double())
    native = NativeExecutor(slice_, inputs="native", catani=True)
    _assert_equal_predictions(
        _evaluate(native, name, Precision.double()), python, 1e-12
    )
    # Without the option the native route forms none (the default).
    plain = NativeExecutor(slice_, inputs="native")
    assert _evaluate(plain, name, Precision.double()).ir_poles is None


@pytest.mark.loop_physics
def test_native_catani_in_double_double(slices):
    slice_ = slices("uux_ddxg")
    python = _evaluate(NativeExecutor(slice_), "uux_ddxg", Precision.double_double())
    native = NativeExecutor(slice_, inputs="native", catani=True)
    _assert_equal_predictions(
        _evaluate(native, "uux_ddxg", Precision.double_double()), python, 1e-24
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ("gg_gg", "uux_ddxg", "gg_ttx"))
def test_stored_process_forms_catani_from_its_record(slices, loop_native, name):
    slice_ = slices(name)
    live = NativeExecutor(slice_, inputs="native")
    stored = NativeProcess.of(live)
    loaded = NativeProcess.from_members(stored.plan, stored.members())
    executor = loaded.executor(loop_native, catani=True)
    result = _evaluate(executor, name, Precision.double())
    if name == "gg_ttx":
        # A massive coloured leg: the stored route records no Catani record.
        assert "catani" not in loaded.record
        assert result.ir_poles is None
        return
    record = loaded.record["catani"]
    assert record["strong"] == "QCD"
    assert all(Fraction(gamma) for _label, _rep, gamma in record["legs"])
    python = _evaluate(NativeExecutor(slice_), name, Precision.double())
    _assert_equal_predictions(result, python, 1e-12)
    # The record holds only where its conditions do (the zero-ness of each
    # adjustable mass the leg masses and n_f read: MT here, MB was fixed at
    # generation); a flipped one forms no prediction.
    from pyamplicol.loops.v2 import dipoles

    conditions = [tuple(condition) for condition in record["conditions"]]
    assert conditions
    assert dipoles.record_applies(record, {**PARAMETERS, **FIXED})
    for mass, zero in conditions:
        assert not dipoles.record_applies(record, {mass: "1" if zero else "0"})
    mass, zero = conditions[0]
    point, scale = PROCESSES[name][2]()
    moved = executor.evaluate(
        [point],
        [scale],
        parameters={**PARAMETERS, mass: "1" if zero else "0"},
        precision=Precision.double(),
    )[0]
    assert moved.ir_poles is None


@pytest.mark.loop_physics
def test_native_correlations_conserve_colour(slices):
    from pyamplicol.loops.v2.assemble import _casimir

    slice_ = slices("uux_ddxg")
    executor = NativeExecutor(slice_, inputs="native", catani=True)
    point, scale = PROCESSES["uux_ddxg"][2]()
    precision = Precision.double()
    precision, evaluations, failed = executor._evaluated(
        [point], [scale], PARAMETERS, precision, "all", None, catani=True
    )
    assert not failed
    (evaluation,) = evaluations
    correlations = evaluation.correlations
    assert correlations is not None
    born = sum(complex(v).real for v in evaluation.born.values())
    representation = {leg.label: leg.representation for leg in slice_.basis.legs}
    coloured = [label for label, _rep in executor._correlator().coloured]
    average = slice_.normalisation.average
    for i in coloured:
        total = sum(
            values[i, j]
            for values in correlations.values()
            for j in coloured
            if j != i
        )
        expected = -float(_casimir(representation[i])) * born * average
        assert abs(total - expected) <= 1e-12 * abs(expected)
