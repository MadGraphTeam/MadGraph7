# SPDX-License-Identifier: 0BSD
"""Bounded source-semantic CMS control, not physical scheme certification."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


def _prepare_cms_source(reglog_convention="source"):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source graph generation requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_EW_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_EW_UFO_REFERENCE to the full Gmu UFO")
    from pyamplicol.loops.counterterms import prepare_counterterm_model
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_cms import prepare_gmu_complex_mass_declarations
    from pyamplicol.loops.model_electroweak import (
        neutral_lepton_counterterm_validations,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    view = prepare_gmu_complex_mass_declarations(
        load_raw_loop_model(path, restriction="full"),
        reglog_convention=reglog_convention,
    )
    zeros = (
        "aEW0M1",
        "MD",
        "MU",
        "MS",
        "MC",
        "MB",
        "Me",
        "MM",
        "MTA",
        "lamWS",
        "AWS",
        "rhoWS",
        "etaWS",
    )
    ones = ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice")
    structure = build_covariant_structure(
        view.derived,
        fixed_parameters={**dict.fromkeys(zeros, "0"), **dict.fromkeys(ones, "1")},
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )
    ir = build_loop_model(view.derived, scalar_program=structure.scalars)
    prepared = prepare_counterterm_model(
        ir,
        structure.native,
        primitive_validations=neutral_lepton_counterterm_validations(ir),
        vertex_spinor_frame="ufo",
    )
    return view, structure, prepared


@pytest.fixture(scope="module")
def cms_source():
    return _prepare_cms_source()


CASES = (
    ("double", None, 0, "2e-11"),
    ("double-double", None, 80, "5e-25"),
    ("arbitrary", 80, 180, "2e-64"),
)


def _prepare_cms_program(cms_source, case, reference):
    from symbolica import E

    from pyamplicol.loops.physical_source import source_model_program

    arithmetic, precision, bit, tolerance = case
    view, structure, prepared = cms_source
    assert not view.derived.metadata["complex_mass_derivation"][
        "physical_certification"
    ]
    program = source_model_program(
        structure,
        prepared,
        ("e-", "e+"),
        ("m-", "m+"),
        selected_corrections=["QED"],
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("4"),
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
        parameters={
            "Gf": "0.0000116639",
            "aS": "0.118",
            "MU_R": "1000",
            "MT": "173.3",
            "MZ": "91.188",
            "MW": "80.419",
            "MH": "125",
            "WZ": "2.49877",
            "WW": "2.09291",
            "WT": "1.37758",
            "WH": "0.00407468",
        },
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )
    return view, structure, program, reference, bit, tolerance


@pytest.fixture(scope="module")
def cms_program(cms_source, request):
    reference = json.loads(
        (
            Path(__file__).resolve().parents[1] / "fixtures/loop/cms-source-gmu.json"
        ).read_text()
    )
    return _prepare_cms_program(cms_source, request.param, reference)


def _compare(result, expected):
    np.testing.assert_allclose(
        result.born[:, 0].astype(float),
        np.asarray(expected["born"], dtype=float),
        rtol=3e-12,
    )
    for name, field in (
        ("bare", "bare"),
        ("uv", "uv"),
        ("r2", "r2"),
        ("full", "virtual"),
    ):
        values = getattr(result, field)
        np.testing.assert_allclose(
            values[:, :, 0].astype(float),
            np.asarray(expected[name], dtype=float),
            rtol=2e-10,
            atol=2e-13,
        )
        assert np.max(np.abs(values[:, :, 1].astype(float))) < 1e-12


@pytest.mark.parametrize("cms_program", CASES, indirect=True)
def test_source_semantic_cms_components_and_precise_update(cms_program):
    _view, structure, program, reference, bit, tolerance = cms_program
    program.update_parameters(
        {
            "Gf": "0.0000116639",
            "WZ": "2.49877",
            "WW": "2.09291",
            "WT": "1.37758",
            "WH": "0.00407468",
        }
    )
    finite_points = np.asarray(reference["finite"]["points"], dtype=str)
    finite = program.evaluate_decimal(finite_points, ["1000000"] * len(finite_points))
    _compare(finite, reference["finite"])
    # Genuine precision evidence independent of the binary64 reference: exact
    # Gf rebinding scales Born by f² and every one-loop component by f³.
    with localcontext() as context:
        context.prec = 110
        factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        program.update_parameters({"Gf": str(Decimal("0.0000116639") * factor)})
        rebound = program.evaluate_decimal(finite_points[:1], ["1000000"])
        assert Decimal(rebound.born[0, 0]) != Decimal(finite.born[0, 0])
        assert abs(
            Decimal(rebound.born[0, 0]) / Decimal(finite.born[0, 0]) - factor**2
        ) < Decimal(tolerance)
        for field in ("bare", "uv", "r2", "virtual"):
            assert abs(
                Decimal(getattr(rebound, field)[0, 0, 0])
                / Decimal(getattr(finite, field)[0, 0, 0])
                - factor**3
            ) < Decimal(tolerance)
    assert program.source_bare.structure is structure


@pytest.mark.parametrize(
    "cms_program",
    [
        CASES[0],
        CASES[1],
        pytest.param(
            CASES[2],
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "NinjaRS 5f35b07: AP80 sqrt-square mass precision changes "
                    "the scalar box branch; see reproduce_cms_zero_box.py. "
                    "Mandatory AP zero-width validation remains open."
                ),
            ),
        ),
    ],
    indirect=True,
)
def test_source_semantic_cms_zero_width(cms_program):
    _view, _structure, program, reference, _bit, _tolerance = cms_program
    # The same graph/CT program consumes updated original width inputs. The
    # original unmodified CMS reference equals the real-mass reference here.
    program.update_parameters(
        {"Gf": "0.0000116639", "WZ": "0", "WW": "0", "WT": "0", "WH": "0"}
    )
    points = np.asarray(reference["zero"]["points"], dtype=str)
    zero = program.evaluate_decimal(points, ["1000000"] * len(points))
    _compare(zero, reference["zero"])


@pytest.fixture(scope="module")
def standard_source():
    return _prepare_cms_source("ufo-standard")


@pytest.mark.parametrize("case", CASES[:2])
def test_standard_cms_complete_components_across_width_updates(standard_source, case):
    reference = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/loop/cms-ufo-standard-gmu.json"
        ).read_text()
    )
    view, _structure, program, _reference, _bit, _tolerance = _prepare_cms_program(
        standard_source, case, reference
    )
    assert (
        view.derived.metadata["complex_mass_derivation"]["reglog_convention"]
        == "ufo-standard"
    )
    assert not view.derived.metadata["complex_mass_derivation"][
        "physical_certification"
    ]
    results = {}
    # One program per carrier, with the same original global width inputs
    # rebound at runtime; no graph regeneration or scalar formula replacement.
    for mode in ("finite", "small", "zero"):
        expected = reference[mode]
        program.update_parameters(expected["widths"])
        points = np.asarray(expected["points"], dtype=str)
        result = program.evaluate_decimal(points, ["1000000"] * len(points))
        _compare(result, expected)
        np.testing.assert_allclose(
            result.virtual[:, :, 0].astype(float),
            result.bare[:, :, 0].astype(float)
            + result.uv[:, :, 0].astype(float)
            + result.r2[:, :, 0].astype(float),
            rtol=1e-13,
            atol=1e-14,
        )
        results[mode] = {
            "born": result.born.tolist(),
            **{
                name: getattr(result, field).tolist()
                for name, field in (
                    ("bare", "bare"),
                    ("uv", "uv"),
                    ("r2", "r2"),
                    ("full", "virtual"),
                )
            },
        }
    # Keep optional numerical evidence outside the checkout. The assertions
    # above do not depend on this diagnostic report being requested.
    if directory := os.environ.get("PYAMPLICOL_CMS_CONTROL_OUTPUT"):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"standard-{case[0]}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
