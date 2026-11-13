# SPDX-License-Identifier: 0BSD
"""Actual model-owned photon sources and crossed ports against Recola (the v2
engine on the public CLI route)."""

import io
import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, CorrelatorConfig, Runtime
from pyamplicol.cli import run_cli
from pyamplicol.models.loading import compile_model_source
from tools.loop_reference.model_compatibility import prepare_loop_sm_scalar_declarations

from .test_loop_generation_public_runtime import _card

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module", params=(1, 2), ids=("outgoing", "incoming"))
def photon_artifact(loop_sm_ufo, request, tmp_path_factory):
    source = str(loop_sm_ufo)
    root = tmp_path_factory.mktemp(f"photon-{request.param}")
    retained = os.environ.get("PYAMPLICOL_PHOTON_ARTIFACTS")
    output = (
        (Path(retained) / str(request.param)).resolve()
        if retained
        else root / "artifact"
    )
    photon_leg = 3 if request.param == 1 else 2
    if not retained:
        model = compile_model_source(
            source, restriction="no_b_mass", cache_dir=root / "model-cache"
        )
        model = replace(
            model,
            raw_loop_model=prepare_loop_sm_scalar_declarations(
                model.raw_loop_model
            ).derived,
            _serialized_path=None,
            _prepared_bundle=None,
        )
        model_path = model.write(root / "model.json")
        expression = "u u~ > a g" if request.param == 1 else "u a > u g"
        card = _card(model_path, output).replace(
            'entries = [{expression="u u~ > e- e+", name="dy"},\n'
            '           {expression="g g > g g", name="gg"}]',
            f'entries = [{{expression="{expression}", name="photon"}}]\n'
            "max_coupling_orders = {QCD=1, QED=1}",
        )
        card += "relative_tolerance = 3e-9\nabsolute_tolerance = 2e-12\n"
        card_path = root / "virtual.toml"
        card_path.write_text(card)
        correlators = CorrelatorConfig(
            spin_correlations=((photon_leg,), (4,), (photon_leg, 4))
        )
        declaration = root / "correlators.json"
        declaration.write_text(json.dumps(correlators.to_json_dict()))
        stdout, stderr = io.StringIO(), io.StringIO()
        code = run_cli(
            (
                "generate",
                "--card",
                str(card_path),
                "--correlators",
                str(declaration),
                "--json",
                "--progress",
                "off",
            ),
            stdout=stdout,
            stderr=stderr,
        )
        assert code == 0, stderr.getvalue()
        print(f"PHOTON_ARTIFACT {request.param} {output}", flush=True)
    reference = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/photon_sources.json"
    )
    rows = [
        row
        for row in json.loads(reference.read_text())["rows"]
        if row["process"] == request.param
    ]
    return output, photon_leg, rows


def _runtime(path, arithmetic="double", precision=16):
    runtime = Runtime.load(path, arithmetic=arithmetic, precision=precision)
    # Recola's chosen alpha differs slightly from the source-model default.
    runtime.set_model_parameters({"aS": "0.118", "MT": "173", "aEWM1": "132.50698"})
    return runtime


def _real(value):
    return Decimal(str(value.real)) if hasattr(value, "real") else Decimal(str(value))


def _flat(result):
    return [result.born_values[0][0], *result.values[0][0]]


@pytest.mark.parametrize(
    "arithmetic,precision",
    [
        ("double", 16),
        ("double-double", 32),
        ("arbitrary", 70),
    ],
)
def test_photon_sources_match_recola_and_cold_literal_replay(
    photon_artifact, arithmetic, precision
):
    path, photon_leg, rows = photon_artifact
    runtime = _runtime(path, arithmetic, precision)
    assert tuple(map(tuple, runtime.inspect()["spin_correlations"])) == (
        (photon_leg,),
        (photon_leg, 4),
        (4,),
    )
    for row in rows:
        vectors = {row["leg"]: row["vector"]} if row["leg"] else {}
        result = runtime.evaluate_correlated_many(
            [row["point"]],
            {"control": CorrelatedRequest("born", vectors)},
            mu_r2=row["mu_r2"],
        )["control"]
        for actual, expected in zip(_flat(result), row["cepsilon"], strict=True):
            assert float(actual.real) == pytest.approx(expected, rel=3e-9, abs=2e-12)
            assert abs(float(actual.imag)) < 2e-12
    # Literal nontransverse vectors are not independently certified by Recola's
    # physical-helicity projection. Test this contract using exact complex
    # homogeneity and source ownership instead, without modifying the vector.
    with localcontext() as ctx:
        ctx.prec = 100
        tiny = (
            Decimal(2)
            ** {"double": -25, "double-double": -70, "arbitrary": -170}[arithmetic]
        )
        vector = [["0.1", "0.2"], ["0.8", "-0.3"], ["0.4", "1"], ["-0.6", "0.5"]]
        scaled = [
            [str(Decimal(a) * (1 + tiny)), str(Decimal(b) * (1 + tiny))]
            for a, b in vector
        ]
        result = runtime.evaluate_correlated_many(
            [rows[0]["point"]] * 3,
            {
                "literal": CorrelatedRequest(
                    "born", {photon_leg: [vector, scaled, vector]}
                )
            },
            mu_r2=rows[0]["mu_r2"],
        )["literal"]
        first, second = map(_real, (result.born_values[0][0], result.born_values[1][0]))
        assert abs((second - first) - first * (2 * tiny + tiny * tiny)) < abs(
            first * tiny
        ) * Decimal("1e-5")
        assert result.values[0] == result.values[2]
        runtime.set_spin_correlation_vectors({photon_leg: vector})
        runtime.clear()
        assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)
        after = runtime.evaluate_correlated([rows[0]["point"]], mu_r2=rows[0]["mu_r2"])
        assert after.values[0] == result.values[0]


def test_photon_joint_polarization_sum(photon_artifact):
    path, photon_leg, rows = photon_artifact
    runtime = _runtime(path)
    point, scale = rows[0]["point"], rows[0]["mu_r2"]
    photon_a = ["0", "0.8", "0", "-0.6"] if photon_leg == 3 else ["0", "1", "0", "0"]
    gluon_a, b = ["0", "0.8", "0", "-0.6"], ["0", "0", "1", "0"]
    joint = runtime.evaluate_correlated_many(
        [point] * 4,
        {
            "joint": CorrelatedRequest(
                "born",
                {photon_leg: [photon_a, photon_a, b, b], 4: [gluon_a, b, gluon_a, b]},
            )
        },
        mu_r2=scale,
    )["joint"]
    ordinary = runtime.evaluate([point], mu_r2=scale)
    assert sum(float(v[0].real) for v in joint.born_values) == pytest.approx(
        ordinary.born_values[0][0], rel=3e-9, abs=2e-12
    )
    for e in range(3):
        assert sum(float(v[0][e].real) for v in joint.values) == pytest.approx(
            ordinary.values[0][0][e], rel=3e-9, abs=2e-12
        )
