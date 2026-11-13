# SPDX-License-Identifier: 0BSD
"""Fresh public joint-source publication and independent singleton-spin physics
(the v2 engine on the public CLI route against Recola)."""

import io
import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, CorrelatorConfig, Runtime
from pyamplicol.cli import run_cli
from pyamplicol.models.loading import compile_model_source
from tools.loop_reference.model_compatibility import prepare_loop_sm_scalar_declarations

from .test_loop_generation_public_runtime import _card

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def public_spin_artifact(loop_sm_ufo, loop_reference, tmp_path_factory):
    source = str(loop_sm_ufo)
    rows = loop_reference("spin-recola-control")["rows"]
    retained = os.environ.get("PYAMPLICOL_PUBLIC_SPIN_ARTIFACT")
    if retained:
        return Path(retained), rows
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    root = tmp_path_factory.mktemp("public-spin")
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
    output = root / "artifact"
    card = _card(model_path, output).replace(
        'entries = [{expression="u u~ > e- e+", name="dy"},\n'
        '           {expression="g g > g g", name="gg"}]',
        'entries = [{expression="u u~ > g g", name="uugg"}]\n'
        "max_coupling_orders = {QCD=2, QED=0}",
    )
    # The shared generation default targets tree comparisons at 1e-12 relative.
    # This loop fixture uses the same binary64 tolerance as its independently
    # matched provider control, retaining every ordinary and source-class check.
    card += "relative_tolerance = 3e-9\nabsolute_tolerance = 2e-12\n"
    card_path = root / "virtual.toml"
    card_path.write_text(card)
    declarations = CorrelatorConfig(spin_correlations=((3,), (4,), (3, 4)))
    declaration_path = root / "correlators.json"
    declaration_path.write_text(json.dumps(declarations.to_json_dict()))
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        (
            "generate",
            "--card",
            str(card_path),
            "--correlators",
            str(declaration_path),
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    # Normal staged validation checks every declared source class and Laurent
    # order with generic literal complex vectors. No native identity is patched.
    print(f"PUBLISHED_PUBLIC_SPIN {output}", flush=True)
    return output, rows


def _real(value):
    return (
        float(value.real)
        if isinstance(value, (CorrelatedValue, complex))
        else float(value)
    )


@pytest.mark.parametrize(
    "arithmetic,precision", [("double", 16), ("double-double", 32), ("arbitrary", 70)]
)
def test_public_singleton_spin_matches_independent_recola(
    public_spin_artifact, arithmetic, precision
):
    path, rows = public_spin_artifact
    runtime = Runtime.load(path, arithmetic=arithmetic, precision=precision)
    assert tuple(map(tuple, runtime.inspect()["spin_correlations"])) == (
        (3,),
        (3, 4),
        (4,),
    )
    result = runtime.evaluate_correlated_many(
        [row["point"] for row in rows],
        {"spin": CorrelatedRequest("born", {3: [row["vector"] for row in rows]})},
        mu_r2=[row["mu_r2"] for row in rows],
    )["spin"]
    assert result.shape == (2, 1, 3) and result.correlated
    for index, row in enumerate(rows):
        for actual, expected in zip(
            (result.born_values[index][0], *result.values[index][0]),
            row["cepsilon"],
            strict=True,
        ):
            assert _real(actual) == pytest.approx(expected, rel=3e-11, abs=2e-13)
            assert abs(float(actual.imag)) < 2e-13
            assert isinstance(
                actual, complex if arithmetic == "double" else CorrelatedValue
            )
    # The external reference is binary64. This is provider agreement, not a
    # new claim of 70 independently validated digits.


def test_public_joint_sources_preserve_normalization_and_defaults(public_spin_artifact):
    path, rows = public_spin_artifact
    runtime = Runtime.load(path)
    point = rows[0]["point"]
    a = rows[0]["vector"]
    b = ["0", "0", "1", "0"]
    assert Decimal(point[2][2]) == 0  # the frozen control lies in the x-z plane
    joint = runtime.evaluate_correlated_many(
        [point] * 4,
        {"joint": CorrelatedRequest("born", {3: [a, a, b, b], 4: [a, b, a, b]})},
        mu_r2=rows[0]["mu_r2"],
    )["joint"]
    ordinary = runtime.evaluate([point], mu_r2=rows[0]["mu_r2"])
    assert sum(_real(row[0]) for row in joint.born_values) == pytest.approx(
        ordinary.born_values[0][0], rel=3e-9
    )
    for epsilon in range(3):
        assert sum(_real(row[0][epsilon]) for row in joint.values) == pytest.approx(
            ordinary.values[0][0][epsilon], rel=3e-9
        )
    runtime.set_spin_correlation_vectors({3: a})
    same = runtime.evaluate_correlated_many(
        [point],
        {
            "inherited": CorrelatedRequest(),
            "ordinary": CorrelatedRequest(spin_vectors={}),
        },
        mu_r2=rows[0]["mu_r2"],
    )
    runtime.clear()
    after = runtime.evaluate_correlated([point], mu_r2=rows[0]["mu_r2"])
    assert after.values == same["inherited"].values
    assert _real(same["ordinary"].born_values[0][0]) == pytest.approx(
        ordinary.born_values[0][0]
    )
    assert runtime.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)
