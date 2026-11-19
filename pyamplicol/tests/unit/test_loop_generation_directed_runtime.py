# SPDX-License-Identifier: 0BSD
"""Fresh public colour-history generation (v2) with an independent Recola dipole."""

import io
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, CorrelatedValue, CorrelatorConfig, Runtime
from pyamplicol.cli import run_cli
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.models.loading import compile_model_source
from tools.loop_reference.model_compatibility import prepare_loop_sm_scalar_declarations

from .test_loop_generation_public_runtime import _card

pytestmark = pytest.mark.loop_fast


def test_fresh_public_virtual_dipole_matches_recola(
    loop_reference_path, loop_sm_ufo, tmp_path
):
    source = str(loop_sm_ufo)
    reference = str(loop_reference_path("directed-recola-control"))
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    # A real unified extension/source identity is required. No companion import,
    # target_info, artifact check or runtime method is replaced in this test.
    native = load_loop_native()
    from pyamplicol import _rusticol

    assert native is _rusticol._loop
    verify_native_module(_rusticol)
    model = compile_model_source(
        source, restriction="no_b_mass", cache_dir=tmp_path / "model-cache"
    )
    # The declared no_b_mass source card fixes MB=0; MT=173 and alpha_s=.118
    # match the independent provider. Only the established four-G0 source
    # Lorentz arity defect is corrected, explicitly, before loop compilation.
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    model_path = model.write(tmp_path / "model.json")
    output = tmp_path / "artifact"
    card = _card(model_path, output).replace(
        'entries = [{expression="u u~ > e- e+", name="dy"},\n'
        '           {expression="g g > g g", name="gg"}]',
        'entries = [{expression="u u~ > d d~", name="uudd"}]\n'
        "max_coupling_orders = {QCD=2, QED=0}",
    )
    card_path = tmp_path / "virtual.toml"
    card_path.write_text(card)
    declarations = CorrelatorConfig(
        color_correlations=(
            ColorCorrelator.dipole("T11", 1, 1),
            ColorCorrelator.dipole("T12", 1, 2),
        )
    )
    declaration_path = tmp_path / "correlators.json"
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
    assert output.is_dir()
    # Successful publication includes staged ordinary AND every directed Born,
    # order and pole coefficient at two generated points, against source kernels.
    runtime = Runtime.load(output, process="uudd")
    assert [item.id for item in runtime.available_color_correlations()] == [
        "born",
        "T11",
        "T12",
    ]
    assert runtime.available_color_correlations() == declarations.color_requests
    metadata = runtime.inspect()
    assert metadata["engine"] == "v2"
    # v2 records the declared histories by id (the born history is implicit).
    assert metadata["color_correlations"] == ["T11", "T12"]
    rows = json.loads(Path(reference).read_text())["rows"]
    points, scales = [row["point"] for row in rows], [row["mu_r2"] for row in rows]
    for arithmetic, precision in (
        ("double", 16),
        ("double-double", 32),
        ("arbitrary", 70),
    ):
        runtime = Runtime.load(
            output, process="uudd", arithmetic=arithmetic, precision=precision
        )
        grouped = runtime.evaluate_correlated_many(
            points,
            {name: CorrelatedRequest(name) for name in ("T12", "born", "T11")},
            mu_r2=scales,
        )
        assert list(grouped) == ["T12", "born", "T11"]
        for name in grouped:
            assert grouped[name].correlated and grouped[name].shape == (2, 1, 3)
            assert all(row.status == "success" for row in grouped[name].diagnostics)
        for point, row in enumerate(rows):
            for name, dipole, multiplier in (
                ("born", "1", 1),
                ("T11", "1", 4 / 3),
                ("T12", "2", 4 / 3),
            ):
                result = grouped[name]
                values = (result.born_values[point][0], *result.values[point][0])
                # Recola has already been converted to C_epsilon in this
                # independent reference; its quark dipoles carry 1/C_F.
                for actual, target in zip(
                    values, row["dipoles"][dipole]["cepsilon"], strict=True
                ):
                    if isinstance(actual, CorrelatedValue):
                        assert isinstance(actual.real, Decimal)
                        actual = complex(float(actual.real), float(actual.imag))
                    assert actual.real == pytest.approx(
                        multiplier * target, rel=3e-10, abs=2e-13
                    )
                    assert abs(actual.imag) < 2e-13
        ordinary = runtime.evaluate(points, mu_r2=scales)
        for point in range(2):
            born_history = grouped["born"].values[point][0]
            for actual, expected in zip(
                born_history, ordinary.values[point][0], strict=True
            ):
                real = actual.real
                assert float(real) == pytest.approx(
                    float(expected), rel=3e-10, abs=2e-13
                )
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        ("inspect", str(output), "--process", "uudd", "--full-physics", "--json"),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    inspection = json.loads(stdout.getvalue())
    assert "T12" in json.dumps(inspection)
