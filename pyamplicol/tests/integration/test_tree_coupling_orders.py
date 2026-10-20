# SPDX-License-Identifier: 0BSD
"""Squared coupling-order partials of tree results against MadGraph split orders.

The committed reference was produced by ``tools/developer/madgraph_split_orders.py``
from MadGraph 3.7.0 standalone output of the UFO ``sm`` model (default
restriction, t-channel widths kept) at the momenta it records.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime, TreeOrderEvaluation
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.cli.main import main as cli_main
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
)

REFERENCE = (
    Path(__file__).resolve().parents[1] / "fixtures/tree/madgraph-split-orders.json"
)
ORDER_LIMITS = {"QCD": 99, "QED": 99}


def _reference(process: str) -> dict[str, object]:
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    (entry,) = (item for item in payload["processes"] if item["process"] == process)
    return entry


@pytest.fixture(scope="module")
def sm_model(
    madgraph_source: Path, tmp_path_factory: pytest.TempPathFactory
) -> ModelSource:
    root = tmp_path_factory.mktemp("madgraph-sm")
    bundle = root / "sm.pyamplicol-model"
    ModelSource.from_path(madgraph_source / "models/sm", restriction="default").compile(
        cache_dir=root / "cache",
        prepared_output=bundle,
        evaluator=EvaluatorConfig(),
    )
    return ModelSource.from_path(bundle)


def _runtime(
    model: ModelSource,
    process: str,
    artifact: Path,
    *,
    accuracy: str = "full",
    execution_mode: str = "recurrence",
    lc_flow_layout: str = "topology-replay",
) -> Runtime:
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy=accuracy, lc_flow_layout=lc_flow_layout),
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders=ORDER_LIMITS
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, samples=1, seed=7, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )
    Generator(config).generate(process, artifact, model=model)
    return Runtime.load(artifact)


def _assert_matches_madgraph(
    result: TreeOrderEvaluation, reference, *, interference: bool
) -> None:
    """Compare squared orders that are (not) interference between distinct
    amplitude orders, relative to the summed magnitude of all partials."""

    expected_orders = tuple(
        tuple(sorted((name, power) for name, power in order.items() if power))
        for order in reference["squared_orders"]
    )
    assert set(result.orders) == set(expected_orders)
    mixed = {
        sector.order
        for sector in result.sectors
        if any(left != right for left, right in sector.pairs)
    }
    compared = [order for order in expected_orders if (order in mixed) == interference]
    assert compared
    for point, partials in enumerate(reference["partials"]):
        scale = math.fsum(abs(value) for value in partials)
        for order, expected in zip(expected_orders, partials, strict=True):
            if order not in compared:
                continue
            actual = complex(result.for_order(order)[point])
            assert actual.imag == 0.0
            assert abs(actual.real - expected) <= 1e-12 * scale, (point, order)


@pytest.mark.parametrize(
    "process",
    ["u u~ > d d~", "g g > t t~", "u u~ > e- e+ g", "u u~ > d d~ g"],
)
def test_partials_match_madgraph_and_keep_the_total(
    sm_model: ModelSource, tmp_path: Path, process: str
) -> None:
    reference = _reference(process)
    runtime = _runtime(sm_model, process, tmp_path / "artifact")
    points = reference["points"]

    result = runtime.evaluate_orders(points)

    assert result.total == runtime.evaluate(points)
    _assert_matches_madgraph(result, reference, interference=False)
    for total, summed in zip(result.total, result.sum_orders(), strict=True):
        assert math.isclose(complex(summed).real, complex(total).real, rel_tol=1e-13)
    assert all(
        pair[0] in result.amplitude_orders and pair[1] in result.amplitude_orders
        for sector in result.sectors
        for pair in sector.pairs
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pre-existing tree-engine sign: the recurrence's QCD amplitude of "
        "coloured four-fermion processes has the opposite sign to its "
        "electroweak amplitude, so totals and QCD x EW interference differ "
        "from MadGraph (diagonal orders agree)"
    ),
)
@pytest.mark.parametrize("process", ["u u~ > d d~", "u u~ > d d~ g"])
def test_interference_partials_match_madgraph(
    sm_model: ModelSource, tmp_path: Path, process: str
) -> None:
    reference = _reference(process)
    runtime = _runtime(sm_model, process, tmp_path / "artifact")

    _assert_matches_madgraph(
        runtime.evaluate_orders(reference["points"]), reference, interference=True
    )


def test_exact_partials_agree_and_keep_the_exact_total(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    reference = _reference("u u~ > d d~ g")
    runtime = _runtime(sm_model, "u u~ > d d~ g", tmp_path / "artifact")
    points = reference["points"][:1]

    exact = runtime.evaluate_orders(points, precision=32)
    native = runtime.evaluate_orders(points)

    assert exact.total == runtime.evaluate(points, precision=32)
    assert exact.orders == native.orders
    assert all(isinstance(value, Decimal) for value in exact.values[0])
    assert exact.sum_orders()[0] == pytest.approx(exact.total[0], rel=Decimal("1e-30"))
    scale = math.fsum(abs(float(value)) for value in exact.values[0])
    for exact_value, native_value in zip(
        exact.values[0], native.values[0], strict=True
    ):
        assert abs(float(exact_value) - complex(native_value).real) <= 1e-13 * scale


def _assert_matches_exact(runtime: Runtime, points: list[object]) -> None:
    """Partials keep the total, sum to it and agree with the exact executor."""

    result = runtime.evaluate_orders(points)
    exact = runtime.evaluate_orders(points[:1], precision=32)

    assert result.total == runtime.evaluate(points)
    assert exact.total == runtime.evaluate(points[:1], precision=32)
    assert len(result.amplitude_orders) > 1
    for total, summed in zip(result.total, result.sum_orders(), strict=True):
        assert math.isclose(complex(summed).real, complex(total).real, rel_tol=1e-13)
    scale = math.fsum(abs(float(value)) for value in exact.values[0])
    for exact_value, native_value in zip(
        exact.values[0], result.values[0], strict=True
    ):
        assert abs(float(exact_value) - complex(native_value).real) <= 1e-13 * scale


@pytest.mark.parametrize(
    ("accuracy", "layout"),
    [("lc", "topology-replay"), ("lc", "all-flow-union"), ("nlc", "topology-replay")],
)
def test_truncated_colour_partials_agree_with_exact(
    sm_model: ModelSource, tmp_path: Path, accuracy: str, layout: str
) -> None:
    runtime = _runtime(
        sm_model,
        "u u~ > d d~ g",
        tmp_path / "artifact",
        accuracy=accuracy,
        lc_flow_layout=layout,
    )
    _assert_matches_exact(runtime, _reference("u u~ > d d~ g")["points"])


def test_symmetric_group_colour_partials_agree_with_exact(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    # Full colour with two gluons selects the symmetric-group Fourier reducer.
    runtime = _runtime(sm_model, "u u~ > d d~ g g", tmp_path / "artifact")
    manifest = json.loads((tmp_path / "artifact/artifact.json").read_text())
    assert (
        "rusticol.color-contraction.symmetric-group-fft.v1"
        in (manifest["processes"][0]["required_runtime_capabilities"])
    )
    _assert_matches_exact(runtime, [runtime._backend.validation_momenta()[0]])


def test_cli_prints_partials_and_rejects_non_recurrence_lanes(
    sm_model: ModelSource, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _reference("u u~ > d d~")
    _runtime(sm_model, "u u~ > d d~", tmp_path / "artifact")
    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps(reference["points"]), encoding="utf-8")

    assert (
        cli_main(
            [
                "evaluate",
                str(tmp_path / "artifact"),
                "--momenta",
                str(momenta),
                "--orders",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["contribution_kind"] == "tree"
    assert payload["shape"] == [len(reference["points"]), 3]
    assert len(payload["values"]) == len(reference["points"])

    eager = _runtime(
        sm_model, "u u~ > d d~", tmp_path / "eager", execution_mode="eager"
    )
    with pytest.raises(CompatibilityError, match="recurrence artifact"):
        eager.evaluate_orders(reference["points"])
