# SPDX-License-Identifier: 0BSD
"""Selector-only full colour (``generation.selector_only_full_colour``).

Eligible full-colour classes are generated with their helicity-selector
companion only: no main contracted-colour schedule. Per helicity they must
equal the default artifact's companion values at the same batch, bitwise;
helicity sums, resolved output and the other APIs that need the main schedule
are refused. Flavour-union templates keep their main schedule.

``p p > j j j`` runs only with ``PYAMPLICOL_RUN_SELECTOR_ONLY=pp3j``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from pyamplicol import EvaluationError, Generator, ProcessSet, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.artifacts import inspect_artifact
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import flat_final_states
from tests.integration.test_runtime_selector_api_parity import native_sdk  # noqa: F401

_SELECTOR_ONLY = "rusticol.recurrence-selector-only-fc.v1"
_CASES = {"pp2j": "p p > j j", "pp3j": "p p > j j j"}
_OPT_IN = {"pp3j"}
_OPT_IN_ENV = "PYAMPLICOL_RUN_SELECTOR_ONLY"
Point = tuple[tuple[float, float, float, float], ...]


def _config(
    *,
    unions: bool,
    selector_only: bool,
    validate: bool = False,
    emit_api: bool = False,
    qed: int = 0,
    contraction: str = "auto",
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full", contraction=contraction),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=True,
            flavor_unions=unions,
            max_coupling_orders={"QED": qed},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=emit_api,
            selector_only_full_colour=selector_only,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=validate, post_build_validation=validate
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module", params=tuple(_CASES))
def artifacts(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    if request.param in _OPT_IN and request.param not in os.environ.get(
        _OPT_IN_ENV, ""
    ).split(","):
        pytest.skip(f"set {_OPT_IN_ENV}={request.param} to run it")
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    root = tmp_path_factory.mktemp(f"selector-only-{request.param}")
    processes = ProcessSet.from_expressions((_CASES[request.param],))
    paths = {}
    try:
        for unions in (False, True):
            for selector_only in (False, True):
                name = f"{'selector' if selector_only else 'default'}-u{int(unions)}"
                paths[name] = root / name
                # MG7's configuration: with the UMAMI/API bundle.
                Generator(
                    _config(unions=unions, selector_only=selector_only, emit_api=True)
                ).generate(processes, paths[name])
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _manifest(artifact: Path) -> dict[str, object]:
    return json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))


def _execution(artifact: Path, process_id: str) -> dict[str, object]:
    return json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )


def _selector_only_ids(artifact: Path) -> list[str]:
    return [
        str(process["id"])
        for process in _manifest(artifact)["processes"]
        if _SELECTOR_ONLY
        in _execution(artifact, str(process["id"]))["required_runtime_capabilities"]
    ]


def _points(expression: str, count: int) -> list[Point]:
    """Flat massless 2 -> n points at sqrt(s) = 1 TeV, beams along z."""

    final_count = len(expression.split(">")[1].split())
    initial = ((500.0, 0.0, 0.0, 500.0), (500.0, 0.0, 0.0, -500.0))
    states = flat_final_states(
        [0.0] * final_count, sqrt_s=1000.0, seeds=range(1, count + 1)
    )
    return [
        initial + tuple(tuple(float(x) for x in leg) for leg in state.momenta)
        for state in states
    ]


def _real(values: Sequence[object]) -> list[float]:
    return [complex(value).real for value in values]  # type: ignore[arg-type]


def _reducer(artifact: Path, process_id: str) -> tuple[object, object]:
    color = _execution(artifact, process_id)["runtime_metadata"]["color_contraction"]
    factorization = color.get("factorization") or {}
    return color["storage"], factorization.get("kind")


def test_selector_only_processes_omit_the_main_schedule(
    artifacts: dict[str, Path],
) -> None:
    for unions in (0, 1):
        selector = artifacts[f"selector-u{unions}"]
        default = artifacts[f"default-u{unions}"]
        assert not _selector_only_ids(default)
        selected = _selector_only_ids(selector)
        assert selected
        for process in _manifest(selector)["processes"]:
            process_id = str(process["id"])
            root = selector / "processes" / process_id
            execution = _execution(selector, process_id)
            assert not (root / "recurrence-bootstrap.bin").exists()
            if process_id not in selected:
                # Union templates keep the main schedule (no companion yet).
                assert unions == 1
                assert execution["plan"] is not None
                assert execution["helicity_selector_companion"] is None
                continue
            assert execution["plan"] is None
            assert execution["recurrence_summary"]["lanes"] == (
                "helicity-selector-only"
            )
            companion = execution["helicity_selector_companion"]
            assert companion["color_contraction"] == {
                "source": "selector-local",
                "view": "selector-local-color-view-v1",
            }
            assert (
                execution["runtime_metadata"]["color_contraction"]["component_count"]
                == 1
            )
            assert not (root / "recurrence-binding.bin").exists()
            assert (root / "helicity-selector-binding.bin").exists()
            # Same reducer as the default artifact's local block.
            assert _reducer(selector, process_id) == _reducer(default, process_id)
            physics = json.loads((root / "physics.json").read_text("utf-8"))
            default_physics = json.loads(
                (default / "processes" / process_id / "physics.json").read_text("utf-8")
            )
            assert [h["id"] for h in physics["helicities"]] == [
                h["id"] for h in default_physics["helicities"]
            ]
            assert not any(h["structural_zero"] for h in physics["helicities"])
        umami = json.loads((selector / "API/umami/metadata.json").read_text())
        assert umami["helicity_groups"]
        inspected = {
            process.id: process for process in inspect_artifact(selector).processes
        }
        for process_id in selected:
            assert inspected[process_id].recurrence_lanes == (
                "selector-only (no main schedule)"
            )


def test_per_helicity_values_equal_the_default_companion_bitwise(
    artifacts: dict[str, Path],
) -> None:
    generator = random.Random(20261121)
    for unions in (0, 1):
        selector_artifact = artifacts[f"selector-u{unions}"]
        default_artifact = artifacts[f"default-u{unions}"]
        for process_id in _selector_only_ids(selector_artifact):
            selector = Runtime.load(selector_artifact, process=process_id)
            default = Runtime.load(default_artifact, process=process_id)
            helicities = list(default.physics.helicity_ids)
            zeros = {
                helicity.id
                for helicity in default.physics.helicities
                if helicity.structural_zero
            }
            points = _points(selector.physics.process, 33)
            scale = max(
                abs(value)
                for h in helicities
                if h not in zeros
                for value in _real(default.evaluate(points[:1], helicities=[h]))
            )
            for helicity in helicities:
                actual = _real(
                    selector.evaluate(points, helicity_by_point=[helicity] * 33)
                )
                if helicity in zeros:
                    assert max(map(abs, actual)) <= 1.0e-12 * scale, helicity
                    continue
                expected = _real(
                    default.evaluate(points, helicity_by_point=[helicity] * 33)
                )
                assert actual == expected, (process_id, helicity)
                assert (
                    _real(selector.evaluate(points, helicities=[helicity])) == expected
                )
            mixed = [generator.choice(helicities) for _ in points]
            actual = _real(selector.evaluate(points, helicity_by_point=mixed))
            expected = _real(default.evaluate(points, helicity_by_point=mixed))
            for helicity, value, reference in zip(mixed, actual, expected, strict=True):
                if helicity in zeros:
                    assert abs(value) <= 1.0e-12 * scale
                else:
                    assert value == reference, (process_id, helicity)


def test_apis_needing_the_main_schedule_are_refused(
    artifacts: dict[str, Path],
) -> None:
    artifact = artifacts["selector-u1"]
    process_id = _selector_only_ids(artifact)[0]
    runtime = Runtime.load(artifact, process=process_id)
    helicities = list(runtime.physics.helicity_ids)
    points = _points(runtime.physics.process, 3)
    with pytest.raises(CompatibilityError, match="helicity-summed evaluation"):
        runtime.evaluate(points)
    with pytest.raises(CompatibilityError, match="helicity-summed evaluation"):
        runtime.evaluate(points, helicities=helicities[:2])
    with pytest.raises(CompatibilityError, match="resolved helicity/colour output"):
        runtime.evaluate_resolved(points, helicities=helicities[:1])
    # Colour selectors stop at the NLC/full selector check, as in default mode.
    with pytest.raises(EvaluationError, match="LC color-flow selection is unavailable"):
        runtime.evaluate(
            points, helicities=helicities[:1], color_flows=runtime.physics.color_ids[:1]
        )
    # Selector-only processes are never union templates in S1 (section 6).
    for helicity_by_point in (None, helicities[:1] * len(points)):
        with pytest.raises(CompatibilityError, match="flavour-union assignments"):
            runtime.evaluate_assignments(
                points, [[0]] * len(points), helicity_by_point=helicity_by_point
            )
    # Native profiles (backend API): single helicity only.
    backend = runtime._backend  # type: ignore[attr-defined]
    with pytest.raises(CompatibilityError, match="helicity-summed evaluation"):
        backend.profile(points)
    with pytest.raises(CompatibilityError, match="arena profiling"):
        backend._profile_arena_repeated(points, 1, helicities=helicities[:1])
    profile = backend.profile(points, helicities=helicities[:1], include_values=True)
    assert [complex(v).real for v in profile["values"]] == _real(
        runtime.evaluate(points, helicities=helicities[:1])
    )
    # Supported: one helicity, globally, per point, or as order partials.
    single = _real(runtime.evaluate(points, helicities=helicities[:1]))
    orders = runtime.evaluate_orders(points, helicities=helicities[:1])
    assert _real(orders.total) == single
    if any(
        _SELECTOR_ONLY
        not in _execution(artifact, str(p["id"]))["required_runtime_capabilities"]
        for p in _manifest(artifact)["processes"]
    ):
        template = next(
            str(p["id"])
            for p in _manifest(artifact)["processes"]
            if _execution(artifact, str(p["id"]))["plan"] is not None
        )
        assert Runtime.load(artifact, process=template).flavour_union_assignments


def test_model_parameters_reach_the_selector_lane(
    artifacts: dict[str, Path],
) -> None:
    selector_artifact = artifacts["selector-u0"]
    process_id = _selector_only_ids(selector_artifact)[0]
    runtimes = [
        Runtime.load(artifact, process=process_id)
        for artifact in (selector_artifact, artifacts["default-u0"])
    ]
    points = _points(runtimes[0].physics.process, 5)
    helicity = next(
        helicity.id
        for helicity in runtimes[1].physics.helicities
        if not helicity.structural_zero
    )
    before = [_real(r.evaluate(points, helicities=[helicity])) for r in runtimes]
    physics = json.loads(
        (selector_artifact / "processes" / process_id / "physics.json").read_text(
            "utf-8"
        )
    )
    name = next(
        str(parameter["name"])
        for parameter in physics["model_parameters"]
        if "alpha_s" in str(parameter["name"])
    )
    for runtime in runtimes:
        runtime.set_model_parameters({name: 0.2})
    after = [_real(r.evaluate(points, helicities=[helicity])) for r in runtimes]
    assert before[0] == before[1]
    assert after[0] == after[1]


def test_generation_validation_checks_selector_only_processes(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    output = tmp_path / "validated"
    Generator(_config(unions=False, selector_only=True, validate=True)).generate(
        ProcessSet.from_expressions(("u u~ > g g g",)), output
    )
    assert _selector_only_ids(output)


@pytest.mark.parametrize(
    ("expression", "factorization"),
    [("g g > g g g", "klein-four-walsh"), ("u u~ > g g g", None)],
)
def test_forced_direct_contraction_keeps_the_reducer_bitwise(
    tmp_path: Path, expression: str, factorization: str | None
) -> None:
    """color.contraction=direct: both modes use the same Walsh or direct block."""
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    paths = {}
    for selector_only in (False, True):
        paths[selector_only] = tmp_path / f"direct-{int(selector_only)}"
        Generator(
            _config(unions=False, selector_only=selector_only, contraction="direct")
        ).generate(ProcessSet.from_expressions((expression,)), paths[selector_only])
    (process_id,) = _selector_only_ids(paths[True])
    reducer = _reducer(paths[True], process_id)
    assert reducer == _reducer(paths[False], process_id) == ("repeated", factorization)
    selector = Runtime.load(paths[True], process=process_id)
    default = Runtime.load(paths[False], process=process_id)
    # Default-mode structural zeros are computed (roundoff) in selector-only mode.
    helicities = [h.id for h in default.physics.helicities if not h.structural_zero]
    points = _points(selector.physics.process, 33)
    for helicity in helicities:
        expected = _real(default.evaluate(points, helicity_by_point=[helicity] * 33))
        assert (
            _real(selector.evaluate(points, helicity_by_point=[helicity] * 33))
            == expected
        ), helicity
    mixed = [helicities[(7 * index + 3) % len(helicities)] for index in range(33)]
    assert _real(selector.evaluate(points, helicity_by_point=mixed)) == _real(
        default.evaluate(points, helicity_by_point=mixed)
    )


def test_mixed_order_partials_are_refused(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    output = tmp_path / "mixed-orders"
    Generator(_config(unions=False, selector_only=True, qed=2)).generate(
        ProcessSet.from_expressions(("u d > u d",)), output
    )
    process_id = _selector_only_ids(output)[0]
    runtime = Runtime.load(output, process=process_id)
    helicities = list(runtime.physics.helicity_ids)
    points = _points(runtime.physics.process, 3)
    single = _real(runtime.evaluate(points, helicities=helicities[:1]))
    assert any(single)
    with pytest.raises(CompatibilityError, match="amplitude-order partials"):
        runtime.evaluate_orders(points, helicities=helicities[:1])


# C probe: per-helicity, per-point and mixed helicities via the C ABI, raw doubles out.
_C_PROBE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "rusticol.h"

static int write_values(FILE *out, const double *values, size_t count) {
    return fwrite(values, sizeof(double), count, out) == count ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const size_t point_count = strtoul(argv[4], NULL, 10);
    FILE *input = fopen(argv[3], "rb");
    if (input == NULL) return 3;
    fseek(input, 0, SEEK_END);
    const size_t momentum_count = (size_t)ftell(input) / sizeof(double);
    fseek(input, 0, SEEK_SET);
    double *momenta = malloc(momentum_count * sizeof(double));
    double *values = malloc(point_count * sizeof(double));
    uint32_t *by_point = malloc(point_count * sizeof(uint32_t));
    if (!momenta || !values || !by_point) return 3;
    if (fread(momenta, sizeof(double), momentum_count, input) != momentum_count)
        return 3;
    fclose(input);
    RusticolRuntimeHandle *runtime = NULL;
    if (rusticol_runtime_load(argv[1], argv[2], NULL, &runtime) != RUSTICOL_STATUS_OK)
        return 10;
    size_t helicity_count = 0;
    if (rusticol_runtime_helicity_count(runtime, &helicity_count) != RUSTICOL_STATUS_OK)
        return 11;
    FILE *out = fopen(argv[5], "wb");
    if (out == NULL) return 3;
    for (size_t helicity = 0; helicity < helicity_count; ++helicity) {
        char id[512];
        size_t required = 0;
        if (rusticol_runtime_helicity_id(runtime, helicity, id, sizeof id, &required)
            != RUSTICOL_STATUS_OK) return 12;
        const char *ids[] = {id};
        if (rusticol_runtime_evaluate_selected_f64(
                runtime, momenta, momentum_count, point_count, ids, 1, NULL, 0,
                NULL, 0, NULL, 0, values, point_count) != RUSTICOL_STATUS_OK) return 13;
        if (write_values(out, values, point_count)) return 3;
        for (size_t point = 0; point < point_count; ++point)
            by_point[point] = (uint32_t)helicity;
        if (rusticol_runtime_evaluate_selected_f64(
                runtime, momenta, momentum_count, point_count, NULL, 0, NULL, 0,
                by_point, point_count, NULL, 0, values, point_count)
            != RUSTICOL_STATUS_OK) return 14;
        if (write_values(out, values, point_count)) return 3;
    }
    for (size_t point = 0; point < point_count; ++point)
        by_point[point] = (uint32_t)((point * 7 + 3) % helicity_count);
    if (rusticol_runtime_evaluate_selected_f64(
            runtime, momenta, momentum_count, point_count, NULL, 0, NULL, 0,
            by_point, point_count, NULL, 0, values, point_count)
        != RUSTICOL_STATUS_OK) return 15;
    if (write_values(out, values, point_count)) return 3;
    fclose(out);
    /* Helicity sums need the omitted main schedule. */
    if (rusticol_runtime_evaluate_f64(runtime, momenta, momentum_count, point_count,
                                      values,
                                      point_count) == RUSTICOL_STATUS_OK) return 16;
    if (rusticol_runtime_free(runtime) != RUSTICOL_STATUS_OK) return 17;
    puts("ok");
    return 0;
}
"""


def test_c_abi_values_equal_python_bitwise(
    artifacts: dict[str, Path],
    native_sdk: object,  # noqa: F811
    tmp_path: Path,
) -> None:
    import struct
    import subprocess

    from tests.integration.test_runtime_selector_api_parity import (
        _compile_probe,
        _source_environment,
    )

    sdk = native_sdk
    binary = _compile_probe("c", _C_PROBE, sdk, tmp_path)  # type: ignore[arg-type]
    artifact = artifacts["selector-u0"]
    count = 255  # odd: the last tile ends on an unpaired point
    for process_id in _selector_only_ids(artifact)[:3]:
        runtime = Runtime.load(artifact, process=process_id)
        helicities = list(runtime.physics.helicity_ids)
        points = _points(runtime.physics.process, count)
        momenta = tmp_path / "momenta.bin"
        momenta.write_bytes(
            struct.pack(
                f"{len(points) * len(points[0]) * 4}d",
                *(x for point in points for leg in point for x in leg),
            )
        )
        values = tmp_path / "values.bin"
        completed = subprocess.run(
            [
                str(binary),
                str(artifact),
                process_id,
                str(momenta),
                str(count),
                str(values),
            ],
            env=_source_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        raw = values.read_bytes()
        native = list(struct.unpack(f"{len(raw) // 8}d", raw))
        rows = [native[i : i + count] for i in range(0, len(native), count)]
        assert len(rows) == 2 * len(helicities) + 1
        for index, helicity in enumerate(helicities):
            expected = _real(runtime.evaluate(points, helicities=[helicity]))
            assert rows[2 * index] == expected, (process_id, helicity)
            by_point = _real(
                runtime.evaluate(points, helicity_by_point=[helicity] * count)
            )
            assert rows[2 * index + 1] == by_point, (process_id, helicity)
        mixed = [
            helicities[(point * 7 + 3) % len(helicities)] for point in range(count)
        ]
        assert rows[-1] == _real(runtime.evaluate(points, helicity_by_point=mixed))
