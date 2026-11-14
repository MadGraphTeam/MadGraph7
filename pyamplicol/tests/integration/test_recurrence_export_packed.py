# SPDX-License-Identifier: 0BSD
"""MadGraph7: the static recurrence export works on an artifact generated with
packed payloads (``generation.pack_payloads = true``).

One LC recurrence artifact of ``d d~ > z g`` is generated loose and one packed.
Both are exported through ``pyamplicol export-recurrence`` with reference
points. The two exports are equal except for ``artifact_id`` (the packed
artifact declares other payload files, so its content identity differs), and
the reference values equal a fresh runtime evaluation bit for bit.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.artifacts import load_manifest
from pyamplicol.cli import run_cli
from tests.integration.test_recurrence_generation_runtime import (
    _generation_config,
    _require_native_recurrence,
    _seeded_validation_points,
)

PROCESS = "d d~ > z g"
# Fields of the export that legitimately differ between a loose and a packed
# artifact of the same process.
ARTIFACT_IDENTITY_FIELDS = ("artifact_id",)


@pytest.fixture(scope="module")
def builtin_sm_recurrence_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    _require_native_recurrence()
    root = tmp_path_factory.mktemp("builtin-sm-recurrence-packed")
    return ModelSource.built_in_sm().compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=_generation_config("recurrence").evaluator,
    )


def _export(artifact: Path, points: Path, output: Path) -> dict[str, object]:
    errors = io.StringIO()
    status = run_cli(
        [
            "export-recurrence",
            str(artifact),
            "-o",
            str(output),
            "--reference-points",
            str(points),
            "--json",
        ],
        stdout=io.StringIO(),
        stderr=errors,
    )
    assert status == 0, errors.getvalue()
    return json.loads(output.read_text(encoding="utf-8"))


def test_recurrence_export_of_a_packed_artifact_equals_the_loose_one(
    tmp_path: Path,
    builtin_sm_recurrence_model: CompiledModel,
) -> None:
    config = _generation_config("recurrence")
    packed_config = replace(
        config, generation=replace(config.generation, pack_payloads=True)
    )
    loose, packed = tmp_path / "loose", tmp_path / "packed"
    Generator(config).generate(PROCESS, loose, model=builtin_sm_recurrence_model)
    Generator(packed_config).generate(
        PROCESS, packed, model=builtin_sm_recurrence_model
    )
    assert not (loose / "payloads.pacbin").exists()
    assert (packed / "payloads.pacbin").is_file()
    assert not any(
        path.name in {"physics.json", "execution.json"}
        for path in (packed / "processes").rglob("*")
    ), "the per-process records must live in payloads.pacbin"
    kept = os.environ.get("PYAMPLICOL_RECURRENCE_EXPORT_ARTIFACTS")
    if kept:
        # CI's Rust parity test interprets this export bit for bit as well.
        destination = Path(kept) / "packed"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(packed, destination)

    points = _seeded_validation_points(PROCESS, 17, 23)
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps([[list(leg) for leg in point] for point in points]),
        encoding="utf-8",
    )
    exports = {
        name: _export(artifact, points_file, tmp_path / f"{name}.json")
        for name, artifact in (("loose", loose), ("packed", packed))
    }

    for name, artifact in (("loose", loose), ("packed", packed)):
        assert exports[name]["artifact_id"] == load_manifest(artifact).artifact_id
    assert exports["loose"]["artifact_id"] != exports["packed"]["artifact_id"]
    stripped = [
        {
            key: value
            for key, value in export.items()
            if key not in ARTIFACT_IDENTITY_FIELDS
        }
        for export in exports.values()
    ]
    assert stripped[0] == stripped[1]
    programs = exports["packed"]["programs"]
    assert isinstance(programs, list) and programs

    # The reference values equal a fresh evaluation of the packed artifact, bit
    # for bit (two points: the batched path the export describes).
    runtime = Runtime.load(packed)
    expected = runtime.evaluate_resolved(list(points), precision=16)
    for program in programs:
        reference = program["reference_values"]
        assert reference["helicity_ids"] == list(expected.helicity_ids)
        color = expected.color_ids.index(program["colour_id"])
        values = [
            [complex(helicity[color]).real for helicity in point]
            for point in expected.values
        ]
        assert [[value.hex() for value in row] for row in reference["values"]] == [
            [value.hex() for value in row] for row in values
        ]
