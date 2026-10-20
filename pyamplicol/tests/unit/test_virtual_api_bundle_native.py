# SPDX-License-Identifier: 0BSD
"""Opt-in emitted-client proof against the ordinary installed single archive."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.artifacts.api_bundle import emit_api_bundle, format_validation_points

pytestmark = pytest.mark.loop_fast


def test_real_writer_emits_virtual_bundle_without_tree_umami_provider(
    saved_loop_fixture, loop_sm_ufo, tmp_path
):
    # The saved mixed-order uudd artifact was produced from loop_sm/full.
    fixture = str(saved_loop_fixture("sdk-mixed-source"))
    source_path = str(loop_sm_ufo)
    from pyamplicol.api import ModelSource, Runtime
    from pyamplicol.config import resolve_config
    from pyamplicol.generation.artifact_writer import (
        _GenerationConfigProvenance,
        write_schema_v3_artifact,
    )
    from pyamplicol.generation.loop_artifact import (
        LoopExecutionMetadata,
        LoopProcessArtifact,
        LoopSchemeDeclaration,
    )
    from pyamplicol.generation.validation import ValidationPointRecord
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.physical_orders import BornSquaredSector, PhysicalOrderLayout
    from pyamplicol.models.loading import compile_model_source

    original = load_manifest(fixture)
    selected = original.processes[0]
    raw = json.loads((Path(fixture) / selected["physics_path"]).read_text())["metadata"]
    layout = raw.pop("orders")

    def order(value):
        return tuple((name, power) for name, power in value)

    raw["scheme"] = LoopSchemeDeclaration(**raw["scheme"])
    raw["orders"] = PhysicalOrderLayout(
        tuple(
            BornSquaredSector(
                order(s["order"]), tuple((order(a), order(b)) for a, b in s["pairs"])
            )
            for s in layout["born"]
        ),
        tuple(
            VirtualOrderSector(
                order(s["order"]),
                tuple(
                    BornLoopPair(order(p["born"]), order(p["loop"])) for p in s["pairs"]
                ),
            )
            for s in layout["virtual"]
        ),
        tuple(layout["correction_names"]),
    )
    for name in ("arithmetic", "available_r2_strategies", "epsilon_powers"):
        raw[name] = tuple(raw[name])
    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (500.0, 300.0, 0.0, 400.0),
        (500.0, -300.0, 0.0, -400.0),
    )
    process = LoopProcessArtifact(
        selected["id"],
        selected["expression"],
        tuple(selected["external_pdgs"]),
        Path(fixture) / "processes" / selected["id"] / "loop-runtime.pacbin",
        LoopExecutionMetadata(**raw),
        ValidationPointRecord(
            selected["id"],
            selected["expression"],
            0,
            tuple(zip(selected["external_pdgs"], point, strict=True)),
        ),
        {},
        color_accuracy=raw["color_accuracy"],
    )
    config = resolve_config(
        {
            "action": "generate",
            "process": {"loop_coupling_orders": ["QCD"]},
            "color": {"accuracy": raw["color_accuracy"], "contraction": "direct"},
            "generation": {"emit_api_bundle": True},
        }
    )
    # The retained program came from loop_sm/full with the explicit four-G0
    # scalar-arity declaration view. Recompile that SAME normalized model with
    # the current compiler; assert source identity, exact raw declarations and
    # defaults before attaching it. This does not regenerate graphs/numerators
    # and does not relabel the plan as the distinct packaged built-in SM.
    previous_model = json.loads(
        (Path(fixture) / "model/compiled-model.json").read_text()
    )
    compiled_model = compile_model_source(
        source_path, restriction="full", cache_dir=tmp_path / "model-cache"
    )
    assert compiled_model.raw_loop_model is not None
    compiled_model = replace(
        compiled_model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            compiled_model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    refreshed = compiled_model.to_dict()
    assert refreshed["source"] == previous_model["source"]
    assert refreshed["raw_loop_model"] == previous_model["raw_loop_model"]
    assert refreshed["parameter_defaults"] == previous_model["parameter_defaults"]
    output = tmp_path / "published"
    result = write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(source_path, restriction="full"),
        compiled_model=compiled_model,
        configuration=_GenerationConfigProvenance.from_config(config),
        processes=(process,),
        timings={"total": 0.0},
        staged_validation=lambda staged: Runtime.load(staged, process=selected["id"]),
    )
    assert result.api_bundle_path == "API"
    paths = {record.path for record in load_manifest(output).payloads}
    assert "API/umami/check_standalone.c" in paths
    assert "API/common/input.h" in paths
    assert not any(path.startswith("API/umami/provider") for path in paths)
    assert "API/umami/umami.c" not in paths
    assert "API/umami/umami_provider.h" not in paths


def test_installed_virtual_bundle_clients_preserve_mixed_order_resolved_axes(
    saved_loop_fixture, installed_virtual_bundle, tmp_path
):
    fixture = str(saved_loop_fixture("sdk-mixed-source"))
    installed = str(installed_virtual_bundle)
    make = shutil.which("make")
    assert make is not None, "the installed SDK client check requires make"
    artifact = tmp_path / "artifact"
    shutil.copytree(fixture, artifact)
    points = [
        [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400], [500, -300, 0, -400]],
        [[500, 0, 0, 500], [500, 0, 0, -500], [500, 400, 0, 300], [500, -400, 0, -300]],
    ]
    manifest = load_manifest(artifact)
    with ArtifactBuilder(artifact, mode="append") as builder:
        emit_api_bundle(builder, {"uudd": points[0]}, contribution_kind="virtual")
        # Repeated process records exercise two distinct rows in one native call.
        both = (
            format_validation_points({"uudd": points[0]})
            + format_validation_points({"uudd": points[1]}).split(b"\n", 1)[1]
        )
        builder.add_bytes(
            "API/validation_points.dat",
            both,
            role="validation-momenta",
            media_type="text/tab-separated-values",
        )
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            runtime={**manifest.runtime, "api_bundle_path": "API"},
            dependencies=manifest.dependencies,
            default_process_id=manifest.default_process_id,
            extensions=manifest.extensions,
        )
    environment = {**os.environ, "PYTHONPATH": installed}
    config = str(Path(installed) / "bin/rusticol-config")
    assert Path(config).is_file()
    python_result = subprocess.run(
        [
            sys.executable,
            "API/python/check_standalone.py",
            "uudd",
            "1000000",
            "--resolved",
        ],
        cwd=artifact,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = json.loads(python_result.stdout)
    assert expected["shape"][:2] == [2, 3]
    assert expected["born_shape"][:2] == [2, 3]
    assert expected["epsilon_powers"] == [0, -1, -2]

    def flatten(value):
        if isinstance(value, list):
            return [item for part in value for item in flatten(part)]
        return [float(value)]

    born, virtual = flatten(expected["born_values"]), flatten(expected["values"])
    for language in ("c", "cpp", "fortran", "rust", "umami"):
        directory = artifact / "API" / language
        build = tmp_path / "build" / language
        subprocess.run(
            [make, f"RUSTICOL_CONFIG={config}", f"BUILD_DIR={build}"],
            cwd=directory,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        completed = subprocess.run(
            [str(build / "check_standalone"), "uudd", "1000000", "resolved"],
            cwd=artifact,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        (tmp_path / f"{language}.log").write_text(completed.stdout + completed.stderr)
        lines = completed.stdout.splitlines()
        metadata = json.loads(next(line for line in lines if line.startswith("{")))
        assert len(metadata["plan"]["order_layout"]["virtual"]) == 3
        assert len(metadata["plan"]["order_layout"]["born"]) == 3
        if language == "fortran":
            actual_born = [
                float(line.split()[-1]) for line in lines if line.startswith("born ")
            ]
            actual_virtual = [
                float(line.split()[-1]) for line in lines if line.startswith("virtual ")
            ]
        elif language == "rust":
            actual_born = json.loads(
                next(line[5:] for line in lines if line.startswith("born "))
            )
            actual_virtual = json.loads(
                next(line[8:] for line in lines if line.startswith("virtual "))
            )
        else:
            actual_born = [
                float(x)
                for x in next(
                    line[5:] for line in lines if line.startswith("born ")
                ).split()
            ]
            actual_virtual = [
                float(x)
                for x in next(
                    line[8:] for line in lines if line.startswith("virtual ")
                ).split()
            ]
        assert actual_born == pytest.approx(born, rel=3e-11, abs=1e-13), language
        assert actual_virtual == pytest.approx(virtual, rel=3e-11, abs=1e-13), language
        if language == "umami":
            for p in range(2):
                for name in ("born", "virtual"):
                    prefix = f"umami {name} point={p} "
                    actual = [
                        float(x)
                        for x in next(
                            line[len(prefix) :]
                            for line in lines
                            if line.startswith(prefix)
                        ).split()
                    ]
                    if name == "born":
                        source = expected["born_values"][p]
                        columns = [
                            source[o][h][c]
                            for c in range(len(source[0][0]))
                            for h in range(len(source[0]))
                            for o in range(len(source))
                        ]
                    else:
                        source = expected["values"][p]
                        columns = [
                            source[o][h][c][e]
                            for e in range(3)
                            for c in range(len(source[0][0]))
                            for h in range(len(source[0]))
                            for o in range(len(source))
                        ]
                    assert actual == pytest.approx(columns, rel=3e-11, abs=1e-13)
