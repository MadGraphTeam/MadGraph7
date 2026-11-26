# SPDX-License-Identifier: 0BSD
"""Native checks of the UMAMI colour-accuracy ratio providers."""

from __future__ import annotations

import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.artifacts.umami_c import helicity_export
from pyamplicol.artifacts.umami_reweight import write_reweight_provider
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from tests.integration.test_umami_api import (
    _config_executable,
    _library,
    _runtime_point,
    _unavailable,
)

EXPRESSIONS = ("g g > g g", "d d~ > g g")
NAMES = ("gg", "dd_gg")


def _generate(artifact: Path, accuracy: str, *, selector_only: bool = False) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=True,
            selector_only_full_colour=selector_only,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            ProcessSet.from_expressions(EXPRESSIONS, names=NAMES),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )


@pytest.fixture(scope="module")
def reweight(tmp_path_factory):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp("umami-reweight")
    lc, full, output = root / "lc", root / "full", root / "reweight"
    _generate(lc, "lc")
    _generate(full, "full")
    assert write_reweight_provider(lc, full, output) == ["p0"]
    completed = subprocess.run(
        ["make", "-C", str(output), f"RUSTICOL_CONFIG={config_executable}"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads((lc / "API/umami/metadata.json").read_text())
    return lc, full, output, data, _library(output / "build/libreweight_p0.so")


def _ratio(library, handle, point, channel, flavour, helicity):
    n = len(point)
    momenta = (ct.c_double * (4 * n))(
        *(float(point[leg][component]) for component in range(4) for leg in range(n))
    )
    inputs = {
        0: momenta,
        2: (ct.c_int * 1)(flavour),
        6: (ct.c_int * 1)(helicity),
        8: (ct.c_uint * 1)(channel),
    }
    output = (ct.c_double * 1)(-1.0)
    status = library.umami_matrix_element(
        handle,
        1,
        1,
        0,
        len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(
            *(ct.cast(value, ct.c_void_p) for value in inputs.values())
        ),
        1,
        (ct.c_int * 1)(12),  # UMAMI_OUT_MATRIX_ELEMENT_RATIO
        (ct.c_void_p * 1)(ct.cast(output, ct.c_void_p)),
    )
    return status, output[0]


def test_ratio_matches_native_runtimes(reweight):
    lc, full, output, data, library = reweight
    handle = ct.c_void_p()
    assert library.umami_initialize(ct.byref(handle), os.fsencode(output)) == 0
    try:
        checked = 0
        for channel, description in enumerate(data["channels"]):
            for flavour, entry in enumerate(description["processes"]):
                runtime = entry["runtime"]
                process = data["runtime_processes"][runtime["process_index"]]
                lc_runtime = Runtime.load(lc, process=process["id"])
                full_runtime = Runtime.load(full, process=process["id"])
                runtime_point = _runtime_point_of(lc, process["id"])
                point = [None] * len(runtime_point)
                for runtime_slot, input_slot in enumerate(runtime["permutation"]):
                    point[input_slot] = runtime_point[runtime_slot]
                exports = helicity_export(data, runtime)
                for runtime_index, exported in enumerate(exports):
                    identifier = process["helicity_ids"][runtime_index]
                    reference = complex(
                        lc_runtime.evaluate((runtime_point,), helicities=[identifier])[0]
                    ).real
                    target = complex(
                        full_runtime.evaluate(
                            (runtime_point,), helicities=[identifier]
                        )[0]
                    ).real
                    status, value = _ratio(
                        library, handle, point, channel, flavour, exported
                    )
                    assert status == 0
                    expected = target / reference if reference != 0.0 else 0.0
                    assert value == pytest.approx(expected, rel=1e-10, abs=1e-15)
                    checked += reference != 0.0
        assert checked > 0
        assert _ratio(library, handle, point, 0, 0, len(data["helicities"]))[0] == 1
    finally:
        assert library.umami_free(handle) == 0


def test_batched_ratios_match_single_events(reweight):
    lc, _, output, data, library = reweight
    events = []
    for channel, description in enumerate(data["channels"]):
        for flavour, entry in enumerate(description["processes"]):
            runtime = entry["runtime"]
            process = data["runtime_processes"][runtime["process_index"]]
            runtime_point = _runtime_point_of(lc, process["id"])
            point = [None] * len(runtime_point)
            for runtime_slot, input_slot in enumerate(runtime["permutation"]):
                point[input_slot] = runtime_point[runtime_slot]
            # Ratios of structurally zero helicities are rounding noise.
            native = Runtime.load(lc, process=process["id"])
            total = complex(native.evaluate((runtime_point,))[0]).real
            for runtime_index, exported in enumerate(helicity_export(data, runtime)):
                value = complex(
                    native.evaluate(
                        (runtime_point,),
                        helicities=[process["helicity_ids"][runtime_index]],
                    )[0]
                ).real
                if abs(value) > 1e-12 * abs(total):
                    events.append((channel, flavour, exported, point))
    # Interleaved selections, more events than one native batch, behind an offset.
    events = [events[(i * 7) % len(events)] for i in range(600)]
    offset, n = 3, len(events[0][3])
    rows = [events[0]] * offset + events
    stride = len(rows)
    momenta = (ct.c_double * (4 * n * stride))(
        *(
            float(rows[event][3][leg][component])
            for component in range(4)
            for leg in range(n)
            for event in range(stride)
        )
    )
    inputs = {
        0: momenta,
        2: (ct.c_int * stride)(*(row[1] for row in rows)),
        6: (ct.c_int * stride)(*(row[2] for row in rows)),
        8: (ct.c_uint * stride)(*(row[0] for row in rows)),
    }
    ratios = (ct.c_double * stride)(*[-1.0] * stride)
    handle = ct.c_void_p()
    assert library.umami_initialize(ct.byref(handle), os.fsencode(output)) == 0
    try:
        status = library.umami_matrix_element(
            handle,
            len(events),
            stride,
            offset,
            len(inputs),
            (ct.c_int * len(inputs))(*inputs),
            (ct.c_void_p * len(inputs))(
                *(ct.cast(value, ct.c_void_p) for value in inputs.values())
            ),
            1,
            (ct.c_int * 1)(12),  # UMAMI_OUT_MATRIX_ELEMENT_RATIO
            (ct.c_void_p * 1)(ct.cast(ratios, ct.c_void_p)),
        )
        assert status == 0
        assert list(ratios[:offset]) == [-1.0] * offset
        for index, (channel, flavour, helicity, point) in enumerate(events):
            status, value = _ratio(library, handle, point, channel, flavour, helicity)
            assert status == 0
            assert ratios[offset + index] == pytest.approx(value, rel=1e-12, abs=1e-300)
    finally:
        assert library.umami_free(handle) == 0


@pytest.fixture(scope="module")
def selector_only_reweight(reweight, tmp_path_factory):
    """The provider with a selector-only full-colour target (MG7's emit_api setup)."""
    lc = reweight[0]
    root = tmp_path_factory.mktemp("umami-reweight-selector-only")
    full, output = root / "full", root / "reweight"
    _generate(full, "full", selector_only=True)
    assert write_reweight_provider(lc, full, output) == ["p0"]
    completed = subprocess.run(
        ["make", "-C", str(output), f"RUSTICOL_CONFIG={_config_executable()}"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return full, output, _library(output / "build/libreweight_p0.so")


def _single(runtime, point, helicity):
    return complex(runtime.evaluate((point,), helicities=[helicity])[0]).real


def test_selector_only_target_matches_the_default_target(
    reweight, selector_only_reweight
):
    lc, default_full, _, data, _ = reweight
    full, output, library = selector_only_reweight
    selected = [
        process["id"]
        for process in json.loads((full / "artifact.json").read_text())["processes"]
        if "rusticol.recurrence-selector-only-fc.v1"
        in json.loads(
            (full / "processes" / process["id"] / "execution.json").read_text()
        )["required_runtime_capabilities"]
    ]
    assert selected
    # Selector-only physics has no helicity aliases: every helicity is its own group.
    target = json.loads((full / "API/umami/metadata.json").read_text())
    assert target["helicity_groups"]
    assert all(
        len(group) == 1 for entry in target["helicity_groups"] for group in entry
    )
    handle = ct.c_void_p()
    assert library.umami_initialize(ct.byref(handle), os.fsencode(output)) == 0
    try:
        checked = 0
        for channel, description in enumerate(data["channels"]):
            for flavour, entry in enumerate(description["processes"]):
                runtime = entry["runtime"]
                process = data["runtime_processes"][runtime["process_index"]]
                lc_runtime = Runtime.load(lc, process=process["id"])
                full_runtime = Runtime.load(default_full, process=process["id"])
                runtime_point = _runtime_point_of(lc, process["id"])
                point = [None] * len(runtime_point)
                for runtime_slot, input_slot in enumerate(runtime["permutation"]):
                    point[input_slot] = runtime_point[runtime_slot]
                exports = helicity_export(data, runtime)
                for runtime_index, exported in enumerate(exports):
                    identifier = process["helicity_ids"][runtime_index]
                    reference = _single(lc_runtime, runtime_point, identifier)
                    expected = _single(full_runtime, runtime_point, identifier)
                    status, value = _ratio(
                        library, handle, point, channel, flavour, exported
                    )
                    assert status == 0
                    if reference == 0.0:
                        continue
                    assert value == pytest.approx(
                        expected / reference, rel=1e-10, abs=1e-15
                    )
                    checked += 1
        assert checked > 0
    finally:
        assert library.umami_free(handle) == 0


def _runtime_point_of(artifact, process_id):
    class _Artifact:
        pass

    provider = _Artifact()
    provider.artifact = artifact
    return _runtime_point(provider, process_id)


def test_reference_must_be_leading_colour(reweight, tmp_path):
    lc, full, _, _, _ = reweight
    with pytest.raises(ValueError, match="leading-colour"):
        write_reweight_provider(full, lc, tmp_path / "wrong")
