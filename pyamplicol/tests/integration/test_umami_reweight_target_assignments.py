# SPDX-License-Identifier: 0BSD
"""Reweight providers check the flavour-union assignments of the target artifact.

A target runtime is used with the assignment keys of the provider's flavour rows
and switch targets; a key that is not the runtime's makes the load fail (J0).
"""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet
from pyamplicol.artifacts.umami_c import helicity_export
from pyamplicol.artifacts.umami_reweight import (
    load_umami_metadata,
    write_reweight_provider,
)
from pyamplicol.artifacts.umami_reweight_data import (
    decode_reweight_data,
    write_reweight_data,
)
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
    ProcessConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point
from tests.integration.test_umami_api import (
    _config_executable,
    _library,
    _unavailable,
)
from tests.integration.test_umami_flavor_switch import IN, OUT

EXPRESSIONS = (
    "g g > g g",
    "d d~ > g g",
    "g g > d d~",
    "d d~ > d d~",
    "d d~ > u u~",
    "u u~ > d d~",
    "u d > u d",
    "d d > d d",
)
NAMES = tuple(f"p{index}" for index in range(len(EXPRESSIONS)))
def _config(accuracy: str, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=True,
            flavor_unions=unions,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=True,
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


def _make(directory: Path, config_executable: str) -> None:
    completed = subprocess.run(
        ["make", "-C", str(directory), f"RUSTICOL_CONFIG={config_executable}"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    config_executable = _config_executable()
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp("reweight-targets")
    lc, full = root / "lc", root / "full-union"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for path, accuracy, unions in ((lc, "lc", False), (full, "full", True)):
            Generator(_config(accuracy, unions)).generate(
                ProcessSet.from_expressions(EXPRESSIONS, names=NAMES),
                path,
                model=ModelSource.from_path(prepared_model),
            )
    output = root / "reweight"
    assert write_reweight_provider(lc, full, output, switch_lines=2) == ["p0"]
    return root, lc, output, config_executable


def _corrupted(provider, block: str) -> Path:
    """A copy of the provider with every target assignment key of ``block``
    (the flavours or the switch targets) replaced; returns the copy."""

    root, _lc, output, config_executable = provider
    copy = root / f"reweight-{block}"
    shutil.copytree(output, copy, ignore=shutil.ignore_patterns("build"))
    tables = decode_reweight_data((copy / "reweight_p0.rwd").read_bytes())
    changed = 0
    if block == "flavours":
        for rows in tables.channels:
            for k, row in enumerate(rows):
                if row.target_assignment_key is not None:
                    rows[k] = replace(row, target_assignment_key="not-an-assignment")
                    changed += 1
    else:
        keys = tables.switch.targets.assignment_key
        for k, key in enumerate(keys):
            if key is not None:
                keys[k] = "not-an-assignment"
                changed += 1
    assert changed, f"no flavour-union target key in the {block} rows"
    write_reweight_data(copy, tables)
    _make(copy, config_executable)
    return copy


def _status(directory: Path, lc: Path, *, switch: bool) -> int:
    """Status of one ratio call with an event of every flavour of the provider,
    or of umami_initialize if that fails (it validates the first target)."""

    library = _library(directory / "build/libreweight_p0.so")
    handle = ct.c_void_p()
    status = library.umami_initialize(ct.byref(handle), os.fsencode(directory))
    if status != 0:
        return int(status)
    try:
        (data,) = load_umami_metadata(lc)
        point = [
            tuple(map(float, p.momentum)) for p in generic_validation_point("g g > g g")
        ]
        events = [
            (channel, flavour, helicity_export(data, entry["runtime"])[0])
            for channel, description in enumerate(data["channels"])
            for flavour, entry in enumerate(description["processes"])
        ]
        count, n = len(events), len(point)
        assert (
            library.umami_set_parameter(handle, b"flavor_switch", float(switch), 0.0)
            == 0
        )
        inputs = {
            IN["UMAMI_IN_MOMENTA"]: (ct.c_double * (4 * n * count))(
                *(point[leg][k] for k in range(4) for leg in range(n) for _ in events)
            ),
            IN["UMAMI_IN_FLAVOR_INDEX"]: (ct.c_int * count)(*(e[1] for e in events)),
            IN["UMAMI_IN_HELICITY_INDEX"]: (ct.c_int * count)(*(e[2] for e in events)),
            IN["UMAMI_IN_CHANNEL_INDEX"]: (ct.c_uint * count)(*(e[0] for e in events)),
            # columns PDG id + 6, without the top quark
            IN["UMAMI_IN_PARTON_DENSITIES"]: (ct.c_double * (26 * count))(
                *(
                    0.0 if column in (0, 12) else 1.0
                    for column in range(13)
                    for _ in range(2 * count)
                )
            ),
            # close to 1: switched events take a target flavour
            IN["UMAMI_IN_RANDOM_FLAVOR"]: (ct.c_double * count)(*[0.999] * count),
            IN["UMAMI_IN_RANDOM_HELICITY"]: (ct.c_double * count)(*[0.5] * count),
            IN["UMAMI_IN_RANDOM_COLOR"]: (ct.c_double * count)(*[0.5] * count),
        }
        names = ("UMAMI_OUT_MATRIX_ELEMENT_RATIO", "UMAMI_OUT_FLAVOR_INDEX")
        outputs = [(ct.c_double * count)(), (ct.c_int * count)()]
        status = library.umami_matrix_element(
            handle,
            count,
            count,
            0,
            len(inputs),
            (ct.c_int * len(inputs))(*inputs),
            (ct.c_void_p * len(inputs))(
                *(ct.cast(v, ct.c_void_p) for v in inputs.values())
            ),
            len(outputs),
            (ct.c_int * len(outputs))(*(OUT[name] for name in names)),
            (ct.c_void_p * len(outputs))(*(ct.cast(v, ct.c_void_p) for v in outputs)),
        )
        if status == 0 and switch:
            assert any(k >= 0 for k in outputs[1]), "no event was switched"
        return int(status)
    finally:
        assert library.umami_free(handle) == 0


def test_target_assignment_keys_are_those_of_the_target_runtime(provider):
    _root, lc, output, config_executable = provider
    tables = decode_reweight_data((output / "reweight_p0.rwd").read_bytes())
    keys = [row.target_assignment_key for rows in tables.channels for row in rows]
    assert "not-an-assignment" not in keys + tables.switch.targets.assignment_key
    _make(output, config_executable)
    assert _status(output, lc, switch=False) == 0
    assert _status(output, lc, switch=True) == 0


@pytest.mark.parametrize(("block", "switch"), [("flavours", False), ("switch", True)])
def test_a_corrupted_target_assignment_key_fails_the_load(provider, block, switch):
    _root, lc, _output, _config_executable = provider
    assert _status(_corrupted(provider, block), lc, switch=switch) != 0
