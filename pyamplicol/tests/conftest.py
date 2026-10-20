# SPDX-License-Identifier: 0BSD
"""Shared inputs for the one-loop tests.

Loop modules carry one of the ``loop_fast``, ``loop_physics`` or
``loop_nightly`` markers. External inputs (UFO models, MadGraph source and
saved PACBIN/artifact controls) resolve here once: an environment variable
overrides the default location, and a missing input skips with its name.
Committed provider references under ``tests/fixtures/loop`` are mandatory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
LOOP_REFERENCES = TESTS / "fixtures" / "loop"
LOOP_MARKERS = ("loop_fast", "loop_physics", "loop_nightly")
DEFERRED_MARKER = "deferred_direct_r2"

_MAIN_CHECKOUT = Path("/shared/external/pyLoopAmpliCol/pyamplicol")
_WORKSPACE = Path(
    os.environ.get(
        "PYAMPLICOL_LOOP_WORKSPACE", "/shared/external/pyLoopAmpliCol/workspace"
    )
)
_MG5 = Path("FOR_REFERENCE_ONLY/MG5_aMC_v3_7_0")

# fixture name -> (overriding environment variables, default candidates)
_EXTERNAL_INPUTS: dict[str, tuple[tuple[str, ...], tuple[Path, ...]]] = {
    "loop_sm_ufo": (
        ("PYAMPLICOL_NLO_UFO_REFERENCE",),
        (
            TESTS.parent / _MG5 / "models/loop_sm",
            _MAIN_CHECKOUT / _MG5 / "models/loop_sm",
            _WORKSPACE / "reference/sources/MG5_aMC_v3_7_0/models/loop_sm",
        ),
    ),
    "loop_gmu_ufo": (
        ("PYAMPLICOL_EW_UFO_REFERENCE", "PYAMPLICOL_EW_UFO"),
        (_WORKSPACE / "reference/models/loop_qcd_qed_sm_Gmu_py3",),
    ),
    "loop_qcd_qed_sm_ufo": (
        ("PYAMPLICOL_EW_NLO_UFO_REFERENCE",),
        (_WORKSPACE / "reference/models/loop_qcd_qed_sm",),
    ),
    "sgluon_nlo_ufo": (
        ("PYAMPLICOL_SGLUON_UFO_REFERENCE",),
        (_WORKSPACE / "reference/models/bsm-candidates/sgluons_NLO_py3",),
    ),
    "madgraph_source": (
        ("PYAMPLICOL_MADGRAPH_SOURCE",),
        (_WORKSPACE / "reference/sources/MG5_aMC_v3_7_0",),
    ),
    "saved_loop_fixtures": (
        ("PYAMPLICOL_LOOP_SAVED_FIXTURES",),
        (_WORKSPACE / "fixtures",),
    ),
}


def _external_input(name: str) -> Path:
    variables, candidates = _EXTERNAL_INPUTS[name]
    for variable in variables:
        if configured := os.environ.get(variable):
            path = Path(configured)
            if not path.exists():
                pytest.fail(f"{variable}={configured} does not exist")
            return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip(f"{name} unavailable; set {variables[0]}")


def _input_fixture(name: str):
    def fixture() -> Path:
        return _external_input(name)

    fixture.__name__ = name
    fixture.__doc__ = f"Path of the external {name} input."
    return pytest.fixture(scope="session", name=name)(fixture)


for _name in _EXTERNAL_INPUTS:
    globals()[_name] = _input_fixture(_name)


@pytest.fixture(scope="session")
def loop_sm_compiled_model(loop_sm_ufo, tmp_path_factory):
    """loop_sm compiled once per session with the checked scalar-arity view."""

    from dataclasses import replace

    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.loading import compile_model_source

    root = tmp_path_factory.mktemp("loop-sm-model")
    model = compile_model_source(loop_sm_ufo, cache_dir=root / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / "loop-sm.json")


@pytest.fixture(scope="session")
def loop_sm_colour_model(loop_sm_ufo, tmp_path_factory):
    """loop_sm with the committed explicit QCD loop/colour declarations."""

    import importlib.util

    script = TESTS.parent / "examples/loop/source_colour/prepare_model.py"
    spec = importlib.util.spec_from_file_location("_loop_colour_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("loop-sm-colour-model")
    return module.prepare(loop_sm_ufo, root / "loop-sm-colour.json")


@pytest.fixture(scope="session")
def loop_native():
    """The loop core registered in the staged ``pyamplicol._rusticol``."""

    from pyamplicol.api.errors import DependencyError
    from pyamplicol.loops.native import load_loop_native

    try:
        return load_loop_native()
    except DependencyError as error:
        pytest.skip(f"staged native loop runtime unavailable: {error}")


@pytest.fixture(scope="session")
def loop_reference_path():
    """Path of a committed provider reference; a missing file is a failure."""

    def resolve(name: str) -> Path:
        path = LOOP_REFERENCES / f"{name}.json"
        if not path.is_file():
            pytest.fail(f"committed loop reference {path} is missing")
        return path

    return resolve


@pytest.fixture(scope="session")
def loop_reference(loop_reference_path):
    """Load a committed provider reference from ``tests/fixtures/loop``."""

    return lambda name: json.loads(loop_reference_path(name).read_text())


@pytest.fixture(scope="session")
def saved_loop_fixture(saved_loop_fixtures):
    """Resolve one saved PACBIN/artifact control below the fixture root."""

    def resolve(relative: str) -> Path:
        path = saved_loop_fixtures / relative
        if not path.exists():
            pytest.fail(f"saved loop fixture {path} is missing")
        return path

    return resolve


def pytest_collection_modifyitems(config, items):
    expression = config.getoption("markexpr", "") or ""
    if DEFERRED_MARKER not in expression:
        deferred = [item for item in items if DEFERRED_MARKER in item.keywords]
        if deferred:
            config.hook.pytest_deselected(items=deferred)
            items[:] = [item for item in items if DEFERRED_MARKER not in item.keywords]
    if os.environ.get("SYMBOLICA_LICENSE"):
        return
    # An unlicensed Symbolica aborts the interpreter on this class of host, so
    # loop tests are skipped before any of them can touch it.
    skip = pytest.mark.skip(reason="loop tests require SYMBOLICA_LICENSE")
    for item in items:
        if any(marker in item.keywords for marker in LOOP_MARKERS):
            item.add_marker(skip)
