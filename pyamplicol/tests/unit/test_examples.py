# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ast
import json
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from pyamplicol.config import FIELD_REGISTRY, ProcessEntry, resolve_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
RUN_CARDS = tuple(sorted(EXAMPLES.glob("*.toml")))


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _schema_leaf_paths(
    payload: Mapping[str, object],
    *,
    prefix: str = "",
) -> set[str]:
    paths: set[str] = set()
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if path in FIELD_REGISTRY:
            paths.add(path)
        elif isinstance(value, Mapping):
            paths.update(_schema_leaf_paths(value, prefix=path))
        else:
            paths.add(path)
    return paths


@pytest.mark.parametrize("card", RUN_CARDS, ids=lambda path: path.stem)
def test_run_cards_parse_and_resolve_without_domain_backends(card: Path) -> None:
    payload = _read_toml(card)
    resolution = resolve_config(card)
    config = resolution.effective

    assert payload["schema_version"] == 1
    assert config.schema_version == 1
    assert config.action == payload["action"]
    assert not resolution.was_clamped

    if config.action == "generate":
        assert config.process.entries
        assert config.generation.output is not None
    elif config.action == "evaluate":
        assert config.evaluation.artifact is not None
        assert config.evaluation.momenta is not None
    elif config.action == "benchmark":
        assert config.evaluation.artifact is not None


def test_all_options_is_an_exhaustive_commented_registry_snapshot() -> None:
    card = EXAMPLES / "all_options.toml"
    payload = _read_toml(card)
    text = card.read_text(encoding="utf-8")

    assert _schema_leaf_paths(payload) == set(FIELD_REGISTRY)
    for path in FIELD_REGISTRY:
        assert f"# {path}:" in text, f"missing all-options comment for {path}"


def test_example_matrix_covers_required_models_and_modes() -> None:
    color_cards = {
        "builtin_sm_lc.toml": "lc",
        "builtin_sm_nlc.toml": "nlc",
        "builtin_sm_full.toml": "full",
    }
    for name, accuracy in color_cards.items():
        payload = _read_toml(EXAMPLES / name)
        assert payload["model"] == {"source": "built-in-sm"}
        assert payload["color"]["accuracy"] == accuracy  # type: ignore[index]

    process_set = resolve_config(EXAMPLES / "process_set_mixed_multiplicity.toml")
    assert process_set.effective.process.entries == (
        ProcessEntry("d d~ > z g", "ddbar_zg"),
        ProcessEntry("d d~ > z g g", "ddbar_zgg"),
    )

    external_sources = {
        "external_ufo_sm.toml": "models/ufo/sm",
        "external_json_sm.toml": "models/json/sm/sm.json",
        "external_json_scalars.toml": "models/json/scalars/scalars.json",
        "external_json_scalar_gravity.toml": (
            "models/json/scalar_gravity/scalar_gravity.json"
        ),
    }
    for name, expected in external_sources.items():
        payload = _read_toml(EXAMPLES / name)
        source = payload["model"]["source"]  # type: ignore[index]
        assert source == expected
        assert not Path(str(source)).is_absolute()
        assert ".." not in Path(str(source)).parts

    total = resolve_config(EXAMPLES / "evaluate_total.toml").effective
    resolved = resolve_config(EXAMPLES / "evaluate_resolved.toml").effective
    assert not total.evaluation.resolved
    assert resolved.evaluation.resolved
    assert resolve_config(EXAMPLES / "benchmark.toml").effective.action == "benchmark"


def test_example_data_has_finite_momenta_and_scalar_parameters() -> None:
    momenta = json.loads(
        (EXAMPLES / "data/ddbar_zg_momenta.json").read_text(encoding="utf-8")
    )
    assert isinstance(momenta, list) and momenta
    for point in momenta:
        assert isinstance(point, list) and len(point) == 4
        for particle in point:
            assert isinstance(particle, list) and len(particle) == 4
            assert all(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(component)
                for component in particle
            )

    parameters = json.loads(
        (EXAMPLES / "data/model_parameters.json").read_text(encoding="utf-8")
    )
    assert set(parameters) == {
        "normalization.alpha_ew",
        "normalization.alpha_s_me_check",
    }
    assert all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in parameters.values()
    )


def test_example_source_is_syntax_valid_public_and_spdx_marked() -> None:
    source_files = [
        *EXAMPLES.glob("*.toml"),
        *EXAMPLES.glob("*.md"),
        *EXAMPLES.glob("python/*.py"),
        *EXAMPLES.glob("native/*.cpp"),
        *EXAMPLES.glob("native/*.f90"),
        EXAMPLES / "native/Makefile",
    ]
    for source in source_files:
        first_line = source.read_text(encoding="utf-8").splitlines()[0]
        assert "SPDX-License-Identifier: 0BSD" in first_line, source

    for source in EXAMPLES.glob("python/*.py"):
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        compile(tree, str(source), "exec")
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(name.startswith("pyamplicol._") for name in imported)


def test_native_examples_use_installed_sdk_discovery_and_public_wrappers() -> None:
    makefile = (EXAMPLES / "native/Makefile").read_text(encoding="utf-8")
    cpp = (EXAMPLES / "native/runtime.cpp").read_text(encoding="utf-8")
    fortran = (EXAMPLES / "native/runtime.f90").read_text(encoding="utf-8")

    for option in ("--cflags", "--libs", "--fortran-source"):
        assert f"$(RUSTICOL_CONFIG) {option}" in makefile
    assert "CXX = c++" in makefile
    assert "FC = gfortran" in makefile
    assert "$(RUSTICOL_CONFIG) --library" not in makefile
    assert "rusticol::Runtime" in cpp
    assert ".evaluate_resolved(" in cpp
    assert "set_model_parameters_json" in cpp
    assert 'set_model_parameter("normalization.alpha_s_me_check"' in cpp
    assert "use rusticol, only: rusticol_runtime" in fortran
    assert "runtime%evaluate_resolved" in fortran
    assert "model_parameters=trim(parameters)" in fortran
    assert "runtime%set_model_parameter" in fortran
    assert '"normalization.alpha_s_me_check"' in fortran
    assert '"alpha_s"' not in cpp
    assert '"alpha_s"' not in fortran

    combined = "\n".join((makefile, cpp, fortran))
    assert "src/pyamplicol" not in combined
    assert str(ROOT) not in combined


def test_readme_states_current_release_boundary_and_available_utilities() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs/user/release-status.md").read_text(encoding="utf-8")
    assert "pyamplicol==0.1.0` is not published yet" in readme
    assert "schema-v3 generation" in readme
    assert "root `API/` bundle" in readme
    assert "Python-to-Rusticol runtime adapter" in readme
    for command in ("examples", "config", "doctor", "self-test"):
        assert command in status
        assert command in readme
    assert "Transactional schema-v3 generation" in status
    assert "lazy Python-to-Rusticol adapter" in status
    assert "generated root `API/` bundle is not emitted" not in status
    assert "GenerationBackend.generate()` raises" not in status
    assert "artifact writing is not integrated" not in readme
