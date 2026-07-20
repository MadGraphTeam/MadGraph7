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
        assert payload["process"]["entries"][0]["expression"] == "u u~ > g g"  # type: ignore[index]

    eager = resolve_config(EXAMPLES / "builtin_sm_eager.toml").effective
    assert eager.model.source == "built-in-sm"
    assert eager.evaluator.execution_mode.value == "eager"
    assert eager.evaluator.backend.value == "jit"
    assert eager.evaluator.jit.optimization_level == 3

    process_set = resolve_config(EXAMPLES / "process_set_mixed_multiplicity.toml")
    assert process_set.effective.process.entries == (
        ProcessEntry("u u~ > Z g", "uubar_Zg"),
        ProcessEntry("u u~ > Z g g", "uubar_Zgg"),
    )

    primary = resolve_config(EXAMPLES / "generate_pp_zjj_from_ufo_sm.toml").effective
    assert primary.process.entries == (ProcessEntry("p p > Z j j"),)
    assert primary.process.multiparticles == {
        "p": ("d", "d~", "g"),
        "j": ("d", "d~", "g"),
    }
    assert primary.process.flavor_scheme == 2
    assert primary.model.restriction == "default"
    assert primary.generation.output == EXAMPLES / "artifacts/pp_zjj"

    external_sources = {
        "external_ufo_sm.toml": "models/ufo/sm",
        "generate_pp_zjj_from_ufo_sm.toml": "models/json/sm/sm.json",
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
    assert total.evaluation.process == resolved.evaluation.process == "d d~ > z g g"
    assert resolve_config(EXAMPLES / "benchmark.toml").effective.action == "benchmark"


def test_z6g_benchmark_examples_encode_the_report_specializations() -> None:
    selected = resolve_config(
        EXAMPLES / "benchmark_z6g_single_flow_helicity_sum.toml"
    ).effective
    all_flows = resolve_config(
        EXAMPLES / "benchmark_z6g_all_flows_single_helicity.toml"
    ).effective

    expected_process = (ProcessEntry("u u~ > Z g g g g g g", "uubar_Z_6g"),)
    for config in (selected, all_flows):
        assert config.process.entries == expected_process
        assert config.model.source == "built-in-sm"
        assert config.color.accuracy.value == "lc"
        assert config.evaluator.execution_mode.value == "compiled"
        assert config.evaluator.backend.value == "jit"
        assert config.evaluator.jit.optimization_level == 3
        assert config.evaluator.batch_size == 64
        assert config.evaluator.output_chunk_size == 512
        assert config.benchmark.batch_size == 64
        assert config.benchmark.target_runtime == 20.0
        assert config.benchmark.warmup_runs == 2
        assert config.benchmark.minimum_samples == 5

    assert selected.process.reference_color_order == (2, 4, 5, 6, 7, 8, 9, 1, 3)
    assert selected.process.selected_color_sector_ids == (0,)
    assert selected.process.selected_source_helicities == {}

    assert all_flows.process.reference_color_order == ()
    assert all_flows.process.selected_color_sector_ids == ()
    assert all_flows.process.selected_source_helicities == {
        "1": -1,
        "2": 1,
        "3": -1,
        "4": 1,
        "5": -1,
        "6": 1,
        "7": -1,
        "8": 1,
        "9": -1,
    }


def test_example_data_has_finite_momenta_and_scalar_parameters() -> None:
    momenta = json.loads(
        (EXAMPLES / "data/pp_zjj_momenta.json").read_text(encoding="utf-8")
    )
    assert isinstance(momenta, list) and momenta
    for point in momenta:
        assert isinstance(point, list) and len(point) == 5
        for particle in point:
            assert isinstance(particle, list) and len(particle) == 4
            assert all(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(component)
                for component in particle
            )
        incoming = [
            sum(particle[index] for particle in point[:2]) for index in range(4)
        ]
        outgoing = [
            sum(particle[index] for particle in point[2:]) for index in range(4)
        ]
        assert incoming == pytest.approx(outgoing, rel=1e-13, abs=1e-13)

    parameters = json.loads(
        (EXAMPLES / "data/model_parameters.json").read_text(encoding="utf-8")
    )
    assert set(parameters) == {"aS", "MZ"}
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
    assert "CXX ?= c++" in makefile
    assert "FC ?= gfortran" in makefile
    assert "$(RUSTICOL_CONFIG) --library" not in makefile
    assert "rusticol::Runtime" in cpp
    assert ".evaluate_resolved(" in cpp
    assert "set_model_parameters_json" in cpp
    assert 'set_model_parameter("aS"' in cpp
    assert "p_p_to_z_j_j_4" in cpp
    assert "use rusticol, only: rusticol_runtime" in fortran
    assert "runtime%evaluate_resolved" in fortran
    assert "model_parameters=trim(parameters)" in fortran
    assert "runtime%set_model_parameter" in fortran
    assert '"aS"' in fortran
    assert "p_p_to_z_j_j_4" in fortran
    assert "momenta(20)" in fortran

    generated_rust = (
        ROOT / "src/pyamplicol/assets/api_templates/rust/check_standalone.rs"
    ).read_text(encoding="utf-8")
    generated_makefile = (
        ROOT / "src/pyamplicol/assets/api_templates/rust/Makefile"
    ).read_text(encoding="utf-8")
    assert 'include!(env!("RUSTICOL_RUST_SOURCE"))' in generated_rust
    assert "Runtime::load" in generated_rust
    assert "rusticol_runtime_" not in generated_rust
    assert "unsafe" not in generated_rust
    assert "the Rust Rusticol API supports only double precision" in generated_rust
    assert "$(RUSTICOL_CONFIG) --rustflags" in generated_makefile

    combined = "\n".join((makefile, cpp, fortran))
    assert "src/pyamplicol" not in combined
    assert str(ROOT) not in combined


def test_readme_states_current_release_boundary_and_available_utilities() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs/user/release-status.md").read_text(encoding="utf-8")
    assert "pyamplicol==0.1.0` is not published yet" in readme
    assert "p p > Z j j" in readme
    assert "p_p_to_z_j_j_4" in readme
    assert "rust/check_standalone.rs" in readme
    assert "aS" in readme and "MZ" in readme
    for command in ("examples", "config", "doctor", "self-test"):
        assert command in status
        assert command in readme
    assert "transactional schema-v3 generation" in status
    assert "dependency-free Rust 2021" in status


def test_user_docs_and_examples_exclude_retired_workflows() -> None:
    sources = [
        ROOT / "README.md",
        *sorted((ROOT / "docs/user").glob("*.md")),
        ROOT / "docs/development/PACKAGING_CONTRACT.md",
        ROOT / "docs/development/ARCHITECTURE_DECISIONS.md",
        *sorted(EXAMPLES.rglob("*.md")),
        *sorted(EXAMPLES.rglob("*.toml")),
        *sorted(EXAMPLES.rglob("*.py")),
        *sorted(EXAMPLES.rglob("*.cpp")),
        *sorted(EXAMPLES.rglob("*.f90")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for retired in (
        "ddbar_zg",
        "normalization.alpha_s_me_check",
        "CycloneDX",
        "SBOM",
        "RECORD signatures",
        "signed source tag",
        "parent repository",
        "parent workspace",
        "parent project",
    ):
        assert retired not in combined
