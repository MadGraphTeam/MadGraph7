# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "legacy_amplicol.py"
REPORT = ROOT / "tests" / "fixtures" / "reference" / "legacy-fortran-v1.json"
PHYSICS = ROOT / "tests" / "fixtures" / "reference" / "physics-v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("legacy_amplicol_oracle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_process_file_parser_preserves_fortran_external_order(tmp_path: Path) -> None:
    module = _module()
    process_file = tmp_path / "processes.txt"
    process_file.write_text(
        """4 1
1 -1 21 23


1

1   1   1   1 4 2 3
1   1   1 -1 21 23   2 3 1 4   1.0
""",
        encoding="utf-8",
    )

    entries = module.parse_process_file(process_file)

    assert entries == (
        module.ProcessEntry(
            group=1,
            integral=1,
            process_pdgs=(1, -1, 21, 23),
            color_order=(2, 3, 1, 4),
        ),
    )
    assert module.select_process_entry(entries, "d d~ > z g") == entries[0]
    assert module._permutation((1, -1, 23, 21), entries[0].process_pdgs) == (
        0,
        1,
        3,
        2,
    )


def test_process_selection_prefers_exact_external_order() -> None:
    module = _module()
    entries = (
        module.ProcessEntry(1, 1, (1, -1, 21, 23), (1, 2, 3, 4)),
        module.ProcessEntry(2, 1, (1, -1, 23, 21), (1, 2, 4, 3)),
    )

    assert module.select_process_entry(entries, "d d~ > z g") == entries[1]
    assert module.matching_process_entries(entries, "d d~ > z g") == (
        entries[1],
        entries[0],
    )


def test_color_probe_runs_outside_the_legacy_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "legacy"
    repository.mkdir()
    process_file = tmp_path / "processes.txt"
    process_file.write_text("fixture\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, capture=True):
        observed.update(command=tuple(command), cwd=cwd, capture=capture)
        output = """AMPICOL_COLOR_PROBE_CURRENTS 1
AMPICOL_COLOR_PROBE_VERTICES 1
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 1.0 1.0 1.0
AMPICOL_COLOR_PROBE_VALUE lc 1 1 1.0
"""
        return module.subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module, "_run", fake_run)
    entry = module.ProcessEntry(1, 1, (1, -1, 23), (1, 2, 3))

    result = module.run_color_probe(
        repository,
        process_file=process_file,
        entry=entry,
        source_pdgs=(1, -1, 23),
        momenta=(
            (50.0, 0.0, 0.0, 50.0),
            (50.0, 0.0, 0.0, -50.0),
            (100.0, 0.0, 0.0, 0.0),
        ),
        color_accuracy="lc",
    )

    assert result.value == 1.0
    assert observed["cwd"] != repository
    assert observed["command"][0] == str(
        (repository / "amplicol_color_probe").resolve()
    )


def test_probe_output_parser_records_values_and_topology() -> None:
    module = _module()
    result = module._parse_probe_output(
        """AMPICOL_COLOR_PROBE_CURRENTS 7
AMPICOL_COLOR_PROBE_VERTICES 4
AMPICOL_COLOR_PROBE_AMPLITUDES 1
AMPICOL_COLOR_PROBE_COLOR_ORDERS 1
AMPICOL_COLOR_PROBE_COMPONENTS 4.0E-1 3.5E-1 3.4E-1
AMPICOL_COLOR_PROBE_VALUE full 1 1 3.4E-1
"""
    )

    assert result.value == pytest.approx(0.34)
    assert result.components == pytest.approx((0.4, 0.35, 0.34))
    assert result.currents == 7
    assert result.vertices == 4
    assert result.amplitudes == 1
    assert result.color_orders == 1


def test_tracked_fortran_report_is_pinned_and_matches_physics_fixture() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    with (ROOT / "dependencies" / "release-lock.toml").open("rb") as stream:
        lock = tomllib.load(stream)

    assert report["schema_version"] == 1
    assert report["oracle"] == "legacy-fortran-amplicol-color-probe"
    assert report["revision"] == lock["legacy_amplicol"]["revision"]
    assert set(report["cases"]) == {
        "builtin_sm_ddbar_z_lc",
        "builtin_sm_ddbar_z_nlc",
        "builtin_sm_ddbar_z_full",
        "builtin_sm_ddbar_zg_lc",
        "builtin_sm_ddbar_zg_nlc",
        "builtin_sm_ddbar_zg_full",
    }
    for name, observation in report["cases"].items():
        assert observation["total"] == pytest.approx(
            physics["cases"][name]["total"], rel=1.0e-12, abs=1.0e-15
        )
        assert observation["max_resolved_relative_difference"] < 1.0e-12
