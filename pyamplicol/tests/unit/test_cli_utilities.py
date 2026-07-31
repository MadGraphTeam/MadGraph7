# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import os
from pathlib import Path

from pyamplicol.cli import UtilityInvocation, parse_cli, run_cli


def test_examples_list_is_checkout_independent_and_descriptive() -> None:
    invocation = parse_cli(("examples", "list", "--format", "json"))
    assert isinstance(invocation, UtilityInvocation)
    stdout = io.StringIO()
    assert run_cli(("examples", "list", "--format", "json"), stdout=stdout) == 0
    entries = json.loads(stdout.getvalue())
    assert any(entry["name"] == "builtin_sm_lc" for entry in entries)
    assert all("SPDX" not in entry["description"] for entry in entries)


def test_examples_copy_requires_force_for_nonempty_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "examples"
    assert run_cli(("examples", "copy", str(destination))) == 0
    assert (destination / "builtin_sm_lc.toml").is_file()
    assert (destination / "benchmark_z6g_single_flow_helicity_sum.toml").is_file()
    assert (destination / "benchmark_z6g_all_flows_single_helicity.toml").is_file()
    selected_card = (
        destination / "benchmark_z6g_single_flow_helicity_sum.toml"
    ).read_text(encoding="utf-8")
    union_card = (
        destination / "benchmark_z6g_all_flows_single_helicity.toml"
    ).read_text(encoding="utf-8")
    assert 'lc_flow_layout = "all-flow-union"' not in selected_card
    assert 'lc_flow_layout = "all-flow-union"' in union_card
    assert (destination / "models/json/sm/sm.json").is_file()
    assert (destination / "models/ufo/sm/vertices.py").is_file()
    stderr = io.StringIO()
    assert run_cli(("examples", "copy", str(destination)), stderr=stderr) == 2
    assert "not empty" in stderr.getvalue()
    assert run_cli(("examples", "copy", str(destination), "--force")) == 0


def test_profiling_campaign_copy_is_reset_and_requires_force(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "profiling-campaign"
    invocation = parse_cli(("profiling-campaign", "copy", str(destination)))
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.kind == "profiling-campaign-copy"
    assert invocation.path == destination
    assert invocation.force is False

    assert run_cli(("profiling-campaign", "copy", str(destination))) == 0
    copied = tuple(path for path in destination.rglob("*") if path.is_file())
    assert len(copied) == 55
    assert (destination / "steer_performance_campaign.py").is_file()
    assert os.access(destination / "steer_performance_campaign.py", os.X_OK)
    assert (destination / "TABLE_FILLING.md").is_file()
    assert (destination / "results/report-cache.schema.json").is_file()
    assert not (destination / "pyAmpliCol.pdf").exists()
    assert not (destination / ".artifacts").exists()
    workspace = json.loads(
        (destination / "report-workspace.json").read_text(encoding="utf-8")
    )
    assert workspace["measurement_state"] == "reset"
    for path in sorted((destination / "results").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", ()):
            assert entry["measurement"]["status"] == "not_available"

    stderr = io.StringIO()
    assert (
        run_cli(
            ("profiling-campaign", "copy", str(destination)),
            stderr=stderr,
        )
        == 2
    )
    assert "not empty" in stderr.getvalue()
    assert run_cli(("profiling-campaign", "copy", str(destination), "--force")) == 0


def test_profiling_campaign_copy_records_local_amplicol_default(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "original-amplicol"
    checkout.mkdir()
    checkout = checkout.resolve()
    destination = tmp_path / "profiling-campaign"
    invocation = parse_cli(
        (
            "profiling-campaign",
            "copy",
            str(destination),
            "--local-amplicol",
            str(checkout),
        )
    )
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.local_amplicol == checkout

    assert (
        run_cli(
            (
                "profiling-campaign",
                "copy",
                str(destination),
                "--local-amplicol",
                str(checkout),
            )
        )
        == 0
    )
    configured = destination / ".pyamplicol-original-amplicol"
    assert configured.read_text(encoding="utf-8") == f"{checkout}\n"

    assert run_cli(("profiling-campaign", "copy", str(destination), "--force")) == 0
    assert not configured.exists()


def test_config_template_and_resolve(tmp_path: Path) -> None:
    target = tmp_path / "all.toml"
    assert run_cli(("config", "template", str(target))) == 0
    template = target.read_text(encoding="utf-8")
    assert "schema_version" in template
    for path in (
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    ):
        assert path not in template
    stdout = io.StringIO()
    assert (
        run_cli(
            (
                "config",
                "resolve",
                str(target),
                "--set",
                "generation.workers=1",
                "--format",
                "json",
            ),
            stdout=stdout,
        )
        == 0
    )
    payload = json.loads(stdout.getvalue())
    assert payload["effective"]["generation"]["workers"] == 1


def test_generate_dry_run_does_not_require_output() -> None:
    invocation = parse_cli(("generate", "d d~ > z", "--dry-run"))
    assert invocation.dry_run is True
    assert invocation.resolve().effective.generation.output is None


def test_example_run_materializes_outside_package_resources(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("PYAMPLICOL_EXAMPLE_CACHE", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    from pyamplicol.cli.utilities import example_card

    card = example_card("builtin_sm_lc")
    assert card == tmp_path / "cache/builtin_sm_lc.toml"
    assert (tmp_path / "cache/data/pp_zjj_momenta.json").is_file()
    assert (tmp_path / "cache/models/json/scalars/scalars.json").is_file()
    assert (tmp_path / "cache/models/ufo/sm/vertices.py").is_file()
