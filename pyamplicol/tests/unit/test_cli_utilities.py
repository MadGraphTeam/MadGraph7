# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.cli import UtilityInvocation, parse_cli, run_cli
from pyamplicol.cli.utilities import list_examples, profiling_campaign_root
from tools.performance_report.catalog import REPORT_CATALOG


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_examples_list_is_checkout_independent_and_descriptive() -> None:
    invocation = parse_cli(("examples", "list", "--json"))
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.output_format == "json"
    assert invocation.output_color == "auto"
    stdout = io.StringIO()
    assert run_cli(("examples", "list", "--json"), stdout=stdout) == 0
    entries = json.loads(stdout.getvalue())
    assert any(entry["name"] == "builtin_sm_lc" for entry in entries)
    heft = next(entry for entry in entries if entry["name"] == "builtin_sm_heft")
    assert "scalar HEFT" in heft["description"]
    assert "g g > H g g" in heft["description"]
    assert all("SPDX" not in entry["description"] for entry in entries)
    otf = next(entry for entry in entries if entry["name"] == "otf_pp_zjj")
    assert "first explicit warm_up or evaluation" in otf["description"]
    assert "complete helicity sum" in otf["description"]


def test_examples_list_presentation_options_are_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocation = parse_cli(("examples", "list", "--color", "always"))
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.output_format == "human"
    assert invocation.output_color == "always"

    machine = parse_cli(("examples", "list", "--json"))
    assert isinstance(machine, UtilityInvocation)
    assert machine.output_format == "json"

    with pytest.raises(SystemExit) as exc_info:
        parse_cli(("examples", "list", "--help"))
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--json" in help_text
    assert "--color {auto,always,never}" in help_text
    assert "--format" not in help_text
    with pytest.raises(SystemExit):
        parse_cli(("examples", "list", "--format", "json"))


def test_utility_output_auto_colors_a_terminal_and_json_is_uncolored(
    tmp_path: Path,
) -> None:
    human = _TTYStringIO()
    assert run_cli(("examples", "list"), stdout=human) == 0
    assert "Packaged Examples" in human.getvalue()
    assert "\x1b[" in human.getvalue()
    with pytest.raises(json.JSONDecodeError):
        json.loads(human.getvalue())

    target = tmp_path / "template.toml"
    machine = _TTYStringIO()
    assert (
        run_cli(
            ("config", "template", str(target), "--json"),
            stdout=machine,
        )
        == 0
    )
    assert json.loads(machine.getvalue()) == {
        "destination": str(target.resolve()),
        "operation": "configuration template",
        "status": "complete",
    }
    assert "\x1b[" not in machine.getvalue()


def test_examples_run_help_lists_every_runnable_card(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_cli(("examples", "run", "--help"))

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    runnable_names = {
        entry.name for entry in list_examples() if entry.action != "reference"
    }
    assert runnable_names
    assert all(f"  {name}\n" in help_text for name in runnable_names)
    assert "all_options" not in help_text


def test_examples_copy_requires_force_for_nonempty_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "examples"
    assert run_cli(("examples", "copy", str(destination))) == 0
    assert (destination / "builtin_sm_lc.toml").is_file()
    assert (destination / "builtin_sm_heft.toml").is_file()
    assert (destination / "benchmark_z6g_single_flow_helicity_sum.toml").is_file()
    assert (destination / "benchmark_z6g_all_flows_single_helicity.toml").is_file()
    assert (destination / "otf_pp_zjj.toml").is_file()
    assert (destination / "python/otf_pp_zjj_warm_up.py").is_file()
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
    expected_progress = {
        "evaluate_total.toml": "off",
        "evaluate_resolved.toml": "off",
        "benchmark.toml": "log",
    }
    for name, progress in expected_progress.items():
        card = parse_cli((str(destination / name),))
        assert not isinstance(card, UtilityInvocation)
        output = card.resolve().effective.output
        assert output.format == "human"
        assert output.color == "auto"
        assert output.progress == progress
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
    assert len(copied) == 69
    expected_cache_names = {
        f"{cell.dataset_id}.json" for cell in REPORT_CATALOG.measurement_cells()
    } | {"report-cache.schema.json"}
    assert {
        path.name for path in (destination / "results").glob("*.json")
    } == expected_cache_names
    assert (destination / "steer_performance_campaign.py").is_file()
    assert os.access(destination / "steer_performance_campaign.py", os.X_OK)
    launcher = destination / "steer_performance_campaign.py"
    assert launcher.read_text(encoding="utf-8").splitlines()[0] == (
        f"#!{sys.executable}"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = "/usr/bin:/bin"
    direct = subprocess.run(
        (str(launcher), "--help"),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    table_filling = (destination / "TABLE_FILLING.md").read_text(encoding="utf-8")
    assert "compact retained evidence" in table_filling
    assert "keep attempts and large artifacts" not in table_filling
    assert (destination / "results/report-cache.schema.json").is_file()
    assert not (destination / "pyAmpliCol.pdf").exists()
    assert not (destination / ".artifacts").exists()
    assert (destination / "campaign_artifacts").is_dir()
    assert not tuple((destination / "campaign_artifacts").iterdir())
    assert not (profiling_campaign_root() / "campaign_artifacts").exists()
    copied_readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "moves its state" in copied_readme
    assert "legacy `.artifacts`" in copied_readme
    workspace = json.loads(
        (destination / "report-workspace.json").read_text(encoding="utf-8")
    )
    assert workspace["measurement_state"] == "reset"
    assert workspace["initialized_from"] == "src/pyamplicol/_profiling_campaign"
    assert workspace["report_source_revision"] == "unknown"
    assert workspace["report_source_tree"] == "unknown"
    assert workspace["initialized_source_identity"] == {
        "clean": False,
        "dirty_paths": [],
        "revision": "unknown",
        "schema": "pyamplicol-report-source-v1",
        "tree": "unknown",
    }
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
    (destination / "campaign_artifacts/attempts/cell-a").mkdir(parents=True)
    (destination / "campaign_artifacts/attempts/cell-a/payload.bin").write_bytes(
        b"attempt"
    )
    (destination / "campaign_summary_ids").mkdir()
    (destination / "campaign_summary_ids/error.txt").write_text(
        "cell-a\n", encoding="utf-8"
    )
    (destination / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n")
    (destination / "measurement_lineage.json").write_text(
        "stale lineage\n",
        encoding="utf-8",
    )
    (destination / "pyAmpliCol.aux").write_text("stale aux\n", encoding="utf-8")
    (destination / "pyAmpliCol.log").write_text("stale log\n", encoding="utf-8")
    (destination / "notes.aux").write_text("unrelated\n", encoding="utf-8")
    (destination / ".artifacts").mkdir()
    (destination / ".artifacts/legacy.bin").write_bytes(b"legacy")
    (destination / "unrelated.txt").write_text("keep\n", encoding="utf-8")
    madgraph_cache = destination / "results/reference_madgraph_full.json"
    madgraph_cache.write_text('{"stale":true}\n', encoding="ascii")
    assert run_cli(("profiling-campaign", "copy", str(destination), "--force")) == 0
    assert (destination / "campaign_artifacts").is_dir()
    assert not tuple((destination / "campaign_artifacts").iterdir())
    assert not (destination / "campaign_summary_ids").exists()
    assert not (destination / "pyAmpliCol.pdf").exists()
    assert not (destination / "measurement_lineage.json").exists()
    assert not (destination / "pyAmpliCol.aux").exists()
    assert not (destination / "pyAmpliCol.log").exists()
    assert (destination / "notes.aux").read_text(encoding="utf-8") == "unrelated\n"
    assert (destination / ".artifacts/legacy.bin").read_bytes() == b"legacy"
    assert (destination / "unrelated.txt").read_text(encoding="utf-8") == "keep\n"
    reset_madgraph_cache = json.loads(madgraph_cache.read_text(encoding="ascii"))
    assert reset_madgraph_cache["dataset_id"] == "reference_madgraph_full"
    assert {
        entry["measurement"]["status"] for entry in reset_madgraph_cache["entries"]
    } == {"not_available"}


def test_profiling_campaign_force_refuses_an_active_directory_lock(
    tmp_path: Path,
) -> None:
    import fcntl

    destination = tmp_path / "active-campaign"
    assert run_cli(("profiling-campaign", "copy", str(destination))) == 0
    sentinel = destination / "campaign_artifacts/active-attempt.bin"
    sentinel.write_bytes(b"active")
    descriptor = os.open(
        destination,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        stderr = io.StringIO()
        assert (
            run_cli(
                ("profiling-campaign", "copy", str(destination), "--force"),
                stderr=stderr,
            )
            == 2
        )
        assert "campaign is active" in stderr.getvalue()
        assert sentinel.read_bytes() == b"active"
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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
    assert configured.read_text(encoding="utf-8") == f"{checkout}\n"

    replacement = tmp_path / "replacement-amplicol"
    replacement.mkdir()
    assert (
        run_cli(
            (
                "profiling-campaign",
                "copy",
                str(destination),
                "--force",
                "--local-amplicol",
                str(replacement),
            )
        )
        == 0
    )
    assert configured.read_text(encoding="utf-8") == f"{replacement.resolve()}\n"


def test_profiling_campaign_copy_records_local_madgraph_default(
    tmp_path: Path,
) -> None:
    installation = tmp_path / "madgraph"
    (installation / "bin").mkdir(parents=True)
    (installation / "bin/mg5_aMC").write_text("#!/bin/sh\n", encoding="utf-8")
    (installation / "bin/mg5_aMC").chmod(0o755)
    installation = installation.resolve()
    destination = tmp_path / "profiling-campaign"
    invocation = parse_cli(
        (
            "profiling-campaign",
            "copy",
            str(destination),
            "--local-madgraph",
            str(installation),
        )
    )
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.local_madgraph == installation

    assert (
        run_cli(
            (
                "profiling-campaign",
                "copy",
                str(destination),
                "--local-madgraph",
                str(installation),
            )
        )
        == 0
    )
    configured = destination / ".pyamplicol-madgraph"
    assert configured.read_text(encoding="utf-8") == f"{installation}\n"

    assert run_cli(("profiling-campaign", "copy", str(destination), "--force")) == 0
    assert configured.read_text(encoding="utf-8") == f"{installation}\n"

    replacement = tmp_path / "replacement-madgraph"
    (replacement / "bin").mkdir(parents=True)
    (replacement / "bin/mg5_aMC").write_text("#!/bin/sh\n", encoding="utf-8")
    (replacement / "bin/mg5_aMC").chmod(0o755)
    assert (
        run_cli(
            (
                "profiling-campaign",
                "copy",
                str(destination),
                "--force",
                "--local-madgraph",
                str(replacement),
            )
        )
        == 0
    )
    assert configured.read_text(encoding="utf-8") == f"{replacement.resolve()}\n"


def test_profiling_campaign_copy_rejects_unsafe_local_madgraph_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "profiling-campaign"
    assert run_cli(("profiling-campaign", "copy", str(destination))) == 0
    sentinel = tmp_path / "external-madgraph-config"
    sentinel.write_text("keep\n", encoding="utf-8")
    (destination / ".pyamplicol-madgraph").symlink_to(sentinel)
    replacement = tmp_path / "replacement-madgraph"
    (replacement / "bin").mkdir(parents=True)
    (replacement / "bin/mg5_aMC").write_text("#!/bin/sh\n", encoding="utf-8")
    (replacement / "bin/mg5_aMC").chmod(0o755)
    stderr = io.StringIO()

    assert (
        run_cli(
            (
                "profiling-campaign",
                "copy",
                str(destination),
                "--force",
                "--local-madgraph",
                str(replacement),
            ),
            stderr=stderr,
        )
        == 2
    )
    assert (
        "unsafe profiling campaign output" in stderr.getvalue()
        or "escapes its destination" in stderr.getvalue()
    )
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_profiling_campaign_force_rejects_unsafe_exact_reset_targets(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("keep\n", encoding="utf-8")

    for name in ("campaign_artifacts", "campaign_summary_ids"):
        destination = tmp_path / f"campaign-{name}"
        destination.mkdir()
        (destination / name).symlink_to(external, target_is_directory=True)
        stderr = io.StringIO()
        assert (
            run_cli(
                ("profiling-campaign", "copy", str(destination), "--force"),
                stderr=stderr,
            )
            == 2
        )
        assert "unsafe" in stderr.getvalue() or "escapes" in stderr.getvalue()
        assert sentinel.read_text(encoding="utf-8") == "keep\n"

    destination = tmp_path / "campaign-pdf"
    destination.mkdir()
    (destination / "pyAmpliCol.pdf").symlink_to(sentinel)
    stderr = io.StringIO()
    assert (
        run_cli(
            ("profiling-campaign", "copy", str(destination), "--force"),
            stderr=stderr,
        )
        == 2
    )
    assert "unsafe" in stderr.getvalue() or "escapes" in stderr.getvalue()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_profiling_campaign_force_rejects_unsafe_managed_output(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-readme"
    external.write_text("keep\n", encoding="utf-8")
    destination = tmp_path / "campaign"
    destination.mkdir()
    (destination / "README.md").symlink_to(external)

    stderr = io.StringIO()
    assert (
        run_cli(
            ("profiling-campaign", "copy", str(destination), "--force"),
            stderr=stderr,
        )
        == 2
    )
    assert "unsafe" in stderr.getvalue() or "escapes" in stderr.getvalue()
    assert external.read_text(encoding="utf-8") == "keep\n"


def test_profiling_campaign_force_rejects_symlinked_destination(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    destination = tmp_path / "campaign"
    destination.symlink_to(real, target_is_directory=True)

    stderr = io.StringIO()
    assert (
        run_cli(
            ("profiling-campaign", "copy", str(destination), "--force"),
            stderr=stderr,
        )
        == 2
    )
    assert "must not traverse a symlink" in stderr.getvalue()
    assert not tuple(real.iterdir())


def test_profiling_campaign_force_rejects_special_state_members(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "campaign"
    state = destination / "campaign_artifacts"
    state.mkdir(parents=True)
    os.mkfifo(state / "active.pipe")

    stderr = io.StringIO()
    assert (
        run_cli(
            ("profiling-campaign", "copy", str(destination), "--force"),
            stderr=stderr,
        )
        == 2
    )
    assert "contains a special file" in stderr.getvalue()
    assert (state / "active.pipe").exists()


def test_profiling_campaign_copy_help_describes_local_reset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        parse_cli(("profiling-campaign", "copy", "--help"))

    assert stopped.value.code == 0
    rendered = capsys.readouterr().out
    assert "DEST/campaign_artifacts" in rendered
    assert "moves with DEST" in rendered
    assert "unrelated files" in rendered
    assert "--local-madgraph" in rendered
    assert "MadGraph installation" in rendered


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
                "--json",
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
