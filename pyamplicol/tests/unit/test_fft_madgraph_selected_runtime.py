# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.developer import fft_madgraph_selected_runtime as madgraph


def _event_text(
    family: str,
    n: int,
    point: int,
    helicity: tuple[int, ...],
) -> str:
    header = (
        f"AMPLIGLUON_EVENT_V1\nFINAL_GLUONS {n}\nSTRONG_COUPLING 1.0\n"
        if family == "gg"
        else "PYAMPLICOL_SCALING_EVENT_V1\n"
    )
    momenta = "\n".join(
        f"{point + index}.0 {index + 1}.0 0.0 0.0"
        for index in range(n + 2)
    )
    return (
        f"{header}BEGIN_MOMENTA\n{momenta}\nEND_MOMENTA\n"
        "NHELICITIES 1\nBEGIN_HELICITIES\n"
        f"{' '.join(str(value) for value in helicity)}\n"
        "END_HELICITIES\n"
    )


def _source_cell(
    *,
    family: str,
    mode: str,
    n: int,
    helicity: tuple[int, ...],
    event_paths: tuple[Path, ...],
) -> dict[str, object]:
    return {
        "family": family,
        "mode": mode,
        "n": n,
        "total_external": n + 2,
        "process": madgraph.process_expression(family, n),
        "status": "measured",
        "label": f"test {family} {mode} n={n}",
        "helicity_workload": "fixed",
        "warm_fixed_helicity": True,
        "warm_helicity_sum": False,
        "helicity": list(helicity),
        "event_paths": [str(path) for path in event_paths],
        "point_values": [float(point) for point in range(1, 11)],
    }


def _make_source_report(root: Path) -> Path:
    events_root = root / "events"
    cells: dict[str, dict[str, dict[str, dict[str, object]]]] = {
        "gg": {"reference-fft": {}, "recurrence-fft": {}},
        "ddbar": {"recurrence-fft": {}},
    }
    for family in madgraph.FAMILIES:
        modes = tuple(cells[family])
        for n in madgraph.FINAL_MULTIPLICITIES:
            helicity = tuple(
                1 if family == "gg" or index % 2 else -1
                for index in range(n + 2)
            )
            event_directory = events_root / family / f"n{n}"
            event_directory.mkdir(parents=True)
            event_paths: list[Path] = []
            for point in range(1, madgraph.POINT_COUNT + 1):
                event = event_directory / f"point-{point:02d}.event"
                event.write_text(
                    _event_text(family, n, point, helicity),
                    encoding="utf-8",
                )
                event_paths.append(event)
            for mode in modes:
                cells[family][mode][str(n)] = _source_cell(
                    family=family,
                    mode=mode,
                    n=n,
                    helicity=helicity,
                    event_paths=tuple(event_paths),
                )
    report = root / "source-report.json"
    report.write_text(
        json.dumps(
            {
                "kind": madgraph.SOURCE_KIND,
                "schema_version": 1,
                "policy": {
                    "measurement": {
                        "alpha_s": 0.118,
                        "helicity_workload": "fixed",
                        "warm_fixed_helicity": True,
                        "warm_helicity_sum": False,
                    }
                },
                "cells": cells,
            }
        ),
        encoding="utf-8",
    )
    return report


@pytest.fixture(scope="module")
def source_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_source_report(tmp_path_factory.mktemp("madgraph-source"))


@pytest.fixture(scope="module")
def matrix_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("madgraph-matrix") / "matrix.f"
    path.write_text(
        """      REAL*8 FUNCTION MATRIX(P,NHEL,IC)
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER NHEL(NEXTERNAL)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=3)
      INTEGER NCOLOR
      PARAMETER (NCOLOR=2)
      MATRIX=1D0
      END
      SUBROUTINE SMATRIX(P,ANS)
      INTEGER NCOMB
      PARAMETER (NCOMB=16)
      INTEGER NHEL(NEXTERNAL,NCOMB)
      INTEGER IDEN
      DATA IDEN/512/
      ANS=1D0
      ANS=ANS/DBLE(IDEN)
      END
""",
        encoding="utf-8",
    )
    return path


def test_publication_process_grid_is_final_state_n2_through_n9() -> None:
    assert tuple(range(2, 10)) == madgraph.FINAL_MULTIPLICITIES
    assert madgraph.process_expression("gg", 2) == "g g > g g"
    assert madgraph.process_expression("gg", 9) == "g g > g g g g g g g g g"
    assert madgraph.process_expression("ddbar", 2) == "d d~ > d d~"
    assert madgraph.process_expression("ddbar", 9) == "d d~ > d d~ g g g g g g g"
    with pytest.raises(madgraph.SelectedMadGraphError, match="unsupported"):
        madgraph.process_expression("qq", 2)
    with pytest.raises(madgraph.SelectedMadGraphError, match="unsupported"):
        madgraph.process_expression("gg", 1)
    assert madgraph.process_expression("gg", 10).endswith("g g g g g g g g")


def test_cli_requires_explicit_source_report_and_madgraph_installation(
    tmp_path: Path,
) -> None:
    common = (
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "report.json"),
    )
    with pytest.raises(SystemExit):
        madgraph._parser().parse_args(common)
    with pytest.raises(SystemExit):
        madgraph._parser().parse_args(
            (*common, "--source-report", str(tmp_path / "source.json"))
        )
    with pytest.raises(SystemExit):
        madgraph._parser().parse_args(
            (*common, "--mg5-root", str(tmp_path / "mg5"))
        )

    arguments = madgraph._parser().parse_args(
        (
            *common,
            "--source-report",
            str(tmp_path / "source.json"),
            "--mg5-root",
            str(tmp_path / "mg5"),
        )
    )
    assert arguments.source_report == tmp_path / "source.json"
    assert arguments.mg5_root == tmp_path / "mg5"


def test_source_selection_authenticates_both_families_and_all_points(
    source_report: Path,
) -> None:
    source = madgraph.load_source_selection(source_report)
    assert source.alpha_s == 0.118
    assert set(source.cells) == {"gg", "ddbar"}
    assert set(source.cells["gg"]) == set(range(2, 10))
    assert set(source.cells["ddbar"]) == set(range(2, 10))
    assert source.unavailable == {"gg": {}, "ddbar": {}}
    assert source.cells["gg"][2].source_mode == "reference-fft"
    assert source.cells["ddbar"][2].source_mode == "recurrence-fft"

    for family in madgraph.FAMILIES:
        for n in madgraph.FINAL_MULTIPLICITIES:
            cell = source.cells[family][n]
            assert cell.process == madgraph.process_expression(family, n)
            assert len(cell.helicity) == n + 2
            assert len(cell.events) == madgraph.POINT_COUNT
            assert len(cell.reference_point_values) == madgraph.POINT_COUNT
            assert len({event.sha256 for event in cell.events}) == 10
            assert all(event.helicity == cell.helicity for event in cell.events)

    assert source.cells["gg"][2].events[0].strong_coupling == 1.0
    assert source.cells["ddbar"][2].events[0].strong_coupling == pytest.approx(
        math.sqrt(4.0 * math.pi * source.alpha_s)
    )


def test_source_selection_rejects_changed_event_helicity(
    tmp_path: Path,
    source_report: Path,
) -> None:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    payload["cells"]["ddbar"]["recurrence-fft"]["2"]["helicity"][0] *= -1
    report = tmp_path / "source.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(madgraph.SelectedMadGraphError, match="event helicities"):
        madgraph.load_source_selection(report)


def test_source_selection_rejects_duplicate_event_payloads(tmp_path: Path) -> None:
    report = _make_source_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    paths = payload["cells"]["gg"]["reference-fft"]["2"]["event_paths"]
    Path(paths[1]).write_bytes(Path(paths[0]).read_bytes())
    with pytest.raises(madgraph.SelectedMadGraphError, match="unique event payloads"):
        madgraph.load_source_selection(report, multiplicities=(2,))


def _summed_source_report(tmp_path: Path, source_report: Path) -> Path:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    measurement = payload["policy"]["measurement"]
    measurement.update(
        {
            "helicity_workload": "sum",
            "warm_fixed_helicity": False,
            "warm_helicity_sum": True,
        }
    )
    for family in madgraph.FAMILIES:
        mode = madgraph.SUM_SOURCE_MODE[family]
        for cell in payload["cells"][family][mode].values():
            cell.update(
                {
                    "helicity_workload": "sum",
                    "warm_fixed_helicity": False,
                    "warm_helicity_sum": True,
                }
            )
    report = tmp_path / "summed-source.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report


def test_summed_source_uses_candidate_convention_and_native_smatrix(
    tmp_path: Path,
    source_report: Path,
) -> None:
    source = madgraph.load_source_selection(
        _summed_source_report(tmp_path, source_report),
        multiplicities=(2,),
        helicity_workload="sum",
    )
    assert source.helicity_workload == "sum"
    assert source.cells["gg"][2].source_mode == "recurrence-fft"
    assert source.cells["ddbar"][2].source_mode == "recurrence-fft"

    body = madgraph.render_check_source(
        source.cells["gg"][2], summed_helicity_coverage_count=16
    )
    compact = body.replace(" ", "")
    assert body.count("CALL SMATRIX(") == 3
    assert "USERHEL=-1" in compact
    assert "MATRIX(P(0,1,E),HEL,IC)" not in body
    assert f"ASVALUE={madgraph._fortran_double(source.alpha_s)}" in body
    assert "HELICITY_COVERAGE_COUNT" in body

    parsed = madgraph.parse_driver_output(
        _driver_output(helicity_workload="sum"),
        batches=madgraph.TIMING_DRIVER_BATCH_COUNT,
        expected_total_external=4,
        expected_events=10,
        helicity_workload="sum",
        expected_helicity_coverage_count=16,
    )
    assert parsed.evaluations_per_sweep == 1
    assert parsed.helicity_coverage_count == 16


def test_rendered_check_fixes_only_timing_helicity_and_calls_direct_matrix(
    source_report: Path,
) -> None:
    source = madgraph.load_source_selection(source_report)
    for family, n in (("gg", 2), ("ddbar", 9)):
        cell = source.cells[family][n]
        body = madgraph.render_check_source(cell)
        assert "SMATRIX" not in body.upper()
        assert "MATRIX(P(0,1,E),HEL,IC)" in body
        assert "MATRIX(P(0,1,1),HEL,IC)" in body
        assert "CALL SETPARA('../../Cards/param_card.dat')" in body
        assert "CALL UPDATE_AS_PARAM2(1D0,ASVALUE)" in body
        assert "CALL SYSTEM_CLOCK(CLOCK0,CLOCKRATE)" in body
        assert "DBLE(CLOCK1-CLOCK0)/DBLE(CLOCKRATE)" in body
        assert f"DATA HEL /{','.join(str(value) for value in cell.helicity)}/" in body
        assert f"PARAMETER (NSAMPLES={madgraph.TIMING_DRIVER_BATCH_COUNT})" in body
        assert max(map(len, body.splitlines())) <= 132


def _driver_output(
    *,
    total_external: int = 4,
    batches: int = 11,
    point_values: tuple[float, ...] | None = None,
    initialization_seconds: float = 1.0e-3,
    helicity_workload: str = "fixed",
) -> str:
    points = point_values or tuple(float(index) for index in range(1, 11))
    lines = [
        (
            "BACKEND MadGraph5_aMCatNLOHelicitySum"
            if helicity_workload == "sum"
            else "BACKEND MadGraph5_aMCatNLOFixedHelicity"
        ),
        (
            "HELICITY_EVALUATOR SMATRIX_GENERATED_COMPLETE_HELICITY_SUM"
            if helicity_workload == "sum"
            else "HELICITY_EVALUATOR MATRIX_DIRECT_VECTOR"
        ),
        f"TOTAL_EXTERNAL {total_external}",
        f"INITIALIZATION_SECONDS {initialization_seconds:.17e}",
        "FIRST_SAMPLE_PASS_SECONDS 2.0D-3",
    ]
    lines.extend(
        f"MATRIX_ELEMENT {index} 1 {value:.17e}"
        for index, value in enumerate(points, start=1)
    )
    lines.append("EVALUATIONS_PER_SWEEP 1")
    lines.append(
        "HELICITY_COVERAGE_COUNT "
        + (str(2**total_external) if helicity_workload == "sum" else "1")
    )
    for batch in range(1, batches + 1):
        lines.extend(
            (
                f"EVALUATION_SWEEP_SECONDS {batch} {batch}.0D-6",
                f"EVALUATION_CELL_SECONDS {batch} 1 1 {batch}.0D-6",
            )
        )
    lines.append("CHECKSUM 1.0D+0")
    return "\n".join(lines) + "\n"


def test_driver_output_requires_ten_points_and_eleven_timing_batches() -> None:
    parsed = madgraph.parse_driver_output(
        _driver_output(),
        batches=madgraph.TIMING_DRIVER_BATCH_COUNT,
        expected_total_external=4,
        expected_events=10,
    )
    assert parsed.point_values == tuple(float(index) for index in range(1, 11))
    assert len(parsed.cell_seconds) == 11
    assert parsed.initialization_seconds == 1.0e-3

    incomplete = _driver_output().replace(
        "EVALUATION_CELL_SECONDS 11 1 1 11.0D-6\n", ""
    )
    with pytest.raises(madgraph.SelectedMadGraphError, match="incomplete timing"):
        madgraph.parse_driver_output(
            incomplete,
            batches=11,
            expected_total_external=4,
            expected_events=10,
        )


def test_external_time_command_selects_darwin_and_linux_dialects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(madgraph.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        madgraph.Path,
        "is_file",
        lambda path: str(path) == "/usr/bin/time",
    )
    assert madgraph._external_time_command(("probe", "argument")) == [
        "/usr/bin/time",
        "-l",
        "probe",
        "argument",
    ]

    monkeypatch.setattr(madgraph.platform, "system", lambda: "Linux")
    monkeypatch.setattr(madgraph.shutil, "which", lambda name: f"/opt/bin/{name}")
    assert madgraph._external_time_command(("probe",)) == [
        "/opt/bin/time",
        "-v",
        "probe",
    ]


def test_external_time_command_falls_back_to_watchdog_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(madgraph.platform, "system", lambda: "Linux")
    monkeypatch.setattr(madgraph.shutil, "which", lambda _name: None)
    assert madgraph._external_time_command(("probe", "argument")) == [
        "probe",
        "argument",
    ]

    monkeypatch.setattr(madgraph.platform, "system", lambda: "FreeBSD")
    assert madgraph._external_time_command(("probe",)) == ["probe"]


@pytest.mark.parametrize(
    ("system", "stderr", "expected_kib"),
    (
        ("Darwin", "  1048577  maximum resident set size\n", 1025),
        (
            "Linux",
            "\tMaximum resident set size (kbytes): 123456\n",
            123456,
        ),
        ("FreeBSD", "123 maximum resident set size\n", None),
    ),
)
def test_external_time_max_rss_parsers(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    stderr: str,
    expected_kib: int | None,
) -> None:
    monkeypatch.setattr(madgraph.platform, "system", lambda: system)
    assert madgraph._parse_time_max_rss(stderr) == expected_kib


def test_generated_matrix_audit_proves_general_helicity_and_direct_vector(
    matrix_source: Path,
) -> None:
    metadata = madgraph._validate_matrix_source(
        matrix_source, family="gg", final_multiplicity=2
    )
    assert metadata["generation_helicity_coverage"] == "all"
    assert metadata["generated_helicity_coverage_count"] == 16
    assert metadata["generated_matrix_graphs"] > 0
    assert metadata["colour_flows"] > 0
    assert metadata["smatrix_iden"] == 512
    assert metadata["matrix_sha256"] == hashlib.sha256(
        matrix_source.read_bytes()
    ).hexdigest()


def test_generated_matrix_audit_rejects_missing_duplicate_or_wrong_iden(
    tmp_path: Path,
    matrix_source: Path,
) -> None:
    fixture = matrix_source.read_text(encoding="utf-8")
    variants = {
        "missing": fixture.replace("      DATA IDEN/512/\n", ""),
        "comment-only": fixture.replace(
            "      DATA IDEN/512/",
            "C     DATA IDEN/512/",
        ),
        "duplicate": fixture.replace(
            "      DATA IDEN/512/\n",
            "      DATA IDEN/512/\n      DATA IDEN/512/\n",
        ),
        "wrong": fixture.replace("      DATA IDEN/512/", "      DATA IDEN/513/"),
    }
    for name, source in variants.items():
        path = tmp_path / f"{name}.f"
        path.write_text(source, encoding="utf-8")
        with pytest.raises(
            madgraph.SelectedMadGraphError,
            match="IDEN denominator",
        ):
            madgraph._validate_matrix_source(path, family="gg", final_multiplicity=2)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ("PARAMETER (NCOMB=8)", "complete-helicity coverage"),
        ("INTEGER NHEL(NEXTERNAL,16)", "NHEL\\(NEXTERNAL,NCOMB\\)"),
    ),
)
def test_generated_matrix_audit_requires_exact_ncomb_nhel_coverage(
    tmp_path: Path,
    matrix_source: Path,
    replacement: str,
    message: str,
) -> None:
    fixture = matrix_source.read_text(encoding="utf-8")
    original = (
        "PARAMETER (NCOMB=16)"
        if replacement.startswith("PARAMETER")
        else "INTEGER NHEL(NEXTERNAL,NCOMB)"
    )
    path = tmp_path / "bad-coverage.f"
    path.write_text(fixture.replace(original, replacement), encoding="utf-8")
    with pytest.raises(madgraph.SelectedMadGraphError, match=message):
        madgraph._validate_matrix_source(path, family="gg", final_multiplicity=2)


def test_generated_matrix_audit_requires_executable_smatrix_iden_use(
    tmp_path: Path,
    matrix_source: Path,
) -> None:
    fixture = matrix_source.read_text(encoding="utf-8")
    statement = "      ANS=ANS/DBLE(IDEN)"
    variants = {
        "missing": fixture.replace(f"{statement}\n", ""),
        "commented": fixture.replace(statement, "C     ANS=ANS/DBLE(IDEN)"),
        "duplicate": fixture.replace(
            f"{statement}\n",
            f"{statement}\n{statement}\n",
        ),
    }
    for name, source in variants.items():
        path = tmp_path / f"{name}.f"
        path.write_text(source, encoding="utf-8")
        with pytest.raises(
            madgraph.SelectedMadGraphError,
            match="does not apply its IDEN denominator exactly once",
        ):
            madgraph._validate_matrix_source(path, family="gg", final_multiplicity=2)


def test_numerical_convention_map_is_fixed_not_fitted() -> None:
    assert madgraph._expected_smatrix_iden("gg", 2) == 512
    assert madgraph._expected_smatrix_iden("ddbar", 2) == 36
    assert madgraph._expected_smatrix_iden("ddbar", 4) == 72
    assert madgraph._expected_smatrix_iden("ddbar", 5) == 216
    assert madgraph._numerical_normalization_factor("gg", smatrix_iden=512) == 1.0
    assert (
        madgraph._numerical_normalization_factor("ddbar", smatrix_iden=36) == 1.0 / 36.0
    )
    assert (
        madgraph._numerical_normalization_factor("ddbar", smatrix_iden=72) == 1.0 / 72.0
    )
    with pytest.raises(madgraph.SelectedMadGraphError, match="unsupported"):
        madgraph._numerical_normalization_factor("bad", smatrix_iden=1)
    with pytest.raises(madgraph.SelectedMadGraphError, match="positive"):
        madgraph._numerical_normalization_factor("ddbar", smatrix_iden=0)


def test_watchdog_resource_uses_conservative_peak_guard() -> None:
    report = {
        "passes": True,
        "execution": {"elapsed_wall_seconds": 2.5, "outcome": "command-finished"},
        "enforcement": {
            "peak_guard_bytes": 2049,
            "peak_rss_bytes": 1024,
            "peak_physical_footprint_bytes": 2049,
            "metric": "max(process-tree-rss,darwin-process-tree-physical-footprint)",
            "limit_bytes": 30 * 1024**3,
        },
    }
    resource = madgraph._watchdog_resource(report)
    assert resource["elapsed_wall_seconds"] == 2.5
    assert resource["peak_guard_kib"] == 3
    assert resource["passes"] is True


def test_disk_exhaustion_is_infrastructure_not_physics_frontier() -> None:
    assert madgraph._is_disk_exhaustion(OSError(28, "No space left on device"))
    assert madgraph._is_disk_exhaustion("No space left on device")
    assert not madgraph._is_disk_exhaustion("generation reached memory limit")


def test_time_and_memory_limits_are_strict_publication_caps() -> None:
    assert madgraph.DEFAULT_GENERATION_TIMEOUT_SECONDS < 3600.0
    assert madgraph.MAX_MEMORY_GIB == 30.0
    assert madgraph.TARGET_SECONDS == 0.25
    assert madgraph.WARM_SAMPLE_COUNT == 10


def test_hidden_timeout_worker_propagates_exit_and_terminates_late_child() -> None:
    assert (
        madgraph.main(
            [
                "_run-with-timeout",
                "--timeout-seconds",
                "1",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ]
        )
        == 7
    )
    assert (
        madgraph.main(
            [
                "_run-with-timeout",
                "--timeout-seconds",
                "0.01",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(10)",
            ]
        )
        == 124
    )


def _watchdog_report(*, elapsed: float, peak_kib: int = 1024) -> dict[str, object]:
    return {
        "passes": True,
        "execution": {
            "elapsed_wall_seconds": elapsed,
            "outcome": "command-finished",
        },
        "enforcement": {
            "peak_guard_bytes": peak_kib * 1024,
            "peak_rss_bytes": peak_kib * 1024,
            "peak_physical_footprint_bytes": peak_kib * 1024,
            "metric": "process-tree-rss",
            "limit_bytes": 30 * 1024**3,
        },
    }


def test_timeout_frontier_requires_authenticated_wrapper_marker() -> None:
    natural_124 = subprocess.CompletedProcess([], 124, "", "child failed")
    timed_out = subprocess.CompletedProcess(
        [], 124, "", f"prefix\n{madgraph.TIMEOUT_MARKER}\nsuffix"
    )
    assert not madgraph._bounded_command_timed_out(natural_124)
    assert madgraph._bounded_command_timed_out(timed_out)


def test_measurement_runtime_consumes_remaining_cold_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
    matrix_source: Path,
) -> None:
    source = madgraph.load_source_selection(source_report)
    selected = source.cells["gg"][2]
    cell_dir = tmp_path / "gg" / "n2"
    process_dir = cell_dir / "generated" / "SubProcesses" / "P1_test"
    process_dir.mkdir(parents=True)
    shutil.copy2(matrix_source, process_dir / "matrix.f")
    (process_dir / "makefile").write_text("check:\n\t@true\n", encoding="ascii")
    mg5_root = tmp_path / "mg5"
    mg5_root.mkdir()
    (mg5_root / "VERSION").write_text("test-version\n", encoding="ascii")
    calls: list[tuple[list[str], float | None, Path]] = []

    def bounded(
        command: list[str],
        *,
        cwd: Path,
        report: Path,
        limit_gib: float,
        timeout_seconds: float | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        del limit_gib
        calls.append((list(command), timeout_seconds, report))
        if command[0] == "make":
            executable = cwd / "check"
            executable.write_bytes(b"test executable")
            executable.chmod(0o755)
            return (
                subprocess.CompletedProcess(command, 0, "", ""),
                _watchdog_report(elapsed=2.0, peak_kib=2000),
            )
        return (
            subprocess.CompletedProcess(
                command,
                0,
                _driver_output(
                    point_values=selected.reference_point_values,
                    initialization_seconds=1.0,
                ),
                "",
            ),
            _watchdog_report(elapsed=3.0, peak_kib=3000),
        )

    monkeypatch.setattr(madgraph, "_run_bounded_command", bounded)
    cell = madgraph._measure_generated_cell(
        source=source,
        selected=selected,
        generation={
            "output": str(cell_dir / "generated"),
            "process_dirs": [str(process_dir)],
        },
        generation_watchdog=_watchdog_report(elapsed=5.0, peak_kib=1000),
        cell_dir=cell_dir,
        fc="gfortran",
        fflags="-O3",
        limit_gib=30.0,
        mg5_root=mg5_root,
        cold_to_ready_limit_seconds=20.0,
    )
    assert [call[1] for call in calls] == [15.0, 13.0]
    assert all(
        call[2].parent == cell_dir / "measurement-attempts" / "attempt-001"
        for call in calls
    )
    assert cell["metrics"]["generation_seconds"] == 8.0
    assert cell["provenance"]["generation_helicity_coverage"] == "all"
    with pytest.raises(madgraph.ResourceFrontierError) as exhausted:
        madgraph._remaining_cold_to_ready_budget(
            5.0, madgraph._watchdog_resource(_watchdog_report(elapsed=5.0))
        )
    assert exhausted.value.category == "generation-time-limit"


def test_attempt_paths_resume_without_reusing_watchdog_reports(tmp_path: Path) -> None:
    cell_dir = tmp_path / "gg" / "n2"
    cell_dir.mkdir(parents=True)
    first = madgraph._new_attempt_directory(cell_dir, "measurement")
    (first / "build-watchdog.json").write_text("{}\n", encoding="ascii")
    (first / "runtime-watchdog.json").write_text("{}\n", encoding="ascii")
    second = madgraph._new_attempt_directory(cell_dir, "measurement")
    assert first.name == "attempt-001"
    assert second.name == "attempt-002"
    assert (first / "build-watchdog.json").is_file()
    assert not (second / "build-watchdog.json").exists()


def test_cache_lock_survives_cache_tree_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(madgraph, "CACHE_LOCK_ROOT", lock_root)
    run_directory = tmp_path / "run"
    cache = run_directory / "madgraph" / "cache"
    first = madgraph._acquire_cache_lock(cache)
    try:
        cache.mkdir(parents=True)
        shutil.rmtree(run_directory)
        with pytest.raises(
            madgraph.SelectedMadGraphError,
            match="another MadGraph profiler holds cache lock",
        ):
            madgraph._acquire_cache_lock(cache)
    finally:
        first.close()

    resumed = madgraph._acquire_cache_lock(cache)
    resumed.close()
    assert madgraph._cache_lock_path(cache).parent == lock_root


@pytest.mark.parametrize("keep_generated", [False, True])
def test_checkpoint_resume_safely_prunes_only_cell_generated(
    tmp_path: Path, keep_generated: bool
) -> None:
    cell_dir = tmp_path / "gg" / "n2"
    generated = cell_dir / "generated"
    generated.mkdir(parents=True)
    sentinel = generated / "sentinel"
    sentinel.write_text("owned\n", encoding="ascii")
    checkpoint = cell_dir / "cell.json"
    identity = {"producer": "test"}
    madgraph._json_atomic(
        checkpoint,
        {"checkpoint_identity": identity, "cell": {"status": "measured"}},
    )
    cell = madgraph._load_checkpoint_cell(
        checkpoint,
        identity=identity,
        cell_dir=cell_dir,
        keep_generated=keep_generated,
    )
    assert cell["status"] == "measured"
    assert generated.exists() is keep_generated
    assert tmp_path.exists()


@pytest.mark.parametrize("family", madgraph.FAMILIES)
def test_checkpoint_producer_change_requires_fresh_cache(
    tmp_path: Path, family: str
) -> None:
    cell_dir = tmp_path / family / "n2"
    cell_dir.mkdir(parents=True)
    checkpoint = cell_dir / "cell.json"
    current = {
        "producer_sha256": "current-producer",
        "family": family,
        "n": 2,
        "sentinel": "same",
    }
    madgraph._json_atomic(
        checkpoint,
        {
            "checkpoint_identity": {
                **current,
                "producer_sha256": "previous-producer",
            },
            "cell": {
                "status": "failed",
                "failure_category": "generation-time-limit",
            },
        },
    )
    with pytest.raises(
        madgraph.SelectedMadGraphError,
        match="checkpoint identity changed",
    ):
        madgraph._load_checkpoint_cell(
            checkpoint,
            identity=current,
            cell_dir=cell_dir,
            keep_generated=True,
        )


def test_checkpoint_reuse_ignores_only_unrelated_source_report_expansion(
    tmp_path: Path,
) -> None:
    cell_dir = tmp_path / "gg" / "n2"
    cell_dir.mkdir(parents=True)
    checkpoint = cell_dir / "cell.json"
    current = {
        "producer_sha256": "same-producer",
        "source_report_sha256": "expanded-report",
        "source_cell_sha256": "same-cell",
        "family": "gg",
        "n": 2,
    }
    madgraph._json_atomic(
        checkpoint,
        {
            "checkpoint_identity": {
                **current,
                "source_report_sha256": "sparse-report",
            },
            "cell": {"status": "measured"},
        },
    )

    assert (
        madgraph._load_checkpoint_cell(
            checkpoint,
            identity=current,
            cell_dir=cell_dir,
            keep_generated=True,
        )["status"]
        == "measured"
    )


def test_generated_cleanup_refuses_symlink_target(tmp_path: Path) -> None:
    cell_dir = tmp_path / "gg" / "n2"
    cell_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve\n", encoding="ascii")
    (cell_dir / "generated").symlink_to(outside, target_is_directory=True)
    with pytest.raises(madgraph.SelectedMadGraphError, match="symlink"):
        madgraph._prune_cell_generated(cell_dir)
    assert (outside / "sentinel").read_text(encoding="ascii") == "preserve\n"


def _fake_mg5_root(tmp_path: Path) -> Path:
    root = tmp_path / "mg5"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "mg5_aMC").write_text("#!/bin/sh\n", encoding="ascii")
    (root / "VERSION").write_text("test-version\n", encoding="ascii")
    return root


@pytest.mark.parametrize(
    "category",
    [
        "generation-error",
        "generation-launch-error",
        "generation-structure-error",
        "generation-watchdog-error",
        "infrastructure-disk-exhaustion",
    ],
)
def test_generation_diagnostics_abort_without_frontier_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
    category: str,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(madgraph, "FINAL_MULTIPLICITIES", (2, 3))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_kwargs: (
            {
                "status": "failed",
                "failure_category": category,
                "failure_reason": "diagnostic",
            },
            {},
        ),
    )
    cache = tmp_path / "cache"
    with pytest.raises(madgraph.SelectedMadGraphError, match="diagnostic"):
        madgraph.build_runtime_report(
            source_report=source_report,
            cache_dir=cache,
            fc="gfortran",
            fflags="-O3",
            timeout_seconds=10.0,
            mg5_root=_fake_mg5_root(tmp_path),
            memory_limit_gib=1.0,
        )
    assert not (cache / "gg" / "n2" / "cell.json").exists()


@pytest.mark.parametrize(
    "message",
    [
        "generated standalone build failed",
        "MadGraph points disagree with reference",
        "MadGraph output has incomplete timing cells",
    ],
)
def test_measurement_diagnostics_abort_without_frontier_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
    message: str,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(madgraph, "FINAL_MULTIPLICITIES", (2, 3))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_kwargs: ({"status": "measured"}, {}),
    )

    def fail_measurement(**_kwargs: object) -> dict[str, object]:
        raise madgraph.SelectedMadGraphError(message)

    monkeypatch.setattr(madgraph, "_measure_generated_cell", fail_measurement)
    cache = tmp_path / "cache"
    with pytest.raises(madgraph.SelectedMadGraphError, match=message):
        madgraph.build_runtime_report(
            source_report=source_report,
            cache_dir=cache,
            fc="gfortran",
            fflags="-O3",
            timeout_seconds=10.0,
            mg5_root=_fake_mg5_root(tmp_path),
            memory_limit_gib=1.0,
        )
    assert not (cache / "gg" / "n2" / "cell.json").exists()


@pytest.mark.parametrize("category", ["generation-time-limit", "memory-limit"])
def test_only_resource_failures_create_and_censor_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
    category: str,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(madgraph, "FINAL_MULTIPLICITIES", (2, 3))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_kwargs: (
            {
                "status": "failed",
                "failure_category": category,
                "failure_reason": "resource cap",
            },
            {},
        ),
    )
    cache = tmp_path / "cache"
    report = madgraph.build_runtime_report(
        source_report=source_report,
        cache_dir=cache,
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=10.0,
        mg5_root=_fake_mg5_root(tmp_path),
        memory_limit_gib=1.0,
    )
    n2 = report["runtime_series"]["gg"][madgraph.MODE]["2"]
    n3 = report["runtime_series"]["gg"][madgraph.MODE]["3"]
    assert n2["status"] == "failed"
    assert n2["failure_category"] == category
    assert n3["status"] == "skipped"
    assert n3["skipped_after_frontier"] is True


def test_diagnostic_family_omission_is_not_an_availability_frontier() -> None:
    cell = madgraph._diagnostic_skip_cell("gg", 2, "diagnostic")
    assert cell["status"] == "skipped"
    assert cell["censors_higher_multiplicities"] is False
    assert cell["skipped_after_frontier"] is False
    assert "availability_frontier_n" not in cell


def _measured_stub(**arguments: object) -> dict[str, object]:
    selected = arguments["selected"]
    source = arguments["source"]
    assert isinstance(selected, madgraph.SelectedCell)
    assert isinstance(source, madgraph.SourceSelection)
    return {
        "status": "measured",
        "family": selected.family,
        "mode": madgraph.MODE,
        "label": madgraph.LABEL,
        "n": selected.n,
        "total_external": selected.total_external,
        "process": selected.process,
        "helicity_workload": selected.helicity_workload,
        "provenance": {
            "source_report": {
                "path": madgraph._display_path(source.path),
                "sha256": source.sha256,
                "cell": (
                    f"cells.{selected.family}.{selected.source_mode}.{selected.n}"
                ),
            }
        },
    }


def _write_source_payload(
    tmp_path: Path, payload: dict[str, object], name: str
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scaling_source_records_dependency_frontier_without_losing_lower_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    payload["kind"] = madgraph.SCALING_STUDY_KIND
    failed = payload["cells"]["ddbar"]["recurrence-fft"]["3"]
    failed.update(
        {
            "status": "failed",
            "failure_category": "memory-limit",
            "failure_reason": "source reached its configured RAM cap",
            "censors_higher_multiplicities": True,
        }
    )
    skipped = payload["cells"]["ddbar"]["recurrence-fft"]["4"]
    skipped.update(
        {
            "status": "skipped",
            "failure_reason": "skipped after source frontier",
            "censors_higher_multiplicities": True,
        }
    )
    source_path = _write_source_payload(tmp_path, payload, "source.json")
    source = madgraph.load_source_selection(
        source_path, multiplicities=(2, 3, 4)
    )
    assert set(source.cells["ddbar"]) == {2}
    assert set(source.unavailable["ddbar"]) == {3, 4}

    monkeypatch.setattr(madgraph, "FAMILIES", ("ddbar",))
    attempted: list[int] = []

    def generation_attempt(**arguments: object) -> tuple[dict[str, object], dict]:
        selected = arguments["selected"]
        assert isinstance(selected, madgraph.SelectedCell)
        attempted.append(selected.n)
        return {"status": "measured"}, {}

    monkeypatch.setattr(madgraph, "_generation_attempt", generation_attempt)
    monkeypatch.setattr(madgraph, "_measure_generated_cell", _measured_stub)
    progress_path = tmp_path / "overlay.progress.json"
    report = madgraph.build_runtime_report(
        source_report=source_path,
        cache_dir=tmp_path / "cache",
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=7200.0,
        mg5_root=_fake_mg5_root(tmp_path),
        multiplicities=(2, 3, 4),
        memory_limit_gib=64.0,
        progress_output=progress_path,
    )

    cells = report["runtime_series"]["ddbar"][madgraph.MODE]
    assert attempted == [2]
    assert cells["2"]["status"] == "measured"
    assert cells["3"]["failure_category"] == "dependency-unavailable"
    assert cells["4"]["status"] == "skipped"
    assert report["policy"]["final_state_multiplicities"] == [2, 3, 4]
    assert report["policy"]["generation_timeout_seconds"] == 7200.0
    assert report["policy"]["outer_memory_watchdog_gib"] == 64.0
    progress = madgraph.load_runtime_progress(progress_path)
    assert progress["status"] == "complete-with-failures"
    assert progress["summary"]["completed_cell_count"] == 3
    assert progress["pending_cells"] == []


def test_progress_is_atomic_sparse_and_names_the_current_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_arguments: ({"status": "measured"}, {}),
    )
    monkeypatch.setattr(madgraph, "_measure_generated_cell", _measured_stub)
    progress_path = tmp_path / "overlay.progress.json"
    observed: list[dict[str, object]] = []
    original_atomic = madgraph._json_atomic

    def capture_atomic(
        path: Path, payload: dict[str, object], *, replace: bool = True
    ) -> None:
        original_atomic(path, payload, replace=replace)
        if path == progress_path.resolve(strict=False):
            observed.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(madgraph, "_json_atomic", capture_atomic)
    report = madgraph.build_runtime_report(
        source_report=source_report,
        cache_dir=tmp_path / "cache",
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=7201.0,
        mg5_root=_fake_mg5_root(tmp_path),
        multiplicities=(2, 3),
        memory_limit_gib=96.0,
        progress_output=progress_path,
    )

    current = [entry["current_cell"] for entry in observed if entry["current_cell"]]
    assert [(entry["family"], entry["n"]) for entry in current] == [
        ("gg", 2),
        ("gg", 3),
    ]
    completed = [entry["summary"]["completed_cell_count"] for entry in observed]
    assert completed[0] == 0
    assert 1 in completed
    assert completed[-1] == 2
    progress = madgraph.load_runtime_progress(progress_path)
    assert progress["status"] == "complete"
    assert progress["policy"]["final_state_multiplicities"] == [2, 3]
    assert progress["policy"]["generation_timeout_seconds"] == 7201.0
    assert progress["policy"]["outer_memory_watchdog_gib"] == 96.0
    assert progress["source_report"]["sha256"] == hashlib.sha256(
        source_report.read_bytes()
    ).hexdigest()
    assert set(progress["runtime_series"]["gg"][madgraph.MODE]) == {"2", "3"}
    assert report["status"] == "complete"


def test_resumed_progress_prehydrates_compatible_checkpoint_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_arguments: ({"status": "measured"}, {}),
    )
    monkeypatch.setattr(madgraph, "_measure_generated_cell", _measured_stub)
    cache = tmp_path / "cache"
    mg5_root = _fake_mg5_root(tmp_path)
    madgraph.build_runtime_report(
        source_report=source_report,
        cache_dir=cache,
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=10.0,
        mg5_root=mg5_root,
        multiplicities=(2,),
        memory_limit_gib=1.0,
    )

    progress_path = tmp_path / "overlay.progress.json"
    observed: list[dict[str, object]] = []
    original_atomic = madgraph._json_atomic

    def capture_atomic(
        path: Path, payload: dict[str, object], *, replace: bool = True
    ) -> None:
        original_atomic(path, payload, replace=replace)
        if path == progress_path.resolve(strict=False):
            observed.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(madgraph, "_json_atomic", capture_atomic)

    def stop_on_new_cell(**arguments: object) -> dict[str, object]:
        selected = arguments["selected"]
        assert isinstance(selected, madgraph.SelectedCell)
        raise madgraph.SelectedMadGraphError(f"stop at n={selected.n}")

    monkeypatch.setattr(madgraph, "_measure_generated_cell", stop_on_new_cell)
    with pytest.raises(madgraph.SelectedMadGraphError, match="stop at n=3"):
        madgraph.build_runtime_report(
            source_report=source_report,
            cache_dir=cache,
            fc="gfortran",
            fflags="-O3",
            timeout_seconds=10.0,
            mg5_root=mg5_root,
            multiplicities=(2, 3),
            memory_limit_gib=1.0,
            progress_output=progress_path,
        )

    assert observed
    first = observed[0]
    assert first["summary"]["completed_cell_count"] == 1
    assert set(first["runtime_series"]["gg"][madgraph.MODE]) == {"2"}
    assert first["pending_cells"][0]["n"] == 3


def test_final_plot_protocol_never_launches_madgraph_above_n6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    attempted: list[int] = []

    def generation_attempt(**arguments: object) -> tuple[dict[str, object], dict]:
        selected = arguments["selected"]
        assert isinstance(selected, madgraph.SelectedCell)
        attempted.append(selected.n)
        return {"status": "measured"}, {}

    monkeypatch.setattr(madgraph, "_generation_attempt", generation_attempt)
    monkeypatch.setattr(madgraph, "_measure_generated_cell", _measured_stub)
    progress_path = tmp_path / "overlay.progress.json"

    report = madgraph.build_runtime_report(
        source_report=source_report,
        cache_dir=tmp_path / "cache",
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=10.0,
        mg5_root=_fake_mg5_root(tmp_path),
        multiplicities=tuple(range(2, 10)),
        memory_limit_gib=1.0,
        progress_output=progress_path,
    )

    assert attempted == list(range(2, 7))
    cells = report["runtime_series"]["gg"][madgraph.MODE]
    assert {cells[str(n)]["status"] for n in range(7, 10)} == {
        "not-applicable"
    }
    assert report["policy"]["maximum_measured_multiplicity"] == 6
    assert report["summary"]["runtime_series_status_counts"] == {
        "measured": 5,
        "not-applicable": 3,
    }
    progress = madgraph.load_runtime_progress(progress_path)
    assert progress["status"] == "complete"


def test_diagnostic_exit_leaves_completed_progress_renderable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    monkeypatch.setattr(
        madgraph,
        "_generation_attempt",
        lambda **_arguments: ({"status": "measured"}, {}),
    )

    def measure(**arguments: object) -> dict[str, object]:
        selected = arguments["selected"]
        assert isinstance(selected, madgraph.SelectedCell)
        if selected.n == 3:
            raise madgraph.SelectedMadGraphError("synthetic diagnostic")
        return _measured_stub(**arguments)

    monkeypatch.setattr(madgraph, "_measure_generated_cell", measure)
    progress_path = tmp_path / "overlay.progress.json"
    with pytest.raises(madgraph.SelectedMadGraphError, match="synthetic diagnostic"):
        madgraph.build_runtime_report(
            source_report=source_report,
            cache_dir=tmp_path / "cache",
            fc="gfortran",
            fflags="-O3",
            timeout_seconds=10.0,
            mg5_root=_fake_mg5_root(tmp_path),
            multiplicities=(2, 3),
            memory_limit_gib=1.0,
            progress_output=progress_path,
        )
    progress = madgraph.load_runtime_progress(progress_path)
    assert progress["status"] == "running"
    assert progress["summary"]["completed_cell_count"] == 1
    assert progress["current_cell"]["n"] == 3
    assert set(progress["runtime_series"]["gg"][madgraph.MODE]) == {"2"}


def test_cell_scoped_source_identity_reuses_and_rebinds_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_report: Path,
) -> None:
    monkeypatch.setattr(madgraph, "FAMILIES", ("gg",))
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    payload["kind"] = madgraph.SCALING_STUDY_KIND
    first_source = _write_source_payload(tmp_path, payload, "source-1.json")
    attempts = 0

    def generation_attempt(**_arguments: object) -> tuple[dict[str, object], dict]:
        nonlocal attempts
        attempts += 1
        return {"status": "measured"}, {}

    monkeypatch.setattr(madgraph, "_generation_attempt", generation_attempt)
    monkeypatch.setattr(madgraph, "_measure_generated_cell", _measured_stub)
    cache = tmp_path / "cache"
    mg5_root = _fake_mg5_root(tmp_path)
    madgraph.build_runtime_report(
        source_report=first_source,
        cache_dir=cache,
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=10.0,
        mg5_root=mg5_root,
        multiplicities=(2,),
        memory_limit_gib=1.0,
    )
    assert attempts == 1

    payload["cells"]["gg"]["reference-fft"]["3"]["label"] = "unrelated edit"
    second_source = _write_source_payload(tmp_path, payload, "source-2.json")
    report = madgraph.build_runtime_report(
        source_report=second_source,
        cache_dir=cache,
        fc="gfortran",
        fflags="-O3",
        timeout_seconds=10.0,
        mg5_root=mg5_root,
        multiplicities=(2,),
        memory_limit_gib=1.0,
    )
    assert attempts == 1
    rebound = report["runtime_series"]["gg"][madgraph.MODE]["2"]
    assert rebound["provenance"]["source_report"]["path"] == str(second_source)
    assert rebound["provenance"]["source_report"]["sha256"] == hashlib.sha256(
        second_source.read_bytes()
    ).hexdigest()
    checkpoint = json.loads((cache / "gg" / "n2" / "cell.json").read_text())
    assert checkpoint["checkpoint_identity"]["source_report_sha256"] == rebound[
        "provenance"
    ]["source_report"]["sha256"]
    assert len(checkpoint["checkpoint_identity"]["source_cell_sha256"]) == 64


@pytest.mark.parametrize(
    ("timeout", "memory", "message"),
    (
        (0.0, 1.0, "generation timeout"),
        (math.inf, 1.0, "generation timeout"),
        (True, 1.0, "generation timeout"),
        (1.0, 0.0, "memory limit"),
        (1.0, math.nan, "memory limit"),
        (1.0, True, "memory limit"),
    ),
)
def test_custom_caps_must_only_be_positive_finite_numbers(
    tmp_path: Path,
    source_report: Path,
    timeout: float,
    memory: float,
    message: str,
) -> None:
    with pytest.raises(madgraph.SelectedMadGraphError, match=message):
        madgraph.build_runtime_report(
            source_report=source_report,
            cache_dir=tmp_path / "cache",
            fc="gfortran",
            fflags="-O3",
            timeout_seconds=timeout,
            mg5_root=tmp_path / "unused",
            multiplicities=(2,),
            memory_limit_gib=memory,
        )


def test_default_progress_path_stays_beside_overlay(tmp_path: Path) -> None:
    output = tmp_path / "overlay.json"
    assert madgraph._default_progress_output(output) == (
        tmp_path / "overlay.progress.json"
    )
