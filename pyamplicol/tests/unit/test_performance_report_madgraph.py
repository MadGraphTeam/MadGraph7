# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.performance_report import madgraph
from tools.performance_report.models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    MeasurementSpec,
    ModelKey,
    ResultStatus,
    Workload,
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_UFO_MODEL = _REPOSITORY / "src/pyamplicol/assets/models/ufo/sm"
_JSON_MODEL = _REPOSITORY / "src/pyamplicol/assets/models/json/sm/sm.json"
_JSON_RESTRICTION = (
    _REPOSITORY / "src/pyamplicol/assets/models/json/sm/restrict_default.json"
)


def _command_result(
    *,
    cwd: Path,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> madgraph.CommandResult:
    return madgraph.CommandResult(
        args=("fake-command",),
        cwd=cwd,
        elapsed_seconds=0.25,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _madgraph_cell() -> CellSpec:
    return CellSpec(
        dataset_id="reference_madgraph_full",
        process="g g > g g",
        n_final=2,
        process_key="gg",
        measurement=MeasurementSpec(
            execution_mode=ExecutionMode.MADGRAPH,
            model=ModelKey.UFO_SM,
            accuracy=Accuracy.FULL,
            backend="fortran",
            jit_optimization_level=None,
        ),
        workload=Workload.CONTRACTED,
    )


def _installation(tmp_path: Path) -> Path:
    installation = tmp_path / "madgraph"
    launcher = installation / "bin/mg5_aMC"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    launcher.chmod(0o755)
    shutil.copytree(_UFO_MODEL, installation / "models/sm")
    (installation / "VERSION").write_text("test-version\n", encoding="ascii")
    return installation


class _FakeExecutor:
    def __init__(self, *, fail_import: bool = False) -> None:
        self.fail_import = fail_import
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
    ) -> madgraph.CommandResult:
        rendered = tuple(os.fspath(item) for item in args)
        self.calls.append((rendered, cwd))
        if Path(rendered[0]).name == "mg5_aMC":
            if self.fail_import:
                return madgraph.CommandResult(
                    rendered,
                    cwd,
                    1.75,
                    0,
                    "UFOImportError: installed model rejected\n",
                    "",
                )
            self._write_standalone(
                cwd,
                Path(rendered[0]).parents[1] / "models/sm",
            )
            return madgraph.CommandResult(rendered, cwd, 1.75, 0, "", "")
        if "-o" in rendered:
            output = cwd / rendered[rendered.index("-o") + 1]
            output.write_text("fake driver\n", encoding="ascii")
            output.chmod(0o755)
            return madgraph.CommandResult(rendered, cwd, 0.4, 0, "", "")
        points = int(rendered[3])
        value = 2.5
        seconds = points * 0.01
        stdout = "\n".join(
            (
                f"PYAMPLICOL_MG_VALUE {value:.17e}",
                f"PYAMPLICOL_MG_POINTS {points}",
                f"PYAMPLICOL_MG_SECONDS {seconds:.17e}",
                f"PYAMPLICOL_MG_CHECKSUM {points * value:.17e}",
                "",
            )
        )
        return madgraph.CommandResult(rendered, cwd, seconds, 0, stdout, "")

    @staticmethod
    def _write_standalone(artifact: Path, model: Path) -> None:
        standalone = artifact / "standalone"
        cards = standalone / "Cards"
        subprocess_dir = standalone / "SubProcesses/P0_gg_gg"
        library = standalone / "lib"
        source = standalone / "Source"
        for directory in (cards, subprocess_dir, library, source):
            directory.mkdir(parents=True)
        (cards / "proc_card_mg5.dat").write_text(
            "import model sm\n",
            encoding="utf-8",
        )
        (cards / "param_card.dat").write_text(
            (model / "restrict_default.dat").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (subprocess_dir / "matrix.f").write_text("C fake matrix\n", encoding="ascii")
        (subprocess_dir / "matrix.o").write_bytes(b"fake object")
        (library / "libdhelas.a").write_bytes(b"fake library")
        (library / "libmodel.a").write_bytes(b"fake library")
        (source / "make_opts").write_text(
            "DEFAULT_F_COMPILER = fake-fortran\n", encoding="ascii"
        )


class _QuantizedTimerExecutor(_FakeExecutor):
    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
    ) -> madgraph.CommandResult:
        result = super().run(args, cwd=cwd)
        rendered = tuple(os.fspath(item) for item in args)
        if Path(rendered[0]).name == "mg5_aMC" or "-o" in rendered:
            return result
        points = int(rendered[3])
        seconds = {10: 0.001, 20: 0.0}.get(points, points * 1.0e-5)
        value = 2.5
        stdout = "\n".join(
            (
                f"PYAMPLICOL_MG_VALUE {value:.17e}",
                f"PYAMPLICOL_MG_POINTS {points}",
                f"PYAMPLICOL_MG_SECONDS {seconds:.17e}",
                f"PYAMPLICOL_MG_CHECKSUM {points * value:.17e}",
                "",
            )
        )
        return madgraph.CommandResult(rendered, cwd, seconds, 0, stdout, "")


def test_command_card_uses_exact_standalone_launch_sequence_and_installed_sm() -> None:
    assert madgraph.madgraph_command_card("g g > g g") == (
        "import model sm\n"
        "generate g g > g g\n"
        "output standalone standalone -f\n"
        "launch -f\n"
    )
    with pytest.raises(ValueError, match="one non-empty line"):
        madgraph.madgraph_command_card("g g > g g\nimport model sm")
    assert "iso_fortran_env, only: int64" in madgraph._DRIVER_SOURCE
    assert "integer(kind=int64) :: clock_start" in madgraph._DRIVER_SOURCE
    assert "if (answer /= reference) error stop 16" in madgraph._DRIVER_SOURCE
    assert (
        hashlib.sha256(madgraph._DRIVER_SOURCE.encode("utf-8")).hexdigest()
        == madgraph.MADGRAPH_DRIVER_SOURCE_SHA256
    )


def test_generation_guard_rejects_model_fallback_and_import_failures(
    tmp_path: Path,
) -> None:
    cards = tmp_path / "standalone/Cards"
    cards.mkdir(parents=True)
    process_card = cards / "proc_card_mg5.dat"
    process_card.write_text("import model fallback\n", encoding="utf-8")

    with pytest.raises(madgraph.MadGraphAdapterError, match="bound exclusively"):
        madgraph._reject_failed_generation(
            _command_result(cwd=tmp_path), tmp_path / "standalone"
        )

    process_card.write_text("import model sm\n", encoding="utf-8")
    madgraph._reject_failed_generation(
        _command_result(cwd=tmp_path), tmp_path / "standalone"
    )
    with pytest.raises(madgraph.MadGraphAdapterError, match="possible model fallback"):
        madgraph._reject_failed_generation(
            _command_result(cwd=tmp_path, stderr="UFOImportError: bad model"),
            tmp_path / "standalone",
        )


def test_exact_parameter_card_matches_json_default_restriction(tmp_path: Path) -> None:
    card = tmp_path / "param_card.dat"

    record = madgraph._write_exact_param_card(
        card,
        parameters_path=_UFO_MODEL / "parameters.py",
        model_path=_JSON_MODEL,
        restriction_path=_JSON_RESTRICTION,
    )

    expected = madgraph._expected_external_parameters(_JSON_MODEL, _JSON_RESTRICTION)
    restriction = json.loads(_JSON_RESTRICTION.read_text(encoding="utf-8"))
    rendered = card.read_text(encoding="ascii")
    assert record == {
        "external_parameter_count": len(expected),
        "external_parameters_sha256": record["external_parameters_sha256"],
        "binary64_exact_match": True,
        "format": "%.14e",
    }
    assert set(restriction) == {name for name, _value in expected.values()}
    for name, value in expected.values():
        assert f"{value:.14e} # {name}" in rendered
        assert float(f"{value:.14e}") == float(restriction[name][0])
    assert (
        madgraph._validate_exact_param_card(card, _JSON_MODEL, _JSON_RESTRICTION)
        == record
    )


def test_driver_output_parser_accepts_fortran_numbers_and_checks_checksum(
    tmp_path: Path,
) -> None:
    result = _command_result(
        cwd=tmp_path,
        stdout=(
            "PYAMPLICOL_MG_VALUE 1.25D+01\n"
            "PYAMPLICOL_MG_POINTS 4\n"
            "PYAMPLICOL_MG_SECONDS 2.0D-03\n"
        ),
        stderr="PYAMPLICOL_MG_CHECKSUM 5.0D+01\n",
    )

    parsed = madgraph._parse_driver_output(result, 4)

    assert parsed.value == 12.5
    assert parsed.points == 4
    assert parsed.seconds == 0.002
    assert parsed.checksum == 50.0
    with pytest.raises(madgraph.MadGraphAdapterError, match="changed their value"):
        madgraph._parse_driver_output(
            _command_result(
                cwd=tmp_path,
                stdout=result.stdout + "PYAMPLICOL_MG_CHECKSUM 4.9D+01\n",
            ),
            4,
        )

    # Sequential floating-point accumulation has an O(points * epsilon)
    # forward error even when every evaluated value is identical.  This is a
    # real n=1 campaign calibration sample whose individual-value guard passed.
    repeated_points = 65_790
    repeated = _command_result(
        cwd=tmp_path,
        stdout=(
            "PYAMPLICOL_MG_VALUE 2.29970567613276160E+002\n"
            f"PYAMPLICOL_MG_POINTS {repeated_points}\n"
            "PYAMPLICOL_MG_SECONDS 6.97789999999999938E-002\n"
            "PYAMPLICOL_MG_CHECKSUM 1.51297636432885565E+007\n"
        ),
    )
    parsed_repeated = madgraph._parse_driver_output(repeated, repeated_points)
    assert parsed_repeated.value == 229.97056761327616
    assert parsed_repeated.points == repeated_points


def test_installation_validation_requires_regular_executable_launcher(
    tmp_path: Path,
) -> None:
    installation = _installation(tmp_path)
    launcher = installation / "bin/mg5_aMC"

    assert madgraph._validate_installation(installation) == (
        installation.resolve(),
        launcher.resolve(),
        (installation / "models/sm").resolve(),
    )
    launcher.chmod(0o644)
    with pytest.raises(madgraph.MadGraphAdapterError, match="regular executable"):
        madgraph._validate_installation(installation)
    launcher.unlink()
    target = tmp_path / "other-launcher"
    target.write_text("#!/bin/sh\n", encoding="ascii")
    target.chmod(0o755)
    launcher.symlink_to(target)
    with pytest.raises(madgraph.MadGraphAdapterError, match="regular executable"):
        madgraph._validate_installation(installation)


def test_adapter_fake_executor_happy_path_never_invokes_madgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        madgraph.shutil,
        "which",
        lambda compiler: f"/fake/bin/{compiler}",
    )
    installation = _installation(tmp_path)
    artifact = tmp_path / "artifact"
    executor = _FakeExecutor()
    settings = madgraph.MadGraphSettings(
        installation=installation,
        target_runtime_seconds=0.05,
        minimum_calls=2,
        maximum_calls=2,
        minimum_profile_chunks=5,
        json_model_path=_JSON_MODEL,
        restriction_json_path=_JSON_RESTRICTION,
    )

    measurement = madgraph.MadGraphMeasurementAdapter(executor=executor).measure(
        _madgraph_cell(), artifact_path=artifact, settings=settings
    )

    assert measurement["status"] == ResultStatus.OK.value
    assert measurement["generation_seconds"] == 1.75
    assert measurement["matrix_element"] == 2.5
    assert measurement["sample_count"] == 10
    assert measurement["wall_seconds_per_point"] == pytest.approx(0.01)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["command_card"] == madgraph.madgraph_command_card(
        _madgraph_cell().process
    )
    assert provenance["exact_param_card"]["format"] == "%.14e"
    assert provenance["generation_includes_madgraph_compilation"] is True
    assert provenance["model"]["name"] == "sm"
    assert provenance["model"]["source_directory"] == os.fspath(
        installation / "models/sm"
    )
    assert len(executor.calls) == 8
    assert Path(executor.calls[0][0][0]) == installation / "bin/mg5_aMC"


def test_adapter_retries_a_quantized_zero_duration_profile_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        madgraph.shutil,
        "which",
        lambda compiler: f"/fake/bin/{compiler}",
    )
    installation = _installation(tmp_path)
    executor = _QuantizedTimerExecutor()
    settings = madgraph.MadGraphSettings(
        installation=installation,
        target_runtime_seconds=0.01,
        minimum_calls=10,
        maximum_calls=2_000,
        minimum_profile_chunks=5,
        json_model_path=_JSON_MODEL,
        restriction_json_path=_JSON_RESTRICTION,
    )

    measurement = madgraph.MadGraphMeasurementAdapter(executor=executor).measure(
        _madgraph_cell(),
        artifact_path=tmp_path / "artifact",
        settings=settings,
    )

    driver_points = [
        int(args[3])
        for args, _cwd in executor.calls
        if Path(args[0]).name == "pyamplicol_madgraph_driver"
    ]
    assert driver_points == [10, 20, 200, 200, 200, 200, 200]
    assert measurement["sample_count"] == 1_000
    assert measurement["wall_seconds_per_point"] == pytest.approx(1.0e-5)
    provenance = measurement["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["driver_timer"] == "fortran-system-clock-int64"
    assert provenance["profile_calls_per_chunk"] == 200
    assert provenance["profile_attempted_chunk_count"] == 6
    assert provenance["profile_discarded_chunk_count"] == 1
    assert provenance["profile_zero_duration_chunk_count"] == 1


def test_adapter_fake_executor_fails_closed_on_import_error(tmp_path: Path) -> None:
    executor = _FakeExecutor(fail_import=True)
    settings = madgraph.MadGraphSettings(
        installation=_installation(tmp_path),
        json_model_path=_JSON_MODEL,
        restriction_json_path=_JSON_RESTRICTION,
    )

    with pytest.raises(madgraph.MadGraphAdapterError, match="possible model fallback"):
        madgraph.MadGraphMeasurementAdapter(executor=executor).measure(
            _madgraph_cell(),
            artifact_path=tmp_path / "artifact",
            settings=settings,
        )
    assert len(executor.calls) == 1
