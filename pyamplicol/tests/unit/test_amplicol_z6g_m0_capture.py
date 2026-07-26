# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = ROOT / "tools" / "developer" / "amplicol_z6g_m0_capture.py"
M0_SCRIPT = ROOT / "tools" / "developer" / "eager_compiled_arena_m0.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture = _load("_test_amplicol_z6g_m0_capture", CAPTURE_SCRIPT)
m0 = _load("_test_amplicol_z6g_m0_capture_m0", M0_SCRIPT)

REVISION = "b" * 40
PYAMPLICOL_REVISION = "a" * 40
POINTS_SHA256 = "1" * 64
NORMALIZATION_SHA256 = "2" * 64
FLOW_ID = "flow:2,4,5,6,7,8,9,1"
FLOW_WORD = [2, 4, 5, 6, 7, 8, 9, 1]
HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
HELICITY_VALUES = [-1, 1, -1, 1, -1, 1, -1, 1, -1]
HOST = {
    "platform": "Darwin-test",
    "system": "Darwin",
    "release": "24.0.0",
    "version": "test",
    "machine": "arm64",
    "processor": "arm",
    "cpu_model": "test-cpu",
    "logical_cpu_count": 8,
}


def _write(path: Path, raw: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if executable:
        path.chmod(0o755)
    return capture._file_ref(path)


def _git_repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=path, check=True)
    tracked = path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Capture Test",
            "-c",
            "user.email=capture@example.invalid",
            "commit",
            "-qm",
            "initial",
        ),
        cwd=path,
        check=True,
    )
    revision = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return path, revision


def _contract(tmp_path: Path) -> capture.CaptureContract:
    fixture = _write(tmp_path / "validation-momenta.json", b"fixture\n")
    return capture.CaptureContract(
        expected={
            "pyamplicol_source_revision": PYAMPLICOL_REVISION,
            "amplicol_source_revision": REVISION,
            "momenta_points_sha256": POINTS_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
            "host_sha256": capture._canonical_sha256(HOST),
            "color_flow": {"id": FLOW_ID, "word": FLOW_WORD},
            "helicity": {"id": HELICITY_ID, "values": HELICITY_VALUES},
            "external_leg_permutation": list(capture.AMPLICOL_EXTERNAL_LEG_PERMUTATION),
        },
        input_files=(),
        host=HOST,
        fixture={
            "path": Path(fixture["path"]),
            "file": fixture,
            "point_count": 2,
            "points_sha256": POINTS_SHA256,
        },
        color_axis={
            "count": 720,
            "ordered_ids_sha256": "3" * 64,
        },
        helicity_axis={
            "count": 768,
            "ordered_ids_sha256": "4" * 64,
        },
        color_flow_request="1",
        helicity_request="1",
        selected_values=(complex(1.25, 0.0), complex(2.5, 0.0)),
        union_values=(complex(3.75, 0.0), complex(5.0, 0.0)),
    )


def _context(tmp_path: Path) -> capture.ProbeContext:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    process_file = runtime / "processes.txt"
    process_file.write_text("process\n", encoding="utf-8")
    selected = runtime / "amplicol_library_benchmark"
    union = runtime / "amplicol_color_probe"
    selected.write_bytes(b"selected")
    union.write_bytes(b"union")
    worker = runtime / "amplicol_z6g_m0_sample"
    worker.write_bytes(b"worker")
    selected.chmod(0o755)
    union.chmod(0o755)
    worker.chmod(0o755)
    return capture.ProbeContext(
        runtime=runtime,
        worker_executable=worker,
        process_file=process_file,
        selected_binary=selected,
        union_binary=union,
        linked_files=(selected, union, process_file),
        group=11,
        integral=7,
        source_pdgs=(2, -2, 23, 21, 21, 21, 21, 21, 21),
        generated_pdgs=(2, -2, 21, 21, 21, 21, 21, 21, 23),
        generated_color_order=(2, 3, 4, 5, 6, 7, 8, 1, 9),
        permutation=capture.AMPLICOL_EXTERNAL_LEG_PERMUTATION,
    )


def _source(tmp_path: Path) -> capture.SourceEvidence:
    identity = _write(tmp_path / "amplicol-source.f90", b"program probe\n")
    return capture.SourceEvidence(
        source={
            "revision": REVISION,
            "dirty": False,
            "compiler": {
                "id": "gfortran",
                "version": "GNU Fortran 14.2",
                "target": "arm64-apple-darwin",
                "flags_sha256": capture._canonical_sha256(["-O3"]),
            },
            "source_tree_sha256": capture._source_tree_sha256([identity]),
        },
        source_files=(identity,),
    )


def _role_samples(
    tmp_path: Path,
    *,
    role: str,
    executable: dict[str, Any],
    fixture: dict[str, Any],
    values: list[list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    offset = 0 if role == capture.SELECTED_ROLE else 1
    for index in range(capture.MIN_SAMPLES):
        evaluated = 1000
        seconds_per_point = 4.0e-5 * (1.0 + index * 0.01)
        elapsed = evaluated * seconds_per_point
        selector = (
            f"--color-flow-id={FLOW_ID}"
            if role == capture.SELECTED_ROLE
            else f"--helicity-id={HELICITY_ID}"
        )
        command = [
            executable["path"],
            "_sample",
            f"--workload={role}",
            f"--round={index}",
            f"--momenta={fixture['path']}",
            f"--source-revision={REVISION}",
            selector,
        ]
        command_sha256 = capture._canonical_sha256(command)
        stdout_payload = {
            "kind": "amplicol-m0-probe-result",
            "schema_version": 1,
            "role": role,
            "sample_index": index,
            "evaluated_point_count": evaluated,
            "elapsed_seconds": elapsed,
            "seconds_per_point": seconds_per_point,
            "selected_totals": values,
            "resolved_sums": values,
        }
        stdout = capture._canonical_bytes(stdout_payload).decode() + "\n"
        raw = capture._raw_sample(
            role=role,
            sample_index=index,
            command_sha256=command_sha256,
            stdout=stdout,
            parsed=stdout_payload,
        )
        raw_ref = capture._write_json(
            tmp_path / "raw" / f"{2 * index + offset:02d}-{role}.json",
            raw,
        )
        start = 4 * index + 2 * offset
        started = f"2026-07-26T01:00:{start:02d}+00:00"
        finished = f"2026-07-26T01:00:{start + 1:02d}+00:00"
        sample = {
            "sample_index": index,
            "interleave_round": index,
            "interleave_position": 2 * index + offset,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "subprocess": True,
            "command": command,
            "command_sha256": command_sha256,
            "evaluated_point_count": evaluated,
            "elapsed_seconds": elapsed,
            "seconds_per_point": seconds_per_point,
            "interrupted": False,
            "raw_output_file": raw_ref,
        }
        samples.append(sample)
        projections.append(
            {
                "role": role,
                "round": index,
                "position": 2 * index + offset,
                "started_at_utc": started,
                "finished_at_utc": finished,
                "command_sha256": command_sha256,
            }
        )
    return samples, projections


def test_generated_manifests_pass_the_real_m0_amplicol_validator(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    for name, raw in (
        ("amplicol_library_benchmark", b"selected"),
        ("amplicol_color_probe", b"union"),
        ("libamp_1.so", b"library"),
    ):
        path = repository / name
        path.write_bytes(raw)
        path.chmod(0o755)
    process_file = tmp_path / "processes.txt"
    process_file.write_text("process\n", encoding="utf-8")
    context = replace(
        capture._copy_runtime(repository, tmp_path / "capture", process_file),
        group=11,
        integral=7,
        source_pdgs=(2, -2, 23, 21, 21, 21, 21, 21, 21),
        generated_pdgs=(2, -2, 21, 21, 21, 21, 21, 21, 23),
        generated_color_order=(2, 3, 4, 5, 6, 7, 8, 1, 9),
        permutation=capture.AMPLICOL_EXTERNAL_LEG_PERMUTATION,
    )
    source = _source(tmp_path)
    executable = capture._file_ref(context.worker_executable)
    linked = [capture._file_ref(path) for path in context.linked_files]
    binary_evidence = {
        "executable": executable,
        "linked_libraries": linked,
        "source_files": list(source.source_files),
    }
    selected_values = capture._complex_pairs(contract.selected_values)
    union_values = capture._complex_pairs(contract.union_values)
    selected_samples, selected_projection = _role_samples(
        tmp_path,
        role=capture.SELECTED_ROLE,
        executable=executable,
        fixture=contract.fixture["file"],
        values=selected_values,
    )
    union_samples, union_projection = _role_samples(
        tmp_path,
        role=capture.UNION_ROLE,
        executable=executable,
        fixture=contract.fixture["file"],
        values=union_values,
    )
    evidence_paths = [
        executable["path"],
        *(row["path"] for row in linked),
        *(row["path"] for row in source.source_files),
    ]
    assert len(evidence_paths) == len(set(evidence_paths))
    assert capture._file_ref(context.process_file) in linked
    combined = sorted(
        [*selected_projection, *union_projection],
        key=lambda row: row["position"],
    )
    interleave_sha256 = capture._interleave_group_sha256(combined)
    assert interleave_sha256 == m0._amplicol_interleave_group_sha256(combined)

    validated = {}
    for role, samples, values in (
        (capture.SELECTED_ROLE, selected_samples, contract.selected_values),
        (capture.UNION_ROLE, union_samples, contract.union_values),
    ):
        manifest = capture._manifest(
            role=role,
            contract=contract,
            context=context,
            source=source,
            binary_evidence=binary_evidence,
            samples=samples,
            interleave_group_sha256=interleave_sha256,
            selected=values,
            resolved=values,
        )
        path = tmp_path / f"{role}.json"
        ref = capture._write_json(path, manifest)
        m0_ref = m0._file_ref(ref, base=tmp_path, label=role)
        loaded = m0._load_json_ref(m0_ref, role)
        validated[role] = m0._validate_amplicol(
            loaded,
            role=role,
            expected=contract.expected,
        )

    assert validated[capture.SELECTED_ROLE].timing["timing_boundary"] == (
        "amplitude-evaluation"
    )
    assert validated[capture.UNION_ROLE].timing["timing_boundary"] == (
        "direct-library-total"
    )
    assert (
        validated[capture.SELECTED_ROLE].interleave_group_sha256
        == validated[capture.UNION_ROLE].interleave_group_sha256
        == interleave_sha256
    )


def test_capture_orchestrates_exactly_seven_selected_union_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _write(tmp_path / "request-template.json", b"request\n")
    contract = replace(
        _contract(tmp_path),
        input_files=(request,),
    )
    context = _context(tmp_path)
    source = _source(tmp_path)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(capture, "_contract_from_request", lambda _path: contract)
    monkeypatch.setattr(capture, "_host_identity", lambda: HOST)
    monkeypatch.setattr(
        capture,
        "_require_clean_revision",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(capture, "_prepare_probes", lambda **_kwargs: context)
    monkeypatch.setattr(capture, "_source_evidence", lambda _repository: source)
    monkeypatch.setattr(
        capture.legacy_amplicol,
        "validate_checkout",
        lambda _repository: None,
    )

    def fake_run_sample(
        _command: list[str],
        *,
        role: str,
        round_index: int,
        points: int,
    ) -> tuple[str, dict[str, Any], str, str]:
        calls.append((role, round_index))
        position = len(calls) - 1
        values = (
            contract.selected_values
            if role == capture.SELECTED_ROLE
            else contract.union_values
        )
        elapsed = points * 4.0e-5 * (1.0 + round_index * 0.01)
        payload = {
            "kind": "amplicol-m0-probe-result",
            "schema_version": 1,
            "role": role,
            "sample_index": round_index,
            "evaluated_point_count": points,
            "elapsed_seconds": elapsed,
            "seconds_per_point": elapsed / points,
            "selected_totals": capture._complex_pairs(values),
            "resolved_sums": capture._complex_pairs(values),
        }
        stdout = capture._canonical_bytes(payload).decode() + "\n"
        started = f"2026-07-26T02:00:{2 * position:02d}+00:00"
        finished = f"2026-07-26T02:00:{2 * position + 1:02d}+00:00"
        return stdout, payload, started, finished

    monkeypatch.setattr(capture, "_run_sample_command", fake_run_sample)
    output = tmp_path / "output"
    arguments = SimpleNamespace(
        request_template=Path(request["path"]),
        repository=tmp_path / "repository",
        output_directory=output,
        jobs=1,
        selected_points=1000,
        union_points=2000,
        warmup_points=100,
        target_seconds=5.0,
        minimum_points=100,
        maximum_points=100_000,
    )

    index = capture._capture(arguments)

    assert calls == [
        (role, round_index)
        for round_index in range(capture.MIN_SAMPLES)
        for role in capture.ROLES
    ]
    assert index["complete"] is True
    assert index["paired_round_count"] == capture.MIN_SAMPLES
    assert index["evaluated_points"] == {
        capture.SELECTED_ROLE: 1000,
        capture.UNION_ROLE: 2000,
    }
    for role in capture.ROLES:
        ref = index["amplicol_evidence"][role]
        m0_ref = m0._file_ref(ref, base=output, label=role)
        loaded = m0._load_json_ref(m0_ref, role)
        evidence = m0._validate_amplicol(
            loaded,
            role=role,
            expected=contract.expected,
        )
        assert evidence.interleave_group_sha256 == (index["interleave_group_sha256"])


def test_sample_command_binds_all_required_physical_inputs(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    context = _context(tmp_path)

    selected = capture._sample_command(
        role=capture.SELECTED_ROLE,
        round_index=3,
        points=1234,
        contract=contract,
        context=context,
    )
    union = capture._sample_command(
        role=capture.UNION_ROLE,
        round_index=3,
        points=4321,
        contract=contract,
        context=context,
    )

    for command, role in (
        (selected, capture.SELECTED_ROLE),
        (union, capture.UNION_ROLE),
    ):
        assert command[0] == str(context.worker_executable)
        assert f"--workload={role}" in command
        assert "--round=3" in command
        assert f"--momenta={contract.fixture['path']}" in command
        assert f"--source-revision={REVISION}" in command
        assert f"--color-flow-word={','.join(map(str, FLOW_WORD))}" in command
        assert f"--helicity-values={','.join(map(str, HELICITY_VALUES))}" in command
        assert "--source-to-generated-permutation=0,1,3,4,5,6,7,8,2" in command
    assert f"--color-flow-id={FLOW_ID}" in selected
    assert f"--helicity-id={HELICITY_ID}" in union
    assert capture._canonical_sha256(selected) != capture._canonical_sha256(union)


def test_runtime_launcher_pins_interpreter_producer_and_probe_inputs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for name, raw in (
        ("amplicol_library_benchmark", b"selected"),
        ("amplicol_color_probe", b"union"),
        ("libamp_1.so", b"library"),
    ):
        path = repository / name
        path.write_bytes(raw)
        path.chmod(0o755)
    process_file = tmp_path / "processes.txt"
    process_file.write_text("process\n", encoding="utf-8")

    context = capture._copy_runtime(
        repository,
        tmp_path / "capture",
        process_file,
    )

    launcher = context.worker_executable.read_text(encoding="utf-8")
    bundled_producer = (
        context.runtime
        / "python-src"
        / "tools"
        / "developer"
        / "amplicol_z6g_m0_capture.py"
    ).resolve()
    assert launcher.startswith("#!/bin/sh\nexec ")
    assert str(Path(sys.executable).resolve()) in launcher
    assert " -I -S -B " in launcher
    assert str(bundled_producer) in launcher
    assert str(CAPTURE_SCRIPT) not in launcher
    assert context.worker_executable.stat().st_mode & 0o111
    assert Path(sys.executable).resolve() in context.linked_files
    assert CAPTURE_SCRIPT not in context.linked_files
    for relative in capture._SAMPLE_RUNTIME_RELATIVE_FILES:
        copied = (context.runtime / "python-src" / relative).resolve()
        assert copied.is_file()
        assert copied in context.linked_files
    assert context.process_file in context.linked_files
    assert context.selected_binary in context.linked_files
    assert context.union_binary in context.linked_files
    assert len(context.linked_files) == len(set(context.linked_files))

    completed = subprocess.run(
        (str(context.worker_executable), "_sample", "--help"),
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_sample_runtime_inventory_covers_every_imported_legacy_oracle_module() -> None:
    expected = {
        path.relative_to(ROOT)
        for path in (ROOT / "tools" / "developer" / "legacy_oracle").glob("*.py")
    }
    assert expected <= set(capture._SAMPLE_RUNTIME_RELATIVE_FILES)


def test_copied_helper_tampering_is_detected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for name in (
        "amplicol_library_benchmark",
        "amplicol_color_probe",
        "libamp_1.so",
    ):
        path = repository / name
        path.write_bytes(name.encode())
        path.chmod(0o755)
    process_file = tmp_path / "processes.txt"
    process_file.write_text("process\n", encoding="utf-8")
    context = capture._copy_runtime(repository, tmp_path / "capture", process_file)
    identities = tuple(capture._file_ref(path) for path in context.linked_files)
    helper = (
        context.runtime / "python-src" / "tools" / "developer" / "legacy_amplicol.py"
    )
    helper.write_text(
        helper.read_text(encoding="utf-8") + "\n# adversarial drift\n",
        encoding="utf-8",
    )

    with pytest.raises(capture.CaptureError, match="capture input drifted"):
        capture._verify_immutable_files(identities)


def test_clean_revision_gate_rejects_revision_and_untracked_drift(
    tmp_path: Path,
) -> None:
    repository, revision = _git_repository(tmp_path / "repository")

    capture._require_clean_revision(
        repository,
        expected_revision=revision,
        label="test checkout",
    )
    with pytest.raises(capture.CaptureError, match="revision differs"):
        capture._require_clean_revision(
            repository,
            expected_revision="0" * 40,
            label="test checkout",
        )

    (repository / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(capture.CaptureError, match="tracked or untracked"):
        capture._require_clean_revision(
            repository,
            expected_revision=revision,
            label="test checkout",
        )


def test_prepare_probes_rejects_untracked_checkout_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, revision = _git_repository(tmp_path / "amplicol")
    (repository / "untracked-before-build.txt").write_text(
        "drift\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capture.legacy_amplicol,
        "prepare_checkout",
        lambda _repository: None,
    )
    monkeypatch.setattr(
        capture.legacy_amplicol,
        "expected_revision",
        lambda: revision,
    )

    with pytest.raises(capture.CaptureError, match="tracked or untracked"):
        capture._prepare_probes(
            repository=repository,
            output=tmp_path / "output",
            fixture_path=tmp_path / "not-read-before-cleanliness.json",
            expected_flow_word=FLOW_WORD,
            expected_permutation=capture.AMPLICOL_EXTERNAL_LEG_PERMUTATION,
            jobs=1,
        )


def test_capture_rejects_current_source_revision_and_cleanliness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, revision = _git_repository(tmp_path / "pyamplicol")
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    base_contract = _contract(tmp_path)
    monkeypatch.setattr(capture, "ROOT", repository)
    prepared = False

    def unexpected_prepare(**_kwargs: Any) -> None:
        nonlocal prepared
        prepared = True
        raise AssertionError("probe preparation must not run")

    monkeypatch.setattr(capture, "_prepare_probes", unexpected_prepare)

    def arguments(output: str) -> SimpleNamespace:
        return SimpleNamespace(
            request_template=request,
            repository=tmp_path / "amplicol",
            output_directory=tmp_path / output,
        )

    wrong_expected = dict(base_contract.expected)
    wrong_expected["pyamplicol_source_revision"] = "0" * 40
    monkeypatch.setattr(
        capture,
        "_contract_from_request",
        lambda _path: replace(base_contract, expected=wrong_expected),
    )
    with pytest.raises(capture.CaptureError, match="revision differs"):
        capture._capture(arguments("wrong-revision"))
    assert prepared is False

    clean_expected = dict(base_contract.expected)
    clean_expected["pyamplicol_source_revision"] = revision
    monkeypatch.setattr(
        capture,
        "_contract_from_request",
        lambda _path: replace(base_contract, expected=clean_expected),
    )
    (repository / "untracked-current-source.txt").write_text(
        "drift\n",
        encoding="utf-8",
    )
    with pytest.raises(capture.CaptureError, match="tracked or untracked"):
        capture._capture(arguments("dirty-source"))
    assert prepared is False


def test_process_generation_subprocess_receives_lowercase_z(
    tmp_path: Path,
) -> None:
    process_payload = (
        "9 1\n"
        "2 -2 21 21 21 21 21 21 23\n"
        "\n"
        "\n"
        "1\n"
        "\n"
        "1 1 1 1 9 2 3 4 5 6 7 8\n"
        "1 1 2 -2 21 21 21 21 21 21 23 "
        "2 3 4 5 6 7 8 1 9 720.0\n"
        "\n"
        "\n"
    )
    repository = tmp_path / "case-sensitive-amplicol"
    repository.mkdir()
    process_list = repository / "process_list.py"
    process_list.write_text(
        "\n".join(
            (
                "import json",
                "import sys",
                "from pathlib import Path",
                "arguments = sys.argv[1:]",
                "Path('received.json').write_text(json.dumps(arguments))",
                (f"if arguments != ['--serial', {capture.NORMALIZED_PROCESS!r}]:"),
                "    raise SystemExit(19)",
                f"Path('processes.txt').write_text({process_payload!r})",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    setup = tmp_path / "setup"
    setup.mkdir()

    generated = capture._generate_process_file(
        repository=repository,
        setup=setup,
        log_path=setup / "generation.log",
    )

    assert generated == setup / "processes.txt"
    assert generated.read_text(encoding="utf-8") == process_payload
    assert json.loads((setup / "received.json").read_text(encoding="utf-8")) == [
        "--serial",
        "u u~ > z g g g g g g",
    ]
    assert " Z " in f" {capture.PROCESS_EXPRESSION} "
    source_pdgs = capture.legacy_amplicol.process_pdgs(capture.NORMALIZED_PROCESS)
    entry, matches = capture.legacy_amplicol.select_generated_process_entry(
        capture.legacy_amplicol.parse_process_file(generated),
        generated_process=capture.NORMALIZED_PROCESS,
        wanted_pdgs=source_pdgs,
    )
    assert len(matches) == 1
    assert (
        capture.legacy_amplicol._permutation(
            source_pdgs,
            entry.process_pdgs,
        )
        == capture.AMPLICOL_EXTERNAL_LEG_PERMUTATION
    )
    mapped = capture.legacy_amplicol.source_mapped_color_order(
        entry,
        source_pdgs=source_pdgs,
    )
    colored_labels = {
        index
        for index, pdg in enumerate(source_pdgs, start=1)
        if abs(pdg) == 21 or 1 <= abs(pdg) <= 6
    }
    assert [label for label in mapped if label in colored_labels] == FLOW_WORD


def test_probe_stdout_is_strict_and_identity_bound() -> None:
    payload = {
        "kind": "amplicol-m0-probe-result",
        "schema_version": 1,
        "role": capture.SELECTED_ROLE,
        "sample_index": 2,
        "evaluated_point_count": 100,
        "elapsed_seconds": 0.5,
        "seconds_per_point": 0.005,
        "selected_totals": [[1.0, 0.0]],
        "resolved_sums": [[1.0, 0.0]],
    }
    stdout = capture._canonical_bytes(payload).decode() + "\n"
    assert (
        capture._parse_probe_stdout(
            stdout,
            role=capture.SELECTED_ROLE,
            sample_index=2,
            evaluated_points=100,
        )
        == payload
    )

    payload["unexpected"] = True
    with pytest.raises(capture.CaptureError, match="unknown or missing"):
        capture._parse_probe_stdout(
            json.dumps(payload),
            role=capture.SELECTED_ROLE,
            sample_index=2,
            evaluated_points=100,
        )
    with pytest.raises(capture.CaptureError, match="duplicate JSON key"):
        capture._parse_probe_stdout(
            '{"kind":"x","kind":"y"}',
            role=capture.SELECTED_ROLE,
            sample_index=2,
            evaluated_points=100,
        )


def test_timing_boundaries_and_adaptive_count_fail_closed() -> None:
    output = """\
Timing summary
generation setup 1.25
amplitude evaluation 0.5
total 2.0
"""
    assert capture._timing_value(output, "amplitude evaluation") == 0.5
    assert capture._timing_value(output, "total") == 2.0
    assert (
        capture._adaptive_points(
            0.5,
            warmup_points=100,
            target_seconds=5.0,
            minimum_points=100,
            maximum_points=10_000,
        )
        == 1000
    )
    with pytest.raises(capture.CaptureError, match="positive finite"):
        capture._adaptive_points(
            0.0,
            warmup_points=100,
            target_seconds=5.0,
            minimum_points=100,
            maximum_points=10_000,
        )
    with pytest.raises(capture.CaptureError, match="no unique"):
        capture._timing_value("Timing summary\nother 1.0\n", "total")
