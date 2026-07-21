# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.ci import eager_portability as portability
from tools.ci.eager_portability_contract import symjit_storage_v3_target

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "eager-portability.yml"


@pytest.fixture
def contracts() -> portability.RuntimeContracts:
    return portability.RuntimeContracts(
        bundle_kind="pyamplicol-prepared-model",
        bundle_schema_version=1,
        eager_kernel_abi="pyamplicol-eager-kernel-v1",
        compiled_model_schema_version=9,
        symbolica_serialization_abi="symbolica-bincode2-v1",
        symjit_application_abi="symjit-application-storage-v3",
        symjit_runtime_capability="symjit.application.complex-f64.v1",
        eager_runtime_capability="rusticol.eager-dag.complex-f64.v1",
        package_version="0.1.0.dev0+candidate.portability",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle(
    path: Path,
    contracts: portability.RuntimeContracts,
    *,
    architecture: str = "x86_64",
    mutate: Callable[[dict[str, object], dict[str, bytes]], None] | None = None,
) -> Path:
    application_path = "kernels/000000/application-0.symjit"
    state_path = "kernels/000000/evaluator-state-0.evaluator.bin"
    model_path = "model/model.pyAmplicol-model.json"
    payloads = {
        application_path: b"architecture-scoped-symjit-storage-v3-fixture\x00\x01",
        state_path: b"architecture-neutral-symbolica-state-fixture\x00\x02",
        model_path: json.dumps(
            {
                "schema_version": contracts.compiled_model_schema_version,
                "source": {"kind": "built-in-sm"},
            },
            sort_keys=True,
        ).encode("utf-8"),
    }
    f64_manifest: dict[str, object] = {
        "application_abi": contracts.symjit_application_abi,
        "application_path": application_path,
        "backend": "jit",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "element_layout": "complex-f64",
        "endianness": "little",
        "evaluator_state_path": state_path,
        "kind": "symjit-application-evaluator",
        "optimization_level": 3,
        "required_defuns": [],
        "runtime_capability": contracts.symjit_runtime_capability,
        "settings": {
            "backend": "jit",
            "compiled_inline_asm": "none",
            "compiled_native": False,
            "compiler_path": None,
            "effective_compiler_flags": [],
            "jit_optimization_level": 3,
        },
        "translation_mode": "indirect",
        "word_bits": 64,
    }
    manifest: dict[str, object] = {
        "eager_kernel_abi": contracts.eager_kernel_abi,
        "kernel_pack": {
            "backend": "jit",
            "dependency_abis": {
                "symbolica_serialization": contracts.symbolica_serialization_abi,
                "symjit_application": contracts.symjit_application_abi,
            },
            "kernels": [
                {
                    "f64_evaluator_manifest": f64_manifest,
                    "kernel_id": 0,
                }
            ],
            "kernel_variants": [],
            "optimization_settings": {
                "backend": "jit",
                "compiled_inline_asm": "none",
                "compiled_native": False,
                "compiler_path": None,
                "effective_compiler_flags": [],
                "jit_optimization_level": 3,
            },
            "producer": {
                "compiled_model_schema": contracts.compiled_model_schema_version,
                "distribution": "pyamplicol",
                "model_compiler_version": 13,
                "version": contracts.package_version,
            },
            "provenance": {
                "compiled_model_digest": "a" * 64,
                "model_name": "built-in-sm",
                "model_source": {"kind": "built-in-sm"},
            },
            "target": symjit_storage_v3_target(architecture),
        },
        "kind": contracts.bundle_kind,
        "schema_version": contracts.bundle_schema_version,
    }
    if mutate is not None:
        mutate(manifest, payloads)
    manifest["members"] = [
        {
            "path": member,
            "sha256": _sha256(payload),
            "size": len(payload),
        }
        for member, payload in sorted(payloads.items())
    ]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in {
            "manifest.json": json.dumps(manifest, sort_keys=True).encode("utf-8"),
            **payloads,
        }.items():
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return path


def _write_transfer_fixture(
    directory: Path,
    contracts: portability.RuntimeContracts,
    *,
    architecture: str,
) -> Path:
    directory.mkdir()
    bundle = _write_bundle(
        directory / portability.DEFAULT_BUNDLE_NAME,
        contracts,
        architecture=architecture,
    )
    fixture = {
        "bundle": {
            "architecture_class": architecture,
            "bundle_sha256": portability._sha256_file(bundle),
            "filename": bundle.name,
            "preflight_evaluator_count": 1,
        },
        "expected": {
            "atol": portability.DEFAULT_ATOL,
            "imaginary": 0.0,
            "real": 1.0,
            "rtol": portability.DEFAULT_RTOL,
        },
        "kind": portability.TRANSFER_KIND,
        "process": {
            "expression": portability.DEFAULT_PROCESS,
            "id": portability.DEFAULT_PROCESS_ID,
            "momenta": [],
        },
        "producer": {
            "architecture_class": architecture,
            "git_commit": "fixture-commit",
            "machine": "x86_64" if architecture == "x86_64" else "arm64",
            "system": "Linux",
        },
        "schema_version": portability.TRANSFER_SCHEMA_VERSION,
    }
    (directory / portability.DEFAULT_FIXTURE_NAME).write_text(
        json.dumps(fixture),
        encoding="utf-8",
    )
    return directory


@pytest.mark.parametrize("architecture", ("x86_64", "aarch64"))
def test_architecture_jit_bundle_audit_accepts_matching_storage_v3_pack(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    architecture: str,
) -> None:
    bundle = _write_bundle(
        tmp_path / "model.pyamplicol-model",
        contracts,
        architecture=architecture,
    )

    result = portability.audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_sha256=portability._sha256_file(bundle),
        expected_architecture_class=architecture,
    )

    assert result["backend"] == "jit"
    assert result["kernel_count"] == 1
    assert result["symjit_application_count"] == 1
    assert result["exact_state_count"] == 1
    assert result["architecture_class"] == architecture
    assert result["target"] == symjit_storage_v3_target(architecture)


def test_architecture_jit_bundle_audit_rejects_wrong_storage_abi(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        dependencies = pack["dependency_abis"]
        assert isinstance(dependencies, dict)
        dependencies["symjit_application"] = "symjit-application-storage-v999"

    bundle = _write_bundle(
        tmp_path / "wrong-abi.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="application storage ABI mismatch",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"\x7fELF" + b"\0" * 64, "contains ELF image"),
        (b"!<arch>\n" + b"\0" * 64, "contains static archive"),
        (b"\xcf\xfa\xed\xfe" + b"\0" * 64, "contains Mach-O image"),
    ),
)
def test_architecture_jit_bundle_audit_rejects_native_machine_code(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    payload: bytes,
    message: str,
) -> None:
    def mutate(_manifest: dict[str, object], payloads: dict[str, bytes]) -> None:
        payloads["kernels/000000/application-0.symjit"] = payload

    bundle = _write_bundle(
        tmp_path / "native.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(portability.PortabilityError, match=message):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("member", "payload"),
    (
        ("kernels/000000/generated.cpp", b"double kernel(double x);\n"),
        ("kernels/000000/generated.asm", b"section .text\n"),
        ("kernels/000000/kernel.o", b"opaque native object fixture"),
    ),
)
def test_architecture_jit_bundle_audit_rejects_native_source_or_object_payload(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    member: str,
    payload: bytes,
) -> None:
    def mutate(_manifest: dict[str, object], payloads: dict[str, bytes]) -> None:
        payloads[member] = payload

    bundle = _write_bundle(
        tmp_path / "native-source.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="native/source payload",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


def test_architecture_jit_bundle_audit_rejects_legacy_portable_target(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        target = pack["target"]
        assert isinstance(target, dict)
        target["portable"] = True
        target["target_triple"] = "portable-symjit-mir"

    bundle = _write_bundle(
        tmp_path / "falsely-portable.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="storage v3 must not be marked portable",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


def test_architecture_jit_bundle_audit_rejects_cross_architecture_before_load(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(
        tmp_path / "x86-pack.pyamplicol-model",
        contracts,
        architecture="x86_64",
    )

    with pytest.raises(
        portability.PortabilityError,
        match=(
            "architecture class mismatch: bundle is 'x86_64', consumer is "
            "'aarch64'; refusing before SymJIT load"
        ),
    ):
        portability.audit_architecture_jit_bundle(
            bundle,
            contracts=contracts,
            expected_architecture_class="aarch64",
        )


@pytest.mark.parametrize(
    ("machine", "expected"),
    (("AMD64", "x86_64"), ("x86_64", "x86_64"), ("arm64", "aarch64")),
)
def test_machine_names_normalize_to_storage_v3_architecture_classes(
    machine: str,
    expected: str,
) -> None:
    assert portability.architecture_class(machine) == expected


def test_consumer_generation_command_uses_eager_pack_without_model_compile(
    tmp_path: Path,
) -> None:
    command = portability._generation_command(
        Path("/python"),
        model=tmp_path / "transferred.pyamplicol-model",
        output=tmp_path / "artifact",
        execution_mode="eager",
    )

    assert command[:4] == ["/python", "-m", "pyamplicol", "generate"]
    assert "compile" not in command
    assert command[command.index("--execution-mode") + 1] == "eager"
    assert command[command.index("--model") + 1].endswith(
        "transferred.pyamplicol-model"
    )
    assert "--no-post-build-validation" in command
    assert "--no-emit-api-bundle" in command


def test_python_executable_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "python"
    launcher.symlink_to(Path(sys.executable))

    result = portability._python_executable(launcher)

    assert result == launcher.absolute()
    assert result.is_symlink()


def test_producer_rejects_an_unexpected_architecture_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portability.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portability.platform, "machine", lambda: "aarch64")

    with pytest.raises(
        portability.PortabilityError,
        match=r"producer architecture class.*expected 'x86_64'",
    ):
        portability.produce_transfer(
            tmp_path / "transfer",
            python=Path(sys.executable),
            expected_system="Linux",
            expected_machine="x86_64",
        )

    assert not (tmp_path / "transfer").exists()


def test_consumer_rejects_cross_architecture_before_generation_or_symjit_load(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer",
        contracts,
        architecture="x86_64",
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(portability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(portability.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("cross-architecture preflight started consumer generation")

    def unexpected_load(*_args: object, **_kwargs: object) -> complex:
        pytest.fail("cross-architecture preflight attempted SymJIT-backed evaluation")

    monkeypatch.setattr(portability, "_run", unexpected_run)
    monkeypatch.setattr(portability, "_evaluate_artifact", unexpected_load)

    with pytest.raises(
        portability.PortabilityError,
        match="refusing before SymJIT load",
    ):
        portability.consume_transfer(
            transfer,
            python=Path(sys.executable),
            report_path=report,
            expected_system="Darwin",
            expected_machine="arm64",
        )

    assert not report.exists()


def test_consumer_accepts_same_architecture_pack_across_operating_systems(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer",
        contracts,
        architecture="x86_64",
    )
    monkeypatch.setattr(portability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(portability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(portability, "_git_commit", lambda: "fixture-commit")
    monkeypatch.setattr(
        portability,
        "_preflight_all_prepared_applications",
        lambda _bundle: 1,
    )

    class GenerationStarted(RuntimeError):
        pass

    def stop_at_generation(*_args: object, **_kwargs: object) -> None:
        raise GenerationStarted

    monkeypatch.setattr(portability, "_run", stop_at_generation)

    with pytest.raises(GenerationStarted):
        portability.consume_transfer(
            transfer,
            python=Path(sys.executable),
            report_path=tmp_path / "report.json",
            expected_system="Darwin",
            expected_machine="AMD64",
        )


def test_compiler_guard_records_and_denies_external_tool_execution() -> None:
    with portability.compiler_guard() as (environment, marker):
        completed = subprocess.run(
            ["c++", "--version"],
            check=False,
            capture_output=True,
            env=environment,
        )
        assert completed.returncode == 97
        assert marker.read_text(encoding="utf-8").strip().endswith("c++ --version")


def test_portability_workflow_transfers_matching_architecture_packs() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    trigger = workflow.split("on:\n", maxsplit=1)[1].split(
        "\npermissions:\n", maxsplit=1
    )[0]

    assert "pull_request:" in trigger
    assert "push:" in trigger
    assert "workflow_dispatch:" in trigger
    assert "src/pyamplicol/assets/prepared_models/**" in trigger
    assert "rust/crates/rusticol-core/src/eager_runtime/**" in trigger
    assert "rust/crates/rusticol-core/src/engine/**" in trigger
    assert "rust/crates/rusticol-core/src/evaluator/symjit.rs" in trigger
    assert "dependencies/contributor-lock.toml" in trigger
    assert "src/pyamplicol/evaluators/symbolica*.py" in trigger
    assert workflow.count("eager_portability.py produce") == 1
    assert workflow.count("eager_portability.py consume") == 1
    assert workflow.count("pyamplicol-eager-jit-transfer-${{") == 2
    assert "needs: produce-architecture-packs" in workflow
    assert "ubuntu-24.04" in workflow
    assert "macos-15-intel" in workflow
    assert "macos-15" in workflow
    assert "target: linux-x86-64" in workflow
    assert "target: macos-x86-64" in workflow
    assert "target: macos-arm64" in workflow
    assert workflow.count("pack_architecture: x86_64") == 3
    assert workflow.count("pack_architecture: aarch64") == 2
    assert (
        "target: macos-x86-64\n"
        "            system: Darwin\n"
        "            machine: x86_64\n"
        "            pack_architecture: x86_64"
    ) in workflow
    assert (
        "target: macos-arm64\n"
        "            system: Darwin\n"
        "            machine: arm64\n"
        "            pack_architecture: aarch64"
    ) in workflow
    assert "--expected-system ${{ matrix.system }}" in workflow
    assert "--expected-machine ${{ matrix.machine }}" in workflow
    assert workflow.count("tools/ci/memory_watchdog.py --limit-gib 30 --") == 4
    assert workflow.count("dependencies/install_dependencies.py") == 2
    assert workflow.count("--without-legacy-amplicol") == 2
    assert "contents: read" in workflow
    assert "continue-on-error" not in workflow
    assert "publish-pypi" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "PYAMPLICOL_BUILD_MODE: candidate" in workflow
    assert workflow.count('PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP: "1"') == 2
    assert (
        "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0" in workflow
    )
    assert (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    )


def test_portability_workflow_does_not_expose_credentials() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "persist-credentials: false" in workflow
    assert "id-token: write" not in workflow
    assert "PYPI" not in workflow
    assert "secrets." not in workflow
    assert os.fspath(ROOT) not in workflow
