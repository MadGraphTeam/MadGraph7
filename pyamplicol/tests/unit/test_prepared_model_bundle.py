# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from pyamplicol.models import prepared as prepared_module
from pyamplicol.models.prepared import (
    EAGER_KERNEL_ABI,
    PREPARED_MODEL_BUNDLE_KIND,
    PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
    PREPARED_MODEL_COMPILED_MODEL_PATH,
    PREPARED_MODEL_MANIFEST_PATH,
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedModelBundleError,
    load_prepared_model_bundle,
    write_prepared_model_bundle,
)


def _kernel(
    kernel_id: int = 0,
    *,
    signature: str = "vertex:q-qbar-g:v1",
) -> PreparedKernelRecord:
    root = f"kernels/{kernel_id}"
    return PreparedKernelRecord(
        kernel_id=kernel_id,
        contract_kind="vertex",
        canonical_signature=signature,
        input_arity=2,
        output_arity=1,
        input_layout=("left-current", "right-current"),
        input_contracts=(
            {
                "role": "left-current",
                "component": 0,
                "symbol": "pyamplicol::left",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
            {
                "role": "right-current",
                "component": 0,
                "symbol": "pyamplicol::right",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("current-contribution",),
        exact_expressions=("pyamplicol::left+pyamplicol::right",),
        exact_evaluator_state_path=f"{root}/exact.evaluator.bin",
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "input_len": 2,
            "output_len": 1,
            "application_path": f"{root}/application.symjit",
            "evaluator_state_path": f"{root}/exact.evaluator.bin",
        },
    )


def _pack(*kernels: PreparedKernelRecord) -> PreparedKernelPack:
    return PreparedKernelPack(
        backend="jit",
        optimization_settings={"optimization_level": 3, "cpe_rounds": "default"},
        producer={"distribution": "pyamplicol", "version": "0.1.0"},
        dependency_abis={
            "symbolica_serialization": "symbolica-community-v1",
            "symjit_application": "symjit-application-complex-f64-v1",
        },
        provenance={
            "model_content_sha256": "1" * 64,
            "compiler_revision": "2" * 40,
        },
        target={
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "portable-symjit-mir",
            "cpu_features": [],
        },
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
        },
        kernels=kernels or (_kernel(),),
    )


def _compiled_model() -> dict[str, object]:
    return {
        "kind": "pyamplicol-compiled-model",
        "schema_version": 9,
        "model_compiler_version": 13,
        "model": {"name": "test-model"},
    }


def _payloads(*kernels: PreparedKernelRecord) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for kernel in kernels or (_kernel(),):
        result[kernel.exact_evaluator_state_path] = f"exact:{kernel.kernel_id}".encode()
        application_path = str(kernel.f64_evaluator_manifest["application_path"])
        result[application_path] = f"jit:{kernel.kernel_id}".encode()
    return result


def _valid_bundle(tmp_path: Path) -> Path:
    kernel = _kernel()
    return write_prepared_model_bundle(
        tmp_path / "prepared",
        compiled_model=_compiled_model(),
        kernel_pack=_pack(kernel),
        payloads=_payloads(kernel),
    )


def _entries(path: Path) -> list[tuple[zipfile.ZipInfo, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [(info, archive.read(info)) for info in archive.infolist()]


def _rewrite(
    path: Path,
    entries: list[tuple[zipfile.ZipInfo, bytes]],
) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for original, data in entries:
            info = zipfile.ZipInfo(original.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = original.external_attr
            archive.writestr(info, data)


def _mutate_manifest(
    path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    entries = _entries(path)
    updated: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename == PREPARED_MODEL_MANIFEST_PATH:
            manifest = json.loads(data)
            assert isinstance(manifest, dict)
            mutate(manifest)
            data = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("ascii")
        updated.append((info, data))
    _rewrite(path, updated)


def test_prepared_model_bundle_round_trip_and_payload_copy(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    assert path == (tmp_path / "prepared.pyamplicol-model").resolve()
    bundle = load_prepared_model_bundle(path)

    assert bundle.backend == "jit"
    assert bundle.compiled_model == _compiled_model()
    assert bundle.kernel_pack.kernels[0].canonical_signature == "vertex:q-qbar-g:v1"
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"
    with pytest.raises(TypeError):
        bundle.compiled_model["kind"] = "changed"  # type: ignore[index]

    extracted = bundle.copy_referenced_payloads(tmp_path / "extracted")
    extracted_paths = tuple(
        path.relative_to(tmp_path / "extracted").as_posix() for path in extracted
    )
    assert extracted_paths == (
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    )
    assert extracted[0].read_bytes() == b"jit:0"
    assert extracted[1].read_bytes() == b"exact:0"
    assert not (tmp_path / "extracted" / "model").exists()

    with zipfile.ZipFile(path, "r") as archive:
        assert archive.namelist() == [
            PREPARED_MODEL_MANIFEST_PATH,
            "kernels/0/application.symjit",
            "kernels/0/exact.evaluator.bin",
            PREPARED_MODEL_COMPILED_MODEL_PATH,
        ]
        manifest = json.loads(archive.read(PREPARED_MODEL_MANIFEST_PATH))
    assert manifest["kind"] == PREPARED_MODEL_BUNDLE_KIND
    assert manifest["schema_version"] == PREPARED_MODEL_BUNDLE_SCHEMA_VERSION
    assert manifest["eager_kernel_abi"] == EAGER_KERNEL_ABI
    assert manifest["kernel_pack"]["backend"] == "jit"
    assert {record["path"] for record in manifest["members"]} == {
        PREPARED_MODEL_COMPILED_MODEL_PATH,
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    }


def test_payload_reference_index_is_reused_after_bundle_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _valid_bundle(tmp_path)
    bundle = load_prepared_model_bundle(path)
    expected = (
        "kernels/0/application.symjit",
        "kernels/0/exact.evaluator.bin",
    )

    def fail_manifest_traversal(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        raise AssertionError("manifest payload paths were recomputed")

    monkeypatch.setattr(
        prepared_module,
        "_collect_manifest_paths",
        fail_manifest_traversal,
    )
    assert bundle.kernel_pack.kernels[0].referenced_payload_paths == expected
    assert bundle.kernel_pack.kernels[0].referenced_payload_paths == expected

    def fail_kernel_scan(_kernel: PreparedKernelRecord) -> tuple[str, ...]:
        raise AssertionError("kernel payload paths were rescanned")

    monkeypatch.setattr(
        PreparedKernelRecord,
        "referenced_payload_paths",
        property(fail_kernel_scan),
    )
    assert bundle.kernel_pack.referenced_payload_paths == expected
    assert bundle.kernel_pack.referenced_payload_paths == expected
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"
    assert bundle.read_payload("kernels/0/application.symjit") == b"jit:0"


def test_cached_payload_index_keeps_path_and_checksum_validation(
    tmp_path: Path,
) -> None:
    path = _valid_bundle(tmp_path)
    bundle = load_prepared_model_bundle(path)

    with pytest.raises(PreparedModelBundleError, match="normalized relative POSIX"):
        bundle.read_payload("../kernels/0/application.symjit")

    entries = [
        (info, b"bad!!" if info.filename.endswith("application.symjit") else data)
        for info, data in _entries(path)
    ]
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="SHA-256 mismatch"):
        bundle.read_payload("kernels/0/application.symjit")


def test_prepared_model_bundle_bytes_are_deterministic(tmp_path: Path) -> None:
    first_kernel = _kernel(5, signature="closure:q-qbar:v1")
    second_kernel = _kernel(2, signature="vertex:g-g-g:v1")
    pack = _pack(first_kernel, second_kernel)
    payloads = _payloads(first_kernel, second_kernel)

    first = write_prepared_model_bundle(
        tmp_path / "first.pyamplicol-model",
        compiled_model=_compiled_model(),
        kernel_pack=pack,
        payloads=payloads,
    )
    second = write_prepared_model_bundle(
        tmp_path / "second.pyamplicol-model",
        compiled_model=dict(reversed(tuple(_compiled_model().items()))),
        kernel_pack=pack,
        payloads=dict(reversed(tuple(payloads.items()))),
    )

    assert first.read_bytes() == second.read_bytes()
    assert tuple(kernel.kernel_id for kernel in pack.kernels) == (2, 5)


@pytest.mark.parametrize(
    ("payload_path", "pattern"),
    (
        ("../escape.bin", "normalized relative POSIX"),
        ("/absolute.bin", "normalized relative POSIX"),
        ("nested\\windows.bin", "normalized relative POSIX"),
    ),
)
def test_writer_rejects_unsafe_payload_references(
    payload_path: str,
    pattern: str,
) -> None:
    with pytest.raises(PreparedModelBundleError, match=pattern):
        PreparedKernelRecord(
            kernel_id=0,
            contract_kind="closure",
            canonical_signature="closure:test",
            input_arity=1,
            output_arity=1,
            input_layout=("input",),
            input_contracts=(
                {
                    "role": "current",
                    "component": 0,
                    "symbol": "pyamplicol::input",
                    "model_parameter_name": None,
                    "model_parameter_index": None,
                },
            ),
            output_layout=("output",),
            exact_expressions=("pyamplicol::input",),
            exact_evaluator_state_path=payload_path,
            f64_evaluator_manifest={
                "kind": "test",
                "application_path": "kernels/application.bin",
            },
        )


def test_writer_requires_exact_referenced_payload_set(tmp_path: Path) -> None:
    kernel = _kernel()
    payloads = _payloads(kernel)
    del payloads["kernels/0/application.symjit"]
    payloads["kernels/0/unreferenced.bin"] = b"extra"

    with pytest.raises(
        PreparedModelBundleError,
        match=r"missing: kernels/0/application\.symjit.*unreferenced",
    ):
        write_prepared_model_bundle(
            tmp_path / "invalid",
            compiled_model=_compiled_model(),
            kernel_pack=_pack(kernel),
            payloads=payloads,
        )


def test_writer_rejects_symlink_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    link = tmp_path / "payload-link.bin"
    link.symlink_to(source)
    kernel = _kernel()
    payloads: dict[str, bytes | Path] = _payloads(kernel)
    payloads[kernel.exact_evaluator_state_path] = link

    with pytest.raises(PreparedModelBundleError, match="must not be a symlink"):
        write_prepared_model_bundle(
            tmp_path / "invalid",
            compiled_model=_compiled_model(),
            kernel_pack=_pack(kernel),
            payloads=payloads,
        )


def test_reader_rejects_traversal_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    info = zipfile.ZipInfo("../escape.bin")
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    entries.append((info, b"escape"))
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="normalized relative POSIX"):
        load_prepared_model_bundle(path)


def test_reader_rejects_duplicate_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    duplicate = next(
        entry
        for entry in entries
        if entry[0].filename == "kernels/0/application.symjit"
    )
    entries.append(duplicate)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="duplicate member"):
        load_prepared_model_bundle(path)


def test_reader_rejects_symlink_archive_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    updated: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        if info.filename == "kernels/0/application.symjit":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        updated.append((info, data))
    _rewrite(path, updated)

    with pytest.raises(PreparedModelBundleError, match="regular file"):
        load_prepared_model_bundle(path)


def test_reader_rejects_missing_manifest_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = [
        entry
        for entry in _entries(path)
        if entry[0].filename != "kernels/0/application.symjit"
    ]
    _rewrite(path, entries)

    with pytest.raises(
        PreparedModelBundleError,
        match=r"members do not match.*missing",
    ):
        load_prepared_model_bundle(path)


def test_reader_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = [
        (info, b"other" if info.filename.endswith("application.symjit") else data)
        for info, data in _entries(path)
    ]
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="SHA-256 mismatch"):
        load_prepared_model_bundle(path)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    (
        ("kind", "not-pyamplicol", "bundle kind"),
        ("schema_version", 99, "bundle schema"),
        ("eager_kernel_abi", "eager-v99", "kernel ABI"),
    ),
)
def test_reader_rejects_bad_bundle_contract(
    tmp_path: Path,
    field: str,
    value: object,
    pattern: str,
) -> None:
    path = _valid_bundle(tmp_path)
    _mutate_manifest(path, lambda manifest: manifest.__setitem__(field, value))

    with pytest.raises(PreparedModelBundleError, match=pattern):
        load_prepared_model_bundle(path)


def test_reader_rejects_bad_backend(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)

    def mutate(manifest: dict[str, object]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        pack["backend"] = "cuda"

    _mutate_manifest(path, mutate)
    with pytest.raises(PreparedModelBundleError, match="unsupported prepared backend"):
        load_prepared_model_bundle(path)


def test_exactly_one_pack_and_unique_kernel_contracts(tmp_path: Path) -> None:
    first = _kernel(1, signature="same")
    with pytest.raises(PreparedModelBundleError, match="kernel IDs must be unique"):
        _pack(first, _kernel(1, signature="other"))
    with pytest.raises(PreparedModelBundleError, match="signatures must be unique"):
        _pack(first, _kernel(2, signature="same"))

    path = _valid_bundle(tmp_path)

    def replace_singular_pack(manifest: dict[str, object]) -> None:
        pack = manifest.pop("kernel_pack")
        manifest["kernel_packs"] = [pack, pack]

    _mutate_manifest(path, replace_singular_pack)
    with pytest.raises(PreparedModelBundleError, match="missing fields: kernel_pack"):
        load_prepared_model_bundle(path)


def test_reader_rejects_unreferenced_payload_member(tmp_path: Path) -> None:
    path = _valid_bundle(tmp_path)
    entries = _entries(path)
    manifest_index = next(
        index
        for index, (info, _) in enumerate(entries)
        if info.filename == PREPARED_MODEL_MANIFEST_PATH
    )
    manifest = json.loads(entries[manifest_index][1])
    assert isinstance(manifest, dict)
    members = manifest["members"]
    assert isinstance(members, list)
    members.append(
        {
            "path": "kernels/unreferenced.bin",
            "size": 5,
            "sha256": hashlib.sha256(b"extra").hexdigest(),
        }
    )
    entries[manifest_index] = (
        entries[manifest_index][0],
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        ),
    )
    extra = zipfile.ZipInfo("kernels/unreferenced.bin")
    extra.create_system = 3
    extra.external_attr = (stat.S_IFREG | 0o644) << 16
    entries.append((extra, b"extra"))
    _rewrite(path, entries)

    with pytest.raises(PreparedModelBundleError, match="unreferenced"):
        load_prepared_model_bundle(path)


def test_manifest_payload_mapping_is_deeply_immutable() -> None:
    kernel = _kernel()
    manifest = kernel.f64_evaluator_manifest
    assert isinstance(manifest, Mapping)
    with pytest.raises(TypeError):
        manifest["kind"] = "changed"  # type: ignore[index]
