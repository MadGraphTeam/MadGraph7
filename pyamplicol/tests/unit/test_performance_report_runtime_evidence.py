# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report import runtime_evidence


def _isolate_bytecode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix = tmp_path / "absent-cache-prefix"
    assert not prefix.exists()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    monkeypatch.setattr(
        runtime_evidence,
        "_isolated_startup_flags",
        lambda: (True, True, True),
    )


def test_source_only_bytecode_policy_requires_both_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runtime_evidence,
        "_isolated_startup_flags",
        lambda: (True, True, True),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    with pytest.raises(runtime_evidence.RuntimeEvidenceError, match="Python -B"):
        runtime_evidence.source_only_bytecode_policy()

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="PYTHONPYCACHEPREFIX",
    ):
        runtime_evidence.source_only_bytecode_policy()

    prefix = tmp_path / "populated"
    prefix.mkdir()
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    with pytest.raises(runtime_evidence.RuntimeEvidenceError, match="remain absent"):
        runtime_evidence.source_only_bytecode_policy()


def test_package_identity_is_stable_and_ignores_ineligible_cache_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    package = tmp_path / "pyamplicol"
    cache = package / "__pycache__"
    submodule = package / "submodule"
    cache.mkdir(parents=True)
    submodule.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (submodule / "data.bin").write_bytes(b"\x00\x01")
    (cache / "__init__.cpython-test.pyc").write_bytes(b"ineligible")

    first = runtime_evidence.python_package_tree_identity(package)
    second = runtime_evidence.python_package_tree_identity(package)

    assert first == second
    assert first["kind"] == "pyamplicol-python-package-tree-v2"
    assert first["file_count"] == 2
    assert first["member_set_stable"] is True
    assert first["namespace_bound_to_root_fd"] is True
    assert first["bytecode_policy"] == {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }

    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert runtime_evidence.python_package_tree_identity(package)["sha256"] != first[
        "sha256"
    ]


def test_package_identity_rejects_sourceless_bytecode_and_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    package = tmp_path / "pyamplicol"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    bytecode = package / "injected.pyc"
    bytecode.write_bytes(b"eligible sourceless bytecode")
    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="sourceless bytecode",
    ):
        runtime_evidence.python_package_tree_identity(package)

    bytecode.unlink()
    target = package / "target.py"
    target.write_bytes(b"VALUE = 1\n")
    (package / "alias.py").symlink_to(target)
    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="not a regular file",
    ):
        runtime_evidence.python_package_tree_identity(package)


def test_package_identity_binds_ordered_shadowing_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    first = tmp_path / "first" / "pyamplicol"
    second = tmp_path / "second" / "pyamplicol"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "__init__.py").write_text("ORIGIN = 'first'\n", encoding="utf-8")
    (second / "__init__.py").write_text("ORIGIN = 'second'\n", encoding="utf-8")

    forward = runtime_evidence.python_package_tree_identity((first, second))
    reverse = runtime_evidence.python_package_tree_identity((second, first))

    assert forward["roots"] == [str(first.resolve()), str(second.resolve())]
    assert reverse["roots"] == [str(second.resolve()), str(first.resolve())]
    assert forward["sha256"] != reverse["sha256"]


def _module(
    origin: Path,
    *,
    cached: Path | None = None,
    namespace: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        __file__=None if namespace else str(origin),
        __cached__=None if cached is None else str(cached),
        __spec__=SimpleNamespace(
            origin=None if namespace else str(origin),
            submodule_search_locations=[str(origin)] if namespace else None,
        ),
    )


def test_loaded_origins_are_exact_authenticated_members(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    package = tmp_path / "pyamplicol"
    package.mkdir()
    package_init = package / "__init__.py"
    package_init.write_text("VALUE = 1\n", encoding="utf-8")
    cache = package / "__pycache__" / "__init__.cpython-test.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"ordinary pip-created bytecode cache")
    native = package / "_rusticol.so"
    native.write_bytes(b"native")
    expected_tree = runtime_evidence.python_package_tree_identity(package)
    expected_native = runtime_evidence._path_file_identity(native)
    fake_sys = SimpleNamespace(
        dont_write_bytecode=True,
        pycache_prefix=sys.pycache_prefix,
        modules={
            "pyamplicol": _module(package_init, cached=cache),
            "pyamplicol._rusticol": _module(native),
        },
    )
    monkeypatch.setattr(runtime_evidence, "sys", fake_sys)

    policy = runtime_evidence.loaded_pyamplicol_origin_policy(
        package,
        native_extension=native,
        expected_package_identity=expected_tree,
        expected_native_identity=expected_native,
    )

    assert policy["observed_module_count"] == 2
    assert len(policy["observations"]) == 2
    assert len(policy["observations_sha256"]) == 64


def test_loaded_origins_reject_actual_sourceless_bytecode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    package = tmp_path / "pyamplicol"
    package.mkdir()
    package_init = package / "__init__.py"
    package_init.write_text("VALUE = 1\n", encoding="utf-8")
    native = package / "_rusticol.so"
    native.write_bytes(b"native")
    expected_tree = runtime_evidence.python_package_tree_identity(package)
    expected_native = runtime_evidence._path_file_identity(native)
    bytecode = package / "__pycache__" / "injected.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"sourceless bytecode")
    fake_sys = SimpleNamespace(
        dont_write_bytecode=True,
        pycache_prefix=sys.pycache_prefix,
        modules={"pyamplicol.injected": _module(bytecode)},
    )
    monkeypatch.setattr(runtime_evidence, "sys", fake_sys)

    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="used sourceless bytecode",
    ):
        runtime_evidence.loaded_pyamplicol_origin_policy(
            package,
            native_extension=native,
            expected_package_identity=expected_tree,
            expected_native_identity=expected_native,
        )


def test_loaded_origins_reject_unrecorded_and_excluded_members(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_bytecode(monkeypatch, tmp_path)
    package = tmp_path / "pyamplicol"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    package_init = package / "__init__.py"
    package_init.write_text("VALUE = 1\n", encoding="utf-8")
    native = package / "_rusticol.so"
    native.write_bytes(b"native")
    expected_tree = runtime_evidence.python_package_tree_identity(package)
    expected_native = runtime_evidence._path_file_identity(native)
    injected = package / "injected.py"
    injected.write_text("VALUE = 2\n", encoding="utf-8")
    fake_sys = SimpleNamespace(
        dont_write_bytecode=True,
        pycache_prefix=sys.pycache_prefix,
        modules={"pyamplicol": _module(injected)},
    )
    monkeypatch.setattr(runtime_evidence, "sys", fake_sys)
    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="preimport identity",
    ):
        runtime_evidence.loaded_pyamplicol_origin_policy(
            package,
            native_extension=native,
            expected_package_identity=expected_tree,
            expected_native_identity=expected_native,
        )

    injected.unlink()
    cached_source = cache / "injected.py"
    cached_source.write_text("VALUE = 3\n", encoding="utf-8")
    fake_sys.modules = {"pyamplicol": _module(cached_source)}
    with pytest.raises(
        runtime_evidence.RuntimeEvidenceError,
        match="excluded cache bytes",
    ):
        runtime_evidence.loaded_pyamplicol_origin_policy(
            package,
            native_extension=native,
            expected_package_identity=expected_tree,
            expected_native_identity=expected_native,
        )
