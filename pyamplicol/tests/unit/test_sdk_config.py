# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import lzma
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import platformdirs
import pytest

import rusticol_config as config


def _sdk(tmp_path: Path) -> Path:
    root = tmp_path / "installed path with spaces" / "_sdk"
    (root / "include").mkdir(parents=True)
    (root / "fortran").mkdir()
    (root / "lib").mkdir()
    (root / "rust").mkdir()
    (root / "include" / "rusticol.h").write_text("", encoding="utf-8")
    (root / "include" / "rusticol.hpp").write_text("", encoding="utf-8")
    (root / "fortran" / "rusticol.f90").write_text("", encoding="utf-8")
    (root / "rust" / "rusticol.rs").write_text("", encoding="utf-8")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"archive")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": "0.1.0",
                "target": "aarch64-apple-darwin",
                "archive": "lib/librusticol_capi.a",
                "rust_source": "rust/rusticol.rs",
            }
        ),
        encoding="utf-8",
    )
    (root / "link.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "aarch64-apple-darwin",
                "system_libraries": ["System", "m"],
                "frameworks": ["Security"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_sdk_info_returns_a_complete_native_link_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    info = config.load_sdk_info()
    assert info.library == root / "lib" / "librusticol_capi.a"
    assert info.rust_source == root / "rust" / "rusticol.rs"
    assert info.link_flags == (
        str(root / "lib" / "librusticol_capi.a"),
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    )
    assert info.rust_flags == tuple(
        token
        for link_flag in info.link_flags
        for token in ("-C", f"link-arg={link_flag}")
    )
    assert info.to_json()["rust_flags"] == list(info.rust_flags)
    assert info.to_json()["rust_source"] == str(info.rust_source)
    assert info.cargo_encoded_rust_flags.split("\x1f") == list(info.rust_flags)


def test_shipped_shared_libraries_are_linked_by_path_with_a_run_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    names = ("libmpc.3.dylib", "libmpfr.6.dylib", "libgmp.10.dylib")
    for name in names:
        (root / "lib" / name).write_bytes(b"dylib")
    link = json.loads((root / "link.json").read_text(encoding="utf-8"))
    link["shared_libraries"] = [f"lib/{name}" for name in names]
    (root / "link.json").write_text(json.dumps(link), encoding="utf-8")
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    info = config.load_sdk_info()
    lib = root / "lib"
    assert info.link_flags == (
        str(lib / "librusticol_capi.a"),
        *(str(lib / name) for name in names),
        f"-Wl,-rpath,{lib}",
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    )
    assert info.to_json()["shared_libraries"] == [str(lib / name) for name in names]

    link["shared_libraries"] = ["../../outside.dylib"]
    (root / "link.json").write_text(json.dumps(link), encoding="utf-8")
    with pytest.raises(config.SdkUnavailableError, match="escapes"):
        config.load_sdk_info()


def test_rustflags_cli_emits_shell_safe_rustc_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    assert config.main(["--rustflags"]) == 0
    output = capsys.readouterr().out.strip()
    assert shlex.split(output) == list(config.load_sdk_info().rust_flags)
    assert str(root / "lib" / "librusticol_capi.a") in output

    assert config.main(["--rust-source"]) == 0
    assert capsys.readouterr().out.strip() == str(root / "rust" / "rusticol.rs")

    assert config.main(["--cargo-rustflags"]) == 0
    encoded = capsys.readouterr().out.rstrip("\n")
    assert encoded.split("\x1f") == list(config.load_sdk_info().rust_flags)


@pytest.mark.parametrize("field", ("archive", "rust_source"))
def test_sdk_metadata_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root = _sdk(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata[field] = "../../outside"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    with pytest.raises(config.SdkUnavailableError, match="escapes"):
        config.load_sdk_info()


def test_sdk_metadata_requires_exact_version_and_uses_wheel_record_for_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(
        config,
        "_package_version",
        lambda: "0.1.0.dev0+candidate.deadbeef",
    )
    with pytest.raises(config.SdkUnavailableError, match="version"):
        config.load_sdk_info()

    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"tampered")
    info = config.load_sdk_info()
    assert info.library.read_bytes() == b"tampered"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert "archive_sha256" not in metadata


@pytest.fixture
def standalone_sdk(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A wheel-shaped SDK whose package initializer must never be executed."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    site = tmp_path / "site packages"
    package = site / "pyamplicol"
    shutil.copytree(_sdk(tmp_path), package / "_sdk")
    (package / "__init__.py").write_text(
        "raise RuntimeError('numerical package initialization is forbidden')\n",
        encoding="utf-8",
    )
    (site / "symbolica.py").write_text(
        "raise RuntimeError('Symbolica initialization is forbidden')\n",
        encoding="utf-8",
    )
    (package / "_internal").mkdir()
    shutil.copyfile(
        source_root / "pyamplicol" / "_internal" / "versions.py",
        package / "_internal" / "versions.py",
    )
    (package / "_build_info.json").write_text(
        json.dumps({"version": "0.1.0", "publishable": True}),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(site), str(source_root)))
    return package, environment


@pytest.mark.parametrize("invocation", ("console", "module"))
def test_standalone_sdk_query_does_not_initialize_numerical_packages(
    standalone_sdk: tuple[Path, dict[str, str]], invocation: str
) -> None:
    package, environment = standalone_sdk
    command = (
        [sys.executable, "-m", "rusticol_config"]
        if invocation == "module"
        else [
            sys.executable,
            "-c",
            "import sys; from rusticol_config import main; result = main(); "
            "assert 'pyamplicol' not in sys.modules; "
            "assert 'symbolica' not in sys.modules; raise SystemExit(result)",
        ]
    )
    result = subprocess.run(
        [*command, "--json"],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert payload["package_version"] == "0.1.0"
    assert payload["library"] == str(package / "_sdk/lib/librusticol_capi.a")
    assert payload["link_flags"] == [
        payload["library"],
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    ]


def test_standalone_sdk_retains_authoritative_version_validation(
    standalone_sdk: tuple[Path, dict[str, str]],
) -> None:
    package, environment = standalone_sdk
    (package / "_build_info.json").write_text(
        json.dumps({"version": None, "publishable": True}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-m", "rusticol_config", "--version"],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "wheel build version provenance is invalid" in result.stderr
    assert "initialization is forbidden" not in result.stderr


def test_sdk_console_entry_and_package_are_declared() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["rusticol-config"] == "rusticol_config:main"
    assert "rusticol_config" in project["tool"]["maturin"]["python-packages"]


def test_existing_sdk_import_forwards_to_the_same_implementation() -> None:
    source = Path(__file__).resolve().parents[2] / "src/pyamplicol/_sdk/config.py"
    compatibility = runpy.run_path(str(source))
    assert compatibility["main"] is config.main
    assert compatibility["load_sdk_info"] is config.load_sdk_info
    assert compatibility["SdkInfo"] is config.SdkInfo


# Distribution wheels ship the archive xz-compressed; rusticol-config unpacks
# it on first use into the user cache (PYAMPLICOL_CACHE_DIR overrides it).
_ARCHIVE = b"!<arch>\n" + bytes(range(256)) * 4096


def _compressed_sdk(tmp_path: Path, archive: bytes = _ARCHIVE) -> Path:
    root = _sdk(tmp_path)
    (root / "lib" / "librusticol_capi.a").unlink()
    (root / "lib" / "librusticol_capi.a.xz").write_bytes(
        lzma.compress(archive, format=lzma.FORMAT_XZ)
    )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["archive"] = "lib/librusticol_capi.a.xz"
    metadata["archive_compression"] = {
        "format": "xz",
        "sha256": hashlib.sha256(archive).hexdigest(),
        "size": len(archive),
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


@pytest.fixture
def compressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = _compressed_sdk(tmp_path)
    cache = tmp_path / "cache root"
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(cache))
    return root, cache


def _expected_cache_path(cache: Path, archive: bytes = _ARCHIVE) -> Path:
    digest = hashlib.sha256(archive).hexdigest()
    return cache / "sdk" / f"0.1.0-{digest}" / "librusticol_capi.a"


def test_compressed_archive_is_unpacked_on_first_use_and_reused(
    compressed: tuple[Path, Path],
) -> None:
    root, cache = compressed
    expected = _expected_cache_path(cache)
    info = config.load_sdk_info()
    # Metadata-only queries never unpack.
    assert info.library_name == "librusticol_capi.a"
    assert info.cflags == (f"-I{root / 'include'}",)
    assert not cache.exists()

    assert info.library == expected
    assert expected.read_bytes() == _ARCHIVE
    assert info.link_flags[0] == str(expected)
    assert info.to_json()["library"] == str(expected)
    assert sorted(path.name for path in expected.parent.iterdir()) == [
        "librusticol_capi.a"
    ]

    # Warm cache: the checked copy is reused, not rewritten.
    before = expected.stat()
    assert config.load_sdk_info().library == expected
    after = expected.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_cached_copy_failing_its_check_is_replaced(
    compressed: tuple[Path, Path],
) -> None:
    _, cache = compressed
    expected = _expected_cache_path(cache)
    expected.parent.mkdir(parents=True)
    for damaged in (_ARCHIVE[:-1], _ARCHIVE[:-1] + b"x"):
        expected.write_bytes(damaged)
        assert config.load_sdk_info().library == expected
        assert expected.read_bytes() == _ARCHIVE


def test_cache_is_keyed_by_version_and_archive_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(cache))
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")
    other = _ARCHIVE + b"other build"
    paths = []
    for name, archive in (("a", _ARCHIVE), ("b", other)):
        root = _compressed_sdk(tmp_path / name, archive)
        monkeypatch.setattr(config, "_resource_root", lambda root=root: root)
        paths.append(config.load_sdk_info().library)
        assert paths[-1].read_bytes() == archive
    assert paths == [_expected_cache_path(cache), _expected_cache_path(cache, other)]


def test_default_cache_is_the_platform_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYAMPLICOL_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        platformdirs, "user_cache_path", lambda name: tmp_path / "platform" / name
    )
    assert config.sdk_cache_root() == tmp_path / "platform" / "pyamplicol" / "sdk"
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", "~/override")
    assert config.sdk_cache_root() == Path("~/override").expanduser() / "sdk"


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"sha256": "0" * 64}, "does not unpack to the recorded size and SHA-256"),
        ({"size": 1}, "does not unpack to the recorded size and SHA-256"),
        ({"format": "zstd"}, "invalid compressed SDK archive metadata"),
        ({"size": True}, "invalid compressed SDK archive metadata"),
        ({"sha256": "ABC"}, "invalid compressed SDK archive metadata"),
    ),
)
def test_compressed_archive_metadata_and_digest_are_checked(
    compressed: tuple[Path, Path], change: dict[str, object], message: str
) -> None:
    root, cache = compressed
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["archive_compression"].update(change)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(config.SdkUnavailableError, match=message):
        config.load_sdk_info().archive.resolve()
    # Nothing unverified is published, and no partial file is left behind.
    published = [path for path in cache.rglob("*") if path.is_file()]
    assert published == []


def test_corrupt_compressed_archive_is_refused(
    compressed: tuple[Path, Path],
) -> None:
    root, _ = compressed
    (root / "lib" / "librusticol_capi.a.xz").write_bytes(b"\xfd7zXZ\x00 broken")
    with pytest.raises(config.SdkUnavailableError, match="is corrupt"):
        config.load_sdk_info().archive.resolve()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_read_only_installation_and_unwritable_cache(
    compressed: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = compressed
    modes = {path: path.stat().st_mode for path in (root, root / "lib")}
    for path in modes:
        path.chmod(0o555)
    try:
        assert config.load_sdk_info().library.read_bytes() == _ARCHIVE
        blocked = cache.parent / "blocked"
        blocked.mkdir()
        blocked.chmod(0o555)
        monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(blocked / "cache"))
        with pytest.raises(config.SdkUnavailableError, match="PYAMPLICOL_CACHE_DIR"):
            config.load_sdk_info().archive.resolve()
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def test_concurrent_first_uses_publish_one_complete_archive(
    compressed: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both first uses find no cached copy and unpack into their own files;
    # neither renames before both are complete.
    _, cache = compressed
    barrier = threading.Barrier(2, timeout=60)
    replace = os.replace

    def synchronized_replace(source, destination) -> None:
        assert Path(source).read_bytes() == _ARCHIVE
        barrier.wait()
        replace(source, destination)

    monkeypatch.setattr(config.os, "replace", synchronized_replace)
    results: list[Path] = []
    errors: list[BaseException] = []

    def first_use() -> None:
        try:
            results.append(config.load_sdk_info().library)
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=first_use) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    expected = _expected_cache_path(cache)
    assert results == [expected, expected]
    assert expected.read_bytes() == _ARCHIVE
    assert [path.name for path in expected.parent.iterdir()] == ["librusticol_capi.a"]


def test_standalone_compressed_sdk_cli_unpacks_into_the_override(
    standalone_sdk: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    package, environment = standalone_sdk
    shutil.rmtree(package / "_sdk")
    shutil.copytree(_compressed_sdk(tmp_path / "compressed"), package / "_sdk")
    cache = tmp_path / "cli cache"
    environment["PYAMPLICOL_CACHE_DIR"] = str(cache)
    expected = _expected_cache_path(cache)

    def query(*arguments: str) -> str:
        return subprocess.run(
            [sys.executable, "-m", "rusticol_config", *arguments],
            cwd=package.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()

    assert query("--cflags") == shlex.join((f"-I{package / '_sdk/include'}",))
    assert not cache.exists()
    assert query("--library") == str(expected)
    assert shlex.split(query("--libs"))[0] == str(expected)
    assert json.loads(query("--json"))["library"] == str(expected)
    assert expected.read_bytes() == _ARCHIVE
