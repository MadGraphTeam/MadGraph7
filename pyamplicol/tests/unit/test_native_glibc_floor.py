# SPDX-License-Identifier: 0BSD
"""The native libraries' glibc floor and the load-time platform check.

The loop engine's release natives may need at most glibc 2.42 (the documented
floor: ``GLIBC_ABI_GNU2_TLS``, TLS descriptors, defined by glibc 2.43 and by
glibc 2.42 builds with the stable-branch update that added it). A toolchain
change that raises it fails
``test_staged_natives_stay_within_the_documented_glibc_floor`` instead of users.
When the running glibc lacks a version a library needs (an older release, or an
ABI marker its libc.so.6 does not define), ``import pyamplicol._rusticol``
raises ``NativePlatformError`` before the library is opened, and a loader error
naming a missing glibc version becomes the same error.
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol import _native_loader as native_loader
from pyamplicol.api.errors import DependencyError, NativePlatformError

pytestmark = pytest.mark.loop_fast

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTED_FLOOR = (2, 42)
NATIVES = ("_rusticol.so", "librusticol_loop_sdk.so")
# ld.so's text for the slice's SDK library under glibc 2.40 (2026-10-28; the
# libc path, a Nix store path there, shortened).
LOADER_MESSAGE = (
    "/lib64/libc.so.6: version `GLIBC_ABI_GNU2_TLS' not found (required by "
    "/opt/pyamplicol/librusticol_loop_sdk.so)"
)


def _staged_build() -> Path:
    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    build = Path(json.loads(info.read_text())["native_build"]["directory"])
    missing = [name for name in NATIVES if not (build / name).is_file()]
    if missing:
        pytest.fail(f"{build} has no {', '.join(missing)}")
    return build


def _assert_within_floor(path: Path) -> None:
    names = native_loader.glibc_version_requirements(path)
    assert names, f"{path} declares no glibc version requirement"
    unknown = [name for name in names if native_loader.glibc_release(name) is None]
    assert not unknown, (
        f"{path} requires glibc versions of unknown release {unknown}: add them "
        "to pyamplicol._native_loader.GLIBC_ABI_MARKERS"
    )
    floor = native_loader.glibc_floor(names)
    assert floor is not None
    release, name = floor
    assert release <= DOCUMENTED_FLOOR, (
        f"{path} needs glibc {'.'.join(map(str, release))} ({name}), above the "
        "documented floor 2.42: update DOCUMENTED_FLOOR, docs/user/installation.md, "
        "the SDK README and headers, and DESIGN §12.1"
    )


def _readelf_requirements(path: Path) -> tuple[str, ...]:
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.skip("readelf (binutils) is not on PATH: no independent ELF reader")
    output = subprocess.run(
        [readelf, "-V", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names: set[str] = set()
    needs = False
    for line in output.splitlines():
        if line.startswith("Version "):
            needs = line.startswith("Version needs section")
        elif needs and "Name: GLIBC_" in line:
            names.add(line.split("Name: ", 1)[1].split()[0])
    return tuple(sorted(names))


def _patched(source: Path, destination: Path, old: str, new: str) -> Path:
    """A copy of ``source`` whose ELF version requirement ``old`` reads ``new``."""

    assert len(new) <= len(old)
    data = source.read_bytes()
    needle = b"\0" + old.encode() + b"\0"
    offset = data.index(needle) + 1
    replacement = new.encode().ljust(len(old), b"\0")
    destination.write_bytes(
        data[:offset] + replacement + data[offset + len(replacement) :]
    )
    assert new in (native_loader.glibc_version_requirements(destination) or ())
    return destination


def test_staged_natives_stay_within_the_documented_glibc_floor() -> None:
    build = _staged_build()
    for name in NATIVES:
        _assert_within_floor(build / name)


def test_elf_requirements_match_readelf(tmp_path) -> None:
    build = _staged_build()
    raised = _patched(
        build / "librusticol_loop_sdk.so",
        tmp_path / "raised.so",
        "GLIBC_2.35",
        "GLIBC_2.43",
    )
    for path in (*(build / name for name in NATIVES), raised):
        assert native_loader.glibc_version_requirements(path) == (
            _readelf_requirements(path)
        )


def test_the_floor_pin_fails_on_a_raised_or_unknown_requirement(tmp_path) -> None:
    sdk = _staged_build() / "librusticol_loop_sdk.so"
    raised = _patched(sdk, tmp_path / "raised.so", "GLIBC_2.35", "GLIBC_2.43")
    with pytest.raises(AssertionError, match=r"needs glibc 2\.43 \(GLIBC_2\.43\)"):
        _assert_within_floor(raised)
    unknown = _patched(sdk, tmp_path / "unknown.so", "GLIBC_2.35", "GLIBC_NEW")
    with pytest.raises(AssertionError, match=r"unknown release \[.GLIBC_NEW.\]"):
        _assert_within_floor(unknown)


def test_a_library_above_the_running_glibc_is_refused_before_loading(
    tmp_path,
) -> None:
    """A real ELF requirement above this host's glibc, with no faked version."""

    running = native_loader.running_glibc()
    assert running is not None, "this test host runs glibc"
    major, minor = map(int, running.split(".")[:2])
    above = f"GLIBC_{major}.{minor + 1}"
    library = _patched(
        _staged_build() / "librusticol_loop_sdk.so",
        tmp_path / "above.so",
        "GLIBC_2.35",
        above,
    )
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    error = refused.value
    assert isinstance(error, DependencyError)
    assert not isinstance(error, ImportError)
    assert (error.library, error.required, error.found, error.version_name) == (
        str(library),
        f"{major}.{minor + 1}",
        running,
        above,
    )
    assert f"requires glibc >= {major}.{minor + 1}" in str(error)
    assert f"this process runs glibc {running}" in str(error)
    assert "docs/user/installation.md" in str(error)

    # The dynamic loader refuses the same file; its error becomes the same type.
    with pytest.raises(OSError, match=f"version `{above}' not found") as loader:
        ctypes.CDLL(str(library))
    translated = native_loader.translate_loader_error(loader.value, library)
    assert isinstance(translated, NativePlatformError)
    assert (translated.library, translated.required, translated.version_name) == (
        str(library),
        f"{major}.{minor + 1}",
        above,
    )
    assert "The dynamic loader reported" in str(translated)


def test_an_older_faked_glibc_refuses_the_rusticol_import_before_dlopen() -> None:
    """``import pyamplicol._rusticol`` below its floor: typed error, no mapping."""

    extension = importlib.machinery.PathFinder.find_spec(
        "_rusticol", [str(ROOT / "src" / "pyamplicol")]
    )
    assert extension is not None and extension.origin is not None
    floor = native_loader.glibc_floor(
        native_loader.glibc_version_requirements(extension.origin) or ()
    )
    assert floor is not None
    (major, minor, *_), version_name = floor
    older = f"{major}.{minor - 1}"
    script = textwrap.dedent(
        f"""
        import os, sys
        real = os.confstr
        os.confstr = lambda name: (
            "glibc {older}" if name == "CS_GNU_LIBC_VERSION" else real(name)
        )
        from pyamplicol import NativePlatformError
        try:
            import pyamplicol._rusticol
        except NativePlatformError as error:
            mapped = "_rusticol" in open("/proc/self/maps").read()
            print(repr((error.required, error.found, error.version_name, mapped,
                        "pyamplicol._rusticol" in sys.modules)))
            print(error)
        else:
            raise SystemExit("the import was not refused")
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(ROOT / "src"), environment.get("PYTHONPATH")) if entry
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary, message = completed.stdout.splitlines()[:2]
    expected = (f"{major}.{minor}", older, version_name, False, False)
    assert summary == repr(expected)
    assert message.startswith(f"Native library {extension.origin} requires glibc")
    assert f"but this process runs glibc {older}." in message


def test_loader_version_errors_become_the_typed_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(native_loader, "running_glibc", lambda: "2.40")
    error = native_loader.translate_loader_error(ImportError(LOADER_MESSAGE))
    assert isinstance(error, NativePlatformError)
    assert error.library == "/opt/pyamplicol/librusticol_loop_sdk.so"
    assert (error.required, error.found, error.version_name) == (
        "2.42",
        "2.40",
        "GLIBC_ABI_GNU2_TLS",
    )
    assert "requires glibc >= 2.42 (it needs the ELF version GLIBC_ABI_GNU2_TLS, " in (
        str(error)
    )
    assert "glibc 2.43 or newer, or an updated glibc 2.42 build" in str(error)
    assert "but this process runs glibc 2.40. Run it on" in str(error)

    # A glibc 2.42 without the stable-branch update: the release is not older,
    # this build lacks the marker. No "requires >= 2.42 but runs 2.42".
    monkeypatch.setattr(native_loader, "running_glibc", lambda: "2.42")
    lacking = native_loader.translate_loader_error(ImportError(LOADER_MESSAGE))
    assert lacking is not None
    assert (lacking.required, lacking.found) == ("2.42", "2.42")
    text = str(lacking)
    assert text.startswith(
        "Native library /opt/pyamplicol/librusticol_loop_sdk.so needs the ELF "
        "version GLIBC_ABI_GNU2_TLS, which glibc 2.43 and newer define, as do "
        "glibc 2.42 builds with the stable-branch update that added it, but not "
        "the glibc 2.42 release, but this process runs glibc 2.42, which does "
        "not define it. Update this system's glibc (glibc 2.43 or newer, or an "
        "updated glibc 2.42 build)"
    ), text
    assert "requires glibc >= 2.42" not in text

    # A third-party library (Symbolica's extension): its own build is the remedy.
    dependency = native_loader.translate_loader_error(
        ImportError(
            "/lib/libc.so.6: version `GLIBC_2.43' not found (required by "
            "/site-packages/symbolica/core.abi3.so)"
        )
    )
    assert dependency is not None
    assert dependency.library == "/site-packages/symbolica/core.abi3.so"
    assert "Install a build of that library made for this glibc" in str(dependency)
    assert "pyAmpliCol's native libraries" not in str(dependency)
    numbered = native_loader.translate_loader_error(
        OSError("libc.so.6: version `GLIBC_2.43' not found (required by x.so)")
    )
    assert numbered is not None and numbered.required == "2.43"
    unknown = native_loader.translate_loader_error(
        OSError("libc.so.6: version `GLIBC_ABI_NEW' not found (required by x.so)")
    )
    assert unknown is not None and unknown.required is None
    assert "a glibc that provides the ELF version GLIBC_ABI_NEW" in str(unknown)
    assert native_loader.translate_loader_error(ImportError("no module x")) is None

    # Through the guarded extension loader: the pre-check passes (the library
    # needs nothing), the loader fails with its glibc message.
    library = tmp_path / "plain.so"
    library.write_bytes(b"not an ELF file")
    raw = ImportError(LOADER_MESSAGE)

    def fail(self, spec):
        raise raw

    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, "create_module", fail)
    loader = native_loader._GuardedExtensionFileLoader("plain", str(library))
    spec = importlib.machinery.ModuleSpec("plain", loader, origin=str(library))
    with pytest.raises(NativePlatformError) as translated:
        loader.create_module(spec)
    assert translated.value.__cause__ is raw
    other = ImportError("undefined symbol: foo")

    def fail_otherwise(self, spec):
        raise other

    monkeypatch.setattr(
        importlib.machinery.ExtensionFileLoader, "create_module", fail_otherwise
    )
    with pytest.raises(ImportError) as untouched:
        loader.create_module(spec)
    assert untouched.value is other


def test_a_c_library_that_is_not_glibc_is_named(monkeypatch) -> None:
    sdk = _staged_build() / "librusticol_loop_sdk.so"
    monkeypatch.setattr(native_loader, "running_glibc", lambda: None)
    with pytest.raises(NativePlatformError, match="does not run on glibc") as error:
        native_loader.check_native_library(sdk)
    assert error.value.found is None
    assert error.value.required is not None


def test_abi_markers_are_decided_by_the_running_libc_definitions(
    monkeypatch,
) -> None:
    """A marker missing from this libc.so.6 is refused whatever the release."""

    sdk = _staged_build() / "librusticol_loop_sdk.so"
    requirements = native_loader._glibc_requirements(sdk) or []
    markers = sorted(
        {name for _, name in requirements if name in native_loader.GLIBC_ABI_MARKERS}
    )
    assert "GLIBC_ABI_GNU2_TLS" in markers, f"{sdk}: no TLS-descriptor marker"
    files = {file for file, name in requirements if name in markers}
    assert files == {"libc.so.6"}
    numbered = native_loader.glibc_floor(
        name for _, name in requirements if name not in markers
    )
    assert numbered is not None and numbered[0] < (2, 42)
    asked: list[str] = []

    def definitions(defined):
        def read(soname):
            asked.append(soname)
            return defined

        return read

    def check(found, defined):
        monkeypatch.setattr(native_loader, "running_glibc", lambda: found)
        monkeypatch.setattr(
            native_loader, "glibc_version_definitions", definitions(defined)
        )
        native_loader.check_native_library(sdk)

    every = frozenset(markers)
    without = every - {"GLIBC_ABI_GNU2_TLS"}
    # glibc 2.42 without the stable-branch update (or a newer glibc that lacks
    # it): refused, "does not define it", not "requires 2.42 but runs 2.42".
    for found in ("2.42", "2.44"):
        with pytest.raises(NativePlatformError) as refused:
            check(found, without)
        error = refused.value
        assert (error.required, error.found, error.version_name) == (
            "2.42",
            found,
            "GLIBC_ABI_GNU2_TLS",
        )
        assert f"this process runs glibc {found}, which does not define it" in (
            str(error)
        )
        assert "Update this system's glibc (glibc 2.43 or newer" in str(error)
    check("2.42", every)
    check("2.44", every)
    assert set(asked) == {"libc.so.6"}
    # Below the floor the release decides, before any definition is read.
    asked.clear()
    with pytest.raises(NativePlatformError, match=r"runs glibc 2\.41\. Run it"):
        check("2.41", every)
    assert asked == []
    # Definitions unreadable: the release decides.
    check("2.42", None)


def test_a_marker_this_libc_does_not_define_is_refused_before_the_loader(
    tmp_path,
) -> None:
    """Real definitions, no faked version: the i386 marker on x86-64."""

    running = native_loader.running_glibc()
    defined = native_loader.glibc_version_definitions()
    assert running is not None and defined is not None
    assert "GLIBC_ABI_GNU_TLS" not in defined, "an x86-64 libc.so.6 lacks it"
    library = _patched(
        _staged_build() / "librusticol_loop_sdk.so",
        tmp_path / "marker.so",
        "GLIBC_ABI_GNU2_TLS",
        "GLIBC_ABI_GNU_TLS",
    )
    lacking = native_loader._release(running) >= (2, 42)
    expected = (
        f"this process runs glibc {running}, which does not define it"
        if lacking
        else f"this process runs glibc {running}."
    )
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    assert refused.value.version_name == "GLIBC_ABI_GNU_TLS"
    assert expected in str(refused.value)
    with pytest.raises(OSError, match="version `GLIBC_ABI_GNU_TLS' not found") as raw:
        ctypes.CDLL(str(library))
    translated = native_loader.translate_loader_error(raw.value, library)
    assert translated is not None and expected in str(translated)


def test_running_libc_definitions_match_readelf() -> None:
    path = native_loader._mapped_object("libc.so.6")
    assert path is not None and Path(path).is_file()
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.skip("readelf (binutils) is not on PATH: no independent ELF reader")
    output = subprocess.run(
        [readelf, "-V", "--wide", path], check=True, capture_output=True, text=True
    ).stdout
    section = output.split("Version definition section", 1)[1]
    section = section.split("Version needs section", 1)[0]
    expected = {
        line.split("Name: ", 1)[1].split()[0]
        for line in section.splitlines()
        if "Name: GLIBC_" in line and "Parent" not in line
    }
    assert native_loader.glibc_version_definitions() == frozenset(expected)
    assert native_loader._mapped_object("no-such-library.so.1") is None


def test_running_glibc_reads_confstr(monkeypatch) -> None:
    monkeypatch.setattr(os, "confstr", lambda name: "glibc 2.40")
    assert native_loader.running_glibc() == "2.40"
    monkeypatch.setattr(os, "confstr", lambda name: "musl 1.2.5")
    monkeypatch.setattr(native_loader, "_gnu_get_libc_version", lambda: None)
    assert native_loader.running_glibc() is None

    def unavailable(name):
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(os, "confstr", unavailable)
    assert native_loader.running_glibc() is None


def test_files_without_glibc_requirements_pass(tmp_path) -> None:
    text = tmp_path / "library.dylib"
    text.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(60))
    assert native_loader.glibc_version_requirements(text) is None
    native_loader.check_native_library(text)
    assert native_loader.glibc_version_requirements(tmp_path / "missing.so") is None
    assert native_loader.glibc_floor(["GLIBC_2.2.5", "GLIBC_2.38"]) == (
        (2, 38),
        "GLIBC_2.38",
    )
    assert native_loader.glibc_floor(["GLIBC_PRIVATE"]) is None


def test_doctor_reports_the_platform_error(monkeypatch) -> None:
    from pyamplicol import diagnostics

    def refuse(name):
        raise NativePlatformError("Native library x requires glibc >= 2.42")

    monkeypatch.setattr(diagnostics, "importlib", SimpleNamespace(import_module=refuse))
    check = diagnostics._native_check(required=True)
    assert (check.status, check.name) == ("fail", "rusticol-python")
    assert "requires glibc >= 2.42" in check.detail


HEADER = ROOT / "rust" / "loop-reduction" / "sdk" / "include" / "rusticol_virtual.h"


def test_the_sdk_header_states_the_documented_floor() -> None:
    floor = {}
    for line in HEADER.read_text().splitlines():
        for part in ("MAJOR", "MINOR"):
            prefix = f"#define RUSTICOL_VIRTUAL_GLIBC_FLOOR_{part} "
            if line.startswith(prefix):
                floor[part] = int(line[len(prefix) :])
    assert (floor["MAJOR"], floor["MINOR"]) == DOCUMENTED_FLOOR


def test_the_sdk_header_stops_an_older_glibc_toolchain_at_compile_time(
    tmp_path,
) -> None:
    """``-DRUSTICOL_VIRTUAL_SHARED``: an older glibc (faked) is an #error."""

    compiler = os.environ.get("CC") or shutil.which("cc")
    if compiler is None:
        pytest.fail("no cc on PATH for the SDK header check; set CC")
    client = tmp_path / "client.c"
    client.write_text('#include "rusticol_virtual.h"\n')
    older = tmp_path / "older_glibc.h"
    older.write_text(
        "#include <features.h>\n#undef __GLIBC_MINOR__\n"
        f"#define __GLIBC_MINOR__ {DOCUMENTED_FLOOR[1] - 1}\n"
    )

    command = (compiler, "-std=c11", "-fsyntax-only", f"-I{HEADER.parent}")

    def compile_client(*flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*command, *flags, str(client)],
            capture_output=True,
            text=True,
            check=False,
        )

    assert compile_client("-DRUSTICOL_VIRTUAL_SHARED").returncode == 0
    assert compile_client("-include", str(older)).returncode == 0  # opt-in only
    refused = compile_client("-DRUSTICOL_VIRTUAL_SHARED", "-include", str(older))
    assert refused.returncode != 0
    assert "librusticol_loop_sdk.so needs glibc >= 2.42" in refused.stderr
