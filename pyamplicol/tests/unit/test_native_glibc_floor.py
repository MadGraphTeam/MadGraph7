# SPDX-License-Identifier: 0BSD
"""The native libraries' glibc requirements and the load-time platform check.

The release natives built by ``tools/developer/build_loop_native.py`` need
``GLIBC_2.38`` at most and no glibc ABI marker (the build compiles its C
dependencies with the traditional TLS dialect, so no ``GLIBC_ABI_GNU2_TLS``).
A toolchain change that moves that floor fails
``test_staged_natives_stay_within_the_documented_glibc_floor`` (the measured
requirement is pinned exactly, so the documented floor follows it) instead of
users. Before ``pyamplicol._rusticol`` is loaded, every glibc version it needs
is looked up in the version definitions of the glibc libraries this process
has loaded; a missing one raises ``NativePlatformError``, and the dynamic
loader's own missing-version error becomes the same error. Libraries and
"older" C libraries with chosen versions are linked here against stub shared
objects (any C toolchain, whatever its glibc), so only the floor pin and the
readelf cross-check depend on the staged native. The build script's pin of the
C dependencies' TLS dialect is checked against a gnu2-default compiler wrapper.
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import json
import os
import shutil
import struct
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
DOCUMENTED_FLOOR = (2, 38)
NATIVES = ("_rusticol.so", "librusticol_loop_sdk.so")
# ld.so's text for the slice's SDK library 6e123fdb under glibc 2.40
# (2026-10-28; the libc path, a Nix store path there, shortened).
LOADER_MESSAGE = (
    "/lib64/libc.so.6: version `GLIBC_ABI_GNU2_TLS' not found (required by "
    "/opt/pyamplicol/librusticol_loop_sdk.so)"
)


def _staged_build() -> Path:
    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    directory = json.loads(info.read_text()).get("native_build", {}).get("directory")
    if not directory:
        pytest.fail(
            f"{info} names no native build directory: stage one with "
            "tools/developer/build_loop_native.py stage"
        )
    build = Path(directory)
    missing = [name for name in NATIVES if not (build / name).is_file()]
    if missing:
        pytest.fail(f"{build} has no {', '.join(missing)}")
    return build


def _assert_documented_floor(path: Path) -> None:
    names = native_loader.glibc_version_requirements(path)
    assert names, f"{path} declares no glibc version requirement"
    markers = [name for name in names if native_loader.numbered_release(name) is None]
    assert not markers, (
        f"{path} needs {markers}, which only some glibc builds define (an ABI "
        "marker such as GLIBC_ABI_GNU2_TLS: C objects compiled with TLS "
        "descriptors); build_loop_native.py compiles the C dependencies with "
        "-mtls-dialect=gnu to avoid it"
    )
    newest = max(names, key=lambda name: native_loader.numbered_release(name) or ())
    release = native_loader.numbered_release(newest)
    assert release == DOCUMENTED_FLOOR, (
        f"{path} needs {newest} at most, not the documented floor GLIBC_2.38: "
        "update DOCUMENTED_FLOOR, docs/user/installation.md, the SDK README and "
        "headers, and DESIGN §12.1"
    )


def _compiler() -> str:
    compiler = os.environ.get("CC") or shutil.which("cc")
    if compiler is None:
        pytest.fail("no cc on PATH for the glibc floor tests; set CC")
    return compiler


def _link(*command: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


class _Stubs:
    """Shared objects with chosen glibc version needs and definitions."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.compiler = _compiler()
        self.count = 0
        self.built: dict[tuple[str, tuple[str, ...]], Path] = {}

    def _stem(self) -> Path:
        self.count += 1
        stem = self.directory / f"stub{self.count}"
        stem.mkdir()
        return stem

    def _define(self, stem: Path, versions: tuple[str, ...]) -> Path:
        """A ``libc.so.6`` (soname and file name) defining exactly ``versions``."""

        functions = [f"pyamplicol_glibc_floor_{k}" for k in range(len(versions))]
        nodes = [
            f"{version} {{ global: {functions[k]};{' local: *;' if k == 0 else ''} }};"
            for k, version in enumerate(versions)
        ]
        (stem / "versions.map").write_text("\n".join(nodes) + "\n")
        (stem / "libc.c").write_text(
            "".join(f"void {function}(void) {{}}\n" for function in functions)
        )
        libc = stem / "libc.so.6"
        _link(
            self.compiler,
            "-shared",
            "-fPIC",
            "-nostdlib",
            "-Wl,-soname,libc.so.6",
            f"-Wl,--version-script={stem / 'versions.map'}",
            str(stem / "libc.c"),
            "-o",
            str(libc),
        )
        return libc

    def libc(self, *versions: str) -> Path:
        """A C library that defines exactly ``versions`` (an "older glibc")."""

        key = ("libc", versions)
        if key not in self.built:
            libc = self._define(self._stem(), versions)
            assert native_loader._version_definitions(libc) == (
                "libc.so.6",
                frozenset(versions),
            )
            self.built[key] = libc
        return self.built[key]

    def library(self, *versions: str) -> Path:
        """A library that needs exactly ``versions`` from ``libc.so.6``.

        It calls one function per version of a stub ``libc.so.6`` defining them,
        so its version needs name the running ``libc.so.6`` at load: the dynamic
        loader checks them against this process's glibc before resolving any
        symbol.
        """

        key = ("library", versions)
        if key not in self.built:
            stem = self._stem()
            stub = self._define(stem, versions)
            calls = "".join(
                f" pyamplicol_glibc_floor_{k}();" for k in range(len(versions))
            )
            (stem / "client.c").write_text(
                "".join(
                    f"void pyamplicol_glibc_floor_{k}(void);\n"
                    for k in range(len(versions))
                )
                + f"void pyamplicol_glibc_floor_client(void) {{{calls} }}\n"
            )
            library = stem / "needs.so"
            _link(
                self.compiler,
                "-shared",
                "-fPIC",
                "-nostdlib",
                str(stem / "client.c"),
                str(stub),
                "-o",
                str(library),
            )
            assert sorted(native_loader.glibc_version_requirements(library) or ()) == (
                sorted(versions)
            )
            self.built[key] = library
        return self.built[key]


@pytest.fixture(scope="module")
def stubs(tmp_path_factory) -> _Stubs:
    return _Stubs(tmp_path_factory.mktemp("glibc-stubs"))


def _running_libc() -> tuple[str, frozenset[str]]:
    loaded = native_loader._loaded_definitions(
        "libc.so.6", native_loader._mapped_files()
    )
    assert loaded is not None, "this test host runs glibc: libc.so.6 is mapped"
    return loaded


def _readelf_versions(path: Path) -> str:
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.fail("readelf (binutils) is not on PATH: no independent ELF reader")
    return subprocess.run(
        [readelf, "-V", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _readelf_requirements(path: Path) -> tuple[str, ...]:
    output = _readelf_versions(path)
    names: set[str] = set()
    needs = False
    for line in output.splitlines():
        if line.startswith("Version "):
            needs = line.startswith("Version needs section")
        elif needs and "Name: GLIBC_" in line:
            # A weak need ("Flags: WEAK") is not enforced by the dynamic loader.
            name, _, flags = line.split("Name: ", 1)[1].partition("Flags: ")
            if "WEAK" not in flags.split("Version:")[0]:
                names.add(name.split()[0])
    return tuple(sorted(names))


def test_staged_natives_stay_within_the_documented_glibc_floor() -> None:
    build = _staged_build()
    for name in NATIVES:
        _assert_documented_floor(build / name)


def test_elf_requirements_match_readelf(stubs) -> None:
    build = _staged_build()
    built = stubs.library("GLIBC_2.29", "GLIBC_ABI_GNU2_TLS", "GLIBC_2.43", "GLIBC_2.3")
    # By release (GLIBC_2.3 before GLIBC_2.29), then the markers.
    assert native_loader.glibc_version_requirements(built) == (
        "GLIBC_2.3",
        "GLIBC_2.29",
        "GLIBC_2.43",
        "GLIBC_ABI_GNU2_TLS",
    )
    for path in (*(build / name for name in NATIVES), built):
        requirements = native_loader.glibc_version_requirements(path)
        assert sorted(requirements or ()) == list(_readelf_requirements(path))


def test_the_floor_pin_fails_on_a_changed_requirement_or_an_abi_marker(stubs) -> None:
    _assert_documented_floor(stubs.library("GLIBC_2.2.5", "GLIBC_2.38"))
    for changed in ("GLIBC_2.39", "GLIBC_2.37"):
        with pytest.raises(AssertionError, match=rf"needs {changed} at most, not"):
            _assert_documented_floor(stubs.library("GLIBC_2.2.5", changed))
    with pytest.raises(AssertionError, match=r"needs \['GLIBC_ABI_GNU2_TLS'\]"):
        _assert_documented_floor(stubs.library("GLIBC_2.2.5", "GLIBC_ABI_GNU2_TLS"))


def test_a_version_the_running_glibc_lacks_is_refused_before_loading(stubs) -> None:
    """A real requirement above this host's glibc, nothing faked."""

    running = native_loader.running_glibc()
    assert running is not None, "this test host runs glibc"
    major, minor = map(int, running.split(".")[:2])
    above = f"GLIBC_{major}.{minor + 1}"
    library = stubs.library("GLIBC_2.2.5", above)
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    error = refused.value
    assert isinstance(error, DependencyError)
    assert not isinstance(error, ImportError)
    assert (error.library, error.missing, error.required, error.found) == (
        str(library),
        (above,),
        f"{major}.{minor + 1}",
        running,
    )
    assert str(error).startswith(
        f"Native library {library} needs the ELF version {above}, which this "
        f"process's glibc {running} does not define. Run it on a Linux system or "
        f"container whose glibc defines it (glibc {major}.{minor + 1} or newer); "
        "`ldd --version` shows the glibc release. Or build pyAmpliCol's native "
        "libraries from source on this host"
    ), str(error)
    assert "docs/user/installation.md" in str(error)

    # The dynamic loader refuses the same file; its error becomes the same type.
    with pytest.raises(OSError, match=f"version `{above}' not found") as loader:
        ctypes.CDLL(str(library))
    translated = native_loader.translate_loader_error(loader.value, library)
    assert isinstance(translated, NativePlatformError)
    assert (translated.library, translated.missing, translated.required) == (
        str(library),
        (above,),
        f"{major}.{minor + 1}",
    )
    assert f"The dynamic loader reported: {loader.value}" in str(translated)


def _weakened(library: Path, version: str, into: Path) -> Path:
    """A copy of ``library`` whose need for ``version`` is weak (VER_FLG_WEAK).

    GNU ld emits no weak version needs, so the flag is set in the file.
    """

    data = bytearray(library.read_bytes())
    with open(library, "rb") as stream:
        elf = native_loader._Elf(stream)
        need = struct.Struct(elf.order + "HHIII")
        auxiliary = struct.Struct(elf.order + "IHHII")
        patched = 0
        for kind, offset, _, link, entries in elf.sections:
            if kind != native_loader._SHT_GNU_VERNEED:
                continue
            strings = elf._strings(link)
            position = offset
            for _ in range(entries):
                _, count, _, first, following = need.unpack_from(data, position)
                at = position + first
                for _ in range(count):
                    _, flags, _, name, step = auxiliary.unpack_from(data, at)
                    if native_loader._string(strings, name) == version:
                        struct.pack_into(elf.order + "H", data, at + 4, flags | 0x2)
                        patched += 1
                    at += step
                position += following
    assert patched == 1
    into.write_bytes(bytes(data))
    return into


def test_a_weak_version_need_is_not_enforced(stubs, tmp_path) -> None:
    """ld.so accepts a library whose weak version need the libc lacks."""

    running = native_loader.running_glibc()
    assert running is not None, "this test host runs glibc"
    major, minor = map(int, running.split(".")[:2])
    above = f"GLIBC_{major}.{minor + 1}"
    library = _weakened(
        stubs.library("GLIBC_2.2.5", above), above, tmp_path / "weak.so"
    )
    readelf = _readelf_versions(library)
    assert any(
        f"Name: {above}" in line and "Flags: WEAK" in line
        for line in readelf.splitlines()
    ), readelf
    assert native_loader.glibc_version_requirements(library) == ("GLIBC_2.2.5",)
    native_loader.check_native_library(library)
    # The loader passes its version check too: it fails only on the stub
    # symbols, which the running libc.so.6 does not define.
    with pytest.raises(OSError, match="undefined symbol") as loader:
        ctypes.CDLL(str(library))
    assert "not found" not in str(loader.value)


def test_abi_markers_are_looked_up_in_the_running_libc(stubs) -> None:
    """No release number decides a marker: the running libc.so.6's definitions do.

    i386's ``GLIBC_ABI_GNU_TLS`` is never defined by an x86-64 glibc; every
    version the running libc.so.6 defines passes, markers included.
    """

    running = native_loader.running_glibc()
    _, defined = _running_libc()
    library = stubs.library("GLIBC_2.2.5", "GLIBC_ABI_GNU_TLS")
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    error = refused.value
    assert (error.missing, error.required, error.found) == (
        ("GLIBC_ABI_GNU_TLS",),
        None,
        running,
    )
    assert str(error).startswith(
        f"Native library {library} needs the ELF version GLIBC_ABI_GNU_TLS, which "
        f"this process's glibc {running} does not define. Run it on a Linux "
        "system or container whose glibc defines it; `readelf -V` of a system's "
        "libc.so.6 lists the versions it defines. Or build"
    ), str(error)
    newest = max(defined, key=lambda name: native_loader.numbered_release(name) or ())
    markers = sorted(name for name in defined if name.startswith("GLIBC_ABI_"))
    native_loader.check_native_library(stubs.library("GLIBC_2.2.5", newest, *markers))


def test_a_faked_older_glibc_is_refused_before_loading(monkeypatch, stubs) -> None:
    """An "older glibc": a libc.so.6 defining the versions up to GLIBC_2.37."""

    older = stubs.libc("GLIBC_2.2.5", "GLIBC_2.34", "GLIBC_2.37")
    monkeypatch.setattr(native_loader, "_mapped_files", lambda: [str(older)])
    monkeypatch.setattr(native_loader, "running_glibc", lambda: "2.37")
    native_loader.check_native_library(stubs.library("GLIBC_2.2.5", "GLIBC_2.37"))
    library = stubs.library("GLIBC_2.2.5", "GLIBC_ABI_GNU2_TLS", "GLIBC_2.38")
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    error = refused.value
    assert (error.missing, error.required, error.found) == (
        ("GLIBC_2.38", "GLIBC_ABI_GNU2_TLS"),
        "2.38",
        "2.37",
    )
    assert (
        "needs the ELF versions GLIBC_2.38, GLIBC_ABI_GNU2_TLS, which this "
        "process's glibc 2.37 does not define. Run it on a Linux system or "
        "container whose glibc defines them (at least glibc 2.38); "
        "`readelf -V`"
    ) in str(error)
    # A glibc library not loaded into this process is left to the loader.
    monkeypatch.setattr(native_loader, "_mapped_files", lambda: [])
    native_loader.check_native_library(library)


def test_an_older_faked_glibc_refuses_the_rusticol_import_before_dlopen(
    stubs,
) -> None:
    """``import pyamplicol._rusticol`` on an older libc.so.6: typed, no mapping.

    The faked libc.so.6 defines every version the extension needs from it but
    the newest numbered one and any ABI marker.
    """

    extension = importlib.machinery.PathFinder.find_spec(
        "_rusticol", [str(ROOT / "src" / "pyamplicol")]
    )
    assert extension is not None and extension.origin is not None
    needs = native_loader._version_needs(extension.origin)
    assert needs is not None
    from_libc = needs["libc.so.6"]
    newest = max(from_libc, key=lambda name: native_loader.numbered_release(name) or ())
    release = native_loader.numbered_release(newest)
    assert release is not None
    markers = sorted(n for n in from_libc if native_loader.numbered_release(n) is None)
    kept = tuple(n for n in from_libc if n != newest and n not in markers)
    older_release = f"{release[0]}.{release[1] - 1}"
    script = textwrap.dedent(
        f"""
        import os, sys
        from pyamplicol import NativePlatformError, _native_loader
        real_files, real_confstr = _native_loader._mapped_files, os.confstr
        _native_loader._mapped_files = lambda: [
            {str(stubs.libc(*kept))!r} if os.path.basename(path) == "libc.so.6"
            else path
            for path in real_files()
        ]
        os.confstr = lambda name: (
            "glibc {older_release}" if name == "CS_GNU_LIBC_VERSION"
            else real_confstr(name)
        )
        try:
            import pyamplicol._rusticol
        except NativePlatformError as error:
            mapped = "_rusticol" in open("/proc/self/maps").read()
            print(repr((error.missing, error.required, error.found, mapped,
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
    expected = (
        (newest, *markers),
        ".".join(map(str, release)),
        older_release,
        False,
        False,
    )
    assert summary == repr(expected)
    assert message.startswith(f"Native library {extension.origin} needs the ELF")
    assert f"which this process's glibc {older_release} does not define" in message


def test_the_real_loader_error_is_translated_by_the_guarded_loader(
    monkeypatch, stubs
) -> None:
    """The fallback, with the real dynamic loader and libc.so.6.

    With no glibc library visible to the pre-load check, the extension loader
    reaches ``dlopen``, whose version check refuses the library before any of
    its code runs; the guarded loader raises the translated error.
    """

    running = native_loader.running_glibc()
    library = stubs.library("GLIBC_2.2.5", "GLIBC_ABI_GNU_TLS")
    monkeypatch.setattr(native_loader, "_mapped_files", lambda: [])
    loader = native_loader._GuardedExtensionFileLoader("needs_marker", str(library))
    spec = importlib.machinery.ModuleSpec("needs_marker", loader, origin=str(library))
    with pytest.raises(NativePlatformError) as refused:
        loader.create_module(spec)
    error = refused.value
    assert (error.library, error.missing, error.required, error.found) == (
        str(library),
        ("GLIBC_ABI_GNU_TLS",),
        None,
        running,
    )
    assert isinstance(error.__cause__, ImportError)
    assert "version `GLIBC_ABI_GNU_TLS' not found" in str(error.__cause__)
    assert "does not define. Run it on" in str(error)
    assert str(error).endswith(f"The dynamic loader reported: {error.__cause__}")


def test_loader_version_errors_become_the_typed_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(native_loader, "running_glibc", lambda: "2.40")
    error = native_loader.translate_loader_error(ImportError(LOADER_MESSAGE))
    assert isinstance(error, NativePlatformError)
    assert (error.library, error.missing, error.required, error.found) == (
        "/opt/pyamplicol/librusticol_loop_sdk.so",
        ("GLIBC_ABI_GNU2_TLS",),
        None,
        "2.40",
    )
    assert str(error) == (
        "Native library /opt/pyamplicol/librusticol_loop_sdk.so needs the ELF "
        "version GLIBC_ABI_GNU2_TLS, which this process's glibc 2.40 does not "
        "define. Run it on a Linux system or container whose glibc defines it; "
        "`readelf -V` of a system's libc.so.6 lists the versions it defines. Or "
        "build pyAmpliCol's native libraries from source on this host so that "
        "they link against its glibc (see 'Native libraries and glibc' in "
        "docs/user/installation.md). The dynamic loader reported: "
        f"{LOADER_MESSAGE}"
    )

    # A third-party library (Symbolica's extension): its own build is the remedy.
    dependency = native_loader.translate_loader_error(
        ImportError(
            "/lib/libc.so.6: version `GLIBC_2.43' not found (required by "
            "/site-packages/symbolica/core.abi3.so)"
        )
    )
    assert dependency is not None
    assert dependency.library == "/site-packages/symbolica/core.abi3.so"
    assert (dependency.missing, dependency.required) == (("GLIBC_2.43",), "2.43")
    assert "Install a build of that library made for this glibc" in str(dependency)
    assert "pyAmpliCol's native libraries" not in str(dependency)
    named = native_loader.translate_loader_error(
        OSError("libc.so.6: version `GLIBC_2.43' not found"), tmp_path / "x.so"
    )
    assert named is not None and named.library == str(tmp_path / "x.so")
    assert native_loader.translate_loader_error(ImportError("no module x")) is None

    # Through the guarded extension loader with a faked loader error; any
    # other loader error passes unchanged.
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


def test_a_c_library_that_is_not_glibc_is_named(monkeypatch, stubs) -> None:
    library = stubs.library("GLIBC_2.2.5", "GLIBC_2.38")
    monkeypatch.setattr(native_loader, "running_glibc", lambda: None)
    with pytest.raises(NativePlatformError) as refused:
        native_loader.check_native_library(library)
    error = refused.value
    assert (error.found, error.required, error.missing) == (
        None,
        "2.38",
        ("GLIBC_2.2.5", "GLIBC_2.38"),
    )
    assert (
        "needs glibc >= 2.38 (the ELF versions GLIBC_2.2.5, GLIBC_2.38), but this "
        "process does not run on glibc"
    ) in str(error)


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
    assert native_loader.numbered_release("GLIBC_2.2.5") == (2, 2, 5)
    assert native_loader.numbered_release("GLIBC_ABI_GNU2_TLS") is None
    assert native_loader.numbered_release("GCC_3.0") is None


def test_doctor_reports_the_platform_error(monkeypatch) -> None:
    from pyamplicol import diagnostics

    def refuse(name):
        raise NativePlatformError("Native library x needs the ELF version GLIBC_2.43")

    monkeypatch.setattr(diagnostics, "importlib", SimpleNamespace(import_module=refuse))
    check = diagnostics._native_check(required=True)
    assert (check.status, check.name) == ("fail", "rusticol-python")
    assert "needs the ELF version GLIBC_2.43" in check.detail


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
    """``-DRUSTICOL_VIRTUAL_SHARED``: an older glibc is an #error.

    Both glibc releases are faked, so the toolchain's own glibc does not matter.
    """

    compiler = _compiler()
    client = tmp_path / "client.c"
    client.write_text('#include "rusticol_virtual.h"\n')

    def glibc(name: str, major: int, minor: int) -> str:
        header = tmp_path / f"{name}_glibc.h"
        header.write_text(
            "#include <features.h>\n#undef __GLIBC__\n#undef __GLIBC_MINOR__\n"
            f"#define __GLIBC__ {major}\n#define __GLIBC_MINOR__ {minor}\n"
        )
        return str(header)

    major, minor = DOCUMENTED_FLOOR
    floor, older = glibc("floor", major, minor), glibc("older", major, minor - 1)
    newer = glibc("newer", major + 1, 0)

    command = (compiler, "-std=c11", "-fsyntax-only", f"-I{HEADER.parent}")

    def compile_client(*flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*command, *flags, str(client)],
            capture_output=True,
            text=True,
            check=False,
        )

    for accepted in (floor, newer):
        completed = compile_client("-DRUSTICOL_VIRTUAL_SHARED", "-include", accepted)
        assert completed.returncode == 0, completed.stderr
    assert compile_client("-include", older).returncode == 0  # opt-in only
    refused = compile_client("-DRUSTICOL_VIRTUAL_SHARED", "-include", older)
    assert refused.returncode != 0
    assert "librusticol_loop_sdk.so needs glibc >= 2.38" in refused.stderr


def _build_loop_native():
    import importlib.util

    script = ROOT / "tools" / "developer" / "build_loop_native.py"
    spec = importlib.util.spec_from_file_location("_glibc_build_loop_native", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tls_descriptor_relocations(path: Path) -> int:
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.fail("readelf (binutils) is not on PATH")
    output = subprocess.run(
        [readelf, "-rW", str(path)], check=True, capture_output=True, text=True
    ).stdout
    return output.count("TLSDESC")


def test_the_c_dependencies_are_compiled_without_tls_descriptors(tmp_path) -> None:
    """A compiler wrapper defaulting to -mtls-dialect=gnu2 is overridden.

    The wrapper puts its dialect before the caller's flags, as the nixpkgs
    cc-wrapper's cc-cflags-before does; the pinned CC appends the traditional
    dialect, so a thread-local access compiles without TLS descriptors.
    """

    build_loop_native = _build_loop_native()
    wrapper = tmp_path / "gnu2-cc"
    wrapper.write_text(f'#!/bin/sh\nexec {_compiler()} -mtls-dialect=gnu2 "$@"\n')
    wrapper.chmod(0o755)
    settings = build_loop_native.c_dependency_environment(
        dict(os.environ, CC=str(wrapper))
    )
    if not (sys.platform.startswith("linux") and os.uname().machine == "x86_64"):
        assert settings == {}  # one TLS dialect elsewhere
        return
    assert settings == {"CC": f"{wrapper} -mtls-dialect=gnu"}
    source = tmp_path / "tls.c"
    source.write_text("__thread int counter;\nint bump(void) { return ++counter; }\n")

    def relocations(compiler: str, name: str) -> int:
        target = tmp_path / name
        _link(*compiler.split(), "-fPIC", "-O2", "-c", str(source), "-o", str(target))
        return _tls_descriptor_relocations(target)

    assert relocations(str(wrapper), "wrapped.o") > 0  # the wrapper's own dialect
    assert relocations(settings["CC"], "pinned.o") == 0


def test_a_target_directory_with_other_c_settings_drops_gmp_mpfr_sys(
    monkeypatch, tmp_path
) -> None:
    build_loop_native = _build_loop_native()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        build_loop_native.subprocess,
        "run",
        lambda command, **options: calls.append(command),
    )
    target = tmp_path / "target"
    settings = {"CC": "cc -mtls-dialect=gnu"}
    stamp = target / "pyamplicol-c-dependencies.json"

    def drops() -> bool:
        calls.clear()
        build_loop_native._drop_stale_c_dependencies(tmp_path, target, {}, settings)
        return bool(calls)

    assert not drops()  # nothing built yet
    (target / "release" / "build" / "gmp-mpfr-sys-0123").mkdir(parents=True)
    assert drops()  # built before the C settings were recorded
    assert calls[0][:2] == ["cargo", "clean"] and calls[0][-2:] == [
        "-p",
        "gmp-mpfr-sys",
    ]
    stamp.write_text(json.dumps({"CC": "gcc"}))
    assert drops()
    stamp.write_text(json.dumps(settings))
    assert not drops()
