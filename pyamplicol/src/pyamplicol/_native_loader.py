# SPDX-License-Identifier: 0BSD
"""Refuse a native library the running glibc cannot load, before loading it.

A Linux native library records which symbol versions it needs from each shared
library it links (its ELF version needs, ``readelf -V``): ``GLIBC_2.38`` from
``libc.so.6``, for instance. The dynamic loader refuses the library when a
loaded library does not define one of them. Before ``pyamplicol._rusticol`` is
loaded, :func:`check_native_library` makes that check itself for glibc: it
compares the ``GLIBC_*`` versions the library needs from each glibc library
already loaded into this process (``libc.so.6``, ``libm.so.6``, ...) with the
versions that library defines (its ELF version definitions), and raises
:class:`pyamplicol.api.errors.NativePlatformError` naming every missing version
and the running glibc in place of the loader's raw "version `GLIBC_2.38' not
found". No glibc release number is assumed: whether a glibc build defines a
version, an ABI marker such as ``GLIBC_ABI_GNU2_TLS`` included, is read from
that build. A process that does not run on glibc gets the same error. A
library not loaded yet is left to the dynamic loader, whose error
:func:`translate_loader_error` turns into the same error (a dependency's own
requirement, Symbolica's extension for instance, included).

The normal path runs once per process, when the extension is first imported:
it reads ``/proc/self/maps`` and the ELF section headers and version sections
of the library and of the glibc libraries it needs (about 1 ms in all; libc's
version definitions and their string table are about 37 kB). Nothing here
imports :mod:`pyamplicol.api` or :mod:`re` unless an error is raised.
"""

from __future__ import annotations

import importlib.machinery
import os
import struct
import sys
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from types import ModuleType

    from .api.errors import NativePlatformError

#: Extension modules loaded through the check.
GUARDED_MODULES = frozenset({"pyamplicol._rusticol"})

# ld.so: "LIBC: version `GLIBC_2.38' not found (required by LIBRARY)".
_LOADER_MISSING_VERSION = r"version [`'\"]?(GLIBC_[A-Za-z0-9_.]+?)[`'\"]? not found"
_LOADER_REQUIRED_BY = r" \(required by ([^()]+)\)"
_SHT_GNU_VERDEF = 0x6FFFFFFD
_SHT_GNU_VERNEED = 0x6FFFFFFE
_VER_FLG_BASE = 0x1
_VER_FLG_WEAK = 0x2


def running_glibc() -> str | None:
    """The running glibc's version (``"2.42"``), or None if it is not glibc."""

    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        value = None
    if not value:
        value = _gnu_get_libc_version()
    name, _, version = (value or "").partition(" ")
    return version if name == "glibc" and _release(version) is not None else None


def _gnu_get_libc_version() -> str | None:
    # Only reached where confstr reports no GNU C library.
    try:
        import ctypes

        function = ctypes.CDLL(None).gnu_get_libc_version
    except (AttributeError, OSError, TypeError):
        return None
    function.restype = ctypes.c_char_p
    version = function()
    return f"glibc {version.decode('ascii', 'replace')}" if version else None


def _release(text: str) -> tuple[int, ...] | None:
    """``"2.38"`` -> ``(2, 38)``; None unless dot-separated numbers."""

    parts = text.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def numbered_release(version_name: str) -> tuple[int, ...] | None:
    """The glibc release a numbered version names (``GLIBC_2.38`` -> ``(2, 38)``).

    None for any other name, an ABI marker such as ``GLIBC_ABI_GNU2_TLS``
    included: no release number tells whether a glibc build defines a marker.
    """

    prefix, _, number = version_name.partition("_")
    return _release(number) if prefix == "GLIBC" else None


def _ordered(names: Iterable[str]) -> tuple[str, ...]:
    """Numbered versions by release, then the other names alphabetically."""

    def key(name: str) -> tuple[int, tuple[int, ...], str]:
        release = numbered_release(name)
        return (0, release, name) if release is not None else (1, (), name)

    return tuple(sorted(set(names), key=key))


def _newest_numbered(names: Iterable[str]) -> str | None:
    numbered = [name for name in names if numbered_release(name) is not None]
    return _ordered(numbered)[-1] if numbered else None


# ELF reading: section headers, version needs (SHT_GNU_verneed) and version
# definitions (SHT_GNU_verdef) of a file, as ``readelf -V`` shows them.


class _Elf:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        ident = stream.read(16)
        if len(ident) != 16 or ident[:4] != b"\x7fELF":
            raise ValueError("not an ELF file")
        if ident[4] not in (1, 2) or ident[5] not in (1, 2):
            raise ValueError("unknown ELF class or byte order")
        wide = ident[4] == 2
        self.order = "<" if ident[5] == 1 else ">"
        header = struct.Struct(
            self.order + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
        )
        fields = header.unpack(self.read(16, header.size))
        table_offset, entry_size, count = fields[5], fields[10], fields[11]
        section = struct.Struct(self.order + ("IIQQQQIIQQ" if wide else "IIIIIIIIII"))
        if table_offset == 0 or entry_size < section.size:
            raise ValueError("no section headers")
        if count == 0:  # extended numbering: section 0's sh_size holds the count
            count = section.unpack(self.read(table_offset, section.size))[5]
        table = self.read(table_offset, count * entry_size)
        # (sh_type, sh_offset, sh_size, sh_link, sh_info) per section.
        self.sections = [
            (fields[1], fields[4], fields[5], fields[6], fields[7])
            for fields in (
                section.unpack_from(table, k * entry_size) for k in range(count)
            )
        ]

    def read(self, offset: int, size: int) -> bytes:
        self.stream.seek(offset)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated ELF file")
        return data

    def _strings(self, link: int) -> bytes:
        if link >= len(self.sections):
            raise ValueError("bad string table link")
        _, offset, size, _, _ = self.sections[link]
        return self.read(offset, size)

    def version_needs(self) -> dict[str, list[str]]:
        """Needed library (``vn_file``) -> the version names needed from it.

        Weak needs (``VER_FLG_WEAK``) are left out: the dynamic loader accepts
        a library whose weak version need the loaded library does not define.
        """

        need = struct.Struct(self.order + "HHIII")
        auxiliary = struct.Struct(self.order + "IHHII")
        needs: dict[str, list[str]] = {}
        for kind, offset, size, link, entries in self.sections:
            if kind != _SHT_GNU_VERNEED:
                continue
            strings, data = self._strings(link), self.read(offset, size)
            position = 0
            for _ in range(entries):
                _, count, file, first, following = need.unpack_from(data, position)
                names = needs.setdefault(_string(strings, file), [])
                at = position + first
                for _ in range(count):
                    _, flags, _, name, step = auxiliary.unpack_from(data, at)
                    if not flags & _VER_FLG_WEAK:
                        names.append(_string(strings, name))
                    if step == 0:
                        break
                    at += step
                if following == 0:
                    break
                position += following
        return needs

    def version_definitions(self) -> tuple[str | None, frozenset[str]]:
        """The base definition (the soname) and the version names defined."""

        definition = struct.Struct(self.order + "HHHHIII")
        auxiliary = struct.Struct(self.order + "II")
        base, names = None, set()
        for kind, offset, size, link, entries in self.sections:
            if kind != _SHT_GNU_VERDEF:
                continue
            strings, data = self._strings(link), self.read(offset, size)
            position = 0
            for _ in range(entries):
                _, flags, _, count, _, first, following = definition.unpack_from(
                    data, position
                )
                if count:
                    name = _string(
                        strings, auxiliary.unpack_from(data, position + first)[0]
                    )
                    if flags & _VER_FLG_BASE:
                        base = name
                    else:
                        names.add(name)
                if following == 0:
                    break
                position += following
        return base, frozenset(names)


def _string(strings: bytes, offset: int) -> str:
    return strings[offset : strings.index(b"\0", offset)].decode("ascii", "replace")


def _version_needs(path: str | os.PathLike[str]) -> dict[str, list[str]] | None:
    try:
        with open(path, "rb") as stream:
            return _Elf(stream).version_needs()
    except (OSError, ValueError, struct.error):
        return None


def _version_definitions(
    path: str | os.PathLike[str],
) -> tuple[str | None, frozenset[str]] | None:
    try:
        with open(path, "rb") as stream:
            return _Elf(stream).version_definitions()
    except (OSError, ValueError, struct.error):
        return None


def glibc_version_requirements(path: str | os.PathLike[str]) -> tuple[str, ...] | None:
    """The ``GLIBC_*`` versions an ELF file needs (``readelf -V``), by release.

    Numbered versions by release, then ABI markers; weak needs, which the
    dynamic loader does not enforce, are left out.

    None if the file is not a readable ELF file with section headers (a Mach-O
    or PE library, a stripped section table): then nothing is checked.
    """

    needs = _version_needs(path)
    if needs is None:
        return None
    return _ordered(
        name for names in needs.values() for name in names if _is_glibc(name)
    )


def _is_glibc(name: str) -> bool:
    return name.startswith("GLIBC_")


def _mapped_files() -> list[str]:
    """The files mapped into this process (Linux ``/proc/self/maps``)."""

    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            lines = maps.read().splitlines()
    except OSError:
        return []
    files: dict[str, None] = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            files.setdefault(fields[5], None)
    return list(files)


def _loaded_definitions(
    soname: str, files: Sequence[str]
) -> tuple[str, frozenset[str]] | None:
    """The path and version definitions of the loaded library ``soname``.

    The mapped file named ``soname`` (``/lib64/libc.so.6``), else one in
    glibc's older naming whose soname it is (``libc-2.31.so``); None when no
    such library is loaded.
    """

    stem = soname.partition(".so")[0] + "-"
    named = [path for path in files if os.path.basename(path) == soname]
    older = [
        path
        for path in files
        if os.path.basename(path).startswith(stem) and path.endswith(".so")
    ]
    for path in (*named, *older):
        definitions = _version_definitions(path)
        if definitions is not None and definitions[0] == soname:
            return path, definitions[1]
    return None


def _is_pyamplicol_library(path: str) -> bool:
    package = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(path).startswith(package + os.sep) or os.path.basename(
        path
    ).startswith(("_rusticol", "librusticol"))


def _error(
    library: str | None,
    *,
    missing: Sequence[str],
    found: str | None,
    loader_message: str | None = None,
    own: bool = True,
) -> NativePlatformError:
    """The NativePlatformError for ELF versions ``missing`` from glibc ``found``."""

    from .api.errors import NativePlatformError

    missing = _ordered(missing)
    newest = _newest_numbered(missing)
    required = None if newest is None else newest.partition("_")[2]
    plural = len(missing) > 1
    versions = f"the ELF version{'s' if plural else ''} {', '.join(missing)}"
    subject = f"Native library {library}" if library else "A native library"
    newer = "" if required is None else f"glibc {required} or newer"
    if found is None:
        problem = (
            f"needs glibc{f' >= {required}' if required else ''} ({versions}), "
            "but this process does not run on glibc (no GNU C library version "
            f"is reported on {sys.platform})"
        )
        where = f"Run it on a Linux system or container with {newer or 'glibc'}"
    else:
        it = "them" if plural else "it"
        markers = [name for name in missing if numbered_release(name) is None]
        problem = (
            f"needs {versions}, which this process's glibc {found} does not define"
        )
        if markers:
            # No release number tells whether a glibc build defines a marker.
            check = "`readelf -V` of a system's libc.so.6 lists the versions it defines"
        else:
            check = "`ldd --version` shows the glibc release"
        if required is None:
            release = ""
        elif markers:
            release = f" (at least glibc {required})"
        else:
            release = f" ({newer})"
        where = (
            f"Run it on a Linux system or container whose glibc defines {it}"
            f"{release}; {check}"
        )
    if own:
        remedy = (
            f"{where}. Or build pyAmpliCol's native libraries from source on this "
            "host so that they link against its glibc (see 'Native libraries and "
            "glibc' in docs/user/installation.md)."
        )
    else:
        remedy = (
            "Install a build of that library made for this glibc (for a Python "
            "package, a wheel for this platform or a build from source on this "
            f"host). Or {where[0].lower()}{where[1:]}."
        )
    message = f"{subject} {problem}. {remedy}"
    if loader_message:
        message += f" The dynamic loader reported: {loader_message}"
    return NativePlatformError(
        message, library=library, required=required, found=found, missing=missing
    )


def check_native_library(path: str | os.PathLike[str]) -> None:
    """Raise NativePlatformError if this process's glibc cannot load ``path``.

    Every ``GLIBC_*`` version ``path`` needs from a glibc library already
    loaded into this process must be one that library defines (the dynamic
    loader's own version check, made before loading); a process that does not
    run on glibc is refused when ``path`` needs any. Versions needed from a
    library not loaded yet are left to the dynamic loader, whose version check
    runs before any of the library's code: :func:`translate_loader_error` turns
    its error into the same NativePlatformError. The ``pyamplicol._rusticol``
    import calls it before ``dlopen``. Internal: not a public API.
    """

    needs = _version_needs(path)
    if not needs:
        return
    glibc: dict[str, list[str]] = {}
    for soname, names in needs.items():
        wanted = [name for name in names if _is_glibc(name)]
        if wanted:
            glibc[soname] = wanted
    if not glibc:
        return
    found = running_glibc()
    if found is None:
        everything = [name for names in glibc.values() for name in names]
        raise _error(os.fspath(path), missing=everything, found=None)
    files = _mapped_files()
    missing: list[str] = []
    for soname, names in glibc.items():
        loaded = _loaded_definitions(soname, files)
        if loaded is not None:
            missing.extend(name for name in names if name not in loaded[1])
    if missing:
        raise _error(os.fspath(path), missing=missing, found=found)


def translate_loader_error(
    error: BaseException, library: str | os.PathLike[str] | None = None
) -> NativePlatformError | None:
    """The NativePlatformError for a loader's missing-glibc-version error, else None.

    Matches the dynamic loader's ``version `GLIBC_...' not found (required by
    LIBRARY)`` in the ImportError of an extension module or the OSError of
    ``ctypes.CDLL``; the error names LIBRARY (else ``library``, the library
    being loaded, which is taken to be pyAmpliCol's).
    """

    import re

    text = str(error)
    match = re.search(_LOADER_MISSING_VERSION, text)
    if match is None:
        return None
    loading = None if library is None else os.fspath(library)
    needing = re.compile(_LOADER_REQUIRED_BY).search(text, match.end())
    named = needing.group(1) if needing is not None else loading
    if named is None:
        named = getattr(error, "path", None)
    return _error(
        named,
        missing=(match.group(1),),
        found=running_glibc(),
        loader_message=text,
        # A dependency's own requirement (Symbolica's extension): its remedy.
        own=named is None or named == loading or _is_pyamplicol_library(named),
    )


class _GuardedExtensionFileLoader(importlib.machinery.ExtensionFileLoader):
    """An extension loader that checks the platform before ``dlopen``."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        check_native_library(self.path)
        try:
            return super().create_module(spec)
        except (ImportError, OSError) as error:
            translated = translate_loader_error(error, self.path)
            if translated is None:
                raise
            raise translated from error


class _NativeLibraryFinder:
    """Hand pyAmpliCol's native extension modules to the guarded loader."""

    @classmethod
    def find_spec(
        cls,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in GUARDED_MODULES:
            return None
        for finder in sys.meta_path:
            find_spec = None if finder is cls else getattr(finder, "find_spec", None)
            spec = None if find_spec is None else find_spec(fullname, path, target)
            if not isinstance(spec, importlib.machinery.ModuleSpec):
                continue
            if (
                type(spec.loader) is importlib.machinery.ExtensionFileLoader
                and spec.origin is not None
            ):
                spec.loader = _GuardedExtensionFileLoader(fullname, spec.origin)
            return spec
        return None


def install() -> None:
    """Put the finder first on ``sys.meta_path`` (idempotent)."""

    if _NativeLibraryFinder not in sys.meta_path:
        sys.meta_path.insert(0, _NativeLibraryFinder)


__all__ = [
    "GUARDED_MODULES",
    "check_native_library",
    "glibc_version_requirements",
    "install",
    "numbered_release",
    "running_glibc",
    "translate_loader_error",
]
