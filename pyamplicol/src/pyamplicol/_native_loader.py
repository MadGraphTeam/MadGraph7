# SPDX-License-Identifier: 0BSD
"""Refuse a native library the running C library cannot load, before loading it.

A Linux native library records the glibc versions it needs (its ELF version
requirements, ``readelf -V``). The release natives of the loop engine (the
``_rusticol`` extension and the loop SDK's shared library, built with thin LTO
by ``tools/developer/build_loop_native.py``) need ``GLIBC_ABI_GNU2_TLS`` (TLS
descriptors), so glibc 2.42 or newer; a wheel built in a manylinux container
needs less. Before ``pyamplicol._rusticol`` is loaded, :func:`check_native_library`
reads that file's requirements and compares the newest with the running glibc
(``os.confstr("CS_GNU_LIBC_VERSION")``): an older glibc, or a C library that is
not glibc, gets :class:`pyamplicol.api.errors.NativePlatformError` naming both,
instead of the dynamic loader's raw "version `GLIBC_ABI_GNU2_TLS' not found".
A loader error of that kind that still happens (a dependency's requirement, a
library whose requirements could not be read) becomes the same error
(:func:`translate_loader_error`).

The normal path reads a few kilobytes of ELF headers once per process; nothing
here imports :mod:`pyamplicol.api` unless an error is raised.
"""

from __future__ import annotations

import importlib.machinery
import os
import re
import struct
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from types import ModuleType

    from .api.errors import NativePlatformError

#: Extension modules loaded through the check.
GUARDED_MODULES = frozenset({"pyamplicol._rusticol"})

#: glibc's ABI marker versions carry no release number; each is first defined
#: by the release below (read from the version definitions of glibc 2.40, 2.42
#: and 2.44's libc.so.6). Numbered versions GLIBC_x.y[.z] name their release.
#: Another GLIBC_ name is unknown: the check then leaves the decision to the
#: loader (and translates its error), and tests/unit/test_native_glibc_floor.py
#: fails so that this table is extended.
GLIBC_ABI_MARKERS: dict[str, tuple[int, ...]] = {
    "GLIBC_ABI_DT_RELR": (2, 36),
    "GLIBC_ABI_DT_X86_64_PLT": (2, 42),
    "GLIBC_ABI_GNU2_TLS": (2, 42),
    "GLIBC_ABI_GNU_TLS": (2, 42),
}

_NUMBERED = re.compile(r"GLIBC_(\d+(?:\.\d+)+)")
_RELEASE = re.compile(r"\d+(?:\.\d+)+")
# ld.so: "version `GLIBC_2.38' not found (required by ...)".
_LOADER_MISSING_VERSION = re.compile(
    r"version [`'\"]?(GLIBC_[A-Za-z0-9_.]+?)[`'\"]? not found"
)
_LOADER_REQUIRED_BY = re.compile(r" \(required by ([^()]+)\)")
_SHT_GNU_VERNEED = 0x6FFFFFFE


def running_glibc() -> str | None:
    """The running glibc's version (``"2.42"``), or None if it is not glibc."""

    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        value = None
    if not value:
        value = _gnu_get_libc_version()
    name, _, version = (value or "").partition(" ")
    return version if name == "glibc" and _RELEASE.match(version) else None


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


def _release(text: str) -> tuple[int, ...]:
    match = _RELEASE.match(text)
    assert match is not None
    return tuple(int(part) for part in match.group().split("."))


def glibc_release(version_name: str) -> tuple[int, ...] | None:
    """The glibc release that first provides an ELF version name, if known."""

    marker = GLIBC_ABI_MARKERS.get(version_name)
    if marker is not None:
        return marker
    match = _NUMBERED.fullmatch(version_name)
    return None if match is None else _release(match.group(1))


def glibc_floor(version_names: Iterable[str]) -> tuple[tuple[int, ...], str] | None:
    """The newest known glibc release among ELF version names, with its name."""

    known = [
        (release, name)
        for name in version_names
        if (release := glibc_release(name)) is not None
    ]
    return max(known) if known else None


def glibc_version_requirements(path: str | os.PathLike[str]) -> tuple[str, ...] | None:
    """The sorted ``GLIBC_*`` versions an ELF file requires (as ``readelf -V``).

    None if the file is not a readable ELF file with section headers (a Mach-O
    or PE library, a stripped section table): then nothing is checked.
    """

    try:
        with open(path, "rb") as stream:
            names = _verneed_names(stream)
    except (OSError, ValueError, struct.error):
        return None
    if names is None:
        return None
    return tuple(sorted({name for name in names if name.startswith("GLIBC_")}))


def _read(stream: BinaryIO, offset: int, size: int) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated ELF file")
    return data


def _verneed_names(stream: BinaryIO) -> list[str] | None:
    """Every version name of the SHT_GNU_verneed sections (Verneed/Vernaux)."""

    ident = stream.read(16)
    if len(ident) != 16 or ident[:4] != b"\x7fELF":
        return None
    if ident[4] not in (1, 2) or ident[5] not in (1, 2):
        return None
    wide = ident[4] == 2
    order = "<" if ident[5] == 1 else ">"
    header = struct.Struct(order + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"))
    fields = header.unpack(_read(stream, 16, header.size))
    table_offset, entry_size, count = fields[5], fields[10], fields[11]
    section = struct.Struct(order + ("IIQQQQIIQQ" if wide else "IIIIIIIIII"))
    if table_offset == 0 or entry_size < section.size:
        return None
    if count == 0:  # extended numbering: section 0's sh_size holds the count
        count = section.unpack(_read(stream, table_offset, section.size))[5]
    table = _read(stream, table_offset, count * entry_size)
    sections = [section.unpack_from(table, k * entry_size) for k in range(count)]
    need = struct.Struct(order + "HHIII")
    auxiliary = struct.Struct(order + "IHHII")
    names: list[str] = []
    for _, kind, _, _, offset, size, link, entries, _, _ in sections:
        if kind != _SHT_GNU_VERNEED or link >= count:
            continue
        strings = _read(stream, sections[link][4], sections[link][5])
        data = _read(stream, offset, size)
        position = 0
        for _ in range(entries):
            _, auxiliaries, _, first, following = need.unpack_from(data, position)
            at = position + first
            for _ in range(auxiliaries):
                _, _, _, name, step = auxiliary.unpack_from(data, at)
                end = strings.index(b"\0", name)
                names.append(strings[name:end].decode("ascii", "replace"))
                if step == 0:
                    break
                at += step
            if following == 0:
                break
            position += following
    return names


def _error(
    library: str | None,
    *,
    required: tuple[int, ...] | None,
    version_name: str,
    found: str | None,
    loader_message: str | None = None,
) -> NativePlatformError:
    from .api.errors import NativePlatformError

    floor = None if required is None else ".".join(map(str, required))
    subject = f"Native library {library}" if library else "A native library"
    needs = f"glibc >= {floor}" if floor else f"a glibc that provides {version_name}"
    if found is None:
        running = (
            "this process does not run on glibc (no GNU C library version is "
            f"reported on {sys.platform})"
        )
    else:
        running = f"this process runs glibc {found}"
    newer = f"glibc {floor} or newer" if floor else "a newer glibc"
    message = (
        f"{subject} requires {needs} (it needs the ELF version {version_name}), "
        f"but {running}. Run it on a Linux system or container with {newer} "
        "(`ldd --version` shows the version), or build pyAmpliCol's native "
        "libraries from source on this host so that they link against its "
        "glibc (see 'Native libraries and glibc' in docs/user/installation.md)."
    )
    if loader_message:
        message += f" The dynamic loader reported: {loader_message}"
    return NativePlatformError(
        message,
        library=library,
        required=floor,
        found=found,
        version_name=version_name,
    )


def check_native_library(path: str | os.PathLike[str]) -> None:
    """Raise NativePlatformError if the running glibc is older than ``path`` needs.

    Call it before opening one of pyAmpliCol's native libraries with
    ``ctypes``; the ``pyamplicol._rusticol`` import calls it itself.
    """

    names = glibc_version_requirements(path)
    floor = glibc_floor(names) if names else None
    if floor is None:
        return
    required, version_name = floor
    found = running_glibc()
    if found is not None and _release(found) >= required:
        return
    raise _error(
        os.fspath(path), required=required, version_name=version_name, found=found
    )


def translate_loader_error(
    error: BaseException, library: str | os.PathLike[str] | None = None
) -> NativePlatformError | None:
    """The NativePlatformError for a loader's missing-glibc-version error, else None.

    Matches the dynamic loader's ``version `GLIBC_...' not found (required by
    LIBRARY)`` in the ImportError of an extension module or the OSError of
    ``ctypes.CDLL``; the error names LIBRARY (else ``library``).
    """

    text = str(error)
    match = _LOADER_MISSING_VERSION.search(text)
    if match is None:
        return None
    needing = _LOADER_REQUIRED_BY.search(text, match.end())
    if needing is not None:
        library = needing.group(1)
    elif library is None:
        library = getattr(error, "path", None)
    name = match.group(1)
    return _error(
        None if library is None else os.fspath(library),
        required=glibc_release(name),
        version_name=name,
        found=running_glibc(),
        loader_message=text,
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
    "GLIBC_ABI_MARKERS",
    "GUARDED_MODULES",
    "check_native_library",
    "glibc_floor",
    "glibc_release",
    "glibc_version_requirements",
    "install",
    "running_glibc",
    "translate_loader_error",
]
