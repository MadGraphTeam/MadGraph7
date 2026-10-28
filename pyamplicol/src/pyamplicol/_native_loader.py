# SPDX-License-Identifier: 0BSD
"""Refuse a native library the running C library cannot load, before loading it.

A Linux native library records the glibc versions it needs (its ELF version
requirements, ``readelf -V``). The release natives of the loop engine (the
``_rusticol`` extension and the loop SDK's shared library, built by
``tools/developer/build_loop_native.py``) can need ``GLIBC_ABI_GNU2_TLS`` (TLS
descriptors), which glibc 2.43 and newer define, and glibc 2.42 builds that
include the stable-branch update adding it; a wheel built in a manylinux
container needs less. Before ``pyamplicol._rusticol`` is loaded,
:func:`check_native_library` reads that file's requirements and compares their
newest release (for an ABI marker such as ``GLIBC_ABI_GNU2_TLS``, which carries
no number, the oldest release that can define it) with the running glibc
(``os.confstr("CS_GNU_LIBC_VERSION")``). A glibc below that floor, or a C
library that is not glibc, gets :class:`pyamplicol.api.errors.NativePlatformError`
naming the required and the running version instead of the dynamic loader's
raw "version `GLIBC_ABI_GNU2_TLS' not found". A release number cannot decide
whether a glibc build defines an ABI marker (a glibc 2.42 without the update
does not); the dynamic loader's own version check decides that, before any of
the library's code runs, and :func:`translate_loader_error` turns its error
(and any other missing-glibc-version error: a dependency's requirement, a
library whose requirements could not be read) into the same error.

The normal path reads a few kilobytes of the library's ELF headers once per
process (~0.1 ms); nothing here imports :mod:`pyamplicol.api` unless an error
is raised.
"""

from __future__ import annotations

import importlib.machinery
import os
import re
import struct
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, BinaryIO, NamedTuple

if TYPE_CHECKING:
    from types import ModuleType

    from .api.errors import NativePlatformError

#: Extension modules loaded through the check.
GUARDED_MODULES = frozenset({"pyamplicol._rusticol"})


class AbiMarker(NamedTuple):
    """The glibc releases that define an ABI marker version (``GLIBC_ABI_*``)."""

    #: The oldest release some of whose builds define it.
    earliest: tuple[int, ...]
    #: The first upstream release that defines it.
    release: tuple[int, ...]


#: glibc's ABI marker versions carry no release number. This table names the
#: releases for the pre-load check, the messages and the floor pin; whether a
#: glibc build at or above the earliest release defines a marker is decided by
#: the dynamic loader (its error is translated). GLIBC_ABI_DT_RELR came with
#: glibc 2.36. The x86 markers were added on glibc master after the 2.42
#: release (BZ #33129, #33212, #33221): the glibc 2.42 release does not define
#: them (its source tarball has none), glibc 2.43 does, and they were
#: cherry-picked to the release/2.42/master stable branch (3970785b, d0f72b96),
#: so updated 2.42 builds do too. Checked on x86-64 libc.so.6 builds: glibc
#: 2.40 defines only GLIBC_ABI_DT_RELR; updated 2.42 builds and 2.44 also
#: GLIBC_ABI_DT_X86_64_PLT and GLIBC_ABI_GNU2_TLS. GLIBC_ABI_GNU_TLS is i386's
#: (same patch series; no i386 libc.so.6 checked). Numbered versions
#: GLIBC_x.y[.z] name their release. Another GLIBC_ name is unknown: the check
#: then leaves the decision to the loader (and translates its error), and
#: tests/unit/test_native_glibc_floor.py fails so that this table is extended.
GLIBC_ABI_MARKERS: dict[str, AbiMarker] = {
    "GLIBC_ABI_DT_RELR": AbiMarker((2, 36), (2, 36)),
    "GLIBC_ABI_DT_X86_64_PLT": AbiMarker((2, 42), (2, 43)),
    "GLIBC_ABI_GNU2_TLS": AbiMarker((2, 42), (2, 43)),
    "GLIBC_ABI_GNU_TLS": AbiMarker((2, 42), (2, 43)),
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
    """The oldest glibc release that can provide an ELF version name, if known.

    For an ABI marker that is the oldest release with builds that define it
    (``(2, 42)`` for ``GLIBC_ABI_GNU2_TLS``, defined by glibc 2.43 and by
    updated 2.42 builds), see :data:`GLIBC_ABI_MARKERS`.
    """

    marker = GLIBC_ABI_MARKERS.get(version_name)
    if marker is not None:
        return marker.earliest
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
            needs = _elf_version_needs(stream)
    except (OSError, ValueError, struct.error):
        return None
    if needs is None:
        return None
    return tuple(sorted({name for name in needs if name.startswith("GLIBC_")}))


def _read(stream: BinaryIO, offset: int, size: int) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated ELF file")
    return data


def _string(strings: bytes, offset: int) -> str:
    return strings[offset : strings.index(b"\0", offset)].decode("ascii", "replace")


def _elf_version_needs(stream: BinaryIO) -> list[str] | None:
    """The version names an ELF file needs from its dependencies.

    From its SHT_GNU_verneed sections (Verneed/Vernaux entries); None if it is
    not an ELF file with sections.
    """

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
    needs: list[str] = []
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
                needs.append(_string(strings, name))
                if step == 0:
                    break
                at += step
            if following == 0:
                break
            position += following
    return needs


def _version(release: tuple[int, ...]) -> str:
    return ".".join(map(str, release))


def _is_pyamplicol_library(path: str) -> bool:
    package = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(path).startswith(package + os.sep) or os.path.basename(
        path
    ).startswith(("_rusticol", "librusticol"))


def _error(
    library: str | None,
    *,
    required: tuple[int, ...] | None,
    version_name: str,
    found: str | None,
    loader_message: str | None = None,
    own: bool = True,
) -> NativePlatformError:
    """The NativePlatformError for ``version_name`` missing from glibc ``found``.

    ``found`` at or above ``required`` means that this glibc build lacks an ABI
    marker its release can define (a glibc 2.42 build without the stable-branch
    update for ``GLIBC_ABI_GNU2_TLS``).
    """

    from .api.errors import NativePlatformError

    marker = GLIBC_ABI_MARKERS.get(version_name)
    floor = None if required is None else _version(required)
    lacking = found is not None and required is not None and _release(found) >= required
    defined_by = release_note = ""
    if floor is None:
        needs = f"a glibc that provides the ELF version {version_name}"
        newer = "a newer glibc"
    elif marker is not None and marker.release != marker.earliest:
        release, earliest = _version(marker.release), _version(marker.earliest)
        defined_by = (
            f", which glibc {release} and newer define, as do glibc {earliest} "
            "builds with the stable-branch update that added it"
        )
        release_note = f"; the glibc {earliest} release does not"
        needs = f"glibc >= {floor}"
        newer = f"glibc {release} or newer, or an updated glibc {earliest} build"
    else:
        needs = f"glibc >= {floor}"
        newer = f"glibc {floor} or newer"
    if found is None:
        running = (
            "this process does not run on glibc (no GNU C library version is "
            f"reported on {sys.platform})"
        )
    elif lacking:
        running = f"this process runs glibc {found}, which does not define it"
    else:
        running = f"this process runs glibc {found}"
    where = (
        f"Update this system's glibc ({newer}), or run it on a Linux system or "
        "container that has one"
        if lacking
        else f"Run it on a Linux system or container with {newer}"
    )
    subject = f"Native library {library}" if library else "A native library"
    if own:
        remedy = (
            f"{where} (`ldd --version` shows the version), or build pyAmpliCol's "
            "native libraries from source on this host so that they link "
            "against its glibc (see 'Native libraries and glibc' in "
            "docs/user/installation.md)."
        )
    else:
        remedy = (
            "Install a build of that library made for this glibc (for a Python "
            "package, a wheel for this platform or a build from source on this "
            f"host), or {where[0].lower()}{where[1:]} (`ldd --version` shows "
            "the version)."
        )
    if lacking:
        what = f"needs the ELF version {version_name}{defined_by}"
    elif floor is None:
        what = f"requires {needs}"
    else:
        what = (
            f"requires {needs} (it needs the ELF version {version_name}"
            f"{defined_by}{release_note})"
        )
    message = f"{subject} {what}, but {running}. {remedy}"
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

    The library's floor is the newest release among its requirements (for an
    ABI marker, the oldest release that can define it, :data:`GLIBC_ABI_MARKERS`:
    2.42 for ``GLIBC_ABI_GNU2_TLS``); a running glibc below it, or a C library
    that is not glibc, is refused. A glibc at or above it that still lacks a
    marker (a glibc 2.42 build without the stable-branch update; markers are
    per architecture) is left to the dynamic loader, whose version check runs
    before any of the library's code: :func:`translate_loader_error` turns its
    error into the same NativePlatformError. The ``pyamplicol._rusticol``
    import calls it before ``dlopen``. Internal: not a public API.
    """

    floor = glibc_floor(glibc_version_requirements(path) or ())
    if floor is None:
        return
    found = running_glibc()
    if found is None or _release(found) < floor[0]:
        raise _error(
            os.fspath(path), required=floor[0], version_name=floor[1], found=found
        )


def translate_loader_error(
    error: BaseException, library: str | os.PathLike[str] | None = None
) -> NativePlatformError | None:
    """The NativePlatformError for a loader's missing-glibc-version error, else None.

    Matches the dynamic loader's ``version `GLIBC_...' not found (required by
    LIBRARY)`` in the ImportError of an extension module or the OSError of
    ``ctypes.CDLL``; the error names LIBRARY (else ``library``, the library
    being loaded, which is taken to be pyAmpliCol's).
    """

    text = str(error)
    match = _LOADER_MISSING_VERSION.search(text)
    if match is None:
        return None
    loading = None if library is None else os.fspath(library)
    needing = _LOADER_REQUIRED_BY.search(text, match.end())
    named = needing.group(1) if needing is not None else loading
    if named is None:
        named = getattr(error, "path", None)
    name = match.group(1)
    return _error(
        named,
        required=glibc_release(name),
        version_name=name,
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
    "GLIBC_ABI_MARKERS",
    "GUARDED_MODULES",
    "AbiMarker",
    "check_native_library",
    "glibc_floor",
    "glibc_release",
    "glibc_version_requirements",
    "install",
    "running_glibc",
    "translate_loader_error",
]
