# SPDX-License-Identifier: 0BSD
"""Report native SDK metadata without initializing pyAmpliCol or Symbolica."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import lzma
import os
import re
import runpy
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The pyAmpliCol cache root (also used by the compiled-model cache). A
# compressed SDK archive is unpacked under ``<root>/sdk/``.
CACHE_ENVIRONMENT = "PYAMPLICOL_CACHE_DIR"
ARCHIVE_COMPRESSION = "xz"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class SdkUnavailableError(RuntimeError):
    """Raised when a source tree has no staged native SDK."""


@dataclass(frozen=True)
class SdkArchive:
    """The SDK's static archive as the installation ships it.

    A plain archive is used in place. A compressed one (``.xz``; distribution
    wheels) is unpacked on first use into the user cache, keyed by the package
    version and the archive's SHA-256 recorded in the SDK metadata, and checked
    against that digest and size whenever it is used.
    """

    path: Path
    package_version: str
    sha256: str | None = None
    size: int | None = None

    @property
    def compressed(self) -> bool:
        return self.sha256 is not None

    @property
    def name(self) -> str:
        """File name of the usable (unpacked) archive."""

        suffix = f".{ARCHIVE_COMPRESSION}"
        return (
            self.path.name.removesuffix(suffix) if self.compressed else self.path.name
        )

    @property
    def cache_path(self) -> Path | None:
        """Where a compressed archive is unpacked; None for a plain archive."""

        if not self.compressed:
            return None
        return sdk_cache_root() / f"{self.package_version}-{self.sha256}" / self.name

    def resolve(self) -> Path:
        """Return the path of the usable archive, unpacking it if needed."""

        if not self.compressed:
            return self.path
        destination = self.cache_path
        assert destination is not None
        if _matches(destination, self.sha256, self.size):
            return destination
        return _unpack(self, destination)


@dataclass(frozen=True)
class SdkInfo:
    abi_version: int
    package_version: str
    target: str
    include_dir: Path
    archive: SdkArchive
    fortran_source: Path
    rust_source: Path
    system_libraries: tuple[str, ...]
    frameworks: tuple[str, ...]
    # Shared libraries shipped next to the archive (a distribution wheel's
    # GMP, MPFR and MPC), in link order; linked by path with a run path.
    shared_libraries: tuple[Path, ...] = ()
    # The resolved archive path, once known (functools.cached_property would
    # serialize every instance's first use behind one lock on Python 3.11).
    _resolved: dict[str, Path] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    @property
    def library(self) -> Path:
        """The usable static archive (a compressed one is unpacked first)."""

        if "library" not in self._resolved:
            self._resolved["library"] = self.archive.resolve()
        return self._resolved["library"]

    @property
    def library_name(self) -> str:
        """The static archive's file name, without unpacking it."""

        return self.archive.name

    @property
    def cflags(self) -> tuple[str, ...]:
        return (f"-I{self.include_dir}",)

    @property
    def link_flags(self) -> tuple[str, ...]:
        shared = tuple(str(path) for path in self.shared_libraries)
        run_paths = tuple(
            f"-Wl,-rpath,{directory}"
            for directory in dict.fromkeys(
                path.parent for path in self.shared_libraries
            )
        )
        libraries = tuple(f"-l{name}" for name in self.system_libraries)
        frameworks = tuple(
            token for name in self.frameworks for token in ("-framework", name)
        )
        return (str(self.library), *shared, *run_paths, *libraries, *frameworks)

    @property
    def rust_flags(self) -> tuple[str, ...]:
        return tuple(
            token
            for link_flag in self.link_flags
            for token in ("-C", f"link-arg={link_flag}")
        )

    @property
    def cargo_encoded_rust_flags(self) -> str:
        return "\x1f".join(self.rust_flags)

    def to_json(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "package_version": self.package_version,
            "target": self.target,
            "include_dir": str(self.include_dir),
            "library": str(self.library),
            "fortran_source": str(self.fortran_source),
            "rust_source": str(self.rust_source),
            "system_libraries": list(self.system_libraries),
            "frameworks": list(self.frameworks),
            "shared_libraries": [str(path) for path in self.shared_libraries],
            # The shipped archive's compression record (None: used in place).
            "archive_compression": (
                {
                    "format": ARCHIVE_COMPRESSION,
                    "sha256": self.archive.sha256,
                    "size": self.archive.size,
                }
                if self.archive.compressed
                else None
            ),
            "cflags": list(self.cflags),
            "link_flags": list(self.link_flags),
            "rust_flags": list(self.rust_flags),
        }


def sdk_cache_root() -> Path:
    """The directory holding unpacked SDK archives.

    ``$PYAMPLICOL_CACHE_DIR/sdk`` when the variable is set, otherwise ``sdk``
    in the platform's user cache directory (``$XDG_CACHE_HOME/pyamplicol`` or
    ``~/.cache/pyamplicol`` on Linux, ``~/Library/Caches/pyamplicol`` on macOS).
    """

    configured = os.environ.get(CACHE_ENVIRONMENT)
    if configured:
        root = Path(configured).expanduser()
    else:
        from platformdirs import user_cache_path

        root = user_cache_path("pyamplicol")
    return root.absolute() / "sdk"


def _matches(path: Path, sha256: str | None, size: int | None) -> bool:
    """Whether ``path`` is a regular file with the recorded size and digest."""

    try:
        if not path.is_file() or path.stat().st_size != size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == sha256


def _unpack(archive: SdkArchive, destination: Path) -> Path:
    """Unpack ``archive`` to ``destination`` atomically.

    Each process writes and checks its own temporary file in the destination
    directory and renames it into place, so concurrent first uses never see
    a partial archive; an existing copy that fails its check is replaced.
    """

    hint = (
        f"set {CACHE_ENVIRONMENT} to a writable directory to unpack the "
        "native SDK archive elsewhere"
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
    except OSError as error:
        raise SdkUnavailableError(
            f"cannot create the SDK cache directory {destination.parent}: "
            f"{error}; {hint}"
        ) from error
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(descriptor, "wb") as output,
            lzma.open(archive.path, "rb", format=lzma.FORMAT_XZ) as source,
        ):
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
                size += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != archive.size or digest.hexdigest() != archive.sha256:
            raise SdkUnavailableError(
                f"the compressed SDK archive {archive.path} does not unpack to "
                "the recorded size and SHA-256; reinstall pyamplicol"
            )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    except lzma.LZMAError as error:
        raise SdkUnavailableError(
            f"the compressed SDK archive {archive.path} is corrupt: {error}; "
            "reinstall pyamplicol"
        ) from error
    except OSError as error:
        raise SdkUnavailableError(
            f"cannot unpack the SDK archive into {destination.parent}: {error}; {hint}"
        ) from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return destination


def _resource_root() -> Path:
    # Finding the top-level package does not execute its __init__.py. SDK
    # compiler/linker queries must not acquire a numerical-runtime licence.
    spec = importlib.util.find_spec("pyamplicol")
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        raise SdkUnavailableError(
            "Rusticol's static SDK requires a normally installed, unpacked "
            "pyamplicol wheel"
        )
    return Path(spec.origin).resolve().parent / "_sdk"


def _package_version() -> str:
    # Reuse the authoritative, stdlib-only version/source-runtime validation.
    # Importing it through pyamplicol would first initialize Symbolica.
    versions = runpy.run_path(
        str(_resource_root().parent / "_internal" / "versions.py")
    )
    return str(versions["package_version"]())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SdkUnavailableError(
            "Rusticol's native SDK is unavailable in this source tree; "
            "install a pyamplicol binary wheel or build one with 'just wheel'"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SdkUnavailableError(f"invalid SDK metadata object: {path}")
    return payload


def _confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SdkUnavailableError(f"SDK metadata escapes its resource root: {relative}")
    if not path.is_file():
        raise SdkUnavailableError(f"missing SDK resource: {relative}")
    return path


def _sdk_archive(root: Path, metadata: dict[str, Any], version: str) -> SdkArchive:
    """The shipped archive: plain, or compressed with its unpacked digest."""

    relative = str(metadata["archive"])
    compression = metadata.get("archive_compression")
    if compression is None:
        return SdkArchive(_confined(root, relative), version)
    if (
        not isinstance(compression, dict)
        or compression.get("format") != ARCHIVE_COMPRESSION
        or not relative.endswith(f".{ARCHIVE_COMPRESSION}")
        or not isinstance(compression.get("sha256"), str)
        or _SHA256.fullmatch(compression["sha256"]) is None
        or not isinstance(compression.get("size"), int)
        or isinstance(compression["size"], bool)
        or compression["size"] <= 0
    ):
        raise SdkUnavailableError("invalid compressed SDK archive metadata")
    return SdkArchive(
        _confined(root, relative),
        version,
        sha256=compression["sha256"],
        size=compression["size"],
    )


def load_sdk_info() -> SdkInfo:
    """Read the installed SDK's metadata.

    Nothing is unpacked here: a compressed archive is unpacked when
    ``SdkInfo.library`` (or a link line using it) is first needed.
    """

    root = _resource_root()
    metadata = _load_json(root / "metadata.json")
    link = _load_json(root / "link.json")
    if metadata.get("schema_version") != 1 or link.get("schema_version") != 1:
        raise SdkUnavailableError("unsupported Rusticol SDK metadata schema")
    if metadata.get("target") != link.get("target"):
        raise SdkUnavailableError("Rusticol SDK target metadata is inconsistent")

    package_version = _package_version()
    metadata_version = str(metadata.get("version"))
    if package_version != metadata_version:
        raise SdkUnavailableError(
            "Rusticol SDK metadata version does not match pyamplicol"
        )
    archive = _sdk_archive(root, metadata, package_version)
    for required in (
        "include/rusticol.h",
        "include/rusticol.hpp",
        "fortran/rusticol.f90",
        "rust/rusticol.rs",
    ):
        _confined(root, required)

    return SdkInfo(
        abi_version=int(metadata["abi_version"]),
        package_version=package_version,
        target=str(metadata["target"]),
        include_dir=(root / "include").resolve(),
        archive=archive,
        fortran_source=_confined(root, "fortran/rusticol.f90"),
        rust_source=_confined(
            root, str(metadata.get("rust_source", "rust/rusticol.rs"))
        ),
        system_libraries=tuple(map(str, link.get("system_libraries", []))),
        frameworks=tuple(map(str, link.get("frameworks", []))),
        shared_libraries=tuple(
            _confined(root, str(relative))
            for relative in link.get("shared_libraries", [])
        ),
    )


def prune_other_versions(info: SdkInfo) -> tuple[Path, ...]:
    """Remove the SDK cache copies of other package versions or archives.

    A compressed archive is unpacked once per package version and archive
    digest (``<root>/sdk/<version>-<sha256>/``), so upgrades leave the copies
    of earlier versions in the user cache. Nothing deletes them
    automatically: this explicit, opt-in call (``rusticol-config
    --prune-other-versions``) removes every ``<version>-<sha256>`` directory
    under ``<root>/sdk/`` other than this installation's own and returns the
    removed directories. Other files and directories under the cache root are
    never touched; a plain (uncompressed) installation keeps no copy, so every
    copy found is another installation's."""

    import shutil

    root = sdk_cache_root()
    own = info.archive.cache_path
    keep = None if own is None else own.parent
    removed = []
    if not root.is_dir():
        return ()
    for entry in sorted(root.iterdir()):
        version, _sep, digest = entry.name.rpartition("-")
        if not (entry.is_dir() and version and _SHA256.fullmatch(digest)):
            continue
        if keep is not None and entry.resolve() == keep.resolve():
            continue
        shutil.rmtree(entry)
        removed.append(entry)
    return tuple(removed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rusticol-config",
        description="Report paths and typed linker arguments for Rusticol's SDK.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--abi-version", action="store_true")
    group.add_argument("--version", action="store_true")
    group.add_argument("--target", action="store_true")
    group.add_argument("--include-dir", action="store_true")
    group.add_argument("--library", action="store_true")
    group.add_argument("--fortran-source", action="store_true")
    group.add_argument("--rust-source", action="store_true")
    group.add_argument("--cflags", action="store_true")
    group.add_argument("--libs", action="store_true")
    group.add_argument("--rustflags", action="store_true")
    group.add_argument("--cargo-rustflags", action="store_true")
    group.add_argument("--json", action="store_true")
    group.add_argument(
        "--prune-other-versions",
        action="store_true",
        help="remove the user-cache SDK copies of other package versions "
        "(opt-in; never done automatically); prints the removed directories",
    )
    return parser


def _output(args: argparse.Namespace, info: SdkInfo) -> str:
    if args.abi_version:
        return str(info.abi_version)
    if args.version:
        return info.package_version
    if args.target:
        return info.target
    if args.include_dir:
        return str(info.include_dir)
    if args.library:
        return str(info.library)
    if args.fortran_source:
        return str(info.fortran_source)
    if args.rust_source:
        return str(info.rust_source)
    if args.cflags:
        return shlex.join(info.cflags)
    if args.libs:
        return shlex.join(info.link_flags)
    if args.rustflags:
        return shlex.join(info.rust_flags)
    if args.cargo_rustflags:
        return info.cargo_encoded_rust_flags
    if args.prune_other_versions:
        return "\n".join(str(path) for path in prune_other_versions(info))
    return json.dumps(info.to_json(), indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # A compressed archive is unpacked here by the outputs that name it.
        output = _output(args, load_sdk_info())
    except SdkUnavailableError as error:
        _parser().error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
