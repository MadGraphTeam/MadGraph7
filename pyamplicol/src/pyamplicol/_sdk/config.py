# SPDX-License-Identifier: 0BSD
"""Locate and report native Rusticol SDK resources from an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import shlex
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SdkUnavailableError(RuntimeError):
    """Raised when a source tree has no staged native SDK."""


@dataclass(frozen=True)
class SdkInfo:
    abi_version: int
    package_version: str
    target: str
    include_dir: Path
    library: Path
    fortran_source: Path
    sbom: Path
    system_libraries: tuple[str, ...]
    frameworks: tuple[str, ...]

    @property
    def cflags(self) -> tuple[str, ...]:
        return (f"-I{self.include_dir}",)

    @property
    def link_flags(self) -> tuple[str, ...]:
        libraries = tuple(f"-l{name}" for name in self.system_libraries)
        frameworks = tuple(
            token for name in self.frameworks for token in ("-framework", name)
        )
        return (str(self.library), *libraries, *frameworks)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["include_dir"] = str(self.include_dir)
        payload["library"] = str(self.library)
        payload["fortran_source"] = str(self.fortran_source)
        payload["sbom"] = str(self.sbom)
        payload["cflags"] = list(self.cflags)
        payload["link_flags"] = list(self.link_flags)
        return payload


def _resource_root() -> Path:
    resource = importlib.resources.files("pyamplicol._sdk")
    if not isinstance(resource, os.PathLike):
        raise SdkUnavailableError(
            "Rusticol's static SDK requires a normally installed, unpacked "
            "pyamplicol wheel"
        )
    return Path(os.fspath(resource)).resolve()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_sdk_info() -> SdkInfo:
    root = _resource_root()
    metadata = _load_json(root / "metadata.json")
    link = _load_json(root / "link.json")
    if metadata.get("schema_version") != 1 or link.get("schema_version") != 1:
        raise SdkUnavailableError("unsupported Rusticol SDK metadata schema")
    if metadata.get("target") != link.get("target"):
        raise SdkUnavailableError("Rusticol SDK target metadata is inconsistent")

    package_version = importlib.metadata.version("pyamplicol")
    metadata_version = str(metadata.get("version"))
    if package_version != metadata_version:
        raise SdkUnavailableError(
            "Rusticol SDK metadata version does not match pyamplicol"
        )
    library = _confined(root, str(metadata["archive"]))
    expected_digest = str(metadata.get("archive_sha256", ""))
    if len(expected_digest) != 64 or _sha256(library) != expected_digest:
        raise SdkUnavailableError("Rusticol SDK archive digest does not match metadata")
    sbom = _confined(root, str(metadata.get("sbom", "")))
    expected_sbom_digest = str(metadata.get("sbom_sha256", ""))
    if len(expected_sbom_digest) != 64 or _sha256(sbom) != expected_sbom_digest:
        raise SdkUnavailableError("Rusticol SDK SBOM digest does not match metadata")
    for required in (
        "include/rusticol.h",
        "include/rusticol.hpp",
        "fortran/rusticol.f90",
    ):
        _confined(root, required)

    return SdkInfo(
        abi_version=int(metadata["abi_version"]),
        package_version=package_version,
        target=str(metadata["target"]),
        include_dir=(root / "include").resolve(),
        library=library,
        fortran_source=_confined(root, "fortran/rusticol.f90"),
        sbom=sbom,
        system_libraries=tuple(map(str, link.get("system_libraries", []))),
        frameworks=tuple(map(str, link.get("frameworks", []))),
    )


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
    group.add_argument("--sbom", action="store_true")
    group.add_argument("--cflags", action="store_true")
    group.add_argument("--libs", action="store_true")
    group.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        info = load_sdk_info()
    except (SdkUnavailableError, importlib.metadata.PackageNotFoundError) as error:
        _parser().error(str(error))

    if args.abi_version:
        print(info.abi_version)
    elif args.version:
        print(info.package_version)
    elif args.target:
        print(info.target)
    elif args.include_dir:
        print(info.include_dir)
    elif args.library:
        print(info.library)
    elif args.fortran_source:
        print(info.fortran_source)
    elif args.sbom:
        print(info.sbom)
    elif args.cflags:
        print(shlex.join(info.cflags))
    elif args.libs:
        print(shlex.join(info.link_flags))
    elif args.json:
        print(json.dumps(info.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
