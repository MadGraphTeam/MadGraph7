#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate release legal metadata against release or candidate Cargo inputs."""

from __future__ import annotations

import argparse
import json
import string
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import check_rust_licenses as strict_legal

ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE = Path("config/release-dependencies.toml")
CANDIDATE_LOCK_RELATIVE = Path("dependencies/candidate-Cargo.lock")
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


@dataclass(frozen=True, order=True)
class InventoryIdentity:
    name: str
    version: str
    source: str
    checksum: str


@dataclass(frozen=True)
class CandidatePackage:
    name: str
    version: str
    manifest: Path
    license: str
    license_file: str | None
    published_version: str
    published_checksum: str


@dataclass(frozen=True)
class LegalInventoryConfig:
    cargo_lock: Path
    candidate_cargo_lock: Path
    rust_inventory: Path
    registry_source: str
    candidate_packages: tuple[CandidatePackage, ...]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a TOML table")
    return payload


def _safe_relative(value: object, *, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a relative path string")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} is unsafe: {value!r}")
    return path


def _required_string(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _table_list(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be an array of tables")
    return value


def _checksum(entry: dict[str, Any], field: str) -> str:
    value = _required_string(entry, field)
    if len(value) != 64 or any(
        character not in string.hexdigits for character in value
    ):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest")
    return value.lower()


def _load_config(root: Path) -> LegalInventoryConfig:
    payload = _read_toml(root / CONFIG_RELATIVE)
    if payload.get("schema_version") != 1:
        raise ValueError("release dependency inventory must use schema_version = 1")
    cargo_lock = _safe_relative(payload.get("cargo_lock"), field="cargo_lock")
    candidate_cargo_lock = _safe_relative(
        payload.get("candidate_cargo_lock"), field="candidate_cargo_lock"
    )
    rust_inventory = _safe_relative(
        payload.get("rust_inventory"), field="rust_inventory"
    )
    if cargo_lock != Path("Cargo.lock"):
        raise ValueError("cargo_lock must be 'Cargo.lock'")
    if candidate_cargo_lock != CANDIDATE_LOCK_RELATIVE:
        raise ValueError(
            f"candidate_cargo_lock must be {CANDIDATE_LOCK_RELATIVE.as_posix()!r}"
        )
    if rust_inventory != strict_legal.INVENTORY_RELATIVE:
        raise ValueError(
            f"rust_inventory must be {strict_legal.INVENTORY_RELATIVE.as_posix()!r}"
        )
    registry_source = _required_string(payload, "registry_source")
    if registry_source != CRATES_IO_SOURCE:
        raise ValueError(f"registry_source must be {CRATES_IO_SOURCE!r}")

    candidate_packages: list[CandidatePackage] = []
    for entry in _table_list(payload, "candidate_path_package"):
        license_expression = _required_string(entry, "license")
        if not strict_legal._valid_spdx(license_expression):
            raise ValueError(
                f"candidate package has invalid SPDX expression {license_expression!r}"
            )
        raw_license_file = entry.get("license_file")
        if raw_license_file is not None and not isinstance(raw_license_file, str):
            raise ValueError("license_file must be a string when present")
        if isinstance(raw_license_file, str):
            _safe_relative(raw_license_file, field="license_file")
        manifest = _safe_relative(entry.get("manifest"), field="manifest")
        if manifest.parts[:2] != ("dependencies", "checkouts"):
            raise ValueError("candidate manifests must be under dependencies/checkouts")
        if isinstance(raw_license_file, str) and Path(raw_license_file).parts[0] != (
            "licenses"
        ):
            raise ValueError("candidate license files must be under licenses/")
        candidate_packages.append(
            CandidatePackage(
                name=_required_string(entry, "name"),
                version=_required_string(entry, "version"),
                manifest=manifest,
                license=license_expression,
                license_file=raw_license_file,
                published_version=_required_string(entry, "published_version"),
                published_checksum=_checksum(entry, "published_checksum"),
            )
        )
    candidate_keys = [(package.name, package.version) for package in candidate_packages]
    if candidate_keys != sorted(candidate_keys):
        raise ValueError("candidate path packages must use name/version order")
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate path package declarations contain duplicates")

    published = {
        InventoryIdentity(
            package.name,
            package.published_version,
            registry_source,
            package.published_checksum,
        )
        for package in candidate_packages
    }
    if len(published) != len(candidate_packages):
        raise ValueError("candidate packages repeat a published inventory identity")
    return LegalInventoryConfig(
        cargo_lock=cargo_lock,
        candidate_cargo_lock=candidate_cargo_lock,
        rust_inventory=rust_inventory,
        registry_source=registry_source,
        candidate_packages=tuple(candidate_packages),
    )


def _inventory_identity(package: dict[str, Any]) -> InventoryIdentity:
    return InventoryIdentity(
        name=str(package.get("name", "")),
        version=str(package.get("version", "")),
        source=str(package.get("source", "")),
        checksum=str(package.get("checksum", "")),
    )


def _package_key(package: dict[str, Any]) -> tuple[str, str]:
    return str(package.get("name", "")), str(package.get("version", ""))


def candidate_package_keys(root: Path = ROOT) -> set[tuple[str, str]]:
    """Return candidate-only path identities from the candidate Cargo lock."""

    root = root.resolve()
    config = _load_config(root)
    lock = _read_toml(root / config.candidate_cargo_lock)
    inventory = _read_toml(root / config.rust_inventory)
    first_party = {
        _package_key(package)
        for package in strict_legal._table_list(inventory, "first_party")
    }
    source_less = {
        _package_key(package)
        for package in strict_legal._table_list(lock, "package")
        if not package.get("source")
    }
    return source_less - first_party


def _configuration_issues(
    inventory: dict[str, Any], config: LegalInventoryConfig
) -> list[strict_legal.LicenseIssue]:
    issues: list[strict_legal.LicenseIssue] = []
    inventory_packages = strict_legal._table_list(inventory, "package")
    inventory_identities = [
        _inventory_identity(package) for package in inventory_packages
    ]
    if len(inventory_identities) != len(set(inventory_identities)):
        issues.append(
            strict_legal.LicenseIssue(
                "candidate-inventory-duplicate",
                "curated inventory contains duplicate Cargo identities",
            )
        )
    inventory_keys = [
        strict_legal._package_key(package) for package in inventory_packages
    ]
    if inventory_keys != sorted(inventory_keys):
        issues.append(
            strict_legal.LicenseIssue(
                "candidate-inventory-order",
                "curated inventory is not in Cargo identity order",
            )
        )
    by_identity = {
        _inventory_identity(package): package for package in inventory_packages
    }
    for candidate in config.candidate_packages:
        published = InventoryIdentity(
            candidate.name,
            candidate.published_version,
            config.registry_source,
            candidate.published_checksum,
        )
        entry = by_identity.get(published)
        if entry is None:
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-published-inventory",
                    f"missing pinned published inventory row for {candidate.name} "
                    f"{candidate.published_version}",
                )
            )
            continue
        if (
            entry.get("license") != candidate.license
            or entry.get("license_file") != candidate.license_file
        ):
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-license-drift",
                    f"candidate legal declaration for {candidate.name} does not "
                    "match its pinned published inventory row",
                )
            )
    return issues


def _manifest_issues(
    root: Path, config: LegalInventoryConfig
) -> list[strict_legal.LicenseIssue]:
    issues: list[strict_legal.LicenseIssue] = []
    resolved_root = root.resolve()
    for candidate in config.candidate_packages:
        manifest_path = root / candidate.manifest
        try:
            resolved_manifest = manifest_path.resolve(strict=True)
            if not resolved_manifest.is_relative_to(resolved_root):
                raise ValueError("manifest resolves outside the repository")
            manifest = _read_toml(resolved_manifest)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-manifest",
                    f"cannot inspect {candidate.manifest}: {exc}",
                )
            )
            continue
        package = manifest.get("package")
        if not isinstance(package, dict):
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-manifest",
                    f"{candidate.manifest} has no package table",
                )
            )
            continue
        actual_key = str(package.get("name", "")), str(package.get("version", ""))
        if actual_key != (candidate.name, candidate.version):
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-manifest-identity",
                    f"{candidate.manifest} declares {actual_key}, expected "
                    f"{(candidate.name, candidate.version)}",
                )
            )
        manifest_license = package.get("license")
        if isinstance(manifest_license, str) and manifest_license != candidate.license:
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-manifest-license",
                    f"{candidate.manifest} declares {manifest_license!r}, expected "
                    f"{candidate.license!r}",
                )
            )
        license_file = package.get("license-file")
        if manifest_license is None and isinstance(license_file, str):
            source_license = resolved_manifest.parent / license_file
            if not source_license.is_file() or not source_license.read_bytes():
                issues.append(
                    strict_legal.LicenseIssue(
                        "candidate-manifest-license-file",
                        f"{candidate.manifest} has no nonempty {license_file}",
                    )
                )
        elif manifest_license is None:
            issues.append(
                strict_legal.LicenseIssue(
                    "candidate-manifest-license",
                    f"{candidate.manifest} declares neither license nor license-file",
                )
            )
    return issues


def _candidate_projection(
    root: Path,
    lock: dict[str, Any],
    inventory: dict[str, Any],
    config: LegalInventoryConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[strict_legal.LicenseIssue]]:
    issues = [
        *_configuration_issues(inventory, config),
        *_manifest_issues(root, config),
    ]
    lock_packages = strict_legal._table_list(lock, "package")
    inventory_first_party = strict_legal._table_list(inventory, "first_party")
    first_party = {_package_key(package) for package in inventory_first_party}
    configured = {
        (package.name, package.version): package
        for package in config.candidate_packages
    }
    actual_path = {
        _package_key(package) for package in lock_packages if not package.get("source")
    }
    expected_path = first_party | set(configured)
    if actual_path != expected_path:
        issues.append(
            strict_legal.LicenseIssue(
                "candidate-path-inventory",
                "source-less Cargo packages do not match first-party plus declared "
                f"candidate packages; missing={sorted(expected_path - actual_path)}, "
                f"extra={sorted(actual_path - expected_path)}",
            )
        )

    projected_lock_packages: list[dict[str, Any]] = []
    for package in lock_packages:
        projected = dict(package)
        candidate = configured.get(_package_key(package))
        if not package.get("source") and candidate is not None:
            projected["source"] = f"candidate+path:{candidate.manifest.as_posix()}"
            projected.pop("checksum", None)
        projected_lock_packages.append(projected)

    published_replacements = {
        InventoryIdentity(
            package.name,
            package.published_version,
            config.registry_source,
            package.published_checksum,
        )
        for package in config.candidate_packages
    }
    projected_inventory_packages = [
        dict(package)
        for package in strict_legal._table_list(inventory, "package")
        if _inventory_identity(package) not in published_replacements
    ]
    for candidate in config.candidate_packages:
        entry: dict[str, Any] = {
            "name": candidate.name,
            "version": candidate.version,
            "source": f"candidate+path:{candidate.manifest.as_posix()}",
            "license": candidate.license,
        }
        if candidate.license_file is not None:
            entry["license_file"] = candidate.license_file
        projected_inventory_packages.append(entry)
    projected_inventory_packages.sort(key=strict_legal._package_key)

    projected_lock = dict(lock)
    projected_lock["package"] = projected_lock_packages
    projected_inventory = dict(inventory)
    projected_inventory["package"] = projected_inventory_packages
    return projected_lock, projected_inventory, issues


def _candidate_issues(
    root: Path,
    lock: dict[str, Any],
    inventory: dict[str, Any],
    config: LegalInventoryConfig,
) -> list[strict_legal.LicenseIssue]:
    projected_lock, projected_inventory, issues = _candidate_projection(
        root, lock, inventory, config
    )
    issues.extend(
        [
            *strict_legal._inventory_issues(projected_lock, projected_inventory),
            *strict_legal._legal_file_issues(root, projected_inventory),
            *strict_legal._packaging_issues(root, projected_inventory),
            *strict_legal._notice_issues(root),
            *strict_legal._static_link_compliance_issues(
                root, projected_lock, projected_inventory
            ),
        ]
    )
    return issues


def check_repository(
    root: Path = ROOT, *, mode: str
) -> list[strict_legal.LicenseIssue]:
    """Run the strict release gate or its declared candidate projection."""

    if mode not in {"candidate", "release"}:
        raise ValueError(f"unsupported legal inventory mode {mode!r}")
    root = root.resolve()
    strict_release_issues = (
        strict_legal.check_repository(root) if mode == "release" else []
    )
    if mode == "release" and not (root / CONFIG_RELATIVE).is_file():
        return strict_release_issues
    try:
        config = _load_config(root)
        inventory = _read_toml(root / config.rust_inventory)
        config_issues = _configuration_issues(inventory, config)
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        issues = [strict_legal.LicenseIssue("legal-inventory-config", str(exc))]
        return sorted(
            [*strict_release_issues, *issues],
            key=lambda issue: (issue.code, issue.message),
        )

    if mode == "release":
        issues = [*config_issues, *strict_release_issues]
    else:
        try:
            lock = _read_toml(root / config.candidate_cargo_lock)
            issues = _candidate_issues(root, lock, inventory, config)
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            tomllib.TOMLDecodeError,
        ) as exc:
            issues = [
                *config_issues,
                strict_legal.LicenseIssue("legal-inventory-read", str(exc)),
            ]
    return sorted(issues, key=lambda issue: (issue.code, issue.message))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (defaults to the script's checkout)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--mode",
        choices=("candidate", "release"),
        default="release",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = check_repository(args.root, mode=args.mode)
    candidate_blockers = strict_legal.blocking_issues(issues, mode="candidate")
    release_blockers = strict_legal.blocking_issues(issues, mode="release")
    requested_blockers = strict_legal.blocking_issues(issues, mode=args.mode)
    if args.json:
        print(
            json.dumps(
                {
                    "candidate_ready": not candidate_blockers,
                    "issues": [asdict(issue) for issue in issues],
                    "release_ready": not release_blockers,
                    "requested_mode": args.mode,
                    "requested_mode_ready": not requested_blockers,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif requested_blockers:
        for issue in requested_blockers:
            print(f"[{issue.code}] {issue.message}", file=sys.stderr)
    elif release_blockers:
        print("Candidate legal checks passed; release readiness is FALSE")
        for issue in release_blockers:
            print(f"[{issue.code}] {issue.message}")
    else:
        print("Release legal inventory passed")
    return 0 if not requested_blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
