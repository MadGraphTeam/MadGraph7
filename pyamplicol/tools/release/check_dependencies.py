#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate the small published or contributor dependency contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "release-lock.toml"
CONTRIBUTOR_LOCK_PATH = ROOT / "dependencies" / "contributor-lock.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CARGO_LOCK_PATH = ROOT / "Cargo.lock"
RUST_TOOLCHAIN_PATH = ROOT / "rust-toolchain.toml"
STATE_PATH = ROOT / "dependencies" / "install-state.json"
CANDIDATE_LOCK_PATH = ROOT / "dependencies" / "candidate-Cargo.lock"
CARGO_CONFIG_PATH = ROOT / "dependencies" / "candidate-cargo-config.toml"
CHECKOUTS_PATH = ROOT / "dependencies" / "checkouts"

_REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_LOCAL_CRATES = {"rusticol-capi", "rusticol-core", "rusticol-python"}
_CANDIDATE_LOCAL_CRATES = {
    *_LOCAL_CRATES,
    "graphica",
    "numerica",
    "symbolica",
    "symjit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_NAME = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


def canonicalize_name(value: str) -> str:
    return _CANONICAL_NAME.sub("-", value).lower()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return payload


def _load_lock() -> dict[str, Any]:
    payload = _load_toml(LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError("dependencies/release-lock.toml must use schema_version = 1")
    return payload


def _load_contributor_lock() -> dict[str, Any]:
    payload = _load_toml(CONTRIBUTOR_LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "dependencies/contributor-lock.toml must use schema_version = 1"
        )
    return payload


def _locked_python_dependencies(lock: dict[str, Any]) -> dict[str, str]:
    raw = lock.get("python_dependencies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("release lock must list exact Python dependencies")
    dependencies: dict[str, str] = {}
    ordered_names: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"distribution", "version"}:
            raise ValueError(
                f"python_dependencies[{index}] must contain only distribution/version"
            )
        distribution = entry["distribution"]
        version = entry["version"]
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ValueError(
                f"python_dependencies[{index}] needs string distribution/version"
            )
        name = canonicalize_name(distribution)
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ValueError(f"invalid locked version for {name}: {version}") from error
        if name in dependencies:
            raise ValueError(f"release lock repeats Python dependency {name}")
        ordered_names.append(name)
        dependencies[name] = normalized_version
    if ordered_names != sorted(ordered_names):
        raise ValueError("release-lock Python dependencies must be name-sorted")
    return dependencies


def _project_python_dependencies() -> dict[str, str]:
    project = _load_toml(PYPROJECT_PATH).get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table")
    raw = project.get("dependencies")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("pyproject runtime dependencies must be strings")
    dependencies: dict[str, str] = {}
    for value in raw:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as error:
            raise ValueError(f"invalid pyproject dependency: {value}") from error
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            raise ValueError(
                f"release runtime dependency must be one exact published pin: {value}"
            )
        try:
            version = str(Version(specifiers[0].version))
        except InvalidVersion as error:
            raise ValueError(
                f"invalid pyproject dependency version: {value}"
            ) from error
        if name in dependencies:
            raise ValueError(f"pyproject repeats runtime dependency {name}")
        dependencies[name] = version
    return dependencies


def _release_contract_issues(
    lock: dict[str, Any], *, candidate: bool
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    forbidden_tables = {"legal_status", "python_runtime_lock"}.intersection(lock)
    if forbidden_tables:
        issues.append(
            GateIssue(
                "release-lock-scope",
                "release-lock.toml contains retired release machinery: "
                + ", ".join(sorted(forbidden_tables)),
            )
        )
    try:
        locked = _locked_python_dependencies(lock)
        project = _project_python_dependencies()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [*issues, GateIssue("python-dependency-contract", str(error))]
    if locked != project:
        issues.append(
            GateIssue(
                "python-dependency-contract",
                "pyproject exact runtime pins disagree with release-lock.toml; "
                f"lock={locked}, pyproject={project}",
            )
        )

    symbolica = lock.get("symbolica")
    loader = lock.get("ufo_model_loader")
    if not isinstance(symbolica, dict) or not isinstance(loader, dict):
        issues.append(
            GateIssue(
                "published-dependency-contract",
                "release lock needs Symbolica and ufo-model-loader compatibility data",
            )
        )
        return issues
    allowed_symbolica = {
        "python_distribution",
        "python_version",
        "rust_crate",
        "rust_version",
        "published_symjit_version",
        "serialization_abi",
        "release_status",
    }
    allowed_loader = {
        "python_distribution",
        "required_version",
        "latest_verified_published_version",
        "release_status",
    }
    if set(symbolica) != allowed_symbolica or set(loader) != allowed_loader:
        issues.append(
            GateIssue(
                "release-lock-scope",
                "published dependency sections must contain compatibility data only",
            )
        )
    symbolica_name = canonicalize_name(str(symbolica.get("python_distribution", "")))
    loader_name = canonicalize_name(str(loader.get("python_distribution", "")))
    if locked.get(symbolica_name) != str(symbolica.get("python_version", "")):
        issues.append(
            GateIssue(
                "symbolica-pin",
                "Symbolica compatibility data disagrees with the exact Python pin",
            )
        )
    if locked.get(loader_name) != str(loader.get("required_version", "")):
        issues.append(
            GateIssue(
                "ufo-loader-pin",
                "ufo-model-loader compatibility data disagrees with the exact pin",
            )
        )
    if loader.get("latest_verified_published_version") != loader.get(
        "required_version"
    ):
        issues.append(
            GateIssue(
                "ufo-loader-unverified",
                "the required ufo-model-loader version is not the verified release",
            )
        )
    if not candidate and symbolica.get("release_status") != "verified":
        issues.append(
            GateIssue(
                "symbolica-unverified",
                "the exact published Symbolica/SymJIT pair is not yet verified to "
                "contain the fixes used by contributor builds",
            )
        )
    if not candidate and loader.get("release_status") != "verified":
        issues.append(
            GateIssue(
                "ufo-loader-unverified",
                "the exact published ufo-model-loader release is not verified",
            )
        )
    return issues


def _cargo_packages(path: Path) -> list[dict[str, Any]]:
    payload = _load_toml(path)
    packages = payload.get("package")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        raise ValueError(f"{path} has no Cargo package array")
    return packages


def _registry_source_issues(
    packages: list[dict[str, Any]], *, local_crates: set[str], prefix: str
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    for package in packages:
        name = str(package.get("name", ""))
        source = package.get("source")
        if name in local_crates:
            if source is not None:
                issues.append(
                    GateIssue(
                        f"{prefix}-cargo-local-source",
                        f"local crate {name} unexpectedly has source {source}",
                    )
                )
            continue
        if source != _REGISTRY_SOURCE:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-nonregistry",
                    f"Cargo.lock package {name} is not an exact crates.io package",
                )
            )
            continue
        checksum = package.get("checksum")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-checksum",
                    f"Cargo.lock package {name} has no crates.io checksum",
                )
            )
    return issues


def _cargo_manifest_pin_issues(lock: dict[str, Any]) -> list[GateIssue]:
    symbolica = lock["symbolica"]
    expected = {
        "symbolica": f"={symbolica['rust_version']}",
        "symjit": f"={symbolica['published_symjit_version']}",
    }
    try:
        root = _load_toml(ROOT / "Cargo.toml")
        core = _load_toml(ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml")
        root_dependencies = root["workspace"]["dependencies"]
        core_dependencies = core["dependencies"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-manifest", f"invalid Cargo manifest: {error}")]
    issues: list[GateIssue] = []
    if "patch" in root:
        issues.append(
            GateIssue(
                "release-cargo-patch",
                "release Cargo.toml may not contain a [patch] table",
            )
        )
    for name, version in expected.items():
        tables = [core_dependencies]
        if name == "symbolica":
            tables.append(root_dependencies)
        for table in tables:
            entry = table.get(name) if isinstance(table, dict) else None
            if not isinstance(entry, dict) or entry.get("version") != version:
                issues.append(
                    GateIssue(
                        "release-cargo-pin",
                        f"Cargo manifest must require {name} {version} exactly",
                    )
                )
            elif "git" in entry or "path" in entry:
                issues.append(
                    GateIssue(
                        "release-cargo-source",
                        f"release Cargo dependency {name} may not use git/path",
                    )
                )
    return issues


def _release_cargo_lock_issues(lock: dict[str, Any]) -> list[GateIssue]:
    try:
        packages = _cargo_packages(CARGO_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-lock", str(error))]
    issues = _registry_source_issues(
        packages, local_crates=_LOCAL_CRATES, prefix="release"
    )
    symbolica = lock.get("symbolica", {})
    expected = {
        str(symbolica.get("rust_crate", "symbolica")): str(
            symbolica.get("rust_version", "")
        ),
        "symjit": str(symbolica.get("published_symjit_version", "")),
    }
    for name, version in expected.items():
        matches = [
            package
            for package in packages
            if package.get("name") == name and package.get("version") == version
        ]
        if len(matches) != 1 or matches[0].get("source") != _REGISTRY_SOURCE:
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    f"Cargo.lock must resolve published {name}=={version} exactly",
                )
            )
    return [*issues, *_cargo_manifest_pin_issues(lock)]


def _toolchain_issues(lock: dict[str, Any]) -> list[GateIssue]:
    toolchain = lock.get("toolchain")
    if not isinstance(toolchain, dict):
        return [GateIssue("toolchain-contract", "release lock has no toolchain table")]
    issues: list[GateIssue] = []
    try:
        rust = _load_toml(RUST_TOOLCHAIN_PATH)["toolchain"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("toolchain-contract", f"invalid rust-toolchain.toml: {error}")
        ]
    if not isinstance(rust, dict) or rust.get("channel") != toolchain.get(
        "rust_toolchain"
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "rust-toolchain.toml disagrees with release-lock.toml",
            )
        )
    digest = toolchain.get("manylinux_image_digest")
    image = toolchain.get("manylinux_image")
    if (
        not isinstance(image, str)
        or not image
        or not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "manylinux image and digest must be pinned exactly",
            )
        )
    build_system = _load_toml(PYPROJECT_PATH).get("build-system", {})
    requirements = (
        build_system.get("requires", []) if isinstance(build_system, dict) else []
    )
    expected = {
        "maturin": str(toolchain.get("maturin", "")),
        "packaging": str(toolchain.get("packaging", "")),
    }
    observed: dict[str, str] = {}
    for raw in requirements if isinstance(requirements, list) else []:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            observed[canonicalize_name(requirement.name)] = specifiers[0].version
    if observed != expected:
        issues.append(
            GateIssue(
                "toolchain-contract",
                f"build-system pins disagree with release lock: {observed}",
            )
        )
    return issues


def _candidate_revisions(lock: dict[str, Any]) -> dict[str, str]:
    return {
        "gammaloop": str(lock["gammaloop_candidate"]["revision"]),
        "symbolica": str(lock["symbolica"]["candidate_revision"]),
        "symbolica-community": str(lock["symbolica"]["community_revision"]),
        "symjit": str(lock["symjit"]["candidate_revision"]),
    }


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _candidate_config_issues() -> list[GateIssue]:
    try:
        config = _load_toml(CARGO_CONFIG_PATH)
        patches = config["patch"]["crates-io"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("candidate-cargo-config", f"invalid Cargo patch config: {error}")
        ]
    if not isinstance(patches, dict) or not patches:
        return [
            GateIssue("candidate-cargo-config", "candidate Cargo patch table is empty")
        ]
    issues: list[GateIssue] = []
    checkout_root = CHECKOUTS_PATH.resolve()
    for name, entry in patches.items():
        path_value = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path_value, str):
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} has no local path",
                )
            )
            continue
        path = Path(path_value).resolve()
        try:
            path.relative_to(checkout_root)
        except ValueError:
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} escapes dependencies/checkouts",
                )
            )
            continue
        if not path.exists():
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} path is missing: {path}",
                )
            )
    return issues


def _candidate_issues(_release_lock: dict[str, Any]) -> list[GateIssue]:
    required = (
        CONTRIBUTOR_LOCK_PATH,
        STATE_PATH,
        CANDIDATE_LOCK_PATH,
        CARGO_CONFIG_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        return [
            GateIssue(
                "candidate-input-missing",
                "run 'just dev-install' before a candidate build; missing: "
                + ", ".join(str(path) for path in missing),
            )
        ]
    try:
        contributor = _load_contributor_lock()
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        packages = _cargo_packages(CANDIDATE_LOCK_PATH)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        return [GateIssue("candidate-input-invalid", str(error))]
    issues = _registry_source_issues(
        packages,
        local_crates=_CANDIDATE_LOCAL_CRATES,
        prefix="candidate",
    )
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        issues.append(
            GateIssue("candidate-state-invalid", "installer state must use schema 1")
        )
        return issues
    if state.get("publishable") is not False:
        issues.append(
            GateIssue(
                "candidate-state-publishable",
                "candidate installer state must explicitly be non-publishable",
            )
        )
    sources = state.get("sources")
    revisions = _candidate_revisions(contributor)
    if not isinstance(sources, dict):
        issues.append(
            GateIssue("candidate-state-invalid", "installer state has no source map")
        )
    else:
        for name, revision in revisions.items():
            entry = sources.get(name)
            checkout = CHECKOUTS_PATH / name
            if not isinstance(entry, dict) or entry.get("revision") != revision:
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate source {name} does not match contributor-lock.toml",
                    )
                )
            if name == "symjit":
                symjit = contributor["symjit"]
                archive_matches = (
                    isinstance(entry, dict)
                    and entry.get("version") == symjit.get("candidate_version")
                    and entry.get("archive_sha256") == symjit.get("archive_sha256")
                )
                if not checkout.is_dir() or not archive_matches:
                    issues.append(
                        GateIssue(
                            "candidate-source-revision",
                            "candidate SymJIT archive does not match "
                            "contributor-lock.toml",
                        )
                    )
            elif not checkout.is_dir() or _git_head(checkout) != revision:
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate checkout {name} is not at {revision}",
                    )
                )
    for entry in contributor.get("patches", []):
        relative = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(relative, str)
            or not (ROOT / "dependencies" / relative).is_file()
        ):
            issues.append(
                GateIssue(
                    "candidate-patch-missing",
                    f"contributor patch is missing: {relative}",
                )
            )
    return [*issues, *_candidate_config_issues()]


def _published(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "pyamplicol-release-gate"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _published_dependency_issues(lock: dict[str, Any]) -> list[GateIssue]:
    try:
        dependencies = _locked_python_dependencies(lock)
    except ValueError as error:
        return [GateIssue("python-dependency-contract", str(error))]
    issues: list[GateIssue] = []
    for name, version in dependencies.items():
        if not _published(f"https://pypi.org/pypi/{name}/{version}/json"):
            issues.append(
                GateIssue(
                    "python-release-unavailable",
                    f"{name}=={version} is unavailable from PyPI",
                )
            )
    symbolica = lock["symbolica"]
    for crate, version in (
        (symbolica["rust_crate"], symbolica["rust_version"]),
        ("symjit", symbolica["published_symjit_version"]),
    ):
        if not _published(f"https://crates.io/api/v1/crates/{crate}/{version}"):
            issues.append(
                GateIssue(
                    "rust-release-unavailable",
                    f"{crate}=={version} is unavailable from crates.io",
                )
            )
    return issues


def check(*, candidate: bool, online: bool) -> list[GateIssue]:
    try:
        lock = _load_lock()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-lock-invalid", str(error))]
    issues = [
        *_release_contract_issues(lock, candidate=candidate),
        *_release_cargo_lock_issues(lock),
        *_toolchain_issues(lock),
    ]
    if candidate:
        issues.extend(_candidate_issues(lock))
    elif online:
        issues.extend(_published_dependency_issues(lock))
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="validate source-checkout candidate inputs as non-publishable",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip package-index availability checks",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = check(candidate=args.candidate, online=not args.offline)
    payload = {
        "mode": "candidate" if args.candidate else "release",
        "ready": not issues,
        "issues": [{"code": issue.code, "message": issue.message} for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.message}", file=sys.stderr)
    else:
        print(f"{payload['mode']} dependency gate passed")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
