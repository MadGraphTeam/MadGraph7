#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Create the isolated patched contributor environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / "dependencies"
RELEASE_LOCK = DEPENDENCIES / "release-lock.toml"
CONTRIBUTOR_LOCK = DEPENDENCIES / "contributor-lock.toml"
# Retained as the public constant used by older contributor-side callers.
LOCK = RELEASE_LOCK
PYTHON_LOCK = DEPENDENCIES / "python-runtime-lock.toml"
CHECKOUTS = DEPENDENCIES / "checkouts"
WHEELHOUSE = DEPENDENCIES / "wheelhouse"
VENV = ROOT / ".venv"
STATE = DEPENDENCIES / "install-state.json"
CANDIDATE_LOCK = DEPENDENCIES / "candidate-Cargo.lock"
CARGO_CONFIG = DEPENDENCIES / "candidate-cargo-config.toml"
ARTIFACTS = ROOT / ".artifacts" / "candidate"
TRASH = ROOT / ".trash"

sys.path.insert(0, str(ROOT / "build_backend"))
from python_lock import load_python_runtime_lock  # noqa: E402

_SOURCE_TREE_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
}
_WORKSPACE_CRATES = frozenset(
    {
        "rusticol-capi",
        "rusticol-core",
        "rusticol-python",
    }
)
_CANDIDATE_PATH_CRATES = frozenset(
    {
        "graphica",
        "numerica",
        "symbolica",
        "symjit",
    }
)
_CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SetupError(RuntimeError):
    """Contributor setup could not be completed safely."""


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    revision: str
    branch: str | None = None

    @property
    def path(self) -> Path:
        return CHECKOUTS / self.key


class Runner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        rendered = [str(item) for item in command]
        suffix = f"  # cwd={cwd}" if cwd else ""
        print(f"$ {shlex.join(rendered)}{suffix}")
        if self.dry_run:
            return subprocess.CompletedProcess(rendered, 0, "", "")
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=capture,
        )
        if completed.returncode != 0:
            if capture:
                print(completed.stdout, end="")
                print(completed.stderr, end="", file=sys.stderr)
            raise SetupError(
                f"command exited with {completed.returncode}: {shlex.join(rendered)}"
            )
        return completed


def _load_lock(path: Path, description: str) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if payload.get("schema_version") != 1:
        raise SetupError(f"unsupported {description} schema")
    return payload


def _release_lock() -> dict[str, Any]:
    return _load_lock(RELEASE_LOCK, "dependency release-lock")


def _contributor_lock() -> dict[str, Any]:
    return _load_lock(CONTRIBUTOR_LOCK, "dependency contributor-lock")


def _lock() -> dict[str, Any]:
    """Return the contributor setup view without polluting release metadata."""

    release = _release_lock()
    contributor = _contributor_lock()
    payload = dict(release)
    for key, value in contributor.items():
        if (
            isinstance(value, dict)
            and isinstance(payload.get(key), dict)
        ):
            merged = dict(payload[key])
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload


def _sources(payload: dict[str, Any], *, with_legacy: bool) -> tuple[Source, ...]:
    symbolica = payload["symbolica"]
    gammaloop = payload["gammaloop_candidate"]
    sources = [
        Source(
            "symbolica",
            str(symbolica["source_url"]),
            str(symbolica["candidate_revision"]),
        ),
        Source(
            "symbolica-community",
            str(symbolica["community_url"]),
            str(symbolica["community_revision"]),
        ),
        Source(
            "symjit",
            str(payload["symjit"]["source_url"]),
            str(payload["symjit"]["candidate_revision"]),
        ),
        Source(
            "gammaloop",
            str(gammaloop["source_url"]),
            str(gammaloop["revision"]),
        ),
    ]
    if with_legacy:
        legacy = payload["legacy_amplicol"]
        sources.append(
            Source(
                "legacy-amplicol",
                str(legacy["source_url"]),
                str(legacy["revision"]),
                str(legacy["branch"]),
            )
        )
    return tuple(sources)


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["VIRTUAL_ENV"] = str(VENV)
    binary = VENV / ("Scripts" if os.name == "nt" else "bin")
    environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
    return environment


def _require_tools() -> None:
    missing = [name for name in ("cargo", "git", "rustc") if shutil.which(name) is None]
    if missing:
        raise SetupError("missing contributor tools: " + ", ".join(sorted(missing)))


def _archive_managed_state(runner: Runner) -> None:
    managed = (
        VENV,
        CHECKOUTS,
        WHEELHOUSE,
        STATE,
        CANDIDATE_LOCK,
        CARGO_CONFIG,
        ARTIFACTS,
    )
    present = [path for path in managed if path.exists() or path.is_symlink()]
    if not present:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = TRASH / f"dependency-reset-{stamp}"
    for path in present:
        relative = path.relative_to(ROOT)
        target = destination / relative
        print(f"$ mv {shlex.quote(str(path))} {shlex.quote(str(target))}")
        if runner.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def _git_head(runner: Runner, path: Path) -> str:
    completed = runner.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture=True,
    )
    return completed.stdout.strip()


def _source_tree_sha256(root: Path) -> str:
    """Hash every candidate source byte outside deterministic build caches."""

    digest = hashlib.sha256()
    for raw_directory, raw_directories, raw_files in os.walk(root, topdown=True):
        directory = Path(raw_directory)
        directories = sorted(
            name for name in raw_directories if name not in _SOURCE_TREE_EXCLUDES
        )
        raw_directories[:] = [
            name for name in directories if not (directory / name).is_symlink()
        ]
        entries = [
            *(
                directory / name
                for name in directories
                if (directory / name).is_symlink()
            ),
            *(
                directory / name
                for name in sorted(raw_files)
                if name not in _SOURCE_TREE_EXCLUDES
                and not name.endswith((".pyc", ".pyo"))
            ),
        ]
        for path in entries:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            mode = path.lstat().st_mode & 0o111
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(mode.to_bytes(2, "big"))
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"L")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            elif path.is_file():
                digest.update(b"F")
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            else:
                digest.update(b"O")
    return digest.hexdigest()


def _checkout(runner: Runner, source: Source, *, update: bool) -> None:
    if not source.path.exists():
        source.path.parent.mkdir(parents=True, exist_ok=True)
        clone_command = [
            "git",
            "clone",
            "--filter=blob:none",
        ]
        if source.branch is not None:
            clone_command.extend(
                ["--branch", source.branch, "--single-branch"]
            )
        clone_command.extend(
            ["--no-checkout", source.url, str(source.path)]
        )
        runner.run(
            clone_command
        )
        runner.run(
            ["git", "checkout", "--detach", source.revision],
            cwd=source.path,
        )
        return
    if runner.dry_run:
        print(f"# verify {source.key} at {source.revision}")
        return
    head = _git_head(runner, source.path)
    if head == source.revision:
        return
    if not update:
        raise SetupError(
            f"{source.path} is at {head}, expected {source.revision}; "
            "rerun with --update or --reset"
        )
    fetch_ref = source.branch or source.revision
    runner.run(["git", "fetch", "origin", fetch_ref], cwd=source.path)
    runner.run(
        ["git", "checkout", "--detach", source.revision],
        cwd=source.path,
    )


def _apply_patch(runner: Runner, checkout: Path, patch: Path) -> None:
    if runner.dry_run:
        print(f"$ git apply {shlex.quote(str(patch))}  # cwd={checkout}")
        return
    check = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        runner.run(["git", "apply", patch], cwd=checkout)
        return
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        return
    raise SetupError(
        f"patch does not apply cleanly to {checkout}: {patch.name}\n"
        f"{check.stderr}{reverse.stderr}"
    )


def _unapply_patch_if_present(runner: Runner, checkout: Path, patch: Path) -> None:
    if runner.dry_run:
        print(f"$ git apply --reverse {shlex.quote(str(patch))}  # cwd={checkout}")
        return
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        runner.run(["git", "apply", "--reverse", patch], cwd=checkout)


def _replace_section(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(?ms)^\[{re.escape(name)}\]\n.*?(?=^\[[^\n]+\]\n|\Z)")
    replacement = f"[{name}]\n{body.strip()}\n\n"
    if not pattern.search(text):
        return text.rstrip() + "\n\n" + replacement
    return pattern.sub(replacement, text, count=1)


def _patch_source_manifests(runner: Runner) -> None:
    if runner.dry_run:
        print("# rewrite candidate Cargo manifests to pinned local paths")
        return
    symbolica = CHECKOUTS / "symbolica"
    symjit = CHECKOUTS / "symjit"
    community = CHECKOUTS / "symbolica-community"
    gammaloop = CHECKOUTS / "gammaloop"

    symjit_cargo = symjit / "Cargo.toml"
    text = symjit_cargo.read_text(encoding="utf-8")
    text, count = re.subn(
        r'(?m)^crate-type\s*=\s*\["cdylib"\]\s*$',
        'crate-type = ["rlib"]',
        text,
        count=1,
    )
    if count == 0 and 'crate-type = ["rlib"]' not in text:
        raise SetupError("could not make SymJIT an rlib")
    symjit_cargo.write_text(text, encoding="utf-8")

    symbolica_cargo = symbolica / "Cargo.toml"
    text = symbolica_cargo.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^symjit\s*=.*$",
        'symjit = { path = "../symjit" }',
        text,
        count=1,
    )
    if count != 1:
        raise SetupError("could not point Symbolica at managed SymJIT")
    symbolica_cargo.write_text(text, encoding="utf-8")

    dependencies = (
        """
example_extension = { path = "example_extension" }
idenso = { path = "../gammaloop/crates/idenso", features = ["bincode", "python"] }
spenso = { path = "../gammaloop/crates/spenso", features = ["shadowing", "python"] }
spynso3 = { path = "../gammaloop/crates/spynso3" }
symbolica = { path = "../symbolica", features = ["python_export"] }
pyo3 = { version = "0.28", features = ["abi3"] }
"""
        'pyo3-stub-gen = { version = "0.17", optional = true, '
        'default-features = false, features = ["numpy"] }\n'
        """
mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }
"""
    )
    patches = """
graphica = { path = "../symbolica/lib/graphica" }
idenso = { path = "../gammaloop/crates/idenso" }
linnet = { path = "../gammaloop/crates/linnet" }
linnest = { path = "../gammaloop/crates/linnest" }
numerica = { path = "../symbolica/lib/numerica" }
spenso = { path = "../gammaloop/crates/spenso" }
spenso-hep-lib = { path = "../gammaloop/crates/spenso-hep-lib" }
spenso-macros = { path = "../gammaloop/crates/spenso-macros" }
spynso3 = { path = "../gammaloop/crates/spynso3" }
symbolica = { path = "../symbolica" }
"""
    community_cargo = community / "Cargo.toml"
    text = community_cargo.read_text(encoding="utf-8")
    if not re.search(r"(?m)^\[workspace\]\s*$", text):
        # Managed checkouts live below pyAmpliCol's workspace directory but are
        # independent build inputs.  An explicit empty workspace prevents Cargo
        # from adopting this package into the nearest ancestor workspace.
        text = "[workspace]\n\n" + text
    text = _replace_section(text, "dependencies", dependencies)
    text = _replace_section(text, "patch.crates-io", patches)
    text = text.replace('    "vakint/python_stubgen",\n', "")
    text = re.sub(
        r"(?m)^numerica\s*=\s*\{[^\n]*\}\s*$",
        'numerica = { path = "../symbolica/lib/numerica" }',
        text,
        count=1,
    )
    community_cargo.write_text(text.rstrip() + "\n", encoding="utf-8")

    example = community / "example_extension" / "Cargo.toml"
    text = example.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^symbolica\s*=\s*\{[^\n]*\}\s*$",
        'symbolica = { path = "../../symbolica", features = ["python_export"] }',
        text,
        count=1,
    )
    example.write_text(text.rstrip() + "\n", encoding="utf-8")

    community_lib = community / "src" / "lib.rs"
    text = community_lib.read_text(encoding="utf-8")
    text = text.replace(
        "    register_module!(m, vakint::symbolica_community_module::VakintWrapper);\n",
        "",
    )
    community_lib.write_text(text, encoding="utf-8")

    gammaloop_cargo = gammaloop / "Cargo.toml"
    text = gammaloop_cargo.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^symbolica\s*=\s*\{[^\n]*\}\s*$",
        (
            'symbolica = { path = "../symbolica", '
            'default-features = false, features = ["gmp"] }'
        ),
        text,
        count=1,
    )
    text = _replace_section(
        text,
        "patch.crates-io",
        """
graphica = { path = "../symbolica/lib/graphica" }
numerica = { path = "../symbolica/lib/numerica" }
symbolica = { path = "../symbolica" }
""",
    )
    gammaloop_cargo.write_text(text.rstrip() + "\n", encoding="utf-8")


def _patch_sources(runner: Runner, payload: dict[str, Any]) -> None:
    patch_series: list[tuple[Path, Path]] = []
    for entry in payload["patches"]:
        dependency = str(entry["dependency"])
        checkout_name = {
            "symbolica": "symbolica",
            "symjit": "symjit",
            "gammaloop": "gammaloop",
        }.get(dependency)
        if checkout_name is None:
            continue
        checkout = CHECKOUTS / checkout_name
        patch = DEPENDENCIES / str(entry["path"])
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SetupError(f"dependency patch digest changed: {patch}")
        patch_series.append((checkout, patch))
    # Later patches may overlap lines introduced by earlier ones, so checking
    # each patch independently is not an idempotent test of the final series.
    # Unwind our managed series in reverse order, then replay it from the lock.
    for checkout, patch in reversed(patch_series):
        _unapply_patch_if_present(runner, checkout, patch)
    for checkout, patch in patch_series:
        _apply_patch(runner, checkout, patch)
    _patch_source_manifests(runner)
    runner.run(["cargo", "update", "-p", "symjit"], cwd=CHECKOUTS / "symbolica")
    runner.run(
        ["cargo", "update", "-p", "symjit"],
        cwd=CHECKOUTS / "symbolica-community",
    )


def _write_cargo_config(runner: Runner) -> None:
    entries = {
        "graphica": CHECKOUTS / "symbolica" / "lib" / "graphica",
        "numerica": CHECKOUTS / "symbolica" / "lib" / "numerica",
        "symbolica": CHECKOUTS / "symbolica",
        "symjit": CHECKOUTS / "symjit",
    }
    text = ["# Generated by dependencies/install_dependencies.py", "[patch.crates-io]"]
    text.extend(
        f"{name} = {{ path = {json.dumps(str(path.resolve()))} }}"
        for name, path in entries.items()
    )
    print(f"# write {CARGO_CONFIG}")
    if runner.dry_run:
        return
    CARGO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CARGO_CONFIG.write_text("\n".join(text) + "\n", encoding="utf-8")


def _cargo_lock_packages(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SetupError(f"invalid Cargo lock file {path}: {error}") from error
    packages = payload.get("package")
    if payload.get("version") != 4 or not isinstance(packages, list):
        raise SetupError(f"Cargo lock file {path} must use format version 4")
    return packages


def _validate_release_cargo_lock(path: Path) -> None:
    """Require the canonical lock to contain published registry crates only."""

    invalid: list[str] = []
    for package in _cargo_lock_packages(path):
        name = str(package.get("name", "<unnamed>"))
        source = package.get("source")
        checksum = package.get("checksum")
        if name in _WORKSPACE_CRATES and source is None and checksum is None:
            continue
        if source != _CRATES_IO_SOURCE:
            invalid.append(f"{name} has non-registry source {source!r}")
            continue
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            invalid.append(f"{name} has no valid registry checksum")
    if invalid:
        raise SetupError(
            "canonical Cargo.lock is candidate/path-resolved; regenerate it "
            "without the candidate Cargo patch configuration:\n  "
            + "\n  ".join(invalid)
        )


def _validate_candidate_cargo_lock(path: Path) -> None:
    """Require only the managed patch crates to use local path resolution."""

    packages = _cargo_lock_packages(path)
    local_names = {
        str(package.get("name"))
        for package in packages
        if package.get("source") is None and package.get("checksum") is None
    }
    expected_local = _WORKSPACE_CRATES | _CANDIDATE_PATH_CRATES
    missing = sorted(expected_local - local_names)
    unexpected = sorted(local_names - expected_local)
    invalid_registry: list[str] = []
    for package in packages:
        name = str(package.get("name", "<unnamed>"))
        if name in expected_local:
            continue
        source = package.get("source")
        checksum = package.get("checksum")
        if (
            source != _CRATES_IO_SOURCE
            or not isinstance(checksum, str)
            or _SHA256_PATTERN.fullmatch(checksum) is None
        ):
            invalid_registry.append(name)
    if missing or unexpected or invalid_registry:
        details = []
        if missing:
            details.append("missing local crates: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected local crates: " + ", ".join(unexpected))
        if invalid_registry:
            details.append(
                "invalid registry crates: " + ", ".join(sorted(invalid_registry))
            )
        raise SetupError("invalid candidate Cargo lock: " + "; ".join(details))


def _write_candidate_lock(runner: Runner) -> None:
    if runner.dry_run:
        print(f"# resolve {CANDIDATE_LOCK} from canonical Cargo.lock")
        return
    release_lock = ROOT / "Cargo.lock"
    release_lock_bytes = release_lock.read_bytes()
    _validate_release_cargo_lock(release_lock)
    with tempfile.TemporaryDirectory(prefix="pyamplicol-candidate-lock-") as raw:
        temporary = Path(raw)
        shutil.copy2(ROOT / "Cargo.toml", temporary / "Cargo.toml")
        shutil.copy2(release_lock, temporary / "Cargo.lock")
        shutil.copytree(ROOT / "rust", temporary / "rust")
        runner.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        _rewrite_candidate_symjit_requirement(temporary)
        config = temporary / ".cargo" / "config.toml"
        config.parent.mkdir(parents=True)
        shutil.copy2(CARGO_CONFIG, config)
        # Resolving from the release lock preserves every unrelated registry
        # version while replacing only the explicitly patched candidate crates.
        runner.run(
            ["cargo", "metadata", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        _validate_candidate_cargo_lock(temporary / "Cargo.lock")
        runner.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        shutil.copy2(temporary / "Cargo.lock", CANDIDATE_LOCK)
    if release_lock.read_bytes() != release_lock_bytes:
        raise SetupError("candidate lock generation modified canonical Cargo.lock")


def _rewrite_candidate_symjit_requirement(root: Path) -> None:
    """Project the published release manifest onto the pinned candidate source."""

    lock = _lock()
    published = str(lock["symbolica"]["published_symjit_version"])
    candidate = str(lock["symjit"]["candidate_version"])
    manifest = root / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    pattern = rf'(?m)^(symjit\s*=\s*\{{\s*version\s*=\s*)"={re.escape(published)}"'
    updated, count = re.subn(pattern, rf'\g<1>"={candidate}"', text, count=1)
    if count != 1:
        raise SetupError(
            "could not project rusticol-core from the published SymJIT "
            f"requirement {published} to candidate {candidate}"
        )
    manifest.write_text(updated, encoding="utf-8")


def _runtime_requirements_text() -> str:
    runtime_lock = load_python_runtime_lock(PYTHON_LOCK)
    excluded = {"symbolica"}
    lines: list[str] = []
    for package in runtime_lock.packages:
        if package.name in excluded:
            continue
        if not package.artifacts:
            raise SetupError(
                f"locked runtime package {package.name} has no wheel artifacts"
            )
        lines.append(f"{package.distribution}=={package.version} \\")
        for index, artifact in enumerate(package.artifacts):
            continuation = " \\" if index < len(package.artifacts) - 1 else ""
            lines.append(f"    --hash=sha256:{artifact.sha256}{continuation}")
    return "\n".join(lines) + "\n"


def _ensure_venv(runner: Runner, payload: dict[str, Any]) -> None:
    if not _venv_python().is_file():
        runner.run([sys.executable, "-m", "venv", VENV])
    python = _venv_python()
    if runner.dry_run:
        print(f"# ensure pip is available in {VENV}")
    else:
        pip_probe = subprocess.run(
            [python, "-m", "pip", "--version"],
            env=_venv_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if pip_probe.returncode != 0:
            runner.run(
                [python, "-m", "ensurepip", "--upgrade"],
                env=_venv_environment(),
            )
    toolchain = payload["toolchain"]
    contributor_tools = [
        "pip",
        "build>=1.2,<2",
        "jsonschema>=4.22,<5",
        f"maturin=={toolchain['maturin']}",
        "mypy>=1.13,<2",
        "pytest>=8.3,<9",
        "ruff>=0.9,<1",
        "twine>=6,<7",
        "wheel>=0.45,<1",
    ]
    runner.run(
        [python, "-m", "pip", "install", "--upgrade", *contributor_tools],
        env=_venv_environment(),
    )
    requirements = _runtime_requirements_text()
    if runner.dry_run:
        print("# install the hash-locked non-candidate Python runtime closure")
    else:
        with tempfile.TemporaryDirectory(
            prefix="pyamplicol-runtime-requirements-"
        ) as raw_directory:
            requirement_path = Path(raw_directory) / "requirements.txt"
            requirement_path.write_text(requirements, encoding="utf-8")
            runner.run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--no-deps",
                    "--requirement",
                    requirement_path,
                ],
                env=_venv_environment(),
            )


def _ensure_just(runner: Runner) -> None:
    if shutil.which("just") is None:
        runner.run(["cargo", "install", "just", "--locked"])


def _single_wheel(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.whl"))
    if len(candidates) != 1:
        raise SetupError(
            f"expected one {prefix} wheel in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _archive_candidate_wheels(directory: Path, prefix: str) -> None:
    candidates = sorted(directory.glob(f"{prefix}*.whl"))
    if not candidates:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = TRASH / f"candidate-wheel-refresh-{stamp}" / directory.relative_to(ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in candidates:
        shutil.move(str(wheel), str(destination / wheel.name))


def _build_candidate_wheels(runner: Runner) -> None:
    python = _venv_python()
    environment = _venv_environment()
    symbolica_wheels = WHEELHOUSE / "symbolica"
    project_wheels = ARTIFACTS
    for directory in (symbolica_wheels, project_wheels):
        if not runner.dry_run:
            directory.mkdir(parents=True, exist_ok=True)

    runner.run(
        [
            python,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--interpreter",
            python,
            "--out",
            symbolica_wheels,
        ],
        cwd=CHECKOUTS / "symbolica-community",
        env=environment,
    )
    if not runner.dry_run:
        runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                _single_wheel(symbolica_wheels, "symbolica"),
            ],
            env=environment,
        )

    build_environment = dict(
        environment,
        PYAMPLICOL_BUILD_MODE="candidate",
    )
    if not runner.dry_run:
        _archive_candidate_wheels(project_wheels, "pyamplicol")
    runner.run(
        [
            python,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            project_wheels,
        ],
        cwd=ROOT,
        env=build_environment,
    )
    if not runner.dry_run:
        runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                _single_wheel(project_wheels, "pyamplicol"),
            ],
            env=environment,
        )


def _write_state(
    runner: Runner,
    payload: dict[str, Any],
    sources: tuple[Source, ...],
) -> None:
    if runner.dry_run:
        print(f"# write {STATE}")
        return
    source_state: dict[str, dict[str, str]] = {}
    for source in sources:
        head = _git_head(runner, source.path)
        source_state[source.key] = {
            "url": source.url,
            "revision": head,
            "worktree_sha256": _source_tree_sha256(source.path),
        }
        if source.branch is not None:
            source_state[source.key]["branch"] = source.branch
    state = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "publishable": False,
        "release_lock_sha256": hashlib.sha256(RELEASE_LOCK.read_bytes()).hexdigest(),
        "contributor_lock_sha256": hashlib.sha256(
            CONTRIBUTOR_LOCK.read_bytes()
        ).hexdigest(),
        "python_runtime_lock_sha256": hashlib.sha256(
            PYTHON_LOCK.read_bytes()
        ).hexdigest(),
        "candidate_lock_sha256": hashlib.sha256(
            CANDIDATE_LOCK.read_bytes()
        ).hexdigest(),
        "cargo_config_sha256": hashlib.sha256(CARGO_CONFIG.read_bytes()).hexdigest(),
        "sources": source_state,
        "patches": [
            {
                "dependency": entry["dependency"],
                "path": entry["path"],
                "sha256": entry["sha256"],
            }
            for entry in payload["patches"]
        ],
    }
    STATE.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--without-legacy-amplicol", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = _lock()
    runner = Runner(dry_run=args.dry_run)
    if args.reset:
        _archive_managed_state(runner)
    if not args.dry_run:
        _require_tools()
    sources = _sources(
        payload,
        with_legacy=not args.without_legacy_amplicol,
    )
    _ensure_just(runner)
    _ensure_venv(runner, payload)
    for source in sources:
        _checkout(runner, source, update=args.update)
    _patch_sources(runner, payload)
    _write_cargo_config(runner)
    _write_candidate_lock(runner)
    _write_state(runner, payload, sources)
    if not args.no_build:
        _build_candidate_wheels(runner)
    print(f"Contributor environment ready at {VENV}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
