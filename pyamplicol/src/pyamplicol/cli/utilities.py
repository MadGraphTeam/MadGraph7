# SPDX-License-Identifier: 0BSD
"""Checkout-independent utility commands for configuration and examples."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal

from pyamplicol.api.errors import ConfigurationError
from pyamplicol.config import resolution_to_dict, resolve_config
from pyamplicol.diagnostics import run_doctor, run_self_test

UtilityKind = Literal[
    "config-template",
    "config-resolve",
    "examples-list",
    "examples-copy",
    "examples-run",
    "profiling-campaign-copy",
    "doctor",
    "self-test",
]

_PROFILING_CAMPAIGN_REQUIRED_FILES = (
    Path("README.md"),
    Path("TABLE_FILLING.md"),
    Path("build_pdf.py"),
    Path("pyAmpliCol.tex"),
    Path("report-workspace.json"),
    Path("report_environment.json"),
    Path("report_environment.tex"),
    Path("result_tables.py"),
    Path("result_validation_summary.tex"),
    Path("steer_performance_campaign.py"),
)
_PROFILING_CAMPAIGN_GLOBS = (
    "result_*_table.tex",
    "section_*.tex",
    "results/*.json",
)
_PROFILING_CAMPAIGN_FILE_COUNT = 55
_PROFILING_CAMPAIGN_LOCAL_AMPLICOL = ".pyamplicol-original-amplicol"


@dataclass(frozen=True, slots=True)
class UtilityInvocation:
    kind: UtilityKind
    output_format: Literal["human", "json"] = "human"
    path: Path | None = None
    name: str | None = None
    force: bool = False
    overrides: tuple[str, ...] = ()
    local_amplicol: Path | None = None


@dataclass(frozen=True, slots=True)
class ExampleEntry:
    name: str
    action: str
    description: str


def _utility_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyamplicol")
    commands = parser.add_subparsers(dest="utility", required=True)

    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    template = config_commands.add_parser("template")
    template.add_argument("output", type=Path, nargs="?")
    template.add_argument("--force", action="store_true")
    resolve = config_commands.add_parser("resolve")
    resolve.add_argument("card", type=Path)
    resolve.add_argument("--set", dest="overrides", action="append", default=[])
    resolve.add_argument("--format", choices=("human", "json"), default="human")

    examples = commands.add_parser("examples")
    example_commands = examples.add_subparsers(dest="examples_command", required=True)
    listing = example_commands.add_parser("list")
    listing.add_argument("--format", choices=("human", "json"), default="human")
    copy = example_commands.add_parser("copy")
    copy.add_argument("destination", type=Path)
    copy.add_argument("--force", action="store_true")
    run = example_commands.add_parser("run")
    run.add_argument("name")
    run.add_argument("--set", dest="overrides", action="append", default=[])
    run.add_argument("--format", choices=("human", "json"), default="human")

    profiling_campaign = commands.add_parser("profiling-campaign")
    profiling_commands = profiling_campaign.add_subparsers(
        dest="profiling_campaign_command", required=True
    )
    campaign_copy = profiling_commands.add_parser("copy")
    campaign_copy.add_argument("destination", type=Path)
    campaign_copy.add_argument("--force", action="store_true")
    campaign_copy.add_argument(
        "--local-amplicol",
        type=Path,
        metavar="PATH_TO_COMPLETE_CHECKOUT",
        help=(
            "Record a clean, complete original-AmpliCol checkout as the "
            "default for this copied campaign."
        ),
    )

    for name in ("doctor", "self-test"):
        command = commands.add_parser(name)
        command.add_argument("--format", choices=("human", "json"), default="human")
    return parser


def parse_utility(argv: Sequence[str]) -> UtilityInvocation:
    namespace = _utility_parser().parse_args(argv)
    if namespace.utility == "config":
        if namespace.config_command == "template":
            return UtilityInvocation(
                "config-template",
                path=namespace.output,
                force=bool(namespace.force),
            )
        return UtilityInvocation(
            "config-resolve",
            output_format=namespace.format,
            path=namespace.card,
            overrides=tuple(namespace.overrides),
        )
    if namespace.utility == "examples":
        if namespace.examples_command == "list":
            return UtilityInvocation("examples-list", output_format=namespace.format)
        if namespace.examples_command == "copy":
            return UtilityInvocation(
                "examples-copy",
                path=namespace.destination,
                force=bool(namespace.force),
            )
        return UtilityInvocation(
            "examples-run",
            output_format=namespace.format,
            name=namespace.name,
            overrides=tuple(namespace.overrides),
        )
    if namespace.utility == "profiling-campaign":
        return UtilityInvocation(
            "profiling-campaign-copy",
            path=namespace.destination,
            force=bool(namespace.force),
            local_amplicol=namespace.local_amplicol,
        )
    return UtilityInvocation(namespace.utility, output_format=namespace.format)


def _source_examples_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples"
        if (candidate / "all_options.toml").is_file():
            return candidate
    raise ConfigurationError("packaged examples are unavailable")


def examples_root() -> Path:
    packaged = resources.files("pyamplicol").joinpath("_examples")
    if not isinstance(packaged, os.PathLike):
        return _source_examples_root()
    path = Path(os.fspath(packaged))
    if (path / "all_options.toml").is_file():
        return path.resolve()
    return _source_examples_root()


def _profiling_campaign_inventory(source: Path) -> tuple[Path, ...]:
    selected = set(_PROFILING_CAMPAIGN_REQUIRED_FILES)
    for pattern in _PROFILING_CAMPAIGN_GLOBS:
        selected.update(path.relative_to(source) for path in source.glob(pattern))
    missing = tuple(
        relative
        for relative in _PROFILING_CAMPAIGN_REQUIRED_FILES
        if not (source / relative).is_file()
    )
    if missing or len(selected) != _PROFILING_CAMPAIGN_FILE_COUNT:
        detail = ", ".join(path.as_posix() for path in missing)
        raise ConfigurationError(
            "packaged profiling campaign has an invalid inventory"
            + (f": missing {detail}" if detail else "")
        )
    for relative in selected:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ConfigurationError(
                f"packaged profiling campaign has an unsafe member: {relative}"
            )
    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def _source_profiling_campaign_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs" / "performance_reports" / "macbook_M3_manual"
        if (candidate / "steer_performance_campaign.py").is_file():
            _profiling_campaign_inventory(candidate)
            return candidate
    raise ConfigurationError("packaged profiling campaign is unavailable")


def profiling_campaign_root() -> Path:
    packaged = resources.files("pyamplicol").joinpath("_profiling_campaign")
    if not isinstance(packaged, os.PathLike):
        return _source_profiling_campaign_root()
    path = Path(os.fspath(packaged))
    if (path / "steer_performance_campaign.py").is_file():
        _profiling_campaign_inventory(path)
        return path.resolve()
    return _source_profiling_campaign_root()


def _copy_tree(source: Path, destination: Path, *, force: bool) -> Path:
    target = destination.expanduser().resolve(strict=False)
    if target.exists():
        if not target.is_dir():
            raise ConfigurationError(f"destination is not a directory: {target}")
        if any(target.iterdir()) and not force:
            raise ConfigurationError(
                f"destination is not empty: {target}; pass --force to merge"
            )
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _resolved_original_amplicol_checkout(path: Path) -> Path:
    try:
        repository = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ConfigurationError(
            f"--local-amplicol checkout is unavailable: {path}"
        ) from error
    if not repository.is_dir():
        raise ConfigurationError(f"--local-amplicol is not a directory: {repository}")
    return repository


def _copy_profiling_campaign(
    destination: Path,
    *,
    force: bool,
    local_amplicol: Path | None = None,
) -> Path:
    validated_amplicol = (
        None
        if local_amplicol is None
        else _resolved_original_amplicol_checkout(local_amplicol)
    )
    source = profiling_campaign_root()
    target = destination.expanduser().resolve(strict=False)
    if target.exists():
        if not target.is_dir():
            raise ConfigurationError(f"destination is not a directory: {target}")
        if any(target.iterdir()) and not force:
            raise ConfigurationError(
                f"destination is not empty: {target}; pass --force to merge"
            )
    target.mkdir(parents=True, exist_ok=True)
    for relative in _profiling_campaign_inventory(source):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, output)
    launcher = target / "steer_performance_campaign.py"
    launcher_lines = launcher.read_bytes().splitlines(keepends=True)
    if not launcher_lines or not launcher_lines[0].startswith(b"#!"):
        raise ConfigurationError("profiling campaign launcher has no shebang")
    launcher.write_bytes(
        f"#!{sys.executable}\n".encode() + b"".join(launcher_lines[1:])
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    local_amplicol_path = target / _PROFILING_CAMPAIGN_LOCAL_AMPLICOL
    if validated_amplicol is None:
        local_amplicol_path.unlink(missing_ok=True)
    else:
        local_amplicol_path.write_text(
            f"{validated_amplicol}\n",
            encoding="utf-8",
        )
    return target


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__":
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())


def _copy_packaged_models(destination: Path) -> None:
    source = resources.files("pyamplicol.assets").joinpath("models")
    if not source.is_dir():
        raise ConfigurationError("packaged model assets are unavailable")
    _copy_resource_tree(source, destination)


def _card_description(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        description = stripped.lstrip("# ") if stripped.startswith("#") else ""
        if description and not description.startswith("SPDX-License-Identifier:"):
            return description
        if stripped and not stripped.startswith("#"):
            break
    return "Packaged run-card example"


def list_examples() -> tuple[ExampleEntry, ...]:
    entries: list[ExampleEntry] = []
    for card in sorted(examples_root().glob("*.toml")):
        if card.name == "all_options.toml":
            action = "reference"
        else:
            action = str(resolve_config(card).effective.action)
        entries.append(ExampleEntry(card.stem, action, _card_description(card)))
    return tuple(entries)


def example_card(name: str) -> Path:
    if not name or Path(name).name != name:
        raise ConfigurationError("example name must not contain a path")
    source = examples_root()
    card = source / f"{name.removesuffix('.toml')}.toml"
    if not card.is_file() or card.name == "all_options.toml":
        available = ", ".join(entry.name for entry in list_examples())
        raise ConfigurationError(f"unknown example {name!r}; available: {available}")
    override = os.environ.get("PYAMPLICOL_EXAMPLE_CACHE")
    if override:
        workspace = Path(override).expanduser().resolve(strict=False)
    else:
        from platformdirs import user_cache_path

        from pyamplicol import __version__

        workspace = user_cache_path("pyamplicol") / "examples" / __version__
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace, dirs_exist_ok=True)
    _copy_packaged_models(workspace / "models")
    return workspace / card.name


def execute_utility(invocation: UtilityInvocation) -> object:
    if invocation.kind == "config-template":
        destination = (
            Path("pyamplicol.toml") if invocation.path is None else invocation.path
        )
        source = examples_root() / "all_options.toml"
        target = destination.expanduser().resolve(strict=False)
        if target.exists() and not invocation.force:
            raise ConfigurationError(f"configuration file exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target)
    if invocation.kind == "config-resolve":
        assert invocation.path is not None
        return resolution_to_dict(
            resolve_config(invocation.path, overrides=invocation.overrides)
        )
    if invocation.kind == "examples-list":
        return list_examples()
    if invocation.kind == "examples-copy":
        assert invocation.path is not None
        destination = _copy_tree(
            examples_root(), invocation.path, force=invocation.force
        )
        _copy_packaged_models(destination / "models")
        return str(destination)
    if invocation.kind == "profiling-campaign-copy":
        assert invocation.path is not None
        return str(
            _copy_profiling_campaign(
                invocation.path,
                force=invocation.force,
                local_amplicol=invocation.local_amplicol,
            )
        )
    if invocation.kind == "doctor":
        return run_doctor()
    if invocation.kind == "self-test":
        return run_self_test()
    raise ConfigurationError(f"utility {invocation.kind!r} must be dispatched by CLI")


__all__ = [
    "ExampleEntry",
    "UtilityInvocation",
    "example_card",
    "examples_root",
    "execute_utility",
    "list_examples",
    "parse_utility",
    "profiling_campaign_root",
]
