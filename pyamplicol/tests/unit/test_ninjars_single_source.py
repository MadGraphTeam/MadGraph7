# SPDX-License-Identifier: 0BSD
"""One NinjaRS in the build: every NinjaRS crate has exactly one source.

``rusticol_core::symbol_order`` reserves NinjaRS's positional slots and
master-integral symbols through ``ninjars_symbolica::reserve``; the pool is a
process-wide static of the ``ninjars-positional`` crate. A lock file with two
sources of a NinjaRS crate (one ``Cargo.toml`` pin moved, another not) links
two copies, hence two pools: the reservation would build one while the
reductions use the other, and NinjaRS's extended-carrier results would depend
on the process history again, silently. Every NinjaRS pin of the workspace,
the lock file and ``dependencies/manifest.json`` must name one revision.

NinjaRS's reserved names (``ninjars-symbolica``'s ``src/master_symbols.txt``)
are those of the oneloopmaster revision NinjaRS pins and tests: the lock must
hold ONE ``oneloop`` package, from alphal00p/oneloopmaster at pyAmpliCol's own
pin. A different NinjaRS pin of oneloop would add a second source; a
``[patch]`` could move either crate silently, so no manifest may patch a
NinjaRS crate or oneloop, and every ``ninjars*`` package comes from the
alphal00p git source (not a path, a fork or a registry).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NINJARS = "https://github.com/alphal00p/ninjars.git"
ONELOOP = "https://github.com/alphal00p/oneloopmaster.git"
GUARDED = ("ninjars", "oneloop")


def _manifests() -> list[Path]:
    found = (ROOT / "rust").rglob("Cargo.toml")
    return [
        *sorted(path for path in found if "target" not in path.parts),
        ROOT / "Cargo.toml",
    ]


def _pins() -> dict[str, set[str]]:
    """Per manifest, the revisions of its NinjaRS git dependencies."""

    found: dict[str, set[str]] = {}
    for path in _manifests():
        data = tomllib.loads(path.read_text())
        tables = [data.get("dependencies", {}), data.get("dev-dependencies", {})]
        tables.append(data.get("workspace", {}).get("dependencies", {}))
        for target in data.get("target", {}).values():
            tables.append(target.get("dependencies", {}))
        for table in tables:
            for name, spec in table.items():
                if isinstance(spec, dict) and spec.get("git") == NINJARS:
                    found.setdefault(str(path.relative_to(ROOT)), set()).add(
                        spec["rev"]
                    )
                    assert name.startswith("ninjars"), (path, name)
    return found


def test_every_ninjars_crate_has_one_source() -> None:
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    sources: dict[str, set[str]] = {}
    for package in lock["package"]:
        source = package.get("source", "")
        if package["name"].startswith("ninjars"):
            # Not a path, a fork or a registry: alphal00p's git source only.
            assert source.startswith(f"git+{NINJARS}?rev="), package
        if source.startswith(f"git+{NINJARS}"):
            sources.setdefault(package["name"], set()).add(source)
    assert {"ninjars", "ninjars-positional", "ninjars-symbolica"} <= set(sources)
    assert all(len(found) == 1 for found in sources.values()), sources
    locked = {re.sub(r".*#", "", next(iter(found))) for found in sources.values()}
    assert len(locked) == 1, sources
    (revision,) = locked
    pins = _pins()
    assert pins, "no NinjaRS pin in the workspace manifests"
    assert all(found == {revision} for found in pins.values()), (revision, pins)
    (entry,) = (
        entry
        for entry in json.loads((ROOT / "dependencies" / "manifest.json").read_text())[
            "dependencies"
        ]
        if entry["name"] == "ninjars"
    )
    assert entry["revision"] == revision, (entry["revision"], revision)


def test_oneloop_has_one_source_at_the_pinned_revision() -> None:
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    (oneloop,) = (p for p in lock["package"] if p["name"] == "oneloop")
    assert oneloop["source"].startswith(f"git+{ONELOOP}?rev="), oneloop
    revision = re.sub(r".*#", "", oneloop["source"])
    pins = set()
    for path in _manifests():
        data = tomllib.loads(path.read_text())
        for spec in data.get("dependencies", {}).values():
            if isinstance(spec, dict) and spec.get("git") == ONELOOP:
                pins.add(spec["rev"])
    assert pins == {revision}, (pins, revision)


def test_no_manifest_patches_ninjars_or_oneloop() -> None:
    for path in _manifests():
        data = tomllib.loads(path.read_text())
        for source, table in data.get("patch", {}).items():
            for name, spec in table.items():
                url = spec.get("git", "") if isinstance(spec, dict) else ""
                assert not name.startswith(GUARDED), (path, source, name)
                assert "alphal00p/ninjars" not in url, (path, source, name)
                assert "oneloopmaster" not in url, (path, source, name)
