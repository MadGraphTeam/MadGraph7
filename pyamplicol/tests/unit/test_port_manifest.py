# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_port_manifest_has_unique_git_objects() -> None:
    with (ROOT / "docs" / "development" / "PORT_MANIFEST.toml").open("rb") as stream:
        payload = tomllib.load(stream)

    entries = payload["reference_inputs"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert all(re.fullmatch(r"[a-f0-9]{40}", entry["blob"]) for entry in entries)
    assert all(entry["status"] == "ported" for entry in entries)
    assert all(
        entry["destination"]
        and all(
            path and not path.startswith("/") and (ROOT / path).exists()
            for path in entry["destination"]
        )
        for entry in entries
    )
    assert {entry["component"] for entry in entries} >= {
        "models",
        "generation",
        "rust-runtime",
        "rust-c-api",
        "rust-sdk",
    }
