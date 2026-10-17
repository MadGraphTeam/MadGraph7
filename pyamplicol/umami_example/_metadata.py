"""Small shared reader for the generated UMAMI example metadata."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


def require_current_exporter() -> Path:
    """Reject the old selectable exporter before starting expensive generation."""
    from pyamplicol.artifacts import umami

    path = Path(umami.__file__).resolve()
    if "grouping" in inspect.signature(umami.build_umami_metadata).parameters:
        raise RuntimeError(
            f"Outdated pyAmpliCol UMAMI exporter loaded by {sys.executable}: {path}. "
            "Reinstall pyAmpliCol from the current umami branch in this Python "
            "environment; git pull alone only updates the example scripts. "
            "Then regenerate into a fresh --output without --reuse. "
            "See umami_example/README.md for the source-checkout workflow."
        )
    return path


def grouping_summary(data: dict) -> str:
    """Describe the loaded tables, without claiming that a policy was applied."""
    entries = [entry for channel in data["channels"] for entry in channel["processes"]]
    physical_count = sum(len(entry["members"]) for entry in entries)
    return (
        f"{len(data['channels'])} integration channels, "
        f"{len(entries)} representative contributions, "
        f"{physical_count} physical contributions"
    )


def load_metadata(path: str | Path, provider: str | None = None) -> dict:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("kind") == "pyamplicol-umami-provider-index":
        choices = data["providers"]
        selected = [item for item in choices if item["id"] == provider]
        if len(selected) != 1:
            names = ", ".join(item["id"] for item in choices)
            raise ValueError(f"select one provider with --provider: {names}")
        return load_metadata(path.parent / selected[0]["metadata"], provider)
    if provider is not None and data["provider"]["id"] != provider:
        raise ValueError(f"metadata does not describe provider {provider!r}")
    if data["provider"]["incoming_count"] != 2:
        raise ValueError("this scattering example requires two incoming particles")
    grouping = data.get("grouping")
    if (
        not isinstance(grouping, dict)
        or "mode" in grouping
        or not {"physical_contributions", "exported_contributions"} <= grouping.keys()
    ):
        raise ValueError(
            f"Outdated or missing automatic-grouping metadata in {path}. "
            "Reinstall pyAmpliCol from the current umami branch, then regenerate "
            "into a fresh --output without --reuse. Rebuilding the .so alone "
            "does not update metadata; do not edit channel counts by hand."
        )
    return data
