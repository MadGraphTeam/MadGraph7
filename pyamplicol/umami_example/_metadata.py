"""Small shared reader for the generated UMAMI example metadata."""

from __future__ import annotations

import json
from pathlib import Path


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
    return data
