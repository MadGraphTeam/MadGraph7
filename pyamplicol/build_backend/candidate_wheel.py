# SPDX-License-Identifier: 0BSD
"""Explicit Community wheel input for a locked-core development build."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename

ENVIRONMENT = "PYAMPLICOL_CANDIDATE_SYMBOLICA_WHEEL"


def existing_community_wheel(
    root: Path, value: str | Path | None = None
) -> Path | None:
    """Validate the standard wheel name against the Python distribution pin.

    Pip owns installation and wheel metadata checks. Cargo independently owns
    the kernel's version and source resolution; the wheel is not a native build
    input and is never hashed into the compiled core's identity.
    """
    value = os.environ.get(ENVIRONMENT) if value is None else value
    if value is None:
        return None
    wheel = Path(value).expanduser().resolve()
    if not wheel.is_file():
        raise RuntimeError(f"existing Community wheel is missing: {wheel}")
    name, version, _build, _tags = parse_wheel_filename(wheel.name)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(
        Requirement(item)
        for item in project["project"]["dependencies"]
        if Requirement(item).name == "symbolica"
    )
    if name != "symbolica" or version not in requirement.specifier:
        raise RuntimeError(f"existing Community wheel must satisfy {requirement}")
    # This opt-in lane consumes only the checked-in Cargo inputs. Refuse an
    # ambiguous mixture with the historical generated path-patch lane.
    if any(
        (root / "dependencies" / filename).exists()
        for filename in (
            "candidate-Cargo.lock",
            "candidate-cargo-config.toml",
            "install-state.json",
        )
    ):
        raise RuntimeError("existing-wheel candidate cannot mix managed Cargo inputs")
    return wheel
