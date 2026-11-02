# SPDX-License-Identifier: 0BSD
"""UFO models shipped with the installed package, selected by name.

``model.source = "loop_sm"`` (or ``ModelSource.from_path("loop_sm")``) selects
the package's ``assets/models/ufo/loop_sm`` when no file or directory of that
name exists: an existing path always wins. The shipped directories and their
provenance are listed in ``assets/models/PROVENANCE.toml``.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def packaged_ufo_models() -> tuple[str, ...]:
    """The names of the UFO models shipped with the installed package."""

    root = Path(str(resources.files("pyamplicol").joinpath("assets", "models", "ufo")))
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        )
    )


def packaged_ufo_model(name: str) -> Path | None:
    """The installed directory of the shipped UFO model ``name``, or None
    when ``name`` is not a bare model name or no such model is shipped."""

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return None
    if name not in packaged_ufo_models():
        return None
    return Path(
        str(resources.files("pyamplicol").joinpath("assets", "models", "ufo", name))
    ).resolve()


def resolve_model_path(value: str, base_dir: Path | None = None) -> Path:
    """``value`` as a model path (relative to ``base_dir``, else the working
    directory); a bare name of a shipped UFO model selects it when no such
    path exists."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    candidate = candidate.resolve(strict=False)
    if candidate.exists():
        return candidate
    packaged = packaged_ufo_model(value)
    return candidate if packaged is None else packaged


__all__ = ["packaged_ufo_model", "packaged_ufo_models", "resolve_model_path"]
