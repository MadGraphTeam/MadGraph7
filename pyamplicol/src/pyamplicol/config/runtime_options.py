# SPDX-License-Identifier: 0BSD
"""Runtime options of a loaded loop runtime from a runtime-options TOML
(``Runtime.load(..., options=)``; docs/user/configuration.md, "Runtime
options").

A runtime-options file is a TOML with an ``[evaluation]`` table; a full run
card is accepted too: only ``[evaluation]`` is read (with the run card's
rules, ``config.resolver``), its runtime options (``RUNTIME_OPTIONS``) are
applied, its other keys (the inputs of a command-line run) are not read,
and an unknown key is refused by name. The native SDKs read the same TOML
(``rusticol_virtual_set_options``, ``rusticol_loop_reduction::
runtime_options``): the same keys, values and refusal texts, held equal by
``tests/unit/test_runtime_options.py``.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

# The runtime options of ``[evaluation]`` (the native reader's ``KEYS``; the
# stability checks are the ``[evaluation.stability_checks]`` table, given or
# not). Python reads the same keys from a run card for ``evaluate``.
RUNTIME_OPTIONS: tuple[str, ...] = (
    "arithmetic",
    "precision",
    "poles",
    "on_failure",
    "reduction",
    "reduction_target",
    "reduction_unchecked",
    "lane_rescue",
    "kinematics_improvement",
    "kinematics_improvement_binary64",
    "stability_checks",
    "threads",
)

# The options a loaded runtime holds for its calls (``Runtime.evaluate`` and
# its variants take them per call): the others are load-time settings.
CALL_OPTIONS: tuple[str, ...] = ("poles", "on_failure", "stability_checks")

RuntimeOptionsSource = os.PathLike[str] | str | Mapping[str, object]


def read_runtime_options(source: RuntimeOptionsSource) -> dict[str, Any]:
    """The runtime options ``source`` gives (a path to a TOML file, a TOML
    text, or a parsed TOML document as a mapping), validated by the run
    card's rules, by name: ``stability_checks`` is a
    ``VirtualStabilityChecks`` (None: the table requests no check), the
    others their run-card values. A key the TOML does not give is absent."""

    document = _document(source)
    table = document.get("evaluation", {})
    if not isinstance(table, Mapping):
        raise ConfigurationError("configuration section 'evaluation' must be a table")
    from .resolver import resolve_config

    evaluation = resolve_config(
        {"action": "evaluate", "evaluation": dict(table)}
    ).requested.evaluation
    found: dict[str, Any] = {}
    for name in RUNTIME_OPTIONS:
        if name not in table:
            continue
        if name == "stability_checks":
            found[name] = evaluation.stability_checks.to_checks()
        else:
            value = getattr(evaluation, name)
            found[name] = getattr(value, "value", value)
    return found


def _document(source: RuntimeOptionsSource) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    # A path: a PathLike, or one line without "=" that is not a table header
    # (a TOML text with a setting has one; a mistyped path then says "no such
    # file" instead of failing as TOML).
    if isinstance(source, os.PathLike) or (
        isinstance(source, str)
        and source.strip()
        and "\n" not in source
        and "=" not in source
        and not source.lstrip().startswith("[")
    ):
        path = Path(os.fspath(source)).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigurationError(
                f"cannot read runtime options {path}: {error}"
            ) from error
    elif isinstance(source, str):
        text = source
    else:
        raise TypeError("options takes a TOML path, a TOML text or a mapping")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"invalid TOML: {error}") from error


__all__ = [
    "CALL_OPTIONS",
    "RUNTIME_OPTIONS",
    "RuntimeOptionsSource",
    "read_runtime_options",
]
