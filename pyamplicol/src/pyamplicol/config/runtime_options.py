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
    document = source if isinstance(source, Mapping) else _parse(source)
    _check_integers(document, "")
    return document


def _parse(source: os.PathLike[str] | str) -> Mapping[str, object]:
    """The TOML of ``source``: a PathLike is a path; a one-line string is a
    path when it names an existing file, else a TOML text (``[evaluation]``
    is an empty table: no option); a one-line string that is neither an
    existing file nor valid TOML (a mistyped path) is refused naming both
    readings."""

    if isinstance(source, os.PathLike):
        return _read(Path(os.fspath(source)))
    if not isinstance(source, str):
        raise TypeError("options takes a TOML path, a TOML text or a mapping")
    one_line = bool(source.strip()) and "\n" not in source
    if one_line and Path(source).expanduser().is_file():
        return _read(Path(source))
    try:
        return tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        if one_line:
            raise ConfigurationError(
                f"runtime options {source!r} are not an existing file and not "
                f"valid TOML: {error}"
            ) from error
        raise ConfigurationError(f"invalid TOML: {error}") from error


def _read(path: Path) -> Mapping[str, object]:
    path = path.expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(
            f"cannot read runtime options {path}: {error}"
        ) from error
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"invalid TOML: {error}") from error


# TOML integers are signed 64-bit (TOML 1.0: a value that cannot be represented
# losslessly is an error); Python's tomllib reads larger ones, the native
# reader (the toml crate) refuses them: refused here too, so that both read the
# same documents.
_INT64 = (-(2**63), 2**63 - 1)


def _check_integers(value: object, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if not _INT64[0] <= value <= _INT64[1]:
            raise ConfigurationError(
                f"invalid TOML: the integer {value} at {path or 'the top level'} is "
                "outside the signed 64-bit range of TOML integers"
            )
    elif isinstance(value, Mapping):
        for key, entry in value.items():
            _check_integers(entry, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, entry in enumerate(value):
            _check_integers(entry, f"{path}[{index}]")


__all__ = [
    "CALL_OPTIONS",
    "RUNTIME_OPTIONS",
    "RuntimeOptionsSource",
    "read_runtime_options",
]
