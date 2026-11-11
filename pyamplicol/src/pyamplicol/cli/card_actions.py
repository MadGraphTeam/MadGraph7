# SPDX-License-Identifier: 0BSD
"""``pyamplicol CARD ACTION [options]``: one run card, the action on the
command line.

The card carries the settings of every action (``examples/loop_generate.toml``);
the command line picks the action (the card's own ``action`` is what
``pyamplicol CARD`` alone runs). ``--output DIR`` names the artifact: the
directory ``generate`` writes (``generation.output``) and the one ``check``,
``evaluate``, ``profile`` and ``inspect`` read (``evaluation.artifact``),
relative to the working directory. Every other option is the action's own
(``pyamplicol ACTION --help``), e.g. ``--process`` (MG5_aMC syntax accepted)
and ``--set PATH=VALUE``.

The rewrite is ``ACTION --card CARD [--set <path>=<DIR>] [options]``; the
older forms (``pyamplicol CARD``, ``pyamplicol check CARD``,
``pyamplicol ACTION --card CARD``) are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pyamplicol.config import ConfigurationError

#: The actions selectable after a card, and the setting ``--output`` names.
CARD_ACTIONS = {
    "generate": "generation.output",
    "check": "evaluation.artifact",
    "evaluate": "evaluation.artifact",
    "profile": "evaluation.artifact",
    "benchmark": "evaluation.artifact",
    "inspect": "evaluation.artifact",
}


def _is_card(argument: str) -> bool:
    if argument.startswith("-"):
        return False
    path = Path(argument).expanduser()
    return path.suffix.lower() == ".toml" or path.is_file()


def rewrite_card_action(arguments: Sequence[str]) -> list[str] | None:
    """The ``ACTION --card CARD ...`` form of ``CARD ACTION ...`` (None for
    any other command line)."""

    if len(arguments) < 2 or not _is_card(arguments[0]):
        return None
    action = arguments[1]
    if action not in CARD_ACTIONS:
        return None
    card = Path(arguments[0]).expanduser()
    if not card.is_file():
        raise ConfigurationError(f"run card {arguments[0]!r} does not exist")
    rest: list[str] = []
    output: str | None = None
    items = list(arguments[2:])
    index = 0
    while index < len(items):
        item = items[index]
        if item == "--output":
            if index + 1 >= len(items):
                raise ConfigurationError("--output needs a directory")
            output = items[index + 1]
            index += 2
            continue
        if item.startswith("--output="):
            output = item.removeprefix("--output=")
            index += 1
            continue
        rest.append(item)
        index += 1
    rewritten = [action, "--card", str(card.resolve(strict=False))]
    if output is not None:
        if not output:
            raise ConfigurationError("--output needs a directory")
        directory = Path(output).expanduser()
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        rewritten += ["--set", f"{CARD_ACTIONS[action]}={directory}"]
    return rewritten + rest


__all__ = ["CARD_ACTIONS", "rewrite_card_action"]
