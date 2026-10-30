# SPDX-License-Identifier: 0BSD
"""Listed refusals of process requests (tree and one-loop alike).

A process request the product does not serve is refused, before any model
work, with a typed error (``ProcessRequestUnsupported``) whose ``key`` names
a listed rule (``PROCESS_REQUEST_REFUSALS``; a loop dry run lists them as
``process_request_refusals``). The rules read the request's syntax and the
number of its initial-state particles, never a particle or process name:

* ``decay-process``: a 1 -> n request (one initial particle). pyAmpliCol
  evaluates 2 -> n scattering matrix elements; decays (with or without QCD
  or electroweak corrections) are not served (user ruling 2026-10-28 ~15:25
  (1): decays come after M-QCD; m-sm PLAN Q3 (a): after M-SM).
* ``madloop-syntax``: MG5_aMC process syntax that has a pyAmpliCol run-card
  setting instead (``[virt=QCD]``, ``[QCD]`` and the other NLO brackets,
  coupling-order constraints such as ``QED=2`` or ``QCD^2==4``, ``/ x``
  particle exclusions) or no counterpart (``$``/``$$`` s-channel exclusions,
  decay chains ``p p > t t~, t > b w+``, ``@N`` process tags). The message
  names the run-card setting.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pyamplicol.config.errors import PyAmpliColError


@dataclass(frozen=True)
class ProcessRequestRefusal:
    """A listed process request the product refuses: ``key`` names it,
    ``feature`` says what is refused and ``remedy`` what to request instead."""

    key: str
    feature: str
    remedy: str


PROCESS_REQUEST_REFUSALS = (
    ProcessRequestRefusal(
        "decay-process",
        "a 1 -> n decay request (one initial-state particle)",
        "a 2 -> n scattering process; decays, with or without QCD or "
        "electroweak corrections, are not served yet",
    ),
    ProcessRequestRefusal(
        "madloop-syntax",
        "MG5_aMC process syntax inside the process expression: NLO brackets "
        "([virt=QCD], [QCD], [real=QCD], [noborn=QCD], ...), coupling-order "
        "constraints (QED=2, QCD^2==4, QED<=2), particle exclusions (/ x), "
        "s-channel exclusions ($ x, $$ x), decay chains (a > b c, b > d e) and "
        "process tags (@N)",
        "plain 'a b > c d ...' with the run-card settings: "
        'process.loop_coupling_orders = ["QCD"] for [virt=QCD] or [QCD]; '
        "process.max_coupling_orders = {QED = 2} for QED=2 (squared-order "
        "sectors are read out of the result, QCD^2==); "
        'generation.particle_veto = ["x"] for / x; s-channel exclusions, '
        "decay chains and loop-induced [noborn=...] requests are not served",
    ),
)
_REFUSALS = {refusal.key: refusal for refusal in PROCESS_REQUEST_REFUSALS}


class ProcessRequestUnsupported(PyAmpliColError, ValueError):
    """A listed process request the product refuses
    (``PROCESS_REQUEST_REFUSALS``); ``key`` is its key. The message starts
    with the key and names the remedy. A ``ValueError`` as the untyped
    refusals it replaces."""

    def __init__(self, key: str, message: str) -> None:
        refusal = _REFUSALS.get(key)
        if refusal is None:
            raise ValueError(f"unknown process request refusal {key!r}")
        super().__init__(f"{key}: {message}. Instead: {refusal.remedy}")
        self.key = key


def listed_process_refusal_keys() -> tuple[str, ...]:
    return tuple(_REFUSALS)


# MG5_aMC process-line syntax with no place in a pyAmpliCol expression. A
# bracket holding '=' is an NLO bracket ([virt=QCD]); a bare '[QCD]' is
# recognised by the caller that knows the model's coupling orders
# (``refuse_order_brackets``), since '[u d]' is an anonymous multiparticle.
_ORDER_CONSTRAINT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\^2)?(==|<=|>=|=|<|>)-?\d+$")


def refuse_madloop_syntax(expression: str) -> None:
    """Refuse (``madloop-syntax``) an expression that uses MG5_aMC process
    syntax (``ProcessRequestRefusal`` 'madloop-syntax'): an NLO bracket
    holding '=', a coupling-order constraint, '/', '$', '@', or a decay
    chain (a second ',' separated clause)."""

    text = " ".join(str(expression).split())
    found: list[str] = []
    if re.search(r"\[[^\]]*=[^\]]*\]", text):
        found.append("an NLO bracket ('[virt=QCD]' style)")
    if "$" in text:
        found.append("an s-channel exclusion ('$')")
    if "/" in text:
        found.append("a particle exclusion ('/')")
    if "@" in text:
        found.append("a process tag ('@')")
    if _decay_chain(text):
        found.append("a decay chain (',')")
    if any(
        _ORDER_CONSTRAINT.match(token)
        for token in re.split(r"\s+", re.sub(r"\[[^\]]*\]", " ", text))
        if token
    ):
        found.append("a coupling-order constraint ('QED=2' style)")
    if found:
        raise ProcessRequestUnsupported(
            "madloop-syntax",
            f"the process {text!r} uses MG5_aMC syntax: " + ", ".join(found),
        )


def _decay_chain(text: str) -> bool:
    """Whether ``text`` has a ',' outside brackets (MG5_aMC decay chains)."""

    depth = 0
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            return True
    return False


def refuse_order_brackets(
    expression: str, bracket_items: Iterable[str], order_names: Iterable[str]
) -> None:
    """Refuse (``madloop-syntax``) an anonymous bracket of ``expression``
    whose items are coupling-order names of the model (``order_names``, any
    case) rather than particles: MG5_aMC's '[QCD]'."""

    orders = {name.casefold() for name in order_names}
    items = [item for item in bracket_items if item.casefold() in orders]
    if items:
        raise ProcessRequestUnsupported(
            "madloop-syntax",
            f"the process {' '.join(str(expression).split())!r} brackets the "
            f"coupling order(s) {', '.join(items)} ('[QCD]' style NLO syntax)",
        )


def refuse_decay_syntax(expression: str) -> None:
    """Refuse (``decay-process``) an expression one of whose processes has a
    single initial-state slot (repetitions such as ``2*g`` count their
    slots; a multiparticle label or anonymous bracket is one slot)."""

    from .core_syntax import _split_repeat_token, _tokenize_side, split_process_set

    for process in split_process_set(" ".join(str(expression).split())):
        initial, separator, _final = process.partition(">")
        if not separator:
            continue
        slots = sum(
            _split_repeat_token(token)[0] for token in _tokenize_side(initial.strip())
        )
        refuse_decay(process, slots)


def refuse_decay(expression: str, initial_count: int) -> None:
    """Refuse (``decay-process``) a request with one initial particle."""

    if initial_count == 1:
        raise ProcessRequestUnsupported(
            "decay-process",
            f"{' '.join(str(expression).split())!r} has one initial-state "
            "particle (a 1 -> n decay)",
        )


__all__ = [
    "PROCESS_REQUEST_REFUSALS",
    "ProcessRequestRefusal",
    "ProcessRequestUnsupported",
    "listed_process_refusal_keys",
    "refuse_decay",
    "refuse_decay_syntax",
    "refuse_madloop_syntax",
    "refuse_order_brackets",
]
