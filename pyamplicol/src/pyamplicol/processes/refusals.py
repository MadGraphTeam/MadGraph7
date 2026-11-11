# SPDX-License-Identifier: 0BSD
"""Listed refusals of process requests (tree and one-loop alike).

A process request the product does not serve is refused, before any model
work, with a typed error (``ProcessRequestUnsupported``) whose ``key`` names
a listed rule (``PROCESS_REQUEST_REFUSALS``; a loop dry run lists them as
``process_request_refusals``). The rules read the request's syntax, never a
particle or process name:

* ``madloop-syntax``: MG5_aMC process syntax that is not served (decay
  chains ``p p > t t~, t > b w+``, ``$``/``$$`` s-channel exclusions,
  ``@N`` process tags, ``[real=QCD]`` and ``[LOonly=QCD]``: only the
  one-loop virtual is computed, amplitude-level ``==``/``>`` limits). The
  served MG5_aMC syntax (NLO brackets, coupling-order constraints, ``/ x``
  exclusions) is translated into run-card settings by
  ``processes.mg5_syntax`` wherever a process expression enters (TOML
  ``process.entries``, the CLI, ``pyamplicol.generate`` / ``Generator``); a
  ``ProcessRequest`` is a plain expression and refuses it with that
  pointer.

A process has one initial particle (a 1 -> n decay, MG5_aMC's ``t > b w+``)
or two (a 2 -> n scattering; ``processes.model.require_external_leg_counts``).
Tree-level decays (M-FINAL MF-DEC-1) and one-loop decays (MF-DEC-2) are
served.
"""

from __future__ import annotations

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
        "madloop-syntax",
        "MG5_aMC process syntax that is not served: decay chains (a > b c, "
        "b > d e), s-channel exclusions ($ x, $$ x), process tags (@N), NLO "
        "brackets other than the virtual ([real=QCD], [LOonly=QCD]: only the "
        "one-loop virtual is computed), amplitude-level limits other than "
        "QED=2 / QED<=2; and, in a plain ProcessRequest, any MG5_aMC syntax",
        "request each decay ('t > b w+') as its own process; [QCD], "
        "[virt=QCD], [QCD QED], [noborn=QCD], [sqrvirt=QCD], QED=2, "
        "QED<=2, QCD^2<=4 (also ==, <, >, >=) and '/ x' are translated into "
        "the run-card settings (process.loop_coupling_orders, "
        "process.loop_induced, process.max_coupling_orders, "
        "process.squared_coupling_orders, generation.particle_veto) when the "
        "expression is given as a run-card process entry, with --process, or "
        "to pyamplicol.generate / Generator.plan / Generator.generate",
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


def refuse_madloop_syntax(expression: str) -> None:
    """Refuse (``madloop-syntax``) MG5_aMC syntax in a plain process
    expression (a ``ProcessRequest``): the unserved syntax with its reason
    (``mg5_syntax.translate_process_expression``), the served syntax with
    the entry points that translate it and what it translates to."""

    from .mg5_syntax import translate_process_expression

    translation = translate_process_expression(expression)
    if translation.translated:
        raise ProcessRequestUnsupported(
            "madloop-syntax",
            f"a ProcessRequest takes a plain expression; {translation.describe()}",
        )


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


__all__ = [
    "PROCESS_REQUEST_REFUSALS",
    "ProcessRequestRefusal",
    "ProcessRequestUnsupported",
    "listed_process_refusal_keys",
    "refuse_madloop_syntax",
    "refuse_order_brackets",
]
