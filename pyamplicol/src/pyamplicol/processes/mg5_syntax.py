# SPDX-License-Identifier: 0BSD
"""MG5_aMC process syntax, translated to pyAmpliCol run-card settings.

A process expression may be written as on an MG5_aMC ``generate`` line. The
MG5_aMC annotations are translated, in this module only, into the run-card
settings that already express them; the expression keeps its particles:

========================================  ====================================
MG5_aMC                                   pyAmpliCol setting
========================================  ====================================
``[QCD]``, ``[virt=QCD]``, ``[all=QCD]``  ``process.loop_coupling_orders``
``[QCD QED]``, ``[virt=QCD QED]``         (the corrections, sorted)
``[noborn=QCD]``, ``[sqrvirt=QCD]``       the same, plus
                                          ``process.loop_induced = "only"``
``QED=2``, ``QED<=2`` (amplitude)         ``process.max_coupling_orders``
``QCD^2<=4``, ``QED^2==6``, ...           ``process.squared_coupling_orders``
``/ z h``                                 ``generation.particle_veto``
========================================  ====================================

Only the one-loop virtual is served: ``[QCD]`` (MG5_aMC's full NLO) means
the virtual here, ``[real=QCD]`` and ``[LOonly=QCD]`` are refused. An
amplitude- or squared-order constraint switches off the minimal-order
policy (``process.coupling_order_policy = "explicit"``), as MG5_aMC drops
its WEIGHTED minimisation when the user names coupling orders.

The NLO bracket is the expression's last element; a bracket there is read
as MG5_aMC's when it holds ``=`` or when every item is spelt as a coupling
order (upper-case letters, digits and ``_``, MG5_aMC's convention); any
other bracket is pyAmpliCol's anonymous multiparticle (``[d g]``). Decay
chains (``,``), s-channel exclusions (``$``, ``$$``) and process tags
(``@N``) stay refused (``madloop-syntax``).

A translated setting that the run card, a dedicated option or an override
also sets to another value is refused (``ProcessSyntaxConflict``, naming
both); several process entries must translate to the same settings (they
are run-wide). ``apply_to_values`` (the resolver's flat values) and
``apply_to_run_config`` (a typed ``RunConfig``) are the two adapters of one
merge (``merge_translations``), used by the TOML card, the CLI and the
Python API alike.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pyamplicol.config.errors import ConfigurationError

from .refusals import ProcessRequestUnsupported

#: The NLO bracket keywords served (the virtual), by what they route to.
VIRTUAL_KEYWORDS = ("all", "virt")
LOOP_INDUCED_KEYWORDS = ("noborn", "sqrvirt")
#: MG5_aMC keywords whose contribution pyAmpliCol does not compute.
UNSERVED_KEYWORDS = ("real", "loonly")

#: Comparison operators of a squared-order constraint (MG5_aMC: ``==``,
#: ``<=``, ``>``; ``<``, ``>=`` and ``=`` (as ``<=``) are accepted too).
SQUARED_OPERATORS = ("<=", "==", ">=", "<", ">")

_ORDER_NAME = r"[A-Za-z][A-Za-z0-9_]*"
_CONSTRAINT = re.compile(
    rf"^(?P<name>{_ORDER_NAME})(?P<squared>\^2)?(?P<op>==|<=|>=|=|<|>)(?P<value>-?\d+)$"
)
_ORDER_SPELLING = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TRAILING_BRACKET = re.compile(r"\[(?P<body>[^\[\]]*)\]\s*$")

PROCESS_PATHS = (
    "process.loop_coupling_orders",
    "process.loop_induced",
    "process.coupling_order_policy",
    "process.max_coupling_orders",
    "process.squared_coupling_orders",
    "generation.particle_veto",
)


@dataclass(frozen=True, slots=True)
class SquaredOrderConstraint:
    """One squared-coupling-order constraint, ``name^2 operator value``: a
    squared-order sector whose power of ``name`` (in the fundamental
    coupling, as MG5_aMC counts it) fails it is not served."""

    name: str
    operator: str
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(_ORDER_NAME, self.name):
            raise ValueError(f"invalid coupling-order name {self.name!r}")
        if self.operator not in SQUARED_OPERATORS:
            raise ValueError(
                f"squared-order operator must be one of {list(SQUARED_OPERATORS)}"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError("a squared-order limit is an integer")
        if self.value < 0:
            raise ValueError("a squared-order limit is non-negative")

    @property
    def text(self) -> str:
        return f"{self.name}^2{self.operator}{self.value}"

    def admits(self, power: int) -> bool:
        value = self.value
        return {
            "<=": power <= value,
            "==": power == value,
            ">=": power >= value,
            "<": power < value,
            ">": power > value,
        }[self.operator]

    @property
    def upper_bound(self) -> int | None:
        """The largest power the constraint admits (None: unbounded)."""

        if self.operator in ("<=", "=="):
            return self.value
        if self.operator == "<":
            return self.value - 1
        return None


def parse_squared_order_constraint(text: str) -> SquaredOrderConstraint:
    """``"QCD^2<=4"`` (MG5_aMC's spelling; ``"QCD<=4"`` and ``=`` for ``<=``
    are accepted in the run-card setting, where every constraint is
    squared)."""

    if not isinstance(text, str):
        raise ValueError("a squared-order constraint is a string such as 'QCD^2<=4'")
    match = _CONSTRAINT.match("".join(text.split()))
    if match is None:
        raise ValueError(
            f"invalid squared-order constraint {text!r}; expected NAME^2<=N, "
            "NAME^2==N, NAME^2<N, NAME^2>N or NAME^2>=N (e.g. 'QCD^2<=4')"
        )
    operator = "<=" if match["op"] == "=" else match["op"]
    return SquaredOrderConstraint(match["name"], operator, int(match["value"]))


def canonical_squared_orders(values: Iterable[str]) -> tuple[str, ...]:
    """The canonical texts of ``values`` (sorted, unique): the stored form of
    ``process.squared_coupling_orders``."""

    return tuple(
        sorted({parse_squared_order_constraint(value).text for value in values})
    )


def squared_orders_admit(
    constraints: Iterable[SquaredOrderConstraint],
    order: Mapping[str, int] | Iterable[tuple[str, int]],
) -> bool:
    """Whether the squared-order sector ``order`` (powers by coupling name, a
    missing name a zero power) passes every constraint."""

    powers = dict(order.items() if isinstance(order, Mapping) else order)
    return all(
        constraint.admits(int(powers.get(constraint.name, 0)))
        for constraint in constraints
    )


@dataclass(frozen=True, slots=True)
class ProcessSyntaxTranslation:
    """One process expression and the settings its MG5_aMC annotations
    translate to (``settings``: configuration path -> value; empty for a
    plain pyAmpliCol expression)."""

    source: str
    expression: str
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    @property
    def translated(self) -> bool:
        """Whether the expression carries MG5_aMC annotations (every
        annotation translates to a setting)."""

        return bool(self.settings)

    def describe(self) -> str:
        """``'<source>' -> '<expression>' with path = value, ...``."""

        parts = ", ".join(
            f"{path} = {_render(value)}" for path, value in self.settings.items()
        )
        text = f"{self.source!r} (MG5_aMC syntax) -> {self.expression!r}"
        return f"{text} with {parts}" if parts else text


def _render(value: object) -> str:
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k} = {v}" for k, v in value.items()) + "}"
    if isinstance(value, tuple):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    return f'"{value}"' if isinstance(value, str) else str(value)


def _refuse(source: str, what: str) -> ProcessRequestUnsupported:
    return ProcessRequestUnsupported(
        "madloop-syntax", f"the process {source!r} uses MG5_aMC syntax {what}"
    )


def _outside_brackets(text: str, character: str) -> bool:
    depth = 0
    for item in text:
        if item == "[":
            depth += 1
        elif item == "]":
            depth -= 1
        elif item == character and depth == 0:
            return True
    return False


def _nlo_bracket(source: str, text: str) -> tuple[str, dict[str, object]]:
    """The expression without its trailing NLO bracket and the bracket's
    settings (none when the trailing bracket is an anonymous
    multiparticle)."""

    match = _TRAILING_BRACKET.search(text)
    if match is None:
        return text, {}
    body = match["body"].strip()
    if "=" in body:
        keyword, _, orders_text = body.partition("=")
        keyword = keyword.strip().casefold()
        orders = tuple(orders_text.split())
    else:
        items = tuple(body.split())
        if not items or not all(_ORDER_SPELLING.match(item) for item in items):
            return text, {}
        keyword, orders = "all", items
    if keyword in UNSERVED_KEYWORDS:
        raise _refuse(
            source,
            f"'[{body}]': only the one-loop virtual is served ([QCD] or "
            "[virt=QCD]: the virtual; [noborn=QCD]: a loop-induced |A_loop|^2)",
        )
    if keyword not in (*VIRTUAL_KEYWORDS, *LOOP_INDUCED_KEYWORDS):
        raise _refuse(
            source,
            f"'[{body}]': unknown NLO bracket keyword {keyword!r} (served: "
            + ", ".join(f"{k}=" for k in (*VIRTUAL_KEYWORDS, *LOOP_INDUCED_KEYWORDS))
            + ", or a bare [QCD])",
        )
    if not orders or not all(re.fullmatch(_ORDER_NAME, name) for name in orders):
        raise _refuse(source, f"'[{body}]': the bracket names no coupling orders")
    settings: dict[str, object] = {
        "process.loop_coupling_orders": tuple(sorted(set(orders)))
    }
    if keyword in LOOP_INDUCED_KEYWORDS:
        settings["process.loop_induced"] = "only"
    return text[: match.start()].rstrip(), settings


def translate_process_expression(expression: str) -> ProcessSyntaxTranslation:
    """Translate one process expression (``a b > c d ...``, a pyAmpliCol
    process set ``... | ...`` included) written in MG5_aMC syntax: the
    plain expression and its settings. Refuses (``madloop-syntax``) the
    MG5_aMC syntax that is not served and an expression whose parts of a
    process set translate differently."""

    if not isinstance(expression, str):
        raise TypeError("process expression must be a string")
    source = " ".join(expression.split())
    if "$" in source:
        raise _refuse(source, "'$' (an s-channel exclusion), which is not served")
    if "@" in source:
        raise _refuse(source, "'@' (a process tag), which is not served")
    if _outside_brackets(source, ","):
        raise _refuse(
            source,
            "',' (a decay chain), which is not served: request each decay "
            "('t > b w+') as its own process",
        )
    parts = _split_set(source)
    translated = [_translate_one(source, part) for part in parts]
    if not any(settings for _part, settings in translated):
        # A plain expression is kept as written.
        return ProcessSyntaxTranslation(source, source)
    settings = translated[0][1]
    for _part, other in translated[1:]:
        if other != settings:
            raise _refuse(
                source,
                "with different annotations in the parts of one process set "
                "('|'); the settings are run-wide",
            )
    plain = " | ".join(part for part, _settings in translated)
    return ProcessSyntaxTranslation(source, plain, settings)


def _split_set(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "|" and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _translate_one(source: str, text: str) -> tuple[str, dict[str, object]]:
    text, settings = _nlo_bracket(source, text)
    for body in re.findall(r"\[([^\[\]]*)\]", text):
        if "=" in body:
            # A bracket holding '=' elsewhere than at the end.
            raise _refuse(
                source,
                f"'[{body}]' before the end of the process (the NLO bracket "
                "comes last, as in MG5_aMC)",
            )
    tokens = _tokens(text)
    if ">" not in tokens:
        return text, settings
    arrow = tokens.index(">")
    particles: list[str] = []
    veto: list[str] = []
    amplitude: dict[str, int] = {}
    squared: list[SquaredOrderConstraint] = []
    in_veto = False
    for position, token in enumerate(tokens):
        constraint = _CONSTRAINT.match(token)
        if constraint is not None and not token.startswith("["):
            in_veto = False
            if position < arrow:
                raise _refuse(
                    source,
                    f"'{token}' before '>' (coupling-order constraints follow "
                    "the final state)",
                )
            name, value = constraint["name"], int(constraint["value"])
            operator = constraint["op"]
            if constraint["squared"]:
                parsed = parse_squared_order_constraint(token)
                squared.append(parsed)
                continue
            if operator not in ("=", "<="):
                raise _refuse(
                    source,
                    f"'{token}': an amplitude-level constraint is an upper "
                    "limit (QED=2 or QED<=2); select squared-order sectors "
                    f"with {name}^2{operator}N",
                )
            if value < 0:
                raise _refuse(source, f"'{token}': a negative coupling-order limit")
            if name in amplitude and amplitude[name] != value:
                raise _refuse(source, f"two amplitude-level limits on {name}")
            amplitude[name] = value
            continue
        if token == "/":
            if position < arrow:
                raise _refuse(source, "'/' before '>'")
            in_veto = True
            continue
        if in_veto:
            veto.append(token)
            continue
        particles.append(token)
    if any(token == "/" for token in tokens) and not veto:
        raise _refuse(source, "'/' without a particle to exclude")
    if amplitude:
        settings["process.max_coupling_orders"] = dict(sorted(amplitude.items()))
    if squared:
        settings["process.squared_coupling_orders"] = canonical_squared_orders(
            constraint.text for constraint in squared
        )
    if amplitude or squared:
        settings["process.coupling_order_policy"] = "explicit"
    if veto:
        settings["generation.particle_veto"] = tuple(dict.fromkeys(veto))
    return " ".join(particles), settings


def _tokens(text: str) -> list[str]:
    """Whitespace tokens with ``>`` and ``/`` split off, brackets kept whole
    (``u u~ > [d g]``) and constraints joined (``QCD^2 <= 4``). A bare
    ``>`` after a squared order (``QED^2 > 2``) is a comparison, any other
    one the arrow."""

    text = re.sub(rf"({_ORDER_NAME}\^2)\s*(==|<=|>=|=|<|>)\s*(-?\d+)", r"\1\2\3", text)
    text = re.sub(
        rf"(?<![^\s])({_ORDER_NAME})\s*(==|<=|>=|=|<)\s*(-?\d+)", r"\1\2\3", text
    )
    text = text.replace("/", " / ")
    raw: list[str] = []
    depth = 0
    current = ""
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character.isspace() and depth == 0:
            if current:
                raw.append(current)
            current = ""
        else:
            current += character
    if current:
        raw.append(current)
    tokens: list[str] = []
    for token in raw:
        if ">" in token and token != ">" and not _CONSTRAINT.match(token):
            before, _, after = token.partition(">")
            tokens.extend(item for item in (before, ">", after) if item)
        else:
            tokens.append(token)
    return tokens


# ------------------------------------------------------------------ merging


class ProcessSyntaxConflict(ConfigurationError):
    """A setting that a process expression's MG5_aMC syntax translates to,
    which the run card (or a CLI option, an override, the Python
    configuration) sets to another value; or two process entries whose
    annotations translate to different run-wide settings. The message names
    both values; nothing is chosen silently."""


def merge_translations(
    translations: Sequence[ProcessSyntaxTranslation],
    *,
    current: Callable[[str], object],
    explicit: Callable[[str], bool],
) -> dict[str, object]:
    """The setting updates of ``translations`` (path -> value) against the
    configuration: ``current(path)`` its value, ``explicit(path)`` whether
    the user set it. A translated value that differs from an explicit one is
    a ``ProcessSyntaxConflict``; ``process.max_coupling_orders`` merges per
    coupling (a coupling the card limits differently conflicts)."""

    carrying = [item for item in translations if item.settings]
    if not carrying:
        return {}
    first = carrying[0]
    for other in translations:
        if dict(other.settings) != dict(first.settings):
            raise ProcessSyntaxConflict(
                f"process entries {first.source!r} and {other.source!r} "
                "translate to different run-wide settings ("
                + _settings_text(first)
                + " against "
                + _settings_text(other)
                + "); give every entry the same MG5_aMC annotations, or use "
                "the run-card settings"
            )
    updates: dict[str, object] = {}
    errors: list[str] = []
    for path, value in first.settings.items():
        present = current(path)
        if path == "process.max_coupling_orders":
            merged = dict(present) if isinstance(present, Mapping) else {}
            for name, limit in dict(value).items():  # type: ignore[call-overload]
                if name in merged and merged[name] != limit:
                    errors.append(
                        f"the process {first.source!r} limits {name} to {limit} "
                        f"(process.max_coupling_orders.{name}) but the "
                        f"configuration sets {name} = {merged[name]}"
                    )
                merged[name] = limit
            updates[path] = merged
            continue
        if explicit(path) and _normal(present) != _normal(value):
            errors.append(
                f"the process {first.source!r} sets {path} = {_render(value)} "
                f"but the configuration sets {path} = {_render(_normal(present))}"
            )
            continue
        updates[path] = value
    if errors:
        raise ProcessSyntaxConflict(
            "; ".join(errors) + "; state it in one place: remove the setting "
            "from the run card (or the option) or the annotation from the "
            "process"
        )
    return updates


def _normal(value: object) -> object:
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (list, tuple)):
        return tuple(sorted(str(item) for item in value))
    return value


def _settings_text(translation: ProcessSyntaxTranslation) -> str:
    if not translation.settings:
        return "none"
    return ", ".join(
        f"{path} = {_render(value)}" for path, value in translation.settings.items()
    )


def translate_entries(
    expressions: Iterable[str],
) -> tuple[ProcessSyntaxTranslation, ...]:
    return tuple(translate_process_expression(item) for item in expressions)


def apply_to_values(
    values: dict[str, object], explicit_paths: Iterable[str]
) -> tuple[ProcessSyntaxTranslation, ...]:
    """Translate the process entries of the resolver's flat ``values`` in
    place (``config.resolver``): plain expressions and the merged settings.
    ``explicit_paths``: the paths the card, the dedicated options and the
    overrides set. Returns the translations."""

    from pyamplicol.config.models import ProcessEntry

    entries = values.get("process.entries") or ()
    translations = translate_entries(
        entry.expression
        for entry in entries  # type: ignore[union-attr]
    )
    if not any(item.translated for item in translations):
        return translations
    explicit = set(explicit_paths)
    updates = merge_translations(
        translations,
        current=lambda path: values.get(path),
        explicit=lambda path: path in explicit,
    )
    values["process.entries"] = tuple(
        ProcessEntry(expression=item.expression, name=entry.name)  # type: ignore[union-attr]
        for item, entry in zip(translations, entries, strict=True)  # type: ignore[arg-type]
    )
    values.update(updates)
    return translations


def apply_to_run_config(
    run: Any, translations: Sequence[ProcessSyntaxTranslation]
) -> Any:
    """``run`` (a ``RunConfig``) with the settings of ``translations``
    merged (a setting is explicit when it differs from its default): the
    Python API's adapter (``api.services``)."""

    from dataclasses import replace

    from pyamplicol.config.registry import default_values

    defaults = default_values()

    def current(path: str) -> object:
        section, name = path.split(".", 1)
        return getattr(getattr(run, section), name)

    def explicit(path: str) -> bool:
        return _normal(current(path)) != _normal(defaults.get(path))

    updates = merge_translations(translations, current=current, explicit=explicit)
    if not updates:
        return run
    by_section: dict[str, dict[str, object]] = {}
    for path, value in updates.items():
        section, name = path.split(".", 1)
        by_section.setdefault(section, {})[name] = value
    changed = {
        section: replace(getattr(run, section), **values)
        for section, values in by_section.items()
    }
    if (
        run.process.loop_coupling_orders is None
        and "process.loop_coupling_orders" in updates
    ):
        # A tree configuration made one-loop by the expression takes the
        # one-loop defaults of what it resolved for a tree request: the
        # execution mode ('auto') and the colour accuracy (full colour).
        from pyamplicol.config.models import (
            ColorAccuracy,
            EvaluatorExecutionMode,
        )

        if run.evaluator.execution_mode is not EvaluatorExecutionMode.LOOP_RECURRENCE:
            changed["evaluator"] = replace(
                run.evaluator, execution_mode=EvaluatorExecutionMode.AUTO
            )
        if run.color.accuracy is ColorAccuracy.LC:
            changed["color"] = replace(run.color, accuracy=None)
    return replace(run, **changed)


__all__ = [
    "PROCESS_PATHS",
    "ProcessSyntaxConflict",
    "ProcessSyntaxTranslation",
    "SquaredOrderConstraint",
    "apply_to_run_config",
    "apply_to_values",
    "canonical_squared_orders",
    "merge_translations",
    "parse_squared_order_constraint",
    "squared_orders_admit",
    "translate_api_request",
    "translate_entries",
    "translate_process_expression",
]


def translate_api_request(processes: Any, config: Any) -> tuple[Any, Any]:
    """The Python API's entry (``Generator.plan`` / ``generate``): the
    process strings of ``processes`` (a string, or an iterable mixing
    strings and ``ProcessRequest`` objects; a ``ProcessSet`` and a
    ``ProcessRequest`` are plain already) translated, and ``config`` with
    their settings merged (``RunConfig`` or ``ConfigResolution``, both its
    configurations; None: a default generation ``RunConfig``)."""

    from pyamplicol.api.requests import ProcessRequest, ProcessSet

    if isinstance(processes, (ProcessSet, ProcessRequest)):
        return processes, config
    items = [processes] if isinstance(processes, str) else list(processes)
    translations = tuple(
        translate_process_expression(item) for item in items if isinstance(item, str)
    )
    if not any(item.translated for item in translations):
        return processes, config
    if len(translations) != len(items):
        # ProcessRequest objects carry no annotations: as plain entries.
        translations = tuple(
            translate_process_expression(item)
            if isinstance(item, str)
            else ProcessSyntaxTranslation(item.expression, item.expression)
            for item in items
        )
    from pyamplicol.config import (
        Action,
        ConfigResolution,
        GenerationConfig,
        RunConfig,
    )

    if config is None:
        config = RunConfig(action=Action.GENERATE)
    if isinstance(config, GenerationConfig):
        raise ProcessSyntaxConflict(
            "the MG5_aMC syntax of "
            + ", ".join(repr(t.source) for t in translations if t.translated)
            + " translates to run-card settings ("
            + _settings_text(next(t for t in translations if t.translated))
            + "), which a GenerationConfig does not hold; pass a RunConfig"
        )
    if isinstance(config, ConfigResolution):
        from dataclasses import replace

        requested = apply_to_run_config(config.requested, translations)
        effective = apply_to_run_config(config.effective, translations)
        if effective is not config.effective:
            # As ``resolve_config`` resolves its effective configuration,
            # from the requested colour settings (the tree request's were
            # resolved for its lane).
            effective = replace(effective, color=requested.color)
            effective = replace(
                effective,
                evaluator=replace(
                    effective.evaluator,
                    execution_mode=effective.resolved_execution_mode,
                ),
            )
            effective = replace(effective, color=effective.resolved_color)
        config = ConfigResolution(
            requested=requested, effective=effective, clamps=config.clamps
        )
    else:
        config = apply_to_run_config(config, translations)
    plain = [
        translation.expression if isinstance(item, str) else item
        for item, translation in zip(items, translations, strict=True)
    ]
    return (plain[0] if isinstance(processes, str) else plain), config
