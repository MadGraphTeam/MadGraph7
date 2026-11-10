# SPDX-License-Identifier: 0BSD
"""The kinematics improvement of the input points (m-lit lit-psimprove).

Before a point is evaluated in a carrier above binary64 (double-double or
arbitrary precision), its external momenta are moved onto exact
energy-momentum conservation and the external mass shells in that carrier's
arithmetic (USER 2026-11-10 ~00:56), by default with MadLoop's default
(``ImprovePSPoint = 2`` with MadLoop's own fallback to ``ImprovePSPoint =
1``; USER ~01:56, DECISIONS 2026-11-10 01:56 (1)); binary64 evaluations are
improved only under the binary64 switch (DECISIONS 01:56 (2); what counts as
a binary64 evaluation is the carrier kind, DECISIONS 02:29 (G3)).

The one implementation is Rust's (``rusticol_core::kinematics_improvement``,
through ``improve_kinematics``); this module reads its policy (the values,
the default variant, the binary64 switch's default, the refusal texts), and
forms the public record (``api.virtual_results.VirtualKinematics``). Python
never re-implements the algorithm.

Settings: ``kinematics_improvement`` = ``"z-shift"`` (MadLoop 2, the
default) | ``"rescaling"`` (MadLoop 1) | ``"projection"`` (the
frame-independent minimal-norm projection) | ``"off"``, and
``kinematics_improvement_binary64`` (bool, default False).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

SETTING = "kinematics_improvement"
BINARY64 = "kinematics_improvement_binary64"


def _native() -> Any:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


@functools.lru_cache(maxsize=1)
def policy() -> Mapping[str, Any]:
    """The native policy record (``kinematics_policy()``), read once."""

    return json.loads(_native().kinematics_policy())


def default_setting() -> str:
    """The default variant (MadLoop's, the native policy's one constant)."""

    return str(policy()["default"])


def default_binary64() -> bool:
    """The binary64 switch's default (off)."""

    return bool(policy()["binary64_default"])


def setting_value(value: object) -> str:
    """Validate a variant setting (the native policy's values and refusal)."""

    if not isinstance(value, str) or value not in policy()["settings"]:
        raise ValueError(policy()["setting_refusal"])
    return value


def binary64_value(value: object) -> bool:
    """Validate the binary64 switch (a boolean; the native refusal text)."""

    if not isinstance(value, bool):
        raise ValueError(policy()["binary64_refusal"])
    return value


@dataclass(frozen=True, slots=True)
class KinematicsSettings:
    """A call's resolved settings: the variant setting and the binary64
    switch."""

    setting: str
    binary64: bool

    def variant(self, carrier: str) -> str | None:
        """The variant an evaluation in ``carrier`` (``"double"``,
        ``"double-double"``, ``"arbitrary"``) runs, or None (the native key,
        ``kinematics_applies``)."""

        return _native().kinematics_applies(self.setting, self.binary64, carrier)

    def records(self, carrier: str) -> bool:
        """Whether results in ``carrier`` carry a kinematics record: every
        evaluation the settings reach (binary64 only under the switch), also
        when the setting is ``"off"`` (the record then reads ``"off"``; the
        native key, ``kinematics_reaches``)."""

        return bool(_native().kinematics_reaches(self.binary64, carrier))


def resolve(
    setting: str | None,
    binary64: bool | None,
    *,
    default_setting_: str | None = None,
    default_binary64_: bool | None = None,
) -> KinematicsSettings:
    """A call's settings: its own, else the runtime's (load-time) ones, else
    the policy's defaults."""

    chosen = (
        setting_value(setting)
        if setting is not None
        else default_setting_
        if default_setting_ is not None
        else default_setting()
    )
    switch = (
        binary64_value(binary64)
        if binary64 is not None
        else default_binary64_
        if default_binary64_ is not None
        else default_binary64()
    )
    return KinematicsSettings(chosen, switch)


def improve(
    points: Sequence[Sequence[Sequence[Any]]],
    n_initial: int,
    masses2: Sequence[Fraction | None],
    arithmetic: str,
    digits: int | None,
    variant: str,
    *,
    fallback: bool = True,
    tolerances: Mapping[str, float] | None = None,
) -> tuple[list[list[list[str]]], list[dict[str, Any]]]:
    """Improve ``points`` (physical momenta, the ``n_initial`` initial states
    first, exact components) in ``arithmetic`` (``"arbitrary"`` at
    ``digits`` working digits, the native executor's) with ``variant``
    toward ``masses2`` (exact; None: a leg's own p²): the improved points
    (exact fraction texts; unchanged where no variant improves the point or
    it is already exactly physical) and the per-point records (the native
    JSON records, ``kinematics_improvement::Record``). ``fallback=False``
    runs the variant alone (a forced variant)."""

    texts = [
        [[str(Fraction(str(x))) for x in leg] for leg in point] for point in points
    ]
    masses = [None if m is None else str(Fraction(m)) for m in masses2]
    extra = dict(tolerances or {})
    out, records = _native().improve_kinematics(
        texts,
        n_initial,
        masses,
        arithmetic,
        digits,
        variant,
        fallback,
        extra.get("fourmom"),
        extra.get("onshell"),
        extra.get("close"),
    )
    found = [json.loads(record) for record in records]
    # An improved point as the exact decimal texts of its carrier values
    # (dyadic rationals: finite decimal expansions, the inputs' syntax); any
    # other point is the caller's own, unchanged.
    improved = [
        [[_exact_decimal(x) for x in leg] for leg in point]
        if record["status"] == "improved"
        else [[str(x) for x in leg] for leg in original]
        for point, record, original in zip(out, found, points, strict=True)
    ]
    return improved, found


def _exact_decimal(text: str) -> str:
    """The exact decimal text of a dyadic rational (a carrier value's exact
    rational, ``n / 2^k``)."""

    from decimal import Decimal

    value = Fraction(text)
    k = value.denominator.bit_length() - 1
    if value.denominator != 1 << k:
        raise ValueError(f"not a carrier value: {text}")
    scaled = value.numerator * 5**k
    digits = tuple(int(c) for c in str(abs(scaled)))
    # Constructed from its digits: exact (no context rounding).
    return str(Decimal((int(scaled < 0), digits, -k)))


def off_record(setting: str) -> dict[str, Any]:
    """The record of a point the settings reach but do not improve (the
    setting ``"off"``; the native record, ``kinematics_off_record``, the
    loop SDK's)."""

    return dict(json.loads(_native().kinematics_off_record(setting)))


def notes(
    records: Sequence[Mapping[str, Any]], carrier: str, precision: int
) -> list[str]:
    """The tree runtimes' warnings-channel lines of one call's records (the
    native formatter, ``kinematics_improvement::notes_json``, the tree C
    ABI's): a summary line and every typed warning."""

    return list(
        _native().kinematics_notes(
            [json.dumps(dict(record)) for record in records], carrier, precision
        )
    )


# A leg-mass provider: (v2 Precision, a caller-order point, its mu_R^2, the
# batch's point count) -> the squared mass of each caller-order leg (exact;
# None: no bound mass), exactly the masses the evaluation binds.
MassProvider = Callable[[Any, Sequence[Sequence[Any]], Any, int], list[Fraction | None]]


def initial_count(legs: Sequence[Any]) -> int:
    """The number of initial states of a process's legs (``is_initial``,
    ``label``): they come first (the improvement's convention, MadLoop's)."""

    initial = sorted(leg.label for leg in legs if leg.is_initial)
    if initial != list(range(1, len(initial) + 1)):
        raise ValueError("the kinematics improvement needs the initial states first")
    return len(initial)


class LevelImprover:
    """One call's kinematics improvement, level by level: each carrier's
    evaluation of the user's points (a call without a rescue policy, or one
    rescue level) improves the ORIGINAL caller-order points once in its own
    carrier (MadLoop's SET_MP_PS: never an already improved point), with the
    masses the evaluation binds (``masses``). A point improved for a carrier
    is reused by every later evaluation of it at the same carrier."""

    def __init__(
        self, settings: KinematicsSettings, n_initial: int, masses: MassProvider
    ) -> None:
        self.settings = settings
        self.n_initial = n_initial
        self._masses = masses
        self._cache: dict[tuple[Any, ...], tuple[list[list[str]], dict[str, Any]]] = {}

    def level(
        self,
        points: Sequence[Sequence[Sequence[Any]]],
        scales: Sequence[Any],
        precision: Any,
    ) -> tuple[list[Any], list[Any]]:
        """The points of one carrier's evaluation (``precision``, a v2
        ``Precision``) and each point's record (``VirtualKinematics``; None:
        no record, a binary64 evaluation without the binary64 switch)."""

        from pyamplicol.api.virtual_results import VirtualKinematics

        carrier = precision.arithmetic
        variant = self.settings.variant(carrier)
        if variant is None:
            if not self.settings.records(carrier):
                return list(points), [None] * len(points)
            off = VirtualKinematics.from_record(off_record(self.settings.setting))
            return list(points), [off] * len(points)
        digits = precision.working_digits if carrier == "arbitrary" else None
        out: list[Any] = list(points)
        records: list[Any] = [None] * len(points)
        groups: dict[Fraction, list[int]] = {}
        for index, scale in enumerate(scales):
            groups.setdefault(Fraction(str(scale)), []).append(index)
        for indices in groups.values():
            first = indices[0]
            masses = self._masses(precision, points[first], scales[first], len(points))
            keys = {
                i: (carrier, digits, variant, tuple(masses), _key(points[i]))
                for i in indices
            }
            missing = [i for i in indices if keys[i] not in self._cache]
            if missing:
                improved, found = improve(
                    [points[i] for i in missing],
                    self.n_initial,
                    masses,
                    carrier,
                    digits,
                    variant,
                )
                for i, point, record in zip(missing, improved, found, strict=True):
                    self._cache[keys[i]] = (point, record)
            for i in indices:
                point, record = self._cache[keys[i]]
                out[i] = point
                records[i] = VirtualKinematics.from_record(record)
        return out, records


def _key(point: Sequence[Sequence[Any]]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(str(x)) for x in leg) for leg in point)


def with_kinematics(evaluation: Any, records: Sequence[Any]) -> Any:
    """``evaluation`` (a ``VirtualEvaluation``) with each lane diagnostic's
    kinematics record; unchanged when no point has one."""

    from dataclasses import replace

    if all(record is None for record in records):
        return evaluation
    diagnostics = tuple(
        replace(diagnostic, kinematics=record)
        for diagnostic, record in zip(evaluation.diagnostics, records, strict=True)
    )
    return replace(evaluation, diagnostics=diagnostics)


__all__ = [
    "BINARY64",
    "SETTING",
    "KinematicsSettings",
    "LevelImprover",
    "MassProvider",
    "binary64_value",
    "default_binary64",
    "default_setting",
    "improve",
    "initial_count",
    "notes",
    "off_record",
    "policy",
    "resolve",
    "setting_value",
    "with_kinematics",
]
