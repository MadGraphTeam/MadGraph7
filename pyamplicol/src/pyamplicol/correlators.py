# SPDX-License-Identifier: 0BSD
"""Opt-in declarations for tree-level colour and spin correlations.

Colour connections are ordered and use all-outgoing colour charges. External
legs keep the process's one-based labels; negative labels are convenient for
auxiliary emitted partons. These are colour operations, not additional external
momenta. The ordinary Born colour metric is always available as ``"born"``.

Spin classes declare the exact sets of vector legs that may be replaced
simultaneously. Numerical vectors are supplied to the runtime, not generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from pyamplicol.color.connections import EmitGluon, SplitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator

__all__ = [
    "ColorCorrelator",
    "CorrelatedValue",
    "CorrelatorConfig",
    "EmitGluon",
    "SplitGluon",
]


@dataclass(frozen=True)
class CorrelatedValue:
    """One possibly complex correlation, retaining the requested precision.

    ``real`` and ``imag`` are decimals. ``complex(value)`` converts to ordinary
    double precision. Ordered colour connections need not give a real result;
    their imaginary part must not be discarded before physical combinations.
    """

    real: Decimal
    imag: Decimal = Decimal(0)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))


def _fields(value: object, allowed: set[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {context} fields: {sorted(unknown, key=str)}")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a list")
    return value


def _emission_from_json(value: object) -> EmitGluon | SplitGluon:
    if not isinstance(value, Mapping):
        raise TypeError("a colour connection step must be an object")
    kind = value.get("kind")
    if kind == "emit-gluon":
        data = _fields(value, {"kind", "emitter", "emitted"}, "gluon emission")
        try:
            return EmitGluon(cast(int, data["emitter"]), cast(int, data["emitted"]))
        except KeyError as exc:
            raise ValueError(f"gluon emission is missing {exc.args[0]!r}") from exc
    if kind == "split-gluon":
        data = _fields(
            value, {"kind", "parent", "quark", "antiquark"}, "gluon splitting"
        )
        try:
            return SplitGluon(
                cast(int, data["parent"]),
                cast(int, data["quark"]),
                cast(int, data["antiquark"]),
            )
        except KeyError as exc:
            raise ValueError(f"gluon splitting is missing {exc.args[0]!r}") from exc
    raise ValueError(f"unknown colour connection step {kind!r}")


@dataclass(frozen=True)
class CorrelatorConfig:
    """The correlations to prepare alongside a complete tree amplitude.

    ``color_correlations`` contains the requested ``bra† ket`` connections,
    keyed by unique IDs. Each side has at most three ordered emissions.
    ``spin_correlations=((3,), (3, 4))`` permits replacing leg 3 alone or legs
    3 and 4 together. An empty runtime vector map always restores the ordinary
    helicity sum. Each replaced leg contributes one vector state; the usual
    initial-state averaging and identical-particle factors are retained.

    The first correlated executor uses the generic compiled amplitude and
    direct exact colour algebra. It does not use FFT or helicity-specific
    current recycling. Uncorrelated generation does not construct this object.
    """

    color_correlations: tuple[ColorCorrelator, ...] = ()
    spin_correlations: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        colors = tuple(self.color_correlations)
        if any(not isinstance(item, ColorCorrelator) for item in colors):
            raise TypeError("color_correlations must contain ColorCorrelator records")
        ids = tuple(item.id for item in colors)
        if "born" in ids:
            raise ValueError("the colour correlation ID 'born' is reserved")
        if len(set(ids)) != len(ids):
            raise ValueError("colour correlation IDs must be unique")
        classes: list[tuple[int, ...]] = []
        for group in self.spin_correlations:
            legs = tuple(group)
            if not legs:
                raise ValueError("a spin-correlation class must contain a leg")
            if any(type(leg) is not int or leg <= 0 for leg in legs):
                raise ValueError("spin-correlation legs must be positive integers")
            if len(set(legs)) != len(legs):
                raise ValueError("a spin-correlation class repeats a leg")
            canonical = tuple(sorted(legs))
            if canonical in classes:
                raise ValueError("spin-correlation classes must be distinct")
            classes.append(canonical)
        object.__setattr__(self, "color_correlations", colors)
        object.__setattr__(self, "spin_correlations", tuple(sorted(classes)))

    @property
    def spin_legs(self) -> tuple[int, ...]:
        """All external legs whose source vectors must remain unrestricted."""
        return tuple(sorted({leg for group in self.spin_correlations for leg in group}))

    @property
    def color_requests(self) -> tuple[ColorCorrelator, ...]:
        """Requested operators, including the ordinary Born overlap."""
        return (ColorCorrelator("born"), *self.color_correlations)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "color_correlations": [
                item.to_json_dict() for item in self.color_correlations
            ],
            "spin_correlations": [list(group) for group in self.spin_correlations],
        }

    @classmethod
    def from_json_dict(cls, value: object) -> CorrelatorConfig:
        """Read the same declaration used by Python and ``--correlators``."""
        data = _fields(
            value, {"color_correlations", "spin_correlations"}, "correlator declaration"
        )
        colors: list[ColorCorrelator] = []
        for record in _sequence(
            data.get("color_correlations", ()), "color_correlations"
        ):
            item = _fields(record, {"id", "bra", "ket"}, "colour correlation")
            if "id" not in item:
                raise ValueError("a colour correlation requires an ID")
            colors.append(
                ColorCorrelator(
                    item["id"],  # type: ignore[arg-type]
                    tuple(
                        _emission_from_json(step)
                        for step in _sequence(item.get("bra", ()), "bra")
                    ),
                    tuple(
                        _emission_from_json(step)
                        for step in _sequence(item.get("ket", ()), "ket")
                    ),
                )
            )
        spins = tuple(
            tuple(_sequence(group, "spin-correlation class"))
            for group in _sequence(
                data.get("spin_correlations", ()), "spin_correlations"
            )
        )
        return cls(tuple(colors), spins)  # type: ignore[arg-type]
