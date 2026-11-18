# SPDX-License-Identifier: 0BSD
"""Degenerate phase-space points of stored point sets (USER 2026-11-14 ~20:50 UTC:
tests and profiles use points sampling the whole phase space, "rambo on diet",
except where they intend to probe a limit).

``classify(point)`` names the degeneracies of one point ``[[E, px, py, pz], ...]``
(all legs, initial first; components as numbers, decimal or fraction texts or
binary64 hex texts) at the precision its own representation carries
(``precision``: 2^-53 for binary64 values, 10^-d for decimal texts of d
significant digits, exact for fractions): a configuration is degenerate when it is
indistinguishable from the limit at that precision, i.e. its scale-free measure
is at most ``ROUNDINGS`` times the representation's precision:

Decimal texts of at least ``ROUNDED_DIGITS`` significant digits are rounded
values, taken at the quantum of the point's finest last digit (the printing
quantum of a point printed on a common quantum); shorter terminating decimals,
integers and fractions are exact (a degenerate exact point is exactly
degenerate).

Kinematically forced alignments are not degeneracies: the two initial legs of a
2 -> n point and the two final legs of a 2 -> 2 or 1 -> 2 point (back to back in
their centre-of-mass frame) are not compared; a leg at rest has no direction.

* ``planar`` (n >= 5 legs): every four momenta have eps(p_a, p_b, p_c, p_d) /
  (E_a E_b E_c E_d) at the rounding level (the four-dimensional Gram determinant
  of any four momenta, eps^2, vanishes: the momenta span at most three
  dimensions);
* ``collinear i j``: two legs with nonzero three-momenta whose three-momenta are
  parallel or antiparallel, |p_i x p_j| / (|p_i| |p_j|) at the rounding level (a
  final leg along a beam is a forward configuration);
* ``soft i``: a massless leg whose energy is at the rounding level of the
  largest energy.

Everything is computed in exact rational arithmetic on the stored values, so the
measure is that of the stored point, not of a further rounding. Generic: no
process, mass or file is special-cased.
"""

from __future__ import annotations

import itertools
import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

# A measure within this many roundings of the representation is the limit.
ROUNDINGS = 1000
# A decimal text with at least this many significant digits is a rounded value
# (binary64's shortest round-trip texts have up to 17, printed points more); a
# shorter one is an exact terminating decimal.
ROUNDED_DIGITS = 16

_PERMUTATIONS = [
    ((-1) ** sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]), p)
    for p in itertools.permutations(range(4))
]


def component(value: Any) -> tuple[Fraction, Fraction | None, Fraction | None]:
    """The exact value of a stored component, its relative precision (binary64:
    2^-53) and its absolute quantum (a decimal text: the unit of its last
    digit); None where it has none (an integer, a fraction: exact)."""

    if isinstance(value, bool):
        raise TypeError("a boolean is not a momentum component")
    if isinstance(value, int):
        return Fraction(value), None, None
    if isinstance(value, float):
        return Fraction(value), Fraction(1, 2**53), None
    if isinstance(value, Decimal):
        value = str(value)
    if not isinstance(value, str):
        raise TypeError(f"not a momentum component: {value!r}")
    text = value.strip()
    if "/" in text:
        return Fraction(text), None, None
    if text.lower().lstrip("+-").startswith("0x"):
        return Fraction(float.fromhex(text)), Fraction(1, 2**53), None
    number = Decimal(text)
    exponent = number.as_tuple().exponent
    digits = len(number.as_tuple().digits)
    if not isinstance(exponent, int) or exponent >= 0 or digits < ROUNDED_DIGITS:
        # An integer or a short terminating decimal: an exact value (a rounded
        # generic real has as many digits as its carrier).
        return Fraction(number), None, None
    return Fraction(number), None, Fraction(1, 10**-exponent)


def is_point(value: Any) -> bool:
    """Whether a JSON value is a phase-space point: at least three legs of four
    numeric components each, physical (nonnegative energies, timelike or
    lightlike)."""

    if not isinstance(value, list) or len(value) < 3:
        return False
    exact = []
    for leg in value:
        if not isinstance(leg, list) or len(leg) != 4:
            return False
        for x in leg:
            if isinstance(x, bool) or not isinstance(x, (int, float, str)):
                return False
        try:
            exact.append([component(x)[0] for x in leg])
        except (ValueError, ArithmeticError, TypeError):
            return False
    # Physical momenta: nonnegative energies, timelike or lightlike (to a
    # relative 1e-6: rounded massless legs), not all zero (helicity or colour
    # tables, shift vectors and other four-entry rows are not points).
    tolerance = Fraction(1, 10**6)
    for e, x, y, z in exact:
        if e < 0 or e * e - x * x - y * y - z * z < -tolerance * e * e:
            return False
    return any(leg[0] > 0 for leg in exact)


def precision(point: list) -> Fraction | None:
    """The relative precision the point is stored at: the finest of its
    components' (binary64: 2^-53; decimal texts: the finest quantum of the
    point's last digits relative to its largest energy, the common printing
    quantum of a printed point); None when every component is exact (integers,
    fractions)."""

    scale = max(abs(component(leg[0])[0]) for leg in point)
    finest = None
    for leg in point:
        for x in leg:
            _, relative, quantum = component(x)
            candidate = (
                relative if quantum is None else (quantum / scale if scale else None)
            )
            if candidate is not None and (finest is None or candidate < finest):
                finest = candidate
    return finest


def _eps(rows) -> Fraction:
    total = Fraction(0)
    for sign, perm in _PERMUTATIONS:
        term = Fraction(sign)
        for row, column in enumerate(perm):
            term *= rows[row][column]
        total += term
    return total


def classify(point: list) -> list[str]:
    """The degeneracies of ``point`` (module docstring); empty for a generic
    point."""

    legs = [[component(x)[0] for x in leg] for leg in point]
    rel = precision(point)
    tolerance = Fraction(0) if rel is None else ROUNDINGS * rel
    found = []
    energies = [abs(leg[0]) for leg in legs]
    scale = max(energies)
    if scale == 0:
        return ["zero point"]
    decay = all(x == 0 for x in legs[0][1:])
    if len(legs) >= 5:
        largest = Fraction(0)
        for combo in itertools.combinations(range(len(legs)), 4):
            norm = math.prod(energies[k] for k in combo)
            if norm == 0:
                continue
            value = abs(_eps([legs[k] for k in combo])) / norm
            largest = max(largest, value)
            if largest > tolerance:
                break
        if largest <= tolerance:
            found.append("planar")

    def norm3(v):
        return v[1] * v[1] + v[2] * v[2] + v[3] * v[3]

    initial = 1 if decay else 2
    forced = {(0, 1)} if initial == 2 else set()
    if len(legs) - initial == 2:
        forced.add((initial, initial + 1))
    for i, j in itertools.combinations(range(len(legs)), 2):
        if (i, j) in forced:
            continue
        a, b = legs[i], legs[j]
        na, nb = norm3(a), norm3(b)
        if na == 0 or nb == 0:
            continue
        cross = (
            (a[2] * b[3] - a[3] * b[2]) ** 2
            + (a[3] * b[1] - a[1] * b[3]) ** 2
            + (a[1] * b[2] - a[2] * b[1]) ** 2
        )
        # |a x b|^2 / (|a|^2 |b|^2) <= tolerance^2.
        if cross <= tolerance * tolerance * na * nb:
            found.append(f"collinear {i} {j}")
    for k, leg in enumerate(legs):
        mass2 = leg[0] ** 2 - norm3(leg)
        massless = abs(mass2) <= tolerance * leg[0] ** 2 if leg[0] else True
        if massless and energies[k] <= tolerance * scale:
            found.append(f"soft {k}")
    return found


def points_in(value: Any, path: str = ""):
    """Every point stored in a JSON value, with its path."""

    if is_point(value):
        yield path, value
        return
    if (
        isinstance(value, list)
        and value
        and all(isinstance(leg, dict) and "momentum" in leg for leg in value)
    ):
        # Legs as records (a validation point: {"pdg": ..., "momentum": [...]}).
        legs = [leg["momentum"] for leg in value]
        if is_point(legs):
            yield path, legs
            return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from points_in(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from points_in(item, f"{path}[{index}]")


# The stored point sets tests, benchmarks, profiles and stability tools read.
ROOTS = (
    "tests/fixtures",
    "tools/loop_reference",
    "tools/loop_benchmark",
    "docs/performance_reports/loop/data",
    "examples",
    "src/pyamplicol/assets/selftest",
)


def intended_limits(data: Any) -> dict[str, str]:
    """A point set's intended limit points: its top-level ``intended_limits``
    mapping from a point's path (``points_in``) to the limit it probes and the
    expected behaviour."""

    if isinstance(data, dict) and isinstance(data.get("intended_limits"), dict):
        return {str(k): str(v) for k, v in data["intended_limits"].items()}
    return {}


def violations(root) -> list[tuple[str, str, list[str]]]:
    """Every degenerate stored point under the directories ``ROOTS`` of the
    checkout ``root`` that its file does not declare an intended limit (with
    a reason): (file, path, degeneracies)."""

    import json
    from pathlib import Path

    found = []
    for directory in ROOTS:
        for path in sorted((Path(root) / directory).rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            limits = intended_limits(data)
            for where, point in points_in(data):
                kinds = classify(point)
                if kinds and not limits.get(where):
                    found.append((str(path.relative_to(root)), where, kinds))
    return found


__all__ = [
    "ROOTS",
    "ROUNDED_DIGITS",
    "ROUNDINGS",
    "classify",
    "component",
    "intended_limits",
    "is_point",
    "points_in",
    "precision",
    "violations",
]
