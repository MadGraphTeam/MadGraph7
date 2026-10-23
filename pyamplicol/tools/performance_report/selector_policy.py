# SPDX-License-Identifier: 0BSD
"""Backend-independent selector policy for LC report comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class SelectorPolicyError(ValueError):
    """Raised when a physical selector axis cannot be chosen unambiguously."""


def canonical_lc_flow_word(word: Sequence[int]) -> tuple[int, ...]:
    """Validate and return one labelled physical LC flow word."""

    normalized = tuple(word)
    if (
        not normalized
        or any(
            isinstance(label, bool) or not isinstance(label, int)
            for label in normalized
        )
        or any(label < 1 for label in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise SelectorPolicyError(
            "LC selector flow words must contain distinct positive source labels"
        )
    return normalized


def canonical_lc_selector_word(
    words: Iterable[Sequence[int]],
) -> tuple[int, ...]:
    """Choose the backend-independent representative of a physical LC axis.

    The choice is deliberately structural.  Selecting by evaluated magnitude
    makes two correct backends choose different labelled components whenever
    their physical axes have different enumeration orders or roundoff.
    """

    normalized = tuple(canonical_lc_flow_word(word) for word in words)
    if not normalized:
        raise SelectorPolicyError("LC selector derivation requires a color-flow axis")
    if len(set(normalized)) != len(normalized):
        raise SelectorPolicyError("LC selector color-flow words must be unique")
    return min(normalized)


def selector_color_flow_id(word: Sequence[int]) -> str:
    normalized = canonical_lc_flow_word(word)
    return "flow:" + ",".join(str(label) for label in normalized)


def _preferred_helicities(pdg: int) -> tuple[int, ...]:
    absolute = abs(int(pdg))
    if absolute in {
        1,
        2,
        3,
        4,
        5,
        6,
        11,
        12,
        13,
        14,
        15,
        16,
        21,
        22,
    }:
        return (-1, 1)
    if absolute in {23, 24}:
        return (-1, 0, 1)
    return (0,)


def fixed_selector_helicity(pdgs: Sequence[int]) -> tuple[int, ...]:
    """Return the deterministic physical helicity shared by report backends."""

    if not pdgs:
        raise SelectorPolicyError("selector helicity derivation requires external PDGs")
    result: list[int] = []
    charged_current_fermions = any(abs(int(pdg)) in {12, 14, 16} for pdg in pdgs)
    for index, pdg in enumerate(pdgs, start=1):
        domain = _preferred_helicities(pdg)
        if -1 in domain and 1 in domain:
            # Prefer the left-chiral particle/right-chiral antiparticle state
            # for charged-current fermion chains.  Preserve the established
            # alternating vector/neutral-current report convention otherwise.
            if charged_current_fermions and (
                1 <= abs(int(pdg)) <= 6 or 11 <= abs(int(pdg)) <= 16
            ):
                result.append(-1 if int(pdg) > 0 else 1)
            else:
                result.append(-1 if index % 2 else 1)
        elif 0 in domain:
            result.append(0)
        else:
            result.append(domain[0])
    return tuple(result)


def selector_helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


__all__ = [
    "SelectorPolicyError",
    "canonical_lc_flow_word",
    "canonical_lc_selector_word",
    "fixed_selector_helicity",
    "selector_color_flow_id",
    "selector_helicity_id",
]
