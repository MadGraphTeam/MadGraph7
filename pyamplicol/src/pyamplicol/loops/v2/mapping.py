# SPDX-License-Identifier: 0BSD
"""Processes served by a stored plan v4 process through a leg mapping (M-FINAL
MF-MAP; design §9).

A *mapped* process is evaluated by a stored representative at permuted
momenta (``observables.FlavourMember``: ``legs[i]`` is the representative leg
of public leg i, within the initial and within the final state):

* a **permutation alias**: an explicit ``ProcessAlias`` (the manifest's alias
  record of its target), or a selector expression equal to a stored
  process's up to a permutation within each side (``side_permutation``); a
  relabelling of legs that carry the same particle is exact, no proof is
  needed;
* a **process-group member** (``process-groups.json``, basis
  ``exact-loop-flavor``): a species substitution proved exactly on the
  representative's retained source plan at generation
  (``observables.admit_member``), possibly composed with such a permutation.

Correlations compose with the mapping (design §9 "Composition"): a declared
colour history read on the mapped process's labels is relabelled onto the
representative's and served by the representative's stored history of equal
structure (``mapped_histories``); a spin class likewise
(``mapped_spin_classes``). A history or class whose image the representative
does not store is not served (``MappedObservables`` lists only those served).
Numerical process-group analysis (a relation found only by numerical
comparison) is refused at generation (``refusals``: ``process-group-analysis``).

Nothing here touches Symbolica: the stored columns are the representative's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def side_permutation(representative: str, requested: str) -> tuple[int, ...] | None:
    """The representative-to-public permutation (``permutation[r]`` is the
    public leg of representative leg r) under which ``requested`` lists the
    particles of ``representative`` in another order within each side, or
    None. Tokens compare letter-case-insensitively, as the artifact selector
    does (rusticol-core ``normalize_process_expression``); repeated particles
    take the first unused leg (``side_preserving_process_permutation``)."""

    rep = _sides(representative)
    public = _sides(requested)
    if (
        rep is None
        or public is None
        or [len(s) for s in rep] != [len(s) for s in public]
    ):
        return None
    permutation = [0] * sum(len(side) for side in rep)
    offset = 0
    for rep_side, public_side in zip(rep, public, strict=True):
        free = list(range(len(rep_side)))
        for public_index, token in enumerate(public_side):
            found = next((k for k in free if rep_side[k] == token), None)
            if found is None:
                return None
            free.remove(found)
            permutation[offset + found] = offset + public_index
        offset += len(rep_side)
    return tuple(permutation)


def _sides(expression: str) -> list[list[str]] | None:
    parts = expression.split(">")
    if len(parts) != 2:
        return None
    sides = [[token.lower() for token in part.split()] for part in parts]
    return sides if all(sides) else None


def permuted_expression(expression: str, permutation: Sequence[int]) -> str:
    """``expression`` with its particles in the public order of
    ``permutation`` (representative leg r -> public leg ``permutation[r]``),
    in the representative's spelling."""

    initial, final = (part.split() for part in expression.split(">"))
    tokens = [*initial, *final]
    public: list[str] = [""] * len(tokens)
    for r, p in enumerate(permutation):
        public[p] = tokens[r]
    return " ".join(public[: len(initial)]) + " > " + " ".join(public[len(initial) :])


def inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    out = [0] * len(permutation)
    for index, image in enumerate(permutation):
        out[image] = index
    return tuple(out)


def mapped_histories(member: Any, histories: Sequence[Any]) -> dict[str, str]:
    """Mapped history id -> representative history id: a declared history
    (``histories``, the representative's stored ones) read on the mapped
    process's labels is relabelled onto the representative's
    (``FlavourMember.history``) and served by the stored history of equal
    bra and ket; one without such an image is absent."""

    stored = {(history.bra, history.ket): history.id for history in histories}
    out = {}
    for history in histories:
        image = member.history(history)
        found = stored.get((image.bra, image.ket))
        if found is not None:
            out[history.id] = found
    return out


def mapped_spin_classes(
    member: Any, classes: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """The declared spin classes (one-based legs) a mapped process is served:
    those whose legs, relabelled onto the representative, form a stored
    class."""

    stored = {tuple(sorted(group)) for group in classes}
    return tuple(
        tuple(group)
        for group in classes
        if tuple(sorted(member.label(leg) for leg in group)) in stored
    )


class MappedObservables:
    """A stored process's observable columns (``native_observables.
    StoredObservableSlice``) as a permuted mapped process reads them: its
    served colour histories (ids as declared, read on its own labels) and spin
    classes, and correlated column sets whose outputs carry its history ids.
    Momenta, spin legs and resolved helicity rows are relabelled by the
    member itself (``StoredColumns``, ``V2ObservablesRuntime``)."""

    def __init__(self, observables: Any, member: Any):
        self._observables = observables
        self._ids = mapped_histories(member, observables.histories)
        self.histories = tuple(
            history for history in observables.histories if history.id in self._ids
        )
        self.spin_classes = mapped_spin_classes(member, observables.spin_classes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._observables, name)

    def history(self, identifier: str) -> Any:
        from .observables import INCLUSIVE

        if identifier != INCLUSIVE and identifier not in self._ids:
            raise KeyError(f"unserved colour correlation {identifier!r}")
        return self._observables.history(identifier)

    def native_columns(self, kind: str, value: Any, **options: Any) -> Any:
        if kind != "correlated":
            return self._observables.native_columns(kind, value, **options)
        from .observables import INCLUSIVE

        wanted = tuple(dict.fromkeys(value))
        missing = [n for n in wanted if n != INCLUSIVE and n not in self._ids]
        if missing:
            raise KeyError(f"unserved colour correlations {missing!r}")
        keys = {name: self._ids.get(name, name) for name in wanted}
        found = self._observables.native_columns(
            kind, tuple(dict.fromkeys(keys.values())), **options
        )
        return _Rekeyed(found, keys)


class _Rekeyed:
    """A correlated column selection whose outputs are re-keyed from the
    representative's history ids to the mapped process's."""

    def __init__(self, selection: Any, keys: Mapping[str, str]):
        self._selection = selection
        self._keys = dict(keys)

    def _rekey(self, result: Any) -> Any:
        if result is None:
            return None
        return {name: result[key] for name, key in self._keys.items()}

    def evaluate(self, points: Any, scales: Any, **options: Any) -> Any:
        from .plan_export import FailedPoints

        try:
            results = self._selection.evaluate(points, scales, **options)
        except FailedPoints as failed:
            raise FailedPoints(
                [self._rekey(result) for result in failed.results], failed.failures
            ) from failed
        return [self._rekey(result) for result in results]


__all__ = [
    "MappedObservables",
    "inverse",
    "mapped_histories",
    "mapped_spin_classes",
    "permuted_expression",
    "side_permutation",
]
