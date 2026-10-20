# SPDX-License-Identifier: 0BSD
"""Transport typed colour labels between physical slots and public leg labels."""

from collections.abc import Callable

from pyamplicol.color.connections import ColorEmission, EmitGluon, SplitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator


def _map(history: ColorCorrelator, label: Callable[[int], int]) -> ColorCorrelator:
    def operation(step: ColorEmission) -> ColorEmission:
        if isinstance(step, EmitGluon):
            return EmitGluon(label(step.emitter_label), label(step.emitted_label))
        return SplitGluon(
            label(step.parent_label),
            label(step.quark_label),
            label(step.antiquark_label),
        )

    return ColorCorrelator(
        history.id,
        tuple(map(operation, history.bra)),
        tuple(map(operation, history.ket)),
    )


def public_color_history(history: ColorCorrelator) -> ColorCorrelator:
    """Keep negative auxiliaries; shift nonnegative native labels by one.

    Shifting every nonnegative label also preserves distinct positive auxiliary
    labels. No permutation, charge sign or tensor operation is applied.
    """
    return _map(history, lambda value: value + 1 if value >= 0 else value)


def native_color_history(history: ColorCorrelator) -> ColorCorrelator:
    def label(value: int) -> int:
        if value == 0:
            raise ValueError(
                "public colour labels are positive legs or negative auxiliaries"
            )
        return value - 1 if value > 0 else value

    return _map(history, label)


def permute_native_color_history(
    history: ColorCorrelator,
    permutation: tuple[int, ...],
    *,
    physical_labels: tuple[int, ...],
) -> ColorCorrelator:
    """Relabel native physical ports; auxiliary identities never move.

    A physical label remains physical when a split retires its initial gluon.
    Only actual initial coloured ports identify physical colour labels. An
    auxiliary may equal a colourless external slot and must still remain fixed.
    The caller's existing ColorConnection validates the resulting typed ports
    and creation history.
    """
    permutation = tuple(permutation)
    if any(type(value) is not int for value in permutation) or tuple(
        sorted(permutation)
    ) != tuple(range(len(permutation))):
        raise ValueError("colour history permutation must cover physical ports")
    labels = tuple(physical_labels)
    if (
        any(
            type(value) is not int or not 0 <= value < len(permutation)
            for value in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("colour history requires unique actual physical colour ports")
    physical = frozenset(labels)
    return _map(
        history, lambda value: permutation[value] if value in physical else value
    )
