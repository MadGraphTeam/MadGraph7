# SPDX-License-Identifier: 0BSD
"""Prepare UFO color words through public native tensor constructors.

HepKit lowers the ordinary UFO color heads itself. Its current dispatcher does
not recognize ``Tr``; represent that ordered fundamental trace with the native
trace API before HepKit localizes UFO leg indices. No generator components,
matrix multiplication, trace identities or index-contraction rules live here.
The original UFO expression remains in the model sidecar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .model_expressions import parse_exact_ufo


def prepare_color_expression(source: str, particle_colors: Sequence[int]) -> Any:
    """Return an exact expression accepted by the native local-rule loader.

    The initial trace boundary is SU(3), with distinct positive adjoint external
    leg indices. Negative dummy indices and other representations are explicitly
    unsupported here. This does not certify a model's general-Nc continuation.
    Unknown UFO function heads fail before they can become unresolved scalars.
    """
    from symbolica import AtomType, S
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor

    expression = parse_exact_ufo(source)
    pending = [expression]
    traces = set()
    native_heads = {"Identity", "T", "f", "d", "Epsilon", "EpsilonBar"}
    while pending:
        atom = pending.pop()
        kind = atom.get_type()
        if kind == AtomType.Fn:
            name = atom.get_name()
            if name == "UFO::Tr":
                traces.add(atom)
            elif name.startswith("UFO::") and name[5:] not in native_heads:
                raise NotImplementedError(f"unsupported UFO color head {name!r}")
        if kind in {AtomType.Fn, AtomType.Add, AtomType.Mul, AtomType.Pow}:
            pending.extend(atom)

    generator = tensor.TensorExpression.color_t(8, 3)
    for trace in traces:
        try:
            legs = tuple(int(argument) for argument in trace)
        except (TypeError, ValueError) as error:
            raise NotImplementedError(
                "Tr indices must be integer external legs"
            ) from error
        if len(legs) < 2 or len(set(legs)) != len(legs):
            raise NotImplementedError("Tr requires at least two distinct external legs")
        if any(
            leg < 1 or leg > len(particle_colors) or particle_colors[leg - 1] != 8
            for leg in legs
        ):
            raise NotImplementedError(
                "Tr requires positive SU(3) adjoint external legs"
            )
        # Public tensor constructors require atomic spectator labels. Native UFO
        # idx wrappers are inserted afterward, then localized by HepKit itself.
        labels = tuple(S(f"pyamplicol_color_trace::leg{leg}") for leg in legs)
        replacement = tensor.trace(
            tensor.Representation.cof(3),
            *(generator(label, tensor.AUTO, tensor.AUTO) for label in labels),
        ).to_expression()
        for label, leg in zip(labels, legs, strict=True):
            replacement = replacement.replace(label, hep.Symbols.ufo_index()(1, leg))
        expression = expression.replace(trace, replacement)
    return expression
