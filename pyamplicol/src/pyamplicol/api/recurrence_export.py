# SPDX-License-Identifier: 0BSD
"""Optional numerical references for a static recurrence export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from .errors import EvaluationError
from .protocols import Momenta
from .results import ResolvedEvaluation

if TYPE_CHECKING:
    from .services import Runtime


def attach_reference_points(
    export: dict[str, object], runtime: Runtime, momenta: Momenta
) -> dict[str, object]:
    """Attach f64 caller-order inputs and per-helicity outputs to each program."""

    requested_points = list(momenta)
    if not requested_points:
        raise ValueError("reference_momenta requires at least one phase-space point")
    programs = cast(Sequence[Mapping[str, object]], export["programs"])
    colours = list(dict.fromkeys(str(program["colour_id"]) for program in programs))
    # A singleton may activate a distinct interaction rewrite in the runtime.
    # Evaluate two copies and retain one reference to describe the batched path.
    evaluated = runtime.evaluate_resolved(
        requested_points if len(requested_points) > 1 else requested_points * 2,
        color_flows=colours,
        precision=16,
    )
    if not isinstance(evaluated, ResolvedEvaluation):
        raise EvaluationError("recurrence reference evaluation requires tree values")
    points = [
        [[float(component) for component in leg] for leg in point]
        for point in requested_points
    ]
    helicities = {
        identifier: index for index, identifier in enumerate(evaluated.helicity_ids)
    }
    colors = {identifier: index for index, identifier in enumerate(evaluated.color_ids)}
    with_references = []
    for program in programs:
        output_ids = [
            str(output["id"])
            for output in cast(
                Sequence[Mapping[str, object]], program["helicity_outputs"]
            )
        ]
        try:
            color = colors[str(program["colour_id"])]
            output_indices = [helicities[identifier] for identifier in output_ids]
        except KeyError as exc:
            raise EvaluationError(
                "recurrence export axes do not match the resolved reference evaluation"
            ) from exc
        with_references.append(
            {
                **program,
                "reference_values": {
                    "momenta": points,
                    "helicity_ids": output_ids,
                    "values": [
                        [
                            complex(evaluated.values[point][helicity][color]).real
                            for helicity in output_indices
                        ]
                        for point in range(len(points))
                    ],
                },
            }
        )
    return {**export, "programs": with_references}
