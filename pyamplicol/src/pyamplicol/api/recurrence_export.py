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
    export: dict[str, object],
    runtime: Runtime,
    momenta: Momenta,
    *,
    crossed_member: str | None = None,
) -> dict[str, object]:
    """Attach f64 caller-order inputs and per-helicity outputs to each program."""

    requested_points = list(momenta)
    if not requested_points:
        raise ValueError("reference_momenta requires at least one phase-space point")
    programs = cast(Sequence[Mapping[str, object]], export["programs"])
    if crossed_member is not None:
        return _attach_crossed_reference_points(
            export, runtime, requested_points, crossed_member
        )
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


def _attach_crossed_reference_points(
    export: dict[str, object],
    runtime: Runtime,
    requested_points: list[object],
    member: str,
) -> dict[str, object]:
    """References of a crossed member: its template's assignment at
    ``q[program_leg] = sign * p[caller_leg]``, template helicity and flow axes."""

    programs = cast(Sequence[Mapping[str, object]], export["programs"])
    assignment = next(
        (
            index
            for index, entry in enumerate(runtime.flavour_union_assignments)
            if index and entry["process_id"] == member and "relabelled" not in entry
        ),
        None,
    )
    if assignment is None or not programs:
        raise EvaluationError(f"{member!r} is not an assignment of this runtime")
    points = [
        [[float(component) for component in leg] for leg in point]
        for point in cast(Sequence[Sequence[Sequence[float]]], requested_points)
    ]
    crossing = cast(Sequence[Mapping[str, object]], programs[0]["crossing"])
    template_points = []
    for point in points:
        crossed: list[list[float]] = [[] for _ in point]
        for entry in crossing:
            sign = float(cast(float, entry["sign"]))
            crossed[cast(int, entry["program_leg"])] = [
                sign * value for value in point[cast(int, entry["caller_leg"])]
            ]
        template_points.append(crossed)
    # Two copies of a single point select batched arithmetic, as above.
    batch = template_points if len(template_points) > 1 else template_points * 2
    rows = runtime.evaluate_assignments(
        batch, [[assignment]] * len(batch), output="helicity-flow"
    )
    colors = {
        str(color.id): index for index, color in enumerate(runtime.physics.color_flows)
    }
    with_references = []
    for program in programs:
        outputs = cast(Sequence[Mapping[str, object]], program["helicity_outputs"])
        try:
            color = colors[str(program["representative_colour_id"])]
        except KeyError as exc:
            raise EvaluationError(
                "recurrence export axes do not match the assignment evaluation"
            ) from exc
        values = [
            [
                complex(cast(Sequence[Sequence[object]], rows[point][0])[k][color]).real
                for k in range(len(outputs))
            ]
            for point in range(len(points))
        ]
        with_references.append(
            {
                **program,
                "reference_values": {
                    "momenta": points,
                    "helicity_ids": [str(output["id"]) for output in outputs],
                    "values": values,
                },
            }
        )
    return {**export, "programs": with_references}
