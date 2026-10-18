# SPDX-License-Identifier: 0BSD
"""Output-only transport from the native model-rule verification boundary.

Evaluation, reduction and sums have already run in native batches. This adapter
restores the requested point order and formats each returned coefficient once;
it performs no numerical evaluation or implicit sum over coupling sectors.
"""

from __future__ import annotations

from decimal import Decimal

from pyamplicol.api.virtual_results import VirtualEvaluation, VirtualLaneDiagnostic
from pyamplicol.config import R2Strategy

from .stability import NativeStableModelResult, NumericalVerificationError
from .strategy import R2Selection


def stable_model_evaluation(
    result: NativeStableModelResult,
    *,
    r2: R2Selection,
    allow_partial: bool = False,
) -> VirtualEvaluation:
    """Format ordinary model-R2 results without losing orders or precision.

    The output point axis follows ``result.requested_point_ids``, including
    ``None`` for failed points only when explicitly requested. The native
    diagnostic remains the authority for numerical verification; successful
    reduction alone never becomes an accuracy claim. This internal adapter does
    not admit additional physical processes or enable public loop generation.
    """
    if not isinstance(result, NativeStableModelResult):
        raise TypeError("expected a complete native model-rule verification result")
    if not isinstance(r2, R2Selection) or r2.effective != R2Strategy.MODEL:
        raise ValueError("native model-rule results require the model R2 strategy")
    requested, present = result.requested_point_ids, result.point_ids
    if len(set(requested)) != len(requested) or len(set(present)) != len(present):
        raise ValueError("native result has duplicate point identities")
    if not set(present) <= set(requested):
        raise ValueError("native result contains an unrequested point")
    if tuple(item.point_id for item in result.diagnostics) != requested:
        raise ValueError("native diagnostics do not match requested point identities")
    born, virtual = result.born, result.virtual
    born_width, virtual_width = len(result.orders.born), len(result.orders.virtual)
    if (
        tuple(born.row_ids) != present
        or tuple(virtual.row_ids) != present
        or born.columns != born_width
        or virtual.columns != 3 * virtual_width
    ):
        raise ValueError("native result buffers do not match their point/order layout")
    # Native compatibility distinguishes arithmetic even when AP happens to use
    # exactly the same bit count as double or DoubleFloat. Check empty views so
    # there is no copy of the numerical packet just to validate metadata.
    empty = born.take_rows([])
    checks = [empty, virtual.take_rows([])]
    if result.diagnostics:
        output = result.diagnostics[0].output_arithmetic
        if any(item.output_arithmetic != output for item in result.diagnostics):
            raise ValueError(
                "one native call must have one requested output arithmetic"
            )
        checks.append(type(born).from_decimal([], 0, 0, **output.options))
    type(born).concat_columns(checks)

    present_set = set(present)
    for item in result.diagnostics:
        if item.status not in {"success", "rescued", "failed"}:
            raise ValueError("unknown native model result status")
        if (item.status != "failed") != (item.point_id in present_set):
            raise ValueError("native status and available point values disagree")
        if item.accuracy not in {
            "not-established",
            "empirical-cross-precision-agreement",
        }:
            raise ValueError("unknown native numerical verification evidence")
        if item.status == "failed" and item.accuracy != "not-established":
            raise ValueError("failed native point cannot establish accuracy")
    if not allow_partial and len(present) != len(requested):
        raise NumericalVerificationError(result)

    def real_values(batch):
        values = []
        for real, imaginary in batch.to_decimal():
            # Check before binary64 conversion: a tiny unwanted imaginary part
            # must not be silently rounded to zero at the output boundary.
            if Decimal(imaginary) != 0:
                raise ValueError("ordinary Born-loop interference must be real")
            values.append(Decimal(real))
        return values

    born_values, virtual_values = real_values(born), real_values(virtual)
    rows = {point: row for row, point in enumerate(present)}
    output_born, output_virtual, diagnostics = [], [], []
    for item in result.diagnostics:
        detail = [item.accuracy]
        if item.evaluated_arithmetic is not None:
            spec = item.evaluated_arithmetic
            detail.append(
                f"evaluated in {spec.arithmetic}"
                + (f" at {spec.precision} decimal digits" if spec.precision else "")
            )
        if item.detail:
            detail.append(item.detail)
        diagnostics.append(
            VirtualLaneDiagnostic(
                item.status,
                item.output_arithmetic.arithmetic,
                born.precision_bits,
                item.accuracy == "empirical-cross-precision-agreement",
                "; ".join(detail),
            )
        )
        if item.point_id not in rows:
            output_born.append(None)
            output_virtual.append(None)
            continue
        row = rows[item.point_id]
        convert = (
            float if item.output_arithmetic.arithmetic == "double" else lambda x: x
        )
        output_born.append(
            tuple(
                convert(value)
                for value in born_values[row * born_width : (row + 1) * born_width]
            )
        )
        start = row * 3 * virtual_width
        output_virtual.append(
            tuple(
                tuple(
                    convert(virtual_values[start + 3 * sector + power])
                    for power in range(3)
                )
                for sector in range(virtual_width)
            )
        )
    return VirtualEvaluation(
        tuple(output_born),
        tuple(output_virtual),
        tuple(sector.order for sector in result.orders.born),
        result.orders.virtual,
        tuple(diagnostics),
        r2,
    )
