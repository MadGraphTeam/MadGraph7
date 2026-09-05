# SPDX-License-Identifier: 0BSD
"""Reconstruct ME normalization in the arithmetic requested by the caller."""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import lru_cache

from pyamplicol.api.errors import ArtifactError


@lru_cache(maxsize=8)
def _pi(precision: int) -> Decimal:
    from symbolica import E

    # Do not construct an optimized evaluator of the constant Pi: that
    # construction requires a target precision in current Symbolica.
    return Decimal(E("pi").evaluate({}, decimal_digit_precision=precision)[0])


def exact_normalization(
    physics: Mapping[str, object],
    parameters: Sequence[Decimal],
    precision: int,
    parameter_schema: Sequence[Mapping[str, object]] | None = None,
) -> Decimal:
    """Use exact combinatorial factors and current, not default, couplings.

    The native cached scalar is deliberately not upcast: independent rounding
    of the Born and real-emission normalizations spoils local subtraction.
    This function is called only by the non-f64 executors.
    """
    extensions = physics.get("extensions", {})
    if not isinstance(extensions, Mapping):
        raise ArtifactError("exact normalization has invalid physics extensions")
    normalization = extensions.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ArtifactError("exact normalization metadata is absent")
    # Full/NLC contractions already include their colour metric. Only LC
    # needs the common colour factor, exactly as in the native executor.
    color = (
        Decimal(str(normalization.get("color_factor", 1)))
        if physics.get("color_accuracy") == "lc"
        else Decimal(1)
    )
    average = Decimal(str(normalization.get("average_factor", 1)))
    identical = Decimal(str(normalization.get("identical_factor", 1)))
    if average <= 0 or identical <= 0:
        raise ArtifactError("exact normalization has a nonpositive averaging factor")
    qcd_power = int(normalization.get("qcd_coupling_power", 0))
    ew_power = int(normalization.get("electroweak_coupling_power", 0))
    coupling = Decimal(1)
    if qcd_power or ew_power:
        records = parameter_schema or physics.get("model_parameters", ())
        if not isinstance(records, Sequence):
            raise ArtifactError("exact normalization parameter schema is invalid")
        by_name = {
            str(record["name"]): parameters[int(record.get("parameter_index", index))]
            for index, record in enumerate(records)
            if isinstance(record, Mapping) and "name" in record
        }
        try:
            pi = _pi(precision)
            if qcd_power:
                coupling *= (
                    4 * pi * by_name["normalization.alpha_s_me_check"]
                ) ** qcd_power
            if ew_power:
                coupling *= (8 * pi * by_name["normalization.alpha_ew"]) ** ew_power
        except KeyError as exc:
            raise ArtifactError(
                "exact normalization lacks its current coupling parameter"
            ) from exc
    else:
        coupling = Decimal(str(normalization.get("global_coupling_factor", 1)))
    return color * coupling / (average * identical)
