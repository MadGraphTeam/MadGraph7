# SPDX-License-Identifier: 0BSD
"""Bind saved family bilinears without changing their algebra or column order.

The default remains the native dependency batch interpreter. The explicit JIT
option uses one node in the existing coefficient scheduler, including its row
tiling and shape/precision checks. It is only available for binary64.
"""


def validate_projector_engine(engine, arithmetic):
    if engine not in {"interpreted", "jit"}:
        raise ValueError("projector engine must be interpreted or jit")
    if engine == "jit" and arithmetic != "double":
        raise ValueError("projector JIT requires double arithmetic")
    return engine


def family_projector_shape(
    helicity_count, born_widths, loop_widths, graph_indices, sector_count, power_count
):
    """Recover packed uncorrelated widths from retained execution layouts."""
    return (
        helicity_count
        * (sum(born_widths) + sum(loop_widths[index] for index in graph_indices)),
        sector_count * power_count,
    )


def bind_family_projector(
    state, input_count, output_count, *, backend, options, engine="interpreted"
):
    """Return a fresh native workspace over exactly the supplied instructions."""
    validate_projector_engine(engine, options["arithmetic"])
    if any(
        type(width) is not int or width < 0 for width in (input_count, output_count)
    ):
        raise ValueError("projector widths must be nonnegative integers")
    if engine == "interpreted":
        return backend.CoefficientEvaluator(state, **options)
    return backend.CoefficientPlan(
        [state],
        [list(range(input_count))],
        input_count,
        list(range(input_count, input_count + output_count)),
        engine=engine,
        **options,
    )
