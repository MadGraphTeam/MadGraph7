# SPDX-License-Identifier: 0BSD
"""Cold dependency templates share compilation, never mutable event workspace."""

import gc
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext

import pytest
from _loop_reduction import CoefficientKernelCache, CoefficientPlan, NativeBatch
from symbolica import Expression, S


def saved(expressions, parameters):
    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


@pytest.mark.parametrize(
    "arithmetic,precision,engine,exponent,tolerance",
    [
        ("double", None, "interpreted", 3, Decimal("1e-13")),
        ("double", None, "jit", 3, Decimal("1e-13")),
        ("double-double", None, "interpreted", 80, Decimal("1e-28")),
        ("arbitrary", 80, "interpreted", 180, Decimal("1e-74")),
    ],
)
def test_cold_cache_clones_independent_workspaces_and_retains_code_lifetime(
    arithmetic, precision, engine, exponent, tolerance
):
    x = S("kernel_cache_x")
    state = saved([x - 1, x * x], [x])
    cache = CoefficientKernelCache(arithmetic, precision, engine)

    def plan(second, cache=None):
        return CoefficientPlan(
            [state, state],
            [[0], [second]],
            1,
            [1, 2, 3, 4],
            arithmetic,
            precision,
            engine,
            kernel_cache=cache,
        )

    a, b = plan(2, cache), plan(1, cache)
    a_oracle, b_oracle = plan(2), plan(1)
    assert cache.entries == 1 and cache.hits == 3
    assert cache.settings == (
        "use_threads=false;use_simd=true"
        if engine == "jit"
        else "dependency-batch-interpreter"
    )
    reverse = saved([x * x, x - 1], [x])
    CoefficientPlan(
        [reverse], [[0]], 1, [1, 2], arithmetic, precision, engine, kernel_cache=cache
    )
    assert cache.entries == 2
    retained = []
    with localcontext() as context:
        context.prec = 160
        tiny = Decimal(2) ** -exponent
        for call, rows in enumerate((1, 33, 0, 65, 7, 129, 1)):
            ids = [9000 - 5 * i for i in range(rows)]
            inputs = NativeBatch.from_decimal(
                [[str(1 + (i + call + 1) * tiny), "0"] for i in range(rows)],
                rows,
                1,
                arithmetic,
                precision,
                ids,
            )
            for actual, oracle in ((a, a_oracle), (b, b_oracle)):
                value = actual.evaluate_batch(inputs)
                target = oracle.evaluate_batch(inputs)
                assert value.row_ids == target.row_ids == ids
                assert value.precision_bits == inputs.precision_bits
                for pair, expected in zip(
                    value.to_decimal(), target.to_decimal(), strict=True
                ):
                    assert all(
                        abs(Decimal(left) - Decimal(right))
                        <= tolerance * max(1, abs(Decimal(right)))
                        for left, right in zip(pair, expected, strict=True)
                    )
                if rows:
                    assert Decimal(value.to_decimal()[0][0]) != 0
                retained.append((value, value.to_decimal()))
        # Parallel calls exercise separately cloned evaluator scratch.
        inputs = NativeBatch.from_decimal(
            [["1.125", "0"]] * 33, 33, 1, arithmetic, precision
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            parallel = list(
                pool.map(lambda p: p.evaluate_batch(inputs).to_decimal(), (a, b))
            )
        assert parallel == [
            a_oracle.evaluate_batch(inputs).to_decimal(),
            b_oracle.evaluate_batch(inputs).to_decimal(),
        ]
    assert all(value.to_decimal() == snapshot for value, snapshot in retained)
    del cache
    gc.collect()
    assert (
        a.evaluate_batch(inputs).to_decimal()
        == a_oracle.evaluate_batch(inputs).to_decimal()
    )


def test_cache_rejects_carrier_precision_or_engine_aliasing():
    x = S("cache_options_x")
    state = saved([x], [x])
    for cache, arithmetic, precision, engine in (
        (CoefficientKernelCache(), "double-double", None, "interpreted"),
        (CoefficientKernelCache("arbitrary", 60), "arbitrary", 80, "interpreted"),
        (CoefficientKernelCache(), "double", None, "jit"),
    ):
        with pytest.raises(ValueError, match="cache arithmetic, precision or engine"):
            CoefficientPlan(
                [state],
                [[0]],
                1,
                [1],
                arithmetic,
                precision,
                engine,
                kernel_cache=cache,
            )
