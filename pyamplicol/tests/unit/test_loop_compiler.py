# SPDX-License-Identifier: 0BSD
"""Generic source preparation against independent complete component controls."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol.loops.compiler import prepare_loop_request_model
from pyamplicol.models.loading import CompiledModel

pytestmark = pytest.mark.loop_physics


def _compiled(raw):
    # Only the existing raw sidecar and declared compile options belong to the
    # loop ingress. Deliberately incompatible normalized defaults must not leak.
    return CompiledModel(
        source={"options": {"simplify": True}},
        producer={},
        model={},
        ir=None,
        parameter_defaults={"aS": (1000.0, 0.0)},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0,
        raw_loop_model=raw,
    )


def test_compiler_requires_captured_source_not_normalized_tree_defaults():
    with pytest.raises(NotImplementedError, match="raw NLO sidecar"):
        prepare_loop_request_model(_compiled(None))
    with pytest.raises(TypeError, match="CompiledModel or RawLoopModel"):
        prepare_loop_request_model({"model": "arbitrary display name"})


@pytest.fixture(scope="module")
def prepared(loop_sm_ufo, loop_sm_compiled_model, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    path = str(loop_sm_ufo)
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(path)
    # The actual compiled tree IR supplies the source particle contracts; its
    # normalized aS default is made deliberately incompatible and must not leak.
    compiled = load_compiled_model(loop_sm_compiled_model)
    compiled = replace(
        compiled,
        raw_loop_model=raw,
        parameter_defaults={**compiled.parameter_defaults, "aS": (1000.0, 0.0)},
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )


def _exact_born_uv(point, scale, alpha_s, mass):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo as parse

    p = [[parse(value) for value in leg] for leg in point]

    def square(v):
        return v[0] ** 2 - sum((x * x for x in v[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    alpha = parse(alpha_s)
    born = E("4/9") * (4 * E("pi") * alpha) ** 2 * (t * t + u * u) / s**2
    finite = born * alpha / (3 * E("pi")) * (parse(scale) / parse(mass) ** 2).log()
    single = -7 * born * alpha / (2 * E("pi"))
    values = Expression.evaluator_multiple(
        [born, finite, single], [], n_cores=1, jit_compile=False
    ).evaluate_complex_with_prec([], 110)
    return [value.to_decimal_tuple()[0] for value in values]
