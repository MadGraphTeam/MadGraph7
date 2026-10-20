# SPDX-License-Identifier: 0BSD
"""One-pass exact sums give the projectors of sequential Symbolica addition."""

from __future__ import annotations

import pytest

pytest.importorskip("symbolica")

_PROJECTOR_MODULES = (
    "physical",
    "physical_process",
    "dual_weights",
    "physical_counterterms",
    "prepared_correlations",
)


def _sequential(terms):
    """The replaced accumulation: one Python ``+`` per term onto the total."""
    from symbolica import E

    return sum(terms, E("0"))


@pytest.mark.loop_fast
def test_symbolic_sum_equals_sequential_addition():
    from symbolica import E, S

    from pyamplicol.loops.symbolic_sums import symbolic_sum

    a, b, c = S(
        "loop_symbolic_sum_test::a",
        "loop_symbolic_sum_test::b",
        "loop_symbolic_sum_test::c",
    )
    assert symbolic_sum([]) == E("0")
    assert symbolic_sum(iter([a])) == a
    assert symbolic_sum([a, -a, b - b]) == E("0")
    assert symbolic_sum([2, E("1/3"), a]) == E("7/3") + a
    symbols = [S(f"loop_symbolic_sum_test::x{i}") for i in range(23)]
    terms = [
        E(f"{i % 5 + 1}/{i % 3 + 2} + {i % 4 - 2}*1i")
        * symbols[i % len(symbols)].conj()
        * symbols[(7 * i) % len(symbols)]
        for i in range(2000)
    ]
    # Nested sums, conjugate pairs, exact cancellations and repeated monomials.
    terms += [value + value.conj() for value in terms[:50]]
    terms += [-value for value in terms[100:300]]
    terms += [a * (b + c), -(a * b), E("0"), (a + b) ** 2]
    assert symbolic_sum(terms) == _sequential(terms)
    assert symbolic_sum(terms).to_canonical_string() == (
        _sequential(terms).to_canonical_string()
    )


def _states(monkeypatch, build, summation):
    """Capture every exact projector expression saved while compiling."""
    from importlib import import_module

    from pyamplicol.loops import physical

    original, captured = physical._saved, []

    def saved(expressions, parameters):
        expressions, parameters = list(expressions), list(parameters)
        captured.append((expressions, parameters))
        return original(expressions, parameters)

    with monkeypatch.context() as patch:
        for name in _PROJECTOR_MODULES:
            module = import_module(f"pyamplicol.loops.{name}")
            patch.setattr(module, "_saved", saved, raising=False)
            patch.setattr(module, "symbolic_sum", summation)
        build()
    return captured


@pytest.fixture(scope="module")
def loop_sm(loop_sm_ufo):
    from pyamplicol.models.loop_import import load_raw_loop_model

    return load_raw_loop_model(loop_sm_ufo)


def _source_program(backend, raw, outgoing, born, loop, *, directed=False):
    from pyamplicol.loops.compiler import prepare_loop_request_model

    def build():
        prepared = prepare_loop_request_model(
            raw,
            born_order_bounds={"QCD": born, "QED": 0},
            loop_order_bounds={"QCD": loop, "QED": 0},
            fixed_parameters={"MB": "0"},
        )
        program = prepared.compile(
            (2, -2),
            outgoing,
            selected_corrections=["QCD"],
            backend=backend,
            parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        )
        if directed:
            from pyamplicol.loops.physical_model_correlations import (
                NativeDirectedModelProgram,
            )

            from .test_loop_physical_correlations import _histories

            NativeDirectedModelProgram(program, _histories(program.virtual.bare))

    return build


def _gluon_program(raw):
    def build():
        from pyamplicol.loops.model import build_loop_model
        from pyamplicol.loops.physical_counterterms import (
            gluon_scattering_model_virtual_program,
        )

        gluon_scattering_model_virtual_program(build_loop_model(raw))

    return build


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "case",
    ["uu~>dd~ directed", "uu~>dd~g", "gg>gg model R2"],
)
def test_projectors_equal_sequential_accumulation(
    loop_native, loop_sm, monkeypatch, case
):
    """Born, dual-weight, family, UV/R2/field and directed projectors agree.

    The reference re-runs the same compilation with every collected target
    added term by term, as the replaced ``sum``/``+=`` code did; the outputs
    must be identical canonical expressions, not merely numerically close.
    """
    from pyamplicol.loops.symbolic_sums import symbolic_sum

    build = {
        "uu~>dd~ directed": _source_program(
            loop_native, loop_sm, (1, -1), 2, 4, directed=True
        ),
        "uu~>dd~g": _source_program(loop_native, loop_sm, (1, -1, 21), 3, 5),
        "gg>gg model R2": _gluon_program(loop_sm),
    }[case]
    calls = []

    def counted(terms):
        terms = tuple(terms)
        calls.append(len(terms))
        return symbolic_sum(terms)

    reference = _states(monkeypatch, build, _sequential)
    actual = _states(monkeypatch, build, counted)
    assert calls and max(calls) > 1
    assert len(actual) == len(reference)
    for (outputs, parameters), (expected, expected_parameters) in zip(
        actual, reference, strict=True
    ):
        assert parameters == expected_parameters
        assert len(outputs) == len(expected)
        for value, target in zip(outputs, expected, strict=True):
            assert value == target
