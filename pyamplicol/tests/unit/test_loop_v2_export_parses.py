# SPDX-License-Identifier: 0BSD
"""The export's native parse stages against their Python oracles (M-FINAL
MF-RUST, HANDOFF §0.4).

* The exact kernels' term reading (``reference_executor.ExactKernel``):
  natively (``loop_kernel_terms``) and with ``_kernel_terms``
  (``native_kernel_terms(False)``), the same terms in the same order.
* The parameter program's canonical stage (``native_trees._parameter_stage``):
  renamed in Symbolica's binary export (``loop_rename_export``) and on text
  or by ``replace_multiple`` (``assemble._canonical_stage``), the same
  expressions and the same saved evaluator.

The plans of whole exports are compared on an electroweak rung in
``test_loop_v2_catalogue.py`` (nightly).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.loop_fast


# ------------------------------------------------------------- kernel terms


def _kernel_spec(expressions: list[str]) -> SimpleNamespace:
    from symbolica import E

    def symbol(name: str) -> str:
        return E(f"mfrust_kernel::{name}").to_canonical_string()

    inputs = [
        SimpleNamespace(symbol=symbol("x0"), role="current", model_parameter_name=None),
        SimpleNamespace(symbol=symbol("x1"), role="block", model_parameter_name=None),
        *(
            SimpleNamespace(
                symbol=symbol(name), role="model-parameter", model_parameter_name=value
            )
            for name, value in (("g", "G"), ("m", "M"), ("w", "W"))
        ),
    ]
    return SimpleNamespace(
        inputs=inputs,
        exact_expressions=tuple(
            E(text.replace("@", "mfrust_kernel::")).to_canonical_string()
            for text in expressions
        ),
    )


def _kernel_lanes(spec):
    """("read", ``ExactKernel(spec).outputs``) or ("refused", message of its
    ``NotImplementedError``) in the oracle and the native lane."""

    from pyamplicol.loops.v2 import reference_executor as rx

    def lane(native: bool):
        with rx.native_kernel_terms(native):
            try:
                return "read", rx.ExactKernel(spec).outputs
            except NotImplementedError as error:
                return "refused", str(error)

    return lane(False), lane(True)


# Every reading rule of ``_TermReading``: integer and negative powers of
# parameters and complex numbers, fractional powers of rationals (primes),
# products and powers a fractional power is distributed over (positive
# bases), pi, parameter-only sums and functions kept whole (with and without
# a fractional power), nested powers, constants without an input.
_READ = [
    "3/2*@g^2*@x0*@x1^2 - 1𝑖/3*@m*@x0 + 7",
    "(2*@m)^(1/2)*@x0 + 𝜋*@x1*@g^(-2) - 𝜋^2/6",
    "(@g+@m)^(-1/3)*@x0 + log(@m)^2*@x1 + (@g*@m)^(1/2)*@x0*@x1",
    "(12/7)^(3/2)*@x0 + (1+2𝑖)^(-2)*@x1 + 5/9",
    "((@g+@w)^(1/2)*@m)^(1/3)*@x0 + @x0*(@g*@m^2)^(-3/2)",
    "cos(@g)*exp(@m/@w)*@x0^3 + (1/3)^(-5/2)*@x1*@w^(7/2)",
]

# Refusals (or the readings Symbolica's normal form leaves): an input under
# a root, an unknown symbol, a symbolic exponent, an input inside a kept
# factor, fractional and complex powers of numbers.
_REFUSED = [
    "@x0^(1/2)",
    "@z*@x0",
    "@x0^@g",
    "(@x0+@g)^(-1)*@x1",
    "(-2)^(1/3)*@x0",
    "(1+1𝑖)^(1/2)*@x0",
    "@g^(1𝑖)*@x0",
]


def test_kernel_terms_read_natively_as_the_oracle(loop_native):
    """Every reading rule: the native term reading gives the oracle's terms
    (constant keys and monomials, in the canonical order); the native lane
    is the default."""

    from pyamplicol.loops.v2 import reference_executor as rx

    assert rx.NATIVE_KERNEL_TERMS is True
    oracle, native = _kernel_lanes(_kernel_spec(_READ))
    assert oracle[0] == "read"
    assert native == oracle
    keys = [key for terms in oracle[1] for key, _monomial in terms]
    kinds = {base[0] for key in keys for part in key[1:3] for base, _e in part}
    assert kinds == {"param", "prime", "pi", "expr"}
    assert any(key[3] for key in keys)  # a distributed (positive) base


@pytest.mark.parametrize("expression", _REFUSED)
def test_kernel_term_refusals_follow_the_oracle(loop_native, expression):
    """A kernel the oracle refuses (``NotImplementedError``) is refused
    natively; one it reads is read alike."""

    oracle, native = _kernel_lanes(_kernel_spec([expression]))
    assert native[0] == oracle[0]
    if oracle[0] == "read":
        assert native == oracle


def test_catalogue_kernels_read_natively_as_the_oracle(
    loop_sm_compiled_model, loop_native
):
    """Every kernel of loop_sm's loop catalogue reads natively as in the
    oracle."""

    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue
    from pyamplicol.models.loading import load_compiled_model

    catalogue = build_loop_kernel_catalogue(load_compiled_model(loop_sm_compiled_model))
    assert len(catalogue.kernels) > 10
    for kernel in catalogue.kernels:
        oracle, native = _kernel_lanes(kernel)
        assert native == oracle, kernel.kernel_id


@pytest.mark.loop_nightly
def test_ew_catalogue_kernels_read_natively_as_the_oracle(loop_gmu_ufo, loop_native):
    """The same on the G_mu UFO's catalogue in 't Hooft-Feynman gauge with
    every propagating particle as a loop species (an EW virtual's kernels;
    about 3 min)."""

    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue
    from pyamplicol.models.loading import compile_model_source, with_model_gauge

    compiled = with_model_gauge(
        compile_model_source(loop_gmu_ufo, require_supported=False), "feynman"
    )
    species = tuple(
        particle.pdg_code
        for particle in compiled.ir.particles
        if particle.auxiliary_kind is None and particle.propagating
    )
    catalogue = build_loop_kernel_catalogue(
        compiled, loop_species=species, counterterm_entries=()
    )
    assert len(catalogue.kernels) > 300
    for kernel in catalogue.kernels:
        oracle, native = _kernel_lanes(kernel)
        assert native == oracle, kernel.kernel_id


# --------------------------------------------------------- parameter stage


@functools.cache
def _stage_symbols():
    from symbolica import S

    from pyamplicol.loops.scalar_frontend import _producer_predicate

    return (
        [S(f"mfrust_stage::e{i}", is_real=i % 2 == 0) for i in range(13)],
        S("mfrust_stage::f", "mfrust_stage::g"),
        S("mfrust_stage::h", is_symmetric=True),
        S("mfrust_stage::equal", eval={"complex": _producer_predicate("equal")}),
        S("mfrust_stage::x", "mfrust_stage::y"),
        S("mfrust_plain_symbol"),
    )


def _stage(definitions: bool):
    """A stage like the parameter program's: positional inputs with
    prefix-sharing names (``e1``/``e12``), real inputs, nested and
    argument-only helper calls, a head with attributes and one with an
    evaluation callback (a scalar-library predicate; neither renamed), a
    default-namespace symbol and a namespaced one that is no input."""

    from symbolica import E, Expression

    inputs, (f, g), sym, equal, (x, y), plain = _stage_symbols()
    e1, e12 = inputs[1], inputs[12]
    outputs = [
        e12**2 - E("1/3") * e1 * inputs[4] + Expression.I * e12,
        equal(e1, e1) * plain + E("mfrust_stage::e123") / (e12 + 1),
        Expression.num(0),
        inputs[2].conj() * inputs[3].conj() + E("2^(1/2)") * inputs[0],
    ]
    if definitions:
        outputs += [
            f(e1, f(e12, E("2"))) * g(e12) + sym(e1, e12) * plain,
            g(f(g(f(e1, e1)), e12)) + sym(g(e1), x),
        ]
    functions = (
        [
            (f, (x, y), x * y + g(x)),
            (g, (x,), x**2 + e1 + equal(x, e12)),
            (sym, (x, y), x * y + 1),
        ]
        if definitions
        else []
    )
    return (
        SimpleNamespace(
            output_expressions=outputs,
            parameter_symbols=inputs,
            symbolica_functions=functions,
        ),
        [plain, E("mfrust_stage::e123"), x],
    )


def _evaluator(result, extra) -> bytes:
    from symbolica import Expression, FunctionDefinition

    canonical, parameters, functions = result
    return bytes(
        Expression.evaluator_multiple(
            canonical,
            [*parameters, *extra],
            functions=[
                FunctionDefinition(head, list(arguments), body, inlining="never")
                for head, arguments, body in functions
            ],
            jit_compile=False,
        ).save()
    )


@pytest.mark.parametrize("definitions", [True, False], ids=["helpers", "positional"])
def test_parameter_stage_renames_in_the_export_as_the_oracles(loop_native, definitions):
    """The parameter program's canonical stage renamed in Symbolica's binary
    export gives ``_canonical_stage``'s expressions (on text and by
    ``replace_multiple``, the oracles), parameters and definitions, and the
    same saved evaluator; the native lane is the default."""

    from pyamplicol.loops.v2 import assemble, native_trees

    assert native_trees.NATIVE_PARAMETER_STAGE is True
    stage, extra = _stage(definitions)
    oracles = []
    for text in (True, False):
        with assemble.text_canonical_stage(text):
            oracles.append(assemble._canonical_stage(stage))
    with native_trees.native_parameter_stage(False):
        assert native_trees._parameter_stage(stage)[0] == oracles[0][0]
    native = native_trees._parameter_stage(stage)
    for oracle in oracles:
        assert native[0] == oracle[0]
        assert [e.to_canonical_string() for e in native[0]] == [
            e.to_canonical_string() for e in oracle[0]
        ]
        assert native[1] == oracle[1]
        assert len(native[2]) == len(oracle[2])
        for (head, arguments, body), (head0, arguments0, body0) in zip(
            native[2], oracle[2], strict=True
        ):
            assert (head, arguments) == (head0, arguments0)
            assert body == body0
        assert _evaluator(native, extra) == _evaluator(oracle, extra)
    text = "".join(repr(e) for e in native[0])
    assert "mfrust_stage::e1" not in text.replace("mfrust_stage::e123", "")
    assert "mfrust_stage::equal(" in text
    if definitions:
        assert "mfrust_stage::h(" in text
