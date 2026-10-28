# SPDX-License-Identifier: 0BSD
"""The model compiler's kernel-equivalence keys and proofs run natively
(``rusticol_core::model_proofs`` through ``model_kernel_equivalence``; WP3.2
EWI-3R), with the Python stage as the oracle
(``compiler_entry.native_kernel_proofs(False)``,
``compiler_entry._kernel_keys_and_proofs``); and the compile's memoised
stages rely on the kind independence of the kernel networks.

* loop_sm (R2 and UV vertices, four-point contacts, complex couplings whose
  canonical strings hold the imaginary unit) compiles to the same IR with the
  native and the Python keys and proofs, on one worker and on several;
* a kernel whose oriented component Symbolica's ``expand`` does not bring to
  its normal form in one pass (a sum to a half-integer power times its square
  root, as the G_mu target's CT contact finals hold) gets the oracle's proof
  natively, so the native lane expands the signed component again as the
  oracle does (review of EWI-3R: a native lane that skipped it changed the
  class of 14 of the target's 13,403 kernels);
* the oriented components and the contact partials of a structure formed for
  one kernel kind, renamed to another kind, equal those formed for that kind
  directly (``compiler_entry._compile_oriented_kernels`` and
  ``compiler_contacts._contact_partial_component_expressions`` reuse them),
  and so do the contact finals (``_contact_final_component_expressions``);
* the kind renamed on the canonical text (``_renamed_kernel_text``) equals
  Symbolica's renaming, and loop_sm compiles to the same IR with the
  compile's text renamings, fused-component memo (``_fused_components``) and
  contact-partial product memo (``_canonical_product_text``) as with
  Symbolica's renaming and every fusion and product formed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def loop_sm_payload(loop_sm_ufo, tmp_path_factory):
    from pyamplicol.models.loading import compile_model_source

    model = compile_model_source(
        loop_sm_ufo, cache_dir=tmp_path_factory.mktemp("kernel-equivalence-cache")
    )
    return model


def _compiled_ir(model, *, native: bool) -> dict:
    from pyamplicol.models import compiler_entry

    compiler_entry._KERNEL_PROOFS.clear()
    with compiler_entry.native_kernel_proofs(native):
        ir = compiler_entry.compile_ufo_model_ir(
            model.model,
            tree_kernel_variant=model.ir.tree_kernel_variant,
            gauge=model.ir.gauge,
        )
    return ir.to_dict()


def test_native_kernel_equivalence_equals_the_python_oracle(
    loop_sm_payload, monkeypatch
):
    oracle = _compiled_ir(loop_sm_payload, native=False)
    kernels = oracle["oriented_kernels"]
    assert len({kernel["evaluation_class"] for kernel in kernels}) < len(kernels)
    assert any(kernel["evaluation_input_exchange_factor"] for kernel in kernels)
    for threads in ("1", "4"):
        monkeypatch.setenv("RAYON_NUM_THREADS", threads)
        assert _compiled_ir(loop_sm_payload, native=True) == oracle, threads


def test_native_proofs_expand_the_signed_components_again(monkeypatch):
    from types import SimpleNamespace

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.models import compiler_entry
    from pyamplicol.models import compiler_symbolica as _sym

    _sym._ensure_symbolica()
    model_symbols = symbols.model("kernel_proof_expansion")
    kind = 5

    def leg(side: str, index: int) -> str:
        return model_symbols.kernel_component(kind, side, index).to_canonical_string()

    a, g = (model_symbols.expression(name).to_canonical_string() for name in "ag")
    factored = f"((1-{a})^(3/2)+1/2)*(1-{a})^(1/2)"
    once = _sym.E(factored).expand()
    expanded = (1 * once).expand().to_canonical_string()
    # One expansion leaves (1-a)^2 unexpanded: the case stays discriminating.
    assert once.to_canonical_string() != expanded
    kernels = [
        SimpleNamespace(
            kind=kind,
            coupling_expression=g,
            lc_color_normalization_power=0,
            component_expressions=(
                f"{sign}({factor})*{leg('left', 0)}*{leg('right', 1)}",
                f"{sign}{leg('left', 1)}*{leg('right', 0)}",
            ),
        )
        # The same polynomial factored and expanded, with both signs: the
        # least signature of one of each sign pair has the sign 1.
        for sign in ("", "-")
        for factor in (factored, expanded)
    ]
    monkeypatch.setattr(compiler_entry, "_KERNEL_PROOFS", {})
    keys, proofs = compiler_entry._kernel_keys_and_proofs(
        kernels, {}, lambda kernel: [], lambda kernel: (4, 4, 4), model_symbols
    )
    assert len(set(keys)) == 4
    # Equal polynomials up to a sign: one class, both signs.
    assert len({proof[0] for proof in proofs.values()}) == 1
    assert {proof[1] for proof in proofs.values()} == {(1.0, 0.0), (-1.0, 0.0)}
    prefix = model_symbols.kernel_component(0, "left", 0).get_name()
    for threads in ("1", "4"):
        monkeypatch.setenv("RAYON_NUM_THREADS", threads)
        native_keys, found = load_loop_native().model_kernel_equivalence(
            [
                (
                    kind,
                    kernel.coupling_expression,
                    0,
                    list(kernel.component_expressions),
                    [],
                    (4, 4, 4),
                )
                for kernel in kernels
            ],
            prefix.rsplit("::", 1)[0] + "::",
            set(),
        )
        assert native_keys == keys
        assert {
            key: (
                digest,
                (float(sign), 0.0),
                tuple(order),
                None if exchange is None else (float(exchange), 0.0),
            )
            for key, (digest, sign, order, exchange) in found
        } == proofs, threads


def test_kernel_networks_do_not_depend_on_the_kind(loop_sm_payload):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_symbolica as _sym
    from pyamplicol.models.compiler_contacts import (
        _formed_contact_partial_component_expressions,
    )
    from pyamplicol.models.compiler_kernels import (
        _oriented_component_expressions,
        _remap_kernel_symbols,
        _renamed_kernel_text,
    )

    _sym._ensure_symbolica()
    ir = loop_sm_payload.ir
    model_symbols = symbols.model(ir.name)
    particle_by_name = {particle.name: particle for particle in ir.particles}

    def renamed(values, old, new):
        symbolica = tuple(
            _remap_kernel_symbols(
                _sym.E(value), old_kind=old, new_kind=new, model_symbols=model_symbols
            ).to_canonical_string()
            for value in values
        )
        text = tuple(
            _renamed_kernel_text(
                value, old_kind=old, new_kind=new, model_symbols=model_symbols
            )
            for value in values
        )
        assert text == symbolica
        return symbolica

    oriented, partials = set(), set()
    for term in ir.vertex_terms:
        spins = tuple(particle_by_name[name].spin for name in term.particles)
        if term.valence == 3:
            for result_leg in range(3):
                left_leg, right_leg = (leg for leg in range(3) if leg != result_leg)
                key = (term.lorentz_expression, spins, left_leg, right_leg, result_leg)
                if key in oriented:
                    continue
                oriented.add(key)
                formed = {
                    kind: _oriented_component_expressions(
                        term,
                        particle_by_name,
                        left_leg=left_leg,
                        right_leg=right_leg,
                        result_leg=result_leg,
                        kind=kind,
                        model_symbols=model_symbols,
                    )
                    for kind in (3, 120)
                }
                assert formed[3].ordering == formed[120].ordering
                assert renamed(formed[3].values, 3, 120) == formed[120].values, key
        elif term.valence == 4:
            key = (term.lorentz_expression, spins)
            if key in partials:
                continue
            partials.add(key)
            options = {"left_leg": 0, "right_leg": 1, "open_legs": (2, 3)}
            formed = {
                kind: _formed_contact_partial_component_expressions(
                    term,
                    particle_by_name,
                    kind=kind,
                    model_symbols=model_symbols,
                    **options,
                )
                for kind in (9, 41)
            }
            assert renamed(formed[9], 9, 41) == formed[41], key
    assert len(oriented) > 20 and len(partials) > 3


def test_contact_finals_do_not_depend_on_the_kind(loop_sm_payload):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_contacts
    from pyamplicol.models import compiler_symbolica as _sym
    from pyamplicol.models.compiler_entry import _compile_four_point_contact_kernels
    from pyamplicol.models.compiler_kernels import _remap_kernel_symbols

    _sym._ensure_symbolica()
    ir = loop_sm_payload.ir
    model_symbols = symbols.model(ir.name)

    def compiled(start_kind):
        # Each structure's first final is formed for its own kind.
        compiler_contacts._FINALS.clear()
        compiler_contacts._PARTIALS.clear()
        return _compile_four_point_contact_kernels(
            ir.vertex_terms,
            ir.particles,
            start_kind=start_kind,
            model_symbols=model_symbols,
        )[1]

    first, second = compiled(1000), compiled(5000)
    finals = [k for k in first if k.vertex.endswith("::contact-final")]
    assert len(first) == len(second) and len(finals) > 3
    for a, b in zip(first, second, strict=True):
        assert b.kind == a.kind + 4000
        assert b.component_expressions == tuple(
            _remap_kernel_symbols(
                _sym.E(value),
                old_kind=a.kind,
                new_kind=b.kind,
                model_symbols=model_symbols,
            ).to_canonical_string()
            for value in a.component_expressions
        ), a.vertex


def test_text_renamings_and_memos_equal_the_formed_compile(
    loop_sm_payload, monkeypatch
):
    from pyamplicol.models import (
        compiler_contacts,
        compiler_entry,
        compiler_kernels,
    )
    from pyamplicol.models import compiler_symbolica as _sym

    def memos_cleared():
        compiler_kernels._FUSED.clear()
        compiler_kernels._CANONICAL_COMPONENTS.clear()
        compiler_kernels._CANONICAL_PRODUCTS.clear()
        compiler_contacts._FINALS.clear()
        compiler_contacts._PARTIALS.clear()

    original = compiler_kernels._renamed_fused_texts
    renamed = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        renamed.append(result is not None)
        return result

    memos_cleared()
    with monkeypatch.context() as patch:
        patch.setattr(compiler_kernels, "_renamed_fused_texts", counted)
        fast = _compiled_ir(loop_sm_payload, native=True)
    assert sum(renamed) > 10 and compiler_kernels._CANONICAL_PRODUCTS

    def product_formed(text, factor, *, kind, model_symbols):
        return compiler_kernels._canonicalize_oriented_kernel_component(
            _sym.E(text) * _sym.E(factor)
        ).to_canonical_string()

    def symbolica_renamed(text, *, old_kind, new_kind, model_symbols):
        if old_kind == new_kind:
            return text
        return compiler_kernels._remap_kernel_symbols(
            _sym.E(text),
            old_kind=old_kind,
            new_kind=new_kind,
            model_symbols=model_symbols,
        ).to_canonical_string()

    memos_cleared()
    with monkeypatch.context() as patch:
        patch.setattr(compiler_kernels, "_renamed_fused_texts", lambda *a, **k: None)
        patch.setattr(compiler_entry, "_canonical_product_text", product_formed)
        for module in (compiler_kernels, compiler_contacts, compiler_entry):
            patch.setattr(module, "_renamed_kernel_text", symbolica_renamed)
        formed = _compiled_ir(loop_sm_payload, native=True)
    memos_cleared()
    assert formed == fast


def test_contact_partial_products_are_renamed_from_the_formed_kind(loop_sm_payload):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_kernels
    from pyamplicol.models import compiler_symbolica as _sym
    from pyamplicol.models.compiler_contacts import (
        _formed_contact_partial_component_expressions,
    )

    _sym._ensure_symbolica()
    ir = loop_sm_payload.ir
    model_symbols = symbols.model(ir.name)
    particle_by_name = {particle.name: particle for particle in ir.particles}
    options = {"left_leg": 0, "right_leg": 1, "open_legs": (2, 3)}
    checked = 0
    for term in ir.vertex_terms:
        if term.valence != 4 or "ufo_momentum_" in term.lorentz_expression:
            continue
        formed = {
            kind: _formed_contact_partial_component_expressions(
                term,
                particle_by_name,
                kind=kind,
                model_symbols=model_symbols,
                **options,
            )
            for kind in (9, 41)
        }
        for index, text in enumerate(formed[9]):
            if text == "0":
                continue
            compiler_kernels._CANONICAL_PRODUCTS.clear()
            first = compiler_kernels._canonical_product_text(
                text, "-1/2", kind=9, model_symbols=model_symbols
            )
            assert len(compiler_kernels._CANONICAL_PRODUCTS) == 1
            renamed = compiler_kernels._canonical_product_text(
                formed[41][index], "-1/2", kind=41, model_symbols=model_symbols
            )
            assert len(compiler_kernels._CANONICAL_PRODUCTS) == 1  # a memo hit
            assert (
                first
                == compiler_kernels._canonicalize_oriented_kernel_component(
                    _sym.E(text) * _sym.E("-1/2")
                ).to_canonical_string()
            )
            assert (
                renamed
                == compiler_kernels._canonicalize_oriented_kernel_component(
                    _sym.E(formed[41][index]) * _sym.E("-1/2")
                ).to_canonical_string()
            )
            checked += 1
    compiler_kernels._CANONICAL_PRODUCTS.clear()
    assert checked > 3
