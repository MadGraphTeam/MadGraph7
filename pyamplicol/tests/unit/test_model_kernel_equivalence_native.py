# SPDX-License-Identifier: 0BSD
"""The model compiler's kernel-equivalence keys and proofs run natively
(``rusticol_core::model_proofs`` through ``model_kernel_equivalence``; WP3.2
EWI-3R), with the Python stage as the oracle
(``compiler_entry.native_kernel_proofs(False)``,
``compiler_entry._kernel_keys_and_proofs``); and the compile's memoised
stages rely on the kind independence of the kernel networks.

* loop_sm (R2 and UV vertices, four-point contacts, complex couplings whose
  canonical strings hold ``𝑖``) compiles to the same IR with the native and
  the Python keys and proofs, on one worker and on several;
* the oriented components and the contact partials of a structure formed for
  one kernel kind, renamed to another kind, equal those formed for that kind
  directly (``compiler_entry._compile_oriented_kernels`` and
  ``compiler_contacts._contact_partial_component_expressions`` reuse them).
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


def test_kernel_networks_do_not_depend_on_the_kind(loop_sm_payload):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_symbolica as _sym
    from pyamplicol.models.compiler_contacts import (
        _formed_contact_partial_component_expressions,
    )
    from pyamplicol.models.compiler_kernels import (
        _oriented_component_expressions,
        _remap_kernel_symbols,
    )

    _sym._ensure_symbolica()
    ir = loop_sm_payload.ir
    model_symbols = symbols.model(ir.name)
    particle_by_name = {particle.name: particle for particle in ir.particles}

    def renamed(values, old, new):
        return tuple(
            _remap_kernel_symbols(
                _sym.E(value), old_kind=old, new_kind=new, model_symbols=model_symbols
            ).to_canonical_string()
            for value in values
        )

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
