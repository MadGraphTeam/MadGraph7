# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_contact_trees as contact_trees
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contacts import _execute_dense_tensor
from pyamplicol.models.compiler_kernels import (
    _spin_axis_labels,
    _spin_representations,
    _spin_slots,
)
from pyamplicol.models.contracts import CompiledParticleRecord


def _vector(name: str, pdg: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
        spin=3,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def test_rank_zero_plain_expression_bypasses_tensor_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sym._ensure_symbolica()
    expression = _sym.E("contact_left*contact_right")
    library = _sym.TensorLibrary.hep_lib_atom()

    def unexpected_tensor_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plain scalar expression must not create a tensor network")

    monkeypatch.setattr(_sym, "TensorNetwork", unexpected_tensor_network)

    assert _execute_dense_tensor(expression, library, axis_labels=()) == (expression,)


def test_rank_zero_symbol_detection_is_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReorderedSymbols:
        def get_all_symbols(
            self, *, include_function_symbols: bool = True
        ) -> list[str]:
            return ["left", "right"] if include_function_symbols else ["right", "left"]

    expression = ReorderedSymbols()

    def unexpected_tensor_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbol order must not create a tensor network")

    monkeypatch.setattr(_sym, "TensorNetwork", unexpected_tensor_network)

    assert _execute_dense_tensor(expression, object(), axis_labels=()) == (expression,)


def test_staged_contact_contraction_preserves_dense_axis_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sym._ensure_symbolica()
    registry = symbols.model("staged-contact-test")
    particles = tuple(_vector(f"v{index}", 9_100_000 + index) for index in range(4))
    payload = {
        leg: (
            tuple(registry.symbol(f"input_{leg}_{index}") for index in range(4)),
            tuple(_sym.E("0") for _ in range(4)),
        )
        for leg in range(3)
    }
    representations = tuple(
        representation
        for particle in particles
        for representation in _spin_representations(particle.spin)
    )
    slots = tuple(
        slot
        for leg, particle in enumerate(particles, start=1)
        for slot in _spin_slots(particle.spin, leg)
    )
    source_name = _sym.TensorName(registry.qualified_name("test_contact"))
    source_components = tuple(_sym.E(str(index + 1)) for index in range(4**4))

    def source_tensor() -> tuple[object, object]:
        library = _sym.TensorLibrary.hep_lib_atom()
        library.register(
            _sym.LibraryTensor.dense(
                source_name(*representations),
                source_components,
            )
        )
        return source_name(*slots).to_expression(), library

    direct_expression, direct_library = source_tensor()
    for leg, (components, _momentum) in sorted(payload.items()):
        direct_expression *= contact_trees._contact_tree_physical_tensor_expression(
            direct_library,
            kind=17,
            leg=leg,
            spin=particles[leg].spin,
            components=components,
            model_symbols=registry,
        )
    direct = _execute_dense_tensor(
        direct_expression,
        direct_library,
        axis_labels=_spin_axis_labels(particles[3].spin, 4),
    )

    monkeypatch.setattr(contact_trees, "_MAX_STAGED_CONTACT_INPUT_WORK", 4)
    staged_expression, staged_library = source_tensor()
    staged = contact_trees._execute_contact_tensor_staged(
        staged_expression,
        staged_library,
        particles,
        payload_by_leg=payload,
        result_leg=3,
        kind=17,
        model_symbols=registry,
    )

    assert tuple(item.to_canonical_string() for item in staged) == tuple(
        item.to_canonical_string() for item in direct
    )
