# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models import compiler_contact_trees as contact_trees
from pyamplicol.models import compiler_contacts as contacts
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contacts import _execute_dense_tensor
from pyamplicol.models.compiler_kernels import (
    _spin_axis_labels,
    _spin_representations,
    _spin_slots,
)
from pyamplicol.models.contracts import CompiledParticleRecord, CompiledVertexTerm


def _particle(name: str, pdg: int, *, spin: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
        spin=spin,
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


def _vector(name: str, pdg: int) -> CompiledParticleRecord:
    return _particle(name, pdg, spin=3)


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


def test_staged_four_point_partial_preserves_dense_axis_order() -> None:
    _sym._ensure_symbolica()
    registry = symbols.model("staged-contact-partial-test")
    particles = tuple(_vector(f"v{index}", 9_200_000 + index) for index in range(4))
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
    source_name = _sym.TensorName(registry.qualified_name("test_contact_partial"))
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

    input_legs = ((0, "left"), (1, "right"))
    direct_expression, direct_library = source_tensor()
    for leg, side in input_legs:
        direct_expression *= contacts._input_tensor_expression(
            direct_library,
            kind=23,
            side=side,
            spin=particles[leg].spin,
            leg=leg + 1,
            components=contacts._component_symbols(
                23,
                side,
                particles[leg].spin,
                model_symbols=registry,
            ),
            model_symbols=registry,
        )
    direct = _execute_dense_tensor(
        direct_expression,
        direct_library,
        axis_labels=tuple(
            label
            for leg in (2, 3)
            for label in _spin_axis_labels(particles[leg].spin, leg + 1)
        ),
    )

    staged_expression, staged_library = source_tensor()
    staged = contacts._execute_contact_partial_staged(
        staged_expression,
        staged_library,
        particles,
        input_legs=input_legs,
        open_legs=(2, 3),
        kind=23,
        model_symbols=registry,
    )

    assert tuple(item.to_canonical_string() for item in staged) == tuple(
        item.to_canonical_string() for item in direct
    )


def test_sliced_spin2_contact_partial_preserves_dense_axis_order() -> None:
    _sym._ensure_symbolica()
    registry = symbols.model("sliced-spin2-contact-partial-test")
    particles = (
        _particle("s0", 9_300_000, spin=1),
        _particle("s1", 9_300_001, spin=1),
        _particle("h0", 9_300_002, spin=5),
        _particle("h1", 9_300_003, spin=5),
    )
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
    source_name = _sym.TensorName(registry.qualified_name("spin2_contact_partial"))
    source_components = tuple(_sym.E(str(index + 1)) for index in range(16**2))

    def source_expression(library: object) -> object:
        library.register(
            _sym.LibraryTensor.dense(
                source_name(*representations),
                source_components,
            )
        )
        return source_name(*slots).to_expression()

    direct_library = _sym.TensorLibrary.hep_lib_atom()
    direct_expression = source_expression(direct_library)
    for leg, side in ((0, "left"), (1, "right")):
        direct_expression *= contacts._input_tensor_expression(
            direct_library,
            kind=29,
            side=side,
            spin=particles[leg].spin,
            leg=leg + 1,
            components=contacts._component_symbols(
                29,
                side,
                particles[leg].spin,
                model_symbols=registry,
            ),
            model_symbols=registry,
        )
    direct = _execute_dense_tensor(
        direct_expression,
        direct_library,
        axis_labels=tuple(
            label
            for leg in (2, 3)
            for label in _spin_axis_labels(particles[leg].spin, leg + 1)
        ),
    )

    sliced_library = _sym.TensorLibrary.hep_lib_atom()
    sliced = contacts._execute_contact_partial_sliced(
        source_expression(sliced_library),
        sliced_library,
        particles,
        input_legs=((0, "left"), (1, "right")),
        open_legs=(2, 3),
        kind=29,
        model_symbols=registry,
    )

    assert tuple(item.to_canonical_string() for item in sliced) == tuple(
        item.to_canonical_string() for item in direct
    )


def test_sliced_spin2_contact_bounds_symbolic_inputs_and_preserves_mixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sym._ensure_symbolica()
    registry = symbols.model("bounded-spin2-contact-partial-test")
    particles = (
        _particle("scalar_input", 9_350_000, spin=1),
        _particle("spin2_input", 9_350_001, spin=5),
        _vector("vector_output", 9_350_002),
        _particle("spin2_output", 9_350_003, spin=5),
    )
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
    source_name = _sym.TensorName(registry.qualified_name("bounded_spin2_contact"))
    source_components = tuple(
        _sym.E(str(index % 19 - 9)) for index in range(16 * 4 * 16)
    )

    def source_expression(library: object) -> object:
        library.register(
            _sym.LibraryTensor.dense(
                source_name(*representations),
                source_components,
            )
        )
        return source_name(*slots).to_expression()

    input_legs = ((0, "left"), (1, "right"))
    direct_library = _sym.TensorLibrary.hep_lib_atom()
    direct_expression = source_expression(direct_library)
    for leg, side in input_legs:
        direct_expression *= contacts._input_tensor_expression(
            direct_library,
            kind=31,
            side=side,
            spin=particles[leg].spin,
            leg=leg + 1,
            components=contacts._component_symbols(
                31,
                side,
                particles[leg].spin,
                model_symbols=registry,
            ),
            model_symbols=registry,
        )
    direct = _execute_dense_tensor(
        direct_expression,
        direct_library,
        axis_labels=tuple(
            label
            for leg in (2, 3)
            for label in _spin_axis_labels(particles[leg].spin, leg + 1)
        ),
    )

    original_input_tensor = contacts._input_tensor_expression
    spin2_component_counts: list[int] = []

    def recording_input_tensor(*args: object, **kwargs: object) -> object:
        if kwargs.get("spin") == 5:
            components = kwargs["components"]
            assert isinstance(components, tuple)
            spin2_component_counts.append(
                sum(component.to_canonical_string() != "0" for component in components)
            )
        return original_input_tensor(*args, **kwargs)

    monkeypatch.setattr(contacts, "_input_tensor_expression", recording_input_tensor)
    sliced_library = _sym.TensorLibrary.hep_lib_atom()
    sliced = contacts._execute_contact_partial_sliced(
        source_expression(sliced_library),
        sliced_library,
        particles,
        input_legs=input_legs,
        open_legs=(2, 3),
        kind=31,
        model_symbols=registry,
    )

    assert spin2_component_counts
    assert set(spin2_component_counts) == {1}
    assert len(sliced) == len(direct) == 4 * 16
    for actual, expected in zip(sliced, direct, strict=True):
        difference = (
            _sym.E(actual.to_canonical_string())
            - _sym.E(expected.to_canonical_string())
        ).expand()
        assert difference.to_canonical_string() == "0"


def test_linux_single_spin2_output_uses_sliced_contact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sym._ensure_symbolica()
    particles = (
        _particle("s0", 9_400_000, spin=1),
        _particle("s1", 9_400_001, spin=1),
        _particle("s2", 9_400_002, spin=1),
        _particle("h0", 9_400_003, spin=5),
    )
    particle_by_name = {particle.name: particle for particle in particles}
    term = CompiledVertexTerm(
        id=1,
        vertex="single-spin2-contact",
        particles=tuple(particle.name for particle in particles),
        color_index=0,
        lorentz_index=0,
        color_source="1",
        color_expression="1",
        lorentz_name="L_single_spin2",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC_single_spin2",
        coupling_expression="1",
        coupling_orders=(),
    )
    calls: list[tuple[int, ...]] = []

    def sliced(
        _expression: object,
        _library: object,
        _particles: object,
        *,
        input_legs: object,
        open_legs: tuple[int, ...],
        kind: int,
        model_symbols: object,
    ) -> tuple[object, ...]:
        del input_legs, kind, model_symbols
        calls.append(open_legs)
        return tuple(_sym.E(str(index + 1)) for index in range(16))

    def unexpected_dense(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Linux spin-2 outputs must use the sliced path")

    monkeypatch.setattr(contacts, "_HOST_PLATFORM", "linux")
    monkeypatch.setattr(contacts, "_execute_contact_partial_sliced", sliced)
    monkeypatch.setattr(contacts, "_execute_dense_tensor", unexpected_dense)

    result = contacts._contact_partial_component_expressions(
        term,
        particle_by_name,
        left_leg=0,
        right_leg=1,
        open_legs=(2, 3),
        kind=41,
        model_symbols=symbols.model("single-spin2-contact-test"),
    )

    assert calls == [(2, 3)]
    assert result == tuple(str(index + 1) for index in range(16))
