# SPDX-License-Identifier: 0BSD
"""Loop kernel catalogue of the v2 engine against independent UFO lowerings.

Oracles: the UFO Lorentz structure contracted numerically with spenso, the
tree engine's own full-variant oriented kernels or, for the quartic vertices
it has no kernel for, the raw UFO Metric structures evaluated by hand, and the
model-declared UFO propagator numerators (not the tree propagator rules the
catalogue uses). Segments are checked at several random exact points.
"""

from __future__ import annotations

import random
from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


@pytest.fixture(scope="module")
def catalogue(compiled):
    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue

    return build_loop_kernel_catalogue(compiled)


@pytest.fixture(scope="module")
def full_model(compiled):
    from pyamplicol.loops.v2.catalogue import compile_tree_kernel_variant
    from pyamplicol.models.external import CompiledUFOModel

    return CompiledUFOModel(compile_tree_kernel_variant(compiled, "full"))


class ExactPoint:
    """Random exact complex-rational inputs shared by kernels and oracles."""

    def __init__(self, compiled, seed: int) -> None:
        from symbolica import E

        self.random = random.Random(seed)
        self.E = E
        # Parameters with a vanishing default lower as massless/zero in the
        # tree engine, so they stay zero here as well.
        self.parameters = {
            name: E("0") if complex(*value) == 0 else self.rational(positive=True)
            for name, value in compiled.parameter_defaults.items()
        }

    def rational(self, *, positive: bool = False):
        numerator = self.random.randint(1 if positive else -40, 40)
        return self.E(f"{numerator or 1}/{self.random.randint(1, 13)}")

    def complex(self):
        from symbolica import Expression

        return self.rational() + self.rational() * Expression.I

    def vector(self, length: int, *, real: bool = False):
        return tuple(self.rational() if real else self.complex() for _ in range(length))


def _decimal(value: Fraction) -> str:
    # Symbolica derives the working precision from each input's digits.
    return format(Decimal(value.numerator) / Decimal(value.denominator), ".70e")


def _substitute(expression, values):
    from symbolica import E, Replacement

    return (
        E(expression)
        .replace_multiple([Replacement(E(symbol), value) for symbol, value in values])
        .expand()
    )


def _zero(expression) -> bool:
    return expression.expand().to_canonical_string() == "0"


def _monomial(q, exponents):
    from symbolica import E

    value = E("1")
    for component, power in zip(q, exponents, strict=True):
        value *= component**power
    return value


def _kernel_inputs(kernel, point, *, shift=(), blocks=(), block_momenta=(), loop=()):
    values = []
    for item in kernel.inputs:
        if item.role == "model-parameter":
            values.append((item.symbol, point.parameters[item.model_parameter_name]))
        else:
            source = {
                "loop-momentum": shift,
                "block-current": blocks,
                "block-momentum": block_momenta,
                "loop-current": loop,
            }[item.role]
            values.append((item.symbol, source[item.component]))
    return values


def _apply_kernel(kernel, entries, dimension, values, q, loop, block_values=()):
    """Sum entries of a fused/split segment or numerator applied to ``loop``."""

    from symbolica import E

    coefficients = [_substitute(item, values) for item in kernel.exact_expressions]
    outputs = [E("0")] * dimension
    for entry, coefficient in zip(entries, coefficients, strict=True):
        term = coefficient * _monomial(q, entry.exponents) * loop[entry.loop_in]
        for block, component in zip(block_values, entry.blocks, strict=True):
            term *= block[component]
        outputs[entry.out] += term
    return [value.expand() for value in outputs]


def _ufo_axes(spin: int, leg: int):
    """(representation, index) of a UFO leg, named as in the lowered expressions.

    Kept local so that the oracle shares only the slot names of the lowered
    expression with the implementation, not its axis bookkeeping. Scalars and
    ghosts (UFO spin 1, -1) have none; tensor lines (spin 5) are refused.
    """

    from pyamplicol.models import compiler_symbolica as sym

    if spin == 2:
        return ((sym.Representation.bis(4), f"ufo_s_1_{leg}"),)
    if spin == 3:
        return ((sym.Representation.mink(4), f"ufo_l_1_{leg}"),)
    if spin in (-1, 1):
        return ()
    raise ValueError(f"no oracle slots for UFO spin {spin}")


def _ufo_contraction(compiled, lorentz, spins, currents, momenta, open_leg):
    """Numerically contract a normalized UFO Lorentz tensor with spenso."""

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models import compiler_symbolica as sym
    from pyamplicol.models.tensors import levi_civita_tensor

    model_symbols = symbols.model(compiled.name)
    library = sym.TensorLibrary.hep_lib_atom()
    library.register(levi_civita_tensor())
    names = [
        sym.TensorName(model_symbols.ufo_momentum_tensor_name(leg + 1))
        for leg in range(len(spins))
    ]
    expression = sym.E(lorentz)
    for leg, values in currents.items():
        axes = _ufo_axes(spins[leg], leg + 1)
        if not axes:
            expression *= values[0]
            continue
        name = sym.TensorName(f"loop_v2_oracle::current_{leg}")
        library.register(
            sym.Tensor.dense(name(*(rep for rep, _ in axes)), list(values))
        )
        expression *= name(*(rep(index) for rep, index in axes)).to_expression()
    for leg, values in momenta.items():
        library.register(
            sym.Tensor.dense(names[leg](sym.Representation.mink(4)), list(values))
        )
    # At most one open axis, so the dense components are in index order.
    open_axes = () if open_leg is None else _ufo_axes(spins[open_leg], open_leg + 1)
    network = sym.TensorNetwork(
        sym.as_tensor(expression) if open_axes else expression, library
    )
    network.execute(library=library)
    result = network.result_tensor(library).to_dense()
    assert len(result) == (4 if open_axes else 1)
    return [(sym.E("0") + result[index]).expand() for index in range(len(result))]


def _ufo_numerator(compiled, model, pdg, value, momentum, point):
    """The model-declared UFO propagator numerator applied to ``value``."""

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models.tensors import normalize_lorentz_expression

    record = model._propagator_record(pdg)
    spin = model.particle(pdg).spin
    model_symbols = symbols.model(compiled.name)
    normalized = normalize_lorentz_expression(
        record.numerator, (spin, spin), model_symbols=model_symbols, vertex=False
    )
    numerator = model_symbols.expression(normalized.expression)
    for record in compiled.ir.parameters:
        if record.nature == "internal":
            numerator = numerator.replace(
                model_symbols.symbol(record.name),
                model_symbols.expression(record.resolved_expression),
            )
    for name, parameter in point.parameters.items():
        numerator = numerator.replace(model_symbols.symbol(name), parameter)
    return _ufo_contraction(
        compiled,
        numerator.to_canonical_string(),
        (spin, spin),
        {0: value},
        {0: momentum, 1: momentum},
        open_leg=1,
    )


def _coupling(compiled, term, point):
    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols

    value = E(term.coupling_expression)
    for name, parameter in point.parameters.items():
        value = value.replace(
            symbols.runtime_model_parameter(compiled.name, name), parameter
        )
    power = term.lc_color_normalization_power
    return value * E(f"2^(-{power}/2)") if power else value


def _particle(compiled, name):
    return next(particle for particle in compiled.ir.particles if particle.name == name)


def _ufo_lorentz_structures(ufo) -> dict[str, str]:
    """Raw UFO Lorentz structures by name, parsed from lorentz.py, not imported."""

    import ast

    structures = {}
    for node in ast.walk(ast.parse((ufo / "lorentz.py").read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Lorentz":
            fields = {keyword.arg: keyword.value for keyword in node.keywords}
            name = ast.literal_eval(fields["name"])
            structures[name] = ast.literal_eval(fields["structure"])
    return structures


class _NotMetric(Exception):
    pass


def _metric_vertex(structure: str, vectors, open_leg: int):
    """A pure-Metric UFO structure applied to explicit vectors, by hand.

    The second oracle for vertices the tree engine has no kernel for (the
    quartic V_4): Metric(a, b) contracts two closed legs with diag(1,-1,-1,-1)
    and passes the partner's component through on the open leg. Legs are
    1-based UFO positions; returns None for anything but Metric products.
    """

    import ast

    from symbolica import E

    signature = (1, -1, -1, -1)

    def evaluate(node, component):
        match node:
            case ast.BinOp(op=ast.Add()):
                return evaluate(node.left, component) + evaluate(node.right, component)
            case ast.BinOp(op=ast.Sub()):
                return evaluate(node.left, component) - evaluate(node.right, component)
            case ast.BinOp(op=ast.Mult()):
                return evaluate(node.left, component) * evaluate(node.right, component)
            case ast.UnaryOp(op=ast.USub()):
                return -evaluate(node.operand, component)
            case ast.Constant(value=int() as value):
                return E(str(value))
            case ast.Call(
                func=ast.Name(id="Metric"),
                args=[ast.Constant(value=int() as a), ast.Constant(value=int() as b)],
            ):
                if open_leg in (a, b):
                    return vectors[b if a == open_leg else a][component]
                pairs = zip(signature, vectors[a], vectors[b], strict=True)
                return sum((s * x * y for s, x, y in pairs), E("0"))
        raise _NotMetric

    try:
        tree = ast.parse(structure, mode="eval").body
        return [evaluate(tree, component).expand() for component in range(4)]
    except _NotMetric:
        return None


def _segment_point(model, binding, point):
    shift = point.vector(4, real=True)
    q = point.vector(4, real=True)
    loop = point.vector(binding.loop_in_dimension)
    blocks = [point.vector(dimension) for dimension in binding.block_dimensions]
    momenta = [point.vector(4, real=True) for _ in binding.attached_legs]
    embedded = [
        tuple(
            value if not isinstance(value, float) else point.E("0")
            for value in model._embed_weyl_current(species, chirality, block)
        )
        for species, chirality, block in zip(
            binding.block_species,
            binding.key.block_chiralities,
            blocks,
            strict=True,
        )
    ]
    return shift, q, loop, blocks, momenta, embedded


@pytest.mark.parametrize("seed", (7, 101, 202))
def test_segments_equal_ufo_vertex_times_propagator_numerator(
    compiled, catalogue, full_model, loop_sm_ufo, seed
):
    """Every segment equals UFO vertex x UFO propagator numerator, exactly.

    The second oracle is the tree engine's full-variant kernel where one
    exists, else (quartic vertices) the raw UFO Metric structure by hand.
    """

    point = ExactPoint(compiled, seed=seed)
    model = full_model.with_runtime_parameters(point.parameters)
    terms = {term.id: term for term in compiled.ir.vertex_terms}
    structures = _ufo_lorentz_structures(loop_sm_ufo)
    tree_kernels = {
        (kernel.term_ids, kernel.source_particle_legs): kernel
        for kernel in full_model.compiled.ir.oriented_kernels
    }
    checked_against_tree = checked_by_hand = 0
    for binding in catalogue.segments:
        term = terms[binding.key.term_id]
        kernel = catalogue.kernels[binding.kernel_id]
        shift, q, loop, blocks, momenta, embedded = _segment_point(
            model, binding, point
        )
        values = _kernel_inputs(
            kernel,
            point,
            shift=shift,
            blocks=[value for block in blocks for value in block],
            block_momenta=[value for momentum in momenta for value in momentum],
        )
        split = binding.key.form == "split"
        actual = _apply_kernel(
            kernel,
            binding.entries,
            binding.loop_out_dimension,
            values,
            q,
            loop,
            blocks if split else (),
        )

        loop_momentum = [q[index] + shift[index] for index in range(4)]
        out_momentum = [
            loop_momentum[index] + sum(momentum[index] for momentum in momenta)
            for index in range(4)
        ]
        currents = {binding.key.loop_in_leg: loop}
        incoming = {binding.key.loop_in_leg: loop_momentum}
        for leg, current, momentum in zip(
            binding.attached_legs, embedded, momenta, strict=True
        ):
            currents[leg] = current
            incoming[leg] = momentum
        incoming[binding.key.loop_out_leg] = [-value for value in out_momentum]
        spins = tuple(_particle(compiled, name).spin for name in term.particles)
        coupling = _coupling(compiled, term, point)
        vertex = [
            coupling * value
            for value in _ufo_contraction(
                compiled,
                term.lorentz_expression,
                spins,
                currents,
                incoming,
                binding.key.loop_out_leg,
            )
        ]
        expected = _ufo_numerator(
            compiled, model, binding.loop_out_species, vertex, out_momentum, point
        )
        assert len(actual) == len(expected) == binding.loop_out_dimension
        assert all(
            _zero(left - right) for left, right in zip(actual, expected, strict=True)
        ), binding.key

        legs = (
            binding.key.loop_in_leg,
            *binding.attached_legs,
            binding.key.loop_out_leg,
        )
        tree = tree_kernels.get(((term.id,), legs))
        if tree is None:
            if set(spins) != {3}:
                continue
            by_hand = _metric_vertex(
                structures[term.lorentz_name],
                {leg + 1: current for leg, current in currents.items()},
                binding.key.loop_out_leg + 1,
            )
            if by_hand is None:
                continue
            expected = _ufo_numerator(
                compiled,
                model,
                binding.loop_out_species,
                [coupling * value for value in by_hand],
                out_momentum,
                point,
            )
            assert all(
                _zero(left - right)
                for left, right in zip(actual, expected, strict=True)
            ), binding.key
            checked_by_hand += 1
            continue
        # Tree couplings may be derived runtime parameters of the model.
        tree_parameters = {
            name: model._parameter_value(name) for name in tree.runtime_parameters
        }
        tree_vertex = model._projected_kernel_components(
            tree,
            loop,
            blocks[0],
            left_chirality=0,
            right_chirality=binding.key.block_chiralities[0],
            result_chirality=0,
            left_momentum=loop_momentum,
            right_momentum=momenta[0],
            runtime_parameter_values=tree_parameters,
        )
        tree_coupling = model._resolved_kernel_coupling_expression(
            tree, tree_parameters
        )
        expected = _ufo_numerator(
            compiled,
            model,
            binding.loop_out_species,
            [tree_coupling * value for value in tree_vertex],
            out_momentum,
            point,
        )
        assert all(
            _zero(left - right) for left, right in zip(actual, expected, strict=True)
        ), binding.key
        checked_against_tree += 1
    assert checked_against_tree > len(catalogue.segments) // 3
    assert checked_by_hand == sum(
        binding.vertex == "V_4" for binding in catalogue.segments
    )


def test_propagator_numerators_equal_the_ufo_declaration(compiled, catalogue):
    from pyamplicol.models.external import CompiledUFOModel

    point = ExactPoint(compiled, seed=11)
    model = CompiledUFOModel(compiled)
    for binding in catalogue.propagator_numerators:
        kernel = catalogue.kernels[binding.kernel_id]
        shift, q = point.vector(4, real=True), point.vector(4, real=True)
        loop = point.vector(binding.dimension)
        actual = _apply_kernel(
            kernel,
            binding.entries,
            binding.dimension,
            _kernel_inputs(kernel, point, shift=shift),
            q,
            loop,
        )
        momentum = [q[index] + shift[index] for index in range(4)]
        expected = _ufo_numerator(
            compiled, model, binding.species, loop, momentum, point
        )
        assert all(
            _zero(left - right) for left, right in zip(actual, expected, strict=True)
        ), binding.species


def test_counterterm_sites_equal_the_ufo_counterterm_vertex(compiled, catalogue):
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.tensors import normalize_lorentz_expression

    point = ExactPoint(compiled, seed=13)
    model = CompiledUFOModel(compiled)
    lorentz = {item["name"]: item for item in compiled.model["lorentz_structures"]}
    assert catalogue.counterterm_sites
    for binding in catalogue.counterterm_sites:
        kernel = catalogue.kernels[binding.kernel_id]
        blocks = [point.vector(dimension) for dimension in binding.block_dimensions]
        momenta = [point.vector(4, real=True) for _ in binding.block_species]
        (actual,) = (
            _substitute(expression, values)
            for expression in kernel.exact_expressions
            for values in [
                _kernel_inputs(
                    kernel,
                    point,
                    blocks=[value for block in blocks for value in block],
                    block_momenta=[value for momentum in momenta for value in momentum],
                )
            ]
        )
        spins = tuple(_particle(compiled, name).spin for name in binding.particles)
        currents = {
            leg: tuple(
                value if not isinstance(value, float) else point.E("0")
                for value in model._embed_weyl_current(species, chirality, block)
            )
            for leg, (species, chirality, block) in enumerate(
                zip(
                    binding.block_species,
                    binding.key.block_chiralities,
                    blocks,
                    strict=True,
                )
            )
        }
        normalized = normalize_lorentz_expression(
            lorentz[binding.lorentz_name]["structure"],
            spins,
            model_symbols=symbols.model(compiled.name),
        ).expression
        (expected,) = _ufo_contraction(
            compiled, normalized, spins, currents, dict(enumerate(momenta)), None
        )
        assert _zero(actual - expected), binding.key
    # The Levi-Civita R2 vertices with a Z have kernels (Epsilon with ALOHA's
    # components, ``models.tensors.LEVI_CIVITA``, ``test_loop_v2_levi_civita``);
    # nothing is unsupported.
    assert not catalogue.unsupported_counterterms
    assert {"V_GGZ", "V_GGGZ"} <= {
        binding.vertex
        for binding in catalogue.counterterm_sites
        if "Epsilon" in lorentz[binding.lorentz_name]["structure"]
    }


def test_loop_sm_catalogue_structure(compiled, catalogue):
    """Delta <= 1, both orientations, ghosts, massive Dirac lines and closures."""

    by_name = {particle.name: particle.pdg_code for particle in compiled.ir.particles}
    gluon, ghost, top = by_name["G"], by_name["gh"], by_name["t"]
    assert {gluon, ghost, -ghost, top, -top} <= set(catalogue.loop_species)
    assert max(binding.degree for binding in catalogue.segments) <= 1
    assert max(binding.degree for binding in catalogue.propagator_numerators) <= 1
    assert {binding.degree for binding in catalogue.segments} == {0, 1}

    pairs = {
        (binding.key.term_id, binding.key.loop_in_leg, binding.key.loop_out_leg)
        for binding in catalogue.segments
    }
    assert all((term, out, into) in pairs for term, into, out in pairs)
    forms = {}
    for binding in catalogue.segments:
        key = binding.key
        forms.setdefault(
            (key.term_id, key.loop_in_leg, key.loop_out_leg, key.block_chiralities),
            set(),
        ).add(key.form)
    assert set(map(frozenset, forms.values())) == {frozenset({"fused", "split"})}

    vertices = {binding.vertex for binding in catalogue.segments}
    assert {"V_3", "V_4", "V_Ggh", "V_51"} <= vertices
    ghost_segments = [
        binding
        for binding in catalogue.segments
        if binding.loop_in_species in (ghost, -ghost)
    ]
    assert ghost_segments and all(
        binding.block_species == (gluon,) for binding in ghost_segments
    )
    top_line = catalogue.propagator_numerator(top)
    assert (top_line.propagator_kind, top_line.mass_parameter) == (
        "dirac-fermion",
        "MT",
    )
    assert any(entry.degree == 0 for entry in top_line.entries)
    quartic = [binding for binding in catalogue.segments if binding.vertex == "V_4"]
    assert quartic and all(len(binding.attached_legs) == 2 for binding in quartic)

    # Closures are pure traces: the closed-loop -1 of ghosts and fermions is
    # the skeleton's epsilon_s (DESIGN.md §3.3), never part of the kernel.
    assert [binding.species for binding in catalogue.closures] == sorted(
        catalogue.loop_species
    )
    point = ExactPoint(compiled, seed=29)
    for binding in catalogue.closures:
        kernel = catalogue.kernels[binding.kernel_id]
        assert kernel.contract_kind == "loop-closure"
        dimension = binding.dimension
        matrix = point.vector(dimension * dimension)
        values = _kernel_inputs(kernel, point, loop=matrix)
        (value,) = kernel.exact_expressions
        trace = sum(
            (matrix[index * (dimension + 1)] for index in range(dimension)),
            point.E("0"),
        )
        assert _zero(_substitute(value, values) - trace)
    assert {
        binding.dimension
        for binding in catalogue.closures
        if binding.species in (gluon, ghost, top)
    } == {1, 4}


def test_tree_kernel_variants_differ_exactly_by_self_contractions(compiled, full_model):
    """Transverse and full agree on conserved inputs, else differ by p_i.J_i."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols

    transverse = {kernel.kind: kernel for kernel in compiled.ir.oriented_kernels}
    full = {kernel.kind: kernel for kernel in full_model.compiled.ir.oriented_kernels}
    assert transverse.keys() == full.keys()
    changed = [
        kind
        for kind in transverse
        if transverse[kind].component_expressions != full[kind].component_expressions
    ]
    assert [transverse[kind].particles for kind in changed] == [("G", "G", "G")]
    assert all(
        (transverse[kind].particles, transverse[kind].term_ids)
        == (full[kind].particles, full[kind].term_ids)
        for kind in transverse
    )

    model_symbols = symbols.model(compiled.name)
    for kind in changed:
        left = [model_symbols.kernel_component(kind, "left", i) for i in range(4)]
        right = [model_symbols.kernel_component(kind, "right", i) for i in range(4)]
        pl = [model_symbols.kernel_momentum(kind, "left", i) for i in range(4)]
        pr = [model_symbols.kernel_momentum(kind, "right", i) for i in range(4)]

        def dot(a, b):
            return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]

        difference = [
            (E(f) - E(t)).expand()
            for f, t in zip(
                full[kind].component_expressions,
                transverse[kind].component_expressions,
                strict=True,
            )
        ]
        assert not all(_zero(value) for value in difference)
        # The difference is a combination of (p_L.L) R and (p_R.R) L whose
        # coefficients (couplings) are free of currents and momenta.
        left_scale = difference[0].coefficient(pl[0] * left[0] * right[0])
        right_scale = difference[0].coefficient(pr[0] * right[0] * left[0])
        kinematic = {
            symbol.to_canonical_string() for symbol in (*left, *right, *pl, *pr)
        }
        for scale in (left_scale, right_scale):
            assert not _zero(scale)
            assert not kinematic & {
                symbol.to_canonical_string() for symbol in scale.get_all_symbols(False)
            }
        for component, value in enumerate(difference):
            expected = (
                left_scale * dot(pl, left) * right[component]
                + right_scale * dot(pr, right) * left[component]
            )
            assert _zero(value - expected)

        # Conserved inputs: project random currents onto p.J = 0.
        point = ExactPoint(compiled, seed=17)
        momenta = [point.vector(4, real=True) for _ in range(2)]
        currents = []
        for momentum in momenta:
            current = list(point.vector(4))
            scale = dot(momentum, current) / dot(momentum, momentum)
            currents.append(
                [c - scale * p for c, p in zip(current, momentum, strict=True)]
            )
        values = [
            (symbol.to_canonical_string(), value)
            for symbols_, data in (
                (left, currents[0]),
                (right, currents[1]),
                (pl, momenta[0]),
                (pr, momenta[1]),
            )
            for symbol, value in zip(symbols_, data, strict=True)
        ]
        for f, t in zip(
            full[kind].component_expressions,
            transverse[kind].component_expressions,
            strict=True,
        ):
            assert _zero(_substitute(f, values) - _substitute(t, values))


def test_prepared_tree_catalog_carries_its_variant(compiled, full_model):
    from pyamplicol.loops.v2.catalogue import (
        compile_tree_kernel_variant,
        prepared_tree_kernel_catalog,
    )
    from pyamplicol.models.contracts import CompiledModelIR
    from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog

    assert compile_tree_kernel_variant(compiled, "transverse") is compiled
    full_ir = full_model.compiled.ir
    assert (compiled.ir.tree_kernel_variant, full_ir.tree_kernel_variant) == (
        "transverse",
        "full",
    )
    # The variant is part of the IR's serialized identity; the transverse form
    # is unchanged from before the flag existed.
    assert "tree_kernel_variant" not in compiled.ir.to_dict()
    assert CompiledModelIR.from_dict(full_ir.to_dict()) == full_ir
    assert compile_tree_kernel_variant(full_model.compiled, "full") is (
        full_model.compiled
    )
    assert compile_tree_kernel_variant(full_model.compiled, "transverse").ir == (
        compiled.ir
    )

    transverse = prepared_tree_kernel_catalog(compiled, "transverse")
    full = prepared_tree_kernel_catalog(compiled, "full")
    assert (transverse.tree_kernel_variant, full.tree_kernel_variant) == (
        "transverse",
        "full",
    )
    # Derived from the IR, not stamped by the loop helper.
    assert build_prepared_kernel_catalog(full_model).tree_kernel_variant == "full"
    assert "tree_kernel_variant" not in transverse.resolver_manifest()
    assert full.resolver_manifest()["tree_kernel_variant"] == "full"
    assert [binding.key for binding in transverse.vertex_bindings] == [
        binding.key for binding in full.vertex_bindings
    ]
    changed = {
        binding.key.particles
        for left, binding in zip(
            transverse.vertex_bindings, full.vertex_bindings, strict=True
        )
        if transverse.kernels[left.kernel_id].exact_expressions
        != full.kernels[binding.kernel_id].exact_expressions
    }
    assert changed == {(21, 21, 21)}


def test_symjit_and_exact_evaluators(compiled, catalogue):
    """Binary64 SymJIT, arbitrary precision and the exact saved state."""

    import numpy as np
    from symbolica import E, Evaluator, Expression

    from pyamplicol.loops.v2.catalogue import compile_loop_kernel

    getcontext().prec = 80
    point = ExactPoint(compiled, seed=19)

    def fraction() -> Fraction:
        return Fraction(point.random.randint(-40, 40), point.random.randint(1, 13))

    by_name = {particle.name: particle.pdg_code for particle in compiled.ir.particles}
    gluon_segment = next(
        binding
        for binding in catalogue.segments
        if binding.vertex == "V_3" and binding.key.form == "fused"
    )
    selected = {
        "gluon segment": catalogue.kernels[gluon_segment.kernel_id],
        "top numerator": catalogue.kernels[
            catalogue.propagator_numerator(by_name["t"]).kernel_id
        ],
        "counterterm site": catalogue.kernels[catalogue.counterterm_sites[0].kernel_id],
    }
    for label, kernel in selected.items():
        evaluator = compile_loop_kernel(kernel)
        restored = Evaluator.load(evaluator.exact_state())
        pairs = [
            [
                (
                    Fraction(
                        point.parameters[
                            item.model_parameter_name
                        ].to_canonical_string()
                    ),
                    Fraction(0),
                )
                if item.role == "model-parameter"
                else (fraction(), Fraction(0))
                if "momentum" in item.role
                else (fraction(), fraction())
                for item in kernel.inputs
            ]
            for _ in range(3)
        ]
        inputs = [
            [E(str(re)) + E(str(im)) * Expression.I for re, im in row] for row in pairs
        ]
        binary64 = evaluator.evaluate_complex(
            np.array(
                [[complex(value.evaluate({})) for value in row] for row in inputs],
                dtype=np.complex128,
            )
        )
        for row, values, exact_pairs in zip(binary64, inputs, pairs, strict=True):
            substitutions = [
                (item.symbol, value)
                for item, value in zip(kernel.inputs, values, strict=True)
            ]
            exact = [
                _substitute(expression, substitutions)
                for expression in kernel.exact_expressions
            ]
            reference = np.array([complex(value.evaluate({})) for value in exact])
            scale = max(1.0, float(np.max(np.abs(reference))))
            assert np.max(np.abs(row - reference)) <= 1e-14 * scale, label

            precise = [(_decimal(re), _decimal(im)) for re, im in exact_pairs]
            expected = [value.evaluate({}, 80).to_decimal_tuple() for value in exact]
            for result in (
                evaluator.evaluate_with_precision(precise, 60),
                restored.evaluate_complex_with_prec(precise, 60),
            ):
                for value, (re, im) in zip(result, expected, strict=True):
                    actual_re, actual_im = value.to_decimal_tuple()
                    assert abs(Decimal(actual_re) - Decimal(re)) <= Decimal("1e-50")
                    assert abs(Decimal(actual_im) - Decimal(im)) <= Decimal("1e-50")


def test_built_in_models_refuse_the_full_variant():
    """Built-in kernels are transverse only; ``full`` must not return them."""

    from pyamplicol.loops.v2.catalogue import (
        LoopKernelCatalogueError,
        compile_tree_kernel_variant,
    )
    from pyamplicol.models.loading import compile_model_source

    builtin = compile_model_source("built-in-sm", use_cache=False)
    assert builtin.ir.tree_kernel_variant == "transverse"
    assert compile_tree_kernel_variant(builtin, "transverse") is builtin
    with pytest.raises(LoopKernelCatalogueError, match="built-in"):
        compile_tree_kernel_variant(builtin, "full")


# ------------------------------------------- native polynomial stage (MS-GEN)


def _catalogue_lanes(compiled, **kwargs):
    """The catalogue with the segments' polynomial stage in Python (the
    oracle) and natively (``loop_catalogue_polynomials``)."""

    from pyamplicol.loops.v2.catalogue import (
        build_loop_kernel_catalogue,
        native_catalogue_polynomials,
    )

    with native_catalogue_polynomials(False):
        oracle = build_loop_kernel_catalogue(compiled, **kwargs)
    with native_catalogue_polynomials(True):
        native = build_loop_kernel_catalogue(compiled, **kwargs)
    return oracle, native


def _assert_identical(oracle, native):
    # Every kernel (canonical signature, exact outputs, inputs, layout) and
    # every binding with its entries, in order.
    assert native.kernels == oracle.kernels
    for kind in (
        "segments",
        "propagator_numerators",
        "closures",
        "counterterm_sites",
        "unsupported_counterterms",
    ):
        assert getattr(native, kind) == getattr(oracle, kind), kind
    assert [binding.entries for binding in native.segments] == [
        binding.entries for binding in oracle.segments
    ]
    assert native == oracle


def test_native_polynomial_stage_equals_the_python_oracle(
    compiled, catalogue, loop_native
):
    """The segments' polynomial stage in Rust (M-SM MS-GEN, HANDOFF §0.4)
    gives the Python stage's catalogue byte for byte on loop_sm: every
    kernel's exact outputs, inputs, layout and signature, every segment's
    entries; the default lane is the native one."""

    from pyamplicol.loops.v2 import catalogue as module

    assert module.NATIVE_CATALOGUE_POLYNOMIALS is True
    oracle, native = _catalogue_lanes(compiled)
    _assert_identical(oracle, native)
    assert catalogue == native
    assert oracle.segments and all(
        binding.key.form in ("fused", "split") for binding in oracle.segments
    )


@pytest.mark.loop_nightly
def test_native_polynomial_stage_equals_the_python_oracle_on_the_ew_ufo(
    loop_gmu_ufo, loop_native
):
    """The same on the G_mu UFO in 't Hooft-Feynman gauge with every
    propagating particle as a loop species (weak bosons, Goldstones, ghosts,
    every fermion: the catalogue of an EW virtual; about 2 min in the Python
    lane)."""

    from pyamplicol.models.loading import compile_model_source, with_model_gauge

    compiled = with_model_gauge(
        compile_model_source(loop_gmu_ufo, require_supported=False), "feynman"
    )
    species = tuple(
        particle.pdg_code
        for particle in compiled.ir.particles
        if particle.auxiliary_kind is None and particle.propagating
    )
    oracle, native = _catalogue_lanes(
        compiled, loop_species=species, counterterm_entries=()
    )
    _assert_identical(oracle, native)
    assert len(native.segments) > 1000


@pytest.mark.loop_nightly
def test_native_stages_give_the_oracle_plans_on_an_ew_rung(loop_gmu_ufo, loop_native):
    """g g > t t~ with QCD and QED corrections on the G_mu UFO (the EW
    ladder's first rung, ``tools/loop_v2_generate._fixture_request``)
    generated and exported with the native stages (catalogue polynomials,
    text canonical stages) and with their Python oracles: the same plan v4
    JSON and the same native tree programs, saved evaluators included
    (``snapshot.programs_digest``). About 5 min, 7 GB."""

    import argparse
    import json
    from pathlib import Path

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.catalogue import native_catalogue_polynomials
    from pyamplicol.loops.v2.plan_export import export_plan
    from pyamplicol.loops.v2.snapshot import programs_digest
    from pyamplicol.models.loading import compile_model_source
    from tools import loop_v2_generate

    root = Path(__file__).resolve().parents[2]
    reference = json.loads(
        (root / "tests/fixtures/loop/required/gg_ttx_ew.json").read_text()
    )
    compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
    request, _ = loop_v2_generate._fixture_request(
        reference, compiled, argparse.Namespace()
    )

    def lane(native: bool):
        with (
            native_catalogue_polynomials(native),
            assemble.text_canonical_stage(native),
        ):
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            exported = export_plan(slice_, forward=False, trees=True, born_squared=True)
        return exported.json(), programs_digest(exported.trees)

    oracle = lane(False)
    native = lane(True)
    assert native[1] == oracle[1]
    assert native[0] == oracle[0]
