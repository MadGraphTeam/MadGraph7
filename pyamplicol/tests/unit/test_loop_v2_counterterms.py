# SPDX-License-Identifier: 0BSD
"""Counterterm sites and external fields of the v2 engine (design §6, WP1.7).

Oracle: the frozen legacy engine (O1, ``tests/fixtures/loop/legacy``), i.e. its
insertion multisets and its UV_local, R2 and external-field components in
binary64 and at 200 bits.

* Legacy multisets count CT trees per alternative. A site sums the trees of
  its blocks, so here a site weighs the number of Feynman trees in its blocks,
  counted by a field-level recursion over the model's vertex rules (oracle
  only; the production modules never count or list trees).
* The numerical oracle closes the v2 sites on the tree runtime's block rows at
  the fixture points and contracts them with a Born that the same site closure
  builds from the ordinary vertices rooted at leg 1, so CT and Born share one
  set of conventions; the Born itself is checked against O1.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any, ClassVar

import pytest

from pyamplicol.color.connections import ExactColorCoefficient
from pyamplicol.color.contraction_types import NC
from pyamplicol.loops.v2.ct_sites import (
    COMPONENT_BY_KIND,
    LAURENT_POWERS,
    Carrier,
    CertifiedZero,
    ColourOverlaps,
    CountertermCompletenessError,
    SiteClosure,
    add_colour,
    generate_counterterm_sites,
    ufo_colour_network,
)
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelLoopCatalogue,
    _set_partitions,
    add_orders,
    generate_families,
)
from pyamplicol.loops.v2.fields import (
    FIELD_COMPONENT,
    counterterm_scalar_plan,
    external_field_insertions,
)
from tools.loop_reference import legacy_capture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "legacy"
# Design §16.3: components against the legacy engine.
TOLERANCE = {"double": Decimal("1e-12"), "arbitrary-200": Decimal("1e-40")}
WORKING_DIGITS = {"double": 40, "arbitrary-200": 80}
COMPONENTS = ("uv_local", "r2", FIELD_COMPONENT)
ACCEPTANCE = ("gg_gg", "uux_ddxg", "uux_ggg", "dy", "uudd")
CONTROLS = (*ACCEPTANCE, "tt", "uux_uux", "uux_uuxg")
# legacy_capture process strings veto with MG5 aliases.
VETO_PDGS = {"z": 23}


# --------------------------------------------------------------------- slices


@dataclass
class CountertermSlice:
    """The v2 generation objects of one O1 control."""

    name: str
    compiled: Any
    prepared: Any
    data: dict
    blocks: Any
    table: Any
    selection: LoopOrderSelection
    families: Any
    vetoed: tuple[int, ...]
    _caches: dict = field(default_factory=dict)

    @property
    def control(self):
        return legacy_capture.CONTROLS[self.name]

    @property
    def fixture(self) -> dict:
        return json.loads((FIXTURES / f"{self.name}.json").read_text())

    @property
    def rules(self):
        return self.prepared.counterterms.rules

    def sites(self, rules=None, zeros=()):
        return generate_counterterm_sites(
            self.families,
            self.table,
            self.rules if rules is None else rules,
            self.data,
            self.selection,
            forbidden=self.vetoed,
            zeros=zeros,
        )

    def fields(self):
        from pyamplicol.loops.wavefunction import prepare_external_field_rules

        return external_field_insertions(
            self.table.legs,
            prepare_external_field_rules(self.prepared.counterterms.source),
            self.data,
            self.selection,
            self.blocks.born_orders,
            forbidden=self.vetoed,
        )


@pytest.fixture(scope="module")
def ct_slice(loop_sm_compiled_model):
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.blocks import build_tree_blocks
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, CountertermSlice] = {}

    def build(name: str) -> CountertermSlice:
        if name in built:
            return built[name]
        control = legacy_capture.CONTROLS[name]
        text, *vetoes = control.process.split(" / ")
        vetoed = tuple(VETO_PDGS[token] for veto in vetoes for token in veto.split())
        prepared = prepare_loop_request_model(
            compiled,
            born_order_bounds=control.born_order_bounds,
            loop_order_bounds=control.loop_order_bounds,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        )
        structure = prepared.structure
        data = json.loads(structure.native.to_json())
        loop = add_orders(tuple(control.loop_order_bounds.items()))
        selection = LoopOrderSelection(frozenset({"QCD"}), frozenset({loop}))
        blocks = build_tree_blocks(
            build_model_process_ir(text, compiled.ir, color_accuracy="full"),
            CompiledUFOModel(compiled),
            born_coupling_orders=control.born_order_bounds,
            block_coupling_orders=control.loop_order_bounds,
            ignored_particle_ids=vetoed,
        )
        table = blocks.block_table(data)
        catalogue = ModelLoopCatalogue(
            data, species_orders=structure.perturbative_species_orders
        )
        built[name] = CountertermSlice(
            name,
            compiled,
            prepared,
            data,
            blocks,
            table,
            selection,
            generate_families(table, catalogue, selection),
            vetoed,
        )
        return built[name]

    return build


# ------------------------------------------------------------------ tree counts


def _tree_counts(ct: CountertermSlice):
    """Feynman trees per block current (mask, incoming field, orders): oracle."""

    data = ct.data
    conjugate = {entry["name"]: entry["antiname"] for entry in data["particles"]}
    coupling_orders = {
        entry["name"]: add_orders(tuple(tuple(item) for item in entry["orders"]))
        for entry in data["couplings"]
    }
    merges: dict[tuple[str, ...], list[tuple[str, tuple]]] = defaultdict(list)
    arity = 0
    for rule in data["vertex_rules"]:
        names = rule["particles"]
        arity = max(arity, len(names))
        orders = {
            coupling_orders[name] for row in rule["couplings"] for name in row if name
        }
        for order in orders:
            pairs = {
                (
                    tuple(sorted(n for i, n in enumerate(names) if i != result)),
                    conjugate[names[result]],
                )
                for result in range(len(names))
            }
            for attached, produced in pairs:
                merges[attached].append((produced, order))
    legs = ct.table.legs

    @cache
    def count(mask: int, name: str, orders: tuple) -> int:
        if mask & (mask - 1) == 0:
            leg = legs[mask.bit_length() - 1]
            return int(leg.field == name and not orders)
        total = 0
        for parts in _set_partitions(mask, 2, arity - 1):
            for choice in product(*(ct.table.currents(part) for part in parts)):
                attached = tuple(sorted(block.field for block in choice))
                for produced, order in merges.get(attached, ()):
                    if produced != name:
                        continue
                    if add_orders(order, *(b.orders for b in choice)) != orders:
                        continue
                    weight = 1
                    for block in choice:
                        weight *= count(block.mask, block.field, block.orders)
                    total += weight
        return total

    return lambda block: count(block.mask, block.field, block.orders), count


def _legacy_multisets(ct: CountertermSlice, sites, fields) -> dict:
    weight, count = _tree_counts(ct)
    counterterms = Counter()
    for site in sites.sites:
        trees = 1
        for block in site.blocks:
            trees *= weight(block)
        counterterms[site.insertion_key] += trees * len(site.alternatives)
    # A field factor multiplies every Born tree of its order: the trees of the
    # legs other than leg 0 that join leg 0.
    legs = ct.table.legs
    conjugate = {entry["name"]: entry["antiname"] for entry in ct.data["particles"]}
    rest = (1 << len(legs)) - 2
    external = Counter()
    for insertion in fields:
        born = count(rest, conjugate[legs[0].field], insertion.born_order)
        external[insertion.insertion_key] += born
    return {
        "counterterms": sorted([key, n] for key, n in counterterms.items() if n),
        "external_fields": sorted([key, n] for key, n in external.items() if n),
    }


# --------------------------------------------------------------------- numerics


def _upcast(value: Decimal, digits: int) -> str:
    if value.is_zero():
        return f"0E-{digits}"
    return format(value, f".{digits - 1}E")


def _carrier(label: str) -> tuple[Carrier, Any]:
    """The arithmetic of a fixture carrier and a Decimal-pair converter."""

    from pyamplicol.loops.v2.catalogue import compile_loop_kernel

    evaluators: dict[int, Any] = {}

    def evaluator(spec):
        if spec.kernel_id not in evaluators:
            evaluators[spec.kernel_id] = compile_loop_kernel(spec)
        return evaluators[spec.kernel_id]

    if label == "double":
        import numpy as np

        def kernel(spec, values):
            rows = np.array([values], dtype=np.complex128)
            return list(evaluator(spec).evaluate_complex(rows)[0])

        def number(value: ExactColorCoefficient, power: int):
            return complex(value) * 2.0 ** (-power / 2)

        def convert(pair):
            return complex(float(pair[0]), float(pair[1]))

        return Carrier(number, kernel, 0j), convert

    digits = WORKING_DIGITS[label]

    def kernel(spec, values):
        inputs = [
            (_upcast(value.real, digits), _upcast(value.imag, digits))
            for value in values
        ]
        return [
            DecimalComplex(*map(Decimal, value.to_decimal_tuple()))
            for value in evaluator(spec).evaluate_with_precision(inputs, digits)
        ]

    sqrt2 = DecimalComplex.CONTEXT.sqrt(Decimal(2))

    def number(value: ExactColorCoefficient, power: int):
        context = DecimalComplex.CONTEXT
        scale = context.power(sqrt2, -power)
        return DecimalComplex(
            *(
                context.multiply(
                    context.divide(Decimal(part.numerator), Decimal(part.denominator)),
                    scale,
                )
                for part in (value.real, value.imag)
            )
        )

    def convert(pair):
        return DecimalComplex(Decimal(pair[0]), Decimal(pair[1]))

    return Carrier(number, kernel, DecimalComplex(Decimal(0), Decimal(0))), convert


@dataclass(frozen=True, slots=True)
class DecimalComplex:
    """Complex Decimal at a fixed working precision (the AP oracle carrier).

    Symbolica's ComplexFloat tracks accuracy and lowers its precision after
    cancellations, which the gauge cancellations of the Born trigger; the
    oracle needs a fixed working precision instead.
    """

    real: Decimal
    imag: Decimal
    CONTEXT: ClassVar[Context] = Context(prec=max(WORKING_DIGITS.values()) + 10)

    def __add__(self, other):
        other = _complex(other)
        c = self.CONTEXT
        return DecimalComplex(
            c.add(self.real, other.real), c.add(self.imag, other.imag)
        )

    __radd__ = __add__

    def __neg__(self):
        return DecimalComplex(self.real.copy_negate(), self.imag.copy_negate())

    def __mul__(self, other):
        other = _complex(other)
        c = self.CONTEXT
        return DecimalComplex(
            c.subtract(
                c.multiply(self.real, other.real), c.multiply(self.imag, other.imag)
            ),
            c.add(c.multiply(self.real, other.imag), c.multiply(self.imag, other.real)),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: int):
        c = self.CONTEXT
        return DecimalComplex(
            c.divide(self.real, Decimal(other)), c.divide(self.imag, Decimal(other))
        )

    def conjugate(self):
        return DecimalComplex(self.real, self.imag.copy_negate())


def _complex(value) -> DecimalComplex:
    if isinstance(value, DecimalComplex):
        return value
    return DecimalComplex(Decimal(value), Decimal(0))


@contextmanager
def _exact_u1_weights():
    """Evaluate tree rows with the exact U(1)-subtraction colour weight 1/N_c.

    The tree engine stores that interaction weight as binary64 1/3
    (``GenericDAG`` ``color_weight``), which limits multi-line block rows to
    ~1e-17 in any carrier (design §3.2 item 4: graded U(1) currents, WP2.4).
    The oracle restores the exact weight so that the 200-bit comparison tests
    the sites, not that rounding.
    """

    from symbolica import E

    from pyamplicol.generation import stage_expressions

    rounded = complex(1 / NC)
    original = stage_expressions._exact_complex_expression

    def exact(value):
        return E(f"1/{NC}") if value == rounded else original(value)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stage_expressions, "_exact_complex_expression", exact)
        yield


def _block_rows(ct: CountertermSlice, point, parameters, digits):
    """Tree-runtime rows {(current id, variant): (re, im) Decimals}, exactly.

    The interpreted stage evaluators of the numerical-current warm-up, with
    every input upcast to ``digits`` and the runtime-derived model values
    computed from their exact definitions (as the exact runtime does).
    """

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
    from pyamplicol.generation.contracts import StageCompilationInput
    from pyamplicol.generation.numerical_current_warmup import (
        _build_interpreted_stage_evaluator,
        _evaluate_interpreted_stage,
    )
    from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
    from pyamplicol.generation.stage_planning import (
        build_generic_stage_compiler_blueprint,
    )
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.runtime.symbolica_exact import (
        _fill_momenta,
        _fill_sources,
        _upcast_complex_inputs,
    )

    compiled, dag = ct.compiled, ct.blocks.blocks_dag
    model = CompiledUFOModel(compiled)
    runtime_schema = build_runtime_expression_schema(dag, model, process_id="ct")
    schema = runtime_schema.to_mapping()
    blueprint = build_generic_stage_compiler_blueprint(
        StageCompilationInput(dag, model, runtime_schema)
    )
    assert blueprint.expression_ready, blueprint.blockers
    settings = SymbolicaEvaluatorSettings(
        iterations=1,
        cpe_iterations=0,
        n_cores=1,
        direct_translation=False,
        jit_direct_translation=False,
        jit_optimization_level=0,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        compiled_chunk_compile_workers=1,
    )
    records = sorted(
        schema["model_parameters"], key=lambda record: record["parameter_index"]
    )
    external = {
        record["name"]: Decimal(parameters.get(record["name"], repr(record["default"])))
        for record in records
        if record["kind"] == "external_parameter"
    }
    model_symbols = symbols.model(compiled.name)
    substitutions = dict(
        zip(
            (E(model_symbols.symbol(name).to_canonical_string()) for name in external),
            _upcast_complex_inputs(
                [(value, Decimal(0)) for value in external.values()], digits
            ),
            strict=True,
        )
    )
    definitions = {
        record.name: record.resolved_expression
        for record in compiled.ir.parameters
        if record.nature == "internal"
    }
    definitions.update(
        {
            f"derived_coupling_{term.id}": term.coupling_expression
            for term in compiled.ir.vertex_terms
        }
    )
    derived, values = {}, []
    for record in records:
        if record["kind"] == "external_parameter":
            values.append(external[record["name"]])
            continue
        name = record["runtime_name"]
        if name not in derived:
            value = E(definitions[name]).evaluate(
                substitutions, decimal_digit_precision=digits
            )
            derived[name] = tuple(Decimal(x) for x in value.to_decimal_tuple())
        values.append(derived[name][record["complex_component"] == "imag"])
    layout = schema["parameter_layout"]
    value_count = int(layout["value_component_count"])
    offset = value_count + int(layout["momentum_parameter_count"])
    momenta = tuple(tuple(Decimal(x) for x in vector) for vector in point)
    state = [(Decimal(0), Decimal(0))] * max(
        int(layout["parameter_count_if_flattened"]), offset + len(values)
    )
    with localcontext() as context:
        context.prec = digits + 10
        _fill_sources(state, momenta, schema, tuple(values))
        _fill_momenta(state, momenta, schema)
    for index, value in enumerate(values):
        state[offset + index] = (value, Decimal(0))
    for stage in blueprint.stages:
        evaluator = _build_interpreted_stage_evaluator(
            stage, settings=settings, output_partitions=((0, stage.output_length),)
        )
        inputs = [None] * stage.parameter_count
        for component in stage.input_components:
            inputs[component.parameter_index] = state[component.global_component]
        outputs = _evaluate_interpreted_stage(
            evaluator, _upcast_complex_inputs(tuple(inputs), digits), precision=digits
        )
        for slot in stage.output_slots:
            state[slot.component_start : slot.component_stop] = outputs[
                slot.output_start : slot.output_stop
            ]
    return {
        (int(slot["current_id"]), str(slot["variant"])): tuple(
            state[slot["component_start"] : slot["component_stop"]]
        )
        for slot in schema["value_storage"]["value_slots"]
    }


def _scalars(ct: CountertermSlice, plan, parameters, label):
    """Exact source scalars in the carrier's native arithmetic."""

    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    arithmetic, digits, _, _ = legacy_capture.CARRIERS[label]
    state = NativeScalarBindings(
        plan,
        arithmetic=arithmetic,
        precision=digits,
        parameters={
            name: value
            for name, value in parameters.items()
            if name in plan.external_names
        },
    )
    return {
        output: (Decimal(re), Decimal(im))
        for output, (re, im) in zip(
            plan.outputs, state.values.to_decimal(), strict=True
        )
    }


def _ordinary_entries(ct: CountertermSlice):
    """Non-ghost ordinary vertex entries by vertex, for the Born oracle."""

    spins = {entry["name"]: entry["spin"] for entry in ct.data["particles"]}
    out = defaultdict(list)
    for entry in ct.prepared.model.kernels:
        if entry.kind == "tree" and all(
            spins.get(name, -1) != -1 for name in entry.particle_names
        ):
            out[entry.vertex_id].append(entry)
    return tuple(out.values())


def _born_terms(ct: CountertermSlice, closure: SiteClosure):
    """Born terms rooted at the vertex of leg 1, by amplitude order."""

    full = (1 << len(ct.table.legs)) - 1
    born = set(ct.blocks.born_orders)
    out = defaultdict(list)
    for entries in _ordinary_entries(ct):
        fields = entries[0].particle_names
        order = add_orders(tuple(entries[0].order))
        for parts in _set_partitions(full, len(fields), len(fields)):
            if 1 not in parts:
                continue
            for choice in product(*(ct.table.currents(part) for part in parts)):
                if sorted(b.field for b in choice) != sorted(fields):
                    continue
                total = add_orders(order, *(b.orders for b in choice))
                if total not in born:
                    continue
                free, attachments = sorted(choice), []
                for name in fields:
                    block = next(b for b in free if b.field == name)
                    free.remove(block)
                    attachments.append(block)
                out[total].extend(closure.vertex_terms(entries, attachments))
    return out


def _components(ct: CountertermSlice, label: str, index: int):
    """Born and the CT components at one fixture point, by squared order."""

    from pyamplicol.loops.request import prepare_loop_external_request
    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue

    if "closure" not in ct._caches:
        sites, fields = ct.sites(), ct.fields()
        entries = [e for site in sites.sites for e in site.rule.entries]
        entries += [e for rows in _ordinary_entries(ct) for e in rows]
        unique = list({id(e): e for e in entries}.values())
        catalogue = build_loop_kernel_catalogue(ct.compiled, counterterm_entries=unique)
        closure = SiteClosure(ct.blocks, ct.data, catalogue)
        plan, names = counterterm_scalar_plan(ct.prepared.structure, sites, fields)
        ct._caches["closure"] = (
            sites,
            fields,
            closure,
            _born_terms(ct, closure),
            {site: closure.terms(site) for site in sites.sites},
            plan,
            names,
        )
    sites, fields, closure, born_terms, site_terms, plan, names = ct._caches["closure"]
    fixture, control = ct.fixture, ct.control
    point = fixture["points"][index]
    scale = DecimalComplex.CONTEXT.sqrt(Decimal(fixture["scales_squared"][index]))
    parameters = {**legacy_capture.PARAMETERS, "MU_R": str(scale)}
    parameters.update(legacy_capture.FIXED_PARAMETERS)
    if control.drell_yan:
        parameters["aEWM1"] = legacy_capture.DRELL_YAN_MODEL_VALUES[1]
    carrier, convert = _carrier(label)
    with _exact_u1_weights():
        rows = _block_rows(ct, point, parameters, WORKING_DIGITS[label])
    scalars = {
        key: convert(value)
        for key, value in _scalars(ct, plan, parameters, label).items()
    }
    ordinary = sorted(
        {term.entry.coupling.name for terms in born_terms.values() for term in terms}
    )
    ordinary_plan = ct.prepared.structure.scalar_bindings(
        tuple((name, 0) for name in ordinary)
    )
    scalars.update(
        {
            key: convert(value)
            for key, value in _scalars(ct, ordinary_plan, parameters, label).items()
        }
    )
    labels = sorted(leg.label for leg in ct.blocks.process.legs)
    initial = {leg.label for leg in ct.blocks.process.legs if leg.is_initial}
    momenta = {}
    for mask in range(1, 1 << len(labels)):
        total = [Decimal(0)] * 4
        for label_ in labels:
            if mask >> (label_ - 1) & 1:
                for mu in range(4):
                    value = Decimal(point[label_ - 1][mu])
                    total[mu] = DecimalComplex.CONTEXT.add(
                        total[mu], value.copy_negate() if label_ in initial else value
                    )
        momenta[mask] = [convert((value, Decimal(0))) for value in total]
    helicities = defaultdict(set)
    for block in ct.blocks.blocks:
        if block.is_source:
            for row in block.rows:
                ((leg, value),) = row.helicities
                helicities[leg].add(value)
    request = prepare_loop_external_request(
        ct.prepared.structure.native, control.incoming, control.outgoing
    )
    average = int(request.initial_average)
    overlaps = ColourOverlaps(closure.legs)

    def term_rows(term, helicity):
        out = []
        for block in term.blocks:
            want = tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
            row = next((r for r in block.rows if r.helicities == want), None)
            if row is None:
                return None
            variant = "source" if block.is_source else "propagated"
            out.append([convert(v) for v in rows[(row.current_id, variant)]])
        return out

    def kinematic(term, helicity):
        values = term_rows(term, helicity)
        if values is None:
            return None
        vectors = [momenta[block.key.external_mask] for block in term.blocks]
        return term.sign * closure.value(term, values, vectors, carrier)

    born = defaultdict(lambda: carrier.zero)
    components = defaultdict(lambda: carrier.zero)
    for values in product(*(sorted(helicities[leg]) for leg in labels)):
        helicity = dict(zip(labels, values, strict=True))
        amplitudes = {}
        for order, terms in born_terms.items():
            vector = {}
            for term in terms:
                value = kinematic(term, helicity)
                if value is not None:
                    coupling = scalars[term.entry.coupling.name, 0]
                    add_colour(vector, term.colour, coupling * value, carrier)
            amplitudes[order] = vector
        for left, right in product(amplitudes, repeat=2):
            born[add_orders(left, right)] += overlaps.interfere(
                amplitudes[left], amplitudes[right], carrier
            )
        loops = defaultdict(dict)
        for site, terms in site_terms.items():
            for term in terms:
                value = kinematic(term, helicity)
                if value is None:
                    continue
                for power in LAURENT_POWERS:
                    coupling = scalars[term.entry.coupling.name, power]
                    add_colour(
                        loops[site.component, power, site.orders],
                        term.colour,
                        len(site.alternatives) * coupling * value,
                        carrier,
                    )
        # Every (Born, loop) order pair of these controls is one admitted sector.
        for (component, power, loop_order), vector in loops.items():
            for order, amplitude in amplitudes.items():
                value = overlaps.interfere(amplitude, vector, carrier)
                components[component, add_orders(order, loop_order), power] += (
                    value + value.conjugate()
                )
        for insertion in fields:
            name = names[insertion.identity]
            ket = amplitudes.get(insertion.born_order)
            if ket is None:
                continue
            for order, amplitude in amplitudes.items():
                overlap = overlaps.interfere(amplitude, ket, carrier)
                for power in LAURENT_POWERS:
                    value = scalars[name, power] * overlap
                    components[
                        FIELD_COMPONENT, add_orders(order, insertion.orders), power
                    ] += value + value.conjugate()
    return (
        {order: value / average for order, value in born.items()},
        {key: value / average for key, value in components.items()},
    )


def _real(value) -> Decimal:
    return Decimal(value.real)


def _sector_orders(fixture, kind):
    return [add_orders(tuple(item.items())) for item in fixture["orders"][kind]]


# ------------------------------------------------------------------------ tests


@pytest.mark.loop_fast
def test_ufo_colour_lowering_matches_the_tree_normaliser(loop_sm_compiled_model):
    """Tr-free UFO colours lower exactly like ``normalize_color_expression``;
    Tr(a, b) is delta^ab / 2."""

    from pyamplicol.color.connections import ColorLeg
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.v2.colour import (
        _vertex_label,
        reduce_network,
        relabel,
        spenso_colour_network,
    )
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.tensors import normalize_color_expression

    compiled = load_compiled_model(loop_sm_compiled_model)
    source = build_loop_model(compiled.raw_loop_model)
    colours = {p.name: p.color for p in compiled.ir.particles}
    checked = set()
    for entry in source.kernels:
        reps = tuple(colours[name] for name in entry.particle_names)
        if (entry.color, reps) in checked or set(reps) == {1}:
            continue
        checked.add((entry.color, reps))
        # Literal tensors type their legs all-outgoing: incoming triplets flip.
        legs = tuple(
            ColorLeg(i + 1, -rep if abs(rep) == 3 else rep)
            for i, rep in enumerate(reps)
            if rep != 1
        )
        external = {("leg", leg.label): leg.label for leg in legs}
        mine = reduce_network(ufo_colour_network(entry.color, reps), legs, external)
        if "Tr" in entry.color:
            assert entry.color == "Tr(1,2)" and reps[:2] == (8, 8)
            expected = reduce_network(
                ufo_colour_network("Identity(1,2)", reps), legs, external
            )
            assert mine.terms == tuple(
                type(term)(term.coefficient.scaled(Fraction(1, 2)), term.tensor)
                for term in expected.terms
            )
            continue
        spenso = normalize_color_expression(
            parse_exact_ufo(entry.color).to_canonical_string(), reps
        ).expression
        network = relabel(
            spenso_colour_network(spenso), lambda name: _vertex_label(str(name))
        )
        reference = reduce_network(network, legs, external)
        assert (mine.terms, mine.normalization_power) == (
            reference.terms,
            reference.normalization_power,
        ), entry.color
    assert any("d(" in colour for colour, _ in checked)


@pytest.mark.loop_fast
def test_gg_gg_insertions_equal_legacy(ct_slice):
    ct = ct_slice("gg_gg")
    sites = ct.sites()
    assert sites.k_max == 4
    assert _legacy_multisets(ct, sites, ct.fields()) == ct.fixture["insertions"]
    # Divergent witnesses: bubbles, triangles and boxes of gluon, ghost and
    # quark loops; every one is covered by a species-matched R2 rule.
    assert all(entry.rules and entry.certificate is None for entry in sites.coverage)
    assert {len(signature.attachments) for signature in sites.witnesses} == {2, 3, 4}


@pytest.mark.loop_fast
def test_removed_rule_makes_generation_fail(ct_slice):
    """Dropping the gluon-loop R2 of the three-gluon vertex leaves the gluon and
    ghost triangles uncovered; generation names them. A certified zero for
    exactly those witnesses would be accepted instead."""

    ct = ct_slice("gg_gg")
    removed = {
        name
        for name, rule in ct.rules.items()
        if rule.kind == "R2"
        and {entry.vertex_name for entry in rule.entries} == {"V_R23G"}
        and rule.alternatives == ((21,),)
    }
    assert len(removed) == 1
    rules = {name: rule for name, rule in ct.rules.items() if name not in removed}
    with pytest.raises(CountertermCompletenessError) as failure:
        ct.sites(rules)
    message = str(failure.value)
    assert "G{0}, G{1}, G{2,3}" in message and "(21,)" in message
    assert "(82,)" in message  # ghost loops belong to the gluon's loop group
    uncovered = [
        CertifiedZero(signature, species, "test: declared zero")
        for signature, rows in ct.sites().witnesses.items()
        if signature.fields == ("G", "G", "G")
        for species in rows
        if species in ((21,), (82,))
    ]
    sites = ct.sites(rules, zeros=uncovered)
    assert {entry.certificate for entry in sites.coverage} == {
        None,
        "test: declared zero",
    }
    # Only the removed rule's sites are gone.
    assert sites.sites == tuple(
        site for site in ct.sites().sites if site.rule_name not in removed
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", CONTROLS)
def test_insertion_multisets_equal_legacy(ct_slice, name):
    ct = ct_slice(name)
    sites = ct.sites()
    assert _legacy_multisets(ct, sites, ct.fields()) == ct.fixture["insertions"]
    assert set(COMPONENT_BY_KIND) >= {site.kind for site in sites.sites}


@pytest.mark.loop_physics
@pytest.mark.parametrize("label", tuple(TOLERANCE))
@pytest.mark.parametrize("name", CONTROLS)
def test_components_equal_legacy(ct_slice, loop_native, name, label):
    """UV_local, R2 and field components equal O1 at every fixture point."""

    ct = ct_slice(name)
    fixture = ct.fixture
    results = fixture["results"][label]
    born_sectors = _sector_orders(fixture, "born")
    sectors = _sector_orders(fixture, "virtual")
    tolerance = TOLERANCE[label]
    for index in range(len(fixture["points"])):
        born, components = _components(ct, label, index)
        for b, order in enumerate(born_sectors):
            reference = Decimal(results["born"][index][b][0])
            assert abs(_real(born[order]) - reference) <= tolerance * abs(reference)
        scale_born = max(abs(Decimal(row[0])) for row in results["born"][index])
        for component in COMPONENTS:
            for s, order in enumerate(sectors):
                reference = [
                    Decimal(results[component][index][s][k][0]) for k in range(3)
                ]
                mine = [
                    _real(components.get((component, order, power), 0j))
                    for power in LAURENT_POWERS
                ]
                scale = max(map(abs, reference)) or scale_born
                for k in range(3):
                    assert abs(mine[k] - reference[k]) <= tolerance * scale, (
                        component,
                        index,
                        k,
                        mine[k],
                        reference[k],
                    )
