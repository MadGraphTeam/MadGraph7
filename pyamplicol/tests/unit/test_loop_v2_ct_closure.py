# SPDX-License-Identifier: 0BSD
"""Counterterm amplitudes from the tree recursion, closed at leg 1 (design §6.1).

The production form of the counterterms (``loops.v2.ct_closure``): CT = 1
currents of the CT-lowered model's loop-request recursion, closed at the
vertex of leg 1 with exact colour, as the slice closes its Born.

Oracles:

* O6: per component, squared order and Laurent coefficient, the closure of
  the lowered rules equals their site closure (``loops.v2.ct_sites``);
* O1: with the sites of the rules the recursion leaves to site closure, the
  UV_local and R2 components equal the frozen legacy engine.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from itertools import product

import pytest

from pyamplicol.config import PoleSelection
from pyamplicol.loops.v2.ct_sites import LAURENT_POWERS, ColourOverlaps, add_colour
from pyamplicol.loops.v2.families import add_orders
from tools.loop_reference import legacy_capture

from .loop_v2_stage_rows import stage_rows
from .test_loop_v2_counterterms import (
    CONTROLS,
    TOLERANCE,
    WORKING_DIGITS,
    DecimalComplex,
    _carrier,
    _components,
    _exact_u1_weights,
    _ordinary_entries,
    _sector_orders,
)
from .test_loop_v2_ct_recursion import (  # noqa: F401 (fixtures)
    _recursion,
    ct_slice,
    lowered_ct_model,
)

CT_COMPONENTS = ("uv_local", "r2")


@pytest.fixture(scope="module")
def closure_ct_model(loop_sm_compiled_model):
    """``vetoed |PDG| codes -> CountertermModel``: loop_sm lowered for the
    closure's loop-request blocks (multi-trace colours included), per veto
    (and per set ``dirac`` of Weyl fields propagated as Dirac fields)."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import lower_model_counterterms
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    prepared = prepare_loop_request_model(
        compiled, fixed_parameters=legacy_capture.FIXED_PARAMETERS
    )
    built = {}

    def build(veto=(), dirac=()):
        key = (tuple(sorted(veto)), tuple(sorted(dirac)))
        if key not in built:
            built[key] = lower_model_counterterms(
                compiled,
                prepared.counterterms.rules,
                fixed_parameters=prepared.structure.fixed_parameters,
                veto=key[0],
                multi_trace=True,
                dirac_fields=key[1],
            )
        return built[key]

    return build


def _closure(ct, lowered):
    """The CT program, its closure and the source scalar plan, cached."""

    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue
    from pyamplicol.loops.v2.ct_closure import (
        CountertermBlocks,
        CountertermClosure,
        counterterm_closure_entries,
    )
    from pyamplicol.processes.model import build_model_process_ir

    key = ("ct-closure", id(lowered))
    if key in ct._caches:
        return ct._caches[key]
    recursion = _recursion(ct, lowered)
    compiled = lowered.lowered.compiled
    text = ct.control.process.split(" / ")[0]
    process = build_model_process_ir(text, compiled.ir, color_accuracy="full")
    program = CountertermBlocks.build(
        recursion, process, ignored_particle_ids=ct.vetoed
    )
    ordinary = _ordinary_entries(ct)
    sites = CountertermClosure.root_sites(program, ct.data, ct.selection)
    # The request's rules declare the loop species (as the slice's sites do).
    every = [entry for rule in ct.rules.values() for entry in rule.entries]
    entries = [*counterterm_closure_entries(sites, ordinary), *every]
    catalogue = build_loop_kernel_catalogue(
        ct.compiled, counterterm_entries=list({id(e): e for e in entries}.values())
    )
    closure = CountertermClosure.build(program, sites, ordinary, ct.data, catalogue)
    outputs = tuple(sorted({term.coupling for term in closure.terms}))
    plan = ct.prepared.structure.scalar_bindings(outputs)
    ct._caches[key] = (recursion, closure, plan)
    return ct._caches[key]


def _program_rows(closure, parameters, digits):
    """Rows of the CT program's blocks, exactly (as ``_block_rows``)."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

    compiled = closure.program.recursion.model.lowered.compiled
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
    model_symbols = symbols.model(compiled.name)

    def values(records):
        external = {
            record["name"]: Decimal(
                str(parameters.get(record["name"], repr(record["default"])))
            )
            for record in records
            if record["kind"] == "external_parameter"
        }
        substitutions = dict(
            zip(
                (
                    E(model_symbols.symbol(name).to_canonical_string())
                    for name in external
                ),
                _upcast_complex_inputs(
                    [(value, Decimal(0)) for value in external.values()], digits
                ),
                strict=True,
            )
        )
        derived, out = {}, []
        for record in records:
            if record["kind"] == "external_parameter":
                out.append(external[record["name"]])
                continue
            name = record["runtime_name"]
            if name not in derived:
                value = E(definitions[name]).evaluate(
                    substitutions, decimal_digit_precision=digits
                )
                derived[name] = tuple(Decimal(x) for x in value.to_decimal_tuple())
            out.append(derived[name][record["complex_component"] == "imag"])
        return out

    return values, compiled


def closure_components(ct, lowered, label: str, index: int) -> dict:
    """CT amplitudes of the closure interfered with the Born, per (component,
    squared order, power), at one fixture point (the ``_components`` oracle
    arithmetic: exact colour at N_c = 3, rows at the carrier's digits)."""

    from pyamplicol.loops.scalar_bindings import NativeScalarBindings
    from pyamplicol.loops.v2.ct_closure import model_value, program_propagated_currents
    from pyamplicol.models.external import CompiledUFOModel

    recursion, closure, plan = _closure(ct, lowered)
    fixture, control = ct.fixture, ct.control
    point = fixture["points"][index]
    scale = DecimalComplex.CONTEXT.sqrt(Decimal(fixture["scales_squared"][index]))
    parameters = {**legacy_capture.PARAMETERS, "MU_R": str(scale)}
    parameters.update(legacy_capture.FIXED_PARAMETERS)
    if control.drell_yan:
        parameters["aEWM1"] = legacy_capture.DRELL_YAN_MODEL_VALUES[1]
    digits = WORKING_DIGITS[label]
    carrier, convert = _carrier(label)
    model_parameters = {**parameters, **recursion.parameter_values()}
    values, compiled = _program_rows(closure, model_parameters, digits)
    momenta_rows = [[Decimal(x) for x in vector] for vector in point]
    dag = closure.program.blocks.blocks_dag
    with _exact_u1_weights():
        rows = stage_rows(
            dag,
            CompiledUFOModel(compiled),
            momenta_rows,
            values,
            digits,
            propagated_currents=program_propagated_currents(dag),
        )
    arithmetic, scalar_digits, _, _ = legacy_capture.CARRIERS[label]
    state = NativeScalarBindings(
        plan,
        arithmetic=arithmetic,
        precision=scalar_digits,
        parameters={
            name: value
            for name, value in parameters.items()
            if name in plan.external_names
        },
    )
    scalars = {
        output: convert((Decimal(re), Decimal(im)))
        for output, (re, im) in zip(
            plan.outputs, state.values.to_decimal(), strict=True
        )
    }
    scales = {}
    for term in closure.terms:
        if term.scale is not None and term.scale not in scales:
            pair = model_value(compiled, term.scale, model_parameters, digits + 10)
            scales[term.scale] = convert(tuple(Decimal(x) for x in pair))
    blocks = closure.program.blocks
    labels = sorted(leg.label for leg in blocks.process.legs)
    initial = {leg.label for leg in blocks.process.legs if leg.is_initial}
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
    for block in blocks.blocks:
        if block.is_source:
            for row in block.rows:
                ((leg, value),) = row.helicities
                helicities[leg].add(value)

    def kinematic(term, helicity):
        values = []
        for block in term.blocks:
            want = tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
            row = next((r for r in block.rows if r.helicities == want), None)
            if row is None:
                return None
            variant = "source" if block.is_source else "propagated"
            values.append([convert(v) for v in rows[(row.current_id, variant)]])
        vectors = [momenta[block.key.external_mask] for block in term.blocks]
        return term.sign * closure.closure.value(term, values, vectors, carrier)

    born_amplitudes = _born_amplitudes(ct, label, index)
    overlaps = ColourOverlaps(closure.closure.legs)
    components = defaultdict(lambda: carrier.zero)
    for choice in product(*(sorted(helicities[leg]) for leg in labels)):
        helicity = dict(zip(labels, choice, strict=True))
        loops = defaultdict(dict)
        for item in closure.terms:
            value = kinematic(item.term, helicity)
            if value is None:
                continue
            value = item.multiplicity * scalars[item.coupling] * value
            if item.scale is not None:
                value = scales[item.scale] * value
            add_colour(
                loops[item.component, item.power, item.orders],
                item.term.colour,
                value,
                carrier,
            )
        amplitudes = born_amplitudes[tuple(sorted(helicity.items()))]
        for (component, power, loop_order), vector in loops.items():
            for order, amplitude in amplitudes.items():
                value = overlaps.interfere(amplitude, vector, carrier)
                components[component, add_orders(order, loop_order), power] += (
                    value + value.conjugate()
                )
    average = _average(ct)
    return {key: value / average for key, value in components.items()}


def _average(ct) -> int:
    from pyamplicol.loops.request import prepare_loop_external_request

    request = prepare_loop_external_request(
        ct.prepared.structure.native, ct.control.incoming, ct.control.outgoing
    )
    return int(request.initial_average)


def _born_amplitudes(ct, label: str, index: int) -> dict:
    """Born colour vectors per helicity of the site oracle (``_components``)."""

    key = ("born-amplitudes", label, index)
    if key in ct._caches:
        return ct._caches[key]
    out = _born_vectors(ct, label, index)
    ct._caches[key] = out
    return out


def _born_vectors(ct, label: str, index: int) -> dict:
    from pyamplicol.loops.v2.ct_sites import SiteClosure

    _components(ct, label, index)  # builds the oracle's closure cache
    (cached,) = (
        value
        for key, value in ct._caches.items()
        if isinstance(key, tuple) and key[:2] == ("closure", None)
    )
    _sites, _fields, closure, born_terms, _site_terms, _plan, _names = cached
    assert isinstance(closure, SiteClosure)
    return _born_vectors_from(ct, closure, born_terms, label, index)


def _born_vectors_from(ct, closure, born_terms, label: str, index: int) -> dict:
    from .test_loop_v2_counterterms import _block_rows, _scalars

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
    ordinary = sorted(
        {term.entry.coupling.name for terms in born_terms.values() for term in terms}
    )
    plan = ct.prepared.structure.scalar_bindings(tuple((name, 0) for name in ordinary))
    scalars = {
        key: convert(value)
        for key, value in _scalars(ct, plan, parameters, label).items()
    }
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
    out = {}
    for choice in product(*(sorted(helicities[leg]) for leg in labels)):
        helicity = dict(zip(labels, choice, strict=True))
        amplitudes = {}
        for order, terms in born_terms.items():
            vector = {}
            for term in terms:
                values = []
                for block in term.blocks:
                    want = tuple(
                        sorted((leg, helicity[leg]) for leg in block.key.labels)
                    )
                    row = next((r for r in block.rows if r.helicities == want), None)
                    if row is None:
                        values = None
                        break
                    variant = "source" if block.is_source else "propagated"
                    values.append([convert(v) for v in rows[(row.current_id, variant)]])
                if values is None:
                    continue
                vectors = [momenta[block.key.external_mask] for block in term.blocks]
                value = term.sign * closure.value(term, values, vectors, carrier)
                coupling = scalars[term.entry.coupling.name, 0]
                add_colour(vector, term.colour, coupling * value, carrier)
            amplitudes[order] = vector
        out[tuple(sorted(helicity.items()))] = amplitudes
    return out


def _site_value(rows, key) -> Decimal:
    return Decimal(rows[key].real) if key in rows else Decimal(0)


def _assert_closure_equals_sites_and_legacy(ct, lowered, label: str) -> None:
    from .test_loop_v2_ct_recursion import _site_components

    recursion = _recursion(ct, lowered)
    fixture = ct.fixture
    results = fixture["results"][label]
    sectors = _sector_orders(fixture, "virtual")
    tolerance = TOLERANCE[label]
    with localcontext() as context:
        context.prec = 120
        for index in range(len(fixture["points"])):
            mine_all = closure_components(ct, lowered, label, index)
            sites = _site_components(ct, label, index, recursion.rules)
            residual = _site_components(ct, label, index, recursion.residual)
            scale_born = max(abs(Decimal(row[0])) for row in results["born"][index])
            assert {key[0] for key in mine_all} <= set(CT_COMPONENTS)
            for component in CT_COMPONENTS:
                for s, order in enumerate(sectors):
                    for k, power in enumerate(LAURENT_POWERS):
                        key = (component, order, power)
                        reference = Decimal(results[component][index][s][k][0])
                        scale = abs(reference) or scale_born
                        mine = _site_value(mine_all, key)
                        site = _site_value(sites, key)
                        total = mine + _site_value(residual, key)
                        assert abs(mine - site) <= tolerance * scale, (key, index)
                        assert abs(total - reference) <= tolerance * scale, (
                            key,
                            index,
                        )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", CONTROLS)
def test_closure_equals_site_closure(ct_slice, closure_ct_model, name):  # noqa: F811
    """O6 and O1 of the closed CT amplitudes in binary64."""

    ct = ct_slice(name)
    _assert_closure_equals_sites_and_legacy(ct, closure_ct_model(ct.vetoed), "double")


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg", "uux_ddxg", "dy"))
def test_closure_equals_site_closure_at_200_bits(
    ct_slice,  # noqa: F811
    closure_ct_model,
    name,
):
    """O6 and O1 of the closed CT amplitudes in the 200-bit carrier."""

    ct = ct_slice(name)
    _assert_closure_equals_sites_and_legacy(
        ct, closure_ct_model(ct.vetoed), "arbitrary-200"
    )


# ------------------------------------------------------------ slice-level O6

# Requests beyond the legacy controls: the gggA R2 (a d-symbol with a
# colour-singlet leg), the ggAA R2 (two singlet legs) and the four-gluon R2
# inside five-gluon blocks need the multi-trace colour state and the contact
# splits of the lowered model; with a gluon at leg 1, a light-quark clone line
# meets the root with its Weyl factor and a top clone line carries the top
# width of the tree propagators. In the six-leg requests these R2 vertices
# take a quark pair's gluon inside a block (uu~ -> dd~gg, gg -> uu~ag) or
# feed their gluon into the quark line of the block (uu~ -> aggg,
# uu~ -> aagg): the U(1) projection of their gluon legs (design §6.1).
# uu~ -> dd~ss~ joins three quark lines by gluon exchanges, each with its
# U(1)-subtraction auxiliary. In gg -> uu~uu~ the four-gluon R2 closing the
# gluons of one pairing of the identical-flavour lines and the U(1)-subtraction
# auxiliary of the other pairing give one block derivations of both
# colour-connection parities, whose -1 the compilation folds
# (``generation.fermion_ordering.apply_colour_connection_signs``, design §3.3
# "Mixed connection parities"). uu~ -> gggg closes multi-gluon currents into
# closed traces of three labels (nightly; its CT-level check runs in the
# physics tier).
# ``name -> (process, Born QCD, QED, PDG codes of the rule the request is
# about, leg masses)``.
SLICE_O6 = {
    "uux_agg": ("u u~ > a g g", 2, 1, (21, 21, 21, 22), (0,) * 5),
    "uux_aag": ("u u~ > a a g", 1, 2, (21, 21, 22, 22), (0,) * 5),
    "gg_ggg": ("g g > g g g", 3, 0, (21, 21, 21, 21), (0,) * 5),
    "gg_uuxg": ("g g > u u~ g", 3, 0, (2, 2), (0,) * 5),
    "gg_ttxg": ("g g > t t~ g", 3, 0, (6, 6), (0, 0, 173, 173, 0)),
    "uux_ddxgg": ("u u~ > d d~ g g", 4, 0, (21, 21, 21, 21), (0,) * 6),
    "gg_uuxag": ("g g > u u~ a g", 3, 1, (21, 21, 21, 22), (0,) * 6),
    "uux_aggg": ("u u~ > a g g g", 3, 1, (21, 21, 21, 22), (0,) * 6),
    "uux_aagg": ("u u~ > a a g g", 2, 2, (21, 21, 22, 22), (0,) * 6),
    "gg_ttxgg": ("g g > t t~ g g", 4, 0, (21, 21, 21, 21), (0, 0, 173, 173, 0, 0)),
    "uux_ttxgg": ("u u~ > t t~ g g", 4, 0, (2, 2, 21), (0, 0, 173, 173, 0, 0)),
    "uux_ddxssx": ("u u~ > d d~ s s~", 4, 0, (1, 1, 21), (0,) * 6),
    "gg_uuxuux": ("g g > u u~ u u~", 4, 0, (21, 21, 21, 21), (0,) * 6),
    "uux_gggg": ("u u~ > g g g g", 4, 0, (21, 21, 21, 21), (0,) * 6),
}
SLICE_PARAMETERS = {"aS": "0.118", "MT": "173"}
# Requests whose trees propagate a top: its width (a runtime input) breaks the
# Ward identity of their gluon currents, so their blocks and Born take the
# full tree kernels (``blocks.with_ward_kernels``), the Feynman rules of the
# lowered model's recursion; recursion and sites then agree at the model's
# top width.
SLICE_FULL_KERNELS = frozenset({"gg_ttxg", "gg_ttxgg", "uux_ttxgg"})
SLICE_SCALE_SQUARED = "1000000"
# Recursion and sites differ only by rounding (binary64) or not at all within
# the 200-bit carrier's working precision; the legacy O1 tolerances.
SLICE_TOLERANCE = {"double": Decimal("1e-12"), "arbitrary-200": Decimal("1e-40")}
# The 200-bit carrier and the six-leg request with tops in every block (about
# 15 minutes per carrier) run in the nightly tier; gg -> tt~g keeps the top
# width and the full kernels in the physics tier.
SLICE_CARRIERS = (
    "double",
    pytest.param("arbitrary-200", marks=pytest.mark.loop_nightly),
)
SLICE_NIGHTLY = frozenset({"gg_ttxgg", "uux_gggg"})


@pytest.fixture(scope="module")
def slice_routes(loop_sm_compiled_model, loop_native):
    """``(name, counterterms) -> LoopSlice`` for the ``SLICE_O6`` requests."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built = {}

    def build(name: str, counterterms: str):
        if (name, counterterms) not in built:
            process, qcd, qed, _rule, _masses = SLICE_O6[name]
            built[name, counterterms] = LoopSlice.generate(
                compiled,
                SliceRequest(
                    process,
                    {"QCD": qcd, "QED": qed},
                    {"QCD": qcd + 2, "QED": qed},
                    fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                    counterterms=counterterms,
                ),
                loop_native,
            )
        return built[name, counterterms]

    return build


# The quark triangles with one or two photons and the rest gluons have no rule
# in loop_sm: they vanish (Furry's theorem; a single gluon's colour trace),
# which the model certifier of design §6.3 is to certify. Until it exists the
# O6 requests with photons certify them here; both routes see the same sites.
FURRY_ZEROS = {("A", "G", "G"), ("A", "A", "G")}


def _certify_furry_zeros(monkeypatch) -> None:
    from pyamplicol.loops.v2 import ct_sites

    coverage = ct_sites._coverage

    def certified(witnesses, rules, names, corrections, partners, zeros):
        furry = tuple(
            ct_sites.CertifiedZero(signature, species, "Furry or colour zero (test)")
            for signature, group in witnesses.items()
            if signature.fields in FURRY_ZEROS
            for species in group
        )
        return coverage(
            witnesses, rules, names, corrections, partners, (*zeros, *furry)
        )

    monkeypatch.setattr(ct_sites, "_coverage", certified)


def _slice_point(masses) -> list[list[str]]:
    from tools.loop_reference import exact_points

    (point,) = exact_points.generate(list(masses), count=1)
    return [[exact_points.decimal_string(x) for x in p] for p in point]


def _route_differences(slice_routes, name, carrier, parameters, masses=None):
    """Largest |recursion - sites| per component over the slice's sectors and
    Laurent coefficients, relative to the sites value (the Born if it is
    zero), at one exact point (leg ``masses``, by default ``SLICE_O6``'s)."""

    from pyamplicol.loops.v2.assemble import COMPONENTS, Precision

    recursion = slice_routes(name, "recursion")
    sites = slice_routes(name, "sites")
    precision = {
        "double": Precision.double(),
        "arbitrary-200": Precision.arbitrary(60),
    }[carrier]
    point = _slice_point(SLICE_O6[name][4] if masses is None else masses)
    mine, reference = (
        route.evaluate(
            point, SLICE_SCALE_SQUARED, parameters=parameters, precision=precision
        )
        for route in (recursion, sites)
    )
    worst = {}
    with localcontext() as context:
        context.prec = 120
        assert mine.born.keys() == reference.born.keys()
        born = max(abs(Decimal(str(value))) for value in reference.born.values())
        assert born > 0
        for component in COMPONENTS:
            assert mine.components[component].keys() == (
                reference.components[component].keys()
            ), component
            worst[component] = Decimal(0)
            for sector, laurent in reference.components[component].items():
                values = [Decimal(str(x)) for x in laurent.coefficients]
                ours = [
                    Decimal(str(x))
                    for x in mine.components[component][sector].coefficients
                ]
                scale = max(map(abs, values)) or born
                for a, b in zip(ours, values, strict=True):
                    worst[component] = max(worst[component], abs(a - b) / scale)
        # The counterterm components are not trivially zero.
        assert any(
            Decimal(str(x)) != 0
            for laurent in reference.components["r2"].values()
            for x in laurent.coefficients
        )
    return worst


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", SLICE_CARRIERS)
@pytest.mark.parametrize(
    "name",
    tuple(
        pytest.param(name, marks=pytest.mark.loop_nightly)
        if name in SLICE_NIGHTLY
        else name
        for name in SLICE_O6
    ),
)
def test_slice_counterterms_from_recursion_equal_sites(
    slice_routes, monkeypatch, name, carrier
):
    """O6 on the slice: every component of the request with its counterterms
    from the recursion (production: the plain admission, no site closure)
    equals the request with every CT rule closed at sites (witness
    admission); the recursion lowers every QCD rule."""

    _certify_furry_zeros(monkeypatch)
    recursion = slice_routes(name, "recursion")
    assert slice_routes(name, "sites").counterterm_program is None
    program = recursion.counterterm_program
    residual = program.recursion.residual
    # Every rule is lowered: no rule is refused for the U(1) part of a gluon,
    # and the Levi-Civita cells of the R2 of the Z with gluons cancel between
    # the quark doublets (design §6.1).
    rules = recursion.prepared.counterterms.rules
    assert not residual, residual
    # The rules of interest are lowered and admitted.
    pdgs = SLICE_O6[name][3]

    def about(rule):
        return tuple(sorted(abs(int(pdg)) for pdg in rule.particle_pdgs)) == pdgs

    lowered = [n for n, rule in program.recursion.rules.items() if about(rule)]
    assert lowered and all(program.recursion.admission[n] for n in lowered), lowered
    assert not {n for n, rule in rules.items() if about(rule)} & set(residual)
    variant = "full" if name in SLICE_FULL_KERNELS else "transverse"
    assert recursion.blocks.kernel_variant == variant
    worst = _route_differences(slice_routes, name, carrier, SLICE_PARAMETERS)
    assert all(value <= SLICE_TOLERANCE[carrier] for value in worst.values()), worst


@pytest.mark.loop_physics
def test_generation_leaves_the_program_rows_to_the_first_point(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """Generation builds the CT program and its root terms but no Python
    tree rows: a native evaluation never needs them, and at seven legs their
    evaluators would cost hours. The first point compiles the block rows,
    then the program's (the rows are lazy, RUST-MIGRATION.md §10 decision 5;
    their rounding does not depend on the CT route,
    ``test_lazy_block_rows_do_not_depend_on_the_counterterm_route``)."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    built = []
    tree_rows = assemble.TreeRows

    class Recorded(tree_rows):
        def __init__(self, dag, compiled, **options):
            built.append(id(dag))
            super().__init__(dag, compiled, **options)

    monkeypatch.setattr(assemble, "TreeRows", Recorded)
    _certify_furry_zeros(monkeypatch)
    route = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > a g",
            {"QCD": 1, "QED": 1},
            {"QCD": 3, "QED": 1},
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        ),
        loop_native,
    )
    program = route.counterterm_program
    assert program is not None and program.closure.terms
    blocks = id(route.blocks.blocks_dag)
    counterterms = id(program.blocks.blocks.blocks_dag)
    assert built == []
    route.evaluate(
        _slice_point((0,) * 4),
        SLICE_SCALE_SQUARED,
        parameters=SLICE_PARAMETERS,
        precision=Precision.double(),
    )
    assert built == [blocks, counterterms]


# ------------------------------------------------------- CT-level O6, pure QCD

# Six legs with four gluons: multi-trace CT currents merge with further
# gluons (an R2 current CT{3,4,5} meeting gluon 6 in an ordinary vertex) and
# the four-gluon R2 takes multi-gluon currents inside a block, which no
# five-leg request does. The comparison stops at the Born and the
# counterterms (``LoopSlice._point``): no loop is reduced.
CT_LEVEL_TOLERANCE = Decimal("1e-12")
# MadLoop's binary64 R2 at six legs (sites route: 1.1e-12 off).
PROVIDER_R2_TOLERANCE = 1e-10


def _counterterms_per_route(compiled, native, process, qcd, point, **options):
    """``route -> (Born, counterterms)`` of ``process`` at one exact point
    (decimal strings), binary64."""

    from pyamplicol.loops.v2.assemble import (
        LoopSlice,
        Precision,
        SliceRequest,
        _fraction,
        _Numbers,
    )

    exact = tuple(tuple(_fraction(x) for x in momentum) for momentum in point)
    values = {}
    for route in ("recursion", "sites"):
        generated = LoopSlice.generate(
            compiled,
            SliceRequest(
                process,
                {"QCD": qcd, "QED": 0},
                {"QCD": qcd + 2, "QED": 0},
                fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                counterterms=route,
                **options,
            ),
            native,
        )
        if route == "recursion":
            residual = generated.counterterm_program.recursion.residual
            assert not residual, residual
        with _Numbers(Precision.double()).scope():
            evaluation = generated._point(
                exact,
                _fraction(SLICE_SCALE_SQUARED),
                dict(SLICE_PARAMETERS),
                Precision.double(),
                PoleSelection.ALL,
            )
        values[route] = (dict(evaluation.born), dict(evaluation.counterterms))
    return values


def _assert_counterterm_routes_agree(values) -> None:
    """Per (component, sector, Laurent power), relative to the sector's
    largest coefficient at the sites (a key one route leaves out is zero)."""

    (born, mine), (born_sites, theirs) = values["recursion"], values["sites"]
    assert born.keys() == born_sites.keys()
    with localcontext() as context:
        context.prec = 60
        for order, value in born.items():
            reference = Decimal(repr(float(born_sites[order])))
            difference = abs(Decimal(repr(float(value))) - reference)
            assert difference <= CT_LEVEL_TOLERANCE * abs(reference), order
        scales = defaultdict(Decimal)
        for (component, sector, _power), value in theirs.items():
            key = (component, sector)
            scales[key] = max(scales[key], abs(Decimal(repr(float(value)))))
        for key in mine.keys() | theirs.keys():
            component, sector, _power = key
            value = mine.get(key, 0.0)
            reference = Decimal(repr(float(theirs.get(key, 0.0))))
            difference = abs(Decimal(repr(float(value))) - reference)
            scale = scales[component, sector] or abs(
                Decimal(repr(float(next(iter(born_sites.values())))))
            )
            assert difference <= CT_LEVEL_TOLERANCE * scale, (key, value, reference)
        assert any(value != 0 for (c, _s, _p), value in theirs.items() if c == "r2")


@pytest.mark.loop_physics
def test_six_leg_four_gluon_counterterms_from_recursion_equal_sites(
    loop_sm_compiled_model, loop_native
):
    """O6 at u u~ > g g g g (CT level, full colour)."""

    from pyamplicol.models.loading import load_compiled_model

    values = _counterterms_per_route(
        load_compiled_model(loop_sm_compiled_model),
        loop_native,
        "u u~ > g g g g",
        4,
        _slice_point((0,) * 6),
    )
    _assert_counterterm_routes_agree(values)


# A fresh interpreter without a personal key whose only Symbolica calls happen
# inside pyAmpliCol: Spenso's Rayon workers then met the package-scoped
# Symbolica key without a thread-wide license and waited for the GIL (design
# §6.1 "Spenso's threads"); a personal key in the environment, or an earlier
# Symbolica call outside the package, hides that. (No Symbolica call may run
# outside the package here: without a key it would take the user's one
# restricted Symbolica thread, which the native loop module's own Symbolica
# then cannot start.) The script reports the package's Spenso policy.
_SIX_GLUON_SCRIPT = """
import json, sys
from pyamplicol import licensing
from pyamplicol.config import PoleSelection
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.v2.assemble import (
    LoopSlice, Precision, SliceRequest, _fraction, _Numbers,
)
from pyamplicol.models.loading import load_compiled_model
model, point, fixed, scale, parameters = sys.argv[1:6]
compiled, native = load_compiled_model(model), load_loop_native()
exact = tuple(tuple(_fraction(x) for x in p) for p in json.loads(point))
out = {}
for route in ("recursion", "sites"):
    generated = LoopSlice.generate(compiled, SliceRequest(
        "g g > g g g g", {"QCD": 4, "QED": 0}, {"QCD": 6, "QED": 0},
        fixed_parameters=json.loads(fixed), counterterms=route), native)
    if route == "recursion":
        assert not generated.counterterm_program.recursion.residual
    with _Numbers(Precision.double()).scope():
        evaluation = generated._point(exact, _fraction(scale),
            json.loads(parameters), Precision.double(), PoleSelection.ALL)
    out[route] = [[[repr(k), float(v)] for k, v in part.items()]
                  for part in (evaluation.born, evaluation.counterterms)]
out["serial"] = (licensing._SPENSO_POLICY_APPLIED, licensing._SPENSO_SERIAL)
print(json.dumps(out))
"""


@pytest.mark.loop_nightly
def test_six_gluon_generation_completes_without_a_personal_key(
    loop_sm_compiled_model, loop_reference
):
    """g g > g g g g through ``LoopSlice.generate`` (recursion and sites) in a
    new interpreter without a personal Symbolica key (the package's library
    key alone, as users without a license run it: Spenso is set serial);
    O6 at the CT level at the first point of the three-provider reference,
    and the R2 component of both routes against MadLoop's (an independent
    oracle of the four-gluon R2 at six legs). About an hour. The native loop
    module's own Symbolica runs restricted without a key, on the user's one
    restricted Symbolica thread: no other unlicensed Symbolica process of
    the user may run at the same time."""

    import ast
    import json
    import os
    import subprocess
    import sys

    control = loop_reference("gg-gggg-r2-component-control")
    point = loop_reference("required/gg_gggg")["points"][0]
    assert point == control["point"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _SIX_GLUON_SCRIPT,
            str(loop_sm_compiled_model),
            json.dumps(point),
            json.dumps(legacy_capture.FIXED_PARAMETERS),
            SLICE_SCALE_SQUARED,
            json.dumps(SLICE_PARAMETERS),
        ],
        capture_output=True,
        text=True,
        timeout=4 * 3600,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in ("SYMBOLICA_LICENSE", "SYMBOLICA_MASTER_LICENSE")
            },
            "RAYON_NUM_THREADS": "4",
            "SYMBOLICA_HIDE_BANNER": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    routes = json.loads(result.stdout.strip().splitlines()[-1])
    # The package set Spenso serial (no thread-wide key).
    assert routes.pop("serial") == [True, True]
    values = {
        route: tuple(
            {ast.literal_eval(key): value for key, value in part} for part in pair
        )
        for route, pair in routes.items()
    }
    _assert_counterterm_routes_agree(values)
    reference = control["r2"]["finite"]
    for route, (born, counterterms) in values.items():
        (value,) = born.values()
        assert abs(value - control["born"]) <= 1e-12 * abs(control["born"]), route
        r2 = counterterms[("r2", (("QCD", 10),), 0)]
        assert abs(r2 - reference) <= PROVIDER_R2_TOLERANCE * abs(reference), route


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "accuracy"),
    (
        ("u u~ > d d~ g g", "lc"),
        ("u u~ > d d~ g g", "nlc"),
        pytest.param("u u~ > g g g g", "nlc", marks=pytest.mark.loop_nightly),
    ),
)
def test_colour_projected_counterterms_from_recursion_equal_sites(
    loop_sm_compiled_model, loop_native, process, accuracy
):
    """O6 at LC and NLC with the colour declaration (CT level): the
    four-gluon R2 (the general-N_c asset) takes the d line's gluon at
    u u~ > d d~ g g, and multi-gluon currents at u u~ > g g g g, graded by
    N_c in the recursion's couplings and at the sites."""

    import json

    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    values = _counterterms_per_route(
        load_compiled_model(loop_sm_compiled_model),
        loop_native,
        process,
        4,
        _slice_point((0,) * 6),
        accuracy=accuracy,
        colour_declaration=declaration,
    )
    _assert_counterterm_routes_agree(values)


# Spin and Ward modes (observables, design §8.4) with the counterterms of
# either route: (process, Born orders, loop orders, spin legs, sources).
_OBSERVABLE_ROUTES = {
    "gg_gg-spin": ("g g > g g", 2, 4, ((3, 4),), {"spin": (3, 4)}),
    "uux_ggg-spin": ("u u~ > g g g", 3, 5, ((4,),), {"spin": (4,)}),
    "uux_ggg-ward": ("u u~ > g g g", 3, 5, (), {"ward": (3, (0, 1))}),
}


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", tuple(_OBSERVABLE_ROUTES))
def test_spin_and_ward_counterterms_from_recursion_equal_sites(
    loop_sm_compiled_model, loop_native, name
):
    """Observables x ct-recursion (slice DECISIONS 10:00 UTC): the CT program
    takes the spin sources on both sides and the Ward vector on the loop side
    (``CountertermProgram.point(sources=...)``), so with complex
    non-transverse spin sources, or the gauge vector i p_3 on the loop side
    only, the Born x CT interference of every component, sector and Laurent
    coefficient equals the site closure's within 1e-45 of the largest one,
    at 200 bits (the slice point, CT level: no reduction). In Ward mode the
    R2 part is nonzero (it cancels the bare loop's violation)."""

    import json

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.observables import CorrelatorConfig

    from .test_loop_v2_observables import FIXTURES, VECTORS, observable_builder

    process, born, loop, classes, sources = _OBSERVABLE_ROUTES[name]
    generate = observable_builder(loop_sm_compiled_model, loop_native)
    record = name.split("-")[0]
    fixture = json.loads((FIXTURES / "legacy" / f"{record}.json").read_text())
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    options = {}
    if "spin" in sources:
        options["spin"] = {leg: VECTORS[k] for k, leg in enumerate(sources["spin"])}
    else:
        options["ward"] = sources["ward"]
    values = {}
    for route in ("recursion", "sites"):
        observables = generate(
            process,
            born={"QCD": born, "QED": 0},
            loop={"QCD": loop, "QED": 0},
            correlators=CorrelatorConfig(spin_correlations=classes),
            ward="ward" in sources,
            counterterms=route,
        )
        evaluation = observables.point(
            point,
            scale,
            parameters=SLICE_PARAMETERS,
            precision=Precision.arbitrary(60),
            **options,
        )
        values[route] = {
            key: complex(value) for key, value in evaluation.counterterms.items()
        }
    mine, theirs = values["recursion"], values["sites"]
    scale_ = max(abs(value) for value in theirs.values())
    assert scale_ > 0
    for key in mine.keys() | theirs.keys():
        difference = abs(mine.get(key, 0) - theirs.get(key, 0))
        assert difference <= 1e-45 * scale_, (key, mine.get(key), theirs.get(key))
    assert any(abs(v) > 1e-6 * scale_ for (c, _s, _p), v in theirs.items() if c == "r2")
