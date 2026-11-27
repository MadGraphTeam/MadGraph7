# SPDX-License-Identifier: 0BSD
"""The shipped N_c continuation of loop_sm (M-QCD MQ-NC; design §6.7).

``models.loop_color_declarations`` ships the formal-N_c data of MG5_aMC
3.7.0's loop_sm keyed by structural vertex digests; LC and NLC requests on
loop_sm take it when they name no declaration. Checked here:

* coverage: every counterterm vertex (103; the source's 104th, ``V_R24G``,
  is commented out) and every external-field factor of loop_sm resolves,
  under every shipped restriction, also through the checked scalar-arity
  view of the tests' compiled model; the four-gluon R2 by its whole-rule
  recipes (both loop groups), every other vertex by the identities;
* the literature colour factor of every coupling: the source coupling with
  the declared identities is the entry's factor (N_c, C_A, C_F or T_F, from
  the literature, not from the source) times an N_c-independent rest in
  every branch of its conditionals, so no N_c = 3 number hides in it (a
  control with C_F written as 4/3 fails the check);
* the digest is structural: a vertex renamed or moved keeps its entry, one
  whose coupling changes loses it; the data resolves only for a source that
  states its loop scheme (shipped profile or model metadata);
* the guard (``colour-declaration-coverage``): an LC/NLC request whose
  counterterm program can insert an uncovered vertex is refused, naming it;
  an uncovered vertex no tree of the request holds is left alone, and so is
  every vertex at full colour;
* LC and NLC of a request that names no declaration equal those with the
  example declaration bitwise, and the LC/NLC smoke passes on every shipped
  restriction (nightly).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "examples/loop/source_colour/loop_sm_qcd_subset.json"
RESTRICTIONS = (
    "default",
    "no_b_mass",
    "no_widths",
    "no_masses",
    "no_tau_mass",
    "lepton_masses",
    "c_mass",
    "ckm",
    "zeromass_ckm",
)
REFUSAL = "colour-declaration-coverage"


@pytest.fixture(scope="module")
def raw_models(loop_sm_ufo, tmp_path_factory):
    """loop_sm's raw source per shipped restriction (compiled once each)."""

    from pyamplicol.models.loading import compile_model_source

    cache = tmp_path_factory.mktemp("nc-declaration-models")
    found = {}

    def raw(restriction: str = "default"):
        if restriction not in found:
            found[restriction] = compile_model_source(
                loop_sm_ufo, restriction=restriction, cache_dir=cache
            ).raw_loop_model
        return found[restriction]

    return raw


def _subset():
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    return LoopColorDeclaration.from_dict(json.loads(SUBSET.read_text()))


# ------------------------------------------------------------------ coverage


@pytest.mark.loop_fast
@pytest.mark.parametrize("restriction", RESTRICTIONS)
def test_shipped_data_covers_every_loop_sm_counterterm_vertex(raw_models, restriction):
    """Every counterterm vertex and field factor of loop_sm resolves under
    each shipped restriction (same digests: a restriction lives in the
    metadata); the declaration lists every counterterm kernel but those of
    the four-gluon R2, which its whole-rule recipes cover (both loop groups)."""

    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )
    from pyamplicol.models.loop_color_declarations import colour_coverage

    raw = raw_models(restriction)
    vertices = raw.collections["counterterm_vertices"]
    assert len(vertices) == 103
    (coverage,) = colour_coverage(raw)
    assert coverage.uncovered_vertices == () and coverage.uncovered_fields == ()
    declaration = resolve_loop_color_declaration(raw)
    assert declaration is not None
    whole = {
        index
        for index, data in coverage.vertices.items()
        if data.continuation == "whole-rule"
    }
    assert [vertices[i]["name"] for i in whole] == ["V_R2RGA"]
    (four,) = whole
    groups = len(vertices[four]["fields"]["loop_particles"])
    assert sorted(r.loop_group_index for r in declaration.whole_rules) == list(
        range(groups)
    )
    assert {r.contribution for r in declaration.whole_rules} == {
        "gauge",
        "fundamental-dirac",
    }
    expected = {
        ("counterterm_vertices", vertex["id"], *key["items"])
        for vertex in vertices
        if vertex["id"] != four
        for key, _ref in vertex["fields"]["couplings"]["entries"]
    }
    assert set(declaration.kernel_keys) == expected
    assert len(declaration.field_keys) == 6
    # The example QCD subset is a subset of the shipped data, with the same
    # field keys and whole-rule recipes.
    subset = _subset()
    assert {
        key for key in subset.kernel_keys if key[0] == "counterterm_vertices"
    } <= set(declaration.kernel_keys)
    assert set(subset.field_keys) == set(declaration.field_keys)
    assert subset.whole_rules == declaration.whole_rules


@pytest.mark.loop_fast
def test_the_tests_scalar_arity_view_resolves_the_same_declaration(raw_models):
    """The checked four-G0 scalar-arity view of the tests' compiled loop_sm
    (a tree-vertex repair) keeps every counterterm digest."""

    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    raw = raw_models()
    view = prepare_loop_sm_scalar_declarations(raw).derived
    assert resolve_loop_color_declaration(view) == resolve_loop_color_declaration(raw)


def _with_vertex(raw, index, change):
    """``raw`` with counterterm vertex ``index``'s record changed."""

    from pyamplicol.models.loop_import import RawLoopModel

    payload = raw.to_dict()
    change(payload, payload["collections"]["counterterm_vertices"][index])
    return RawLoopModel.from_dict(payload)


@pytest.mark.loop_fast
def test_the_vertex_digest_is_structural(raw_models):
    """A vertex's digest ignores its name and the model's name; a changed
    coupling expression changes it (the vertex is then uncovered), and so
    does a changed parameter definition it reads."""

    from pyamplicol.models.loop_color_declarations import (
        colour_coverage,
        vertex_digest,
    )

    raw = raw_models()
    vertices = raw.collections["counterterm_vertices"]
    index = next(i for i, v in enumerate(vertices) if v["name"] == "V_R2GUU")

    def rename(payload, vertex):
        vertex["name"] = vertex["fields"]["name"] = "V_RENAMED"
        if "name" in payload:
            payload["name"] = "renamed_loop_model"

    renamed = _with_vertex(raw, index, rename)
    assert vertex_digest(
        renamed, renamed.collections["counterterm_vertices"][index]
    ) == (vertex_digest(raw, vertices[index]))
    (coverage,) = colour_coverage(renamed)
    assert coverage.uncovered_vertices == ()

    coupling = vertices[index]["fields"]["couplings"]["entries"][0][1]["id"]

    def change(payload, _vertex):
        record = payload["collections"]["couplings"][coupling]
        expression = record["fields"]["value"]["expression"]
        record["fields"]["value"]["expression"] = expression.replace(
            "((Ncol**2-1)/(2.0*Ncol))", "(4.0/3.0)"
        )
        assert record["fields"]["value"]["expression"] != expression

    changed = _with_vertex(raw, index, change)
    (coverage,) = colour_coverage(changed)
    # R2_GQQ is shared by the six q q~ g R2 vertices.
    uncovered = {vertices[i]["name"] for i in coverage.uncovered_vertices}
    assert "V_R2GUU" in uncovered and len(uncovered) == 6, uncovered


@pytest.mark.loop_fast
def test_the_shipped_data_needs_the_sources_own_loop_scheme(raw_models):
    """The shipped data resolves only for a source stating its scheme: the
    shipped [loop.model] profile of stock loop_sm, or the loop declaration in
    the model's metadata (the public route attaches the effective one). A
    model that shares vertex digests but states another scheme, or none (a
    changed vertex leaves the profile's digest), gets no declaration, so LC
    and NLC refuse it for want of colour data (e.g. loop_qcd_qed_sm, which
    shares 3 of its 499 counterterm vertices with loop_sm)."""

    from dataclasses import replace as replace_scheme

    from pyamplicol.loops.shipped_profiles import shipped_loop_profile
    from pyamplicol.models.loop_color_declarations import (
        colour_coverage,
        resolve_shipped_declaration,
    )
    from pyamplicol.models.loop_declaration import with_loop_model_declaration

    raw = raw_models()
    stock = resolve_shipped_declaration(raw)
    assert stock is not None
    declared = shipped_loop_profile(raw).declaration
    assert (
        resolve_shipped_declaration(with_loop_model_declaration(raw, declared)) == stock
    )
    other = replace(
        declared,
        scheme=replace_scheme(declared.scheme, model_convention="another model"),
    )
    assert resolve_shipped_declaration(with_loop_model_declaration(raw, other)) is None

    vertices = raw.collections["counterterm_vertices"]
    index = next(i for i, v in enumerate(vertices) if v["name"] == "V_R2GUU")

    def rename(payload, vertex):
        vertex["name"] = vertex["fields"]["name"] = "V_RENAMED"

    # A renamed vertex changes the declarations digest (no shipped profile)
    # but no vertex digest: covered, yet without a stated scheme unresolved.
    renamed = _with_vertex(raw, index, rename)
    assert shipped_loop_profile(renamed) is None
    (coverage,) = colour_coverage(renamed)
    assert coverage.uncovered_vertices == ()
    assert resolve_shipped_declaration(renamed) is None
    assert (
        resolve_shipped_declaration(with_loop_model_declaration(renamed, declared))
        == stock
    )


# ------------------------------------------------- literature colour factors


def _nc_check(raw):
    """``check(expression text, factor) -> bool``: the source expression with
    the counterterm parameters inlined and the shipped identities is the
    literature factor times an N_c-independent rest, in every branch of its
    conditionals (module docstring)."""

    from symbolica import E, S

    from pyamplicol._internal.physics.symbols import symbols as registry
    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.models.loop_color_declarations import shipped_colour_data

    (data,) = shipped_colour_data()
    nc, eps = S("mqnc_check_N", "mqnc_check_eps")
    ufo = {
        record["name"]: record for record in raw.collections["counterterm_parameters"]
    }
    identities = {
        registry.qualified(f"UFO::{name}"): parse_exact_ufo(text).replace(
            registry.qualified(f"UFO::{data.formal_symbol}"), nc
        )
        for name, text in data.parameter_identities.items()
    }
    factors = {
        "Nc": nc,
        "CA": nc,
        "CF": (nc**2 - 1) / (2 * nc),
        "TF": E("1/2"),
    }
    (branch,) = (
        s
        for s in parse_exact_ufo("cond(MT,0,1)").get_all_symbols(True)
        if s.get_name() == "symbolica::if"
    )
    x_, y_, z_ = S("x_", "y_", "z_")

    def inline(expression):
        for _ in range(16):
            symbols = {s.get_name() for s in expression.get_all_symbols(False)}
            hits = [name for name in ufo if f"UFO::{name}" in symbols]
            if not hits:
                return expression
            for name in hits:
                terms = ufo[name]["fields"]["value"]["terms"]
                value = sum(
                    (parse_exact_ufo(str(t)) * eps ** int(p) for p, t in terms),
                    E("0"),
                )
                expression = expression.replace(
                    registry.qualified(f"UFO::{name}"), value
                )
        raise AssertionError("counterterm parameters do not resolve")

    def check(text: str, factor: str) -> bool:
        continued = inline(parse_exact_ufo(text))
        for symbol, value in identities.items():
            continued = continued.replace(symbol, value)
        expected = factors[factor]
        for pick in (y_, z_):
            value = continued
            for _ in range(4):
                value = value.replace(branch(x_, y_, z_), pick)
            rest = value * expected.replace(nc, 3) - value.replace(nc, 3) * expected
            if rest.expand().together().expand() != 0:
                return False
        return True

    return check


@pytest.mark.loop_fast
def test_every_shipped_coupling_carries_its_literature_colour_factor(raw_models):
    """Each coupling entry of every identity-continued vertex and each field
    factor of loop_sm is its entry's literature colour factor (N_c, C_A,
    C_F or T_F) times an N_c-independent rest (module docstring); C_F
    written as the number 4/3 fails the check."""

    from pyamplicol.models.loop_color_declarations import (
        colour_coverage,
        field_factors,
    )

    raw = raw_models()
    check = _nc_check(raw)
    (coverage,) = colour_coverage(raw)
    couplings = raw.collections["couplings"]
    checked = 0
    for index, data in coverage.vertices.items():
        if data.continuation != "identities":
            continue
        vertex = raw.collections["counterterm_vertices"][index]
        for key, reference in vertex["fields"]["couplings"]["entries"]:
            indices = ",".join(map(str, key["items"]))
            value = couplings[reference["id"]]["fields"]["value"]
            texts = (
                [value["expression"]]
                if value["kind"] == "expression"
                else [text for _power, text in value["terms"]]
            )
            for text in texts:
                assert check(text, data.factors[indices]), (data.label, indices, text)
            checked += 1
    assert checked == 198
    for factor in field_factors(raw):
        entry = coverage.fields[factor.key]
        particle = raw.collections["particles"][factor.key[0]]["fields"]
        for key, value in particle["counterterm"]["entries"]:
            if key["items"][-1] != factor.key[2]:
                continue
            for _power, text in value["entries"]:
                assert check(str(text), entry.factor), (factor.label, text)
    # The control: the q q~ g R2 with C_F as a number.
    hidden = "-complex(0,1)*G**3/(16.0*cmath.pi**2)*(4.0/3.0)*(1.0+lhv)"
    assert not check(hidden, "CF")
    assert check(hidden.replace("(4.0/3.0)", "((Ncol**2-1)/(2.0*Ncol))"), "CF")


# --------------------------------------------------------- generation guard


def _request(process, born, accuracy, **changes):
    from pyamplicol.loops.v2.assemble import SliceRequest

    return SliceRequest(
        process,
        born,
        {**born, "QCD": born["QCD"] + 2},
        accuracy=accuracy,
        colour_contraction="direct",
        **changes,
    )


@pytest.fixture(scope="module")
def loop_sm(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


def _values(result):
    return (
        {repr(k): repr(v) for k, v in result.born.items()},
        {
            component: {
                repr(sector): [repr(c) for c in laurent.coefficients]
                for sector, laurent in sectors.items()
            }
            for component, sectors in result.components.items()
        },
    )


POINT_GG = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "300", "0", "400"],
    ["500", "-300", "0", "-400"],
]


@pytest.mark.loop_physics
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_projected_request_without_a_declaration_takes_the_shipped_one(
    loop_sm, loop_native, accuracy
):
    """g g > g g at LC and NLC naming no declaration resolves loop_sm's
    shipped one and equals the example declaration's evaluation bitwise."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, resolve_request
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    born = {"QCD": 2, "QED": 0}
    request = _request("g g > g g", born, accuracy)
    resolved = resolve_request(loop_sm, request)
    assert resolved.colour_declaration == resolve_loop_color_declaration(
        loop_sm.raw_loop_model
    )
    values = []
    for declaration in (None, _subset()):
        slice_ = LoopSlice.generate(
            loop_sm, replace(request, colour_declaration=declaration), loop_native
        )
        assert slice_.request.colour_declaration is not None
        values.append(
            _values(
                slice_.evaluate(
                    POINT_GG,
                    "1000000",
                    parameters={"aS": "0.118"},
                    precision=Precision.double(),
                )
            )
        )
    assert values[0] == values[1]


@pytest.mark.loop_physics
def test_an_uncovered_reachable_vertex_is_refused(loop_sm, loop_native):
    """With the example QCD subset, u u~ > e- e+ g at NLC can insert the
    closed-quark-loop R2 of g g Z, which the subset does not cover: refused
    (``colour-declaration-coverage``, naming it); the shipped declaration
    covers it; full colour needs none. u u~ > d d~ g (pure QCD, the subset's
    scope) does not reach it."""

    from pyamplicol.loops.v2.assemble import LoopSlice, prepare_request_model
    from pyamplicol.loops.v2.refusals import (
        LoopRequestUnsupported,
        listed_refusal_keys,
    )

    assert REFUSAL in listed_refusal_keys()

    born = {"QCD": 1, "QED": 2}
    with pytest.raises(LoopRequestUnsupported, match=REFUSAL) as refused:
        LoopSlice.generate(
            loop_sm,
            _request("u u~ > e- e+ g", born, "nlc", colour_declaration=_subset()),
            loop_native,
        )
    assert refused.value.key == REFUSAL
    assert "[G G Z]" in str(refused.value)
    # The guard reads the CT program, not the model preparation.
    prepare_request_model(
        loop_sm, _request("u u~ > e- e+ g", born, "nlc", colour_declaration=_subset())
    )
    # Full colour reads the source couplings at N_c = 3: a declaration that
    # does not cover a reached vertex is no reason to refuse it.
    LoopSlice.generate(
        loop_sm,
        _request("u u~ > e- e+ g", born, "full", colour_declaration=_subset()),
        loop_native,
    )
    LoopSlice.generate(
        loop_sm,
        _request(
            "u u~ > d d~ g", {"QCD": 3, "QED": 0}, "lc", colour_declaration=_subset()
        ),
        loop_native,
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("restriction", RESTRICTIONS)
def test_lc_and_nlc_smoke_on_every_shipped_restriction(
    loop_sm_ufo, loop_native, tmp_path, restriction
):
    """LC and NLC of g g > g g (the heavy-quark loops' mass-dependent UV and
    gluon field counterterms) and u u~ > e- e+ g (electroweak legs: the
    closed quark loops of g g Z and g g A) on each shipped restriction of
    the stock loop_sm, naming no declaration: generated with the shipped one
    and evaluated to finite values."""

    import math

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference import exact_points
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    model = compile_model_source(
        loop_sm_ufo, restriction=restriction, cache_dir=tmp_path / "cache"
    )
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    for process, born in (
        ("g g > g g", {"QCD": 2, "QED": 0}),
        ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}),
    ):
        legs = len(process.replace(">", "").split())
        point = exact_points.generate([0] * legs, count=1)[0]
        point = [[exact_points.decimal_string(x) for x in p] for p in point]
        for accuracy in ("lc", "nlc"):
            slice_ = LoopSlice.generate(
                model, _request(process, born, accuracy), loop_native
            )
            assert slice_.request.colour_declaration is not None
            result = slice_.evaluate(
                point,
                "1000000",
                parameters={"aS": "0.118"},
                precision=Precision.double(),
            )
            values = [
                c
                for laurent in result.components["virtual"].values()
                for c in laurent.coefficients
            ]
            assert values and all(math.isfinite(float(c)) for c in values)


@pytest.mark.loop_fast
def test_shipped_data_scheme_is_the_shipped_loop_model_profile():
    """The shipped continuation's scheme is the shipped [loop.model]
    profile's (``loops.shipped_profiles.LOOP_SM_SCHEME``): a projected request
    refuses a continuation whose scheme differs from the effective one, so a
    drift would refuse LC and NLC on stock loop_sm."""

    from dataclasses import asdict

    from pyamplicol.loops.shipped_profiles import LOOP_SM_SCHEME

    data = json.loads(
        (
            ROOT / "src/pyamplicol/models/loop_color_declarations/loop_sm.json"
        ).read_text()
    )
    assert data["scheme"] == asdict(LOOP_SM_SCHEME)


@pytest.mark.loop_physics
def test_public_route_lc_on_loop_sm_needs_no_declaration(loop_native, tmp_path):
    """The public route on the shipped loop_sm (``model.source = "loop_sm"``,
    the wheel's copy, its default restriction with MB = 4.7): an LC card with
    no [loop.model], no colour declaration and no fixed parameters plans and
    generates g g > g g (the shipped [loop.model] profile and the shipped
    colour declaration), and the stored process evaluates at LC."""

    from dataclasses import asdict

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet, Runtime
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.shipped_profiles import LOOP_SM_SCHEME
    from pyamplicol.models.loading import compile_model_source

    output = tmp_path / "gg-lc"
    card = tmp_path / "card.toml"
    card.write_text(
        f"""
action = "generate"
[model]
source = "loop_sm"
[process]
entries = [{{expression="g g > g g", name="gg"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "lc"
contraction = "direct"
[loop]
r2_strategy = "model"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""
    )
    # The dry run compiles no model source: populate the model cache first.
    compile_model_source("loop_sm", restriction="default")
    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    generator = Generator(resolve_config(card))
    plan = generator.plan(processes)
    assert not plan.unsupported_features, plan.unsupported_features
    coverage = plan.estimated_coverage
    assert REFUSAL in " ".join(coverage["loop_admission_refusals"])
    assert coverage["model_declaration"]["scheme"] == asdict(LOOP_SM_SCHEME)
    generator.generate(processes, output)
    runtime = Runtime.load(output)
    result = runtime.evaluate([POINT_GG], mu_r2="1000000")
    assert result.color_accuracy == "lc"
