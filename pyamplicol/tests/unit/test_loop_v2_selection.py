# SPDX-License-Identifier: 0BSD
"""Leading-colour helicity and colour-flow selection of v2 virtuals (MQ-SELECT).

``loops.v2.selection``; m-qcd PLAN §3.2 MQ-SELECT and slice-status DECISIONS
2026-10-28 ~15:50/~15:55. Oracles:

* exact identities: the selections of a complete set of colour flows (and of
  helicity rows) sum to the inclusive leading-colour value (Born and every
  Laurent coefficient), Python and native, binary64 and double-double; a
  selection of one helicity row and one flow equals its ``evaluate_resolved``
  entry;
* a generation-time selection (``SliceRequest.selection``; the public
  route's ``[process]`` selectors) prunes the recursion (fewer skeletons and
  Born rows) and equals the runtime selection of the inclusive slice; the
  ring-aware pruning is held on g g > g g g, where per-reading liveness
  loses contributions;
* the tree engine: the helicity and flow IDs and the flow order are the tree
  runtime's (LC Born per helicity and flow proportional to the tree
  engine's built-in SM values, zeros at the same entries);
* the public route: a selected artifact equals the inclusive artifact's
  runtime selection; NLC and full colour keep refusing selectors; requests
  without a selection keep their request records (no ``selection`` entry);
* MadLoop: the helicity IDs are MadLoop's helicity configurations, the
  full-colour rows equal ``SLOOPMATRIXHEL`` (MQ-REF's per-helicity
  references) at 1e-12 in double-double.

Exhaustive configurations (2->4 complete sets, a 2->5 row) run nightly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.loops.v2 import selection as lc
from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
PARAMETERS = {"aS": "0.118", "MT": "173"}
ALPHA_S = float(PARAMETERS["aS"])
DOUBLE = 1e-12
DOUBLE_DOUBLE = Decimal("1e-24")
# Born and loop QCD orders by external leg count (QED 0).
QCD = {
    legs: ({"QCD": legs - 2, "QED": 0}, {"QCD": legs, "QED": 0})
    for legs in (4, 5, 6, 7)
}


def _points(name: str, count: int = 1, start: int = 0):
    fixture = json.loads((LEGACY / f"{name}.json").read_text())
    stop = start + count
    return fixture["points"][start:stop], fixture["scales_squared"][start:stop]


def _options() -> dict:
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    return {
        "fixed_parameters": legacy_capture.FIXED_PARAMETERS,
        "accuracy": "lc",
        "colour_declaration": LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        ),
    }


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    """``ObservableSlice`` of an LC request (with an optional selection),
    generated once per module."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict = {}

    def build(process: str, selection=None):
        key = (process, selection)
        if key not in built:
            born, loop = QCD[len(process.split()) - 1]
            request = SliceRequest(
                process, born, loop, selection=selection, **_options()
            )
            built[key] = ObservableSlice.of(
                LoopSlice.generate(compiled, request, loop_native)
            )
        return built[key]

    return build


def _runtime(observables, executor="native"):
    from pyamplicol.runtime.virtual import V2ObservablesRuntime

    return V2ObservablesRuntime(
        observables, executor=executor, model_parameters=PARAMETERS
    )


def _values(result, index=0) -> list:
    """Born then the Laurent coefficients of the one sector at a point."""

    ((laurent,),) = (result.values[index],)
    ((born,),) = (result.born_values[index],)
    return [born, *laurent]


def _close(actual, expected, tolerance, born=None) -> None:
    """Per coefficient (m-qcd PLAN §1.4): rel(a, b) = |a - b| / max(|a|, |b|)
    <= ``tolerance``, strictly relative; with the point's inclusive Born
    ``born`` a coefficient that vanishes identically (both below ``tolerance``
    F_s, F_s = |born| alpha_s/2pi) is held absolutely to ``tolerance`` F_s.
    Binary64 or ``Decimal`` values."""

    if isinstance(tolerance, Decimal):
        actual, expected = [Decimal(x) for x in actual], [Decimal(x) for x in expected]
        born = None if born is None else Decimal(born)
        factor = Decimal(ALPHA_S) / (2 * Decimal(math.pi))
    else:
        factor = ALPHA_S / (2 * math.pi)
    floor = 0 if born is None else tolerance * abs(born) * factor
    for a, b in zip(actual, expected, strict=True):
        size = max(abs(a), abs(b))
        if size <= floor:
            continue
        assert abs(a - b) <= tolerance * size, (actual, expected, a, b)


def _summed(results) -> list:
    out = None
    for result in results:
        values = _values(result)
        out = (
            values if out is None else [a + b for a, b in zip(out, values, strict=True)]
        )
    return out


# ------------------------------------------------------------------ fast checks


@pytest.mark.loop_fast
def test_selection_records_catalogue_resolution_and_refusals():
    """Selections normalise and round-trip through the request record, which
    holds them only when set; selectors resolve by tree ID, runtime ID and
    ordinal and refuse unknown ones; NLC and full colour refuse them."""

    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.runtime.virtual import v2_options

    selection = lc.Selection(((5, 1), (1, -1)), ((2, 4, 3, 5, 1),))
    assert selection.helicities == ((1, -1), (5, 1))
    assert lc.Selection.from_record(selection.record()) == selection
    assert set(selection.record()) == {"helicities", "flows"} and selection.restricts
    # The tree engine's plan options order the flows and select nothing.
    ordered = lc.Selection((), None, [2, 4, 3, 5, 1], None)
    assert ordered.record()["color_order"] == [2, 4, 3, 5, 1]
    assert lc.Selection.from_record(ordered.record()) == ordered
    assert not ordered.restricts
    capped = lc.Selection((), ((2, 1, 3, 5, 4),), None, 2)
    assert lc.Selection.from_record(json.loads(json.dumps(capped.record()))) == capped
    with pytest.raises(ValueError, match="one helicity per leg"):
        lc.Selection(((1, -1), (1, 1)))
    with pytest.raises(ValueError, match="at least one flow"):
        lc.Selection((), ())
    plain = SliceRequest("u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}, accuracy="lc")
    assert "selection" not in request_record(plain)
    assert request_from_record(request_record(plain)) == plain
    chosen = replace(plain, selection=selection)
    record = request_record(chosen)
    assert record["selection"] == selection.record()
    assert request_from_record(json.loads(json.dumps(record))) == chosen
    for accuracy in ("full", "nlc"):
        with pytest.raises(ValueError, match="leading-colour option"):
            SliceRequest(
                "g g > g g",
                {"QCD": 2},
                {"QCD": 4},
                accuracy=accuracy,
                selection=selection,
            )
        with pytest.raises(CompatibilityError, match="leading colour"):
            lc.require_leading_colour(accuracy, ["h:-1,-1,+1,+1"], None)
    with pytest.raises(CompatibilityError, match=r"at next-to-leading colour.*without"):
        lc.require_leading_colour("nlc", None, ["1"], resolved=True)
    lc.require_leading_colour("full", None, None)
    # What evaluate / evaluate_resolved do not take (the correlated ones).
    with pytest.raises(CompatibilityError, match=r"correlated.*evaluate_resolved"):
        v2_options(helicities=["h:-1,-1,+1,+1"])
    rows = (((0, -1), (1, 1)), ((0, 1), (1, -1)))
    flows = (
        lc.Flow("flow:2,1", (2, 1), 1, 0),
        lc.Flow("flow:1,2", (1, 2), None, 1),
        lc.Flow("flow:3,4", (3, 4), 0, 2),
    )
    assert lc.Flow.from_record(json.loads(json.dumps(flows[1].record()))) == flows[1]
    catalogue = lc.Catalogue(rows, flows)
    assert lc.helicity_id(rows[0]) == "h:-1,+1"
    assert catalogue.resolve(["h:+1,-1"], ["1", "flow:3,4"]) == (
        frozenset({rows[1]}),
        frozenset({0, 1}),
    )
    assert catalogue.resolve(["h0"], None) == (frozenset({rows[0]}), None)
    with pytest.raises(EvaluationError, match="unknown helicity"):
        catalogue.resolve(["h:+1,+1"], None)
    with pytest.raises(EvaluationError, match="out of range"):
        catalogue.resolve(None, ["4"])
    with pytest.raises(EvaluationError, match="unknown colour-flow"):
        catalogue.resolve(None, ["flow:9"])
    with pytest.raises(EvaluationError, match="no Born amplitude"):
        catalogue.resolve(None, ["flow:1,2"])


@pytest.mark.loop_fast
def test_the_selection_cache_keeps_the_most_recently_used(monkeypatch):
    """Evaluation-time selections are kept least-recently-used first out:
    a hit forms nothing, the oldest entry is dropped beyond the size, size 0
    keeps none; ``$PYAMPLICOL_LOOP_SELECTION_CACHE`` sets the default count
    and refuses anything but an integer >= 0."""

    built = []

    def build(key):
        return lambda: built.append(key) or [key]

    cache = lc.SelectionCache(2)
    first = cache.get("a", build("a"))
    assert cache.get("a", build("a")) is first and built == ["a"]
    cache.get("b", build("b"))
    cache.get("a", build("a"))  # "b" is now the least recently used
    cache.get("c", build("c"))
    assert [key for key, _value in cache.items()] == ["a", "c"]
    assert built == ["a", "b", "c"] and len(cache) == 2
    none = lc.SelectionCache(0)
    none.get("a", build("a"))
    none.get("a", build("a"))
    assert len(none) == 0 and built[-2:] == ["a", "a"]
    monkeypatch.delenv(lc.SELECTION_CACHE_ENVIRONMENT, raising=False)
    assert lc.SelectionCache().size == lc.SELECTION_CACHE_DEFAULT
    monkeypatch.setenv(lc.SELECTION_CACHE_ENVIRONMENT, " 3 ")
    assert lc.SelectionCache().size == 3
    for text in ("-1", "x", "2.5"):
        monkeypatch.setenv(lc.SELECTION_CACHE_ENVIRONMENT, text)
        with pytest.raises(EvaluationError, match="integer >= 0"):
            lc.SelectionCache()


# ------------------------------------------------------------------ identities


def assert_complete_sets(
    observables,
    points,
    *,
    executors=("python", "native"),
    groups=None,
    double_double=True,
    resolved=True,
) -> None:
    """Complete sets of selections sum to the inclusive value: the flows
    (``groups`` of flow ordinals, default one each), and the helicity rows
    split in two; in binary64 for each executor and in double-double
    natively; one (row, flow) equals its resolved entry."""

    pts, scales = points
    for executor in executors:
        runtime = _runtime(observables, executor)
        catalogue = runtime.selection_catalogue()
        inclusive = _values(runtime.evaluate(pts, mu_r2=scales))
        partition = groups or [[str(k + 1)] for k in range(len(catalogue.flows))]
        _close(
            _summed(
                runtime.evaluate(pts, mu_r2=scales, color_flows=g) for g in partition
            ),
            inclusive,
            DOUBLE,
        )
        ids = [lc.helicity_id(row) for row in catalogue.rows]
        half = len(ids) // 2
        _close(
            _summed(
                runtime.evaluate(pts, mu_r2=scales, helicities=part)
                for part in (ids[:half], ids[half:])
            ),
            inclusive,
            DOUBLE,
        )
        if not resolved:
            continue
        flow = next(f for f in catalogue.flows if f.basis is not None)
        row = catalogue.rows[len(catalogue.rows) // 3]
        resolved = runtime.evaluate_resolved(pts, mu_r2=scales)
        entry = resolved.values[0][0][catalogue.rows.index(row)][
            resolved.virtual_color_ids.index(f"basis-{flow.basis}")
        ]
        one = runtime.evaluate(
            pts, mu_r2=scales, helicities=[lc.helicity_id(row)], color_flows=[flow.id]
        )
        _close(list(one.values[0][0]), list(entry), DOUBLE, born=inclusive[0])
    if not double_double:
        return
    runtime = _runtime(observables, "native")
    flows = runtime.selection_catalogue().flows
    partition = groups or [[flow.id] for flow in flows]
    inclusive = _values(runtime.evaluate(pts, mu_r2=scales, arithmetic="double-double"))
    total = _summed(
        runtime.evaluate(pts, mu_r2=scales, color_flows=g, arithmetic="double-double")
        for g in partition
    )
    _close(total, inclusive, DOUBLE_DOUBLE)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "points"),
    [("g g > g g", "gg_gg"), ("u u~ > d d~ g", "uux_ddxg")],
)
def test_runtime_selections_partition_the_inclusive_value(generate, process, points):
    """Pure gluon and quark lines: complete sets of flows and of helicity
    rows sum to the inclusive LC value, Python and native, binary64 and
    double-double; a (row, flow) selection is its resolved entry."""

    assert_complete_sets(generate(process), _points(points))


def _thirds(count: int) -> list[list[str]]:
    """A complete set of three flow groups (ordinals)."""

    ordinals = [str(k + 1) for k in range(count)]
    return [ordinals[k::3] for k in range(3)]


@pytest.mark.loop_physics
@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("process", "points", "start", "executors", "resolved"),
    [
        ("u u~ > g g g", "uux_ggg", 0, ("python", "native"), True),
        # The first point is outside binary64's 1e-12 for g g > g g g (LC
        # estimate 1.7e-9, true error 1.6e-10 against double-double; the
        # binary64 flow groups sum to it within 8.4e-11), the second at its
        # edge (estimate 1.1e-12); the third has estimate 7.9e-14 (every
        # flow group's below 1.1e-13) and true error 6.1e-15.
        ("g g > g g g", "uux_ggg", 2, ("native",), True),
    ],
)
def test_runtime_selections_partition_the_inclusive_value_nightly(
    generate, process, points, start, executors, resolved
):
    """Mixed and pure-gluon 2->3 rows of the Python and native slices (three
    flow groups and the two helicity halves as complete sets; binary64 and
    double-double); 2->4 and 2->5 run on their stored artifacts
    (``test_stored_selections_partition_the_inclusive_value``)."""

    observables = generate(process)
    flows = _runtime(observables).selection_catalogue().flows
    assert_complete_sets(
        observables,
        _points(points, start=start),
        executors=executors,
        groups=_thirds(len(flows)),
        resolved=resolved,
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "points", "flow", "helicities"),
    [
        ("u u~ > d d~ g", "uux_ddxg", (2, 4, 3, 5, 1), ((1, -1), (5, 1))),
        ("g g > g g g", "uux_ggg", (1, 2, 4, 5, 3), ()),
        # Identical-flavour quark lines: Born roots of mixed N_c grades,
        # graded by the slice's own grade split.
        ("u u~ > u u~ g", "uux_uuxg", (2, 4, 3, 5, 1), ()),
    ],
)
def test_generation_selection_prunes_and_equals_the_runtime_selection(
    generate, process, points, flow, helicities
):
    """A generation-time selection keeps fewer skeletons (and Born rows),
    keeps the process's leading N_c degrees and equals the runtime selection
    of the inclusive slice, Python and native, binary64 and (natively)
    double-double. g g > g g g holds the ring-aware pruning (per-reading
    liveness lost 40 % of the double pole); u u~ > u u~ g identical-flavour
    lines (its 60 of 196 skeletons)."""

    from pyamplicol.loops.v2 import reference_executor as rx

    inclusive = generate(process)
    chosen = generate(process, lc.Selection(helicities, (flow,)))
    assert len(rx.reading_skeletons(chosen.slice.plan)) < len(
        rx.reading_skeletons(inclusive.slice.plan)
    )
    assert len(chosen.slice.basis) == 1
    assert chosen.slice.degrees() == inclusive.slice.degrees()
    fixed = dict(helicities)
    rows = [
        r
        for r in inclusive.rows
        if all(dict(r)[leg - 1] == h for leg, h in fixed.items())
    ]
    assert len(chosen.rows) == len(rows) < len(inclusive.rows) or not helicities
    pts, scales = _points(points)
    selectors = {
        "helicities": [lc.helicity_id(row) for row in rows] if helicities else None,
        "color_flows": [lc.flow_id(flow)],
    }
    reference = _runtime(inclusive).evaluate(pts, mu_r2=scales, **selectors)
    for executor in ("python", "native"):
        value = _runtime(chosen, executor).evaluate(pts, mu_r2=scales)
        _close(_values(value), _values(reference), DOUBLE)
    reference = _runtime(inclusive).evaluate(
        pts, mu_r2=scales, arithmetic="double-double", **selectors
    )
    value = _runtime(chosen).evaluate(pts, mu_r2=scales, arithmetic="double-double")
    _close(_values(value), _values(reference), DOUBLE_DOUBLE)


@pytest.mark.loop_physics
def test_a_flow_below_the_leading_colour_is_zero_and_refused_at_generation(
    generate, loop_sm_compiled_model
):
    """u u~ > d d~ g's tree flows joining each quark to its own antiquark are
    below the squared matrix element's leading power: zero when selected at
    evaluation; at generation refused on their own (listed
    ``subleading-colour-flows``), exact zeros next to a leading flow (the
    tree engine's ``max_color_sectors = 3``: two such flows and a leading
    one), whose selection equals the runtime selection of the three flows
    in double-double."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loading import load_compiled_model

    observables = generate("u u~ > d d~ g")
    runtime = _runtime(observables)
    pts, scales = _points("uux_ddxg")
    value = runtime.evaluate(pts, mu_r2=scales, color_flows=["flow:2,1,3,5,4"])
    assert _values(value) == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(LoopRequestUnsupported) as refused:
        generate("u u~ > d d~ g", lc.Selection((), ((2, 1, 3, 5, 4),)))
    assert refused.value.key == "subleading-colour-flows"
    assert "[[2, 1, 3, 5, 4]]" in str(refused.value)
    compiled = load_compiled_model(loop_sm_compiled_model)
    capped = _chosen(compiled, "u u~ > d d~ g", max_color_sectors=3)
    assert capped.flows == ((2, 1, 3, 5, 4), (2, 4, 3, 5, 1), (2, 5, 1, 3, 4))
    mixed = generate("u u~ > d d~ g", capped)
    assert mixed.slice.degrees() == observables.slice.degrees()
    chosen = _runtime(mixed)
    flows = [flow.id for flow in chosen.selection_catalogue().flows]
    # The tree plan's first three sectors, in its order.
    assert flows == [flow.id for flow in runtime.selection_catalogue().flows][:3]
    assert sorted(flows) == ["flow:2,1,3,5,4", "flow:2,4,3,5,1", "flow:2,5,1,3,4"]
    for flow in ("flow:2,1,3,5,4", "flow:2,5,1,3,4"):
        value = chosen.evaluate(pts, mu_r2=scales, color_flows=[flow])
        assert _values(value) == [0.0, 0.0, 0.0, 0.0]
    reference = runtime.evaluate(
        pts, mu_r2=scales, color_flows=flows, arithmetic="double-double"
    )
    value = chosen.evaluate(pts, mu_r2=scales, arithmetic="double-double")
    _close(_values(value), _values(reference), DOUBLE_DOUBLE)


@pytest.mark.loop_physics
def test_a_flow_interfering_between_born_orders_is_refused_at_generation(
    loop_sm_compiled_model, loop_native
):
    """u u~ > d d~ with QCD and photon Born amplitudes: at QCD^2 QED^2 the
    QCD flow (2,4,3,1) interferes at the leading power with the photon
    Born's (2,1,3,4). Its evaluation-time selection keeps that term (the
    flows sum to the inclusive value in double-double); a generation-time
    selection would drop it, so it is refused (listed
    ``interfering-colour-flows``, naming the unselected flow and the order),
    (2,1,3,4) alone is below the leading colour at QCD^4
    (``subleading-colour-flows``), and both flows together are the
    inclusive value."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)

    def generate(selection=None):
        request = SliceRequest(
            "u u~ > d d~",
            {"QCD": 2, "QED": 2},
            ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
            particle_veto=(23, 24, 25, 250, 251),
            selection=selection,
            # The shipped loop_sm colour declaration (it covers the photon's
            # counterterm vertices).
            **{**_options(), "colour_declaration": None},
        )
        return ObservableSlice.of(LoopSlice.generate(compiled, request, loop_native))

    inclusive = generate()
    runtime = _runtime(inclusive)
    pts, scales = _points("uudd")
    flows = [flow.id for flow in runtime.selection_catalogue().flows]
    assert sorted(flows) == ["flow:2,1,3,4", "flow:2,4,3,1"]

    def coefficients(result) -> list:
        return [
            *result.born_values[0],
            *(c for laurent in result.values[0] for c in laurent),
        ]

    expected = coefficients(
        runtime.evaluate(pts, mu_r2=scales, arithmetic="double-double")
    )
    parts = [
        coefficients(
            runtime.evaluate(
                pts, mu_r2=scales, color_flows=[flow], arithmetic="double-double"
            )
        )
        for flow in flows
    ]
    born = max(abs(x) for x in expected[:3])
    _close(list(map(sum, zip(*parts, strict=True))), expected, DOUBLE_DOUBLE, born)
    # The cross order's (QCD^2 QED^2, entry 0) Born of the QCD flow alone
    # is nonzero (the inclusive one vanishes at the leading power).
    cross = parts[flows.index("flow:2,4,3,1")][0]
    assert abs(cross) > Decimal("1e-4")
    with pytest.raises(LoopRequestUnsupported) as refused:
        generate(lc.Selection((), ((2, 4, 3, 1),)))
    assert refused.value.key == "interfering-colour-flows"
    assert "flows [2, 1, 3, 4] at {'QCD': 2, 'QED': 2}" in str(refused.value)
    with pytest.raises(LoopRequestUnsupported) as refused:
        generate(lc.Selection((), ((2, 1, 3, 4),)))
    assert refused.value.key == "subleading-colour-flows"
    both = generate(lc.Selection((), ((2, 1, 3, 4), (2, 4, 3, 1))))
    value = coefficients(
        _runtime(both).evaluate(pts, mu_r2=scales, arithmetic="double-double")
    )
    _close(value, expected, DOUBLE_DOUBLE, born)


@pytest.mark.loop_physics
def test_helicity_selections_keep_split_block_currents(
    loop_sm_ufo, loop_native, tmp_path
):
    """u u~ > w+ w- g at leading colour on its public artifact (stored
    path): the u current of legs {1, 3, 4} is two tree blocks of one name and
    chirality (flavour flows u and u d u, ``test_loop_v2_vv_jet.py``), which
    the stored native tree program adds per row (``block_row_terms``). A
    runtime helicity selection forms its column set from the stored resolved
    records and restricts that tree program to the state its kept rows read
    (``TreeProgram::set_columns``, ``loop_v2/trees_rows.rs``), the further
    terms of a read row included (``block_slots``): the two halves of the
    helicity rows sum to the inclusive value, and single helicities equal
    their resolved rows, in double-double at the first point of MadLoop's
    reference (``mqcd/uux_wpwmg_4fs.json``). Without the further terms the
    halves sum to 1.34 of the inclusive finite part."""

    from fractions import Fraction

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.artifact import read_native_process, read_plan

    process = "u u~ > w+ w- g"
    output = tmp_path / "artifact"
    card = tmp_path / "card.toml"
    card.write_text(
        _card(
            model=json.dumps(str(loop_sm_ufo)),
            output=json.dumps(str(output)),
            selectors="max_coupling_orders = {QED = 2}",
            process=json.dumps(process),
            fixed='{WT = "0", WZ = "0", WW = "0", WH = "0"}',
        )
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(process, name="p"),)), output
    )
    stored = output / "processes" / "p" / "loop-runtime.pacbin"
    trees = read_native_process(stored, read_plan(stored)).trees.program
    assert trees.get("block_row_terms")
    runtime = Runtime.load(output, process="p", model_parameters=PARAMETERS)
    reference = json.loads((MQCD / "uux_wpwmg_4fs.json").read_text())
    point = [reference["points"][0]]
    scale = [str(Fraction(reference["mu_r"][0]) ** 2)]
    helicities = [h.id for h in runtime.physics.helicities]
    half = len(helicities) // 2

    def values(**selectors) -> list:
        result = runtime.evaluate(
            point, mu_r2=scale, arithmetic="double-double", **selectors
        )
        return [
            *result.born_values[0],
            *(c for laurent in result.values[0] for c in laurent),
        ]

    expected = values()
    born = abs(expected[0])  # one squared Born order
    parts = [
        values(helicities=helicities[:half]),
        values(helicities=helicities[half:]),
    ]
    assert all(any(abs(x) > 0 for x in part) for part in parts)
    _close(list(map(sum, zip(*parts, strict=True))), expected, DOUBLE_DOUBLE, born)
    # Single helicities against the resolved rows (the stored inclusive
    # columns, no row restriction): the first nonzero row and the largest.
    resolved = runtime.evaluate_resolved(point, mu_r2=scale, arithmetic="double-double")
    assert list(resolved.helicity_ids) == helicities
    ((rows,),) = (resolved.values[0],)
    ((born_rows,),) = (resolved.born_values[0],)
    nonzero = [k for k, (entry,) in enumerate(rows) if entry.double_pole != 0]
    assert 0 < len(nonzero) < len(helicities)
    largest = max(nonzero, key=lambda k: abs(rows[k][0].finite))
    for k in sorted({nonzero[0], largest}):
        (entry,) = rows[k]
        (row_born,) = born_rows[k]
        _close(
            values(helicities=[helicities[k]]),
            [row_born, entry.finite, entry.single_pole, entry.double_pole],
            DOUBLE_DOUBLE,
            born,
        )


# ------------------------------------------------------- tree-engine parity


def _chosen(compiled, process: str, **selectors):
    """The public route's selection (``loop_service._v2_selection``) of the
    tree engine's ``[process]`` selectors."""

    from types import SimpleNamespace

    from pyamplicol.generation.loop_service import _SELECTORS, _v2_selection

    fields = dict.fromkeys(_SELECTORS)
    fields.update(selectors)
    backend = SimpleNamespace(_process_selection=SimpleNamespace(**fields))
    return _v2_selection(backend, compiled, process, "p")


def _toml(selectors: dict) -> str:
    return "".join(f"{key} = {json.dumps(value)}\n" for key, value in selectors.items())


def _tree_artifact(tmp: Path, process: str, selectors: str = "") -> Path:
    """A tree LC artifact of ``process`` on the built-in Standard Model (with
    ``[process]`` selector lines)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    tmp.mkdir(parents=True, exist_ok=True)
    card = tmp / "tree.toml"
    output = tmp / "tree"
    card.write_text(
        'action = "generate"\n[process]\n'
        f'entries = [{{expression={json.dumps(process)}, name="t"}}]\n'
        f'coupling_order_policy = "minimal"\n{selectors}\n[color]\n'
        'accuracy = "lc"\n'
        f"[generation]\noutput = {json.dumps(str(output))}\n"
        "emit_api_bundle = false\n[generation.validation]\n"
        "post_build_validation = false\n"
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(process, name="t"),)), output
    )
    return output


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "points"),
    [
        ("u u~ > d d~ g", "uux_ddxg"),
        pytest.param("g g > g g g", "uux_ggg", marks=pytest.mark.loop_nightly),
    ],
)
def test_helicity_and_flow_ids_are_the_tree_engines(
    generate, tmp_path, process, points
):
    """The flow IDs and their order are the tree runtime's (built-in SM, LC);
    per (helicity, flow) the loop's LC Born is the tree engine's up to one
    constant factor (couplings, and the tree's flows below the leading
    power, which the loop's LC Born drops): zeros at the same entries."""

    from pyamplicol import Runtime

    observables = generate(process)
    runtime = _runtime(observables)
    catalogue = runtime.selection_catalogue()
    tree = Runtime.load(_tree_artifact(tmp_path, process), process="t")
    assert [flow.id for flow in catalogue.flows] == list(tree.physics.color_ids)
    pts, scales = _points(points)
    floats = [[[float(x) for x in p] for p in point] for point in pts]
    ratios = []
    for flow in catalogue.flows[:6]:
        for row in catalogue.rows[::5]:
            ident = lc.helicity_id(row)
            assert ident in tree.physics.helicity_ids
            (value,) = tree.evaluate(floats, helicities=[ident], color_flows=[flow.id])
            expected = complex(value[0] if isinstance(value, tuple) else value).real
            (born,) = runtime.evaluate(
                pts, mu_r2=scales, helicities=[ident], color_flows=[flow.id]
            ).born_values[0]
            if born == 0 or expected == 0:
                assert abs(born) <= 1e-12 * abs(
                    _values(runtime.evaluate(pts, mu_r2=scales))[0]
                )
                continue
            ratios.append(born / expected)
    assert ratios and max(ratios) - min(ratios) <= 1e-10 * abs(ratios[0])


@pytest.mark.loop_physics
def test_public_colour_subset_selectors_are_the_tree_engines(
    loop_sm_compiled_model, tmp_path
):
    """The public route's colour subset selectors keep the tree engine's
    flows: ``selected_color_sector_ids`` those its leading-colour artifact
    with the same selector computes (a folded sector: its flow and the
    reflection), ``max_color_sectors`` and ``reference_color_order`` its
    plan's sectors; a missing sector id is refused as the tree engine
    refuses it; helicities alone keep every flow; no selector is no
    selection."""

    from functools import partial

    from pyamplicol import Runtime
    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    chosen = partial(_chosen, compiled)

    assert chosen("g g > g g g") is None
    assert chosen("g g > g g g", selected_source_helicities={1: -1}) == (
        lc.Selection(((1, -1),))
    )
    # Sector ids and the reference order: the tree engine's artifact with the
    # same selectors computes these flows and numbers them in this order.
    for k, (process, selectors) in enumerate(
        (
            ("g g > g g g", {"selected_color_sector_ids": [0, 7]}),
            ("u u~ > d d~ g", {"selected_color_sector_ids": [2, 1]}),
            ("g g > g g g", {"reference_color_order": [1, 3, 2, 4, 5]}),
            (
                "u u~ > d d~ g",
                {
                    "reference_color_order": [2, 4, 3, 5, 1],
                    "selected_color_sector_ids": [1, 3],
                },
            ),
        )
    ):
        selection = chosen(process, **selectors)
        legs = len(process.split()) - 1
        request = SliceRequest(process, *QCD[legs], accuracy="lc", selection=selection)
        tree = Runtime.load(
            _tree_artifact(tmp_path / str(k), process, _toml(selectors)),
            process="t",
        )
        assert [flow.id for flow in lc.request_flows(compiled, request, ())] == list(
            tree.physics.color_ids
        )
    # A truncated plan (the tree engine's recurrence route refuses it, its
    # plan keeps the reference sector and then enumerates up to the cap): the
    # reference sector (1 3 2 4 5, folded with its reflection) and the first
    # enumerated one (1 2 3 4 5 and its reflection).
    assert chosen(
        "g g > g g g", max_color_sectors=1, reference_color_order=(1, 3, 2, 4, 5)
    ).flows == ((1, 2, 3, 4, 5), (1, 3, 2, 4, 5), (1, 5, 4, 2, 3), (1, 5, 4, 3, 2))
    assert chosen("u u~ > d d~ g", max_color_sectors=2).flows == (
        (2, 1, 3, 5, 4),
        (2, 5, 1, 3, 4),
    )
    with pytest.raises(GenerationError, match="did not materialize"):
        chosen("g g > g g g", selected_color_sector_ids=[99])
    # Selected helicities name legs and helicities the process has (the
    # tree engine's source states), before any generation.
    with pytest.raises(GenerationError, match=r"legs \[9\] the process does not"):
        chosen("u u~ > d d~ g", selected_source_helicities={9: 1})
    with pytest.raises(
        GenerationError,
        match=r"helicity 0 is unavailable for external leg 1 \(its helicities: "
        r"\[-1, 1\]\)",
    ):
        chosen("u u~ > d d~ g", selected_source_helicities={1: 0})


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "selectors",
    [
        {"selected_color_sector_ids": [2]},
        {"reference_color_order": [2, 4, 3, 5, 1]},
    ],
    ids=["sector-ids", "reference-order"],
)
def test_flow_ordinals_are_the_tree_engines_under_colour_selectors(
    generate, loop_sm_compiled_model, tmp_path, selectors
):
    """A generation with the tree engine's colour selectors lists and numbers
    its flows as the tree runtime of the same selectors does (sector ids: the
    flows it computes; ``reference_color_order``: its plan's order, which
    selects nothing): ordinal k is the tree runtime's k-th flow, whose value
    is the inclusive slice's selection of that flow ID."""

    from pyamplicol import Runtime
    from pyamplicol.models.loading import load_compiled_model

    process = "u u~ > d d~ g"
    selection = _chosen(
        load_compiled_model(loop_sm_compiled_model), process, **selectors
    )
    tree = Runtime.load(
        _tree_artifact(tmp_path, process, _toml(selectors)), process="t"
    )
    runtime = _runtime(generate(process, selection))
    assert [f.id for f in runtime.selection_catalogue().flows] == list(
        tree.physics.color_ids
    )
    inclusive = _runtime(generate(process))
    pts, scales = _points("uux_ddxg")
    born = _values(inclusive.evaluate(pts, mu_r2=scales))[0]
    for k, ident in enumerate(tree.physics.color_ids):
        _close(
            _values(runtime.evaluate(pts, mu_r2=scales, color_flows=[str(k + 1)])),
            _values(inclusive.evaluate(pts, mu_r2=scales, color_flows=[ident])),
            DOUBLE,
            born=born,
        )


@pytest.mark.loop_physics
def test_declared_correlators_with_a_selection_are_refused(
    loop_sm_colour_model, loop_sm_compiled_model, loop_native, tmp_path
):
    """As the tree engine's correlated generation, declared colour or spin
    correlations (or the Ward mode) with a selection are refused with the
    listed ``selected-correlators`` before any generation: on the public route
    with any of the tree engine's selectors (a spin source on a leg of fixed
    helicity would ignore the selection), and for a ``SliceRequest``
    selection that keeps fewer helicities or flows."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loading import load_compiled_model

    process = "g g > g g"
    spin = CorrelatorConfig(spin_correlations=((3,),))
    colour = CorrelatorConfig(color_correlations=(ColorCorrelator.dipole("T12", 1, 2),))
    for k, (selectors, correlators) in enumerate(
        (
            ('selected_source_helicities = {"3" = -1}', spin),
            ("selected_color_sector_ids = [0]", colour),
            ("reference_color_order = [1, 3, 2, 4]", colour),
        )
    ):
        output = tmp_path / f"artifact-{k}"
        card = tmp_path / f"card-{k}.toml"
        card.write_text(
            _card(
                model=json.dumps(str(loop_sm_colour_model)),
                output=json.dumps(str(output)),
                selectors=selectors,
                accuracy="lc",
                process=json.dumps(process),
            )
        )
        with pytest.raises(LoopRequestUnsupported) as refused:
            Generator(resolve_config(card)).generate(
                ProcessSet((ProcessRequest.parse(process, name="p"),)),
                output,
                correlators=correlators,
            )
        assert refused.value.key == "selected-correlators"
        assert "process." in str(refused.value)
        assert not output.exists()
    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        process, *QCD[4], selection=lc.Selection(((3, -1),)), **_options()
    )
    for options in ({"correlators": spin}, {"correlators": colour}, {"ward": True}):
        with pytest.raises(LoopRequestUnsupported, match="selected-correlators"):
            ObservableSlice.generate(compiled, request, loop_native, **options)


# The stored complete-set rows: per process MadLoop's committed reference
# whose first point is taken, and the (helicity rows, flows) counts.
STORED_ROWS = {
    "u u~ > g g g g": ("uux_uuxuux", (64, 24)),
    "u u~ > d d~ g g": ("uux_uuxuux", (64, 12)),
    "g g > g g g g": ("uux_uuxuux", (64, 120)),
    "u u~ > g g g g g": ("uux_ggggg", (128, 120)),
}
O1_CAP = 1e-9


@pytest.mark.loop_physics
@pytest.mark.loop_nightly
@pytest.mark.parametrize("process", list(STORED_ROWS))
def test_stored_selections_partition_the_inclusive_value(artifacts, process):
    """2->4 and the 2->5 row at leading colour on their public artifacts
    (nightly): three groups of the flows and the two halves of the helicity
    rows (stored selections, ``native_observables.selection_parts``) sum to
    the inclusive value at the first point of MadLoop's committed reference,
    per coefficient: in double-double to 1e-24 of each evaluation; in binary64 under
    the O1 rule (m-qcd PLAN §1.4 rule 2) with the per-coefficient stability
    estimates E (the recommended checks, ``stability.RECOMMENDED_CHECKS``:
    one generic exact rotation and the rescaled reduction direction; each E
    at most 1e-9) propagated: |sum - inclusive| <= sum over the evaluations of
    max(1e-12, E) |value| (the Born to 1e-12)."""

    from fractions import Fraction

    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks

    name, counts = STORED_ROWS[process]
    key = process.replace(" ", "").replace(">", "_").replace("~", "x")
    runtime = Runtime.load(
        artifacts(key, "", process=process), process="p", model_parameters=PARAMETERS
    )
    flows = [a.id for a in runtime.physics.born_anchored_color_components]
    helicities = [h.id for h in runtime.physics.helicities]
    assert (len(helicities), len(flows)) == counts
    reference = json.loads(
        (ROOT / f"tests/fixtures/loop/required/{name}.json").read_text()
    )
    point = [reference["points"][0]]
    scale = [str(Fraction(reference["mu_r"][0]) ** 2)]
    half = len(helicities) // 2
    partitions = (
        [{"color_flows": flows[k::3]} for k in range(3)],
        [{"helicities": helicities[:half]}, {"helicities": helicities[half:]}],
    )
    checks = VirtualStabilityChecks(RECOMMENDED_CHECKS.rotations, True, rescaled=True)
    for arithmetic in ("double", "double-double"):

        def values(arithmetic=arithmetic, **selectors):
            """(values, per-coefficient error bounds) of one evaluation."""

            result = runtime.evaluate(
                point,
                mu_r2=scale,
                arithmetic=arithmetic,
                stability_checks=checks if arithmetic == "double" else None,
                **selectors,
            )
            values = [Decimal(str(x)) for x in _values(result)]
            if result.stability is None:
                return values, [DOUBLE_DOUBLE * abs(x) for x in values]
            estimate = result.stability[0]
            relative = [
                0.0,
                estimate.finite,
                estimate.single_pole,
                estimate.double_pole,
            ]
            assert max(relative) <= O1_CAP, (process, estimate)
            return values, [
                Decimal(max(DOUBLE, e)) * abs(x)
                for e, x in zip(relative, values, strict=True)
            ]

        inclusive, inclusive_error = values()
        for partition in partitions:
            parts = [values(**s) for s in partition]
            for c, b in enumerate(inclusive):
                total = sum(part[0][c] for part in parts)
                bound = inclusive_error[c] + sum(part[1][c] for part in parts)
                assert abs(total - b) <= bound, (arithmetic, c, total, b, bound)


# ------------------------------------------------------- MadLoop per helicity

MQCD = ROOT / "tests" / "fixtures" / "loop" / "required" / "mqcd"
# MadLoop's per-helicity runs (``SLOOPMATRIXHEL``, MQ-REF; the fixture's
# ``per_helicity`` rows): every point in MadLoop's quadruple-precision stage,
# each entry's Born its double pole over the universal factor, and a
# ``reference`` record per entry (``tools/loop_reference/README.md``).
PER_HELICITY = {
    "gg_ttx_sectors": (0, 1, 2, 3),
    "gg_ggg_hel": (0, 1, 2),
}
# Reference outliers by coefficient (m-qcd PLAN §1.4 rule 1 (i): the
# provider's own precision covers the difference). Parity makes the pole
# coefficients of helicities h and -h equal at a point (the Born's times real
# kinematic functions), so MadLoop's spread |ML(h) - ML(-h)| on such a pair
# is its own precision there. At gg_ggg_hel point 1 MadLoop's single poles
# of the pair below (~1e-8 of the summed single pole) differ by 4.6e-12
# relative; ours are 1.1e-12 and 3.5e-12 from them. A listed coefficient is
# held to: ours equal on the pair to 1e-24 (double-double), and within
# MadLoop's spread of each of its values; every other coefficient to 1e-12
# (the other coefficients of the three points agree to 1.2e-13).
PER_HELICITY_OUTLIERS = {
    ("gg_ggg_hel", 1): {
        "single_pole": frozenset({(-1, 1, -1, 1, -1), (1, -1, 1, -1, 1)}),
    },
}


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    [
        "gg_ttx_sectors",
        pytest.param("gg_ggg_hel", marks=pytest.mark.loop_nightly),
    ],
)
def test_single_helicities_equal_madloop_per_helicity(
    loop_sm_compiled_model, loop_native, name
):
    """The helicity rows a selection names (tree-engine IDs) are MadLoop's
    helicity configurations: per row, the full-colour ``evaluate_resolved``
    entry (double-double) times the helicity average equals MadLoop's
    ``SLOOPMATRIXHEL`` (colour-averaged, not helicity-averaged) to 1e-12 per
    coefficient (m-qcd PLAN §1.4 rule 1) as the entry's reference record
    says: relatively, or, for a helicity whose Born vanishes, absolutely
    against 1e-12 of F_s = |B| alpha_s/2pi of the point's summed Born; a
    listed reference outlier (``PER_HELICITY_OUTLIERS``) by parity: ours
    equal on h and -h, MadLoop within its own spread on them. The
    leading-colour selection of the same IDs is the leading-colour resolved
    row (the identities above), so this ties single-helicity selections to
    MadLoop's labels and normalisation."""

    import argparse
    import math
    from fractions import Fraction

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.observables import ObservableSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.runtime.virtual import V2ObservablesRuntime
    from tools import loop_v2_generate

    reference = json.loads((MQCD / f"{name}.json").read_text())
    compiled = load_compiled_model(loop_sm_compiled_model)
    request, ignored = loop_v2_generate._fixture_request(
        reference,
        compiled,
        argparse.Namespace(compatibility=None, scale_parameter=None),
    )
    assert not ignored and request.accuracy == "full"
    observables = ObservableSlice.of(LoopSlice.generate(compiled, request, loop_native))
    runtime = V2ObservablesRuntime(
        observables, executor="native", model_parameters=reference["parameters"]
    )
    points = PER_HELICITY[name]
    result = runtime.evaluate_resolved(
        [reference["points"][p] for p in points],
        mu_r2=[str(Fraction(reference["mu_r"][p]) ** 2) for p in points],
        arithmetic="double-double",
    )
    # The helicity average: the incoming legs' helicity counts.
    average = math.prod(
        len({h for row in observables.rows for leg, h in row if leg == k})
        for k in (0, 1)
    )
    keys = ("born", "finite", "single_pole", "double_pole")
    alpha = float(reference["parameters"]["aS"])
    for i, p in enumerate(points):
        madloop = {
            lc.helicity_id(tuple(enumerate(row["helicity"]))): row
            for row in reference["per_helicity"]["rows"][p]
        }
        ids = [lc.helicity_id(row) for row in observables.rows]
        assert sorted(ids) == sorted(madloop)
        exact = {
            ident: [
                Decimal(str(result.born_values[i][0][k][0])) * average,
                *(Decimal(str(x)) * average for x in result.values[i][0][k][0]),
            ]
            for k, ident in enumerate(ids)
        }
        ours = {ident: [float(x) for x in values] for ident, values in exact.items()}
        outliers = {
            (lc.helicity_id(tuple(enumerate(h))), coefficient)
            for coefficient, pairs in PER_HELICITY_OUTLIERS.get((name, p), {}).items()
            for h in pairs
        }
        for ident, row in madloop.items():
            record = row["reference"]
            assert record["status"] == "ok", (p, ident, record["reason"])
            expected = [row[key] for key in keys]
            for c, (a, b) in enumerate(zip(ours[ident], expected, strict=True)):
                if record["comparison"] == "absolute":
                    scale = record["absolute_scale"] * alpha / (2 * math.pi)
                    assert abs(a) <= 1e-12 * scale, (p, ident, keys[c], a)
                elif (ident, keys[c]) in outliers:
                    parity = lc.helicity_id(
                        tuple(enumerate(-h for h in row["helicity"]))
                    )
                    mine, image = exact[ident][c], exact[parity][c]
                    assert abs(mine - image) <= DOUBLE_DOUBLE * max(
                        abs(mine), abs(image)
                    ), (p, ident, keys[c], mine, image)
                    spread = abs(Decimal(b) - Decimal(madloop[parity][keys[c]]))
                    assert abs(mine - Decimal(b)) <= spread, (p, ident, keys[c])
                else:
                    error = abs(a - b) / max(abs(a), abs(b))
                    assert error <= 1e-12, (p, ident, keys[c], a, b, error)


# ------------------------------------------------------------------ public route


@pytest.fixture(scope="module")
def artifacts(loop_sm_colour_model, loop_native, tmp_path_factory):
    """The public LC artifacts of a process (u u~ > d d~ g by default):
    inclusive, or with the tree engine's selectors."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    built = {}

    def build(
        key: str,
        selectors: str,
        accuracy: str = "lc",
        process: str = "u u~ > d d~ g",
    ) -> Path:
        if key not in built:
            tmp = tmp_path_factory.mktemp(key)
            card = tmp / "card.toml"
            card.write_text(
                _card(
                    model=json.dumps(str(loop_sm_colour_model)),
                    output=json.dumps(str(tmp / "artifact")),
                    selectors=selectors,
                    accuracy=accuracy,
                    process=json.dumps(process),
                )
            )
            processes = ProcessSet((ProcessRequest.parse(process, name="p"),))
            generator = Generator(resolve_config(card))
            if accuracy != "lc":
                built[key] = generator.plan(processes)
            else:
                generator.generate(processes, tmp / "artifact")
                built[key] = tmp / "artifact"
        return built[key]

    return build


_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{{expression={process}, name="p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
{selectors}
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {fixed}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _card(**fields: str) -> str:
    """``_CARD`` with its fields (by default no selectors, leading colour and
    MB = 0)."""

    defaults = {"selectors": "", "accuracy": "lc", "fixed": '{MB = "0"}'}
    return _CARD.format(**{**defaults, **fields})


_SELECTORS = (
    'selected_color_sector_ids = [2]\nselected_source_helicities = {"1" = -1, "5" = 1}'
)


@pytest.mark.loop_physics
def test_public_selectors_generate_the_selection_of_the_inclusive_artifact(artifacts):
    """The stored selected artifact (tree sector 2, legs 1 and 5 fixed)
    equals the inclusive artifact's runtime selection (stored resolved
    columns summed), binary64 and double-double; its physics lists its flow;
    stored complete flow sets sum to the inclusive value; the inclusive
    record has no selection, the selected one names it; a flow below the
    leading colour is refused at generation, selectors at NLC in the dry
    run."""

    from pyamplicol import Runtime
    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.api.errors import GenerationError

    inclusive = Runtime.load(
        artifacts("inclusive", ""), process="p", model_parameters=PARAMETERS
    )
    chosen = Runtime.load(
        artifacts("selected", _SELECTORS), process="p", model_parameters=PARAMETERS
    )
    anchors = inclusive.physics.born_anchored_color_components
    assert [a.id for a in anchors] == [
        "flow:2,1,3,5,4",
        "flow:2,5,1,3,4",
        "flow:2,4,3,5,1",
        "flow:2,5,4,3,1",
    ]
    assert inclusive.physics.selector_capabilities == ("helicity", "color_flow")
    assert [a.id for a in chosen.physics.born_anchored_color_components] == [
        "flow:2,4,3,5,1"
    ]
    assert len(chosen.physics.helicities) == 8
    pts, scales = _points("uux_ddxg", 2)
    rows = [
        h.id
        for h in inclusive.physics.helicities
        if h.values[0] == -1 and h.values[4] == 1
    ]
    for arithmetic, tolerance in (("double", DOUBLE), ("double-double", DOUBLE_DOUBLE)):
        reference = inclusive.evaluate(
            pts, mu_r2=scales, helicities=rows, color_flows=["3"], arithmetic=arithmetic
        )
        value = chosen.evaluate(pts, mu_r2=scales, arithmetic=arithmetic)
        for index in range(len(pts)):
            _close(_values(value, index), _values(reference, index), tolerance)
    checked = chosen.evaluate(
        pts,
        mu_r2=scales,
        color_flows=["flow:2,4,3,5,1"],
        stability_checks=VirtualStabilityChecks(
            RECOMMENDED_CHECKS.rotations, True, rescaled=True
        ),
    )
    assert checked.stability is not None and checked.stability[0].relative < 1e-9
    # A stored selection is one column set formed from the stored resolved
    # records (``native_observables.selection_parts``): a (row, flow)
    # selection is its resolved entry, and a flow's set runs only the
    # families whose seeds it reads.
    resolved = inclusive.evaluate_resolved(pts, mu_r2=scales)
    total = inclusive.evaluate(pts, mu_r2=scales)
    flow = anchors[2]
    (b,) = flow.born_term_indices
    k = [h.id for h in inclusive.physics.helicities].index(rows[1])
    one = inclusive.evaluate(
        pts, mu_r2=scales, helicities=[rows[1]], color_flows=[flow.id]
    )
    for index in range(len(pts)):
        entry = resolved.values[index][0][k][
            resolved.virtual_color_ids.index(f"basis-{b}")
        ]
        _close(
            list(one.values[index][0]),
            list(entry),
            DOUBLE,
            born=_values(total, index)[0],
        )
    # evaluate_resolved takes the same selectors: the selected rows and the
    # selected flows' basis elements, each entry as without them.
    subset = inclusive.evaluate_resolved(
        pts, mu_r2=scales, helicities=[rows[1]], color_flows=[flow.id]
    )
    assert subset.helicity_ids == (f"h{k}",)
    assert subset.virtual_color_ids == (f"basis-{b}",)
    column = resolved.virtual_color_ids.index(f"basis-{b}")
    for index in range(len(pts)):
        assert subset.values[index][0][0][0] == resolved.values[index][0][k][column]
        assert (
            subset.born_values[index][0][0][0]
            == resolved.born_values[index][0][k][column]
        )
    halves = inclusive.evaluate_resolved(pts, mu_r2=scales, helicities=rows)
    assert halves.helicity_ids == tuple(
        f"h{i}" for i, h in enumerate(inclusive.physics.helicities) if h.id in rows
    )
    assert halves.virtual_color_ids == resolved.virtual_color_ids
    inclusive.evaluate(pts, mu_r2=scales, color_flows=[flow.id])
    observables = inclusive._backend._observable_runtime()

    def formed(key):
        """The stored column set the runtime keeps for a selection."""

        (found,) = (
            value.columns
            for (_kind, column), value in observables._selections.items()
            if column.key == key
        )
        return found

    flow_set = formed(lc.column(None, frozenset({b})).key)
    kept = flow_set.families[0]
    assert kept is not None and len(kept) < int(flow_set.meta["families"])
    # Its (row, flow) set's plan keeps the rows that row reaches (the native
    # live-row pruning, ``plan/live.rs``): a fraction of the flow's dressing
    # work, on the same families.
    catalogue = observables.selection_catalogue()
    row_set = formed(lc.column(*catalogue.resolve([rows[1]], [flow.id])).key)
    assert row_set.families == flow_set.families
    work = "dressing work (modelled complex MAC per lane)"
    one_row, every_row = (
        found.last_report["counts"][work] for found in (row_set, flow_set)
    )
    assert 0 < one_row <= every_row / 4, (one_row, every_row)
    _close(
        _summed(
            inclusive.evaluate(pts, mu_r2=scales, color_flows=[a.id]) for a in anchors
        ),
        _values(total),
        DOUBLE,
    )
    # The selected artifact lists and numbers the flows it computes, as the
    # tree runtime of the same selector does: ordinal 1 is its one flow;
    # another flow is no flow of it.
    _close(
        _values(chosen.evaluate(pts, mu_r2=scales, color_flows=["1"])),
        _values(chosen.evaluate(pts, mu_r2=scales)),
        DOUBLE,
    )
    with pytest.raises(EvaluationError, match="unknown colour-flow selector"):
        chosen.evaluate(pts, mu_r2=scales, color_flows=["flow:2,5,4,3,1"])
    with pytest.raises(EvaluationError, match="out of range"):
        chosen.evaluate(pts, mu_r2=scales, color_flows=["2"])
    assert (inclusive.physics.helicity_coverage, inclusive.physics.color_coverage) == (
        "all",
        "all",
    )
    assert (chosen.physics.helicity_coverage, chosen.physics.color_coverage) == (
        "selected",
        "selected",
    )
    request = lambda runtime: runtime.inspect()["request"]  # noqa: E731
    assert "selection" not in request(inclusive)
    assert request(chosen)["selection"] == {
        "flows": [[2, 4, 3, 5, 1]],
        "helicities": [[1, -1], [5, 1]],
    }
    with pytest.raises(GenerationError, match="subleading-colour-flows"):
        artifacts("subleading", "selected_color_sector_ids = [0]")
    blockers = artifacts("nlc", _SELECTORS, accuracy="nlc").unsupported_features
    assert any("leading colour (color.accuracy='lc')" in item for item in blockers)


@pytest.mark.loop_physics
def test_the_dry_run_refuses_selected_helicities_the_process_lacks(
    loop_sm_colour_model, tmp_path
):
    """The public dry run (``Generator.plan``) reports a selected helicity of
    an unknown leg or one the leg does not have, as the generation refuses
    it (``loop_service._v2_selection``)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    processes = ProcessSet((ProcessRequest.parse("u u~ > d d~ g", name="p"),))
    for k, (selector, message) in enumerate(
        (
            ('{"9" = 1}', "legs [9] the process does not have"),
            ('{"1" = 0}', "selected helicity 0 is unavailable for external leg 1"),
        )
    ):
        card = tmp_path / f"card{k}.toml"
        card.write_text(
            _card(
                model=json.dumps(str(loop_sm_colour_model)),
                output=json.dumps(str(tmp_path / f"artifact{k}")),
                selectors=f"selected_source_helicities = {selector}",
                accuracy="lc",
                process=json.dumps("u u~ > d d~ g"),
            )
        )
        blockers = Generator(resolve_config(card)).plan(processes).unsupported_features
        assert any(message in item for item in blockers), blockers


@pytest.mark.loop_physics
def test_evaluation_time_selections_are_kept_most_recently_used(artifacts, monkeypatch):
    """A loaded artifact keeps the formed column sets of its most recently
    used evaluation-time selections only (``$PYAMPLICOL_LOOP_SELECTION_CACHE
    = 2`` here): a third distinct selection releases the least recently used
    one's set and native executors, a re-formed selection returns bitwise
    its first value, and ``Runtime.clear`` releases every set."""

    import gc
    import weakref

    from pyamplicol import Runtime

    monkeypatch.setenv(lc.SELECTION_CACHE_ENVIRONMENT, "2")
    runtime = Runtime.load(
        artifacts("inclusive", ""), process="p", model_parameters=PARAMETERS
    )
    pts, scales = _points("uux_ddxg")
    flows = [a.id for a in runtime.physics.born_anchored_color_components]
    first = _values(runtime.evaluate(pts, mu_r2=scales, color_flows=[flows[2]]))
    observables = runtime._backend._observable_runtime()
    (entry,) = observables._selections.items()
    released = weakref.ref(entry[1].columns)
    del entry
    for flow in (flows[3], flows[0]):
        runtime.evaluate(pts, mu_r2=scales, color_flows=[flow])
    assert len(observables._selections) == 2
    gc.collect()
    assert released() is None
    again = _values(runtime.evaluate(pts, mu_r2=scales, color_flows=[flows[2]]))
    assert again == first and first[0] != 0
    runtime.clear()
    assert observables._selections is None
