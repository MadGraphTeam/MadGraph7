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
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.loops.v2 import selection as lc

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
PARAMETERS = {"aS": "0.118", "MT": "173"}
DOUBLE = 1e-12
DOUBLE_DOUBLE = Decimal("1e-24")
# Born and loop QCD orders by external leg count (QED 0).
QCD = {
    legs: ({"QCD": legs - 2, "QED": 0}, {"QCD": legs, "QED": 0})
    for legs in (4, 5, 6, 7)
}


def _points(name: str, count: int = 1):
    fixture = json.loads((LEGACY / f"{name}.json").read_text())
    return fixture["points"][:count], fixture["scales_squared"][:count]


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


def _close(actual, expected, tolerance) -> None:
    scale = max(abs(x) for x in expected)
    for a, b in zip(actual, expected, strict=True):
        assert abs(a - b) <= tolerance * scale, (actual, expected)


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

    selection = lc.Selection(((5, 1), (1, -1)), ((2, 4, 3, 5, 1),))
    assert selection.helicities == ((1, -1), (5, 1))
    assert lc.Selection.from_record(selection.record()) == selection
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
    lc.require_leading_colour("full", None, None)
    rows = (((0, -1), (1, 1)), ((0, 1), (1, -1)))
    flows = (
        lc.Flow("flow:2,1", (2, 1), 1, True, "flow:2,1", 0),
        lc.Flow("flow:1,2", (1, 2), None, True, "flow:1,2", 1),
        lc.Flow("flow:3,4", (3, 4), 0, True, "flow:3,4", 2),
    )
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
        _close(list(one.values[0][0]), list(entry), DOUBLE)
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
    scale = max(abs(x) for x in inclusive)
    for a, b in zip(total, inclusive, strict=True):
        assert abs(Decimal(a) - Decimal(b)) <= DOUBLE_DOUBLE * Decimal(scale)


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
    ("process", "points", "executors", "resolved"),
    [
        ("u u~ > g g g", "uux_ggg", ("python", "native"), True),
        ("g g > g g g", "uux_ggg", ("native",), True),
        ("u u~ > g g g g", "uux_uuxuux.lc", ("native",), False),
        ("g g > g g g g", "uux_uuxuux.lc", ("native",), False),
    ],
)
def test_runtime_selections_partition_the_inclusive_value_nightly(
    generate, process, points, executors, resolved
):
    """Mixed and pure-gluon rows to 2->4 (native; three flow groups and the
    two helicity halves as complete sets; binary64 and double-double)."""

    observables = generate(process)
    flows = _runtime(observables).selection_catalogue().flows
    assert_complete_sets(
        observables,
        _points(points),
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
    ],
)
def test_generation_selection_prunes_and_equals_the_runtime_selection(
    generate, process, points, flow, helicities
):
    """A generation-time selection keeps fewer skeletons (and Born rows) and
    equals the runtime selection of the inclusive slice, Python and native,
    binary64 and (natively) double-double. g g > g g g holds the ring-aware
    pruning (per-reading liveness lost 40 % of the double pole)."""

    from pyamplicol.loops.v2 import reference_executor as rx

    inclusive = generate(process)
    chosen = generate(process, lc.Selection(helicities, (flow,)))
    assert len(rx.reading_skeletons(chosen.slice.plan)) < len(
        rx.reading_skeletons(inclusive.slice.plan)
    )
    assert len(chosen.slice.basis) == 1
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
    expected = _values(reference)
    scale = max(abs(x) for x in expected)
    for a, b in zip(_values(value), expected, strict=True):
        assert abs(Decimal(a) - Decimal(b)) <= DOUBLE_DOUBLE * Decimal(scale)


@pytest.mark.loop_physics
def test_a_flow_below_the_leading_colour_is_zero_and_refused_at_generation(generate):
    """u u~ > d d~ g's tree flows joining each quark to its own antiquark are
    below the squared matrix element's leading power: zero when selected at
    evaluation, refused when selected at generation."""

    observables = generate("u u~ > d d~ g")
    runtime = _runtime(observables)
    pts, scales = _points("uux_ddxg")
    value = runtime.evaluate(pts, mu_r2=scales, color_flows=["flow:2,1,3,5,4"])
    assert _values(value) == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="below the process's leading colour"):
        generate("u u~ > d d~ g", lc.Selection((), ((2, 1, 3, 5, 4),)))


# ------------------------------------------------------- tree-engine parity


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

    from types import SimpleNamespace

    from pyamplicol import Runtime
    from pyamplicol.api.errors import GenerationError
    from pyamplicol.generation.loop_service import _v2_selection
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)

    def chosen(process: str, **selectors):
        fields = dict.fromkeys(
            (
                "max_color_sectors",
                "reference_color_order",
                "selected_color_sector_ids",
                "selected_source_helicities",
            )
        )
        fields.update(selectors)
        backend = SimpleNamespace(_process_selection=SimpleNamespace(**fields))
        return _v2_selection(backend, compiled, process, "p")

    assert chosen("g g > g g g") is None
    assert chosen("g g > g g g", selected_source_helicities={1: -1}) == (
        lc.Selection(((1, -1),))
    )
    # Sector ids: the tree engine's artifact with the same selector.
    for k, (process, ids) in enumerate(
        (("g g > g g g", [0, 7]), ("u u~ > d d~ g", [1, 2]))
    ):
        selection = chosen(process, selected_color_sector_ids=ids)
        tree = Runtime.load(
            _tree_artifact(
                tmp_path / str(k), process, f"selected_color_sector_ids = {ids}"
            ),
            process="t",
        )
        assert sorted(lc.flow_id(word) for word in selection.flows) == sorted(
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


# ------------------------------------------------------- MadLoop per helicity

MQCD = ROOT / "tests" / "fixtures" / "loop" / "required" / "mqcd"
# MadLoop's per-helicity values (``SLOOPMATRIXHEL``, MQ-REF's per-helicity
# mode; ``per_helicity`` of the fixture) at the points whose every row meets
# the 1e-12 standard, except the listed reference outliers: g g > g g g's
# helicities (-1,-1,-1,+1,-1) and (+1,+1,+1,-1,+1) are suppressed to 1e-10 of
# the largest row at its points, where MadLoop's double-precision Born (the
# loop runs in quadruple precision, status 3xx) is good to about 1e-20 of the
# largest row only: Born 3.3e-11, single pole 2.6e-12 relative to the row at
# point 1, the double pole (Born times Catani's constant) with it; every
# other row there agrees to 4.4e-13. g g > t t~ (MadLoop in double
# precision, status 2xx): points 0 and 2 (every row within 7.6e-14); at point
# 1 MadLoop's two parity-equal rows of one single pole differ by 2.6e-13 and
# ours (equal) by 1.06e-12 from one of them.
PER_HELICITY = {
    "gg_ttx_sectors": (0, 2),
    "gg_ggg": (1,),
}
PER_HELICITY_OUTLIERS = {
    ("gg_ggg", 1): frozenset({(-1, -1, -1, 1, -1), (1, 1, 1, -1, 1)}),
}


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    [
        "gg_ttx_sectors",
        pytest.param("gg_ggg", marks=pytest.mark.loop_nightly),
    ],
)
def test_single_helicities_equal_madloop_per_helicity(
    loop_sm_compiled_model, loop_native, name
):
    """The helicity rows a selection names (tree-engine IDs) are MadLoop's
    helicity configurations: per row, the full-colour ``evaluate_resolved``
    entry (double-double) times the helicity average equals MadLoop's
    ``SLOOPMATRIXHEL`` (colour-averaged, not helicity-averaged) to 1e-12 per
    coefficient (m-qcd PLAN §1.4 rule 1); rows that vanish identically
    absolutely to 1e-12 F_s (F_s = |B| alpha_s/2pi of the summed Born). A listed
    reference outlier (``PER_HELICITY_OUTLIERS``) is held to 1e-12 of that
    coefficient's largest row. The leading-colour selection of the same IDs
    is the leading-colour resolved row (the identities above), so this ties
    single-helicity selections to MadLoop's labels and normalisation."""

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
        ours = {
            ident: [
                float(result.born_values[i][0][k][0]) * average,
                *(float(x) * average for x in result.values[i][0][k][0]),
            ]
            for k, ident in enumerate(ids)
        }
        summed = sum(row["born"] for row in madloop.values())
        assert abs(summed / average - reference["rows"][p]["born"]) <= 1e-12 * abs(
            summed
        )
        scale = abs(summed) * alpha / (2 * math.pi)
        largest = [max(abs(row[key]) for row in madloop.values()) for key in keys]
        outliers = {
            lc.helicity_id(tuple(enumerate(h)))
            for h in PER_HELICITY_OUTLIERS.get((name, p), ())
        }
        for ident, row in madloop.items():
            expected = [row[key] for key in keys]
            # A row that vanishes identically (helicity-violating; MadLoop's
            # numerical zeros are ~1e-33 of the summed Born).
            vanishing = abs(row["born"]) <= 1e-20 * abs(summed)
            for c, (a, b) in enumerate(zip(ours[ident], expected, strict=True)):
                if vanishing or b == 0:
                    assert abs(a) <= 1e-12 * scale, (p, ident, keys[c], a)
                elif ident in outliers:
                    assert abs(a - b) <= 1e-12 * largest[c], (p, ident, keys[c], a, b)
                else:
                    error = abs(a - b) / max(abs(a), abs(b))
                    assert error <= 1e-12, (p, ident, keys[c], a, b, error)


# ------------------------------------------------------------------ public route


@pytest.fixture(scope="module")
def artifacts(loop_sm_colour_model, loop_native, tmp_path_factory):
    """The public LC artifacts of u u~ > d d~ g: inclusive, and with the tree
    engine's selectors (sector 2, legs 1 and 5 fixed)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    built = {}

    def build(key: str, selectors: str, accuracy: str = "lc") -> Path:
        if key not in built:
            tmp = tmp_path_factory.mktemp(key)
            card = tmp / "card.toml"
            card.write_text(
                _CARD.format(
                    model=json.dumps(str(loop_sm_colour_model)),
                    output=json.dumps(str(tmp / "artifact")),
                    selectors=selectors,
                    accuracy=accuracy,
                )
            )
            processes = ProcessSet((ProcessRequest.parse("u u~ > d d~ g", name="p"),))
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
entries = [{{expression="u u~ > d d~ g", name="p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
{selectors}
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
engine = "v2"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""
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
    for arithmetic, tolerance in (("double", DOUBLE), ("double-double", 1e-24)):
        reference = inclusive.evaluate(
            pts, mu_r2=scales, helicities=rows, color_flows=["3"], arithmetic=arithmetic
        )
        value = chosen.evaluate(pts, mu_r2=scales, arithmetic=arithmetic)
        for index in range(len(pts)):
            expected = [Decimal(x) for x in _values(reference, index)]
            scale = max(abs(x) for x in expected)
            for a, b in zip(_values(value, index), expected, strict=True):
                assert abs(Decimal(a) - b) <= Decimal(tolerance) * scale
    checked = chosen.evaluate(
        pts,
        mu_r2=scales,
        color_flows=["flow:2,4,3,5,1"],
        stability_checks=VirtualStabilityChecks(("z:3/5:4/5",), True),
    )
    assert checked.stability is not None and checked.stability[0].relative < 1e-9
    # A stored selection is one column set formed from the stored resolved
    # records (``native_observables.selection_parts``): a (row, flow)
    # selection is its resolved entry, and a flow's set runs only the
    # families whose seeds it reads.
    resolved = inclusive.evaluate_resolved(pts, mu_r2=scales)
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
        _close(list(one.values[index][0]), list(entry), DOUBLE)
    inclusive.evaluate(pts, mu_r2=scales, color_flows=[flow.id])
    stored = inclusive._backend._observable_runtime()._observables
    (flow_set,) = (
        found
        for key, found in stored._sets.items()
        if key[:2] == ("selection", lc.column(None, frozenset({b})).key)
    )
    kept = flow_set.families[0]
    assert kept is not None and len(kept) < int(flow_set.meta["families"])
    _close(
        _summed(
            inclusive.evaluate(pts, mu_r2=scales, color_flows=[a.id]) for a in anchors
        ),
        _values(inclusive.evaluate(pts, mu_r2=scales)),
        DOUBLE,
    )
    with pytest.raises(EvaluationError, match="no Born amplitude"):
        chosen.evaluate(pts, mu_r2=scales, color_flows=["flow:2,5,4,3,1"])
    request = lambda runtime: runtime.inspect()["request"]  # noqa: E731
    assert "selection" not in request(inclusive)
    assert request(chosen)["selection"] == {
        "flows": [[2, 4, 3, 5, 1]],
        "helicities": [[1, -1], [5, 1]],
    }
    with pytest.raises(GenerationError, match="below the process's leading colour"):
        artifacts("subleading", "selected_color_sector_ids = [0]")
    blockers = artifacts("nlc", _SELECTORS, accuracy="nlc").unsupported_features
    assert any("leading colour (color.accuracy='lc')" in item for item in blockers)
