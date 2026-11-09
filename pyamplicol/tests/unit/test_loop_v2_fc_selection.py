# SPDX-License-Identifier: 0BSD
"""Run-time helicity selection of Born-interfered v2 virtuals at full and
next-to-leading colour (lit-fc-hel; USER 2026-11-09 ~19:53, DECISIONS
2026-11-09 19:53 (3)).

At NLC and full colour the colour is contracted (one resolved component per
Born helicity row), so a selection keeps some helicity rows with every
colour component of their Born; colour flows exist at leading colour only
and are refused there. Oracles:

* exact identities: a single helicity's ``evaluate`` is its
  ``evaluate_resolved`` entry bitwise up to the sign of a zero; a complete
  set of single helicities and two complementary halves sum to the
  unselected value within the binary64 summation bound gamma_n S of both
  sides (n the summed helicity rows, S the unselected call's
  pre-cancellation scale) plus the reduction estimates E of both sides
  (each side reduces its own numerators, so their reduction rounding
  differs; ``rounding_bound``);
* "only the selected helicities, nothing more" (USER 2026-11-08 ~22:34): a
  selected call forms the kept rows' trees and dresses the links its rows
  reach (``plan::LiveRows``), counted against the loader's row restriction
  of the same rows (the per-selection column set, ``selection_parts``); a
  resolved selection forms its kept rows' columns only;
* an artifact generated before this package (the shipped self-test
  ``selftest_virtual/ew``) serves the selection without regeneration.
"""

from __future__ import annotations

import json
import math
import struct
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pyamplicol.api.errors import CompatibilityError

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
PARAMETERS = {"aS": "0.118", "MT": "173"}
UNIT = 2.0**-53

_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{{expression={process}, name="p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""

CASES = [
    ("u u~ > d d~ g", "uux_ddxg", "full"),
    ("g g > g g", "gg_gg", "full"),
    ("u u~ > d d~ g", "uux_ddxg", "nlc"),
]


def generate_born(root: Path, model: Path, process: str, accuracy: str) -> Path:
    """The public Born-interfered artifact of ``process`` (QCD corrections,
    ``loop_sm`` with the committed colour declarations) at a colour
    accuracy, generated under ``root`` (process name ``p``)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    card = root / "card.toml"
    card.write_text(
        _CARD.format(
            model=json.dumps(str(model)),
            output=json.dumps(str(root / "artifact")),
            accuracy=accuracy,
            process=json.dumps(process),
        )
    )
    processes = ProcessSet((ProcessRequest.parse(process, name="p"),))
    Generator(resolve_config(card)).generate(processes, root / "artifact")
    return root / "artifact"


@pytest.fixture(scope="module")
def artifacts(loop_sm_colour_model, loop_native, tmp_path_factory):
    """The public artifact of a process at an accuracy, generated once."""

    del loop_native
    built: dict[tuple[str, str], Path] = {}

    def build(process: str, accuracy: str) -> Path:
        key = (process, accuracy)
        if key not in built:
            built[key] = generate_born(
                tmp_path_factory.mktemp("fc"), loop_sm_colour_model, process, accuracy
            )
        return built[key]

    return build


def _points(name: str, count: int = 1) -> tuple[list[Any], list[Any]]:
    fixture = json.loads((LEGACY / f"{name}.json").read_text())
    return fixture["points"][:count], fixture["scales_squared"][:count]


def _runtime(path: Path) -> Any:
    from pyamplicol import Runtime

    return Runtime.load(path, process="p", model_parameters=PARAMETERS)


def _bits(value: Any) -> Any:
    """A coefficient's bits up to the sign of a zero (binary64: its IEEE
    bytes; extended carriers: the exact decimal)."""

    if isinstance(value, complex):
        return (_bits(value.real), _bits(value.imag))
    if isinstance(value, float):
        return struct.pack("<d", value + 0.0)
    return Decimal(value) + 0


def _entry(result: Any, point: int) -> list[Any]:
    """Born per squared order, then every sector's Laurent coefficients, of
    an ordinary evaluation's point."""

    return [*result.born_values[point], *(c for s in result.values[point] for c in s)]


def _resolved_entry(result: Any, point: int, row: int) -> list[Any]:
    """The same of a resolved evaluation's row (one colour component)."""

    (component,) = range(len(result.virtual_color_ids))
    born = [order[row][component] for order in result.born_values[point]]
    laurent = [c for s in result.values[point] for c in s[row][component]]
    return [*born, *laurent]


def _checks() -> Any:
    from pyamplicol.api import VirtualStabilityChecks

    # The reduction-direction check: the estimate E and the scale S.
    return VirtualStabilityChecks((), True)


def rounding_bound(
    total: Any, parts: list[Any], point: int, unit: Any, rows: int
) -> list[Any]:
    """Per entry of ``_entry``: the bound on |sum of parts - total| of a
    partition of the ``rows`` Born helicity rows (lit-fc-hel (H4)): both
    sides sum the same per-row terms in another grouping (the summed call
    adds the rows in its seeds and trees, each part its own rows, then the
    parts are added), so each is within the summation bound gamma_n S of the
    exact sum (n the rows, the summed terms; S the pre-cancellation scale:
    the total's ``magnitude``, the Born's absolute sum), plus the reduction
    estimates E of the total and of every part (each side reduces its own
    numerators: the summed call one per family, a part its rows' only)."""

    n = rows
    gamma = n * unit / (1 - n * unit)
    # Every output is rounded once more to its carrier (the parts and the
    # total).
    outputs = len(parts) + 1
    born = len(total.born_values[point])
    out = []
    for k, value in enumerate(_entry(total, point)):
        if k < born:
            scale = sum(abs(_entry(p, point)[k]) for p in parts)
            out.append(2 * gamma * scale + outputs * abs(value) * unit)
            continue
        s, c = divmod(k - born, len(total.values[point][0]))
        estimate = total.stability[point]
        scale = estimate.magnitude[s][c]
        errors = estimate.absolute[s][c] + sum(
            p.stability[point].absolute[s][c] for p in parts
        )
        out.append(
            2 * gamma * type(value)(scale)
            + type(value)(errors)
            + outputs * abs(value) * unit
        )
    return out


def _assert_partition(
    total: Any, parts: list[Any], points: int, unit: Any, rows: int
) -> None:
    from decimal import localcontext

    for point in range(points):
        with localcontext() as context:
            # Exact sums of the parts' decimal values.
            context.prec = 200
            summed = [
                sum(values) for values in zip(*(_entry(p, point) for p in parts))
            ]
        bound = rounding_bound(total, parts, point, unit, rows)
        for got, expected, limit in zip(
            summed, _entry(total, point), bound, strict=True
        ):
            assert abs(got - expected) <= limit, (point, got, expected, limit)


@pytest.mark.loop_physics
@pytest.mark.parametrize(("process", "points", "accuracy"), CASES)
def test_single_helicities_are_their_resolved_entries_and_partition_the_virtual(
    artifacts, process, points, accuracy
):
    """Every single helicity's selection is its resolved entry bitwise up to
    the sign of a zero (Born and every Laurent coefficient); the complete set
    of single helicities and two complementary halves sum to the unselected
    virtual within ``rounding_bound``; colour flows are refused typed with a
    text naming the helicity selection."""

    runtime = _runtime(artifacts(process, accuracy))
    pts, scales = _points(points, 2)
    hel = [h.id for h in runtime.physics.helicities]
    resolved = runtime.evaluate_resolved(pts, mu_r2=scales)
    assert resolved.virtual_color_ids == (accuracy,)
    assert list(resolved.helicity_ids) == [f"h{k}" for k in range(len(hel))]
    total = runtime.evaluate(pts, mu_r2=scales, stability_checks=_checks())
    singles = []
    for k, h in enumerate(hel):
        one = runtime.evaluate(
            pts, mu_r2=scales, helicities=[h], stability_checks=_checks()
        )
        assert one.color_accuracy == accuracy
        for point in range(len(pts)):
            assert [_bits(x) for x in _entry(one, point)] == [
                _bits(x) for x in _resolved_entry(resolved, point, k)
            ], (h, point)
        singles.append(one)
    _assert_partition(total, singles, len(pts), UNIT, len(hel))
    half = len(hel) // 2
    halves = [
        runtime.evaluate(pts, mu_r2=scales, helicities=part, stability_checks=_checks())
        for part in (hel[:half], hel[half:])
    ]
    _assert_partition(total, halves, len(pts), UNIT, len(hel))
    # The tree engine's IDs name the same rows as the runtime IDs.
    by_tree = runtime.evaluate(
        pts, mu_r2=scales, helicities=[runtime.physics.helicities[1].id]
    )
    by_index = runtime.evaluate(pts, mu_r2=scales, helicities=["h1"])
    assert [_bits(x) for x in _entry(by_tree, 0)] == [
        _bits(x) for x in _entry(by_index, 0)
    ]
    for call in (runtime.evaluate, runtime.evaluate_resolved):
        with pytest.raises(CompatibilityError, match="helicities=... selects Born"):
            call(pts, mu_r2=scales, color_flows=["1"])


def _largest(runtime: Any, pts: Any, scales: Any) -> str:
    """The helicity with the largest Born at the first point (a vanishing
    helicity measures nothing)."""

    resolved = runtime.evaluate_resolved(pts, mu_r2=scales)
    born = [abs(x[0]) for x in resolved.born_values[0][0]]
    return resolved.helicity_ids[born.index(max(born))]


@pytest.mark.loop_physics
@pytest.mark.parametrize(("process", "points"), [("u u~ > d d~ g", "uux_ddxg")])
def test_a_selection_forms_only_its_rows(artifacts, monkeypatch, process, points):
    """USER 2026-11-08 ~22:34 ('only the selected helicities are evaluated
    and nothing more'): a selected call on the selection template forms the
    kept rows' Born and counterterm trees only and dresses exactly the links
    of the loader's row restriction of the same rows (the per-selection
    column set ``selection_parts``, whose plan the load prunes to its rows):
    the same dressing work, reduction lanes and tree closure rows, every
    family that drops a link a pruned copy (plan::LiveRows, lit-fc-hel
    (G4)), values bitwise up to the sign of a zero; one helicity forms one
    tree row against every row of the summed call. A resolved selection
    forms only its kept rows' columns (one tree row and one colour component
    per kept row) with the unselected resolved call's values."""

    from pyamplicol.loops.v2 import native_observables as no
    from pyamplicol.loops.v2 import selection as lc
    from pyamplicol.runtime.virtual import _inputs

    runtime = _runtime(artifacts(process, "full"))
    pts, scales = _points(points, 2)
    hel = [h.id for h in runtime.physics.helicities]
    one = _largest(runtime, pts, scales)
    backend = runtime._backend._observable_runtime()
    observables = backend._observables
    summed = runtime._backend._executor
    runtime.evaluate(pts, mu_r2=scales)
    summed_counts = summed.last_report["counts"]
    assert summed_counts["tree closure rows"] == len(hel)
    catalogue = backend.selection_catalogue()
    points_, scales_ = _inputs(pts, scales, backend.external_count)
    settings = {
        "parameters": PARAMETERS,
        "precision": backend._precision("double", None)[1],
        "poles": "all",
        "stability": None,
    }
    for chosen in ([one], hel[: len(hel) // 2], hel[1:]):
        column = lc.column(*catalogue.resolve(chosen, None))
        masked = observables.native_columns(lc.SELECTION, column)
        oracle = no.StoredColumns(observables, "resolved", selection=column)
        got = masked.evaluate(points_, scales_, **settings)
        expected = oracle.evaluate(points_, scales_, **settings)
        assert [[_bits(x) for x in r[column.key].totals] for r in got] == [
            [_bits(x) for x in r[column.key].totals] for r in expected
        ], chosen
        counts = masked.columns.last_report["counts"]
        restricted = oracle.last_report["counts"]
        for name in (
            "dressing work (modelled complex MAC per lane)",
            "reduction lanes",
        ):
            assert counts[name] == restricted[name], (chosen, name)
        assert (
            counts["tree closure rows"]
            == restricted["tree closure rows"]
            == len(chosen)
        )
        assert 0 < counts["live-row links"]
        assert 0 < counts["row-pruned families"]
    resolved = runtime.evaluate_resolved(pts, mu_r2=scales)
    selections = backend._selections
    picked = runtime.evaluate_resolved(pts, mu_r2=scales, helicities=[one, hel[0]])
    (latest,) = [value for key, value in selections.items() if key[0] == "resolved"]
    report = latest.columns.last_report["counts"]
    assert len(latest.columns.plan.columns) == 2
    assert report["tree closure rows"] == 2
    rows = [resolved.helicity_ids.index(h) for h in picked.helicity_ids]
    assert sorted(picked.helicity_ids) == sorted({one, hel[0]})
    for point in range(len(pts)):
        for k, row in enumerate(rows):
            assert [_bits(x) for x in _resolved_entry(picked, point, k)] == [
                _bits(x) for x in _resolved_entry(resolved, point, row)
            ]


@pytest.mark.loop_physics
def test_double_double_selections_and_the_correlated_refusal(artifacts):
    """Double-double: a single helicity is its resolved entry exactly, and
    two complementary halves sum to the unselected value within
    ``rounding_bound`` at the double-double unit. The correlated routes keep
    refusing selections (the v2 rule of both process types, DECISIONS
    2026-11-09 19:53 (3): the same semantics as the loop-induced and
    leading-colour selections)."""

    runtime = _runtime(artifacts("u u~ > d d~ g", "full"))
    pts, scales = _points("uux_ddxg", 1)
    hel = [h.id for h in runtime.physics.helicities]
    dd = {"arithmetic": "double-double"}
    one = _largest(runtime, pts, scales)
    resolved = runtime.evaluate_resolved(pts, mu_r2=scales, **dd)
    single = runtime.evaluate(pts, mu_r2=scales, helicities=[one], **dd)
    row = resolved.helicity_ids.index(one)
    assert [_bits(x) for x in _entry(single, 0)] == [
        _bits(x) for x in _resolved_entry(resolved, 0, row)
    ]
    total = runtime.evaluate(pts, mu_r2=scales, stability_checks=_checks(), **dd)
    half = len(hel) // 2
    halves = [
        runtime.evaluate(
            pts, mu_r2=scales, helicities=part, stability_checks=_checks(), **dd
        )
        for part in (hel[:half], hel[half:])
    ]
    from pyamplicol.loops.v2.assemble import Precision

    unit = Decimal(Precision.double_double().unit_roundoff)
    _assert_partition(total, halves, 1, unit, len(hel))
    from pyamplicol import CorrelatedRequest

    with pytest.raises(CompatibilityError, match="take no helicity"):
        runtime.evaluate_correlated(pts, mu_r2=scales, helicities=[one])
    with pytest.raises(CompatibilityError, match="take no helicity"):
        runtime.evaluate_correlated_many(
            pts,
            {"born": CorrelatedRequest("born", {})},
            mu_r2=scales,
            helicities=[one],
        )


@pytest.mark.loop_physics
def test_a_rescued_selection_evaluates_its_rows_at_every_level(
    artifacts, monkeypatch
):
    """A rescue policy with a selection (``stability=``): every level and
    every rotation evaluates the selected rows only (each native call of the
    process is the selection's masks on the template, keeping those rows),
    and the rescued value is the selection's double-double value."""

    from pyamplicol.api import VirtualRescueLevel, VirtualRescuePolicy
    from pyamplicol.loops.v2 import native_observables as no

    runtime = _runtime(artifacts("u u~ > d d~ g", "full"))
    pts, scales = _points("uux_ddxg", 1)
    one = _largest(runtime, pts, scales)
    forced = VirtualRescuePolicy(
        (
            VirtualRescueLevel("double", target=1e-300, rotations=1),
            VirtualRescueLevel("double-double", rotations=1),
        )
    )
    calls: list[Any] = []
    evaluations = no.StoredColumns._evaluations

    def recorded(self: Any, *args: Any) -> Any:
        calls.append((self.kind, self.template is not None, args[-1], args[4]))
        return evaluations(self, *args)

    monkeypatch.setattr(no.StoredColumns, "_evaluations", recorded)
    result = runtime.evaluate(pts, mu_r2=scales, helicities=[one], stability=forced)
    monkeypatch.undo()
    assert result.rescue[0].level == 1
    assert result.diagnostics[0].status == "rescued"
    carriers = {str(precision) for *_x, precision in calls}
    assert len(carriers) == 2, carriers
    # Each level: the reference and its rotation, each on the template with
    # the selection's rows.
    assert len(calls) == 4
    backend = runtime._backend._observable_runtime()
    (row,) = backend.selection_catalogue().resolve([one], None)[0]
    kept = tuple(r == row for r in backend.selection_catalogue().rows)
    for kind, template, masks, _precision in calls:
        assert (kind, template) == ("resolved", True)
        assert masks.rows == kept
    direct = runtime.evaluate(
        pts,
        mu_r2=scales,
        helicities=[one],
        arithmetic="double-double",
        stability_checks=forced.levels[1].checks,
    )
    assert [_bits(x) for x in _entry(result, 0)] == [
        _bits(x) for x in _entry(direct, 0)
    ]


@pytest.mark.loop_physics
def test_an_artifact_generated_before_the_selection_serves_it():
    """The shipped self-test artifact (``selftest_virtual/ew``, full colour,
    generated before lit-fc-hel) serves the selection without regeneration:
    a single helicity is its resolved entry bitwise and the halves sum to the
    unselected virtual within ``rounding_bound``."""

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_ew as ew

    expected = json.loads((ew.fixture_root() / "expected.json").read_text())
    pts = [
        [[str(Decimal(c)) for c in p] for p in point]
        for point in expected["points"][:1]
    ]
    scales = [str(Decimal(s)) for s in expected["mu_r2"][:1]]
    runtime = Runtime.load(ew.fixture_root() / "artifact", process=ew.NAME)
    assert runtime.physics.color_accuracy != "lc"
    hel = [h.id for h in runtime.physics.helicities]
    resolved = runtime.evaluate_resolved(pts, mu_r2=scales)
    one = _largest(runtime, pts, scales)
    single = runtime.evaluate(pts, mu_r2=scales, helicities=[one])
    row = resolved.helicity_ids.index(one)
    assert [_bits(x) for x in _entry(single, 0)] == [
        _bits(x) for x in _resolved_entry(resolved, 0, row)
    ]
    total = runtime.evaluate(pts, mu_r2=scales, stability_checks=_checks())
    half = len(hel) // 2
    halves = [
        runtime.evaluate(pts, mu_r2=scales, helicities=part, stability_checks=_checks())
        for part in (hel[:half], hel[half:])
    ]
    _assert_partition(total, halves, 1, UNIT, len(hel))
    assert not math.isnan(float(_entry(single, 0)[0]))


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "accuracy"),
    [("u u~ > d d~ g", "full"), ("u u~ > d d~ g", "lc"), ("g g > g g", "nlc")],
)
def test_the_native_formation_is_the_python_one(artifacts, process, accuracy):
    """lit-fc-hel (G5): the selection template and a resolved selection's
    columns are formed by the one native formation the SDKs share (Rust
    ``v2_runtime::template``), equal to their Python formation (the oracle,
    ``PythonSelectionTemplate`` / ``resolved_parts_oracle``): the same seed
    records, interference programs, output columns and lanes, and the same
    masks of single, several and every helicity row (and, at leading
    colour, flows); at leading colour exactly the template formed before
    lit-fc-hel."""

    from pyamplicol.loops.v2 import native_observables as no
    from pyamplicol.loops.v2 import selection as lc

    runtime = _runtime(artifacts(process, accuracy))
    backend = runtime._backend._observable_runtime()
    observables = backend._observables
    native = no.SelectionTemplate(observables)
    python = no.PythonSelectionTemplate(observables)

    def records(formed: Any) -> Any:
        companion = formed.set.companion_overlay_json
        return (
            json.loads(formed.set.overlay_json),
            None if companion is None else json.loads(companion),
            json.loads(formed.set.columns_json),
            formed.set.plan_components,
            formed.set.companion_components,
        )

    assert records(native) == records(python)
    assert native.columns == python.columns
    assert native.lanes == python.lanes
    assert native.classes == python.classes
    catalogue = backend.selection_catalogue()
    hel = [lc.helicity_id(row) for row in catalogue.rows]
    selections: list[tuple[Any, Any]] = [
        (hel[:1], None),
        (hel[: len(hel) // 2], None),
        (hel[1:], None),
        (hel, None),
    ]
    flows = [flow.id for flow in catalogue.flows if flow.basis is not None]
    if flows:
        selections += [(None, flows[:1]), (hel[::3], flows[1::2])]
    for helicities, chosen in selections:
        column = lc.column(*catalogue.resolve(helicities, chosen))
        assert native.masks(column) == python.masks(column), (helicities, chosen)
        rows = column.rows
        basis = column.basis
        formed = no.resolved_parts(observables, rows, basis)
        oracle = no.resolved_parts_oracle(observables, rows, basis)
        assert records(formed) == records(oracle)
        assert formed.columns == oracle.columns
        assert formed.lanes == oracle.lanes
