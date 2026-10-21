# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


costmodel = _load("_test_loop_v2_costmodel", ROOT / "tools" / "loop_v2_costmodel.py")


def _row(process: str, colour: str = "FC"):
    return next(
        r for r in costmodel.PROCESSES if (r.process, r.colour) == (process, colour)
    )


@pytest.mark.loop_fast
def test_design_recipe_reproduces_table_1a() -> None:
    # DESIGN Table 1a targets (with the default re-test).
    expected = {
        ("gg>gggg", "FC"): (0.174, 0.465),
        ("gg>ggggg", "FC"): (3.38, 9.03),
        ("uu~>e-e+dd~g", "FC"): (0.00677, 0.0201),
        ("uu~>e-e+ggg", "LC"): (0.0045, 0.0106),
    }
    for (process, colour), band in expected.items():
        low, high = costmodel.predict(
            _row(process, colour), None, "reduction-direction"
        )
        assert low == pytest.approx(band[0], rel=5e-3)
        assert high == pytest.approx(band[1], rel=5e-3)


@pytest.mark.loop_fast
def test_node_work_matches_the_design_constants() -> None:
    # Late step: rows 1, X 64, rank 6 -> C(10,4) = 210 coefficients.
    assert costmodel.node_work(1, 6, 64, costmodel.STRUCTURED) == 210 * min(
        200 * 64, 64 * 64 + 300
    )
    assert costmodel.node_work(2, 0, 3, costmodel.DENSE) == 2 * 80 * 3
    assert costmodel.node_work(2, 0, 3, costmodel.DENSE, seeded=False) == 2 * 320 * 3


def _synthetic_report(
    seconds_scale: float, rates: dict[int, float] | None = None
) -> dict[str, object]:
    """Nodes that run at ``rates[lanes]`` design-CMAC/s (default 1 G at 8
    lanes, 2 G at 16), plus a slow small-X node outside the calibration."""
    rates = rates or {8: 1e9, 16: 2e9}
    records: list[dict[str, object]] = []
    shapes = [("probe", 64, 4, 0), ("probe", 32, 12, 1), ("probe", 1, 64, 6)]
    shapes.append(("small-x", 16, 2, 2))
    for lanes, rate in rates.items():
        for shape_set, rows_out, x, rank_in in shapes:
            for form in ("split-dense", "fused-structured", "fused-blocked"):
                constants = (
                    costmodel.STRUCTURED if "structured" in form else costmodel.DENSE
                )
                work = costmodel.node_work(
                    rows_out, rank_in, x, constants, seeded=False
                )
                seconds = work * lanes / rate * seconds_scale
                if shape_set == "small-x":
                    seconds *= 1e3 if form != "fused-blocked" else 1e2
                shape = {
                    "set": shape_set,
                    "line": "gluon",
                    "rows_out": rows_out,
                    "x": x,
                    "rank_in": rank_in,
                }
                records.append(
                    {
                        "kind": "node",
                        "lanes": lanes,
                        "form": form,
                        "shape": shape,
                        "seconds_per_node": seconds,
                        "gcmac_per_s": work * lanes / seconds / 1e9,
                    }
                )
                if form == "fused-structured" and shape_set == "probe":
                    records.append(
                        {
                            "kind": "poly-layout",
                            "lanes": lanes,
                            "form": "poly-apply-segment",
                            "shape": shape,
                            "gcmac_per_s": 0.5 * work * lanes / seconds / 1e9,
                        }
                    )
            records.append(
                {
                    "kind": "dispatch",
                    "lanes": lanes,
                    "form": "fused-blocked",
                    "per_node_seconds": 1e-7,
                    "per_edge_call_seconds": 1e-8,
                }
            )
    return {"target_features": {}, "records": records}


@pytest.mark.loop_fast
def test_calibration_uses_measured_throughput_and_picks_lanes() -> None:
    report = _synthetic_report(1.0)
    calibrations = [costmodel.calibrate(report, lanes) for lanes in (8, 16)]
    assert [c.throughput[0] for c in calibrations] == pytest.approx([1e9, 2e9])
    assert [c.throughput[1] for c in calibrations] == pytest.approx([1e9, 2e9])
    chosen = costmodel.recommended_lanes(calibrations)
    assert chosen.lanes == 16
    assert chosen.per_shape[0]["best_form"] == "fused-structured"
    assert chosen.per_shape[0]["best_dense_form"] in costmodel.DENSE_FORMS

    row = _row("uu~>e-e+dd~g")
    low, high = costmodel.predict(row, chosen, "none")
    dispatch = (row.trie_nodes * 1e-7 + (row.trie_nodes + row.skeletons) * 1e-8) / 16
    assert low == pytest.approx(
        (row.dress_structured * 1.20 / 2e9 + dispatch + row.ninjars_ms[0] / 1e3) * 1.05
    )
    # The upper end keeps the design's allowance for gathers and dispatch.
    upper = 2e9 * (1 - costmodel.ALLOWANCE)
    assert high == pytest.approx(
        (row.dress_dense * 1.35 / upper + dispatch + row.ninjars_ms[1] / 1e3) * 1.10
    )
    _, measured_only = costmodel.predict(row, chosen, "none", allowance=0.0)
    assert measured_only == pytest.approx(
        (row.dress_dense * 1.35 / 2e9 + dispatch + row.ninjars_ms[1] / 1e3) * 1.10
    )


@pytest.mark.loop_fast
def test_small_x_shapes_inform_forms_but_not_the_calibration() -> None:
    report = _synthetic_report(1.0)
    calibration = costmodel.calibrate(report, 16)
    assert [s["rank_in"] for s in calibration.per_shape] == [0, 1, 6]
    small = [s for s in costmodel.fastest_forms(report, 16) if s["set"] == "small-x"]
    assert len(small) == 1 and small[0]["best_form"] == "fused-blocked"
    assert costmodel.poly_layout_ratio(report, 16) == pytest.approx(0.5)


def _program_report(
    seconds: dict[tuple[str, str, int], float], policy: str = "rule"
) -> dict[str, object]:
    """Program records of the bench: ``seconds[(process, colour, lanes)]`` per
    point for ``policy``, and each policy twice as slow at the other lanes."""
    records: list[dict[str, object]] = []
    for (process, colour, lanes), value in seconds.items():
        work = {"best/seed": 1_000, "best/step": 9_000}
        for name, scale in ((policy, 1.0), ("oracle", 0.9), ("split-structured", 1.5)):
            records.append(
                {
                    "kind": "program",
                    "process": process,
                    "colour": colour,
                    "lanes": lanes,
                    "policy": name,
                    "steps": 10,
                    "edges": 20,
                    "links": 300,
                    "build_seconds": 0.0,
                    "walk_seconds": value * scale * lanes,
                    "seconds_per_point": value * scale,
                    "reference_best_gcmac_per_s": 1e4 / (value * scale) / 1e9,
                    "reference_work": work,
                    "details": {},
                }
            )
        records.append(
            {
                "kind": "program-walk",
                "process": process,
                "colour": colour,
                "lanes": lanes,
                "policy": "walk",
                "steps": 10,
                "edges": 20,
                "links": 300,
                "build_seconds": 0.0,
                "walk_seconds": 1e-6 * lanes,
                "seconds_per_point": 1e-6,
                "reference_work": work,
            }
        )
    return {"target_features": {}, "records": records}


GATED = ("gg>gggg", "uu~>e-e+dd~g", "uu~>e-e+ggg", "gg>tt~gg", "gg>tt~e-e+g")


@pytest.mark.loop_fast
def test_measured_programs_replace_the_dressing_term() -> None:
    timings = {(p, "FC", 16): 1e-3 for p in GATED}
    timings[("gg>gggg", "LC", 16)] = 5e-4
    calibration = costmodel.program_calibration(_program_report(timings), 16)
    row = _row("uu~>e-e+dd~g")
    assert costmodel.dressing_seconds(row, calibration) == (1e-3, 1e-3)
    low, high = costmodel.predict_measured(row, calibration, "none")
    assert low == pytest.approx((1e-3 + row.ninjars_ms[0] / 1e3) * 1.05)
    assert high == pytest.approx((1e-3 + row.ninjars_ms[1] / 1e3) * 1.10)
    _, checked = costmodel.predict_measured(row, calibration, "reduction-direction")
    assert checked == pytest.approx((1e-3 + 2.0 * row.ninjars_ms[1] / 1e3) * 1.10)

    # Unmeasured rows scale from their class by Appendix B structured work
    # (lower end) and from the costliest measured row per unit work (upper).
    seven = _row("gg>ggggg")
    base = _row("gg>gggg")
    per_work = {
        key: timings[(*key, 16)] / _row(*key).dress_structured
        for key in {(p, c) for p, c, _ in timings}
    }
    low, high = costmodel.dressing_seconds(seven, calibration)
    assert low == pytest.approx(1e-3 * seven.dress_structured / base.dress_structured)
    assert high == pytest.approx(max(per_work.values()) * seven.dress_structured)
    lc = costmodel.dressing_seconds(_row("gg>ggggg", "LC"), calibration)[0]
    assert lc == pytest.approx(
        5e-4
        * _row("gg>ggggg", "LC").dress_structured
        / _row("gg>gggg", "LC").dress_structured
    )


@pytest.mark.loop_fast
def test_g0_verdict_follows_the_headroom(tmp_path: Path) -> None:
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps(_program_report({(p, "FC", 16): 1e-3 for p in GATED})))
    out = tmp_path / "out.json"
    assert costmodel.main([str(fast), "--json", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert summary[0]["gate_lanes"] == 16
    assert all(row["pass"] and row["measured"] for row in summary[0]["g0"])

    # uu~ -> e-e+dd~g: gate 1.3 x 34.7 ms, so x1.3 headroom needs the upper
    # target (dressing + NinjaRS 3.27 ms) x 1.10 below 34.7 ms.
    slow = dict.fromkeys(((p, "FC", 16) for p in GATED), 1e-3)
    slow[("uu~>e-e+dd~g", "FC", 16)] = 0.030
    path = tmp_path / "slow.json"
    path.write_text(json.dumps(_program_report(slow)))
    assert costmodel.main([str(path), "--json", str(out)]) == 1
    verdict = {
        row["process"]: row["pass"] for row in json.loads(out.read_text())[0]["g0"]
    }
    assert verdict == {p: p != "uu~>e-e+dd~g" for p in GATED}


@pytest.mark.loop_fast
def test_reports_without_programs_do_not_pass_the_gate(tmp_path: Path) -> None:
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_synthetic_report(1.0)))
    assert costmodel.main([str(probe)]) == 1


@pytest.mark.loop_fast
def test_lane_count_holds_up_in_every_program_and_report() -> None:
    def report(at: dict[int, float]):
        return _program_report(
            {(p, "FC", lanes): seconds for p in GATED for lanes, seconds in at.items()}
        )

    # Fastest at 8 lanes in one code generation, collapsed at 8 in the other.
    reports = [report({8: 1.0, 16: 1.05, 32: 1.3}), report({8: 2.5, 16: 1.0, 32: 1.1})]
    assert costmodel.program_lanes(reports[:1]) == 8
    assert costmodel.program_lanes(reports) == 16
    # More lanes only when measurably (> 5 %) faster.
    assert costmodel.program_lanes([report({16: 1.0, 32: 0.97})]) == 16
    assert costmodel.program_lanes([report({8: 1.2, 16: 1.1, 32: 1.0})]) == 32
    with pytest.raises(ValueError, match="every report"):
        costmodel.program_lanes([report({8: 1.0}), report({16: 1.0})])
    with pytest.raises(ValueError, match="tolerance"):
        costmodel.program_lanes([report({8: 1.0, 16: 2.0}), report({8: 2.0, 16: 1.0})])


@pytest.mark.loop_fast
def test_selection_rule_recovers_step_models_and_picks_the_cheaper_form() -> None:
    lanes = 16
    truth = {
        "fused-structured": costmodel.StepModel(0.0, 3e-8, 1 / 1.5e9),
        "split-structured": costmodel.StepModel(2e-6, 1e-9, 1 / 1.4e9),
        "fused-blocked": costmodel.StepModel(0.0, 1.2e-8, 1 / 1.4e9),
        "fused-dense": costmodel.StepModel(0.0, 4e-8, 1 / 1.4e9),
        "split-dense": costmodel.StepModel(3e-6, 1e-9, 1 / 1.3e9),
    }
    # Seeded-like steps (few coefficients per iteration) favour split, deep
    # steps (many coefficients) favour fused.
    shapes = [(rows, m) for rows in (1, 4, 16, 64) for m in (1, 5, 15, 35, 70)]
    steps, forms = [], {}
    for form, model in truth.items():
        scale = 1.5 if "split" in form else 1.0
        costs = [
            [int(rows * 400 * m * scale), int(rows * 400 * scale)] for rows, m in shapes
        ]
        forms[form] = {
            "cost": costs,
            "own_seconds": [model.seconds(cost, lanes) for cost in costs],
        }
    for rows, m in shapes:
        steps.append([0, rows, 2, 2 * rows, 0, 4, 4, 400 * rows * m, 600 * rows * m])
    record = {
        "kind": "program-steps",
        "process": "gg>gggg",
        "colour": "FC",
        "lanes": lanes,
        "fields": [
            "seeded",
            "rows",
            "edges",
            "links",
            "rank_in",
            "d_alpha",
            "d_out",
            "reference_fused",
            "reference_split",
        ],
        "steps": steps,
        "forms": forms,
    }
    fitted = costmodel.fit_step_model(
        forms["split-structured"]["cost"],
        forms["split-structured"]["own_seconds"],
        lanes,
    )
    for got, want in zip(
        (fitted.per_step, fitted.per_iteration, fitted.per_cmac),
        (2e-6, 1e-9, 1 / 1.4e9),
        strict=True,
    ):
        assert got == pytest.approx(want, rel=1e-6)
    rules = costmodel.selection_rules({"records": [record]}, lanes)
    (program,) = rules["programs"]
    assert program["rule_over_oracle"] == pytest.approx(1.0)
    assert program["best_single_form_over_oracle"] > 1.0
    assert program["counted_over_oracle"] > 1.0  # counted work misses the overheads
    with pytest.raises(ValueError, match="per-step records"):
        costmodel.selection_rules({"records": [record]}, 8)


def _check_program(program: dict) -> None:
    """Row links and vertex groups stay inside the objects they index."""
    steps, contents, seeds = program["steps"], program["contents"], program["seeds"]
    consumed = collections.Counter()
    for step in steps:
        for edge in step["edges"]:
            if edge["seed"] >= 0:
                assert edge["child"] == -1 and step["seeded"]
                source_rows = seeds[edge["seed"]]
            else:
                consumed[edge["child"]] += 1
                assert steps[edge["child"]]["depth"] == step["depth"] - 1
                source_rows = steps[edge["child"]]["rows"]
            links = list(zip(*[iter(edge["links"])] * 3, strict=True))
            assert links
            for output, source, block in links:
                assert output < step["rows"] and source < source_rows
                assert block < contents[edge["content"]]["rows"]
            for part in edge["split"]:
                assert set(zip(*[iter(part["links"])] * 3, strict=True)) <= set(links)
                assert part["group"] < len(step["groups"])
        for group in step["groups"]:
            assert set(group["targets"]) <= set(range(step["rows"]))
        if step["trace"]:
            assert step["rows"] == 1 and step["d_alpha"] == step["d_out"]
    assert consumed == {i: 1 for i, step in enumerate(steps) if not step["trace"]}


@pytest.mark.loop_physics
def test_programs_are_the_reference_executor_walk(loop_sm_compiled_model) -> None:
    """The exported program is the schedule-B walk of the reference executor:
    the same per-form work, step by step, for gluon, quark-line, several
    quark-line and massive loops, in full and leading colour."""

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.models.loading import load_compiled_model

    counts = costmodel._operation_counts_tool()
    prepared = counts.prepare(load_compiled_model(loop_sm_compiled_model))
    cases = [
        ({"incoming": ["G", "G"], "outgoing": ["G", "G"], "born_orders": {"QCD": 2}}, c)
        for c in ("LC", "FC")
    ]
    cases += [
        (
            {
                "incoming": ["u", "u~"],
                "outgoing": ["d", "d~", "G"],
                "born_orders": {"QCD": 3},
            },
            "FC",
        ),
        (
            {
                "incoming": ["G", "G"],
                "outgoing": ["t", "t~"],
                "born_orders": {"QCD": 2},
            },
            "FC",
        ),
    ]
    for entry, colour in cases:
        plan, kinematics, seeds = costmodel.program_plan(prepared, entry, colour)
        program = costmodel.trie_program(plan, kinematics, seeds, "test", colour)
        executor = rx.ReferenceExecutor(kinematics, seeds, numeric=False)
        assert program["work"] == dict(executor.run_b(plan).counts)
        assert len(program["steps"]) == program["work"]["dress-steps"]
        _check_program(program)
    assert costmodel.program_file_name("gg>tt~e-e+g", "FC") == "gg_ttxe-e+g-FC.json"
