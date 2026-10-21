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


@pytest.mark.loop_fast
def test_g0_verdict_follows_the_headroom(tmp_path: Path) -> None:
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps(_synthetic_report(1.0)))
    assert costmodel.main([str(fast), "--json", str(tmp_path / "out.json")]) == 0
    summary = json.loads((tmp_path / "out.json").read_text())
    assert summary[0]["gate_lanes"] == summary[0]["fastest_lanes"] == 16
    assert all(row["pass"] for row in summary[0]["g0"])

    slow = tmp_path / "slow.json"
    slow.write_text(json.dumps(_synthetic_report(50.0)))
    assert costmodel.main([str(slow)]) == 1


@pytest.mark.loop_fast
def test_robust_lanes_avoids_a_collapse_in_any_report() -> None:
    def calibrations(*rates: float):
        return [
            costmodel.Calibration(lanes, (rate, rate), (), 0.0, 0.0)
            for lanes, rate in zip((8, 16, 32), rates, strict=True)
        ]

    # Best at 8 lanes in two code generations, collapsed at 8 in the third.
    reports = [
        calibrations(1.4, 1.34, 1.2),
        calibrations(1.9, 1.75, 1.5),
        calibrations(0.8, 1.9, 1.8),
    ]
    assert costmodel.robust_lanes(reports) == 16


@pytest.mark.loop_fast
def test_gate_is_evaluated_at_one_lane_count_for_every_report(tmp_path: Path) -> None:
    # Fastest at 8 lanes in the first report, collapsed at 8 in the second.
    paths = []
    for name, rates in (("a", {8: 2.0e9, 16: 1.9e9}), ("b", {8: 0.5e9, 16: 2e9})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_synthetic_report(1.0, rates)))
        paths.append(str(path))
    out = tmp_path / "out.json"
    assert costmodel.main([*paths, "--json", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert [entry["fastest_lanes"] for entry in summary] == [8, 16]
    assert [entry["gate_lanes"] for entry in summary] == [16, 16]
    for entry in summary:
        at16 = next(c for c in entry["calibrations"] if c["lanes"] == 16)
        calibration = costmodel.Calibration(
            16,
            tuple(at16["throughput_design_cmac_per_s"]),
            (),
            at16["per_node_seconds_lower_bound"],
            at16["per_edge_call_seconds_lower_bound"],
        )
        assert entry["g0"] == json.loads(json.dumps(costmodel.g0_rows(calibration)))

    # At an explicit 8 lanes the collapsed report fails the gate.
    assert costmodel.main([*paths, "--lanes", "8", "--json", str(out)]) == 1
    summary = json.loads(out.read_text())
    assert {entry["gate_lanes"] for entry in summary} == {8}
    passed = [all(row["pass"] for row in entry["g0"]) for entry in summary]
    assert passed == [True, False]


@pytest.mark.loop_fast
def test_trie_shapes_accept_the_families_export() -> None:
    nodes = [
        {"rows_out": 4, "x": 8, "rank_in": 2, "count": 3},
        {"rows_out": 2, "x_rows": 6, "rank_in": 0, "edges": 2, "work": 0},
    ]
    structured, dense, count = costmodel.trie_shape_work(nodes)
    assert count == 4
    assert structured == 3 * costmodel.node_work(
        4, 2, 8, costmodel.STRUCTURED
    ) + costmodel.node_work(2, 0, 6, costmodel.STRUCTURED)
    assert dense == 3 * costmodel.node_work(4, 2, 8, costmodel.DENSE) + (
        costmodel.node_work(2, 0, 6, costmodel.DENSE)
    )


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
