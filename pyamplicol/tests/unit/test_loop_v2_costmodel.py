# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.loop_fast

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


def test_node_work_matches_the_design_constants() -> None:
    # Late step: rows 1, X 64, rank 6 -> C(10,4) = 210 coefficients.
    assert costmodel.node_work(1, 6, 64, costmodel.STRUCTURED) == 210 * min(
        200 * 64, 64 * 64 + 300
    )
    assert costmodel.node_work(2, 0, 3, costmodel.DENSE) == 2 * 80 * 3
    assert costmodel.node_work(2, 0, 3, costmodel.DENSE, seeded=False) == 2 * 320 * 3


def _synthetic_report(seconds_scale: float) -> dict[str, object]:
    records: list[dict[str, object]] = []
    shapes = [(64, 4, 0), (32, 12, 1), (1, 64, 6)]
    for lanes in (8, 16):
        for rows_out, x, rank_in in shapes:
            for form in ("split-dense", "fused-structured", "fused-blocked"):
                constants = (
                    costmodel.STRUCTURED if "structured" in form else costmodel.DENSE
                )
                work = costmodel.node_work(
                    rows_out, rank_in, x, constants, seeded=False
                )
                # 1 G design-CMAC/s at 8 lanes, 2 G at 16 lanes.
                seconds = work * lanes / (lanes / 8 * 1e9) * seconds_scale
                records.append(
                    {
                        "kind": "node",
                        "lanes": lanes,
                        "form": form,
                        "shape": {
                            "line": "gluon",
                            "rows_out": rows_out,
                            "x": x,
                            "rank_in": rank_in,
                        },
                        "seconds_per_node": seconds,
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
    assert high == pytest.approx(
        (row.dress_dense * 1.35 / 2e9 + dispatch + row.ninjars_ms[1] / 1e3) * 1.10
    )


def test_g0_verdict_follows_the_headroom(tmp_path: Path) -> None:
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps(_synthetic_report(1.0)))
    assert costmodel.main([str(fast), "--json", str(tmp_path / "out.json")]) == 0
    summary = json.loads((tmp_path / "out.json").read_text())
    assert summary[0]["recommended_lanes"] == 16
    assert all(row["pass"] for row in summary[0]["g0"])

    slow = tmp_path / "slow.json"
    slow.write_text(json.dumps(_synthetic_report(50.0)))
    assert costmodel.main([str(slow)]) == 1
