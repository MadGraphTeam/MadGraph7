# SPDX-License-Identifier: 0BSD
"""The unchecked Table 2 re-time command (m-lit lit-unchecked (5),
``tools/loop_benchmark/retime_unchecked.py``): its dry run prints one pinned
systemd unit per row with the public-API settings and the collect step, and
launches nothing; the rows default to lit-perf's timed rows; the stage split
of the instrumented run is labelled and complete."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_fast

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "retime_unchecked", ROOT / "tools" / "loop_benchmark" / "retime_unchecked.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["retime_unchecked"] = module
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path) -> Path:
    records = [
        {
            "key": "gg_hgg",
            "tir_options": "shipped",
            "source": "lit-perf gate tool (process li_gg_hgg, ...)",
        },
        {
            "key": "li51",
            "tir_options": "shipped",
            "source": "lit-perf gate tool (process li_gg_emepmupmumg, ...)",
        },
        {"key": "gg_h", "source": "an earlier configuration (process li_gg_h)"},
    ]
    target = tmp_path / "docs" / "performance_reports" / "loop" / "data"
    target.mkdir(parents=True)
    (target / "li-timing.json").write_text(json.dumps({"records": records}))
    return tmp_path


def test_the_dry_run_prints_the_units_and_launches_nothing(tmp_path, capsys):
    module = _module()
    checkout = _checkout(tmp_path)
    assert module.timed_rows(checkout) == [
        ("li_gg_hgg", "gg_hgg"),
        ("li_gg_emepmupmumg", "li51"),
    ]
    code = module.main(
        [
            "campaign",
            "--checkout",
            str(checkout),
            "--native",
            str(tmp_path / "native"),
            "--out",
            str(tmp_path / "out"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--tag",
            "t",
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out.splitlines()
    units = [shlex.split(line) for line in printed if line.startswith("systemd-run")]
    assert [unit[2] for unit in units] == [
        "--unit=lit-lit-unchecked-retime-t-li-gg-hgg",
        "--unit=lit-lit-unchecked-retime-t-li-gg-emepmupmumg",
    ]
    for unit, process in zip(units, ("li_gg_hgg", "li_gg_emepmupmumg"), strict=True):
        assert unit[:2] == ["systemd-run", "--user"] and "--collect" in unit
        # A property is its own argument after -p (systemd-run refuses " X=").
        assert unit[unit.index("-p") + 1] == "MemoryMax=150G"
        shell = unit[-1]
        assert f"row {process}" in shell
        assert "taskset -c" in shell and "RAYON_NUM_THREADS=1" in shell
        # The licence is read inside the unit's command, never written.
        assert "SYMBOLICA_LICENSE=$(grep -o" in shell
        assert ("--auto" in shell) == (process == "li_gg_hgg")
    assert printed[-1].endswith(
        shlex.join(
            [
                "collect",
                "--checkout",
                str(checkout),
                "--native",
                str(tmp_path / "native"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
    )
    # Nothing launched, nothing written.
    assert not (tmp_path / "out").exists()


def test_the_settings_are_the_public_unchecked_call():
    module = _module()
    assert module.SETTINGS["Runtime.load"] == {"reduction_unchecked": True}
    assert module.SETTINGS["evaluate"] == {"poles": "none", "on_failure": "return"}
    assert module.SETTINGS["arms"] == {
        "tirust": {"reduction": "tirust"},
        "ninjars": {"reduction": "ninjars"},
    }


def test_the_stage_split_is_labelled_and_complete():
    module = _module()
    stages = {
        "backend_runs": {
            "tirust": {
                "effective": "tirust",
                "batches": {
                    "128": {
                        "median_seconds_per_point": 1.0,
                        "stages_per_point": {
                            "trees + CT kets": 0.1,
                            "dressing": 0.5,
                            "scatter": 0.1,
                            "reduction": 0.2,
                            "reduction: tensor moments": 0.15,
                            "squaring": 0.01,
                        },
                    }
                },
            }
        }
    }
    split = module._stage_split(stages, "tirust")
    assert split["label"] == "instrumented run: stage shares, not the timed call"
    # The run's stages verbatim (the reduction with its backend's own
    # sub-stages): the report groups them (build_summary.py _split).
    per = stages["backend_runs"]["tirust"]["batches"]["128"]["stages_per_point"]
    assert split["seconds_per_point"] == per
    assert split["total"] == 1.0 and split["batch"] == 128
    assert module._stage_split(stages, "ninjars") is None
    assert module._stage_split(None, "tirust") is None
    # The report's grouping of the record: TIRust's reduction as its tensor
    # moments and contraction, the timer's rest outside them in glue.
    import importlib.util

    path = (
        ROOT / "docs" / "performance_reports" / "loop" / "scripts" / "build_summary.py"
    )
    spec = importlib.util.spec_from_file_location("li_report_build_summary", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    grouped = builder._li_stages({"stages": split})
    assert grouped["inputs"] == 0.1 and grouped["numerator"] == 0.5
    assert grouped["moments"] == 0.15 and grouped["contraction"] == 0.0
    assert grouped["glue"] == pytest.approx(0.01 + 0.05)
    assert "ninjars" not in grouped and "masters" not in grouped


def test_cross_backend_agreement_and_born_records():
    """The two backends' agreement at the timed points (finite part, the
    largest over sectors, points either fails counted apart), and a
    Born-interfered row's record for the backend table (lit-report R4)."""

    import types

    module = _module()
    tirust = types.SimpleNamespace(
        values=[[(2.0, 0, 0)], [(1.0, 0, 0), (4.0, 0, 0)], None]
    )
    ninjars = types.SimpleNamespace(
        values=[[(2.0, 0, 0)], [(1.0, 0, 0), (4.4, 0, 0)], [(1.0, 0, 0)]]
    )
    cross = module.cross_backend(tirust, ninjars)
    assert cross["points"] == 2 and cross["failed"] == 1
    assert cross["max_relative"] == pytest.approx(0.4 / 4.4)
    assert cross["above_1e-5"] == 1 and cross["per_point"][0] == 0.0
    record = {
        "key": "gg_ggg",
        "process": "gg_ggg",
        "revision": "b" * 40,
        "backend": "tirust",
        "batch": 128,
        "seconds_per_point": 0.01,
        "failed_points": [0],
        "cross_backend": {k: v for k, v in cross.items() if k != "per_point"},
    }
    born = module._born_record(record)
    assert born["seconds_per_point"] == 0.01 and "failed_points" not in born
    assert born["agreement"] == {
        "max_relative": cross["max_relative"],
        "points": 2,
        "metric": cross["metric"],
    }
    failed = dict(record, backend="ninjars", failed_points=[3])
    del failed["seconds_per_point"]
    failed = module._born_record(failed)
    assert failed["failed_points"] == 3 and "agreement" not in failed
