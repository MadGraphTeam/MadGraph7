# SPDX-License-Identifier: 0BSD
"""MadGraph7: a one-loop artifact generated with packed payloads
(``generation.pack_payloads = true``) evaluates to the same bits as the loose
one, and its loop runtime plan is the pacbin member kind 12
(``LOOP_RUNTIME_PLAN``; upstream pyAmpliCol numbers it 11, which is
``DECLARED_PAYLOAD`` here).

Generation and evaluation run in child interpreters through the CLI, as a
user runs them, so this process never drives Symbolica itself.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.generation.evaluator_container import PacbinMemberKind, PacbinReader

pytestmark = pytest.mark.loop_fast

# The cheapest virtual process of the shipped loop model.
PROCESS = "d d~ > z [QCD]"
PROCESS_ID = "d_dbar_to_z"
PLAN = "schedule/loop-plan-v4.bin.zlib"


def _cli(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pyamplicol", *arguments],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _plan_kind(artifact: Path) -> PacbinMemberKind:
    runtime = artifact / "processes" / PROCESS_ID / "loop-runtime.pacbin"
    with PacbinReader.open(runtime) as reader:
        return reader.member(PLAN).kind


def test_packed_loop_artifact_matches_the_loose_one(tmp_path: Path) -> None:
    loose, packed = tmp_path / "loose", tmp_path / "packed"
    _cli("generate", PROCESS, str(loose), "--model", "loop_sm")
    _cli(
        "generate",
        PROCESS,
        str(packed),
        "--model",
        "loop_sm",
        "--set",
        "generation.pack_payloads=true",
    )

    assert not (loose / "payloads.pacbin").exists()
    assert (packed / "payloads.pacbin").is_file()
    # The per-process records live in the container; the loop runtime stays
    # its own (loose) pacbin next to it.
    assert not (packed / "processes" / PROCESS_ID / "execution.json").exists()
    with PacbinReader.open(packed / "payloads.pacbin") as reader:
        kinds = {member.logical_path: member.kind for member in reader.members}
    assert kinds[f"processes/{PROCESS_ID}/execution.json"] is (
        PacbinMemberKind.DECLARED_PAYLOAD
    )
    for artifact in (loose, packed):
        assert _plan_kind(artifact) is PacbinMemberKind.LOOP_RUNTIME_PLAN
        assert int(PacbinMemberKind.LOOP_RUNTIME_PLAN) == 12

    results = [
        json.loads(_cli("evaluate", str(artifact), "--json"))
        for artifact in (loose, packed)
    ]
    for result in results:
        assert result["contribution"] == "virtual"
        assert result["statuses"] and set(result["statuses"]) == {"success"}
        values = [v for point in result["values"] for row in point for v in row]
        assert values and all(math.isfinite(v) for v in values)
    for key in ("values", "born_values", "statuses", "poles"):
        assert results[0][key] == results[1][key], key
