# SPDX-License-Identifier: 0BSD
"""``pyamplicol check`` (m-lit lit-examples, USER 2026-11-09 ~08:30): the pole
and Ward checks of a one-loop request regenerated in memory
(``loops.v2.checks``, ``api.services.check``).

* **Born-interfered pass** (u u~ > g g, NLO QCD, loop_sm): the poles of every
  point and sector against Catani's operator and the Ward identity of both
  gluons (c = 1 and i) pass in binary64, from the run card and from the
  generated artifact; the Goldstone row reads "not applicable" with its
  reason; the CLI exits 0 and its JSON carries every row.
* **Loop-induced pass** (g g > a a, five massless quark flavours): the
  finiteness probe and the Ward probe of one leg per gauge-boson species
  pass in binary64 at every point.
* **The failing case**: the same request on a copy of loop_sm without the
  R2 vertex of its external particles (``loop_r2_dropped``) fails the Ward
  check in binary64 far beyond ten times its estimate and again after the
  double-double fallback; the CLI exits 1.
* **Exit codes and statuses** of the report (pure records), and a process
  with nothing to check (an exact colour zero, h > g a) exits 3.
* **Rows without a criterion** (pure records): a binary64 pole or Ward row
  without its estimate or out of the binary64 scope is "not run" (the next
  carrier decides; with binary64 alone no pass), and a sector without a
  pole row is listed "no prediction", never omitted.
* **An artifact is checked on what it stores**: an API-generated artifact
  whose configuration names another process is checked on its stored
  process; a model that is not the stored one (content hash) or a source
  that does not resolve is refused (``ArtifactError``), and ``--model`` /
  ``model=`` names the stored model's source.

The Goldstone-equivalence path itself (an external massive vector in
't Hooft-Feynman gauge) is lit-probe's (``test_loop_v2_induced_probes``,
``test_loop_v2_induced_ew_gauge``): ``check`` calls the same
``induced.goldstone_point``; here its "not applicable" row is tested.
"""

from __future__ import annotations

import io
import json
import shutil
from importlib import resources
from pathlib import Path

import pytest

from .loop_r2_dropped import declaration, drop_r2_vertex

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"


def _card(
    root: Path,
    expression: str,
    *,
    source: str = "loop_sm",
    fixed: dict[str, str] | None = None,
    veto: tuple[str, ...] = (),
    action: str = "check",
    points: int = 1,
) -> Path:
    fixed = fixed or {"WT": "0"}
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(
        f"""action = "{action}"
[model]
source = {json.dumps(source)}
restriction = "no_b_mass"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
[loop]
r2_strategy = "model"
{declaration(source)}[loop.preparation]
fixed_parameters = {{{", ".join(f"{k} = {json.dumps(v)}" for k, v in fixed.items())}}}
[generation]
output = "artifact"
emit_api_bundle = false
particle_veto = {json.dumps(list(veto))}
[check]
points = {points}
"""
    )
    return card


def _nf5() -> dict[str, str]:
    fixture = json.loads((FIXTURES / "li_gg_aa_nf5.json").read_text())
    return dict(fixture["scheme_data"]["fixed_parameters"])


def _cli(*arguments: str) -> tuple[int, dict]:
    from pyamplicol.cli.main import run_cli

    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        ["check", *arguments, "--json", "--progress", "off"],
        stdout=stdout,
        stderr=stderr,
    )
    return code, json.loads(stdout.getvalue()) if stdout.getvalue() else {}


def _rows(report, check: str):
    (process,) = report.processes
    return [row for row in process.rows if row.check == check]


# --------------------------------------------------------------- records


@pytest.mark.loop_fast
def test_statuses_and_exit_codes_follow_the_rows():
    from pyamplicol.loops.v2 import checks

    def process(*verdicts):
        rows = tuple(
            checks.CheckRow("poles", "s", 0, "double", verdict) for verdict in verdicts
        )
        return checks.ProcessCheck(
            "p", "a > b", "virtual", "zero", (), "ninjars", rows, 0.0
        )

    def report(*processes):
        return checks.CheckReport(
            "x", "card", 1, 0, "auto", ("double",), processes, "", 0.0
        )

    passed = process(checks.PASS, checks.NOT_APPLICABLE, checks.NO_PREDICTION)
    failed = process(checks.PASS, checks.FAIL)
    empty = process(checks.NOT_APPLICABLE, checks.NO_PREDICTION)
    not_run = process(checks.PASS, checks.NOT_RUN)
    assert passed.status == checks.PASSED
    assert failed.status == checks.FAILED
    assert empty.status == checks.NOTHING_CHECKED
    assert not_run.status == checks.NOTHING_CHECKED
    assert report(passed).exit_code == checks.EXIT_PASSED == 0
    assert report(passed, failed).exit_code == checks.EXIT_FAILED == 1
    assert report(passed, empty).exit_code == checks.EXIT_NOTHING_CHECKED == 3
    assert report(failed, empty).exit_code == checks.EXIT_FAILED
    assert len({0, checks.EXIT_FAILED, checks.EXIT_NOTHING_CHECKED, 2}) == 4


@pytest.mark.loop_fast
def test_the_catani_criterion_is_the_harness_rule():
    """|pole - Catani| <= 10 E + 1e-12 |Catani| per coefficient (binary64)
    and 1e-12 relative in double-double, as stored_checks.py applied it."""

    from pyamplicol.loops.v2.checks import catani_comparison, catani_dd_comparison

    values = [0.0, 2.0 + 1e-9, -3.0]
    poles = (2.0, -3.0)
    _c, deviations, bounds, ok = catani_comparison(values, poles, [0, 1e-10, 0])
    assert ok and bounds[0] == pytest.approx(10 * 1e-10 + 2e-12)
    assert deviations[0] == pytest.approx(1e-9)
    assert not catani_comparison(values, poles, [0, 1e-11, 0])[3]
    assert not catani_comparison(values, poles, None)[3]
    assert catani_dd_comparison([0.0, 2.0 * (1 + 1e-13), -3.0], poles)[2]
    assert not catani_dd_comparison([0.0, 2.0 * (1 + 1e-11), -3.0], poles)[2]


@pytest.mark.loop_fast
def test_rows_without_a_criterion_are_not_run_and_no_sector_is_omitted():
    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks

    first, second = (("QCD", 4),), (("QCD", 2), ("QED", 2))

    def laurent(*values):
        return NS(coefficients=list(values))

    result = NS(
        stability=None,
        components={"virtual": {first: laurent(0.0, 2.0, -3.0)}},
        ir_poles={first: (2.0, -3.0)},
    )
    rows = checks._pole_rows_binary64(result, 0, 1e-5)
    assert {row.verdict for row in rows.values()} == {checks.NOT_RUN}
    assert all(row.reason.startswith("no criterion") for row in rows.values())
    scoped = NS(
        relative=1e-3,
        absolute={first: laurent(0.0, 1e-12, 1e-12)},
        scale={first: laurent(1.0, 1.0, 1.0)},
    )
    rows = checks._pole_rows_binary64(
        NS(**{**vars(result), "stability": scoped}), 0, 1e-5
    )
    assert {row.verdict for row in rows.values()} == {checks.NOT_RUN}
    assert all("binary64 scope" in row.reason for row in rows.values())
    # A sector with neither a pole row nor a listed reason: "no prediction".
    slice_ = NS(unpredicted_sectors=lambda: {})
    (row,) = checks._unpredicted_rows(slice_, [first, second], True, 0, {first})
    assert row.verdict == checks.NO_PREDICTION and row.subject == "QCD^2 QED^2"
    # A Ward row whose virtual part is out of the binary64 scope: not run
    # (its Born part passed), a failure when its Born part fails.
    ward = {
        "leg": 3,
        "factor": (1, 0),
        "born": 0.0,
        "born_passed": True,
        "parts": {"virtual": [1e-3]},
        "estimate": [1e-6],
        "estimate_over": [1.0],
        "o1_over": [0.5],
        "in_scope": False,
        "passed": False,
    }
    assert checks._ward_row(ward, "double", 0).verdict == checks.NOT_RUN
    failed = checks._ward_row({**ward, "born": 1.0, "born_passed": False}, "double", 0)
    assert failed.verdict == checks.FAIL


# --------------------------------------------------------------- physics


@pytest.mark.loop_physics
def test_a_born_interfered_request_passes_from_card_and_artifact(tmp_path):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.api.services import check
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2 import checks

    card = _card(tmp_path / "card", "u u~ > g g")
    report = check(card)
    assert report.exit_code == 0 and report.source_kind == "card"
    (process,) = report.processes
    assert process.contribution == "virtual" and process.status == checks.PASSED
    poles = _rows(report, "poles")
    assert {row.subject.split()[-1] for row in poles} == {"1/eps", "1/eps^2"}
    assert all(row.verdict == checks.PASS for row in poles)
    assert all(row.carrier == "double" and row.ratio <= 1 for row in poles)
    # A binary64 rung: no improvement (off by default in binary64).
    assert all(row.kinematics is None for row in poles)
    ward = _rows(report, "ward")
    assert sorted(row.subject for row in ward) == [
        "leg 3, c = 1",
        "leg 3, c = i",
        "leg 4, c = 1",
        "leg 4, c = i",
    ]
    assert all(row.verdict == checks.PASS for row in ward)
    (goldstone,) = _rows(report, "goldstone")
    assert goldstone.verdict == checks.NOT_APPLICABLE and goldstone.reason
    # The generated artifact: its request regenerated from config/effective.toml.
    config = resolve_config(_card(tmp_path / "gen", "u u~ > g g", action="generate"))
    output = Path(config.effective.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse("u u~ > g g", name="p"),)), output
    )
    code, payload = _cli(str(output), "--points", "1")
    assert code == 0 and payload["exit_code"] == 0
    assert payload["source_kind"] == "artifact"
    (entry,) = payload["processes"]
    assert entry["status"] == "passed"
    assert {row["check"] for row in entry["rows"]} == {"poles", "ward", "goldstone"}
    assert set(entry["rows"][0]) >= {
        "check",
        "subject",
        "point",
        "carrier",
        "verdict",
        "ratio",
        "deviation",
        "estimate",
        "fallback",
        "reason",
    }


@pytest.mark.loop_physics
def test_a_loop_induced_request_passes_its_probes(tmp_path):
    from pyamplicol.api.services import check
    from pyamplicol.loops.v2 import checks

    card = _card(tmp_path, "g g > a a", fixed=_nf5(), veto=("t",), points=2)
    report = check(card)
    assert report.exit_code == 0
    (process,) = report.processes
    assert process.contribution == "loop-induced"
    poles, ward = _rows(report, "poles"), _rows(report, "ward")
    assert [row.point for row in poles] == [0, 1]
    assert all(row.subject.startswith("finiteness") for row in poles)
    # One Ward probe per gauge-boson species: the first gluon and photon.
    assert sorted({row.subject for row in ward}) == ["leg 1", "leg 3"]
    for row in poles + ward:
        assert row.verdict == checks.PASS and row.carrier == "double"
        assert row.ratio <= 1 and row.fallback is None
    (goldstone,) = _rows(report, "goldstone")
    assert goldstone.verdict == checks.NOT_APPLICABLE


@pytest.mark.loop_physics
def test_a_dropped_r2_vertex_fails_the_ward_check(tmp_path):
    from pyamplicol.api.services import check
    from pyamplicol.loops.v2 import checks

    shipped = Path(
        str(resources.files("pyamplicol").joinpath("assets", "models", "ufo"))
    )
    dropped = tmp_path / "loop_sm_r2_dropped"
    shutil.copytree(shipped / "loop_sm", dropped)
    assert drop_r2_vertex(dropped, [21, 21, 22, 22])
    card = _card(
        tmp_path / "card",
        "g g > a a",
        source=str(dropped),
        fixed=_nf5(),
        veto=("t",),
    )
    report = check(card)
    assert report.exit_code == checks.EXIT_FAILED
    ward = _rows(report, "ward")
    assert ward
    for row in ward:
        # binary64 far beyond ten times its estimate, then double-double too.
        assert row.verdict == checks.FAIL and row.carrier == "double-double"
        # The double-double rung evaluates its lifted point improved in
        # double-double (m-lit lit-psimprove (G5)); beams along z: MadLoop's
        # z-shift (the default) produces it.
        assert row.kinematics == "z-shift", row
        assert row.fallback.startswith("double ratio")
        assert float(row.fallback.split()[2]) > 1e5
        assert row.ratio > 1e5
    # R2 is finite: the finiteness probe passes.
    assert all(row.verdict == checks.PASS for row in _rows(report, "poles"))
    code, payload = _cli(str(card))
    assert code == 1 and payload["exit_code"] == 1
    assert payload["processes"][0]["status"] == "failed"
    # The report names the card it checked.
    assert payload["source"] == str(card.resolve())
    assert payload["source_kind"] == "card"


@pytest.mark.loop_physics
def test_an_artifact_is_checked_on_its_stored_processes_and_model(tmp_path):
    """The artifact's configuration names u u~ > g g, the API call generated
    g g > a a (``api_p``) from a copy of loop_sm given by path: check takes
    the stored process; the copy's name alone (``effective.toml``) does not
    resolve, a model that is not the stored one is refused."""

    from pyamplicol.api import ArtifactError, Generator, ProcessRequest, ProcessSet
    from pyamplicol.api.services import check
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2 import checks

    shipped = Path(
        str(resources.files("pyamplicol").joinpath("assets", "models", "ufo"))
    )
    copy = tmp_path / "loop_sm_copy"
    shutil.copytree(shipped / "loop_sm", copy)
    card = _card(
        tmp_path / "gen",
        "u u~ > g g",
        source=str(copy),
        fixed=_nf5(),
        veto=("t",),
        action="generate",
    )
    config = resolve_config(card)
    output = Path(config.effective.generation.output)
    Generator(config).generate(
        ProcessSet((ProcessRequest.parse("g g > a a", name="api_p"),)), output
    )
    # effective.toml keeps the UFO directory's name only.
    with pytest.raises(ArtifactError, match="cannot be resolved"):
        check(output, points=1)
    code, payload = _cli(str(output), "--points", "1", "--model", str(copy))
    assert code == 0 and payload["source_kind"] == "artifact"
    (entry,) = payload["processes"]
    assert entry["process"] == "api_p" and entry["contribution"] == "loop-induced"
    assert entry["status"] == checks.PASSED
    # Another model (the copy without an R2 vertex): not the stored one.
    dropped = tmp_path / "loop_sm_r2_dropped"
    shutil.copytree(shipped / "loop_sm", dropped)
    assert drop_r2_vertex(dropped, [21, 21, 22, 22])
    with pytest.raises(ArtifactError, match="not the artifact's stored model"):
        check(output, points=1, model=str(dropped))


@pytest.mark.loop_physics
def test_nothing_to_check_exits_with_its_own_code(tmp_path):
    """h > g a: every loop has one coloured attachment (Tr T^a = 0), an exact
    colour zero; nothing is checked, which is no pass (exit 3)."""

    from pyamplicol.loops.v2 import checks

    card = _card(tmp_path, "h > g a", fixed={"WT": "0", "WH": "0"})
    code, payload = _cli(str(card))
    assert code == checks.EXIT_NOTHING_CHECKED == payload["exit_code"]
    (entry,) = payload["processes"]
    assert entry["status"] == "nothing checked"
    assert {row["verdict"] for row in entry["rows"]} == {"not applicable"}
