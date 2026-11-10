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
* **Complex-mass scheme, untagged photons** (u u~ > a a with QED
  corrections, WT != 0): the QED pole prediction's gamma_gamma counts the
  massless charged fermions; the top's complex pole mass is a massive
  internal species, not the cms-external-width refusal of an unstable leg.
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
    # A sector the operators predict but without a pole row at this point:
    # an applicable check that did not run, "not run" (exit 3), not a neutral
    # "no prediction" (DECISIONS 2026-11-09 14:25, lit-examples re-audit
    # issue 8); a sector they do not predict keeps "no prediction".
    slice_ = NS(unpredicted_sectors=lambda: {})
    (row,) = checks._unpredicted_rows(slice_, [first, second], True, 0, {first})
    assert row.verdict == checks.NOT_RUN and row.subject == "QCD^2 QED^2"
    assert row.reason.startswith("an applicable check did not run")
    slice_ = NS(unpredicted_sectors=lambda: {second: "uncorrected exchange"})
    (row,) = checks._unpredicted_rows(slice_, [first, second], True, 0, {first})
    assert row.verdict == checks.NO_PREDICTION
    assert row.reason.startswith("uncorrected exchange")
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


@pytest.mark.loop_fast
def test_a_refused_double_double_run_is_named_on_the_binary64_rows(monkeypatch):
    """When binary64 evaluated a point and double-double is refused, the rows
    binary64 left undecided keep its verdict (never a pass) and say that
    double-double was refused, pole and Ward rows alike (lit-examples
    re-audit issue 3)."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks, induced, plan_export, ward

    sector = (("QCD", 4),)
    carriers = (NS(arithmetic="double"), NS(arithmetic="double-double"))

    def laurent(*values):
        return NS(coefficients=list(values))

    out_of_scope = NS(
        relative=1e-3,
        absolute={sector: laurent(0.0, 1e-12, 1e-12)},
        scale={sector: laurent(1.0, 1.0, 1.0)},
    )

    def evaluate(_executor, _point, _scale, carrier, _checks):
        if carrier.arithmetic == "double-double":
            raise plan_export.FailedPoints([None], {0: ["ill-conditioned"]})
        return NS(
            stability=out_of_scope,
            components={"virtual": {sector: laurent(0.0, 2.0, -3.0)}},
            ir_poles={sector: (2.0, -3.0)},
        )

    def ward_check(_executor, _point, _scale, name, *, legs, factors):
        if name == "double-double":
            raise plan_export.FailedPoints([None], {0: ["ill-conditioned"]})
        row = {
            "born": 0.0,
            "born_passed": True,
            "parts": {"virtual": [1e-3]},
            "estimate": [1e-6],
            "estimate_over": [1.0],
            "o1_over": [0.5],
            "in_scope": False,
            "passed": False,
        }
        return {
            "rows": [
                {**row, "leg": leg, "factor": factor}
                for leg in legs
                for factor in factors
            ]
        }

    monkeypatch.setattr(plan_export, "NativeExecutor", lambda *a, **k: None)
    monkeypatch.setattr(checks, "_evaluate", evaluate)
    monkeypatch.setattr(ward, "gauge_legs", lambda _target: [3])
    monkeypatch.setattr(ward, "ward_check", ward_check)
    monkeypatch.setattr(induced, "goldstone_applicability", lambda _t: "n/a")
    target = NS(unpredicted_sectors=lambda: {})
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers)
    decided = [row for row in rows if row.check in ("poles", "ward")]
    assert {row.check for row in decided} == {"poles", "ward"}
    for row in decided:
        assert row.verdict == checks.NOT_RUN and row.carrier == "double"
        assert "double-double refused" in row.reason, row


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


@pytest.mark.loop_physics
def test_an_untagged_photon_under_the_complex_mass_scheme_with_a_top_width(tmp_path):
    """u u~ > a a with QED corrections, complex-mass scheme, WT != 0, the
    photons untagged: the QED pole prediction's gamma_gamma counts the
    massless charged fermions only; the top's complex pole mass makes it a
    massive internal species (it raised the untyped cms-external-width
    ValueError before, DECISIONS 2026-11-09 22:40 side item (b)). Both poles
    pass at both points, so the top is not counted."""

    from pyamplicol.api.services import check
    from pyamplicol.loops.v2 import checks

    tmp_path.mkdir(parents=True, exist_ok=True)
    card = tmp_path / "card.toml"
    card.write_text(
        """action = "check"
[model]
source = "loop_qcd_qed_sm_Gmu"
restriction = "default"
[process]
entries = [{expression = "u u~ > a a", name = "p"}]
loop_coupling_orders = ["QED"]
coupling_order_policy = "explicit"
max_coupling_orders = { QCD = 0, QED = 2 }
[loop]
external_photons = "none"
[loop.preparation]
fixed_parameters = { MB = "0", MTA = "0", WT = "1.37758", WZ = "2.49877", \
WW = "2.09291", WH = "0.00407468" }
[loop.model.scheme]
mass_scheme = "complex-mass"
[check]
points = 2
"""
    )
    report = check(card)
    assert report.exit_code == checks.EXIT_PASSED
    poles = _rows(report, "poles")
    assert len(poles) == 2 * 2 and all(row.verdict == checks.PASS for row in poles)
