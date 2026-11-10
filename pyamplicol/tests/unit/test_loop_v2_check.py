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
* **The arbitrary-precision rung** (DECISIONS 2026-11-10 00:08; scripted
  carrier results): a row double-double fails while within ten times its own
  estimate (asked for only then) is re-run alone in arbitrary precision at
  the derived digits and decided there, by the double-double criterion; one
  still undecided there reads "undecided" and fails its process (exit 1); a
  double-double residual far above its estimate fails with no
  arbitrary-precision run; ``check.arbitrary_digits`` sets the digits; an
  explicit carrier runs no rung (its undecided rows read "undecided"); the
  loop-induced probes' criterion is their own estimate, so the predicate
  never holds for them. The real decision is a nightly id
  (``test_loop_v2_check_ap``).
* **Ward rows not applicable** (USER 2026-11-10 ~00:17; records on a toy
  model): a request whose corrections include one other than the model's
  strong correction AND with an on-shell zero-width external massive quark
  (an antiquark alone included) has one "not applicable" Ward row with the
  reason; its Ward evaluation is never called and its passing poles pass
  the process. QCD-only corrections with external tops, EW corrections with
  massless quarks only, a massive colour-singlet fermion or a quark with a
  non-zero width are checked as before. Ward rows are not split by
  squared-order sector: there is no per-sector Ward row to keep checked.

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
    # A stand-in target without request data: its Ward rows apply.
    monkeypatch.setattr(checks, "ward_applicability", lambda _t: None)
    target = NS(unpredicted_sectors=lambda: {})
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers)
    decided = [row for row in rows if row.check in ("poles", "ward")]
    assert {row.check for row in decided} == {"poles", "ward"}
    for row in decided:
        assert row.verdict == checks.NOT_RUN and row.carrier == "double"
        assert "double-double refused" in row.reason, row


def _toy_target(legs, corrections, fixed):
    """A probe-slice stand-in on a toy model (records): a gluon, a photon,
    a massive top (MT, width WT), a massless up quark, a massive muon (MM,
    width WM) whose orders are QCD (the gluon's vertices) and QED."""

    from types import SimpleNamespace as NS

    def particle(name, anti, pdg, spin, colour, charge=0.0, mass="ZERO", width="ZERO"):
        return NS(
            name=name,
            antiname=anti,
            pdg_code=pdg,
            spin=spin,
            color=colour,
            charge=charge,
            mass=mass,
            width=width,
            ghost_number=0,
            goldstoneboson=False,
            auxiliary_kind=None,
        )

    particles = [
        particle("g", "g", 21, 3, 8),
        particle("a", "a", 22, 3, 1),
        particle("t", "t~", 6, 2, 3, 2 / 3, "MT", "WT"),
        particle("t~", "t", -6, 2, -3, -2 / 3, "MT", "WT"),
        particle("u", "u~", 2, 2, 3, 2 / 3),
        particle("u~", "u", -2, 2, -3, -2 / 3),
        particle("mu-", "mu+", 13, 2, 1, -1.0, "MM", "WM"),
        particle("mu+", "mu-", -13, 2, 1, 1.0, "MM", "WM"),
    ]

    def vertex(names, order):
        return NS(particles=names, coupling_orders=((order, 1),))

    vertices = [
        vertex(("g", "g", "g"), "QCD"),
        vertex(("t~", "t", "g"), "QCD"),
        vertex(("u~", "u", "g"), "QCD"),
        vertex(("t~", "t", "a"), "QED"),
        vertex(("u~", "u", "a"), "QED"),
        vertex(("mu+", "mu-", "a"), "QED"),
    ]
    compiled = NS(ir=NS(particles=particles, parameters=[], vertex_terms=vertices))
    return NS(
        compiled=compiled,
        request=NS(corrections=corrections, fixed_parameters=fixed),
        blocks=NS(
            process=NS(legs=[NS(label=i + 1, pdg=pdg) for i, pdg in enumerate(legs)])
        ),
        unpredicted_sectors=lambda: {},
    )


@pytest.mark.loop_fast
def test_ew_corrections_with_a_zero_width_massive_quark_make_ward_not_applicable():
    """The class of the USER ruling 2026-11-10 ~00:17 from the request's own
    data: corrections other than the strong one (identified by the gluon's
    vertices) and an external massive quark with zero width (an antiquark's
    record carries colour -3). No sector split exists: the class takes the
    process's whole (leg, factor) Ward rows."""

    from pyamplicol.loops.v2 import checks

    zero = {"WT": "0", "WM": "0"}
    both = ("QCD", "QED")
    reason = checks.ward_applicability(_toy_target([21, 21, 6, -6], both, zero))
    assert reason and "QED" in reason and "(legs 3, 4)" in reason
    # An antiquark as the only external massive quark.
    reason = checks.ward_applicability(_toy_target([21, 2, 2, -6], both, zero))
    assert reason and "(leg 4)" in reason
    # QED corrections alone (no strong one) with an external top.
    assert checks.ward_applicability(_toy_target([22, 6, -6], ("QED",), zero))
    # Outside the class: QCD-only corrections with external tops; EW
    # corrections with massless quarks only, or a massive colour-singlet
    # fermion; a massive quark whose width is not zero.
    for legs, corrections, fixed in (
        ([21, 21, 6, -6], ("QCD",), zero),
        ([2, -2, 21, 22], both, zero),
        ([22, 22, 13, -13], both, zero),
        ([21, 21, 6, -6], both, {"WM": "0"}),
    ):
        target = _toy_target(legs, corrections, fixed)
        assert checks.ward_applicability(target) is None, (legs, corrections)


@pytest.mark.loop_fast
def test_a_process_in_the_class_reports_one_not_applicable_ward_row(monkeypatch):
    """The Born-interfered rows of an in-class request: its passing poles,
    one "not applicable" Ward row naming the legs and the reason, the Ward
    evaluation never called; the process passes (the row is neutral). The
    loop-induced rows probe finiteness only, with the same Ward row."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks, induced, plan_export, stability, ward

    sector = (("QCD", 4),)
    estimate = NS(
        relative=1e-12,
        absolute={sector: NS(coefficients=[0.0, 1e-12, 1e-12])},
        scale={sector: NS(coefficients=[1.0, 1.0, 1.0])},
    )

    def evaluate(_executor, _point, _scale, carrier, checks_):
        assert (
            carrier.arithmetic == "double" and checks_ is stability.RECOMMENDED_CHECKS
        )
        return NS(
            stability=estimate,
            components={"virtual": {sector: NS(coefficients=[0.0, 2.0, -3.0])}},
            ir_poles={sector: (2.0, -3.0)},
        )

    def never(*_args, **_options):
        raise AssertionError("the Ward evaluation runs for a not-applicable class")

    monkeypatch.setattr(plan_export, "NativeExecutor", lambda *a, **k: None)
    monkeypatch.setattr(checks, "_evaluate", evaluate)
    monkeypatch.setattr(ward, "ward_check", never)
    monkeypatch.setattr(ward, "gauge_legs", lambda _target: (1, 2))
    monkeypatch.setattr(induced, "goldstone_applicability", lambda _t: "n/a")
    target = _toy_target([21, 21, 6, -6], ("QCD", "QED"), {"WT": "0"})
    carriers = (NS(arithmetic="double"), NS(arithmetic="double-double"))
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers)
    (row,) = [row for row in rows if row.check == "ward"]
    assert row.verdict == checks.NOT_APPLICABLE and row.point is None
    assert row.reason.startswith("legs 1, 2 not checked: EW corrections (QED)")
    assert "(legs 3, 4)" in row.reason
    poles = [row for row in rows if row.check == "poles"]
    assert len(poles) == 2 and {row.verdict for row in poles} == {checks.PASS}
    process = checks.ProcessCheck(
        "p", "g g > t t~", "virtual", "zero", (), "ninjars", tuple(rows), 0.0
    )
    assert process.status == checks.PASSED
    # Loop-induced: the Ward probes are not evaluated (no legs probed).
    probed = []

    def point_probes(_executor, _point, _scale, legs, _carriers, _floors):
        probed.append(list(legs))
        return [{"probe": "finiteness", "carrier": "double"}], [
            induced.ProbeResult("finiteness", {"k": 0.1})
        ]

    monkeypatch.setattr(induced, "probe_legs", lambda _target: [1, 2])
    monkeypatch.setattr(induced, "probe_floors", lambda *a: ({}, {}))
    monkeypatch.setattr(induced, "point_probes", point_probes)
    target.amplitude = NS(colour_zero=False)
    rows = checks.induced_rows(target, None, None, None, [(None, 1)], carriers)
    assert probed == [[]]
    assert [row.check for row in rows] == ["poles", "ward", "goldstone"]
    assert rows[1].verdict == checks.NOT_APPLICABLE
    assert rows[1].reason.startswith("legs 1, 2 not checked: EW corrections")


def _scripted_poles(monkeypatch, script):
    """``checks._evaluate`` replaced by ``script(carrier, checked)`` -> the
    1/eps coefficient and its estimate E (None: no checks) of one sector
    whose Catani 1/eps pole is 2 (1/eps^2: -3, always exact); the calls are
    recorded as (carrier name, digits, checked)."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks, induced, plan_export, ward

    sector = (("QED", 4),)
    calls = []

    def laurent(*values):
        return NS(coefficients=list(values))

    def evaluate(_executor, _point, _scale, carrier, checks_):
        calls.append((carrier.arithmetic, carrier.digits, checks_ is not None))
        single, error = script(carrier, checks_ is not None)
        stability = None
        if error is not None:
            stability = NS(
                relative=error,
                absolute={sector: laurent(0.0, error, 1e-40)},
                scale={sector: laurent(1.0, 1.0, 1.0)},
            )
        return NS(
            stability=stability,
            components={"virtual": {sector: laurent(0.0, single, -3.0)}},
            ir_poles={sector: (2.0, -3.0)},
        )

    monkeypatch.setattr(plan_export, "NativeExecutor", lambda *a, **k: None)
    monkeypatch.setattr(checks, "_evaluate", evaluate)
    monkeypatch.setattr(ward, "gauge_legs", lambda _target: ())
    monkeypatch.setattr(induced, "goldstone_applicability", lambda _t: "n/a")
    return calls


def _rung_rows(rows):
    return {row.subject: row for row in rows if row.check == "poles"}


@pytest.mark.loop_fast
def test_a_row_double_double_cannot_decide_is_decided_in_arbitrary_precision(
    monkeypatch,
):
    """Point 0: binary64 out of scope; double-double 1/eps off by 1e-11
    relative (criterion 1e-12) with its own estimate E = 1e-11 (asked for
    only now): undecided, re-run alone in arbitrary precision at d =
    ceil(32.0 + log10(10 E / (1e-12 x 2))) = 34 digits (the point lifted
    for it), where it passes by the double-double criterion. Point 1 passes
    in binary64: no other carrier runs. The 1/eps^2 row keeps its
    double-double pass."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks
    from pyamplicol.loops.v2.assemble import Precision

    def script(carrier, checked):
        if carrier.arithmetic == "double":
            return 2.0, (1e-3 if script.point == 0 else 1e-14)
        if carrier.arithmetic == "double-double":
            return 2.0 * (1 + 1e-11), (1e-11 if checked else None)
        return 2.0 * (1 + 1e-30), 1e-31

    calls = _scripted_poles(monkeypatch, script)
    lifted = []

    def lift(index, carrier):
        lifted.append((index, carrier.digits))
        return (f"point {index} at {carrier.working_digits}", 1)

    rung = checks.Rung(lift)
    carriers = (Precision.double(), Precision.double_double())
    target = NS(unpredicted_sectors=lambda: {})
    rows = []
    for point in (0, 1):
        script.point = point
        rows.append(checks.born_interfered_rows(target, [(None, 1)], carriers, rung))
    first, second = _rung_rows(rows[0]), _rung_rows(rows[1])
    single = first["QED^4 1/eps"]
    assert (single.verdict, single.carrier, single.digits) == (
        checks.PASS,
        "arbitrary",
        34,
    )
    assert single.fallback.startswith("double-double undecided: residual 2e-11")
    assert "34 digits derived" in single.fallback
    assert first["QED^4 1/eps^2"].carrier == "double-double"
    assert first["QED^4 1/eps^2"].verdict == checks.PASS
    assert lifted == [(0, 34)]
    assert calls[:4] == [
        ("double", None, True),
        ("double-double", None, False),
        ("double-double", None, True),
        ("arbitrary", 34, True),
    ]
    assert calls[4:] == [("double", None, True)]
    assert {row.carrier for row in second.values()} == {"double"}
    # The rule itself: the derivation and the predicate's two failures.
    part = checks.Part(2e-11, 2e-12, 1e-11)
    assert checks.undecided([part])
    assert checks.rung_digits([part], Precision.double_double().unit_roundoff) == 34
    assert not checks.undecided([checks.Part(2e-11, 2e-12, 1e-13)])
    assert not checks.undecided([checks.Part(2e-11, 2e-12, None)])
    assert not checks.undecided([checks.Part(1e-12, 2e-12, 1e-13)])


@pytest.mark.loop_fast
def test_still_undecided_far_above_the_estimate_option_and_explicit_carrier(
    monkeypatch,
):
    """A row still undecided at the derived digits reads "undecided" with
    both carriers' numbers and fails its process (exit 1); a double-double
    residual far above its own estimate fails with no arbitrary-precision
    run; ``check.arbitrary_digits`` sets the digits; an explicit carrier runs
    no rung and its undecided row reads "undecided"."""

    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import checks
    from pyamplicol.loops.v2.assemble import Precision

    target = NS(unpredicted_sectors=lambda: {})
    carriers = (Precision.double(), Precision.double_double())
    rung = checks.Rung(lambda index, carrier: (None, 1))

    def still(carrier, checked):
        if carrier.arithmetic == "double":
            return 2.0, 1e-3
        if carrier.arithmetic == "double-double":
            return 2.0 * (1 + 1e-11), (1e-11 if checked else None)
        return 2.0 * (1 + 1e-11), 1e-11

    _scripted_poles(monkeypatch, still)
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers, rung)
    row = _rung_rows(rows)["QED^4 1/eps"]
    assert row.verdict == checks.UNDECIDED and row.carrier == "arbitrary"
    assert row.fallback.startswith("double-double undecided")
    assert row.reason.startswith("double-double undecided: residual 2e-11")
    assert "; still arbitrary (34 digits) undecided" in row.reason
    process = checks.ProcessCheck(
        "p", "u u~ > a a", "virtual", "zero", (), "ninjars", tuple(rows), 0.0
    )
    assert process.status == checks.FAILED
    report = checks.CheckReport(
        "x", "card", 1, 0, "auto", ("double",), (process,), "", 0.0
    )
    assert report.exit_code == checks.EXIT_FAILED

    def far(carrier, checked):
        if carrier.arithmetic == "double":
            return 2.0, 1e-3
        return 2.0 * (1 + 1e-11), (1e-20 if checked else None)

    calls = _scripted_poles(monkeypatch, far)
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers, rung)
    row = _rung_rows(rows)["QED^4 1/eps"]
    assert row.verdict == checks.FAIL and row.carrier == "double-double"
    assert row.estimate is not None and row.digits is None
    assert all(carrier != "arbitrary" for carrier, _digits, _checked in calls)

    calls = _scripted_poles(monkeypatch, still)
    option = checks.Rung(lambda index, carrier: (None, 1), digits=60)
    rows = checks.born_interfered_rows(target, [(None, 1)], carriers, option)
    row = _rung_rows(rows)["QED^4 1/eps"]
    assert row.digits == 60 and "option: 60 digits" in row.fallback
    assert ("arbitrary", 60, True) in calls

    calls = _scripted_poles(monkeypatch, still)
    explicit = (Precision.double_double(),)
    rows = checks.born_interfered_rows(target, [(None, 1)], explicit)
    row = _rung_rows(rows)["QED^4 1/eps"]
    assert row.verdict == checks.UNDECIDED and row.carrier == "double-double"
    assert "explicit carrier runs no arbitrary-precision rung" in row.reason
    assert all(carrier != "arbitrary" for carrier, _digits, _checked in calls)
    # The loop-induced probes: the criterion is the probe's own estimate, a
    # failing probe is decided.
    from pyamplicol.loops.v2.induced import ProbeResult

    failed = checks._probe_row(
        {"probe": "ward", "leg": 1, "carrier": "double-double"},
        ProbeResult("ward", {"k": 11.0}),
        0,
    )
    assert failed.verdict == checks.FAIL


@pytest.mark.loop_fast
def test_ward_rows_take_the_rung_by_the_double_double_criterion(monkeypatch):
    """A Ward row double-double fails (virtual 4e-22 of the physical scale,
    bound 1e-25) with its own reduction-direction estimate 1.5e-21 (the u u~
    > d d~ a a point of lit-ew-blocks §3 (d) in miniature) is re-run in
    arbitrary precision at ceil(32.0 + log10(10 x 1.5e-21 / 1e-25)) = 38
    digits and passes the double-double criterion there; the double-double
    pass criterion is unchanged (judged without the estimate)."""

    from dataclasses import dataclass, field
    from types import SimpleNamespace as NS

    from pyamplicol.loops.v2 import assemble, checks, plan_export, ward
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import REDUCTION

    sector = (("QED", 10),)

    @dataclass(frozen=True)
    class Result:
        components: dict = field(default_factory=dict)
        born: dict = field(default_factory=dict)
        loop_born: dict | None = None
        stability: object = None

    def laurent(*values):
        return NS(coefficients=list(values))

    def result(virtual, error, loop_born=None):
        components = {
            name: {sector: laurent(1.0, 1.0, 1.0)} for name in assemble.COMPONENTS
        }
        components["virtual"] = {sector: laurent(virtual, 0.0, 0.0)}
        estimate = None
        if error is not None:
            estimate = NS(
                absolute={sector: laurent(error, 0.0, 0.0)},
                budgets={REDUCTION: {}},
            )
        return Result(components, {sector: 1.0}, loop_born, estimate)

    calls = []

    def ward_check(_executor, point, _scale, name, *, legs, factors, **options):
        precision = options.get("precision")
        checked = bool(options.get("checks"))
        arithmetic = name if precision is None else precision.arithmetic
        calls.append((arithmetic, getattr(precision, "digits", None), checked))
        virtual = {"double": 1e-6, "double-double": 4e-22, "arbitrary": 1e-30}
        error = {"double": 1e-5, "double-double": 1.5e-21, "arbitrary": 1e-31}
        physical = result(1.0, None)
        checked = checked or arithmetic == "double"
        results = [
            result(
                virtual[arithmetic],
                error[arithmetic] if checked else None,
                {sector: 0.0},
            )
            for _leg in legs
            for _factor in factors
        ]
        rows = [
            {**ward.ward_row(physical, r, name), "leg": leg, "factor": factor}
            for r, (leg, factor) in zip(
                results, [(leg, f) for leg in legs for f in factors], strict=True
            )
        ]
        return {"physical": physical, "results": results, "rows": rows}

    monkeypatch.setattr(ward, "ward_check", ward_check)
    lifted = []
    rung = checks.Rung(
        lambda index, carrier: lifted.append(carrier.digits) or (None, 1)
    )
    carriers = (Precision.double(), Precision.double_double())
    rows = checks._born_ward_rows(None, None, 1, [5], carriers, 2, rung)
    assert len(rows) == 2
    for row in rows:
        assert (row.verdict, row.carrier, row.digits) == (checks.PASS, "arbitrary", 38)
        assert row.fallback.startswith("double-double undecided: residual 4e-22")
        assert row.estimate == pytest.approx(1e-31)
    assert lifted == [38]
    # (carrier, digits, checks asked for): binary64 runs WARD_CHECKS by
    # ward_check's default, double-double on the ladder none.
    assert calls == [
        ("double", None, False),
        ("double-double", None, False),
        ("double-double", None, True),
        ("arbitrary", 38, True),
    ]
    assert plan_export  # the executor is not touched: ward_check is scripted


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
