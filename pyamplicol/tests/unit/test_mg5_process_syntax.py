# SPDX-License-Identifier: 0BSD
"""MG5_aMC process syntax translated to run-card settings (ux-process):
``processes.mg5_syntax`` for the TOML card, the CLI (``--process`` and
``pyamplicol CARD ACTION --output``) and the Python API alike; conflicts
with explicit settings are typed; unserved syntax keeps its refusal; the
flat phase-space points of ``evaluate`` / ``profile`` without a momenta
file."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pyamplicol.config import Action, RunConfig, resolve_config
from pyamplicol.processes.mg5_syntax import (
    ProcessSyntaxConflict,
    parse_squared_order_constraint,
    squared_orders_admit,
    translate_api_request,
    translate_process_expression,
)
from pyamplicol.processes.refusals import ProcessRequestUnsupported

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("expression", "plain", "settings"),
    [
        (
            "u u~ > d d~ QCD^2<=4 QED^2<=6 [QCD QED]",
            "u u~ > d d~",
            {
                "process.loop_coupling_orders": ("QCD", "QED"),
                "process.squared_coupling_orders": ("QCD^2<=4", "QED^2<=6"),
                "process.coupling_order_policy": "explicit",
            },
        ),
        (
            "g g > h [noborn=QCD]",
            "g g > h",
            {
                "process.loop_coupling_orders": ("QCD",),
                "process.loop_induced": "only",
            },
        ),
        (
            "g g > z z [ sqrvirt = QCD ] ",
            "g g > z z",
            {
                "process.loop_coupling_orders": ("QCD",),
                "process.loop_induced": "only",
            },
        ),
        (
            "u u~ > t t~ g [virt=QCD]",
            "u u~ > t t~ g",
            {"process.loop_coupling_orders": ("QCD",)},
        ),
        (
            "u u~ > d d~ / z h QED<=2 [QCD]",
            "u u~ > d d~",
            {
                "process.loop_coupling_orders": ("QCD",),
                "process.max_coupling_orders": {"QED": 2},
                "process.coupling_order_policy": "explicit",
                "generation.particle_veto": ("z", "h"),
            },
        ),
        (
            "u u~>d d~ QED=2 QCD^2 > 2 [all=QED]",
            "u u~ > d d~",
            {
                "process.loop_coupling_orders": ("QED",),
                "process.max_coupling_orders": {"QED": 2},
                "process.squared_coupling_orders": ("QCD^2>2",),
                "process.coupling_order_policy": "explicit",
            },
        ),
    ],
)
def test_served_mg5_syntax_translates_to_settings(expression, plain, settings):
    translation = translate_process_expression(expression)
    assert translation.expression == plain
    assert dict(translation.settings) == settings
    assert translation.translated
    assert plain in translation.describe()


@pytest.mark.parametrize(
    "expression",
    ["u u~ > d d~", "u u~ > d [d g]", "t > b w+", "p p > z j j", "u u~>d d~"],
)
def test_plain_expressions_and_anonymous_multiparticles_are_kept(expression):
    translation = translate_process_expression(expression)
    assert not translation.translated
    assert translation.expression == " ".join(expression.split())


@pytest.mark.parametrize(
    ("expression", "found"),
    [
        ("p p > t t~ [real=QCD]", "only the one-loop virtual is served"),
        ("p p > t t~ [LOonly=QCD]", "only the one-loop virtual is served"),
        ("p p > t t~ [foo=QCD]", "unknown NLO bracket keyword"),
        ("t > b w+, w+ > e+ ve", "a decay chain"),
        ("p p > z $ h", "s-channel exclusion"),
        ("p p > z $$ h", "s-channel exclusion"),
        ("p p > t t~ @1", "process tag"),
        ("u u~ > d d~ QED==2", "an amplitude-level constraint is an upper limit"),
        ("u u~ > d d~ [virt=QCD] g", "the NLO bracket comes last"),
        ("u u~ > d d~ /", "without a particle"),
    ],
)
def test_unserved_mg5_syntax_is_the_listed_refusal(expression, found):
    with pytest.raises(ProcessRequestUnsupported) as refused:
        translate_process_expression(expression)
    assert refused.value.key == "madloop-syntax"
    assert found in str(refused.value)


def test_squared_order_constraints():
    constraint = parse_squared_order_constraint("QCD^2 <= 4")
    assert constraint.text == "QCD^2<=4"
    assert parse_squared_order_constraint("QED=6").text == "QED^2<=6"
    assert [constraint.admits(p) for p in (2, 4, 6)] == [True, True, False]
    parsed = [
        parse_squared_order_constraint(t) for t in ("QCD^2<=4", "QED^2>2", "QED^2<6")
    ]
    assert squared_orders_admit(parsed, {"QCD": 2, "QED": 4})
    assert not squared_orders_admit(parsed, (("QCD", 6),))
    assert not squared_orders_admit(parsed, {"QCD": 4, "QED": 2})
    with pytest.raises(ValueError, match="invalid squared-order constraint"):
        parse_squared_order_constraint("QCD^2~4")


def test_the_card_translates_and_requested_keeps_the_expression():
    resolution = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > d d~ QCD^2<=4 QED^2<=6 [QCD QED]"}]
            },
        }
    )
    effective = resolution.effective
    assert effective.process.entries[0].expression == "u u~ > d d~"
    assert effective.process.loop_coupling_orders == ("QCD", "QED")
    assert effective.process.squared_coupling_orders == ("QCD^2<=4", "QED^2<=6")
    assert str(effective.process.coupling_order_policy) == "explicit"
    assert str(effective.color.accuracy) == "full"
    assert resolution.requested.process.entries[0].expression.endswith("[QCD QED]")
    # Idempotent: the requested configuration resolves to the same.
    again = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > d d~ QCD^2<=4 QED^2<=6 [QCD QED]"}],
                "loop_coupling_orders": ["QED", "QCD"],
                "coupling_order_policy": "explicit",
            },
        }
    )
    assert again.effective == effective


@pytest.mark.parametrize(
    ("process", "match"),
    [
        (
            {"loop_coupling_orders": ["QED"]},
            r"sets process.loop_coupling_orders = \[\"QCD\"\] but the "
            r"configuration sets process.loop_coupling_orders = \[\"QED\"\]",
        ),
        (
            {"max_coupling_orders": {"QED": 4}},
            "limits QED to 2 .* but the configuration sets QED = 4",
        ),
        (
            {"coupling_order_policy": "minimal"},
            "process.coupling_order_policy",
        ),
    ],
)
def test_a_conflicting_setting_is_a_typed_error_naming_both(process, match):
    with pytest.raises(ProcessSyntaxConflict, match=match):
        resolve_config(
            {
                "action": "generate",
                "process": {
                    "entries": [{"expression": "u u~ > d d~ QED=2 [QCD]"}],
                    **process,
                },
            }
        )


def test_entries_must_agree_and_settings_merge_per_coupling():
    with pytest.raises(ProcessSyntaxConflict, match="different run-wide settings"):
        resolve_config(
            {
                "action": "generate",
                "process": {
                    "entries": [
                        {"expression": "u u~ > d d~ [QCD]"},
                        {"expression": "u u~ > s s~"},
                    ]
                },
            }
        )
    merged = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > d d~ QED=2 [QCD]"}],
                "max_coupling_orders": {"QCD": 2},
            },
        }
    ).effective.process
    assert dict(merged.max_coupling_orders) == {"QCD": 2, "QED": 2}


def test_the_squared_setting_is_validated_and_one_loop_only():
    process = resolve_config(
        {
            "action": "generate",
            "process": {
                "entries": [{"expression": "u u~ > d d~"}],
                "loop_coupling_orders": ["QCD"],
                "squared_coupling_orders": ["QED<=4", "QCD^2 == 6"],
            },
        }
    ).effective.process
    assert process.squared_coupling_orders == ("QCD^2==6", "QED^2<=4")
    with pytest.raises(ValueError, match=r"requires process\.loop_coupling_orders"):
        resolve_config(
            {
                "action": "generate",
                "process": {
                    "entries": [{"expression": "u u~ > d d~"}],
                    "squared_coupling_orders": ["QCD^2<=4"],
                },
            }
        )


def test_the_python_api_translates_strings_into_its_configuration():
    processes, config = translate_api_request(
        "g g > h [noborn=QCD]", RunConfig(action=Action.GENERATE)
    )
    assert processes == "g g > h"
    assert config.process.loop_coupling_orders == ("QCD",)
    assert str(config.process.loop_induced) == "only"
    assert str(config.color.accuracy) == "full"
    tree = resolve_config({"action": "generate", "model": {"source": "loop_sm"}})
    processes, resolution = translate_api_request(["u u~ > d d~ / z [QCD]"], tree)
    card = resolve_config(
        {
            "action": "generate",
            "model": {"source": "loop_sm"},
            "process": {"entries": [{"expression": "u u~ > d d~ / z [QCD]"}]},
        }
    )
    assert processes == ["u u~ > d d~"]
    # The same effective settings as the card's (its entries aside).
    for name in ("loop", "color", "generation", "evaluator", "evaluation"):
        assert getattr(resolution.effective, name) == getattr(card.effective, name)
    process = resolution.effective.process
    assert process.loop_coupling_orders == ("QCD",)
    plain, unchanged = translate_api_request("u u~ > d d~", tree)
    assert plain == "u u~ > d d~" and unchanged is tree


def test_a_process_request_refuses_with_the_translation():
    from pyamplicol.api import ProcessRequest

    with pytest.raises(ProcessRequestUnsupported) as refused:
        ProcessRequest.parse("g g > h [noborn=QCD]")
    message = str(refused.value)
    assert "a ProcessRequest takes a plain expression" in message
    assert 'process.loop_induced = "only"' in message
    assert "pyamplicol.generate" in message


# ------------------------------------------------------------- the CLI


def test_card_action_form_selects_the_action_and_the_artifact(tmp_path, monkeypatch):
    from pyamplicol.cli.parser import parse_cli

    card = tmp_path / "loop.toml"
    card.write_text((ROOT / "examples" / "loop_generate.toml").read_text())
    monkeypatch.chdir(tmp_path)
    invocation = parse_cli(
        [
            "loop.toml",
            "generate",
            "--process=g g > h [noborn=QCD]",
            "--output=./out",
        ]
    )
    effective = invocation.resolve().effective
    assert str(effective.action) == "generate"
    assert effective.generation.output == tmp_path / "out"
    assert effective.process.entries[0].expression == "g g > h"
    assert str(effective.process.loop_induced) == "only"
    for action, expected in (
        ("check", "check"),
        ("evaluate", "evaluate"),
        ("profile", "benchmark"),
    ):
        effective = (
            parse_cli(["loop.toml", action, "--output", "out"]).resolve().effective
        )
        assert str(effective.action) == expected
        assert effective.evaluation.artifact == tmp_path / "out"
    # The card alone runs its own action; the older forms are unchanged.
    assert str(parse_cli(["loop.toml"]).resolve().effective.action) == "generate"
    assert str(parse_cli(["check", "loop.toml"]).resolve().effective.action) == (
        "check"
    )


def test_the_master_card_resolves_for_every_action():
    card = ROOT / "examples" / "loop_generate.toml"
    effective = resolve_config(card).effective
    assert effective.model.source.endswith("loop_qcd_qed_sm")
    assert effective.model.restriction == "default"
    assert effective.loop.external_widths == "zero"
    assert effective.process.squared_coupling_orders == ("QCD^2<=4", "QED^2<=6")
    assert effective.evaluation.momenta is None
    assert effective.evaluation.artifact == effective.generation.output
    for action in (Action.CHECK, Action.EVALUATE, Action.BENCHMARK):
        assert resolve_config(card, action=action).effective.action == action


# ------------------------------------------------------- flat points


def _fake_artifact(root: Path, expression: str, legs: list[list[str]]) -> Path:
    (root / "processes" / "p").mkdir(parents=True)
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "default_process_id": "p",
                "processes": [{"id": "p", "expression": expression, "aliases": []}],
            }
        )
    )
    (root / "processes" / "p" / "validation-momenta.json").write_text(
        json.dumps(
            {
                "available": True,
                "points": [[{"momentum": leg, "pdg": 0} for leg in legs]],
            }
        )
    )
    return root


def _mass2(momentum) -> float:
    e, x, y, z = momentum
    return e * e - x * x - y * y - z * z


def test_flat_points_scattering_with_masses(tmp_path):
    from pyamplicol.cli.flat_points import draw_flat_points

    top = 173.0
    e1 = math.sqrt(300.0**2 + 400.0**2 + 100.0**2 + top**2)
    e2 = math.sqrt(300.0**2 + 400.0**2 + top**2)
    beam = (e1 + e2 + 100.0) / 2
    artifact = _fake_artifact(
        tmp_path,
        "u u~ > t t~ g",
        [
            [repr(beam), "0", "0", repr(beam)],
            [repr(beam), "0", "0", repr(-beam)],
            [repr(e1), "300", "400", "100"],
            [repr(e2), "-300", "-400", "0"],
            ["100", "0", "0", "-100"],
        ],
    )
    flat = draw_flat_points(artifact, process=None, points=3, seed=7, sqrt_s=None)
    assert flat.sqrt_s == pytest.approx(2 * beam)
    assert flat.masses[:2] == (0.0, 0.0) and flat.masses[4] == 0.0
    assert flat.masses[2] == pytest.approx(top, rel=1e-12)
    assert len(flat.points) == 3 and len({p for p in flat.points}) == 3
    for point in flat.points:
        incoming = [sum(leg[i] for leg in point[:2]) for i in range(4)]
        outgoing = [sum(leg[i] for leg in point[2:]) for i in range(4)]
        assert incoming == pytest.approx(outgoing, abs=1e-9)
        assert incoming[0] == pytest.approx(2 * beam)
        for leg, mass in zip(point, flat.masses, strict=True):
            assert _mass2(leg) == pytest.approx(mass * mass, abs=1e-7)
    # Point i depends on seed + i only.
    again = draw_flat_points(artifact, process="p", points=1, seed=8, sqrt_s=None)
    assert again.points[0] == flat.points[1]
    other = draw_flat_points(artifact, process=None, points=1, seed=7, sqrt_s=2000)
    assert sum(leg[0] for leg in other.points[0][:2]) == pytest.approx(2000.0)
    assert flat.mu_r2()[0] == repr(flat.sqrt_s * flat.sqrt_s)
    with pytest.raises(ValueError, match="below the threshold"):
        draw_flat_points(artifact, process=None, points=1, seed=1, sqrt_s=300.0)


def test_flat_points_decay_and_two_to_one(tmp_path):
    from pyamplicol.cli.flat_points import draw_flat_points

    decay = _fake_artifact(
        tmp_path / "decay",
        "t > b w+",
        [["173", "0", "0", "0"], ["60", "0", "0", "60"], ["113", "0", "0", "-60"]],
    )
    flat = draw_flat_points(decay, process=None, points=2, seed=1, sqrt_s=None)
    assert flat.initial == 1 and flat.sqrt_s == pytest.approx(173.0)
    for point in flat.points:
        assert point[0] == (173.0, 0.0, 0.0, 0.0)
        assert sum(leg[0] for leg in point[1:]) == pytest.approx(173.0)
    single = _fake_artifact(
        tmp_path / "h",
        "g g > h",
        [
            ["62.5", "0", "0", "62.5"],
            ["62.5", "0", "0", "-62.5"],
            ["125", "0", "0", "0"],
        ],
    )
    flat = draw_flat_points(single, process=None, points=2, seed=1, sqrt_s=None)
    assert flat.points[0] == flat.points[1]
    assert flat.points[0][2] == (125.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="fixed by its masses"):
        draw_flat_points(single, process=None, points=1, seed=1, sqrt_s=500.0)


def test_a_card_with_its_own_actions_options(tmp_path, monkeypatch):
    """``pyamplicol X_profile.toml --stability`` (m-lit ux-polish item 6):
    options only the card's own action takes select that action."""

    from pyamplicol.cli.parser import parse_cli

    card = tmp_path / "loop_gg_zgg_profile.toml"
    card.write_text((ROOT / "examples" / "loop_gg_zgg_profile.toml").read_text())
    monkeypatch.chdir(tmp_path)
    for options in (["--stability"], ["--runtime", "--stability"]):
        invocation = parse_cli(["loop_gg_zgg_profile.toml", *options])
        assert str(invocation.resolve().effective.action) == "benchmark"
        # The same invocation as the documented explicit form.
        assert invocation == parse_cli(["profile", "--card", str(card), *options])
        assert invocation != parse_cli(["profile", "--card", str(card)])
    plain = parse_cli(["loop_gg_zgg_profile.toml", "--json"])
    assert str(plain.resolve().effective.action) == "benchmark"
