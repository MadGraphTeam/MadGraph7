# SPDX-License-Identifier: 0BSD
"""Multiparticle one-loop and loop-induced requests end to end (m-lit
lit-multiparticle; DECISIONS 2026-11-07 ~23:34, "MULTIPARTICLE").

A one-loop request expands like a tree request (``loop_service._prepare``
-> ``service._expand_process_set``: the same labels, filters and
side-permutation dedup, one artifact entry per concrete process);
``process.loop_induced`` routes the members without a Born
(``loop_service._born_members``); with ``process.flavor_blind`` a member is
served by an earlier process only by an exact proof (``loop_service.
_v2_member``). Generated through ``Generator`` and evaluated through
``Runtime`` on loop_sm (no_b_mass, QCD corrections, zero widths):

* ``p p > h h``, purely loop-induced: G G > H H is a real amplitude (MadLoop
  within the lax rule); u u~, d d~ and s s~ > H H are exact colour zeros
  (stored, served as exactly 0 with their colour record); c c~ and
  b b~ > H H route no loop family but keep the counterterm kets of their
  Yukawa R2/UV vertices, whose couplings the restriction makes vanish: a
  numerical zero, exactly 0 within its absolute estimate (the term scale of
  an amplitude without a family is empty, ``induced.term_values``); each
  member bitwise equal to the same process generated as its own request.
* ``p p > z z`` with ``process.loop_induced = 'include'``: the five
  q q~ > Z Z members Born-interfered (``rusticol.loop-runtime.v4``) and
  G G > Z Z loop-induced (v5) in one artifact, never summed, each bitwise
  equal to its own request's process.
* ``process.flavor_blind``: on the QCD request ``q qb > g g`` (q = u d s c
  b) s s~ > G G is served by d d~ > G G (exact species proof), bitwise its
  representative (its own generation to rounding); u u~, c c~ and b b~ > G G are
  generated on their own (charge metadata; the zero-coupling Yukawa
  counterterm rules of c and b change the retained plan). On ``p p > z z``
  with ``include`` nothing is folded (a down-type substitution changes the
  diagonal-CKM W couplings the electroweak orders retain), loop-induced
  member included, and neither on ``p p > h h`` (its five loop-induced
  q q~ > H H candidates), each bitwise equal to the plain generation.
* LC / NLC: a request with a member routed loop-induced is refused as a
  whole (``loop-induced-colour-accuracy``, ``generate`` writes nothing);
  the mixed ``auto`` request passes, G G > Z Z skipped with its typed
  ``loop-induced`` record.

Tiers: the term-scale unit, the counterterm-only process alone, the
``flavor_blind`` fold and the LC / NLC dry runs are in the default tier; the
end-to-end multiparticle generations (two runs of six processes each) are
``loop_nightly``. Measured times (one process each, 2026-11-08, on the
shared host at load ~430 on 384 cores) are in the test docstrings.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
# The loop-induced fixtures' no_b_mass scheme: massless b and tau, zero widths.
FIXED = {
    "MB": "0",
    "ymb": "0",
    "MTA": "0",
    "ymtau": "0",
    "WT": "0",
    "WW": "0",
    "WZ": "0",
    "WH": "0",
}
LAX = 1e-5
V4 = "rusticol.loop-runtime.v4"
V5 = "rusticol.loop-runtime.v5"
# A massless 2 -> 2 point (q q~ > g g).
MASSLESS = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "300", "0", "400"],
    ["500", "-300", "0", "-400"],
]
FLAVOURS = '{q = ["u", "d", "s", "c", "b"], qb = ["u~", "d~", "s~", "c~", "b~"]}'


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _card(output: Path, entries, *, process: str = "", colour: str = "") -> str:
    fixed = ", ".join(f"{key} = {json.dumps(value)}" for key, value in FIXED.items())
    listed = ", ".join(
        f"{{expression = {json.dumps(expression)}, name = {json.dumps(name)}}}"
        for name, expression in entries
    )
    return f"""
action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{listed}]
loop_coupling_orders = ["QCD"]
{process}
[color]
{colour}
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
loop_induced_probes = true
[generation.validation]
post_build_validation = false
"""


def _generator(root: Path, entries, **settings):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config
    from pyamplicol.models.loading import compile_model_source

    compile_model_source("loop_sm", restriction="no_b_mass")
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(_card(root / "artifact", entries, **settings))
    return Generator(resolve_config(card))


def _processes(entries):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet(
        tuple(
            ProcessRequest.parse(expression, name=name) for name, expression in entries
        )
    )


def _generate(root: Path, entries, **settings) -> Path:
    output = root / "artifact"
    _generator(root, entries, **settings).generate(_processes(entries), output)
    return output


def _plan(root: Path, entries, **settings):
    return _generator(root, entries, **settings).plan(_processes(entries))


def _members(artifact: Path) -> dict[str, tuple[str, list[str]]]:
    """Stored process id -> (expression, required runtime capabilities)."""

    manifest = json.loads((artifact / "artifact.json").read_text())
    return {
        record["id"]: (record["expression"], record["required_runtime_capabilities"])
        for record in manifest["processes"]
    }


def _singles(root: Path, members) -> tuple[Path, dict[str, str]]:
    """Every member expression generated as its own request in one run: the
    artifact and expression -> its process id."""

    entries = tuple((f"single_{i}", e) for i, e in enumerate(sorted(members)))
    return _generate(root, entries), {e: n for n, e in entries}


def _evaluate(artifact: Path, process: str, point, scale, parameters, **options):
    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process=process, model_parameters=parameters)
    return runtime, runtime.evaluate([point], mu_r2=[str(scale)], **options)


def _bits(value) -> list[str]:
    """The finite values, poles and Born values of an evaluation as float
    hex strings (bitwise; -0.0 differs from 0.0)."""

    flat = [x for sector in value.values[0] for x in sector]
    flat += list(value.born_values[0])
    return [float(x).hex() for x in flat]


def _keys(plan) -> list[str]:
    return [feature.split(":")[0] for feature in plan.unsupported_features]


def _point(name: str):
    fixture = _fixture(name)
    return (
        fixture,
        fixture["points"][0],
        Fraction(fixture["mu_r"][0]) ** 2,
        dict(fixture["scheme_data"]["parameters"]),
    )


# ------------------------------------------------------------- artifacts


@pytest.fixture(scope="module")
def pphh(tmp_path_factory):
    """p p > h h, purely loop-induced (generated once)."""

    return _generate(tmp_path_factory.mktemp("pphh"), (("p", "p p > h h"),))


@pytest.fixture(scope="module")
def ppzz(tmp_path_factory):
    """p p > z z with process.loop_induced = 'include' (generated once)."""

    return _generate(
        tmp_path_factory.mktemp("ppzz"),
        (("p", "p p > z z"),),
        process='loop_induced = "include"',
    )


# ------------------------------------------------- loop-induced: p p > h h


def _probes(artifact: Path) -> dict[str, dict]:
    """Process id -> its generation-time probe record (m-lit Q12)."""

    manifest = json.loads((artifact / "artifact.json").read_text())
    return {
        record["id"]: record["filters"]["loop_induced_probes"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
    }


def _assert_zero_member(artifact, key, expression, value, info, probes, inputs):
    """A q q~ > H H member: exactly 0. u, d, s: every loop has a single
    coloured attachment (the s-channel gluon on the top loop) and no
    counterterm ket, the exact colour zero with its record. c, b: no loop
    family either, but the counterterm kets of their Yukawa R2 / UV vertices
    (R2_ccH, UV_Hcc, R2_bbH, UV_Hbb: y_c = y_b = 0 at no_b_mass) carry a
    colour tensor at the root of leg 1 (H H > c c~ has none): not
    a colour zero (m-lit Q13), evaluated numerically, the probes passed,
    exactly 0 within its absolute estimate. ``inputs``: the evaluated
    (point, scale, parameters)."""

    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    ((finite, *poles),) = value.values[0]
    assert all(x == 0 for x in (finite, *poles)), expression
    (lane,) = value.diagnostics
    if expression.split()[0] in ("u", "d", "s"):
        assert info["exact_colour_zero"] is True, expression
        assert lane.detail.startswith("exact colour zero")
        assert probes[key] == {"exact_colour_zero": True}
        return
    assert info["exact_colour_zero"] is False, expression
    assert probes[key]["finiteness"]["passed"], expression
    _runtime, checked = _evaluate(
        artifact,
        key,
        *inputs,
        stability_checks=VirtualStabilityChecks(1, True, rescaled=True),
    )
    assert _bits(checked) == _bits(value)
    (estimate,) = checked.stability
    (error,) = estimate.absolute
    assert abs(finite) <= error.finite


@pytest.mark.loop_fast
def test_an_amplitude_without_a_slot_has_an_empty_term_scale():
    """``induced.term_values``: a native point without a (loop order, row)
    slot (no family routed at any order: counterterm kets alone) has the
    empty term scale; a slot without its term scale stays an error."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.induced import term_values

    empty = SimpleNamespace(native_bare={}, native_terms=None)
    assert term_values(empty) == {} and empty.term_cache == {}
    slot = SimpleNamespace(native_bare={("orders", "row"): [0j]}, native_terms=None)
    with pytest.raises(ValueError, match="no term scale"):
        term_values(slot)


@pytest.mark.loop_physics
def test_a_counterterm_only_loop_induced_process_generates(tmp_path):
    """c c~ > h h alone (a member of p p > h h): no loop family, the
    zero-coupling Yukawa counterterm kets only; generated (it failed in the
    generation-time probes for want of a term scale), served as exactly 0
    within its absolute estimate. Measured 25 s."""

    _unused, *inputs = _point("li_gg_hh")
    artifact = _generate(tmp_path / "cchh", (("cc", "c c~ > h h"),))
    runtime, value = _evaluate(artifact, "cc", *inputs)
    assert value.contribution == "loop-induced" and value.born_values == ((),)
    info, probes = runtime.inspect(), _probes(artifact)
    _assert_zero_member(artifact, "cc", "c c~ > h h", value, info, probes, inputs)


@pytest.mark.loop_nightly
def test_loop_induced_multiparticle_request_end_to_end(pphh, tmp_path):
    """p p > h h: six members, all served loop-induced (v5), each bitwise
    equal to the same process generated as its own request; G G > H H
    against MadLoop, the q q~ > H H members exactly 0 (``_assert_zero_member``:
    u, d, s exact colour zeros with their record; c, b numerical zeros within
    their absolute estimates). Measured 124-136 s (two generations of six
    processes)."""

    fixture, point, scale, parameters = _point("li_gg_hh")
    artifact = pphh
    members = _members(artifact)
    expressions = {
        expression.casefold(): key for key, (expression, _c) in members.items()
    }
    assert set(expressions) == {
        f"{q} {q}~ > h h" for q in ("d", "u", "s", "c", "b")
    } | {"g g > h h"}
    assert all(capabilities == [V5] for _e, capabilities in members.values())
    manifest = json.loads((artifact / "artifact.json").read_text())
    assert manifest["runtime"]["required_runtime_capabilities"] == [V5]
    assert "loop_skipped_members" not in manifest["extensions"]
    probes = _probes(artifact)
    singles, single = _singles(tmp_path / "singles", [e for e, _c in members.values()])
    for expression, key in sorted(expressions.items()):
        runtime, value = _evaluate(artifact, key, point, scale, parameters)
        _own, alone = _evaluate(
            singles, single[members[key][0]], point, scale, parameters
        )
        assert _bits(value) == _bits(alone), expression
        assert value.contribution == "loop-induced"
        assert value.born_orders == () and value.born_values == ((),)
        info = runtime.inspect()
        if expression != "g g > h h":
            inputs = (point, scale, parameters)
            _assert_zero_member(artifact, key, expression, value, info, probes, inputs)
            continue
        madloop = fixture["rows"][0]["providers"]["madloop"]["value"]
        assert abs(value.values[0][0].finite / madloop - 1) <= LAX
        assert info["exact_colour_zero"] is False
        assert probes[key]["finiteness"]["passed"] and probes[key]["ward"]["passed"]


# --------------------------------------------- include on a mixed request


@pytest.mark.loop_nightly
def test_include_serves_both_kinds_of_a_mixed_multiparticle_request(ppzz, tmp_path):
    """p p > z z, include: q q~ > Z Z Born-interfered (v4), G G > Z Z
    loop-induced (v5) in one artifact, nothing skipped; each member bitwise
    equal to its own request's process (never summed: the loop-induced
    member has no Born, the Born members no loop-squared part). Measured
    90 s (two generations of six processes)."""

    _unused, point, scale, parameters = _point("li_gg_zz")
    members = _members(ppzz)
    manifest = json.loads((ppzz / "artifact.json").read_text())
    assert manifest["runtime"]["required_runtime_capabilities"] == [V4, V5]
    assert "loop_skipped_members" not in manifest["extensions"]
    kinds = {expression.casefold(): caps for expression, caps in members.values()}
    assert kinds == {
        **{f"{q} {q}~ > z z": [V4] for q in ("d", "u", "s", "c", "b")},
        "g g > z z": [V5],
    }
    singles, single = _singles(tmp_path / "singles", [e for e, _c in members.values()])
    for key, (expression, capabilities) in sorted(members.items()):
        _runtime, value = _evaluate(ppzz, key, point, scale, parameters)
        _own, alone = _evaluate(singles, single[expression], point, scale, parameters)
        assert _bits(value) == _bits(alone), expression
        if capabilities == [V5]:
            assert value.contribution == "loop-induced"
            assert value.born_orders == () and value.born_values == ((),)
            assert value.values[0][0].finite > 0
        else:
            assert value.contribution == "virtual"
            assert value.born_values[0][0] > 0


# ------------------------------------------------------------ flavor_blind


@pytest.mark.loop_physics
def test_flavor_blind_folds_a_one_loop_member_only_by_exact_proof(tmp_path):
    """q qb > g g (q = u d s c b), flavor_blind: s s~ > G G is served by
    d d~ > G G (exact species proof, ``exact-loop-flavor``), bitwise its
    representative and its own generation to rounding; the others are
    generated on their own. Measured 47 s."""

    from pyamplicol.artifacts.process_groups import read_process_groups

    parameters = dict(_fixture("li_gg_hh")["scheme_data"]["parameters"])
    scale = "1000000"
    artifact = _generate(
        tmp_path / "fold",
        (("p", "q qb > g g"),),
        process=f"flavor_blind = true\nmultiparticles = {FLAVOURS}",
    )
    stored = {e.casefold(): key for key, (e, _c) in _members(artifact).items()}
    assert set(stored) == {f"{q} {q}~ > g g" for q in ("u", "d", "c", "b")}
    groups = read_process_groups(artifact)
    folded = {
        member.process_id: (
            representative,
            tuple(member.permutation),
            str(member.basis),
        )
        for representative, group in groups.groups.items()
        for member in group.members
        if not member.generated
    }
    ((member, (representative, permutation, basis)),) = folded.items()
    assert representative == stored["d d~ > g g"]
    assert permutation == (0, 1, 2, 3) and basis == "exact-loop-flavor"
    runtime, value = _evaluate(artifact, member, MASSLESS, scale, parameters)
    assert runtime.representative_process_key == representative
    assert runtime.physics.process_id == member
    _stored, served = _evaluate(artifact, representative, MASSLESS, scale, parameters)
    assert _bits(value) == _bits(served)
    assert value.born_values[0][0] > 0
    # Its own generation orders its terms by its own species: equal to
    # rounding (2 ulp measured; as the mapped members of test_loop_v2_mapping).
    alone = _generate(tmp_path / "alone", (("ssx", "s s~ > g g"),))
    _own, direct = _evaluate(alone, "ssx", MASSLESS, scale, parameters)
    mine = [float(x) for x in (*value.values[0][0], *value.born_values[0])]
    theirs = [float(x) for x in (*direct.values[0][0], *direct.born_values[0])]
    size = max(map(abs, theirs))
    assert all(abs(a - b) <= 1e-12 * size for a, b in zip(mine, theirs, strict=True))


@pytest.mark.loop_nightly
def test_flavor_blind_on_a_mixed_request_folds_nothing_without_a_proof(ppzz, tmp_path):
    """p p > z z, include, flavor_blind: no member has an exact proof (a
    down-type substitution changes the diagonal-CKM W couplings the
    electroweak orders retain; up and down types differ in charge), so every
    member, the loop-induced G G > Z Z included, is generated on its own and
    evaluates bitwise as the plain generation (``ppzz``), even where two
    members' values agree (d d~ and s s~ > Z Z). Measured 50 s (with the
    shared ``ppzz`` artifact)."""

    from pyamplicol.artifacts.process_groups import read_process_groups

    _unused, point, scale, parameters = _point("li_gg_zz")
    artifact = _generate(
        tmp_path / "blind",
        (("p", "p p > z z"),),
        process='loop_induced = "include"\nflavor_blind = true',
    )
    members = _members(artifact)
    assert members == _members(ppzz)
    groups = read_process_groups(artifact)
    assert all(m.generated for g in groups.groups.values() for m in g.members)
    assert sorted(groups.groups) == sorted(members)
    values = {}
    for key, (expression, _capabilities) in sorted(members.items()):
        _runtime, value = _evaluate(artifact, key, point, scale, parameters)
        _plain, plain = _evaluate(ppzz, key, point, scale, parameters)
        assert _bits(value) == _bits(plain), expression
        values[expression.casefold()] = _bits(value)
    assert values["d d~ > z z"] == values["s s~ > z z"]


@pytest.mark.loop_nightly
def test_flavor_blind_on_loop_induced_members_folds_nothing_without_a_proof(
    pphh, tmp_path
):
    """p p > h h, flavor_blind: the five q q~ > H H members are candidates of
    one another (d, s, b and u, c share charge and mass) but none has an
    exact proof (the electroweak orders retain the diagonal-CKM W couplings
    a down-type substitution changes; the zero-coupling Yukawa counterterm
    rules of c and b), so all six are generated on their own and evaluate
    bitwise as the plain generation (``pphh``). Measured 66 s (with the
    shared ``pphh`` artifact)."""

    from pyamplicol.artifacts.process_groups import read_process_groups

    _unused, *inputs = _point("li_gg_hh")
    artifact = _generate(
        tmp_path / "blind", (("p", "p p > h h"),), process="flavor_blind = true"
    )
    members = _members(artifact)
    assert members == _members(pphh)
    groups = read_process_groups(artifact)
    assert all(m.generated for g in groups.groups.values() for m in g.members)
    assert sorted(groups.groups) == sorted(members)
    for key, (expression, _capabilities) in sorted(members.items()):
        _runtime, value = _evaluate(artifact, key, *inputs)
        _plain, plain = _evaluate(pphh, key, *inputs)
        assert _bits(value) == _bits(plain), expression


# ------------------------------------------------------------- LC and NLC


@pytest.mark.loop_physics
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_truncated_colour_refuses_routed_loop_induced_members_as_a_whole(
    tmp_path, accuracy
):
    """LC/NLC: p p > h h, p p > z z with include or only, and a run mixing
    p p > z z with p p > h h are refused as a whole with
    ``loop-induced-colour-accuracy`` (also by ``generate``, which writes no
    artifact); p p > z z under auto passes at that accuracy, G G > Z Z
    skipped with its typed ``loop-induced`` record. Measured 16 s each."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    colour = f"accuracy = {json.dumps(accuracy)}"
    zz = (("p", "p p > z z"),)
    for tag, entries, process in (
        ("pphh", (("p", "p p > h h"),), ""),
        ("include", zz, 'loop_induced = "include"'),
        ("only", zz, 'loop_induced = "only"'),
        ("two", (("z", "p p > z z"), ("h", "p p > h h")), ""),
    ):
        plan = _plan(tmp_path / tag, entries, process=process, colour=colour)
        assert _keys(plan) == ["loop-induced-colour-accuracy"], tag
        assert f"color.accuracy = {accuracy!r}" in plan.unsupported_features[0]
    root = tmp_path / "generate"
    with pytest.raises(LoopRequestUnsupported) as refused:
        _generate(root, zz, process='loop_induced = "include"', colour=colour)
    assert refused.value.key == "loop-induced-colour-accuracy"
    assert not (root / "artifact").exists()
    plan = _plan(tmp_path / "auto", zz, colour=colour)
    assert plan.unsupported_features == ()
    records = plan.estimated_coverage["processes"]
    assert sorted(record["process"].casefold() for record in records) == sorted(
        f"{q} {q}~ > z z" for q in ("d", "u", "s", "c", "b")
    )
    assert all(str(record["color_accuracy"]) == accuracy for record in records)
    assert all("contribution" not in record for record in records)
    ((skipped),) = plan.estimated_coverage["skipped_members"]
    assert skipped["process"].casefold() == "g g > z z"
    assert skipped["refusal"] == "loop-induced"
