# SPDX-License-Identifier: 0BSD
"""Loop-induced request semantics, routing and order inference (m-lit
LIT-REQ; PLAN §1.2-§1.4, §2.a.10, rulings Q2 (b), Q4 (a), Q7, Q8 (a)).

* ``process.loop_induced = auto | skip | only | include`` (ProcessConfig,
  TOML, CLI ``--loop-induced``, Python ``LoopInducedPolicy``) routes the
  concrete processes without a Born amplitude within their coupling-order
  limits (loop-induced, |A_loop|^2): every value by dry run on the mixed
  ``p p > z z`` and the purely loop-induced ``g g > h``.
* Their loop orders are inferred from the field-level loop family recursion
  (``assemble.reachable_loop_orders``, ``loop_service.loop_induced_orders``):
  the PLAN §1.3 table on loop_sm and loop_qcd_qed_sm; explicit bounds are
  honoured; no reachable amplitude is the typed ``no-loop-amplitude``.
* Full colour only (``loop-induced-colour-accuracy``); fixed tree widths
  admitted (the coloured-width refusal stays the Born-interfered guard),
  unstable external states refused under the complex-mass scheme.
* Generation refuses ``loop-induced-engine-pending`` until the amplitude
  engine lands (never a silent zero); Born-interfered dry runs and request
  records are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
# The complex-mass scheme with a stable Higgs, Z and W (the top keeps its
# width; ``test_loop_v2_cms.py``).
STABLE = {"WZ": "0", "WW": "0", "WH": "0"}


def _card(
    output: Path,
    entries,
    *,
    model: str = "loop_sm",
    restriction: str | None = "no_b_mass",
    orders=("QCD",),
    policy: str = "minimal",
    loop_induced: str | None = None,
    process: str = "",
    colour: str = "",
    loop: str = "",
    fixed: dict[str, str] | None = None,
    scheme: str | None = None,
) -> str:
    fixed = ZERO_WIDTHS if fixed is None else fixed
    parameters = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    listed = ", ".join(
        f"{{expression = {json.dumps(expression)}, name = {json.dumps(name)}}}"
        for name, expression in entries
    )
    restricted = (
        "" if restriction is None else f"restriction = {json.dumps(restriction)}"
    )
    mode = "" if loop_induced is None else f"loop_induced = {json.dumps(loop_induced)}"
    declared = (
        "" if scheme is None else f'[loop.model.scheme]\nmass_scheme = "{scheme}"'
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(model)}
{restricted}
[process]
entries = [{listed}]
loop_coupling_orders = {json.dumps(list(orders))}
coupling_order_policy = {json.dumps(policy)}
{mode}
{process}
[color]
{colour}
[loop]
r2_strategy = "model"
{loop}
{declared}
[loop.preparation]
fixed_parameters = {{{parameters}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _generator(tmp_path: Path, entries, **settings):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    card = tmp_path / "card.toml"
    card.write_text(_card(tmp_path / "out", entries, **settings))
    return Generator(resolve_config(card))


def _processes(entries):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet(
        tuple(
            ProcessRequest.parse(expression, name=name) for name, expression in entries
        )
    )


def _plan(tmp_path: Path, entries, **settings):
    from pyamplicol.models.loading import compile_model_source

    compile_model_source(
        settings.get("model", "loop_sm"),
        restriction=settings.get("restriction", "no_b_mass") or "default",
    )
    return _generator(tmp_path, entries, **settings).plan(_processes(entries))


def _keys(plan) -> list[str]:
    return [feature.split(":")[0] for feature in plan.unsupported_features]


def _induced(plan) -> dict[str, dict]:
    return {
        record["key"]: record
        for record in plan.estimated_coverage["processes"]
        if record.get("contribution") == "loop-induced"
    }


# The keys of a Born-interfered dry-run record (unchanged by LIT-REQ).
BORN_RECORD_KEYS = {
    "key",
    "process",
    "engine",
    "born_orders",
    "loop_orders",
    "corrections",
    "color_accuracy",
    "scale_parameter",
    "loop_graph_count",
    "numerator_compilation_deferred",
    "loop_admission_checked",
    "native_export_checked",
}


# ---------------------------------------------------------- configuration


@pytest.mark.loop_fast
def test_loop_induced_key_in_every_configuration_surface(tmp_path):
    """``process.loop_induced``: ProcessConfig default and choices, TOML,
    the CLI option and override, refused values; ``only`` and ``include``
    need a one-loop request; not a removed field."""

    from pyamplicol.cli.parser import parse_cli
    from pyamplicol.config import (
        FIELD_REGISTRY,
        ConfigurationError,
        LoopInducedPolicy,
        ProcessConfig,
        resolve_config,
    )
    from pyamplicol.config.resolver import REMOVED_FIELDS

    assert ProcessConfig().loop_induced is LoopInducedPolicy.AUTO
    assert ProcessConfig(loop_induced="only").loop_induced is LoopInducedPolicy.ONLY
    with pytest.raises(ConfigurationError, match=r"process\.loop_induced must be one"):
        ProcessConfig(loop_induced="never")
    assert tuple(LoopInducedPolicy) == ("auto", "skip", "only", "include")
    assert FIELD_REGISTRY["process.loop_induced"].default == "auto"
    assert "process.loop_induced" not in REMOVED_FIELDS

    def cli(*argv):
        return parse_cli(("generate", *argv)).resolve().effective.process

    assert cli("--loop-coupling-order", "QCD").loop_induced == "auto"
    for value in LoopInducedPolicy:
        assert (
            cli("--loop-coupling-order", "QCD", "--loop-induced", value).loop_induced
            == value
        )
    assert (
        cli("--loop-coupling-order", "QCD", "--set", "process.loop_induced=only")
    ).loop_induced == "only"
    with pytest.raises(SystemExit):
        parse_cli(("generate", "--loop-induced", "never"))
    for value in ("only", "include"):
        with pytest.raises(ConfigurationError, match="requires one-loop generation"):
            parse_cli(("generate", "--loop-induced", value)).resolve()
    assert cli("--loop-induced", "skip").loop_induced == "skip"
    card = tmp_path / "card.toml"
    card.write_text(_card(tmp_path / "out", (("p", "g g > h"),), loop_induced="only"))
    assert resolve_config(card).effective.process.loop_induced == "only"


@pytest.mark.loop_fast
def test_loop_induced_orders_minimal_and_explicit():
    """``loop_induced_orders``: ``explicit`` keeps every reachable order;
    ``minimal`` every reachable order of the least weighted degree by the
    hierarchy (MG5_aMC's WEIGHTED bound); zero powers named."""

    from pyamplicol.generation.loop_service import loop_induced_orders

    assert loop_induced_orders(
        [(("QCD", 2), ("QED", 1))], policy="minimal", names=("QCD", "QED")
    ) == {"QCD": 2, "QED": 1}
    assert loop_induced_orders([(("QED", 4),)], policy="minimal", names=("QCD",)) == {
        "QCD": 0,
        "QED": 4,
    }
    weights = {"QCD": 1, "QED": 2}
    a = (("QCD", 2), ("QED", 1))  # degree 4
    b = (("QED", 2),)  # degree 4
    c = (("QCD", 2), ("QED", 2))  # degree 6
    d = (("QCD", 4), ("QED", 1))  # degree 6
    reached = [d, c, b, a]
    assert loop_induced_orders(reached, policy="minimal", hierarchy=weights) == (
        {"QCD": 2, "QED": 1},
        {"QCD": 0, "QED": 2},
    )
    assert loop_induced_orders(reached, policy="explicit", hierarchy=weights) == (
        {"QCD": 2, "QED": 1},
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        {"QCD": 4, "QED": 1},
    )
    # Equal weights: the lowest total power wins.
    assert loop_induced_orders([d, a], policy="minimal") == {"QCD": 2, "QED": 1}
    with pytest.raises(ValueError, match="reachable loop amplitude"):
        loop_induced_orders([], policy="minimal")
    with pytest.raises(ValueError, match="unknown coupling-order policy"):
        loop_induced_orders([a], policy="lowest")


@pytest.mark.loop_fast
def test_slice_request_contribution_and_its_record():
    """``SliceRequest.contribution``: recorded only for a loop-induced
    request (Born-interfered records keep their bytes), restored from the
    record; full colour only (Q8); its generation is refused typed before
    anything is built."""

    from dataclasses import replace

    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.refusals import (
        LoopRequestUnsupported,
        LoopRequestUnsupportedValue,
    )

    virtual = SliceRequest("u u~ > d d~", {"QCD": 2}, {"QCD": 4}, corrections=("QCD",))
    record = request_record(virtual)
    assert "contribution" not in record
    assert request_from_record(record) == virtual
    induced = SliceRequest("g g > h", {}, {"QCD": 2, "QED": 1}, corrections=("QCD",))
    induced = replace(induced, contribution="loop-induced")
    record = request_record(induced)
    assert record["contribution"] == "loop-induced"
    assert request_from_record(record) == induced
    with pytest.raises(ValueError, match="unknown contribution"):
        replace(virtual, contribution="loop-squared")
    for accuracy in ("lc", "nlc"):
        with pytest.raises(LoopRequestUnsupportedValue) as refused:
            replace(induced, accuracy=accuracy)
        assert refused.value.key == "loop-induced-colour-accuracy"
    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(None, induced, None)
    assert refused.value.key == "loop-induced-engine-pending"


@pytest.mark.loop_fast
def test_loop_induced_refusals_are_listed_and_name_the_key():
    from pyamplicol.loops.v2.refusals import (
        listed_configuration_keys,
        listed_refusal_keys,
        refusal_text,
    )
    from pyamplicol.processes.refusals import (
        ProcessRequestUnsupported,
        refuse_madloop_syntax,
    )

    admission = (
        "loop-induced",
        "born-in-loop-induced-mode",
        "no-loop-amplitude",
        "loop-induced-colour-accuracy",
        "loop-induced-engine-pending",
    )
    assert set(admission) <= set(listed_refusal_keys())
    assert "loop-induced-include-pending" in listed_configuration_keys()
    for key in ("loop-induced", "born-in-loop-induced-mode"):
        assert "process.loop_induced = 'auto'" in refusal_text(key, "m")
    assert "color.accuracy = 'full'" in refusal_text(
        "loop-induced-colour-accuracy", "m"
    )
    with pytest.raises(ProcessRequestUnsupported) as refused:
        refuse_madloop_syntax("g g > h [noborn=QCD]")
    message = str(refused.value)
    assert refused.value.key == "madloop-syntax"
    assert "a loop-induced NLO bracket ('[noborn=QCD]' style" in message
    assert 'process.loop_induced = "auto"' in message
    with pytest.raises(ProcessRequestUnsupported) as refused:
        refuse_madloop_syntax("u u~ > d d~ [virt=QCD]")
    assert "an NLO bracket ('[virt=QCD]' style)" in str(refused.value)


# ------------------------------------------------------------ order table

# PLAN §1.3 "Expected orders" (QCD-induced: [QCD] corrections).
QCD_INDUCED = {
    "gg_h": ("g g > h", {"QCD": 2, "QED": 1}),
    "gg_hh": ("g g > h h", {"QCD": 2, "QED": 2}),
    "gg_zz": ("g g > z z", {"QCD": 2, "QED": 2}),
    "gg_aa": ("g g > a a", {"QCD": 2, "QED": 2}),
    "gg_ww": ("g g > w+ w-", {"QCD": 2, "QED": 2}),
    "gg_zh": ("g g > z h", {"QCD": 2, "QED": 2}),
    "gg_hg": ("g g > h g", {"QCD": 3, "QED": 1}),
    "h_gg": ("h > g g", {"QCD": 2, "QED": 1}),
}
# EW-induced ([QED] corrections; the complete a a > a a, Q38 (a)).
EW_INDUCED = {
    "aa_aa": ("a a > a a", {"QCD": 0, "QED": 4}),
    "h_aa": ("h > a a", {"QCD": 0, "QED": 3}),
    "h_za": ("h > z a", {"QCD": 0, "QED": 3}),
}


def _check_table(plan, table, corrections):
    induced = _induced(plan)
    assert set(induced) == set(table), plan.unsupported_features
    assert _keys(plan) == ["loop-induced-engine-pending"]
    for key, (_expression, orders) in table.items():
        record = induced[key]
        assert dict(record["loop_orders"]) == orders, key
        assert dict(record["born_orders"]) == {}
        assert record["interference"] == "conj(loop)*loop"
        assert record["loop_order_policy"] == "minimal"
        assert record["color_accuracy"] == "full"
        assert list(record["corrections"]) == list(corrections)
        assert record["width_policy"] == "fixed-tree-widths"
        ((sector),) = record["sectors"]
        assert dict(sector["order"]) == {k: 2 * v for k, v in orders.items()}
        assert [list(map(dict, pair)) for pair in sector["pairs"]] == [[orders, orders]]


@pytest.mark.loop_physics
def test_order_table_is_inferred_on_loop_sm(tmp_path):
    """The QCD-induced rows of PLAN §1.3 by inference (minimal) on loop_sm,
    one dry run: each is served loop-induced at its expected orders."""

    entries = tuple((key, expression) for key, (expression, _) in QCD_INDUCED.items())
    _check_table(_plan(tmp_path, entries), QCD_INDUCED, ("QCD",))


@pytest.mark.loop_physics
def test_order_table_is_inferred_on_loop_qcd_qed_sm(tmp_path):
    """The PLAN §1.3 table on loop_qcd_qed_sm (one dry run, QCD and QED
    corrections: MadLoop's joint selection admits each loop under either):
    the QCD-induced rows keep their orders, the EW-induced ones (the
    complete a a > a a, h > a a, h > z a) take QED = 4, 3, 3."""

    table = {
        key: QCD_INDUCED[key] for key in ("gg_h", "gg_zz", "gg_hg", "h_gg")
    } | EW_INDUCED
    entries = tuple((key, expression) for key, (expression, _) in table.items())
    plan = _plan(
        tmp_path,
        entries,
        model="loop_qcd_qed_sm",
        restriction=None,
        orders=("QCD", "QED"),
        # loop_qcd_qed_sm has no alpha(0) input: no photon is tagged.
        loop='external_photons = "none"',
    )
    _check_table(plan, table, ("QCD", "QED"))


@pytest.mark.loop_physics
def test_native_and_python_recursions_reach_the_same_orders():
    """``reachable_loop_orders`` on the native family recursion equals its
    Python oracle (``families.generate_families``)."""

    from pyamplicol.loops.covariant_model import with_source_repairs
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import (
        SliceRequest,
        prepare_request_model,
        reachable_loop_orders,
    )
    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.loop_declaration import (
        resolve_loop_model_declaration,
        with_loop_model_declaration,
    )

    compiled = with_source_repairs(
        compile_model_source("loop_sm", restriction="no_b_mass")
    )
    compiled = with_loop_model_declaration(
        compiled,
        resolve_loop_model_declaration(
            compiled.raw_loop_model, None, selected_corrections=("QCD",)
        ),
    )
    for process, orders in (
        ("g g > z z", (("QCD", 2), ("QED", 2))),
        ("g g > h g", (("QCD", 3), ("QED", 1))),
    ):
        request = SliceRequest(
            process,
            {},
            {},
            corrections=("QCD",),
            fixed_parameters=ZERO_WIDTHS,
            scale_parameter="MU_R",
            contribution="loop-induced",
        )
        model = prepare_request_model(compiled, request)
        native = reachable_loop_orders(
            compiled, request, {}, model=model, native=load_loop_native()
        )
        python = reachable_loop_orders(compiled, request, {}, model=model)
        assert native == python == frozenset({orders}), process
        # An explicit bound below the lowest order reaches nothing.
        assert not reachable_loop_orders(
            compiled, request, {"QED": 0}, model=model, native=load_loop_native()
        )


# ------------------------------------------------------- routing by value


@pytest.mark.loop_physics
@pytest.mark.parametrize("value", ["auto", "skip", "only", "include"])
def test_each_value_on_mixed_and_purely_loop_induced_requests(tmp_path, value):
    """Every value of ``process.loop_induced`` by dry run on p p > z z (Born
    members and g g > z z) and g g > h (purely loop-induced)."""

    mixed = _plan(tmp_path / "ppzz", (("p", "p p > z z"),), loop_induced=value)
    induced = _plan(tmp_path / "ggh", (("p", "g g > h"),), loop_induced=value)
    if value == "include":
        for plan in (mixed, induced):
            assert _keys(plan) == ["loop-induced-include-pending"]
        return
    skipped = mixed.estimated_coverage["skipped_members"]
    records = mixed.estimated_coverage["processes"]
    if value in ("auto", "skip"):
        # Born members served as before, g g > z z skipped with today's record.
        assert mixed.unsupported_features == ()
        assert len(records) == 5
        assert all(set(record) == BORN_RECORD_KEYS for record in records)
        assert all(dict(record["born_orders"])["QED"] == 2 for record in records)
        ((record),) = skipped
        assert record["process"].casefold() == "g g > z z"
        assert record["refusal"] == "loop-induced"
        assert record["reason"].startswith("loop-induced: ")
    else:
        # only: g g > z z served loop-induced, the Born members skipped.
        assert _keys(mixed) == ["loop-induced-engine-pending"]
        ((record),) = records
        assert record["contribution"] == "loop-induced"
        assert record["process"].casefold() == "g g > z z"
        assert dict(record["loop_orders"]) == {"QCD": 2, "QED": 2}
        assert len(skipped) == 5
        assert {item["refusal"] for item in skipped} == {"born-in-loop-induced-mode"}
    if value == "skip":
        assert _keys(induced) == ["loop-induced"]
        assert "process.loop_induced = 'auto'" in induced.unsupported_features[0]
        return
    assert _keys(induced) == ["loop-induced-engine-pending"]
    (record,) = induced.estimated_coverage["processes"]
    assert record["contribution"] == "loop-induced"
    assert dict(record["loop_orders"]) == {"QCD": 2, "QED": 1}
    assert dict(record["coupling_order_limits"]) == {}
    if value == "only":
        born = _plan(tmp_path / "uudd", (("p", "u u~ > d d~"),), loop_induced=value)
        assert _keys(born) == ["born-in-loop-induced-mode"]


@pytest.mark.loop_physics
def test_members_without_any_amplitude_and_explicit_bounds(tmp_path):
    """A Born-less member that reaches no loop amplitude is skipped with
    ``no-loop-amplitude`` (p p > h h: the massless quark members; refused
    for a single process: a a > a a under QCD, g g > h with QED = 0);
    explicit bounds bound the loop orders under either policy (u u~ > d d~ h
    with QED <= 1 excludes the VBF Born)."""

    plan = _plan(tmp_path / "pphh", (("p", "p p > h h"),))
    assert _keys(plan) == ["loop-induced-engine-pending"]
    (record,) = _induced(plan).values()
    assert record["process"].casefold() == "g g > h h"
    skipped = plan.estimated_coverage["skipped_members"]
    assert skipped and {item["refusal"] for item in skipped} == {"no-loop-amplitude"}
    for name, expression, process in (
        ("aa", "a a > a a", ""),
        ("gg", "g g > h", "max_coupling_orders = {QED = 0}"),
    ):
        plan = _plan(tmp_path / name, (("p", expression),), process=process)
        assert _keys(plan) == ["no-loop-amplitude"]
    for policy in ("minimal", "explicit"):
        plan = _plan(
            tmp_path / policy,
            (("p", "u u~ > d d~ h"),),
            policy=policy,
            process="max_coupling_orders = {QED = 1}",
        )
        (record,) = _induced(plan).values()
        assert dict(record["coupling_order_limits"]) == {"QED": 1}
        assert dict(record["loop_orders"]) == {"QCD": 4, "QED": 1}
        assert record["loop_order_policy"] == policy
    # Without the bound the VBF tree is its Born (Born-interfered).
    plan = _plan(tmp_path / "vbf", (("p", "u u~ > d d~ h"),))
    assert plan.unsupported_features == ()
    (record,) = plan.estimated_coverage["processes"]
    assert set(record) == BORN_RECORD_KEYS


# ------------------------------------------------- colour, widths, engine


@pytest.mark.loop_physics
def test_colour_accuracy_and_generation_refusals(tmp_path):
    """LC/NLC loop-induced requests are refused typed (Q8); the generation
    of an admitted loop-induced request is refused typed before any output
    (never a silent zero)."""

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    for accuracy in ("lc", "nlc"):
        plan = _plan(
            tmp_path / accuracy,
            (("p", "g g > h"),),
            colour=f'accuracy = "{accuracy}"',
        )
        assert _keys(plan) == ["loop-induced-colour-accuracy"]
    generator = _generator(tmp_path / "gen", (("p", "g g > h"),))
    output = tmp_path / "gen" / "out"
    with pytest.raises(GenerationError) as refused:
        generator.generate(_processes((("p", "g g > h"),)), output)
    assert isinstance(refused.value, LoopRequestUnsupported)
    assert refused.value.key == "loop-induced-engine-pending"
    assert not output.exists()


@pytest.mark.loop_physics
def test_width_policy_on_the_loop_induced_branch_only(tmp_path, monkeypatch):
    """Q7: the loop-induced branch takes no coloured-width refusal (fixed
    tree widths admitted) while Born-interfered requests keep it: with the
    refusal forced on every DAG, u u~ > d d~ is refused and g g > h h (at
    the card's top width) is served. Under the complex-mass scheme a
    loop-induced request records it and an unstable external state is
    refused (``cms-external-width``, ahead of the routing)."""

    from pyamplicol.loops.v2 import artifact
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    def refuse(process, *_args):
        raise LoopRequestUnsupported("coloured-width", f"{process}: forced")

    plan = _plan(tmp_path / "ggtt", (("p", "g g > t t~"),), fixed={"WZ": "0"})
    assert _keys(plan) == ["coloured-width"]
    assert 'WT = "0"' in plan.unsupported_features[0]
    plan = _plan(tmp_path / "gghh", (("p", "g g > h h"),), fixed=STABLE)
    assert _keys(plan) == ["loop-induced-engine-pending"]
    (record,) = _induced(plan).values()
    assert record["width_policy"] == "fixed-tree-widths"
    monkeypatch.setattr(artifact, "require_zero_coloured_widths", refuse)
    plan = _plan(tmp_path / "uudd", (("p", "u u~ > d d~"),))
    assert _keys(plan) == ["coloured-width"]
    plan = _plan(tmp_path / "gghh2", (("p", "g g > h h"),), fixed=STABLE)
    assert _keys(plan) == ["loop-induced-engine-pending"]
    monkeypatch.undo()
    plan = _plan(
        tmp_path / "cms", (("p", "g g > h"),), fixed=STABLE, scheme="complex-mass"
    )
    assert _keys(plan) == ["loop-induced-engine-pending"]
    (record,) = _induced(plan).values()
    assert record["width_policy"] == "complex-mass-scheme"
    plan = _plan(
        tmp_path / "cmsh",
        (("p", "g g > h"),),
        fixed={"WZ": "0", "WW": "0"},
        scheme="complex-mass",
    )
    assert _keys(plan) == ["cms-external-width"]
