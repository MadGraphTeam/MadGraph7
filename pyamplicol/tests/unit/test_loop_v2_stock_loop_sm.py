# SPDX-License-Identifier: 0BSD
"""Stock loop_sm on the public route (m-qcd PLAN MQ-MODEL).

* The production source repair (``covariant_model.repair_source_declarations``)
  equals the checkout's explicit scalar-arity view
  (``tools/loop_reference/model_compatibility.py``, the independent oracle):
  the same derived declarations, found structurally, idempotent, recorded.
* The wheel ships MG5_aMC 3.7.0's loop_sm verbatim; ``model.source =
  "loop_sm"`` selects it, and its ``[loop.model]`` declaration is the shipped
  profile of its exact declarations (one digest for the shipped copy, the
  MG5_aMC directory and every restriction).
* A one-loop request without ``color.accuracy`` is full colour.
* The public route on the shipped loop_sm (no ``tools/`` import, no
  ``[loop.model]``, no colour setting) generates the QED-Born requests the
  scalar-arity defect blocked, and its binary64 values are bitwise equal to
  the tools-view route's (default tier: u u~ > e- e+; nightly: the
  MQ-MODEL must-generate list, also on the raw MG5_aMC directory).
* Every listed refusal key a request on loop_sm can reach fires in a dry run
  (``Generator.plan`` / ``pyamplicol generate --dry-run``), and the dry run
  lists every key.
* A multiparticle request (``p p > t t~``, campaign row M01) holds MadLoop's
  subprocesses, each bitwise its own single-process generation (nightly), and
  a loop process loads under any letter case of the model's spelling.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}


def _ufo_card(
    model: str,
    output: Path,
    expression: str,
    *,
    restriction: str | None = None,
    fixed: dict[str, str] | None = None,
    scheme: bool = False,
) -> str:
    """A public-route card: no colour setting, no [loop.model] unless
    ``scheme`` (the tools-view route states it, as its tests always did)."""

    fixed = ZERO_WIDTHS if fixed is None else fixed
    parameters = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    declared = (
        """
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
"""
        if scheme
        else ""
    )
    restricted = (
        "" if restriction is None else f"restriction = {json.dumps(restriction)}"
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(model)}
{restricted}
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[loop]
r2_strategy = "model"
{declared}
[loop.preparation]
fixed_parameters = {{{parameters}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _generator(card_text: str, path: Path):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    path.write_text(card_text)
    return Generator(resolve_config(path))


def _processes(expression: str):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet((ProcessRequest.parse(expression, name="p"),))


# ------------------------------------------------------------ source repair


@pytest.mark.loop_fast
def test_source_repair_equals_the_scalar_arity_view(loop_sm_ufo):
    """The production rule derives exactly the checkout view's declarations
    (one vertex: V_1001's SSS1 -> SSSS1, found by structure), records the
    repair, and is a no-op on a repaired source (the view's included)."""

    from pyamplicol.loops.covariant_model import (
        SOURCE_REPAIRS_KEY,
        repair_source_declarations,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    raw = load_raw_loop_model(loop_sm_ufo)
    original = raw.to_dict()
    repaired = repair_source_declarations(raw)
    assert raw.to_dict() == original
    view = prepare_loop_sm_scalar_declarations(raw).derived
    assert repaired.to_dict()["collections"] == view.to_dict()["collections"]
    (record,) = repaired.metadata[SOURCE_REPAIRS_KEY]
    assert record["rule"] == "constant-tensor-slot-arity"
    assert record["vertex"][1] == "V_1001"
    assert record["original_lorentz"][1] == "SSS1"
    assert record["derived_lorentz"][1] == "SSSS1"
    assert repair_source_declarations(repaired) is repaired
    assert repair_source_declarations(view) is view


@pytest.mark.loop_fast
def test_unrepairable_slot_mismatch_is_the_listed_refusal(loop_sm_ufo):
    """A slot mismatch that is not a constant scalar tensor stays as declared
    and refuses, typed, once a request reaches the vertex
    (``lorentz-slot-arity``); unreached, it is inert."""

    from symbolica import is_licensed

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.covariant_model import repair_source_declarations
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

    if not is_licensed():
        pytest.skip("source preparation requires licensed Symbolica")
    raw = load_raw_loop_model(loop_sm_ufo)
    payload = raw.to_dict()
    # V_1001 (four G0) on a three-slot tensor that is not a constant scalar
    # one (a vector slot).
    vertex = raw.record("vertices", "V_1001")
    other = next(
        record
        for record in raw.collections["lorentz_structures"]
        if len(record["fields"]["spins"]) == 3 and set(record["fields"]["spins"]) != {1}
    )
    payload["collections"]["vertices"][vertex["id"]]["fields"]["lorentz"] = [
        dict(
            kind="reference",
            collection="lorentz_structures",
            id=other["id"],
            name=other["name"],
        )
    ]
    doctored = RawLoopModel.from_dict(payload)
    assert repair_source_declarations(doctored) is doctored
    fixed = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0"}
    # QED = 0: the request scope never reaches the four-scalar vertex.
    prepare_loop_request_model(
        doctored,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters=fixed,
    )
    with pytest.raises(LoopRequestUnsupported) as refused:
        prepare_loop_request_model(
            doctored,
            born_order_bounds={"QCD": 2, "QED": 2},
            loop_order_bounds={"QCD": 4, "QED": 2},
            fixed_parameters=fixed,
        )
    assert refused.value.key == "lorentz-slot-arity"
    assert "V_1001" in str(refused.value) and other["name"] in str(refused.value)


# -------------------------------------------------- shipped model and profile


@pytest.mark.loop_fast
def test_shipped_loop_sm_is_verbatim_and_selected_by_name(loop_sm_ufo, tmp_path):
    """The installed loop_sm is MG5_aMC 3.7.0's byte for byte; a bare name
    selects it (card, API), an existing path of that name wins."""

    from pyamplicol._internal.packaged_models import (
        packaged_ufo_model,
        packaged_ufo_models,
    )
    from pyamplicol.api.requests import ModelSource
    from pyamplicol.config import resolve_config

    shipped = packaged_ufo_model("loop_sm")
    assert shipped is not None and "loop_sm" in packaged_ufo_models()
    names = sorted(path.name for path in shipped.iterdir() if path.is_file())
    assert names == sorted(
        path.name
        for path in Path(loop_sm_ufo).iterdir()
        if path.is_file()
        and (path.suffix == ".py" or path.name.startswith("restrict_"))
    )
    for name in names:
        assert (shipped / name).read_bytes() == (Path(loop_sm_ufo) / name).read_bytes()
    assert ModelSource.from_path("loop_sm").path == shipped
    card = tmp_path / "card.toml"
    card.write_text('action = "generate"\n[model]\nsource = "loop_sm"\n')
    assert Path(resolve_config(card).effective.model.source) == shipped
    (tmp_path / "loop_sm").mkdir()
    (tmp_path / "loop_sm" / "__init__.py").write_text("")
    assert Path(resolve_config(card).effective.model.source) == tmp_path / "loop_sm"
    assert packaged_ufo_model("../loop_sm") is None
    assert packaged_ufo_model("no_such_model") is None


@pytest.mark.loop_fast
def test_shipped_profile_declares_loop_sm_by_its_declarations(loop_sm_ufo):
    """One declarations digest for the shipped copy, the MG5_aMC directory,
    every restriction and the repaired source; the profile is the
    declaration the loop_sm tests state; an explicit or source declaration
    wins; another source gets none."""

    from pyamplicol._internal.packaged_models import packaged_ufo_model
    from pyamplicol.loops.shipped_profiles import (
        LOOP_SM_SCHEME,
        shipped_loop_profile,
        source_declarations_digest,
    )
    from pyamplicol.models.loop_declaration import (
        LoopModelDeclaration,
        resolve_loop_model_declaration,
        with_loop_model_declaration,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    digests = set()
    for source in (loop_sm_ufo, packaged_ufo_model("loop_sm")):
        for restriction in ("default", "no_b_mass", "no_widths", "zeromass_ckm"):
            raw = load_raw_loop_model(source, restriction=restriction)
            digests.add(source_declarations_digest(raw))
            profile = shipped_loop_profile(raw)
            assert profile is not None
            chosen = resolve_loop_model_declaration(raw, selected_corrections=("QCD",))
            assert chosen == profile.declaration
            assert chosen.scheme == LOOP_SM_SCHEME
            assert chosen.loop_coupling_orders == ("QCD",)
            assert chosen.uv_complete and chosen.model_r2_complete
    assert len(digests) == 1
    with pytest.raises(ValueError, match="does not cover selected corrections: QED"):
        resolve_loop_model_declaration(raw, selected_corrections=("QED",))
    other = LoopModelDeclaration(
        replace(LOOP_SM_SCHEME, model_convention="explicit"), ("QCD",), True, True
    )
    declared = with_loop_model_declaration(raw, other)
    assert resolve_loop_model_declaration(declared, selected_corrections=("QCD",)) == (
        other
    )
    payload = raw.to_dict()
    payload["collections"]["couplings"][0]["fields"]["expression"] = "0"
    from pyamplicol.models.loop_import import RawLoopModel

    assert shipped_loop_profile(RawLoopModel.from_dict(payload)) is None


@pytest.mark.loop_fast
def test_one_loop_requests_default_to_full_colour(tmp_path):
    """color.accuracy unset: full colour for a one-loop request, LC for a
    tree request; an explicit setting is kept."""

    from pyamplicol.config import ColorAccuracy, resolve_config

    card = tmp_path / "card.toml"
    for loop, accuracy, expected in (
        (True, None, ColorAccuracy.FULL),
        (False, None, ColorAccuracy.LC),
        (True, "lc", ColorAccuracy.LC),
        (True, "nlc", ColorAccuracy.NLC),
        (False, "full", ColorAccuracy.FULL),
    ):
        text = 'action = "generate"\n'
        text += '[process]\nloop_coupling_orders = ["QCD"]\n' if loop else ""
        if accuracy is not None:
            text += f'[color]\naccuracy = "{accuracy}"\n'
        card.write_text(text)
        resolution = resolve_config(card)
        assert resolution.effective.color.accuracy is expected
        assert resolution.requested.color.accuracy is expected


# ------------------------------------------------------------- public route


def _compile(source, restriction=None):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(source, restriction=restriction or "default")


def _point(masses_in, masses_out, seed, sqrt_s=1000.0):
    """One physical binary64 point: incoming along +-z, RAMBO final state."""

    from pyamplicol.generation.phase_space import massive_rambo_final_state

    m1, m2 = masses_in
    p = math.sqrt((sqrt_s**2 - (m1 + m2) ** 2) * (sqrt_s**2 - (m1 - m2) ** 2)) / (
        2 * sqrt_s
    )
    incoming = [
        [math.sqrt(p * p + m1 * m1), 0.0, 0.0, p],
        [math.sqrt(p * p + m2 * m2), 0.0, 0.0, -p],
    ]
    final = massive_rambo_final_state(
        len(masses_out), sqrt_s=sqrt_s, masses=masses_out, seed=seed
    )
    return incoming + [list(map(float, momentum)) for momentum in final]


def _masses(model, expression: str) -> tuple[list[float], list[float]]:
    """The external masses of ``expression`` at the compiled ``model``'s
    parameter values (incoming, outgoing)."""

    from pyamplicol.processes.model import ModelParticleCatalog

    catalog = ModelParticleCatalog(
        model.ir.name, model.ir.particles, model.ir.parameters
    )
    values = {record.name: record.value for record in model.ir.parameters}

    def mass(token: str) -> float:
        name = catalog.resolve(token).mass
        return 0.0 if name.casefold() == "zero" else float(values[name][0])

    initial, final = expression.split(">")
    return [mass(t) for t in initial.split()], [mass(t) for t in final.split()]


def _values(result):
    return [
        (
            [complex(value) for value in point_born],
            [
                [complex(c) if c is not None else None for c in laurent]
                for laurent in point_values
            ],
        )
        for point_born, point_values in zip(
            result.born_values, result.values, strict=True
        )
    ]


def _public_and_view(
    model, view, expression, tmp_path, *, restriction=None, fixed=None
):
    """The binary64 values of ``expression`` through the public route on
    ``model`` (no [loop.model], no colour setting) and through the
    tools-view compiled model with the explicit declaration, at two points."""

    from pyamplicol import Runtime

    runs = {}
    for name, source, scheme, chosen in (
        ("public", model, False, restriction),
        ("view", view, True, None),
    ):
        output = tmp_path / name
        generator = _generator(
            _ufo_card(
                str(source),
                output,
                expression,
                restriction=chosen,
                fixed=fixed,
                scheme=scheme,
            ),
            tmp_path / f"{name}.toml",
        )
        generator.generate(_processes(expression), output)
        runs[name] = Runtime.load(output, process="p")
    masses_in, masses_out = _masses(_compile(model, restriction), expression)
    points = [_point(masses_in, masses_out, seed) for seed in (11, 12)]
    values = {
        name: _values(runtime.evaluate(points, mu_r2=[8315.0] * len(points)))
        for name, runtime in runs.items()
    }
    for born, laurent in values["public"]:
        assert all(value != 0 for value in born)
        assert all(sector[0] != 0 for sector in laurent)
    return runs, values


@pytest.mark.loop_physics
def test_public_route_on_the_shipped_loop_sm_equals_the_tools_view(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """u u~ > e- e+ (a QED Born: the four-G0 vertex is in the request's
    scope) on ``model.source = "loop_sm"``: full colour by default, the
    shipped declaration, the repair recorded in the published compiled model
    and in the dry run, binary64 values bitwise equal to the tools view."""

    _compile("loop_sm")
    plan = _generator(
        _ufo_card("loop_sm", tmp_path / "plan", "u u~ > e- e+"), tmp_path / "plan.toml"
    ).plan(_processes("u u~ > e- e+"))
    assert not plan.unsupported_features, plan.unsupported_features
    coverage = plan.estimated_coverage
    (record,) = coverage["processes"]
    assert record["color_accuracy"] == "full"
    assert coverage["model_declaration"]["scheme"]["model_convention"] == (
        "loop_sm lhv=1 supplied QCD rules"
    )
    runs, values = _public_and_view(
        "loop_sm", loop_sm_compiled_model, "u u~ > e- e+", tmp_path
    )
    assert values["public"] == values["view"]
    assert runs["public"].inspect()["color_accuracy"] == "full"
    compiled = json.loads(
        (tmp_path / "public" / "model" / "compiled-model.json").read_text()
    )
    (repair,) = compiled["raw_loop_model"]["metadata"]["pyamplicol_source_repairs"]
    assert repair["vertex"][1] == "V_1001"


# The MQ-MODEL must-generate list (m-qcd PLAN §3.2): QED-Born requests the
# scalar-arity defect blocked on the public route.
MUST_GENERATE = (
    "u u~ > e- e+ g",
    "u d~ > e+ ve g",
    "u u~ > z z",
    "u u~ > w+ w-",
    "u d~ > w+ z",
    "u d~ > w+ h",
    "u d~ > t b~",
    "u b > d t",
    "u u~ > z h",
    "u d > u d h",
    "e+ e- > t t~",
)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("source", ["shipped", "mg5"])
@pytest.mark.parametrize("expression", MUST_GENERATE)
def test_public_route_generates_the_blocked_list_bitwise_equal_to_the_tools_view(
    loop_sm_ufo, loop_sm_compiled_model, loop_native, tmp_path, expression, source
):
    model = "loop_sm" if source == "shipped" else str(loop_sm_ufo)
    _runs, values = _public_and_view(
        model, loop_sm_compiled_model, expression, tmp_path
    )
    assert values["public"] == values["view"]


# ---------------------------------------------------------------- refusals


def _plan(model, expression, tmp_path, *, restriction=None, fixed=None):
    _compile(model, restriction)
    return _generator(
        _ufo_card(
            model, tmp_path / "plan", expression, restriction=restriction, fixed=fixed
        ),
        tmp_path / "plan.toml",
    ).plan(_processes(expression))


def _blocker(model, expression, tmp_path, *, restriction=None, fixed=None):
    plan = _plan(model, expression, tmp_path, restriction=restriction, fixed=fixed)
    (blocker,) = plan.unsupported_features
    return blocker, plan


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,expression,restriction,fixed",
    [
        ("loop-induced", "g g > h h", None, None),
        ("coloured-width", "g g > t t~", None, {"WZ": "0"}),
    ],
)
def test_loop_sm_refusals_fire_in_the_dry_run(
    tmp_path, key, expression, restriction, fixed
):
    """Each refusal a [virt=QCD] request on loop_sm reaches is the listed key
    in the dry run, which lists every key with its remedy."""

    from pyamplicol.loops.v2.refusals import listed_refusal_keys
    from pyamplicol.processes.refusals import listed_process_refusal_keys

    blocker, plan = _blocker(
        "loop_sm", expression, tmp_path, restriction=restriction, fixed=fixed
    )
    assert blocker.startswith(f"{key}: "), blocker
    listed = [
        item.split(":")[0]
        for item in plan.estimated_coverage["loop_admission_refusals"]
    ]
    assert listed == list(listed_refusal_keys())
    assert [
        item.split(":")[0]
        for item in plan.estimated_coverage["process_request_refusals"]
    ] == list(listed_process_refusal_keys())
    if key == "coloured-width":
        assert 'model.restriction = "no_widths"' in blocker
        assert 'WT = "0"' in blocker


@pytest.mark.loop_physics
def test_loop_sm_admits_dirac_weyl_closures_in_the_dry_run(tmp_path):
    """b b~ > w+ w- on no_b_mass, whose top t channel closes a Dirac current
    on a massless (Weyl) b: the loop Born admits the class (m-qcd MQ-5F,
    ``test_loop_v2_dirac_weyl.py``), so the dry run has no blocker, and the
    tree lanes' key ``dirac-weyl-closure`` is no loop admission refusal."""

    from pyamplicol.loops.v2.refusals import listed_refusal_keys

    plan = _plan(
        "loop_sm",
        "b b~ > w+ w-",
        tmp_path,
        restriction="no_b_mass",
        fixed={"WT": "0", "WZ": "0", "WW": "0", "WH": "0"},
    )
    assert not plan.unsupported_features, plan.unsupported_features
    assert "dirac-weyl-closure" not in listed_refusal_keys()


MADLOOP_SYNTAX = (
    "u u~ > d d~ [virt=QCD]",
    "u u~ > d d~ QED=0",
    "u u~ > d d~ QCD^2==4",
    "u u~ > d d~ / z",
    "u u~ > d d~ $ z",
    "g g > t t~, t > b w+",
    "u u~ > d d~ @1",
)


@pytest.mark.loop_fast
@pytest.mark.parametrize("expression", MADLOOP_SYNTAX)
def test_madloop_syntax_is_the_listed_refusal_naming_the_setting(expression):
    from pyamplicol.api import ProcessRequest
    from pyamplicol.processes.refusals import ProcessRequestUnsupported

    with pytest.raises(ProcessRequestUnsupported) as refused:
        ProcessRequest.parse(expression)
    assert refused.value.key == "madloop-syntax"
    assert isinstance(refused.value, ValueError)
    text = str(refused.value)
    assert "process.loop_coupling_orders" in text
    assert "process.max_coupling_orders" in text
    assert "generation.particle_veto" in text


@pytest.mark.loop_physics
def test_madloop_syntax_and_decays_refuse_in_the_cli_dry_run(tmp_path):
    """``pyamplicol generate --dry-run``: '[QCD]' (a bracket of no particles,
    found at the model's process expansion) is the plan's unsupported
    feature; a decay chain and a 1 -> n decay are refused when the request
    is parsed (exit 2). Each names its key."""

    from pyamplicol.cli import run_cli

    _compile("loop_sm")

    def dry_run(expression):
        card = tmp_path / "card.toml"
        card.write_text(_ufo_card("loop_sm", tmp_path / "out", expression))
        stdout, stderr = io.StringIO(), io.StringIO()
        code = run_cli(
            (
                "generate",
                "--card",
                str(card),
                "--dry-run",
                "--json",
                "--progress",
                "off",
            ),
            stdout=stdout,
            stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue()

    code, stdout, _ = dry_run("u u~ > d d~ [QCD]")
    assert code == 0
    (feature,) = json.loads(stdout)["unsupported_features"]
    assert feature.startswith("madloop-syntax: "), feature
    for expression, key in (
        ("t > b w+, w+ > e+ ve", "madloop-syntax"),
        ("t > b w+", "decay-process"),
    ):
        code, _, stderr = dry_run(expression)
        assert code == 2
        assert f"{key}: " in stderr, stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    "expression", ["t > b w+", "z > e+ e- g", "p > t b~ e+ ve", "2*g > g g g"]
)
def test_decays_are_the_listed_refusal_tree_and_loop_alike(expression):
    """One initial-state slot is a 1 -> n decay (``decay-process``), refused
    by the product-wide parser; two slots (``2*g``) are a collision."""

    from pyamplicol.api import ProcessRequest
    from pyamplicol.processes.refusals import ProcessRequestUnsupported

    if expression.startswith("2*"):
        assert ProcessRequest.parse(expression).expression == expression
        return
    with pytest.raises(ProcessRequestUnsupported) as refused:
        ProcessRequest.parse(expression)
    assert refused.value.key == "decay-process"


# ---------------------------------------------------------------- self-test


@pytest.mark.loop_physics
def test_self_test_evaluates_a_virtual_on_the_shipped_loop_sm(loop_native):
    """``pyamplicol self-test``'s virtual: generated through the public route
    on the shipped loop_sm (no checkout input), = MadLoop at its points."""

    from pyamplicol.diagnostics import _virtual_check, run_self_test

    check = _virtual_check()
    assert check.status == "pass", check.detail
    assert "MadLoop" in check.detail and "full colour" in check.detail
    assert "Symbolica" in check.detail
    assert "physics-virtual" in {item.name for item in run_self_test().checks}


@pytest.mark.loop_fast
def test_self_test_virtual_failure_names_the_reason_and_licence(monkeypatch):
    from pyamplicol import diagnostics
    from pyamplicol._internal import selftest_virtual

    def refuse():
        raise RuntimeError("generation refused")

    monkeypatch.setattr(selftest_virtual, "run_virtual_self_test", refuse)
    check = diagnostics._virtual_check()
    assert check.status == "fail"
    assert "generation refused" in check.detail and "Symbolica" in check.detail


@pytest.mark.loop_fast
@pytest.mark.parametrize("key", ["majorana-fermion", "multi-fermion-contact"])
def test_majorana_and_contact_vertices_are_listed_refusals(loop_sm_ufo, key):
    """The model preparation refuses a self-conjugate fermion and a vertex of
    more than two fermions with their listed keys (loop_sm doctored: the
    electron neutrino made self-conjugate; the u u~ G vertex given a d d~
    pair)."""

    from symbolica import is_licensed

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

    if not is_licensed():
        pytest.skip("source preparation requires licensed Symbolica")
    raw = load_raw_loop_model(loop_sm_ufo)
    payload = raw.to_dict()
    particles = {
        record["name"]: record for record in payload["collections"]["particles"]
    }
    if key == "majorana-fermion":
        particles["ve"]["fields"]["antiname"] = "ve"
    else:
        vertex = next(
            record
            for record in payload["collections"]["vertices"]
            if sorted(ref["name"] for ref in record["fields"]["particles"])
            == ["G", "u", "u~"]
        )

        def ref(name):
            record = particles[name]
            return dict(
                kind="reference", collection="particles", id=record["id"], name=name
            )

        vertex["fields"]["particles"] = [ref(n) for n in ("u~", "u", "d~", "d")]
    with pytest.raises(LoopRequestUnsupported) as refused:
        prepare_loop_request_model(
            RawLoopModel.from_dict(payload),
            born_order_bounds={"QCD": 2, "QED": 0},
            loop_order_bounds={"QCD": 4, "QED": 0},
            fixed_parameters={"MB": "0", "ymb": "0"},
        )
    assert refused.value.key == key


# ------------------------------------------------- MadLoop on the public route

# MadLoop 3.7.0 accurate-mode controls on loop_sm-no_b_mass (committed):
# Drell-Yan with the Z vetoed and e+ e- > t t~, QCD corrections, widths 0.
MADLOOP_ROWS = {
    "dy": ("drell-yan-control", "u u~ > e- e+", ["Z"]),
    "ee": ("ee-ttx-madloop-control", "e+ e- > t t~", []),
}


def _row_rel(mine, reference) -> float:
    from decimal import Decimal

    mine = Decimal(str(complex(mine).real)) if not isinstance(mine, Decimal) else mine
    reference = Decimal(repr(reference))
    scale = max(abs(mine), abs(reference))
    return 0.0 if scale == 0 else float(abs(mine - reference) / scale)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("dy", marks=pytest.mark.loop_physics),
        pytest.param("ee", marks=pytest.mark.loop_nightly),
    ],
)
def test_public_route_on_the_shipped_loop_sm_equals_madloop(
    loop_reference, loop_native, tmp_path, row
):
    """The public route on ``model.source = "loop_sm"`` (no [loop.model],
    default full colour) against MadLoop (m-qcd PLAN §1.4): double-double
    within 1e-12 of MadLoop (rule 1; MadLoop's own accuracy is below that at
    every point), binary64 within 1e-12 of our double-double (rule 2,
    well-conditioned 2 -> 2 points), Born and every Laurent coefficient."""

    from pyamplicol import Runtime

    name, expression, veto = MADLOOP_ROWS[row]
    reference = loop_reference(name)
    assert all(item["accuracy"] < 1e-11 for item in reference["results"])
    _compile("loop_sm", "no_b_mass")
    card = tmp_path / "card.toml"
    text = _ufo_card(
        "loop_sm", tmp_path / "artifact", expression, restriction="no_b_mass"
    ).replace(
        "[generation]\n",
        f"[generation]\nparticle_veto = {json.dumps(veto)}\n",
    )
    _generator(text, card).generate(_processes(expression), tmp_path / "artifact")
    runtime = Runtime.load(tmp_path / "artifact", process="p")
    assert runtime.inspect()["color_accuracy"] == "full"
    # The artifact records the model's spelling; any letter case selects it.
    recorded = runtime.inspect()["process"]
    assert (
        Runtime.load(tmp_path / "artifact", process=recorded.swapcase()).inspect()[
            "process_id"
        ]
        == "p"
    )
    points = reference["points"]
    scales = [1.0e6] * len(points)
    carriers = {
        arithmetic: runtime.evaluate(
            [[[str(c) for c in p] for p in point] for point in points]
            if arithmetic != "double"
            else [[[float(c) for c in p] for p in point] for point in points],
            mu_r2=scales,
            arithmetic=arithmetic,
        )
        for arithmetic in ("double", "double-double")
    }

    def coefficients(result, index):
        ((born,),) = (result.born_values[index],)
        ((laurent,),) = (result.values[index],)
        return (born, laurent.finite, laurent.single_pole, laurent.double_pole)

    for index, expected in enumerate(reference["results"]):
        dd = coefficients(carriers["double-double"], index)
        double = coefficients(carriers["double"], index)
        for k, key in enumerate(("born", "finite", "single_pole", "double_pole")):
            assert _row_rel(dd[k], expected[key]) <= 1e-12, (index, key)
            assert _row_rel(double[k], float(dd[k])) <= 1e-12, (index, key)


@pytest.mark.loop_fast
def test_self_test_virtual_is_a_named_warning_without_symbolica(monkeypatch):
    """The f64 deployment gate runs ``run_self_test`` with Symbolica's import
    blocked and needs ``report.ok``: the virtual check reports why it did not
    run and imports nothing of Symbolica."""

    import builtins

    from pyamplicol import diagnostics

    original = builtins.__import__

    def reject(name, *args, **kwargs):
        if name == "symbolica" or name.startswith("symbolica."):
            raise ModuleNotFoundError("blocked", name="symbolica")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject)
    check = diagnostics._virtual_check()
    assert check.status == "warning"
    assert "needs Symbolica" in check.detail


@pytest.mark.loop_fast
def test_process_selection_takes_any_letter_case_of_the_model_spelling(monkeypatch):
    """A loop artifact records the model's spelling (loop_sm's gluon is
    ``G``; a multiparticle request's subprocesses have no other name), and
    ``Runtime.load(..., process=...)`` follows the public spelling rule of
    requests: the exact spelling first, then a unique letter-case-insensitive
    match, as the tree artifacts' selector; a match that is not unique names
    the stable ids."""

    from types import SimpleNamespace

    from pyamplicol.api.errors import EvaluationError
    from pyamplicol.artifacts import process_groups
    from pyamplicol.loops.v2 import artifact

    monkeypatch.setattr(process_groups, "read_process_groups", lambda root: None)

    def manifest(*expressions):
        return SimpleNamespace(
            processes=tuple(
                {"id": f"p_{k}", "expression": expression}
                for k, expression in enumerate(expressions, 1)
            ),
            default_process_id=None,
            root=None,
        )

    found = manifest("u u~ > t t~", "G G > t t~")
    assert artifact._select(found, "g  g > t t~")[0]["id"] == "p_2"
    assert artifact._select(found, "G G > T T~")[0]["id"] == "p_2"
    assert artifact._select(found, "p_1")[0]["id"] == "p_1"
    with pytest.raises(EvaluationError, match="has no process 'g g > b b~'"):
        artifact._select(found, "g g > b b~")
    twins = manifest("a a > t t~", "A A > t t~")
    assert artifact._select(twins, "A A > t t~")[0]["id"] == "p_2"
    with pytest.raises(EvaluationError, match=r"ambiguous.* p_1, p_2"):
        artifact._select(twins, "a A > t t~")


@pytest.mark.loop_nightly
def test_multiparticle_subprocesses_are_madloops_and_their_own_generations(
    loop_native, tmp_path
):
    """Campaign row M01 on the public route: ``p p > t t~`` on
    loop_sm-no_b_mass is one artifact holding MadLoop's subprocesses
    (P0_uux_ttx: the five q q~, the b included; P2_gg_ttx: g g), and each
    subprocess, loaded by its expression in MadLoop's spelling, is bitwise its
    own single-process generation (binary64, two points)."""

    from pyamplicol import Runtime

    multi = tmp_path / "multi"
    _generator(
        _ufo_card("loop_sm", multi, "p p > t t~", restriction="no_b_mass"),
        tmp_path / "multi.toml",
    ).generate(_processes("p p > t t~"), multi)
    stored = json.loads((multi / "artifact.json").read_text())["processes"]
    assert {
        (tuple(sorted(p["external_pdgs"][:2])), tuple(sorted(p["external_pdgs"][2:])))
        for p in stored
    } == {((-q, q), (-6, 6)) for q in range(1, 6)} | {((21, 21), (-6, 6))}
    masses_in, masses_out = _masses(_compile("loop_sm", "no_b_mass"), "g g > t t~")
    points = [_point(masses_in, masses_out, seed) for seed in (11, 12)]
    for name, expression in (("uux", "u u~ > t t~"), ("gg", "g g > t t~")):
        single = tmp_path / name
        _generator(
            _ufo_card("loop_sm", single, expression, restriction="no_b_mass"),
            tmp_path / f"{name}.toml",
        ).generate(_processes(expression), single)
        found = [
            _values(runtime.evaluate(points, mu_r2=[8315.0] * len(points)))
            for runtime in (
                Runtime.load(multi, process=expression),
                Runtime.load(single, process="p"),
            )
        ]
        assert found[0] == found[1], expression
        assert all(sector[0] != 0 for _, laurent in found[0] for sector in laurent)
