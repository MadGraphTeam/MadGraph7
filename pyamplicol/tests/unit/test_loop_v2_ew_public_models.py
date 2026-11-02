# SPDX-License-Identifier: 0BSD
"""The QCD+QED loop models on the public route (m-sm PLAN MS-MODEL).

* The Goldstone-flag source repair: the legacy UFO object library's
  ``Particle.anti`` loses the Goldstone flag of an antiparticle (MG5_aMC's
  loop_qcd_qed_sm and loop_qcd_qed_sm_Gmu: G-); the production source repair
  (``covariant_model.repair_source_declarations``, rule ``GOLDSTONE_FLAG``)
  restores it by structure, records it and is idempotent, so the public route
  needs no ``compatibility = "gmu_particle_anti"`` (still accepted). loop_sm
  has no such defect and keeps its declarations digest.
* The wheel ships both models (m-sm PLAN Q5 (a)): loop_qcd_qed_sm as
  downloaded, loop_qcd_qed_sm_Gmu as the Python-3 conversion of its Python-2
  download (``assets/models/PROVENANCE.toml``); a bare name selects each.
* Their ``[loop.model]`` declarations are shipped profiles keyed by the
  declarations digest after the repair (one digest for the shipped copy, the
  reference copy and every restriction), covering QCD and QED jointly; the
  scheme differs from loop_sm's, so the shipped N_c colour data stays
  loop_sm's alone.
* A Python-2 UFO is the listed refusal ``ufo-python2``
  (``models.source_refusals``), at model loading and in the dry run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

GOLDSTONE_ANTIPARTICLE = "G-"
LOOP_SM_DIGEST = "e2512823936595ee4848974146d20b1290de6ecd7cf48c5d03bb47e1466b1a77"
EW_MODELS = {
    # shipped name -> the conftest fixture of the reference copy
    "loop_qcd_qed_sm": "loop_qcd_qed_sm_ufo",
    "loop_qcd_qed_sm_Gmu": "loop_gmu_ufo",
}


def _shipped(name: str) -> Path:
    from pyamplicol._internal.packaged_models import packaged_ufo_model

    path = packaged_ufo_model(name)
    assert path is not None, name
    return path


# ------------------------------------------------------------ source repair


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(EW_MODELS))
def test_goldstone_flag_repair_is_structural_recorded_and_idempotent(name):
    """The repair restores exactly the lost flag (G-), records it, leaves
    the input unchanged and is a no-op on its result; the repaired source
    builds the covariant structure of the explicit profile, which stays
    accepted on both."""

    from pyamplicol.loops.covariant_model import (
        GOLDSTONE_FLAG,
        SOURCE_REPAIRS_KEY,
        build_covariant_structure,
        repair_source_declarations,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(_shipped(name), restriction="full")
    original = raw.to_dict()
    assert (
        raw.record("particles", GOLDSTONE_ANTIPARTICLE)["fields"]["goldstoneboson"]
        is False
    )
    with pytest.raises(ValueError, match="antiparticle metadata mismatch for G"):
        build_covariant_structure(raw)
    repaired = repair_source_declarations(raw)
    assert raw.to_dict() == original
    (record,) = repaired.metadata[SOURCE_REPAIRS_KEY]
    assert record["rule"] == GOLDSTONE_FLAG
    assert record["particle"][1] == GOLDSTONE_ANTIPARTICLE
    assert (record["field"], record["original"], record["derived"]) == (
        "goldstoneboson",
        False,
        True,
    )
    assert (
        repaired.record("particles", GOLDSTONE_ANTIPARTICLE)["fields"]["goldstoneboson"]
        is True
    )
    assert repair_source_declarations(repaired) is repaired
    changed = {
        key
        for key, collection in original["collections"].items()
        if collection != repaired.to_dict()["collections"][key]
    }
    assert changed == {"particles"}
    structure = build_covariant_structure(repaired)
    profile = build_covariant_structure(raw, compatibility="gmu_particle_anti")
    again = build_covariant_structure(repaired, compatibility="gmu_particle_anti")
    assert structure.native.to_json() == profile.native.to_json()
    assert again.native.to_json() == profile.native.to_json()
    assert [item["rule"] for item in structure.diagnostics] == [GOLDSTONE_FLAG]


@pytest.mark.loop_fast
def test_loop_sm_has_no_goldstone_flag_defect(loop_sm_ufo):
    from pyamplicol.loops.covariant_model import (
        GOLDSTONE_FLAG,
        SOURCE_REPAIRS_KEY,
        repair_source_declarations,
    )
    from pyamplicol.loops.shipped_profiles import source_declarations_digest
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(loop_sm_ufo)
    rules = [
        item["rule"]
        for item in repair_source_declarations(raw).metadata[SOURCE_REPAIRS_KEY]
    ]
    assert GOLDSTONE_FLAG not in rules
    assert source_declarations_digest(raw) == LOOP_SM_DIGEST


# ------------------------------------------------------------------ assets


def _ufo_files(directory: Path) -> list[str]:
    """The files of a UFO directory a model consists of: its modules and its
    restriction cards (no caches, backups, AppleDouble or hidden files)."""

    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and (
            (path.suffix == ".py" and path.stem.isidentifier())
            or (path.name.startswith("restrict_") and path.suffix == ".dat")
        )
    )


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(EW_MODELS))
def test_shipped_qcd_qed_model_is_the_reference_and_selected_by_name(
    request, tmp_path, name
):
    """Each shipped file equals the reference copy's (UFOmodeldb's with the
    issue-1 model patch; loop_qcd_qed_sm_Gmu: its Python-3 conversion, whose
    restriction cards are the download's but the patched with_b_mass), and a
    bare name selects the shipped model."""

    import tomllib

    from pyamplicol._internal.packaged_models import packaged_ufo_models
    from pyamplicol.api.requests import ModelSource
    from pyamplicol.config import resolve_config

    reference = Path(request.getfixturevalue(EW_MODELS[name]))
    shipped = _shipped(name)
    assert name in packaged_ufo_models()
    names = _ufo_files(shipped)
    assert names == _ufo_files(reference)
    assert names == sorted(path.name for path in shipped.iterdir() if path.is_file())
    for file in names:
        assert (shipped / file).read_bytes() == (reference / file).read_bytes(), file
    provenance = tomllib.loads(
        (shipped.parents[1] / "PROVENANCE.toml").read_text(encoding="utf-8")
    )
    records = {
        entry["canonical_package_path"].rsplit("/", 1)[1]: entry
        for entry in provenance["files"]
        if f"/ufo/{name}/" in entry["canonical_package_path"]
    }
    assert sorted(records) == names
    # The G_mu modules take the Python-3 conversion, alone or after the
    # UFOmodeldb issue-1 model patch (whose charge appends are folded).
    converted = sorted(
        file
        for file, entry in records.items()
        if entry["transformation_id"]
        in (
            "ufo-python3-conversion-v1",
            "ufomodeldb-issue-1-patch-python3-v1",
            "ufomodeldb-issue-1-patch-python3-fold-v1",
        )
    )
    expected = (
        [file for file in names if file.endswith(".py")]
        if name.endswith("_Gmu")
        else []
    )
    assert converted == expected
    assert ModelSource.from_path(name).path == shipped
    card = tmp_path / "card.toml"
    card.write_text(f'action = "generate"\n[model]\nsource = "{name}"\n')
    assert Path(resolve_config(card).effective.model.source) == shipped


# ---------------------------------------------------------------- profiles


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", sorted(EW_MODELS))
def test_shipped_profile_declares_the_qcd_qed_model_by_its_declarations(request, name):
    """One digest for the shipped copy, the reference copy and every
    restriction; the profile declares joint QCD and QED corrections (and
    covers each alone); its scheme is not loop_sm's, so no shipped N_c colour
    data attaches; a modified copy gets no profile."""

    from pyamplicol.loops.shipped_profiles import (
        LOOP_SM_SCHEME,
        shipped_loop_profile,
        source_declarations_digest,
    )
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )
    from pyamplicol.models.loop_declaration import resolve_loop_model_declaration
    from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

    restrictions = ["default", "with_b_mass"]
    if name.endswith("_Gmu"):
        restrictions.append("a0")
    digests = set()
    for source in (_shipped(name), Path(request.getfixturevalue(EW_MODELS[name]))):
        for restriction in restrictions:
            raw = load_raw_loop_model(source, restriction=restriction)
            digests.add(source_declarations_digest(raw))
            profile = shipped_loop_profile(raw)
            assert profile is not None, (source, restriction)
            for corrections in (("QCD", "QED"), ("QED",), ("QCD",)):
                chosen = resolve_loop_model_declaration(
                    raw, selected_corrections=corrections
                )
                assert chosen == profile.declaration
            assert chosen.loop_coupling_orders == ("QCD", "QED")
            assert chosen.uv_complete and chosen.model_r2_complete
            assert chosen.scheme != LOOP_SM_SCHEME
            assert name in chosen.scheme.model_convention
    assert len(digests) == 1
    assert LOOP_SM_DIGEST not in digests
    assert resolve_loop_color_declaration(raw) is None
    payload = raw.to_dict()
    payload["collections"]["couplings"][0]["fields"]["expression"] = "0"
    assert shipped_loop_profile(RawLoopModel.from_dict(payload)) is None


# ---------------------------------------------------------------- python 2


def _python2_copy(tmp_path: Path, label: str, edit) -> Path:
    """A copy of the shipped G_mu model (as a UFO named ``label``) with one
    module edited back to Python 2."""

    target = tmp_path / label
    shutil.copytree(_shipped("loop_qcd_qed_sm_Gmu"), target)
    edit(target)
    return target


def _print_statement(target: Path) -> None:
    path = target / "write_param_card.py"
    path.write_text(path.read_text() + "\nif False:\n    print 'param_card'\n")


def _iteritems(target: Path) -> None:
    path = target / "object_library.py"
    text = path.read_text()
    assert "self.__dict__.items()" in text
    path.write_text(text.replace("self.__dict__.items()", "self.__dict__.iteritems()"))


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("edit", "evidence"),
    [
        (_print_statement, "write_param_card.py is not Python 3 source"),
        (_iteritems, "Python-2 name 'iteritems'"),
    ],
)
def test_python2_ufo_is_the_listed_refusal(tmp_path, edit, evidence):
    """A Python-2 UFO (a module that is not Python-3 source, or an import
    failing on a Python-2-only name) is refused with ``ufo-python2`` by the
    raw loop loader and the model compiler, naming the evidence, the remedy
    and the shipped models; another import failure keeps its own error."""

    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.loop_import import load_raw_loop_model
    from pyamplicol.models.source_refusals import (
        ModelSourceUnsupported,
        listed_model_source_refusal_keys,
    )

    assert "ufo-python2" in listed_model_source_refusal_keys()
    source = _python2_copy(tmp_path, "py2_model", edit)
    for load in (
        lambda: load_raw_loop_model(source),
        lambda: compile_model_source(
            source, cache_dir=tmp_path / "cache", use_cache=False
        ),
    ):
        with pytest.raises(ModelSourceUnsupported) as caught:
            load()
        assert caught.value.key == "ufo-python2"
        message = str(caught.value)
        assert message.startswith("ufo-python2: the UFO model 'py2_model'")
        assert evidence in message
        assert "convert model" in message and "loop_qcd_qed_sm_Gmu" in message

    def broken(target: Path) -> None:
        path = target / "particles.py"
        path.write_text(path.read_text() + "\nraise RuntimeError('not python 2')\n")

    other = _python2_copy(tmp_path, "broken_model", broken)
    with pytest.raises(RuntimeError, match="not python 2"):
        load_raw_loop_model(other)


@pytest.mark.loop_fast
def test_python2_ufo_refuses_in_the_dry_run(tmp_path):
    """The dry run, which loads no model it has not compiled, finds a module
    of the UFO that is not Python-3 source without importing it: the plan's
    unsupported feature (``Generator.plan``, ``pyamplicol generate
    --dry-run``) is the listed refusal."""

    import io
    import json

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.cli import run_cli
    from pyamplicol.config import resolve_config

    source = _python2_copy(tmp_path, "py2_model", _print_statement)
    card = tmp_path / "card.toml"
    card.write_text(
        f"""
action = "generate"
[model]
source = "{source}"
cache_dir = "{tmp_path / "cache"}"
[process]
entries = [{{expression = "u u~ > d d~", name = "p"}}]
loop_coupling_orders = ["QED"]
[generation]
output = "{tmp_path / "out"}"
"""
    )
    plan = Generator(resolve_config(card)).plan(
        ProcessSet((ProcessRequest.parse("u u~ > d d~", name="p"),))
    )
    (feature,) = plan.unsupported_features
    assert feature.startswith("ufo-python2: the UFO model 'py2_model'"), feature
    stdout = io.StringIO()
    code = run_cli(
        ("generate", "--card", str(card), "--dry-run", "--json", "--progress", "off"),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    (feature,) = json.loads(stdout.getvalue())["unsupported_features"]
    assert feature.startswith("ufo-python2: "), feature
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- tree API


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", sorted(EW_MODELS))
def test_qcd_qed_tree_models_are_supported(tmp_path, name):
    """The NLO UFO scalar helpers are registered (``models.loading.
    SUPPORTED_FUNCTION_ARITIES``): the tree model of each shipped QCD+QED
    loop model compiles as supported, so Born and real-emission trees come
    from the same UFO as the virtual."""

    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        _shipped(name), cache_dir=tmp_path / "cache", require_supported=True
    )
    assert compiled.supported
    assert compiled.issues == ()
    assert {"arg", "crecms", "grreglog", "recms", "regsqrt"} <= set(
        compiled.capabilities["declared_functions"]
    )


# The tree check of the MadLoop Borns (m-sm PLAN MS-MODEL item 6): the
# required EW fixtures whose Born has no photon (no alpha(0) convention):
# fixture -> shipped model.
TREE_ROWS = {
    "emep_emep_ew": "loop_qcd_qed_sm",
    "emep_ttxh_ew": "loop_qcd_qed_sm_Gmu",
    "gg_ttx_ew": "loop_qcd_qed_sm_Gmu",
    "uux_emepg_ew": "loop_qcd_qed_sm_Gmu",
    "gg_ttxemep_ew": "loop_qcd_qed_sm_Gmu",
}


@pytest.mark.loop_nightly
def test_tree_matrix_elements_on_the_qcd_qed_models_equal_madloops_borns(tmp_path):
    """Five tree MEs (2 -> 2 to 2 -> 4, both models) on the shipped models'
    prepared kernel packs (``ModelSource.compile``, JIT), full colour, at the
    fixtures' points and parameters (the restriction's own zeros kept), equal
    MadLoop's Born to 1e-13 (measured <= 2.3e-15)."""

    import json

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ModelSource, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.config.models import EvaluatorConfig

    root = Path(__file__).resolve().parents[1] / "fixtures/loop/required"
    prepared = {}
    for name in sorted(set(TREE_ROWS.values())):
        output = tmp_path / f"{name}.pyamplicol-model"
        ModelSource.from_path(name).compile(
            cache_dir=tmp_path / "cache",
            prepared_output=output,
            evaluator=EvaluatorConfig(),
        )
        prepared[name] = output
    for row, name in TREE_ROWS.items():
        fixture = json.loads((root / f"{row}.json").read_text())
        expression = " ".join(
            token
            for token in fixture["process"].split(" [")[0].split()
            if "=" not in token
        )
        orders = ", ".join(f"{k} = {v}" for k, v in fixture["born_orders"].items())
        output = tmp_path / row
        card = tmp_path / f"{row}.toml"
        card.write_text(
            f"""
action = "generate"
[model]
source = "{prepared[name]}"
[process]
entries = [{{expression = "{expression}", name = "p"}}]
coupling_order_policy = "explicit"
max_coupling_orders = {{{orders}}}
[color]
accuracy = "full"
[generation]
output = "{output}"
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""
        )
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse(expression, name="p"),)), output
        )
        runtime = Runtime.load(output, process="p")
        declared = {p.name: p for p in runtime.physics.model_parameters}
        values = {**fixture["parameters"], **fixture["fixed_parameters"]}
        for key, value in values.items():
            if key in declared and not declared[key].mutable:
                # a restriction's own zero (MB, ymb, MTA, ymtau): as the card
                assert declared[key].default_real == float(value), (row, key)
        runtime.set_model_parameters(
            {
                key: float(value)
                for key, value in values.items()
                if key in declared and declared[key].mutable
            }
        )
        for point, reference in zip(fixture["points"], fixture["rows"], strict=True):
            momenta = [[float(x) for x in momentum] for momentum in point]
            (value,) = runtime.evaluate([momenta])
            born = reference["born"]
            assert abs(complex(value).real - born) <= 1e-13 * abs(born), (row, value)


# ------------------------------------------------------------ public route

ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
# The declaration the electroweak tests state in their cards (the tools
# view); the public route states none.
_DECLARED = """
[loop.model]
loop_coupling_orders = {corrections}
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO on-shell electroweak"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_qcd_qed_sm lhv=1 supplied QED rules"
"""


def _ew_card(source, output, expression, corrections, *, restriction, fixed, view):
    """A one-loop card on ``source``: the public route (``view`` False: no
    ``[loop.model]``, no compatibility profile, no colour setting) or the
    tools view (the explicit declaration and ``gmu_particle_anti``)."""

    import json

    parameters = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    restricted = (
        "" if restriction is None else f"restriction = {json.dumps(restriction)}"
    )
    declared = _DECLARED.format(corrections=json.dumps(corrections)) if view else ""
    compatibility = 'compatibility = "gmu_particle_anti"' if view else ""
    return f"""
action = "generate"
[model]
source = {json.dumps(str(source))}
{restricted}
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = {json.dumps(corrections)}
coupling_order_policy = "minimal"
[loop]
r2_strategy = "model"
{declared}
[loop.preparation]
fixed_parameters = {{{parameters}}}
{compatibility}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _generator(text: str, path: Path):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    path.write_text(text)
    return Generator(resolve_config(path))


def _processes(expression: str):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet((ProcessRequest.parse(expression, name="p"),))


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "restriction", "expression"),
    [
        ("loop_qcd_qed_sm", None, "u u~ > d d~"),
        ("loop_qcd_qed_sm_Gmu", "a0", "u u~ > a g"),
    ],
)
def test_public_route_dry_run_takes_the_shipped_profile(
    tmp_path, name, restriction, expression
):
    """``model.source = <name>`` with no ``[loop.model]`` and no
    compatibility profile: the dry run admits a joint QCD+QED request with
    the shipped profile's declaration and the repair recorded."""

    from pyamplicol.api.requests import ModelSource
    from pyamplicol.loops.shipped_profiles import shipped_loop_profile
    from pyamplicol.models.loop_import import load_raw_loop_model

    ModelSource.from_path(name, restriction=restriction or "default").compile()
    plan = _generator(
        _ew_card(
            name,
            tmp_path / "out",
            expression,
            ["QCD", "QED"],
            restriction=restriction,
            fixed=ZERO_WIDTHS,
            view=False,
        ),
        tmp_path / "card.toml",
    ).plan(_processes(expression))
    assert plan.unsupported_features == ()
    profile = shipped_loop_profile(load_raw_loop_model(_shipped(name)))
    coverage = plan.estimated_coverage
    assert coverage["model_declaration"] == profile.declaration.to_dict()
    (record,) = coverage["processes"]
    assert record["color_accuracy"] == "full"
    assert not (tmp_path / "out").exists()


def _masses(model, expression: str) -> tuple[list[float], list[float]]:
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


def _point(masses_in, masses_out, seed, sqrt_s=1000.0):
    import math

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


def _values(result):
    return [
        (
            [complex(value) for value in born],
            [[None if c is None else complex(c) for c in laurent] for laurent in point],
        )
        for born, point in zip(result.born_values, result.values, strict=True)
    ]


# The MS-MODEL acceptance list (m-sm PLAN §3.2): (shipped model,
# restriction, process, corrections, extra fixed parameters, the MadLoop
# fixture the public route is also checked against: a required fixture or an
# MS-REF one under ``msm/``, at its parameters).
_MT173 = {"MT": "173", "ymt": "173"}
PUBLIC_ROUTE = {
    "ee_mumu": ("loop_qcd_qed_sm_Gmu", None, "e- e+ > m- m+", ["QED"], {}, None),
    "uu_ee": (
        "loop_qcd_qed_sm_Gmu",
        None,
        "u u~ > e- e+",
        ["QCD", "QED"],
        _MT173,
        "msm/uux_emep_ew",
    ),
    "uu_dd": ("loop_qcd_qed_sm", None, "u u~ > d d~", ["QCD", "QED"], {}, None),
    "gg_tt": (
        "loop_qcd_qed_sm_Gmu",
        None,
        "g g > t t~",
        ["QCD", "QED"],
        _MT173,
        "gg_ttx_ew",
    ),
    "ud_wg": (
        "loop_qcd_qed_sm_Gmu",
        None,
        "u d~ > w+ g",
        ["QCD", "QED"],
        _MT173,
        "msm/udx_wpg_ew",
    ),
    "uu_zz": (
        "loop_qcd_qed_sm_Gmu",
        None,
        "u u~ > z z",
        ["QED"],
        _MT173,
        "msm/uux_zz_ew",
    ),
    "uu_ag": ("loop_qcd_qed_sm_Gmu", "a0", "u u~ > a g", ["QCD", "QED"], {}, None),
}


def _madloop_sectors(reference, row):
    """MadLoop's Born and Laurent coefficients of a fixture row by squared
    order (``canonical_order``): an MS-REF fixture's per-sector record
    (``sectors``: every split order), or a required fixture's one sector
    (its Born and loop amplitude orders)."""

    from pyamplicol.loops.orders import canonical_order

    def order(record):
        return canonical_order(
            tuple(
                (name, power)
                for name, power in record.items()
                if isinstance(power, int) and not isinstance(power, bool)
            )
        )

    if "sectors" in row:
        sectors = row["sectors"]
        return (
            {order(record): record["born"] for record in sectors["born"]},
            {
                order(record): (
                    record["finite"],
                    record["single_pole"],
                    record["double_pole"],
                )
                for record in sectors["loop"]
            },
        )
    born, loop = reference["born_orders"], reference["loop_orders"]
    madloop = row["providers"]["madloop"]
    return (
        {order({name: 2 * power for name, power in born.items()}): madloop["born"]},
        {
            order(
                {name: born.get(name, 0) + loop.get(name, 0) for name in {*born, *loop}}
            ): (madloop["finite"], madloop["single_pole"], madloop["double_pole"])
        },
    )


# u u~ > z z [virt=QED] on G_mu was refused at load ("a trace step above the
# family rank") until the plan export split such trace steps by source storage
# (``plan_export._trace_groups``).
@pytest.mark.loop_nightly
@pytest.mark.parametrize("case", sorted(PUBLIC_ROUTE))
def test_public_route_on_the_shipped_qcd_qed_models_equals_the_tools_view(
    request, loop_native, tmp_path, case
):
    """The public route (``model.source = <name>``: the shipped UFO, its
    shipped profile, the automatic Goldstone-flag repair, no ``tools/``
    import) generates each request; its binary64 Born and virtual at two
    points are bitwise those of the tools view (the reference UFO with the
    explicit declaration and ``compatibility = "gmu_particle_anti"``). With a
    MadLoop fixture (generated at its parameters, asserted), the public
    route's Born and every loop sector of the request equal MadLoop's
    quadruple-precision values at the fixture's points (its metric:
    relative, Laurent floor |Born| alpha/2pi; 1e-8)."""

    import json

    from pyamplicol import Runtime
    from pyamplicol.models.loading import compile_model_source

    name, restriction, expression, corrections, extra, fixture = PUBLIC_ROUTE[case]
    fixed = {**ZERO_WIDTHS, **extra}
    view_source = request.getfixturevalue(EW_MODELS[name])
    runs = {}
    for route, source, view in (
        ("public", name, False),
        ("view", view_source, True),
    ):
        output = tmp_path / route
        _generator(
            _ew_card(
                source,
                output,
                expression,
                corrections,
                restriction=restriction,
                fixed=fixed,
                view=view,
            ),
            tmp_path / f"{route}.toml",
        ).generate(_processes(expression), output)
        runs[route] = Runtime.load(output, process="p")
    compiled = compile_model_source(
        _shipped(name), restriction=restriction or "default"
    )
    masses_in, masses_out = _masses(compiled, expression)
    points = [_point(masses_in, masses_out, seed) for seed in (11, 12)]
    values = {
        route: _values(runtime.evaluate(points, mu_r2=[8315.0] * len(points)))
        for route, runtime in runs.items()
    }
    for born, laurent in values["public"]:
        assert all(value != 0 for value in born)
        assert all(sector[0] != 0 for sector in laurent)
    assert values["public"] == values["view"]
    if fixture is None:
        return
    reference = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / f"fixtures/loop/required/{fixture}.json"
        ).read_text()
    )
    # The generation's parameters are the fixture's (a restriction's own
    # zeros, e.g. MB and ymtau, are absent from the card).
    defaults = compiled.parameter_defaults
    for key, value in {
        **reference["parameters"],
        **reference["fixed_parameters"],
    }.items():
        have = fixed[key] if key in fixed else defaults.get(key, (0.0, 0.0))[0]
        assert float(have) == float(value), (fixture, key, have, value)
    alpha = float(reference["derived_parameters"]["alpha"])
    points = [[[float(x) for x in p] for p in point] for point in reference["points"]]
    scales = [float(mu) ** 2 for mu in reference["mu_r"]]
    result = runs["public"].evaluate(points, mu_r2=scales)
    sectors = [sector.order for sector in result.sectors]
    for k, row in enumerate(reference["rows"]):
        borns, loops = _madloop_sectors(reference, row)
        if "sectors" in row:
            # An MS-REF fixture has every split order of the request.
            assert set(result.born_orders) <= set(borns), (fixture, borns)
            assert set(sectors) <= set(loops), (fixture, sectors, loops)
        floor = abs(row["born"]) * alpha / (2 * 3.141592653589793)
        compared = []
        for b, order in enumerate(result.born_orders):
            if order in borns:
                ours = complex(result.born_values[k][b]).real
                theirs = borns[order]
                error = abs(ours - theirs) / max(abs(ours), abs(theirs))
                assert error <= 1e-8, (fixture, k, order, "born", ours, theirs)
                compared.append(order)
        for v, order in enumerate(sectors):
            if order not in loops:
                continue
            for i, key in enumerate(("finite", "single_pole", "double_pole")):
                ours = complex(result.values[k][v][i]).real
                theirs = loops[order][i]
                error = abs(ours - theirs) / max(abs(ours), abs(theirs), floor)
                assert error <= 1e-8, (fixture, k, order, key, ours, theirs)
            compared.append(order)
        assert len(compared) >= 2, (fixture, compared)


# ---------------------------------------------------------------- self-test


@pytest.mark.loop_physics
def test_self_test_evaluates_the_prepared_ew_virtual(loop_native):
    """``pyamplicol self-test`` loads the shipped prepared EW virtual (no
    generation) and its Born and Laurent coefficients equal MadLoop's."""

    from pyamplicol._internal.selftest_virtual_ew import RELATIVE_TOLERANCE
    from pyamplicol.diagnostics import _prepared_virtual_check, run_self_test

    check = _prepared_virtual_check()
    assert check.status == "pass", check.detail
    assert "loop_qcd_qed_sm" in check.detail and "3 points" in check.detail
    assert RELATIVE_TOLERANCE <= 1e-9
    assert "physics-virtual-prepared" in {item.name for item in run_self_test().checks}


@pytest.mark.loop_fast
def test_prepared_self_test_is_a_named_warning_without_symbolica(monkeypatch):
    import builtins

    from pyamplicol import diagnostics

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "symbolica" or name.startswith("symbolica."):
            raise ModuleNotFoundError("blocked", name="symbolica")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    check = diagnostics._prepared_virtual_check()
    assert check.status == "warning"
    assert "Symbolica" in check.detail


@pytest.mark.loop_nightly
def test_prepared_ew_self_test_artifact_is_the_public_route_output(
    loop_native, tmp_path
):
    """Regenerating the shipped artifact through the public route
    (``generate_ew_self_test_artifact``) gives bitwise its values: the shipped
    copy is current, also in its conventions and in the "uv" selection
    (MS-POLES: the counterterms' poles split by loop_qcd_qed_sm's pole
    regulators, which an artifact written before records none of)."""

    import json

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_ew as selftest

    root = selftest.fixture_root()
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    output = tmp_path / "artifact"
    selftest.generate_ew_self_test_artifact(output, tmp_path / "cache")
    assert selftest.evaluate(output, expected) == selftest.evaluate(
        root / "artifact", expected
    )
    points = [[list(map(float, p)) for p in point] for point in expected["points"]]
    scales = [float(x) for x in expected["mu_r2"]]
    runs = []
    for artifact in (output, root / "artifact"):
        runtime = Runtime.load(artifact, process=selftest.NAME, mute_warnings=True)
        result = runtime.evaluate(points, mu_r2=scales, poles="uv")
        runs.append(
            (
                runtime.inspect()["conventions"],
                result.values,
                result.pole_part("uv", "uv"),
                result.pole_part("ir", "uv"),
            )
        )
    assert runs[0][0]["pole_split"] == "regulators"
    assert runs[0] == runs[1]
