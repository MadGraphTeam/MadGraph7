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
    """Each shipped file equals the reference copy's (loop_qcd_qed_sm: the
    download; loop_qcd_qed_sm_Gmu: its Python-3 conversion, whose restriction
    cards are the download's), and a bare name selects the shipped model."""

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
    converted = sorted(
        file
        for file, entry in records.items()
        if entry["transformation_id"] == "ufo-python3-conversion-v1"
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
