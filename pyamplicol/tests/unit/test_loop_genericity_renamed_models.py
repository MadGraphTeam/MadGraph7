# SPDX-License-Identifier: 0BSD
"""Production decisions do not depend on how a model spells its content.

The renamed-model harness of the 2026-10-25 genericity audit (REPORT.md §3;
``loop_renamed_models``): UFO copies of MadLoop's loop_sm and of
loop_qcd_qed_sm_Gmu with

* ``orders``: coupling orders renamed (QCD -> QS, QED -> Qe) and an unused
  mixed-case order NPprop (M3);
* ``particles``: particles renamed (gluon -> gl, G+/G-/G0 -> Gp/Gm/Gz,
  ghosts), MW/WW -> MWB/WWB and the Z, G0, G+ PDG codes renumbered to
  9000023, 9000250, 9000251 (m5, m6);
* ``scale``: MU_R -> muR (same LOOP code), aS moved off SMINPUTS 3;
* ``mass``: MW internal (G_mu) or external (loop_sm) with the same value.

Each test generates the same request on the original and on the variant,
renamed through the variant's map, and compares the decisions (dry-run
route, orders, corrections, scale binding, EW classification, refusals,
certificates, repairs) and the values (binary64, within 1e-12 of the
largest coefficient). A decision another package owns is a strict xfail
naming the owner and slice-status DECISIONS 2026-10-26 ~00:20: it passes
once the owner's fix lands, and then fails here until the marker goes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from .loop_renamed_models import RenameMap, write_variant

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-12
GMU = "gmu_particle_anti"
FIXED = {"MB": "0"}
PARAMETERS = {"aS": "0.118", "MT": "173"}
ORIGINAL = RenameMap("original")
DY_POINT = [
    [500, 0, 0, 500],
    [500, 0, 0, -500],
    [500, 300, 0, 400],
    [500, -300, 0, -400],
]
OWNED = "slice-status DECISIONS 2026-10-26 ~00:20"
M1_M2_OWNER = (
    f"M1/M2 (native point inputs of derived masses): generation-scale-2, {OWNED} item 1"
)
M2_ALPHA_OWNER = f"m2 (one alpha_s rule): generation-scale-2, {OWNED} item 1"
M1_EW_OWNER = (
    f"m1 (corrections named 'QCD'/'QED'; Catani): WP3.2 ew-poles batch 2, {OWNED} "
    "item 3"
)


def _uux_ddxg_point():
    fixture = ROOT / "tests/fixtures/loop/legacy/uux_ddxg.json"
    return json.loads(fixture.read_text())["points"][0]


@pytest.fixture(scope="module")
def models(loop_sm_ufo, loop_gmu_ufo, tmp_path_factory):
    """``models(name, variant)``: (compiled model, rename map); ``variant``
    None is the original. loop_sm takes the structural scalar-arity view."""

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    sources = {"loop_sm": Path(loop_sm_ufo), "gmu": Path(loop_gmu_ufo)}
    root = tmp_path_factory.mktemp("renamed-models")
    built: dict = {}

    def build(name: str, variant: str | None = None):
        key = (name, variant)
        if key not in built:
            if variant is None:
                source, rename = sources[name], ORIGINAL
            else:
                source = root / f"{name}-{variant}"
                rename = write_variant(sources[name], source, variant)
            compiled = compile_model_source(
                source, cache_dir=root / "cache", require_supported=False
            )
            if name == "loop_sm":
                compiled = replace(
                    compiled,
                    raw_loop_model=prepare_loop_sm_scalar_declarations(
                        compiled.raw_loop_model
                    ).derived,
                    _serialized_path=None,
                    _prepared_bundle=None,
                )
            built[key] = compiled, rename
        return built[key]

    return build


def _request(rename, process, born, loop, corrections, **options):
    from pyamplicol.loops.v2.assemble import SliceRequest

    options["fixed_parameters"] = {
        rename.parameter(name): value
        for name, value in options.get("fixed_parameters", FIXED).items()
    }
    return SliceRequest(
        rename.process(process),
        rename.orders_of(born),
        rename.orders_of(loop),
        corrections=tuple(rename.order(name) for name in corrections),
        **options,
    )


def _slice(models, loop_native, name, variant, *request, **options):
    from pyamplicol.loops.v2.assemble import LoopSlice

    compiled, rename = models(name, variant)
    if name == "gmu":
        # G_mu keeps its scale parameter at LOOP 666 (the public route binds
        # the only LOOP entry, ``test_renamed_scale_parameter_is_bound``).
        options.setdefault("compatibility", GMU)
        options.setdefault("scale_parameter", rename.parameter("MU_R"))
    return LoopSlice.generate(
        compiled, _request(rename, *request, **options), loop_native
    ), rename


def _evaluate(slice_, point, rename):
    parameters = {rename.parameter(k): v for k, v in PARAMETERS.items()}
    return slice_.evaluate(point, 1e6, parameters=parameters)


def _values(result, rename):
    """Born and virtual of ``result`` keyed by the original order names."""

    back = {new: old for old, new in rename.orders.items()}

    def key(orders):
        return tuple(sorted((back.get(name, name), power) for name, power in orders))

    born = {key(k): float(v) for k, v in result.born.items()}
    virtual = {
        key(k): tuple(float(c) for c in laurent.coefficients)
        for k, laurent in result.components["virtual"].items()
    }
    return born, virtual


def _assert_same_values(original, variant, rename=ORIGINAL):
    born, virtual = _values(original, ORIGINAL)
    born_v, virtual_v = _values(variant, rename)
    assert set(born_v) == set(born) and set(virtual_v) == set(virtual)
    scale = max(abs(x) for x in born.values())
    assert all(abs(born_v[k] - v) <= TOLERANCE * scale for k, v in born.items())
    scale = max(abs(x) for row in virtual.values() for x in row)
    for key, row in virtual.items():
        assert all(
            abs(a - b) <= TOLERANCE * scale
            for a, b in zip(row, virtual_v[key], strict=True)
        )


# ------------------------------------------------------- public dry run


def _card(
    model,
    output,
    process,
    rename,
    *,
    engine="v2",
    process_extra="",
    compatibility=None,
    extra="",
):
    qcd = rename.order("QCD")
    preparation = 'fixed_parameters = {MB = "0"}'
    if compatibility is not None:
        preparation += f'\ncompatibility = "{compatibility}"'
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression={json.dumps(rename.process(process))}, name="p"}}]
loop_coupling_orders = ["{qcd.lower()}"]
coupling_order_policy = "minimal"
{process_extra}
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
engine = "{engine}"
[loop.preparation]
{preparation}
[loop.model]
loop_coupling_orders = ["{qcd}"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "the source UFO's supplied QCD rules"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
{extra}"""


def _plan_entry(models, name, variant, process, tmp_path, **options):
    """The single process entry of the public dry run (``Generator.plan``)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    compiled, rename = models(name, variant)
    directory = tmp_path / f"{name}-{variant}"
    directory.mkdir()
    model = compiled.write(directory / "model.json")
    card = directory / "card.toml"
    if name == "gmu":
        options.setdefault("compatibility", GMU)
    card.write_text(_card(model, directory / "out", process, rename, **options))
    plan = Generator(resolve_config(card)).plan(
        ProcessSet((ProcessRequest.parse(rename.process(process), name="p"),))
    )
    assert plan.unsupported_features == ()
    (entry,) = plan.estimated_coverage["processes"]
    return dict(entry), rename


def _renamed_orders(orders, rename):
    out = {rename.order(k): v for k, v in dict(orders).items()}
    return {**out, **dict.fromkeys(rename.extra_orders, 0)}


@pytest.mark.parametrize("engine", ["v2", "legacy"])
def test_renamed_orders_take_the_same_dry_run(models, engine, tmp_path):
    """u u~ > d d~ [QS] (the configuration names the correction in lower
    case): on both engines the dry run equals the original's under the map,
    the unused NPprop bounded at zero."""

    original, _ = _plan_entry(
        models, "loop_sm", None, "u u~ > d d~", tmp_path, engine=engine
    )
    renamed, rename = _plan_entry(
        models, "loop_sm", "orders", "u u~ > d d~", tmp_path, engine=engine
    )
    if engine == "legacy":
        for key in ("born_graph_count", "loop_graph_count", "denominator_family_count"):
            assert renamed[key] == original[key] > 0
        return
    assert dict(renamed["born_orders"]) == _renamed_orders(
        original["born_orders"], rename
    )
    assert dict(renamed["loop_orders"]) == _renamed_orders(
        original["loop_orders"], rename
    )
    assert renamed["corrections"] == ("QS",)
    assert (renamed["engine"], renamed["scale_parameter"]) == (
        original["engine"],
        original["scale_parameter"],
    )


def test_renamed_orders_give_the_same_legacy_virtual(models, tmp_path):
    """u u~ > d d~ [QS] on the legacy engine: public generation (with its
    post-build validation) and runtime values equal the original's."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    values = []
    for variant in (None, "orders"):
        compiled, rename = models("loop_sm", variant)
        directory = tmp_path / str(variant)
        directory.mkdir()
        model = compiled.write(directory / "model.json")
        card = directory / "card.toml"
        card.write_text(
            _card(
                model,
                directory / "out",
                "u u~ > d d~",
                rename,
                engine="legacy",
                extra="[generation.validation]\npost_build_validation = true\n"
                "samples = 1\n",
            )
        )
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse("u u~ > d d~", name="p"),)),
            directory / "out",
        )
        runtime = Runtime.load(directory / "out", process="p")
        ((laurent,),) = runtime.evaluate([DY_POINT], mu_r2="1000000").values
        values.append((laurent.finite, laurent.single_pole, laurent.double_pole))
    scale = max(abs(x) for x in values[0])
    assert all(abs(a - b) <= TOLERANCE * scale for a, b in zip(*values, strict=True))


def test_explicit_limits_resolve_in_any_case(models, tmp_path):
    extra = "[process.max_coupling_orders]\nqs = 2\nQE = 0\nnpprop = 0"
    entry, _ = _plan_entry(
        models, "loop_sm", "orders", "u u~ > d d~", tmp_path, process_extra=extra
    )
    assert dict(entry["born_orders"]) == {"NPprop": 0, "QS": 2, "Qe": 0}


@pytest.mark.parametrize("name", ["loop_sm", "gmu"])
def test_renamed_scale_parameter_is_bound(models, name):
    """MU_R -> muR at (LOOP, 1) on loop_sm, at the only LOOP entry (666) on
    G_mu: the public route's scale binding (``loop_service._v2_scale_parameter``)
    takes the renamed parameter."""

    from pyamplicol.generation.loop_service import _v2_scale_parameter

    bound = [
        _v2_scale_parameter(models(name, variant)[0], FIXED)
        for variant in (None, "scale")
    ]
    assert bound == ["MU_R", "muR"]


def test_order_names_resolve_once_in_any_case():
    from pyamplicol.models.order_names import (
        CouplingOrderNameError,
        CouplingOrderNames,
    )

    names = CouplingOrderNames.of(("QS", "Qe", "NPprop"))
    assert names.resolve("qe") == names.resolve("QE") == "Qe"
    assert names.resolve_all(("npprop", "QS")) == ("NPprop", "QS")
    assert names.orders((("NPPROP", 1), ("QE", 2), ("QS", 1))) == (
        ("NPprop", 1),
        ("QS", 1),
        ("Qe", 2),
    )
    assert names.mapping({"QE": 0, "QS": 2}) == {"QS": 2, "Qe": 0}
    assert names.declared_name("CT_INSERTION") == "CT_INSERTION"
    with pytest.raises(CouplingOrderNameError, match="not a coupling order"):
        names.resolve("QCD")
    with pytest.raises(CouplingOrderNameError, match="differ only in case"):
        CouplingOrderNames.of(("QCD", "qcd"))
    shipped = CouplingOrderNames.of(("QCD", "QED"))
    orders = (("QCD", 2),)
    assert shipped.identity and shipped.orders(orders) is orders


def test_orders_one_key_apart_are_refused_at_load(loop_sm_ufo, tmp_path):
    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.order_names import CouplingOrderNameError

    source = tmp_path / "loop_sm-qs"
    write_variant(Path(loop_sm_ufo), source, "orders")
    path = source / "coupling_orders.py"
    path.write_text(
        path.read_text(encoding="latin-1").replace("'NPprop'", "'qs'"),
        encoding="latin-1",
    )
    with pytest.raises(CouplingOrderNameError, match="'QS' and 'qs'"):
        compile_model_source(source, cache_dir=tmp_path / "cache", use_cache=False)


# ------------------------------------------------------------ values


@pytest.mark.parametrize(
    ("process", "born", "loop"),
    [
        ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}),
        ("u u~ > e+ e-", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}),
    ],
    ids=["qcd-born", "ew-born"],
)
def test_renamed_orders_give_the_same_virtual(models, loop_native, process, born, loop):
    """QCD corrections to a QCD Born and to an electroweak Born (the
    mixed-case Qe at a nonzero power through the tree blocks and roots)."""

    point = _uux_ddxg_point() if process.endswith("g") else DY_POINT
    request = (process, born, loop, ("QCD",))
    fixed = {**FIXED, "WZ": "0"}
    results = []
    for variant in (None, "orders"):
        slice_, rename = _slice(
            models, loop_native, "loop_sm", variant, *request, fixed_parameters=fixed
        )
        results.append(_evaluate(slice_, point, rename))
    _assert_same_values(*results, rename)


@pytest.mark.parametrize(
    "name",
    ["loop_sm", pytest.param("gmu", marks=pytest.mark.loop_nightly)],
)
def test_renamed_particles_give_the_same_virtual(models, loop_native, name):
    """u u~ > e+ e- with QCD corrections: the Born's Z exchange under PDG
    9000023, the gluon, Goldstones and ghosts renamed, MW -> MWB."""

    request = ("u u~ > e+ e-", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}, ("QCD",))
    fixed = {**FIXED, "WZ": "0"}
    results = [
        _evaluate(
            _slice(
                models, loop_native, name, variant, *request, fixed_parameters=fixed
            )[0],
            DY_POINT,
            ORIGINAL,
        )
        for variant in (None, "particles")
    ]
    _assert_same_values(*results)


def test_renamed_scale_and_moved_alpha_s_give_the_same_virtual(models, loop_native):
    """u u~ > d d~ g, QCD, alpha_s named explicitly: the slice binds muR and
    the virtual equals the original's."""

    request = ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, ("QCD",))
    results = []
    for variant in (None, "scale"):
        slice_, rename = _slice(
            models, loop_native, "loop_sm", variant, *request, alpha_s_parameter="aS"
        )
        assert slice_.scale_parameter == rename.parameter("MU_R")
        results.append(_evaluate(slice_, _uux_ddxg_point(), rename))
    _assert_same_values(*results)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=M1_EW_OWNER)
def test_renamed_orders_keep_the_ew_classification_and_catani_poles(
    models, loop_native
):
    from pyamplicol.loops.v2 import ew_policy
    from pyamplicol.loops.v2.assemble import prepare_request_model

    request = ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, ("QCD",))
    decisions = []
    for variant in (None, "orders"):
        compiled, rename = models("loop_sm", variant)
        slice_, _ = _slice(models, loop_native, "loop_sm", variant, *request)
        prepared = prepare_request_model(compiled, _request(rename, *request))
        result = _evaluate(slice_, _uux_ddxg_point(), rename)
        decisions.append(
            (
                ew_policy.electroweak(
                    compiled,
                    _request(rename, *request),
                    prepared.prepared.structure.perturbative_species_orders,
                ),
                prepared.alpha_s_parameter,
                result.ir_poles is not None,
            )
        )
    assert decisions[1] == decisions[0] == (False, "aS", True)


@pytest.mark.xfail(strict=True, raises=ValueError, reason=M2_ALPHA_OWNER)
def test_alpha_s_off_sminputs_does_not_refuse_a_qcd_request(models, loop_native):
    from pyamplicol.loops.v2.assemble import prepare_request_model

    compiled, rename = models("loop_sm", "scale")
    request = ("u u~ > d d~", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0}, ("QCD",))
    prepare_request_model(compiled, _request(rename, *request))


# ---------------------------------------------------- particles, PDGs (m5, m6)


def _structure(source: Path):
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(source, restriction="full")
    return raw, build_covariant_structure(raw, compatibility=GMU)


def test_renamed_goldstones_are_repaired_identically(loop_gmu_ufo, tmp_path):
    """m6: the G_mu UFO's Particle.anti loses the Goldstone flag of G-;
    renamed (Gp/Gm), renumbered (9000251) and with MWB/WWB it is repaired
    alike; with the flag already right the profile is refused."""

    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.models.loop_import import RawLoopModel

    # A UFO directory is imported as a package of its name: not "particles".
    variant = tmp_path / "gmu_particles_variant"
    write_variant(Path(loop_gmu_ufo), variant, "particles")
    _, original = _structure(Path(loop_gmu_ufo))
    raw, renamed = _structure(variant)
    flags = {
        p["name"]: p["goldstoneboson"]
        for p in json.loads(renamed.native.to_json())["particles"]
    }
    assert flags["Gm"] is flags["Gp"] is True
    (diagnostic,) = original.diagnostics
    (repaired,) = renamed.diagnostics
    assert dict(repaired) == {**dict(diagnostic), "particle": "Gm"}
    payload = raw.to_dict()
    for record in payload["collections"]["particles"]:
        if record["name"] == "Gm":
            record["fields"]["goldstoneboson"] = True
    with pytest.raises(ValueError, match="recognized Gmu anti metadata mismatch"):
        build_covariant_structure(RawLoopModel.from_dict(payload), compatibility=GMU)


def test_renumbered_singlets_receive_the_same_certificates(loop_gmu_ufo, tmp_path):
    """m5: the counterterm bases of the renumbered Z (9000023) and G0
    (9000250) get the certificates of the original's, entry by entry."""

    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases

    variant = tmp_path / "gmu_particles_variant"
    rename = write_variant(Path(loop_gmu_ufo), variant, "particles")
    certificates = []
    for source in (Path(loop_gmu_ufo), variant):
        _, structure = _structure(source)
        ir = build_loop_model(structure.raw, scalar_program=structure.scalars)
        certificates.append(
            {
                v.source_key: (tuple(v.particle_pdgs), v.conformance_control)
                for v in certified_source_counterterm_bases(ir)
            }
        )
    original, renamed = certificates
    assert renamed == {
        key: (tuple(rename.pdg(p) for p in pdgs), control)
        for key, (pdgs, control) in original.items()
    }
    assert any(9000023 in pdgs for pdgs, _ in renamed.values())


# ------------------------------------------------------------ mass nature


def _w_point(mw: float) -> list[list[float]]:
    from pyamplicol.generation.phase_space import massive_rambo_final_state

    final = massive_rambo_final_state(2, sqrt_s=1000.0, masses=(mw, 0.0), seed=3)
    return [[500.0, 0.0, 0.0, 500.0], [500.0, 0.0, 0.0, -500.0], *map(list, final)]


W_REQUEST = ("u d~ > w+ g", {"QCD": 1, "QED": 1}, {"QCD": 3, "QED": 1}, ("QCD",))


def _route(slice_) -> str:
    from pyamplicol.loops.v2.plan_export import export_plan

    try:
        export_plan(slice_, forward=False, trees=True, born_squared=True)
    except NotImplementedError:
        return "plan-only"
    return "native-v2"


@pytest.fixture(scope="module")
def w_slices(models, loop_native):
    built = {}

    def get(name, variant):
        if (name, variant) not in built:
            built[name, variant] = _slice(
                models, loop_native, name, variant, *W_REQUEST
            )[0]
        return built[name, variant]

    return get


# MW of loop_sm is derived (MW external in its variant), that of G_mu a card
# input (an alias of one in its variant); the G_mu pair is the slower one.
MW_MODELS = ["loop_sm", pytest.param("gmu", marks=pytest.mark.loop_nightly)]


@pytest.mark.parametrize("name", MW_MODELS)
def test_mw_nature_gives_the_same_virtual(w_slices, models, name):
    """u d~ > w+ g, QCD: the same values to rounding whether MW is a card
    input or derived (its value the same to rounding)."""

    from pyamplicol.models.external import CompiledUFOModel

    mw = CompiledUFOModel(models(name, None)[0]).mass(24)
    point = _w_point(float(mw))
    _assert_same_values(
        _evaluate(w_slices(name, None), point, ORIGINAL),
        _evaluate(w_slices(name, "mass"), point, ORIGINAL),
    )


# loop_sm with MW a card input reaches the native export, whose SymJIT plane
# check (rusticol-core loop_v2/trees.rs ``verify_plane``) probes the parameter
# program at artificial inputs in (0.5, 1.5): with MW > MZ there the W
# coupling's (1 - MW^2/MZ^2)^(-1/2) sits on its branch cut, where the plane
# and the exact evaluator take opposite branches (a sign), so the export is
# refused; which input gets which probe value depends on the parameter names.
PLANE_CHECK_OWNER = (
    "SymJIT plane check at non-physical probe inputs (branch cut of "
    "(1 - MW^2/MZ^2)^(-1/2)): native tree export (native_trees.py / trees.rs "
    f"verify_plane) owner, reported by genericity-fixes; and {M1_M2_OWNER}"
)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "loop_sm",
            marks=pytest.mark.xfail(
                strict=True,
                raises=(AssertionError, ValueError),
                reason=PLANE_CHECK_OWNER,
            ),
        ),
        pytest.param(
            "gmu",
            marks=[
                pytest.mark.loop_nightly,
                pytest.mark.xfail(
                    strict=True, raises=AssertionError, reason=M1_M2_OWNER
                ),
            ],
        ),
    ],
)
def test_mw_nature_takes_the_same_native_route(w_slices, name):
    assert _route(w_slices(name, "mass")) == _route(w_slices(name, None))
