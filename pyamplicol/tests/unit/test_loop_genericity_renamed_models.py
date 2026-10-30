# SPDX-License-Identifier: 0BSD
"""Production decisions do not depend on how a model spells its content.

The renamed-model harness of the 2026-10-25 genericity audit (REPORT.md §3;
``loop_renamed_models``): UFO copies of MadLoop's loop_sm and of
loop_qcd_qed_sm_Gmu with

* ``orders``: coupling orders renamed (QCD -> QS, QED -> Qe) and an unused
  mixed-case order NPprop (M3);
* ``lower``: coupling orders spelt in lower case (qcd, qed; M3);
* ``particles``: particles renamed (gluon -> gl, G+/G-/G0 -> Gp/Gm/Gz,
  ghosts), MW/WW -> MWB/WWB and the Z, G0, G+ PDG codes renumbered to
  9000023, 9000250, 9000251 (m5, m6);
* ``scale``: MU_R -> muR (same LOOP code), aS moved off SMINPUTS 3;
* ``mass``: MW internal (G_mu) or external (loop_sm) with the same value.

The orders variant is also written of loop_qcd_qed_sm, for a renamed
electroweak correction ('Qe'). Each test generates the same request on the
original and on the variant, renamed through the variant's map, and compares
the decisions (dry-run route, orders, corrections, scale binding, EW
classification, refusals, certificates, repairs) and the values (binary64,
every component within 1e-12 of the largest coefficient). A decision another
package owns is a strict xfail naming the owner and its slice-status DECISIONS
entry: it passes once the owner's fix lands, and then fails here until the
marker goes.
"""

from __future__ import annotations

import json
import re
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


def _uux_ddxg_point():
    fixture = ROOT / "tests/fixtures/loop/legacy/uux_ddxg.json"
    return json.loads(fixture.read_text())["points"][0]


# The UFO fixture of each model name (``qcd_qed``: MG5_aMC's loop_qcd_qed_sm,
# resolved only by the tests that use it).
SOURCES = {
    "loop_sm": "loop_sm_ufo",
    "gmu": "loop_gmu_ufo",
    "qcd_qed": "loop_qcd_qed_sm_ufo",
}


@pytest.fixture(scope="module")
def models(request, tmp_path_factory):
    """``models(name, variant)``: (compiled model, rename map); ``variant``
    None is the original. loop_sm takes the structural scalar-arity view."""

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("renamed-models")
    built: dict = {}

    def build(name: str, variant: str | None = None):
        key = (name, variant)
        if key not in built:
            source = Path(request.getfixturevalue(SOURCES[name]))
            if variant is None:
                rename = ORIGINAL
            else:
                rename = write_variant(source, root / f"{name}-{variant}", variant)
                source = root / f"{name}-{variant}"
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
    if name != "loop_sm":
        # G_mu and loop_qcd_qed_sm keep their scale parameter at LOOP 666 (the
        # public route binds the only LOOP entry,
        # ``test_renamed_scale_parameter_is_bound``).
        options.setdefault("compatibility", GMU)
        options.setdefault("scale_parameter", rename.parameter("MU_R"))
    return LoopSlice.generate(
        compiled, _request(rename, *request, **options), loop_native
    ), rename


def _evaluate(slice_, point, rename):
    parameters = {rename.parameter(k): v for k, v in PARAMETERS.items()}
    return slice_.evaluate(point, 1e6, parameters=parameters)


def _values(result, rename):
    """Born and every component (virtual, bare, uv, r2, ...) of ``result``,
    keyed by the original order names."""

    back = {new: old for old, new in rename.orders.items()}

    def key(orders):
        return tuple(sorted((back.get(name, name), power) for name, power in orders))

    born = {key(k): float(v) for k, v in result.born.items()}
    components = {
        (name, key(k)): tuple(float(c) for c in laurent.coefficients)
        for name, sectors in result.components.items()
        for k, laurent in sectors.items()
    }
    return born, components


def _assert_same_values(original, variant, rename=ORIGINAL):
    born, components = _values(original, ORIGINAL)
    born_v, components_v = _values(variant, rename)
    assert set(born_v) == set(born) and set(components_v) == set(components)
    assert any(name == "virtual" for name, _ in components)
    scale = max(abs(x) for x in born.values())
    assert all(abs(born_v[k] - v) <= TOLERANCE * scale for k, v in born.items())
    scale = max(abs(x) for row in components.values() for x in row)
    for key, row in components.items():
        assert all(
            abs(a - b) <= TOLERANCE * scale
            for a, b in zip(row, components_v[key], strict=True)
        ), key


def _route(slice_) -> str:
    """The artifact writer's decision on the native export of the plan,
    trees and Born squared: ``native-v2``, or the listed refusal's key
    (``native_trees.NATIVE_EXPORT_REFUSALS``; there is no plan-only route,
    and any other error propagates)."""

    from pyamplicol.loops.v2.native_trees import NativeExportUnsupported
    from pyamplicol.loops.v2.plan_export import export_plan

    try:
        export_plan(slice_, forward=False, trees=True, born_squared=True)
    except NativeExportUnsupported as error:
        return f"refused: {error.feature}"
    return "native-v2"


def _frames(slice_, rename):
    """Every family's schedule-B frame, keyed by the family key in the
    original's mass and width parameter names."""

    from pyamplicol.loops.v2.families import MassId

    back = {new: old for old, new in rename.parameters.items()}

    def mass(m):
        return MassId(back.get(m.mass, m.mass), back.get(m.width, m.width))

    return {
        tuple((mask, mass(m)) for mask, m in key): schedule.trie.frame
        for key, schedule in slice_.plan.schedules.items()
    }


def _electroweak(slice_) -> bool:
    from pyamplicol.loops.v2 import ew_policy

    return ew_policy.electroweak(
        slice_.compiled,
        slice_.request,
        slice_.prepared.structure.perturbative_species_orders,
    )


def _assert_same_decisions(original, variant, rename, *, electroweak=True):
    """The production decisions of REPORT.md §3 on ``variant`` equal the
    original's under ``rename``: the native route or refusal, the frames,
    the scale binding and (``electroweak``) the EW classification."""

    assert _route(variant) == _route(original)
    frames = _frames(original, ORIGINAL)
    assert frames and _frames(variant, rename) == frames
    assert variant.scale_parameter == rename.parameter(original.scale_parameter)
    if electroweak:
        assert _electroweak(variant) is _electroweak(original)


# ------------------------------------------------------- public dry run


def _card(
    model,
    output,
    process,
    rename,
    *,
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


def test_renamed_orders_take_the_same_dry_run(models, tmp_path):
    """u u~ > d d~ [QS] (the configuration names the correction in lower
    case): the dry run equals the original's under the map, the unused
    NPprop bounded at zero."""

    original, _ = _plan_entry(models, "loop_sm", None, "u u~ > d d~", tmp_path)
    renamed, rename = _plan_entry(models, "loop_sm", "orders", "u u~ > d d~", tmp_path)
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


def test_renamed_orders_give_the_same_public_virtual(models, tmp_path):
    """u u~ > d d~ [QS] through the public route: generation (with its
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


def test_lower_case_orders_give_the_same_public_v2_virtual(models, tmp_path):
    """u u~ > d d~ [qcd] on the v2 engine through the public route
    (``Generator.generate`` with its staged validation, ``Runtime.load``) on
    loop_sm with every order spelt in lower case and the limits given in
    upper case: the original's values, the Born orders in the model's
    spelling."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    results = []
    for variant in (None, "lower"):
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
                process_extra="[process.max_coupling_orders]\nQCD = 2\nQED = 0",
            )
        )
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse("u u~ > d d~", name="p"),)),
            directory / "out",
        )
        runtime = Runtime.load(directory / "out", process="p")
        results.append(runtime.evaluate([DY_POINT], mu_r2="1000000"))
    original, lower = results
    assert lower.born_orders == tuple(
        tuple((rename.order(name), power) for name, power in orders)
        for orders in original.born_orders
    )
    (((laurent,),), ((laurent_v,),)) = original.values, lower.values
    values = [
        (row.finite, row.single_pole, row.double_pole) for row in (laurent, laurent_v)
    ]
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


# ------------------------------------------------------------ values


@pytest.mark.parametrize(
    ("name", "process", "born", "loop"),
    [
        ("loop_sm", "u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}),
        ("loop_sm", "u u~ > e+ e-", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}),
        pytest.param(
            "gmu",
            "u u~ > e+ e-",
            {"QCD": 0, "QED": 2},
            {"QCD": 2, "QED": 2},
            marks=pytest.mark.loop_nightly,
        ),
    ],
    ids=["qcd-born", "ew-born", "gmu-ew-born"],
)
def test_renamed_orders_give_the_same_virtual(
    models, loop_native, name, process, born, loop
):
    """QCD corrections to a QCD Born and to an electroweak Born (the
    mixed-case Qe at a nonzero power through the tree blocks and roots), on
    loop_sm and on G_mu: the same route, frames, scale binding and EW
    classification (the strong order found by the gluon's vertices, m1),
    and the same values."""

    point = _uux_ddxg_point() if process.endswith("g") else DY_POINT
    request = (process, born, loop, ("QCD",))
    fixed = {**FIXED, "WZ": "0"}
    slices, results = [], []
    for variant in (None, "orders"):
        slice_, rename = _slice(
            models, loop_native, name, variant, *request, fixed_parameters=fixed
        )
        slices.append(slice_)
        results.append(_evaluate(slice_, point, rename))
    _assert_same_decisions(*slices, rename)
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
    slices = [
        _slice(models, loop_native, name, variant, *request, fixed_parameters=fixed)
        for variant in (None, "particles")
    ]
    _assert_same_decisions(slices[0][0], *slices[1])
    _assert_same_values(
        *(_evaluate(slice_, DY_POINT, ORIGINAL) for slice_, _ in slices)
    )


def test_renamed_scale_and_moved_alpha_s_give_the_same_virtual(models, loop_native):
    """u u~ > d d~ g, QCD, alpha_s named explicitly: the slice binds muR and
    the virtual equals the original's."""

    request = ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, ("QCD",))
    slices, results = [], []
    for variant in (None, "scale"):
        slice_, rename = _slice(
            models, loop_native, "loop_sm", variant, *request, alpha_s_parameter="aS"
        )
        assert slice_.scale_parameter == rename.parameter("MU_R")
        slices.append(slice_)
        results.append(_evaluate(slice_, _uux_ddxg_point(), rename))
    _assert_same_decisions(*slices, rename)
    _assert_same_values(*results)


# Identical-flavour lines at leading and next-to-leading colour (genericity
# audit 2, G2-M1 acceptance (iii)): the blocks of one colour word reached
# with several U(1)-subtraction grades are split by grade
# (``generation.dag_grades``) whatever the quark is called or numbered and
# wherever its legs stand. Leg i of the permuted request is leg
# PERMUTED_LEGS[i] of u u~ > u u~ g: the loop requests close at an antiquark
# instead of a quark, and the gluon comes first in the final state.
UUX_UUXG = ("u u~ > u u~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, ("QCD",))
PERMUTED_UUX_UUXG = "u~ u > g u~ u"
PERMUTED_LEGS = (1, 0, 4, 3, 2)


def _mixed_grade_words(slice_) -> int:
    """How many (word, particle) pairs of the slice's loop-side blocks come
    in several N_c grades (separate blocks after the split)."""

    grades: dict = {}
    for block in slice_.blocks.blocks:
        if block.key.auxiliary_kind is None:
            grades.setdefault((block.key.word, block.key.particle_id), set()).add(
                block.colour_flow.nc_grade
            )
    return sum(len(found) > 1 for found in grades.values())


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_renamed_quarks_and_permuted_legs_give_the_same_projections(
    models, loop_native, accuracy
):
    """LC and NLC of u u~ > u u~ g (two identical-flavour lines): with the
    quark renamed and renumbered (``quarks``: uq, PDG 9000002) the slice
    splits the same blocks by grade and gives the same decisions and values,
    and so does the leg order u~ u > g u~ u at the permuted point (binary64,
    every component within 1e-12 of the largest coefficient)."""

    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    options = {
        "accuracy": accuracy,
        "colour_declaration": LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        ),
        "colour_contraction": "direct",
    }
    fixture = ROOT / "tests/fixtures/loop/legacy/uux_uuxg.json"
    point = json.loads(fixture.read_text())["points"][0]
    original, _ = _slice(models, loop_native, "loop_sm", None, *UUX_UUXG, **options)
    renamed, rename = _slice(
        models, loop_native, "loop_sm", "quarks", *UUX_UUXG, **options
    )
    permuted, _ = _slice(
        models,
        loop_native,
        "loop_sm",
        None,
        PERMUTED_UUX_UUXG,
        *UUX_UUXG[1:],
        **options,
    )
    split = _mixed_grade_words(original)
    assert split > 0
    assert _mixed_grade_words(renamed) == split
    assert _mixed_grade_words(permuted) > 0
    _assert_same_decisions(original, renamed, rename)
    reference = _evaluate(original, point, ORIGINAL)
    _assert_same_values(reference, _evaluate(renamed, point, rename), rename)
    _assert_same_values(
        reference, _evaluate(permuted, [point[i] for i in PERMUTED_LEGS], ORIGINAL)
    )


# Widths zero outside the complex-mass scheme (``test_loop_v2_ew`` LADDER
# rung uudd_qed).
ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
MASSLESS_POINT = [
    ["500", "0", "0", "500"],
    ["500", "0", "0", "-500"],
    ["500", "240", "320", "300"],
    ["500", "-240", "-320", "-300"],
]


@pytest.mark.loop_nightly
def test_renamed_correction_order_gives_the_same_virtual(models, loop_native):
    """u u~ > d d~ with QED corrections named 'Qe' on loop_qcd_qed_sm (UV
    pole selection, every component): the same EW classification, route,
    frames and scale binding, and the same values. (Photon-only corrections,
    the Z, W, H and Goldstones vetoed, are refused on this model since WP3.2
    EWI-5: its electroweak counterterms do not label their loop content,
    ``refusals`` "veto-unlabelled-counterterms".)"""

    request = ("u u~ > d d~", {"QCD": 2, "QED": 0}, {"QCD": 2, "QED": 2}, ("QED",))
    slices, results = [], []
    for variant in (None, "orders"):
        slice_, rename = _slice(
            models,
            loop_native,
            "qcd_qed",
            variant,
            *request,
            fixed_parameters=ZERO_WIDTHS,
        )
        assert slice_.request.corrections == (rename.order("QED"),)
        slices.append(slice_)
        results.append(
            slice_.evaluate(
                MASSLESS_POINT,
                "1000000",
                parameters={"aS": "0.118", "MT": "173"},
                poles="uv",
            )
        )
    assert _electroweak(slices[0]) is True
    _assert_same_decisions(*slices, rename)
    _assert_same_values(*results, rename)


@pytest.mark.parametrize(
    ("process", "born", "loop"),
    [
        # A unitary-gauge loop line (WP3.2 blocker B3).
        ("u u~ > e+ e-", {"QCD": 0, "QED": 2}, {"QCD": 0, "QED": 4}),
        # Two colour-singlet fermion lines (WP3.2 blocker B4, EWI-5).
        ("e- e+ > m- m+", {"QCD": 0, "QED": 2}, {"QCD": 0, "QED": 4}),
    ],
    ids=["unitary-gauge-line", "two-singlet-lines"],
)
def test_renamed_orders_refuse_alike(models, loop_native, process, born, loop):
    """QED corrections ('Qe') on loop_sm: the outcome is the original's, a
    refusal of the same type and message (or, once its blocker is lifted, a
    generation on both)."""

    outcomes = []
    for variant in (None, "orders"):
        try:
            _slice(
                models,
                loop_native,
                "loop_sm",
                variant,
                process,
                born,
                loop,
                ("QED",),
                fixed_parameters={**FIXED, **ZERO_WIDTHS},
            )
        except Exception as error:  # the outcome is compared
            outcomes.append((type(error).__name__, str(error)))
        else:
            outcomes.append(None)
    _, rename = models("loop_sm", "orders")
    if outcomes[1] is not None:
        # A message naming an order names it in the variant's spelling.
        name, message = outcomes[1]
        for old, new in rename.orders.items():
            message = re.sub(rf"\b{re.escape(new)}\b", old, message)
        outcomes[1] = name, message
    assert outcomes[1] == outcomes[0]


def test_renamed_orders_keep_the_ew_classification_and_catani_poles(
    models, loop_native
):
    """m1 (WP3.2 ew-poles): the strong correction is the order the gluon's
    vertices carry, whatever it is called ('QS'): the request stays QCD-only
    (not electroweak), alpha_s is resolved and Catani's operator predicts
    the same poles under the map."""

    from pyamplicol.loops.v2 import ew_policy
    from pyamplicol.loops.v2.assemble import prepare_request_model

    request = ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}, ("QCD",))
    decisions, poles = [], []
    for variant in (None, "orders"):
        compiled, rename = models("loop_sm", variant)
        slice_, _ = _slice(models, loop_native, "loop_sm", variant, *request)
        prepared = prepare_request_model(compiled, _request(rename, *request))
        result = _evaluate(slice_, _uux_ddxg_point(), rename)
        back = {new: old for old, new in rename.orders.items()}
        poles.append(
            {
                tuple(sorted((back.get(n, n), p) for n, p in sector)): tuple(
                    float(x) for x in values
                )
                for sector, values in (result.ir_poles or {}).items()
            }
        )
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
    assert set(poles[1]) == set(poles[0])
    scale = max(abs(x) for row in poles[0].values() for x in row)
    for sector, row in poles[0].items():
        assert all(
            abs(a - b) <= TOLERANCE * scale
            for a, b in zip(row, poles[1][sector], strict=True)
        ), sector


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


# ------------------------------------------------------------ mass nature


def _w_point(mw: float) -> list[list[float]]:
    from pyamplicol.generation.phase_space import massive_rambo_final_state

    final = massive_rambo_final_state(2, sqrt_s=1000.0, masses=(mw, 0.0), seed=3)
    return [[500.0, 0.0, 0.0, 500.0], [500.0, 0.0, 0.0, -500.0], *map(list, final)]


W_REQUEST = ("u d~ > w+ g", {"QCD": 1, "QED": 1}, {"QCD": 3, "QED": 1}, ("QCD",))


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
# check (rusticol-core loop_v2/trees.rs ``verify_plane``) once compared the
# plane application with the exact evaluator at real inputs 0.5 + frac(0.618
# (k + 1)), assigned by input index: where MW > MZ there, the argument of the
# W coupling's (1 - MW^2/MZ^2)^(-1/2) was a negated real, which SymJIT signs
# -0 and Symbolica +0, so the export was refused depending on the parameter
# order (G_mu's card-input MW passed). Since WP3.2 EWI-N (ew-native; slice-
# status DECISIONS 2026-10-26 ~21:10 item 6, user-confirmed 2026-10-27
# ~19:20) the check certifies the translation at complex rows off the real
# axis, whatever the order and names, and the real-axis evaluation at the
# request's parameter values (``native_trees._request_points``), where the
# argument is positive: both variants take the native route
# (``test_loop_v2_ew_native.py`` holds the model-free tests). Since M-SM
# MS-MASTERS the translation also makes a cut argument's zero imaginary part
# +0, so at a real point on the cut the plane takes exact arithmetic's side.
def _former_probes(count: int) -> list[tuple[float, ...]]:
    """The former real probes of ``count`` inputs: two rows, by input index."""

    values = [0.5 + (0.6180339887498949 * (k + 1)) % 1.0 for k in range(2 * count)]
    return [tuple(values[:count]), tuple(values[count:])]


@pytest.mark.parametrize(
    "name", ["loop_sm", pytest.param("gmu", marks=pytest.mark.loop_nightly)]
)
def test_mw_nature_takes_the_same_native_route(w_slices, name):
    """Derived masses are parameter-program outputs (generation-scale-2's
    M1/M2), and the plane check does not depend on the parameter order: MW
    takes the native route whether a card input or derived."""

    assert _route(w_slices(name, "mass")) == _route(w_slices(name, None))
    assert _route(w_slices(name, None)) == "native-v2"


def test_plane_mismatch_names_the_parameter_program_output(w_slices, monkeypatch):
    """At the former real probes as its real point (MW > MZ there: the W
    coupling's argument on its cut), the loop_sm variant's parameter program
    exports natively (the translation takes exact arithmetic's side of a
    cut, M-SM MS-MASTERS); without a real point it is refused as the listed
    ``plane-mismatch`` (a ``GenerationError`` at generation, no raw
    ``ValueError``), naming the parameter-program output, with no stale
    advice."""

    from pyamplicol.loops.v2 import native_trees
    from pyamplicol.loops.v2.native_trees import NativeExportUnsupported
    from pyamplicol.loops.v2.plan_export import export_plan

    slice_ = w_slices("loop_sm", "mass")
    assert _route(slice_) == "native-v2"
    monkeypatch.setattr(
        native_trees,
        "_request_points",
        lambda _slice, record: _former_probes(len(record["names"])),
    )
    assert export_plan(slice_, forward=False, trees=True, born_squared=True)
    monkeypatch.setattr(native_trees, "_request_points", lambda _slice, record: [])
    with pytest.raises(NativeExportUnsupported) as refused:
        export_plan(slice_, forward=False, trees=True, born_squared=True)
    assert refused.value.feature == "plane-mismatch"
    message = str(refused.value)
    assert re.search(r"parameter-program output \d+ \(\w", message), message
    # A derived coupling is named by its UFO coupling and vertex, not by the
    # runtime's term id (``native_trees._output_label``).
    assert re.search(r"\(model coupling \w+ of vertex \w+ \(", message), message
    assert "derived_coupling_" not in message, message
    assert "reads a branch-cut" in message, message
    assert "exact evaluator;" not in message


@pytest.mark.parametrize(
    "text", ["sqrt(1 - x^2/y^2)", "(1 - x^2/y^2)^(-1/2)", "sqrt(1 - y^2/x^2)"]
)
def test_plane_check_accepts_the_w_coupling_in_either_parameter_order(
    loop_native, text
):
    """The mechanism of the former loop_sm refusal, model-free and in a
    second: a square-root argument that the former real probes (x = 1.118 >
    y = 0.736) put on its cut, checked at the request's point with x = MW,
    y = MZ, or with the inputs declared the other way round (the last
    case): accepted, and also at a real point on the cut (exact arithmetic's
    side there, M-SM MS-MASTERS)."""

    from symbolica import E, Expression, S

    state = bytes(
        Expression.evaluator_multiple(
            [E(text)], [S("x"), S("y")], jit_compile=False
        ).save()
    )
    point = [80.419, 91.188] if "x^2/y^2" in text else [91.188, 80.419]
    assert loop_native.compile_v2_plane(state, 2, False, [point])
    assert loop_native.compile_v2_plane(state, 2, False, [point[::-1]])
