# SPDX-License-Identifier: 0BSD
"""Dirac-Weyl closures in the loop Born admission (m-qcd MQ-5F).

A fermion current whose line passed a massive fermion is a Dirac
(chirality-0) current, a massless external fermion a Weyl (chirality +-1)
state, and models install no Dirac-Weyl closure contraction. Where a chiral
vertex takes a line between massive and massless fermions (the top and a
massless b through a W on loop_sm-no_b_mass, the top and a light quark
through a non-diagonal CKM element, a massive charm and a strange quark on
c_mass), the tree DAG would close a Dirac current on a Weyl sink: the tree
lanes refuse it.

The loop engine needs no such contraction. It reads its Born DAG's colour
plan and, per amplitude root, the coupling orders, colour sector and
helicities; its Born amplitude is its own closure of the tree blocks at a
model vertex holding leg 1, whose kernels embed Weyl currents in the Dirac
basis. ``test_loop_slices_never_read_the_root_contraction`` records that
decision: a slice whose Born DAG carries the marker contraction on every
direct root is byte-identical. The loop Born DAG (``loop_born``) therefore
records a Dirac-Weyl closure as a root with the marker
``dag_ordering.DIRAC_WEYL_CLOSURE``, which no tree evaluator lowers.

The class then generates on loop_sm's own restrictions, and its values equal
the default restriction with the b massless at run time (MB = ymb = 0, where
the b is a Dirac field everywhere) to rounding: binary64 within 1e-13, and
double-double within 1e-28. The MadLoop rows of m-qcd PLAN §4.2 (D17, F01,
F02, F03, F04, C09, H02, H03) run in MQ-VALIDATE's campaign.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from pyamplicol.generation import dag_compiler
from pyamplicol.generation.dag_ordering import DIRAC_WEYL_CLOSURE
from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
# The default restriction with a massless b and tau at run time: the 5FS of
# loop_sm-no_b_mass with every b and tau a Dirac field.
MASSLESS_B = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0", **ZERO_WIDTHS}
PARAMETERS = {"aS": "0.118", "MT": "173"}
MT, MW = "173", "80.41900244575616010551713717311353321699"
MU_R2 = "1000000"
QED_BORN = ({"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2})
TW_BORN = ({"QCD": 1, "QED": 1}, {"QCD": 3, "QED": 1})
MIXED_BORN = ({"QCD": 2, "QED": 2}, ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}))
# process: (Born orders, loop orders, external masses)
PROCESSES = {
    "g b > t w-": (*TW_BORN, ("0", "0", MT, MW)),
    "b b~ > w+ w-": (*QED_BORN, ("0", "0", MW, MW)),
    "u b > d t": (*QED_BORN, ("0", "0", "0", MT)),
    "b u > d t": (*QED_BORN, ("0", "0", "0", MT)),
    "t t~ > b b~": (*MIXED_BORN, (MT, MT, "0", "0")),
    "g s > t w-": (*TW_BORN, ("0", "0", MT, MW)),
    "g d > t w-": (*TW_BORN, ("0", "0", MT, MW)),
    "u s > d t": (*QED_BORN, ("0", "0", "0", MT)),
    "s s~ > w+ w-": (*QED_BORN, ("0", "0", MW, MW)),
    "g s > w- c": (*TW_BORN, ("0", "0", MW, None)),
}
# m-qcd PLAN §3.2 MQ-5F: the restricted-UFO class that must generate.
CLASS = [
    *(
        ("no_b_mass", process)
        for process in (
            "g b > t w-",
            "b b~ > w+ w-",
            "u b > d t",
            "b u > d t",
            "t t~ > b b~",
        )
    ),
    *(
        (restriction, process)
        for restriction in ("zeromass_ckm", "ckm")
        for process in ("g s > t w-", "g d > t w-", "u s > d t", "s s~ > w+ w-")
    ),
    ("c_mass", "s s~ > w+ w-"),
    ("c_mass", "g s > w- c"),
]


@pytest.fixture(scope="module")
def restricted(loop_sm_ufo, tmp_path_factory):
    """loop_sm under one of its restriction cards (the scalar-arity view of
    ``tests/conftest.py``'s ``loop_sm_compiled_model``), compiled once."""

    from pyamplicol.models.loading import compile_model_source, load_compiled_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    built = {}

    def compile_(restriction: str, *, path: bool = False):
        if restriction not in built:
            root = tmp_path_factory.mktemp(f"loop-sm-{restriction}")
            model = compile_model_source(
                loop_sm_ufo, restriction=restriction, cache_dir=root / "cache"
            )
            model = replace(
                model,
                raw_loop_model=prepare_loop_sm_scalar_declarations(
                    model.raw_loop_model
                ).derived,
                _serialized_path=None,
                _prepared_bundle=None,
            )
            written = model.write(root / "model.json")
            built[restriction] = (written, load_compiled_model(written))
        return built[restriction][0 if path else 1]

    return compile_


def _slice(compiled, native, process, fixed):
    born, loop, _masses = PROCESSES[process]
    return LoopSlice.generate(
        compiled,
        SliceRequest(process, born, loop, fixed_parameters=dict(fixed)),
        native,
    )


def _mass(compiled, process: str, value: str | None) -> str:
    """A leg mass of ``PROCESSES``; ``None``: the charm's at the model's
    default (the only massive leg whose value the restriction keeps)."""

    if value is not None:
        return value
    (charm,) = (record for record in compiled.ir.particles if int(record.pdg_code) == 4)
    return str(compiled.parameter_defaults[str(charm.mass)][0])


def _point(masses) -> list[list[str]]:
    """A 2 -> 2 point at sqrt(s) = 1000, on shell to 45 digits, with a
    generic scattering angle."""

    with localcontext() as context:
        context.prec = 60
        m1, m2, m3, m4 = (Decimal(m) for m in masses)
        root = Decimal(1000)
        s = root * root

        def momentum(a, b):
            return ((s - (a + b) ** 2) * (s - (a - b) ** 2)).sqrt() / (2 * root)

        p, q = momentum(m1, m2), momentum(m3, m4)
        e1, e2 = (p * p + m1 * m1).sqrt(), (p * p + m2 * m2).sqrt()
        e3, e4 = (q * q + m3 * m3).sqrt(), (q * q + m4 * m4).sqrt()
        transverse = q * Decimal(4) / 5
        v = [transverse * 5 / 13, transverse * 12 / 13, q * Decimal(3) / 5]
        legs = [[e1, 0, 0, p], [e2, 0, 0, -p], [e3, *v], [e4, *(-x for x in v)]]
        return [[format(+Decimal(x), "f") for x in leg] for leg in legs]


def _evaluate(slice_, point, precision):
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    (result,) = executor.evaluate(
        [point], [MU_R2], parameters=PARAMETERS, precision=precision
    )
    return result


def _values(result) -> dict[tuple, Decimal]:
    values = {("born", sector): Decimal(str(v)) for sector, v in result.born.items()}
    for sector, laurent in result.components["virtual"].items():
        for k, x in enumerate(laurent.coefficients):
            values[("virtual", sector, k)] = Decimal(str(x))
    return values


def _relative(a: Decimal, b: Decimal) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else float(abs(a - b) / scale)


def _assert_catani(result) -> None:
    """The renormalised virtual's poles equal Catani's in every predicted
    sector (double-double)."""

    assert result.ir_poles
    virtual = result.components["virtual"]
    for sector, (single, double) in result.ir_poles.items():
        coefficients = virtual[sector].coefficients
        for mine, theirs in ((coefficients[1], single), (coefficients[2], double)):
            assert _relative(Decimal(str(mine)), Decimal(str(theirs))) <= 1e-24, (
                sector,
                mine,
                theirs,
            )


@pytest.mark.loop_physics
def test_loop_slices_never_read_the_root_contraction(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """The decision of MQ-5F item 1: a loop slice reads no root contraction
    of its Born DAG. With the marker on every direct root (the Born DAG and
    the Born-order preflight alike), the exported plan is byte-identical and
    the Born and virtual are identical at a point."""

    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    process = "b u > d t"
    point = _point(("0", "0", "0", MT))
    plain = _slice(compiled, loop_native, process, MASSLESS_B)
    native = dag_compiler._direct_contraction_ir

    def marked(model, left, right, *, loop_born=False):
        contraction = native(model, left, right, loop_born=loop_born)
        if contraction is None or not loop_born:
            return contraction
        return DIRAC_WEYL_CLOSURE

    monkeypatch.setattr(dag_compiler, "_direct_contraction_ir", marked)
    marker = _slice(compiled, loop_native, process, MASSLESS_B)
    kinds = {
        (root.kind, root.contraction_ir == DIRAC_WEYL_CLOSURE)
        for root in marker.blocks.born_dag.amplitude_roots
    }
    assert ("direct-contraction", True) in kinds
    assert ("direct-contraction", False) not in kinds
    assert not any(
        root.contraction_ir == DIRAC_WEYL_CLOSURE
        for root in plain.blocks.born_dag.amplitude_roots
    )
    assert export_plan(marker).json() == export_plan(plain).json()
    double = Precision.double()
    assert _values(_evaluate(marker, point, double)) == _values(
        _evaluate(plain, point, double)
    )


@pytest.mark.loop_fast
def test_the_loop_born_dag_records_a_dirac_weyl_root(restricted):
    """b b~ > w+ w- with a massless b: the top t channel closes a Dirac
    current on a Weyl b. The tree DAG refuses it; the loop Born DAG records
    the root with the marker contraction at the Born's order."""

    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.loops.v2.blocks import _born_roots
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled = restricted("no_b_mass")
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir("b b~ > w+ w-", compiled.ir, color_accuracy="full")
    bounds = {"QCD": 0, "QED": 2}
    with pytest.raises(
        ValueError, match="dirac-weyl-closure: a Dirac \\(chirality-0\\) current cannot"
    ):
        compile_generic_dag(process, model=model, max_coupling_orders=bounds)
    dag = compile_generic_dag(
        process, model=model, max_coupling_orders=bounds, loop_born=True
    )
    marked = [
        root
        for root in dag.amplitude_roots
        if root.contraction_ir == DIRAC_WEYL_CLOSURE
    ]
    assert marked
    assert {root.coupling_orders for root in _born_roots(dag, model)} == {(("QED", 2),)}


@pytest.mark.loop_fast
def test_the_born_order_preflight_admits_the_class(restricted):
    """``artifact.born_amplitude_orders`` (the public route's preflight)
    reads the Born orders from a loop Born DAG: g b > t w- with a massless b,
    whose every root closes a Dirac current on a Weyl sink, has its Born."""

    from pyamplicol.loops.v2.artifact import born_amplitude_orders

    orders, _dag = born_amplitude_orders(
        restricted("no_b_mass"), "g b > t w-", {"QCD": 1, "QED": 1}
    )
    assert orders == ((("QCD", 1), ("QED", 1)),)


def _equal_routes(restricted, loop_native, process, carriers):
    """The process on loop_sm-no_b_mass equals the default restriction with
    MB = ymb = 0 at a point, per Born sector and virtual coefficient."""

    masses = PROCESSES[process][2]
    point = _point(masses)
    weyl = _slice(restricted("no_b_mass"), loop_native, process, ZERO_WIDTHS)
    dirac = _slice(restricted("default"), loop_native, process, MASSLESS_B)
    for precision, bound in carriers:
        mine = _values(_evaluate(weyl, point, precision))
        theirs = _values(_evaluate(dirac, point, precision))
        assert set(mine) == set(theirs)
        assert any(value for key, value in mine.items() if key[0] == "virtual")
        worst = max(_relative(mine[key], theirs[key]) for key in mine)
        assert worst <= bound, (precision, worst)


@pytest.mark.loop_physics
def test_a_massless_b_line_through_the_top_equals_the_massive_b_route(
    restricted, loop_native
):
    """b b~ > w+ w- (the top t channel, MadLoop row D17): binary64 within
    1e-13 of the massive-b route with MB = ymb = 0."""

    _equal_routes(
        restricted, loop_native, "b b~ > w+ w-", [(Precision.double(), 1e-13)]
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "process",
    ["g b > t w-", "b b~ > w+ w-", "u b > d t", "b u > d t", "t t~ > b b~"],
)
def test_the_no_b_mass_class_equals_the_massive_b_route(
    restricted, loop_native, process
):
    """Every no_b_mass process of the class: binary64 within 1e-13 and
    double-double within 1e-28 of the massive-b route with MB = ymb = 0."""

    _equal_routes(
        restricted,
        loop_native,
        process,
        [(Precision.double(), 1e-13), (Precision.double_double(), 1e-28)],
    )


# The class member in the default tier: a W line a non-diagonal CKM element
# takes between a light quark and the top (about 16 s); the rest nightly.
_DEFAULT_TIER_MEMBER = ("ckm", "u s > d t")


@pytest.mark.parametrize(
    ("restriction", "process"),
    [
        pytest.param(
            *member,
            marks=pytest.mark.loop_physics
            if member == _DEFAULT_TIER_MEMBER
            else pytest.mark.loop_nightly,
        )
        for member in CLASS
    ],
)
def test_the_dirac_weyl_class_generates(restricted, loop_native, restriction, process):
    """The restricted-UFO class generates and its renormalised virtual has
    Catani's poles (double-double) at a point."""

    compiled = restricted(restriction)
    masses = PROCESSES[process][2]
    point = _point(tuple(_mass(compiled, process, m) for m in masses))
    slice_ = _slice(compiled, loop_native, process, ZERO_WIDTHS)
    result = _evaluate(slice_, point, Precision.double_double())
    assert any(Decimal(str(v)) for v in result.born.values())
    _assert_catani(result)


@pytest.mark.loop_nightly
def test_the_public_route_generates_the_class(restricted, loop_native, tmp_path):
    """b b~ > w+ w- on loop_sm-no_b_mass through the TOML card,
    ``Generator`` and ``Runtime`` (minimal coupling orders: the preflight
    reads the Born orders from the loop Born DAG) equals the slice in
    binary64."""

    import json

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    model = restricted("no_b_mass", path=True)
    fixed = ", ".join(f'{name} = "{value}"' for name, value in ZERO_WIDTHS.items())
    card = tmp_path / "card.toml"
    card.write_text(
        f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression="b b~ > w+ w-", name="bbx_wpwm"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
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
[generation]
output = {json.dumps(str(tmp_path / "artifact"))}
emit_api_bundle = false
"""
    )
    processes = ProcessSet((ProcessRequest.parse("b b~ > w+ w-", name="bbx_wpwm"),))
    Generator(resolve_config(card)).generate(processes, tmp_path / "artifact")
    runtime = Runtime.load(
        tmp_path / "artifact", process="bbx_wpwm", model_parameters=PARAMETERS
    )
    point = _point(PROCESSES["b b~ > w+ w-"][2])
    public = runtime.evaluate([point], mu_r2=[MU_R2])
    slice_ = _slice(restricted("no_b_mass"), loop_native, "b b~ > w+ w-", ZERO_WIDTHS)
    mine = _evaluate(slice_, point, Precision.double())
    ((born,),) = (public.born_values[0],)
    (theirs,) = mine.born.values()
    assert _relative(Decimal(str(born)), Decimal(str(theirs))) <= 1e-12
    ((laurent,),) = (public.values[0],)
    (virtual,) = mine.components["virtual"].values()
    for ours, theirs in zip(laurent, virtual.coefficients, strict=True):
        assert _relative(Decimal(str(ours)), Decimal(str(theirs))) <= 1e-12
