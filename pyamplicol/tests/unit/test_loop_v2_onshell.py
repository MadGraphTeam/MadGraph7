# SPDX-License-Identifier: 0BSD
"""On-shell external invariants (DESIGN §11.1): every carrier agrees on a
binary64-rounded point.

The masters read every invariant of a loop family from its S matrix. An
invariant that a single external leg fixes (a single-leg arc of the family,
or one whose complement is a single leg) is that leg's squared mass, exactly
the value its propagators take, never the p^2 of the given momenta: rounded
momenta leave a leg off its shell by ~1e-16 relative. Binary64's fast engine
used to snap such an invariant (NinjaRS's absolute 1e-9 threshold) while the
generic engine (DoubleFloat, MPFR) tests on-shellness exactly and evaluated
the soft and collinear masters off shell, an O(1) error: b b~ > t t~ QED=2
[virt=QCD] with a massive b gave 3.0434 in binary64 and -50.574 in
arbitrary precision at the point below. The native DoubleFloat trees also
lost the massive spinor of a b moving along -z (a Born of ~1e-33).

Each test evaluates one rounded point in the production path (native
executor, native inputs) in binary64, DoubleFloat and MPFR, and its exactly
on-shell, momentum-conserving 140-digit neighbour in MPFR:

* DoubleFloat equals MPFR on the same inputs within 1e-24 (binary64
  observable rule, DESIGN §10.4);
* binary64 equals MPFR within 1e-12, or, where its own stability estimate E
  (the reduction-direction check) exceeds 1e-12, within the user's tenfold
  estimate margin 10 E (DESIGN §10.4), with E at most 1e-9. At the
  reproducer point binary64 is off by 6.6e-10 (E = 6.3e-10): the fast
  engine's IR-divergent (0, MT, MB) triangle, which the generic engine
  evaluates to 1e-15 in binary64;
* MPFR at the rounded point equals MPFR at the on-shell neighbour within
  1e-13: the rounded momenta only move the value at their own (analytic)
  order.

The SDK's plan v4 runtime (``v2_runtime::V2Runtime``, no Python, binary64
only) takes its leg masses from ``TreeProgram::bind`` as the Python API's
executor does: at the rounded g g > t t~ point its values are bitwise the
API's and agree with MPFR by the rule above. Binary64 alone cannot tell a
missing binding (its fast engine snaps the single-leg invariants: the
Python-input lane gives the same bits with every leg unbound at this point
and at the u u~ > t t~ g one); the carriers it matters for (DoubleFloat,
MPFR) run through the Python API's executor above.

The loop_fast test keeps only the binary64-against-MPFR check at the
reproducer point.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.families import (
    ZERO,
    ExternalLeg,
    Family,
    MassId,
    canonical_family_key,
    onshell_leg,
)
from pyamplicol.loops.v2.plan_export import NativeExecutor
from pyamplicol.loops.v2.stability import Budget, StabilityChecks, deviation

from .loop_v2_ufo_variants import internal_mass_ufo

SCALE = "250000"
PARAMETERS = {"aS": "0.118"}
WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
DOUBLE_DOUBLE = 1e-24
BINARY64 = 1e-12
ESTIMATE_CAP = 1e-9
# USER DECISION (2026-10-23; DESIGN §10.4): a single-point test of a
# stability estimate E holds the true error to at most 10 E.
ESTIMATE_MARGIN = 10
ONSHELL_NEIGHBOUR = 1e-13


def _reproducer_point() -> list[list[str]]:
    """The defect's point: b b~ > t t~ at E = 500 GeV with MB = 4.7 and MT =
    173 (loop_sm's defaults), every component rounded to binary64."""

    energy, mb, mt = 500.0, 4.7, 173.0
    pb = math.sqrt(energy * energy - mb * mb)
    pt = math.sqrt(energy * energy - mt * mt)
    legs = [
        (energy, 0.0, 0.0, pb),
        (energy, 0.0, 0.0, -pb),
        (energy, pt * 0.8, 0.0, pt * 0.6),
        (energy, -pt * 0.8, 0.0, -pt * 0.6),
    ]
    return [[repr(x) for x in leg] for leg in legs]


# A RAMBO point at sqrt(s) = 1 TeV (seed 11) rotated in binary64: every leg
# off shell by rounding (up to 3e-16 relative), momentum conserved to 4e-13.
GG_TTX = [
    ["500.0", "-158.15724824596936", "-246.54930300089077", "405.2156537166818"],
    ["500.0", "158.15724824596936", "246.54930300089077", "-405.2156537166818"],
    ["500.0", "-319.9348540402197", "-224.18373180463956", "-259.7197404211101"],
    [
        "500.00000000000006",
        "319.9348540402199",
        "224.18373180463973",
        "259.71974042110975",
    ],
]
UUX_TTXG = [
    ["500.0", "-158.15724824596936", "-246.54930300089077", "405.2156537166818"],
    ["500.0", "158.15724824596936", "246.54930300089077", "-405.2156537166818"],
    [
        "349.69498487261586",
        "-152.2039402255543",
        "-62.925702607411836",
        "-255.4053620741712",
    ],
    [
        "387.1734985760383",
        "251.5490604307532",
        "226.1555251577807",
        "74.50548058111738",
    ],
    [
        "263.13151655134584",
        "-99.34512020519884",
        "-163.22982255036877",
        "180.89988149305373",
    ],
]

# name: process, Born orders, loop orders, fixed parameters, external masses
# (incoming two first), point.
CASES = {
    "bbx_ttx": (
        "b b~ > t t~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        {},
        ("4.7", "4.7", "173", "173"),
        _reproducer_point,
    ),
    "gg_ttx": (
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        {"MB": "0"},
        ("0", "0", "173", "173"),
        lambda: GG_TTX,
    ),
    "uux_ttxg": (
        "u u~ > t t~ g",
        {"QCD": 3, "QED": 0},
        {"QCD": 5, "QED": 0},
        {"MB": "0"},
        ("0", "0", "173", "173", "0"),
        lambda: UUX_TTXG,
    ),
}


def _onshell_neighbour(point, masses) -> list[list[str]]:
    """The point at 140 digits made exactly on shell and momentum conserving:
    the final 3-momenta kept (the last one by conservation), every energy on
    its shell, the incoming pair along its axis with half the total energy
    each (both incoming legs share their mass here)."""

    with localcontext() as context:
        context.prec = 140
        mass = [Decimal(m) for m in masses]
        incoming = [Decimal(x) for x in point[0][1:]]
        norm = sum(x * x for x in incoming).sqrt()
        axis = [x / norm for x in incoming]
        final = [[Decimal(x) for x in leg[1:]] for leg in point[2:]]
        final[-1] = [-sum(leg[mu] for leg in final[:-1]) for mu in range(3)]
        energies = [
            (m * m + sum(x * x for x in leg)).sqrt()
            for m, leg in zip(mass[2:], final, strict=True)
        ]
        half = sum(energies) / 2
        size = (half * half - mass[0] * mass[0]).sqrt()
        text = lambda x: format(+x, "f")  # noqa: E731
        return [
            [text(half), *(text(size * a) for a in axis)],
            [text(half), *(text(-size * a) for a in axis)],
            *(
                [text(e), *(text(x) for x in leg)]
                for e, leg in zip(energies, final, strict=True)
            ),
        ]


def _builder(compiled, loop_native):
    """Per case name, its slice and native executor on ``compiled``."""

    built: dict[str, tuple[LoopSlice, NativeExecutor]] = {}

    def build(name: str) -> tuple[LoopSlice, NativeExecutor]:
        if name not in built:
            process, born, loop, fixed, _masses, _point = CASES[name]
            slice_ = LoopSlice.generate(
                compiled,
                SliceRequest(process, born, loop, fixed_parameters={**WIDTHS, **fixed}),
                loop_native,
            )
            built[name] = slice_, NativeExecutor(slice_, inputs="native")
        return built[name]

    return build


@pytest.fixture(scope="module")
def executors(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    return _builder(load_compiled_model(loop_sm_compiled_model), loop_native)


@pytest.fixture(scope="module")
def internal_mt_executors(loop_sm_ufo, loop_native, tmp_path_factory):
    """loop_sm with MT internal (``internal_mass_ufo``), compiled as
    ``loop_sm_compiled_model`` is."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("loop-sm-internal-mt")
    ufo = internal_mass_ufo(loop_sm_ufo, root / "loop_sm_internal_mt", "MT")
    model = compile_model_source(ufo, cache_dir=root / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return _builder(model, loop_native)


def _virtual(result) -> dict:
    return result.components["virtual"]


def _binary64(laurents) -> dict:
    """Decimal Laurent coefficients rounded to binary64 (to compare with a
    binary64 result's floats)."""

    return {
        sector: Budget(tuple(float(x) for x in laurent.coefficients))
        for sector, laurent in laurents.items()
    }


def _agree_across_carriers(executors, name: str):
    slice_, executor = executors(name)
    *_, masses, point_of = CASES[name]
    point = point_of()

    def evaluate(at, precision, stability=None):
        (result,) = executor.evaluate(
            [at],
            [SCALE],
            parameters=PARAMETERS,
            precision=precision,
            stability=stability,
        )
        return result

    reference = evaluate(point, Precision.arbitrary(32))
    double_double = evaluate(point, Precision.double_double())
    assert deviation(_virtual(reference), _virtual(double_double)) < DOUBLE_DOUBLE
    double = evaluate(point, None, StabilityChecks(reduction_direction=True))
    estimate = double.stability.relative
    assert estimate <= ESTIMATE_CAP, estimate
    error = deviation(_binary64(_virtual(reference)), _virtual(double))
    assert error <= max(BINARY64, ESTIMATE_MARGIN * estimate), (error, estimate)
    onshell = evaluate(_onshell_neighbour(point, masses), Precision.arbitrary(32))
    assert deviation(_virtual(onshell), _virtual(reference)) < ONSHELL_NEIGHBOUR
    return slice_, point, reference


@pytest.mark.loop_fast
def test_a_single_leg_fixes_its_arc_and_the_complement():
    full = 0b11111
    assert onshell_leg(0b00100, full) == 2
    assert onshell_leg(0b11011, full) == 2
    assert onshell_leg(0b00110, full) is None
    assert onshell_leg(0, full) is None


@pytest.mark.loop_fast
def test_structural_onshell_bindings_follow_the_same_rule():
    """``Family.onshell_bindings`` (the certified-scaleless proof) binds the
    invariants of a single-leg arc or single-leg complement, and only those,
    to that leg's mass: the ``onshell_leg`` rule of every carrier. Expected
    bindings written out by hand on 5-leg rings (every leg a distinct mass,
    so a wrong leg shows)."""

    legs = tuple(ExternalLeg(index, "t", MassId(f"M{index}"), 3) for index in range(5))
    masses = [MassId(f"M{index}") for index in range(5)]
    expected = {
        # 10 invariants; arcs {1}..{4} and the complement of {0}.
        (1, 2, 4, 8, 16): dict(
            zip((2, 4, 8, 16, 30), masses[1:] + masses[:1], strict=True)
        ),
        # Legs 0 and 1 merged: 4, 8, 16 and 12 ({2,3}), 24, 28 (complement {0,1}).
        (3, 4, 8, 16): {4: masses[2], 8: masses[3], 16: masses[4]},
        # Invariants 30 (complement {0}), 6 ({1,2}) and 24 ({3,4}).
        (1, 6, 24): {30: masses[0]},
        # Invariants 2 ({1}), 24 ({3,4}) and 26 (complement {0,2}).
        (5, 2, 24): {2: masses[1]},
    }
    for masks, bindings in expected.items():
        key = canonical_family_key(masks, [MassId(ZERO)] * len(masks))
        family = Family(key, 0, ())
        assert family.onshell_bindings(legs) == bindings, masks


@pytest.mark.loop_fast
def test_the_massive_b_reproducer_agrees_in_binary64_and_arbitrary_precision(
    executors,
):
    """The defect's point, lean: binary64 (3.0434) and MPFR (previously
    -50.574) in the production path agree to binary64's accuracy. The
    loop_physics test below adds DoubleFloat, the on-shell neighbour and the
    Python reference."""

    _slice, executor = executors("bbx_ttx")
    point = _reproducer_point()
    (reference,) = executor.evaluate(
        [point], [SCALE], parameters=PARAMETERS, precision=Precision.arbitrary(32)
    )
    (double,) = executor.evaluate(
        [point],
        [SCALE],
        parameters=PARAMETERS,
        stability=StabilityChecks(reduction_direction=True),
    )
    estimate = double.stability.relative
    assert estimate <= ESTIMATE_CAP, estimate
    error = deviation(_binary64(_virtual(reference)), _virtual(double))
    assert error <= max(BINARY64, ESTIMATE_MARGIN * estimate), (error, estimate)


@pytest.mark.loop_physics
def test_the_massive_b_reproducer_agrees_in_every_carrier(executors):
    """b b~ > t t~ QED=2 [virt=QCD], massive b along the beam: the defect's
    point (binary64 3.0434, previously -50.574 in arbitrary precision; a
    DoubleFloat Born of ~1e-33). The Python reference's MPFR value equals the
    native one."""

    slice_, point, native = _agree_across_carriers(executors, "bbx_ttx")
    reference = slice_.evaluate(
        point, SCALE, parameters=PARAMETERS, precision=Precision.arbitrary(32)
    )
    assert deviation(_virtual(reference), _virtual(native)) < 1e-28


@pytest.mark.loop_physics
def test_a_rounded_massive_qcd_point_agrees_in_every_carrier(executors):
    """g g > t t~ at a rotated rounded point: every leg off shell by rounding."""

    _agree_across_carriers(executors, "gg_ttx")


@pytest.mark.loop_physics
def test_an_internal_top_mass_is_bound_on_shell(executors, internal_mt_executors):
    """loop_sm with MT internal (MT = MTx, an external parameter of the same
    value; genericity audit M2): the top's loop propagators (``["output",
    o]`` plan masses) and legs are bound from the parameter program's output
    natively and from the slice's one mass evaluator in Python, so g g > t t~
    at the rotated rounded point agrees in every carrier as with the external
    MT (DoubleFloat and MPFR with the legs on their shell, the on-shell
    neighbour), binds the external MT's exact masses (173 is a binary64
    value), equals its virtual and keeps Catani's prediction."""

    slice_, point, mpfr = _agree_across_carriers(internal_mt_executors, "gg_ttx")
    _slice, executor = internal_mt_executors("gg_ttx")
    program = executor.exported.trees.program["parameters"]
    assert any(isinstance(mass, list) for mass in program["plan_masses"])
    assert program["derived_masses"]
    external_slice, external = executors("gg_ttx")
    for precision in (None, Precision.double_double(), Precision.arbitrary(32)):
        assert executor.bound_masses(
            [point], [SCALE], PARAMETERS, precision
        ) == external.bound_masses([point], [SCALE], PARAMETERS, precision)
    (reference,) = external.evaluate(
        [point], [SCALE], parameters=PARAMETERS, precision=Precision.arbitrary(32)
    )
    assert deviation(_virtual(reference), _virtual(mpfr)) < 1e-28
    mine = slice_.evaluate(point, SCALE, parameters=PARAMETERS)
    theirs = external_slice.evaluate(point, SCALE, parameters=PARAMETERS)
    assert theirs.ir_poles is not None
    assert mine.ir_poles == theirs.ir_poles


_SDK_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{{expression="g g > t t~", name="ggttx"}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0", WT = "0"}}
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
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


@pytest.mark.loop_physics
def test_the_sdk_runtime_binds_the_legs_of_a_rounded_massive_point(
    executors, loop_sm_compiled_model, loop_native, tmp_path
):
    """The SDK's plan v4 runtime (``v2_runtime::V2Runtime`` through
    ``evaluate_v2_artifact``: no Python generation or assembly; its batch
    takes ``leg_masses`` from ``TreeProgram::bind``) at the rotated rounded
    g g > t t~ point (every leg off shell by rounding): its binary64 Born
    and virtual, with and without poles, are bitwise the Python API's, and
    the virtual agrees with MPFR by the binary64 rule above."""

    import json

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    # The rounded decimals leave the massless legs off shell (exactly; the
    # invariants the leg binding sets to 0, which binary64 snaps anyway).
    assert all(
        e * e - x * x - y * y - z * z != 0
        for e, x, y, z in ([Fraction(v) for v in leg] for leg in GG_TTX[:2])
    )
    card = tmp_path / "card.toml"
    output = tmp_path / "artifact"
    card.write_text(
        _SDK_CARD.format(
            model=json.dumps(str(loop_sm_compiled_model)),
            output=json.dumps(str(output)),
        )
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse("g g > t t~", name="ggttx"),)), output
    )
    runtime = Runtime.load(output, process="ggttx", model_parameters=PARAMETERS)
    for poles in (True, False):
        python = runtime.evaluate(
            [GG_TTX], mu_r2=[SCALE], poles="all" if poles else "none"
        )
        ((values, failures),) = loop_native.evaluate_v2_artifact(
            str(output),
            [[x for leg in GG_TTX for x in leg]],
            [SCALE],
            process="ggttx",
            parameters=PARAMETERS,
            poles=poles,
        )
        assert not failures, failures
        born, virtual = values
        assert born == list(python.born_values[0])
        width = 3 if poles else 1
        assert virtual == [list(laurent)[:width] for laurent in python.values[0]]
    # The SDK's virtual against MPFR (the production executor's arbitrary
    # precision at the same rounded point), by the binary64 rule.
    _slice, executor = executors("gg_ttx")
    (reference,) = executor.evaluate(
        [GG_TTX], [SCALE], parameters=PARAMETERS, precision=Precision.arbitrary(32)
    )
    (double,) = executor.evaluate(
        [GG_TTX],
        [SCALE],
        parameters=PARAMETERS,
        stability=StabilityChecks(reduction_direction=True),
    )
    estimate = double.stability.relative
    assert estimate <= ESTIMATE_CAP, estimate
    ((sdk, _failures),) = loop_native.evaluate_v2_artifact(
        str(output),
        [[x for leg in GG_TTX for x in leg]],
        [SCALE],
        process="ggttx",
        parameters=PARAMETERS,
    )
    ((sector, _laurent),) = _virtual(reference).items()
    (sdk_virtual,) = sdk[1]
    error = deviation(
        _binary64(_virtual(reference)), {sector: Budget(tuple(sdk_virtual))}
    )
    assert error <= max(BINARY64, ESTIMATE_MARGIN * estimate), (error, estimate)


@pytest.mark.loop_nightly
def test_a_rounded_massive_qcd_point_with_a_gluon_agrees_in_every_carrier(executors):
    """u u~ > t t~ g: a massless final-state leg off shell by rounding too
    (exactly on-shell 140-digit decimals left its C0 non-finite before)."""

    _agree_across_carriers(executors, "uux_ttxg")
