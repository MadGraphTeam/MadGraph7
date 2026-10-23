# SPDX-License-Identifier: 0BSD
"""v2 observables (WP2.5; design §7.7, §8.4, §8.5, §9).

Every value comes from the v2 generators, the Python reference executor and
the observable seed columns of ``loops.v2.observables``. Oracles:

* providers (O8): Recola's colour dipoles of uu~ -> dd~ and its singleton
  gluon spin source of uu~ -> gg; MadLoop's split-order control of
  uu~ -> dd~ with QCD and photon Born currents (mixed squared-order sectors);
* legacy (O1): the frozen legacy observables
  (``tests/fixtures/loop/legacy/observables``): directed colour histories
  (dipoles and ordered two-emission histories), joint complex spin sources,
  helicity-resolved outputs and flavour members;
* the asymmetric Ward identity (O10) on gg -> gg and uu~ -> ggg: the Born bra
  keeps physical polarisations, the loop side reads epsilon_j -> p_j;
* exact identities: resolved entries sum to the inclusive result, a directed
  history and its swap are complex conjugates, a complete transverse source
  basis sums to the helicity sum, and members and aliases equal the processes
  generated directly.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.api.errors import CompatibilityError
from pyamplicol.color.connections import EmitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    Precision,
    SliceRequest,
)
from pyamplicol.loops.v2.observables import (
    INCLUSIVE,
    MemberObservables,
    ObservableSlice,
    admit_member,
)
from tools.loop_reference import legacy_capture

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
CARRIERS = {"double": Precision.double(), "arbitrary-200": Precision.arbitrary(60)}
PARAMETERS = {"aS": "0.118", "MT": "173"}
QCD2 = {"QCD": 2, "QED": 0}
QCD4 = {"QCD": 4, "QED": 0}
ORDERED = ColorCorrelator(
    "ordered",
    (EmitGluon(1, -1), EmitGluon(1, -2)),
    (EmitGluon(2, -1), EmitGluon(3, -2)),
)
SWAPPED = ColorCorrelator("swapped", ORDERED.ket, ORDERED.bra)
# Two literal complex four-vectors (neither transverse nor normalised).
VECTORS = (
    (("0.1", "0.2"), ("0.8", "-0.3"), ("0.4", "1"), ("-0.6", "0.5")),
    (("-0.3", "0.7"), ("-0.2", "0.8"), ("0.6", "0.1"), ("0.5", "-0.4")),
)


def _points(name: str) -> list:
    data = json.loads((ROOT / "tools/loop_reference" / f"{name}.json").read_text())
    return [[[str(x) for x in p] for p in point] for point in data["points"]]


def _complex(value) -> complex:
    return complex(value)


# Decimal digits of the comparison arithmetic: deviations of 200-bit values
# are measured without rounding them to binary64 first.
COMPARISON_DIGITS = 120


def _real(value) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, str):
        return Decimal(value)
    return Decimal(value)


def _parts(value) -> tuple[Decimal, Decimal]:
    """The exact (real, imaginary) parts of a carrier value (Python number,
    Decimal or ``rx.Complex``) or of a legacy [real, imaginary] string pair."""

    if isinstance(value, (list, tuple)):
        return _real(value[0]), _real(value[1])
    return _real(getattr(value, "real", value)), _real(getattr(value, "imag", 0))


def _magnitude(value) -> Decimal:
    with localcontext(prec=COMPARISON_DIGITS):
        real, imag = _parts(value)
        return (real * real + imag * imag).sqrt()


def _deviation(actual, expected) -> Decimal:
    with localcontext(prec=COMPARISON_DIGITS):
        (a, b), (c, d) = _parts(actual), _parts(expected)
        return ((a - c) ** 2 + (b - d) ** 2).sqrt()


def _close(actual, expected, rel, floor=0.0) -> bool:
    with localcontext(prec=COMPARISON_DIGITS):
        bound = Decimal(str(rel)) * _magnitude(expected) + Decimal(str(floor))
        return _deviation(actual, expected) <= bound


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict = {}

    def build(process, born=QCD2, loop=QCD4, correlators=None, ward=False, **options):
        key = (process, repr(born), repr(loop), repr(correlators), ward, repr(options))
        if key not in built:
            request = SliceRequest(
                process,
                born,
                loop,
                fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                **options,
            )
            built[key] = ObservableSlice.generate(
                compiled, request, loop_native, correlators=correlators, ward=ward
            )
        return built[key]

    return build


# ------------------------------------------------------------------ providers (O8)


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_directed_dipoles_match_recola(generate, loop_reference, carrier):
    """<T_1.T_j> of uu~ -> dd~ against Recola's dipoles (normalised by C_F)."""

    declarations = CorrelatorConfig(
        color_correlations=tuple(
            ColorCorrelator.dipole(f"T1{j}", 1, j) for j in range(1, 5)
        )
    )
    observables = generate("u u~ > d d~", correlators=declarations)
    rows = loop_reference("directed-recola-control")["rows"]
    for row in rows:
        results = observables.evaluate(
            row["point"],
            row["mu_r2"],
            correlations=(INCLUSIVE, "T11", "T12", "T13", "T14"),
            parameters=PARAMETERS,
            precision=CARRIERS[carrier],
        )
        for name, dipole, factor in (
            (INCLUSIVE, "1", 1),
            *((f"T1{j}", str(j), Fraction(4, 3)) for j in range(1, 5)),
        ):
            values = results[name].totals[:4]
            reference = row["dipoles"][dipole]["cepsilon"]
            for actual, expected in zip(values, reference, strict=True):
                assert _close(actual, float(factor) * expected, 3e-10, 2e-13)
                assert abs(complex(actual).imag) < 1e-13


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_singleton_spin_source_matches_recola(generate, loop_reference, carrier):
    """A real transverse source on gluon 3 of uu~ -> gg against Recola."""

    observables = generate(
        "u u~ > g g", correlators=CorrelatorConfig(spin_correlations=((3,),))
    )
    for row in loop_reference("spin-recola-control")["rows"]:
        (result,) = observables.evaluate(
            row["point"],
            row["mu_r2"],
            spin={3: row["vector"]},
            parameters=PARAMETERS,
            precision=CARRIERS[carrier],
        ).values()
        for actual, expected in zip(result.totals[:4], row["cepsilon"], strict=True):
            assert _close(actual, expected, 3e-11, 2e-13)


def test_mixed_sectors_match_madloop_split_orders(generate, loop_reference):
    """uu~ -> dd~ with QCD and photon Born currents: every squared-order
    sector (QCD^6, QCD^4 QED^2, QCD^2 QED^4) against MadLoop, also for the
    directed Casimir column (C_F times each sector), with the Python
    reference executor and natively."""

    from pyamplicol.loops.v2.observables import NativeObservables

    reference = loop_reference("mixed-uudd-qcd-control")
    orders = {
        key: tuple(sorted((name, power) for name, power in value.items() if power))
        for key, value in reference["loop_orders"].items()
    }
    born_orders = {
        key: tuple(sorted((name, power) for name, power in value.items() if power))
        for key, value in reference["born_orders"].items()
    }
    observables = generate(
        "u u~ > d d~",
        born={"QCD": 2, "QED": 2},
        loop=({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        correlators=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("casimir", 1, 1),)
        ),
        ignored_particle_ids=(23, 24, -24, 25, 250, 251, -251),
    )
    point = _points("uudd_qcd")[0]
    histories = (INCLUSIVE, "casimir")
    python = observables.evaluate(
        point, "1000000", correlations=histories, parameters=PARAMETERS
    )
    (native,) = NativeObservables.correlated(observables, histories).evaluate(
        [point], ["1000000"], parameters=PARAMETERS
    )
    row = reference["rows"][0]
    for results in (python, native):
        inclusive, casimir = results[INCLUSIVE], results["casimir"]
        for key, order in born_orders.items():
            assert _close(inclusive.born[order], row["born"][key], 1e-12, 1e-16)
        for key, sector in orders.items():
            virtual = inclusive.components["virtual"][sector]
            for name in ("finite", "single_pole", "double_pole"):
                expected = row["full"][key][name]
                assert _close(getattr(virtual, name), expected, 1e-11, 1e-16)
                correlated = getattr(casimir.components["virtual"][sector], name)
                assert _close(correlated, 4 / 3 * expected, 1e-11, 2e-16)
        assert len(inclusive.components["virtual"]) == 3


# ------------------------------------------------------------------ identities


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_resolved_entries_sum_to_the_inclusive_result(generate, carrier):
    """Helicity x Born-basis entries of gg -> gg (binary64; Born-basis
    entries at 200 bits) sum exactly (to rounding) to the inclusive Born,
    components and 19 totals."""

    observables = generate("g g > g g")
    fixture = json.loads((FIXTURES / "legacy/gg_gg.json").read_text())
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    precision = CARRIERS[carrier]
    inclusive = observables.slice.evaluate(
        point, scale, parameters=PARAMETERS, precision=precision
    )
    resolution = "helicity-colour" if carrier == "double" else "colour"
    resolved = observables.evaluate_resolved(
        point, scale, resolution=resolution, parameters=PARAMETERS, precision=precision
    )
    basis = len(observables.slice.basis)
    assert resolved.components == tuple(range(basis))
    rows = 16 if carrier == "double" else 1
    assert len(resolved.entries) == rows * basis
    total = resolved.total()
    rel = 1e-12 if carrier == "double" else 1e-50
    scale_ = max(abs(float(x)) for x in inclusive.totals)
    for actual, expected in zip(total.totals, inclusive.totals, strict=True):
        assert abs(float(actual - expected)) <= rel * scale_
    for name in COMPONENTS:
        for sector, value in inclusive.components[name].items():
            for a, b in zip(
                total.components[name][sector].coefficients,
                value.coefficients,
                strict=True,
            ):
                assert abs(float(a - b)) <= rel * scale_


def test_directed_history_and_its_swap_are_conjugate(generate):
    """V(bra, ket) = conj V(ket, bra) for an ordered non-Hermitian history."""

    observables = generate(
        "g g > g g",
        correlators=CorrelatorConfig(color_correlations=(ORDERED, SWAPPED)),
    )
    fixture = json.loads((FIXTURES / "legacy/gg_gg.json").read_text())
    point, scale = fixture["points"][1], fixture["scales_squared"][1]
    results = observables.evaluate(
        point, scale, correlations=("ordered", "swapped"), parameters=PARAMETERS
    )
    ordered, swapped = results["ordered"], results["swapped"]
    assert any(abs(complex(x).imag) > 1e-6 * abs(complex(x)) for x in ordered.totals)
    for a, b in zip(ordered.totals, swapped.totals, strict=True):
        assert _close(a, complex(b).conjugate(), 1e-12, 1e-13)


def test_transverse_source_basis_sums_to_the_helicity_sum(generate, loop_reference):
    """Two real transverse vectors on each of gluons 3 and 4 of uu~ -> gg:
    the four joint sources sum to the ordinary helicity sum."""

    observables = generate(
        "u u~ > g g", correlators=CorrelatorConfig(spin_correlations=((3, 4),))
    )
    row = loop_reference("spin-recola-control")["rows"][0]
    point, scale = row["point"], row["mu_r2"]
    a, b = row["vector"], ["0", "0", "1", "0"]
    inclusive = observables.slice.evaluate(point, scale, parameters=PARAMETERS)
    total = [0j] * len(inclusive.totals)
    for v3 in (a, b):
        for v4 in (a, b):
            (result,) = observables.evaluate(
                point, scale, spin={3: v3, 4: v4}, parameters=PARAMETERS
            ).values()
            total = [x + complex(y) for x, y in zip(total, result.totals, strict=True)]
    for actual, expected in zip(total, inclusive.totals, strict=True):
        assert _close(actual, expected, 1e-12, 1e-12)


def test_finite_only_observables(generate):
    """Finite-only mode returns the same finite parts without pole columns."""

    observables = generate(
        "u u~ > d d~",
        correlators=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
        ),
    )
    point = _points("uudd_qcd")[1]
    full = observables.evaluate(
        point, "1000000", correlations=("T12",), parameters=PARAMETERS
    )["T12"]
    finite = observables.evaluate(
        point, "1000000", correlations=("T12",), parameters=PARAMETERS, poles=False
    )["T12"]
    assert len(finite.totals) == 7
    for name in COMPONENTS:
        for sector, value in finite.components[name].items():
            assert value.single_pole is None
            assert _close(
                value.finite, full.components[name][sector].finite, 1e-13, 1e-15
            )


# ------------------------------------------------------------------ Ward (O10)


def _ward_scale(result) -> list[float]:
    """Per Laurent coefficient the largest physical component magnitude."""

    return [
        max(
            abs(complex(laurent.coefficients[k]))
            for name in COMPONENTS
            for laurent in result.components[name].values()
        )
        for k in range(3)
    ]


def _lane_scale(result, average: int) -> list[float]:
    """Per Laurent coefficient the absolute sum of the family lanes' bare
    contributions, 2 |Omega| / (16 pi^2 average): the terms that cancel."""

    from pyamplicol.loops.v2.conventions import LOOP_FACTOR_DENOMINATOR

    factor = 2 / (LOOP_FACTOR_DENOMINATOR * math.pi**2 * average)
    return [
        factor
        * sum(abs(complex(lane.coefficients[k])) for lane in result.families.values())
        for k in range(3)
    ]


@pytest.mark.parametrize(
    "process,name,legs",
    [("g g > g g", "gg_gg", (1, 3)), ("u u~ > g g g", "uux_ggg", (3, 5))],
)
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_asymmetric_ward_identity(generate, process, name, legs, carrier):
    """O10: with epsilon_j -> p_j on the loop side only (blocks, counterterm
    currents, field insertions; full kernels) the renormalised interference
    with the physical Born vanishes per Laurent coefficient and sector,
    relative to the largest physical component. In binary64 the scale is at
    least the absolute sum of the Ward evaluation's family lanes: the bare
    lanes cancel among themselves by orders of magnitude (gauge cancellation
    across families), so rounding is relative to them. In binary64 the Ward
    vector also runs through the native executor (batch inputs only)."""

    from pyamplicol.loops.v2.observables import NativeObservables

    born = {"QCD": 3, "QED": 0} if name == "uux_ggg" else QCD2
    loop = {"QCD": 5, "QED": 0} if name == "uux_ggg" else QCD4
    observables = generate(process, born=born, loop=loop, ward=True)
    fixture = json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    precision = CARRIERS[carrier]
    physical = observables.slice.evaluate(
        point, scale, parameters=PARAMETERS, precision=precision
    )
    tolerance = 1e-11 if carrier == "double" else 1e-45
    native = NativeObservables.correlated(observables, (INCLUSIVE,), lanes=4)
    for leg in legs:
        (ward,) = observables.evaluate(
            point, scale, ward=leg, parameters=PARAMETERS, precision=precision
        ).values()
        runs = [ward]
        if carrier == "double":
            (result,) = native.evaluate(
                [point], [scale], ward=leg, parameters=PARAMETERS
            )
            runs.append(result[INCLUSIVE])
        for run in runs:
            reference = _ward_scale(physical)
            if carrier == "double":
                lanes = _lane_scale(run, observables.slice.normalisation.average)
                reference = [max(a, b) for a, b in zip(reference, lanes, strict=True)]
            for sector, laurent in run.components["virtual"].items():
                for k, value in enumerate(laurent.coefficients):
                    assert abs(complex(value)) <= tolerance * reference[k], (
                        leg,
                        sector,
                        k,
                    )
            # The test is not empty: the physical Born is not.
            assert abs(float(run.totals[0])) > 0


# ------------------------------------------------------------------ flavour (§9)


def test_members_and_aliases_equal_direct_generation(generate):
    """Identity-port members (proved species substitution) and permutation
    aliases evaluated through d d~ > g g equal the processes generated
    directly, for directed histories and spin sources."""

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("casimir", 1, 1), ORDERED),
        spin_correlations=((3,), (3, 4)),
    )
    representative = generate("d d~ > g g", correlators=declarations)
    point = _points("uudd_qcd")[0]
    histories = (INCLUSIVE, "casimir", "ordered")
    for process, alias in (
        ("s s~ > g g", False),
        ("d~ d > g g", True),
        ("s~ s > g g", True),
    ):
        member = admit_member(representative.slice, process)
        assert member.alias is alias
        through = MemberObservables(representative, member)
        direct = generate(process, correlators=declarations)
        for options in (
            {"correlations": histories},
            {"spin": {3: VECTORS[0], 4: VECTORS[1]}},
        ):
            left = through.evaluate(point, "1000000", parameters=PARAMETERS, **options)
            right = direct.evaluate(point, "1000000", parameters=PARAMETERS, **options)
            for key in right:
                for a, b in zip(left[key].totals, right[key].totals, strict=True):
                    assert _close(a, b, 1e-12, 1e-14), (process, key)


def test_member_admission_refuses_unproved_substitutions(generate):
    representative = generate("d d~ > g g")
    with pytest.raises(ValueError, match="neither a proved flavour member"):
        admit_member(representative.slice, "t t~ > g g")
    with pytest.raises(ValueError):
        admit_member(representative.slice, "d d~ > d d~")


# ------------------------------------------------------------------ legacy (O1)

LEGACY_OBSERVABLES = FIXTURES / "legacy" / "observables"
LEGACY_TOLERANCE = {"double": 1e-12, "arbitrary-200": 1e-40}
# Binary64 O1 holds every output to 1e-12 of its 200-bit legacy value (gate
# S1), except these five-leg outputs (fixture, output id; "spin:<legs>" for a
# spin source) whose binary64 roundoff at the fixtures' first point exceeds
# it in every engine. Measured deviation of the binary64 value relative to the
# sector scale at that point (the second point stays below 3e-14):
#   uux_ggg T12:     v2 native 1.3e-12, v2 Python 8.7e-12, legacy 3.4e-12;
#   uux_ggg_nlc T13: v2 native 2.1e-12, v2 Python 8.0e-13, legacy 3.2e-13;
#   uux_ggg_spin 3:  v2 native 1.4e-13, v2 Python 5.3e-12, legacy 3.0e-12.
# Their binary64 values are held to their own stability estimate instead (it
# must bracket the loss and stay below 1e-9), and the same outputs run
# natively in DoubleFloat within DOUBLE_DOUBLE_TOLERANCE (measured 7.1e-27,
# 9.3e-29 and 3.0e-27), which shows that the loss is binary64 rounding.
BINARY64_LOSSES = frozenset(
    {("uux_ggg", "T12"), ("uux_ggg_nlc", "T13"), ("uux_ggg_spin", "spin:3")}
)
DOUBLE_DOUBLE_TOLERANCE = 1e-24


def _legacy(name: str) -> dict:
    return json.loads((LEGACY_OBSERVABLES / f"{name}.json").read_text())


def _legacy_runners(generate, record):
    """(process name, runner) of the representative and its members."""

    from pyamplicol.correlators import _emission_from_json

    histories = tuple(
        ColorCorrelator(
            item["id"],
            tuple(map(_emission_from_json, item["bra"])),
            tuple(map(_emission_from_json, item["ket"])),
        )
        for item in record["histories"]
    )
    (name, process), *members = record["processes"]
    born = {"QCD": len(process.split()) - 3, "QED": 0}
    options = {}
    if record.get("accuracy", "full") != "full":
        from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

        options = {
            "accuracy": record["accuracy"],
            "colour_declaration": LoopColorDeclaration.from_dict(
                json.loads((ROOT / record["colour_declaration"]).read_text())
            ),
        }
    observables = generate(
        process,
        born=born,
        loop={"QCD": born["QCD"] + 2, "QED": 0},
        correlators=CorrelatorConfig(
            color_correlations=histories,
            spin_correlations=tuple(map(tuple, record["spin_classes"])),
        ),
        **options,
    )
    runners = [(name, observables)]
    for member_name, member_process in members:
        member = admit_member(observables.slice, member_process)
        runners.append((member_name, MemberObservables(observables, member)))
    return observables, runners


def _bound(name: str, output: str, result, carrier: str) -> float:
    """The comparison bound of an output: the carrier's legacy tolerance, or
    for a named binary64 loss (``BINARY64_LOSSES``) the output's own
    stability estimate, which must exist and stay below 1e-9."""

    tolerance = LEGACY_TOLERANCE[carrier]
    if carrier != "double" or (name, output) not in BINARY64_LOSSES:
        return tolerance
    assert result.stability is not None and result.stability.relative < 1e-9
    return max(tolerance, result.stability.relative)


def _assert_correlated(result, expected, index, tolerance):
    """One legacy correlated record (Born per order, virtual per sector)."""

    borns = list(result.born.values())
    sectors = list(result.components["virtual"].values())
    assert len(borns) == len(expected["born"][index])
    assert len(sectors) == len(expected["virtual"][index])
    tolerance = Decimal(str(tolerance))
    scale = max(_magnitude(x) for x in expected["born"][index])
    for actual, value in zip(borns, expected["born"][index], strict=True):
        assert _deviation(actual, value) <= tolerance * scale
    for laurent, values in zip(sectors, expected["virtual"][index], strict=True):
        scale = max(_magnitude(x) for x in values)
        for actual, value in zip(laurent.coefficients, values, strict=True):
            assert _deviation(actual, value) <= tolerance * scale


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize(
    "name",
    (
        "ddgg_ssgg",
        "gg_gg",
        "uux_ggg",
        "uux_ggg_spin",
        "gg_gg_lc",
        "gg_gg_nlc",
        "uux_ggg_lc",
        "uux_ggg_nlc",
    ),
)
def test_directed_and_spin_outputs_equal_legacy(generate, name, carrier):
    """O1: every legacy directed history (dipoles, Casimirs, ordered two-
    emission histories and their swaps, complex) and every joint complex spin
    source, for the representative and its flavour members, against the
    200-bit legacy values; at LC and NLC the histories' complete-family
    degrees (design §7.7) and the projected values. In binary64 the
    representative's directed histories also run natively; every output is
    held to 1e-12, except the named five-leg binary64 losses
    (``BINARY64_LOSSES``), which are held to their own rotation and
    reduction-direction estimate in binary64 and to 1e-24 natively in
    DoubleFloat."""

    from pyamplicol.loops.v2.observables import NativeObservables
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

    checks = None
    if carrier == "double" and any(entry == name for entry, _ in BINARY64_LOSSES):
        checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)

    record = _legacy(name)
    observables, runners = _legacy_runners(generate, record)
    precision = CARRIERS[carrier]
    vectors = record["spin_vectors"]
    process = runners[0][0]
    directed = record["results"]["arbitrary-200"][process].get("directed", {})
    if carrier == "double" and directed:
        native = NativeObservables.correlated(observables, tuple(directed)).evaluate(
            record["points"],
            record["scales_squared"],
            parameters=PARAMETERS,
            stability=checks,
        )
        for index, results in enumerate(native):
            for history, values in directed.items():
                result = results[history]
                bound = _bound(name, history, result, carrier)
                _assert_correlated(result, values, index, bound)
    for process, runner in runners:
        expected = record["results"]["arbitrary-200"][process]
        for index, (point, scale) in enumerate(
            zip(record["points"], record["scales_squared"], strict=True)
        ):
            directed = expected.get("directed", {})
            if directed:
                results = runner.evaluate(
                    point,
                    scale,
                    correlations=tuple(directed),
                    parameters=PARAMETERS,
                    precision=precision,
                    stability=checks,
                )
                for history, values in directed.items():
                    result = results[history]
                    bound = _bound(name, history, result, carrier)
                    _assert_correlated(result, values, index, bound)
            for legs, values in expected.get("spin", {}).items():
                spin = {int(leg): vectors[k] for k, leg in enumerate(legs.split(","))}
                (result,) = runner.evaluate(
                    point,
                    scale,
                    spin=spin,
                    parameters=PARAMETERS,
                    precision=precision,
                    stability=checks,
                ).values()
                bound = _bound(name, f"spin:{legs}", result, carrier)
                _assert_correlated(result, values, index, bound)
    if carrier != "double":
        return
    expected = record["results"]["arbitrary-200"][runners[0][0]]
    for entry, output in sorted(BINARY64_LOSSES):
        if entry != name:
            continue
        options = {"parameters": PARAMETERS, "precision": Precision.double_double()}
        if output.startswith("spin:"):
            legs = output.removeprefix("spin:")
            values = expected["spin"][legs]
            options["spin"] = {
                int(leg): vectors[k] for k, leg in enumerate(legs.split(","))
            }
            history = INCLUSIVE
        else:
            values, history = expected["directed"][output], output
        evaluated = NativeObservables.correlated(observables, (history,)).evaluate(
            record["points"], record["scales_squared"], **options
        )
        for index, results in enumerate(evaluated):
            _assert_correlated(results[history], values, index, DOUBLE_DOUBLE_TOLERANCE)


def test_inclusively_pruned_slice_refuses_directed_histories(
    loop_sm_compiled_model, loop_native
):
    """An LC slice pruned for the inclusive virtual alone may lack skeletons
    a directed history needs: ``ObservableSlice.of`` refuses it (the
    generator prunes with the histories' family tables instead)."""

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "g g > g g",
            QCD2,
            QCD4,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            accuracy="lc",
            colour_declaration=declaration,
        ),
        loop_native,
    )
    assert slice_.colour_pruned
    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
    )
    with pytest.raises(ValueError, match="pruned for the inclusive virtual"):
        ObservableSlice.of(slice_, correlators=declarations)


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg"))
def test_helicity_resolved_outputs_equal_legacy(generate, name, carrier):
    """O1: per Born helicity row, the Born and the virtual summed over colour
    components equal the 200-bit legacy resolved outputs, in binary64 and at
    200 bits (the legacy colour anchors are diagram colour tensors, v2's are
    Born basis elements, so the colour axis is compared through its sum,
    design §1.6 decision 3)."""

    record = _legacy(name)
    observables, _runners = _legacy_runners(generate, record)
    expected = record["results"]["arbitrary-200"][record["processes"][0][0]]
    resolved_record = expected["resolved"]
    legacy_rows = [tuple(row) for row in resolved_record["helicities"]]
    tolerance = Decimal(str(LEGACY_TOLERANCE[carrier]))
    for index, (point, scale) in enumerate(
        zip(record["points"], record["scales_squared"], strict=True)
    ):
        resolved = observables.evaluate_resolved(
            point,
            scale,
            resolution="helicity",
            parameters=PARAMETERS,
            precision=CARRIERS[carrier],
        )
        mine = {
            tuple(h for _leg, h in row): value
            for (row, _b), value in resolved.entries.items()
        }
        born_scale = max(
            _magnitude(x) for order in resolved_record["born"][index] for x in order
        )
        virtual_scale = max(
            _magnitude(x)
            for sector in resolved_record["virtual"][index]
            for row in sector
            for x in row
        )
        for h, row in enumerate(legacy_rows):
            value = mine.get(row)
            expected_born = [order[h] for order in resolved_record["born"][index]]
            expected_virtual = [
                sector[h] for sector in resolved_record["virtual"][index]
            ]
            if value is None:
                # A row v2 does not generate is a structural zero.
                assert max(map(_magnitude, expected_born)) <= tolerance * born_scale
                continue
            for actual, target in zip(value.born.values(), expected_born, strict=True):
                assert _deviation(actual, target) <= tolerance * born_scale
            sectors = list(value.components["virtual"].values())
            for laurent, targets in zip(sectors, expected_virtual, strict=True):
                for actual, target in zip(laurent.coefficients, targets, strict=True):
                    assert _deviation(actual, target) <= tolerance * virtual_scale


# ------------------------------------------------------------------ public surface


@pytest.mark.parametrize("executor", ("native", "python"))
def test_public_runtime_surface(generate, loop_reference, executor):
    """``runtime.virtual.V2VirtualRuntime`` answers the public calls with the
    public result types, natively (the default) or with the Python reference
    executor: ordinary, correlated (colour and spin, grouped by source
    assignment, per-point source vectors) and resolved evaluations, in
    binary64 and arbitrary precision (and natively in DoubleFloat)."""

    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.runtime.virtual import V2VirtualRuntime

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),),
        spin_correlations=((3,),),
    )
    observables = generate("u u~ > g g", correlators=declarations)
    runtime = V2VirtualRuntime(
        observables, model_parameters=PARAMETERS, executor=executor
    )
    rows = loop_reference("spin-recola-control")["rows"]
    points = [row["point"] for row in rows]
    scales = [row["mu_r2"] for row in rows]
    assert [item.id for item in runtime.available_color_correlations()] == [
        "born",
        "T12",
    ]
    ordinary = runtime.evaluate(points, mu_r2=scales)
    assert ordinary.shape == (2, 1, 3) and not ordinary.correlated
    assert executor in ordinary.diagnostics[0].detail.lower()
    grouped = runtime.evaluate_correlated_many(
        points,
        {
            "spin": CorrelatedRequest("born", {3: [row["vector"] for row in rows]}),
            "T12": CorrelatedRequest("T12", {}),
            "born": CorrelatedRequest("born", {}),
        },
        mu_r2=scales,
    )
    assert list(grouped) == ["spin", "T12", "born"]
    for index, row in enumerate(rows):
        spin = grouped["spin"]
        values = (spin.born_values[index][0], *spin.values[index][0])
        for actual, expected in zip(values, row["cepsilon"], strict=True):
            assert _close(actual, expected, 3e-11, 2e-13)
        assert grouped["born"].values[index] == tuple(
            tuple(complex(x) for x in sector) for sector in ordinary.values[index]
        )
    resolved = runtime.evaluate_resolved(points[:1], mu_r2=scales[:1])
    assert resolved.shape == (1, 1, 16, len(observables.components), 3)
    for a, b in zip(resolved.total().values[0][0], ordinary.values[0][0], strict=True):
        assert _close(a, b, 1e-12, 1e-14)
    carriers = [("arbitrary", 40)]
    if executor == "native":
        carriers.append(("double-double", None))
    else:
        with pytest.raises(CompatibilityError, match="double-double"):
            runtime.evaluate(points[:1], mu_r2=scales[:1], arithmetic="double-double")
    for arithmetic, digits in carriers:
        precise = runtime.evaluate(
            points[:1], mu_r2=scales[:1], arithmetic=arithmetic, precision=digits
        )
        assert precise.diagnostics[0].arithmetic == arithmetic
        for a, b in zip(precise.values[0][0], ordinary.values[0][0], strict=True):
            assert _close(float(a), b, 1e-13)


def test_public_runtime_members_run_natively(generate):
    """A flavour member and a permutation alias through the public runtime
    (native executor) equal the runtime of the process generated directly:
    directed histories, per-point spin sources and resolved rows (an alias's
    colour components are its representative's Born basis elements, so its
    rows are compared summed over them)."""

    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.runtime.virtual import V2VirtualRuntime

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("casimir", 1, 1), ORDERED),
        spin_correlations=((3, 4),),
    )
    representative = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    requests = {
        "ordered": CorrelatedRequest("ordered", {}),
        "casimir": CorrelatedRequest("casimir", {}),
        "spin": CorrelatedRequest(
            "born", {3: [VECTORS[0], VECTORS[1]], 4: [VECTORS[1], VECTORS[0]]}
        ),
    }
    for process in ("s s~ > g g", "s~ s > g g"):
        member = admit_member(representative.slice, process)
        through = V2VirtualRuntime(
            representative, model_parameters=PARAMETERS, member=member
        )
        direct = V2VirtualRuntime(
            generate(process, correlators=declarations), model_parameters=PARAMETERS
        )
        left = through.evaluate_correlated_many(points, requests, mu_r2=scales)
        right = direct.evaluate_correlated_many(points, requests, mu_r2=scales)
        for label in requests:
            for a, b in zip(left[label].values, right[label].values, strict=True):
                for x, y in zip(a[0], b[0], strict=True):
                    assert _close(x, y, 1e-12, 1e-14), (process, label)
        left = through.evaluate_resolved(points[:1], mu_r2=scales[:1]).values
        right = direct.evaluate_resolved(points[:1], mu_r2=scales[:1]).values
        if member.alias:
            # An alias keeps the representative's Born basis elements as its
            # colour components; compare the helicity rows summed over them.
            left, right = _colour_summed(left), _colour_summed(right)
        flat_left, flat_right = _flat(left), _flat(right)
        assert len(flat_left) == len(flat_right)
        largest = max(map(abs, flat_right))
        for a, b in zip(flat_left, flat_right, strict=True):
            assert abs(a - b) <= 1e-12 * largest, process


def _colour_summed(values) -> tuple:
    """Public resolved values [point][sector][row][component] summed over
    the colour components."""

    return tuple(
        tuple(
            tuple(
                tuple(sum(float(entry[k]) for entry in row) for k in range(3))
                for row in sector
            )
            for sector in point
        )
        for point in values
    )


def _flat(values) -> list[float]:
    """The scalars of nested public result tuples."""

    if isinstance(values, tuple):
        return [x for value in values for x in _flat(value)]
    return [float(values)]


# ------------------------------------------------------------------ O14


def _rank(vectors: list[list[Fraction]]) -> int:
    """The rank of rational vectors (exact elimination)."""

    rows: list[tuple[int, list[Fraction]]] = []
    for vector in vectors:
        vector = list(vector)
        for lead, row in rows:
            if vector[lead]:
                factor = vector[lead] / row[lead]
                vector = [a - factor * b for a, b in zip(vector, row, strict=True)]
        lead = next((k for k, value in enumerate(vector) if value), None)
        if lead is not None:
            rows.append((lead, vector))
    return len(rows)


def _inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """The exact inverse of a nonsingular rational matrix (Gauss-Jordan)."""

    size = len(matrix)
    rows = [
        [*row, *(Fraction(int(i == j)) for j in range(size))]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        lead = rows[column][column]
        rows[column] = [value / lead for value in rows[column]]
        for r in range(size):
            if r != column and rows[r][column]:
                factor = rows[r][column]
                rows[r] = [
                    a - factor * b for a, b in zip(rows[r], rows[column], strict=True)
                ]
    return [row[size:] for row in rows]


def _generalised_inverse(gram: list[list[Fraction]]) -> list[list[Fraction]]:
    """An exact generalised inverse M (G M G = G) of a symmetric positive
    semidefinite rational matrix: the inverse of its principal submatrix on a
    maximal independent column set, zero elsewhere."""

    size = len(gram)
    columns = [[gram[row][column] for row in range(size)] for column in range(size)]
    chosen: list[int] = []
    for column in range(size):
        if _rank([columns[c] for c in (*chosen, column)]) > len(chosen):
            chosen.append(column)
    inverse = _inverse([[gram[a][b] for b in chosen] for a in chosen])
    out = [[Fraction(0)] * size for _ in range(size)]
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            out[a][b] = inverse[i][j]
    return out


def test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14(generate):
    """O14 with resolved-colour seed columns (design §7.6, §14), at 200 bits:
    the projections of v2's bare loop amplitude of qbar q g g g onto BDK's
    one-loop colour structures (``ObservableSlice.project_loop``) determine
    partial amplitudes A_{5;1}, A_{5;3}, A_{5;4} through an exact generalised
    inverse of the N_c = 3 Gram matrix (which has null directions; the
    projections lie in its range to the carrier's rounding). With the tree
    partial amplitudes of v2's Born, BDK's colour sum (N^2-1)/N sum_s
    A_tree(s)^* bracket(s) (hep-ph/9409393 eq. NLOCrossSection) reproduces
    v2's helicity-resolved bare virtual. Gram, tree and BDK matrices are
    exact; every product runs at the carrier's working digits."""

    from decimal import Decimal, localcontext

    from pyamplicol.color.connections import (
        contract_connected_tensor_nc_terms,
        evaluate_color_nc_terms,
    )
    from pyamplicol.loops.v2.assemble import _single

    from .test_loop_v2_colour import (
        BDK_PREFACTOR,
        _bdk_bracket,
        _bdk_structures,
        _laurent_times,
    )

    observables = generate(
        "u~ u > g g g", born={"QCD": 3, "QED": 0}, loop={"QCD": 5, "QED": 0}
    )
    slice_ = observables.slice
    tree, loop = _bdk_structures()
    assert next(iter(tree.values())).legs == slice_.basis.legs
    fixture = json.loads((FIXTURES / "legacy/uux_ggg.json").read_text())
    point = list(fixture["points"][0])
    point[0], point[1] = point[1], point[0]
    scale = fixture["scales_squared"][0]
    precision = CARRIERS["arbitrary-200"]

    def at_three(cell) -> Fraction:
        value = evaluate_color_nc_terms(cell)
        assert not value.imag
        return value.real

    trees, loops = list(tree), list(loop)
    gram_tree = [
        [at_three(contract_connected_tensor_nc_terms(tree[a], tree[b])) for b in trees]
        for a in trees
    ]
    gram_loop = [
        [at_three(contract_connected_tensor_nc_terms(loop[a], loop[b])) for b in loops]
        for a in loops
    ]
    bdk = [
        [
            sum(
                (
                    v * Fraction(3) ** p
                    for p, v in _laurent_times(
                        _bdk_bracket(s).get(key, {}), BDK_PREFACTOR
                    ).items()
                ),
                Fraction(0),
            )
            for key in loops
        ]
        for s in trees
    ]
    tree_inverse = _inverse(gram_tree)
    loop_inverse = _generalised_inverse(gram_loop)
    assert _rank(gram_loop) < len(loops)
    evaluation = observables.point(point, scale, PARAMETERS, precision)
    with localcontext(prec=precision.working_digits):

        def number(value):
            if isinstance(value, Fraction):
                return Decimal(value.numerator) / value.denominator
            return value

        def apply(matrix, vector):
            """A rational matrix times a vector of (real, imaginary) Decimals."""

            return [
                (
                    sum(
                        (number(m) * x for m, (x, _y) in zip(row, vector, strict=True)),
                        Decimal(0),
                    ),
                    sum(
                        (number(m) * y for m, (_x, y) in zip(row, vector, strict=True)),
                        Decimal(0),
                    ),
                )
                for row in matrix
            ]

        def size(vector):
            return max(abs(x) + abs(y) for x, y in vector)

        def tree_amplitudes(row):
            """Tree partial amplitudes: Born = sum_s a_s tree_s."""

            projected = []
            for s in trees:
                re = im = Decimal(0)
                for vector in evaluation.born_vectors[row].values():
                    for (tensor, power), value in vector.items():
                        overlap = number(
                            at_three(
                                contract_connected_tensor_nc_terms(
                                    tree[s], _single(tensor, slice_.basis.legs)
                                )
                            )
                            * Fraction(3) ** power
                        )
                        re += overlap * value.real
                        im += overlap * value.imag
                projected.append((re, im))
            return apply(tree_inverse, projected)

        amplitudes = {row: tree_amplitudes(row) for row in observables.rows}
        largest = max(size(a) for a in amplitudes.values())
        rows = [row for row, a in amplitudes.items() if size(a) > largest / 1000][:2]
    assert len(rows) == 2
    projections = observables.project_loop(
        point,
        scale,
        [loop[key] for key in loops],
        rows=rows,
        parameters=PARAMETERS,
        precision=precision,
    )
    resolved = observables.evaluate_resolved(
        point, scale, resolution="helicity", parameters=PARAMETERS, precision=precision
    )
    (order,) = {label[2] for label in projections}
    with localcontext(prec=precision.working_digits):
        for row in rows:
            (bare,) = resolved.entries[row, None].components["bare"].values()
            expected = []
            for k in range(3):
                target = [
                    (
                        Decimal(projections[row, j, order].coefficients[k].real),
                        Decimal(projections[row, j, order].coefficients[k].imag),
                    )
                    for j in range(len(loops))
                ]
                partial = apply(loop_inverse, target)
                residual = [
                    (a - c, b - d)
                    for (a, b), (c, d) in zip(
                        apply(gram_loop, partial), target, strict=True
                    )
                ]
                assert size(residual) <= Decimal("1e-45") * size(target)
                contracted = apply(bdk, partial)
                interference = sum(
                    (
                        a * c + b * d
                        for (a, b), (c, d) in zip(
                            amplitudes[row], contracted, strict=True
                        )
                    ),
                    Decimal(0),
                )
                expected.append(2 * interference / slice_.normalisation.average)
            scale_ = max(abs(x) for x in expected)
            for actual, value in zip(bare.coefficients, expected, strict=True):
                assert abs(Decimal(actual) - value) <= Decimal("1e-40") * scale_, row


# ------------------------------------------------------------------ native


@pytest.mark.parametrize("carrier", ("double", "double-double", "arbitrary-200"))
def test_native_observables_equal_python(generate, carrier):
    """The native executor runs the same observable columns (exported seed
    weights; ``column_rows`` for helicity-resolved columns): directed
    histories (ordered, complex), spin sources and helicity x colour entries
    equal the Python reference executor (200 bits for the double-double
    carrier, whose trees are formed at its working digits and dressed and
    reduced in DoubleFloat)."""

    from pyamplicol.loops.v2.observables import NativeObservables

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("casimir", 1, 1), ORDERED),
        spin_correlations=((3,),),
    )
    observables = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    reference_precision = CARRIERS["double" if carrier == "double" else "arbitrary-200"]
    precision = (
        Precision.double_double() if carrier == "double-double" else reference_precision
    )
    tolerance = {"double": 1e-12, "double-double": 1e-26, "arbitrary-200": 1e-50}[
        carrier
    ]
    histories = (INCLUSIVE, "casimir", "ordered")
    native = NativeObservables.correlated(observables, histories, lanes=4)
    for spin in (None, {3: VECTORS[0]}):
        results = native.evaluate(
            points,
            scales,
            spin=spin,
            parameters=PARAMETERS,
            precision=precision,
        )
        for point, scale, result in zip(points, scales, results, strict=True):
            reference = observables.evaluate(
                point,
                scale,
                correlations=histories,
                spin=spin,
                parameters=PARAMETERS,
                precision=reference_precision,
            )
            for key, value in reference.items():
                for a, b in zip(result[key].totals, value.totals, strict=True):
                    assert _close(a, b, tolerance, tolerance * 1e-2), (key, spin)
    resolved = NativeObservables.resolved(observables, lanes=4).evaluate(
        points[:1], scales[:1], parameters=PARAMETERS, precision=precision
    )[0]
    reference = observables.evaluate_resolved(
        points[0], scales[0], parameters=PARAMETERS, precision=reference_precision
    )
    largest = max(_magnitude(x) for v in reference.entries.values() for x in v.totals)
    for key, value in reference.entries.items():
        for a, b in zip(resolved[key].totals, value.totals, strict=True):
            assert _deviation(a, b) <= Decimal(str(tolerance)) * largest, key


def test_native_observables_run_schedule_m(generate):
    """Observable columns on a plan with a schedule-M section (WP2.8): on
    g g > g g, whose families schedule M certifies, a dipole, an ordered
    directed history (direct and swapped columns) and the helicity-resolved
    columns (``column_rows``) run with M as with B, at 32 digits to 1e-25
    and in binary64 to 1e-12 of the point's largest output."""

    from pyamplicol.loops.v2.merged import merged_graph
    from pyamplicol.loops.v2.observables import NativeObservables

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2), ORDERED)
    )
    observables = generate("g g > g g", correlators=declarations)
    assert merged_graph(observables.slice).families
    points = _points("gluon_scattering_qcd")[:1]
    scales = ["1000000"]
    histories = (INCLUSIVE, "T12", "ordered")
    for precision, tolerance in (
        (Precision.arbitrary(32), Decimal("1e-25")),
        (Precision.double(), Decimal("1e-12")),
    ):
        options = {"parameters": PARAMETERS, "precision": precision}
        runs = {}
        for schedule in "BM":
            correlated = NativeObservables.correlated(
                observables, histories, lanes=4, schedule=schedule
            )
            resolved = NativeObservables.resolved(
                observables, "helicity", lanes=4, schedule=schedule
            )
            runs[schedule] = [
                {**c, **r}
                for c, r in zip(
                    correlated.evaluate(points, scales, **options),
                    resolved.evaluate(points, scales, **options),
                    strict=True,
                )
            ]
        for b, m in zip(runs["B"], runs["M"], strict=True):
            assert set(b) == set(m) and len(b) > len(histories)
            largest = max(_magnitude(x) for value in b.values() for x in value.totals)
            for key, value in b.items():
                for x, y in zip(m[key].totals, value.totals, strict=True):
                    assert _deviation(x, y) <= tolerance * largest, (key, precision)


def test_native_members_per_point_sources_and_stability(generate):
    """Natively: a permutation alias through its representative equals the
    Python member evaluation (directed histories, one spin source per point,
    resolved rows); the native rotation and reduction-direction estimates
    bound the deviation from the 200-bit evaluation."""

    from pyamplicol.loops.v2.observables import NativeObservables
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("casimir", 1, 1), ORDERED),
        spin_correlations=((3,),),
    )
    representative = generate("d d~ > g g", correlators=declarations)
    member = admit_member(representative.slice, "s~ s > g g")
    python = MemberObservables(representative, member)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    histories = (INCLUSIVE, "casimir", "ordered")
    native = NativeObservables.correlated(
        representative, histories, member=member, lanes=4
    )
    spins = [{3: VECTORS[0]}, {3: VECTORS[1]}]
    for spin in (None, spins):
        results = native.evaluate(points, scales, spin=spin, parameters=PARAMETERS)
        for index, (point, result) in enumerate(zip(points, results, strict=True)):
            reference = python.evaluate(
                point,
                "1000000",
                correlations=histories,
                spin=None if spin is None else spin[index],
                parameters=PARAMETERS,
            )
            for key, value in reference.items():
                for a, b in zip(result[key].totals, value.totals, strict=True):
                    assert _close(a, b, 1e-12, 1e-14), (key, spin)
    resolved = NativeObservables.resolved(
        representative, "helicity", member=member, lanes=4
    ).evaluate(points[:1], scales[:1], parameters=PARAMETERS)[0]
    reference = python.evaluate_resolved(
        points[0], "1000000", resolution="helicity", parameters=PARAMETERS
    )
    assert resolved.keys() == reference.entries.keys()
    for key, value in reference.entries.items():
        for a, b in zip(resolved[key].totals, value.totals, strict=True):
            assert _close(a, b, 1e-12, 1e-14), key
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
    checked = NativeObservables.correlated(representative, histories, lanes=4)
    for spin in (None, spins):
        results = checked.evaluate(
            points, scales, spin=spin, parameters=PARAMETERS, stability=checks
        )
        for index, (point, result) in enumerate(zip(points, results, strict=True)):
            exact = representative.evaluate(
                point,
                "1000000",
                correlations=histories,
                spin=None if spin is None else spin[index],
                parameters=PARAMETERS,
                precision=CARRIERS["arbitrary-200"],
            )
            for key, value in result.items():
                estimate = value.stability
                assert estimate is not None and 0 < estimate.relative < 1e-9
                for sector, laurent in value.components["virtual"].items():
                    target = exact[key].components["virtual"][sector].coefficients
                    size = max(abs(complex(x)) for x in target)
                    error = max(
                        abs(complex(a) - complex(b))
                        for a, b in zip(laurent.coefficients, target, strict=True)
                    )
                    assert error <= estimate.relative * size, (key, error / size)


# ------------------------------------------------------------------ stability


NATIVE_CARRIERS = {
    "double": Precision.double(),
    "double-double": Precision.double_double(),
    "arbitrary-40": Precision.arbitrary(40),
}
# Largest relative estimate a well-conditioned 2 -> 2 point may report.
ESTIMATE_CEILING = {"double": 1e-9, "double-double": 1e-24, "arbitrary-40": 1e-30}


def _assert_brackets(result, exact, ceiling: float) -> None:
    """Every output's estimate is positive, below ``ceiling`` and bounds the
    deviation from the 200-bit value (measured in Decimal arithmetic)."""

    for key, value in result.items():
        estimate = value.stability
        assert estimate is not None and 0 < estimate.relative < ceiling, key
        for sector, laurent in value.components["virtual"].items():
            target = exact[key].components["virtual"][sector].coefficients
            size = max(_magnitude(x) for x in target)
            error = max(
                _deviation(a, b)
                for a, b in zip(laurent.coefficients, target, strict=True)
            )
            assert error <= Decimal(estimate.relative) * size, (key, error / size)


@pytest.mark.parametrize("carrier", ("double", "arbitrary-40"))
def test_observable_stability_estimates_bracket_the_true_error(generate, carrier):
    """Rotation and reduction-direction checks per observable output
    (REQUIREMENTS_ADDENDUM C) in the Python reference executor: the estimate
    of a directed ordered history, a dipole and a rotated complex spin source
    bounds the deviation from the 200-bit evaluation, in binary64 and at 40
    digits (where it stays near the carrier's roundoff); without checks no
    estimate exists."""

    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2), ORDERED),
        spin_correlations=((3,),),
    )
    observables = generate("d d~ > g g", correlators=declarations)
    precision = NATIVE_CARRIERS[carrier]
    checks = StabilityChecks(
        (
            Rotation.about("z", "3/5", "4/5"),
            Rotation.about("x", "-4/5", "3/5") @ Rotation.about("y", "5/13", "12/13"),
        ),
        True,
    )
    for point in _points("uudd_qcd")[:2]:
        for options in (
            {"correlations": ("T12", "ordered")},
            {"spin": {3: VECTORS[0]}},
        ):
            plain = observables.evaluate(
                point, "1000000", parameters=PARAMETERS, precision=precision, **options
            )
            assert all(value.stability is None for value in plain.values())
            checked = observables.evaluate(
                point,
                "1000000",
                parameters=PARAMETERS,
                precision=precision,
                stability=checks,
                **options,
            )
            exact = observables.evaluate(
                point,
                "1000000",
                parameters=PARAMETERS,
                precision=CARRIERS["arbitrary-200"],
                **options,
            )
            _assert_brackets(checked, exact, ESTIMATE_CEILING[carrier])


class _FailingPoint:
    """A native executor whose evaluations after the first ``after`` report
    a failed reduction lane for point ``index`` (as an ``ImpreciseResult``
    master does)."""

    def __init__(self, inner, index: int, *, after: int = 0):
        self.inner = inner
        self.index = index
        self.after = after
        self.calls = 0

    def evaluate(self, inputs, **options):
        output = self.inner.evaluate(inputs, **options)
        self.calls += 1
        if self.calls > self.after:
            failures = [list(f) for f in output["failures"]]
            failures[self.index] = ["lane 0: ImpreciseResult (injected)"]
            output = dict(output, failures=failures)
        return output

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _assert_budgets_equal(mine, theirs, unit=None) -> None:
    """Two absolute budgets (sector -> ``Budget``) agree coefficient by
    coefficient to 1e-12, plus ``unit`` (sector -> ``Budget``) where given."""

    assert set(mine) == set(theirs)
    for sector, budget in theirs.items():
        width = len(budget.coefficients)
        slack = (0.0,) * width if unit is None else unit[sector].coefficients
        for a, b, c in zip(
            mine[sector].coefficients, budget.coefficients, slack, strict=True
        ):
            assert abs(a - b) <= 1e-12 * abs(b) + c, (sector, a, b)


@pytest.mark.parametrize("carrier", tuple(NATIVE_CARRIERS))
def test_native_observable_estimates_follow_the_stability_api(generate, carrier):
    """The native observables share the integrated stability API (WP2.3)
    in every carrier: for every combination of a rotation and the cached or
    rescaled reduction check, an inclusive observable column has the
    estimate of the native inclusive evaluation (check names, absolute sums
    S, every check's budget per sector and Laurent coefficient, the combined
    estimate, the unit roundoff and the reduction lanes), and the inclusive
    and directed estimates of a two-column plan bracket the deviation from
    200 bits; zero checks give no estimate (explicitly absent)."""

    from pyamplicol.loops.v2.observables import NativeObservables
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import (
        REDUCTION,
        RESCALED,
        ROTATION,
        Budget,
        Rotation,
        StabilityChecks,
    )

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
    )
    observables = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    precision = NATIVE_CARRIERS[carrier]
    alone = NativeObservables.correlated(observables, (INCLUSIVE,), lanes=4)
    inclusive = NativeExecutor(observables.slice, lanes=4)
    rotation = Rotation.about("z", "3/5", "4/5")
    for checks, names in (
        (StabilityChecks((rotation,)), {ROTATION.format(0)}),
        (StabilityChecks(reduction_direction=True), {REDUCTION}),
        (StabilityChecks((rotation,), True), {ROTATION.format(0), REDUCTION}),
        (
            StabilityChecks((rotation,), True, rescaled=True),
            {ROTATION.format(0), RESCALED},
        ),
    ):
        options = {
            "parameters": PARAMETERS,
            "precision": precision,
            "stability": checks,
        }
        results = alone.evaluate(points, scales, **options)
        references = inclusive.evaluate(points, scales, **options)
        for result, reference in zip(results, references, strict=True):
            mine, theirs = result[INCLUSIVE].stability, reference.stability
            assert set(mine.checks) == set(theirs.checks) == names
            assert mine.roundoff == theirs.roundoff == precision.unit_roundoff
            _assert_budgets_equal(mine.scale, theirs.scale)
            # The reduction lanes are the same native lanes; the local
            # components are assembled by two Python paths in different
            # orders, so their rotation differences (of roundoff size) may
            # differ by one unit roundoff of the absolute sums S.
            unit = {
                sector: Budget(
                    tuple(precision.unit_roundoff * x for x in budget.coefficients)
                )
                for sector, budget in theirs.scale.items()
            }
            for name in names:
                _assert_budgets_equal(mine.budgets[name], theirs.budgets[name], unit)
            _assert_budgets_equal(mine.absolute, theirs.absolute, unit)
            assert (mine.families is None) == (theirs.families is None)
            if theirs.families is not None:
                assert len(mine.families) == len(theirs.families)
    exact = [
        observables.evaluate(
            point,
            scale,
            correlations=(INCLUSIVE, "T12"),
            parameters=PARAMETERS,
            precision=CARRIERS["arbitrary-200"],
        )
        for point, scale in zip(points, scales, strict=True)
    ]
    native = NativeObservables.correlated(observables, (INCLUSIVE, "T12"), lanes=4)
    checked = native.evaluate(
        points,
        scales,
        parameters=PARAMETERS,
        precision=precision,
        stability=StabilityChecks((rotation,), True, rescaled=True),
    )
    for result, truth in zip(checked, exact, strict=True):
        _assert_brackets(result, truth, ESTIMATE_CEILING[carrier])
    for checks in (None, StabilityChecks()):
        plain = native.evaluate(
            points, scales, parameters=PARAMETERS, precision=precision, stability=checks
        )
        assert all(v.stability is None for result in plain for v in result.values())


class _ShiftedRotation:
    """A native executor whose evaluations after the first scale every
    reduced lane by 1 + 1e-20, far below one binary64 unit of its value."""

    FACTOR = Decimal("1e-20") + 1

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def evaluate(self, inputs, **options):
        output = self.inner.evaluate(inputs, **options)
        self.calls += 1
        if self.calls == 1:
            return output

        def shifted(value) -> str:
            with localcontext(prec=COMPARISON_DIGITS):
                return str(Decimal(str(value)) * self.FACTOR)

        families = [
            [
                (lane, [(shifted(re), shifted(im)) for re, im in values])
                for lane, values in lanes
            ]
            for lanes in output["families"]
        ]
        return dict(output, families=families)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("carrier", ("double-double", "arbitrary-40"))
def test_rotation_budgets_resolve_deviations_below_binary64(generate, carrier):
    """Outside binary64 the rotation budgets of the observables and of the
    inclusive API (``assemble._PointEvaluation.rotation_budget``) subtract
    in the carrier at its working digits: without a deviation a rotation
    check stays near the carrier's roundoff (no binary64 rounding, no
    28-digit Decimal floor), and a relative shift of 1e-20 of every rotated
    lane, invisible in binary64, shows as a budget of order 1e-20."""

    from pyamplicol.loops.v2.observables import NativeObservables
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import ROTATION, Rotation, StabilityChecks

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
    )
    observables = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:1]
    precision = NATIVE_CARRIERS[carrier]
    options = {
        "parameters": PARAMETERS,
        "precision": precision,
        "stability": StabilityChecks((Rotation.about("z", "3/5", "4/5"),)),
    }
    native = NativeObservables.correlated(observables, (INCLUSIVE, "T12"), lanes=4)
    inclusive = NativeExecutor(observables.slice, lanes=4)
    runs = (
        (native.executor, lambda: native.evaluate(points, ["1000000"], **options)[0]),
        (
            inclusive,
            lambda: {INCLUSIVE: inclusive.evaluate(points, ["1000000"], **options)[0]},
        ),
    )
    name = ROTATION.format(0)
    for executor, run in runs:
        for value in run().values():
            assert value.stability.checks[name] < 1e-29
        arithmetic, digits = executor._arithmetic(precision, False)
        key = (arithmetic, digits, executor.lanes_for(arithmetic, 1))
        inner = executor.native(arithmetic, digits, 1)
        executor._native[key] = _ShiftedRotation(inner)
        try:
            shifted = run()
        finally:
            executor._native[key] = inner
        for value in shifted.values():
            assert 1e-21 < value.stability.checks[name] < 1e-18


@pytest.mark.parametrize("after", [0, 1], ids=["reference", "rotation"])
def test_a_failed_observable_point_fails_alone(generate, after):
    """``NativeObservables.evaluate`` follows ``NativeExecutor.evaluate``: a
    failed lane fails its point only, in the reference evaluation or in a
    rotation check's; ``FailedPoints`` keeps the other points' complete
    outputs (estimates included) and invents none for the failed one."""

    from pyamplicol.loops.v2.observables import NativeObservables
    from pyamplicol.loops.v2.plan_export import FailedPoints
    from pyamplicol.loops.v2.stability import ROTATION, Rotation, StabilityChecks

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),),
        spin_correlations=((3,),),
    )
    observables = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    native = NativeObservables.correlated(observables, (INCLUSIVE, "T12"), lanes=4)
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
    spin = [{3: VECTORS[0]}, {3: VECTORS[1]}]
    options = {"parameters": PARAMETERS, "stability": checks, "spin": spin}
    clean = native.evaluate(points, scales, **options)
    executor = native.executor
    key = ("double", None, executor.lanes_for("double", len(points)))
    inner = executor.native("double", None, len(points))
    executor._native[key] = _FailingPoint(inner, 1, after=after)
    try:
        with pytest.raises(FailedPoints) as caught:
            native.evaluate(points, scales, **options)
    finally:
        executor._native[key] = inner
    failed = caught.value
    assert failed.results[1] is None and set(failed.failures) == {1}
    reason = failed.failures[1][0]
    assert "ImpreciseResult" in reason
    assert reason.startswith(ROTATION.format(0)) == bool(after)
    for name, value in failed.results[0].items():
        assert value.totals == clean[0][name].totals
        assert value.stability.relative == clean[0][name].stability.relative


def test_observables_refuse_a_required_seed_fft(generate):
    """Observable columns are contracted directly (design §7.3): a slice
    whose request requires the seed FFT is refused, not silently served by
    the direct contraction."""

    import dataclasses

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),)
    )
    slice_ = generate("g g > g g", correlators=declarations).slice
    required = dataclasses.replace(
        slice_, request=dataclasses.replace(slice_.request, colour_contraction="fft")
    )
    with pytest.raises(ValueError, match="contracted directly"):
        ObservableSlice.of(required, correlators=declarations)
