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
from fractions import Fraction
from pathlib import Path

import pytest

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


def _close(actual, expected, rel, floor=0.0) -> bool:
    actual, expected = complex(actual), complex(expected)
    return abs(actual - expected) <= rel * abs(expected) + floor


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
    directed Casimir column (C_F times each sector)."""

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
    results = observables.evaluate(
        point, "1000000", correlations=(INCLUSIVE, "casimir"), parameters=PARAMETERS
    )
    row = reference["rows"][0]
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


@pytest.mark.parametrize(
    "process,name,legs",
    [("g g > g g", "gg_gg", (1, 3)), ("u u~ > g g g", "uux_ggg", (3, 5))],
)
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_asymmetric_ward_identity(generate, process, name, legs, carrier):
    """O10: with epsilon_j -> p_j on the loop side only (blocks, counterterm
    currents, field insertions; full kernels) the renormalised interference
    with the physical Born vanishes per Laurent coefficient and sector."""

    born = {"QCD": 3, "QED": 0} if name == "uux_ggg" else QCD2
    loop = {"QCD": 5, "QED": 0} if name == "uux_ggg" else QCD4
    observables = generate(process, born=born, loop=loop, ward=True)
    fixture = json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    precision = CARRIERS[carrier]
    physical = observables.slice.evaluate(
        point, scale, parameters=PARAMETERS, precision=precision
    )
    reference = _ward_scale(physical)
    tolerance = 1e-11 if carrier == "double" else 1e-45
    for leg in legs:
        (ward,) = observables.evaluate(
            point, scale, ward=leg, parameters=PARAMETERS, precision=precision
        ).values()
        for sector, laurent in ward.components["virtual"].items():
            for k, value in enumerate(laurent.coefficients):
                assert abs(complex(value)) <= tolerance * reference[k], (leg, sector, k)
        # The test is not empty: the physical Born and bare amplitude are not.
        assert abs(float(ward.totals[0])) > 0


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


def _legacy(name: str) -> dict:
    return json.loads((LEGACY_OBSERVABLES / f"{name}.json").read_text())


def _pair(value) -> complex:
    return complex(float(value[0]), float(value[1]))


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
    observables = generate(
        process,
        born=born,
        loop={"QCD": born["QCD"] + 2, "QED": 0},
        correlators=CorrelatorConfig(
            color_correlations=histories,
            spin_correlations=tuple(map(tuple, record["spin_classes"])),
        ),
    )
    runners = [(name, observables)]
    for member_name, member_process in members:
        member = admit_member(observables.slice, member_process)
        runners.append((member_name, MemberObservables(observables, member)))
    return observables, runners


def _assert_correlated(result, expected, index, tolerance):
    """One legacy correlated record (Born per order, virtual per sector)."""

    borns = list(result.born.values())
    sectors = list(result.components["virtual"].values())
    assert len(borns) == len(expected["born"][index])
    assert len(sectors) == len(expected["virtual"][index])
    scale = max(abs(_pair(x)) for x in expected["born"][index])
    for actual, value in zip(borns, expected["born"][index], strict=True):
        assert abs(complex(actual) - _pair(value)) <= tolerance * scale
    for laurent, values in zip(sectors, expected["virtual"][index], strict=True):
        scale = max(abs(_pair(x)) for x in values)
        for actual, value in zip(laurent.coefficients, values, strict=True):
            assert abs(complex(actual) - _pair(value)) <= tolerance * scale


@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", ("ddgg_ssgg", "gg_gg", "uux_ggg"))
def test_directed_and_spin_outputs_equal_legacy(generate, name, carrier):
    """O1: every legacy directed history (dipoles, Casimirs, ordered two-
    emission histories and their swaps, complex) and every joint complex spin
    source, for the representative and its flavour members, against the
    200-bit legacy values."""

    record = _legacy(name)
    _observables, runners = _legacy_runners(generate, record)
    precision, tolerance = CARRIERS[carrier], LEGACY_TOLERANCE[carrier]
    vectors = record["spin_vectors"]
    for process, runner in runners:
        expected = record["results"]["arbitrary-200"][process]
        for index, (point, scale) in enumerate(
            zip(record["points"], record["scales_squared"], strict=True)
        ):
            directed = expected.get("directed", {})
            results = runner.evaluate(
                point,
                scale,
                correlations=tuple(directed),
                parameters=PARAMETERS,
                precision=precision,
            )
            for history, values in directed.items():
                _assert_correlated(results[history], values, index, tolerance)
            for legs, values in expected.get("spin", {}).items():
                spin = {int(leg): vectors[k] for k, leg in enumerate(legs.split(","))}
                (result,) = runner.evaluate(
                    point, scale, spin=spin, parameters=PARAMETERS, precision=precision
                ).values()
                _assert_correlated(result, values, index, tolerance)


@pytest.mark.parametrize("name", ("gg_gg", "uux_ggg"))
def test_helicity_resolved_outputs_equal_legacy(generate, name):
    """O1: per Born helicity row, the Born and the virtual summed over colour
    components equal the legacy resolved outputs (the legacy colour anchors
    are diagram colour tensors, v2's are Born basis elements, so the colour
    axis is compared through its sum, design §1.6 decision 3)."""

    record = _legacy(name)
    observables, _runners = _legacy_runners(generate, record)
    expected = record["results"]["arbitrary-200"][record["processes"][0][0]]
    resolved_record = expected["resolved"]
    legacy_rows = [tuple(row) for row in resolved_record["helicities"]]
    tolerance = LEGACY_TOLERANCE["double"]
    for index, (point, scale) in enumerate(
        zip(record["points"], record["scales_squared"], strict=True)
    ):
        resolved = observables.evaluate_resolved(
            point, scale, resolution="helicity", parameters=PARAMETERS
        )
        mine = {
            tuple(h for _leg, h in row): value
            for (row, _b), value in resolved.entries.items()
        }
        born_scale = max(
            abs(_pair(x)) for order in resolved_record["born"][index] for x in order
        )
        virtual_scale = max(
            abs(_pair(x))
            for sector in resolved_record["virtual"][index]
            for row in sector
            for x in row
        )
        for h, row in enumerate(legacy_rows):
            value = mine.get(row)
            expected_born = [
                _pair(order[h]) for order in resolved_record["born"][index]
            ]
            expected_virtual = [
                [_pair(x) for x in sector[h]]
                for sector in resolved_record["virtual"][index]
            ]
            if value is None:
                # A row v2 does not generate is a structural zero.
                assert max(map(abs, expected_born)) <= tolerance * born_scale
                continue
            for actual, target in zip(value.born.values(), expected_born, strict=True):
                assert abs(complex(actual) - target) <= tolerance * born_scale
            sectors = list(value.components["virtual"].values())
            for laurent, targets in zip(sectors, expected_virtual, strict=True):
                for actual, target in zip(laurent.coefficients, targets, strict=True):
                    assert abs(complex(actual) - target) <= tolerance * virtual_scale


# ------------------------------------------------------------------ public surface


def test_public_runtime_surface(generate, loop_reference):
    """``runtime.virtual.V2VirtualRuntime`` answers the public calls with the
    public result types: ordinary, correlated (colour and spin, grouped by
    source assignment) and resolved evaluations."""

    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.runtime.virtual import V2VirtualRuntime

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("T12", 1, 2),),
        spin_correlations=((3,),),
    )
    observables = generate("u u~ > g g", correlators=declarations)
    runtime = V2VirtualRuntime(observables, model_parameters=PARAMETERS)
    rows = loop_reference("spin-recola-control")["rows"]
    points = [row["point"] for row in rows]
    scales = [row["mu_r2"] for row in rows]
    assert [item.id for item in runtime.available_color_correlations()] == [
        "born",
        "T12",
    ]
    ordinary = runtime.evaluate(points, mu_r2=scales)
    assert ordinary.shape == (2, 1, 3) and not ordinary.correlated
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
    precise = runtime.evaluate(
        points[:1], mu_r2=scales[:1], arithmetic="arbitrary", precision=40
    )
    for a, b in zip(precise.values[0][0], ordinary.values[0][0], strict=True):
        assert _close(float(a), b, 1e-13)


# ------------------------------------------------------------------ O14


def test_bdk_partial_amplitudes_reproduce_the_colour_sum_o14(generate):
    """O14 with resolved-colour seed columns (design §7.6, §14): the
    projections of v2's bare loop amplitude of qbar q g g g onto BDK's one-loop
    colour structures (``ObservableSlice.project_loop``) determine partial
    amplitudes A_{5;1}, A_{5;3}, A_{5;4} (least squares on the N_c = 3 Gram
    matrix, which has one null direction; residual at rounding). With the tree
    partial amplitudes of v2's Born, BDK's colour sum (N^2-1)/N sum_s
    A_tree(s)^* bracket(s) (hep-ph/9409393 eq. NLOCrossSection) reproduces
    v2's helicity-resolved bare virtual."""

    import numpy as np

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

    def at_three(cell) -> complex:
        return complex(evaluate_color_nc_terms(cell))

    trees, loops = list(tree), list(loop)
    gram_tree = np.array(
        [
            [
                at_three(contract_connected_tensor_nc_terms(tree[a], tree[b]))
                for b in trees
            ]
            for a in trees
        ]
    )
    gram_loop = np.array(
        [
            [
                at_three(contract_connected_tensor_nc_terms(loop[a], loop[b]))
                for b in loops
            ]
            for a in loops
        ]
    )
    bdk = np.array(
        [
            [
                sum(
                    float(v) * 3.0**p
                    for p, v in _laurent_times(
                        _bdk_bracket(s).get(key, {}), BDK_PREFACTOR
                    ).items()
                )
                for key in loops
            ]
            for s in trees
        ]
    )
    evaluation = observables.point(point, scale, PARAMETERS)

    def born(row) -> np.ndarray:
        """Tree partial amplitudes: Born = sum_s a_s tree_s."""

        projections = []
        for s in trees:
            total = 0j
            for vector in evaluation.born_vectors[row].values():
                for (tensor, power), value in vector.items():
                    overlap = contract_connected_tensor_nc_terms(
                        tree[s], _single(tensor, slice_.basis.legs)
                    )
                    total += complex(value) * 3**power * at_three(overlap)
            projections.append(total)
        return np.linalg.solve(gram_tree, np.array(projections))

    amplitudes = {row: born(row) for row in observables.rows}
    largest = max(np.abs(a).max() for a in amplitudes.values())
    rows = [row for row, a in amplitudes.items() if np.abs(a).max() > 1e-3 * largest]
    rows = rows[:2]
    assert rows
    projections = observables.project_loop(
        point, scale, [loop[key] for key in loops], rows=rows, parameters=PARAMETERS
    )
    resolved = observables.evaluate_resolved(
        point, scale, resolution="helicity", parameters=PARAMETERS
    )
    (order,) = {label[2] for label in projections}
    for row in rows:
        (bare,) = resolved.entries[row, None].components["bare"].values()
        for k in range(3):
            target = np.array(
                [
                    complex(projections[row, j, order].coefficients[k])
                    for j in range(len(loops))
                ]
            )
            partial, *_ = np.linalg.lstsq(gram_loop, target, rcond=None)
            residual = np.abs(gram_loop @ partial - target).max()
            assert residual <= 1e-11 * np.abs(target).max()
            interference = np.conj(amplitudes[row]) @ (bdk @ partial)
            expected = 2 * interference.real / slice_.normalisation.average
            assert _close(bare.coefficients[k], expected, 1e-10)


# ------------------------------------------------------------------ native


@pytest.mark.parametrize("carrier", ("double", "double-double", "arbitrary-200"))
def test_native_observables_equal_python(generate, carrier):
    """The native executor runs the same observable columns (exported seed
    weights; ``column_rows`` for helicity-resolved columns): directed
    histories (ordered, complex), spin sources and helicity x colour entries
    equal the Python reference executor; DoubleFloat dresses and reduces the
    200-bit trees natively."""

    from pyamplicol.loops.v2.observables import NativeObservables

    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("casimir", 1, 1), ORDERED),
        spin_correlations=((3,),),
    )
    observables = generate("d d~ > g g", correlators=declarations)
    points = _points("uudd_qcd")[:2]
    scales = ["1000000"] * 2
    precision = CARRIERS["double" if carrier == "double" else "arbitrary-200"]
    options = {"double_double": carrier == "double-double"}
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
            **options,
        )
        for point, scale, result in zip(points, scales, results, strict=True):
            reference = observables.evaluate(
                point,
                scale,
                correlations=histories,
                spin=spin,
                parameters=PARAMETERS,
                precision=precision,
            )
            for key, value in reference.items():
                for a, b in zip(result[key].totals, value.totals, strict=True):
                    assert _close(a, b, tolerance, tolerance * 1e-2), (key, spin)
    resolved = NativeObservables.resolved(observables, lanes=4).evaluate(
        points[:1], scales[:1], parameters=PARAMETERS, precision=precision, **options
    )[0]
    reference = observables.evaluate_resolved(
        points[0], scales[0], parameters=PARAMETERS, precision=precision
    )
    largest = max(abs(complex(x)) for v in reference.entries.values() for x in v.totals)
    for key, value in reference.entries.items():
        for a, b in zip(resolved[key].totals, value.totals, strict=True):
            assert abs(complex(a) - complex(b)) <= tolerance * largest, key
