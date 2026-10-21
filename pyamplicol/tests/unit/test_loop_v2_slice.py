# SPDX-License-Identifier: 0BSD
"""The v2 vertical slice end to end: gate S1 (design §16.3, WP1.8).

Every value comes from the v2 generators and the Python reference executor;
FeynKit diagrams appear only inside the frozen legacy oracles. The slice
processes are gg -> gg (massless b), uu~ -> dd~g and the identical-flavour
sign controls uu~ -> uu~ and uu~ -> uu~g (O13), in binary64 and at 200 bits,
and the leading- and next-to-leading-colour projections of gg -> gg,
uu~ -> ggg, uu~ -> dd~g, Drell-Yan and uu~ -> tt~. Oracles:

* providers (O8): MadLoop, Recola and OpenLoops within 1e-9 (the
  required-references metric) for the virtual's three Laurent coefficients;
* legacy (O1): Born and the six components within 1e-12 (binary64) and
  1e-40 (200 bits) of the frozen 200-bit legacy values. The legacy binary64
  values themselves deviate from those by up to ~7e-12 on the five-leg
  controls, so the binary64 comparison uses the 200-bit oracle;
* the instrumented MadLoop component control of gg -> gg (1e-9);
* O2: Born-weighted family numerators against the legacy capture after exact
  affine frame maps (rank-r coefficients weighted by sqrt(s)^r, the units- and
  frame-independent error of N(q) at |q| ~ sqrt(s)), family sets equal modulo
  certified-zero families (O7);
* O4: schedules F and B give identical numerators in exact arithmetic on the
  physical tree rows and seeds;
* O9: the virtual's poles equal Catani's operator from the Born and its
  colour correlations (1e-10), also with the model's default parameters;
* the Born closed at leg 1's vertex is the tree runtime's Born (its amplitude
  roots, phase included) on the tree engine's colour basis;
* the sign rule's scope: loop fermion vertices that are not pure vector
  currents are refused at generation;
* finite-only evaluations and the stability estimates of the runtime API
  (REQUIREMENTS_ADDENDUM B, C), every check configuration bracketing the true
  binary64 error.
"""

from __future__ import annotations

import gzip
import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    Laurent,
    LoopSlice,
    Precision,
    SliceRequest,
    _decimal_string,
)
from pyamplicol.loops.v2.conventions import FermionSignError
from pyamplicol.loops.v2.families import shifts
from pyamplicol.loops.v2.poly_affine import size, substitute
from pyamplicol.loops.v2.stability import (
    REDUCTION,
    ROTATION,
    Rotation,
    StabilityChecks,
    StabilityEstimate,
    deviation,
    differences,
    per_coefficient,
)
from tools.loop_reference import legacy_capture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop"
SLICE = ("gg_gg", "uux_ddxg", "uux_uux", "uux_uuxg")
CARRIERS = {"double": Precision.double(), "arbitrary-200": Precision.arbitrary(60)}
LEGACY_TOLERANCE = {"double": Decimal("1e-12"), "arbitrary-200": Decimal("1e-40")}
PROVIDER_TOLERANCE = 1e-9
POLE_TOLERANCE = 1e-10
O2_TOLERANCE = {"double": 1e-13, "arbitrary-200": 1e-50}
ALPHA_S = 0.118
LAURENT = ("finite", "single_pole", "double_pole")
PARAMETERS = {"aS": "0.118", "MT": "173"}
# Two exact Pythagorean rotations (terminating decimals for the 2 -> 2 points).
ROTATIONS = (
    Rotation.about("z", "3/5", "4/5"),
    Rotation.about("x", "-4/5", "3/5") @ Rotation.about("y", "5/13", "12/13"),
)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / "legacy" / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            control = legacy_capture.CONTROLS[name]
            built[name] = LoopSlice.generate(
                compiled,
                SliceRequest(
                    control.process,
                    control.born_order_bounds,
                    control.loop_order_bounds,
                    fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                ),
                loop_native,
            )
        return built[name]

    return build


@pytest.fixture(scope="module")
def evaluated(slices):
    """Every slice point once per carrier; binary64 with all stability checks."""

    cache: dict[tuple[str, str], list] = {}

    def evaluate(name: str, carrier: str):
        key = (name, carrier)
        if key not in cache:
            fixture = _fixture(name)
            checks = StabilityChecks(ROTATIONS, True) if carrier == "double" else None
            cache[key] = [
                slices(name).evaluate(
                    point,
                    scale,
                    parameters=PARAMETERS,
                    precision=CARRIERS[carrier],
                    stability=checks,
                )
                for point, scale in zip(
                    fixture["points"], fixture["scales_squared"], strict=True
                )
            ]
        return cache[key]

    return evaluate


def _only(mapping):
    (value,) = mapping.values()
    return value


# ----------------------------------------------------------------- O1 legacy


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", SLICE)
def test_born_and_components_equal_legacy(evaluated, name, carrier):
    """Born, the six components and the 19 totals against the 200-bit O1."""

    results = _fixture(name)["results"]["arbitrary-200"]
    for index, result in enumerate(evaluated(name, carrier)):
        _assert_legacy(results, index, result, LEGACY_TOLERANCE[carrier])
        totals = [_only(result.born)]
        for component in COMPONENTS:
            totals.extend(_only(result.components[component]).coefficients)
        assert len(result.totals) == 19
        assert list(result.totals) == totals


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_massive_quark_line_equals_legacy(slices, carrier):
    """uu~ -> tt~: an open massive fermion line through the loop, read with
    the transposed mass sign (``conventions.TreeReading.mass_signs``), gives
    the legacy Born and components (no IR-pole prediction with massive legs)."""

    fixture = _fixture("tt")
    results = fixture["results"]["arbitrary-200"]
    for index, (point, scale) in enumerate(
        zip(fixture["points"], fixture["scales_squared"], strict=True)
    ):
        result = slices("tt").evaluate(
            point, scale, parameters=PARAMETERS, precision=CARRIERS[carrier]
        )
        assert result.ir_poles is None
        _assert_legacy(results, index, result, LEGACY_TOLERANCE[carrier])


# ----------------------------------------------------------------- colour


# Controls with legacy LC and NLC projections (``legacy_capture``); Drell-Yan
# is the photon-only control, so the Z is excluded from its v2 request.
PROJECTED = ("gg_gg", "uux_ggg", "uux_ddxg", "dy", "tt")
DRELL_YAN_PARAMETERS = {"aS": "0.118", "aEWM1": "132.507"}


def _request(name: str, **options) -> SliceRequest:
    control = legacy_capture.CONTROLS[name]
    process, ignored = control.process, ()
    if control.drell_yan:
        process, ignored = control.process.removesuffix(" / z"), (23,)
    return SliceRequest(
        process,
        control.born_order_bounds,
        control.loop_order_bounds,
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        ignored_particle_ids=ignored,
        **options,
    )


def _parameters(name: str) -> dict[str, str]:
    return (
        DRELL_YAN_PARAMETERS if legacy_capture.CONTROLS[name].drell_yan else PARAMETERS
    )


@pytest.fixture(scope="module")
def declared(loop_sm_compiled_model, loop_native):
    """A control with the committed QCD colour continuation, per accuracy."""

    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[tuple[str, str], LoopSlice] = {}

    def build(name: str, accuracy: str) -> LoopSlice:
        if (name, accuracy) not in built:
            built[name, accuracy] = LoopSlice.generate(
                compiled,
                _request(name, accuracy=accuracy, colour_declaration=declaration),
                loop_native,
            )
        return built[name, accuracy]

    return build


def _assert_legacy(results, index, result, tolerance):
    born = Decimal(results["born"][index][0][0])
    mine = Decimal(_only(result.born))
    assert abs(mine - born) <= tolerance * abs(born), (index, mine, born)
    for component in COMPONENTS:
        reference = [Decimal(x[0]) for x in results[component][index][0]]
        values = [Decimal(x) for x in _only(result.components[component]).coefficients]
        scale = max(map(abs, reference)) or abs(born)
        for k, (a, b) in enumerate(zip(values, reference, strict=True)):
            assert abs(a - b) <= tolerance * scale, (component, index, k, a, b)


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize("name", PROJECTED)
def test_colour_projections_equal_legacy(declared, name, accuracy, carrier):
    """LC and NLC: leading degrees and every component against the legacy
    projections (design §7.4-§7.6; graded Born amplitudes and U(1) pieces,
    four-gluon R2 from the general-N_c HV rule)."""

    slice_ = declared(name, accuracy)
    fixture = _fixture(name)
    reference = fixture["colour_projections"][accuracy]
    born, virtual = _only(slice_.degrees()["born"]), _only(slice_.degrees()["virtual"])
    assert (born, virtual) == (reference["born_degree"], reference["virtual_degree"])
    results = reference["results"]["arbitrary-200"]
    for index, (point, scale) in enumerate(
        zip(fixture["points"], fixture["scales_squared"], strict=True)
    ):
        result = slice_.evaluate(
            point, scale, parameters=_parameters(name), precision=CARRIERS[carrier]
        )
        assert result.ir_poles is None  # Catani's operator is full colour
        _assert_legacy(results, index, result, LEGACY_TOLERANCE[carrier])


@pytest.mark.loop_physics
def test_declared_continuation_keeps_full_colour(declared):
    """The continued couplings and the HV four-gluon rule reproduce the UFO
    counterterms at N_c = 3 (full colour, O1 and O9)."""

    fixture = _fixture("gg_gg")
    results = fixture["results"]["arbitrary-200"]
    for index, (point, scale) in enumerate(
        zip(fixture["points"], fixture["scales_squared"], strict=True)
    ):
        result = declared("gg_gg", "full").evaluate(point, scale, parameters=PARAMETERS)
        _assert_legacy(results, index, result, LEGACY_TOLERANCE["double"])
        ((single, double),) = result.ir_poles.values()
        virtual = _only(result.components["virtual"])
        scale_ = max(abs(single), abs(double))
        assert abs(virtual.single_pole - single) <= POLE_TOLERANCE * scale_
        assert abs(virtual.double_pole - double) <= POLE_TOLERANCE * scale_


@pytest.mark.loop_physics
def test_seed_fft_equals_direct_on_the_slice(
    slices, declared, loop_sm_compiled_model, loop_native
):
    """Seed contraction policy (design §7.3): "auto" contracts the certified
    orbits of gg -> gg with the native symmetric-group FFT in binary64 and
    reports it; "direct" and the 200-bit carrier contract directly, and
    every component agrees with the direct contraction to binary64
    rounding. LC seeds are always direct."""

    import dataclasses

    from pyamplicol.models.loading import load_compiled_model

    fft = slices("gg_gg")
    assert fft.fourier is not None
    direct = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        _request("gg_gg", colour_contraction="direct"),
        loop_native,
    )
    assert direct.fourier is None
    fixture = _fixture("gg_gg")
    for point, scale in zip(fixture["points"], fixture["scales_squared"], strict=True):
        mine = fft.evaluate(point, scale, parameters=PARAMETERS)
        reference = direct.evaluate(point, scale, parameters=PARAMETERS)
        assert mine.colour_contraction == "symmetric-group-fft"
        assert reference.colour_contraction == "direct"
        for component in COMPONENTS:
            a = _only(mine.components[component]).coefficients
            b = _only(reference.components[component]).coefficients
            size = max(map(abs, b)) or abs(_only(reference.born))
            for x, y in zip(a, b, strict=True):
                assert abs(x - y) <= O2_TOLERANCE["double"] * size, (component, x, y)
    precise = fft.evaluate(
        fixture["points"][0],
        fixture["scales_squared"][0],
        parameters=PARAMETERS,
        precision=CARRIERS["arbitrary-200"],
    )
    assert precise.colour_contraction == "direct"
    leading = declared("gg_gg", "lc")
    assert leading.fourier is None
    forced = dataclasses.replace(
        leading,
        request=dataclasses.replace(leading.request, colour_contraction="fft"),
    )
    with pytest.raises(ValueError, match="leading-colour"):
        forced._plan_contraction()


# ----------------------------------------------------------------- O8 providers


def _relative(a: float, b: float, floor: float = 0.0) -> float:
    """The required-references metric: |a-b| / max(|a|, |b|, floor)."""
    scale = max(abs(a), abs(b), floor)
    return 0.0 if scale == 0 else abs(a - b) / scale


def _provider_rows(loop_reference, name: str) -> list[dict]:
    reference = loop_reference(
        "v2/gg_gg_providers" if name == "gg_gg" else f"required/{name}"
    )
    assert reference["validated_providers"] == ["madloop", "recola", "openloops"]
    fixture = _fixture(name)
    points = [
        [[Decimal(str(x)) for x in leg] for leg in p] for p in reference["points"]
    ]
    assert points == [
        [[Decimal(x) for x in leg] for leg in p] for p in fixture["points"]
    ]
    assert [Decimal(mu) ** 2 for mu in reference["mu_r"]] == [
        Decimal(s) for s in fixture["scales_squared"]
    ]
    return [row["providers"] for row in reference["rows"]]


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", SLICE)
def test_virtual_agrees_with_providers(evaluated, loop_reference, name, carrier):
    rows = _provider_rows(loop_reference, name)
    for index, (result, providers) in enumerate(
        zip(evaluated(name, carrier), rows, strict=True)
    ):
        born = float(_only(result.born))
        virtual = [float(x) for x in _only(result.components["virtual"]).coefficients]
        floor = abs(born) * ALPHA_S / (2 * 3.141592653589793)
        for provider, values in providers.items():
            assert _relative(born, values["born"]) <= PROVIDER_TOLERANCE, provider
            for pole, mine in zip(LAURENT, virtual, strict=True):
                error = _relative(mine, values[pole], floor)
                assert error <= PROVIDER_TOLERANCE, (index, provider, pole, error)


COMPONENT_KEYS = {
    "virtual": "full",
    "bare": "bare",
    "uv": "uv",
    "r2": "r2",
    "uv_local": "uv_vertex",
    "uv_external_fields": "wf",
}


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
def test_gg_gg_components_agree_with_madloop_control(
    evaluated, loop_reference, carrier
):
    control = loop_reference("gluon-scattering-component-control")
    for result, row in zip(evaluated("gg_gg", carrier), control["rows"], strict=True):
        born = float(_only(result.born))
        floor = abs(born) * ALPHA_S / (2 * 3.141592653589793)
        for component, key in COMPONENT_KEYS.items():
            values = [
                float(x) for x in _only(result.components[component]).coefficients
            ]
            for pole, mine in zip(LAURENT, values, strict=True):
                error = _relative(mine, row[key][pole], floor)
                assert error <= PROVIDER_TOLERANCE, (component, pole, error)


# ----------------------------------------------------------------- O9 poles


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", SLICE)
def test_poles_equal_catani(evaluated, name, carrier):
    for result in evaluated(name, carrier):
        for sector, (single, double) in result.ir_poles.items():
            virtual = result.virtual(sector)
            scale = max(abs(float(single)), abs(float(double)))
            assert abs(float(virtual.single_pole - single)) <= POLE_TOLERANCE * scale
            assert abs(float(virtual.double_pole - double)) <= POLE_TOLERANCE * scale


@pytest.mark.loop_physics
def test_default_model_parameters(slices, loop_sm_compiled_model, loop_native):
    """Without overrides alpha_s and mu_R are the model's own parameters,
    found by their UFO/SLHA codes; names must be adjustable externals."""

    from pyamplicol.models.loading import load_compiled_model

    slice_ = slices("uux_uux")
    assert (slice_.scale_parameter, slice_.alpha_s_parameter) == ("MU_R", "aS")
    fixture = _fixture("uux_uux")
    result = slice_.evaluate(fixture["points"][0], fixture["scales_squared"][0])
    ((single, double),) = result.ir_poles.values()
    virtual = _only(result.components["virtual"])
    scale = max(abs(single), abs(double))
    assert abs(virtual.single_pole - single) <= POLE_TOLERANCE * scale
    assert abs(virtual.double_pole - double) <= POLE_TOLERANCE * scale
    control = legacy_capture.CONTROLS["uux_uux"]
    for options in ({"alpha_s_parameter": "alpha"}, {"scale_parameter": "MB"}):
        request = SliceRequest(
            control.process,
            control.born_order_bounds,
            control.loop_order_bounds,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            **options,
        )
        with pytest.raises(ValueError, match="adjustable external"):
            LoopSlice.generate(
                load_compiled_model(loop_sm_compiled_model), request, loop_native
            )


# ----------------------------------------------------------------- conventions


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg", "uux_uux"])
def test_born_closure_is_the_tree_runtime_born(slices, name):
    """The Born closed at the vertex of leg 1 equals the tree runtime's Born,
    the amplitude roots of the GenericDAG compiler summed per helicity and
    colour sector, phase included: <b|M_h> agree on every element b of the
    tree engine's colour basis (identical-flavour signs on uu~ -> uu~)."""

    from pyamplicol.color.connections import contract_connected_tensor_nc_terms
    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.colour import born_basis

    slice_ = slices(name)
    fixture = _fixture(name)
    point = [[Fraction(x) for x in leg] for leg in fixture["points"][0]]
    precision = Precision.arbitrary(30)
    numbers = assemble._Numbers(precision)
    basis = born_basis(slice_.blocks)
    assert not any(tensor.normalization_power for tensor in basis.tensors)
    with numbers.scope():
        closure = slice_._point(
            point, Fraction(fixture["scales_squared"][0]), PARAMETERS, precision, True
        )
        roots = assemble.TreeRows(slice_.blocks.born_dag, slice_.compiled).evaluate(
            [[assemble._to_decimal(x) for x in leg] for leg in point],
            closure.parameters,
            numbers,
            amplitudes=True,
        )

        def overlap(bra, ket):
            cells = contract_connected_tensor_nc_terms(bra, ket)
            return sum(
                (
                    numbers.colour(cell.scaled(Fraction(3) ** power), 0)
                    for power, cell in cells.items()
                ),
                numbers.zero,
            )

        runtime: dict = {}
        for root in slice_.blocks.born_roots:
            (value,) = roots[(root.root_id, "amplitude-root")]
            key = (
                tuple((label - 1, h) for label, h in root.helicities),
                root.coupling_orders,
                0
                if root.color_sector_id is None
                else basis.sector_ids.index(root.color_sector_id),
            )
            runtime[key] = runtime.get(key, numbers.zero) + value
        worst = largest = 0.0
        for row, vectors in closure.born_vectors.items():
            for order, vector in vectors.items():
                for bra in basis.tensors:
                    mine = sum(
                        (
                            value
                            * numbers.real(Fraction(3) ** power)
                            * overlap(bra, assemble._single(tensor, basis.legs))
                            for (tensor, power), value in vector.items()
                        ),
                        numbers.zero,
                    )
                    theirs = sum(
                        (
                            amplitude * overlap(bra, basis.tensors[c])
                            for (r, o, c), amplitude in runtime.items()
                            if (r, o) == (row, order)
                        ),
                        numbers.zero,
                    )
                    worst = max(worst, abs(mine - theirs))
                    largest = max(largest, abs(theirs))
    assert largest > 0 and worst <= 1e-25 * largest, worst / largest


@pytest.mark.loop_physics
def test_loop_fermion_vertices_must_be_vector(loop_sm_compiled_model, loop_native):
    """The -1 per loop fermion propagator holds for vector couplings: a quark
    loop through a chiral Z vertex is refused at generation (O13 EW, WP3.2)."""

    from pyamplicol.models.loading import load_compiled_model

    with pytest.raises(FermionSignError, match="not a pure vector coupling"):
        LoopSlice.generate(
            load_compiled_model(loop_sm_compiled_model),
            SliceRequest(
                "u u~ > d d~",
                {"QCD": 0, "QED": 2},
                {"QCD": 2, "QED": 2},
                fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            ),
            loop_native,
        )


# ----------------------------------------------------------------- O2 / O7


def _legacy_families(name: str, carrier: str):
    data = json.loads(
        gzip.decompress((FIXTURES / "legacy" / f"{name}.families.json.gz").read_bytes())
    )
    return data, data["coefficients"][carrier]


def _normal(vector):
    """Leg-momentum coefficients modulo momentum conservation (last leg zero)."""
    return tuple(x - vector[-1] for x in vector)


def _maps(source, source_masses, target, target_masses):
    """Affine maps q_target = sign q_source + a sending one ring onto the other."""
    out = []
    for sign in (1, -1):
        for anchor in range(len(target)):
            shift = _normal(
                [sign * x - y for x, y in zip(source[0], target[anchor], strict=True)]
            )
            image = {
                (_normal([sign * x - y for x, y in zip(v, shift, strict=True)]), m)
                for v, m in zip(source, source_masses, strict=True)
            }
            if image == set(zip(target, target_masses, strict=True)):
                out.append((sign, shift))
    return out


def _momentum(coefficients, momenta, mu):
    return sum(Fraction(c) * p[mu] for c, p in zip(coefficients, momenta, strict=True))


def _symmetrised(coefficients, layout, momenta, maps):
    """Average of a numerator over the ring's automorphisms (integral-preserving)."""
    total = [0] * len(coefficients)
    for sign, shift in maps:
        vector = [_momentum(shift, momenta, mu) for mu in range(4)]
        mapped = substitute(coefficients, layout, sign=sign, shift=vector)
        total = [a + b for a, b in zip(total, mapped, strict=True)]
    return [x * Fraction(1, len(maps)) for x in total]


def _ring(sequence, legs):
    vectors = [
        _normal([1 if mask >> j & 1 else 0 for j in range(legs)])
        for mask in shifts(sequence)
    ]
    masses = ["0" if m.massless else f"{m.mass}^2" for _mask, m in sequence]
    return vectors, masses


def _magnitude(z) -> float:
    return float(z.real * z.real + z.imag * z.imag) ** 0.5 if z else 0.0


def _weights(layout, length, momenta):
    """sqrt(s)^rank per coefficient: the error of N(q) at |q| ~ sqrt(s).

    Coefficients of rank r carry mass dimension D - r, so a maximum over raw
    coefficients depends on the units and on the frame (a shift by p moves
    weight between ranks): binary64 numerators of every implementation then
    "miss" 1e-13 on structurally cancelling low-rank coefficients. The
    weighted metric is the same in every frame and leaves the binary64 v2
    numerators within the O2 bound (DESIGN §14, O2).
    """

    total = [a + b for a, b in zip(momenta[0], momenta[1], strict=True)]
    energy = math.sqrt(float(total[0] ** 2 - sum(x * x for x in total[1:])))
    return [energy ** sum(layout.exponents[i]) for i in range(length)]


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", (*SLICE, "tt"))
def test_family_numerators_equal_o2(slices, name, carrier):
    """O2 and the O7 family sets: every v2 family numerator equals its legacy
    family's (200-bit capture) after an exact affine frame map, symmetrised
    over the ring's automorphisms, which preserve the integral, coefficients
    weighted by sqrt(s)^rank; the legacy families without a v2 partner are
    exactly v2's certified-scaleless ones."""

    from pyamplicol.loops.v2 import reference_executor as rx

    slice_ = slices(name)
    data, rows = _legacy_families(name, "arbitrary-200")
    point, scale = data["points"][0], data["scales_squared"][0]
    execution, momenta = slice_.family_numerators(
        point, scale, parameters=PARAMETERS, precision=CARRIERS[carrier]
    )
    legs = len(data["pdgs"])
    signs = [1, 1] + [-1] * (legs - 2)  # given momenta -> all incoming
    legacy = [
        (
            [
                _normal([c * s for c, s in zip(v, signs, strict=True)])
                for v in family["shifts"]
            ],
            family["masses_squared"],
            family,
            coefficients,
        )
        for family, coefficients in zip(data["families"], rows, strict=True)
    ]
    layout = slice_.layout
    average = slice_.normalisation.average
    matched = set()
    for (key, _orders), numerator in execution.numerators.items():
        vectors, masses = _ring(numerator.sequence, legs)
        found = [
            (index, maps)
            for index, (shifts_, masses_, _f, _c) in enumerate(legacy)
            if (maps := _maps(vectors, masses, shifts_, masses_))
        ]
        assert len(found) == 1, key
        index, maps = found[0]
        matched.add(index)
        _shifts, _masses, family, (row,) = legacy[index]
        rank = max(sum(p[:4]) for p in family["powers"])
        dense = [rx.Complex(0, 0)] * size(max(rank, numerator.rank))
        for powers, (re, im) in zip(family["powers"], row, strict=True):
            assert powers[4:] == [0, 0]
            dense[layout.index[tuple(powers[:4])]] = rx.Complex(
                Fraction(re), Fraction(im)
            )
        sign, shift = maps[0]
        vector = [_momentum(shift, momenta, mu) for mu in range(4)]
        theirs = substitute(dense, layout, sign=sign, shift=vector)
        (column,) = numerator.columns
        mine = [rx.exact(x) * Fraction(1, average) for x in column]
        mine += [0] * (len(theirs) - len(mine))
        autos = _maps(vectors, masses, vectors, masses)
        if len(autos) > 1:
            mine = _symmetrised(mine, layout, momenta, autos)
            theirs = _symmetrised(theirs, layout, momenta, autos)
        weights = _weights(layout, len(theirs), momenta)
        scale_ = max(_magnitude(b) * w for b, w in zip(theirs, weights, strict=True))
        error = max(
            _magnitude(a - b) * w for a, b, w in zip(mine, theirs, weights, strict=True)
        )
        assert error <= O2_TOLERANCE[carrier] * scale_, (key, error / scale_)
    # v2 also logs the massless tadpoles the legacy selection never generates.
    keys = dict.fromkeys(entry.key for entry in slice_.families.scaleless)
    scaleless = [_ring(key, legs) for key in keys]
    for index, (shifts_, masses_, _family, _rows) in enumerate(legacy):
        if index not in matched:
            assert any(_maps(v, m, shifts_, masses_) for v, m in scaleless), index
    assert all(
        len(key) == 1
        for key in keys
        if not any(
            _maps(*_ring(key, legs), shifts_, masses_)
            for shifts_, masses_, _f, _r in legacy
        )
    )


# ----------------------------------------------------------------- O4


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_schedules_agree_exactly_on_physical_rows(slices, name):
    """F = B per family in exact arithmetic on the physical tree rows and
    seeds (M needs colour-ordered seeds, WP2.4, and falls back to B)."""

    fixture = _fixture(name)
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    runs = [
        slices(name).family_numerators(
            point,
            scale,
            parameters=PARAMETERS,
            precision=Precision.arbitrary(30),
            schedule=schedule,
            exact=True,
        )[0]
        for schedule in ("B", "F")
    ]
    b, f = (run.numerators for run in runs)
    assert b.keys() == f.keys() and b
    for key, mine in b.items():
        other = f[key]
        assert mine.frame == other.frame
        for x, y in zip(mine.columns, other.columns, strict=True):
            width = max(len(x), len(y))
            assert list(x) + [0] * (width - len(x)) == list(y) + [0] * (width - len(y))


# ----------------------------------------------------------------- runtime API


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", tuple(CARRIERS))
@pytest.mark.parametrize("name", SLICE)
def test_finite_only_equals_the_finite_part(slices, evaluated, name, carrier):
    """poles=False: the same slice, finite parts equal, pole parts absent."""

    fixture = _fixture(name)
    for full, point, scale in zip(
        evaluated(name, carrier),
        fixture["points"],
        fixture["scales_squared"],
        strict=True,
    ):
        finite = slices(name).evaluate(
            point,
            scale,
            parameters=PARAMETERS,
            precision=CARRIERS[carrier],
            poles=False,
        )
        assert finite.ir_poles is None and finite.stability is None
        assert len(finite.totals) == 7
        assert finite.born == full.born
        for component in COMPONENTS:
            for sector, value in finite.components[component].items():
                reference = full.components[component][sector]
                assert value.single_pole is None and value.double_pole is None
                assert value.finite == reference.finite, component
        for lane, value in finite.families.items():
            assert value.coefficients == full.families[lane].coefficients[:1]


# Every configuration of the two binary64 checks of ``evaluated``.
CONFIGURATIONS = (
    (REDUCTION,),
    (ROTATION.format(0),),
    (ROTATION.format(1),),
    (ROTATION.format(0), ROTATION.format(1)),
    (ROTATION.format(0), REDUCTION),
    (ROTATION.format(1), REDUCTION),
    (ROTATION.format(0), ROTATION.format(1), REDUCTION),
)


@pytest.mark.loop_physics
def test_stability_checks_are_optional_and_combinable(slices, evaluated):
    fixture = _fixture("uux_uux")
    point, scale = fixture["points"][0], fixture["scales_squared"][0]
    slice_ = slices("uux_uux")

    def run(checks, **options):
        return slice_.evaluate(
            point, scale, parameters=PARAMETERS, stability=checks, **options
        ).stability

    assert run(None) is None
    assert run(StabilityChecks()) is None
    full = evaluated("uux_uux", "double")[0].stability
    assert set(full.checks) == set(CONFIGURATIONS[-1])
    # A configuration returns exactly the combination of its own checks.
    for checks, names in (
        (StabilityChecks(reduction_direction=True), CONFIGURATIONS[0]),
        (StabilityChecks(ROTATIONS[:1]), CONFIGURATIONS[1]),
    ):
        alone, subset = run(checks), full.subset(names)
        assert (alone.relative, alone.coefficients, alone.checks) == (
            subset.relative,
            subset.coefficients,
            subset.checks,
        )
        assert (alone.families is None) == (REDUCTION not in names)
        assert full.relative >= alone.relative
    assert full.families and all(
        float(x) >= 0 for errors in full.families.values() for x in errors
    )
    finite = run(StabilityChecks(ROTATIONS[:1], True), poles=False)
    assert len(finite.coefficients) == 1 and finite.finite == finite.coefficients[0]


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", SLICE)
def test_stability_estimates_bracket_the_true_error(evaluated, name):
    """Every check configuration's binary64 estimate is at least the true
    error against 200 bits, for the Laurent vector and for each coefficient
    (the finite part's own relative error included), and stays a meaningful
    bound on these well-conditioned points."""

    for f64, ap in zip(
        evaluated(name, "double"), evaluated(name, "arbitrary-200"), strict=True
    ):
        _assert_brackets(f64, ap)


# Out of sample: the estimate's two factors were calibrated on the 12 slice
# points with ``ROTATIONS``; these points and checks are disjoint from both, and
# their rotated momenta are non-terminating rationals.
FRESH_FRAME = Rotation.from_axis_angle((2, -1, 3), 1.1)
FRESH_CHECKS = StabilityChecks(
    (
        Rotation.from_axis_angle((1, 1, 0), 0.9),
        Rotation.from_axis_angle((-1, 2, 2), 2.3),
    ),
    True,
)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", SLICE)
def test_stability_estimates_bracket_out_of_sample(slices, name):
    fixture = _fixture(name)
    point = FRESH_FRAME.apply(
        [[Fraction(x) for x in momentum] for momentum in fixture["points"][-1]]
    )
    scale = fixture["scales_squared"][-1]
    f64, ap = (
        slices(name).evaluate(
            point,
            scale,
            parameters=PARAMETERS,
            precision=CARRIERS[carrier],
            stability=FRESH_CHECKS if carrier == "double" else None,
        )
        for carrier in ("double", "arbitrary-200")
    )
    _assert_brackets(f64, ap)


def _assert_brackets(f64, ap):
    """Every check configuration's estimate bounds the true binary64 error."""

    truth = {
        sector: Laurent(*[float(x) for x in value.coefficients])
        for sector, value in ap.components["virtual"].items()
    }
    error = deviation(truth, f64.components["virtual"])
    errors = per_coefficient(truth, differences(truth, f64.components["virtual"]))
    assert error > 0
    for names in CONFIGURATIONS:
        estimate = f64.stability.subset(names)
        assert error <= estimate.relative < 1e-10, (names, error, estimate)
        for k, (mine, bound) in enumerate(
            zip(errors, estimate.coefficients, strict=True)
        ):
            assert mine <= bound, (names, LAURENT[k], mine, bound)


@pytest.mark.loop_fast
def test_rotations_are_exact():
    rotation = ROTATIONS[1]
    point = [
        [Fraction(500), Fraction(0), Fraction(0), Fraction(500)],
        [Fraction(500), Fraction(300), Fraction(0), Fraction(400)],
    ]
    for before, after in zip(point, rotation.apply(point), strict=True):
        assert after[0] == before[0]
        assert sum(x * x for x in after[1:]) == sum(x * x for x in before[1:])
    with pytest.raises(ValueError, match="orthogonal"):
        Rotation(((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    with pytest.raises(ValueError, match="exactly 1"):
        Rotation.about("z", "1/2", "1/2")
    # Any axis and angle: an exact rotation within 1/resolution.
    near = Rotation.from_axis_angle((0, 0, 2), math.atan2(4, 3))
    for row, exact in zip(near.matrix, ROTATIONS[0].matrix, strict=True):
        assert all(
            abs(x - y) < Fraction(1, 10**5) for x, y in zip(row, exact, strict=True)
        )
    axis = (1, -2, 3)
    tilted = Rotation.from_axis_angle(axis, 2.5)
    image = tilted.apply([[Fraction(1), *map(Fraction, axis)]])[0]
    assert all(abs(x - y) < 1e-5 for x, y in zip(image[1:], axis, strict=True))
    with pytest.raises(ValueError, match="identity"):
        StabilityChecks((Rotation.from_axis_angle((1, 0, 0), 0.0),))


@pytest.mark.loop_fast
def test_reducer_inputs_keep_the_carrier_digits():
    """Rotated momenta reach the reducer with the carrier's digits, uncapped."""
    assert _decimal_string(Fraction(1, 3), 130) == "0." + "3" * 130
    assert _decimal_string(Fraction(1, 8), 5) == "0.125"


@pytest.mark.loop_fast
def test_an_estimate_needs_a_check():
    with pytest.raises(ValueError, match="at least one check"):
        StabilityEstimate.from_checks({}, {}, scale={}, roundoff=2.0**-53)
