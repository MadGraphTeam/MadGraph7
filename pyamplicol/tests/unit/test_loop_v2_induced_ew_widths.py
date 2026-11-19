# SPDX-License-Identifier: 0BSD
"""The width policy of loop-induced requests and the complex-mass rows (M-LIT
LIT-EW, m-lit PLAN §3.4, §4.1 LI-E3/E4, LI-43, LI-C1; RULED Q7: "Fixed tree
widths but consistent complex mass scheme everywhere when enabled (and then
unstable particles never in the external states).").

* A loop-induced request keeps fixed widths in its tree propagators and
  widthless loop propagators, recorded in the artifact's conventions:
  e+ e- > h h [QED] and e+ e- > g g g with the model's default Z width
  generate and evaluate (e+ e- > h h does not move with W_Z, e+ e- > g g g
  moves to first order); LI-43 g g > e+ e- mu+ mu- with fixed tree widths
  equals MadLoop and OpenLoops (that scheme); h > z a and a a > a a (no tree
  propagator) do not move with any width.
* The complex-mass scheme rows of LI-C1 (complex m_t in g g > a a and
  g g > a a g, complex M_W and m_t in the complete a a > a a, e+ e- > g g g
  and g g > e+ e- mu+ mu- with complex M_Z in their tree propagators,
  g g > z z with external W_Z = 0 and complex m_t) equal the complex-mass
  references in binary64 and double-double, record the complex-mass width
  policy, and an unstable external state is refused typed
  (``cms-external-width``).

Shared helpers: ``test_loop_v2_induced_ew.py`` (the G_mu model, one row's
slice at a time).

Tiers (slice-status DECISIONS 2026-11-07 ~09:12 (5)): one parametrisation
of each test runs in the default tier (e+ e- > g g g at the model's Z
width, LI-43, h > z a widthless, the complex-mass e+ e- > g g g, the
refused g g > z z), every other one is ``loop_nightly``.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from tests.unit.test_loop_v2_induced_ew import (
    ANALYTIC_DD,
    ANALYTIC_DOUBLE,
    EW_OPTIONS,
    _check_providers,
    _EwRows,
    _fixture,
    _process,
    _relative,
    _total,
    tiers,
)

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def ew(loop_gmu_ufo, loop_native, tmp_path_factory):
    return _EwRows(loop_gmu_ufo, loop_native, tmp_path_factory.mktemp("li-ew-w"))


# The complex-mass rows of LI-C1 (fixture, analytic oracle).
CMS_ROWS = [
    ("li_gg_aa_cms", True),
    ("li_gg_aag_cms", False),
    ("li_aa_aa_cms", True),
    ("li_emep_ggg_cms", False),
    ("li_gg_zz_cms", False),
    # 2 -> 4: complex M_Z in the tree propagators of the two lepton pairs.
    ("li_gg_emepmupmum_cms", False),
]
# The complex-mass rows whose real on-shell twin (same model and points, zero
# widths) has no tree propagator: their difference is the complex masses in
# the loop propagators alone (the complete a a > a a: W and top loops).
LOOP_ONLY_TWINS = {"li_aa_aa_cms": "li_aa_aa"}


# -------------------------------------------------------- the width policy


def _default_widths_request(ew, name, **fields):
    """The fixture's request at the model's default widths (only its
    massless-fermion entries fixed)."""

    from dataclasses import replace

    request = ew.request(_fixture(name), **fields)
    fixed = {
        key: value
        for key, value in request.fixed_parameters.items()
        if not key.startswith("W")
    }
    return replace(request, fixed_parameters=fixed)


# (fixture, how |A|^2 depends on the s-channel Z's width): e+ e- > h h has
# no s-channel Z part (Z* -> h h vanishes: Bose symmetry leaves only q^mu,
# which the massless lepton current annihilates), so its value does not
# move with W_Z beyond rounding; e+ e- > g g g has one (gamma*/Z* -> g g g
# through the quark loop), first order in W_Z.
LEPTON_ROWS = [("li_emep_hh", "none"), ("li_emep_ggg", "first order")]


@pytest.mark.parametrize(("name", "dependence"), tiers(LEPTON_ROWS, "li_emep_ggg"))
def test_lepton_rows_with_the_model_z_width(ew, name, dependence):
    """Q7, LI-E3 / LI-E4: e+ e- > h h [QED] and e+ e- > g g g at the model's
    default widths generate with fixed widths in their tree propagators (the
    s-channel Z) and widthless loop propagators, recorded in their
    conventions (no ``electroweak-width`` refusal: they are loop-induced).
    They evaluate at the model's Z width, a run-time parameter of the tree
    propagators: at W_Z = 0 the value equals the zero-width references
    (MadLoop, Recola: Recola is mandatory for e+ e- > h h, no Ward probe
    reaches a process without an external gauge boson); e+ e- > h h does not
    move with W_Z (its s-channel Z part vanishes), e+ e- > g g g moves to
    first order (the tree propagator's width times the quark loop's
    absorptive part)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS, scheme_conventions
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    fixture = _fixture(name)
    slice_ = LoopSlice.generate(
        ew.compiled, _default_widths_request(ew, name), ew.native
    )
    conventions = scheme_conventions(slice_)
    assert conventions["mass_scheme"] == "real-on-shell"
    policy = conventions["width_policy"]
    assert policy["policy"] == FIXED_TREE_WIDTHS
    assert policy["tree_propagators"] == "fixed widths"
    assert policy["loop_propagators"] == "widthless"
    assert policy["tree_widths"]["Z"] == "WZ"
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    parameters = ew.parameters(name, slice_)
    width = Decimal(str(ew.compiled.parameter_defaults["WZ"][0]))
    assert width > 0

    def values(wz):
        return executor.evaluate(
            fixture["points"],
            [Fraction(mu) ** 2 for mu in fixture["mu_r"]],
            parameters={**parameters, "WZ": str(wz)},
            precision=Precision.double(),
        )

    physical, tenth, zero = values(width), values(width / 10), values(0)
    _check_providers(name, zero, analytic=False)
    for index, (a, b, c) in enumerate(zip(physical, tenth, zero, strict=True)):
        for pole in a.totals[2:]:
            assert abs(float(pole)) <= 1e-9 * abs(float(_total(a)))
        shift, small = float(_total(a) - _total(c)), float(_total(b) - _total(c))
        if dependence == "none":
            # Rounding only (the zero-width references hold at W_Z too).
            assert abs(shift) <= 1e-12 * abs(float(_total(c))), (index, shift)
            continue
        assert abs(shift) > 1e-12 * abs(float(_total(c))), index
        # First order in W_Z: the second-order remainder is of relative size
        # M_Z W_Z / |s - M_Z^2| < 1e-3 at these points.
        assert abs(shift / small / 10 - 1) <= 0.05, (index, shift, small)


@pytest.fixture(scope="module")
def loop_sm_no_b_mass(loop_sm_ufo, tmp_path_factory):
    from tools.loop_benchmark.mqcd import compile_restriction

    return compile_restriction(
        loop_sm_ufo, "no_b_mass", tmp_path_factory.mktemp("li-ew-sm") / "no_b_mass"
    )


def test_fixed_tree_widths_equal_madloop_and_openloops(loop_sm_no_b_mass, loop_native):
    """Q7, LI-43 (the reference row of the width policy, scheme qcd-widths):
    g g > e+ e- mu+ mu- [QCD] on MG5_aMC's loop_sm (no_b_mass) with the Z, W
    and Higgs widths fixed in its tree propagators (the Z of each lepton pair
    and the Higgs of g g > h* > z z) and widthless loop propagators equals
    MadLoop and OpenLoops, which use that scheme, within the lax rule at
    every point in binary64 (Recola's fixed-width scheme is another, lit-ref-b:
    excluded); double-double at the first point equals binary64 within 1e-9;
    the policy names the propagated widths."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS, scheme_conventions
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    name = "li_gg_emepmupmum"
    fixture = _fixture(name)
    compiled = loop_sm_no_b_mass
    orders = dict(fixture["loop_orders"])
    fixed = {
        key: value
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
        if key in compiled.parameter_defaults
    }
    assert all(Decimal(fixed[w]) > 0 for w in ("WZ", "WW", "WH"))
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            _process(fixture, compiled),
            orders,
            orders,
            contribution="loop-induced",
            fixed_parameters=fixed,
        ),
        loop_native,
    )
    policy = scheme_conventions(slice_)["width_policy"]
    assert policy["policy"] == FIXED_TREE_WIDTHS
    assert policy["loop_propagators"] == "widthless"
    assert policy["tree_widths"] == {"H": "WH", "Z": "WZ"}
    external = slice_._external_parameters()
    parameters = {
        key: value
        for key, value in fixture["scheme_data"]["parameters"].items()
        if key in external and key not in slice_.fixed_parameters
    }
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    scales = [Fraction(mu) ** 2 for mu in fixture["mu_r"]]
    results = executor.evaluate(
        fixture["points"], scales, parameters=parameters, precision=Precision.double()
    )
    _check_providers(name, results, analytic=False)
    (dd,) = executor.evaluate(
        fixture["points"][:1],
        scales[:1],
        parameters=parameters,
        precision=Precision.double_double(),
    )
    assert _relative(_total(dd), _total(results[0])) <= ANALYTIC_DOUBLE


@pytest.mark.parametrize("name", tiers(["li_h_za", "li_aa_aa"], "li_h_za"))
def test_widthless_loop_propagators(ew, name):
    """Q7: loop propagators carry no width. h > z a and a a > a a have no
    tree propagator (every tree current of h > z a is amputated; photons
    meet no neutral current at tree level), so at the model's default widths
    their values equal the zero-width ones bit for bit, with the W, Z, top
    and Higgs widths of their loop propagators run-time parameters."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.ew_policy import scheme_conventions
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    fixture = _fixture(name)
    slice_ = LoopSlice.generate(
        ew.compiled, _default_widths_request(ew, name), ew.native
    )
    assert scheme_conventions(slice_)["width_policy"]["tree_widths"] == {}
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    parameters = ew.parameters(name, slice_)
    widths = {"WZ", "WW", "WT", "WH"}
    assert widths <= set(slice_._external_parameters())
    assert not widths & set(slice_.fixed_parameters)
    point, mu = fixture["points"][0], Fraction(fixture["mu_r"][0]) ** 2
    (physical,) = executor.evaluate(
        [point], [mu], parameters=parameters, precision=Precision.double()
    )
    (zero,) = executor.evaluate(
        [point],
        [mu],
        parameters={**parameters, **dict.fromkeys(widths, "0")},
        precision=Precision.double(),
    )
    assert physical.totals == zero.totals


# --------------------------------------------------- complex-mass scheme


@pytest.mark.parametrize(("name", "analytic"), tiers(CMS_ROWS, "li_emep_ggg_cms"))
def test_complex_mass_rows_equal_the_references(ew, name, analytic):
    """LI-C1 (Q7): the complex-mass scheme is consistent everywhere: complex
    pole masses in the tree and the loop propagators, complex couplings.
    Binary64 at every point equals the complex-mass references (Recola,
    MadLoop and OpenLoops where designated, the lax rule; the oracle's
    analytic continuation within 1e-9: the top loop of g g > a a, the W and
    top loops of the complete a a > a a), double-double at the first point
    equals binary64 within 1e-9 (the oracle within 1e-12); the conventions
    record the complex-mass width policy."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.ew_policy import COMPLEX_MASS_WIDTHS, scheme_conventions

    slice_ = ew.slice(name)
    conventions = scheme_conventions(slice_)
    assert conventions["mass_scheme"] == "complex-mass"
    assert conventions["width_policy"]["policy"] == COMPLEX_MASS_WIDTHS
    results = ew.values(name, Precision.double())
    _check_providers(name, results, analytic)
    (dd,) = ew.values(name, Precision.double_double(), points=[0])
    assert _relative(_total(dd), _total(results[0])) <= ANALYTIC_DOUBLE
    if analytic:
        row = _fixture(name)["rows"][0]
        oracle = Decimal(row["providers"]["analytic"]["decimal"])
        assert abs(Decimal(str(_total(dd))) / oracle - 1) <= Decimal(ANALYTIC_DD)
    twin = LOOP_ONLY_TWINS.get(name)
    if twin is not None:
        # The complex masses enter the loop propagators: far beyond the lax
        # rule from the real on-shell value at the same point.
        real = _fixture(twin)
        assert real["points"][0] == _fixture(name)["points"][0]
        on_shell = real["rows"][0]["providers"]["madloop"]["value"]
        assert _relative(_total(results[0]), on_shell) > 1e-4


@pytest.mark.parametrize(
    ("process", "orders", "corrections"),
    [
        pytest.param("g g > z z", {"QCD": 2, "QED": 2}, None, id="gg_zz"),
        pytest.param(
            "a a > z z",
            {"QCD": 0, "QED": 4},
            ("QED",),
            id="aa_zz",
            marks=pytest.mark.loop_nightly,
        ),
    ],
)
def test_an_unstable_external_state_is_refused_under_the_complex_mass_scheme(
    ew, process, orders, corrections
):
    """Q7: under the complex-mass scheme unstable particles are never
    external states: a Z of the card's width is refused typed
    (``cms-external-width``, naming the remedy W_Z = 0) before any tree."""

    from pyamplicol.loops.model_cms import ComplexMassSchemeUnsupported
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    fixture = _fixture("li_aa_aa_cms")
    model = ew.model(fixture)
    fixed = {
        key: value
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
        if key in model.parameter_defaults
    }
    options = dict(EW_OPTIONS)
    if corrections is not None:
        options["corrections"] = corrections
    request = SliceRequest(
        process,
        orders,
        orders,
        contribution="loop-induced",
        fixed_parameters=fixed,
        **options,
    )
    with pytest.raises(ComplexMassSchemeUnsupported) as refused:
        LoopSlice.generate(model, request, ew.native)
    assert refused.value.key == "cms-external-width"
    assert 'WZ = "0"' in str(refused.value)
