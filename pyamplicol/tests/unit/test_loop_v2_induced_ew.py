# SPDX-License-Identifier: 0BSD
"""Electroweak-induced loop-induced rows and the width policy (M-LIT LIT-EW,
m-lit PLAN §3.4, §4.1 LI-A3/A4/A6, LI-E1..E3, LI-C1; RULED Q7, Q38 (a)).

On MG5_aMC's G_mu loop model (UFOmodeldb, 't Hooft-Feynman gauge: W,
Goldstone and ghost loops), against LIT-REF's loop-induced references
(``tests/fixtures/loop/required/li``: MadLoop, Recola and the mpmath oracles
of ``tools/loop_reference/li_analytic.py``), served by NinjaRS per amplitude
(the TIRust rows follow LIT-TIR):

* h > a a, h > z a (the h-Z-gamma mixing through the NLOCT vertex
  counterterm V_UVHZAEW and the R2 vertex V_R2AZH), a a > h, e+ e- > h a,
  the complete [QED] a a > a a (fermion, W, Goldstone and ghost loops:
  Karplus-Neuman + Jikia-Tkabladze and Recola; the only a a > a a row),
  a a > z z, a a > h h and e+ e- > h h equal MadLoop and Recola within the
  lax 1e-5 rule and the analytic oracles within 1e-9 in binary64, 1e-12 in
  double-double; h > a a in arbitrary precision equals the oracle and
  double-double to the on-shell contract;
* the width policy (Q7): a loop-induced request keeps fixed widths in its
  tree propagators and widthless loop propagators, recorded in the
  artifact's conventions; e+ e- > h h [QED] with the model's default Z
  width generates and evaluates, its value moving with the Z width to first
  order and reaching the zero-width references; an electroweak
  Born-interfered request keeps its ``electroweak-width`` refusal;
* the complex-mass scheme rows of LI-C1 (complex m_t in g g > a a and
  g g > a a g, complex M_W and m_t in the complete a a > a a, e+ e- > g g g
  with complex M_Z in its tree propagators, g g > z z with external
  W_Z = 0 and complex m_t) equal the complex-mass references, record the
  complex-mass width policy, and an unstable external state is refused
  typed (``cms-external-width``).
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
LAX = 1e-5
ANALYTIC_DOUBLE = 1e-9
ANALYTIC_DD = 1e-12
# The on-shell contract of double-double against arbitrary precision (L7).
DD_AP = 1e-24
# The G_mu UFO's structural source-compatibility profile and scale
# parameter (``tools/loop_v2_generate``); MadLoop's photons are untagged.
EW_OPTIONS = {
    "compatibility": "gmu_particle_anti",
    "scale_parameter": "MU_R",
    "external_photons": "none",
}

# The electroweak-induced rows (fixture, analytic oracle among its providers).
EW_ROWS = [
    ("li_h_aa", True),
    ("li_h_za", True),
    ("li_aa_h", True),
    ("li_emep_ha", False),
    ("li_aa_aa", True),
    ("li_aa_zz", False),
    ("li_aa_hh", False),
    ("li_emep_hh", False),
    # LI-E4: the lepton tree current on a QCD-corrected quark loop.
    ("li_emep_ggg", False),
]
# The complex-mass rows of LI-C1 (fixture, analytic oracle).
CMS_ROWS = [
    ("li_gg_aa_cms", True),
    ("li_gg_aag_cms", False),
    ("li_aa_aa_cms", True),
    ("li_emep_ggg_cms", False),
    ("li_gg_zz_cms", False),
]


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _process(fixture: dict) -> str:
    text = fixture["madloop_process"].split("[")[0]
    return " ".join(token for token in text.split() if "=" not in token)


def _corrections(fixture: dict) -> tuple[str, ...] | None:
    """The fixture's correction ([sqrvirt=QED]: QED; QCD: the model's
    strong one, None)."""

    return ("QED",) if "QED" in fixture["madloop_process"].split("[")[1] else None


class _EwRows:
    """The G_mu model (real on-shell and, per card, complex-mass), the
    generated slices and native executors of the fixture rows (once each)."""

    def __init__(self, ufo, native, root):
        from pyamplicol.models.loading import compile_model_source

        self.native, self.root = native, root
        self.compiled = compile_model_source(
            ufo, require_supported=False, cache_dir=root / "model-cache"
        )
        self.models: dict = {}
        self.slices: dict = {}
        self.executors: dict = {}

    def model(self, fixture: dict):
        """The fixture's model: the real on-shell G_mu model, or its
        complex-mass model at the fixture's card (``model_cms``)."""

        data = fixture["scheme_data"]
        if not data.get("complex_mass_scheme"):
            return self.compiled
        key = json.dumps(data["fixed_parameters"], sort_keys=True)
        found = self.models.get(key)
        if found is None:
            from pyamplicol.loops.model_cms import complex_mass_model

            found, _scheme = complex_mass_model(
                self.compiled,
                fixed_parameters=dict(data["fixed_parameters"]),
                cache_dir=self.root / "model-cache",
            )
            self.models[key] = found
        return found

    def request(self, fixture: dict, **fields):
        from pyamplicol.loops.v2.assemble import SliceRequest

        compiled = self.model(fixture)
        orders = dict(fixture["loop_orders"])
        fixed = {
            name: value
            for name, value in fixture["scheme_data"]["fixed_parameters"].items()
            if name in compiled.parameter_defaults
        }
        corrections = _corrections(fixture)
        options = {**EW_OPTIONS, **fields}
        if corrections is not None:
            options["corrections"] = corrections
        return SliceRequest(
            _process(fixture),
            orders,
            orders,
            contribution="loop-induced",
            fixed_parameters=fixed,
            **options,
        )

    def slice(self, name: str):
        found = self.slices.get(name)
        if found is None:
            from pyamplicol.loops.v2.assemble import LoopSlice

            fixture = _fixture(name)
            found = self.slices[name] = LoopSlice.generate(
                self.model(fixture), self.request(fixture), self.native
            )
        return found

    def executor(self, name: str):
        found = self.executors.get(name)
        if found is None:
            from pyamplicol.loops.v2.plan_export import NativeExecutor

            found = self.executors[name] = NativeExecutor(
                self.slice(name), forward=False, inputs="native"
            )
        return found

    def parameters(self, name: str, slice_=None) -> dict[str, str]:
        """The fixture's parameters the slice takes at run time."""

        slice_ = slice_ or self.slice(name)
        external = slice_._external_parameters()
        return {
            key: value
            for key, value in _fixture(name)["scheme_data"]["parameters"].items()
            if key in external and key not in slice_.fixed_parameters
        }

    def values(self, name: str, precision, points=None, **parameters):
        """The native results at the fixture's points (all, or ``points``)."""

        fixture = _fixture(name)
        indices = range(len(fixture["points"])) if points is None else points
        return self.executor(name).evaluate(
            [fixture["points"][i] for i in indices],
            [Fraction(fixture["mu_r"][i]) ** 2 for i in indices],
            parameters={**self.parameters(name), **parameters},
            precision=precision,
        )


@pytest.fixture(scope="module")
def ew(loop_gmu_ufo, loop_native, tmp_path_factory):
    return _EwRows(loop_gmu_ufo, loop_native, tmp_path_factory.mktemp("li-ew"))


def _relative(value, reference) -> float:
    return abs(float(value) / float(reference) - 1)


def _total(result):
    """The finite |A|^2 summed over sectors."""

    return result.totals[1]


def _check_providers(name, results, analytic):
    """Every point's binary64 value against each provider (MadLoop, Recola,
    OpenLoops where designated: the lax rule) and the analytic oracle (1e-9);
    the poles of |A|^2 vanish to rounding (a necessary-only cross-check)."""

    for index, result in enumerate(results):
        row = _fixture(name)["rows"][index]
        assert result.born == {}
        total = _total(result)
        for provider, record in row["providers"].items():
            if provider == "analytic":
                continue
            assert _relative(total, record["value"]) <= LAX, (index, provider)
        if analytic:
            oracle = row["providers"]["analytic"]["decimal"]
            assert _relative(total, oracle) <= ANALYTIC_DOUBLE, index
        for pole in result.totals[2:]:
            assert abs(float(pole)) <= 1e-9 * abs(float(total))


# ------------------------------------------------------------------ rows


@pytest.mark.parametrize(("name", "analytic"), EW_ROWS, ids=[r[0] for r in EW_ROWS])
def test_ew_rows_equal_madloop_recola_and_the_oracle(ew, name, analytic):
    """LI-A3, A4, A6, E1-E4 in binary64 at every reference point: MadLoop
    and Recola within the lax rule, the oracle within 1e-9; the [QED] rows
    generated in 't Hooft-Feynman gauge (W, Goldstone and ghost loops), at
    zero widths (the references' scheme), recording the fixed-tree-width
    policy."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS, scheme_conventions

    slice_ = ew.slice(name)
    # [QED] requests take 't Hooft-Feynman gauge; the QCD-corrected
    # e+ e- > g g g keeps the compiled model's.
    if _corrections(_fixture(name)) is not None:
        assert str(slice_.compiled.ir.gauge) == "feynman"
    _check_providers(name, ew.values(name, Precision.double()), analytic)
    policy = scheme_conventions(slice_)["width_policy"]
    assert policy["policy"] == FIXED_TREE_WIDTHS
    assert policy["loop_propagators"] == "widthless"
    assert policy["tree_widths"] == {}


@pytest.mark.parametrize(("name", "analytic"), EW_ROWS, ids=[r[0] for r in EW_ROWS])
def test_ew_rows_in_double_double(ew, name, analytic):
    """Double-double at the first point: the oracle within 1e-12 where there
    is one, else binary64 within 1e-9 and the providers within the lax rule."""

    from pyamplicol.loops.v2.assemble import Precision

    (dd,) = ew.values(name, Precision.double_double(), points=[0])
    row = _fixture(name)["rows"][0]
    if analytic:
        oracle = Decimal(row["providers"]["analytic"]["decimal"])
        assert abs(Decimal(str(_total(dd))) / oracle - 1) <= Decimal(ANALYTIC_DD)
    else:
        (double,) = ew.values(name, Precision.double(), points=[0])
        assert _relative(_total(dd), _total(double)) <= ANALYTIC_DOUBLE
        assert _relative(_total(dd), row["providers"]["madloop"]["value"]) <= LAX


def test_h_to_a_a_in_arbitrary_precision(ew):
    """AP on the tier's analytic row (h > a a: W, Goldstone, ghost and
    fermion triangles): the oracle within 1e-12 and double-double within the
    on-shell contract 1e-24 (L7)."""

    from pyamplicol.loops.v2.assemble import Precision

    name = "li_h_aa"
    (ap,) = ew.values(name, Precision.arbitrary(40), points=[0])
    (dd,) = ew.values(name, Precision.double_double(), points=[0])
    oracle = Decimal(_fixture(name)["rows"][0]["providers"]["analytic"]["decimal"])
    with localcontext() as context:
        context.prec = 50
        value = Decimal(str(_total(ap)))
        assert abs(value / oracle - 1) <= Decimal(ANALYTIC_DD)
        assert abs(Decimal(str(_total(dd))) / value - 1) <= Decimal(DD_AP)


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


@pytest.mark.parametrize("name", ["li_emep_hh", "li_emep_ggg"])
def test_lepton_rows_with_the_model_z_width(ew, name):
    """Q7, LI-E3 / LI-E4: e+ e- > h h [QED] and e+ e- > g g g at the model's
    default widths generate with fixed widths in their tree propagators (the
    s-channel Z) and widthless loop propagators, recorded in their
    conventions (no ``electroweak-width`` refusal: they are loop-induced).
    They evaluate at the model's Z width; the Z width is a run-time
    parameter of the tree propagators: at W_Z = 0 the value equals the
    zero-width references (MadLoop, Recola: Recola is mandatory for
    e+ e- > h h, no Ward probe reaches a process without an external gauge
    boson), and the shift from there is of first order in W_Z (the tree
    propagator's width times the loop amplitudes' absorptive parts)."""

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
        assert shift != 0, index
        # First order in W_Z: the second-order remainder is of relative size
        # M_Z W_Z / |s - M_Z^2| < 1e-3 at these points.
        assert abs(shift / small / 10 - 1) <= 0.05, (index, shift, small)


@pytest.mark.parametrize("name", ["li_h_za", "li_aa_aa"])
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


def _ew_card(output: Path, ufo: Path, entries, fixed: dict[str, str]) -> str:
    listed = ", ".join(
        f"{{expression = {json.dumps(expression)}, name = {json.dumps(name)}}}"
        for name, expression in entries
    )
    parameters = ", ".join(f"{key} = {json.dumps(v)}" for key, v in fixed.items())
    return f"""
action = "generate"
[model]
source = {json.dumps(str(ufo))}
[process]
entries = [{listed}]
loop_coupling_orders = ["QED"]
[loop]
r2_strategy = "model"
external_photons = "none"
[loop.preparation]
fixed_parameters = {{{parameters}}}
compatibility = "gmu_particle_anti"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def test_born_interfered_width_refusal_unchanged_loop_induced_served(
    ew, loop_gmu_ufo, tmp_path
):
    """One dry run at the model's default widths, [QED]: the Born-interfered
    e+ e- > m+ m- keeps its ``electroweak-width`` refusal (the s-channel Z,
    remedy: zero widths or the complex-mass scheme), the loop-induced
    e+ e- > h h is served with the fixed-tree-width policy (Q7)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS

    entries = (("mumu", "e+ e- > m+ m-"), ("hh", "e+ e- > h h"))
    fixed = {
        key: value
        for key, value in _fixture("li_emep_hh")["scheme_data"][
            "fixed_parameters"
        ].items()
        if key in ew.compiled.parameter_defaults and not key.startswith("W")
    }
    card = tmp_path / "card.toml"
    card.write_text(_ew_card(tmp_path / "out", loop_gmu_ufo, entries, fixed))
    plan = Generator(resolve_config(card)).plan(
        ProcessSet(tuple(ProcessRequest.parse(e, name=n) for n, e in entries))
    )
    (blocker,) = plan.unsupported_features
    assert blocker.startswith("electroweak-width: "), blocker
    assert "WZ" in blocker and "complex-mass" in blocker and "m+ m-" in blocker
    (record,) = (
        record
        for record in plan.estimated_coverage["processes"]
        if record.get("contribution") == "loop-induced"
    )
    assert record["key"] == "hh" and record["width_policy"] == FIXED_TREE_WIDTHS


# --------------------------------------------------- complex-mass scheme


@pytest.mark.parametrize(("name", "analytic"), CMS_ROWS, ids=[r[0] for r in CMS_ROWS])
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


def test_complex_masses_enter_the_loop_propagators(ew):
    """The complete a a > a a under the complex-mass scheme differs from the
    real on-shell one at the same point by the widths' effect on its W and
    top loops (no tree propagator): far beyond the lax rule, and each equals
    its own references (``test_complex_mass_rows_equal_the_references``,
    ``test_ew_rows_equal_madloop_recola_and_the_oracle``)."""

    from pyamplicol.loops.v2.assemble import Precision

    cms = _fixture("li_aa_aa_cms")
    real = _fixture("li_aa_aa")
    assert cms["points"][0] == real["points"][0]
    (complex_mass,) = ew.values("li_aa_aa_cms", Precision.double(), points=[0])
    (on_shell,) = ew.values("li_aa_aa", Precision.double(), points=[0])
    assert _relative(_total(complex_mass), _total(on_shell)) > 1e-4


@pytest.mark.parametrize(
    ("process", "orders", "corrections"),
    [
        ("g g > z z", {"QCD": 2, "QED": 2}, None),
        ("a a > z z", {"QCD": 0, "QED": 4}, ("QED",)),
    ],
    ids=["gg_zz", "aa_zz"],
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
