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
* an electroweak Born-interfered request keeps its ``electroweak-width``
  refusal while a loop-induced one is served with the fixed-tree-width
  policy (Q7); the width policy's rows and the complex-mass rows of LI-C1
  are ``test_loop_v2_induced_ew_widths.py``'s (two files: each holds the
  CI job's 12 GiB process-tree limit with room).
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


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _process(fixture: dict, compiled) -> str:
    """The fixture's process in the model's particle names (by PDG)."""

    names = {int(r.pdg_code): str(r.name) for r in compiled.ir.particles}
    pdgs, initial = fixture["pdgs"], fixture["initial"]
    return (
        " ".join(names[pdg] for pdg in pdgs[:initial])
        + " > "
        + " ".join(names[pdg] for pdg in pdgs[initial:])
    )


def _corrections(fixture: dict) -> tuple[str, ...] | None:
    """The fixture's correction ([sqrvirt=QED]: QED; QCD: the model's
    strong one, None)."""

    return ("QED",) if "QED" in fixture["madloop_process"].split("[")[1] else None


class _EwRows:
    """The G_mu model (real on-shell and, per card, complex-mass), and the
    generated slice and native executor of the current fixture row: one row
    is kept at a time (an electroweak slice holds about 1-2 GB; the CI job
    runs a file under a 12 GiB process-tree limit), so each test of a row
    takes all its checks at once."""

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
            _process(fixture, compiled),
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

            # Release the previous row before generating this one.
            self.slices.clear()
            self.executors.clear()
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
    and Recola within the lax rule, the oracle within 1e-9; in double-double
    at the first point: the oracle within 1e-12 where there is one, else
    binary64 within 1e-9 and MadLoop within the lax rule. The [QED] rows are
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
    results = ew.values(name, Precision.double())
    _check_providers(name, results, analytic)
    policy = scheme_conventions(slice_)["width_policy"]
    assert policy["policy"] == FIXED_TREE_WIDTHS
    assert policy["loop_propagators"] == "widthless"
    assert policy["tree_widths"] == {}
    (dd,) = ew.values(name, Precision.double_double(), points=[0])
    row = _fixture(name)["rows"][0]
    if analytic:
        oracle = Decimal(row["providers"]["analytic"]["decimal"])
        assert abs(Decimal(str(_total(dd))) / oracle - 1) <= Decimal(ANALYTIC_DD)
    else:
        assert _relative(_total(dd), _total(results[0])) <= ANALYTIC_DOUBLE
        assert _relative(_total(dd), row["providers"]["madloop"]["value"]) <= LAX


@pytest.mark.parametrize("name", ["li_h_aa", "li_gg_aa_cms"])
def test_arbitrary_precision_on_one_row_per_tier(ew, name):
    """AP on one analytic row per tier (the EW rows: h > a a, W, Goldstone,
    ghost and fermion triangles; LI-C1: g g > a a with a complex top mass):
    the oracle within 1e-12 and double-double within the on-shell contract
    1e-24 (L7)."""

    from pyamplicol.loops.v2.assemble import Precision

    (ap,) = ew.values(name, Precision.arbitrary(40), points=[0])
    (dd,) = ew.values(name, Precision.double_double(), points=[0])
    oracle = Decimal(_fixture(name)["rows"][0]["providers"]["analytic"]["decimal"])
    with localcontext() as context:
        context.prec = 50
        value = Decimal(str(_total(ap)))
        assert abs(value / oracle - 1) <= Decimal(ANALYTIC_DD)
        assert abs(Decimal(str(_total(dd))) / value - 1) <= Decimal(DD_AP)


# -------------------------------------------------- the width refusal


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
    """Dry runs at the model's default widths, [QED]: the Born-interfered
    e+ e- > m+ m- keeps its ``electroweak-width`` refusal (the s-channel Z,
    remedy: zero widths or the complex-mass scheme), the loop-induced
    e+ e- > h h is served with the fixed-tree-width policy (Q7)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS
    from pyamplicol.models.loading import compile_model_source

    # The dry run reads compiled models from the cache only.
    compile_model_source(loop_gmu_ufo, require_supported=False)
    fixed = {
        key: value
        for key, value in _fixture("li_emep_hh")["scheme_data"][
            "fixed_parameters"
        ].items()
        if key in ew.compiled.parameter_defaults and not key.startswith("W")
    }

    def plan(name, expression):
        card = tmp_path / name / "card.toml"
        card.parent.mkdir()
        entries = ((name, expression),)
        card.write_text(_ew_card(card.parent / "out", loop_gmu_ufo, entries, fixed))
        return Generator(resolve_config(card)).plan(
            ProcessSet((ProcessRequest.parse(expression, name=name),))
        )

    (blocker,) = plan("mumu", "e+ e- > m+ m-").unsupported_features
    assert blocker.startswith("electroweak-width: "), blocker
    assert "WZ" in blocker and "complex-mass" in blocker
    served = plan("hh", "e+ e- > h h")
    assert served.unsupported_features == ()
    (record,) = (
        record
        for record in served.estimated_coverage["processes"]
        if record.get("contribution") == "loop-induced"
    )
    assert record["key"] == "hh" and record["width_policy"] == FIXED_TREE_WIDTHS
