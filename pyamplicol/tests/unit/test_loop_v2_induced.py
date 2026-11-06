# SPDX-License-Identifier: 0BSD
"""Loop-induced |A_loop|^2 on route R1 (M-LIT LIT-AMP, ``loops/v2/induced``).

A request without a Born amplitude within its coupling-order limits is
generated in amplitude mode: loop x loop sectors, the amplitude basis T of
LIT-COL, one-hot (row, t) columns seeded with c_{c,t} (schema-5 plans), the
native amplitude sum a = i/16pi^2 sum_f Omega_f, the counterterm kets on T
and the canonical square sum_h a^dagger G a / avg. Checked here, on the
loop-induced reference fixtures of LIT-REF (``tests/fixtures/loop/required/
li``, MadLoop / Recola / OpenLoops and the mpmath oracles of
``tools/loop_reference/li_analytic.py``):

* the native evaluation equals MadLoop within the lax 1e-5 rule and the
  analytic oracles within 1e-9 in binary64, 1e-12 in double-double;
* the Python oracle (the reference executor on R1 seeds, the Python
  amplitude assembly and square) equals the native route to rounding, and
  the native point inputs (tree program, unit Born components, native
  counterterm kets) equal the Python point inputs;
* the F2 oracle: the amplitude fed as the Born of the unchanged seeded path
  gives 2|A|^2;
* the poles of |A|^2 vanish within the stability estimate (finiteness);
* h > g a is an exact colour zero with its record (every loop has a single
  coloured attachment, Tr T^a = 0), g g > h a a numerical zero within its
  absolute estimate;
* the native canonical square equals the Python one, and seeded plans keep
  schema 4.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
LAX = 1e-5
ANALYTIC_DOUBLE = 1e-9
ANALYTIC_DD = 1e-12
# The Python oracle dresses and sums in another order than the native
# executor: rounding only (precision is lax, < 1e-5 is never a finding).
ORACLE = 1e-10
VETO = {"t": 6, "b": 5}


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _request(fixture: dict, compiled):
    from pyamplicol.loops.v2.assemble import SliceRequest

    text = fixture["madloop_process"].split("[")[0]
    veto: tuple[int, ...] = ()
    if "/" in text:
        text, vetoed = text.split("/")
        veto = tuple(VETO[name] for name in vetoed.split())
    process = " ".join(token for token in text.split() if "=" not in token)
    orders = dict(fixture["loop_orders"])
    fixed = {
        name: value
        for name, value in fixture["scheme_data"]["fixed_parameters"].items()
        if name in compiled.parameter_defaults
    }
    return SliceRequest(
        process,
        orders,
        orders,
        contribution="loop-induced",
        fixed_parameters=fixed,
        particle_veto=veto,
    )


class _Rows:
    """Generated slices and executors of the fixture rows (once each)."""

    def __init__(self, ufo, native, root):
        self.ufo, self.native, self.root = ufo, native, root
        self.models: dict = {}
        self.slices: dict = {}
        self.executors: dict = {}

    def model(self, restriction: str):
        found = self.models.get(restriction)
        if found is None:
            from tools.loop_benchmark.mqcd import compile_restriction

            found = self.models[restriction] = compile_restriction(
                self.ufo, restriction, self.root / restriction
            )
        return found

    def slice(self, name: str):
        found = self.slices.get(name)
        if found is None:
            from pyamplicol.loops.v2.assemble import LoopSlice

            fixture = _fixture(name)
            compiled = self.model(
                fixture["scheme_data"].get("restriction") or "default"
            )
            found = self.slices[name] = LoopSlice.generate(
                compiled, _request(fixture, compiled), self.native
            )
        return found

    def executor(self, name: str, inputs: str = "python"):
        key = (name, inputs)
        found = self.executors.get(key)
        if found is None:
            from pyamplicol.loops.v2.plan_export import NativeExecutor

            found = self.executors[key] = NativeExecutor(
                self.slice(name), forward=False, inputs=inputs
            )
        return found

    def point(self, name: str, index: int = 0):
        """(momenta, mu_R^2, parameters) of fixture point ``index``."""

        fixture = _fixture(name)
        slice_ = self.slice(name)
        external = slice_._external_parameters()
        fixed = slice_.fixed_parameters
        parameters = {
            key: value
            for key, value in fixture["scheme_data"]["parameters"].items()
            if key in external and key not in fixed
        }
        return (
            fixture["points"][index],
            Fraction(fixture["mu_r"][index]) ** 2,
            parameters,
        )

    def native_value(self, name, precision, *, inputs="python", index=0, **options):
        point, scale, parameters = self.point(name, index)
        (result,) = self.executor(name, inputs).evaluate(
            [point], [scale], parameters=parameters, precision=precision, **options
        )
        return result

    def python_value(self, name, precision, index=0, **options):
        point, scale, parameters = self.point(name, index)
        return self.slice(name).evaluate(
            point, scale, parameters=parameters, precision=precision, **options
        )


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li"))


def _relative(value, reference) -> float:
    return abs(float(value) / float(reference) - 1)


def _total(result):
    """The finite |A|^2 summed over sectors."""

    return result.totals[1]


# ------------------------------------------------------------------ units


@pytest.mark.loop_fast
def test_loop_sectors_pair_loop_orders_once():
    from pyamplicol.loops.v2.induced import kappa, loop_sectors

    a = (("QCD", 2), ("QED", 2))
    b = (("QCD", 4),)
    sectors = dict(loop_sectors([b, a, a]))
    assert sectors == {
        (("QCD", 4), ("QED", 4)): frozenset({(a, a)}),
        (("QCD", 6), ("QED", 2)): frozenset({(b, a)} if b < a else {(a, b)}),
        (("QCD", 8),): frozenset({(b, b)}),
    }
    assert kappa((a, a)) == 1 and kappa((a, b)) == 2


@pytest.mark.loop_fast
def test_native_square_equals_the_python_square(loop_native):
    """The canonical square (``interference::square_amplitudes``) is the
    Python oracle's operation for operation: equal on random vectors with
    two loop orders, three basis elements, two rows and poles."""

    from pyamplicol.loops.v2.assemble import Precision, _Numbers
    from pyamplicol.loops.v2.induced import native_square, square

    rng = random.Random(7)
    numbers = _Numbers(Precision.double())
    orders = [(("QCD", 2),), (("QCD", 3),)]
    rows = [((0, -1),), ((0, 1),)]
    size, width = 3, 3
    gram = [[complex(rng.uniform(-1, 1)) for _ in range(size)] for _ in range(size)]
    gram = [[gram[t][u] + gram[u][t] for u in range(size)] for t in range(size)]
    amplitudes = {
        (order, row): [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(width)]
            for _ in range(size)
        ]
        for order in orders
        for row in rows
    }
    sectors = (
        ((("QCD", 4),), frozenset({(orders[0], orders[0])})),
        ((("QCD", 5),), frozenset({(orders[0], orders[1])})),
        ((("QCD", 6),), frozenset({(orders[1], orders[1])})),
    )
    python = square(amplitudes, gram, sectors, rows, width, numbers, 4)
    native = native_square(amplitudes, gram, sectors, rows, width, 4, loop_native)
    assert python == native


# ------------------------------------------------------------------ rows

# (fixture, analytic oracle in the fixture's rows)
ROWS = [
    ("li_gg_h", True),
    ("li_gg_h_tb", True),
    ("li_h_gg", True),
    ("li_gg_aa_nf5", True),
    ("li_gg_aa", True),
    ("li_gg_hh", False),
    ("li_gg_zz", False),
    ("li_gg_hg", False),
    ("li_uux_hg", False),
]


@pytest.mark.loop_physics
@pytest.mark.parametrize(("name", "analytic"), ROWS, ids=[row[0] for row in ROWS])
def test_binary64_equals_madloop_the_oracle_and_the_python_route(li, name, analytic):
    from pyamplicol.loops.v2.assemble import Precision

    row = _fixture(name)["rows"][0]
    native = li.native_value(name, Precision.double())
    assert native.born == {}
    assert _relative(_total(native), row["providers"]["madloop"]["value"]) <= LAX
    if analytic:
        oracle = row["providers"]["analytic"]["decimal"]
        assert _relative(_total(native), oracle) <= ANALYTIC_DOUBLE
    python = li.python_value(name, Precision.double())
    assert _relative(_total(python), _total(native)) <= ORACLE
    # Finite one-loop amplitude: the poles of |A|^2 are rounding (a
    # necessary-only cross-check, PLAN §2.a.7).
    for pole in native.totals[2:]:
        assert abs(float(pole)) <= 1e-9 * abs(float(_total(native)))


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name", [name for name, analytic in ROWS if analytic and name != "li_gg_aa"]
)
def test_double_double_equals_the_analytic_oracle(li, name):
    from pyamplicol.loops.v2.assemble import Precision

    oracle = _fixture(name)["rows"][0]["providers"]["analytic"]["decimal"]
    native = li.native_value(name, Precision.double_double())
    from decimal import Decimal

    assert (
        abs(Decimal(str(_total(native))) / Decimal(oracle) - 1) <= ANALYTIC_DD
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["li_gg_h", "li_gg_hg"])
def test_native_point_inputs_equal_the_python_point_inputs(li, name):
    from pyamplicol.loops.v2.assemble import Precision

    python = li.native_value(name, Precision.double())
    native = li.native_value(name, Precision.double(), inputs="native")
    assert _relative(_total(native), _total(python)) <= ORACLE


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["li_gg_h", "li_gg_hg"])
def test_f2_oracle_gives_twice_the_square(li, name):
    """PLAN §2.a.2 item 7: a (bare + CT, finite part) fed as the Born
    components of the unchanged seeded path (the amplitude basis as Born
    basis, overlap cells <t|kappa_c>) returns 2|A|^2: the seeds conjugate
    the Born, so bare + CT interference = 2 sum_h a^dagger G a / avg."""

    from pyamplicol.loops.v2.assemble import Precision, SectorSeeds, _Numbers
    from pyamplicol.loops.v2.colour import build_seed_table
    from pyamplicol.loops.v2 import reference_executor as rx

    slice_ = li.slice(name)
    point, scale, parameters = li.point(name)
    precision = Precision.double()
    numbers = _Numbers(precision)
    with numbers.scope():
        evaluation = slice_._point(
            tuple(tuple(Fraction(x) for x in p) for p in point),
            scale,
            slice_._point_parameters(parameters, scale, precision),
            precision,
            "none",
        ).finish(None)
        squared = evaluation.result.totals[1]
        amplitude = evaluation.amplitude_vectors
        basis = slice_.amplitude.seed_basis
        readings = rx.reading_skeletons(slice_.plan)
        orders = sorted({order for order, _row in amplitude})
        born = {
            row: {
                (t, order): {0: amplitude[(order, row)][t][0]}
                for order in orders
                for t in range(len(basis.tensors))
            }
            for _order, row in amplitude
        }
        basis_orders = type(basis)(
            basis.legs,
            basis.sector_ids,
            basis.tensors,
            tuple(frozenset(orders) for _ in basis.tensors),
        )
        table, class_of = build_seed_table(
            slice_.colour,
            readings,
            basis_orders,
            compress=slice_.request.compress_colour,
        )
        pairs = frozenset((a, b) for a in orders for b in orders)
        seeds = SectorSeeds(table, class_of, born, [pairs], numbers, "full")
        evaluation._reduce(evaluation.dress(seeds=seeds).numerators, None, ["f2"])
        bare = evaluation._bare(evaluation.families)["f2"].coefficients[0]
        gram = evaluation.gram
        counterterm = 0
        for (order, row), ket in evaluation.counterterm_vectors.items():
            for left in orders:
                a = amplitude[(left, row)]
                value = sum(
                    a[t][0].conjugate() * gram[t][u] * ket[u][0]
                    for t in range(len(gram))
                    for u in range(len(gram))
                )
                counterterm += 2 * value.real
        f2 = bare + counterterm / evaluation.average
    assert abs(f2 / (2 * squared) - 1) <= ORACLE


# ------------------------------------------------------------------ zeros


@pytest.mark.loop_physics
def test_h_to_g_a_is_an_exact_colour_zero(li):
    """Every loop of h > g a has a single coloured attachment (Tr T^a = 0):
    the slice records the exact colour zero at its loop order, has no
    family and no counterterm ket, and |A|^2 is exactly 0 (RULED Q13: the
    one exact-zero mechanism)."""

    from pyamplicol.loops.v2.assemble import (
        COLOUR_ZERO_REJECTION,
        LoopSlice,
        Precision,
        SliceRequest,
    )

    compiled = li.model("no_b_mass")
    orders = {"QCD": 1, "QED": 2}
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest("h > g a", orders, orders, contribution="loop-induced"),
        li.native,
    )
    assert slice_.colour_zero_orders == {(("QCD", 1), ("QED", 2))}
    assert not slice_.families.families and slice_.amplitude.colour_zero
    assert any(e.reason == COLOUR_ZERO_REJECTION for e in slice_.families.rejected)
    result = slice_.evaluate(
        [["125", "0", "0", "0"], ["62.5", "0", "0", "62.5"], ["62.5", "0", "0", "-62.5"]],
        Fraction(125) ** 2,
        precision=Precision.double(),
    )
    assert all(value == 0 for value in result.totals)


@pytest.mark.loop_physics
def test_g_g_to_h_a_is_zero_within_its_absolute_estimate(li):
    """g g > h a (Furry: the quark loop's two orientations cancel) is served
    numerically (Q13: no Furry-by-rule zero): |A|^2 lies within ten times
    its absolute stability estimate, in binary64 and double-double."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    reference = _fixture("li_gg_h")["rows"][0]["value"]
    for precision in (Precision.double(), Precision.double_double()):
        result = li.native_value(
            "li_gg_ha", precision, stability=RECOMMENDED_CHECKS
        )
        (sector,) = result.stability.absolute
        estimate = result.stability.absolute[sector].coefficients[0]
        assert abs(float(_total(result))) <= 10 * estimate
        assert estimate < 1e-12 * reference


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["li_gg_h", "li_gg_zz"])
def test_poles_vanish_within_the_stability_estimate(li, name):
    """The finiteness cross-check (PLAN §2.a.7): with stability checks the
    single and double poles of |A|^2 lie within ten times their absolute
    estimate."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    result = li.native_value(name, Precision.double(), stability=RECOMMENDED_CHECKS)
    for sector, laurent in result.components["virtual"].items():
        estimate = result.stability.absolute[sector].coefficients
        for k in (1, 2):
            assert abs(float(laurent.coefficients[k])) <= 10 * estimate[k]


# ------------------------------------------------------------------ plans


@pytest.mark.loop_physics
def test_amplitude_mode_plans_are_schema_5_and_seeded_plans_schema_4(li):
    from pyamplicol.loops.v2.plan_export import (
        AMPLITUDE_PLAN_SCHEMA,
        PLAN_SCHEMA,
        export_plan,
    )

    slice_ = li.slice("li_gg_hg")
    exported = export_plan(slice_, forward=False)
    program = exported.program
    size = len(slice_.amplitude)
    assert program["schema"] == AMPLITUDE_PLAN_SCHEMA and PLAN_SCHEMA == 4
    assert exported.amplitude == size
    assert program["columns"] == program["rows"] * size
    assert program["column_rows"] == [
        [r] for r in range(program["rows"]) for _t in range(size)
    ]
    assert program["column_weights"] == [
        t for _r in range(program["rows"]) for t in range(size)
    ]
    assert program["born_components"] == 1
