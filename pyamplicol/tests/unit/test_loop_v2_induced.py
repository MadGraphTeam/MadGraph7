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
* the poles of |A|^2 vanish within the stability estimate, and the
  amplitude-level finiteness and Ward probes pass within ten times their
  own estimates on the per-row term floor (m-lit DECISIONS 2026-11-06
  ~09:00 (1)), whose native and Python forms agree; a non-gauge vector
  fails the Ward probe (it discriminates); the generation-time probes
  (double-double, zero widths, full kernels) pass, and a failed one refuses
  generation unless allowed;
* the result transport is O(rows x T) per point: no reduction lanes reach
  Python unless requested;
* h > g a is an exact colour zero with its record (every loop has a single
  coloured attachment, Tr T^a = 0), g g > h a and g g > a a a are
  numerical zeros within their absolute estimates;
* the native canonical square equals the Python one, and seeded plans keep
  schema 4;
* the public route: generated through ``Generator``, a loop-induced
  artifact (``rusticol.loop-runtime.v5``) loads through ``Runtime``,
  returns a ``VirtualEvaluation`` with zeroed Born fields and
  ``contribution = "loop-induced"`` equal to the slice's native value,
  refuses the "uv" and "ir" pole selections, serves the exact colour zero
  h > g a as 0, and relocates bitwise.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
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
# The v2 loop runtimes: Born-interfered (plan v4) and loop-induced (v5).
V4 = "rusticol.loop-runtime.v4"
V5 = "rusticol.loop-runtime.v5"


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


@pytest.mark.loop_fast
def test_virtual_evaluation_holds_a_loop_induced_result():
    """m-lit Q36: a loop-induced result is a ``VirtualEvaluation`` with
    ``contribution = "loop-induced"``, an empty Born catalogue and
    zero-length Born entries; a virtual result still needs its Born
    catalogue, and a loop-induced one refuses a Born; the default keeps
    every existing value and equality."""

    from pyamplicol.api.virtual_results import VirtualEvaluation, VirtualLaneDiagnostic
    from pyamplicol.config import R2Strategy
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.strategy import R2Selection

    loop = (("QCD", 2), ("QED", 1))
    sector = VirtualOrderSector((("QCD", 4), ("QED", 2)), (BornLoopPair(loop, loop),))
    strategy = R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "model R2")
    success = VirtualLaneDiagnostic("success", "double", 53, False)
    value = VirtualEvaluation(
        ((),),
        (((0.25, 0.0, 0.0),),),
        (),
        (sector,),
        (success,),
        strategy,
        contribution="loop-induced",
    )
    assert value.contribution == "loop-induced" and value.born_shape == (1, 0)
    assert value.finite == ((0.25,),)
    with pytest.raises(ValueError, match="Born and interference catalogues"):
        VirtualEvaluation(
            ((),), (((0.25, 0.0, 0.0),),), (), (sector,), (success,), strategy
        )
    with pytest.raises(ValueError, match="empty Born catalogue"):
        VirtualEvaluation(
            ((1.0,),),
            (((0.25, 0.0, 0.0),),),
            ((("QCD", 4), ("QED", 2)),),
            (sector,),
            (success,),
            strategy,
            contribution="loop-induced",
        )
    with pytest.raises(ValueError, match="unknown contribution"):
        VirtualEvaluation(
            ((),),
            (((0.25, 0.0, 0.0),),),
            (),
            (sector,),
            (success,),
            strategy,
            contribution="loop-squared",
        )


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
@pytest.mark.parametrize("name", [name for name, analytic in ROWS if analytic])
def test_double_double_equals_the_analytic_oracle(li, name):
    from pyamplicol.loops.v2.assemble import Precision

    oracle = _fixture(name)["rows"][0]["providers"]["analytic"]["decimal"]
    native = li.native_value(name, Precision.double_double())
    from decimal import Decimal

    assert abs(Decimal(str(_total(native))) / Decimal(oracle) - 1) <= ANALYTIC_DD


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

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.loops.v2.assemble import Precision, SectorSeeds, _Numbers
    from pyamplicol.loops.v2.colour import build_seed_table

    slice_ = li.slice(name)
    point, scale, parameters = li.point(name)
    precision = Precision.double()
    numbers = _Numbers(precision)
    with numbers.scope():
        evaluation = slice_._point(
            tuple(tuple(Fraction(x) for x in p) for p in point),
            scale,
            parameters,
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
        for (_order, row), ket in evaluation.counterterm_vectors.items():
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


H_TO_G_A_POINT = [
    ["125", "0", "0", "0"],
    ["62.5", "0", "0", "62.5"],
    ["62.5", "0", "0", "-62.5"],
]


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
        H_TO_G_A_POINT, Fraction(125) ** 2, precision=Precision.double()
    )
    assert all(value == 0 for value in result.totals)


@pytest.mark.loop_physics
def test_stored_colour_zero_agrees_with_the_generated_basis(li):
    """The stored amplitude record (``induced.StoredAmplitude``) states the
    request's exact colour zero as the generation's basis does
    (``AmplitudeBasis.colour_zero``: no nonzero coefficient cell and no
    counterterm ket): u u~ > h g at the order QCD = 1, QED = 3 (the quark
    line's Z/photon into a top loop with one gluon, Tr T^a = 0: an exact
    colour zero) and QCD = 3, QED = 1 (the quark line's gluon into a top
    loop, live) is not a colour zero, though one of its orders is; h > g a
    is one."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.induced import StoredAmplitude
    from pyamplicol.loops.v2.native_process import NativeProcess
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    compiled = li.model("no_b_mass")
    fixed = {
        name: value
        for name, value in _fixture("li_uux_hg")["scheme_data"][
            "fixed_parameters"
        ].items()
        if name in compiled.parameter_defaults
    }
    zero = (("QCD", 1), ("QED", 3))
    for process, limits, orders, colour_zero in (
        (
            "u u~ > h g",
            {"QCD": 3, "QED": 3},
            [{"QCD": 3, "QED": 1}, {"QCD": 1, "QED": 3}],
            False,
        ),
        ("h > g a", {"QCD": 1, "QED": 2}, [{"QCD": 1, "QED": 2}], True),
    ):
        slice_ = LoopSlice.generate(
            compiled,
            SliceRequest(
                process,
                limits,
                orders,
                contribution="loop-induced",
                fixed_parameters=fixed,
            ),
            li.native,
        )
        basis = slice_.amplitude
        assert basis.colour_zero is colour_zero, process
        if not colour_zero:
            assert slice_.colour_zero_orders == {zero}
        record = NativeProcess.of(
            NativeExecutor(slice_, forward=False, inputs="native")
        ).record["amplitude"]
        stored = StoredAmplitude.from_record(record)
        assert stored.colour_zero is colour_zero, (process, basis.counterterms)
        assert stored.colour_zero_orders == slice_.colour_zero_orders
        if not colour_zero:
            # The generation's answer, never a guess from the colour-zero
            # orders: without its counterterm kets the record's live cells
            # still make the request no colour zero.
            bare = {**record, "counterterms": [], "ket_map": []}
            assert StoredAmplitude.from_record(bare).colour_zero is False


@pytest.mark.loop_physics
def test_g_g_to_h_a_is_zero_within_its_absolute_estimate(li):
    """g g > h a (Furry: the quark loop's two orientations cancel) is served
    numerically (Q13: no Furry-by-rule zero): |A|^2 lies within ten times
    its absolute stability estimate, in binary64 and double-double."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    reference = _fixture("li_gg_h")["rows"][0]["value"]
    for precision in (Precision.double(), Precision.double_double()):
        result = li.native_value("li_gg_ha", precision, stability=RECOMMENDED_CHECKS)
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
def test_result_transport_is_rows_times_basis(li):
    """PLAN §2.a.4 / Done (6): an amplitude-mode point hands Python its
    amplitude vectors only (loop orders x rows x T slots), no reduction
    lanes; the raw lanes come only on request (``raw_lanes``, the oracle
    route), with the same values."""

    from pyamplicol.loops.v2.assemble import Precision

    name = "li_gg_hg"
    fixture = _fixture(name)
    _point, _scale, parameters = li.point(name)
    points = fixture["points"]
    scales = [str(Fraction(mu) ** 2) for mu in fixture["mu_r"]]
    executor = li.executor(name, "native")
    exported = executor.exported
    default = executor.evaluate(
        points, scales, parameters=parameters, precision=Precision.double()
    )
    slots = len(exported.orders) * len(exported.rows) * exported.amplitude
    lanes = executor.last_report["counts"]["reduction lanes"]
    assert lanes > slots and all(not result.families for result in default)
    executor.raw_lanes = True
    try:
        raw = executor.evaluate(
            points, scales, parameters=parameters, precision=Precision.double()
        )
    finally:
        executor.raw_lanes = False
    assert sum(len(result.families) for result in raw) == lanes
    assert [r.totals for r in raw] == [r.totals for r in default]


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


# ------------------------------------------------------------------ probes


def _evaluation(executor, point, scale, parameters, precision):
    """The point evaluation on a native executor with the reduction re-test
    and the term scale (``induced.probe``'s run)."""

    executor.term_scale = True
    try:
        _precision, (evaluation,), failures = executor._evaluated(
            [point],
            [scale],
            parameters,
            precision,
            "all",
            "reversed",
        )
    finally:
        executor.term_scale = False
    assert not failures
    return evaluation


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["li_gg_h", "li_gg_hg"])
def test_native_term_floor_equals_the_python_oracle(li, name):
    """The native term scale (unit-numerator reductions,
    ``V2Options::term_scale``) equals the Python reference's (its own unit
    numerators through the reducer) per (loop order, row) and Laurent
    coefficient, after the per-row floor (``induced.row_scales``)."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import Precision, _fraction

    precision = Precision.double()
    slice_ = li.slice(name)
    point, scale, parameters = li.point(name)
    native = induced.row_scales(
        _evaluation(li.executor(name), point, scale, parameters, precision)
    )
    exact = tuple(tuple(_fraction(x) for x in p) for p in point)
    python = slice_._evaluate(
        exact, scale, dict(parameters), precision, "all", "reversed"
    )
    oracle = induced.row_scales(python)
    assert set(native) == set(oracle) and native
    for key, values in native.items():
        for value, reference in zip(values, oracle[key], strict=True):
            assert reference > 0
            assert abs(value / reference - 1) <= ORACLE, key


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "precision"),
    [
        ("li_gg_h", "double"),
        ("li_gg_hh", "double"),
        ("li_gg_aa_nf5", "double"),
        ("li_gg_zz", "double"),
        ("li_gg_zz", "double_double"),
    ],
)
def test_finiteness_probe_passes_on_the_term_floor(li, name, precision):
    """PLAN §2.a.7 L3 with the DECISIONS ~09:00 (1) scale: per (loop order,
    row) the poles of the amplitude lie within ten times their estimate on
    the row's term floor, also where whole families are finite (massive
    loops) and where single boxes carry 1/eps^2 (g g > a a massless)."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import Precision

    point, scale, parameters = li.point(name)
    result = induced.probe(
        li.executor(name),
        point,
        scale,
        parameters=parameters,
        precision=getattr(Precision, precision)(),
    )
    assert result.kind == "finiteness" and result.ratios
    assert result.passed, result.ratios


@pytest.mark.loop_physics
def test_ward_probe_passes_and_discriminates(li):
    """The amplitude-level Ward probe (PLAN §2.a.7, Q12) on g g > a a
    (massless quarks; every leg a massless gauge boson): epsilon_j -> c
    p_j / E_j for c = 1, i on each leg leaves the amplitude within ten
    times its own estimate (R2 restores the bare part's violation); a
    non-gauge vector on leg 1 fails it by far."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.ward import WARD_FACTORS, gauge_ward, ward_slice

    name = "li_gg_aa_nf5"
    fixture = _fixture(name)
    compiled = li.model(fixture["scheme_data"].get("restriction") or "default")
    slice_ = ward_slice(compiled, _request(fixture, compiled), li.native)
    executor = NativeExecutor(slice_, forward=False, inputs="python")
    point, scale, parameters = li.point(name)
    precision = Precision.double()
    for leg in (1, 2, 3, 4):
        for factor in WARD_FACTORS:
            result = induced.probe(
                executor,
                point,
                scale,
                parameters=parameters,
                precision=precision,
                ward=gauge_ward(leg, factor, point),
            )
            assert result.kind == "ward" and result.passed, (leg, factor, result.ratios)
    control = (1, (("1", "0"), ("0", "0"), ("1", "0"), ("0", "0")))
    result = induced.probe(
        executor, point, scale, parameters=parameters, precision=precision, ward=control
    )
    assert result.worst > 1e6


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "orders", "name", "legs"),
    [
        ("g g > h g", {"QCD": 3, "QED": 1}, "li_gg_hg", [1]),
        # A Furry zero: the residues are rounding of the numerators' own
        # cancellation, which the rotation budget covers.
        ("g g > h a", {"QCD": 2, "QED": 2}, "li_gg_ha", [1, 4]),
    ],
    ids=["gg_hg", "gg_ha"],
)
def test_generation_probes_pass_at_zero_widths_on_the_full_kernels(
    li, process, orders, name, legs
):
    """m-lit Q12 / Q7, PLAN §2.a.7: the generation-time probes at the
    model's widths (none fixed): on a probe slice with the widths fixed to
    zero on the full tree kernels, in double-double at the candidate
    farthest from the tree poles (lifted to exact momentum conservation),
    the finiteness probe and the Ward probe of one leg per gauge-boson
    species pass within ten times their estimates."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    compiled = li.model("no_b_mass")
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(process, orders, orders, contribution="loop-induced"),
        li.native,
    )
    points = _fixture(name)["points"]
    candidates = [[[float(x) for x in p] for p in point] for point in points]
    record = induced.generation_probes(slice_, compiled, li.native, candidates)
    assert record["precision"] == "double-double" and record["widths"] == "zero"
    assert record["zeroed_widths"] == ["WH", "WT", "WW", "WZ"]
    distances = [induced.probe_point(slice_, point, 64)[1] for point in candidates]
    assert record["candidate"] == distances.index(max(distances))
    assert record["finiteness"]["passed"] and record["ward"]["passed"]
    assert record["ward"]["legs"] == legs and "failed" not in record


@pytest.mark.loop_physics
def test_a_failed_generation_probe_refuses_unless_allowed(li, monkeypatch):
    """A non-gauge vector in place of c p_j / E_j fails the Ward probe of
    g g > h: generation refuses (``ProbeFailure``, the internal key
    ``loop-induced-probe``) unless the failure is allowed, which records
    it; the exact colour zero h > g a has nothing to probe."""

    from pyamplicol.loops.v2 import induced, ward
    from pyamplicol.loops.v2.refusals import LoopEngineRefusal, keyed_error

    slice_ = li.slice("li_gg_h")
    compiled = li.model("no_b_mass")
    candidates = [[[float(x) for x in p] for p in li.point("li_gg_h")[0]]]
    control = (("1", "0"), ("0", "0"), ("1", "0"), ("0", "0"))
    monkeypatch.setattr(ward, "gauge_ward", lambda leg, factor, point: (leg, control))
    with pytest.raises(induced.ProbeFailure, match="ward probe") as refused:
        induced.generation_probes(slice_, compiled, li.native, candidates)
    typed = keyed_error(refused.value)
    assert isinstance(typed, LoopEngineRefusal) and typed.key == "loop-induced-probe"
    allowed = induced.generation_probes(
        slice_, compiled, li.native, candidates, allow_failure=True
    )
    assert allowed["allowed_failure"] and allowed["failed"]
    assert allowed["finiteness"]["passed"] and not allowed["ward"]["passed"]


@pytest.mark.loop_physics
def test_g_g_to_a_a_a_is_zero_within_its_absolute_estimate(li):
    """g g > a a a (Furry: the quark loop's two orientations cancel) is
    served numerically (Q13: no Furry-by-rule zero): |A|^2 lies within ten
    times its absolute estimate. The two orientations of a quark loop are
    dressed into one family numerator, so here the numerator coefficients
    themselves cancel and the term scale is of the size of the rounding too:
    the estimate rests on the rotation and re-test budgets, which see that
    cancellation's rounding."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS
    from tools.loop_reference import exact_points

    compiled = li.model("no_b_mass")
    orders = {"QCD": 2, "QED": 3}
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest("g g > a a a", orders, orders, contribution="loop-induced"),
        li.native,
    )
    (point,) = exact_points.generate([0] * 5, count=1)
    point = [[exact_points.decimal_string(x) for x in p] for p in point]
    executor = NativeExecutor(slice_, forward=False, inputs="python")
    (result,) = executor.evaluate(
        [point],
        [Fraction(1000) ** 2],
        precision=Precision.double(),
        stability=RECOMMENDED_CHECKS,
    )
    (sector,) = result.stability.absolute
    estimate = result.stability.absolute[sector].coefficients[0]
    total = abs(float(_total(result)))
    assert total <= 10 * estimate


# ------------------------------------------------------------ public route


def _card(output, entries, fixture) -> str:
    fixed = ", ".join(
        f"{key} = {json.dumps(value)}"
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
    )
    listed = ", ".join(
        f"{{expression = {json.dumps(expression)}, name = {json.dumps(name)}}}"
        for name, expression in entries
    )
    return f"""
action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{listed}]
loop_coupling_orders = ["QCD"]
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _generate(root: Path, entries, fixture) -> Path:
    """Generate ``entries`` ((name, expression), ...) through the public
    ``Generator`` into ``root / "artifact"`` (``_card``)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    output = root / "artifact"
    card = root / "card.toml"
    card.write_text(_card(output, entries, fixture))
    Generator(resolve_config(card)).generate(
        ProcessSet(tuple(ProcessRequest.parse(e, name=n) for n, e in entries)), output
    )
    return output


@pytest.fixture(scope="module")
def li_artifact(li, tmp_path_factory):
    """g g > h and the exact colour zero h > g a in one loop-induced artifact
    (``rusticol.loop-runtime.v5`` only), generated once."""

    entries = (("gg_h", "g g > h"), ("h_ga", "h > g a"))
    return _generate(
        tmp_path_factory.mktemp("li-artifact"), entries, _fixture("li_gg_h")
    )


@pytest.mark.loop_physics
def test_public_route_generates_loads_and_relocates(li, li_artifact, tmp_path):
    """Generator -> artifact -> Runtime: g g > h and the exact colour zero
    h > g a in one loop-induced artifact (``rusticol.loop-runtime.v5``)."""

    import importlib.util

    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.loops.v2.assemble import Precision

    fixture = _fixture("li_gg_h")
    output = li_artifact
    manifest = json.loads((output / "artifact.json").read_text())
    assert manifest["runtime"]["required_runtime_capabilities"] == [
        "rusticol.loop-runtime.v5"
    ]
    # m-lit Q12: the generation-time probes, recorded per process.
    records = {
        record["id"]: record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
    }
    probes = records["gg_h"]["loop_induced_probes"]
    assert probes["finiteness"]["passed"] and probes["ward"]["passed"]
    assert probes["ward"]["legs"] == [1]
    assert records["h_ga"]["loop_induced_probes"] == {"exact_colour_zero": True}
    point, scale, parameters = li.point("li_gg_h")
    runtime = Runtime.load(output, process="gg_h", model_parameters=parameters)
    info = runtime.inspect()
    assert info["contribution_kind"] == "loop-induced"
    # m-lit Q36: the kind lives in the metadata, and the facade serves it with
    # virtual result semantics; the native admission names the v5 runtime.
    assert runtime.contribution_kind == "loop-induced"
    assert runtime.physics.contribution_kind == "loop-induced"
    assert info["native_admission"]["runtime_capability"] == V5
    assert info["interference"] == "conj(loop)*loop" and info["born_orders"] == []
    value = runtime.evaluate([point], mu_r2=[str(scale)])
    assert value.contribution == "loop-induced"
    assert value.born_orders == () and value.born_values == ((),)
    ((finite, single, double),) = value.values[0]
    madloop = fixture["rows"][0]["providers"]["madloop"]["value"]
    assert _relative(finite, madloop) <= LAX
    native = li.native_value("li_gg_h", Precision.double())
    assert _relative(finite, _total(native)) <= ORACLE
    assert abs(single) <= 1e-9 * abs(finite) and abs(double) <= 1e-9 * abs(finite)
    for poles in ("uv", "ir"):
        with pytest.raises(CompatibilityError, match="not served"):
            runtime.evaluate([point], mu_r2=[str(scale)], poles=poles)
    zero = Runtime.load(output, process="h_ga", model_parameters=parameters)
    (sectors,) = zero.evaluate([H_TO_G_A_POINT], mu_r2=["15625"]).values
    assert all(x == 0 for laurent in sectors for x in laurent)
    # Relocation (PLAN Done 5): a copy at another path evaluates bitwise
    # equal in a fresh interpreter (``tools/loop_artifact_portability.py``).
    tool = ROOT / "tools" / "loop_artifact_portability.py"
    spec = importlib.util.spec_from_file_location("loop_artifact_portability", tool)
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "points": [[[float(x).hex() for x in p] for p in point]],
                "mu_r2": [float(scale).hex()],
            }
        )
    )
    verdict = portability.check(
        output, "gg_h", tmp_path / "check", points_file=points, parameters=parameters
    )
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]
    assert verdict["no_generation"]


@pytest.mark.loop_physics
def test_cli_evaluates_a_loop_induced_artifact(li, li_artifact, tmp_path):
    """``pyamplicol evaluate`` takes a loop-induced artifact (only the v5
    runtime) as a virtual one: with --mu-r2 and --poles it returns the
    Python API's VirtualEvaluation, its kind named in the JSON."""

    import io

    from pyamplicol import Runtime
    from pyamplicol.cli import run_cli

    point, scale, parameters = li.point("li_gg_h")
    (tmp_path / "momenta.json").write_text(json.dumps([point]))
    (tmp_path / "parameters.json").write_text(json.dumps(parameters))
    expected = Runtime.load(
        li_artifact, process="gg_h", model_parameters=parameters
    ).evaluate([point], mu_r2=[str(scale)], poles="none")
    for poles in ("none", "all"):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = run_cli(
            (
                "evaluate",
                str(li_artifact),
                "--process",
                "gg_h",
                "--momenta",
                str(tmp_path / "momenta.json"),
                "--model-parameters",
                str(tmp_path / "parameters.json"),
                "--mu-r2",
                str(scale),
                "--poles",
                poles,
                "--json",
                "--progress",
                "off",
            ),
            stdout=stdout,
            stderr=stderr,
        )
        assert code == 0, stderr.getvalue()
        payload = json.loads(stdout.getvalue())
        assert payload["contribution_kind"] == payload["contribution"]
        assert payload["contribution"] == "loop-induced"
        assert payload["poles"] == poles and payload["born_values"] == [[]]
        ((finite, *rest),) = payload["values"][0]
        assert finite == expected.values[0][0][0]
        if poles == "none":
            assert rest == [None, None]
        else:
            # Computed poles of |A|^2 vanish (the finiteness cross-check).
            assert all(abs(x) <= 1e-9 * abs(finite) for x in rest)


@pytest.mark.loop_physics
def test_a_mixed_artifact_serves_both_contributions(li, loop_native, tmp_path):
    """One run with a Born-interfered request (u u~ > z) and a loop-induced
    one (g g > h, process.loop_induced = auto) writes one artifact requiring
    both v2 loop runtimes, each process its own (v4, v5): it loads and
    evaluates both through the Python API and the native admission; the
    native SDK runtime serves both (bitwise the Python API's; the
    loop-induced one with an empty Born, m-lit LIT-SDK). u u~ > z equals the same
    process generated alone bitwise, g g > h the slice's native value."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualEvaluation
    from pyamplicol.loops.v2.assemble import Precision

    fixture = _fixture("li_gg_h")
    mixed = _generate(
        tmp_path / "mixed", (("uux_z", "u u~ > z"), ("gg_h", "g g > h")), fixture
    )
    alone = _generate(tmp_path / "alone", (("uux_z", "u u~ > z"),), fixture)
    manifest = json.loads((mixed / "artifact.json").read_text())
    assert manifest["runtime"]["required_runtime_capabilities"] == [V4, V5]
    assert {
        p["id"]: p["required_runtime_capabilities"] for p in manifest["processes"]
    } == {"uux_z": [V4], "gg_h": [V5]}
    gg_point, scale, parameters = li.point("li_gg_h")
    # u u~ > z at rest: the Z on shell (MZ of the fixture's scheme).
    mz = parameters["MZ"]
    half = str(Decimal(mz) / 2)
    z_point = [
        [half, "0", "0", half],
        [half, "0", "0", "-" + half],
        [mz, "0", "0", "0"],
    ]
    for key, point, kind, capability in (
        ("uux_z", z_point, "virtual", V4),
        ("gg_h", gg_point, "loop-induced", V5),
    ):
        admitted = json.loads(loop_native.admit_v2_artifact(str(mixed), key))
        assert admitted["runtime_capability"] == capability
        assert admitted["metadata"]["contribution_kind"] == kind
        runtime = Runtime.load(mixed, process=key, model_parameters=parameters)
        assert runtime.contribution_kind == runtime.physics.contribution_kind == kind
        value = runtime.evaluate([point], mu_r2=[str(scale)])
        assert isinstance(value, VirtualEvaluation) and value.contribution == kind
        flat = [str(c) for momentum in point for c in momentum]
        if kind == "virtual":
            assert value.born_values[0][0] != 0
            (born, virtual), failures = loop_native.evaluate_v2_artifact(
                str(mixed), [flat], [str(scale)], process=key, parameters=parameters
            )[0]
            assert not failures
            assert born == list(value.born_values[0])
            assert virtual == [list(laurent) for laurent in value.values[0]]
            single = Runtime.load(alone, process=key, model_parameters=parameters)
            found = single.evaluate([point], mu_r2=[str(scale)])
            assert found.born_values == value.born_values
            assert found.values == value.values
        else:
            assert value.born_orders == () and value.born_values == ((),)
            native = li.native_value("li_gg_h", Precision.double())
            assert _relative(value.values[0][0][0], _total(native)) <= ORACLE
            # The native SDK runtime serves the loop-induced process of a
            # mixed artifact too (m-lit LIT-SDK): empty Born, |A|^2 bitwise.
            (born, virtual), failures = loop_native.evaluate_v2_artifact(
                str(mixed), [flat], [str(scale)], process=key, parameters=parameters
            )[0]
            assert not failures and born == []
            assert virtual == [list(laurent) for laurent in value.values[0]]
