# SPDX-License-Identifier: 0BSD
"""Schedule M on generated slices, natively (design §2.2, §3.5, WP2.8).

``merged.merged_graph`` splits every certified skeleton into its
colour-ordered paths (one per choice of vertex decomposition terms), seeds
each at the cut by the primitive class of its cut word and shares states
across compositions and families; ``plan_export`` writes the graph into plan
v4 and the native executor runs it (``rusticol_core::loop_v2::merged``).
Checks:

* the planar paths of every certified skeleton sum to its B seed exactly and
  the state graph has one cut-to-closure path per planar path; families with
  generic seeds (several quark lines) stay on schedule B;
* O4 in Rust: the native exact carrier gives M = B bit for bit, with chunk
  keys of depth 0, 1 and 2, every kernel form and both ring readings;
  60-digit arithmetic agrees with B to 1e-70;
* a native evaluation with schedule M equals schedule B at 32 digits (all
  three Laurent coefficients); in binary64 M's virtual lies within 1e-12 of
  the 32-digit value, or within its own reduction-direction estimate on
  ill-conditioned points, and finite-only runs give its finite parts; the
  report has M's memory per lane.

``test_loop_v2_merged_controls`` checks M = B on every control at full,
next-to-leading and leading colour in the exact, binary64 and DoubleFloat
carriers.
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import merged
from pyamplicol.loops.v2 import reference_executor as rx
from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    Laurent,
    LoopSlice,
    Precision,
    SliceRequest,
)
from pyamplicol.loops.v2.plan_export import NativeExecutor, _degree, export_plan
from pyamplicol.loops.v2.stability import StabilityChecks, deviation

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
ARBITRARY_TOLERANCE = "1e-70"


def _points(name: str) -> tuple[list, list[str]]:
    if name.endswith(".json"):
        data = json.loads(
            (ROOT / "tests" / "fixtures" / "loop" / name).read_text(), parse_float=str
        )
        points = data["points"]
        mu = data.get("mu_r", ["1000"] * len(points))
        if isinstance(mu, str):
            mu = [mu] * len(points)
        return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]
    data = json.loads(
        (ROOT / "tools" / "loop_reference" / f"{name}.json").read_text(),
        parse_float=str,
    )
    return data["points"], ["1000000"] * len(data["points"])


# process, Born orders, ignored particles, points
PROCESSES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, (), "gluon_scattering_qcd"),
    "uux_gg": ("u u~ > g g", {"QCD": 2, "QED": 0}, (), "gluon_scattering_qcd"),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        (),
        "required/uux_ddxg.json",
    ),
    "uux_uux": ("u u~ > u u~", {"QCD": 2, "QED": 0}, (), "legacy/uux_uux.json"),
    "uux_uuxg": ("u u~ > u u~ g", {"QCD": 3, "QED": 0}, (), "legacy/uux_uuxg.json"),
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, (), "quark_three_gluon_qcd"),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, (), "quark_three_gluon_qcd"),
    # Photon exchange only (``test_loop_v2_executor.IGNORED``).
    "uux_emepgg": (
        "u u~ > e- e+ g g",
        {"QCD": 2, "QED": 2},
        (23,),
        "dilepton-two-gluon.json",
    ),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    compiled = load_compiled_model(loop_sm_compiled_model)
    # The shipped loop_sm declaration (MQ-NC): the electroweak controls reach
    # closed-quark-loop R2 (g g g A) the example QCD subset does not cover.
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
    built: dict[tuple[str, str], LoopSlice] = {}

    def build(name: str, accuracy: str = "full") -> LoopSlice:
        if (name, accuracy) not in built:
            process, born, ignored, _points = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                fixed_parameters=FIXED,
                particle_veto=ignored,
                accuracy=accuracy,
                colour_declaration=None if accuracy == "full" else declaration,
            )
            built[name, accuracy] = LoopSlice.generate(compiled, request, loop_native)
        return built[name, accuracy]

    return build


def _point(name: str, index: int = 0):
    points, scales = _points(PROCESSES[name][3])
    return points[index], scales[index]


def _conserving(point: list) -> list:
    """The point with its last momentum set to conserve momentum exactly.

    Schedule M closes a ring in its cut frame and maps it into the family's
    trie frame by a loop-momentum shift that assumes momentum conservation;
    decimal fixtures conserve it to their last digit only (the dilepton
    points to 1e-13 GeV), which M and B then read in different frames.
    """

    momenta = [[Fraction(x) for x in momentum] for momentum in point]
    momenta[-1] = [
        momenta[0][mu] + momenta[1][mu] - sum(p[mu] for p in momenta[2:-1])
        for mu in range(4)
    ]
    return [[str(x) for x in momentum] for momentum in momenta]


def _executor(slice_, chunk_depth=1, forms="rule", reading="trie") -> NativeExecutor:
    graph = merged.merged_graph(slice_, chunk_depth=chunk_depth, reading=reading)
    return NativeExecutor(slice_, forward=False, merged=graph, forms=forms)


# ----------------------------------------------------------------- generation


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "everywhere"),
    [("gg_gg", True), ("uux_gg", False), ("uux_ddxg", False), ("gg_ggg", True)],
)
def test_planar_paths_certify_against_the_b_seeds(slices, name, everywhere):
    """Every certified family's planar paths sum to its skeletons' B seeds
    (checked inside ``PlanarExpansion.paths``) and the state graph has one
    closing path per planar path; families with generic seeds fall back to
    B. States are shared: fewer states than path positions."""

    slice_ = slices(name)
    graph = merged.merged_graph(slice_)
    families = set(slice_.plan.schedules)
    assert graph.families | set(graph.fallback) == families
    assert not graph.families & set(graph.fallback)
    assert (not graph.fallback) == everywhere
    assert set(graph.fallback.values()) <= {
        "generic seed (no certified primitive class)",
        "the B seed is not a certified combination",
    }
    expansion = merged.PlanarExpansion(slice_)
    positions = 0
    for skeleton in rx.plan_skeletons(slice_.plan):
        if skeleton.family in graph.families:
            positions += sum(len(p.steps) for p in expansion.paths(skeleton))
    counts = graph.counts()
    assert counts["paths"] >= sum(
        1 for s in rx.plan_skeletons(slice_.plan) if s.family in graph.families
    )
    assert counts["states"] < positions
    # Chunks keyed by the first position lose no merge; deeper keys repeat
    # the shallow states in every chunk below them.
    whole = merged.merged_graph(slice_, chunk_depth=0)
    assert whole.counts()["chunks"] == 1
    assert whole.counts()["states"] == counts["states"]
    assert (
        merged.merged_graph(slice_, chunk_depth=2).counts()["states"]
        >= counts["states"]
    )
    # A family's chunks share its first leg set, and the chunks of a leg set
    # are consecutive: the family streams once that run of chunks ran.
    leg_sets: dict = {}
    run: list = []
    start = 0
    for end in graph.chunks:
        (masks, *_rest), *_ = graph.states[start].key
        if not run or run[-1] != masks:
            assert masks not in run
            run.append(masks)
        for state in graph.states[start:end]:
            if state.closure is not None:
                leg_sets.setdefault(state.closure[0], set()).add(masks)
        start = end
    assert all(len(sets) == 1 for sets in leg_sets.values())


@pytest.mark.loop_physics
def test_plan_carries_schedule_m(slices):
    slice_ = slices("gg_gg")
    graph = merged.merged_graph(slice_)
    program = export_plan(slice_, forward=False, merged=graph).program
    section = program["merged"]
    assert len(section["states"]) == len(graph.states)
    assert section["chunks"][-1] == len(graph.states)
    assert sorted(section["families"]) == section["families"]
    closures = [s["closure"] for s in section["states"] if s["closure"]]
    assert {c["family"] for c in closures} == set(section["families"])
    executor = NativeExecutor(slice_, forward=False, merged=graph)
    assert executor.native("double").merged_families == section["families"]
    assert export_plan(slice_, forward=False).program["merged"] is None


# ----------------------------------------------------------------- numerators


def _assert_exact(left, right, label):
    assert left.keys() == right.keys(), label
    for key, mine in left.items():
        theirs = right[key]
        for x, y in zip(mine.columns, theirs.columns, strict=True):
            width = max(len(x), len(y))
            a = [rx.exact(v) for v in x] + [0] * (width - len(x))
            b = [rx.exact(v) for v in y] + [0] * (width - len(y))
            assert a == b, (label, key)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_gg", "uux_ddxg", "uux_emepgg"])
def test_exact_m_equals_b(slices, name):
    """O4 in Rust: exact rational numerators of M equal B's bit for bit,
    chunked or not, in every kernel form, from the trie readings and from
    the skeletons' own readings (closures through the family frame maps),
    at momentum-conserving points."""

    slice_ = slices(name)
    point, scale = _point(name)
    point = _conserving(point)
    reference = None
    for chunk_depth, forms, reading in (
        (1, "rule", "trie"),
        (0, "fused", "trie"),
        (2, "split", "trie"),
        (1, "rule", "skeleton"),
    ):
        executor = _executor(slice_, chunk_depth, forms, reading)
        options = {"parameters": PARAMETERS, "exact": True}
        if reference is None:
            (reference,) = executor.family_numerators([point], [scale], **options)
        (mine,) = executor.family_numerators([point], [scale], schedule="M", **options)
        _assert_exact(mine, reference, (chunk_depth, forms, reading))


def _magnitude(value) -> Decimal:
    """|value| of a complex number or a carrier pair (Decimal parts)."""

    if isinstance(value, complex):
        return Decimal(abs(value))
    return (value.real * value.real + value.imag * value.imag).sqrt()


def _assert_close(left, right, energy, tolerance, label) -> None:
    """Native numerators ``left`` within ``tolerance`` of ``right`` on each
    family's largest sqrt(s)^r-weighted coefficient.

    Families that vanish identically (Furry's theorem: the two orientations
    of a closed quark loop with one photon and two gluons cancel) have no
    scale of their own. B cancels them leaf by leaf, M only at the closure
    of the two orderings, so their rounding is held to ``tolerance`` of the
    point's largest weighted coefficient instead; a family counts as
    vanishing when B's value is below that bound.
    """

    energy = Decimal(energy)
    tolerance = Decimal(tolerance)
    with localcontext() as context:
        context.prec = 100

        def weighted(numerator):
            return max(
                _magnitude(x) * energy ** _degree(k)
                for column in numerator.columns
                for k, x in enumerate(column)
            )

        assert left.keys() == right.keys(), label
        largest = max(weighted(numerator) for numerator in right.values())
        for key, mine in left.items():
            theirs = right[key]
            scale = weighted(theirs)
            if scale < tolerance * largest:
                scale = largest
            for x, y in zip(mine.columns, theirs.columns, strict=True):
                for k, (a, c) in enumerate(zip(x, y, strict=True)):
                    error = _magnitude(a - c) * energy ** _degree(k)
                    assert error <= tolerance * scale, (label, key, k, error / scale)


@pytest.mark.loop_physics
def test_arbitrary_precision_m_equals_b(slices):
    name = "gg_ggg"
    slice_ = slices(name)
    executor = _executor(slice_)
    point, scale = _point(name)
    precision = Precision.arbitrary(60)
    options = {"parameters": PARAMETERS, "precision": precision}
    (b,) = executor.family_numerators([point], [scale], **options)
    (m,) = executor.family_numerators([point], [scale], schedule="M", **options)
    energy = Decimal(point[0][0]) + Decimal(point[1][0])
    _assert_close(m, b, energy, ARBITRARY_TOLERANCE, name)


# ----------------------------------------------------------------- evaluation


def _assert_laurent_close(left, right, tolerance, components=COMPONENTS):
    """Every sector's Laurent coefficients within ``tolerance`` of the
    sector's largest coefficient in ``right``."""

    assert left.born == right.born
    for component in components:
        assert left.components[component].keys() == right.components[component].keys()
        for sector, value in right.components[component].items():
            mine = left.components[component][sector]
            assert len(mine.coefficients) == len(value.coefficients)
            scale = max(abs(x) for x in value.coefficients) or 1
            for x, y in zip(mine.coefficients, value.coefficients, strict=True):
                assert abs(x - y) <= tolerance * scale, (component, sector)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_ggg", "uux_ggg"])
def test_native_evaluation_with_m_equals_b(slices, name):
    """The complete native evaluation with schedule M equals schedule B,
    streaming M's completed families into the reduction.

    At 32 digits M = B to 1e-25 (all components, all three Laurent
    coefficients). In binary64 the two schedules sum the same numerators in
    different orders and the reduction amplifies that rounding on
    ill-conditioned points (gg -> ggg's first point loses five digits in
    either schedule), so M's binary64 virtual is held to 1e-12 of the
    32-digit value, or to its own reduction-direction estimate where that is
    larger; the counterterm components equal B's to 1e-12, and finite-only
    runs give M's finite parts bit for bit. The report has M's memory per
    lane."""

    slice_ = slices(name)
    executor = _executor(slice_)
    points, scales = _points(PROCESSES[name][3])
    points, scales = points[:2], scales[:2]
    options = {"parameters": PARAMETERS}
    exact = {
        schedule: executor.evaluate(
            points,
            scales,
            precision=Precision.arbitrary(32),
            schedule=schedule,
            **options,
        )
        for schedule in "BM"
    }
    for left, right in zip(exact["M"], exact["B"], strict=True):
        _assert_laurent_close(left, right, Decimal("1e-25"))
    b = executor.evaluate(points, scales, **options)
    m = executor.evaluate(
        points,
        scales,
        schedule="M",
        stability=StabilityChecks(reduction_direction=True),
        **options,
    )
    report = executor.last_report
    counterterms = tuple(c for c in COMPONENTS if c not in ("virtual", "bare"))
    for mine, theirs, truth in zip(m, b, exact["B"], strict=True):
        _assert_laurent_close(mine, theirs, 1e-12, counterterms)
        reference = {
            sector: Laurent(*[float(x) for x in value.coefficients])
            for sector, value in truth.components["virtual"].items()
        }
        error = deviation(reference, mine.components["virtual"])
        assert error <= max(1e-12, mine.stability.relative), (error, mine.stability)
    finite = executor.evaluate(points, scales, schedule="M", poles="none", **options)
    for value, full in zip(finite, m, strict=True):
        for component in COMPONENTS:
            for sector, laurent in value.components[component].items():
                assert laurent.single_pole is None and laurent.double_pole is None
                assert laurent.finite == full.components[component][sector].finite
    reals = report["reals_per_lane"]
    assert reals["merged family accumulators"] > 0
    assert reals["open objects (one merged chunk)"] > 0
    assert report["counts"]["merged states"] > 0
