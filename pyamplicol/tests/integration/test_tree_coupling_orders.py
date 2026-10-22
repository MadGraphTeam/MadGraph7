# SPDX-License-Identifier: 0BSD
"""Squared coupling-order partials of tree results against MadGraph split orders.

The committed references were produced by ``tools/developer/madgraph_split_orders.py``
from MadGraph 3.7.0 standalone output of the UFO ``sm`` model (default
restriction, t-channel widths kept) at the momenta they record: full colour,
and leading colour (``--colour lc``, the diagonal of MadGraph's colour matrix)
for processes without external gluons. The recurrence lane is checked per
squared order; the eager lane (the Python tree compiler, also the loop
engine's Born) has no partials and is checked on totals.

Electroweak triple-gauge vertices (u u~ > d d~ e+ e-, W pairs decaying to
leptons, e+ e- > u u~ d d~) are checked per order on the recurrence lane. A
kernel whose inputs exchange with a proven factor (WWA, WWZ) once kept both
input orientations as template rows, and the recurrence applied the vertex
twice to every unordered parent pair (``_canonical_transition_alias_key``).

u u~ > d d~ s s~ (three quark lines, W exchange) has a triple-gauge vertex
whose three legs carry three different quark lines, and u d > u d s s~ and
u d~ > u d~ s s~ pair the incoming quarks by a W. MadGraph 3.7 leaves
colour-flow amplitudes of these processes unassigned; the tool zeroes them
(``_zero_colour_amplitudes``). Their references are backed by second methods:
separately generated restricted exports (``QCD=0``, ``QCD=1``), whose
amplitudes are all assigned, reproduce the matching column of the all-order
export, and the eager lane, which shares nothing with the recurrence lane's
colour construction, reproduces totals and restricted orders, also at leading
colour where MadGraph has no reference (an external gluon).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime, TreeOrderEvaluation
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    write_process_groups,
)
from pyamplicol.cli.main import main as cli_main
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember
from pyamplicol.models.prepared import load_prepared_model_bundle

REFERENCE = (
    Path(__file__).resolve().parents[1] / "fixtures/tree/madgraph-split-orders.json"
)
LC_REFERENCE = REFERENCE.with_name("madgraph-split-orders-lc.json")
ORDER_LIMITS = {"QCD": 99, "QED": 99}


def _reference(process: str, path: Path = REFERENCE) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    (entry,) = (item for item in payload["processes"] if item["process"] == process)
    return entry


@pytest.fixture(scope="module")
def sm_bundle(madgraph_source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("madgraph-sm")
    bundle = root / "sm.pyamplicol-model"
    ModelSource.from_path(madgraph_source / "models/sm", restriction="default").compile(
        cache_dir=root / "cache",
        prepared_output=bundle,
        evaluator=EvaluatorConfig(),
    )
    return bundle


@pytest.fixture(scope="module")
def sm_model(sm_bundle: Path) -> ModelSource:
    return ModelSource.from_path(sm_bundle)


def _runtime(
    model: ModelSource,
    process: str,
    artifact: Path,
    *,
    accuracy: str = "full",
    execution_mode: str = "recurrence",
    lc_flow_layout: str = "topology-replay",
    limits: dict[str, int] = ORDER_LIMITS,
) -> Runtime:
    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy=accuracy, lc_flow_layout=lc_flow_layout),
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders=limits
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, samples=1, seed=7, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )
    Generator(config).generate(process, artifact, model=model)
    return Runtime.load(artifact)


def _assert_total_matches_madgraph(totals, reference) -> None:
    for point, (total, partials) in enumerate(
        zip(totals, reference["partials"], strict=True)
    ):
        scale = math.fsum(abs(value) for value in partials)
        total = complex(total)
        assert total.imag == 0.0
        assert abs(total.real - math.fsum(partials)) <= 1e-12 * scale, point


def _assert_matches_madgraph(result: TreeOrderEvaluation, reference) -> None:
    """Every squared order and the total agree with MadGraph, relative to the
    summed magnitude of all partials."""

    expected_orders = tuple(
        tuple(sorted((name, power) for name, power in order.items() if power))
        for order in reference["squared_orders"]
    )
    assert result.orders == tuple(sorted(expected_orders))
    for point, partials in enumerate(reference["partials"]):
        scale = math.fsum(abs(value) for value in partials)
        total = complex(result.total[point])
        assert abs(total.real - math.fsum(partials)) <= 1e-12 * scale, point
        for order, expected in zip(expected_orders, partials, strict=True):
            actual = complex(result.for_order(order)[point])
            assert actual.imag == 0.0
            assert abs(actual.real - expected) <= 1e-12 * scale, (point, order)


# Diagonal orders, QCD x EW interference between different fermion-line
# pairings (s/t/u channels, identical quarks, three lines), electroweak
# triple-gauge vertices and single-order processes, whose only partial is the
# total.
_THREE_LINE_TRIPLE_GAUGE = "u u~ > d d~ s s~"
# Three lines where a W pairs the u and d lines in the initial state.
_W_PAIRED_THREE_LINES = "u d > u d s s~"
_TRIPLE_GAUGE_PROCESSES = (
    "u u~ > d d~ e+ e-",
    "u d > u d e+ e-",
    "e+ e- > u u~ d d~",
    # W+ W- through leptons, with and without quarks.
    "u u~ > e+ ve mu- vm~",
    _THREE_LINE_TRIPLE_GAUGE,
    _W_PAIRED_THREE_LINES,
    "u d~ > u d~ s s~",
)
_MADGRAPH_PROCESSES = (
    "u u~ > d d~",
    "u u~ > u u~",
    "u d > u d",
    "u u > u u",
    "u u~ > d d~ g",
    "u u~ > u u~ g",
    "u u~ > u u~ u u~",
    "g g > t t~",
    "u u~ > e- e+ g",
    *_TRIPLE_GAUGE_PROCESSES,
    "u u~ > d d~ e+ e- g",
    "e+ e- > mu+ vm e- ve~",
)


@pytest.mark.parametrize("process", _MADGRAPH_PROCESSES)
def test_partials_match_madgraph_and_keep_the_total(
    sm_model: ModelSource, tmp_path: Path, process: str
) -> None:
    reference = _reference(process)
    runtime = _runtime(sm_model, process, tmp_path / "artifact")
    points = reference["points"]

    result = runtime.evaluate_orders(points)

    assert result.total == runtime.evaluate(points)
    _assert_matches_madgraph(result, reference)
    for total, summed in zip(result.total, result.sum_orders(), strict=True):
        assert math.isclose(complex(summed).real, complex(total).real, rel_tol=1e-13)
    assert all(
        pair[0] in result.amplitude_orders and pair[1] in result.amplitude_orders
        for sector in result.sectors
        for pair in sector.pairs
    )


# Quark-only processes, whose MadGraph colour basis is the colour-flow basis.
_LEADING_COLOUR_PROCESSES = (
    "u u~ > d d~",
    "u u~ > u u~",
    "u d > u d",
    "u u > u u",
    "u u~ > u u~ u u~",
    *_TRIPLE_GAUGE_PROCESSES,
)


@pytest.mark.parametrize("process", _LEADING_COLOUR_PROCESSES)
def test_leading_colour_partials_match_madgraph(
    sm_model: ModelSource, tmp_path: Path, process: str
) -> None:
    runtime = _runtime(sm_model, process, tmp_path / "artifact", accuracy="lc")
    reference = _reference(process, LC_REFERENCE)

    _assert_matches_madgraph(runtime.evaluate_orders(reference["points"]), reference)


def test_recurrence_template_keeps_one_orientation_of_exchange_symmetric_vertices(
    sm_bundle: Path,
) -> None:
    """WWA, WWZ, HWW and Htt keep one row per unordered pair of input states
    and spins: both mirrored rows made the recurrence apply the vertex twice to
    every unordered parent pair. (Contact-orbit steps own their mirrors.)"""

    template = load_prepared_model_bundle(
        sm_bundle
    ).kernel_pack.recurrence_template_catalog
    assert template is not None
    particles = {
        state.template_id: state.particle_id for state in template.current_states
    }
    flows = {flow.template_id: flow for flow in template.quantum_flows}
    rows: Counter[tuple[object, ...]] = Counter()
    for transition in template.transitions:
        if (
            transition.input_exchange_factor is None
            or transition.contact_orbit_step_template_ids
        ):
            continue
        flow = flows[transition.quantum_flow_template_id]
        inputs = tuple(
            sorted(
                zip(
                    transition.input_state_template_ids,
                    flow.input_spin_states,
                    strict=True,
                )
            )
        )
        if inputs[0][0] != inputs[1][0]:
            rows[
                (inputs, transition.result_state_template_id, flow.result_spin_state)
            ] += 1
    assert set(rows.values()) == {1}
    assert {
        (
            tuple(sorted(particles[state] for state, _ in inputs)),
            particles[result],
        )
        for inputs, result, _ in rows
    } >= {((-24, 24), 22), ((-24, 24), 23), ((-24, 24), 25), ((-24, 22), -24)}


@pytest.mark.parametrize("colour", ["full", "lc"])
def test_restricted_madgraph_exports_reproduce_their_all_order_column(
    colour: str,
) -> None:
    """MadGraph 3.7 leaves colour-flow amplitudes of multi-line exports
    unassigned (zeroed by the tool); a restricted export, whose amplitudes are
    all assigned, must reproduce its column of the all-order export."""

    path = REFERENCE if colour == "full" else LC_REFERENCE
    entries = json.loads(path.read_text(encoding="utf-8"))["processes"]
    processes = {entry["process"] for entry in entries}
    restricted = [
        entry
        for entry in entries
        if entry["process"].split(" QCD=")[0] in processes - {entry["process"]}
    ]
    assert len(restricted) == (5 if colour == "full" else 4)
    for entry in restricted:
        reference = _reference(entry["process"].split(" QCD=")[0], path)
        assert entry["points"] == reference["points"]
        (order,) = entry["squared_orders"]
        column = reference["squared_orders"].index(order)
        for total, partials in zip(entry["total"], reference["partials"], strict=True):
            assert math.isclose(total, partials[column], rel_tol=1e-13), entry[
                "process"
            ]


def test_leading_colour_lanes_agree_per_order_with_a_gluon_and_a_lepton_pair(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    """MadGraph's leading colour needs a basis without external gluons; the
    eager lane is the second method for u u~ > d d~ e+ e- g: its total, and at
    QCD=1 the pure-electroweak amplitude with one gluon emitted, QCD^2 QED^8,
    which contains every triple-gauge vertex."""

    process = "u u~ > d d~ e+ e- g"
    points = _reference(process)["points"]
    result = _runtime(
        sm_model, process, tmp_path / "recurrence", accuracy="lc"
    ).evaluate_orders(points)
    eager = _runtime(
        sm_model, process, tmp_path / "eager", accuracy="lc", execution_mode="eager"
    ).evaluate(points)
    electroweak = _runtime(
        sm_model,
        process,
        tmp_path / "electroweak",
        accuracy="lc",
        execution_mode="eager",
        limits={"QCD": 1, "QED": 4},
    ).evaluate(points)
    assert result.orders == (
        (("QCD", 2), ("QED", 8)),
        (("QCD", 4), ("QED", 6)),
        (("QCD", 6), ("QED", 4)),
    )
    for point, (values, total, expected) in enumerate(
        zip(result.values, eager, electroweak, strict=True)
    ):
        scale = math.fsum(abs(complex(value).real) for value in values)
        assert abs(complex(result.total[point]).real - complex(total).real) <= (
            1e-12 * scale
        )
        assert abs(complex(values[0]).real - complex(expected).real) <= 1e-12 * abs(
            complex(expected).real
        )


# QCD x electroweak interference between quark lines (s/t channels, identical
# quarks, a gluon, a lepton pair with triple-gauge vertices, three lines that
# exchange electroweak bosons in turn): the eager lane keeps the electroweak
# roots of every line pairing and signs coloured fermions by their Wick order.
@pytest.mark.parametrize(
    ("process", "accuracy"),
    [
        ("u u~ > d d~", "full"),
        ("u u~ > d d~", "lc"),
        ("u u~ > u u~", "full"),
        ("u u~ > u u~", "lc"),
        ("u d > u d", "full"),
        ("u u~ > d d~ g", "full"),
        ("u u~ > d d~ e+ e-", "full"),
        ("u u~ > d d~ e+ e-", "lc"),
        ("u u~ > u u~ u u~", "full"),
        ("u u~ > u u~ u u~", "lc"),
        (_THREE_LINE_TRIPLE_GAUGE, "full"),
        (_THREE_LINE_TRIPLE_GAUGE, "lc"),
        (_W_PAIRED_THREE_LINES, "full"),
        (_W_PAIRED_THREE_LINES, "lc"),
        ("u d~ > u d~ s s~", "full"),
        ("u d~ > u d~ s s~", "lc"),
        ("u d > u d e+ e-", "full"),
        ("u d > u d e+ e-", "lc"),
        ("e+ e- > u u~ d d~", "full"),
        ("e+ e- > u u~ d d~", "lc"),
        ("u u~ > e+ ve mu- vm~", "full"),
        ("u u~ > d d~ e+ e- g", "full"),
    ],
)
def test_eager_lane_matches_madgraph(
    sm_model: ModelSource, tmp_path: Path, process: str, accuracy: str
) -> None:
    reference = _reference(process, REFERENCE if accuracy == "full" else LC_REFERENCE)
    runtime = _runtime(
        sm_model,
        process,
        tmp_path / "artifact",
        accuracy=accuracy,
        execution_mode="eager",
    )

    _assert_total_matches_madgraph(runtime.evaluate(reference["points"]), reference)


_FOUR_LINES = "u u~ > d d~ s s~ c c~"


def _reordered(points: list[object], process: str, reference: str) -> list[object]:
    """``points`` of ``reference`` with their momenta in the leg order of
    ``process`` (the same particles, each listed once)."""

    names = reference.replace(">", "").split()
    order = [names.index(name) for name in process.replace(">", "").split()]
    return [[point[index] for index in order] for point in points]


# The second order relabels every line, so that other traversal sectors and
# other orientations of each colour-singlet vertex build the same diagrams.
@pytest.mark.parametrize("process", [_FOUR_LINES, "u~ u > c c~ s s~ d d~"])
@pytest.mark.parametrize("accuracy", ["full", "lc"])
def test_eager_lane_pairs_w_exchanges_among_four_quark_lines(
    sm_model: ModelSource, tmp_path: Path, process: str, accuracy: str
) -> None:
    """Pure-electroweak four quark lines, where four-point WWA/WWZ/WWAA/WWZZ
    contact vertices join all four lines. The full-colour traversal sectors
    close on different sinks and built some of these diagrams twice and
    their mirror images not at all (1.48x MadGraph). The W pairings of the
    two generations give four interfering colour flows. MadGraph's QCD=0
    export assigns all its colour-flow amplitudes."""

    reference = _reference(
        f"{_FOUR_LINES} QCD=0", REFERENCE if accuracy == "full" else LC_REFERENCE
    )
    runtime = _runtime(
        sm_model,
        process,
        tmp_path / "artifact",
        accuracy=accuracy,
        execution_mode="eager",
        limits={"QCD": 0, "QED": 6},
    )
    points = _reordered(reference["points"], process, _FOUR_LINES)

    _assert_total_matches_madgraph(runtime.evaluate(points), reference)


@pytest.mark.parametrize("accuracy", ["full", "lc"])
@pytest.mark.parametrize("process", [_THREE_LINE_TRIPLE_GAUGE, _W_PAIRED_THREE_LINES])
def test_eager_lane_counts_a_three_line_triple_gauge_vertex_once(
    sm_model: ModelSource, tmp_path: Path, process: str, accuracy: str
) -> None:
    """QED^8 of three quark lines is the pure-electroweak amplitude squared,
    where WWA/WWZ vertices join the three lines (12x MadGraph for
    u u~ > d d~ s s~ when both cyclic orders of the lines kept them). At
    leading colour the flows without a pure-electroweak root vanish.

    MadGraph's all-order export leaves the colour-flow amplitudes no diagram
    feeds unassigned (the tool zeroes them). A second export generated at
    QCD=0 builds the pure-electroweak amplitudes on their own, every one
    assigned, and must reproduce that export's QED^8."""

    path = REFERENCE if accuracy == "full" else LC_REFERENCE
    reference = _reference(process, path)
    alone = _reference(f"{process} QCD=0", path)
    assert alone["points"] == reference["points"]
    assert alone["squared_orders"] == [{"QCD": 0, "QED": 8}]
    (column,) = (
        index
        for index, order in enumerate(reference["squared_orders"])
        if order == {"QCD": 0, "QED": 8}
    )
    for total, partials in zip(alone["total"], reference["partials"], strict=True):
        assert math.isclose(total, partials[column], rel_tol=1e-13)
    runtime = _runtime(
        sm_model,
        process,
        tmp_path / "artifact",
        accuracy=accuracy,
        execution_mode="eager",
        limits={"QCD": 0, "QED": 4},
    )
    for point, (value, partials) in enumerate(
        zip(runtime.evaluate(reference["points"]), reference["partials"], strict=True)
    ):
        value = complex(value)
        assert value.imag == 0.0
        assert math.isclose(value.real, partials[column], rel_tol=1e-12), point


def test_leading_colour_lanes_agree_on_restricted_three_line_orders(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    """At QCD=2 QED=2, a photon or Z emitted by a quark line after a gluon
    exchange with another line carries both lines' colour; the eager lane
    keeps it (it dropped those roots and then failed to load) and agrees with
    the recurrence lane, which has no triple-gauge vertex at this order."""

    limits = {"QCD": 2, "QED": 2}
    points = _reference(_THREE_LINE_TRIPLE_GAUGE, LC_REFERENCE)["points"]
    recurrence = _runtime(
        sm_model,
        _THREE_LINE_TRIPLE_GAUGE,
        tmp_path / "recurrence",
        accuracy="lc",
        limits=limits,
    )
    eager = _runtime(
        sm_model,
        _THREE_LINE_TRIPLE_GAUGE,
        tmp_path / "eager",
        accuracy="lc",
        execution_mode="eager",
        limits=limits,
    )
    for actual, expected in zip(
        eager.evaluate(points), recurrence.evaluate(points), strict=True
    ):
        assert complex(actual) == pytest.approx(complex(expected), rel=1e-13)


def test_leading_colour_lanes_agree_on_mixed_order_totals(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    """Topology replay, all-flow union, on-the-fly construction and the eager
    lane sign identical quarks joined by gluons and electroweak bosons alike."""

    process = "u u~ > u u~ g"
    replay = _runtime(sm_model, process, tmp_path / "replay", accuracy="lc")
    points = replay._backend.validation_momenta()
    expected = replay.evaluate(points)
    for name, execution_mode, layout in (
        ("union", "recurrence", "all-flow-union"),
        ("on-the-fly", "on-the-fly", "topology-replay"),
        ("eager", "eager", "topology-replay"),
    ):
        runtime = _runtime(
            sm_model,
            process,
            tmp_path / name,
            accuracy="lc",
            execution_mode=execution_mode,
            lc_flow_layout=layout,
        )
        for actual, reference in zip(runtime.evaluate(points), expected, strict=True):
            assert complex(actual) == pytest.approx(complex(reference), rel=1e-13)
    _assert_matches_exact(replay, points)


def test_process_group_members_scale_partials_like_totals(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    representative = _runtime(sm_model, "u d > u d", artifact)
    process_id = representative.physics.process_id
    factor = Fraction(3, 2)
    member = GroupMember(
        process_id="u_d_scaled_member",
        expression="u~ d~ > u~ d~",
        pdgs=(-2, -1, -2, -1),
        initial_count=2,
        factor=factor,
        permutation=(0, 1, 2, 3),
        conjugated=True,
        basis=GroupingBasis.NUMERICAL,
        generated=False,
    )
    write_process_groups(
        artifact,
        initial_process_groups(
            [(process_id, "u d > u d", (2, 1, 2, 1))], {process_id: [member]}
        ),
    )
    scaled = Runtime.load(artifact, process="u~ d~ > u~ d~")
    assert scaled.process_group_factor == factor and scaled.charge_conjugated
    points = _reference("u d > u d")["points"]

    for precision in (16, 32):
        expected = representative.evaluate_orders(points, precision=precision)
        result = scaled.evaluate_orders(points, precision=precision)

        assert result.orders == expected.orders
        assert result.total == scaled.evaluate(points, precision=precision)
        for values, reference_values in zip(
            result.values, expected.values, strict=True
        ):
            for value, reference in zip(values, reference_values, strict=True):
                if precision == 16:
                    assert complex(value) == pytest.approx(
                        1.5 * complex(reference), rel=1e-15
                    )
                    continue
                with localcontext() as context:
                    context.prec = 2 * precision
                    exact = reference * factor.numerator / factor.denominator
                assert abs(value - exact) <= Decimal("1e-30") * abs(exact)


def test_exact_partials_agree_and_keep_the_exact_total(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    reference = _reference("u u~ > d d~ g")
    runtime = _runtime(sm_model, "u u~ > d d~ g", tmp_path / "artifact")
    points = reference["points"][:1]

    exact = runtime.evaluate_orders(points, precision=32)
    native = runtime.evaluate_orders(points)

    assert exact.total == runtime.evaluate(points, precision=32)
    assert exact.orders == native.orders
    assert all(isinstance(value, Decimal) for value in exact.values[0])
    assert exact.sum_orders()[0] == pytest.approx(exact.total[0], rel=Decimal("1e-30"))
    scale = math.fsum(abs(float(value)) for value in exact.values[0])
    for exact_value, native_value in zip(
        exact.values[0], native.values[0], strict=True
    ):
        assert abs(float(exact_value) - complex(native_value).real) <= 1e-13 * scale


def _assert_matches_exact(runtime: Runtime, points: list[object]) -> None:
    """Partials keep the total, sum to it and agree with the exact executor."""

    result = runtime.evaluate_orders(points)
    exact = runtime.evaluate_orders(points[:1], precision=32)

    assert result.total == runtime.evaluate(points)
    assert exact.total == runtime.evaluate(points[:1], precision=32)
    assert len(result.amplitude_orders) > 1
    for total, summed in zip(result.total, result.sum_orders(), strict=True):
        assert math.isclose(complex(summed).real, complex(total).real, rel_tol=1e-13)
    scale = math.fsum(abs(float(value)) for value in exact.values[0])
    for exact_value, native_value in zip(
        exact.values[0], result.values[0], strict=True
    ):
        assert abs(float(exact_value) - complex(native_value).real) <= 1e-13 * scale


@pytest.mark.parametrize(
    ("accuracy", "layout"),
    [("lc", "topology-replay"), ("lc", "all-flow-union"), ("nlc", "topology-replay")],
)
def test_truncated_colour_partials_agree_with_exact(
    sm_model: ModelSource, tmp_path: Path, accuracy: str, layout: str
) -> None:
    runtime = _runtime(
        sm_model,
        "u u~ > d d~ g",
        tmp_path / "artifact",
        accuracy=accuracy,
        lc_flow_layout=layout,
    )
    _assert_matches_exact(runtime, _reference("u u~ > d d~ g")["points"])


def test_symmetric_group_colour_partials_agree_with_exact(
    sm_model: ModelSource, tmp_path: Path
) -> None:
    # Full colour with two gluons selects the symmetric-group Fourier reducer.
    runtime = _runtime(sm_model, "u u~ > d d~ g g", tmp_path / "artifact")
    manifest = json.loads((tmp_path / "artifact/artifact.json").read_text())
    assert (
        "rusticol.color-contraction.symmetric-group-fft.v1"
        in (manifest["processes"][0]["required_runtime_capabilities"])
    )
    _assert_matches_exact(runtime, [runtime._backend.validation_momenta()[0]])


def test_cli_prints_partials_and_rejects_non_recurrence_lanes(
    sm_model: ModelSource, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _reference("u u~ > d d~")
    _runtime(sm_model, "u u~ > d d~", tmp_path / "artifact")
    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps(reference["points"]), encoding="utf-8")

    assert (
        cli_main(
            [
                "evaluate",
                str(tmp_path / "artifact"),
                "--momenta",
                str(momenta),
                "--orders",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["contribution_kind"] == "tree"
    assert payload["shape"] == [len(reference["points"]), 3]
    assert len(payload["values"]) == len(reference["points"])

    eager = _runtime(
        sm_model, "u u~ > d d~", tmp_path / "eager", execution_mode="eager"
    )
    with pytest.raises(CompatibilityError, match="recurrence artifact"):
        eager.evaluate_orders(reference["points"])
