# SPDX-License-Identifier: 0BSD
"""Cold directed instructions retain source ownership and complex column axes."""

import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from .test_loop_portable import _rewrite
from .test_loop_source_correlations import source_directed as source_directed


@pytest.fixture(scope="module")
def directed_artifact(source_directed, tmp_path_factory):
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.loops.portable import write_loop_pacbin

    source, prepared, _ = source_directed
    root = tmp_path_factory.mktemp("portable-directed")
    ordinary, directed = root / "ordinary.pacbin", root / "directed.pacbin"
    write_loop_pacbin(source.virtual, ordinary)
    declarations = tuple(
        ColorCorrelator(name, bra.emissions, ket.emissions)
        for name, (bra, ket) in zip(
            ("identity", "casimir", "ordered", "swapped"),
            prepared.bare.histories,
            strict=True,
        )
    )
    write_loop_pacbin(
        source.virtual, directed, directed=prepared, directed_histories=declarations
    )
    return source, prepared, ordinary, directed


def test_directed_descriptor_reuses_source_schedules_and_exact_states(
    directed_artifact,
):
    native = pytest.importorskip("_loop_reduction")
    _, prepared, ordinary, path = directed_artifact
    baseline = json.loads(native.PortableLoopPlan(str(ordinary)).metadata_json())
    loaded = native.PortableLoopPlan(str(path))
    data = json.loads(loaded.metadata_json())
    projection = data.pop("directed_projection")
    baseline.pop("directed_projection")
    assert data == baseline
    assert projection["kind"] == "directed-complex"
    assert projection["color_legs"] == [
        [leg.label, leg.representation] for leg in prepared.bare.color_legs
    ]
    assert [history["id"] for history in projection["histories"]] == [
        "identity",
        "casimir",
        "ordered",
        "swapped",
    ]
    assert projection["histories"][2]["bra"] == projection["histories"][3]["ket"]
    assert projection["histories"][2]["ket"] == projection["histories"][3]["bra"]
    assert loaded.evaluator_state(projection["born"]) == prepared.bare.born_state
    assert (
        loaded.evaluator_state(projection["postprocess"])
        == prepared.bare.postprocess_state
    )
    assert (
        loaded.evaluator_state(projection["counterterms"]) == prepared.counterterm_state
    )
    assert loaded.evaluator_state(projection["assemble"]) == prepared.assembly_state
    assert (
        loaded.evaluator_state(projection["sum_order_values"])
        == prepared.sum_order_state
    )
    for name, family in zip(
        projection["families"], prepared.bare.families, strict=True
    ):
        assert loaded.evaluator_state(name) == family.state


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 70)],
)
def test_cold_directed_assembly_keeps_complex_terms_without_ordinary_execution(
    directed_artifact, arithmetic, precision
):
    native = pytest.importorskip("_loop_reduction")
    from pyamplicol.loops.portable import PortableLoopPlan

    source, prepared, _, path = directed_artifact
    loaded = native.PortableLoopPlan(str(path))
    projection = json.loads(loaded.metadata_json())["directed_projection"]
    width = (
        len(prepared.bare.histories) * len(source.virtual.bare.order_layout.virtual) * 3
    )
    options = dict(arithmetic=arithmetic, precision=precision)
    # Independent arithmetic control of the eight input component blocks. This
    # checks serialized complex assembly, not an independent physical amplitude.
    pairs = [
        (str(block + 1), str(2 * block - 3)) for block in range(8) for _ in range(width)
    ]
    batch = native.NativeBatch.from_decimal(
        pairs, 1, 8 * width, row_ids=[71], **options
    )
    evaluator = native.CoefficientEvaluator(
        loaded.evaluator_state(projection["assemble"]), **options
    )
    result = evaluator.evaluate_batch(batch)
    assert result.row_ids == [71]
    rows = result.to_decimal()
    expected = [(36, 32), (22, 24), (11, 12), (7, 4), (15, 20), (16, 12), (20, 20)]
    for block, pair in enumerate(expected):
        for real, imaginary in rows[block * width : (block + 1) * width]:
            assert (Decimal(real), Decimal(imaginary)) == pair
    with pytest.raises(NotImplementedError, match=r"directed.*scheduler"):
        PortableLoopPlan.open(path, backend=native).bind(**options)
    # Native mode selection is now explicit; merely storing directed kernels
    # must not prevent binding the ordinary observable.
    assert native.NativeLoopProgram(str(path), **options).metadata_json()

    from pyamplicol.loops.prepared_correlations import _directed_order_sum_state

    # Unequal history values and multiple orders detect accidental reduction of
    # the history axis. The expected sums are elementary arithmetic series.
    totals = _directed_order_sum_state(
        4, SimpleNamespace(born=range(2), virtual=range(3))
    )
    born_values = [10 * history + order for history in range(4) for order in range(2)]
    virtual_values = [
        1000 * (component + 1) + 100 * history + 10 * order + pole
        for component in range(8)
        for history in range(4)
        for order in range(3)
        for pole in range(3)
    ]
    pairs = [(str(value), str(-value)) for value in born_values] + [
        (str(value), str(value + 1)) for value in virtual_values
    ]
    values = native.NativeBatch.from_decimal(pairs, 1, len(pairs), **options)
    summed = (
        native.CoefficientEvaluator(totals, **options)
        .evaluate_batch(values)
        .to_decimal()
    )
    for history in range(4):
        expected = [(20 * history + 1, -(20 * history + 1))]
        expected.extend(
            (3 * base + 30, 3 * base + 33)
            for component in range(8)
            for pole in range(3)
            for base in [1000 * (component + 1) + 100 * history + pole]
        )
        assert [
            (Decimal(re), Decimal(im))
            for re, im in summed[25 * history : 25 * (history + 1)]
        ] == expected


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d: d["directed_projection"].update(kind="ordinary-real"), "kind"),
        (lambda d: d["directed_projection"]["families"].pop(), "family"),
        (lambda d: d["directed_projection"]["histories"].pop(), "shape"),
        (
            lambda d: d["directed_projection"]["color_legs"][0].__setitem__(0, 9),
            "ports",
        ),
        (
            lambda d: d["directed_projection"].update(
                born=d["directed_projection"]["assemble"]
            ),
            "shape",
        ),
        (lambda d: d["directed_projection"].update(spin_sources=[]), "unknown field"),
        (lambda d: d.pop("directed_projection"), "unreferenced"),
    ],
)
def test_bad_directed_descriptor_fails_at_shared_reader(
    directed_artifact, tmp_path, mutation, message
):
    native = pytest.importorskip("_loop_reduction")
    target = tmp_path / "bad.pacbin"
    _rewrite(directed_artifact[-1], target, mutation)
    with pytest.raises(ValueError, match=message):
        native.PortableLoopPlan(str(target))


def test_writer_preserves_owner_and_unsupported_domain_boundaries(
    directed_artifact, tmp_path, monkeypatch
):
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.loops.portable import write_loop_pacbin

    source, prepared, _, _ = directed_artifact
    destination = tmp_path / "unsupported.pacbin"
    declarations = tuple(
        ColorCorrelator(f"request-{index}", bra.emissions, ket.emissions)
        for index, (bra, ket) in enumerate(prepared.bare.histories)
    )
    with pytest.raises(ValueError, match="IDs/operations"):
        write_loop_pacbin(
            source.virtual,
            destination,
            directed=prepared,
            directed_histories=(declarations[1], declarations[0], *declarations[2:]),
        )
    with pytest.raises(ValueError, match="IDs/operations"):
        write_loop_pacbin(
            source.virtual,
            destination,
            directed=prepared,
            directed_histories=tuple(
                replace(item, id="duplicate") for item in declarations
            ),
        )
    with pytest.raises(ValueError, match="different retained model"):
        write_loop_pacbin(
            source.virtual, destination, directed=replace(prepared, plan=object())
        )
    with pytest.raises(NotImplementedError, match="spin/resolved"):
        write_loop_pacbin(source.virtual, destination, directed=prepared, resolved=True)
    spin = replace(prepared, bare=replace(prepared.bare, spin_layout=object()))
    with pytest.raises(NotImplementedError, match="spin/resolved"):
        write_loop_pacbin(source.virtual, destination, directed=spin)
    monkeypatch.setattr(source.virtual.bare, "color_accuracy", "lc", raising=False)
    with pytest.raises(NotImplementedError, match="projected-colour"):
        write_loop_pacbin(source.virtual, destination, directed=prepared)
    assert not destination.exists()
