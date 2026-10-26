# SPDX-License-Identifier: 0BSD
"""Fresh public spin/flavour composition against an independent saved member."""

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import CorrelatedRequest, CorrelatorConfig, Runtime
from pyamplicol.api import (
    Generator,
    ProcessRequest,
    ProcessSet,
    VirtualArithmeticSpec,
    VirtualStabilityPolicy,
)
from pyamplicol.api.errors import EvaluationError
from pyamplicol.artifacts import load_manifest
from pyamplicol.color.connections import EmitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.config import resolve_config

from .test_loop_generation_public_runtime import _card

pytestmark = pytest.mark.loop_nightly

# The opt-in fixture keeps source owners for diagnostic follow-ups in an
# interactive research process; saved numerical controls use ordinary PACBIN.
_LIVE_SOURCE_PAIR = None


def _blocks(result):
    return [
        result.born,
        *result.component_batches,
        result.totals,
        result.directed_first,
        result.directed_second,
    ]


def _vectors():
    return (
        [["0.1", "0.2"], ["0.8", "-0.3"], ["0.4", "1"], ["-0.6", "0.5"]],
        [["-0.3", "0.7"], ["-0.2", "0.8"], ["0.6", "0.1"], ["0.5", "-0.4"]],
    )


@pytest.fixture(scope="module")
def mapped_sources(loop_sm_ufo, tmp_path_factory):
    source = str(loop_sm_ufo)
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.generation import loop_service
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    retained = os.environ.get("PYAMPLICOL_SPIN_FLAVOR_ARTIFACT")
    root = (
        Path(retained).resolve()
        if retained
        else tmp_path_factory.mktemp("spin-flavor-public")
    )
    output, independent = root / "artifact", root / "independent-ssgg.pacbin"
    if not retained:
        model = compile_model_source(
            source, restriction="no_b_mass", cache_dir=root / "model-cache"
        )
        model = replace(
            model,
            raw_loop_model=prepare_loop_sm_scalar_declarations(
                model.raw_loop_model
            ).derived,
            _serialized_path=None,
            _prepared_bundle=None,
        )
        model_path = model.write(root / "model.json")
        card = _card(model_path, output).replace(
            'entries = [{expression="u u~ > e- e+", name="dy"},\n'
            '           {expression="g g > g g", name="gg"}]',
            'entries = [{expression="d d~ > g g", name="ddgg"},\n'
            '           {expression="s s~ > g g", name="ssgg"}]\n'
            "max_coupling_orders = {QCD=2, QED=0}\nflavor_blind = true",
        )
        card += "relative_tolerance = 3e-9\nabsolute_tolerance = 2e-12\n"
        card_path = root / "virtual.toml"
        card_path.write_text(card)
        requests = ProcessSet(
            tuple(
                ProcessRequest.parse(text, name=name)
                for name, text in (("ddgg", "d d~ > g g"), ("ssgg", "s s~ > g g"))
            )
        )
        declarations = CorrelatorConfig(
            spin_correlations=((3,), (4,), (3, 4)),
            color_correlations=(
                ColorCorrelator.dipole("casimir", 1, 1),
                ColorCorrelator(
                    "ordered",
                    (EmitGluon(1, -1), EmitGluon(1, -2)),
                    (EmitGluon(2, -1), EmitGluon(3, -2)),
                ),
                ColorCorrelator(
                    "swapped",
                    (EmitGluon(2, -1), EmitGluon(3, -2)),
                    (EmitGluon(1, -1), EmitGluon(1, -2)),
                ),
            ),
        )
        original = loop_service._group_exact_flavors

        def observe(context, programs, projections=None):
            global _LIVE_SOURCE_PAIR
            # Save both independent programs before any comparison assertion.
            for index, name in enumerate(("ddgg", "ssgg")):
                ordinary, histories, sources = projections[index]
                write_loop_pacbin(
                    programs[index].virtual,
                    root / f"independent-{name}.pacbin",
                    resolved=False,
                    directed=ordinary,
                    directed_histories=histories,
                    spin_directed=sources,
                )
            _LIVE_SOURCE_PAIR = (context, tuple(programs), tuple(projections))
            outcome = original(context, programs, projections)
            assert programs[0].prepared is not programs[1].prepared
            return outcome

        # Observation only: all real generation, proof and publication run.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loop_service, "_group_exact_flavors", observe)
            Generator(resolve_config(card_path)).generate(
                requests, output, correlators=declarations
            )
        print(f"SPIN_FLAVOR_ARTIFACT {root}", flush=True)
    manifest = load_manifest(output)
    assert len(manifest.processes) == 1
    grouping = manifest.extensions["generation"]["concrete_processes"][0]["filters"][
        "flavor_grouping"
    ]
    assert grouping["transport_checked"] and grouping["reuse_applied"]
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"][
        :2
    ]
    return output, independent, points


def test_retained_actual_source_admission_and_independent_physical_comparison(
    mapped_sources,
):
    if _LIVE_SOURCE_PAIR is None:
        pytest.skip(
            "retained artifact run; source comparisons were performed at generation"
        )
    import numpy as np

    from pyamplicol.generation.loop_service import _group_exact_flavors

    from .test_loop_spin_flavor_transport import (
        test_independent_spin_members_orders_components_histories_and_exact_inputs,
        test_paired_source_admission_rejects_foreign_classes_histories_phases_and_owners,
    )

    context, programs, projections = _LIVE_SOURCE_PAIR
    _, admissions, _, _ = _group_exact_flavors(context, programs, projections)
    relation = admissions[0][0].transport
    pair = (
        relation,
        programs,
        tuple(p[0] for p in projections),
        tuple(p[2] for p in projections),
        np.asarray(mapped_sources[2], dtype=str),
    )
    test_paired_source_admission_rejects_foreign_classes_histories_phases_and_owners(
        pair
    )
    for options in (
        ("double", None, "fast", "8e-11", 20),
        ("double-double", None, "generic", "8e-25", 70),
        ("arbitrary", 70, "generic", "8e-57", 170),
    ):
        test_independent_spin_members_orders_components_histories_and_exact_inputs(
            pair, *options
        )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "8e-11", 20),
        ("double-double", None, "generic", "8e-25", 70),
        ("arbitrary", 70, "generic", "8e-57", 170),
    ],
)
def test_saved_mapped_source_classes_all_carriers_exact_updates_and_point_ids(
    mapped_sources, arithmetic, precision, engine, tolerance, bit
):
    from pyamplicol.loops.native import load_loop_native

    path, independent_path, points = mapped_sources
    native = load_loop_native()
    mapped = native.NativeVirtualRuntime(
        str(path), process="ssgg", arithmetic=arithmetic, precision=precision
    )
    independent = native.NativeLoopProgram(
        str(independent_path),
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    metadata = json.loads(mapped.metadata_json())
    assert (
        metadata["process_id"] == "ssgg"
        and metadata["representative_process_id"] == "ddgg"
    )
    assert metadata["external_pdgs"] == [3, -3, 21, 21]
    a, b = _vectors()
    ids, scales = [2**64 - 1, 7, 42], ["900000", "1000000", "900000"]
    batch = [points[1], points[0], points[1]]
    with localcontext() as context:
        context.prec = 120
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        changes = {"aS": [alpha, "0"], "MT": ["173", "0"]}
        mapped.set_model_parameters(changes)
        independent.source_state.update(changes)
        for key in ((), (3,), (4,), (3, 4)):
            vectors = {leg: [a, b, a] if leg == 3 else [b] for leg in key}
            actual = mapped.evaluate_directed(
                batch, scales, point_ids=ids, spin_vectors=vectors
            )
            expected = independent.evaluate_directed(
                batch, scales, point_ids=ids, spin_vectors=vectors
            )
            assert actual.point_ids == expected.point_ids == ids
            assert actual.spin_legs == list(key)
            for left, right in zip(_blocks(actual), _blocks(expected), strict=True):
                assert all(
                    ok for _, ok in left.agreement_rows(right, tolerance, tolerance)
                )
            empty = mapped.evaluate_directed(
                [], [], point_ids=[], spin_vectors={leg: [a] for leg in key}
            )
            assert empty.point_ids == [] and empty.spin_legs == list(key)
        # An admitted mapping must preserve exact per-call sources, including
        # changes below binary64 in the higher carriers. One leg is rescaled.
        tiny = Decimal(2) ** -bit
        scaled = [
            [str(Decimal(x) * (1 + tiny)), str(Decimal(y) * (1 + tiny))] for x, y in a
        ]
        homogeneity = mapped.evaluate_directed(
            [points[0]] * 2,
            ["1000000"] * 2,
            spin_vectors={3: [a, scaled], 4: [b]},
        ).born
        values = homogeneity.to_decimal()
        first, second = Decimal(values[0][0]), Decimal(values[homogeneity.columns][0])
        expected_delta = first * (2 * tiny + tiny * tiny)
        assert abs(second - first - expected_delta) < abs(expected_delta) * Decimal(
            "1e-4"
        )
        original = mapped.parameter_overrides
        with pytest.raises(ValueError, match="fixed"):
            mapped.set_model_parameters({"aS": ["0.12", "0"], "MB": ["1", "0"]})
        assert mapped.parameter_overrides == original
    public = Runtime.load(
        path, process="ssgg", arithmetic=arithmetic, precision=precision
    )
    assert public.representative_process_key == "ddgg"
    assert tuple(p.pdg_id for p in public.physics.external_particles) == (3, -3, 21, 21)
    result = public.evaluate_correlated_many(
        batch,
        {
            "ordered": CorrelatedRequest("ordered", {3: [a, b, a], 4: b}),
            "swapped": CorrelatedRequest("swapped", {3: [a, b, a], 4: b}),
        },
        mu_r2=scales,
    )
    for left, right in zip(
        result["ordered"].values, result["swapped"].values, strict=True
    ):
        for x, y in zip(left[0], right[0], strict=True):
            assert float(x.real) == pytest.approx(float(y.real), rel=1e-10, abs=1e-12)
            assert float(x.imag) == pytest.approx(-float(y.imag), rel=1e-10, abs=1e-12)
    public.set_spin_correlation_vectors({3: a, 4: b})
    before = public.evaluate_correlated(batch[:1], mu_r2=scales[:1])
    public.clear()
    assert (
        public.evaluate_correlated(batch[:1], mu_r2=scales[:1]).values == before.values
    )
    assert public.evaluate_correlated([], mu_r2=[]).shape == (0, 1, 3)


def test_mapped_source_subset_rescue_and_cold_admission(mapped_sources, tmp_path):
    from pyamplicol.loops.native import load_loop_native

    from .test_loop_portable import _rewrite

    path, _, points = mapped_sources
    native = load_loop_native()
    a, b = _vectors()
    runtime = Runtime.load(path, process="ssgg")
    requests = {"joint": CorrelatedRequest("ordered", {3: [a, b, a], 4: b})}
    batch, scales = [points[0]] * 3, ["1000000", "1e-400", "1000000"]
    with pytest.raises(EvaluationError) as failure:
        runtime.evaluate_correlated_many(batch, requests, mu_r2=scales)
    assert {item.point_id for item in failure.value.failures} == {1}
    rescued = runtime.evaluate_correlated_many(
        batch,
        requests,
        mu_r2=scales,
        stability=VirtualStabilityPolicy(
            (
                VirtualArithmeticSpec("arbitrary", 60),
                VirtualArithmeticSpec("arbitrary", 80),
            ),
            "1e-8",
            "1e-9",
        ),
    )["joint"]
    assert [row.status for row in rescued.diagnostics] == [
        "success",
        "rescued",
        "success",
    ]
    assert rescued.values[0] == rescued.values[2]
    saved = path / "processes/ddgg/loop-runtime.pacbin"

    def relabel_vector(plan):
        record = plan["admitted_flavor_members"][0]
        record["species"].extend([[21, 22], [22, 21], [-21, -22], [-22, -21]])
        record["external_pdgs"] = [3, -3, 22, 22]

    invalid = tmp_path / "invalid-source-map.pacbin"
    _rewrite(saved, invalid, relabel_vector)
    with pytest.raises(ValueError, match=r"flavour|member|source"):
        native.PortableLoopPlan(str(invalid))
