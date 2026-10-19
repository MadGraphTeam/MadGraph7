# SPDX-License-Identifier: 0BSD
"""Exact public group selection against independently generated member plans."""

import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol import CorrelatedRequest, CorrelatorConfig, Runtime
from pyamplicol.api import Generator, ProcessRequest, ProcessSet
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.process_groups import (
    read_process_groups,
    write_process_groups,
)
from pyamplicol.color.connections import EmitGluon
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.config import resolve_config
from pyamplicol.loops.flavor_transport import identity_species_candidate
from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
from pyamplicol.models.loading import compile_model_source

from .test_loop_generation_public_runtime import _card


def test_identity_species_candidates_preserve_signed_bijections_without_caps():
    assert identity_species_candidate((2, -2, 1, -1), (2, -2, 3, -3)) == {
        1: 3,
        -1: -3,
        3: 1,
        -3: -1,
    }
    # Three-cycle completion, not a hard-coded pairwise flavour swap.
    assert identity_species_candidate((1, 3), (3, 5)) == {
        1: 3,
        -1: -3,
        3: 5,
        -3: -5,
        5: 1,
        -5: -1,
    }
    assert identity_species_candidate((1, 1), (1, 3)) is None
    assert identity_species_candidate((1, -1), (3, -5)) is None
    assert identity_species_candidate((21,), (21,)) == {}


def test_exact_group_basis_is_weaker_than_identity_and_stronger_than_numerical():
    from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember

    item = GroupMember("a", "a a > b b", (1, -1, 3, -3), 2, permutation=(0, 1, 2, 3))
    exact = replace(item, basis=GroupingBasis.EXACT_LOOP_FLAVOR)
    numerical = replace(item, basis=GroupingBasis.NUMERICAL)
    assert item.through(exact).basis == exact.basis
    assert exact.through(item).basis == exact.basis
    assert exact.through(numerical).basis == numerical.basis


def test_identical_pdg_requests_remain_independent_without_species_admission():
    from pyamplicol.generation.loop_service import _group_exact_flavors

    structure = object()
    program = SimpleNamespace(
        source_bare=SimpleNamespace(
            structure=structure,
            external_request=SimpleNamespace(external_pdgs=(2, -2, 1, -1)),
        )
    )
    context = SimpleNamespace(
        expanded=tuple(
            SimpleNamespace(
                request=SimpleNamespace(name=name, expression="u u~ > d d~")
            )
            for name in ("first", "second")
        )
    )
    retained, admissions, representatives, _ = _group_exact_flavors(
        context, (program, program)
    )
    assert retained == [0, 1] and not admissions
    assert representatives == {"first": "first", "second": "second"}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    source = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not source:
        pytest.skip("original loop_sm source required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    root = tmp_path_factory.mktemp("public-loop-mapping-model")
    model = compile_model_source(
        source, restriction="no_b_mass", cache_dir=root / "cache"
    )
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / "model.json")


@pytest.mark.parametrize("correlated", [False, True])
def test_public_exact_flavor_member_native_resolved_and_directed(
    model_path, tmp_path, monkeypatch, correlated
):
    from pyamplicol.generation import loop_service
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.portable import write_loop_pacbin

    native = load_loop_native()
    captured = {}
    original = loop_service._group_exact_flavors

    def retain_actual_programs(context, programs, projections=None):
        # Observation only: all actual source construction and proof still run.
        captured["programs"] = tuple(programs)
        return original(context, programs, projections)

    monkeypatch.setattr(loop_service, "_group_exact_flavors", retain_actual_programs)
    projected = []
    prepare_directed = loop_service._prepare_directed

    def retain_actual_projectors(*args):
        result = prepare_directed(*args)
        projected.append(result)
        return result

    monkeypatch.setattr(loop_service, "_prepare_directed", retain_actual_projectors)
    output = tmp_path / "artifact"
    card = _card(model_path, output).replace(
        'entries = [{expression="u u~ > e- e+", name="dy"},\n'
        '           {expression="g g > g g", name="gg"}]',
        'entries = [{expression="u u~ > d d~", name="uudd"},\n'
        '           {expression="u u~ > s s~", name="uuss"}]\n'
        "max_coupling_orders = {QCD=2, QED=0}\nflavor_blind = true",
    )
    path = tmp_path / "card.toml"
    path.write_text(card)
    requests = ProcessSet(
        tuple(
            ProcessRequest.parse(text, name=name)
            for name, text in (
                ("uudd", "u u~ > d d~"),
                ("uuss", "u u~ > s s~"),
            )
        )
    )
    declarations = (
        CorrelatorConfig(
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
            )
        )
        if correlated
        else None
    )
    Generator(resolve_config(path)).generate(requests, output, correlators=declarations)
    manifest = load_manifest(output)
    assert len(manifest.processes) == 1
    grouping = manifest.extensions["generation"]["concrete_processes"][0]["filters"][
        "flavor_grouping"
    ]
    assert grouping["requested"] and grouping["transport_checked"]
    assert grouping["reuse_applied"] and grouping["independent_reason"] is None
    programs = captured["programs"]
    assert len(programs) == 2 and programs[0].prepared is not programs[1].prepared
    independent_path = tmp_path / "independent-uuss.pacbin"
    directed, histories = (
        (projected[1][0], projected[1][1]) if correlated else (None, ())
    )
    write_loop_pacbin(
        programs[1].virtual,
        independent_path,
        resolved=not correlated,
        directed=directed,
        directed_histories=histories or None,
    )
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"][
        :2
    ]
    scales, ids = ["1000000", "900000"], [2**64 - 1, 7]
    for arithmetic, precision, engine, tolerance in (
        ("double", None, "fast", "4e-11"),
        ("double-double", None, "generic", "4e-25"),
        ("arbitrary", 70, "generic", "4e-57"),
    ):
        mapped = native.NativeVirtualRuntime(
            str(output), process="uuss", arithmetic=arithmetic, precision=precision
        )
        independent = native.NativeLoopProgram(
            str(independent_path),
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine=engine,
        )
        metadata = json.loads(mapped.metadata_json())
        assert metadata["process_id"] == "uuss"
        assert metadata["representative_process_id"] == "uudd"
        assert metadata["external_pdgs"] == [2, -2, 3, -3]
        assert metadata["plan"]["external_pdgs"] == [2, -2, 1, -1]
        original_parameters = mapped.parameter_overrides
        with pytest.raises(ValueError, match="fixed"):
            mapped.set_model_parameters({"aS": ["0.12", "0"], "MB": ["1", "0"]})
        assert mapped.parameter_overrides == original_parameters
        if correlated:
            actual = mapped.evaluate_directed(points, scales, point_ids=ids)
            expected = independent.evaluate_directed(points, scales, point_ids=ids)
        else:
            actual = mapped.evaluate_resolved(points, scales, point_ids=ids)
            expected = independent.evaluate(
                points, scales, point_ids=ids, resolved=True
            )
        assert actual.point_ids == expected.point_ids == ids
        left = [actual.born, *actual.component_batches, actual.totals]
        right = [expected.born, *expected.component_batches, expected.totals]
        if correlated:
            left += [actual.directed_first, actual.directed_second]
            right += [expected.directed_first, expected.directed_second]
        for a, b in zip(left, right, strict=True):
            assert all(ok for _, ok in a.agreement_rows(b, tolerance, tolerance))
        runtime = Runtime.load(
            output, process="uuss", arithmetic=arithmetic, precision=precision
        )
        assert (
            runtime.physics.process_id == "uuss"
            and runtime.representative_process_key == "uudd"
        )
        assert tuple(p.pdg_id for p in runtime.physics.external_particles) == (
            2,
            -2,
            3,
            -3,
        )
        if correlated:
            result = runtime.evaluate_correlated_many(
                points,
                {name: CorrelatedRequest(name) for name in ("ordered", "swapped")},
                mu_r2=scales,
            )
            assert abs(result["ordered"].values[0][0][0].imag) > Decimal("1e-4")
        else:
            result = runtime.evaluate_resolved(points, mu_r2=scales)
            assert result.shape == (2, 1, 16, 1, 3)
            assert result.total().shape == (2, 1, 3)
    groups = read_process_groups(output)
    original_groups = groups.to_json()
    member = groups.groups["uudd"].members[1]
    assert member.process_id == "uuss" and str(member.basis) == "exact-loop-flavor"
    from pyamplicol.generation.process_grouping import ProcessGroupMap

    for changes in (
        {"factor": 2},
        {"conjugated": True},
        {"permutation": (1, 0, 2, 3)},
        {"pdgs": (2, -2, 4, -4)},
        {"process_id": "unadmitted"},
    ):
        altered = ProcessGroupMap.from_json(original_groups)
        altered.groups["uudd"].members[1] = replace(member, **changes)
        write_process_groups(output, altered)
        with pytest.raises(ValueError, match=r"admission|flavour|group|particle"):
            native.NativeVirtualRuntime(
                str(output), process=changes.get("process_id", "uuss")
            )
    write_process_groups(output, ProcessGroupMap.from_json(original_groups))
