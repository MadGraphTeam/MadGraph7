# SPDX-License-Identifier: 0BSD
"""Public virtual generation preserves correction selection and fails explicitly."""

from types import SimpleNamespace

import pytest

from pyamplicol.api import GenerationError, Generator, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config
from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.generation.correlated import correlated_configuration
from pyamplicol.generation.loop_service import _graph_plan, _order_bounds
from pyamplicol.generation.service import GenerationBackend


def loop_config(**sections):
    data = {
        "action": "generate",
        "process": {"loop_coupling_orders": ["QCD"]},
        "color": {"accuracy": "full", "contraction": "direct"},
        "generation": {"emit_api_bundle": False},
    }
    data.update(sections)
    return resolve_config(data)


def test_public_order_ceilings_keep_lower_sectors_and_joint_corrections():
    # Integers in the internal graph API mean exact orders. Public maxima must
    # keep zero/lower Born orders, and both coordinates grow for joint selection.
    born, loop = _order_bounds({"QCD": 2, "QED": 2, "BSM": 1}, ("QCD", "QED"))
    assert born == {"QCD": (0, 2), "QED": (0, 2), "BSM": (0, 1)}
    assert loop == {"QCD": (0, 4), "QED": (0, 4), "BSM": (0, 1)}
    assert _order_bounds({}, ("QCD",)) == ({}, {})


def test_absent_loop_metadata_is_a_dry_run_blocker_not_tree_generation(tmp_path):
    backend = GenerationBackend(loop_config(), None)
    processes = ProcessSet((ProcessRequest.parse("u u~ > e- e+"),))
    plan = backend.plan(processes)
    assert plan.estimated_coverage["contribution_kind"] == "virtual"
    assert any("exact raw sidecar" in reason for reason in plan.unsupported_features)
    with pytest.raises(GenerationError, match="exact raw sidecar"):
        backend.generate(processes, tmp_path / "tree-model")
    assert not (tmp_path / "tree-model").exists()


def test_virtual_correlators_never_enter_tree_compiled_generation(
    tmp_path, monkeypatch
):
    config = loop_config()
    for value in (config, config.effective):
        with pytest.raises(GenerationError, match="virtual correlated generation"):
            correlated_configuration(value)

    def wrong_route(*args, **kwargs):
        raise AssertionError("tree correlated backend must not be called")

    monkeypatch.setattr(
        "pyamplicol.generation.correlated.generate_correlated", wrong_route
    )
    with pytest.raises(GenerationError, match="virtual correlated generation"):
        Generator(config).generate(
            "u u~ > e- e+", tmp_path / "correlated", correlators=CorrelatorConfig()
        )
    assert not (tmp_path / "correlated").exists()


@pytest.mark.parametrize(
    "process_changes,reason",
    [
        ({"selected_source_helicities": {"1": 1}}, "inclusive helicity"),
        ({"flavor_blind": True}, "flavour/process-group"),
    ],
)
def test_pending_loop_features_are_reported_before_model_work(
    process_changes, reason, monkeypatch
):
    config = loop_config(process={"loop_coupling_orders": ["QCD"], **process_changes})
    backend = GenerationBackend(config, None)
    monkeypatch.setattr(
        backend,
        "_resolve_model_for_plan",
        lambda *_: pytest.fail("unavailable feature reached model generation"),
    )
    plan = backend.plan(ProcessSet((ProcessRequest.parse("g g > g g"),)))
    assert any(reason in item for item in plan.unsupported_features)


def test_absent_correction_bound_does_not_rewrite_other_order_ceilings():
    # An omitted coordinate is unbounded in both the tree and graph APIs.
    # Selecting QCD must not enlarge the explicit QED ceiling or invent a QCD
    # Born constraint; graph eligibility remains authoritative.
    born, loop = _order_bounds({"QED": 2, "BSM": 1}, ("QCD",))
    assert born == loop == {"QED": (0, 2), "BSM": (0, 1)}
    assert "QCD" not in born and "QCD" not in loop


def test_dry_run_transports_joint_selection_to_one_graph_plan(monkeypatch):
    from pyamplicol.loops.graphs import GraphSelectionConvention
    from pyamplicol.loops.orders import (
        BornLoopPair,
        LoopContribution,
        LoopEligibility,
        compile_order_plan,
    )

    born = (("QCD", 2),)
    loop = (("QCD", 2), ("QED", 2))
    mixed = LoopEligibility(
        (frozenset({"QCD", "QED"}), frozenset({"QCD"}), frozenset({"QED"})),
        frozenset({"QCD", "QED"}),
    )
    calls = []

    def graph_plan(process, **kwargs):
        calls.append(kwargs)
        orders = compile_order_plan(
            selected_corrections=kwargs["selected_corrections"],
            perturbative_orders=kwargs["perturbative_orders"],
            born_orders=(born,),
            contributions=(LoopContribution("mixed-box", loop, mixed),),
            allowed_pairs=(BornLoopPair(born, loop),),
        )
        assert [item.id for item in orders.contributions] == ["mixed-box"]
        return SimpleNamespace(
            orders=orders,
            born=(object(),),
            loops=(object(),),
            families=(object(),),
            convention=GraphSelectionConvention(),
        )

    external = SimpleNamespace(
        incoming=(1, -1),
        outgoing=(2, -2),
        external_pdgs=(1, -1, 2, -2),
        to_dict=lambda: {"normalization": {"divisor": 36}},
    )
    prepared = SimpleNamespace(
        structure=SimpleNamespace(
            native=object(),
            perturbative_species_orders={},
            raw=SimpleNamespace(
                collections={
                    "coupling_orders": [
                        {"name": name, "fields": {"perturbative_expansion": 1}}
                        for name in ("QCD", "QED")
                    ]
                }
            ),
        ),
        counterterms=SimpleNamespace(process=lambda *_: object()),
        scope=SimpleNamespace(
            born_order_bounds={"QCD": (0, 2), "QED": (0, 2)},
            loop_order_bounds={"QCD": (0, 4), "QED": (0, 4)},
        ),
    )
    entry = SimpleNamespace(
        request=SimpleNamespace(name="mixed", expression="u u~ > d d~"),
        process_ir=SimpleNamespace(initial_pdgs=(2, -2), final_pdgs=(1, -1)),
    )
    monkeypatch.setattr(
        "pyamplicol.generation.loop_service.prepare_loop_external_request",
        lambda *_: external,
    )
    monkeypatch.setattr(
        "pyamplicol.generation.loop_service.build_loop_process", graph_plan
    )
    result = _graph_plan(prepared, entry, ("QCD", "QED"))
    assert len(calls) == 1
    assert calls[0]["selected_corrections"] == ("QCD", "QED")
    assert result["orders"]["correction_names"] == ("QCD", "QED")
    assert result["orders"]["virtual"][0]["order"] == (("QCD", 4), ("QED", 2))
    assert result["loop_graph_count"] == 1
    assert result["numerator_and_reducer_capabilities_checked"] is False


@pytest.mark.parametrize(
    "tuning,path",
    [
        ({"jit": {"optimization_level": 3}}, "evaluator.jit.optimization_level"),
        (
            {"optimization": {"horner_iterations": 2}},
            "evaluator.optimization.horner_iterations",
        ),
        (
            {"recurrence": {"point_tile_size": 32}},
            "evaluator.recurrence.point_tile_size",
        ),
        ({"output_chunk_size": 100}, "evaluator.output_chunk_size"),
    ],
)
def test_tree_tuning_does_not_silently_change_recorded_loop_settings(tuning, path):
    config = loop_config(evaluator=tuning)
    plan = GenerationBackend(config, None).plan(
        ProcessSet((ProcessRequest.parse("g g > g g"),))
    )
    assert any(path in reason for reason in plan.unsupported_features)


def test_shared_resource_cores_are_not_rejected_as_tree_tuning():
    from pyamplicol.generation.loop_service import _feature_blockers

    backend = GenerationBackend(
        loop_config(evaluator={"optimization": {"cores": 1}}), None
    )
    assert (
        _feature_blockers(backend, ProcessSet((ProcessRequest.parse("g g > g g"),)))
        == ()
    )
