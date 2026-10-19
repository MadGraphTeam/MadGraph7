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
    from pyamplicol.api.results import GenerationResult

    calls = []

    def loop_route(
        self, processes, output, *, model=None, mode="error", correlators=None
    ):
        calls.append(correlators)
        return GenerationResult(output=output, processes=processes, mode=mode, files=())

    monkeypatch.setattr(GenerationBackend, "generate", loop_route)
    declarations = CorrelatorConfig()
    result = Generator(config).generate(
        "u u~ > e- e+", tmp_path / "correlated", correlators=declarations
    )
    assert calls == [declarations]
    assert result.output == tmp_path / "correlated"
    assert not result.output.exists()


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


def _colour_backend(monkeypatch, accuracy, declaration):
    """Transport fixture: no tensor/graph numerical work at this boundary."""
    from pyamplicol.generation import loop_service as service
    from pyamplicol.generation.service import _ResolvedModel
    from pyamplicol.models.loop_declaration import (
        LoopModelDeclaration,
        LoopSchemeDeclaration,
    )

    scheme = LoopSchemeDeclaration(
        "HV", "4d", "source", "source", "real", "source", "fixture"
    )
    loop = LoopModelDeclaration(scheme, ("QCD", "QED"), True, True)
    backend = GenerationBackend(
        loop_config(color={"accuracy": accuracy, "contraction": "direct"}), None
    )
    compiled = SimpleNamespace(raw_loop_model=object())
    resolved = _ResolvedModel(source=None, model=None, compiled=compiled)
    for name in ("_resolve_model_for_plan", "_resolve_model"):
        monkeypatch.setattr(backend, name, lambda *_: resolved)
    monkeypatch.setattr(backend, "_artifact_model", lambda *_: compiled)
    monkeypatch.setattr(backend, "_detect_symbolica_license", lambda: None)
    monkeypatch.setattr(
        backend, "_apply_symbolica_resource_policy", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        service, "resolve_loop_model_declaration", lambda *a, **kw: loop
    )
    monkeypatch.setattr(service, "with_loop_model_declaration", lambda value, _: value)
    monkeypatch.setattr(
        service, "resolve_loop_color_declaration", lambda _: declaration(scheme)
    )
    monkeypatch.setattr(
        "pyamplicol.models.external.CompiledUFOModel", lambda _: object()
    )
    return backend, scheme


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_public_colour_missing_declaration_is_actionable_in_plan_and_generate(
    accuracy, monkeypatch, tmp_path
):
    backend, _ = _colour_backend(monkeypatch, accuracy, lambda _: None)
    processes = ProcessSet((ProcessRequest.parse("g g > g g"),))
    plan = backend.plan(processes)
    assert plan.estimated_coverage["color_accuracy"] == accuracy
    assert not plan.estimated_coverage["source_colour_declared"]
    assert len(plan.unsupported_features) == 1
    assert "with_loop_color_declaration" in plan.unsupported_features[0]
    output = tmp_path / "missing-declaration"
    with pytest.raises(GenerationError, match="with_loop_color_declaration"):
        backend.generate(processes, output)
    assert not output.exists()


def test_public_colour_scheme_mismatch_fails_before_preparation(monkeypatch):
    from dataclasses import replace

    backend, _ = _colour_backend(
        monkeypatch,
        "lc",
        lambda scheme: SimpleNamespace(
            scheme=replace(scheme, dimensional_regularization="FDH")
        ),
    )
    monkeypatch.setattr(
        "pyamplicol.generation.loop_service.prepare_loop_request_model",
        lambda *a, **kw: pytest.fail("mismatched scheme reached source preparation"),
    )
    plan = backend.plan(ProcessSet((ProcessRequest.parse("g g > g g"),)))
    assert plan.unsupported_features == (
        "source colour continuation and effective loop scheme differ",
    )


def test_colour_owner_is_bound_once_per_actual_inferred_order_scope(monkeypatch):
    from pyamplicol.generation import loop_service as service

    backend, _ = _colour_backend(
        monkeypatch, "nlc", lambda scheme: SimpleNamespace(scheme=scheme)
    )
    requests = ProcessSet(
        tuple(
            ProcessRequest.parse(text, name=name)
            for name, text in (
                ("dy", "u u~ > e- e+"),
                ("dy2", "d d~ > e- e+"),
                ("gg", "g g > g g"),
            )
        )
    )
    entries = tuple(
        SimpleNamespace(request=request, process_ir=index)
        for index, request in enumerate(requests.requests)
    )
    monkeypatch.setattr(backend, "_expand_process_set", lambda *a: entries)
    monkeypatch.setattr(
        service,
        "infer_minimal_coupling_order_limits",
        lambda index, **kw: {"QED": 2, "QCD": 0} if index < 2 else {"QCD": 2, "QED": 0},
    )
    fixed = {"lhv": ("1", "0")}
    monkeypatch.setattr(service, "source_color_fixed_parameters", lambda *a: fixed)
    prepared_calls, colour_calls = [], []

    def prepare(*args, **kwargs):
        owner = SimpleNamespace(scope=kwargs)
        prepared_calls.append(owner)
        return owner

    def colour(owner, data, *, scheme):
        continuation = SimpleNamespace(owner=owner, data=data, scheme=scheme)
        colour_calls.append(continuation)
        return owner, continuation

    monkeypatch.setattr(service, "prepare_loop_request_model", prepare)
    monkeypatch.setattr(service, "prepare_declared_source_color", colour)
    context = service._prepare(backend, requests, None, planning=True, phase=None)
    assert len(prepared_calls) == len(colour_calls) == 2
    assert context.prepared[0] is context.prepared[1]
    assert context.source_colors[0] is context.source_colors[1]
    assert context.prepared[2] is not context.prepared[0]
    assert all(
        c.owner is p
        for c, p in zip(context.source_colors, context.prepared, strict=True)
    )
    assert all(p.scope["fixed_parameters"] is fixed for p in context.prepared)
    assert context.prepared[0].scope["loop_order_bounds"] == {
        "QED": (0, 2),
        "QCD": (0, 2),
    }
    assert context.prepared[2].scope["loop_order_bounds"] == {
        "QCD": (0, 4),
        "QED": (0, 0),
    }
    plan = service._plan_result(backend, requests, (), context=context)
    assert plan.estimated_coverage["source_colour_declared"] is True
    assert plan.estimated_coverage["formal_colour_process_coverage_checked"] is False


def test_full_colour_keeps_original_source_path_without_reading_colour_data(
    monkeypatch,
):
    from pyamplicol.generation import loop_service as service

    backend, _ = _colour_backend(
        monkeypatch,
        "full",
        lambda _: pytest.fail("full colour consumed formal declaration"),
    )
    requests = ProcessSet((ProcessRequest.parse("g g > g g"),))
    entry = SimpleNamespace(request=requests.requests[0], process_ir=object())
    monkeypatch.setattr(backend, "_expand_process_set", lambda *a: (entry,))
    monkeypatch.setattr(
        service, "infer_minimal_coupling_order_limits", lambda *a, **kw: {"QCD": 2}
    )
    calls = []
    monkeypatch.setattr(
        service,
        "prepare_loop_request_model",
        lambda *a, **kw: calls.append(kw) or object(),
    )
    monkeypatch.setattr(
        service,
        "prepare_declared_source_color",
        lambda *a, **kw: pytest.fail("full colour bound continuation"),
    )
    context = service._prepare(backend, requests, None, planning=False, phase=None)
    assert calls[0]["fixed_parameters"] is None
    assert context.source_colors == (None,)
    assert context.color_declaration is None


def test_colour_required_constant_cannot_undo_an_actual_restriction_card(
    monkeypatch, tmp_path
):
    from pyamplicol.generation import loop_service as service
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from pyamplicol.models.loop_import import load_raw_loop_model

    from .test_loop_model_restrictions import source as restriction_source

    # Use the existing exact UFO/card fixture: MASS 991 is fixed to zero.
    path = restriction_source.__wrapped__(tmp_path)
    raw = load_raw_loop_model(path)
    backend, _ = _colour_backend(
        monkeypatch,
        "nlc",
        lambda scheme: LoopColorDeclaration(
            scheme=scheme,
            formal_symbol="Nc",
            parameter_identities={},
            kernel_keys=(),
            field_keys=(),
            scheme_parameter_values={"M": "1"},
        ),
    )
    compiled = SimpleNamespace(raw_loop_model=raw)
    monkeypatch.setattr(backend, "_artifact_model", lambda *_: compiled)
    requests = ProcessSet((ProcessRequest.parse("g g > g g"),))
    entry = SimpleNamespace(request=requests.requests[0], process_ir=object())
    monkeypatch.setattr(backend, "_expand_process_set", lambda *a: (entry,))
    monkeypatch.setattr(
        service, "infer_minimal_coupling_order_limits", lambda *a, **kw: {}
    )
    monkeypatch.setattr(
        service,
        "prepare_loop_request_model",
        lambda model, **kwargs: prepare_loop_request_model(
            model.raw_loop_model, **kwargs
        ),
    )
    plan = backend.plan(requests)
    assert plan.unsupported_features == (
        "fixed parameter 'M' contradicts the source restriction",
    )


@pytest.mark.parametrize(
    "sections,declarations,message",
    [
        ({}, CorrelatorConfig(spin_correlations=((1,),)), "spin-source"),
        ({"color": {"accuracy": "lc"}}, CorrelatorConfig(), "full colour"),
        (
            {"process": {"loop_coupling_orders": ["QED"]}},
            CorrelatorConfig(),
            "QCD corrections",
        ),
        ({}, object(), "CorrelatorConfig"),
    ],
)
def test_virtual_correlator_preflight_precedes_model_generation(
    tmp_path, monkeypatch, sections, declarations, message
):
    backend = GenerationBackend(loop_config(**sections), None)
    monkeypatch.setattr(
        backend, "_resolve_model", lambda *_: pytest.fail("reached model compilation")
    )
    with pytest.raises((GenerationError, TypeError), match=message):
        backend.generate(
            ProcessSet((ProcessRequest.parse("g g > g g"),)),
            tmp_path / "unsupported",
            correlators=declarations,
        )
    assert not (tmp_path / "unsupported").exists()


def test_virtual_full_physics_summary_retains_saved_history_labels():
    from pyamplicol.reporting.summary import render_summary

    text = render_summary(
        {
            "process_id": "uudd",
            "contribution_kind": "virtual",
            "virtual": {
                "correlations": True,
                "color_correlations": [
                    {"id": "born", "bra": [], "ket": []},
                    {"id": "T12", "bra": [{"emitter": 1}], "ket": [{"emitter": 2}]},
                ],
            },
        }
    )
    assert text is not None and "T12" in text and "born" in text
    assert "directed interference" in text
