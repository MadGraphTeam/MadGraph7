# SPDX-License-Identifier: 0BSD
"""Public virtual generation preserves correction selection and fails explicitly."""

from types import SimpleNamespace

import pytest

from pyamplicol.api import GenerationError, Generator, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config
from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.generation.correlated import correlated_configuration
from pyamplicol.generation.loop_service import (
    _graph_plan,
    _order_bounds,
    _revalidate_v2_point,
    _validate_v2_points,
)
from pyamplicol.generation.service import GenerationBackend

pytestmark = pytest.mark.loop_fast


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
        ({"analyze_process_groups": "numerical"}, "numerical process-group"),
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


@pytest.mark.parametrize("accuracy", ["full", "lc", "nlc"])
def test_exact_flavor_request_attempts_proof_for_each_accuracy(accuracy):
    from pyamplicol.generation.loop_service import (
        _feature_blockers,
        _flavor_grouping_enabled,
        _flavor_grouping_outcome,
    )

    backend = GenerationBackend(
        loop_config(
            process={"loop_coupling_orders": ["QCD"], "flavor_blind": True},
            color={"accuracy": accuracy, "contraction": "direct"},
        ),
        None,
    )
    processes = ProcessSet((ProcessRequest.parse("g g > g g"),))
    assert not _feature_blockers(backend, processes)
    assert _flavor_grouping_enabled(backend, None)
    assert _flavor_grouping_enabled(
        backend, CorrelatorConfig(spin_correlations=((1,),))
    )
    outcome = _flavor_grouping_outcome(backend, None)
    assert outcome["requested"] and not outcome["reuse_applied"]
    assert outcome["transport_checked"]
    assert outcome["independent_reason"]


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("error", [NotImplementedError, ValueError])
def test_flavor_partition_only_falls_back_for_optional_capability(
    monkeypatch, correlated, error
):
    from types import SimpleNamespace

    from pyamplicol.generation.loop_service import _prepare_flavor_color_components
    from pyamplicol.loops import color_components

    bare = SimpleNamespace(
        born_color_components=None, born_color_tensors=(), _prepared_process=None
    )
    model = SimpleNamespace(color_accuracy="lc", bare=bare, _prepared_model=None)

    def unavailable(_):
        raise error("unavailable canonical tensor")

    monkeypatch.setattr(color_components, "prepare_born_color_components", unavailable)
    if correlated and error is NotImplementedError:
        _prepare_flavor_color_components(model, correlated=True)
        assert bare.born_color_components is None
    else:
        with pytest.raises(error, match="unavailable canonical tensor"):
            _prepare_flavor_color_components(model, correlated=correlated)


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
    from pyamplicol.models.loop_preparation import LoopPreparation

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
        service, "resolve_loop_preparation", lambda *a: LoopPreparation()
    )
    monkeypatch.setattr(service, "with_loop_preparation", lambda value, _: value)
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
        owner = SimpleNamespace(scope=kwargs, structure=SimpleNamespace(diagnostics=()))
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
    assert all(p.scope["fixed_parameters"] == fixed for p in context.prepared)
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


def test_projected_complete_family_request_reaches_authoritative_preparation():
    from pyamplicol.generation.loop_service import _check_correlators

    backend = GenerationBackend(loop_config(color={"accuracy": "lc"}), None)
    _check_correlators(backend, CorrelatorConfig.all_color(through_order=1))


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


@pytest.mark.parametrize("corrections", [("QED",), ("QCD", "QED")])
def test_cold_directed_completion_uses_the_authoritative_producer_scope(corrections):
    from pyamplicol.loops.physical_model_correlations import (
        require_directed_model_scope,
    )
    from pyamplicol.loops.prepared_correlations import prepare_directed_model

    model = SimpleNamespace(
        r2_strategy="direct",
        color_accuracy="full",
        color_projection=None,
        bare=SimpleNamespace(
            order_layout=SimpleNamespace(correction_names=corrections)
        ),
    )
    # The actual completion strategy is checked before prepared state or tensor
    # algebra; a correction name alone does not establish that capability.
    for operation in (
        lambda: require_directed_model_scope(model),
        lambda: prepare_directed_model(model, None),
    ):
        with pytest.raises(NotImplementedError, match="requires model R2"):
            operation()


@pytest.mark.parametrize("failure", ["coverage", "owner"])
def test_full_auto_missing_colour_capability_uses_fresh_direct_owner(
    monkeypatch, failure
):
    from pyamplicol.generation import loop_service as service
    from pyamplicol.loops.source_color import SourceColorCapabilityUnavailable
    from pyamplicol.models.loop_preparation import LoopPreparation

    backend = GenerationBackend(
        loop_config(color={"accuracy": "full", "contraction": "auto"}), None
    )
    original, continuation, direct_result = object(), object(), object()
    context = SimpleNamespace(
        resolved=SimpleNamespace(compiled=original),
        corrections=("QCD",),
        preparation=LoopPreparation(),
    )
    bounds = dict(born_order_bounds={"QCD": (0, 2)}, loop_order_bounds={"QCD": (0, 4)})
    calls = []

    def compile_formal(*args, **options):
        assert options["source_color"] is continuation
        if failure == "owner":
            raise ValueError("source colour and scalar program owners differ")
        raise SourceColorCapabilityUnavailable(
            "selected kernel has no declared continuation"
        )

    prepared = SimpleNamespace(scope=SimpleNamespace(**bounds), compile=compile_formal)

    def prepare_direct(compiled, **options):
        assert compiled is original and options == {
            **bounds,
            "fixed_parameters": None,
            "compatibility": None,
            "covariant_multiplets": None,
        }
        calls.append("fresh")

        def compile_direct(*args, **kwargs):
            assert args == ((21, 21), (21, 21))
            assert kwargs["source_color"] is None and kwargs["color_accuracy"] == "full"
            assert kwargs["selected_corrections"] == ("QCD",)
            return direct_result

        return SimpleNamespace(compile=compile_direct)

    monkeypatch.setattr(service, "prepare_loop_request_model", prepare_direct)
    entry = SimpleNamespace(
        process_ir=SimpleNamespace(initial_pdgs=(21, 21), final_pdgs=(21, 21))
    )
    if failure == "coverage":
        result, reason = service._compile_color_operation(
            backend, context, prepared, entry, continuation
        )
        assert (
            result is direct_result
            and "no declared continuation" in reason
            and calls == ["fresh"]
        )
    else:
        with pytest.raises(ValueError, match="owners differ"):
            service._compile_color_operation(
                backend, context, prepared, entry, continuation
            )
        assert calls == []


def test_loop_fft_policy_admits_only_actual_basis_and_ordinary_banks():
    from pyamplicol.generation.loop_service import _check_correlators, _feature_blockers

    requests = ProcessSet((ProcessRequest.parse("g g > g g"),))
    for contraction in ("auto", "symmetric-group-fft"):
        backend = GenerationBackend(
            loop_config(color={"accuracy": "full", "contraction": contraction}), None
        )
        assert not _feature_blockers(backend, requests)
        if contraction == "symmetric-group-fft":
            with pytest.raises(GenerationError, match="certified ordinary/resolved"):
                _check_correlators(backend, CorrelatorConfig())
    backend = GenerationBackend(
        loop_config(
            color={"accuracy": "full", "contraction": "auto", "fft_basis": "trace"}
        ),
        None,
    )
    assert any(
        "trace-basis conversion" in x for x in _feature_blockers(backend, requests)
    )


@pytest.mark.parametrize(
    "accuracy,contraction",
    [("full", "symmetric-group-fft"), ("nlc", "auto"), ("lc", "auto")],
)
def test_missing_formal_coverage_does_not_fallback_for_forced_or_projected(
    monkeypatch, accuracy, contraction
):
    from pyamplicol.generation import loop_service as service
    from pyamplicol.loops.source_color import SourceColorCapabilityUnavailable

    backend = GenerationBackend(
        loop_config(color={"accuracy": accuracy, "contraction": contraction}), None
    )

    def reject(*args, **kwargs):
        raise SourceColorCapabilityUnavailable("selected formal field is unavailable")

    monkeypatch.setattr(
        service,
        "prepare_loop_request_model",
        lambda *a, **kw: pytest.fail("unexpected fallback"),
    )
    entry = SimpleNamespace(
        process_ir=SimpleNamespace(initial_pdgs=(21, 21), final_pdgs=(21, 21))
    )
    with pytest.raises(SourceColorCapabilityUnavailable, match="formal field"):
        service._compile_color_operation(
            backend,
            SimpleNamespace(corrections=("QCD",)),
            SimpleNamespace(compile=reject),
            entry,
            object(),
        )


# ------------------------------------------------- staged v2 re-validation

BORN_ORDER, LOOP_ORDER = (("QCD", 2),), (("QCD", 4),)
# A reference virtual whose sector size (largest coefficient) is 2.
REFERENCE = (-0.4776756003084305, 1.5, -2.0)
TOLERANCE = 1e-12


def _shifted(relative):
    """REFERENCE with its finite part moved by ``relative`` of the sector."""

    return (REFERENCE[0] + 2.0 * relative, *REFERENCE[1:])


def _native_values(laurents, estimate=None):
    """A runtime evaluation of a point batch as ``_compare_v2`` reads it."""

    return SimpleNamespace(
        born_orders=(BORN_ORDER,),
        born_values=tuple((2.0,) for _ in laurents),
        sectors=(SimpleNamespace(order=LOOP_ORDER),),
        values=tuple((tuple(laurent),) for laurent in laurents),
        stability=None
        if estimate is None
        else tuple(SimpleNamespace(relative=estimate) for _ in laurents),
    )


class _Runtime:
    """Native runtime double: binary64 virtual by point, the double-double
    virtual, and the binary64 stability estimate returned under checks."""

    def __init__(self, binary64, double_double=REFERENCE, estimate=1.5e-11):
        self.binary64, self.double_double = binary64, double_double
        self.estimate, self.fail, self.calls = estimate, False, []

    def _values(self, points, arithmetic, stability_checks):
        if self.fail and arithmetic is not None:
            raise ArithmeticError("no double-double here")
        if arithmetic == "double-double":
            return _native_values([self.double_double] * len(points))
        assert arithmetic is None
        return _native_values(
            [self.binary64[point] for point in points],
            None if stability_checks is None else self.estimate,
        )

    def evaluate(self, points, *, mu_r2, arithmetic=None, stability_checks=None):
        self.calls.append(
            ("evaluate", arithmetic, stability_checks is not None, tuple(points))
        )
        return self._values(points, arithmetic, stability_checks)

    def evaluate_correlated_many(
        self, points, requests, *, mu_r2, arithmetic=None, stability_checks=None
    ):
        self.calls.append(
            ("correlated", arithmetic, stability_checks is not None, tuple(points))
        )
        return {
            history: self._values(points, arithmetic, stability_checks)
            for history in requests
        }


class _Runner:
    """Reference executor double: ``binary64`` virtual without a precision,
    REFERENCE at DOUBLE_DOUBLE_DIGITS."""

    def __init__(self, binary64=REFERENCE, labels=None):
        self.binary64, self.labels, self.calls = binary64, labels, []

    def evaluate(self, point, scale, *, correlations, precision=None):
        self.calls.append((point, precision))
        laurent = REFERENCE if precision is not None else self.binary64
        return {
            label: SimpleNamespace(
                born={BORN_ORDER: 2.0},
                components={
                    "virtual": {
                        LOOP_ORDER: SimpleNamespace(coefficients=tuple(laurent))
                    }
                },
            )
            for label in (correlations if self.labels is None else self.labels)
        }


def _revalidate(runtime, runner=None, point="p", tolerance=TOLERANCE):
    """Re-validate ``point`` of ``runtime`` after a binary64 miss."""

    miss = GenerationError("binary64 pair 1.3e-12 apart")
    binary64 = {
        label: _native_values([runtime.binary64[point]])
        for label in ("born", "casimir")
    }
    _revalidate_v2_point(
        "ddgg",
        runtime,
        _Runner() if runner is None else runner,
        ("casimir",),
        binary64,
        0,
        point,
        "1",
        tolerance,
        miss,
    )


def test_a_binary64_miss_within_its_estimate_is_revalidated_in_both_carriers(
    caplog,
):
    """A point whose binary64 pair misses the tolerance passes when native
    double-double equals the reference at DOUBLE_DOUBLE_DIGITS to the same
    tolerance and the binary64 virtual is within its own estimate (d d~ > g g
    NLC casimir: 2.7e-12 of the sector off, estimate 1.5e-11), every output of
    the point; the accepted miss is logged with its deviations."""

    from pyamplicol.loops.v2.assemble import DOUBLE_DOUBLE_DIGITS, Precision

    runtime, runner = _Runtime({"p": _shifted(2.7e-12)}), _Runner()
    with caplog.at_level("WARNING", logger="pyamplicol.generation"):
        _revalidate(runtime, runner)
    assert runtime.calls == [
        ("evaluate", "double-double", False, ("p",)),
        ("correlated", "double-double", False, ("p",)),
        ("evaluate", None, True, ("p",)),
        ("correlated", None, True, ("p",)),
    ]
    assert runner.calls == [("p", Precision.arbitrary(DOUBLE_DOUBLE_DIGITS))]
    (message,) = [r.getMessage() for r in caplog.records]
    assert "binary64 pair 1.3e-12 apart" in message
    assert "casimir: binary64 2.7e-12 (estimate 1.5e-11), double-double 0" in message


@pytest.mark.parametrize(
    ("binary64", "estimate"),
    [
        # A binary64-only defect (the audit's injection), relative and x2.
        (tuple(c * (1 + 1e-6) for c in REFERENCE), 1.5e-11),
        (tuple(2 * c for c in REFERENCE), 1.5e-11),
        # Beyond its own estimate, which exceeds the tolerance.
        (_shifted(3e-12), 2e-12),
        # Off by more than the tolerance at a well-conditioned point.
        (_shifted(3e-12), 5e-13),
    ],
)
def test_a_binary64_only_defect_is_refused(binary64, estimate):
    """Wrong binary64, right double-double: refused, the binary64 virtual
    being held to max(tolerance, its estimate) against 32 digits."""

    with pytest.raises(GenerationError, match="in binary64 against 32 digits"):
        _revalidate(_Runtime({"p": binary64}, estimate=estimate))


@pytest.mark.parametrize("estimate", [None, 2e-9, float("nan")])
def test_a_binary64_miss_without_an_admissible_estimate_is_refused(estimate):
    """No estimate, or one above the user's 1e-9 cap: refused, even with the
    binary64 value within it."""

    with pytest.raises(GenerationError, match="no admissible stability estimate"):
        _revalidate(_Runtime({"p": _shifted(2.7e-12)}, estimate=estimate))


def test_a_looser_tolerance_raises_the_estimate_cap_to_itself():
    """The cap is max(tolerance, 1e-9): an estimate of 2e-9 is admissible
    under a configured 1e-8, and the virtual is then held to the tolerance."""

    _revalidate(_Runtime({"p": _shifted(5e-9)}, estimate=2e-9), tolerance=1e-8)
    with pytest.raises(GenerationError, match="in binary64 against 32 digits"):
        _revalidate(_Runtime({"p": _shifted(2e-8)}, estimate=2e-9), tolerance=1e-8)


def test_a_double_double_mismatch_is_still_refused_at_the_tolerance():
    """The re-validation keeps the tolerance: 4e-12 relative in double-double
    is refused, the binary64 refusal chained; so is one that cannot run."""

    runtime = _Runtime({"p": _shifted(2.7e-12)}, double_double=_shifted(4e-12))
    with pytest.raises(GenerationError, match="in double-double, after binary64"):
        _revalidate(runtime)
    runtime = _Runtime({"p": _shifted(2.7e-12)})
    runtime.fail = True
    with pytest.raises(GenerationError, match="could not run") as refused:
        _revalidate(runtime)
    assert "binary64 pair 1.3e-12 apart" in str(refused.value)


def _points(runtime, runner):
    _validate_v2_points(
        "ddgg", runtime, runner, ("casimir",), ["p0", "p1"], ["1", "1"], TOLERANCE
    )


def test_staged_validation_rechecks_only_the_missing_point():
    """Binary64 passes: one batch per output, no re-validation. A binary64
    miss at the second point re-validates that point alone; the first is not
    re-evaluated."""

    from pyamplicol.loops.v2.assemble import DOUBLE_DOUBLE_DIGITS, Precision

    batch = [
        ("evaluate", None, False, ("p0", "p1")),
        ("correlated", None, False, ("p0", "p1")),
    ]
    runtime, runner = _Runtime({"p0": REFERENCE, "p1": REFERENCE}), _Runner()
    _points(runtime, runner)
    assert runtime.calls == batch
    assert runner.calls == [("p0", None), ("p1", None)]

    runtime = _Runtime({"p0": REFERENCE, "p1": _shifted(2.7e-12)})
    runner = _Runner()
    _points(runtime, runner)
    assert runtime.calls == [
        *batch,
        ("evaluate", "double-double", False, ("p1",)),
        ("correlated", "double-double", False, ("p1",)),
        ("evaluate", None, True, ("p1",)),
        ("correlated", None, True, ("p1",)),
    ]
    assert runner.calls == [
        ("p0", None),
        ("p1", None),
        ("p1", Precision.arbitrary(DOUBLE_DOUBLE_DIGITS)),
    ]
    runtime = _Runtime({"p0": REFERENCE, "p1": tuple(2 * c for c in REFERENCE)})
    with pytest.raises(GenerationError, match="in binary64 against 32 digits"):
        _points(runtime, _Runner())


def test_staged_validation_catches_only_comparison_refusals():
    """Only ``_compare_v2``'s GenerationError starts a re-validation: a
    reference without a declared output fails as such."""

    runtime = _Runtime({"p0": REFERENCE, "p1": REFERENCE})
    with pytest.raises(KeyError, match="casimir"):
        _points(runtime, _Runner(labels=("born",)))
    assert len(runtime.calls) == 2
