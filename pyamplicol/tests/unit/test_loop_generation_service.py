# SPDX-License-Identifier: 0BSD
"""Public virtual generation preserves correction selection and fails explicitly."""

from types import SimpleNamespace

import pytest

from pyamplicol.api import GenerationError, Generator, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config
from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.generation.correlated import correlated_configuration
from pyamplicol.generation.loop_service import (
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
        "HV", "4d", "source", "source", "real-on-shell", "source", "fixture"
    )
    loop = LoopModelDeclaration(scheme, ("QCD", "QED"), True, True)
    backend = GenerationBackend(
        loop_config(color={"accuracy": accuracy, "contraction": "direct"}), None
    )
    # A compiled model declares its coupling orders (``loops.order_names``
    # refuses a model whose spelling it cannot read).
    compiled = SimpleNamespace(
        raw_loop_model=object(),
        ir=SimpleNamespace(
            orders=tuple(SimpleNamespace(name=n) for n in loop.loop_coupling_orders)
        ),
    )
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
        "pyamplicol.loops.v2.assemble.prepare_request_model",
        lambda *a, **kw: pytest.fail("mismatched scheme reached source preparation"),
    )
    plan = backend.plan(ProcessSet((ProcessRequest.parse("g g > g g"),)))
    (feature,) = plan.unsupported_features
    assert feature.startswith(
        "colour-declaration-coverage: source colour continuation and effective "
        "loop scheme differ"
    ), feature


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


def test_loop_fft_policy_admits_only_actual_basis_and_ordinary_banks():
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.generation.loop_service import _check_correlators, _feature_blockers
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    requests = ProcessSet((ProcessRequest.parse("g g > g g"),))
    for contraction in ("auto", "symmetric-group-fft"):
        backend = GenerationBackend(
            loop_config(color={"accuracy": "full", "contraction": contraction}), None
        )
        assert not _feature_blockers(backend, requests)
        # No declared observable: the forced seed FFT serves the inclusive
        # virtual; a declared one is v2's listed refusal, before preparation.
        _check_correlators(backend, CorrelatorConfig())
        for declared in (
            CorrelatorConfig(color_correlations=(ColorCorrelator.dipole("T", 1, 2),)),
            CorrelatorConfig(spin_correlations=((1,),)),
            CorrelatorConfig.all_color(through_order=1),
        ):
            if contraction == "symmetric-group-fft":
                with pytest.raises(GenerationError) as refused:
                    _check_correlators(backend, declared)
                assert isinstance(refused.value, LoopRequestUnsupported)
                assert refused.value.key == "forced-seed-fft-unserved"
            else:
                _check_correlators(backend, declared)
    backend = GenerationBackend(
        loop_config(
            color={"accuracy": "full", "contraction": "auto", "fft_basis": "trace"}
        ),
        None,
    )
    assert any(
        "color.fft_basis = 'trace' is unavailable" in x
        for x in _feature_blockers(backend, requests)
    )


@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_public_preflight_defers_projected_history_proof_to_actual_owner(accuracy):
    from pyamplicol.color.correlator_matrices import ColorCorrelator
    from pyamplicol.generation.loop_service import _check_correlators

    backend = SimpleNamespace(
        _run_config=SimpleNamespace(
            color=SimpleNamespace(accuracy=accuracy, contraction="direct")
        ),
        # Inclusive helicity and colour selectors (``_check_correlators``
        # refuses correlators with a leading-colour selection).
        _process_selection=SimpleNamespace(
            max_color_sectors=None,
            reference_color_order=None,
            selected_color_sector_ids=None,
            selected_source_helicities=None,
        ),
    )
    _check_correlators(backend, CorrelatorConfig(spin_correlations=((3,), (3, 4))))
    _check_correlators(
        backend, CorrelatorConfig(color_correlations=(ColorCorrelator("identity"),))
    )
    _check_correlators(
        backend,
        CorrelatorConfig(color_correlations=(ColorCorrelator.dipole("T", 1, 2),)),
    )
    _check_correlators(backend, CorrelatorConfig(all_color_through_order=1))
    with pytest.raises(TypeError, match="CorrelatorConfig"):
        _check_correlators(backend, object())


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
