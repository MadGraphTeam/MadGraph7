# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

import pyamplicol.generation.service as service_module
import pyamplicol.licensing as licensing_module
from pyamplicol.api import ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import (
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.licensing import SymbolicaLicenseState


class _RuntimeSchemaStub:
    def to_mapping(self) -> dict[str, object]:
        return {"stages": []}


def test_parallel_process_phase_overlaps_and_preserves_input_order() -> None:
    rendezvous = Barrier(2)
    state_lock = Lock()
    active = 0
    peak_active = 0

    def operation(item: int) -> str:
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            if item < 2:
                rendezvous.wait(timeout=2.0)
            time.sleep(0.002 * (4 - item))
            return f"result-{item}"
        finally:
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = service_module._map_process_phase(
            (0, 1, 2, 3),
            operation,
            executor=executor,
            max_in_flight=2,
            phase_name="test phase",
            item_name=lambda item: f"process-{item}",
        )

    assert peak_active == 2
    assert results == ("result-0", "result-1", "result-2", "result-3")


def test_one_generation_worker_runs_process_phase_serially() -> None:
    backend = service_module.GenerationBackend(GenerationConfig(workers=1), None)
    visited: list[int] = []

    results = service_module._map_process_phase(
        (0, 1, 2),
        lambda item: visited.append(item) or item * 2,
        executor=None,
        max_in_flight=backend._process_worker_count(3),
        phase_name="serial test phase",
        item_name=lambda item: f"process-{item}",
    )

    assert backend._process_worker_count(3) == 1
    assert visited == [0, 1, 2]
    assert results == (0, 2, 4)


def test_process_phase_stops_unscheduled_work_and_adds_process_context() -> None:
    slow_started = Event()
    started: list[str] = []
    started_lock = Lock()

    def operation(item: str) -> str:
        with started_lock:
            started.append(item)
        if item == "slow":
            slow_started.set()
            time.sleep(0.05)
            return item
        if item == "bad":
            assert slow_started.wait(timeout=1.0)
            raise RuntimeError("deliberate failure")
        raise AssertionError("unscheduled process ran")

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(
            GenerationError,
            match=("test compilation failed for process 'bad': deliberate failure"),
        ),
    ):
        service_module._map_process_phase(
            ("slow", "bad", "never"),
            operation,
            executor=executor,
            max_in_flight=2,
            phase_name="test compilation",
            item_name=str,
        )

    assert sorted(started) == ["bad", "slow"]


def test_symbolica_resources_do_not_nest_generation_worker_fanout() -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(workers=3),
            evaluator=EvaluatorConfig(
                optimization=EvaluatorOptimizationConfig(cores=4)
            ),
        ),
        None,
    )

    settings = backend._symbolica_settings()

    assert backend._process_worker_count(5) == 3
    assert settings.n_cores == 4
    assert settings.compiled_chunk_compile_workers == 1


def test_restricted_policy_is_applied_before_model_generation_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            generation=GenerationConfig(workers=4),
            evaluator=EvaluatorConfig(
                optimization=EvaluatorOptimizationConfig(cores=8)
            ),
        ),
        None,
    )
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=False, restricted=True),
    )

    def stop_at_model_loading(_source: object) -> object:
        observed.append(
            (
                backend._generation_config.workers,
                backend._symbolica_settings().n_cores,
            )
        )
        raise GenerationError("stop after restricted preflight")

    monkeypatch.setattr(backend, "_resolve_model", stop_at_model_loading)

    with pytest.raises(GenerationError, match="stop after restricted preflight"):
        backend.generate(
            ProcessSet.from_expressions(("p p > z",)),
            tmp_path / "artifact",
        )

    assert observed == [(1, 1)]
    assert backend._process_worker_count(50) == 1
    assert backend._configuration.requested.generation.workers == 4
    assert backend._configuration.effective.generation.workers == 1


def test_licensed_multiparticle_expansion_drives_workers_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cpu_budget = 12
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(entries=(ProcessEntry("p p > z"),)),
            generation=GenerationConfig(
                workers="auto",
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(
                optimization=EvaluatorOptimizationConfig(cores="auto")
            ),
        ),
        None,
    )
    monkeypatch.setattr(
        backend,
        "_detect_symbolica_license",
        lambda: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(licensing_module, "_cpu_budget", lambda: cpu_budget)
    monkeypatch.setattr(
        backend,
        "_artifact_model",
        lambda _resolved: SimpleNamespace(name="test-model"),
    )

    expansion_calls = 0
    expand_request = backend._expand_request

    def track_expansion(
        request: object,
        resolved_model: object,
    ) -> object:
        nonlocal expansion_calls
        expansion_calls += 1
        return expand_request(request, resolved_model)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "_expand_request", track_expansion)
    monkeypatch.setattr(
        backend,
        "_compile_for_generation",
        lambda entry, _model, _phase: SimpleNamespace(expanded=entry),
    )
    monkeypatch.setattr(
        backend,
        "_prepare_warmup_process",
        lambda process, _model, **_kwargs: SimpleNamespace(
            expanded=process.expanded,
            validation_points=(),
        ),
    )
    schema = _RuntimeSchemaStub()
    monkeypatch.setattr(
        backend,
        "_construct_evaluator",
        lambda process, _model, _phase: SimpleNamespace(
            compiled=process,
            runtime_schema=schema,
        ),
    )
    monkeypatch.setattr(
        backend,
        "_materialize_evaluator",
        lambda process, _model, _root, _phase: SimpleNamespace(
            process_id=process.compiled.expanded.request.name
        ),
    )

    worker_counts: list[int] = []
    real_executor = service_module.ThreadPoolExecutor

    def capture_executor(
        *,
        max_workers: int,
        thread_name_prefix: str,
    ) -> ThreadPoolExecutor:
        worker_counts.append(max_workers)
        return real_executor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    monkeypatch.setattr(service_module, "ThreadPoolExecutor", capture_executor)
    captured: dict[str, object] = {}

    def capture_artifact_configuration(
        destination: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            output=Path(destination),
            files=(),
            api_bundle_path=None,
        )

    monkeypatch.setattr(
        service_module,
        "write_schema_v3_artifact",
        capture_artifact_configuration,
    )

    result = backend.generate(
        ProcessSet.from_expressions(("p p > z",)),
        tmp_path / "artifact",
    )

    assert expansion_calls == 1
    assert len(result.processes.requests) == 5
    assert worker_counts == [5]
    settings = backend._symbolica_settings()
    assert settings.n_cores == 2
    assert worker_counts[0] * settings.n_cores <= cpu_budget

    provenance = captured["configuration"]
    requested = provenance.requested  # type: ignore[union-attr]
    effective = provenance.effective  # type: ignore[union-attr]
    adjustments = provenance.adjustments  # type: ignore[union-attr]
    assert requested.generation.workers == "auto"
    assert requested.evaluator.optimization.cores == "auto"
    assert effective.generation.workers == 5
    assert effective.evaluator.optimization.cores == 2
    assert [adjustment.reason for adjustment in adjustments] == [
        "shared affinity-aware CPU budget for concurrent process generation",
        "shared affinity-aware CPU budget for Symbolica evaluator work",
    ]
