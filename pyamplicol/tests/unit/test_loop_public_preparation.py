# SPDX-License-Identifier: 0BSD
"""Public source preparation reuses the real Gmu cache without process graphs.

This is preparation-boundary coverage, not a successful public generate/dry-run
or another numerical EW reference campaign.
"""

import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import ModelError, ProcessRequest, ProcessSet
from pyamplicol.api.models import _compiled_model_from_payload
from pyamplicol.api.requests import ModelSource
from pyamplicol.config import config_to_toml, resolve_config
from pyamplicol.generation.service import GenerationBackend


@pytest.fixture(scope="module")
def gmu_source():
    paths = tuple(
        os.environ.get(name)
        for name in (
            "PYAMPLICOL_EW_PREPARATION_MODEL",
            "PYAMPLICOL_EW_UFO",
            "PYAMPLICOL_EW_PREPARATION_CACHE",
        )
    )
    if not all(paths):
        pytest.skip(
            "existing full-Gmu compiled source and populated model cache required"
        )
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_declaration import (
        LoopModelDeclaration,
        LoopSchemeDeclaration,
    )

    compiled = load_compiled_model(Path(paths[0]))
    assert compiled.raw_loop_model is not None
    assert not compiled.supported
    assert tuple(issue.code for issue in compiled.issues) == ("unknown-functions",)
    reference = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/real_mass_ew_quarks.json"
    )
    fixture = json.loads(reference.read_text())
    scheme = LoopSchemeDeclaration(
        "HV; source lhv=1, epsUV=epsIR=1",
        "physical-four-dimensional",
        "source-UFO-four-dimensional-prescription",
        "source Gmu; GfRedefinitionChoice=1; "
        "source UV vertices include field renormalization",
        "real-on-shell; all widths zero; diagonal CKM",
        "source-UFO-local-rules; no additional finite restoration",
        "explicit source compatibility and covariant partners; "
        "real_mass_ew_quarks.json",
    )
    return SimpleNamespace(
        compiled=compiled,
        source=Path(paths[1]),
        cache=Path(paths[2]),
        fixture=fixture,
        declaration=LoopModelDeclaration(scheme, ("QED",), True, True),
    )


def _configuration(gmu, *, preparation=None, supplied=True, tree=False):
    declaration = gmu.declaration.to_dict()
    declaration.pop("status")
    data = {
        "action": "generate",
        "model": {"cache_dir": str(gmu.cache)},
        "process": {
            "coupling_order_policy": "explicit",
            "max_coupling_orders": {"QCD": 0, "QED": 2},
        },
        "color": {"accuracy": "full", "contraction": "direct"},
        "generation": {"emit_api_bundle": False},
    }
    if not tree:
        data["process"]["loop_coupling_orders"] = ["QED"]
        data["loop"] = {"model": declaration}
        if supplied:
            data["loop"]["preparation"] = preparation
    return data


def _options(gmu):
    # Preserve the accepted exact card values, with an extra precision-sensitive
    # real fixed input. No numerical scalar owner is bound by this test.
    return {
        "compatibility": "gmu_particle_anti",
        "covariant_multiplets": gmu.fixture["covariant_multiplets"],
        "fixed_parameters": {
            **gmu.fixture["fixed_parameters"],
            "Gf": ["0.0000116639000000000000000000000000000000000000000001", "0"],
        },
    }


def _forbid_process_work(monkeypatch):
    from pyamplicol import _rusticol
    from pyamplicol.generation import loop_service
    from pyamplicol.loops import compiler, graphs, physical_source
    from pyamplicol.loops.counterterms import PreparedCountertermModel

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "model preparation entered graph/process/numerical construction"
        )

    for owner, name in (
        (loop_service, "_graph_plan"),
        (loop_service, "build_loop_process"),
        (graphs, "build_loop_process"),
        (compiler.PreparedLoopRequestModel, "compile"),
        (PreparedCountertermModel, "process"),
        (compiler, "source_model_program"),
        (physical_source, "source_model_program"),
        (_rusticol._loop, "NativeLoopProgram"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def _requests():
    return ProcessSet((ProcessRequest.parse("e- e+ > u u~"),))


@pytest.mark.parametrize("planning", (True, False))
def test_actual_public_gmu_preparation_owns_exact_source_and_counterterms(
    gmu_source, monkeypatch, planning
):
    from pyamplicol.models.loop_preparation import (
        LoopPreparation,
        with_loop_preparation,
    )

    from pyamplicol.generation.loop_service import _prepare

    gmu = gmu_source
    _forbid_process_work(monkeypatch)
    original_raw = gmu.compiled.raw_loop_model.to_dict()
    options = _options(gmu)
    if planning:
        # The actual public planning path reads the already populated UFO cache.
        source = ModelSource.from_path(gmu.source, restriction="full")
        config = resolve_config(_configuration(gmu, preparation=options))
    else:
        attached = with_loop_preparation(gmu.compiled, LoopPreparation(**options))
        source = _compiled_model_from_payload(attached)
        config = resolve_config(_configuration(gmu, supplied=False))
        assert config.effective.loop.preparation is None
        text = config_to_toml(config.effective)
        assert "[loop.preparation]" not in text
        config = resolve_config(tomllib.loads(text))
    backend = GenerationBackend(config, None)
    context = _prepare(backend, _requests(), source, planning=planning, phase=None)
    prepared = context.prepared[0]
    assert len(context.expanded) == len(context.prepared) == 1
    assert context.expanded[0].process_ir.initial_pdgs == (11, -11)
    assert context.expanded[0].process_ir.final_pdgs == (2, -2)
    assert prepared.compiled_model is context.resolved.compiled
    assert prepared.original is context.resolved.compiled.raw_loop_model
    assert prepared.structure.raw is prepared.declarations.derived
    assert prepared.model.raw is prepared.structure.raw
    assert prepared.counterterms.source is prepared.model
    assert prepared.counterterms.ordinary_model is prepared.structure.native
    assert not context.resolved.compiled.supported
    assert context.resolved.compiled.issues == gmu.compiled.issues
    assert prepared.scope.born_order_bounds == {"QCD": (0, 0), "QED": (0, 2)}
    assert prepared.scope.loop_order_bounds == {"QCD": (0, 0), "QED": (0, 4)}
    prepared.scope.require_local_counterterms(prepared.counterterms)
    assert prepared.counterterms.rules and not prepared.counterterms.unavailable_entries
    assert all(
        rule.has_validated_tensor_basis for rule in prepared.counterterms.rules.values()
    )
    payload = json.loads(prepared.structure.native.to_json())
    assert payload["covariant_cut_multiplets"] == options["covariant_multiplets"]
    assert prepared.structure.diagnostics[0]["field"] == "goldstoneboson"
    assert (
        prepared.structure.raw.record("particles", "G-")["fields"]["goldstoneboson"]
        is False
    )
    assert (
        next(p for p in payload["particles"] if p["name"] == "G-")["goldstoneboson"]
        is True
    )
    for name, value in options["fixed_parameters"].items():
        expected = (value, "0") if isinstance(value, str) else tuple(value)
        assert tuple(prepared.structure.fixed_parameters[name]) == expected
    assert context.preparation.compatibility == "gmu_particle_anti"
    assert context.source_colors == (None,)
    assert gmu.compiled.raw_loop_model.to_dict() == original_raw


@pytest.mark.parametrize("mode", ("empty-toml", "partial-set"))
def test_explicit_preparation_replaces_source_record_at_actual_boundary(
    gmu_source, monkeypatch, mode
):
    from pyamplicol.models.loop_preparation import (
        LoopPreparation,
        with_loop_preparation,
    )

    from pyamplicol.generation.loop_service import _prepare

    gmu = gmu_source
    _forbid_process_work(monkeypatch)
    source = _compiled_model_from_payload(
        with_loop_preparation(gmu.compiled, LoopPreparation(**_options(gmu)))
    )
    if mode == "empty-toml":
        base = resolve_config(_configuration(gmu, preparation={}))
        text = config_to_toml(base.effective)
        assert "[loop.preparation]" in text
        config = resolve_config(tomllib.loads(text))
    else:
        config = resolve_config(
            _configuration(gmu, supplied=False),
            overrides=('loop.preparation.fixed_parameters.Gf="0.0000116639"',),
        )
    assert config.effective.loop.preparation is not None
    assert config.effective.loop.preparation.compatibility is None
    backend = GenerationBackend(config, None)
    # Missing repair must fail at the real particle-pair authority, rather than
    # silently retaining the sidecar compatibility profile or partner data.
    with pytest.raises(ValueError, match="goldstoneboson"):
        _prepare(backend, _requests(), source, planning=True, phase=None)


def test_active_unsupported_source_function_still_rejects(gmu_source, monkeypatch):
    from pyamplicol.generation.loop_service import _prepare
    from pyamplicol.models.loop_import import RawLoopModel

    gmu = gmu_source
    _forbid_process_work(monkeypatch)
    payload = gmu.compiled.raw_loop_model.to_dict()
    alpha = next(p for p in payload["collections"]["parameters"] if p["name"] == "aEW")
    assert alpha["fields"]["nature"] == "internal"
    alpha["fields"]["value"] = {
        "kind": "expression",
        "expression": "unsupported_active_source_helper(Gf)",
    }
    compiled = replace(
        gmu.compiled,
        raw_loop_model=RawLoopModel.from_dict(payload),
        _serialized_path=None,
    )
    backend = GenerationBackend(
        resolve_config(_configuration(gmu, preparation=_options(gmu))), None
    )
    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"unsupported_active_source_helper|unknown.*function|unsupported.*function",
    ):
        _prepare(
            backend,
            _requests(),
            _compiled_model_from_payload(compiled),
            planning=True,
            phase=None,
        )


@pytest.mark.parametrize("planning", (True, False))
@pytest.mark.parametrize(
    "kind,issue",
    (
        ("handle", "majorana-fermion"),
        ("compiled", "form-factors"),
        ("prepared", "unsupported-color-representation"),
        ("cached", "unsupported-spin"),
    ),
)
def test_loop_model_resolution_retains_shared_exclusions(
    gmu_source, monkeypatch, tmp_path, planning, kind, issue
):
    from pyamplicol.models import loading

    gmu = gmu_source
    problem = loading.ModelCompatibilityIssue(
        "error", issue, "shared physical exclusion"
    )
    compiled = replace(gmu.compiled, issues=(*gmu.compiled.issues, problem))
    config = resolve_config(_configuration(gmu, preparation=_options(gmu)))
    backend = GenerationBackend(config, None)
    if kind == "handle":
        source = _compiled_model_from_payload(compiled)
    elif kind in {"compiled", "prepared"}:
        source = ModelSource(kind=kind, path=tmp_path / "model.json")
        monkeypatch.setattr(loading, "load_compiled_model", lambda *_: compiled)
    else:
        source = ModelSource.from_path(gmu.source, restriction="full")
        name = "load_cached_model_source" if planning else "compile_model_source"

        def load(*args, **kwargs):
            assert kwargs["require_supported"] is False
            return compiled

        monkeypatch.setattr(loading, name, load)
    resolver = backend._resolve_model_for_plan if planning else backend._resolve_model
    with pytest.raises(ModelError, match=issue):
        resolver(source)


@pytest.mark.parametrize("planning", (True, False))
def test_default_tree_resolution_still_rejects_actual_gmu_registry_issues(
    gmu_source, planning
):
    backend = GenerationBackend(
        resolve_config(_configuration(gmu_source, tree=True)), None
    )
    resolver = backend._resolve_model_for_plan if planning else backend._resolve_model
    with pytest.raises(ModelError, match="unknown-functions"):
        resolver(_compiled_model_from_payload(gmu_source.compiled))


def test_full_auto_fallback_reconstructs_actual_base_source_preparation(
    gmu_source, monkeypatch
):
    from pyamplicol.models.loop_preparation import LoopPreparation

    from pyamplicol.generation.loop_service import _compile_color_operation
    from pyamplicol.loops.compiler import PreparedLoopRequestModel
    from pyamplicol.loops.source_color import SourceColorCapabilityUnavailable

    gmu = gmu_source
    _forbid_process_work(monkeypatch)
    options = _options(gmu)
    config = _configuration(gmu, preparation=options)
    config["color"]["contraction"] = "auto"
    backend = GenerationBackend(resolve_config(config), None)
    base = LoopPreparation(**options)
    context = SimpleNamespace(
        resolved=SimpleNamespace(compiled=gmu.compiled),
        corrections=("QED",),
        preparation=base,
    )
    bounds = dict(
        born_order_bounds={"QCD": (0, 0), "QED": (0, 2)},
        loop_order_bounds={"QCD": (0, 0), "QED": (0, 4)},
    )
    continuation = object()

    def missing(*args, **kwargs):
        assert kwargs["source_color"] is continuation
        raise SourceColorCapabilityUnavailable("explicit test continuation unavailable")

    failed = SimpleNamespace(scope=SimpleNamespace(**bounds), compile=missing)

    # Only the final process construction is replaced. The fallback must run
    # the real structural/scalar/CT preparation from the original source.
    def return_prepared(owner, incoming, outgoing, **kwargs):
        assert incoming == (11, -11) and outgoing == (2, -2)
        assert kwargs["selected_corrections"] == ("QED",)
        assert kwargs["source_color"] is None
        return owner

    monkeypatch.setattr(PreparedLoopRequestModel, "compile", return_prepared)
    entry = SimpleNamespace(
        process_ir=SimpleNamespace(initial_pdgs=(11, -11), final_pdgs=(2, -2))
    )
    direct, reason = _compile_color_operation(
        backend, context, failed, entry, continuation
    )
    assert reason == "explicit test continuation unavailable"
    assert direct.compiled_model is gmu.compiled
    assert direct.original is gmu.compiled.raw_loop_model
    assert direct.structure.diagnostics[0]["field"] == "goldstoneboson"
    assert (
        json.loads(direct.structure.native.to_json())["covariant_cut_multiplets"]
        == options["covariant_multiplets"]
    )
    assert direct.scope.born_order_bounds == bounds["born_order_bounds"]
    assert direct.scope.loop_order_bounds == bounds["loop_order_bounds"]
    direct.scope.require_local_counterterms(direct.counterterms)
    for name, value in options["fixed_parameters"].items():
        expected = (value, "0") if isinstance(value, str) else tuple(value)
        assert tuple(direct.structure.fixed_parameters[name]) == expected
