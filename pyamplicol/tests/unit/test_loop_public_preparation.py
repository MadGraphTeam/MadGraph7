# SPDX-License-Identifier: 0BSD
"""Public loop-model resolution of the real Gmu source keeps shared exclusions.

This is resolution-boundary coverage, not a successful public generate/dry-run
or another numerical EW reference campaign.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import ModelError
from pyamplicol.api.models import _compiled_model_from_payload
from pyamplicol.api.requests import ModelSource
from pyamplicol.config import resolve_config
from pyamplicol.generation.service import GenerationBackend

pytestmark = pytest.mark.loop_physics


@pytest.fixture(scope="module")
def gmu_source(loop_gmu_ufo, tmp_path_factory):
    # Compile the full Gmu source once; this populates the model cache that the
    # public planning path below must read instead of compiling.
    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.loop_declaration import (
        LoopModelDeclaration,
        LoopSchemeDeclaration,
    )

    cache = tmp_path_factory.mktemp("ew-preparation") / "model-cache"
    paths = (None, loop_gmu_ufo, cache)
    compiled = compile_model_source(
        loop_gmu_ufo, restriction="full", cache_dir=cache, require_supported=False
    )
    assert compiled.raw_loop_model is not None
    # The NLO UFO scalar helpers (arg, crecms, grreglog, recms, regsqrt) are
    # registered: the tree model of the G_mu source is supported (MS-MODEL).
    assert compiled.supported
    assert compiled.issues == ()
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
def test_default_tree_resolution_accepts_the_gmu_source(gmu_source, planning):
    backend = GenerationBackend(
        resolve_config(_configuration(gmu_source, tree=True)), None
    )
    resolver = backend._resolve_model_for_plan if planning else backend._resolve_model
    resolved = resolver(_compiled_model_from_payload(gmu_source.compiled))
    assert resolved.compiled.supported
