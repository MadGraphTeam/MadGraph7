# SPDX-License-Identifier: 0BSD
"""Consume the upstream loader's source records through its ordinary JSON API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import pyamplicol
from pyamplicol.models.loading import ModelCompileOptions, _load_external_model
from pyamplicol.models.loop_declaration import resolve_loop_model_declaration
from pyamplicol.models.loop_import import (
    load_raw_loop_model,
    raw_loop_model_from_dict,
)


@pytest.fixture
def nlo_source(tmp_path: Path) -> Path:
    packaged = Path(pyamplicol.__file__).parent / "assets/models/ufo/scalars"
    source = tmp_path / "source_owned_nlo"
    shutil.copytree(packaged, source, ignore=shutil.ignore_patterns("__pycache__"))
    with (source / "object_library.py").open("a") as stream:
        stream.write("""
all_CTparameters = []
all_CTvertices = []

class CTParameter:
    def __init__(self, name, type, value):
        self.name, self.type, self.value = name, type, value
        self.nature = 'internal'
        all_CTparameters.append(self)

class CTVertex:
    def __init__(self, name, particles, color, lorentz, couplings,
                 loop_particles, type):
        self.name, self.particles, self.color = name, particles, color
        self.lorentz, self.couplings = lorentz, couplings
        self.loop_particles, self.type = loop_particles, type
        all_CTvertices.append(self)
""")
    (source / "ct_data.py").write_text("""
from .object_library import CTParameter, CTVertex, Coupling, Parameter
from .object_library import all_CTparameters, all_CTvertices
from . import vertices, particles, coupling_orders

dZ = CTParameter(name='dZ', type='real', value={-1: 'lam/3', 0: 'lam/5'})
UV = Coupling(name='UV', value='lam*dZ', order={'QCD': 2})
R2 = Coupling(name='R2', value={0: 'lam**2/7'}, order={'QCD': 2})
EXACT = Parameter(name='EXACT', nature='external', type='real',
    value=0.12345678901234567890123456789012345678901234567890123456789,
    texname='x', lhablock='EXACT', lhacode=[1])
v = vertices.all_vertices[0]
p = particles.all_particles[0]
uv = CTVertex(name='duplicate', particles=v.particles, color=v.color,
    lorentz=v.lorentz, couplings={(0,0,0): UV, (0,0,1): R2},
    loop_particles=[[[p], [p, p]], [[]]], type='UV')
r2 = CTVertex(name='duplicate', particles=v.particles, color=v.color,
    lorentz=v.lorentz, couplings={(0,0,0): R2},
    loop_particles=[[[p]]], type='R2')
p.counterterm = {(1,0): {-1: 'dZ'}}
p.loop_particles = [[p]]
coupling_orders.QCD.perturbative_expansion = 1
""")
    declaration = {
        "scheme": {
            "dimensional_regularization": "declared test spin prescription",
            "external_states": "four dimensional",
            "gamma5": "no gamma five in fixture",
            "renormalization": "declared test UV",
            "mass_scheme": "real masses",
            "finite_restoration": "none in fixture",
            "model_convention": "synthetic import fixture, not physics certified",
        },
        "loop_coupling_orders": ["QCD"],
        "uv_complete": True,
        "model_r2_complete": True,
    }
    with (source / "__init__.py").open("a") as stream:
        stream.write(
            "\nfrom .ct_data import all_CTparameters, all_CTvertices\n"
            f"pyamplicol_loop = {declaration!r}\n"
        )
    (source / "restrict_default.dat").write_text(
        "BLOCK EXACT\n  1 0.987654321098765432109876543210987654321\n"
    )
    return source


def test_upstream_model_json_preserves_source_once_for_loop_consumer(
    nlo_source: Path, tmp_path: Path, monkeypatch
) -> None:
    from ufo_model_loader.commands import load_model
    from ufo_model_loader.model import Model
    from ufo_model_loader.raw import RawUFOModel

    model, _card = load_model(str(nlo_source), None, True)
    payload = json.loads(model.to_json())
    assert payload["raw_ufo"]["kind"] == "ufo-model-source"
    restored = Model.from_json(json.dumps(payload))
    assert restored.raw_ufo == model.raw_ufo
    assert restored.get_parameter("EXACT").value.real == pytest.approx(0.987654321)
    raw = restored.raw_ufo
    assert isinstance(raw, RawUFOModel)
    assert raw.record("parameters", "EXACT")["fields"]["value"]["expression"] == (
        "0.12345678901234567890123456789012345678901234567890123456789"
    )
    assert raw.metadata["restriction"]["applied"] is False
    assert (
        "0.987654321098765432109876543210987654321"
        in raw.metadata["restriction"]["card"]
    )

    serialized = tmp_path / "model.json"
    serialized.write_text(model.to_json())

    def no_source_import(*_args, **_kwargs):
        pytest.fail("JSON ingress must not import the original UFO again")

    monkeypatch.setattr(Model, "from_ufo_model", no_source_import)
    worker = _load_external_model(
        serialized, options=ModelCompileOptions(restriction="none")
    )
    assert "raw_ufo" not in worker["model"]
    assert "raw_ufo_error" not in worker["model"]
    consumer = raw_loop_model_from_dict(worker["raw_loop_model"])
    assert isinstance(consumer, RawUFOModel)
    assert consumer.collections == raw.collections
    assert consumer.exact_value("counterterm_parameters", 0)["terms"] == (
        (-1, "lam/3"),
        (0, "lam/5"),
    )
    assert consumer.record("couplings", "UV")["fields"]["value"] == {
        "kind": "expression",
        "expression": "lam*dZ",
    }
    ct = consumer.collections["counterterm_vertices"]
    assert [row["name"] for row in ct] == ["duplicate", "duplicate"]
    assert ct[0]["fields"]["couplings"]["entries"][1][0]["items"] == (0, 0, 1)
    groups = ct[0]["fields"]["loop_particles"]
    assert [len(alternative) for alternative in groups[0]] == [1, 2]
    assert groups[1] == ((),)
    assert "counterterm" in consumer.collections["particles"][0]["fields"]
    assert consumer.capabilities["perturbative_orders"] == ("QCD",)
    assert (
        resolve_loop_model_declaration(consumer, selected_corrections=("QCD",)).status
        == "declared-not-certified"
    )


def test_legacy_tree_json_without_source_remains_usable(
    nlo_source: Path, tmp_path: Path
) -> None:
    from ufo_model_loader.commands import load_model

    model, _ = load_model(str(nlo_source), "none", False)
    payload = json.loads(model.to_json())
    del payload["raw_ufo"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    worker = _load_external_model(path, options=ModelCompileOptions(restriction="none"))
    assert worker["model"]["vertex_rules"]
    assert "raw_loop_model" not in worker


def test_source_compile_does_not_evaluate_counterterm_only_conditionals(
    nlo_source: Path,
) -> None:
    path = nlo_source / "ct_data.py"
    path.write_text(path.read_text().replace("0: 'lam/5'", "0: 'lam if lam else 0'"))
    worker = _load_external_model(nlo_source, options=ModelCompileOptions())
    assert all(p["name"] != "dZ" for p in worker["model"]["parameters"])
    assert all(c["name"] != "UV" for c in worker["model"]["couplings"])
    raw = raw_loop_model_from_dict(worker["raw_loop_model"])
    assert raw.exact_value("counterterm_parameters", 0)["terms"] == (
        (-1, "lam/3"),
        (0, "lam if lam else 0"),
    )


def test_raw_json_ingress_never_requires_finite_tree_normalization(
    nlo_source: Path, tmp_path: Path, monkeypatch
) -> None:
    from ufo_model_loader.model import Model

    raw = load_raw_loop_model(nlo_source)
    path = tmp_path / "source-only.json"
    path.write_text(raw.to_json())

    def no_normalization(*_args, **_kwargs):
        pytest.fail("raw source JSON must not normalize or reimport a tree model")

    monkeypatch.setattr(Model, "from_ufo_model", no_normalization)
    monkeypatch.setattr(Model, "from_json", no_normalization)
    assert load_raw_loop_model(path) == raw
    with pytest.raises(ValueError, match="restriction"):
        load_raw_loop_model(path, restriction="none")
