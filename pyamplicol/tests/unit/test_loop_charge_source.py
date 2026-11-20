# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import json

import pytest

from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def charged_source(tmp_path):
    source = tmp_path / "charge_source"
    source.mkdir()
    (source / "__init__.py").write_text(
        "from . import object_library, particles\nfrom .object_library import all_particles\n"
    )
    (source / "object_library.py").write_text("""all_particles=[]
class Particle:
 def __init__(self,pdg_code,name,antiname,spin,color,mass,width,texname,antitexname,charge):
  self.pdg_code=pdg_code;self.name=name;self.antiname=antiname;self.spin=spin;self.color=color
  self.mass=mass;self.width=width;self.texname=texname;self.antitexname=antitexname;self.charge=float(charge)
  all_particles.append(self)
 def anti(self):
  return Particle(-self.pdg_code,self.antiname,self.name,self.spin,-self.color,self.mass,self.width,self.antitexname,self.texname,-self.charge)
""")
    (source / "particles.py").write_text("""from .object_library import Particle
p=Particle(pdg_code=1,name='p',antiname='p~',spin=2,color=3,mass='ZERO',width='ZERO',texname='p',antitexname='pbar',charge=2/3)
anti=p.anti()
unit=Particle(pdg_code=2,name='unit',antiname='unit~',spin=1,color=1,mass='ZERO',width='ZERO',texname='u',antitexname='ubar',charge=1)
""")
    return source


def test_direct_and_antiparticle_exact_charge_survive_roundtrip(charged_source):
    raw = load_raw_loop_model(charged_source, restriction="full")
    raw = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    positive, negative, unit = [p["fields"] for p in raw.collections["particles"]]
    assert positive["charge"] == 2 / 3
    assert positive["charge_exact"]["expression"] == "2/3"
    assert negative["charge_exact"]["expression"] == "-(2/3)"
    assert negative["charge_source"]["kind"] == "source-antiparticle-charge"
    assert unit["charge"] == 1.0 and unit["charge_exact"]["expression"] == "1"
    assert all(p["charge_source"]["exact"] for p in (positive, negative, unit))


@pytest.mark.parametrize(
    "edit", ["p.charge=0.5", "alias=p\nalias.charge=0.5", "p=unit"]
)
def test_changed_charge_or_source_binding_invalidates_anti(charged_source, edit):
    with (charged_source / "particles.py").open("a") as stream:
        stream.write(edit + "\n")
    raw = load_raw_loop_model(charged_source, restriction="full")
    assert not raw.record("particles", "p~")["fields"]["charge_source"]["exact"]


def test_unrecognized_anti_charge_rule_rejects(charged_source):
    path = charged_source / "object_library.py"
    path.write_text(
        path.read_text().replace(
            "self.texname,-self.charge)", "self.texname,-self.charge/2)"
        )
    )
    raw = load_raw_loop_model(charged_source, restriction="full")
    assert not raw.record("particles", "p~")["fields"]["charge_source"]["exact"]


def test_lossy_integer_charge_cast_is_not_certified(charged_source):
    path = charged_source / "particles.py"
    path.write_text(path.read_text().replace("charge=1)", "charge=9007199254740993)"))
    raw = load_raw_loop_model(charged_source, restriction="full")
    assert not raw.record("particles", "unit")["fields"]["charge_source"]["exact"]
