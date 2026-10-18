# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pyamplicol.loops.covariant_model import build_covariant_structure
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "covariant_fixture"
    path.mkdir()
    (path / "__init__.py").write_text(
        "from . import object_library, parameters, particles\nfrom .object_library import all_parameters, all_particles\ngauge=[1]\n"
    )
    (path / "object_library.py").write_text("""all_parameters=[]
all_particles=[]
class Parameter:
 def __init__(self,name,nature,type,value):
  self.name=name;self.nature=nature;self.type=type;self.value=value;all_parameters.append(self)
class Particle:
 def __init__(self,pdg_code,name,antiname,spin,color,mass,width,texname,antitexname,charge):
  self.pdg_code=pdg_code;self.name=name;self.antiname=antiname;self.spin=spin;self.color=color
  self.mass=mass;self.width=width;self.texname=texname;self.antitexname=antitexname;self.charge=charge;all_particles.append(self)
""")
    (path / "parameters.py").write_text("""from .object_library import Parameter
ZERO=Parameter(name='ZERO',nature='internal',type='real',value='0')
M=Parameter(name='M',nature='external',type='real',value=0.0)
Zalias=Parameter(name='Zalias',nature='internal',type='real',value='ZERO+0')
""")
    (path / "particles.py").write_text("""from .object_library import Particle
from . import parameters as P
s=Particle(pdg_code=100,name='s',antiname='s',spin=1,color=1,mass=P.M,width=P.ZERO,texname='s',antitexname='s',charge=0)
t=Particle(pdg_code=101,name='t',antiname='t',spin=1,color=1,mass=P.Zalias,width=P.ZERO,texname='t',antitexname='t',charge=0)
v=Particle(pdg_code=102,name='v',antiname='v',spin=3,color=1,mass=P.M,width=P.ZERO,texname='v',antitexname='v',charge=0)
""")
    return path


def test_native_mass_classification_preserves_exact_zero_aliases(source):
    from symbolica import E, S
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor

    raw = load_raw_loop_model(source, restriction="full")
    result = build_covariant_structure(raw)
    assert result.required_bindings == ("M",)
    assert result.native.particle("s").mass == S("UFO::M")
    assert not result.native.particle("s").is_massless
    assert result.native.particle("t").is_massless
    payload = json.loads(result.native.to_json())
    assert all(parameter["value"] is None for parameter in payload["parameters"])
    # Public covariant projector agrees for massive and massless vectors.
    p, i, j = S("probe::p", "probe::i", "probe::j")
    covariant = result.native.particle("v").spin_sum(p, i, j, covariant=True)
    physical = result.native.particle("v").spin_sum(p, i, j)
    assert covariant != physical
    propagator = next(
        prop for prop in payload["propagators"] if prop["particle"] == "v"
    )
    assert (
        "PSlash" not in propagator["numerator"] and "P(" not in propagator["numerator"]
    )
    assert "Metric" in propagator["numerator"]
    assert result.native.particle("t").mass == S("UFO::Zalias")
    # Materialize the stored propagator as a native two-port local rule, using
    # native index localization rather than interpreting UFO Metric ourselves.
    payload["vertex_rules"] = [
        dict(
            name="PROBE",
            particles=["v", "v"],
            color_structures=["1"],
            lorentz_structures=["PROP_NUMERATOR"],
            couplings=[["ONE"]],
        )
    ]
    payload["lorentz_structures"] = [
        dict(
            name="PROP_NUMERATOR",
            spins=[3, 3],
            structure=propagator["numerator"],
        )
    ]
    payload["couplings"] = [dict(name="ONE", expression="1", orders=[], value=None)]
    probe = hep.Model.from_json(json.dumps(payload))
    actual = _local_numerator(probe, "PROBE", ["v", "v"])
    left, right = (axis.index for axis in actual.axes)
    expected = tensor.TensorExpression(
        E("1i") * probe.particle("v").spin_sum(p, left, right, covariant=True)
    )
    assert (actual - expected).expand().to_expression() == 0


def _local_numerator(native, vertex, particles):
    from symbolica.community import hepkit as hep

    graph = hep.FeynmanDiagram.from_dot(
        native,
        "digraph local { ext [style=invis]; v [int_id="
        + json.dumps(vertex)
        + "];"
        + "".join(
            "ext -> v [particle=" + json.dumps(particle) + "];"
            for particle in particles
        )
        + "}",
    ).apply_feynman_rules()
    return native.expand_couplings(graph.vertices[0].numerator_expression())


def test_structural_adapter_requires_declared_gauge_and_same_scalar_source(source):
    raw = load_raw_loop_model(source, restriction="full")
    result = build_covariant_structure(raw)
    copied = RawLoopModel.from_dict(raw.to_dict())
    with pytest.raises(ValueError, match="another raw model"):
        build_covariant_structure(copied, scalars=result.scalars)
    changed = raw.to_dict()
    changed["metadata"]["gauge"] = [0]
    with pytest.raises(ValueError, match="Feynman gauge"):
        build_covariant_structure(RawLoopModel.from_dict(changed))


def test_invalid_condition_cannot_certify_zero_mass(source):
    parameters = source / "parameters.py"
    parameters.write_text(
        parameters.read_text().replace(
            "value='ZERO+0'", "value='0*(1 if complex(0,1) < 0 else 2)'"
        )
    )
    raw = load_raw_loop_model(source, restriction="full")
    with pytest.raises(NotImplementedError, match="mass validity"):
        build_covariant_structure(raw)


@pytest.fixture(scope="module")
def gmu():
    configured = os.environ.get("PYAMPLICOL_EW_UFO")
    if not configured:
        pytest.skip("set PYAMPLICOL_EW_UFO to the full Gmu fixture")
    raw = load_raw_loop_model(Path(configured), restriction="full")
    with pytest.raises(ValueError, match="goldstoneboson"):
        build_covariant_structure(raw)
    return build_covariant_structure(
        raw,
        compatibility="gmu_particle_anti",
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )


def test_full_gmu_structure_is_covariant_and_exact(gmu):
    from symbolica import E, S

    assert len(gmu.native.particles) == 43
    assert len(gmu.native.vertex_rules) == 163
    assert len(gmu.required_bindings) == 37
    assert gmu.native.particle("u").charge == E("2/3")
    assert gmu.native.particle("u~").charge == -E("2/3")
    assert gmu.native.particle("W+").mass == S("UFO::MW")
    assert gmu.native.particle_by_pdg(22).is_massless
    payload = json.loads(gmu.native.to_json())
    assert payload["covariant_cut_multiplets"]["24"] == [24, 251, 9000003, -9000004]
    assert len(payload["propagators"]) == 43
    assert len(payload["couplings"]) == 132
    assert len(payload["lorentz_structures"]) == 22
    assert gmu.raw.record("particles", "G-")["fields"]["goldstoneboson"] is False
    assert (
        next(p for p in payload["particles"] if p["name"] == "G-")["goldstoneboson"]
        is True
    )
    assert len(gmu.diagnostics) == 1 and gmu.diagnostics[0]["field"] == "goldstoneboson"
    assert all(p["value"] is None for p in payload["parameters"])
    for name in ("Z", "W+", "W-"):
        prop = next(p for p in payload["propagators"] if p["particle"] == name)
        assert "P(" not in prop["numerator"]


@pytest.mark.parametrize("pdgs", [(-2, 2, 21), (21, 21, 21, 21), (-11, 11, 23)])
def test_source_local_components_match_structural_graph(gmu, pdgs):
    from pyamplicol.loops.model import (
        CovariantKernelCatalogue,
        KernelBasisEntry,
        LoopModelIR,
    )

    particles = gmu.raw.collections["particles"]
    vertex = next(
        record
        for record in gmu.raw.collections["vertices"]
        if tuple(
            particles[ref["id"]]["fields"]["pdg_code"]
            for ref in record["fields"]["particles"]
        )
        == pdgs
    )
    fields = vertex["fields"]
    entries = []
    for key, ref in fields["couplings"]["entries"]:
        color, lorentz = key["items"]
        source = gmu.raw.collections["lorentz_structures"][
            fields["lorentz"][lorentz]["id"]
        ]
        entries.append(
            KernelBasisEntry(
                collection="vertices",
                vertex_id=vertex["id"],
                vertex_name=vertex["name"],
                kind="tree",
                color_index=color,
                lorentz_index=lorentz,
                loop_group_index=None,
                particle_ids=tuple(p["id"] for p in fields["particles"]),
                particle_names=tuple(p["name"] for p in fields["particles"]),
                color=fields["color"][color],
                lorentz_id=source["id"],
                lorentz_name=source["name"],
                lorentz=source["fields"]["structure"],
                loop_particles=None,
                coupling=gmu.scalars.definitions["couplings", ref["id"]],
                order=(),
            )
        )
    model = LoopModelIR(
        gmu.raw.name, gmu.raw, gmu.scalars.definitions, tuple(entries), ()
    )
    catalogue = CovariantKernelCatalogue(model, gmu.native)
    expected = None
    for entry in entries:
        component = catalogue.lower(entry)
        contribution = (
            component.color * component.lorentz * entry.coupling.value.expression
        )
        expected = contribution if expected is None else expected + contribution
    actual = _local_numerator(gmu.native, vertex["name"], entries[0].particle_names)
    assert (actual - expected).expand().to_expression() == 0


def test_unknown_gmu_mismatch_and_extra_quantum_mismatch_reject(gmu):
    changed = gmu.raw.to_dict()
    negative = next(p for p in changed["collections"]["particles"] if p["name"] == "G-")
    negative["fields"]["charge_source"]["anti_constructor"]["parameters"][10] = (
        "unrecognized"
    )
    with pytest.raises(ValueError, match="recognized Gmu"):
        build_covariant_structure(
            RawLoopModel.from_dict(changed), compatibility="gmu_particle_anti"
        )
    changed = gmu.raw.to_dict()
    negative = next(p for p in changed["collections"]["particles"] if p["name"] == "G-")
    negative["fields"]["GhostNumber"] = 1
    with pytest.raises(ValueError, match="ghost_number"):
        build_covariant_structure(
            RawLoopModel.from_dict(changed), compatibility="gmu_particle_anti"
        )
