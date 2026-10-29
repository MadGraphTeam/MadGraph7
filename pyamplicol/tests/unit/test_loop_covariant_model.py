# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyamplicol.loops.covariant_model import build_covariant_structure
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_physics


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
    # FeynKit can canonicalize exact internal zero aliases to native ZERO.
    # Source identity belongs to the retained raw declaration in either form.
    assert result.native.particle("t").mass in (0, S("UFO::Zalias"))
    assert result.native.particle("t").mass_parameter in ("ZERO", "Zalias")
    assert result.raw.record("particles", "t")["fields"]["mass"]["name"] == "Zalias"
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


def test_structural_zero_mass_alias_is_named_zero_under_either_hepkit(
    source, monkeypatch
):
    """The published Community 3.0.2 HepKit classifies a mass alias that
    resolves to an exact zero (``t``: ``Zalias = ZERO + 0``) as massless but
    keeps its source symbol, where the integration kernel's (9cdc2949) gives
    mass 0 and exports ZERO. Emulated here on the classification model: the
    structure names ZERO for it, so the native model is byte for byte the one
    of the installed HepKit (the v2 families would otherwise read a derived
    mass of value zero, and the propagator would keep a massive form)."""

    from symbolica import S
    from symbolica.community import hepkit as hep

    raw = load_raw_loop_model(source, restriction="full")
    reference = build_covariant_structure(raw).native.to_json()

    real = hep.Model
    texts: list[str] = []

    class PublishedParticle:
        def __init__(self, particle, mass):
            self._particle, self._mass = particle, mass

        @property
        def mass(self):
            if self._particle.is_massless and self._mass != "ZERO":
                return S(f"UFO::{self._mass}")
            return self._particle.mass

        def __getattr__(self, name):
            return getattr(self._particle, name)

    class PublishedClassification:
        def __init__(self, text):
            self._model = real.from_json(text)
            self._masses = {p["name"]: p["mass"] for p in json.loads(text)["particles"]}

        def particle(self, name):
            return PublishedParticle(self._model.particle(name), self._masses[name])

        def __getattr__(self, name):
            return getattr(self._model, name)

    class PublishedModel:
        @staticmethod
        def from_json(text):
            texts.append(text)
            if len(texts) == 1:
                return PublishedClassification(text)
            return real.from_json(text)

    monkeypatch.setattr(hep, "Model", PublishedModel)
    emulated = build_covariant_structure(raw)
    assert len(texts) == 2
    assert PublishedClassification(texts[0]).particle("t").mass == S("UFO::Zalias")
    assert {p["name"]: p["mass"] for p in json.loads(texts[1])["particles"]}["t"] == (
        "ZERO"
    )
    assert emulated.native.to_json() == reference
    assert emulated.raw.record("particles", "t")["fields"]["mass"]["name"] == "Zalias"


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


def test_unused_propagator_declarations_do_not_change_covariant_kernels(source):
    raw = load_raw_loop_model(source, restriction="full")
    baseline = build_covariant_structure(raw)
    payload = raw.to_dict()
    payload["collections"]["propagators"] = [
        {
            "id": 0,
            "name": "unbound_template",
            "ufo_type": "Propagator",
            "fields": {
                "name": "unbound_template",
                "numerator": "UncertifiedNumerator(id)",
                "denominator": "UncertifiedDenominator(id)",
            },
        }
    ]
    declared = RawLoopModel.from_dict(payload)
    unchanged = declared.to_dict()
    actual = build_covariant_structure(declared)
    assert actual.raw is declared
    assert declared.to_dict() == unchanged
    assert json.loads(actual.native.to_json()) == json.loads(baseline.native.to_json())


@pytest.mark.parametrize("declared", (False, True))
def test_actual_custom_propagator_binding_still_requires_certification(source, declared):
    raw = load_raw_loop_model(source, restriction="full")
    payload = raw.to_dict()
    if declared:
        payload["collections"]["propagators"] = [
            {
                "id": 0,
                "name": "bound_template",
                "ufo_type": "Propagator",
                "fields": {"name": "bound_template", "numerator": "1", "denominator": "1"},
            }
        ]
        binding = {"kind": "reference", "collection": "propagators", "id": 0, "name": "bound_template"}
    else:
        binding = "custom_inline_propagator"
    payload["collections"]["particles"][0]["fields"]["propagator"] = binding
    bound = RawLoopModel.from_dict(payload)
    unchanged = bound.to_dict()
    with pytest.raises(NotImplementedError, match="custom source propagator metadata"):
        build_covariant_structure(bound)
    assert bound.to_dict() == unchanged


@pytest.mark.parametrize("left,right", ((0, None), (None, 0), (0, 0)))
def test_zero_hypercharge_metadata_preserves_unspecified_chirality(source, left, right):
    path = source / "particles.py"
    path.write_text(path.read_text() + """
p=Particle(pdg_code=104,name='p',antiname='p~',spin=1,color=1,mass=P.M,width=P.ZERO,texname='p',antitexname='p~',charge=1)
a=Particle(pdg_code=-104,name='p~',antiname='p',spin=1,color=1,mass=P.M,width=P.ZERO,texname='p~',antitexname='p',charge=-1)
""" + f"p.Y=a.Y={left!r}\np.YRight=a.YRight={right!r}\n")
    raw = load_raw_loop_model(source, restriction="full")
    unchanged = raw.to_dict()
    structure = build_covariant_structure(raw)
    for name in ("p", "p~"):
        particle = structure.native.particle(name)
        assert particle.y_charge == left
        assert particle.y_charge_right == right
    assert raw.to_dict() == unchanged


@pytest.mark.parametrize("key", ("Y", "YRight"))
def test_nonzero_hypercharge_still_requires_chiral_convention(source, key):
    raw = load_raw_loop_model(source, restriction="full")
    payload = raw.to_dict()
    payload["collections"]["particles"][0]["fields"][key] = 1
    marked = RawLoopModel.from_dict(payload)
    with pytest.raises(NotImplementedError, match="chiral hypercharge conjugation"):
        build_covariant_structure(marked)


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo):
    configured = str(loop_gmu_ufo)
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


def test_fixed_masses_are_structural_and_shared_with_numeric_bindings(
    loop_native, source
):
    from decimal import Decimal

    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    raw = load_raw_loop_model(source, restriction="full")
    original = raw.to_dict()
    fixed = build_covariant_structure(raw, fixed_parameters={"M": "0"})
    adjustable = build_covariant_structure(raw)
    tiny = build_covariant_structure(raw, fixed_parameters={"M": "1e-1000"})
    assert fixed.native.particle("s").is_massless
    assert fixed.native.particle("v").is_massless
    assert not adjustable.native.particle("s").is_massless
    assert not tiny.native.particle("s").is_massless
    assert fixed.required_bindings == ()
    assert fixed.fixed_parameters == {"M": ("0", "0")}
    assert fixed.scalars.library.raw is raw and raw.to_dict() == original
    plan = fixed.scalar_bindings((("M", 0), ("Zalias", 0)))
    runtime = NativeScalarBindings(plan)
    assert all(Decimal(x) == 0 for pair in runtime.values.to_decimal() for x in pair)
    with pytest.raises(ValueError, match="fixed during generation"):
        runtime.update({"M": "1"})
    # A zero numerical default is still an adjustable external input. Its
    # runtime update cannot rewrite the retained source or structural model.
    adjustable_runtime = NativeScalarBindings(adjustable.scalar_bindings((("M", 0),)))
    adjustable_runtime.update({"M": "1"})
    assert Decimal(adjustable_runtime.values.to_decimal()[0][0]) == 1
    assert not adjustable.native.particle("s").is_massless
    assert raw.to_dict() == original
    assert all(
        p["value"] is None for p in json.loads(fixed.native.to_json())["parameters"]
    )


def test_fixed_lepton_masses_and_yukawas_remove_exactly_zero_source_vertices(gmu):
    fixed = build_covariant_structure(
        gmu.raw,
        scalars=gmu.scalars,
        compatibility="gmu_particle_anti",
        fixed_parameters={"Me": "0", "MM": "0", "yme": "0", "ymm": "0"},
    )
    assert fixed.native.particle_by_pdg(11).is_massless
    assert fixed.native.particle_by_pdg(13).is_massless
    assert not gmu.native.particle_by_pdg(11).is_massless
    assert len(fixed.required_bindings) == 33
    original = json.loads(gmu.native.to_json())
    specialized = json.loads(fixed.native.to_json())
    zero = set(fixed.diagnostics[1]["couplings"])
    assert zero
    remaining = {vertex["name"]: vertex for vertex in specialized["vertex_rules"]}
    removed = []
    for vertex in original["vertex_rules"]:
        slots = [
            coupling
            for row in vertex["couplings"]
            for coupling in row
            if coupling is not None
        ]
        if all(coupling in zero for coupling in slots):
            assert vertex["name"] not in remaining
            removed.append(vertex)
        else:
            actual = remaining[vertex["name"]]
            assert actual["couplings"] == [
                [None if coupling in zero else coupling for coupling in row]
                for row in vertex["couplings"]
            ]
    # Physical control: Higgs Yukawa vertices disappear only with their explicit
    # Yukawa inputs fixed as well as the kinematic lepton masses.
    assert {tuple(vertex["particles"]) for vertex in removed} >= {
        ("e+", "e-", "H"),
        ("m+", "m-", "H"),
    }
    assert all(p["value"] is None for p in specialized["parameters"])
    assert {coupling["name"] for coupling in specialized["couplings"]}.isdisjoint(zero)


def test_fixed_gmu_masses_preserve_native_graph_family_denominators(gmu):
    from symbolica import S

    from tools.loop_reference.feynkit_loop_graphs import build_loop_process

    zeros = (
        "aEW0M1",
        "MD",
        "MU",
        "MS",
        "MC",
        "MB",
        "Me",
        "MM",
        "MTA",
        "lamWS",
        "AWS",
        "rhoWS",
        "etaWS",
        "ymdo",
        "ymup",
        "yms",
        "ymc",
        "ymb",
        "yme",
        "ymm",
        "ymtau",
    )
    fixed = build_covariant_structure(
        gmu.raw,
        scalars=gmu.scalars,
        compatibility="gmu_particle_anti",
        fixed_parameters={name: "0" for name in zeros},
    )
    for particle, mass in (("e-", "Me"), ("m-", "MM")):
        assert fixed.native.particle(particle).mass in (0, S(f"UFO::{mass}"))
        assert fixed.native.particle(particle).mass_parameter in ("ZERO", mass)
        assert fixed.raw.record("particles", particle)["fields"]["mass"]["name"] == mass
        assert fixed.raw.record("parameters", mass)["fields"]["nature"] == "external"
        assert fixed.fixed_parameters[mass] == ("0", "0")
    assert fixed.native.particle("e-").is_massless
    assert fixed.native.particle("m-").is_massless
    # This exercises the strict custom-denominator gate and native family
    # extraction, not just the structurally massless particle classifier.
    process = build_loop_process(
        fixed.native.process(("e-", "e+"), ("m-", "m+")),
        selected_corrections=["QED"],
        perturbative_orders={"QED": 1},
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
    )
    assert len(process.born) == 2
    assert len(process.loops) == 98
    assert all(graph.family is not None for graph in process.loops)


def test_request_pruning_keeps_full_species_facts_but_not_exact_zero_vertices(source):
    from pyamplicol.loops.request_scope import prepare_loop_request_scope

    with (source / "object_library.py").open("a") as stream:
        stream.write("""
all_couplings=[]
all_lorentz=[]
all_vertices=[]
all_orders=[]
class Coupling:
 def __init__(self,name,value,order):
  self.name=name;self.value=value;self.order=order;all_couplings.append(self)
class Lorentz:
 def __init__(self,name,spins,structure):
  self.name=name;self.spins=spins;self.structure=structure;all_lorentz.append(self)
class Vertex:
 def __init__(self,name,particles,color,lorentz,couplings):
  self.name=name;self.particles=particles;self.color=color;self.lorentz=lorentz;self.couplings=couplings;all_vertices.append(self)
class CouplingOrder:
 def __init__(self,name,expansion_order,hierarchy):
  self.name=name;self.expansion_order=expansion_order;self.hierarchy=hierarchy;all_orders.append(self)
""")
    with (source / "parameters.py").open("a") as stream:
        stream.write("A=Parameter(name='A',nature='external',type='real',value=0.0)\n")
    (
        source / "interactions.py"
    ).write_text("""from .object_library import Coupling, Lorentz, Vertex, CouplingOrder
from . import particles as P
QCD=CouplingOrder('QCD',99,1)
QED=CouplingOrder('QED',99,2)
C1=Coupling(name='C1',value='1',order={'QCD':1})
C2=Coupling(name='C2',value='A',order={'QED':1})
C3=Coupling(name='C3',value='1',order={'QED':7})
L=Lorentz('SSS',[1,1,1],'1')
V1=Vertex('V1',[P.s,P.s,P.s],['1'],[L],{(0,0):C1})
V2=Vertex('V2',[P.s,P.s,P.t],['1'],[L],{(0,0):C2})
V3=Vertex('V3',[P.s,P.s,P.s],['1'],[L],{(0,0):C3})
""")
    with (source / "__init__.py").open("a") as stream:
        stream.write(
            "from . import interactions\nfrom .object_library import all_couplings, all_lorentz, all_vertices, all_orders\n"
        )
    raw = load_raw_loop_model(source, restriction="full")
    scope = prepare_loop_request_scope(
        raw,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    structure = build_covariant_structure(
        raw, request_scope=scope, fixed_parameters={"A": "0"}
    )
    assert [rule.name for rule in structure.native.vertex_rules] == ["V1"]
    # V3 is order-pruned but still proves the full-model QED species fact.
    assert structure.perturbative_species_orders[100] == frozenset({"QCD", "QED"})
    # V2 is exactly zero under the fixed source input, so it supplies no fact.
    assert 101 not in structure.perturbative_species_orders


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"GoldstoneBoson": True}, True),
        ({"goldstoneboson": True}, True),
        ({"GoldstoneBoson": False, "goldstone": True}, True),
        ({"GoldstoneBoson": False, "goldstoneboson": False}, False),
        ({}, False),
    ],
)
def test_goldstone_flag_honours_every_established_ufo_spelling(fields, expected):
    from pyamplicol.processes.model import ufo_goldstone_flag

    assert ufo_goldstone_flag(fields) is expected
